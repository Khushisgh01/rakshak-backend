import yt_dlp
import cv2
import numpy as np
import threading
import time
import requests
import json
import queue
from concurrent.futures import ThreadPoolExecutor


class _LiveSession:
    def __init__(self):
        self.latest_frame          = None
        self.stream_active         = False
        self.t_first_frame_grabbed = None
        self.frame_grab_time       = None
        self.hiccup_count          = 0
        self.hiccup_ts             = None


def _frame_grabber(session, stream_url):
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        session.stream_active = False
        return

    first = True
    while session.stream_active:
        t_before   = time.perf_counter()
        ret, frame = cap.read()
        t_after    = time.perf_counter()

        if ret:
            session.frame_grab_time = (t_after - t_before) * 1000
            if first:
                session.t_first_frame_grabbed = time.perf_counter()
                first = False
            session.latest_frame = frame
        else:
            session.hiccup_count += 1
            session.hiccup_ts     = time.perf_counter()
            time.sleep(0.5)
            ret2, frame2 = cap.read()
            if ret2:
                session.latest_frame = frame2
            else:
                session.stream_active = False
                break

    cap.release()


def _send_to_model(frame, model_server_url):
    try:
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        jpg_bytes = buffer.tobytes()

        t_start  = time.perf_counter()
        response = requests.post(
            f"{model_server_url.rstrip('/')}/detect",
            files={"image": ("frame.jpg", jpg_bytes, "image/jpeg")},
            timeout=15
        )
        roundtrip_ms = (time.perf_counter() - t_start) * 1000

        if response.status_code == 200:
            result = response.json()
            result["_roundtrip_ms"] = round(roundtrip_ms, 1)
            return result

        return {
            "detected": False, "detections": [], "false_positives": [],
            "inference_ms": 0,
            "_roundtrip_ms": round(roundtrip_ms, 1),
            "_error": f"HTTP {response.status_code}"
        }

    except requests.exceptions.Timeout:
        return {
            "detected": False, "detections": [], "false_positives": [],
            "inference_ms": 0, "_roundtrip_ms": -1,
            "_error": "Request timed out (>15s)"
        }
    except Exception as e:
        return {
            "detected": False, "detections": [], "false_positives": [],
            "inference_ms": 0, "_roundtrip_ms": -1,
            "_error": str(e)
        }


def _send_to_both_models(frame, accident_url, fire_url):
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_accident = executor.submit(_send_to_model, frame, accident_url)
        future_fire     = executor.submit(_send_to_model, frame, fire_url)
        accident_result = future_accident.result()
        fire_result     = future_fire.result()
    return accident_result, fire_result


def _build_model_block(model_result):
    if model_result is None:
        return None

    if "_error" in model_result:
        return {"error": model_result["_error"]}

    inference_ms = model_result.get("inference_ms", 0)
    roundtrip_ms = model_result.get("_roundtrip_ms", 0)

    clean_detections = []
    for det in model_result.get("detections", []):
        x1, y1, x2, y2 = det["box"]
        clean_detections.append({
            "type":       det["type"],
            "confidence": det["confidence"],
            "coverage":   det.get("coverage", 0),
            "box":        {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        })

    clean_fp = []
    for fp in model_result.get("false_positives", []):
        x1, y1, x2, y2 = fp["box"]
        clean_fp.append({
            "type":       fp["type"],
            "confidence": fp["confidence"],
            "coverage":   fp.get("coverage", 0),
            "box":        {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "fp_reason":  fp.get("fp_reason", "coverage exceeded threshold")
        })

    return {
        "detected":            model_result.get("detected", False),
        "detections":          clean_detections,
        "false_positives":     clean_fp,
        "inference_ms":        inference_ms,
        "roundtrip_ms":        roundtrip_ms,
        "network_overhead_ms": round(roundtrip_ms - inference_ms, 1)
    }


def _emit(payload):
    return json.dumps(payload) + "\n"


def _generate_non_live(stream_url, title, time_quantum, accident_url, fire_url):
    t_open  = time.perf_counter()
    cap     = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        yield _emit({"status": "error", "message": "Could not open video URL."})
        return
    open_ms = (time.perf_counter() - t_open) * 1000

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / fps if fps > 0 else 0

    yield _emit({
        "status":       "video_info",
        "title":        title,
        "duration_s":   round(duration_s, 2),
        "fps":          round(fps, 2),
        "total_frames": total_frames,
        "time_quantum": time_quantum,
        "timing":       {"video_open_ms": round(open_ms, 1)}
    })

    timestamps = []
    t = 0.0
    while t < duration_s:
        timestamps.append(t)
        t += time_quantum

    last_frame_time = (total_frames - 1) / fps if fps > 0 else 0
    if not timestamps or abs(timestamps[-1] - last_frame_time) > 0.01:
        timestamps.append(last_frame_time)

    frame_count = 0

    for i, ts in enumerate(timestamps):
        is_first = (i == 0)
        is_last  = (i == len(timestamps) - 1)

        t_seek     = time.perf_counter()
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        seek_ms    = (time.perf_counter() - t_seek) * 1000

        if not ret or frame is None:
            yield _emit({
                "status":    "warning",
                "message":   f"Could not read frame at {ts:.2f}s — skipping.",
                "timestamp": ts
            })
            continue

        frame_count += 1

        if is_first and is_last:
            position_label = "first_and_last"
        elif is_first:
            position_label = "first"
        elif is_last:
            position_label = "last_edge_case"
        else:
            position_label = "mid"

        accident_result, fire_result = _send_to_both_models(frame, accident_url, fire_url)

        try:
            yield _emit({
                "status":      "frame",
                "frame_count": frame_count,
                "timestamp_s": round(ts, 2),
                "position":    position_label,
                "timing":      {"seek_and_read_ms": round(seek_ms, 1)},
                "models": {
                    "accident": _build_model_block(accident_result),
                    "fire":     _build_model_block(fire_result)
                }
            })
        except GeneratorExit:
            cap.release()
            return

        if not is_last:
            time.sleep(time_quantum)

    cap.release()
    yield _emit({
        "status":      "video_complete",
        "message":     "Video playback complete.",
        "frame_count": frame_count
    })


def _generate_live(stream_url, title, fetch_ms, time_quantum, accident_url, fire_url):
    session                = _LiveSession()
    session.stream_active  = True
    last_seen_hiccup_count = 0

    t_thread_start  = time.perf_counter()
    grabber         = threading.Thread(
        target=_frame_grabber,
        args=(session, stream_url),
        daemon=True
    )
    grabber.start()
    thread_spawn_ms = (time.perf_counter() - t_thread_start) * 1000

    t_wait = time.perf_counter()
    while session.latest_frame is None and session.stream_active:
        time.sleep(0.05)
        if time.perf_counter() - t_wait > 15:
            yield _emit({"status": "error", "message": "No frame received within 15s."})
            session.stream_active = False
            return

    if not session.stream_active:
        yield _emit({
            "status":      "stream_ended",
            "message":     "Live stream ended before first frame could be captured.",
            "frame_count": 0
        })
        return

    t_first_in_python = time.perf_counter()
    wait_ms     = (t_first_in_python - t_thread_start) * 1000
    grab_lag_ms = (session.t_first_frame_grabbed - t_thread_start) * 1000 \
                  if session.t_first_frame_grabbed else 0

    yield _emit({
        "status": "connected",
        "title":  title,
        "timing": {
            "fetch_ms":                round(fetch_ms, 1),
            "thread_spawn_ms":         round(thread_spawn_ms, 1),
            "buffer_fill_ms":          round(grab_lag_ms, 1),
            "poll_overhead_ms":        round(wait_ms - grab_lag_ms, 1),
            "total_to_first_frame_ms": round(wait_ms + fetch_ms, 1)
        }
    })

    frame_count = 0

    try:
        while True:
            if session.hiccup_count != last_seen_hiccup_count:
                last_seen_hiccup_count = session.hiccup_count
                yield _emit({
                    "status":       "hiccup",
                    "message":      "Stream read failed momentarily — attempting to continue.",
                    "hiccup_count": session.hiccup_count
                })

            if not session.stream_active:
                yield _emit({
                    "status":      "stream_ended",
                    "message":     "The live stream has ended.",
                    "frame_count": frame_count
                })
                return

            if session.latest_frame is not None:
                frame_count    += 1
                current_grab_ms = session.frame_grab_time or 0

                accident_result, fire_result = _send_to_both_models(
                    session.latest_frame, accident_url, fire_url
                )

                yield _emit({
                    "status":      "frame",
                    "frame_count": frame_count,
                    "timing":      {"cap_read_ms": round(current_grab_ms, 1)},
                    "models": {
                        "accident": _build_model_block(accident_result),
                        "fire":     _build_model_block(fire_result)
                    }
                })

            time.sleep(time_quantum)

    except GeneratorExit:
        session.stream_active = False
        return


def generate_stream_detections(youtube_url, live=True, time_quantum=3,
                                accident_model_url=None, fire_model_url=None):
    yield _emit({
        "status":       "initialising",
        "mode":         "live" if live else "non_live",
        "time_quantum": time_quantum
    })

    ydl_opts = {
        "format": "best" if live else "best[ext=mp4]/best",
        "quiet":  True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info       = ydl.extract_info(youtube_url, download=False)
            stream_url = info["url"]
            title      = info.get("title", "Unknown")
    except Exception as e:
        yield _emit({"status": "error", "message": f"Could not fetch stream — {e}"})
        return

    t_fetch_start = time.perf_counter()
    fetch_ms      = (time.perf_counter() - t_fetch_start) * 1000

    yield _emit({
        "status": "url_fetched",
        "title":  title,
        "timing": {"fetch_ms": round(fetch_ms, 1)}
    })

    if not live:
        yield from _generate_non_live(
            stream_url, title, time_quantum, accident_model_url, fire_model_url
        )
    else:
        yield from _generate_live(
            stream_url, title, fetch_ms, time_quantum, accident_model_url, fire_model_url
        )


def generate_multi_camera_stream(cameras, accident_model_url, fire_model_url, time_quantum):
    if not cameras:
        yield _emit({"status": "no_cameras", "message": "No cameras found in database."})
        return

    event_queue = queue.Queue()
    stop_event  = threading.Event()

    yield _emit({
        "status":       "initialising_all",
        "camera_count": len(cameras),
        "camera_ids":   [c["id"] for c in cameras]
    })

    def camera_worker(camera):
        camera_id  = camera["id"]
        camera_tag = {
            "camera_id":        camera_id,
            "camera_latitude":  camera["latitude"],
            "camera_longitude": camera["longitude"],
            "camera_url":       camera["url"]
        }

        try:
            gen = generate_stream_detections(
                youtube_url        = camera["url"],
                live               = True,
                time_quantum       = time_quantum,
                accident_model_url = accident_model_url,
                fire_model_url     = fire_model_url
            )

            for raw in gen:
                if stop_event.is_set():
                    return

                raw = raw.strip()
                if not raw:
                    continue

                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                payload.update(camera_tag)
                event_queue.put(payload)

            event_queue.put({
                "status":  "camera_stream_ended",
                "message": "This camera's stream has ended.",
                **camera_tag
            })

        except Exception as e:
            event_queue.put({
                "status":  "camera_error",
                "message": str(e),
                **camera_tag
            })

    threads = []
    for camera in cameras:
        t = threading.Thread(target=camera_worker, args=(camera,), daemon=True)
        t.start()
        threads.append(t)

    try:
        while True:
            try:
                payload = event_queue.get(timeout=1.0)
                yield json.dumps(payload) + "\n"
            except queue.Empty:
                if all(not t.is_alive() for t in threads):
                    yield _emit({
                        "status":  "all_cameras_ended",
                        "message": "All camera streams have ended."
                    })
                    return
                continue

    except GeneratorExit:
        stop_event.set()
        return