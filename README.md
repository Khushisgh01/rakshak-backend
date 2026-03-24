# Rakshak — Live Incident Detection & Analytics Backend

Rakshak is a Django-based backend for real-time detection of accidents and fire/smoke events
across multiple live camera streams. It uses two independently hosted computer vision models
(served via Google Colab + ngrok) and exposes a streaming API, a camera management API,
and an incident analytics API. A live multi-camera dashboard is included out of the box.

---

## Table of Contents

1. Prerequisites
2. Project Structure
3. Environment Setup
4. Running the Model Servers
5. Configuring Environment Variables
6. Running the Django Server
7. API Reference
   - Stream Detection
   - Multi-Camera Dashboard Stream
   - Camera Management
   - Incident Management

---

## 1. Prerequisites

Before running Rakshak, ensure you have the following ready.

**System requirements**

- Python 3.10 or higher
- pip
- git

**External services**

- Two separate ngrok accounts (or two authtoken-distinct tunnels). Each model server needs its
  own publicly accessible HTTPS URL. A single ngrok free account only allows one tunnel at a
  time, so two accounts are required.

- A Google account to run notebooks in Google Colab (free tier is sufficient for testing, but
  a Colab Pro subscription is recommended for GPU access and longer runtimes).

**Model server notebooks**

Both model servers must be running and publicly exposed before starting Rakshak. The notebooks
are available at:

    https://github.com/unusualcatcher/models_servers

There are two notebooks in that repository:

- Accident detection model server
- Fire and smoke detection model server

Each notebook starts a server that exposes a /detect endpoint, and uses ngrok to tunnel
it to a public HTTPS URL. You must copy those two URLs into your .env file (described in
Section 5).

Open each notebook in Google Colab, paste your respective ngrok authtokens when prompted,
and run all cells. Keep both Colab tabs open and active for the duration of your session.
The tunnel URLs change every time you restart a notebook, so update your .env file accordingly.

---

## 2. Project Structure
```
rakshak/
├── main/
│   ├── migrations/
│   │   ├── 0001_initial.py          # Camera model
│   │   └── 0002_incident.py         # Incident model
│   ├── templates/
│   │   └── main/
│   │       ├── dashboard.html       # Multi-camera live dashboard
│   │       └── temp.html            # Single-stream test UI
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py                    # Camera and Incident models
│   ├── streams.py                   # Core streaming and detection logic
│   ├── tests.py
│   ├── urls.py
│   └── views.py                     # All API views
├── rakshak/
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── .env                             # You create this (see Section 5)
├── .gitignore
├── manage.py
└── requirements.txt
```

---

## 3. Environment Setup

**Step 1 — Download and extract**

Download the project zip file and extract it to a directory of your choice.

    cd path/to/rakshak

**Step 2 — Create a virtual environment**

    python -m venv venv

Activate it:

On Linux/macOS:

    source venv/bin/activate

On Windows:

    venv\Scripts\activate

**Step 3 — Install dependencies**

    pip install -r requirements.txt

Note: The requirements include PyTorch (torch), OpenCV headless, and various others dependencies. The total
download size may be several hundred megabytes. Ensure you have a stable internet connection.

Alternatively run the following command if the above command failed for whatever reason:

    pip install django torch numpy opencv-python-headless requests python-dotenv sympy yt-dlp django-cors-headers


**Step 4 — Apply database migrations**

    python manage.py migrate

This creates the local SQLite database (db.sqlite3) with the Camera and Incident tables.

---

## 4. Running the Model Servers

**Step 1 — Open the notebooks**

Go to https://github.com/unusualcatcher/models_servers and open each notebook in Google Colab
using the "Open in Colab" button or by uploading the .ipynb files manually.

**Step 2 — Set up ngrok authtokens**

You need two separate ngrok accounts. Sign up at https://ngrok.com if you do not already have
two accounts. From each account's dashboard, copy the authtoken.

In each notebook, there will be a cell where you paste your ngrok authtoken. Paste a different
authtoken in each notebook so both tunnels can run simultaneously.

**Step 3 — Run all cells**

Run all cells in both notebooks. Each notebook will:

- Install dependencies
- Load the model
- Start a server on a local port
- Use ngrok to expose that server to a public HTTPS URL

At the end of execution, each notebook will print a public URL in the format:

    https://xxxx-xx-xx-xxx-xx.ngrok-free.app

Copy both URLs. You will need them in the next step.

**Step 4 — Keep the notebooks alive**

Do not close or idle out the Colab tabs. If a session disconnects, re-run all cells and update
your .env file with the new ngrok URLs, then restart the Django server.

---

## 5. Configuring Environment Variables

Create a file named .env in the root of the project (the same directory as manage.py).

    touch .env

Add the following contents:

    ACCIDENT_MODEL_URL=https://your-accident-model-ngrok-url.ngrok-free.app
    FIRE_MODEL_URL=https://your-fire-model-ngrok-url.ngrok-free.app
    TIME_QUANTUM=3

**ACCIDENT_MODEL_URL** — The public ngrok URL for the accident detection model server.
**FIRE_MODEL_URL** — The public ngrok URL for the fire and smoke detection model server.
**TIME_QUANTUM** — How often (in seconds) a frame is sampled and sent to both models.
                   Default is 3. Lower values increase detection frequency but also
                   increase load on the model servers.

These variables are loaded automatically via python-dotenv when Django starts.

---

## 6. Running the Django Server

Once your .env is configured and both model server notebooks are running:

    python manage.py runserver

The server starts at http://127.0.0.1:8000 by default.

To allow connections from other devices on your network:

    python manage.py runserver 0.0.0.0:8000

---

## 7. API Reference

All endpoints are relative to the server root, e.g. http://127.0.0.1:8000.

---

### Stream Detection

**GET /stream/detect/**

Streams real-time detection results for a single YouTube URL (live or recorded). The response
is a newline-delimited stream of JSON objects (application/x-ndjson). Each line is a
self-contained JSON object. The connection stays open until the stream ends or the client
disconnects.

Query parameters:

| Parameter | Type    | Required | Default      | Description                                      |
|-----------|---------|----------|--------------|--------------------------------------------------|
| url       | string  | Yes      | —            | Full YouTube URL of the stream or video          |
| live      | boolean | No       | true         | true for live streams, false for recorded videos |
| tq        | integer | No       | TIME_QUANTUM | Seconds between sampled frames                   |

Example request:

    GET /stream/detect/?url=https://www.youtube.com/watch?v=XXXX&live=true&tq=5

The response is a stream of JSON lines. The sequence of event types you will receive is:

**For live streams:**

    {"status": "initialising", "mode": "live", "time_quantum": 3}

    {"status": "url_fetched", "title": "Stream Title", "timing": {"fetch_ms": 120.4}}

    {
      "status": "connected",
      "title": "Stream Title",
      "timing": {
        "fetch_ms": 120.4,
        "thread_spawn_ms": 0.3,
        "buffer_fill_ms": 840.1,
        "poll_overhead_ms": 12.2,
        "total_to_first_frame_ms": 972.8
      }
    }

    {
      "status": "frame",
      "frame_count": 1,
      "timing": {"cap_read_ms": 14.2},
      "models": {
        "accident": {
          "detected": false,
          "detections": [],
          "false_positives": [],
          "inference_ms": 212,
          "roundtrip_ms": 380.5,
          "network_overhead_ms": 168.5
        },
        "fire": {
          "detected": true,
          "detections": [
            {
              "type": "fire",
              "confidence": 0.91,
              "coverage": 0.04,
              "box": {"x1": 120, "y1": 80, "x2": 300, "y2": 220}
            }
          ],
          "false_positives": [],
          "inference_ms": 198,
          "roundtrip_ms": 360.2,
          "network_overhead_ms": 162.2
        }
      }
    }

    {"status": "stream_ended", "message": "The live stream has ended.", "frame_count": 47}

**For recorded videos:**

Recorded video streams follow the same structure but emit a video_info event instead of
connected, include a timestamp_s and position field on each frame event, and end with
a video_complete event instead of stream_ended.

    {"status": "video_info", "title": "...", "duration_s": 120.5, "fps": 30.0,
     "total_frames": 3615, "time_quantum": 3, "timing": {"video_open_ms": 84.1}}

    {
      "status": "frame",
      "frame_count": 1,
      "timestamp_s": 0.0,
      "position": "first",
      "timing": {"seek_and_read_ms": 22.1},
      "models": { ... }
    }

    {"status": "video_complete", "message": "Video playback complete.", "frame_count": 41}

The position field on each frame can be: "first", "mid", "last_edge_case", or
"first_and_last" (when the video is very short).

**Hiccup events** (live streams only) are emitted if the stream momentarily drops:

    {"status": "hiccup", "message": "Stream read failed momentarily — attempting to continue.",
     "hiccup_count": 1}

**Error events:**

    {"status": "error", "message": "Could not fetch stream — ..."}

---

### Multi-Camera Dashboard Stream

**GET /stream/cameras/**

Streams real-time detection results for all cameras currently stored in the database. All
cameras are processed in parallel on separate threads. Events from all cameras are multiplexed
into a single stream, with each event tagged with the originating camera's metadata.

Query parameters:

| Parameter | Type    | Required | Default      | Description                    |
|-----------|---------|----------|--------------|--------------------------------|
| tq        | integer | No       | TIME_QUANTUM | Seconds between sampled frames |

Example request:

    GET /stream/cameras/

The response stream always begins with an initialising_all event:

    {
      "status": "initialising_all",
      "camera_count": 3,
      "camera_ids": [1, 2, 3]
    }

Every subsequent event object contains additional camera context fields appended to it:

    "camera_id": 1,
    "camera_latitude": "28.613900",
    "camera_longitude": "77.209000",
    "camera_url": "https://www.youtube.com/watch?v=XXXX"

A frame event from this endpoint looks like:

    {
      "status": "frame",
      "frame_count": 4,
      "timing": {"cap_read_ms": 18.7},
      "models": {
        "accident": { ... },
        "fire": { ... }
      },
      "camera_id": 2,
      "camera_latitude": "28.700100",
      "camera_longitude": "77.102300",
      "camera_url": "https://www.youtube.com/watch?v=YYYY"
    }

When a single camera's stream ends:

    {
      "status": "camera_stream_ended",
      "message": "This camera's stream has ended.",
      "camera_id": 1,
      ...
    }

When all cameras have ended:

    {"status": "all_cameras_ended", "message": "All camera streams have ended."}

If no cameras exist in the database:

    {"status": "no_cameras", "message": "No cameras found in database."}

---

### Camera Management

**POST /camera/create/**

Creates a new camera record in the database.

Content-Type: application/x-www-form-urlencoded

Request body fields:

| Field         | Type   | Required | Description                                        |
|---------------|--------|----------|----------------------------------------------------|
| latitude      | float  | Yes      | Latitude of the camera (-90 to 90)                 |
| longitude     | float  | Yes      | Longitude of the camera (-180 to 180)              |
| live_feed_url | string | Yes      | YouTube URL of the camera's live stream            |

Example request:

    curl -X POST http://127.0.0.1:8000/camera/create/ \
      -d "latitude=28.6139&longitude=77.2090&live_feed_url=https://www.youtube.com/watch?v=XXXX"

Success response (HTTP 201):

    {
      "success": true,
      "message": "Camera created successfully.",
      "camera": {
        "id": 1,
        "latitude": "28.613900",
        "longitude": "77.209000",
        "live_feed_url": "https://www.youtube.com/watch?v=XXXX"
      }
    }

Error responses:

- HTTP 400 — Missing required fields or invalid coordinate values
- HTTP 409 — A camera already exists at these exact coordinates, or with this exact URL
- HTTP 500 — Unexpected server error

---

**GET /camera/delete-all/**

Deletes all camera records from the database.

Example request:

    GET /camera/delete-all/

Response:

    {
      "success": true,
      "message": "Deleted 3 camera(s) from the database."
    }

---

### Incident Management

Incidents represent detected or manually reported events such as accidents, fires, or other
emergencies. They are stored in the database with a timestamp and geographic coordinates,
and can be queried by radius to serve as an analytics tool for authorities to identify
where incidents are clustering over time.

---

**POST /incident/create/**

Creates a new incident record. This endpoint can be called both by the detection system
automatically and by operators logging incidents manually.

Content-Type: application/x-www-form-urlencoded

Request body fields:

| Field         | Type     | Required | Description                                                  |
|---------------|----------|----------|--------------------------------------------------------------|
| latitude      | float    | Yes      | Latitude of the incident (-90 to 90)                         |
| longitude     | float    | Yes      | Longitude of the incident (-180 to 180)                      |
| incident_type | string   | Yes      | Type label, e.g. "accident", "fire", "flood"                 |
| date_created  | datetime | Yes      | ISO 8601 datetime of when the incident occurred              |
| description   | string   | No       | Free-text description. Auto-generated if left blank.         |

If description is omitted or empty, the system automatically generates one in the format:

    "An incident of type {incident_type} occurred at latitude {lat} and longitude: {lon}."

The date_created field must be a valid ISO 8601 datetime string, for example:

    2026-03-24T14:30:00Z

Example request:

    curl -X POST http://127.0.0.1:8000/incident/create/ \
      -d "latitude=28.6139&longitude=77.2090&incident_type=fire&date_created=2026-03-24T14:30:00Z&description=Fire spotted near highway"

Success response (HTTP 201):

    {
      "success": true,
      "message": "Incident created successfully.",
      "incident": {
        "id": 1,
        "latitude": "28.613900",
        "longitude": "77.209000",
        "incident_type": "fire",
        "description": "Fire spotted near highway",
        "date_created": "2026-03-24 14:30:00+00:00"
      }
    }

Error responses:

- HTTP 400 — Missing required fields or invalid coordinate values
- HTTP 500 — Unexpected server error

---

**GET /incident/within-radius/**

Returns all incidents within a given radius of a coordinate point. Uses a bounding-box
pre-filter at the database level, followed by a precise Haversine calculation in Python
to return only incidents within the true circular radius. This endpoint is the primary
analytics tool — authorities can query it with a location and radius to get a full picture
of all incidents that have occurred in that area.

Query parameters:

| Parameter   | Type  | Required | Description                                   |
|-------------|-------|----------|-----------------------------------------------|
| latitude    | float | Yes      | Center point latitude (-90 to 90)             |
| longitude   | float | Yes      | Center point longitude (-180 to 180)          |
| distance_km | float | Yes      | Search radius in kilometres (must be > 0)     |

Example request:

    GET /incident/within-radius/?latitude=28.6139&longitude=77.2090&distance_km=5.0

Success response (HTTP 200):

    {
      "success": true,
      "searched_at": {
        "latitude": 28.6139,
        "longitude": 77.209
      },
      "distance_km": 5.0,
      "incident_count": 2,
      "incidents": [
        {
          "id": 1,
          "latitude": "28.613900",
          "longitude": "77.209000",
          "incident_type": "fire",
          "description": "Fire spotted near highway",
          "date_created": "2026-03-24 14:30:00+00:00"
        },
        {
          "id": 2,
          "latitude": "28.620100",
          "longitude": "77.198400",
          "incident_type": "accident",
          "description": "An incident of type accident occurred at latitude 28.6201 and longitude: 77.1984.",
          "date_created": "2026-03-24 15:45:00+00:00"
        }
      ]
    }

Error responses:

- HTTP 400 — Missing fields, invalid numbers, or distance_km is zero or negative
- HTTP 500 — Unexpected server error

---

**GET /incident/delete-all/**

Deletes all incident records from the database.

Example request:

    GET /incident/delete-all/

Response:

    {
      "success": true,
      "message": "Deleted 5 camera(s) from the database."
    }
