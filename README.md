# Rakshak — Live Incident Detection & Analytics Backend

Rakshak is a Django-based backend for real-time detection of accidents and fire/smoke events
across multiple live camera streams. It uses two independently hosted computer vision models
(served via Google Colab and ngrok) and exposes a streaming API, a camera management API,
and an incident analytics API. A live multi-camera dashboard is included out of the box.

---

## Table of Contents

1. Prerequisites
2. Project Structure
3. Database Models
4. Environment Setup
5. Running the Model Servers
6. Configuring Environment Variables
7. Running the Django Server
8. API Reference
   - Stream Detection
   - Multi-Camera Stream
   - Camera Management
   - Incident Management
   - Utility Endpoints

---

## 1. Prerequisites

**System requirements**
- Python 3.10 or higher
- pip
- git

**External services**
- Two separate ngrok accounts (or two authtoken-distinct tunnels). Each model server needs its
  own publicly accessible HTTPS URL. A single ngrok free account only allows one tunnel at a
  time, so two accounts are required.
- A Google account to run notebooks in Google Colab. Free tier is sufficient for testing, but
  Colab Pro is recommended for GPU access and longer runtimes.

**Model server notebooks**
Both model servers must be running and publicly exposed before starting Rakshak. The notebooks
are available at:

    https://github.com/unusualcatcher/model_servers

There are two notebooks in that repository:
- Accident detection model server
- Fire and smoke detection model server

Each notebook starts a FastAPI server that exposes a /detect endpoint, and uses ngrok to
tunnel it to a public HTTPS URL. Copy those two URLs into your .env file (described in
Section 6). The tunnel URLs change every time you restart a notebook, so update your .env
file accordingly and restart the Django server.

---

## 2. Project Structure
```
rakshak/
├── main/
│   ├── migrations/
│   │   ├── 0001_initial.py
│   │   └── 0002_incident.py
│   ├── templates/
│   │   └── main/
│   │       ├── dashboard.html       # Multi-camera live dashboard
│   │       └── temp.html            # Single-stream test UI
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── streams.py                   # Core streaming and detection logic
│   ├── tests.py
│   ├── urls.py
│   └── views.py
├── rakshak/
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── footages/                        # Auto-generated incident clip storage
│   └── .gitkeep
├── .env                             # You create this (see Section 6)
├── .gitignore
├── manage.py
└── requirements.txt
```

---

## 3. Database Models

Rakshak uses three database tables. The schema for each is described below.

---

### Camera

Stores the cameras that Rakshak monitors.
```
+---------------+------------------------+------------------------------------------+
| Field         | Type                   | Notes                                    |
+---------------+------------------------+------------------------------------------+
| id            | AutoField (PK)         | Auto-assigned integer primary key        |
| latitude      | DecimalField(9, 6)     | Range -90 to 90                          |
| longitude     | DecimalField(9, 6)     | Range -180 to 180                        |
| live_feed_url | CharField(500)         | YouTube URL of the camera feed           |
| live          | BooleanField           | true = live stream, false = recorded     |
+---------------+------------------------+------------------------------------------+
```

Uniqueness constraints enforced at the application level:
- No two cameras may share the same live_feed_url
- No two cameras may share the same latitude/longitude pair

---

### Incident

Stores manually or automatically reported incidents. Used for analytics queries.
```
+---------------+------------------------+------------------------------------------+
| Field         | Type                   | Notes                                    |
+---------------+------------------------+------------------------------------------+
| id            | AutoField (PK)         | Auto-assigned integer primary key        |
| latitude      | DecimalField(9, 6)     | Range -90 to 90                          |
| longitude     | DecimalField(9, 6)     | Range -180 to 180                        |
| incident_type | CharField(500)         | e.g. "accident", "fire", "flood"         |
| description   | CharField(1000)        | Nullable. Auto-generated if not supplied |
| date_created  | DateTimeField          | Nullable. Must be ISO 8601 on create     |
+---------------+------------------------+------------------------------------------+
```

If description is left blank on creation, it is automatically set to:
"An incident of type {incident_type} occurred at latitude {lat} and longitude: {lon}."

---

### Camera_Incident

Created automatically when a camera detects an incident. One record per camera — it is
updated in place each time a new incident is detected after the cooldown (RECENT_CUTOFF)
has elapsed.
```
+---------------+------------------------+------------------------------------------+
| Field         | Type                   | Notes                                    |
+---------------+------------------------+------------------------------------------+
| id            | AutoField (PK)         | Auto-assigned integer primary key        |
| camera        | ForeignKey → Camera    | CASCADE on camera delete                 |
| incident_type | CharField(500)         | Detection type code (see below)          |
| date_created  | DateTimeField          | Set by the server at detection time      |
| footage       | CharField(1000)        | Absolute path to the saved .avi clip     |
+---------------+------------------------+------------------------------------------+
```

**Incident type codes**
```
+------+------------------------------------------+
| Code | Meaning                                  |
+------+------------------------------------------+
| c    | Crash only                               |
| f    | Fire only                                |
| s    | Smoke only                               |
| cf   | Crash and fire                           |
| cs   | Crash and smoke                          |
| fs   | Fire and smoke                           |
| cfs  | Crash, fire, and smoke                   |
| o    | Other (detected but unclassified)        |
+------+------------------------------------------+
```

**Cooldown behaviour**

After a Camera_Incident is created for a given camera, no new record is created for that
camera until RECENT_CUTOFF seconds have elapsed. Within that window, every detection frame
returns the existing record with "created": false. Once the cooldown expires, the existing
record is updated in place with the new incident type, timestamp, and footage path.

---

## 4. Environment Setup

**Step 1 — Download the ZIP file and extract it**

    cd extracted-path/rakshak-main

**Step 2 — Create a virtual environment**

    python -m venv venv 
    # the first "venv" is part of the command, the second "venv" is simply the name of the folder which can be anything like "venvpath".

Activate it:

    # "venv" here means the name of the folder containing your virtual environment
    # Linux / macOS
    source venv/bin/activate

    # Windows
    venv\Scripts\activate

**Step 3 — Install dependencies**

    pip install -r requirements.txt

If that fails for any reason, install manually:

    pip install django torch numpy opencv-python-headless requests python-dotenv sympy yt-dlp django-cors-headers

**Step 4 — Apply database migrations**

    python manage.py migrate

This creates the local SQLite database (db.sqlite3) with all required tables.

---

## 5. Running the Model Servers

**Step 1 — Open the notebooks**

Go to https://github.com/unusualcatcher/model_servers and open each notebook in Google Colab.

**Step 2 — Set up ngrok authtokens**

You need two separate ngrok accounts. Sign up at https://ngrok.com. From each account's
dashboard, copy the authtoken. Paste a different authtoken in each notebook so both tunnels
can run simultaneously.

**Step 3 — Run all cells**

Run all cells in both notebooks. Each notebook will install dependencies, load the YOLO model,
start a local server, and expose it via ngrok. At the end, each notebook prints a public URL:

    https://xxxx-xx-xx-xxx-xx.ngrok-free.app

Copy both URLs. You need them in the next step.

**Step 4 — Keep the notebooks alive**

Do not close or idle out the Colab tabs. If a session disconnects, re-run all cells, copy the
new ngrok URLs, update your .env file, and restart the Django server.

---

## 6. Configuring Environment Variables

Create a file named .env in the root of the project (same directory as manage.py):

    touch .env

Paste and fill in the following:
```
ACCIDENT_MODEL_URL=https://your-accident-model-ngrok-url.ngrok-free.app
FIRE_MODEL_URL=https://your-fire-smoke-model-ngrok-url.ngrok-free.app
TIME_QUANTUM=3
ACCIDENT_CONF_THRESHOLD=0.35
RECENT_CUTOFF=6000
```

**Variable reference**
```
+------------------------+--------+---------+------------------------------------------------+
| Variable               | Type   | Default | Description                                    |
+------------------------+--------+---------+------------------------------------------------+
| ACCIDENT_MODEL_URL     | string | —       | ngrok URL for the accident detection server    |
| FIRE_MODEL_URL         | string | —       | ngrok URL for the fire/smoke detection server  |
| TIME_QUANTUM           | int    | 3       | Seconds between sampled frames per camera      |
| ACCIDENT_CONF_THRESHOLD| float  | 0.35    | Minimum confidence to count an accident hit    |
| RECENT_CUTOFF          | int    | 6000    | Cooldown in seconds before a new Camera_       |
|                        |        |         | Incident can be created for the same camera    |
+------------------------+--------+---------+------------------------------------------------+
```

---

## 7. Running the Django Server

Once your .env is configured and both model server notebooks are running:

    python manage.py runserver

The server starts at http://127.0.0.1:8000 by default.

To allow connections from other devices on your network:

    python manage.py runserver 0.0.0.0:8000

---

## 8. API Reference

All endpoints are relative to the server root, e.g. http://127.0.0.1:8000.
All POST endpoints accept application/x-www-form-urlencoded request bodies.
All streaming endpoints return application/x-ndjson — one self-contained JSON object per line.

---

### Stream Detection

**GET /stream/detect/**

Streams real-time detection results for a single YouTube URL. The connection stays open for
the duration of the stream or video, emitting one JSON line per sampled frame. This endpoint
does not create Camera_Incident records — it is for inspecting raw model output only.

**Query parameters**
```
+-------------+---------+----------+--------------+--------------------------------------------------+
| Parameter   | Type    | Required | Default      | Description                                      |
+-------------+---------+----------+--------------+--------------------------------------------------+
| url         | string  | Yes      | —            | Full YouTube URL to analyse                      |
| live        | boolean | No       | true         | true for live streams, false for recorded videos |
| tq          | integer | No       | TIME_QUANTUM | Seconds between sampled frames                   |
+-------------+---------+----------+--------------+--------------------------------------------------+
```

**Example request**

    curl -N "http://127.0.0.1:8000/stream/detect/?url=https://www.youtube.com/watch?v=XXXX&live=true&tq=5"

**Live stream — event sequence**

    {"status": "initialising", "mode": "live", "time_quantum": 5}

    {"status": "url_fetched", "title": "Stream Title", "timing": {"fetch_ms": 118.4}}

    {
      "status": "connected",
      "title": "Stream Title",
      "timing": {
        "fetch_ms": 118.4,
        "thread_spawn_ms": 0.3,
        "buffer_fill_ms": 820.1,
        "poll_overhead_ms": 11.2,
        "total_to_first_frame_ms": 949.8
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
              "box": {"x1": 120.0, "y1": 80.0, "x2": 300.0, "y2": 220.0}
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

**Recorded video — event sequence**

    {"status": "initialising", "mode": "non_live", "time_quantum": 3}

    {"status": "url_fetched", "title": "Video Title", "timing": {"fetch_ms": 95.1}}

    {
      "status": "video_info",
      "title": "Video Title",
      "duration_s": 120.5,
      "fps": 30.0,
      "total_frames": 3615,
      "time_quantum": 3,
      "timing": {"video_open_ms": 84.1}
    }

    {
      "status": "frame",
      "frame_count": 1,
      "timestamp_s": 0.0,
      "position": "first",
      "timing": {"seek_and_read_ms": 22.1},
      "models": {
        "accident": { ... },
        "fire": { ... }
      }
    }

    {"status": "video_complete", "message": "Video playback complete.", "frame_count": 41}

The position field on each frame event is one of:
```
+------------------+----------------------------------------------------------+
| Value            | Meaning                                                  |
+------------------+----------------------------------------------------------+
| first            | First sampled frame of the video                         |
| mid              | Any frame between first and last                         |
| last_edge_case   | Final frame of the video                                 |
| first_and_last   | Video is short enough that only one frame was sampled    |
+------------------+----------------------------------------------------------+
```

**Hiccup events** (live streams only) are emitted if the stream momentarily drops:

    {"status": "hiccup", "message": "Stream read failed momentarily — attempting to continue.", "hiccup_count": 1}

**Error events:**

    {"status": "error", "message": "Could not fetch stream — <reason>"}
    {"status": "error", "message": "url parameter is required."}

---

### Multi-Camera Stream

**GET /stream/cameras/**

Streams real-time detection results for all cameras in the database whose live field matches
the live query parameter. All matching cameras are processed in parallel on separate threads.
Events from all cameras are multiplexed into a single NDJSON stream, with each event tagged
with the originating camera's metadata. This endpoint also creates and updates Camera_Incident
records when detections occur.

**Query parameters**
```
+-------------+---------+----------+--------------+---------------------------------------------------+
| Parameter   | Type    | Required | Default      | Description                                       |
+-------------+---------+----------+--------------+---------------------------------------------------+
| tq          | integer | No       | TIME_QUANTUM | Seconds between sampled frames per camera         |
| live        | boolean | No       | true         | Filters cameras by their live field in the DB     |
+-------------+---------+----------+--------------+---------------------------------------------------+
```

**Example request**

    curl -N "http://127.0.0.1:8000/stream/cameras/?live=true&tq=3"

**Event sequence**

Every event in this stream contains the following extra fields appended to it:

    "camera_id": 1,
    "camera_latitude": "28.613900",
    "camera_longitude": "77.209000",
    "camera_url": "https://www.youtube.com/watch?v=XXXX"

Full event sequence:

    {"status": "initialising_all", "camera_count": 2, "camera_ids": [1, 2]}

    {"status": "initialising", "mode": "live", "time_quantum": 3, "camera_id": 1, ...}
    {"status": "initialising", "mode": "live", "time_quantum": 3, "camera_id": 2, ...}

    {"status": "url_fetched", "title": "Camera 1 Title", "camera_id": 1, ...}
    {"status": "url_fetched", "title": "Camera 2 Title", "camera_id": 2, ...}

    {"status": "connected", "title": "Camera 1 Title", "camera_id": 1, ...}
    {"status": "connected", "title": "Camera 2 Title", "camera_id": 2, ...}

    {
      "status": "frame",
      "frame_count": 1,
      "timing": {"cap_read_ms": 18.7},
      "models": {
        "accident": {
          "detected": true,
          "detections": [
            {
              "type": "accident",
              "confidence": 0.87,
              "coverage": 0.61,
              "box": {"x1": 10.0, "y1": 0.0, "x2": 1910.0, "y2": 1075.0}
            }
          ],
          "false_positives": [],
          "inference_ms": 98,
          "roundtrip_ms": 2240.1,
          "network_overhead_ms": 2142.1
        },
        "fire": {
          "detected": false,
          "detections": [],
          "false_positives": [],
          "inference_ms": 11,
          "roundtrip_ms": 2190.3,
          "network_overhead_ms": 2179.3
        }
      },
      "camera_incident": {
        "created": true,
        "id": 1,
        "incident_type": "c",
        "footage_path": "C:\\...\\footages\\cam_1_1774472806.avi"
      },
      "camera_id": 1,
      "camera_latitude": "28.613900",
      "camera_longitude": "77.209000",
      "camera_url": "https://www.youtube.com/watch?v=XXXX"
    }

The camera_incident field is only present on frames where a detection occurred. Its fields:
```
+---------------+-----------------------------------------------------------+
| Field         | Meaning                                                   |
+---------------+-----------------------------------------------------------+
| created       | true if a new/updated record was written, false if within |
|               | the RECENT_CUTOFF cooldown window                         |
| id            | Database ID of the Camera_Incident record                 |
| incident_type | Type code (c, f, s, cf, cs, fs, cfs, o)                  |
| footage_path  | Absolute path to the saved .avi clip on the server        |
+---------------+-----------------------------------------------------------+
```

When all cameras have ended:

    {"status": "all_cameras_ended", "message": "All camera streams have ended."}

If no cameras match the query:

    {"status": "no_cameras", "message": "No cameras found in database."}

---

### Camera Management

**POST /camera/create/**

Creates a new camera record.

**Request body fields**
```
+---------------+--------+----------+-----------------------------------------------+
| Field         | Type   | Required | Description                                   |
+---------------+--------+----------+-----------------------------------------------+
| latitude      | float  | Yes      | Camera latitude (-90 to 90)                   |
| longitude     | float  | Yes      | Camera longitude (-180 to 180)                |
| live_feed_url | string | Yes      | YouTube URL of the camera feed                |
| live          | bool   | No       | true (default) or false                       |
+---------------+--------+----------+-----------------------------------------------+
```

**Example request**

    curl -X POST http://127.0.0.1:8000/camera/create/ \
      -d "latitude=28.6139&longitude=77.2090&live_feed_url=https://www.youtube.com/watch?v=XXXX&live=true"

**Success response (HTTP 201)**

    {
      "success": true,
      "message": "Camera created successfully.",
      "camera": {
        "id": 1,
        "latitude": "28.613900",
        "longitude": "77.209000",
        "live_feed_url": "https://www.youtube.com/watch?v=XXXX",
        "live": true
      }
    }

**Error responses**

- HTTP 400 — Missing required fields or invalid coordinate values
- HTTP 409 — A camera already exists at these exact coordinates, or with this exact URL
- HTTP 500 — Unexpected server error

---

**GET /camera/delete-all/**

Deletes all camera records from the database.

    curl "http://127.0.0.1:8000/camera/delete-all/"

**Response**

    {
      "success": true,
      "message": "Deleted 3 camera(s) from the database."
    }

---

### Incident Management

**POST /incident/create/**

Creates a new incident record.

**Request body fields**
```
+---------------+----------+----------+--------------------------------------------------------------+
| Field         | Type     | Required | Description                                                  |
+---------------+----------+----------+--------------------------------------------------------------+
| latitude      | float    | Yes      | Incident latitude (-90 to 90)                                |
| longitude     | float    | Yes      | Incident longitude (-180 to 180)                             |
| incident_type | string   | Yes      | Type label, e.g. "accident", "fire", "flood"                 |
| date_created  | datetime | Yes      | ISO 8601 datetime string                                     |
| description   | string   | No       | Free-text description. Auto-generated if omitted or empty.   |
+---------------+----------+----------+--------------------------------------------------------------+
```

**Example request**

    curl -X POST http://127.0.0.1:8000/incident/create/ \
      -d "latitude=28.6139&longitude=77.2090&incident_type=fire&date_created=2026-03-24T14:30:00Z&description=Fire spotted near highway"

**Success response (HTTP 201)**

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

**Error responses**

- HTTP 400 — Missing required fields or invalid coordinate values
- HTTP 500 — Unexpected server error

---

**GET /incident/within-radius/**

Returns all incidents within a given radius of a coordinate point. Uses a bounding-box
pre-filter at the database level, followed by a Haversine calculation in Python to return
only incidents within the true circular radius.

**Query parameters**
```
+-------------+-------+----------+-----------------------------------------------+
| Parameter   | Type  | Required | Description                                   |
+-------------+-------+----------+-----------------------------------------------+
| latitude    | float | Yes      | Center point latitude (-90 to 90)             |
| longitude   | float | Yes      | Center point longitude (-180 to 180)          |
| distance_km | float | Yes      | Search radius in kilometres (must be > 0)     |
+-------------+-------+----------+-----------------------------------------------+
```

**Example request**

    curl "http://127.0.0.1:8000/incident/within-radius/?latitude=28.6139&longitude=77.2090&distance_km=5.0"

**Success response (HTTP 200)**

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

**Error responses**

- HTTP 400 — Missing fields, invalid numbers, or distance_km is zero or negative
- HTTP 500 — Unexpected server error

---

**GET /incident/delete-all/**

Deletes all incident records from the database.

    curl "http://127.0.0.1:8000/incident/delete-all/"

**Response**

    {
      "success": true,
      "message": "Deleted 5 incident(s) from the database."
    }

---

### Utility Endpoints

**POST /delete-all-camera-incidents/**

Deletes all Camera_Incident records. Useful for resetting detection state without removing
camera registrations.

    curl -X POST http://127.0.0.1:8000/delete-all-camera-incidents/

**Response**

    {
      "success": true,
      "message": "Deleted 2 camera incident(s).",
      "deleted_count": 2
    }

---

**POST /clean-database/**

Deletes all Camera_Incident, Incident, and Camera records in one operation. Intended for
development and testing only — remove this endpoint before deploying to production.

    curl -X POST http://127.0.0.1:8000/clean-database/

**Response**

    {
      "success": true,
      "message": "All data deleted successfully.",
      "deleted_counts": {
        "camera_incidents": 2,
        "incidents": 5,
        "cameras": 3
      }
    }

---

**POST /delete-by-coordinates/**

Deletes a single Camera, Incident, or Camera_Incident record matched by its coordinates.

**Request body fields**
```
+------------+--------+----------+----------------------------------------------------------+
| Field      | Type   | Required | Description                                              |
+------------+--------+----------+----------------------------------------------------------+
| latitude   | float  | Yes      | Exact latitude to match (up to 6 decimal places)         |
| longitude  | float  | Yes      | Exact longitude to match (up to 6 decimal places)        |
| to_delete  | string | Yes      | One of: camera, incident, camera_incident                |
+------------+--------+----------+----------------------------------------------------------+
```

**Example request**

    curl -X POST http://127.0.0.1:8000/delete-by-coordinates/ \
      -d "latitude=28.6139&longitude=77.2090&to_delete=camera"

**Success response (HTTP 200)**

    {
      "success": true,
      "message": "Camera record deleted successfully.",
      "deleted": {
        "id": 1,
        "model": "Camera",
        "latitude": "28.613900",
        "longitude": "77.209000"
      }
    }

**Error responses**

- HTTP 400 — Missing fields, invalid coordinates, or invalid to_delete value
- HTTP 404 — No matching record found for the given coordinates
- HTTP 500 — Unexpected server error
