from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.http import require_GET, require_POST
from django.views.decorators.csrf import csrf_exempt
from .streams import generate_stream_detections, generate_multi_camera_stream
from .models import Camera, Incident
import math
import os

ACCIDENT_MODEL_URL = os.getenv("ACCIDENT_MODEL_URL")
FIRE_MODEL_URL     = os.getenv("FIRE_MODEL_URL")
TIME_QUANTUM       = int(os.getenv("TIME_QUANTUM", 3))


@require_GET
def stream_detect(request):
    youtube_url  = request.GET.get("url", "").strip()
    live         = request.GET.get("live", "true").lower() == "true"
    time_quantum = int(request.GET.get("tq", TIME_QUANTUM))

    if not youtube_url:
        return JsonResponse({"error": "url parameter is required."}, status=400)

    gen = generate_stream_detections(
        youtube_url        = youtube_url,
        live               = live,
        time_quantum       = time_quantum,
        accident_model_url = ACCIDENT_MODEL_URL,
        fire_model_url     = FIRE_MODEL_URL
    )

    return StreamingHttpResponse(gen, content_type="application/x-ndjson")


@require_GET
def stream_all_cameras(request):
    time_quantum = int(request.GET.get("tq", TIME_QUANTUM))

    cameras = [
        {
            "id":        cam.id,
            "latitude":  str(cam.latitude),
            "longitude": str(cam.longitude),
            "url":       cam.live_feed_url
        }
        for cam in Camera.objects.all()
    ]

    gen = generate_multi_camera_stream(
        cameras            = cameras,
        accident_model_url = ACCIDENT_MODEL_URL,
        fire_model_url     = FIRE_MODEL_URL,
        time_quantum       = time_quantum
    )

    return StreamingHttpResponse(gen, content_type="application/x-ndjson")


@csrf_exempt
@require_POST
def create_camera(request):
    try:
        latitude      = request.POST.get("latitude",      "").strip()
        longitude     = request.POST.get("longitude",     "").strip()
        live_feed_url = request.POST.get("live_feed_url", "").strip()

        if not latitude or not longitude or not live_feed_url:
            return JsonResponse({
                "success": False,
                "error":   "latitude, longitude and live_feed_url are all required."
            }, status=400)

        try:
            latitude  = float(latitude)
            longitude = float(longitude)
        except ValueError:
            return JsonResponse({
                "success": False,
                "error":   "latitude and longitude must be valid numbers."
            }, status=400)

        if not (-90 <= latitude <= 90):
            return JsonResponse({
                "success": False,
                "error":   "latitude must be between -90 and 90."
            }, status=400)

        if not (-180 <= longitude <= 180):
            return JsonResponse({
                "success": False,
                "error":   "longitude must be between -180 and 180."
            }, status=400)

        if Camera.objects.filter(live_feed_url=live_feed_url).exists():
            return JsonResponse({
                "success": False,
                "error":   "a camera with this live feed URL already exists."
            }, status=409)

        if Camera.objects.filter(latitude=latitude, longitude=longitude).exists():
            return JsonResponse({
                "success": False,
                "error":   "a camera already exists at these exact coordinates."
            }, status=409)

        camera = Camera.objects.create(
            latitude      = latitude,
            longitude     = longitude,
            live_feed_url = live_feed_url
        )

        return JsonResponse({
            "success": True,
            "message": "Camera created successfully.",
            "camera": {
                "id":            camera.id,
                "latitude":      str(camera.latitude),
                "longitude":     str(camera.longitude),
                "live_feed_url": camera.live_feed_url
            }
        }, status=201)

    except Exception as e:
        return JsonResponse({
            "success": False,
            "error":   f"An unexpected error occurred: {str(e)}"
        }, status=500)


def test(request):
    return render(request, "main/temp.html")


def dashboard(request):
    return render(request, "main/dashboard.html")

@require_GET
def delete_all_cameras(request):
    count, _ = Camera.objects.all().delete()
    return JsonResponse({
        "success": True,
        "message": f"Deleted {count} camera(s) from the database."
    })

@csrf_exempt
@require_POST
def create_incident(request):
    try:
        latitude      = request.POST.get("latitude",      "").strip()
        longitude     = request.POST.get("longitude",     "").strip()
        incident_type = request.POST.get("incident_type", "").strip()
        description   = request.POST.get("description",   "").strip()

       
        if not latitude or not longitude or not incident_type:
            return JsonResponse({
                "success": False,
                "error":   "latitude, longitude and incident_type are all required."
            }, status=400)

       
        try:
            latitude  = float(latitude)
            longitude = float(longitude)
        except ValueError:
            return JsonResponse({
                "success": False,
                "error":   "latitude and longitude must be valid numbers."
            }, status=400)

       
        if not (-90 <= latitude <= 90):
            return JsonResponse({
                "success": False,
                "error":   "latitude must be between -90 and 90."
            }, status=400)

        if not (-180 <= longitude <= 180):
            return JsonResponse({
                "success": False,
                "error":   "longitude must be between -180 and 180."
            }, status=400)

       
        incident = Incident.objects.create(
            latitude      = latitude,
            longitude     = longitude,
            incident_type = incident_type,
            description   = description or None   # triggers the auto-default in save()
        )

        return JsonResponse({
            "success": True,
            "message": "Incident created successfully.",
            "incident": {
                "id":            incident.id,
                "latitude":      str(incident.latitude),
                "longitude":     str(incident.longitude),
                "incident_type": incident.incident_type,
                "description":   incident.description
            }
        }, status=201)

    except Exception as e:
        return JsonResponse({
            "success": False,
            "error":   f"An unexpected error occurred: {str(e)}"
        }, status=500)

@require_GET
def delete_all_incidents(request):
    count, _ = Incident.objects.all().delete()
    return JsonResponse({
        "success": True,
        "message": f"Deleted {count} camera(s) from the database."
    })

def get_incidents_within_radius(latitude: float, longitude: float, distance_km: float):
    """
    Returns all Incident objects within `distance_km` kilometres of the
    given (latitude, longitude) point.

    Two-stage approach:
      1. Bounding-box query  — cheap DB-level pre-filter (square).
      2. Haversine check     — precise Python-level post-filter (circle).
    """

    # ── Stage 1: Bounding box ──────────────────────────────────────────────
    # Degrees of lat/lon that correspond to `distance_km` from the target point.
    lat_delta = distance_km / 111.32
    lon_delta = distance_km / (111.32 * math.cos(math.radians(latitude)))

    min_lat = latitude  - lat_delta
    max_lat = latitude  + lat_delta
    min_lon = longitude - lon_delta
    max_lon = longitude + lon_delta

    candidates = Incident.objects.filter(
        latitude__gte  = min_lat,
        latitude__lte  = max_lat,
        longitude__gte = min_lon,
        longitude__lte = max_lon,
    )

    # ── Stage 2: Haversine filter ──────────────────────────────────────────
    EARTH_RADIUS_KM = 6371.0

    def haversine(lat1, lon1, lat2, lon2) -> float:
        """Returns the great-circle distance in km between two coordinate pairs."""
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (math.sin(dlat / 2) ** 2
             + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)

        return EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(a))

    results = [
        incident for incident in candidates
        if haversine(
            latitude,
            longitude,
            float(incident.latitude),
            float(incident.longitude)
        ) <= distance_km
    ]

    return results

@csrf_exempt
@require_GET
def incidents_within_radius(request):
    try:
        latitude    = request.GET.get("latitude",    "").strip()
        longitude   = request.GET.get("longitude",   "").strip()
        distance_km = request.GET.get("distance_km", "").strip()

        
        if not latitude or not longitude or not distance_km:
            return JsonResponse({
                "success": False,
                "error":   "latitude, longitude and distance_km are all required."
            }, status=400)

        
        try:
            latitude    = float(latitude)
            longitude   = float(longitude)
            distance_km = float(distance_km)
        except ValueError:
            return JsonResponse({
                "success": False,
                "error":   "latitude, longitude and distance_km must all be valid numbers."
            }, status=400)

        
        if not (-90 <= latitude <= 90):
            return JsonResponse({
                "success": False,
                "error":   "latitude must be between -90 and 90."
            }, status=400)

        if not (-180 <= longitude <= 180):
            return JsonResponse({
                "success": False,
                "error":   "longitude must be between -180 and 180."
            }, status=400)

        if distance_km <= 0:
            return JsonResponse({
                "success": False,
                "error":   "distance_km must be a positive number."
            }, status=400)

       
        incidents = get_incidents_within_radius(latitude, longitude, distance_km)

        return JsonResponse({
            "success":        True,
            "searched_at":    {"latitude": latitude, "longitude": longitude},
            "distance_km":    distance_km,
            "incident_count": len(incidents),
            "incidents": [
                {
                    "id":            incident.id,
                    "latitude":      str(incident.latitude),
                    "longitude":     str(incident.longitude),
                    "incident_type": incident.incident_type,
                    "description":   incident.description,
                }
                for incident in incidents
            ]
        }, status=200)

    except Exception as e:
        return JsonResponse({
            "success": False,
            "error":   f"An unexpected error occurred: {str(e)}"
        }, status=500)