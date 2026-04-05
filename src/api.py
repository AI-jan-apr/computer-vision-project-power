import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STATUS_JSON_PATH = os.path.join(BASE_DIR, "..", "parking_status.json")
EVENTS_JSON_PATH = os.path.join(BASE_DIR, "..", "parking_events.json")
LATEST_FRAME_PATH = os.path.join(BASE_DIR, "..", "latest_frame.jpg")

app = FastAPI(title="Smart Parking API")

# allow frontend later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def read_json_file(path, default_data):
    if not os.path.exists(path):
        return default_data

    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default_data

@app.get("/")
def root():
    return {"message": "Smart Parking API is running"}

@app.get("/admin/dashboard")
def admin_dashboard():
    status_data = read_json_file(
        STATUS_JSON_PATH,
        {
            "frame": 0,
            "summary": {
                "total_slots": 0,
                "occupied_slots": 0,
                "empty_slots": 0,
                "occupancy_rate": 0,
                "active_alerts": 0,
                "wrong_parking_count": 0
            },
            "slots": []
        }
    )

    events_data = read_json_file(EVENTS_JSON_PATH, [])

    recent_events = events_data[-20:] if isinstance(events_data, list) else []

    return {
        "frame": status_data.get("frame", 0),
        "summary": status_data.get("summary", {}),
        "slots": status_data.get("slots", []),
        "events": recent_events,
        "latest_frame_url": "/latest-frame"
    }

@app.get("/user/availability")
def user_availability():
    status_data = read_json_file(
        STATUS_JSON_PATH,
        {
            "frame": 0,
            "summary": {
                "total_slots": 0,
                "occupied_slots": 0,
                "empty_slots": 0,
                "occupancy_rate": 0,
                "active_alerts": 0,
                "wrong_parking_count": 0
            },
            "slots": []
        }
    )

    summary = status_data.get("summary", {})
    slots = status_data.get("slots", [])

    simplified_slots = [
        {
            "slot_id": slot["slot_id"],
            "status": slot["status"]
        }
        for slot in slots
    ]

    recommended_slot = next(
        (slot["slot_id"] for slot in slots if slot["status"] == "empty"),
        None
    )

    return {
        "frame": status_data.get("frame", 0),
        "total_slots": summary.get("total_slots", 0),
        "available_slots": summary.get("empty_slots", 0),
        "occupied_slots": summary.get("occupied_slots", 0),
        "occupancy_rate": summary.get("occupancy_rate", 0),
        "recommended_slot": recommended_slot,
        "slots": simplified_slots,
        "latest_frame_url": "/latest-frame"
    }

@app.get("/events")
def get_events():
    events_data = read_json_file(EVENTS_JSON_PATH, [])
    return {"events": events_data}

@app.get("/latest-frame")
def latest_frame():
    if not os.path.exists(LATEST_FRAME_PATH):
        return JSONResponse(
            status_code=404,
            content={"message": "Latest frame not found"}
        )

    return FileResponse(LATEST_FRAME_PATH, media_type="image/jpeg")