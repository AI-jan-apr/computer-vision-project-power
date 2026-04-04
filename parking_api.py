import os
import json
import time
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.getcwd()

STATUS_JSON_PATH = os.path.join(BASE_DIR, "parking_status.json")
EVENTS_JSON_PATH = os.path.join(BASE_DIR, "parking_system_events.json")
LATEST_FRAME_PATH = os.path.join(BASE_DIR, "latest_frame.jpg")

SESSIONS_JSON_PATH = os.path.join(BASE_DIR, "parking_sessions.json")
USERS_JSON_PATH = os.path.join(BASE_DIR, "users.json")
NOTIFICATIONS_JSON_PATH = os.path.join(BASE_DIR, "notifications.json")
FEES_JSON_PATH = os.path.join(BASE_DIR, "fees.json")

FREE_PARKING_MINUTES = 120
WARNING_BEFORE_END_MINUTES = 15
FEE_INTERVAL_MINUTES = 30
BASE_INTERVAL_FEE = 10.0
WRONG_PARKING_DEFAULT_FEE = 50.0

app = FastAPI(title="Smart Parking API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# HELPERS
# ============================================================

def read_json_file(path, default_data):
    if not os.path.exists(path):
        return default_data

    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default_data


def write_json_file(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def now_str():
    return datetime.now().isoformat()


def parse_dt(value):
    if not value:
        return None
    return datetime.fromisoformat(value)


def default_status_payload():
    return {
        "frame": 0,
        "summary": {
            "total_slots": 0,
            "occupied_slots": 0,
            "empty_slots": 0,
            "occupancy_rate": 0,
            "active_alerts": 0,
            "wrong_parking_count": 0,
            "total_cars_detected": 0
        },
        "slots": []
    }


def ensure_storage_files():
    if not os.path.exists(SESSIONS_JSON_PATH):
        write_json_file(SESSIONS_JSON_PATH, [])
    if not os.path.exists(USERS_JSON_PATH):
        write_json_file(USERS_JSON_PATH, {})
    if not os.path.exists(NOTIFICATIONS_JSON_PATH):
        write_json_file(NOTIFICATIONS_JSON_PATH, [])
    if not os.path.exists(FEES_JSON_PATH):
        write_json_file(FEES_JSON_PATH, [])


def load_all_state():
    ensure_storage_files()
    return {
        "status": read_json_file(STATUS_JSON_PATH, default_status_payload()),
        "events": read_json_file(EVENTS_JSON_PATH, []),
        "sessions": read_json_file(SESSIONS_JSON_PATH, []),
        "users": read_json_file(USERS_JSON_PATH, {}),
        "notifications": read_json_file(NOTIFICATIONS_JSON_PATH, []),
        "fees": read_json_file(FEES_JSON_PATH, []),
    }


def save_all_state(state):
    write_json_file(SESSIONS_JSON_PATH, state["sessions"])
    write_json_file(USERS_JSON_PATH, state["users"])
    write_json_file(NOTIFICATIONS_JSON_PATH, state["notifications"])
    write_json_file(FEES_JSON_PATH, state["fees"])


def get_slot_from_status(status_data, slot_id):
    return next((slot for slot in status_data.get("slots", []) if slot.get("slot_id") == slot_id), None)


def get_active_session_for_user(sessions, user_id):
    return next(
        (s for s in sessions if s.get("user_id") == user_id and s.get("status") == "active"),
        None
    )


def get_active_session_for_slot(sessions, slot_id):
    return next(
        (s for s in sessions if s.get("slot_id") == slot_id and s.get("status") == "active"),
        None
    )


def add_notification(notifications, title, message, type_, user_id=None, slot_id=None, extra=None):
    notification = {
        "notification_id": len(notifications) + 1,
        "type": type_,
        "title": title,
        "message": message,
        "user_id": user_id,
        "slot_id": slot_id,
        "created_at": now_str(),
        "read": False
    }
    if extra:
        notification.update(extra)
    notifications.append(notification)
    return notification


def fee_already_exists(fees, session_id, interval_index):
    return any(
        fee.get("session_id") == session_id and fee.get("interval_index") == interval_index
        for fee in fees
    )


def update_consistency_score(user, wrong_parking=False, overstay=False):
    if wrong_parking:
        user["wrong_parking_count"] = user.get("wrong_parking_count", 0) + 1
        user["consistency_score"] = max(0, user.get("consistency_score", 100) - 15)

    if overstay:
        user["overstay_count"] = user.get("overstay_count", 0) + 1
        user["consistency_score"] = max(0, user.get("consistency_score", 100) - 10)


def process_live_rules(state):
    status_data = state["status"]
    sessions = state["sessions"]
    users = state["users"]
    notifications = state["notifications"]
    fees = state["fees"]

    now = datetime.now()

    for session in sessions:
        if session.get("status") != "active":
            continue

        user_id = session["user_id"]
        slot_id = session["slot_id"]
        slot_data = get_slot_from_status(status_data, slot_id)

        if user_id not in users:
            users[user_id] = {
                "user_id": user_id,
                "consistency_score": 100,
                "wrong_parking_count": 0,
                "overstay_count": 0
            }

        user = users[user_id]

        start_time = parse_dt(session.get("start_time"))
        if start_time is None:
            continue

        elapsed = now - start_time
        elapsed_minutes = int(elapsed.total_seconds() // 60)

        free_end = FREE_PARKING_MINUTES
        warning_start = FREE_PARKING_MINUTES - WARNING_BEFORE_END_MINUTES

        if slot_data is None or slot_data.get("status") == "empty":
            session["status"] = "ended"
            session["ended_at"] = now_str()
            session["ended_reason"] = "slot_became_empty"
            continue

        if slot_data.get("wrong_parking", False):
            if not session.get("wrong_parking_user_warned", False):
                add_notification(
                    notifications,
                    title="Wrong parking warning",
                    message=f"Your selected slot {slot_id} is marked as wrongly parked. Please correct your parking position.",
                    type_="wrong_parking_user_warning",
                    user_id=user_id,
                    slot_id=slot_id,
                    extra={"session_id": session["session_id"]}
                )
                session["wrong_parking_user_warned"] = True

            session["wrong_parking_seen_count"] = session.get("wrong_parking_seen_count", 0) + 1

            if session["wrong_parking_seen_count"] >= 2 and not session.get("wrong_parking_admin_warned", False):
                add_notification(
                    notifications,
                    title="Wrong parking persisted",
                    message=f"Slot {slot_id} has a wrongly parked car that persists and may require admin action.",
                    type_="wrong_parking_admin_warning",
                    slot_id=slot_id,
                    extra={"session_id": session["session_id"]}
                )
                session["wrong_parking_admin_warned"] = True

                if not session.get("wrong_parking_penalized", False):
                    update_consistency_score(user, wrong_parking=True)
                    session["wrong_parking_penalized"] = True
        else:
            session["wrong_parking_seen_count"] = 0

        if elapsed_minutes >= warning_start and elapsed_minutes < free_end:
            if not session.get("limit_warning_sent", False):
                remaining = free_end - elapsed_minutes
                add_notification(
                    notifications,
                    title="Free parking is ending soon",
                    message=f"Your free parking time in slot {slot_id} is ending soon. About {remaining} minutes remain.",
                    type_="free_limit_warning",
                    user_id=user_id,
                    slot_id=slot_id,
                    extra={"session_id": session["session_id"], "minutes_remaining": remaining}
                )
                session["limit_warning_sent"] = True

        if elapsed_minutes > free_end:
            overdue_minutes = elapsed_minutes - free_end
            interval_index = (overdue_minutes - 1) // FEE_INTERVAL_MINUTES + 1

            if not session.get("overdue_warning_sent", False):
                add_notification(
                    notifications,
                    title="Parking fee started",
                    message=f"Your free parking period for slot {slot_id} has ended. Billing has started.",
                    type_="overdue_warning",
                    user_id=user_id,
                    slot_id=slot_id,
                    extra={"session_id": session["session_id"]}
                )
                session["overdue_warning_sent"] = True

            if not fee_already_exists(fees, session["session_id"], interval_index):
                fee_amount = BASE_INTERVAL_FEE * interval_index
                fees.append({
                    "fee_id": len(fees) + 1,
                    "session_id": session["session_id"],
                    "user_id": user_id,
                    "slot_id": slot_id,
                    "interval_index": interval_index,
                    "amount": fee_amount,
                    "created_at": now_str(),
                    "paid": False,
                    "reason": f"Overdue parking fee interval {interval_index}"
                })

                add_notification(
                    notifications,
                    title="New parking fee added",
                    message=f"A new parking fee has been added for slot {slot_id}. Current interval: {interval_index}.",
                    type_="fee_added",
                    user_id=user_id,
                    slot_id=slot_id,
                    extra={"session_id": session["session_id"], "interval_index": interval_index, "amount": fee_amount}
                )

                if not session.get("overstay_penalized", False):
                    update_consistency_score(user, overstay=True)
                    session["overstay_penalized"] = True

    save_all_state(state)


# ============================================================
# REQUEST MODELS
# ============================================================

class SelectSlotRequest(BaseModel):
    user_id: str
    slot_id: int


class ReleaseSlotRequest(BaseModel):
    user_id: str


class CreateWrongParkingFeeRequest(BaseModel):
    slot_id: int
    amount: float = WRONG_PARKING_DEFAULT_FEE
    reason: str = "Wrong parking fee"


# ============================================================
# ROOT / HEALTH
# ============================================================

@app.get("/")
def root():
    return {"message": "Smart Parking API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


# ============================================================
# ADMIN ENDPOINTS
# ============================================================

@app.get("/admin/dashboard")
def admin_dashboard():
    state = load_all_state()
    process_live_rules(state)

    status_data = state["status"]
    events_data = state["events"]
    notifications_data = state["notifications"]

    recent_events = events_data[-20:] if isinstance(events_data, list) else []
    recent_notifications = notifications_data[-20:] if isinstance(notifications_data, list) else []

    return {
        "frame": status_data.get("frame", 0),
        "summary": status_data.get("summary", {}),
        "slots": status_data.get("slots", []),
        "events": recent_events,
        "notifications": recent_notifications,
        "latest_frame_url": "/latest-frame",
        "video_feed_url": "/video-feed"
    }


@app.get("/admin/summary")
def admin_summary():
    state = load_all_state()
    process_live_rules(state)
    return state["status"].get("summary", {})


@app.get("/admin/slots")
def admin_slots():
    state = load_all_state()
    process_live_rules(state)
    return {
        "frame": state["status"].get("frame", 0),
        "slots": state["status"].get("slots", [])
    }


@app.get("/admin/notifications")
def admin_notifications():
    state = load_all_state()
    process_live_rules(state)
    return {"notifications": state["notifications"][-50:]}


@app.post("/admin/create-wrong-parking-fee")
def create_wrong_parking_fee(payload: CreateWrongParkingFeeRequest):
    state = load_all_state()
    process_live_rules(state)

    slot_data = get_slot_from_status(state["status"], payload.slot_id)
    if not slot_data:
        raise HTTPException(status_code=404, detail="Slot not found")

    active_session = get_active_session_for_slot(state["sessions"], payload.slot_id)

    fee = {
        "fee_id": len(state["fees"]) + 1,
        "session_id": active_session["session_id"] if active_session else None,
        "user_id": active_session["user_id"] if active_session else None,
        "slot_id": payload.slot_id,
        "interval_index": None,
        "amount": payload.amount,
        "created_at": now_str(),
        "paid": False,
        "reason": payload.reason
    }
    state["fees"].append(fee)

    add_notification(
        state["notifications"],
        title="Wrong parking fee created",
        message=f"Admin created a wrong parking fee for slot {payload.slot_id}.",
        type_="admin_wrong_parking_fee_created",
        user_id=active_session["user_id"] if active_session else None,
        slot_id=payload.slot_id,
        extra={"amount": payload.amount}
    )

    save_all_state(state)
    return {"message": "Wrong parking fee created", "fee": fee}


# ============================================================
# USER ENDPOINTS
# ============================================================

@app.get("/user/availability")
def user_availability():
    state = load_all_state()
    process_live_rules(state)

    status_data = state["status"]
    summary = status_data.get("summary", {})
    slots = status_data.get("slots", [])

    simplified_slots = [
        {
            "slot_id": slot.get("slot_id"),
            "status": slot.get("status"),
            "wrong_parking": slot.get("wrong_parking", False),
            "confidence": slot.get("confidence", 0)
        }
        for slot in slots
    ]

    recommended_slot = next(
        (slot.get("slot_id") for slot in slots if slot.get("status") == "empty"),
        None
    )

    return {
        "frame": status_data.get("frame", 0),
        "total_slots": summary.get("total_slots", 0),
        "available_slots": summary.get("empty_slots", 0),
        "occupied_slots": summary.get("occupied_slots", 0),
        "occupancy_rate": summary.get("occupancy_rate", 0),
        "wrong_parking_count": summary.get("wrong_parking_count", 0),
        "total_cars_detected": summary.get("total_cars_detected", 0),
        "recommended_slot": recommended_slot,
        "slots": simplified_slots,
        "latest_frame_url": "/latest-frame",
        "video_feed_url": "/video-feed"
    }


@app.post("/user/select-slot")
def user_select_slot(payload: SelectSlotRequest):
    state = load_all_state()
    process_live_rules(state)

    status_data = state["status"]
    sessions = state["sessions"]
    users = state["users"]
    notifications = state["notifications"]

    slot_data = get_slot_from_status(status_data, payload.slot_id)
    if not slot_data:
        raise HTTPException(status_code=404, detail="Slot not found")

    if slot_data.get("status") != "empty":
        raise HTTPException(status_code=400, detail="Slot is not empty")

    existing_session = get_active_session_for_user(sessions, payload.user_id)
    if existing_session:
        raise HTTPException(status_code=400, detail="User already has an active parking session")

    if payload.user_id not in users:
        users[payload.user_id] = {
            "user_id": payload.user_id,
            "consistency_score": 100,
            "wrong_parking_count": 0,
            "overstay_count": 0
        }

    session = {
        "session_id": len(sessions) + 1,
        "user_id": payload.user_id,
        "slot_id": payload.slot_id,
        "start_time": now_str(),
        "status": "active",
        "wrong_parking_seen_count": 0,
        "wrong_parking_user_warned": False,
        "wrong_parking_admin_warned": False,
        "wrong_parking_penalized": False,
        "limit_warning_sent": False,
        "overdue_warning_sent": False,
        "overstay_penalized": False
    }
    sessions.append(session)

    add_notification(
        notifications,
        title="Slot selected",
        message=f"You selected slot {payload.slot_id}. Parking session started.",
        type_="slot_selected",
        user_id=payload.user_id,
        slot_id=payload.slot_id,
        extra={"session_id": session["session_id"]}
    )

    save_all_state(state)

    return {
        "message": "Slot selected successfully",
        "session": session
    }


@app.post("/user/release-slot")
def user_release_slot(payload: ReleaseSlotRequest):
    state = load_all_state()
    process_live_rules(state)

    session = get_active_session_for_user(state["sessions"], payload.user_id)
    if not session:
        raise HTTPException(status_code=404, detail="No active session found for this user")

    session["status"] = "ended"
    session["ended_at"] = now_str()
    session["ended_reason"] = "user_released"

    add_notification(
        state["notifications"],
        title="Parking session ended",
        message=f"Your parking session for slot {session['slot_id']} has ended.",
        type_="session_ended",
        user_id=payload.user_id,
        slot_id=session["slot_id"],
        extra={"session_id": session["session_id"]}
    )

    save_all_state(state)
    return {"message": "Parking session released", "session": session}


@app.get("/user/{user_id}/dashboard")
def user_dashboard(user_id: str):
    state = load_all_state()
    process_live_rules(state)

    sessions = state["sessions"]
    users = state["users"]
    notifications = state["notifications"]
    fees = state["fees"]
    status_data = state["status"]

    if user_id not in users:
        users[user_id] = {
            "user_id": user_id,
            "consistency_score": 100,
            "wrong_parking_count": 0,
            "overstay_count": 0
        }
        save_all_state(state)

    user = users[user_id]
    active_session = get_active_session_for_user(sessions, user_id)

    session_view = None
    if active_session:
        slot_id = active_session["slot_id"]
        slot_data = get_slot_from_status(status_data, slot_id)
        start_time = parse_dt(active_session["start_time"])
        elapsed_seconds = int((datetime.now() - start_time).total_seconds())
        elapsed_minutes = elapsed_seconds // 60

        remaining_free_minutes = max(0, FREE_PARKING_MINUTES - elapsed_minutes)
        overdue_minutes = max(0, elapsed_minutes - FREE_PARKING_MINUTES)

        session_view = {
            "session_id": active_session["session_id"],
            "slot_id": slot_id,
            "status": active_session["status"],
            "elapsed_seconds": elapsed_seconds,
            "elapsed_minutes": elapsed_minutes,
            "remaining_free_minutes": remaining_free_minutes,
            "overdue_minutes": overdue_minutes,
            "wrong_parking": slot_data.get("wrong_parking", False) if slot_data else False,
            "slot_status": slot_data.get("status") if slot_data else None,
            "warning_near_limit": remaining_free_minutes <= WARNING_BEFORE_END_MINUTES and remaining_free_minutes > 0,
            "billing_started": elapsed_minutes > FREE_PARKING_MINUTES
        }

    user_notifications = [
        n for n in notifications
        if n.get("user_id") == user_id
    ][-50:]

    user_fees = [
        fee for fee in fees
        if fee.get("user_id") == user_id
    ][-50:]

    return {
        "user": user,
        "active_session": session_view,
        "notifications": user_notifications,
        "fees": user_fees
    }


@app.get("/user/{user_id}/notifications")
def user_notifications(user_id: str):
    state = load_all_state()
    process_live_rules(state)

    notes = [n for n in state["notifications"] if n.get("user_id") == user_id]
    return {"notifications": notes[-50:]}


# ============================================================
# EVENTS / FRAME
# ============================================================

@app.get("/events")
def get_events():
    events_data = read_json_file(EVENTS_JSON_PATH, [])
    return {"events": events_data}


@app.get("/events/recent")
def get_recent_events(limit: int = 20):
    events_data = read_json_file(EVENTS_JSON_PATH, [])
    if not isinstance(events_data, list):
        events_data = []
    return {"events": events_data[-limit:]}


@app.get("/latest-frame")
def latest_frame():
    if not os.path.exists(LATEST_FRAME_PATH):
        return JSONResponse(
            status_code=404,
            content={"message": "Latest frame not found"}
        )

    return FileResponse(LATEST_FRAME_PATH, media_type="image/jpeg")


def mjpeg_frame_generator():
    while True:
        if os.path.exists(LATEST_FRAME_PATH):
            try:
                with open(LATEST_FRAME_PATH, "rb") as f:
                    frame = f.read()

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    frame +
                    b"\r\n"
                )
            except Exception:
                pass

        time.sleep(0.1)


@app.get("/video-feed")
def video_feed():
    return StreamingResponse(
        mjpeg_frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )