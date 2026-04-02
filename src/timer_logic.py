# timer_logic.py

FRAME_TO_MINUTES = 15       # project-defined timing
TIME_LIMIT_HOURS = 2
TIME_LIMIT_FRAMES = TIME_LIMIT_HOURS * 60 // FRAME_TO_MINUTES
EMPTY_TOLERANCE = 3        # allow a few empty frames before reset

parking_timer_state = {}


def initialize_timer_state(slots):
    global parking_timer_state
    parking_timer_state = {
        str(slot["slot_id"]): {
            "status": "empty",
            "occupied_frames": 0,
            "empty_streak": 0,
            "alert_sent": False
        }
        for slot in slots
    }


def update_all_timers(slot_results):
    alerts = []

    for result in slot_results:
        slot_id = str(result["slot_id"])
        final_status = result["final_status"]
        slot_state = parking_timer_state[slot_id]

        if final_status == "occupied":
            slot_state["status"] = "occupied"
            slot_state["occupied_frames"] += 1
            slot_state["empty_streak"] = 0

            if (
                slot_state["occupied_frames"] >= TIME_LIMIT_FRAMES
                and not slot_state["alert_sent"]
            ):
                slot_state["alert_sent"] = True
                occupied_minutes = slot_state["occupied_frames"] * FRAME_TO_MINUTES
                alerts.append(
                    {
                        "slot_id": slot_id,
                        "message": f"Admin Alert: Slot {slot_id} exceeded the limit ({occupied_minutes} minutes)."
                    }
                )

        else:
            if slot_state["status"] == "occupied":
                slot_state["empty_streak"] += 1

                if slot_state["empty_streak"] >= EMPTY_TOLERANCE:
                    slot_state["status"] = "empty"
                    slot_state["occupied_frames"] = 0
                    slot_state["empty_streak"] = 0
                    slot_state["alert_sent"] = False
            else:
                slot_state["status"] = "empty"

        result["occupied_minutes"] = slot_state["occupied_frames"] * FRAME_TO_MINUTES
        result["time_exceeded"] = slot_state["alert_sent"]

    return alerts