# =========================================================
# TIMER CONFIGURATION
# These settings control how parking duration is tracked
# =========================================================

FRAME_TO_MINUTES = 15
# Each processed frame is treated as 15 minutes in project time

TIME_LIMIT_HOURS = 2
# Maximum allowed parking duration before triggering an alert

TIME_LIMIT_FRAMES = TIME_LIMIT_HOURS * 60 // FRAME_TO_MINUTES
# Convert time limit from hours to equivalent number of frames

EMPTY_TOLERANCE = 3
# Number of consecutive empty frames required before resetting a slot
# This helps avoid immediate resets caused by temporary detection errors


# =========================================================
# GLOBAL TIMER STATE
# Stores the tracking information for each parking slot
# =========================================================

parking_timer_state = {}


def initialize_timer_state(slots):
    """
    Initialize timer tracking for all parking slots.

    Each slot starts as empty, with:
    - 0 occupied frames
    - 0 empty streak
    - no alert sent
    """
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
    """
    Update timer state for all parking slots based on their final status.

    Args:
        slot_results (list): List of slot results from the main detection pipeline.
                             Each result must contain:
                             - slot_id
                             - final_status ("occupied" or "empty")

    Returns:
        alerts (list): A list of alert messages for slots that exceeded the time limit.

    Notes:
        - Occupied slots increase their occupied frame count.
        - Empty slots are only reset after EMPTY_TOLERANCE consecutive empty frames.
        - This tolerance prevents noisy predictions from instantly resetting the timer.
    """
    alerts = []

    for result in slot_results:
        slot_id = str(result["slot_id"])
        final_status = result["final_status"]
        slot_state = parking_timer_state[slot_id]

        # -------------------------------------------------
        # Case 1: Slot is occupied
        # -------------------------------------------------
        if final_status == "occupied":
            slot_state["status"] = "occupied"
            slot_state["occupied_frames"] += 1
            slot_state["empty_streak"] = 0

            # Trigger alert once if parking time exceeds the limit
            if (
                slot_state["occupied_frames"] >= TIME_LIMIT_FRAMES
                and not slot_state["alert_sent"]
            ):
                slot_state["alert_sent"] = True
                occupied_minutes = slot_state["occupied_frames"] * FRAME_TO_MINUTES

                alerts.append(
                    {
                        "slot_id": slot_id,
                        "message": (
                            f"Admin Alert: Slot {slot_id} exceeded the limit "
                            f"({occupied_minutes} minutes)."
                        )
                    }
                )

        # -------------------------------------------------
        # Case 2: Slot is predicted empty
        # -------------------------------------------------
        else:
            # If the slot was previously occupied, start counting empty frames
            if slot_state["status"] == "occupied":
                slot_state["empty_streak"] += 1

                # Reset timer only after enough consecutive empty frames
                if slot_state["empty_streak"] >= EMPTY_TOLERANCE:
                    slot_state["status"] = "empty"
                    slot_state["occupied_frames"] = 0
                    slot_state["empty_streak"] = 0
                    slot_state["alert_sent"] = False

            else:
                # Keep slot as empty if it was already empty
                slot_state["status"] = "empty"

        # -------------------------------------------------
        # Add timing information back into the current result
        # -------------------------------------------------
        result["occupied_minutes"] = slot_state["occupied_frames"] * FRAME_TO_MINUTES
        result["time_exceeded"] = slot_state["alert_sent"]

    return alerts