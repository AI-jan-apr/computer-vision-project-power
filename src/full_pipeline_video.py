import os
import cv2
import json
from ultralytics import YOLO

# ==============================
# CONFIG
# ==============================

BASE_DIR = os.getcwd()

VIDEO_PATH = os.path.join(BASE_DIR, "src", "testvideo.mp4")
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "parking_output_bigparking.mp4")
EVENTS_JSON_PATH = os.path.join(BASE_DIR, "parking_events.json")

SLOT_MODEL_PATH = os.path.join(BASE_DIR, "src", "parking_slots_detector.pt")
CAR_MODEL_PATH = os.path.join(BASE_DIR, "src", "best-2.pt")

# -------- Initialization --------
INIT_FRAMES_COUNT = 20
SLOT_CONF_THRESHOLD = 0.15
MERGE_IOU_THRESHOLD = 0.40
MIN_SEEN_FOR_STABLE_SLOT = 2

# -------- Car detection --------
CAR_CONF_THRESHOLD = 0.10

# Occupancy logic:
# A slot is occupied if:
# 1) car bottom-center is inside slot
# OR
# 2) IoU overlap with slot is large enough
SLOT_IOU_THRESHOLD = 0.15

# Wrong parking:
# if one car overlaps 2 or more slots enough, it is wrong parking
WRONG_PARKING_IOU_THRESHOLD = 0.30
WRONG_PARKING_MIN_SLOTS = 2

# -------- Temporal smoothing --------
OCCUPIED_CONFIRM_FRAMES = 3
EMPTY_CONFIRM_FRAMES = 3

# -------- Parking limit --------
# For your 30-second demo, use something small like 5 seconds.
# Later change to: 2 * 60 * 60
PARKING_LIMIT_SECONDS = 5

SHOW_VIDEO = True
SAVE_VIDEO = True

# ==============================
# GEOMETRY HELPERS
# ==============================

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    union = areaA + areaB - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


def point_in_box(box, x, y):
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def car_bottom_center(car_box):
    x1, y1, x2, y2 = car_box
    cx = (x1 + x2) / 2.0
    cy = y2
    return cx, cy


# ==============================
# INITIALIZATION HELPERS
# ==============================

def merge_slot(all_slots, new_slot, merge_iou_threshold=0.40):
    """
    Merge repeated slot detections across initialization frames.
    """
    for slot in all_slots:
        overlap = iou(slot["bbox"], new_slot["bbox"])
        if overlap >= merge_iou_threshold:
            slot["seen"] += 1

            # keep higher-confidence box
            if new_slot["conf"] > slot["conf"]:
                slot["bbox"] = new_slot["bbox"]
                slot["conf"] = new_slot["conf"]

            return all_slots

    new_slot["seen"] = 1
    all_slots.append(new_slot)
    return all_slots


# ==============================
# CAR / SLOT ASSOCIATION
# ==============================

def is_car_in_slot(slot_box, car_box, iou_threshold=0.15):
    """
    Stronger occupancy rule than IoU alone:
    - bottom-center inside slot OR
    - IoU overlap exceeds threshold
    """
    cx, cy = car_bottom_center(car_box)

    if point_in_box(slot_box, cx, cy):
        return True

    if iou(slot_box, car_box) >= iou_threshold:
        return True

    return False


def car_bottom_center(car_box):
    x1, y1, x2, y2 = car_box
    return (x1 + x2) / 2.0, y2


def point_in_box(box, x, y):
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def detect_wrong_parking(cars, frozen_slots, overlap_threshold=0.30, min_slots=2):
    """
    Stricter wrong parking:
    - car must overlap multiple slots significantly
    - and its bottom-center should not belong clearly to just one slot
    """
    wrong_events = []

    for car_idx, car in enumerate(cars):
        car_box = car["bbox"]
        cx, cy = car_bottom_center(car_box)

        overlapping_slots = []
        center_inside_slots = []

        for slot in frozen_slots:
            slot_box = slot["bbox"]

            if point_in_box(slot_box, cx, cy):
                center_inside_slots.append(slot["slot_id"])

            overlap = iou(car_box, slot_box)
            if overlap >= overlap_threshold:
                overlapping_slots.append(slot["slot_id"])

        # wrong only if it significantly overlaps 2+ slots
        # AND does not clearly belong to one slot by center
        if len(overlapping_slots) >= min_slots and len(center_inside_slots) != 1:
            wrong_events.append({
                "car_index": car_idx,
                "overlapping_slot_ids": overlapping_slots
            })

    return wrong_events


# ==============================
# MAIN
# ==============================

def main():
    os.makedirs(os.path.join(BASE_DIR, "fusion_outputs"), exist_ok=True)

    slot_model = YOLO(SLOT_MODEL_PATH)
    car_model = YOLO(CAR_MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    frame_idx = 0
    merged_slots = []
    frozen_slots = None

    # Per-slot runtime state
    # Each slot stores:
    # - stable_state
    # - candidate_state
    # - candidate_count
    # - occupied_since_frame
    # - alerted_limit
    slot_runtime = {}

    event_log = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        print(f"Processing frame {frame_idx}")

        # =================================
        # PHASE 1: SLOT INITIALIZATION
        # =================================
        if frozen_slots is None:
            results = slot_model(frame, verbose=False)[0]

            for box in results.boxes:
                conf = float(box.conf[0].item())
                if conf < SLOT_CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                merged_slots = merge_slot(
                    merged_slots,
                    {"bbox": [x1, y1, x2, y2], "conf": conf},
                    merge_iou_threshold=MERGE_IOU_THRESHOLD
                )

            cv2.putText(
                frame,
                f"Initializing slots... {frame_idx}/{INIT_FRAMES_COUNT}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            # show temporary merged slots if any
            for idx, slot in enumerate(merged_slots, start=1):
                x1, y1, x2, y2 = slot["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    f"S{idx}",
                    (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1
                )

            if frame_idx >= INIT_FRAMES_COUNT:
                frozen_slots = []
                slot_id = 1
                for slot in merged_slots:
                    if slot["seen"] >= MIN_SEEN_FOR_STABLE_SLOT:
                        frozen_slots.append({
                            "slot_id": slot_id,
                            "bbox": slot["bbox"],
                            "conf": slot["conf"],
                            "seen": slot["seen"]
                        })
                        slot_runtime[slot_id] = {
                            "stable_state": "empty",
                            "candidate_state": None,
                            "candidate_count": 0,
                            "occupied_since_frame": None,
                            "alerted_limit": False
                        }
                        slot_id += 1

                print(f"Frozen {len(frozen_slots)} slots")

            if writer is not None:
                writer.write(frame)

            if SHOW_VIDEO:
                cv2.imshow("Parking Pipeline Advanced", frame)
                if cv2.waitKey(1) == ord("q"):
                    break

            continue

        # =================================
        # PHASE 2: CAR DETECTION
        # =================================
        cars = []
        car_results = car_model(frame, verbose=False)[0]

        for box in car_results.boxes:
            conf = float(box.conf[0].item())
            if conf < CAR_CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cars.append({
                "bbox": [x1, y1, x2, y2],
                "conf": conf
            })

        # Draw car detections
        for idx, car in enumerate(cars):
            x1, y1, x2, y2 = car["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(
                frame,
                f"car {idx} {car['conf']:.2f}",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 0),
                1
            )

        # =================================
        # WRONG PARKING DETECTION
        # =================================
        wrong_events = detect_wrong_parking(
            cars,
            frozen_slots,
            overlap_threshold=WRONG_PARKING_IOU_THRESHOLD,
            min_slots=WRONG_PARKING_MIN_SLOTS
        )

        wrong_slot_ids = set()
        for event in wrong_events:
            for sid in event["overlapping_slot_ids"]:
                wrong_slot_ids.add(sid)

        # =================================
        # SLOT OCCUPANCY + TEMPORAL SMOOTHING + TIMER
        # =================================
        occupied_count = 0
        empty_count = 0

        for slot in frozen_slots:
            slot_id = slot["slot_id"]
            slot_box = slot["bbox"]
            sx1, sy1, sx2, sy2 = slot_box

            # Raw occupancy from current frame
            raw_occupied = False
            best_iou = 0.0

            for car in cars:
                overlap = iou(slot_box, car["bbox"])
                if overlap > best_iou:
                    best_iou = overlap

                if is_car_in_slot(slot_box, car["bbox"], iou_threshold=SLOT_IOU_THRESHOLD):
                    raw_occupied = True

            raw_state = "occupied" if raw_occupied else "empty"

            # Temporal smoothing
            state = slot_runtime[slot_id]

            if state["candidate_state"] == raw_state:
                state["candidate_count"] += 1
            else:
                state["candidate_state"] = raw_state
                state["candidate_count"] = 1

            # Confirm transitions
            if state["stable_state"] == "empty":
                if raw_state == "occupied" and state["candidate_count"] >= OCCUPIED_CONFIRM_FRAMES:
                    state["stable_state"] = "occupied"
                    state["occupied_since_frame"] = frame_idx
                    state["alerted_limit"] = False
                    event_log.append({
                        "frame": frame_idx,
                        "slot_id": slot_id,
                        "event": "occupied_started"
                    })

            elif state["stable_state"] == "occupied":
                if raw_state == "empty" and state["candidate_count"] >= EMPTY_CONFIRM_FRAMES:
                    duration_frames = frame_idx - state["occupied_since_frame"] if state["occupied_since_frame"] else 0
                    duration_seconds = duration_frames / fps

                    event_log.append({
                        "frame": frame_idx,
                        "slot_id": slot_id,
                        "event": "occupied_ended",
                        "duration_seconds": duration_seconds
                    })

                    state["stable_state"] = "empty"
                    state["occupied_since_frame"] = None
                    state["alerted_limit"] = False

            final_state = state["stable_state"]

            # Timer / limit check
            elapsed_seconds = 0.0
            if final_state == "occupied" and state["occupied_since_frame"] is not None:
                elapsed_seconds = (frame_idx - state["occupied_since_frame"]) / fps

                if elapsed_seconds >= PARKING_LIMIT_SECONDS and not state["alerted_limit"]:
                    state["alerted_limit"] = True
                    event_log.append({
                        "frame": frame_idx,
                        "slot_id": slot_id,
                        "event": "parking_limit_exceeded",
                        "elapsed_seconds": elapsed_seconds
                    })

            # Count
            if final_state == "occupied":
                occupied_count += 1
                color = (0, 0, 255)
            else:
                empty_count += 1
                color = (0, 255, 0)

            # Wrong parking overrides color to orange-ish
            if slot_id in wrong_slot_ids:
                color = (0, 165, 255)

            # Draw slot
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 2)

            label = f"S{slot_id} {final_state}"
            cv2.putText(
                frame,
                label,
                (sx1, max(15, sy1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                color,
                1
            )

            # Draw timer if occupied
            if final_state == "occupied":
                cv2.putText(
                    frame,
                    f"{elapsed_seconds:.1f}s",
                    (sx1, min(height - 10, sy2 + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    color,
                    1
                )

            # Draw wrong parking warning
            if slot_id in wrong_slot_ids:
                cv2.putText(
                    frame,
                    "WRONG",
                    (sx1, min(height - 25, sy2 + 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 165, 255),
                    2
                )

            # Draw limit exceeded warning
            if state["alerted_limit"]:
                cv2.putText(
                    frame,
                    "LIMIT!",
                    (sx1, min(height - 40, sy2 + 45)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    2
                )

            print(
                f"Slot {slot_id}: stable={final_state.upper()} | "
                f"raw={raw_state.upper()} | best_iou={best_iou:.2f} | "
                f"timer={elapsed_seconds:.1f}s"
            )

        # =================================
        # DRAW COUNTERS / STATUS
        # =================================
        cv2.putText(
            frame,
            f"Occupied: {occupied_count}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Empty: {empty_count}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Wrong parked cars: {len(wrong_events)}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 165, 255),
            2
        )

        # =================================
        # SAVE / SHOW
        # =================================
        if writer is not None:
            writer.write(frame)

        if SHOW_VIDEO:
            cv2.imshow("Parking Pipeline Advanced", frame)
            if cv2.waitKey(1) == ord("q"):
                break

    # Cleanup
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # Save events log
    with open(EVENTS_JSON_PATH, "w") as f:
        json.dump(event_log, f, indent=4)

    print("Done.")
    print(f"Output video saved to: {OUTPUT_VIDEO_PATH}")
    print(f"Events saved to: {EVENTS_JSON_PATH}")


if __name__ == "__main__":
    main()