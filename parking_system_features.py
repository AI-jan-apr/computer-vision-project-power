import os
import json
import cv2
from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.getcwd()

VIDEO_PATH = os.path.join(BASE_DIR, "testvideo.mp4")
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "parking_system_output.mp4")
EVENTS_JSON_PATH = os.path.join(BASE_DIR, "parking_system_events.json")
STATUS_JSON_PATH = os.path.join(BASE_DIR, "parking_status.json")
LATEST_FRAME_PATH = os.path.join(BASE_DIR, "latest_frame.jpg")

CAR_MODEL_PATH = os.path.join(BASE_DIR, "best-2.pt")
SLOT_MODEL_PATH = os.path.join(BASE_DIR, "parking_slots_detector.pt")

# Notebook-compatible thresholds
SLOT_CONF_THRESHOLD = 0.15
CAR_CONF_THRESHOLD = 0.25

# Better freezing
INIT_FRAMES_COUNT = 30
MERGE_IOU_THRESHOLD = 0.30
MIN_SEEN_FOR_STABLE_SLOT = 2

# Occupancy
SLOT_IOU_THRESHOLD = 0.15

# Temporal smoothing
OCCUPIED_CONFIRM_FRAMES = 3
EMPTY_CONFIRM_FRAMES = 3

# Wrong parking
WRONG_PARKING_MIN_OVERLAP = 0.10
WRONG_PARKING_PRIMARY_OVERLAP = 0.20
WRONG_PARKING_SECONDARY_OVERLAP = 0.18
WRONG_PARKING_SIMILARITY_RATIO = 0.80
WRONG_PARKING_CONFIRM_FRAMES = 3

# Shrink slot boxes slightly for wrong-parking logic only
WRONG_BOX_SHRINK_X = 0.08
WRONG_BOX_SHRINK_Y = 0.06

# Demo timer
PARKING_LIMIT_SECONDS = 5

# IMPORTANT: compute-saving update interval
PROCESS_EVERY_N_FRAMES = 3

SHOW_VIDEO = True
SAVE_VIDEO = True


# ============================================================
# GEOMETRY HELPERS
# ============================================================

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
    return 0.0 if union <= 0 else inter_area / union


def point_in_box(box, x, y):
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def car_bottom_center(car_box):
    x1, _, x2, y2 = car_box
    return (x1 + x2) / 2.0, y2


def is_car_in_slot(slot_box, car_box):
    cx, cy = car_bottom_center(car_box)
    return point_in_box(slot_box, cx, cy) or iou(slot_box, car_box) >= SLOT_IOU_THRESHOLD


def shrink_box(box, shrink_x_ratio=0.08, shrink_y_ratio=0.06):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    dx = int(w * shrink_x_ratio)
    dy = int(h * shrink_y_ratio)

    nx1 = x1 + dx
    ny1 = y1 + dy
    nx2 = x2 - dx
    ny2 = y2 - dy

    if nx2 <= nx1:
        nx1, nx2 = x1, x2
    if ny2 <= ny1:
        ny1, ny2 = y1, y2

    return [nx1, ny1, nx2, ny2]


# ============================================================
# DETECTION HELPERS
# ============================================================

def detect_slots(frame, slot_model):
    results = slot_model(frame, conf=SLOT_CONF_THRESHOLD, verbose=False)[0]
    slots = []

    for box in results.boxes:
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0].item())
        label = slot_model.names[cls_id]

        slots.append({
            "bbox": [x1, y1, x2, y2],
            "label_from_slot_model": label,
            "conf": conf
        })

    return slots


def detect_cars(frame, car_model):
    results = car_model(frame, conf=CAR_CONF_THRESHOLD, verbose=False)[0]
    cars = []

    for box in results.boxes:
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cars.append({
            "bbox": [x1, y1, x2, y2],
            "conf": conf
        })

    return cars


# ============================================================
# FREEZING HELPERS
# ============================================================

def merge_slot(merged_slots, new_slot):
    for slot in merged_slots:
        if iou(slot["bbox"], new_slot["bbox"]) >= MERGE_IOU_THRESHOLD:
            slot["seen"] += 1
            if new_slot["conf"] > slot["conf"]:
                slot["bbox"] = new_slot["bbox"]
                slot["label_from_slot_model"] = new_slot["label_from_slot_model"]
                slot["conf"] = new_slot["conf"]
            return

    new_slot["seen"] = 1
    merged_slots.append(new_slot)


def freeze_slots(merged_slots, best_frame_slots):
    frozen = []
    slot_id = 1

    stable_slots = [
        slot for slot in merged_slots
        if slot["seen"] >= MIN_SEEN_FOR_STABLE_SLOT
    ]

    source_slots = stable_slots if len(stable_slots) >= len(best_frame_slots) * 0.7 else best_frame_slots

    for slot in source_slots:
        frozen.append({
            "slot_id": slot_id,
            "bbox": slot["bbox"],
            "label_from_slot_model": slot["label_from_slot_model"],
            "conf": slot["conf"]
        })
        slot_id += 1

    return frozen


def init_slot_runtime(frozen_slots):
    return {
        slot["slot_id"]: {
            "stable_state": "empty",
            "candidate_state": None,
            "candidate_count": 0,
            "occupied_since_frame": None,
            "alerted_limit": False,
            "last_conf": slot["conf"]
        }
        for slot in frozen_slots
    }


def init_wrong_runtime(frozen_slots):
    return {slot["slot_id"]: 0 for slot in frozen_slots}


# ============================================================
# WRONG PARKING
# ============================================================

def detect_wrong_parking_candidates(cars, frozen_slots):
    wrong_events = []

    shrunk_slots = {
        slot["slot_id"]: shrink_box(
            slot["bbox"],
            WRONG_BOX_SHRINK_X,
            WRONG_BOX_SHRINK_Y
        )
        for slot in frozen_slots
    }

    for car_idx, car in enumerate(cars):
        car_box = car["bbox"]
        cx, cy = car_bottom_center(car_box)

        overlaps = []
        center_inside_slots = []

        for slot in frozen_slots:
            slot_id = slot["slot_id"]
            slot_box = shrunk_slots[slot_id]

            if point_in_box(slot_box, cx, cy):
                center_inside_slots.append(slot_id)

            overlap = iou(car_box, slot_box)
            if overlap >= WRONG_PARKING_MIN_OVERLAP:
                overlaps.append((slot_id, overlap))

        overlaps.sort(key=lambda x: x[1], reverse=True)

        if len(overlaps) < 2:
            continue

        slot1, ov1 = overlaps[0]
        slot2, ov2 = overlaps[1]

        strong_split = (
            ov1 >= WRONG_PARKING_PRIMARY_OVERLAP and
            ov2 >= WRONG_PARKING_SECONDARY_OVERLAP and
            ov2 >= ov1 * WRONG_PARKING_SIMILARITY_RATIO
        )

        if not strong_split:
            continue

        if len(center_inside_slots) == 1 and center_inside_slots[0] == slot1:
            extremely_split = (ov1 >= 0.34 and ov2 >= 0.28 and ov2 >= ov1 * 0.92)
            if not extremely_split:
                continue

        wrong_events.append({
            "car_index": car_idx,
            "overlapping_slot_ids": [slot1, slot2],
            "top_overlaps": [ov1, ov2]
        })

    return wrong_events


def confirm_wrong_slots(candidate_wrong_slot_ids, wrong_runtime):
    confirmed_wrong_slot_ids = set()

    for sid in wrong_runtime:
        if sid in candidate_wrong_slot_ids:
            wrong_runtime[sid] += 1
        else:
            wrong_runtime[sid] = 0

        if wrong_runtime[sid] >= WRONG_PARKING_CONFIRM_FRAMES:
            confirmed_wrong_slot_ids.add(sid)

    return confirmed_wrong_slot_ids


# ============================================================
# UI HELPERS
# ============================================================

def draw_label_box(frame, text, x, y, bg_color, text_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2

    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 6

    x1 = x
    y1 = max(0, y - h - 2 * pad)
    x2 = x + w + 2 * pad
    y2 = y

    cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
    cv2.putText(
        frame,
        text,
        (x1 + pad, y2 - pad),
        font,
        font_scale,
        text_color,
        thickness
    )


def draw_detection_style_box(frame, box, label, conf, is_wrong=False):
    x1, y1, x2, y2 = box

    if is_wrong:
        color = (0, 165, 255)
    elif label == "occupied":
        color = (255, 255, 0)
    else:
        color = (255, 0, 0)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    draw_label_box(frame, f"{label} {conf:.2f}", x1, y1, color, (0, 0, 0))


def draw_summary_panel(frame, total_cars, occupied_count, empty_count, wrong_count):
    overlay = frame.copy()

    x1, y1 = 10, 10
    x2, y2 = 310, 145

    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.putText(frame, f"Cars: {total_cars}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
    cv2.putText(frame, f"Occupied: {occupied_count}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.putText(frame, f"Empty: {empty_count}", (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(frame, f"Wrong parked: {wrong_count}", (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)


def draw_cars(frame, cars):
    for car in cars:
        x1, y1, x2, y2 = car["bbox"]
        conf = car["conf"]

        color = (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        draw_label_box(
            frame,
            f"car {conf:.2f}",
            x1,
            y1,
            color,
            (0, 0, 0)
        )


# ============================================================
# SLOT STATE UPDATE
# ============================================================

def update_slot_state(frame, slot, cars, slot_state, frame_idx, fps, wrong_slot_ids, event_log):
    slot_id = slot["slot_id"]
    slot_box = slot["bbox"]

    raw_occupied = False
    best_iou = 0.0

    for car in cars:
        overlap = iou(slot_box, car["bbox"])
        best_iou = max(best_iou, overlap)

        if is_car_in_slot(slot_box, car["bbox"]):
            raw_occupied = True

    raw_state = "occupied" if raw_occupied else "empty"

    if slot_state["candidate_state"] == raw_state:
        slot_state["candidate_count"] += 1
    else:
        slot_state["candidate_state"] = raw_state
        slot_state["candidate_count"] = 1

    if slot_state["stable_state"] == "empty":
        if raw_state == "occupied" and slot_state["candidate_count"] >= OCCUPIED_CONFIRM_FRAMES:
            slot_state["stable_state"] = "occupied"
            slot_state["occupied_since_frame"] = frame_idx
            slot_state["alerted_limit"] = False

            event_log.append({
                "frame": frame_idx,
                "slot_id": slot_id,
                "event": "occupied_started"
            })

    elif slot_state["stable_state"] == "occupied":
        if raw_state == "empty" and slot_state["candidate_count"] >= EMPTY_CONFIRM_FRAMES:
            duration_frames = frame_idx - slot_state["occupied_since_frame"] if slot_state["occupied_since_frame"] else 0
            duration_seconds = duration_frames / fps

            event_log.append({
                "frame": frame_idx,
                "slot_id": slot_id,
                "event": "occupied_ended",
                "duration_seconds": duration_seconds
            })

            slot_state["stable_state"] = "empty"
            slot_state["occupied_since_frame"] = None
            slot_state["alerted_limit"] = False

    final_state = slot_state["stable_state"]

    elapsed_seconds = 0.0
    if final_state == "occupied" and slot_state["occupied_since_frame"] is not None:
        elapsed_seconds = (frame_idx - slot_state["occupied_since_frame"]) / fps

        if elapsed_seconds >= PARKING_LIMIT_SECONDS and not slot_state["alerted_limit"]:
            slot_state["alerted_limit"] = True
            event_log.append({
                "frame": frame_idx,
                "slot_id": slot_id,
                "event": "parking_limit_exceeded",
                "elapsed_seconds": elapsed_seconds
            })

    display_conf = slot["conf"]

    draw_detection_style_box(
        frame,
        slot_box,
        final_state,
        display_conf,
        is_wrong=(slot_id in wrong_slot_ids)
    )

    x1, _, _, y2 = slot_box

    if final_state == "occupied":
        cv2.putText(frame, f"{elapsed_seconds:.1f}s", (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)

    if slot_id in wrong_slot_ids:
        cv2.putText(frame, "WRONG", (x1, y2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

    if slot_state["alerted_limit"]:
        cv2.putText(frame, "LIMIT!", (x1, y2 + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    occupied = 1 if final_state == "occupied" else 0
    empty = 1 if final_state == "empty" else 0

    print(
        f"Slot {slot_id}: stable={final_state.upper()} | "
        f"raw={raw_state.upper()} | best_iou={best_iou:.2f} | "
        f"timer={elapsed_seconds:.1f}s"
    )

    return occupied, empty


# ============================================================
# STATUS JSON WRITER
# ============================================================

def write_status_json(
    frame_idx,
    frozen_slots,
    slot_runtime,
    confirmed_wrong_slot_ids,
    total_cars,
    occupied_count,
    empty_count
):
    total_slots = len(frozen_slots)
    occupancy_rate = round((occupied_count / total_slots) * 100, 2) if total_slots > 0 else 0

    slots_payload = []

    for slot in frozen_slots:
        slot_id = slot["slot_id"]
        state = slot_runtime[slot_id]["stable_state"]

        slot_status = {
            "slot_id": slot_id,
            "status": state,
            "wrong_parking": slot_id in confirmed_wrong_slot_ids,
            "confidence": slot["conf"]
        }

        if slot_runtime[slot_id]["occupied_since_frame"] is not None:
            slot_status["occupied_since_frame"] = slot_runtime[slot_id]["occupied_since_frame"]
        else:
            slot_status["occupied_since_frame"] = None

        slots_payload.append(slot_status)

    status_payload = {
        "frame": frame_idx,
        "summary": {
            "total_slots": total_slots,
            "occupied_slots": occupied_count,
            "empty_slots": empty_count,
            "occupancy_rate": occupancy_rate,
            "active_alerts": len(confirmed_wrong_slot_ids),
            "wrong_parking_count": len(confirmed_wrong_slot_ids),
            "total_cars_detected": total_cars
        },
        "slots": slots_payload
    }

    with open(STATUS_JSON_PATH, "w") as f:
        json.dump(status_payload, f, indent=4)


# ============================================================
# MAIN
# ============================================================

def main():
    slot_model = YOLO(SLOT_MODEL_PATH)
    car_model = YOLO(CAR_MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    frame_idx = 0
    merged_slots = []
    frozen_slots = None
    slot_runtime = {}
    wrong_runtime = {}
    event_log = []

    best_frame_slots = []
    best_frame_slot_count = 0

    # Cached results for skipped frames
    last_cars = []
    last_total_cars = 0
    last_confirmed_wrong_slot_ids = set()
    last_occupied_count = 0
    last_empty_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
         cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
         continue

        frame_idx += 1
        print(f"Processing frame {frame_idx}")

        # ---------------- Initialization phase ----------------
        if frozen_slots is None:
            current_slots = detect_slots(frame, slot_model)

            for slot in current_slots:
                merge_slot(merged_slots, slot)

            if len(current_slots) > best_frame_slot_count:
                best_frame_slot_count = len(current_slots)
                best_frame_slots = current_slots.copy()

            cv2.putText(
                frame,
                f"Initializing slots... {frame_idx}/{INIT_FRAMES_COUNT}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            for slot in current_slots:
                draw_detection_style_box(
                    frame,
                    slot["bbox"],
                    slot["label_from_slot_model"],
                    slot["conf"]
                )

            if frame_idx >= INIT_FRAMES_COUNT:
                frozen_slots = freeze_slots(merged_slots, best_frame_slots)
                slot_runtime = init_slot_runtime(frozen_slots)
                wrong_runtime = init_wrong_runtime(frozen_slots)
                print(f"Frozen {len(frozen_slots)} slots")

            # During initialization also save latest frame for API
            cv2.imwrite(LATEST_FRAME_PATH, frame)

        # ---------------- Runtime phase ----------------
        else:
            should_process = ((frame_idx - INIT_FRAMES_COUNT) % PROCESS_EVERY_N_FRAMES == 0)

            if should_process:
                cars = detect_cars(frame, car_model)
                total_cars = len(cars)

                wrong_events = detect_wrong_parking_candidates(cars, frozen_slots)
                candidate_wrong_slot_ids = {
                    sid
                    for event in wrong_events
                    for sid in event["overlapping_slot_ids"]
                }

                confirmed_wrong_slot_ids = confirm_wrong_slots(candidate_wrong_slot_ids, wrong_runtime)

                occupied_count = 0
                empty_count = 0

                for slot in frozen_slots:
                    occ, emp = update_slot_state(
                        frame=frame,
                        slot=slot,
                        cars=cars,
                        slot_state=slot_runtime[slot["slot_id"]],
                        frame_idx=frame_idx,
                        fps=fps,
                        wrong_slot_ids=confirmed_wrong_slot_ids,
                        event_log=event_log
                    )
                    occupied_count += occ
                    empty_count += emp

                # cache latest processed results
                last_cars = cars
                last_total_cars = total_cars
                last_confirmed_wrong_slot_ids = confirmed_wrong_slot_ids
                last_occupied_count = occupied_count
                last_empty_count = empty_count

            else:
                # skipped frame: draw using last known results only
                cars = last_cars
                total_cars = last_total_cars
                confirmed_wrong_slot_ids = last_confirmed_wrong_slot_ids
                occupied_count = last_occupied_count
                empty_count = last_empty_count

                for slot in frozen_slots:
                    slot_id = slot["slot_id"]
                    stable_state = slot_runtime[slot_id]["stable_state"]
                    draw_detection_style_box(
                        frame,
                        slot["bbox"],
                        stable_state,
                        slot["conf"],
                        is_wrong=(slot_id in confirmed_wrong_slot_ids)
                    )

                    x1, _, _, y2 = slot["bbox"]

                    if stable_state == "occupied" and slot_runtime[slot_id]["occupied_since_frame"] is not None:
                        elapsed_seconds = (frame_idx - slot_runtime[slot_id]["occupied_since_frame"]) / fps
                        cv2.putText(
                            frame,
                            f"{elapsed_seconds:.1f}s",
                            (x1, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.50,
                            (0, 0, 255),
                            2
                        )

                    if slot_id in confirmed_wrong_slot_ids:
                        cv2.putText(
                            frame,
                            "WRONG",
                            (x1, y2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 165, 255),
                            2
                        )

                    if slot_runtime[slot_id]["alerted_limit"]:
                        cv2.putText(
                            frame,
                            "LIMIT!",
                            (x1, y2 + 62),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 0, 255),
                            2
                        )

            draw_cars(frame, cars)

            draw_summary_panel(
                frame,
                total_cars,
                occupied_count,
                empty_count,
                len(confirmed_wrong_slot_ids)
            )

            write_status_json(
                frame_idx=frame_idx,
                frozen_slots=frozen_slots,
                slot_runtime=slot_runtime,
                confirmed_wrong_slot_ids=confirmed_wrong_slot_ids,
                total_cars=total_cars,
                occupied_count=occupied_count,
                empty_count=empty_count
            )

            cv2.imwrite(LATEST_FRAME_PATH, frame)

        if writer is not None:
            writer.write(frame)

        if SHOW_VIDEO:
            cv2.imshow("Parking System Features Clean", frame)
            if cv2.waitKey(1) == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    with open(EVENTS_JSON_PATH, "w") as f:
        json.dump(event_log, f, indent=4)

    print("Done.")
    print(f"Output video saved to: {OUTPUT_VIDEO_PATH}")
    print(f"Events saved to: {EVENTS_JSON_PATH}")
    print(f"Status saved to: {STATUS_JSON_PATH}")
    print(f"Latest frame saved to: {LATEST_FRAME_PATH}")


if __name__ == "__main__":
    main()