import os
import json
import cv2
from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================
# This section contains all the main project settings and file paths.
# Instead of hardcoding full absolute paths manually, we build them
# dynamically starting from the current working directory.
# This makes the code easier to move between machines as long as the
# folder structure stays the same.

BASE_DIR = os.getcwd()

# Input video:
# This is the parking lot video that will be processed frame by frame.
# We placed it under the "src" folder
VIDEO_PATH = os.path.join(BASE_DIR, "src","testvideo.mp4")

# Output video:
# This is the annotated video that the system writes after processing.
# It will contain the parking slot boxes, car boxes, labels, timers,
# wrong parking markings, and summary info drawn on each frame.
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "parking_system_output.mp4")

# Events JSON:
# This file stores important parking events discovered by the system,
# such as when a slot becomes occupied, when it becomes empty again,
# and when a parking limit is exceeded.
EVENTS_JSON_PATH = os.path.join(BASE_DIR, "src", "parking_system_events.json")

# Status JSON:
# This file acts like the "live backend state" of the parking lot.
# It stores summary data such as total slots, occupied slots,
# empty slots, wrong parking count, and per-slot status.
# FastAPI reads this file so the frontend can display the latest data.
STATUS_JSON_PATH = os.path.join(BASE_DIR, "src", "parking_status.json")

# Latest frame:
# This file stores the newest processed frame as a JPEG image.
# FastAPI uses it to serve the latest visual result to the interface,
# and it is also used by the MJPEG video feed endpoint.
LATEST_FRAME_PATH = os.path.join(BASE_DIR,"src", "latest_frame.jpg")

# Car model:
# This YOLO model is responsible for detecting cars in the parking lot.
CAR_MODEL_PATH = os.path.join(BASE_DIR, "src","best-2.pt")

# Parking slot model:
# This YOLO model is responsible for detecting parking slots.
# We use it mainly during the initialization phase to "freeze" the slots.
SLOT_MODEL_PATH = os.path.join(BASE_DIR, "src", "parking_slots_detector.pt")

# Detection confidence thresholds:
# These define the minimum confidence that YOLO detections must have
# before we accept them as valid.
# Lower threshold = more detections, but more false positives.
# Higher threshold = fewer detections, but possibly missing some objects.
SLOT_CONF_THRESHOLD = 0.15
CAR_CONF_THRESHOLD = 0.25

# Slot freezing parameters:
# The parking slot model is not run permanently for the final slot structure.
# Instead, we detect slots over an initialization period, merge repeated
# detections, and then freeze the best stable slot layout.
INIT_FRAMES_COUNT = 30
MERGE_IOU_THRESHOLD = 0.30
MIN_SEEN_FOR_STABLE_SLOT = 2

# Occupancy logic:
# This controls how we decide whether a detected car belongs to a slot.
# A car can belong to a slot if:
# - its bottom center lies inside the slot, OR
# - the IoU overlap with the slot exceeds this threshold
SLOT_IOU_THRESHOLD = 0.15

# Temporal smoothing:
# We do not trust a single frame immediately.
# A slot must repeatedly appear occupied or empty for a few frames
# before the stable state is officially changed.
# This reduces flickering and noisy state changes.
OCCUPIED_CONFIRM_FRAMES = 3
EMPTY_CONFIRM_FRAMES = 3

# Wrong parking detection parameters:
# These values control how aggressively we detect a car that is spanning
# more than one parking slot.
# The logic checks overlap with nearby slots and confirms the case only
# if the car appears to significantly occupy multiple slots.
WRONG_PARKING_MIN_OVERLAP = 0.10
WRONG_PARKING_PRIMARY_OVERLAP = 0.20
WRONG_PARKING_SECONDARY_OVERLAP = 0.18
WRONG_PARKING_SIMILARITY_RATIO = 0.80
WRONG_PARKING_CONFIRM_FRAMES = 3

# Slot box shrinking for wrong-parking detection only:
# For wrong parking, we intentionally shrink the slot boxes slightly.
# Why?
# Because full boxes may touch neighboring cars/slots too easily,
# especially when the parking lines are close together.
# Shrinking helps make the wrong parking logic stricter and cleaner.
WRONG_BOX_SHRINK_X = 0.08
WRONG_BOX_SHRINK_Y = 0.06

# Demo timer:
# This is the time limit for demonstrating the parking timer feature.
# In a real system this would likely be much higher, but for demo/testing
# we keep it small so the warning/limit logic can appear quickly.
PARKING_LIMIT_SECONDS = 5

# IMPORTANT: compute-saving update interval
# Instead of running full detection on every single frame, we only
# fully process every Nth frame after initialization.
# This reduces computation cost and still keeps the system responsive.
# Example:
# If this is 3, then full detection happens roughly every 3 frames.
PROCESS_EVERY_N_FRAMES = 3

# Visualization / output switches:
# SHOW_VIDEO controls whether OpenCV opens a live window while processing.
# SAVE_VIDEO controls whether the final annotated output video is written.
SHOW_VIDEO = True
SAVE_VIDEO = True


# ============================================================
# GEOMETRY HELPERS
# ============================================================
# This block contains geometric helper functions.
# These are used repeatedly throughout the logic for comparing boxes,
# checking spatial relationships, and deciding occupancy / wrong parking.

def iou(boxA, boxB):
    # IoU = Intersection over Union
    # This measures how much two boxes overlap relative to their total size.
    # It is a standard metric used in computer vision.
    # Value range:
    # 0.0  -> no overlap
    # 1.0  -> perfect overlap

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
    # Checks if a given point (x, y) lies inside a bounding box.
    # This is especially useful for using the bottom-center of a car
    # as a strong indicator of which slot it belongs to.
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def car_bottom_center(car_box):
    # Returns the bottom-center point of a detected car box.
    # Why bottom-center?
    # Because for parked cars, the lower central region usually aligns
    # better with the actual slot the car occupies than the full box center.
    x1, _, x2, y2 = car_box
    return (x1 + x2) / 2.0, y2


def is_car_in_slot(slot_box, car_box):
    # Main helper for deciding if a car belongs to a slot.
    # We consider the car inside the slot if:
    # 1) the bottom-center point of the car lies inside the slot, OR
    # 2) the IoU overlap is large enough
    # This hybrid approach is much more robust than using only one method.
    cx, cy = car_bottom_center(car_box)
    return point_in_box(slot_box, cx, cy) or iou(slot_box, car_box) >= SLOT_IOU_THRESHOLD


def shrink_box(box, shrink_x_ratio=0.08, shrink_y_ratio=0.06):
    # Creates a slightly smaller version of a bounding box.
    # This is used only for wrong parking detection to avoid overly
    # sensitive overlap with neighboring slots.
    # We reduce the width and height by a percentage from all sides.
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    dx = int(w * shrink_x_ratio)
    dy = int(h * shrink_y_ratio)

    nx1 = x1 + dx
    ny1 = y1 + dy
    nx2 = x2 - dx
    ny2 = y2 - dy

    # Safety check:
    # If shrinking produces an invalid box, fall back to the original box.
    if nx2 <= nx1:
        nx1, nx2 = x1, x2
    if ny2 <= ny1:
        ny1, ny2 = y1, y2

    return [nx1, ny1, nx2, ny2]


# ============================================================
# DETECTION HELPERS
# ============================================================
# These functions wrap the YOLO inference logic for slots and cars.
# They convert raw YOLO outputs into simpler Python dictionaries that
# the rest of the pipeline can use more easily.

def detect_slots(frame, slot_model):
    # Runs the slot detection model on the current frame.
    # Returns a list of detected parking slot boxes with:
    # - bbox
    # - predicted class label
    # - confidence score
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
    # Runs the car detection model on the current frame.
    # Returns detected cars with their box and confidence.
    # We do not care about multiple car classes here; the model is used
    # mainly as a generic car detector for occupancy and wrong parking logic.
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
# The parking slots should not jump around every frame.
# So we detect them for a while, merge similar boxes, and then freeze
# a stable slot layout that the rest of the system will trust.

def merge_slot(merged_slots, new_slot):
    # Tries to merge a newly detected slot into an existing merged slot list.
    # If a new slot overlaps enough with an already known slot,
    # we treat them as the same slot and update the best version of it.
    for slot in merged_slots:
        if iou(slot["bbox"], new_slot["bbox"]) >= MERGE_IOU_THRESHOLD:
            slot["seen"] += 1
            if new_slot["conf"] > slot["conf"]:
                slot["bbox"] = new_slot["bbox"]
                slot["label_from_slot_model"] = new_slot["label_from_slot_model"]
                slot["conf"] = new_slot["conf"]
            return

    # If no similar existing slot was found, add this as a new candidate slot.
    new_slot["seen"] = 1
    merged_slots.append(new_slot)


def freeze_slots(merged_slots, best_frame_slots):
    # Builds the final frozen slot list.
    # We prefer slots that were seen multiple times, because they are more stable.
    # If the merged result is too weak, we fall back to the single best frame.
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
    # Initializes per-slot runtime state.
    # Each frozen slot gets an internal state dictionary to track:
    # - stable occupied/empty state
    # - temporary candidate state
    # - candidate counter for smoothing
    # - when occupancy started
    # - whether parking limit alert was already triggered
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
    # Initializes a counter for wrong parking confirmation per slot.
    # We use this so that wrong parking must persist for a few frames
    # before being confirmed.
    return {slot["slot_id"]: 0 for slot in frozen_slots}


# ============================================================
# WRONG PARKING
# ============================================================
# This block contains the wrong parking logic.
# The idea is:
# - compare each car to nearby slots
# - see whether it strongly overlaps multiple slots
# - use bottom-center logic to avoid false positives
# - confirm only if the situation persists

def detect_wrong_parking_candidates(cars, frozen_slots):
    # Detects candidate wrong parking cases for the current frame.
    # A candidate means:
    # the car appears to significantly occupy more than one slot.
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

        # If the car clearly belongs to just one slot by bottom center,
        # we only call it wrong if the overlap split is extremely strong.
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
    # Wrong parking should not be confirmed from one noisy frame.
    # So for each slot:
    # - if it keeps appearing as wrong, increment its counter
    # - otherwise reset the counter
    # Once the counter reaches the configured threshold,
    # we consider it a confirmed wrong parking slot.
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
# These functions draw the visual overlays on the frame.
# They are only responsible for display, not logic.

def draw_label_box(frame, text, x, y, bg_color, text_color=(0, 0, 0)):
    # Draws a solid colored label background with text inside.
    # This helps make labels readable regardless of the frame content.
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
    # Draws the slot box using the chosen style:
    # - orange if wrongly parked
    # - cyan if occupied
    # - blue if empty
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
    # Draws the summary information panel in the top-left corner.
    # This provides a quick overview of:
    # - total detected cars
    # - occupied slots
    # - empty slots
    # - wrongly parked slots
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
    # Draws the detected car boxes on the frame.
    # Cars are shown separately from parking slots so the user can visually
    # understand how occupancy decisions are being made.
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
# This is one of the most important blocks.
# It updates each slot's runtime state based on current car detections.
# It handles:
# - occupancy smoothing
# - occupancy start / end events
# - time counting
# - parking limit alerts
# - wrong parking visual marking

def update_slot_state(frame, slot, cars, slot_state, frame_idx, fps, wrong_slot_ids, event_log):
    slot_id = slot["slot_id"]
    slot_box = slot["bbox"]

    raw_occupied = False
    best_iou = 0.0

    # Check all cars against this slot and determine whether the slot is occupied.
    for car in cars:
        overlap = iou(slot_box, car["bbox"])
        best_iou = max(best_iou, overlap)

        if is_car_in_slot(slot_box, car["bbox"]):
            raw_occupied = True

    raw_state = "occupied" if raw_occupied else "empty"

    # Candidate logic:
    # We do not immediately trust the raw state.
    # We count how many consecutive frames this state has appeared.
    if slot_state["candidate_state"] == raw_state:
        slot_state["candidate_count"] += 1
    else:
        slot_state["candidate_state"] = raw_state
        slot_state["candidate_count"] = 1

    # If the stable state is empty, we only switch to occupied after
    # enough repeated confirmation frames.
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

    # If the stable state is occupied, we only switch back to empty after
    # enough repeated empty confirmations.
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

    # If the slot is occupied, compute how long it has been occupied.
    # If it exceeds the configured limit, add an event and mark the slot as alerted.
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

    # Draw the slot using the final stable state.
    draw_detection_style_box(
        frame,
        slot_box,
        final_state,
        display_conf,
        is_wrong=(slot_id in wrong_slot_ids)
    )

    x1, _, _, y2 = slot_box

    # If occupied, draw the elapsed time below the slot.
    if final_state == "occupied":
        cv2.putText(frame, f"{elapsed_seconds:.1f}s", (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)

    # If wrongly parked, draw the WRONG warning.
    if slot_id in wrong_slot_ids:
        cv2.putText(frame, "WRONG", (x1, y2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

    # If parking limit exceeded, draw the LIMIT warning.
    if slot_state["alerted_limit"]:
        cv2.putText(frame, "LIMIT!", (x1, y2 + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    occupied = 1 if final_state == "occupied" else 0
    empty = 1 if final_state == "empty" else 0

    # Debug print:
    # This is useful during development to inspect how the slot behaves.
    print(
        f"Slot {slot_id}: stable={final_state.upper()} | "
        f"raw={raw_state.upper()} | best_iou={best_iou:.2f} | "
        f"timer={elapsed_seconds:.1f}s"
    )

    return occupied, empty


# ============================================================
# STATUS JSON WRITER
# ============================================================
# This block writes the live backend status to a JSON file.
# FastAPI uses this file to serve the parking state to the frontend.

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

    # For each slot, export a simplified status object that the API and UI can consume.
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
# This is the main execution loop of the full parking system.
# It handles:
# - model loading
# - video reading
# - initialization/freeze phase
# - runtime detection phase
# - drawing
# - status writing
# - latest frame saving
# - optional video saving and live display

def main():
    # Load both YOLO models:
    # - slot model for parking slot detection
    # - car model for vehicle detection
    slot_model = YOLO(SLOT_MODEL_PATH)
    car_model = YOLO(CAR_MODEL_PATH)

    # Open the input video file.
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

    # Read video properties to support timers and video writing.
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if SAVE_VIDEO:
        # If enabled, prepare the output video writer.
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

    # Cached results for skipped frames:
    # Since we do not fully process every frame, we keep the latest valid
    # processed results and reuse them on skipped frames to save computation.
    last_cars = []
    last_total_cars = 0
    last_confirmed_wrong_slot_ids = set()
    last_occupied_count = 0
    last_empty_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
         # If the video reaches the end, restart from frame 0.
         # This makes the demo run continuously in a loop.
         cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
         continue

        frame_idx += 1
        print(f"Processing frame {frame_idx}")

        # ---------------- Initialization phase ----------------
        # During the first INIT_FRAMES_COUNT frames:
        # - detect parking slots repeatedly
        # - merge stable detections
        # - remember the best slot frame
        # At the end of this phase, freeze the final slot layout.
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

            # Even during initialization, save the latest processed frame
            # so the API can still display something.
            cv2.imwrite(LATEST_FRAME_PATH, frame)

        # ---------------- Runtime phase ----------------
        # Once slots are frozen, the system switches to runtime logic:
        # - detect cars
        # - detect wrong parking
        # - update slot states
        # - draw all overlays
        # - write status JSON and latest frame
        else:
            should_process = ((frame_idx - INIT_FRAMES_COUNT) % PROCESS_EVERY_N_FRAMES == 0)

            if should_process:
                # Full detection/update pass
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

                # Cache latest processed results so skipped frames can reuse them.
                last_cars = cars
                last_total_cars = total_cars
                last_confirmed_wrong_slot_ids = confirmed_wrong_slot_ids
                last_occupied_count = occupied_count
                last_empty_count = empty_count

            else:
                # Skipped frame:
                # Do not run new detections.
                # Instead, reuse the last known processed state and simply redraw it.
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

            # Draw cars and summary panel every runtime frame.
            draw_cars(frame, cars)

            draw_summary_panel(
                frame,
                total_cars,
                occupied_count,
                empty_count,
                len(confirmed_wrong_slot_ids)
            )

            # Export the current system state for FastAPI/frontend.
            write_status_json(
                frame_idx=frame_idx,
                frozen_slots=frozen_slots,
                slot_runtime=slot_runtime,
                confirmed_wrong_slot_ids=confirmed_wrong_slot_ids,
                total_cars=total_cars,
                occupied_count=occupied_count,
                empty_count=empty_count
            )

            # Save the latest processed frame for snapshot/API streaming.
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

    # When the program finishes, dump the full event log to JSON.
    with open(EVENTS_JSON_PATH, "w") as f:
        json.dump(event_log, f, indent=4)

    print("Done.")
    print(f"Output video saved to: {OUTPUT_VIDEO_PATH}")
    print(f"Events saved to: {EVENTS_JSON_PATH}")
    print(f"Status saved to: {STATUS_JSON_PATH}")
    print(f"Latest frame saved to: {LATEST_FRAME_PATH}")


if __name__ == "__main__":
    main()