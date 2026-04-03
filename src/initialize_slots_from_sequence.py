import os
import json
import cv2
from ultralytics import YOLO

# ==============================
# CONFIG
# ==============================

BASE_DIR = os.getcwd()

FRAMES_FOLDER = os.path.join(BASE_DIR, "src", "frames_sorted")
SLOT_MODEL_PATH = os.path.join(BASE_DIR, "src", "parking_slots_detector.pt")
OUTPUT_JSON = os.path.join(BASE_DIR, "frozen_slots.json")

INIT_FRAMES_COUNT = 20
SLOT_CONF_THRESHOLD = 0.10
MERGE_IOU_THRESHOLD = 0.50
SHOW_IMAGE = True

# ==============================
# HELPERS
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
    if union == 0:
        return 0.0

    return inter_area / union


def merge_slot_detections(all_slots, new_slot, iou_threshold=0.5):
    """
    Merge a new detected slot into the existing slot list if it overlaps strongly.
    Keep the higher-confidence version and count how many times it appeared.
    """
    for slot in all_slots:
        overlap = iou(slot["bbox"], new_slot["bbox"])
        if overlap >= iou_threshold:
            slot["seen_count"] += 1

            if new_slot["conf"] > slot["conf"]:
                slot["bbox"] = new_slot["bbox"]
                slot["label_from_slot_model"] = new_slot["label_from_slot_model"]
                slot["conf"] = new_slot["conf"]

            return all_slots

    new_slot["seen_count"] = 1
    all_slots.append(new_slot)
    return all_slots


# ==============================
# LOAD MODEL
# ==============================

slot_model = YOLO(SLOT_MODEL_PATH)

# ==============================
# IMAGE LIST
# ==============================

valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
image_files = sorted([
    f for f in os.listdir(FRAMES_FOLDER)
    if f.lower().endswith(valid_exts)
])

if len(image_files) == 0:
    raise RuntimeError("No images found in frames folder.")

init_files = image_files[:INIT_FRAMES_COUNT]
print(f"Using first {len(init_files)} frames for slot initialization.")

# ==============================
# INITIALIZE SLOTS FROM FIRST N FRAMES
# ==============================

merged_slots = []

for img_name in init_files:
    img_path = os.path.join(FRAMES_FOLDER, img_name)
    frame = cv2.imread(img_path)

    if frame is None:
        print(f"Skipping unreadable frame: {img_name}")
        continue

    results = slot_model(frame, verbose=False)[0]

    for box in results.boxes:
        conf = float(box.conf[0].item())
        if conf < SLOT_CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0].item())
        label = slot_model.names[cls_id]  # expected: "empty" or "occupied"

        slot_candidate = {
            "bbox": [x1, y1, x2, y2],
            "label_from_slot_model": label,
            "conf": conf
        }

        merged_slots = merge_slot_detections(
            merged_slots,
            slot_candidate,
            iou_threshold=MERGE_IOU_THRESHOLD
        )

print(f"Unique merged slots before filtering: {len(merged_slots)}")

# ==============================
# FILTER STABLE SLOTS
# ==============================
# Keep only slots seen more than once if enough frames exist.
# If your detector is weak, you can relax this.

min_seen = 2 if len(init_files) >= 3 else 1

stable_slots = []
slot_id = 1

for slot in merged_slots:
    if slot["seen_count"] >= min_seen:
        stable_slots.append({
            "slot_id": slot_id,
            "bbox": slot["bbox"],
            "label_from_slot_model": slot["label_from_slot_model"],
            "conf": slot["conf"],
            "seen_count": slot["seen_count"]
        })
        slot_id += 1

print(f"Stable frozen slots: {len(stable_slots)}")

# ==============================
# SAVE FROZEN SLOTS
# ==============================

with open(OUTPUT_JSON, "w") as f:
    json.dump(stable_slots, f, indent=4)

print(f"Saved frozen slots to {OUTPUT_JSON}")

# ==============================
# VISUALIZE ON LAST INIT FRAME
# ==============================

last_init_frame = cv2.imread(os.path.join(FRAMES_FOLDER, init_files[-1]))

if last_init_frame is not None:
    for slot in stable_slots:
        x1, y1, x2, y2 = slot["bbox"]
        cv2.rectangle(last_init_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            last_init_frame,
            f"S{slot['slot_id']}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1
        )

    if SHOW_IMAGE:
        cv2.imshow("Frozen Slots from Sequence", last_init_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()