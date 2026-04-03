import os
import json
import cv2
from ultralytics import YOLO

# ==============================
# CONFIG
# ==============================

BASE_DIR = os.getcwd()

FRAMES_FOLDER = os.path.join(BASE_DIR, "src", "frames_sorted")
CAR_MODEL_PATH = os.path.join(BASE_DIR, "src", "best-2.pt")
FROZEN_SLOTS_PATH = os.path.join(BASE_DIR, "frozen_slots.json")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "frozen_outputs")

INIT_FRAMES_COUNT = 10
CAR_CONF_THRESHOLD = 0.10
IOU_THRESHOLD = 0.20

SHOW_IMAGES = True
SAVE_OUTPUTS = True

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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


def detect_cars(frame, car_model, conf_threshold=0.10):
    results = car_model(frame, verbose=False)[0]
    cars = []

    for box in results.boxes:
        conf = float(box.conf[0].item())
        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cars.append({
            "bbox": [x1, y1, x2, y2],
            "conf": conf
        })

    return cars

# ==============================
# LOAD
# ==============================

with open(FROZEN_SLOTS_PATH, "r") as f:
    frozen_slots = json.load(f)

car_model = YOLO(CAR_MODEL_PATH)

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

runtime_files = image_files[INIT_FRAMES_COUNT:]
print(f"Using {len(runtime_files)} frames for runtime occupancy.")

# ==============================
# MAIN LOOP
# ==============================

for img_name in runtime_files:
    img_path = os.path.join(FRAMES_FOLDER, img_name)
    frame = cv2.imread(img_path)

    if frame is None:
        print(f"Skipping unreadable frame: {img_name}")
        continue

    print(f"\nProcessing: {img_name}")

    cars = detect_cars(frame, car_model, conf_threshold=CAR_CONF_THRESHOLD)

    # draw detected cars in cyan
    for car in cars:
        x1, y1, x2, y2 = car["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(
            frame,
            f"car {car['conf']:.2f}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1
        )

    occupied_count = 0
    empty_count = 0

    for slot in frozen_slots:
        sx1, sy1, sx2, sy2 = slot["bbox"]
        slot_box = [sx1, sy1, sx2, sy2]

        occupied = False
        best_iou = 0.0

        for car in cars:
            overlap = iou(slot_box, car["bbox"])
            if overlap > best_iou:
                best_iou = overlap

            if overlap >= IOU_THRESHOLD:
                occupied = True

        final_state = "occupied" if occupied else "empty"

        if final_state == "occupied":
            occupied_count += 1
            color = (0, 0, 255)
        else:
            empty_count += 1
            color = (0, 255, 0)

        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 2)
        cv2.putText(
            frame,
            f"S{slot['slot_id']} {final_state}",
            (sx1, max(15, sy1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1
        )

        print(f"Slot {slot['slot_id']}: {final_state.upper()} | best_iou={best_iou:.2f}")

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

    if SAVE_OUTPUTS:
        out_path = os.path.join(OUTPUT_FOLDER, img_name)
        cv2.imwrite(out_path, frame)

    if SHOW_IMAGES:
        cv2.imshow("Frozen Slot Occupancy - Sequence", frame)
        key = cv2.waitKey(300)
        if key == ord("q"):
            break

cv2.destroyAllWindows()
print("\nDone.")