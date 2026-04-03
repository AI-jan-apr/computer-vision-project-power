import os
import cv2
from ultralytics import YOLO

# ==============================
# CONFIG
# ==============================

BASE_DIR = os.getcwd()

IMAGES_FOLDER = os.path.join(BASE_DIR, "src", "frames_sorted")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "fusion_outputs")

SLOT_MODEL_PATH = os.path.join(BASE_DIR, "src", "parking_slots_detector.pt")
CAR_MODEL_PATH = os.path.join(BASE_DIR, "src", "best-2.pt")

SHOW_IMAGES = True
SAVE_OUTPUTS = True

SLOT_CONF_THRESHOLD = 0.25
CAR_CONF_THRESHOLD = 0.10
IOU_THRESHOLD = 0.20

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==============================
# LOAD MODELS
# ==============================

slot_model = YOLO(SLOT_MODEL_PATH)
car_model = YOLO(CAR_MODEL_PATH)

# ==============================
# IOU FUNCTION
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

# ==============================
# GET IMAGE FILES
# ==============================

valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
image_files = sorted([
    f for f in os.listdir(IMAGES_FOLDER)
    if f.lower().endswith(valid_exts)
])

print(f"Found {len(image_files)} images.")

# ==============================
# MAIN LOOP
# ==============================

for img_name in image_files:
    img_path = os.path.join(IMAGES_FOLDER, img_name)
    frame = cv2.imread(img_path)

    if frame is None:
        print(f"Skipping unreadable image: {img_name}")
        continue

    print(f"\nProcessing: {img_name}")

    # --------------------------
    # SLOT DETECTOR
    # --------------------------
    slot_results = slot_model(frame, verbose=False)[0]
    slots = []

    for box in slot_results.boxes:
        conf = float(box.conf[0].item())
        if conf < SLOT_CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0].item())
        label = slot_model.names[cls_id]   # expected: "empty" or "occupied"

        slots.append({
            "bbox": [x1, y1, x2, y2],
            "label": label,
            "conf": conf
        })

    # --------------------------
    # CAR DETECTOR
    # --------------------------
    car_results = car_model(frame, verbose=False)[0]
    cars = []

    for box in car_results.boxes:
        conf = float(box.conf[0].item())
        if conf < CAR_CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cars.append({
            "bbox": [x1, y1, x2, y2],
            "conf": conf
        })

    # --------------------------
    # DRAW CAR DETECTIONS
    # --------------------------
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

    # --------------------------
    # FUSION
    # --------------------------
    occupied_count = 0
    empty_count = 0

    for slot in slots:
        slot_box = slot["bbox"]
        sx1, sy1, sx2, sy2 = slot_box

        car_found = False
        best_overlap = 0.0

        for car in cars:
            overlap = iou(slot_box, car["bbox"])
            if overlap > best_overlap:
                best_overlap = overlap

            if overlap >= IOU_THRESHOLD:
                car_found = True
                break

        # Final decision
        if car_found:
            final_state = "occupied"
        elif slot["label"].lower() == "occupied":
            final_state = "occupied"
        else:
            final_state = "empty"

        if final_state == "occupied":
            occupied_count += 1
            color = (0, 0, 255)
        else:
            empty_count += 1
            color = (0, 255, 0)

        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 2)
        cv2.putText(
            frame,
            f"{final_state} | s:{slot['conf']:.2f}",
            (sx1, max(15, sy1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1
        )

    # --------------------------
    # SUMMARY TEXT
    # --------------------------
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

    # --------------------------
    # SAVE / SHOW
    # --------------------------
    if SAVE_OUTPUTS:
        out_path = os.path.join(OUTPUT_FOLDER, img_name)
        cv2.imwrite(out_path, frame)

    if SHOW_IMAGES:
        cv2.imshow("Fusion Image Pipeline", frame)
        key = cv2.waitKey(300)  # 300 ms per image, feels like playback
        if key == ord("q"):
            break

cv2.destroyAllWindows()
print("\nDone.")