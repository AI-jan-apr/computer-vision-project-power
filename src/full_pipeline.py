import os
import cv2
from ultralytics import YOLO

# ==============================
# CONFIG
# ==============================

BASE_DIR = os.getcwd()

FRAMES_FOLDER = os.path.join(BASE_DIR, "src", "frames_sorted")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "fusion_outputs")

SLOT_MODEL_PATH = os.path.join(BASE_DIR, "src", "parking_slots_detector.pt")
CAR_MODEL_PATH  = os.path.join(BASE_DIR, "src", "best-2.pt")

INIT_FRAMES_COUNT = 10
SLOT_CONF_THRESHOLD = 0.25
CAR_CONF_THRESHOLD = 0.10
IOU_THRESHOLD = 0.20
MERGE_IOU_THRESHOLD = 0.50

SHOW_IMAGES = True
SAVE_OUTPUTS = True

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==============================
# IOU FUNCTION
# ==============================

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    if areaA + areaB - inter == 0:
        return 0

    return inter / (areaA + areaB - inter)

# ==============================
# LOAD MODELS
# ==============================

slot_model = YOLO(SLOT_MODEL_PATH)
car_model  = YOLO(CAR_MODEL_PATH)

# ==============================
# LOAD IMAGES
# ==============================

valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
image_files = sorted([
    f for f in os.listdir(FRAMES_FOLDER)
    if f.lower().endswith(valid_exts)
])

print(f"Found {len(image_files)} images")

# ==============================
# INITIALIZATION STORAGE
# ==============================

merged_slots = []
frozen_slots = None

def merge_slot(all_slots, new_slot):
    for slot in all_slots:
        if iou(slot["bbox"], new_slot["bbox"]) >= MERGE_IOU_THRESHOLD:
            slot["seen"] += 1
            if new_slot["conf"] > slot["conf"]:
                slot["bbox"] = new_slot["bbox"]
                slot["conf"] = new_slot["conf"]
            return all_slots

    new_slot["seen"] = 1
    all_slots.append(new_slot)
    return all_slots

# ==============================
# MAIN LOOP
# ==============================

for i, img_name in enumerate(image_files):

    img_path = os.path.join(FRAMES_FOLDER, img_name)
    frame = cv2.imread(img_path)

    if frame is None:
        continue

    print(f"\nProcessing {img_name}")

    # ==========================
    # PHASE 1: INITIALIZATION
    # ==========================
    if frozen_slots is None:

        print("Initializing slots...")

        results = slot_model(frame, verbose=False)[0]

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < SLOT_CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            merged_slots = merge_slot(merged_slots, {
                "bbox": [x1,y1,x2,y2],
                "conf": conf
            })

        # Freeze after N frames
        if i >= INIT_FRAMES_COUNT:
            frozen_slots = [
                {"slot_id": idx+1, "bbox": s["bbox"]}
                for idx, s in enumerate(merged_slots)
                if s["seen"] >= 2
            ]

            print(f"Frozen {len(frozen_slots)} slots")

        continue

    # ==========================
    # PHASE 2: RUNTIME
    # ==========================
    cars = []

    car_results = car_model(frame, verbose=False)[0]

    for box in car_results.boxes:
        conf = float(box.conf[0])
        if conf < CAR_CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cars.append({"bbox":[x1,y1,x2,y2]})

    # draw cars
    for car in cars:
        x1,y1,x2,y2 = car["bbox"]
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),2)

    # ==========================
    # SLOT OCCUPANCY + COUNTERS
    # ==========================
    occupied_count = 0
    empty_count = 0

    for slot in frozen_slots:
        sx1,sy1,sx2,sy2 = slot["bbox"]

        occupied = False

        for car in cars:
            if iou(slot["bbox"], car["bbox"]) > IOU_THRESHOLD:
                occupied = True
                break

        if occupied:
            occupied_count += 1
            color = (0,0,255)
            label = "occupied"
        else:
            empty_count += 1
            color = (0,255,0)
            label = "empty"

        cv2.rectangle(frame,(sx1,sy1),(sx2,sy2),color,2)
        cv2.putText(frame,label,(sx1,sy1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

    # ==========================
    # DRAW COUNTERS
    # ==========================
    cv2.putText(frame,
                f"Occupied: {occupied_count}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,0,255),
                2)

    cv2.putText(frame,
                f"Empty: {empty_count}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2)

    # ==========================
    # SHOW / SAVE
    # ==========================
    if SAVE_OUTPUTS:
        cv2.imwrite(os.path.join(OUTPUT_FOLDER,img_name),frame)

    if SHOW_IMAGES:
        cv2.imshow("Final Pipeline",frame)
        if cv2.waitKey(200) == ord('q'):
            break

cv2.destroyAllWindows()