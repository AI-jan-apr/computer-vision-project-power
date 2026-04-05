import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon, Point





# configuration 

BASE_DIR = os.getcwd()

IMAGE_PATH = os.path.join(BASE_DIR, "src/frames_test/img5.jpg")
SLOTS_FILE = os.path.join(BASE_DIR, "slots.json")
MODEL_PATH = "src/best.pt"   

CONF_THRESHOLD = 0.10
OVERLAP_THRESHOLD = 0.40

# load data 

frame = cv2.imread(IMAGE_PATH)
if frame is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

with open(SLOTS_FILE, "r") as f:
    slots = json.load(f)

model = YOLO(MODEL_PATH)


# FUNCTIONS: polygon creation, checking if detection intersects with ROI, computes overlap between detection and slots


def get_slot_polygon(points):
    return Polygon(points)

def bottom_center_in_polygon(x1, y1, x2, y2, polygon):
    bx = (x1 + x2) / 2
    by = y2
    return polygon.contains(Point(bx, by))

def bbox_overlap_ratio_with_slot(x1, y1, x2, y2, polygon):
    bbox_poly = Polygon([
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2)
    ])

    inter_area = polygon.intersection(bbox_poly).area
    slot_area = polygon.area

    if slot_area == 0:
        return 0.0

    return inter_area / slot_area

def detect_objects(frame):
    results = model(frame, verbose=False)[0]
    detections = []

    for box in results.boxes:
        conf = float(box.conf[0].item())
        cls_id = int(box.cls[0].item())

        if conf < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "conf": conf,
            "cls_id": cls_id
        })

    return detections

def classify_slot_with_yolo(slot_points, detections):
    polygon = get_slot_polygon(slot_points)

    best_conf = 0.0
    occupied = False

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["conf"]

        inside = bottom_center_in_polygon(x1, y1, x2, y2, polygon)
        overlap = bbox_overlap_ratio_with_slot(x1, y1, x2, y2, polygon)

        if inside or overlap > OVERLAP_THRESHOLD:
            occupied = True
            best_conf = max(best_conf, conf)

    return occupied, best_conf


# RUN YOLO

detections = detect_objects(frame)

print(f"Detected objects: {len(detections)}")

# draw raw YOLO detections
for det in detections:
    x1, y1, x2, y2 = map(int, det["bbox"])
    conf = det["conf"]

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.putText(
        frame,
        f"{conf:.2f}",
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 0),
        1
    )

# CHECK EACH SLOT

for slot in slots:
    slot_id = str(slot["slot_id"])
    pts = slot["points"]

    occupied, best_conf = classify_slot_with_yolo(pts, detections)
    prediction = "occupied" if occupied else "empty"

    print(f"Slot {slot_id}: {prediction.upper()} | conf={best_conf:.3f}")

    color = (0, 255, 0) if prediction == "empty" else (0, 0, 255)

    pts_np = np.array(pts, np.int32)
    cv2.polylines(frame, [pts_np], True, color, 2)

    x, y = pts_np[0]
    cv2.putText(
        frame,
        f"{slot_id}",
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1
    )


# results: 


cv2.imshow("YOLO ROI Test - parkingwithcar.jpg", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()