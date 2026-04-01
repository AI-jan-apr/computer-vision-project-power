import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon, Point


# CONFIGURATION
# we define all the files paths and system parameters here

BASE_DIR = os.getcwd()

EMPTY_IMAGE = os.path.join(BASE_DIR, "empty_parking.jpg") # empty parking lot reference 
IMAGE_PATH = os.path.join(BASE_DIR, "src", "frames_test", "parkingwithcar.jpg") # test image 
SLOTS_FILE = os.path.join(BASE_DIR, "slots.json") # roi coordinates 
MODEL_PATH = os.path.join(BASE_DIR, "src", "best.pt") # yolo model 
SHOW_IMAGE = True

# threshholds for all models 
# GRAYSCALE SETTINGS 
DIFF_THRESHOLD = 40
OCCUPANCY_RATIO_THRESHOLD = 0.15

# YOLO SETTINGS 
YOLO_CONF_THRESHOLD = 0.10
YOLO_OVERLAP_THRESHOLD = 0.40

# FUSION SETTINGS (YOLO priority)
YOLO_STRONG = 0.20
YOLO_WEAK = 0.10
GRAY_SUPPORT = 0.12
GRAY_STRONG = 0.40


# LOAD DATA

print("MODEL PATH:", MODEL_PATH)
print("EXISTS:", os.path.exists(MODEL_PATH))

empty_img = cv2.imread(EMPTY_IMAGE)
frame = cv2.imread(IMAGE_PATH)

if empty_img is None:
    raise FileNotFoundError("Empty parking image not found")

if frame is None:
    raise FileNotFoundError("Test image not found")

with open(SLOTS_FILE, "r") as f:
    slots = json.load(f)

model = YOLO(MODEL_PATH)

# FUNCTIONS

# this is the ROI extraction function 
def extract_slot_with_mask(image, points):
    pts = np.array(points, np.int32)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    x, y, w, h = cv2.boundingRect(pts)

    return image[y:y+h, x:x+w], mask[y:y+h, x:x+w]

# this is the grayscale classifier 

def grayscale_classifier(empty_crop, current_crop, mask_crop):
    empty_gray = cv2.cvtColor(empty_crop, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_crop, cv2.COLOR_BGR2GRAY)

    current_gray = cv2.resize(current_gray, (empty_gray.shape[1], empty_gray.shape[0]))
    mask_crop = cv2.resize(mask_crop, (empty_gray.shape[1], empty_gray.shape[0]), interpolation=cv2.INTER_NEAREST)

    empty_gray = cv2.GaussianBlur(empty_gray, (5, 5), 0)
    current_gray = cv2.GaussianBlur(current_gray, (5, 5), 0)

    diff = cv2.absdiff(empty_gray, current_gray)
    roi = diff[mask_crop > 0]

    if len(roi) == 0:
        return False, 0.0

    changed = np.sum(roi > DIFF_THRESHOLD)
    ratio = changed / len(roi)

    return ratio > OCCUPANCY_RATIO_THRESHOLD, ratio

# this is the yolo detection function 

def detect_objects(frame):
    results = model(frame, verbose=False)[0]

    detections = []
    for box in results.boxes:
        conf = float(box.conf[0].item())

        if conf < YOLO_CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        detections.append((x1, y1, x2, y2, conf))

    return detections

# geometry function, this is to map the detection to the parking slots, using main and fallback checks 
def point_in_polygon(x1, y1, x2, y2, polygon):
    bx = (x1 + x2) / 2
    by = y2
    return polygon.contains(Point(bx, by)) # main - bottom center point 


def overlap_ratio(x1, y1, x2, y2, polygon):
    box = Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
    inter = polygon.intersection(box).area
    return inter / polygon.area if polygon.area > 0 else 0 # fallback - overlap ratio 

# YOLO slot classifier - to determine if a slot is occupied using YOLO, 
# checks if detection is inside the slot, or over laps with slot, then returns confidence and occupation 
def yolo_classifier(slot_pts, detections):
    poly = Polygon(slot_pts)

    best_conf = 0
    occupied = False

    for x1,y1,x2,y2,conf in detections:
        inside = point_in_polygon(x1,y1,x2,y2,poly)
        overlap = overlap_ratio(x1,y1,x2,y2,poly)

        if inside or overlap > YOLO_OVERLAP_THRESHOLD:
            occupied = True
            best_conf = max(best_conf, conf)

    return occupied, best_conf

# fusion function, 
# combines between grayscale and yolo decisions, however yolo has a higher priority 
# grayscale only supports or acts as fallback 
# this is to improve robustness when lightings and shadows occur 

def fuse(gray_occ, gray_ratio, yolo_occ, yolo_conf):
    # YOLO strong wins
    if yolo_occ and yolo_conf >= YOLO_STRONG:
        return "occupied"

    # YOLO weak + grayscale support
    if yolo_occ and yolo_conf >= YOLO_WEAK and gray_occ and gray_ratio >= GRAY_SUPPORT:
        return "occupied"

    # grayscale only if VERY strong
    if gray_occ and gray_ratio >= GRAY_STRONG:
        return "occupied"

    return "empty"



# RUN THE FULL PIPELINE
# 1 - detect vehicles using yolo 
# 2 - draw bounding boxes 
# 3 - for each slot, grayscale, yolo and fusion to get final decision

detections = detect_objects(frame)

print(f"\nDetected objects: {len(detections)}\n")

# draw YOLO boxes
for x1,y1,x2,y2,conf in detections:
    cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (255,255,0), 2)
    cv2.putText(frame, f"{conf:.2f}", (int(x1), int(y1)-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)

# process slots
for slot in slots:
    slot_id = str(slot["slot_id"])
    pts = slot["points"]

    empty_crop, mask_crop = extract_slot_with_mask(empty_img, pts)
    current_crop, _ = extract_slot_with_mask(frame, pts)

  
    gray_occ, gray_ratio = grayscale_classifier(
        empty_crop,
        current_crop,
        mask_crop
    )

    yolo_occ, yolo_conf = yolo_classifier(pts, detections)

    final = fuse(gray_occ, gray_ratio, yolo_occ, yolo_conf)

    print(f"Slot {slot_id} | G:{gray_ratio:.2f} | Y:{yolo_conf:.2f} | FINAL:{final}")

    color = (0,255,0) if final == "empty" else (0,0,255)

    pts_np = np.array(pts, np.int32)
    cv2.polylines(frame, [pts_np], True, color, 2)

    x,y = pts_np[0]
    cv2.putText(frame, slot_id, (x,y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)



# DISPLAY


if SHOW_IMAGE:
    cv2.imshow("Fusion Output", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()