
import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon, Point

from timer_logic import (
    initialize_timer_state,
    update_all_timers
)

# =========================================================
# CONFIGURATION
# Define all file paths and system parameters
# =========================================================

# Base directory of the current script
BASE_DIR = os.getcwd()

# File paths
EMPTY_IMAGE = os.path.join(BASE_DIR, "..", "empty_parking.jpg")   # Reference image (empty parking lot)
IMAGE_PATH = os.path.join(BASE_DIR, "..", "src", "frames_sorted") # Input frames folder
SLOTS_FILE = os.path.join(BASE_DIR, "..", "slots.json")           # Parking slot coordinates
MODEL_PATH = os.path.join(BASE_DIR, "..", "src", "best-2.pt")     # Trained YOLO model

SHOW_IMAGE = True  # Toggle visualization

# -----------------------------
# Thresholds
# -----------------------------

# Grayscale detection thresholds
DIFF_THRESHOLD = 40                  # Pixel difference threshold
OCCUPANCY_RATIO_THRESHOLD = 0.15     # Ratio to classify as occupied

# YOLO detection thresholds
YOLO_CONF_THRESHOLD = 0.10           # Minimum confidence to keep detection
YOLO_OVERLAP_THRESHOLD = 0.40        # Overlap threshold for slot assignment

# Fusion thresholds (YOLO priority)
YOLO_STRONG = 0.20
YOLO_WEAK = 0.10
GRAY_SUPPORT = 0.12
GRAY_STRONG = 0.40


# =========================================================
# LOAD DATA
# =========================================================

print("MODEL PATH:", MODEL_PATH)
print("EXISTS:", os.path.exists(MODEL_PATH))

# Load reference empty parking image
empty_img = cv2.imread(EMPTY_IMAGE)
if empty_img is None:
    raise FileNotFoundError("Empty parking image not found")

# Load parking slot definitions
with open(SLOTS_FILE, "r") as f:
    slots = json.load(f)

# Initialize timer state for each slot
initialize_timer_state(slots)

# Load YOLO model
model = YOLO(MODEL_PATH)


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def get_ordered_images(folder):
    """
    Retrieve all image paths from a folder, sorted in order.
    Only valid image extensions are included.
    """
    valid_ext = (".jpg", ".jpeg", ".png")
    files = [f for f in os.listdir(folder) if f.lower().endswith(valid_ext)]
    files.sort()
    return [os.path.join(folder, f) for f in files]


def extract_slot_with_mask(image, points):
    """
    Extract a parking slot region using polygon masking.

    Returns:
        cropped_image: Region of interest
        cropped_mask: Binary mask for the slot
    """
    pts = np.array(points, np.int32)

    # Create mask for polygon
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(pts)

    return image[y:y+h, x:x+w], mask[y:y+h, x:x+w]


# =========================================================
# GRAYSCALE-BASED CLASSIFIER
# =========================================================

def grayscale_classifier(empty_crop, current_crop, mask_crop):
    """
    Compare current slot with empty reference using grayscale difference.

    Returns:
        occupied (bool)
        change_ratio (float)
    """
    # Convert to grayscale
    empty_gray = cv2.cvtColor(empty_crop, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_crop, cv2.COLOR_BGR2GRAY)

    # Resize to match shapes
    current_gray = cv2.resize(current_gray, (empty_gray.shape[1], empty_gray.shape[0]))
    mask_crop = cv2.resize(mask_crop, (empty_gray.shape[1], empty_gray.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply smoothing to reduce noise
    empty_gray = cv2.GaussianBlur(empty_gray, (5, 5), 0)
    current_gray = cv2.GaussianBlur(current_gray, (5, 5), 0)

    # Compute absolute difference
    diff = cv2.absdiff(empty_gray, current_gray)

    # Apply mask to focus only on slot area
    roi = diff[mask_crop > 0]

    if len(roi) == 0:
        return False, 0.0

    # Calculate change ratio
    changed = np.sum(roi > DIFF_THRESHOLD)
    ratio = changed / len(roi)

    return ratio > OCCUPANCY_RATIO_THRESHOLD, ratio


# =========================================================
# YOLO DETECTION
# =========================================================

def detect_objects(frame):
    """
    Run YOLO object detection on a frame.

    Returns:
        List of bounding boxes (x1, y1, x2, y2, confidence)
    """
    results = model(frame, verbose=False)[0]

    detections = []
    for box in results.boxes:
        conf = float(box.conf[0].item())

        # Filter low-confidence detections
        if conf < YOLO_CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        detections.append((x1, y1, x2, y2, conf))

    return detections


# =========================================================
# GEOMETRY FUNCTIONS
# =========================================================

def point_in_polygon(x1, y1, x2, y2, polygon):
    """
    Check if the bottom-center of a bounding box lies inside a polygon.
    """
    bx = (x1 + x2) / 2
    by = y2
    return polygon.contains(Point(bx, by))


def overlap_ratio(x1, y1, x2, y2, polygon):
    """
    Compute overlap ratio between bounding box and slot polygon.
    """
    box = Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
    inter = polygon.intersection(box).area
    return inter / polygon.area if polygon.area > 0 else 0


# =========================================================
# YOLO SLOT CLASSIFIER
# =========================================================

def yolo_classifier(slot_pts, detections):
    """
    Determine if a slot is occupied using YOLO detections.

    Uses:
    - Point-in-polygon (primary)
    - Overlap ratio (fallback)
    """
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


# =========================================================
# FUSION LOGIC
# =========================================================

def fuse(gray_occ, gray_ratio, yolo_occ, yolo_conf):
    """
    Combine grayscale and YOLO predictions.

    Priority:
    1. Strong YOLO
    2. Weak YOLO + grayscale support
    3. Strong grayscale fallback
    """
    if yolo_occ and yolo_conf >= YOLO_STRONG:
        return "occupied"

    if yolo_occ and yolo_conf >= YOLO_WEAK and gray_occ and gray_ratio >= GRAY_SUPPORT:
        return "occupied"

    if gray_occ and gray_ratio >= GRAY_STRONG:
        return "occupied"

    return "empty"


# =========================================================
# MAIN PIPELINE
# =========================================================

image_paths = get_ordered_images(IMAGE_PATH)[:30]

for image_path in image_paths:
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Could not read image: {image_path}")
        continue

    # Step 1: YOLO detection
    detections = detect_objects(frame)

    print(f"\nProcessing: {os.path.basename(image_path)}")
    print(f"Detected objects: {len(detections)}\n")

    # Draw YOLO bounding boxes
    for x1,y1,x2,y2,conf in detections:
        cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (255,255,0), 2)
        cv2.putText(frame, f"{conf:.2f}", (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)

    slot_results = []

    # Step 2: Process each slot
    for slot in slots:
        slot_id = str(slot["slot_id"])
        pts = slot["points"]

        empty_crop, mask_crop = extract_slot_with_mask(empty_img, pts)
        current_crop, _ = extract_slot_with_mask(frame, pts)

        # Grayscale detection
        gray_occ, gray_ratio = grayscale_classifier(
            empty_crop,
            current_crop,
            mask_crop
        )

        # YOLO detection
        yolo_occ, yolo_conf = yolo_classifier(pts, detections)

        # Fusion decision
        final = fuse(gray_occ, gray_ratio, yolo_occ, yolo_conf)

        slot_results.append({
            "slot_id": slot_id,
            "final_status": final
        })

    # Step 3: Update timers
    alerts = update_all_timers(slot_results)

    for alert in alerts:
        print(alert["message"])

    # Step 4: Visualization
    for result, slot in zip(slot_results, slots):
        slot_id = result["slot_id"]
        final = result["final_status"]
        occupied_minutes = result["occupied_minutes"]
        exceeded = result["time_exceeded"]
        pts = slot["points"]

        # Color logic
        if exceeded:
            color = (0,165,255)   # orange
        else:
            color = (0,255,0) if final == "empty" else (0,0,255)

        pts_np = np.array(pts, np.int32)
        cv2.polylines(frame, [pts_np], True, color, 2)

        x,y = pts_np[0]
        cv2.putText(frame, f"{slot_id} | {occupied_minutes}m", (x,y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Display output
    if SHOW_IMAGE:
        cv2.imshow("Fusion Output", frame)
        key = cv2.waitKey(800)
        if key == 27:
            break

cv2.destroyAllWindows()