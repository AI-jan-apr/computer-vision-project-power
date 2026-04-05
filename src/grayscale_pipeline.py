import cv2
import json
import numpy as np
import os

<<<<<<< HEAD
# ==============================
# CONFIG
# ==============================

BASE_DIR = os.getcwd()

EMPTY_IMAGE = os.path.join(BASE_DIR, "empty_parking.jpg")
FRAMES_FOLDER = os.path.join(BASE_DIR, "src", "frames_test")
SLOTS_FILE = os.path.join(BASE_DIR, "slots.json")

SHOW_IMAGES = True

# grayscale occupancy settings
DIFF_THRESHOLD = 40
OCCUPANCY_RATIO_THRESHOLD = 0.15

# ==============================
# LOAD DATA
# ==============================

=======

# =========================================================
# CONFIGURATION
# Define file paths and grayscale detection thresholds
# =========================================================

BASE_DIR = os.getcwd()

# File paths
EMPTY_IMAGE = os.path.join(BASE_DIR, "empty_parking.jpg")          # Reference empty parking image
FRAMES_FOLDER = os.path.join(BASE_DIR, "src", "frames_test")       # Input test images
SLOTS_FILE = os.path.join(BASE_DIR, "slots.json")                  # Parking slot coordinates

SHOW_IMAGES = True  # Toggle visualization

# -----------------------------
# Grayscale Detection Thresholds
# -----------------------------
DIFF_THRESHOLD = 40                  # Minimum pixel difference to count as change
OCCUPANCY_RATIO_THRESHOLD = 0.15     # % of changed pixels to classify as occupied


# =========================================================
# LOAD DATA
# =========================================================

# Load empty parking reference image
>>>>>>> test
empty_img = cv2.imread(EMPTY_IMAGE)
print("EMPTY_IMAGE PATH:", EMPTY_IMAGE)
print("empty_img loaded:", empty_img is not None)

if empty_img is None:
    raise FileNotFoundError(f"Could not load empty image: {EMPTY_IMAGE}")

<<<<<<< HEAD
with open(SLOTS_FILE) as f:
    slots = json.load(f)

# ==============================
# GROUND TRUTH
# ==============================

ground_truth = {
    "img1.jpg": {"3": "occupied", "36": "occupied", "11": "empty", "40": "occupied", "78": "occupied"},
=======
# Load slot definitions
with open(SLOTS_FILE) as f:
    slots = json.load(f)


# =========================================================
# GROUND TRUTH (EVALUATION)
# Manually labeled slot states used to evaluate performance
# =========================================================

ground_truth = {
    "img1.jpg": {"60": "occupied", "36": "occupied", "11": "empty", "40": "occupied", "78": "occupied"},
>>>>>>> test
    "img2.jpg": {"13": "occupied", "14": "empty", "9": "empty", "10": "empty", "12": "occupied"},
    "img3.jpg": {"37": "occupied", "36": "occupied", "35": "occupied", "34": "occupied"},
    "img4.jpg": {"3": "empty", "4": "empty", "5": "empty", "6": "empty", "7": "occupied"},
    "img5.jpg": {"3": "empty", "4": "empty", "5": "empty", "6": "empty", "7": "occupied"},
    "img6.jpg": {"76": "occupied", "75": "occupied", "74": "occupied", "73": "occupied", "72": "occupied"},
    "img7.jpg": {"108": "occupied", "106": "occupied", "105": "empty", "104": "empty"},
    "img8.jpg": {"37": "occupied", "36": "empty", "35": "empty", "34": "empty"},
    "img9.jpg": {"37": "empty", "36": "occupied", "35": "occupied", "34": "occupied"},
    "img10.jpg": {"78": "occupied", "77": "occupied", "76": "occupied", "75": "occupied"},
    "img11.jpg": {"51": "occupied", "2": "occupied", "3": "empty", "4": "empty"},
    "img12.jpg": {"60": "occupied", "61": "occupied", "59": "empty"},
    "img13.jpg": {"51": "occupied", "2": "occupied", "3": "empty", "4": "empty"},
    "img14.jpg": {"51": "empty", "2": "empty", "3": "empty", "4": "empty"},
    "img15.jpg": {"79": "empty", "78": "empty", "77": "occupied", "76": "empty"},
    "img16.jpg": {"37": "occupied", "36": "occupied", "35": "occupied", "34": "occupied"},
    "img17.jpg": {"51": "occupied", "2": "occupied", "3": "empty", "4": "empty", "5": "empty"},
    "img18.jpg": {"51": "empty", "2": "empty", "3": "empty", "4": "empty", "5": "empty"},
    "img19.jpg": {"37": "occupied", "2": "empty", "3": "empty", "4": "empty", "51": "empty"},
    "img20.jpg": {"51": "empty", "2": "empty", "3": "empty", "4": "empty", "5": "empty"},
<<<<<<< HEAD
}

=======
    "parkingwithcar.jpg": {"60": "occupied", "2": "empty", "3": "empty", "4": "empty", "5": "empty"},
}

# Evaluation counters
>>>>>>> test
use_ground_truth = len(ground_truth) > 0
correct = 0
total = 0

<<<<<<< HEAD
# ==============================
# FUNCTIONS
# ==============================

def extract_slot_with_mask(image, points):
=======

# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def extract_slot_with_mask(image, points):
    """
    Extract a parking slot region using polygon masking.

    Returns:
        image_crop: cropped ROI image
        mask_crop: binary mask of the ROI
    """
>>>>>>> test
    if image is None:
        raise ValueError("Image is None — check loading paths")

    pts = np.array(points, np.int32)

<<<<<<< HEAD
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

=======
    # Create binary mask for the slot
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    # Bounding box around polygon
>>>>>>> test
    x, y, w, h = cv2.boundingRect(pts)

    image_crop = image[y:y+h, x:x+w]
    mask_crop = mask[y:y+h, x:x+w]

    return image_crop, mask_crop


<<<<<<< HEAD
=======
# =========================================================
# GRAYSCALE OCCUPANCY CLASSIFIER
# =========================================================

>>>>>>> test
def grayscale_classifier_masked(
    empty_crop,
    current_crop,
    mask_crop,
    diff_threshold=40,
    occupancy_ratio_threshold=0.15
):
<<<<<<< HEAD
    empty_gray = cv2.cvtColor(empty_crop, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_crop, cv2.COLOR_BGR2GRAY)

=======
    """
    Compare empty vs current slot using grayscale difference.

    Returns:
        occupied (bool)
        occupancy_ratio (float)
        mean_difference (float)
    """
    # Convert to grayscale
    empty_gray = cv2.cvtColor(empty_crop, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_crop, cv2.COLOR_BGR2GRAY)

    # Resize current image to match reference
>>>>>>> test
    current_gray = cv2.resize(
        current_gray,
        (empty_gray.shape[1], empty_gray.shape[0])
    )
<<<<<<< HEAD
=======

    # Resize mask accordingly
>>>>>>> test
    mask_crop = cv2.resize(
        mask_crop,
        (empty_gray.shape[1], empty_gray.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

<<<<<<< HEAD
    empty_gray = cv2.GaussianBlur(empty_gray, (5, 5), 0)
    current_gray = cv2.GaussianBlur(current_gray, (5, 5), 0)

    diff = cv2.absdiff(empty_gray, current_gray)

=======
    # Apply Gaussian blur to reduce noise
    empty_gray = cv2.GaussianBlur(empty_gray, (5, 5), 0)
    current_gray = cv2.GaussianBlur(current_gray, (5, 5), 0)

    # Compute difference
    diff = cv2.absdiff(empty_gray, current_gray)

    # Focus only on ROI pixels
>>>>>>> test
    roi_diff = diff[mask_crop > 0]

    if len(roi_diff) == 0:
        return False, 0.0, 0.0

<<<<<<< HEAD
=======
    # Calculate occupancy metrics
>>>>>>> test
    changed_pixels = np.sum(roi_diff > diff_threshold)
    total_pixels = len(roi_diff)

    occupancy_ratio = changed_pixels / total_pixels
    occupied = occupancy_ratio > occupancy_ratio_threshold
    mean_diff = float(np.mean(roi_diff))

    return occupied, occupancy_ratio, mean_diff

<<<<<<< HEAD
# ==============================
# MAIN LOOP
# ==============================

valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
images = ["parkingwithcar.jpg"]  
=======

# =========================================================
# MAIN PIPELINE
# =========================================================

# Select images to process (can be expanded later)
valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
images = ["parkingwithcar.jpg"]
>>>>>>> test

print("Found images:", len(images))

for img_name in images:
    frame_path = os.path.join(FRAMES_FOLDER, img_name)
    frame = cv2.imread(frame_path)

    print("Loading frame:", frame_path, "->", frame is not None)

    if frame is None:
        print(f"Skipping unreadable image: {frame_path}")
        continue

    print(f"\nProcessing: {img_name}")

<<<<<<< HEAD
=======
    # Process each parking slot
>>>>>>> test
    for slot in slots:
        slot_id = str(slot["slot_id"])
        pts = slot["points"]

<<<<<<< HEAD
        empty_crop, mask_crop = extract_slot_with_mask(empty_img, pts)
        current_crop, _ = extract_slot_with_mask(frame, pts)

=======
        # Extract ROI for empty and current frames
        empty_crop, mask_crop = extract_slot_with_mask(empty_img, pts)
        current_crop, _ = extract_slot_with_mask(frame, pts)

        # Run grayscale classification
>>>>>>> test
        occupied, occupancy_ratio, mean_diff = grayscale_classifier_masked(
            empty_crop,
            current_crop,
            mask_crop,
            diff_threshold=DIFF_THRESHOLD,
            occupancy_ratio_threshold=OCCUPANCY_RATIO_THRESHOLD
        )

        prediction = "occupied" if occupied else "empty"

        print(
            f"Slot {slot_id}: {prediction.upper()} | "
            f"ratio={occupancy_ratio:.3f} | mean_diff={mean_diff:.2f}"
        )

<<<<<<< HEAD
        # ======================
        # EVALUATION
        # ======================
=======
        # -----------------------------
        # EVALUATION (against ground truth)
        # -----------------------------
>>>>>>> test
        if use_ground_truth and img_name in ground_truth:
            if slot_id in ground_truth[img_name]:
                actual = ground_truth[img_name][slot_id]

                if prediction == actual:
                    correct += 1
                else:
                    print(
                        f"Wrong: Slot {slot_id} | "
                        f"pred={prediction} | actual={actual}"
                    )

                total += 1

<<<<<<< HEAD
        # ======================
        # VISUALIZATION
        # ======================
=======
        # -----------------------------
        # VISUALIZATION
        # -----------------------------
>>>>>>> test
        if SHOW_IMAGES:
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

<<<<<<< HEAD
=======
    # Display result
>>>>>>> test
    if SHOW_IMAGES:
        cv2.imshow("Result", frame)
        key = cv2.waitKey(0)

        if key == ord("q"):
            break

cv2.destroyAllWindows()

<<<<<<< HEAD
# ==============================
# FINAL ACCURACY
# ==============================
=======

# =========================================================
# FINAL ACCURACY
# =========================================================
>>>>>>> test

if use_ground_truth and total > 0:
    accuracy = correct / total
    print("\n======================")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Correct: {correct}")
    print(f"Total:   {total}")
    print("======================")
else:
    print("\nNo ground-truth comparisons were made.")