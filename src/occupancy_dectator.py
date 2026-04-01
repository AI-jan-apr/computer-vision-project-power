import cv2
import json
import numpy as np

# Load images
empty_img = cv2.imread("empty_parking.jpg")
current_img = cv2.imread("somanycars.jpg")  # change this later

# Load slot ROIs
with open("slots.json") as f:
    slots = json.load(f)


def extract_slot(image, points):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    pts = np.array(points, np.int32)
    cv2.fillPoly(mask, [pts], 255)

    result = cv2.bitwise_and(image, image, mask=mask)

    # crop bounding box for efficiency
    x, y, w, h = cv2.boundingRect(pts)
    return result[y:y+h, x:x+w]


def is_occupied(empty_crop, current_crop, threshold=25):
    # Convert to grayscale
    empty_gray = cv2.cvtColor(empty_crop, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_crop, cv2.COLOR_BGR2GRAY)

    # Resize (important if slight mismatch)
    current_gray = cv2.resize(current_gray, (empty_gray.shape[1], empty_gray.shape[0]))

    # Absolute difference
    diff = cv2.absdiff(empty_gray, current_gray)

    score = np.mean(diff)

    return score > threshold, score


# Process each slot
for slot in slots:
    slot_id = slot["slot_id"]
    points = slot["points"]

    empty_crop = extract_slot(empty_img, points)
    current_crop = extract_slot(current_img, points)

    occupied, score = is_occupied(empty_crop, current_crop)

    print(f"Slot {slot_id}: {'OCCUPIED' if occupied else 'EMPTY'} | score={score:.2f}")