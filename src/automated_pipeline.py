import os
import cv2
import numpy as np

BASE_DIR = os.getcwd()
IMAGE_PATH = os.path.join(BASE_DIR, "src/new_empty_parking.jpg")

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

# Step 1: grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Step 2: darken / increase contrast
# alpha = -1   # contrast
# beta = -40    # brightness
# darkened = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

# Save outputs
cv2.imwrite("step1_original.png", img)
cv2.imwrite("step2_grayscale.png", gray)
# cv2.imwrite("step3_darkened.png", darkened)

print("Saved:")
print("step1_original.png")
print("step2_grayscale.png")
# print("step3_darkened.png")



BASE_DIR = os.getcwd()
IMAGE_PATH = os.path.join(BASE_DIR, "step2_grayscale.png")

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

# If the saved image is 3-channel, convert to gray
if len(img.shape) == 3:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    gray = img.copy()

# ==============================
# STEP 1: THRESHOLD BRIGHT LINES
# ==============================
# We keep only bright pixels, which should correspond mostly
# to the painted parking lines.

# ==============================
# IMPROVED WHITE LINE EXTRACTION
# ==============================

# 1. Apply CLAHE (contrast enhancement)
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
enhanced = clahe.apply(gray)

# 2. Blur to remove noise
blur = cv2.GaussianBlur(enhanced, (5,5), 0)

# 3. Strong threshold for white paint only
_, binary = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

# 4. Remove noise
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

# 5. CONNECT BROKEN LINES (VERY IMPORTANT)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))

# ==============================
# STEP 2: DETECT LINE SEGMENTS
# ==============================
lines = cv2.HoughLinesP(
    binary,
    1,
    np.pi / 180,
    threshold=20,
    minLineLength=20,
    maxLineGap=8
)

raw_lines_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
filtered_lines_img = raw_lines_img.copy()

raw_lines = []
if lines is not None:
    raw_lines = [line[0] for line in lines]

# Draw all raw lines in blue
for line in raw_lines:
    x1, y1, x2, y2 = line
    cv2.line(raw_lines_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

# ==============================
# STEP 3: FILTER SEPARATOR-LIKE LINES
# ==============================
# We keep slanted lines, not horizontal ones.
# Parking separator lines in your image are diagonal/slanted.

filtered_lines = []

for line in raw_lines:
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1

    if abs(dx) < 2:
        slope = 999
    else:
        slope = dy / dx

    length = np.sqrt(dx * dx + dy * dy)

    # keep slanted moderate-length lines
    if abs(slope) >= 0.3 and abs(slope) <= 10 and length >= 20:
        filtered_lines.append((x1, y1, x2, y2))

# Draw filtered lines in green
for x1, y1, x2, y2 in filtered_lines:
    cv2.line(filtered_lines_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# ==============================
# SAVE OUTPUTS
# ==============================
cv2.imwrite("step4_binary_lines.png", binary)
cv2.imwrite("step5_raw_lines.png", raw_lines_img)
cv2.imwrite("step6_filtered_lines.png", filtered_lines_img)

print("Saved:")
print("step4_binary_lines.png")
print("step5_raw_lines.png")
print("step6_filtered_lines.png")
print(f"Raw lines detected: {len(raw_lines)}")
print(f"Filtered lines kept: {len(filtered_lines)}")