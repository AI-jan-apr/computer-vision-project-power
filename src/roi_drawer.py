import cv2
import json
import numpy as np

# ================================
# LOAD BASE IMAGE (EMPTY PARKING)
# ================================
# This image is used as a reference to draw parking slots (ROIs)

image_path = "empty_parking.jpg"
img = cv2.imread(image_path)

# Ensure the image is loaded correctly
if img is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

# Create a copy to draw on (we keep the original untouched)
base_img = img.copy()

# ================================
# GLOBAL VARIABLES
# ================================
# points: stores temporary clicks for one slot
# slots: stores all finalized parking slots
# slot_id: unique ID for each slot

points = []
slots = []
slot_id = 1


# ================================
# REDRAW FUNCTION
# ================================
# This function updates the display window every time:
# - a point is clicked
# - a slot is added/removed
# It visualizes both saved slots and current drawing

def redraw():
    display = base_img.copy()

    # ----------------------------
    # Draw all saved slots (green)
    # ----------------------------
    for slot in slots:
        pts = np.array(slot["points"], dtype=np.int32)

        # Draw polygon for slot
        cv2.polylines(display, [pts], True, (0, 255, 0), 2)

        # Display slot ID near first point
        x, y = pts[0]
        cv2.putText(
            display,
            str(slot["slot_id"]),
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # -----------------------------------------
    # Draw currently selected points (in red)
    # -----------------------------------------
    for i, (x, y) in enumerate(points):
        # Draw point
        cv2.circle(display, (x, y), 4, (0, 0, 255), -1)

        # Draw line between consecutive points
        if i > 0:
            cv2.line(display, points[i - 1], points[i], (255, 0, 0), 2)

    # -----------------------------------------
    # If 4 points selected, close the polygon
    # -----------------------------------------
    if len(points) == 4:
        cv2.line(display, points[3], points[0], (255, 0, 0), 2)

    # Show updated image
    cv2.imshow("ROI Drawer", display)


# ================================
# MOUSE CLICK HANDLER
# ================================
# This function is triggered when the user clicks:
# - Collects 4 points per slot
# - Automatically saves slot after 4 points

def click_event(event, x, y, flags, param):
    global points, slots, slot_id

    # Left mouse click
    if event == cv2.EVENT_LBUTTONDOWN:

        # Limit to 4 points (quadrilateral)
        if len(points) < 4:
            points.append((x, y))
            redraw()

            # Once 4 points are selected → save slot
            if len(points) == 4:
                slots.append({
                    "slot_id": slot_id,
                    "points": points.copy()
                })

                print(f"Saved slot {slot_id}")

                slot_id += 1
                points = []  # reset for next slot

                redraw()


# ================================
# WINDOW SETUP
# ================================
cv2.namedWindow("ROI Drawer")
cv2.setMouseCallback("ROI Drawer", click_event)


# ================================
# USER INSTRUCTIONS
# ================================
print("Instructions:")
print("- Left click 4 points for each parking slot")
print("- Slot is saved automatically after 4 clicks")
print("- Press 'u' to undo current unfinished points")
print("- Press 'd' to delete last saved slot")
print("- Press 'q' to save and quit")


# Initial draw
redraw()


# ================================
# MAIN LOOP (KEYBOARD CONTROLS)
# ================================
while True:
    key = cv2.waitKey(1) & 0xFF

    # Undo current drawing (before saving)
    if key == ord('u'):
        points = []
        redraw()

    # Delete last saved slot
    elif key == ord('d'):
        if slots:
            removed = slots.pop()

            # Reuse slot ID if needed
            slot_id = removed["slot_id"]

            print(f"Deleted slot {removed['slot_id']}")
            redraw()

    # Quit and save
    elif key == ord('q'):
        break


# Close window
cv2.destroyAllWindows()


# ================================
# SAVE TO JSON
# ================================
# This file will be used later in the detection pipeline

with open("slots.json", "w") as f:
    json.dump(slots, f, indent=4)

print("Saved all slots to slots.json")