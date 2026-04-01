import cv2
import json
import numpy as np

image_path = "empty_parking.jpg"
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

base_img = img.copy()
points = []
slots = []
slot_id = 1


def redraw():
    display = base_img.copy()

    # draw saved slots
    for slot in slots:
        pts = np.array(slot["points"], dtype=np.int32)
        cv2.polylines(display, [pts], True, (0, 255, 0), 2)

        # label near first point
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

    # draw current points and connecting lines
    for i, (x, y) in enumerate(points):
        cv2.circle(display, (x, y), 4, (0, 0, 255), -1)

        if i > 0:
            cv2.line(display, points[i - 1], points[i], (255, 0, 0), 2)

    # if 4 points are selected, close the shape visually
    if len(points) == 4:
        cv2.line(display, points[3], points[0], (255, 0, 0), 2)

    cv2.imshow("ROI Drawer", display)


def click_event(event, x, y, flags, param):
    global points, slots, slot_id

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            redraw()

            # once 4 points are clicked, save automatically
            if len(points) == 4:
                slots.append({
                    "slot_id": slot_id,
                    "points": points.copy()
                })
                print(f"Saved slot {slot_id}")
                slot_id += 1
                points = []
                redraw()


cv2.namedWindow("ROI Drawer")
cv2.setMouseCallback("ROI Drawer", click_event)

print("Instructions:")
print("- Left click 4 points for each parking slot")
print("- The slot will be saved automatically after the 4th point")
print("- Press 'u' to undo current unfinished points")
print("- Press 'd' to delete last saved slot")
print("- Press 'q' to save and quit")

redraw()

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord('u'):
        points = []
        redraw()

    elif key == ord('d'):
        if slots:
            removed = slots.pop()
            slot_id = removed["slot_id"]
            print(f"Deleted slot {removed['slot_id']}")
            redraw()

    elif key == ord('q'):
        break

cv2.destroyAllWindows()

with open("slots.json", "w") as f:
    json.dump(slots, f, indent=4)

print("Saved all slots to slots.json")