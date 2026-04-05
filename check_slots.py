import json
<<<<<<< HEAD

with open("slots.json") as f:
    data = json.load(f)

print(data)
=======
import cv2
import numpy as np

from src.parking_model import ParkingModel


def load_slots(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def box_center(box):
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy


def point_in_polygon(point, polygon):
    px, py = point
    polygon_np = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(polygon_np, (px, py), False) >= 0


def draw_polygon(img, points, color, thickness=2):
    pts = np.array(points, dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def main():
    image_path = "empty_parking.jpg"
    slots_path = "slots.json"
    model_path = "best.pt"

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    slots = load_slots(slots_path)
    detector = ParkingModel(model_path=model_path)

    detections = detector.detect_cars(image_path=image_path, conf=0.1, imgsz=1024)

    occupied_count = 0
    empty_count = 0

    # رسم الـ detections أول
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        conf = det["conf"]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            img,
            f"car {conf:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )

    # فحص كل موقف
    for slot in slots:
        slot_id = slot["slot_id"]
        points = slot["points"]

        occupied = False

        for det in detections:
            center = box_center(det["box"])

            if point_in_polygon(center, points):
                occupied = True
                break

        if occupied:
            color = (0, 0, 255)
            label = "Occupied"
            occupied_count += 1
        else:
            color = (0, 255, 0)
            label = "Empty"
            empty_count += 1

        draw_polygon(img, points, color, thickness=2)

        # مكان كتابة النص
        text_x = points[0][0]
        text_y = points[0][1] - 5

        cv2.putText(
            img,
            f"{slot_id}: {label}",
            (text_x, max(20, text_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    # عدادات عامة
    cv2.putText(
        img,
        f"Occupied: {occupied_count}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )
    cv2.putText(
        img,
        f"Empty: {empty_count}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imwrite("output_result.jpg", img)
    print("Saved result to output_result.jpg")

    cv2.imshow("Parking Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

