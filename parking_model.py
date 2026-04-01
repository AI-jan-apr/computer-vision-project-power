from ultralytics import YOLO


class ParkingModel:
    def __init__(self, model_path: str = "best.pt"):
        self.model = YOLO(model_path)

    def detect_cars(self, image_path: str, conf: float = 0.1, imgsz: int = 1024):
        """
        ترجع قائمة سيارات بالشكل:
        [
            {
                "box": [x1, y1, x2, y2],
                "conf": 0.92,
                "cls": 0
            },
            ...
        ]
        """
        results = self.model(image_path, conf=conf, imgsz=imgsz, verbose=False)
        detections = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                detections.append({
                    "box": [x1, y1, x2, y2],
                    "conf": confidence,
                    "cls": cls_id
                })

        return detections