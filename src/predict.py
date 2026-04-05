from ultralytics import YOLO
from pathlib import Path
import cv2

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"
IMAGE_PATH = BASE_DIR / "IMG_0822.png"

print("Model path:", MODEL_PATH)
print("Model exists:", MODEL_PATH.exists())
print("Image path:", IMAGE_PATH)
print("Image exists:", IMAGE_PATH.exists())

model = YOLO(str(MODEL_PATH))
results = model(str(IMAGE_PATH), imgsz=1024, conf=0.1)

for r in results:
    img = r.plot()
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()