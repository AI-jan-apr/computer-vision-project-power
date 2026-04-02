import os
import shutil
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FOLDER = os.path.join(BASE_DIR, "..", "src", "frames_test2")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "..", "src", "frames_sorted")


def extract_timestamp(filename):
    """
    Example filename:
    2012-09-15_09_37_08_jpg.rf.5a4dcccc262ce9206f4ba43bd60cbbf3
    """
    parts = filename.split("_")
    timestamp_str = "_".join(parts[:4])
    return datetime.strptime(timestamp_str, "%Y-%m-%d_%H_%M_%S")


def main():
    valid_ext = (".jpg", ".jpeg", ".png")
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(valid_ext)]

    # Sort files by extracted timestamp
    files.sort(key=extract_timestamp)

    # Create output folder if it does not exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Copy files into new folder with ordered names
    for idx, filename in enumerate(files, start=1):
        src_path = os.path.join(INPUT_FOLDER, filename)
        dst_name = f"frame_{idx:04d}.jpg"
        dst_path = os.path.join(OUTPUT_FOLDER, dst_name)

        shutil.copy2(src_path, dst_path)

    print(f"Done. {len(files)} images copied to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()