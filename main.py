"""PNEUMONIA CHEST X-RAY CLASSIFIER
"""
import os
from ultralytics import YOLO

def install_dataset():
    # Download latest version
    import kagglehub
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

    print("Path to dataset files:", path)

if __name__ == "__main__":
    # print("Dataset downloaded successfully!")
    if os.path.exists('./chest_xray'):
        print("Training begins")

        model = YOLO("yolov8n-cls.pt")  # nano = small and fast

        model.train(
            data="chest_xray/",  # points to your train/val folders
            epochs=10,
            imgsz=224
        )
    else:
        install_dataset()
