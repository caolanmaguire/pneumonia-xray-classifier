"""PNEUMONIA CHEST X-RAY CLASSIFIER
"""
import os
from ultralytics import YOLO
import cv2
from random import shuffle

def install_dataset():
    # Download latest version
    import kagglehub
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

    print("Path to dataset files:", path)

if __name__ == "__main__":
    # print("Dataset downloaded successfully!")
    if os.path.exists('runs/'):
        print("Dataset already exists. Skipping download.")

        # Fine tuned model
        model = YOLO("runs/classify/train/weights/best.pt")

        cv2.namedWindow("prediction", cv2.WINDOW_NORMAL)

        all_tests = (
            [("NORMAL", f) for f in os.listdir("chest_xray/test/NORMAL/")] +
            [("PNEUMONIA", f) for f in os.listdir("chest_xray/test/PNEUMONIA/")]
        )

        accuracy_checker = 0

        shuffle(all_tests)

        demo_mode = input("Do you want to run in demo mode? (y/n): ").lower()

        for label, filename in all_tests:
            path = f"chest_xray/test/{label}/{filename}"
            
            results = model.predict(source=path, imgsz=224)
            probs = results[0].probs
            prediction = 'PNEUMONIA' if probs.top1 == 1 else 'NORMAL'
            confidence = float(probs.top1conf) * 100

            if label == prediction:
                accuracy_checker += 1

            img = cv2.imread(path)
            color = (0, 255, 0) if prediction == label else (0, 0, 255)  # green if correct, red if wrong
            # cv2.putText(img, f"GT: {label} | Pred: {prediction} ({confidence:.1f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if demo_mode == 'y':
                cv2.imshow("prediction", img)
                cv2.waitKey(0)


            if cv2.getWindowProperty("prediction", cv2.WND_PROP_VISIBLE) < 1:  # Check if window is closed
                break

            probs = results[0].probs
            print(f"Prediction: {'PNEUMONIA' if probs.top1 == 1 else 'NORMAL'}")
            print(f"Confidence: {float(probs.top1conf) * 100:.1f}%")

            print(f"Current Accuracy: {accuracy_checker}/{len(all_tests)} ({(accuracy_checker/len(all_tests))*100:.2f}%)")
        
        print('Testing complete.')

    else:
        install_dataset()
        print("Training begins...")

        model = YOLO("yolov8n-cls.pt")  # nano = small and fast

        model.train(
            data="rsna_chest_xray/",  # points to your train/val folders
            epochs=10,
            imgsz=224
        )