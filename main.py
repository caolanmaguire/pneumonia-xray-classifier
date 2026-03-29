"""PNEUMONIA CHEST X-RAY CLASSIFIER"""

import os
import json
from ultralytics import YOLO
import cv2
from random import shuffle

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
with open("config.json") as f:
    config = json.load(f)

MODEL_PATH   = config["model"]["weights_path"]
IMAGE_SIZE   = config["model"]["image_size"]
CLASSES      = config["model"]["classes"]  # {"0": "NORMAL", "1": "PNEUMONIA"}
CONF_THRESHOLD = config["model"]["confidence_threshold"]

TRAIN_DATA   = config["datasets"]["kaggle"]["train_dir"]
TEST_NORMAL  = config["datasets"]["kaggle"]["test_dir"] + "/NORMAL/"
TEST_PNEUMONIA = config["datasets"]["kaggle"]["test_dir"] + "/PNEUMONIA/"

EPOCHS       = config["training"]["epochs"]
BATCH_SIZE   = config["training"]["batch_size"]
DEVICE       = config["training"]["device"]
BASE_MODEL   = config["training"]["base_model"]
RUNS_DIR     = config["output"]["runs_dir"]


# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------
def install_dataset():
    import kagglehub
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    print("Path to dataset files:", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    if os.path.exists(RUNS_DIR):
        print("Trained model found. Skipping download and training.")

        model = YOLO(MODEL_PATH)

        cv2.namedWindow("prediction", cv2.WINDOW_NORMAL)

        all_tests = (
            [("NORMAL",    f) for f in os.listdir(TEST_NORMAL)] +
            [("PNEUMONIA", f) for f in os.listdir(TEST_PNEUMONIA)]
        )

        accuracy_checker = 0
        low_conf_count   = 0

        shuffle(all_tests)

        demo_mode = input("Do you want to run in demo mode? (y/n): ").lower()

        for label, filename in all_tests:
            path = f"{config['datasets']['kaggle']['test_dir']}/{label}/{filename}"

            results    = model.predict(source=path, imgsz=IMAGE_SIZE, verbose=False)
            probs      = results[0].probs
            top1_idx   = int(probs.top1)
            prediction = CLASSES[str(top1_idx)]
            confidence = float(probs.top1conf) * 100

            if label == prediction:
                accuracy_checker += 1

            # Flag low confidence predictions
            low_conf = confidence < CONF_THRESHOLD
            if low_conf:
                low_conf_count += 1

            img   = cv2.imread(path)
            color = (0, 255, 0) if prediction == label else (0, 0, 255)

            print(f"Prediction: {prediction} ({confidence:.1f}%) {'⚠ LOW CONFIDENCE' if low_conf else ''}")
            print(f"Current Accuracy: {accuracy_checker}/{len(all_tests)} ({(accuracy_checker / len(all_tests)) * 100:.2f}%)")

            if demo_mode == 'y':
                cv2.imshow("prediction", img)
                cv2.waitKey(0)

            if cv2.getWindowProperty("prediction", cv2.WND_PROP_VISIBLE) < 1:
                break

        print(f"\nTesting complete.")
        print(f"Final Accuracy:     {accuracy_checker}/{len(all_tests)} ({(accuracy_checker / len(all_tests)) * 100:.2f}%)")
        print(f"Low Confidence:     {low_conf_count}/{len(all_tests)} predictions flagged below {CONF_THRESHOLD}%")

    else:
        install_dataset()
        print("Training begins...")

        model = YOLO(BASE_MODEL)

        model.train(
            data="rsna_chest_xray/",
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
        )