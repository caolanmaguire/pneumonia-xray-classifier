"""
RSNA Dataset Preparation Script
Converts DICOM files to JPEGs and organises into YOLOv8 folder structure:
    rsna_chest_xray/
        train/
            NORMAL/
            PNEUMONIA/
        val/
            NORMAL/
            PNEUMONIA/
"""

import os
import pandas as pd
import pydicom
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split

# ── Config ──────────────────────────────────────────────────────
DATASET_DIR  = "new_dataset/stage_2_train_images"
LABELS_CSV   = "new_dataset/stage_2_train_labels.csv"
OUTPUT_DIR   = "rsna_chest_xray"
VAL_SPLIT    = 0.2
# ────────────────────────────────────────────────────────────────

def dicom_to_jpeg(dicom_path, output_path):
    ds = pydicom.dcmread(dicom_path)
    pixel_array = ds.pixel_array.astype(np.float32)
    # Normalise to 0-255
    pixel_array -= pixel_array.min()
    if pixel_array.max() > 0:
        pixel_array /= pixel_array.max()
    pixel_array = (pixel_array * 255).astype(np.uint8)
    img = Image.fromarray(pixel_array).convert("RGB")
    img.save(output_path, "JPEG")

def prepare():
    # Read labels — one row per box, Target 1 = PNEUMONIA, 0 = NORMAL
    df = pd.read_csv(LABELS_CSV)

    # One label per patient (some pneumonia patients have multiple rows)
    labels = df.groupby("patientId")["Target"].max().reset_index()
    labels.columns = ["patientId", "label"]

    normal    = labels[labels["label"] == 0]["patientId"].tolist()
    pneumonia = labels[labels["label"] == 1]["patientId"].tolist()

    print(f"Normal:    {len(normal)}")
    print(f"Pneumonia: {len(pneumonia)}")

    # Train/val split per class
    n_train, n_val = train_test_split(normal,    test_size=VAL_SPLIT, random_state=42)
    p_train, p_val = train_test_split(pneumonia, test_size=VAL_SPLIT, random_state=42)

    splits = {
        "train/NORMAL":    n_train,
        "train/PNEUMONIA": p_train,
        "val/NORMAL":      n_val,
        "val/PNEUMONIA":   p_val,
    }

    for folder, patient_ids in splits.items():
        out_path = os.path.join(OUTPUT_DIR, folder)
        os.makedirs(out_path, exist_ok=True)
        print(f"\nConverting {folder} ({len(patient_ids)} images)...")

        for patient_id in tqdm(patient_ids):
            dicom_path = os.path.join(DATASET_DIR, f"{patient_id}.dcm")
            jpeg_path  = os.path.join(out_path, f"{patient_id}.jpg")

            if not os.path.exists(dicom_path):
                print(f"  Missing: {dicom_path}")
                continue

            try:
                dicom_to_jpeg(dicom_path, jpeg_path)
            except Exception as e:
                print(f"  Error on {patient_id}: {e}")

    print("\n✓ Dataset prepared successfully.")
    print(f"  Output: {OUTPUT_DIR}/")

if __name__ == "__main__":
    prepare()