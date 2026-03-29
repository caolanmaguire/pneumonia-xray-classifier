"""
Pneumonia X-Ray Classifier — Evaluation Script
------------------------------------------------
Scans the runs/ directory for all trained models (best.pt),
evaluates each against a test dataset, and saves results to /results.

Usage:
    python evaluate.py                        # uses dataset from config.json
    python evaluate.py --dataset rsna         # evaluate against RSNA test set
    python evaluate.py --dataset kaggle       # evaluate against Kaggle test set

Outputs per model (saved to results/<run_name>/):
    confusion_matrix.png
    roc_curve.png
    classification_report.txt
    results.json
"""

import os
import json
import argparse
import logging
from datetime import datetime
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, no display needed
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
)
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("evaluate.log")
    ]
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
with open("config.json") as f:
    config = json.load(f)

IMAGE_SIZE = config["model"]["image_size"]
CLASSES    = config["model"]["classes"]  # {"0": "NORMAL", "1": "PNEUMONIA"}
RUNS_DIR   = config["output"]["runs_dir"]
RESULTS_DIR = config["output"]["results_dir"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_all_models(runs_dir):
    """Recursively find all best.pt files in the runs directory."""
    pattern = os.path.join(runs_dir, "**/weights/best.pt")
    models  = glob(pattern, recursive=True)
    if not models:
        raise FileNotFoundError(f"No trained models found in '{runs_dir}'.")
    return models


def get_test_dir(dataset):
    """Return the test directory path for the chosen dataset."""
    if dataset == "kaggle":
        return config["datasets"]["kaggle"]["test_dir"]
    elif dataset == "rsna":
        # RSNA val set used as test — no separate test split
        return os.path.join(config["datasets"]["rsna"]["output_dir"], "val")
    elif dataset == "nih":
        return os.path.join(config["datasets"]["nih"]["output_dir"], "val")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def load_test_images(test_dir):
    """
    Walk the test directory and return a list of (filepath, label_idx) tuples.
    Expects structure: test_dir/CLASSNAME/image.jpg
    """
    samples = []
    class_names = sorted(os.listdir(test_dir))

    for class_name in class_names:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Match class name to index from config
        label_idx = next(
            (int(k) for k, v in CLASSES.items() if v == class_name.upper()),
            None
        )
        if label_idx is None:
            log.warning(f"Skipping unknown class folder: {class_name}")
            continue

        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(class_dir, fname), label_idx))

    log.info(f"Found {len(samples)} test images across {len(class_names)} classes")
    return samples


def evaluate_model(model_path, samples, output_dir):
    """Run evaluation for a single model and save all outputs."""
    os.makedirs(output_dir, exist_ok=True)

    log.info(f"Loading model: {model_path}")
    model = YOLO(model_path)

    y_true   = []
    y_pred   = []
    y_scores = []  # pneumonia probability for ROC curve

    total  = len(samples)
    errors = 0

    for i, (filepath, true_label) in enumerate(samples):
        try:
            results    = model.predict(source=filepath, imgsz=IMAGE_SIZE, verbose=False)
            probs      = results[0].probs
            pred_idx   = int(probs.top1)
            pneumonia_score = float(probs.data[1])  # probability of PNEUMONIA class

            y_true.append(true_label)
            y_pred.append(pred_idx)
            y_scores.append(pneumonia_score)

            if (i + 1) % 100 == 0:
                log.info(f"  Progress: {i + 1}/{total}")

        except Exception as e:
            log.warning(f"  Error on {filepath}: {e}")
            errors += 1

    log.info(f"Inference complete. Errors: {errors}/{total}")

    y_true   = np.array(y_true)
    y_pred   = np.array(y_pred)
    y_scores = np.array(y_scores)

    class_names = [CLASSES[str(i)] for i in sorted(CLASSES.keys(), key=int)]

    # ── Confusion Matrix ─────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix\n{os.path.basename(model_path)}", fontsize=13)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    log.info(f"Saved: {cm_path}")

    # ── ROC Curve ────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#00d4ff", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve\n{os.path.basename(model_path)}", fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    log.info(f"Saved: {roc_path}")

    # ── Classification Report ────────────────────────────────────────────────
    report = classification_report(y_true, y_pred, target_names=class_names)
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Evaluated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(report)
    log.info(f"Saved: {report_path}")
    print(f"\n{report}")

    # ── results.json ─────────────────────────────────────────────────────────
    results_data = {
        "model_path":  model_path,
        "evaluated":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_images": total,
        "errors":      errors,
        "accuracy":    round(float(np.mean(y_true == y_pred)) * 100, 2),
        "auc":         round(float(roc_auc), 4),
        "precision":   round(float(precision_score(y_true, y_pred)), 4),
        "recall":      round(float(recall_score(y_true, y_pred)), 4),
        "f1":          round(float(f1_score(y_true, y_pred)), 4),
        "confusion_matrix": cm.tolist(),
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    log.info(f"Saved: {results_path}")

    return results_data


def print_summary(all_results):
    """Print a comparison table of all evaluated models."""
    print("\n" + "=" * 70)
    print(f"{'Model':<35} {'Accuracy':>9} {'AUC':>7} {'Recall':>8} {'F1':>7}")
    print("=" * 70)
    for r in sorted(all_results, key=lambda x: x["auc"], reverse=True):
        name = os.path.basename(os.path.dirname(os.path.dirname(r["model_path"])))
        print(
            f"{name:<35} {r['accuracy']:>8.2f}% {r['auc']:>7.4f} "
            f"{r['recall']:>8.4f} {r['f1']:>7.4f}"
        )
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all trained models")
    parser.add_argument(
        "--dataset",
        type=str,
        default="kaggle",
        choices=["kaggle", "rsna", "nih"],
        help="Which test dataset to evaluate against (default: kaggle)"
    )
    args = parser.parse_args()

    log.info(f"Starting evaluation against dataset: {args.dataset}")

    # Find all models
    model_paths = find_all_models(RUNS_DIR)
    log.info(f"Found {len(model_paths)} model(s) to evaluate")

    # Load test images
    test_dir = get_test_dir(args.dataset)
    samples  = load_test_images(test_dir)

    # Evaluate each model
    all_results = []
    for model_path in model_paths:
        run_name   = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
        output_dir = os.path.join(RESULTS_DIR, f"{run_name}_{args.dataset}")

        log.info(f"\nEvaluating: {run_name}")
        result = evaluate_model(model_path, samples, output_dir)
        all_results.append(result)

    # Summary table
    print_summary(all_results)
    log.info("Evaluation complete.")