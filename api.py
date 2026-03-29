"""
Pneumonia X-Ray Classifier Flask API
---------------------------------------
Exposes a single POST /predict endpoint.
Expects a multipart form upload with key 'image'.
Returns JSON: { prediction, confidence, probabilities, heatmap }

Requires: pip install flask flask-cors ultralytics pillow pytorch-grad-cam
"""

import io
import os
import json
import base64
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
with open("config.json") as f:
    config = json.load(f)

MODEL_PATH        = config["model"]["weights_path"]
IMAGE_SIZE        = config["model"]["image_size"]
CONF_THRESHOLD    = config["model"]["confidence_threshold"]
TARGET_LAYER_IDX  = config["model"]["target_layer_index"]
CLASSES           = config["model"]["classes"]  # {"0": "NORMAL", "1": "PNEUMONIA"}

API_HOST          = config["api"]["host"]
API_PORT          = config["api"]["port"]
API_DEBUG         = config["api"]["debug"]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Trained model not found at '{MODEL_PATH}'.\n"
        "Please train the model first by running main.py without the 'runs/' directory present."
    )

model = YOLO(MODEL_PATH)
print(f"✓ Model loaded from {MODEL_PATH}")


# ---------------------------------------------------------------------------
# Grad-CAM setup — done once at startup, not per request
# ---------------------------------------------------------------------------
class YOLOClassifierWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        result = self.model(x)
        if isinstance(result, tuple):
            return result[0]
        return result

wrapped      = YOLOClassifierWrapper(model.model)
target_layer = [wrapped.model.model[TARGET_LAYER_IDX]]

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_PATH,
        "confidence_threshold": CONF_THRESHOLD,
        "image_size": IMAGE_SIZE
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided. Send a file under the key 'image'."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not read image: {str(e)}"}), 400

    # Run inference
    results = model.predict(source=image, imgsz=IMAGE_SIZE, verbose=False)
    probs   = results[0].probs

    top1_idx   = int(probs.top1)
    prediction = CLASSES[str(top1_idx)]
    confidence = float(probs.top1conf) * 100

    normal_prob    = float(probs.data[0]) * 100
    pneumonia_prob = float(probs.data[1]) * 100

    # Confidence threshold — flag low confidence predictions
    low_confidence = confidence < CONF_THRESHOLD

    # Grad-CAM
    input_tensor = transform(image).unsqueeze(0)
    rgb_img      = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32) / 255.0

    # Class-specific target
    targets = [ClassifierOutputTarget(top1_idx)]

    with EigenCAM(model=wrapped, target_layers=target_layer) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        heatmap       = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)

    buffered = io.BytesIO()
    Image.fromarray(heatmap).save(buffered, format="PNG")
    heatmap_b64 = base64.b64encode(buffered.getvalue()).decode()

    return jsonify({
        "prediction":    prediction,
        "confidence":    round(confidence, 2),
        "low_confidence": low_confidence,
        "probabilities": {
            "NORMAL":    round(normal_prob, 2),
            "PNEUMONIA": round(pneumonia_prob, 2),
        },
        "heatmap": heatmap_b64
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n🩻 Pneumonia Classifier API")
    print(f"   POST http://{API_HOST}:{API_PORT}/predict")
    print(f"   GET  http://{API_HOST}:{API_PORT}/health\n")
    app.run(host=API_HOST, port=API_PORT, debug=API_DEBUG)