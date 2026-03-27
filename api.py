"""
Pneumonia X-Ray Classifier — Flask API
---------------------------------------
Exposes a single POST /predict endpoint.
Expects a multipart form upload with key 'image'.
Returns JSON: { prediction, confidence, probabilities }

Usage:
    python api.py

Then open index.html in a browser (or serve it via any static server).
Requires: pip install flask flask-cors ultralytics pillow
"""

import io
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)
CORS(app)  # Allow requests from the static HTML frontend

# ---------------------------------------------------------------------------
# Model loading — expects a trained best.pt in the default YOLO output path.
# Update MODEL_PATH if your weights live elsewhere.
# ---------------------------------------------------------------------------
MODEL_PATH = "runs/classify/train/weights/best.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Trained model not found at '{MODEL_PATH}'.\n"
        "Please train the model first by running main.py without the 'runs/' directory present."
    )

model = YOLO(MODEL_PATH)
print(f"✓ Model loaded from {MODEL_PATH}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Simple liveness check."""
    return jsonify({"status": "ok", "model": MODEL_PATH})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a chest X-ray image and returns a classification result.

    Request:
        multipart/form-data with field 'image' (JPEG or PNG)

    Response:
        {
            "prediction": "NORMAL" | "PNEUMONIA",
            "confidence": 94.3,          // percentage, 0–100
            "probabilities": {
                "NORMAL": 5.7,
                "PNEUMONIA": 94.3
            }
        }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image provided. Send a file under the key 'image'."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    # Read image bytes and pass directly to YOLO
    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not read image: {str(e)}"}), 400

    # Run inference
    results = model.predict(source=image, imgsz=224, verbose=False)
    probs = results[0].probs

    # Class indices: 0 = NORMAL, 1 = PNEUMONIA (matches Kaggle dataset folder order)
    top1_idx = int(probs.top1)
    prediction = "PNEUMONIA" if top1_idx == 1 else "NORMAL"
    confidence = float(probs.top1conf) * 100

    normal_prob = float(probs.data[0]) * 100
    pneumonia_prob = float(probs.data[1]) * 100

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "probabilities": {
            "NORMAL": round(normal_prob, 2),
            "PNEUMONIA": round(pneumonia_prob, 2),
        }
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n🩻 Pneumonia Classifier API")
    print("   POST http://localhost:5000/predict")
    print("   GET  http://localhost:5000/health\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
