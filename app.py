import os
import uuid
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

UPLOAD_FOLDER = os.path.join("static", "uploads")
MODEL_PATH    = os.path.join("model", "brain_tumor_model.h5")
ALLOWED_EXT   = {"jpg", "jpeg", "png", "webp"}
IMG_SIZE      = 224  

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("[INFO] Loading model...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\n❌ Model not found at '{MODEL_PATH}'.\n"
        "Please run: python train.py\n"
        "Or place a pre-trained 'brain_tumor_model.h5' inside the model/ folder."
    )

model = tf.keras.models.load_model(MODEL_PATH)
print(f"[INFO] Model loaded from {MODEL_PATH}")

def allowed_file(filename: str) -> bool:
    """Check if uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk and prepare it for the model:
      - Resize to 224×224 (VGG16 input size)
      - Convert to RGB (handles grayscale PNGs etc.)
      - Normalize pixel values to [0, 1]
      - Add batch dimension → shape (1, 224, 224, 3)
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  


def predict(image_path: str) -> dict:
    """
    Run inference on a preprocessed image.
    Returns a dict with:
        label      — 'Tumor Detected' or 'No Tumor Detected'
        confidence — percentage confidence (0–100)
        raw_score  — raw sigmoid output (0–1)
        risk_level — 'High' | 'Moderate' | 'Low'
    """
    processed = preprocess_image(image_path)
    raw_score  = float(model.predict(processed, verbose=0)[0][0])

    is_tumor   = raw_score > 0.5
    confidence = raw_score * 100 if is_tumor else (1 - raw_score) * 100

    if is_tumor:
        label = "Tumor Detected"
        risk  = "High" if raw_score > 0.85 else "Moderate"
    else:
        label = "No Tumor Detected"
        risk  = "Low"

    return {
        "label":      label,
        "is_tumor":   is_tumor,
        "confidence": round(confidence, 2),
        "raw_score":  round(raw_score, 4),
        "risk_level": risk,
    }

@app.route("/")
def index():
    """Serve the main UI page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    """
    POST /predict
    Expects: multipart/form-data with field 'file' containing an MRI image.
    Returns: JSON with prediction results.
    """

    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Upload JPG, JPEG, or PNG."}), 400

    ext      = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        result = predict(filepath)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    result["image_url"] = f"/static/uploads/{filename}"
    return jsonify(result)


@app.route("/static/uploads/<filename>")
def uploaded_file(filename):
    """Serve uploaded images back to the browser."""
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    print("🧠 Brain Tumor Detection App")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
