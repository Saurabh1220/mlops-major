from flask import Flask, request, render_template, redirect, url_for, flash
import joblib
import numpy as np
import os

app = Flask(__name__)
app.secret_key = "replace-me"

MODEL_PATH = "models/savedmodel.pth"

def load_model():
    # Load model artifact (raises if missing)
    data = joblib.load(MODEL_PATH)
    return data["model"]

# Load the model at import/startup time so Gunicorn workers have it available
model = None
try:
    if not os.path.exists(MODEL_PATH):
        # give a clear message for debugging
        raise FileNotFoundError(f"Model not found at {MODEL_PATH} — run train.py first or copy models/savedmodel.pth")
    model = load_model()
    app.logger.info("Model loaded at startup.")
except Exception as e:
    # If you want the container to fail fast, re-raise
    app.logger.error(f"Failed to load model at startup: {e}")
    raise

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
def predict():
    f = request.files.get("file")
    if not f:
        flash("No file uploaded")
        return redirect(url_for("index"))
    try:
        arr = np.load(f)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        pred = model.predict(arr)
        return f"Predicted class: {int(pred[0])}"
    except Exception as e:
        app.logger.exception("Prediction failed")
        return f"Error: {e}", 400

if __name__ == "__main__":
    # For local debug (not used by Gunicorn in container)
    app.run(host="0.0.0.0", port=5000)
