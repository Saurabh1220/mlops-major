from flask import Flask, request, render_template, redirect, url_for, flash
import joblib
import numpy as np
import os

app = Flask(__name__)
app.secret_key = "replace-me"

MODEL_PATH = "models/savedmodel.pth"

def load_model():
    data = joblib.load(MODEL_PATH)
    return data["model"]

model = None

@app.before_first_request
def startup():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run training to create models/savedmodel.pth")
    model = load_model()

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
        return f"Error: {e}", 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
