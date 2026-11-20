import io
import os
import json
import numpy as np
from flask import Flask, request, jsonify, send_file
from PIL import Image
from tensorflow import keras

IMG_SIZE = (128, 128)  # must match training size

app = Flask(__name__)
model = keras.models.load_model("waste_mobilenet.keras")
with open("class_names.json") as f:
    class_names = json.load(f)


@app.route("/")
def home():
    return send_file("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None:
        return jsonify(error="No image uploaded"), 400

    img = Image.open(io.BytesIO(file.read())).convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img, dtype="float32"), axis=0)
    probs = model.predict(arr, verbose=0)[0]

    order = np.argsort(probs)[::-1]
    return jsonify(
        label=class_names[int(order[0])],
        confidence=float(probs[order[0]]),
        probs=[{"label": class_names[int(i)], "prob": float(probs[i])} for i in order],
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
