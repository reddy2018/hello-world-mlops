from flask import Flask, request, jsonify
import joblib 
import os
from pathlib import Path # path helps with file paths across OS

app = Flask(__name__) # create Flask app
MODEL_PATH = Path("artifacts/model.pkl")

if not MODEL_PATH.exists():
    # convenience: train if model missing
    import train as _train
    _train.main()

model = joblib.load(MODEL_PATH) # load model from artifacts/model.pkl

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}) # return JSON response with status ok

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() # get JSON data from request
    if not data or "features" not in data:
        return jsonify({"error": "send JSON with key 'features'"}), 400
    features = data["features"]
    try:
        pred = model.predict([features])
        return jsonify({"prediction": int(pred[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500 

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)