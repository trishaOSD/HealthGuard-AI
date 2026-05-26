from flask import Flask, request, jsonify, render_template
from model import predict_disease

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    features = data["features"]

    disease, confidence = predict_disease(features)

    return jsonify({
        "result": disease,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)