
from flask import Flask, request, jsonify
import numpy as np


from main import load_data, train_model

app = Flask(__name__)


print("Initializing model...")
df = load_data()
if df is None:
    raise RuntimeError("Dataset not found. Cannot start API.")

model, vectorizer = train_model(df)
print("Model loaded successfully.")


def compute_credibility(decision_score):
    """
    Convert decision_function output to 0–100 score
    """
    normalized = 1 / (1 + np.exp(-decision_score))  # sigmoid
    return round(normalized * 100, 2)


@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "status": "running",
        "service": "Fake News Credibility API"
    })

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"]

    if not isinstance(text, str) or len(text.strip()) == 0:
        return jsonify({"error": "Invalid text input"}), 400


    text_vec = vectorizer.transform([text])

    decision_score = model.decision_function(text_vec)[0]
    credibility_score = compute_credibility(decision_score)
    

   
    if credibility_score >= 30:
        risk = "Low Risk (High Confidence)"
    elif credibility_score >= 20:
        risk = "Moderate Risk (Medium Confidence)"
    else:
        risk = "High Risk (Low Confidence)"


    return jsonify({
        "credibility_score": credibility_score,
        "assessment": risk
    })


if __name__ == "__main__":
    app.run(debug=True)
