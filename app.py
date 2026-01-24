
from flask import Flask, request, jsonify, render_template
import numpy as np


from main import load_data, train_model, load_saved_model, save_model

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/docs", methods=["GET"])
def docs():
    return render_template("docs.html")

@app.route("/api", methods=["GET"])
def api():
    return render_template("api.html")

@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")





print("Initializing model...")

#first check if model in model xd
model, vectorizer = load_saved_model()

if model is None or vectorizer is None:
    print("no model found")
    df = load_data()
    if df is None:
        raise RuntimeError("Dataset not found. Cannot start API.")

    model, vectorizer = train_model(df, save=True)
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
    logging.info("Analyze endpoint called")

    data = request.get_json()

    if not data or "text" not in data:
        logging.warning("Invalid request: missing text field")
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"]

    if not isinstance(text, str) or len(text.strip()) == 0:
        logging.warning("Invalid request: empty or non-string text")
        return jsonify({"error": "Invalid text input"}), 400

    logging.info(f"Input text length: {len(text)}")

    text_vec = vectorizer.transform([text])

    decision_score = model.decision_function(text_vec)[0]
    credibility_score = compute_credibility(decision_score)

    logging.info(f"Decision score: {decision_score}")
    logging.info(f"Credibility score: {credibility_score}")

    if credibility_score >= 70:
        risk = "Low Risk (High Credibility)"
    elif credibility_score >= 40:
        risk = "Moderate Risk (Medium Credibility)"
    else:
        risk = "High Risk (Low Credibility)"

    logging.info(f"Risk assessment: {risk}")

    return jsonify({
        "credibility_score": credibility_score,
        "assessment": risk
    })

if __name__ == "__main__":
    app.run()

