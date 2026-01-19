# NLP Misinformation Detector

A machine learning-based tool to detect misinformation and "fake news" using NLP techniques. This project provides a training pipeline to build a classifier and a Flask-based API to analyze text credibility in real-time.

## Features

- **Training Pipeline**: This uses a `PassiveAggressiveClassifier` with TF-IDF vectorization to achieve high accuracy on news datasets.
- **REST API**: This is a Flask application that provides endpoints for status checks and text analysis.
- **Credibility Scoring**: This converts model decision scores into a 0-100 credibility scale with risk assessments.


## Installation

Clone the github repo:
```bash
git clone https://github.com/jai-git4208/nlp-misinformation-detector.git
```
please create a venv and activate it

To install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Model
Run the main pipeline to load data, train the model, and evaluate performance:
```bash
python3 main.py
```

### 2. Start the API
Once the model is trained, start the Flask server:
```bash
python3 app.py
```

### 3. Analyze Text
Send a POST request to the `/analyze` endpoint:
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"text": "Your news snippet here..."}' \
     http://127.0.0.1:5000/analyze
```


## API Endpoints

- `GET /status`: Check if the service is running.
- `POST /analyze`: Analyze text credibility. Requires a JSON body with a `"text"` field.

## Dataset
The project expects `True.csv` and `Fake.csv` in the `dataset/` directory. Each file should contain a `text` column.

## Automated Testing

Automated tests were implemented using `pytest` and Flask’s test client.

The tests validate:
- API health endpoint
- Model inference for credible and fake news
- Risk classification logic
- Invalid input handling

Advanced runtime logging captures:
- Request flow
- Model decision scores
- Credibility normalization
- Risk assessment

Test reports are generated in JUnit XML format (`pytest_report.xml`) for reproducibility.

### With love by JAI and KAMAL