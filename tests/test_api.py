import json
import pytest
from app import app

print("TEST FILE LOADED")


@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client


def test_status_endpoint(client):
    response = client.get("/status")
    assert response.status_code == 200

    data = response.get_json()
    assert data["status"] == "running"
    assert data["service"] == "Fake News Credibility API"


def test_analyze_low_or_moderate_risk(client):
    payload = {
        "text": "The Reserve Bank of India released its official monetary policy statement."
    }

    response = client.post(
        "/analyze",
        data=json.dumps(payload),
        content_type="application/json"
    )

    assert response.status_code == 200
    data = response.get_json()

    assert "credibility_score" in data
    assert data["assessment"] in [
        "Low Risk (High Credibility)",
        "Moderate Risk (Medium Credibility)"
    ]


def test_analyze_high_risk(client):
    payload = {
        "text": "SHOCKING SECRET they don't want you to know!!!"
    }

    response = client.post(
        "/analyze",
        data=json.dumps(payload),
        content_type="application/json"
    )

    assert response.status_code == 200
    data = response.get_json()

    assert data["assessment"] == "High Risk (Low Credibility)"


def test_analyze_invalid_input(client):
    response = client.post(
        "/analyze",
        data=json.dumps({}),
        content_type="application/json"
    )

    assert response.status_code == 400
