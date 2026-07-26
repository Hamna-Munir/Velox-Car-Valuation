import pytest
from fastapi.testclient import TestClient

from api.main import app

VALID_PAYLOAD = {
    "brand": "Maruti", "year": 2021, "km_driven": 40000,
    "fuel": "Petrol", "transmission": "Manual",
    "seller_type": "Individual", "owner": "First Owner",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["n_training_samples"] > 0


def test_brands_endpoint(client):
    r = client.get("/brands")
    assert r.status_code == 200
    assert "Maruti" in r.json()


def test_predict_valid_payload(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_price_inr"] > 0
    assert body["p10"] <= body["p50"] <= body["p90"]
    assert len(body["top_contributions"]) > 0


def test_predict_rejects_invalid_fuel_type(client):
    bad_payload = {**VALID_PAYLOAD, "fuel": "Rocket Fuel"}
    r = client.post("/predict", json=bad_payload)
    assert r.status_code == 422


def test_predict_rejects_future_year_beyond_current(client):
    from velox import config
    bad_payload = {**VALID_PAYLOAD, "year": config.CURRENT_YEAR + 5}
    r = client.post("/predict", json=bad_payload)
    assert r.status_code == 422


def test_predict_rejects_negative_mileage(client):
    bad_payload = {**VALID_PAYLOAD, "km_driven": -100}
    r = client.post("/predict", json=bad_payload)
    assert r.status_code == 422


def test_predict_unknown_brand_falls_back_gracefully(client):
    payload = {**VALID_PAYLOAD, "brand": "SomeNewBrandNotInTraining"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.json()["predicted_price_inr"] > 0
