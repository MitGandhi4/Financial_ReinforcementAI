from fastapi.testclient import TestClient

from src.api.main import app
from src.api.schemas import BacktestRequest

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_train_request_validates_timestep_limit():
    response = client.post("/train", json={"ticker": "aapl", "total_timesteps": -1})
    assert response.status_code == 422


def test_ticker_requests_are_normalized():
    assert BacktestRequest(ticker=" aapl ").ticker == "AAPL"
