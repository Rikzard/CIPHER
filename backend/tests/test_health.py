from fastapi.testclient import TestClient

from backend.app import create_app
from backend.config import Settings


def test_health_returns_service_status() -> None:
    settings = Settings(app_name="CIPHER Test", app_version="9.8.7", environment="test")
    client = TestClient(create_app(settings))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "CIPHER Test",
        "version": "9.8.7",
        "environment": "test",
    }
