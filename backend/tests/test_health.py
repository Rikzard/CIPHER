import unittest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.health import router as health_router
from backend.app import create_app
from backend.config import Settings


class TestHealth(unittest.TestCase):
    def test_health_returns_service_status(self) -> None:
        settings = Settings(app_name="CIPHER Test", app_version="9.8.7", environment="test")
        client = TestClient(create_app(settings, detectors=[]))

        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "ok",
                "service": "CIPHER Test",
                "version": "9.8.7",
                "environment": "test",
            },
        )

    def test_health_uninitialized_app_state_fails_cleanly(self) -> None:
        """Verify that uninitialized app state fails cleanly with 500 error."""
        uninitialized_app = FastAPI()
        uninitialized_app.include_router(health_router)
        client = TestClient(uninitialized_app)

        response = client.get("/health")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Application settings not initialized"})


if __name__ == "__main__":
    unittest.main()
