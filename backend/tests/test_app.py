import unittest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.config import Settings


class TestApp(unittest.TestCase):
    def test_application_initialization_with_explicit_settings(self) -> None:
        custom_settings = Settings(app_name="ExplicitApp", app_version="3.0.0", environment="testing")
        app = create_app(custom_settings, detectors=[])
        self.assertEqual(app.title, "ExplicitApp")
        self.assertEqual(app.version, "3.0.0")
        self.assertEqual(app.state.settings, custom_settings)

    def test_global_500_exception_handler(self) -> None:
        app = create_app(detectors=[])

        @app.get("/trigger-error")
        def trigger_error() -> None:
            raise RuntimeError("Secret internal failure message")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/trigger-error")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Internal server error"})
        self.assertNotIn("Secret internal failure message", response.text)


if __name__ == "__main__":
    unittest.main()
