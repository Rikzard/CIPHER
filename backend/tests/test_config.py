import os
import unittest
from backend.config import ConfigurationError, Settings


class TestConfig(unittest.TestCase):
    def test_valid_configuration(self) -> None:
        settings = Settings(app_name="CustomCIPHER", app_version="1.2.3", environment="staging")
        self.assertEqual(settings.app_name, "CustomCIPHER")
        self.assertEqual(settings.app_version, "1.2.3")
        self.assertEqual(settings.environment, "staging")

    def test_configuration_from_environment(self) -> None:
        os.environ["CIPHER_APP_NAME"] = "EnvCIPHER"
        os.environ["CIPHER_APP_VERSION"] = "2.0.0"
        os.environ["CIPHER_ENVIRONMENT"] = "production"
        try:
            settings = Settings.from_environment()
            self.assertEqual(settings.app_name, "EnvCIPHER")
            self.assertEqual(settings.app_version, "2.0.0")
            self.assertEqual(settings.environment, "production")
        finally:
            os.environ.pop("CIPHER_APP_NAME", None)
            os.environ.pop("CIPHER_APP_VERSION", None)
            os.environ.pop("CIPHER_ENVIRONMENT", None)

    def test_invalid_configuration_empty_app_name(self) -> None:
        os.environ["CIPHER_APP_NAME"] = "   "
        try:
            with self.assertRaises(ConfigurationError):
                Settings.from_environment()
        finally:
            os.environ.pop("CIPHER_APP_NAME", None)

    def test_invalid_configuration_empty_app_version(self) -> None:
        os.environ["CIPHER_APP_VERSION"] = ""
        try:
            with self.assertRaises(ConfigurationError):
                Settings.from_environment()
        finally:
            os.environ.pop("CIPHER_APP_VERSION", None)

    def test_invalid_configuration_empty_environment(self) -> None:
        os.environ["CIPHER_ENVIRONMENT"] = " \t "
        try:
            with self.assertRaises(ConfigurationError):
                Settings.from_environment()
        finally:
            os.environ.pop("CIPHER_ENVIRONMENT", None)


if __name__ == "__main__":
    unittest.main()
