import importlib
import os
import unittest
from unittest import mock


class AuthConfigurationTests(unittest.TestCase):
    def test_import_does_not_install_public_default_keys(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            import api.auth as auth

            importlib.reload(auth)
            self.assertEqual(auth.API_KEYS_STORE, {})

    def test_bootstrap_key_is_explicit(self):
        env = {"SYNC_BOOTSTRAP_API_KEYS": "local-test:secret-value:42"}
        with mock.patch.dict(os.environ, env, clear=True):
            import api.auth as auth

            importlib.reload(auth)
            key = auth.verify_api_key("secret-value")
            self.assertIsNotNone(key)
            self.assertEqual(key.name, "local-test")
            self.assertEqual(key.rate_limit_per_minute, 42)


if __name__ == "__main__":
    unittest.main()
