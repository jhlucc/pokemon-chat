import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from server.main import app


class TestAPIEndpoints(unittest.TestCase):
    def test_healthz(self):
        client = TestClient(app)
        resp = client.get("/healthz")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json().get("status"), "ok")

    def test_readyz_shape(self):
        client = TestClient(app)
        resp = client.get("/readyz")
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertIn("status", payload)
        self.assertIn("checks", payload)
        self.assertIn("features", payload)
        self.assertIn("warnings", payload)

    def test_config_roundtrip_uses_tempfile(self):
        # Avoid writing UI config into the repo during tests.
        with tempfile.TemporaryDirectory() as d:
            cfg_path = Path(d) / "ui_config.json"
            with patch("server.runtime_config._config_file", return_value=cfg_path):
                client = TestClient(app)
                resp = client.get("/config")
                self.assertEqual(resp.status_code, 200)
                self.assertIn("model_provider", resp.json())

                resp2 = client.patch("/config", json={"model_provider": "siliconflow"})
                self.assertEqual(resp2.status_code, 200)
                self.assertEqual(resp2.json().get("model_provider"), "siliconflow")

    def test_restart_endpoint(self):
        client = TestClient(app)
        resp = client.post("/restart")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json().get("message"), "Restarted!")

    def test_admin_runtime_reset_no_auth_when_key_empty(self):
        # Admin endpoints are only protected when ADMIN_API_KEY is set.
        prev = os.environ.get("ADMIN_API_KEY")
        try:
            os.environ["ADMIN_API_KEY"] = ""
            client = TestClient(app)
            resp = client.post("/admin/runtime/reset")
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(resp.json().get("success"))
        finally:
            if prev is None:
                os.environ.pop("ADMIN_API_KEY", None)
            else:
                os.environ["ADMIN_API_KEY"] = prev


if __name__ == "__main__":
    unittest.main()
