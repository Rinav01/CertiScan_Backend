import io
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.main import app


class PredictRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("backend.routes.predict.get_model_version", return_value="unet_finetuned_v2.pth")
    @patch("backend.routes.predict.calculate_confidence", return_value=0.25)
    @patch("backend.routes.predict.run_inference", return_value=np.ones((224, 224), dtype=np.float32))
    def test_predict_returns_frontend_friendly_payload(self, *_mocks):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)

        response = self.client.post(
            "/predict",
            files={"file": ("sample.png", io.BytesIO(encoded.tobytes()), "image/png")},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["prediction"], "Fake")
        self.assertEqual(payload["threshold_used"], 0.1)
        self.assertEqual(payload["model_version"], "unet_finetuned_v2.pth")
        self.assertTrue(payload["mask_path"].startswith("outputs/"))
        self.assertTrue(payload["mask_url"].endswith(payload["mask_path"]))

    def test_predict_rejects_non_image_uploads(self):
        response = self.client.post(
            "/predict",
            files={"file": ("sample.txt", io.BytesIO(b"hello"), "text/plain")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Only image uploads are supported.")


if __name__ == "__main__":
    unittest.main()
