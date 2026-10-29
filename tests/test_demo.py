"""API contract checks, including input failures and response image validity."""
import base64
import io
import unittest

import torch
from fastapi.testclient import TestClient
from PIL import Image

from voc_detection.demo import create_app


class FixtureDetector(torch.nn.Module):
    def forward(self, images):
        return [{"boxes": torch.tensor([[1., 2., 12., 15.]]),
                 "labels": torch.tensor([15]), "scores": torch.tensor([.8])}]


class DemoContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(create_app(detector=FixtureDetector().eval()))
        buffer = io.BytesIO()
        Image.new("RGB", (32, 24), "white").save(buffer, format="PNG")
        cls.image = buffer.getvalue()

    def test_home_health_and_openapi(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/health").json()["status"], "ready")
        schema = self.client.get("/openapi.json").json()
        self.assertIn("/predict", schema["paths"])
        self.assertEqual(self.client.get("/docs").status_code, 200)

    def test_success_returns_boxes_and_a_valid_image(self):
        response = self.client.post("/predict", files={"image": ("a.png", self.image, "image/png")})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["detections"][0]["label"], "person")
        self.assertEqual(result["detections"][0]["box"], [1, 2, 12, 15])
        image = Image.open(io.BytesIO(base64.b64decode(result["annotated_image"].split(",", 1)[1])))
        self.assertEqual(image.size, (32, 24))

    def test_threshold_can_produce_no_detections(self):
        response = self.client.post("/predict", files={"image": ("a.png", self.image)},
                                    data={"threshold": "0.9"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["detections"], [])

    def test_invalid_inputs(self):
        self.assertEqual(self.client.post("/predict").status_code, 422)
        for threshold in ("-1", "2", "nan", "bad"):
            response = self.client.post("/predict", files={"image": ("a.png", self.image)},
                                        data={"threshold": threshold})
            self.assertEqual(response.status_code, 422)
        response = self.client.post("/predict", files={"image": ("a.png", b"invalid")})
        self.assertEqual(response.status_code, 400)

    def test_oversized_upload_is_rejected(self):
        response = self.client.post("/predict", files={"image": ("big.png", b"x" * (11 * 1024 * 1024))})
        self.assertEqual(response.status_code, 413)


if __name__ == "__main__":
    unittest.main()
