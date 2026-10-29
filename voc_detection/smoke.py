"""Exercise a running FastAPI service with real image bytes."""
import argparse
import base64
import io
import json
from pathlib import Path

import httpx
from PIL import Image

from .demo import Prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with httpx.Client(base_url=args.url, timeout=60) as client:
        health = client.get("/health")
        health.raise_for_status()
        assert health.json()["status"] == "ready"
        assert client.get("/docs").status_code == 200
        assert "/predict" in client.get("/openapi.json").json()["paths"]
        with args.image.open("rb") as source:
            response = client.post("/predict", files={"image": (args.image.name, source, "image/jpeg")},
                                   data={"threshold": "0.5"})
        response.raise_for_status()
        result = Prediction.model_validate(response.json()).model_dump()
        photo = Image.open(io.BytesIO(base64.b64decode(result.pop("annotated_image").split(",", 1)[1])))
        assert photo.size == (result["width"], result["height"])
        assert client.post("/predict", files={"image": ("invalid.jpg", b"bad")}).status_code == 400
        assert client.post("/predict").status_code == 422
    evidence = {"url": args.url, "health": health.json(), "real_image": args.image.name,
                "checks": ["health", "docs", "OpenAPI contract", "real-model prediction",
                           "typed response", "valid annotated PNG", "invalid image", "missing image"],
                "prediction": result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
