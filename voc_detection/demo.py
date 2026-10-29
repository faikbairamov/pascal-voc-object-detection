"""Local FastAPI application with image uploads and a documented response schema."""
import argparse
import base64
import io
import threading
import warnings
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from PIL import Image, UnidentifiedImageError

from .inference import DEFAULT_CHECKPOINT, load_detector, predict_image, sha256

MAX_BYTES = 10 * 1024 * 1024


class Detection(BaseModel):
    label: str
    class_id: int = Field(ge=1, le=20)
    score: float = Field(ge=0, le=1)
    box: list[float] = Field(min_length=4, max_length=4)


class Prediction(BaseModel):
    width: int
    height: int
    threshold: float
    inference_seconds: float
    detections: list[Detection]
    box_format: str
    annotated_image: str = Field(description="PNG as a base64 data URL")


def create_app(checkpoint=DEFAULT_CHECKPOINT, detector=None):
    app = FastAPI(title="Pascal VOC Object Detection", version="1.0.0",
                  description="CPU inference for 20 VOC categories. Upload a photo to /predict.")
    if detector is None:
        model, metadata = load_detector(checkpoint)
        provenance = {"model_sha256": sha256(checkpoint),
                      "source_run": metadata.get("mlflow_run_id"), "epoch": metadata.get("epoch"),
                      "inference_config": metadata.get("inference_config", {
                          "min_size": 320, "max_size": 640, "rpn_proposals": 150})}
    else:
        model = detector
        provenance = {}
    lock = threading.Lock()

    @app.middleware("http")
    async def limit_request(request: Request, call_next):
        length = request.headers.get("content-length")
        if length:
            try:
                if int(length) > MAX_BYTES + 65536:
                    return JSONResponse({"error": "Upload an image smaller than 10 MB."}, status_code=413)
            except ValueError:
                return JSONResponse({"error": "Invalid content length."}, status_code=400)
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exception):
        return JSONResponse({"error": "Provide an image and a confidence threshold between 0 and 1."},
                            status_code=422)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home():
        return (Path(__file__).parent / "templates/demo.html").read_text()

    @app.get("/health")
    def health():
        return {"status": "ready", "classes": 20, "device": "cpu", **provenance}

    @app.post("/predict", response_model=Prediction,
              responses={400: {"description": "Invalid image"},
                         413: {"description": "Image exceeds 10 MB"}})
    def predict(image: Annotated[UploadFile, File(description="JPEG, PNG, or another Pillow-readable image")],
                threshold: Annotated[float, Form(ge=0, le=1)] = 0.5):
        data = image.file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            return JSONResponse({"error": "Upload an image smaller than 10 MB."}, status_code=413)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as source:
                    if source.width * source.height > 20_000_000:
                        return JSONResponse({"error": "Use an image with at most 20 million pixels."}, status_code=400)
                    photo = source.copy()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
                Image.DecompressionBombWarning, ValueError):
            return JSONResponse({"error": "The file could not be read as an image."}, status_code=400)
        with lock:
            annotated, result = predict_image(model, photo, threshold)
        buffer = io.BytesIO()
        annotated.save(buffer, format="PNG")
        result["annotated_image"] = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        return result

    return app


def main():
    parser = argparse.ArgumentParser(description="Local FastAPI Pascal VOC demo")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    uvicorn.run(create_app(args.checkpoint), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
