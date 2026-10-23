import asyncio
import base64
import logging
import os
import tempfile
import time
from collections import deque
from functools import lru_cache
from io import BytesIO
from typing import Any

from dotenv import load_dotenv
from fastapi import File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.settings import ModelSettings

logger = logging.getLogger(__name__)

load_dotenv()


def prepare_yolo_config_dir() -> str:
    configured = os.getenv("YOLO_CONFIG_DIR", ".ultralytics")
    try:
        os.makedirs(configured, exist_ok=True)
        return configured
    except OSError:
        fallback = os.path.join(tempfile.gettempdir(), "ultralytics")
        os.makedirs(fallback, exist_ok=True)
        return fallback


os.environ["YOLO_CONFIG_DIR"] = prepare_yolo_config_dir()

from ultralytics import YOLO


class DetectionRecord(BaseModel):
    class_name: str = Field(max_length=100)
    bounding_box: list[float] = Field(min_length=4, max_length=4)
    confidence: float


class QueryRequest(BaseModel):
    query: str = Field(max_length=2000)
    detections: list[DetectionRecord] = Field(max_length=200)


class QAUnavailableError(RuntimeError):
    pass


QA_UNAVAILABLE_DETAIL = (
    "Question answering is not configured on this deployment. Set the "
    "GROQ_API_KEY environment variable and restart the app to enable it. "
    "Object detection is still available."
)

model_path = os.getenv("YOLO_MODEL_PATH", "yolo11m.pt")
max_upload_bytes = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
upload_chunk_size = 1024 * 1024
query_rate_limit = int(os.getenv("QUERY_RATE_LIMIT", "10"))
query_rate_window_seconds = int(os.getenv("QUERY_RATE_WINDOW_SECONDS", "60"))
query_request_counts: dict[str, deque[float]] = {}
_query_requests_since_cleanup = 0


def enforce_query_rate_limit(client_host: str | None) -> None:
    """Allow query_rate_limit requests per client within a sliding window."""
    if query_rate_limit <= 0:
        return

    global _query_requests_since_cleanup
    now = time.monotonic()
    key = client_host or "unknown"
    window_start = now - query_rate_window_seconds

    if _query_requests_since_cleanup >= 1000:
        for stale_key in [
            existing
            for existing, stamps in query_request_counts.items()
            if not stamps or stamps[-1] <= window_start
        ]:
            del query_request_counts[stale_key]
        _query_requests_since_cleanup = 0

    timestamps = query_request_counts.setdefault(key, deque())
    while timestamps and timestamps[0] <= window_start:
        timestamps.popleft()

    if len(timestamps) >= query_rate_limit:
        retry_after = max(int(timestamps[0] + query_rate_window_seconds - now) + 1, 1)
        raise HTTPException(
            status_code=429,
            detail="Too many questions. Please wait before trying again.",
            headers={"Retry-After": str(retry_after)},
        )

    timestamps.append(now)
    _query_requests_since_cleanup += 1


def qa_is_configured() -> bool:
    return bool(os.getenv("GROQ_API_KEY", "").strip())


def build_detection_context(detections: list[DetectionRecord]) -> str:
    return "\n".join(
        (
            f"Detected {detection.class_name} with confidence "
            f"{detection.confidence:.2f} at bounding box {detection.bounding_box}"
        )
        for detection in detections
    )


def serialize_detections(results: Any) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for result in results:
        for box in result.boxes:
            detections.append(
                {
                    "class_name": result.names[int(box.cls)],
                    "bounding_box": box.xyxy[0].tolist(),
                    "confidence": box.conf.item(),
                }
            )
    return detections


@lru_cache(maxsize=1)
def get_detector() -> YOLO:
    return YOLO(model_path)


def detect_objects(image: Image.Image) -> tuple[Image.Image, list[dict[str, Any]]]:
    results = get_detector()(image)
    plotted = results[0].plot()
    annotated_image = Image.fromarray(plotted[..., ::-1])
    return annotated_image, serialize_detections(results)


def encode_image(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode("utf-8")


async def read_upload_with_limit(file: UploadFile) -> bytes:
    total_bytes = 0
    chunks = bytearray()

    while chunk := await file.read(upload_chunk_size):
        total_bytes += len(chunk)
        if total_bytes > max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded file exceeds the {max_upload_bytes // (1024 * 1024)} MB limit",
            )
        chunks.extend(chunk)

    return bytes(chunks)


@lru_cache(maxsize=1)
def get_query_agent() -> Agent:
    if not qa_is_configured():
        raise QAUnavailableError(QA_UNAVAILABLE_DETAIL)

    model = GroqModel("openai/gpt-oss-20b")
    return Agent(
        model,
        model_settings=ModelSettings(
            temperature=0,
            max_tokens=400,
            groq_reasoning_format="hidden",
            extra_body={"reasoning_effort": "low"},
        ),
    )


async def detect_endpoint(file: UploadFile = File(...)) -> dict[str, Any]:
    if file.content_type is not None and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    contents = await read_upload_with_limit(file)
    try:
        input_image = Image.open(BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError as error:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image",
        ) from error

    try:
        annotated_image, detections = await asyncio.to_thread(detect_objects, input_image)
    except Exception as error:
        logger.exception("Detection failed")
        raise HTTPException(
            status_code=503,
            detail="Detection model is unavailable. Check the server logs.",
        ) from error

    image_base64 = encode_image(annotated_image)
    return {"image": f"data:image/png;base64,{image_base64}", "detections": detections}


async def query_rag(request: QueryRequest, http_request: Request) -> dict[str, Any]:
    enforce_query_rate_limit(http_request.client.host if http_request.client else None)

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if not request.detections:
        raise HTTPException(
            status_code=400,
            detail="Detection data is required before asking questions",
        )

    if not qa_is_configured():
        raise HTTPException(status_code=503, detail=QA_UNAVAILABLE_DETAIL)

    prompt = (
        "Answer questions about image detection results such as object counts, "
        "confidence scores, object names, and bounding boxes. If the query is "
        "unrelated, respond with: 'I can only answer questions about image "
        "detection results.' Respond with only the direct answer. Do not include "
        "reasoning, steps, or explanation.\n\n"
        f"Detection information:\n{build_detection_context(request.detections)}\n\n"
        f"Query: {query}"
    )

    try:
        result = await get_query_agent().run(prompt)
    except QAUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return {"data": result.output, "sources": []}
