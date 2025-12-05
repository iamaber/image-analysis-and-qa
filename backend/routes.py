import asyncio
import base64
import os
from functools import lru_cache
from io import BytesIO
from typing import Any

from dotenv import load_dotenv
from fastapi import File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.settings import ModelSettings

load_dotenv()
os.environ.setdefault("YOLO_CONFIG_DIR", ".ultralytics")
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)

from ultralytics import YOLO


class DetectionRecord(BaseModel):
    class_name: str
    bounding_box: list[float] = Field(min_length=4, max_length=4)
    confidence: float


class QueryRequest(BaseModel):
    query: str = Field(max_length=2000)
    detections: list[DetectionRecord]


model_path = os.getenv("YOLO_MODEL_PATH", "yolo11m.pt")
max_upload_bytes = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
upload_chunk_size = 1024 * 1024


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
    annotated_image = Image.fromarray(results[0].plot())
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
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not configured")

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

    annotated_image, detections = await asyncio.to_thread(detect_objects, input_image)
    image_base64 = encode_image(annotated_image)
    return {"image": f"data:image/png;base64,{image_base64}", "detections": detections}


async def query_rag(request: QueryRequest) -> dict[str, Any]:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if not request.detections:
        raise HTTPException(
            status_code=400,
            detail="Detection data is required before asking questions",
        )

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
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return {"data": result.output, "sources": []}
