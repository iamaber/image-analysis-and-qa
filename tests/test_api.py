import asyncio
import io
import logging
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

import main
from backend import routes
from backend.app import app

client = TestClient(app)
main_client = TestClient(main.app)


class FakeValue:
    def __init__(self, value: Any) -> None:
        self._value = value

    def tolist(self) -> list[float]:
        return list(self._value)

    def item(self) -> float:
        return self._value


class FakeBox:
    def __init__(
        self,
        class_index: int = 0,
        bounding_box: tuple[float, float, float, float] = (1.0, 2.0, 3.0, 4.0),
        confidence: float = 0.9,
    ) -> None:
        self.cls = class_index
        self.xyxy = [FakeValue(bounding_box)]
        self.conf = FakeValue(confidence)


class FakeResult:
    def __init__(
        self,
        names: dict[int, str],
        boxes: list[FakeBox],
        plotted: Any = None,
    ) -> None:
        self.names = names
        self.boxes = boxes
        self._plotted = plotted

    def plot(self) -> Any:
        return self._plotted


def make_png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def clear_caches() -> None:
    routes.get_detector.cache_clear()
    routes.get_query_agent.cache_clear()
    routes.query_request_counts.clear()


def test_build_detection_context() -> None:
    detections = [
        routes.DetectionRecord(
            class_name="person",
            bounding_box=[1.0, 2.0, 3.0, 4.0],
            confidence=0.97,
        )
    ]

    context = routes.build_detection_context(detections)

    assert "Detected person with confidence 0.97" in context
    assert "[1.0, 2.0, 3.0, 4.0]" in context


def test_serialize_detections() -> None:
    results = [
        FakeResult(
            names={0: "person", 1: "dog"},
            boxes=[FakeBox(0), FakeBox(1, (5.0, 6.0, 7.0, 8.0), 0.5)],
        )
    ]

    detections = routes.serialize_detections(results)

    assert detections == [
        {
            "class_name": "person",
            "bounding_box": [1.0, 2.0, 3.0, 4.0],
            "confidence": 0.9,
        },
        {
            "class_name": "dog",
            "bounding_box": [5.0, 6.0, 7.0, 8.0],
            "confidence": 0.5,
        },
    ]


def test_health_reports_qa_disabled_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "qa_configured": False}


def test_health_reports_qa_enabled_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "qa_configured": True}


def test_query_returns_503_without_groq_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    response = client.post(
        "/query",
        json={
            "query": "How many people are there?",
            "detections": [
                {
                    "class_name": "person",
                    "bounding_box": [1.0, 2.0, 3.0, 4.0],
                    "confidence": 0.97,
                }
            ],
        },
    )

    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]


def test_query_rejects_empty_query() -> None:
    response = client.post(
        "/query",
        json={"query": "   ", "detections": []},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Query cannot be empty"


def test_query_requires_detections() -> None:
    response = client.post("/query", json={"query": "How many?", "detections": []})

    assert response.status_code == 400
    assert "Detection data is required" in response.json()["detail"]


def test_query_rejects_too_many_detections() -> None:
    detections = [
        {
            "class_name": "person",
            "bounding_box": [1.0, 2.0, 3.0, 4.0],
            "confidence": 0.9,
        }
        for _ in range(201)
    ]

    response = client.post(
        "/query",
        json={"query": "How many?", "detections": detections},
    )

    assert response.status_code == 422


def test_query_rate_limit_returns_429(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(routes, "query_rate_limit", 2)

    payload = {
        "query": "How many?",
        "detections": [
            {
                "class_name": "person",
                "bounding_box": [1.0, 2.0, 3.0, 4.0],
                "confidence": 0.9,
            }
        ],
    }

    class FakeAgent:
        async def run(self, prompt: str) -> Any:
            return type("Result", (), {"output": "One person."})()

    monkeypatch.setattr(routes, "get_query_agent", lambda: FakeAgent())

    assert client.post("/query", json=payload).status_code == 200
    assert client.post("/query", json=payload).status_code == 200

    throttled = client.post("/query", json=payload)

    assert throttled.status_code == 429
    assert "Retry-After" in throttled.headers


QUERY_PAYLOAD = {
    "query": "How many?",
    "detections": [
        {
            "class_name": "person",
            "bounding_box": [1.0, 2.0, 3.0, 4.0],
            "confidence": 0.9,
        }
    ],
}


def test_query_returns_error_when_groq_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic_ai.exceptions import ModelHTTPError

    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    class FailingAgent:
        async def run(self, prompt: str) -> Any:
            raise ModelHTTPError(status_code=401, model_name="openai/gpt-oss-20b")

    monkeypatch.setattr(routes, "get_query_agent", lambda: FailingAgent())

    response = client.post("/query", json=QUERY_PAYLOAD)

    assert response.status_code == 502
    assert "temporarily unavailable" in response.json()["detail"]


def test_query_maps_groq_server_error_to_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic_ai.exceptions import ModelHTTPError

    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    class FailingAgent:
        async def run(self, prompt: str) -> Any:
            raise ModelHTTPError(status_code=503, model_name="openai/gpt-oss-20b")

    monkeypatch.setattr(routes, "get_query_agent", lambda: FailingAgent())

    response = client.post("/query", json=QUERY_PAYLOAD)

    assert response.status_code == 503


def test_query_returns_502_on_unexpected_agent_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    class FailingAgent:
        async def run(self, prompt: str) -> Any:
            raise RuntimeError("socket closed")

    monkeypatch.setattr(routes, "get_query_agent", lambda: FailingAgent())

    response = client.post("/query", json=QUERY_PAYLOAD)

    assert response.status_code == 502
    assert "check the server logs" in response.json()["detail"].lower()


def test_detect_rejects_non_image_content_type() -> None:
    response = client.post(
        "/detect",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file must be an image"


def test_detect_rejects_missing_content_type() -> None:
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(routes.detect_endpoint(file=SimpleNamespace(content_type=None)))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Uploaded file must be an image"


def test_detect_rejects_invalid_image_bytes() -> None:
    response = client.post(
        "/detect",
        files={"file": ("broken.png", b"not-an-image", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file must be an image"


def test_detect_rejects_images_over_pixel_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes, "max_image_pixels", 16)

    response = client.post(
        "/detect",
        files={"file": ("image.png", make_png(), "image/png")},
    )

    assert response.status_code == 413
    assert "pixel limit" in response.json()["detail"]


def test_detect_maps_decompression_bomb_to_413(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 4)

    response = client.post(
        "/detect",
        files={"file": ("image.png", make_png(), "image/png")},
    )

    assert response.status_code == 413
    assert "pixel limit" in response.json()["detail"]


def test_detect_enforces_upload_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes, "max_upload_bytes", 8)

    response = client.post(
        "/detect",
        files={"file": ("image.png", make_png(), "image/png")},
    )

    assert response.status_code == 413
    assert "exceeds" in response.json()["detail"]


def test_detect_returns_annotated_image_and_detections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_detect(image: Image.Image) -> tuple[Image.Image, list[dict[str, Any]]]:
        return image, [
            {
                "class_name": "person",
                "bounding_box": [1.0, 2.0, 3.0, 4.0],
                "confidence": 0.97,
            }
        ]

    monkeypatch.setattr(routes, "detect_objects", fake_detect)

    response = client.post(
        "/detect",
        files={"file": ("image.png", make_png(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["image"].startswith("data:image/png;base64,")
    assert payload["detections"] == [
        {
            "class_name": "person",
            "bounding_box": [1.0, 2.0, 3.0, 4.0],
            "confidence": 0.97,
        }
    ]


def test_detect_returns_503_when_model_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_detect(image: Image.Image) -> None:
        raise RuntimeError("model weights missing")

    monkeypatch.setattr(routes, "detect_objects", failing_detect)

    response = client.post(
        "/detect",
        files={"file": ("image.png", make_png(), "image/png")},
    )

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]


def test_detect_converts_plot_bgr_output_to_rgb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    bgr_red = np.zeros((4, 4, 3), dtype=np.uint8)
    bgr_red[..., 2] = 255

    class FakeModel:
        def __call__(self, image: Image.Image) -> list[FakeResult]:
            return [FakeResult(names={0: "person"}, boxes=[], plotted=bgr_red)]

    monkeypatch.setattr(routes, "get_detector", lambda: FakeModel())

    annotated, detections = routes.detect_objects(Image.new("RGB", (4, 4)))

    assert detections == []
    assert annotated.getpixel((0, 0)) == (255, 0, 0)


def test_query_success_returns_agent_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    class FakeAgent:
        def __init__(self) -> None:
            self.prompt = ""

        async def run(self, prompt: str) -> Any:
            self.prompt = prompt
            return type("Result", (), {"output": "There is 1 person."})()

    agent = FakeAgent()
    monkeypatch.setattr(routes, "get_query_agent", lambda: agent)

    response = client.post("/query", json=QUERY_PAYLOAD)

    assert response.status_code == 200
    assert response.json() == {"data": "There is 1 person.", "sources": []}
    assert "Detected person with confidence 0.90" in agent.prompt
    assert "Query: How many?" in agent.prompt


def test_query_logs_unexpected_agent_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    class FailingAgent:
        async def run(self, prompt: str) -> Any:
            raise RuntimeError("socket closed")

    monkeypatch.setattr(routes, "get_query_agent", lambda: FailingAgent())

    with caplog.at_level(logging.ERROR):
        client.post("/query", json=QUERY_PAYLOAD)

    assert "Question answering failed" in caplog.text


def test_root_serves_application() -> None:
    with main_client:
        response = main_client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_application_redirects_to_root() -> None:
    with main_client:
        response = main_client.get("/application", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/"


def test_static_mount_serves_frontend() -> None:
    with main_client:
        response = main_client.get("/static/application.html")

    assert response.status_code == 200
    assert b"<" in response.content
