import io
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import routes
from backend.app import app

client = TestClient(app)


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
    def __init__(self, names: dict[int, str], boxes: list[FakeBox]) -> None:
        self.names = names
        self.boxes = boxes


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


def test_detect_rejects_non_image_content_type() -> None:
    response = client.post(
        "/detect",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file must be an image"


def test_detect_rejects_invalid_image_bytes() -> None:
    response = client.post(
        "/detect",
        files={"file": ("broken.png", b"not-an-image", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file must be an image"


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
