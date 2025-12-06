# AI Vision Workspace

Single-page image analysis workspace for object detection and question answering. Upload an image, run YOLO detection, then ask follow-up questions grounded in the detection results through a `pydantic-ai` agent backed by Groq's Qwen 3 32B model.

## What Changed

- Authentication and database code were removed.
- The application opens directly into the analysis workspace at `/`.
- `POST /query` is stateless and expects both the user question and the detection data.
- Docker support was removed; the project is intended to run locally.

## Stack

- FastAPI
- Ultralytics YOLO
- `pydantic-ai` with Groq `qwen/qwen3-32b`
- Plain HTML, CSS, and JavaScript frontend

## Environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
YOLO_MODEL_PATH=yolo11m.pt
YOLO_CONFIG_DIR=.ultralytics
```

`YOLO_MODEL_PATH` and `YOLO_CONFIG_DIR` are optional locally. By default the model path is `yolo11m.pt` and Ultralytics config/cache data stays in `.ultralytics/` inside the project.

## Run Locally

```bash
uv sync
uv run python main.py
```

Open `http://localhost:8000`.

## API

### `POST /detect`

Accepts a multipart image upload and returns:

```json
{
  "image": "data:image/png;base64,...",
  "detections": [
    {
      "class_name": "person",
      "bounding_box": [12.3, 24.1, 140.5, 300.7],
      "confidence": 0.97
    }
  ]
}
```

### `POST /query`

Accepts:

```json
{
  "query": "How many people are in the image?",
  "detections": [
    {
      "class_name": "person",
      "bounding_box": [12.3, 24.1, 140.5, 300.7],
      "confidence": 0.97
    }
  ]
}
```

Returns:

```json
{
  "data": "There is 1 person in the image.",
  "sources": []
}
```

## Tests

```bash
uv sync
uv run pytest
```
