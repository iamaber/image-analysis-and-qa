# AI Vision Workspace

Single-page image analysis workspace for object detection and question answering. Upload an image, run YOLO detection, then ask follow-up questions grounded in the detection results through a `pydantic-ai` agent backed by Groq's `openai/gpt-oss-20b` model.

Detection runs without any API key. Question answering requires `GROQ_API_KEY`; without it the app still starts, `/detect` works, and `/query` returns a clear `503` instead of crashing.

**Project links:** [live demo](https://huggingface.co/spaces/iamaber/image-analysis-and-qa-rag) · [source](https://github.com/iamaber/image-analysis-and-qa) · [portfolio](https://iamaber.github.io/portfolio-website/)

## Stack

- FastAPI + Uvicorn
- Ultralytics YOLO (`yolo11m.pt`)
- `pydantic-ai` with Groq `openai/gpt-oss-20b`
- Single-file HTML/CSS/JavaScript frontend

## Quick Start

```bash
uv sync
cp .env.example .env          # optional; add GROQ_API_KEY for question answering
uv run python main.py
```

Open `http://localhost:8000`.

The YOLO weights are not committed to the repository. Ultralytics downloads the stock `yolo11m.pt` on first detection (or the Docker image fetches them at build time).

## Environment

Copy `.env.example` to `.env` and adjust as needed.

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `GROQ_API_KEY` | For `/query` | – | Enables question answering. `/detect` works without it. |
| `YOLO_MODEL_PATH` | No | `yolo11m.pt` | Path to the YOLO weights file. |
| `YOLO_CONFIG_DIR` | No | `.ultralytics` | Writable directory for Ultralytics settings/cache. Falls back to a temp directory if the default is not writable. |
| `CORS_ALLOW_ORIGINS` | No | `http://localhost:8000,http://127.0.0.1:8000` | Comma-separated list of allowed origins. |
| `MAX_UPLOAD_BYTES` | No | `10485760` (10 MB) | Maximum accepted upload size. |

## API

### `GET /health`

```json
{
  "status": "ok",
  "qa_configured": true
}
```

`qa_configured` is `false` when `GROQ_API_KEY` is unset. The frontend uses it to disable the chat panel with a configuration hint.

### `POST /detect`

Multipart upload in the `file` field. Only `image/*` content types are accepted, and uploads larger than `MAX_UPLOAD_BYTES` are rejected with `413`.

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

Returns `400` for non-image or unreadable uploads and `503` if the detection model cannot be loaded.

### `POST /query`

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

`query` is limited to 2000 characters and `detections` must be non-empty (`400` otherwise). Without `GROQ_API_KEY` the endpoint returns `503` with a message explaining that question answering is not configured.

## Tests

The tests mock the YOLO model, so they do not download weights or call Groq.

```bash
uv sync
uv run pytest
```

## Deploying to a Hugging Face Docker Space

The repository includes a `Dockerfile` for a CPU Hugging Face Docker Space (`sdk: docker`, `app_port: 7860`):

- The container runs as a non-root user (UID 1000), as Hugging Face requires.
- The app listens on `0.0.0.0:7860`.
- Ultralytics config/cache and Matplotlib cache are redirected to writable `/tmp` directories.
- `pyproject.toml` resolves `torch`/`torchvision` from the PyTorch CPU index on Linux. Without this, `uv sync` pulls the CUDA build of PyTorch and several GB of `nvidia-*` wheels into a CPU-only Space.
- The stock `yolo11m.pt` weights are downloaded at image build time instead of being committed, keeping the repository and build context small.
- Add `GROQ_API_KEY` as a Space secret (Settings → Variables and secrets). Detection works without it; the UI reports that Q&A is disabled.

## Project Layout

```text
backend/        FastAPI app, routes, detection and query logic
frontend/       Single-page application (application.html)
main.py         Uvicorn entry point and static file mounting
tests/          API and helper tests
Dockerfile      Hugging Face Docker Space image
```

## License

Copyright (c) 2025 iamaber

This application is released under the GNU Affero General Public License,
version 3 or (at your option) any later version (AGPL-3.0-or-later); see
[LICENSE](LICENSE) for the full terms. The app imports and serves Ultralytics
YOLO over a network. The installed `ultralytics==8.3.229` package declares
`License: AGPL-3.0`, classifies itself as AGPLv3+, and includes the AGPL v3
licence, so this project adopts the same copyleft terms.

The [source repository](https://github.com/iamaber/image-analysis-and-qa)
provides the corresponding source for this application, including its build
configuration and dependency lockfile. If you distribute or host a modified
version, provide users access to that version's corresponding source under
the AGPL, including the network source offer required by section 13.
