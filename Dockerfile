FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    YOLO_CONFIG_DIR=/tmp/.ultralytics \
    YOLO_MODEL_PATH=/app/models/yolo11m.pt \
    MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/home/user

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

COPY backend ./backend
COPY frontend ./frontend
COPY main.py ./

RUN mkdir -p /app/models \
    && /app/.venv/bin/python -c "from ultralytics import YOLO; YOLO('yolo11m.pt')" \
    && mv /app/yolo11m.pt /app/models/yolo11m.pt \
    && mkdir -p /tmp/.ultralytics /tmp/matplotlib \
    && chown -R user:user /app /tmp/.ultralytics /tmp/matplotlib

USER user

EXPOSE 7860

CMD ["/app/.venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
