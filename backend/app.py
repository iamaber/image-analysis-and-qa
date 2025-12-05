import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import detect_endpoint, query_rag


def get_allowed_origins() -> list[str]:
    origins = os.getenv("CORS_ALLOW_ORIGINS")
    if origins:
        return [origin.strip() for origin in origins.split(",") if origin.strip()]
    return [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]


app = FastAPI(
    title="AI Vision Platform API",
    description="Advanced object detection and intelligent analysis API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.post("/detect")(detect_endpoint)
app.post("/query")(query_rag)
