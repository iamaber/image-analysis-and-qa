from fastapi import FastAPI
from .routes import detect_endpoint, query_rag

# Create FastAPI app
app = FastAPI(
    title="YOLO Object Detection and RAG API",
    version="1.0.0",
)

# Add routes
app.post("/detect")(detect_endpoint)
app.post("/query")(query_rag)
