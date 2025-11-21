from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import detect_endpoint, query_rag

# Create FastAPI app
app = FastAPI(
    title="YOLO Object Detection and RAG API",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add routes
app.post("/detect")(detect_endpoint)
app.post("/query")(query_rag)
