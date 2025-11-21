from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import detect_endpoint, query_rag
from .auth_routes import router as auth_router
from .database import create_tables

# Create FastAPI app
app = FastAPI(
    title="AI Vision Platform API",
    description="Advanced object detection and intelligent analysis API with JWT authentication",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize database tables on startup
@app.on_event("startup")
def startup_event():
    from .database import check_and_update_schema

    check_and_update_schema()


# Add routes
app.post("/detect")(detect_endpoint)
app.post("/query")(query_rag)

# Add authentication routes
app.include_router(auth_router)
