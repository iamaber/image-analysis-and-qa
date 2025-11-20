from io import BytesIO
from typing import List, Dict
import base64
import asyncio
import uuid  # Added for session management

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from PIL import Image
from ultralytics import YOLO
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load the YOLO model (using a lightweight pre-trained model)
model = YOLO("yolo11n.pt")  # Switched to nano for lighter weight; change back if needed

# In-memory storage for detections (session-based)
detections_store: Dict[str, List[Dict]] = {}


# Define Pydantic models
class DetectionResponse(BaseModel):
    image: str  # base64 encoded annotated image
    detections: List[Dict]
    session_id: str  # Added to track session


class QueryRequest(BaseModel):
    query: str
    session_id: str  # Required to fetch the correct detections


class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]


def detect_objects(image: Image.Image) -> tuple[Image.Image, List[Dict]]:
    # Run YOLO inference on the PIL image
    results = model(image)

    # Get the annotated image as a numpy array and convert to PIL
    annotated_img_np = results[0].plot()  # This draws bounding boxes, labels, etc.
    annotated_img = Image.fromarray(annotated_img_np)

    # Extract structured detections
    detections = []
    for r in results:
        for box in r.boxes:
            det = {
                "class_name": r.names[int(box.cls)],
                "bounding_box": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                "confidence": box.conf.item(),
            }
            detections.append(det)

    return annotated_img, detections


# Create FastAPI app
app = FastAPI(
    title="YOLO Object Detection and RAG API",
    version="1.0.0",
    description="API for object detection in images using YOLO and querying detection results via AI.",
)


@app.post("/detect", response_model=DetectionResponse)
async def detect_endpoint(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        contents = await file.read()
        input_img = Image.open(BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # Perform detection in a separate thread to avoid blocking
    try:
        annotated_img, detections = await asyncio.to_thread(detect_objects, input_img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

    # Generate a unique session ID and store detections
    session_id = str(uuid.uuid4())
    detections_store[session_id] = detections

    # Encode annotated image to base64
    buffer = BytesIO()
    annotated_img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    # Return JSON response
    return DetectionResponse(
        image=f"data:image/png;base64,{img_base64}",
        detections=detections,
        session_id=session_id,
    )


@app.post("/query", response_model=AnswerResponse)
async def query_rag(request: QueryRequest = Body(...)):
    """
    Endpoint to query the RAG system about image detection results.

    - Requires session_id from /detect response.
    - query: The question about image detection data.
    - Returns a JSON with answer and sources.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Fetch detections using session_id
    detections = detections_store.get(request.session_id)
    if not detections:
        raise HTTPException(
            status_code=404,
            detail="No detection data found for this session. Please upload an image first.",
        )

    # Build context from detections
    context_lines = [
        f"Detected {d['class_name']} with confidence {d['confidence']:.2f} at bounding box {d['bounding_box']}"
        for d in detections
    ]
    context = "\n".join(context_lines)

    # Improved prompt: More structured, handles edge cases better
    prompt = (
        "You are an expert analyst for image object detection results. "
        "Answer only questions directly related to the provided detection data, such as confidence scores, number of objects, bounding boxes, class names, or simple aggregations (e.g., count of specific objects). "
        "Do not speculate, add external knowledge, or answer unrelated questions. "
        "If the query is not related to the detection data, respond exactly with: 'I can only answer questions about image detection results.'\n\n"
        f"Detection Data:\n{context}\n\nQuery: {request.query}"
    )

    # Initialize agent with settings
    agent = Agent(
        "google-gla:gemini-2.5-flash",
        model_settings=ModelSettings(temperature=0.2, max_tokens=150),
    )

    try:
        result = await agent.run(prompt)
        # Assuming result is an AnswerResponse-like structure; adjust if pydantic_ai returns differently
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Query processing failed: {str(e)}"
        )


# Optional: Cleanup old sessions periodically (not implemented here, but could use a background task)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8006)
