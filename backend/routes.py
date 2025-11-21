from io import BytesIO
import os
import base64
import asyncio
from typing import List, Dict

from fastapi import UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image
from ultralytics import YOLO
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.google import GoogleModel
from dotenv import load_dotenv


# Request model for query endpoint
class QueryRequest(BaseModel):
    query: str


# Load environment variables from .env file
load_dotenv()

# Load the YOLO model (using a lightweight pre-trained model)
model = YOLO("yolo11m.pt")

# Global variable to store last detections
last_detections: List[Dict] = []


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

    # Store detections globally
    global last_detections
    last_detections = detections

    return annotated_img, detections


async def detect_endpoint(file: UploadFile = File(...)):
    # Read the uploaded file
    contents = await file.read()
    input_img = Image.open(BytesIO(contents))

    # Perform detection asynchronously
    annotated_img, detections = await asyncio.to_thread(detect_objects, input_img)

    # Encode annotated image to base64
    buffer = BytesIO()
    annotated_img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    # Return JSON response
    return {"image": f"data:image/png;base64,{img_base64}", "detections": detections}


async def query_rag(request: QueryRequest):
    """
    Endpoint to query the RAG system about image detection results.

    - query: The question about image detection data.
    - Returns a JSON with answer and sources.
    """
    query = request.query
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Use last detections as context
    global last_detections
    if not last_detections:
        return {
            "data": "No detection data available. Please upload an image first.",
            "sources": [],
        }

    context = "\n".join(
        f"Detected {d['class_name']} with confidence {d['confidence']:.2f} at bounding box {d['bounding_box']}"
        for d in last_detections
    )

    prompt = (
        "Answer questions related to image or picture such as confidence scores, number of objects, bounding boxes, etc. "
        "If the query is not related, say 'I can only answer questions about image detection results.'\n\n"
        f"Information:\n{context}\n\nQuery: {query}"
    )
    provider = GoogleProvider(api_key=os.getenv("GOOGLE_API_KEY"))
    model = GoogleModel("gemini-2.5-flash", provider=provider)
    agent = Agent(
        model,
        model_settings=ModelSettings(temperature=0.3, max_tokens=200),
    )

    result = await agent.run(prompt)
    # Extract just the text output from the result
    answer_text = result.output
    return answer_text
