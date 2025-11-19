import base64
from io import BytesIO
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File
from PIL import Image
from ultralytics import YOLO

# Load the YOLO model (using a lightweight pre-trained model)
model = YOLO(
    "yolov8n.pt"
)  # You can change to other variants like 'yolov8s.pt' for better accuracy


def detect_objects(image: Image.Image) -> tuple[Image.Image, List[Dict]]:
    """
    Detects objects in the given image using YOLOv8.

    Args:
        image (PIL.Image.Image): The input image.

    Returns:
        tuple:
            - Annotated image with bounding boxes drawn (PIL.Image.Image).
            - List of detections, each as a dict with 'class_name', 'bounding_box' (list [x1, y1, x2, y2]), and 'confidence'.
    """
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
    title="YOLO Object Detection API",
    description="API for object detection using YOLOv8. Upload an image to get annotated image and detections.",
    version="1.0.0",
)


@app.post("/detect")
async def detect_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to detect objects in an uploaded image.

    - Upload an image file.
    - Returns a JSON with base64-encoded annotated image and list of detections.
    """
    # Read the uploaded file
    contents = await file.read()
    input_img = Image.open(BytesIO(contents))

    # Perform detection
    annotated_img, detections = detect_objects(input_img)

    # Encode annotated image to base64
    buffer = BytesIO()
    annotated_img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Return JSON response
    return {"annotated_image_base64": img_base64, "detections": detections}
