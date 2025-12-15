from ultralytics import YOLO
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import base64

app = FastAPI()

# CORS (important for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 text detection model
model = YOLO(r"Aksor@1.pt")  # update path if needed

# Root endpoint for sanity check
@app.get("/")
async def root():
    return {"message": "Khmer Text Detection API is running. Use POST /detect to send images."}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Run inference
    result = model(img)[0]

    detections = []

    # Draw boxes + confidence
    for box, conf in zip(result.boxes.xyxy, result.boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        confidence = float(conf) * 100

        label = f"{confidence:.1f}%"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            label,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        detections.append({
            "confidence": round(confidence, 2)
        })

    # Encode image to base64
    _, buffer = cv2.imencode(".jpg", img)
    encoded_image = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse({
        "image": encoded_image,
        "detections": detections
    })
