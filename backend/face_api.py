from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import cv2
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def read_image(file):
    data = np.frombuffer(file.file.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

@app.post("/match_faces")
async def match_faces(existing: UploadFile = File(...), new: UploadFile = File(...)):
    img1 = read_image(existing)
    img2 = read_image(new)

    try:
        result = DeepFace.verify(
            img1,
            img2,
            model_name="Facenet512",
            detector_backend="mtcnn",   # <- BEST STABLE
            enforce_detection=False,
            distance_metric="cosine",
        )

        # manually apply a friendly threshold
        matched = result["distance"] < 0.6

        return {
            "matched": matched,
            "distance": float(result["distance"]),
            "threshold": 0.6,
        }

    except Exception as e:
        return {"matched": False, "error": str(e)}
