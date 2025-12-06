from fastapi import FastAPI, UploadFile, File, HTTPException
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

def read_image(file: UploadFile):
    data = np.frombuffer(file.file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return img


def ensure_single_face(img):
    try:
        faces = DeepFace.extract_faces(
            img_path=img,
            detector_backend="retinaface",
            enforce_detection=False,
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Face detection failed")

    if len(faces) == 0:
        raise HTTPException(status_code=400, detail="No face detected")

    if len(faces) > 1:
        raise HTTPException(status_code=400, detail="Multiple faces detected")


@app.post("/match_faces")
async def match_faces(
    existing: UploadFile = File(...),
    new: UploadFile = File(...)
):
    try:
        img1 = read_image(existing)
        img2 = read_image(new)

        ensure_single_face(img1)
        ensure_single_face(img2)

        result = DeepFace.verify(
            img1,
            img2,
            model_name="Facenet512",
            detector_backend="retinaface",
            distance_metric="cosine",
            enforce_detection=True,
        )

        distance = float(result["distance"])
        THRESHOLD = 0.45

        return {
            "matched": distance <= THRESHOLD,
            "distance": distance,
            "threshold": THRESHOLD,
            "model": "Facenet512",
        }

    except HTTPException as e:
        # ✅ controlled validation errors
        return {
            "matched": False,
            "error": e.detail,
        }

    except Exception:
        # ✅ absolute safety net (no server crash)
        return {
            "matched": False,
            "error": "Face verification failed",
        }
