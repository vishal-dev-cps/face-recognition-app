from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import cv2
import numpy as np
from numpy.linalg import norm

# -------------------------------------------------
# APP INIT
# -------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# IMAGE UTILITIES
# -------------------------------------------------
def read_image(file: UploadFile):
    data = np.frombuffer(file.file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image")
    return img


def is_blurry(img, threshold=15):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray32 = np.float32(gray)
    return cv2.Laplacian(gray32, cv2.CV_32F).var() < threshold


# -------------------------------------------------
# CROSS-DOMAIN NORMALIZATION (PASSPORT ↔ WEBCAM FIX)
# -------------------------------------------------
def normalize_for_cross_domain(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Ensure 8-bit for CLAHE
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    norm_img = clahe.apply(gray)

    # Strong denoising for webcam noise
    norm_img = cv2.fastNlMeansDenoising(norm_img, h=15)

    return cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)


# -------------------------------------------------
# FACE EXTRACTION
# -------------------------------------------------
def extract_clean_face(img):
    faces = DeepFace.extract_faces(
        img_path=img,
        detector_backend="retinaface",
        enforce_detection=True,
        align=True,
    )

    if len(faces) == 0:
        raise HTTPException(400, "No face detected")
    if len(faces) > 1:
        raise HTTPException(400, "Multiple faces detected")

    f = faces[0]
    area = f["facial_area"]
    face = f["face"]

    if area["w"] < 80 or area["h"] < 80:
        raise HTTPException(400, "Face too far from camera")

    # Blur is informational only (do NOT block)
    _ = is_blurry(face)

    return normalize_for_cross_domain(face)


# -------------------------------------------------
# SINGLE-IMAGE LIVENESS (SOFT / NON-BLOCKING)
# -------------------------------------------------
def image_entropy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist /= hist.sum()
    return -np.sum(hist * np.log2(hist + 1e-7))


def skin_texture_variance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    texture = cv2.absdiff(gray, blur)
    return np.mean(texture)


def edge_density(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return np.sum(edges > 0) / edges.size


def illumination_variance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.var(gray)


def single_image_liveness(face):
    entropy = image_entropy(face)
    texture = skin_texture_variance(face)
    edges = edge_density(face)
    illum = illumination_variance(face)

    score = 0
    if entropy > 4.5: score += 1
    if texture > 5.5: score += 1
    if edges < 0.22: score += 1
    if illum > 450: score += 1

    passed = score >= 2  # attendance-friendly

    return passed, {
        "entropy": round(entropy, 2),
        "texture": round(texture, 2),
        "edges": round(edges, 3),
        "illumination": round(illum, 1),
        "score": score,
    }


# -------------------------------------------------
# EMBEDDING + COSINE DISTANCE (✅ STABLE)
# -------------------------------------------------
def get_embedding(face_img):
    emb = DeepFace.represent(
        img_path=face_img,
        model_name="Facenet512",
        detector_backend="skip",
        enforce_detection=False,
    )
    return np.array(emb[0]["embedding"], dtype=np.float32)


def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (norm(a) * norm(b))


# -------------------------------------------------
# ATTENDANCE MATCH ENDPOINT
# -------------------------------------------------
@app.post("/match_faces")
async def match_faces(existing: UploadFile = File(...), new: UploadFile = File(...)):
    try:
        img1 = read_image(existing)
        img2 = read_image(new)

        face1 = extract_clean_face(img1)
        face2 = extract_clean_face(img2)

        # Soft liveness (do not block recognition)
        live1, m1 = single_image_liveness(face1)
        live2, m2 = single_image_liveness(face2)

        # ✅ EMBEDDING-BASED MATCHING (NO DeepFace.verify)
        emb1 = get_embedding(face1)
        emb2 = get_embedding(face2)

        distance = cosine_distance(emb1, emb2)

        MATCH_THRESHOLD = 0.55  # tuned for studio ↔ webcam
        matched = distance <= MATCH_THRESHOLD

        return {
            "matched": bool(matched),
            "distance": float(round(distance, 4)),
            "threshold": float(MATCH_THRESHOLD),
            "liveness_passed": bool(live1 and live2),
            "attendance_approved": bool(matched and (live1 and live2)),
            "liveness_details": {
                "existing": {
                    "entropy": float(m1["entropy"]),
                    "texture": float(m1["texture"]),
                    "edges": float(m1["edges"]),
                    "illumination": float(m1["illumination"]),
                    "score": int(m1["score"]),
                },
                "new": {
                    "entropy": float(m2["entropy"]),
                    "texture": float(m2["texture"]),
                    "edges": float(m2["edges"]),
                    "illumination": float(m2["illumination"]),
                    "score": int(m2["score"]),
                }
            }
        }


    except HTTPException as e:
        return {"matched": False, "error": e.detail}

    except Exception as e:
        return {"matched": False, "error": str(e)}
