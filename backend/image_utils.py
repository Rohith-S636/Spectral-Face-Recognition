"""
image_utils.py
--------------
Image preprocessing pipeline with face detection.

Pipeline
--------
raw bytes
  → PIL open + validate
  → OpenCV face detection (Haar cascade)  ← crops out the background
  → grayscale
  → resize to IMG_SIZE × IMG_SIZE
  → CLAHE histogram equalisation          ← normalises lighting / contrast
  → normalise to float32 [0, 1]
  → flatten to (IMG_DIM,) vector

Why this fixes background sensitivity
--------------------------------------
Without face detection, the (100×100) pixel grid contains whatever
background was behind the person.  PCA then learns "white wall" and
"dark room" as significant directions in face space — so two photos
of Alice in different rooms look further apart than Alice and Bob in
the same room.

Cropping to the detected face region means nearly all pixels describe
facial structure, not background.  CLAHE further removes lighting
differences so the same face under bright / dim light produces similar
vectors.

Fallback
--------
If no face is detected (unusual angle, low-res image, etc.) the full
image is used with a logged warning instead of rejecting the upload.
This keeps the system usable while still being honest that accuracy
will be lower.
"""

from __future__ import annotations

import io
import logging

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import HTTPException

from config import IMG_SIZE, IMG_DIM

logger = logging.getLogger(__name__)

# ── Load Haar cascade once at import time ────────────────────────────
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)

# CLAHE (Contrast Limited Adaptive Histogram Equalisation)
# clipLimit=2  – prevents over-amplifying noise
# tileGridSize – local equalisation in 8×8 tiles
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Extra padding added around detected face (fraction of face size)
_FACE_PAD = 0.20   # 20 % padding on each side


# ── Public API ───────────────────────────────────────────────────────

def preprocess_bytes(data: bytes) -> np.ndarray:
    """
    Convert raw image bytes to a normalised (IMG_DIM,) float32 vector.

    Steps
    -----
    1. Decode bytes → PIL Image (validates format)
    2. Convert to OpenCV uint8 grayscale
    3. Face detection → crop to face region (+ padding)
    4. Resize to IMG_SIZE × IMG_SIZE
    5. CLAHE equalisation
    6. Normalise [0, 1] and flatten

    Raises HTTPException(400) for invalid or too-small images.
    """
    # ── 1. Decode ────────────────────────────────────────────────────
    try:
        pil_img = Image.open(io.BytesIO(data))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    if pil_img.width < 20 or pil_img.height < 20:
        raise HTTPException(status_code=400, detail="Image is too small (minimum 20×20 px).")

    # ── 2. Convert to OpenCV grayscale ───────────────────────────────
    cv_img = _pil_to_cv_gray(pil_img)

    # ── 3. Face detection + crop ─────────────────────────────────────
    cv_img, face_found = _detect_and_crop(cv_img)

    if not face_found:
        logger.warning(
            "No face detected in uploaded image — using full image. "
            "Accuracy may be lower."
        )

    # ── 4. Resize ────────────────────────────────────────────────────
    cv_img = cv2.resize(cv_img, (IMG_SIZE, IMG_SIZE),
                        interpolation=cv2.INTER_LANCZOS4)

    # ── 5. CLAHE equalisation ────────────────────────────────────────
    cv_img = _clahe.apply(cv_img)

    # ── 6. Normalise + flatten ───────────────────────────────────────
    vec = cv_img.astype(np.float32) / 255.0   # [0, 1]
    vec = vec.flatten()                        # (IMG_DIM,)

    assert vec.shape == (IMG_DIM,), f"Unexpected shape: {vec.shape}"
    return vec


def preprocess_bytes_with_info(data: bytes) -> tuple[np.ndarray, bool]:
    """
    Same as preprocess_bytes but also returns whether a face was detected.
    Used by the /register route to warn the user if no face was found.
    """
    try:
        pil_img = Image.open(io.BytesIO(data))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    cv_img = _pil_to_cv_gray(pil_img)
    cv_img, face_found = _detect_and_crop(cv_img)
    cv_img = cv2.resize(cv_img, (IMG_SIZE, IMG_SIZE),
                        interpolation=cv2.INTER_LANCZOS4)
    cv_img = _clahe.apply(cv_img)
    vec = cv_img.astype(np.float32).flatten() / 255.0
    return vec, face_found


# ── Internal helpers ─────────────────────────────────────────────────

def _pil_to_cv_gray(pil_img: Image.Image) -> np.ndarray:
    """PIL Image (any mode) → OpenCV uint8 grayscale array."""
    pil_gray = pil_img.convert("L")
    return np.array(pil_gray, dtype=np.uint8)


def _detect_and_crop(gray: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Detect the largest face in a grayscale image and return a padded crop.

    Parameters
    ----------
    gray : HxW uint8 grayscale array

    Returns
    -------
    (cropped_gray, face_found)
    If no face detected → returns the original image unchanged.
    """
    h, w = gray.shape

    # Try two passes: normal scale then smaller minSize for small photos
    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if not isinstance(faces, np.ndarray) or len(faces) == 0:
        # Second pass — more lenient
        faces = _face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20),
        )

    if not isinstance(faces, np.ndarray) or len(faces) == 0:
        return gray, False   # fallback: no face found

    # Pick the largest detected face
    areas = faces[:, 2] * faces[:, 3]   # width * height
    x, y, fw, fh = faces[int(np.argmax(areas))]

    # Add padding around the face
    pad_x = int(fw * _FACE_PAD)
    pad_y = int(fh * _FACE_PAD)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + fw + pad_x)
    y2 = min(h, y + fh + pad_y)

    return gray[y1:y2, x1:x2], True


def vector_to_image(vec: np.ndarray) -> Image.Image:
    """Reconstruct a PIL Image from a flat pixel vector (debugging only)."""
    arr = (vec * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr.reshape(IMG_SIZE, IMG_SIZE), mode="L")
