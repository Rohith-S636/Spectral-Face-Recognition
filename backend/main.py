"""
main.py
-------
FastAPI application – all HTTP routes.

Routes
------
GET  /health                → server status
POST /register              → register a new face
POST /verify                → verify an uploaded face
GET  /users                 → list all registered users
DELETE /users/{name}        → remove a user and retrain
POST /retrain               → rebuild PCA from scratch

Auto-generated Swagger UI available at http://localhost:8000/docs
"""

from __future__ import annotations

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import RAW_FACES_PATH, K
from image_utils import preprocess_bytes, preprocess_bytes_with_info
from pca_engine import PCAEngine
from chroma_store import ChromaStore

# ── App setup ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Spectral Face Recognition API",
    description="Eigenface-based face recognition using PCA + NumPy vector store.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons ───────────────────────────────────────────────────────────
pca   = PCAEngine(k=K)
store = ChromaStore()

# ── Startup ──────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    loaded = pca.load()
    if loaded:
        print(f"[startup] PCA model loaded (K={pca.K}, U={pca.U.shape})")
    else:
        print("[startup] No saved model — register faces to train.")


# ── Raw face store helpers ────────────────────────────────────────────────

def _load_raw() -> tuple[list[str], np.ndarray]:
    if not RAW_FACES_PATH.exists():
        return [], np.empty((0,), dtype=np.float32)
    data  = np.load(RAW_FACES_PATH, allow_pickle=True)
    names = data["names"].tolist()
    X     = data["X"]
    return names, X


def _save_raw(names: list[str], X: np.ndarray) -> None:
    RAW_FACES_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RAW_FACES_PATH, names=np.array(names), X=X)


def _retrain_and_upsert() -> int:
    """Rebuild PCA from all raw faces, re-project, upsert to vector store."""
    names, X = _load_raw()
    n = len(names)
    if n < 2:
        return n
    pca.train(X)
    omegas = pca.project_many(X)
    store.add_many(names, omegas)
    return n


# ── Routes ───────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
def health():
    return {
        "status":      "ok",
        "model_ready": pca.is_trained,
        "users":       store.count(),
    }


@app.post("/register", tags=["faces"])
async def register(
    name:  str        = Form(..., description="User's full name"),
    image: UploadFile = File(..., description="Face image (jpg, png, etc.)"),
):
    """
    Register a new face.

    1. Detect + crop face region, apply CLAHE normalisation
    2. Append preprocessed vector to raw face store
    3. Retrain PCA with all stored faces
    4. Re-project everyone → upsert Ω vectors into vector store
    """
    if not name.strip():
        raise HTTPException(status_code=400, detail="Name must not be empty.")

    data = await image.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    x, face_found = preprocess_bytes_with_info(data)

    # Append / replace in raw store
    names, X = _load_raw()
    if name in names:
        idx     = names.index(name)
        X[idx]  = x
    else:
        names.append(name)
        X = np.vstack([X, x]) if X.ndim == 2 else x.reshape(1, -1)

    _save_raw(names, X)

    n = _retrain_and_upsert()

    return {
        "success":     True,
        "message":     f"'{name}' registered. Model trained with {n} face(s).",
        "total_users": n,
        "model_ready": pca.is_trained,
        "face_detected": face_found,   # frontend can warn if False
    }


@app.post("/verify", tags=["faces"])
async def verify(
    image: UploadFile = File(..., description="Face image to verify"),
):
    """
    Verify an uploaded face.

    1. Detect + crop face region, apply CLAHE normalisation
    2. Project x → Ω using current PCA model
    3. Query vector store for nearest Ω
    4. Return match result + confidence
    """
    if not pca.is_trained:
        raise HTTPException(
            status_code=400,
            detail="No model trained yet. Register at least 2 faces first.",
        )
    if store.count() == 0:
        raise HTTPException(status_code=400, detail="No registered users.")

    data = await image.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    x, face_found = preprocess_bytes_with_info(data)
    omega  = pca.project(x)
    result = store.query(omega)
    result["face_detected"] = face_found   # extra info for frontend

    return result


@app.get("/users", tags=["faces"])
def list_users():
    return {"users": store.list_users(), "count": store.count()}


@app.delete("/users/{name}", tags=["faces"])
def delete_user(name: str):
    names, X = _load_raw()
    if name not in names:
        raise HTTPException(status_code=404, detail=f"User '{name}' not found.")

    idx = names.index(name)
    names.pop(idx)
    X   = np.delete(X, idx, axis=0)
    _save_raw(names, X)
    store.delete_face(name)

    n = _retrain_and_upsert()
    return {
        "success":     True,
        "message":     f"'{name}' deleted. Model retrained with {n} remaining user(s).",
        "total_users": n,
    }


@app.post("/retrain", tags=["system"])
def retrain():
    n = _retrain_and_upsert()
    if n < 2:
        return {
            "success": False,
            "message": f"Need at least 2 registered faces (have {n}).",
        }
    return {
        "success": True,
        "message": f"Model retrained with {n} face(s).",
        "K":       pca.K,
        "U_shape": list(pca.U.shape),
    }
