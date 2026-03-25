"""
chroma_store.py
---------------
Pure-NumPy vector store — replaces ChromaDB entirely.
No C++ compiler required. Works on all platforms.

Keeps the exact same public interface as the ChromaDB version
so main.py does not need any changes.

Storage layout  (data/vector_store.npz)
----------------------------------------
  names   : (N,)  string array of user names
  vectors : (N, K) float32 array of Ω weight vectors
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import THRESHOLD, BASE_DIR

# Where the vector store is saved on disk
STORE_PATH = BASE_DIR / "data" / "vector_store.npz"


class ChromaStore:
    """
    Drop-in replacement for the ChromaDB wrapper.
    Stores Ω vectors in a .npz file and does cosine
    nearest-neighbour search with NumPy.
    """

    def __init__(self):
        self._names:   list[str]      = []
        self._vectors: np.ndarray     = np.empty((0,), dtype=np.float32)
        self._load()

    # ── Persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        if not STORE_PATH.exists():
            return
        data = np.load(STORE_PATH, allow_pickle=True)
        self._names   = data["names"].tolist()
        self._vectors = data["vectors"]   # (N, K)

    def _save(self) -> None:
        STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            STORE_PATH,
            names=np.array(self._names),
            vectors=self._vectors,
        )

    # ── Write ────────────────────────────────────────────────────────

    def add_face(self, name: str, omega: np.ndarray) -> None:
        """Upsert a single face vector."""
        omega = omega.astype(np.float32)
        if name in self._names:
            idx = self._names.index(name)
            self._vectors[idx] = omega
        else:
            self._names.append(name)
            if self._vectors.ndim == 1 and self._vectors.size == 0:
                self._vectors = omega.reshape(1, -1)
            else:
                self._vectors = np.vstack([self._vectors, omega])
        self._save()

    def add_many(self, names: list[str], omegas: np.ndarray) -> None:
        """
        Bulk upsert — called after every retrain.
        Replaces the entire store with the new projected vectors.
        """
        self._names   = list(names)
        self._vectors = omegas.astype(np.float32)   # (N, K)
        self._save()

    def delete_face(self, name: str) -> None:
        if name not in self._names:
            return
        idx = self._names.index(name)
        self._names.pop(idx)
        self._vectors = np.delete(self._vectors, idx, axis=0)
        self._save()

    # ── Query ────────────────────────────────────────────────────────

    def query(self, omega: np.ndarray, n_results: int = 1) -> dict:
        """
        Find the closest registered face using cosine distance.

        cosine distance = 1 − cosine_similarity   (range 0–2)

        Returns the same dict shape as the old ChromaDB version:
        {
          "match"     : bool,
          "name"      : str | None,
          "distance"  : float,
          "confidence": float,
        }
        """
        if len(self._names) == 0:
            return {"match": False, "name": None, "distance": 2.0, "confidence": 0.0}

        q = omega.astype(np.float32)

        # Cosine similarity between q and every stored vector
        q_norm = q / (np.linalg.norm(q) + 1e-10)
        v_norms = self._vectors / (
            np.linalg.norm(self._vectors, axis=1, keepdims=True) + 1e-10
        )
        similarities = v_norms @ q_norm          # (N,)
        distances    = 1.0 - similarities        # cosine distance (N,)

        best_idx      = int(np.argmin(distances))
        best_distance = float(distances[best_idx])
        best_name     = self._names[best_idx]

        matched    = best_distance < THRESHOLD
        confidence = float(max(0.0, 1.0 - best_distance / THRESHOLD)) if matched else 0.0

        return {
            "match":      matched,
            "name":       best_name if matched else None,
            "distance":   round(best_distance, 4),
            "confidence": round(confidence, 4),
        }

    # ── Read ─────────────────────────────────────────────────────────

    def list_users(self) -> list[str]:
        return list(self._names)

    def count(self) -> int:
        return len(self._names)
