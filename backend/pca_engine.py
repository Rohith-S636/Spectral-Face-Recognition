"""
pca_engine.py
-------------
Implements the Eigenface / PCA pipeline exactly as described in the README.

Four-stage pipeline
-------------------
1. Preprocessing   – flatten N×N images to N² vectors, compute mean face Ψ
2. Covariance      – deviation matrix A, surrogate covariance L = AᵀA  (N×N not d×d)
3. Eigenfaces      – eigenvectors of L projected back → orthonormal basis U  (d×K)
4. Projection      – Ω = Uᵀ(x − Ψ)   maps any face to a K-dim weight vector
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from config import K, IMG_DIM, EIGENFACES_PATH


class PCAEngine:
    """
    Wraps the complete Eigenfaces pipeline.

    Attributes
    ----------
    K          : number of eigenfaces to keep
    mean_face  : Ψ  – shape (IMG_DIM,)
    U          : eigenface basis – shape (IMG_DIM, K), column = one eigenface
    is_trained : True once train() has been called successfully
    """

    def __init__(self, k: int = K):
        self.K: int = k
        self.mean_face: np.ndarray | None = None   # Ψ
        self.U: np.ndarray | None = None           # eigenface basis
        self.is_trained: bool = False

    # ── Training ────────────────────────────────────────────────────────

    def train(self, X: np.ndarray) -> None:
        """
        Build the eigenface basis from a matrix of face vectors.

        Parameters
        ----------
        X : shape (N, IMG_DIM)  – one row per registered face image
        """
        N = X.shape[0]
        if N < 2:
            raise ValueError("Need at least 2 registered face images to train PCA.")

        # ── Stage 1: mean face ──────────────────────────────────────────
        self.mean_face = X.mean(axis=0)                      # Ψ  (IMG_DIM,)

        # ── Stage 2: deviation matrix ───────────────────────────────────
        # A = (X − Ψ).T   shape: (IMG_DIM, N)
        A = (X - self.mean_face).T                           # (IMG_DIM, N)

        # ── Stage 3a: surrogate covariance ──────────────────────────────
        # L = AᵀA   shape (N, N)  — MUCH smaller than AAᵀ which is (IMG_DIM, IMG_DIM)
        # The KEY trick that makes eigenfaces practical.
        L = A.T @ A                                          # (N, N)

        # ── Stage 3b: eigenvalue decomposition ──────────────────────────
        # eigh returns eigenvalues in ascending order → reverse for descending
        eigenvalues, eigenvectors = np.linalg.eigh(L)       # eigenvectors: (N, N)
        idx = np.argsort(eigenvalues)[::-1]                  # descending
        eigenvectors = eigenvectors[:, idx]                  # (N, N)

        # Take only top-K eigenvectors (or fewer if N < K)
        k_actual = min(self.K, N - 1)
        eigenvectors = eigenvectors[:, :k_actual]            # (N, k_actual)

        # ── Stage 3c: project back to image space ───────────────────────
        # uᵢ = A · vᵢ  then normalise to unit length
        U_raw = A @ eigenvectors                             # (IMG_DIM, k_actual)
        norms = np.linalg.norm(U_raw, axis=0, keepdims=True)
        norms[norms == 0] = 1e-10                            # avoid div by zero
        self.U = U_raw / norms                               # (IMG_DIM, k_actual)

        self.is_trained = True
        self.save()

    # ── Projection ──────────────────────────────────────────────────────

    def project(self, x: np.ndarray) -> np.ndarray:
        """
        Project a single face vector into face space.

        Ω = Uᵀ(x − Ψ)

        Parameters
        ----------
        x : shape (IMG_DIM,)

        Returns
        -------
        omega : shape (K,)  – weight vector in face space
        """
        if not self.is_trained:
            raise RuntimeError("PCAEngine is not trained yet.")
        return self.U.T @ (x - self.mean_face)               # (K,)

    def project_many(self, X: np.ndarray) -> np.ndarray:
        """
        Project a matrix of face vectors.

        Parameters
        ----------
        X : shape (N, IMG_DIM)

        Returns
        -------
        Omega : shape (N, K)
        """
        if not self.is_trained:
            raise RuntimeError("PCAEngine is not trained yet.")
        centered = X - self.mean_face                        # (N, IMG_DIM)
        return (self.U.T @ centered.T).T                     # (N, K)

    def reconstruct(self, omega: np.ndarray) -> np.ndarray:
        """Reconstruct a face from its weight vector (for debugging/visualisation)."""
        return self.U @ omega + self.mean_face               # (IMG_DIM,)

    # ── Persistence ─────────────────────────────────────────────────────

    def save(self, path: Path = EIGENFACES_PATH) -> None:
        """Save U and mean_face to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, U=self.U, mean_face=self.mean_face, K=np.array(self.K))

    def load(self, path: Path = EIGENFACES_PATH) -> bool:
        """
        Load U and mean_face from disk.

        Returns True if loaded successfully, False if no saved model exists.
        """
        if not path.exists():
            return False
        data = np.load(path)
        self.U = data["U"]
        self.mean_face = data["mean_face"]
        self.K = int(data["K"])
        self.is_trained = True
        return True

    # ── Diagnostics ─────────────────────────────────────────────────────

    def explained_variance_ratio(self, X: np.ndarray) -> np.ndarray:
        """Return cumulative explained variance ratios for the current basis."""
        if not self.is_trained:
            raise RuntimeError("PCAEngine is not trained.")
        A = (X - self.mean_face).T
        L = A.T @ A
        eigenvalues, _ = np.linalg.eigh(L)
        eigenvalues = np.sort(eigenvalues)[::-1]
        total = eigenvalues.sum()
        return np.cumsum(eigenvalues / total)
