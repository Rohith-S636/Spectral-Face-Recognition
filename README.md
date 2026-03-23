# Spectral-Face-Recognition
Spectral-Face-Recognition is a computer vision implementation that identifies individuals using Principal Component Analysis (PCA) and Orthogonal Projections. By solving the eigenvalue problem, the system extracts unique spectral signatures and maps images into a low-dimensional "Face Space" for efficient classification.

## 🚀 Overview
Traditional facial recognition can be computationally expensive. This project utilizes Eigenfaces, a statistical approach that reduces the dimensionality of image data while preserved the most significant features (variance).
* Key Features
  *  Dimensionality Reduction: Compresses high-resolution images into essential weight vectors.
  *  Spectral Analysis: Uses Eigenvalue Decomposition to find the principal components of facial structures.
  *  Efficient Inference: Recognizes faces using Euclidean distance measurements in a reduced vector space.
    
## 🧬 Mathematical Foundation
The algorithm follows a four-stage pipeline:
  * Preprocessing: Flattening $N \times N$ images into $N^2$ vectors and calculating the Mean Face ($\Psi$).
  * Covariance Matrix: Computing the deviation matrix $A$ and solving the eigenvalue problem for $L = A^TA$.
  * Eigenface Construction: Projecting eigenvectors back to image space to create the Orthonormal Basis.
  * Orthogonal Projection: Representing new faces as a linear combination of these basis vectors:
$$\Omega = U^T (x - \Psi)$$
