# Spectral Face Recognition

Face recognition using **Principal Component Analysis (PCA)** and **ChromaDB** as the vector store.

## How it works

```
Image → grayscale → resize 100×100 → flatten → (10 000,) vector x

Training:
  Ψ = mean of all x vectors          (mean face)
  A = deviation matrix  (x - Ψ).T
  L = AᵀA               (surrogate covariance, N×N not d×d)
  U = top-K eigenvectors of L projected back to image space

Recognition:
  Ω = Uᵀ(x − Ψ)         (project face into K-dim face space)
  ChromaDB finds nearest Ω among all registered users
```

## Stack

| Layer    | Technology |
|----------|-----------|
| Backend  | FastAPI + Uvicorn |
| Math     | NumPy / SciPy |
| Images   | Pillow |
| Vector DB| ChromaDB |
| Frontend | HTML + CSS + Vanilla JS |

## Project structure

```
spectral-face-recognition/
├── backend/
│   ├── main.py           ← FastAPI routes
│   ├── pca_engine.py     ← PCA math (Ψ, A, L, U, Ω)
│   ├── image_utils.py    ← image → vector preprocessing
│   ├── chroma_store.py   ← ChromaDB wrapper
│   └── config.py         ← constants and paths
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── requirements.txt
└── README.md
```

## Setup & run

```bash
# 1. Clone and enter the project
git clone https://github.com/Rohith-S636/Spectral-Face-Recognition.git
cd Spectral-Face-Recognition

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the backend
cd backend
uvicorn main:app --reload

# 5. Open the frontend
# Open frontend/index.html directly in your browser
```

API docs available at **http://localhost:8000/docs** once the server is running.

## API endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/health` | Server + model status |
| POST | `/register` | Register a face (name + image) |
| POST | `/verify` | Verify a face image |
| GET | `/users` | List registered users |
| DELETE | `/users/{name}` | Remove a user |
| POST | `/retrain` | Force PCA rebuild |

## Configuration (`backend/config.py`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `IMG_SIZE` | 100 | Resize images to 100×100 |
| `K` | 50 | Number of eigenfaces |
| `THRESHOLD` | 0.35 | Cosine distance accept threshold |
