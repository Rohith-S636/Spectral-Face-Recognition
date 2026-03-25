from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
MODEL_DIR      = BASE_DIR / "models"
DATA_DIR       = BASE_DIR / "data"
CHROMA_DB_PATH = BASE_DIR / "backend" / "chroma_db"

MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

EIGENFACES_PATH = MODEL_DIR / "eigenfaces.npz"   # stores U and mean_face
RAW_FACES_PATH  = DATA_DIR  / "raw_faces.npz"    # stores raw pixel vectors + labels

# ── PCA settings ────────────────────────────────────────────────────────
IMG_SIZE = 100          # images resized to IMG_SIZE × IMG_SIZE
IMG_DIM  = IMG_SIZE * IMG_SIZE   # flattened vector length = 10 000
K        = 50           # number of eigenfaces (top-K components)

# ── Verification threshold ──────────────────────────────────────────────
# ChromaDB returns cosine distance (0 = identical, 2 = opposite).
# Tune this after testing with real faces.
THRESHOLD = 0.35        # distances below this → accepted match

# ── ChromaDB ────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "faces"
