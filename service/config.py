import os
from pathlib import Path

# Try to load Home Assistant addon configuration
try:
    from addon_config import get_addon_service_config
    addon_config = get_addon_service_config()
    # Apply addon config to environment
    for key, value in addon_config.items():
        os.environ[key] = value
except ImportError:
    # Not running as Home Assistant addon
    pass

# ---------- Paths / Config ----------
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Google Cloud Storage configuration (mandatory)
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "parking-trained-images")
GCS_CREDENTIALS_PATH = os.getenv("GCS_CREDENTIALS_PATH", None)  # Path to service account JSON

DEVICE = os.getenv("DEVICE", "cpu")
CLIP_MODEL = os.getenv("CLIP_MODEL", "ViT-B-32")
CLIP_PRETRAIN = os.getenv("CLIP_PRETRAIN", "laion2b_s34b_b79k")


TOPK = int(os.getenv("TOPK", "5"))
ACCEPT_THRESHOLD = float(os.getenv("ACCEPT_THRESHOLD", "0.80"))
MARGIN_THRESHOLD = float(os.getenv("MARGIN_THRESHOLD", "0.04"))
NEG_ACCEPT_CAP = float(os.getenv("NEG_ACCEPT_CAP", "0.80"))

NEG_PREFIX = os.getenv("NEG_PREFIX", "_")
PROTOTYPE_MODE = os.getenv("PROTOTYPE_MODE", "true").lower() in ("1", "true", "yes")

# Persisted files
INDEX_PATH = DATA_DIR / "index.faiss"          # positives
LABELS_PATH = DATA_DIR / "labels.json"         # order of positives
NEG_INDEX_PATH = DATA_DIR / "index_neg.faiss"  # negatives (per-image)
PROTO_PATH = DATA_DIR / "prototypes.json"      # sums & counts per class (prototype mode)
META_PATH = DATA_DIR / "meta.json"             # embed_dim, mode
