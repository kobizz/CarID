import io
import base64
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from config import DEVICE
from ml_models import get_model, get_preprocessor
from gcs_storage import get_gcs_storage


def pil_from_b64(b64: str) -> Image.Image:
    if "," in b64 and ";base64" in b64.split(",")[0]:
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def download_image(url: str) -> Image.Image:
    import requests
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


@torch.no_grad()
def embed_image(pil: Image.Image) -> np.ndarray:
    """Embed PIL image using OpenCLIP."""
    preprocessor = get_preprocessor()
    model = get_model()
    t = preprocessor(pil).unsqueeze(0).to(DEVICE)  # pylint: disable=not-callable
    feat = model.encode_image(t)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype("float32")  # (1, D)


def list_gallery():
    """List all images in the GCS gallery."""
    gcs = get_gcs_storage()
    return gcs.list_gallery()


def save_image(pil_image: Image.Image, folder_name: str, filename: str) -> str:
    """Save image to GCS storage.

    Args:
        pil_image: PIL Image object
        folder_name: Folder/label name
        filename: Image filename

    Returns:
        GCS blob path where image was saved
    """
    gcs = get_gcs_storage()
    return gcs.save_image(pil_image, folder_name, filename)


def load_image(blob_path: str) -> Image.Image:
    """Load image from GCS storage.

    Args:
        blob_path: GCS blob path

    Returns:
        PIL Image object
    """
    gcs = get_gcs_storage()
    return gcs.load_image(blob_path)


def split_label(lbl: str) -> Tuple[str, str]:
    parts = lbl.replace("-", "_").split("_", 1)
    return (parts[0], parts[1] if len(parts) > 1 else "")
