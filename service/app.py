import uuid
import base64
import io
import logging
import sys
from datetime import datetime

from fastapi import HTTPException, Request, FastAPI

from config import (
    NEG_PREFIX, TOPK, ACCEPT_THRESHOLD,
    MARGIN_THRESHOLD, NEG_ACCEPT_CAP, PROTOTYPE_MODE
)
from models import ClassifyReq, ClassifyResp, AddReq
from image_utils import pil_from_b64, download_image, embed_image, split_label, save_image
from index_manager import (
    index_pos, labels_pos, index_neg,
    rebuild_index, add_to_index, get_index_stats, search_indexes
)
from ml_models import unload_model

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Create logger for this module
logger = logging.getLogger(__name__)

# Configure uvicorn loggers to use the same format
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_error_logger = logging.getLogger("uvicorn.error")
uvicorn_logger = logging.getLogger("uvicorn")

# Set up the formatter for all uvicorn loggers
formatter = logging.Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Apply formatter to all existing handlers
for logger_obj in [uvicorn_access_logger, uvicorn_error_logger, uvicorn_logger]:
    for handler in logger_obj.handlers:
        handler.setFormatter(formatter)

logger.info("Starting CarID service with timestamp logging enabled")

app = FastAPI(title="CarID (OpenCLIP + FAISS + Negatives + Prototype Mode)")


@app.get("/healthz")
def healthz():
    logger.info("Health check requested")
    from ml_models import clip_model
    
    return {
        "ok": True,
        "prototype_mode": PROTOTYPE_MODE,
        "model_loaded": clip_model is not None,
        "pos_count": 0 if index_pos is None else index_pos.ntotal,
        "neg_count": 0 if index_neg is None else index_neg.ntotal,
        "classes": labels_pos
    }


@app.get("/index/stats")
def index_stats():
    return get_index_stats()


@app.post("/system/memory/free")
def free_memory():
    """Free model memory (useful for maintenance or memory pressure)"""
    logger.info("Memory cleanup requested")
    unload_model()
    
    # Force garbage collection
    import gc
    gc.collect()
    
    return {"ok": True, "message": "Model unloaded and memory freed"}


@app.post("/index/rebuild")
def index_rebuild():
    """Recompute positive index (and prototypes if enabled) + negative index from disk."""
    logger.info("Index rebuild requested")
    result = rebuild_index()
    logger.info("Index rebuild completed")
    return result


@app.post("/index/add")
def index_add(req: AddReq):
    """Append a single image to positives/negatives WITHOUT full rebuild.
       - In PROTOTYPE_MODE: updates that class prototype & rebuilds small pos index (fast).
       - In per-image mode: appends to pos index or neg index directly.
       Also saves the image to the gallery on disk.
    """
    logger.info(f"Adding image to index - label: {req.label}, is_negative: {req.is_negative}")
    
    # Load image
    if req.image_b64:
        pil = pil_from_b64(req.image_b64)
    elif req.image_url:
        pil = download_image(req.image_url)
        logger.info(f"Downloaded image from URL: {req.image_url}")
    else:
        raise ValueError("Provide image_b64 or image_url")

    # Save to storage
    label = req.label.strip().lower().replace(" ", "_")
    folder_name = label if not req.is_negative else (
        label if label.startswith(NEG_PREFIX) else f"{NEG_PREFIX}{label}"
    )
    fname = f"{uuid.uuid4().hex}.jpg"
    saved_path = save_image(pil, folder_name, fname)

    # Embed
    vec = embed_image(pil).astype("float32")  # (1, D)

    # Update indexes
    result = add_to_index(label, vec, req.is_negative)
    result["ok"] = True
    result["saved_as"] = saved_path

    return result


@app.post("/classify", response_model=ClassifyResp)
def classify(req: ClassifyReq, request: Request):
    logger.info("Classification request received")
    
    if index_pos is None or (index_pos.ntotal == 0):
        logger.error("Index not built - cannot perform classification")
        raise RuntimeError("Index not built; POST /index/rebuild first and/or add images.")

    # Parse debug flag (either body.debug or ?debug=1)
    dbg = bool(req.debug) or (
        request.query_params.get("debug", "").lower() in ("1", "true", "yes", "on")
    )
    
    if dbg:
        logger.info("Debug mode enabled for this classification request")

    # Load image
    if req.image_b64:
        pil = pil_from_b64(req.image_b64)
    elif req.image_url:
        try:
            pil = download_image(req.image_url)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"download_error: {e}") from e
    else:
        raise ValueError("Provide image_b64 or image_url")

    if req.crop_norm:
        x0 = max(0.0, min(1.0, float(req.crop_norm.get("x0", 0.0))))
        y0 = max(0.0, min(1.0, float(req.crop_norm.get("y0", 0.0))))
        x1 = max(0.0, min(1.0, float(req.crop_norm.get("x1", 1.0))))
        y1 = max(0.0, min(1.0, float(req.crop_norm.get("y1", 1.0))))
        if x1 <= x0:
            x1 = min(1.0, x0 + 1e-3)
        if y1 <= y0:
            y1 = min(1.0, y0 + 1e-3)
        W, H = pil.width, pil.height
        box = (int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H))
        pil = pil.crop(box)

    # Embed query
    q = embed_image(pil)

    # Search indexes
    k = req.topk or TOPK
    ranked, neg_top = search_indexes(q, k)

    if not ranked:
        logger.warning("No results found in classification search")
        return ClassifyResp(
            label=None, make=None, model=None, score=0.0,
            accepted=False, topk=[], debug=None
        )

    top1_lbl, top1_sim = ranked[0]
    top2_sim = ranked[1][1] if len(ranked) > 1 else 0.0
    logger.info(f"Classification result - top match: {top1_lbl} (score: {top1_sim:.3f})")

    thr = (req.accept_threshold
           if req.accept_threshold is not None
           else ACCEPT_THRESHOLD)
    margin = (req.margin_threshold
              if req.margin_threshold is not None
              else MARGIN_THRESHOLD)

    cond_thr = top1_sim >= thr
    cond_margin = (top1_sim - max(top2_sim, neg_top)) >= margin
    cond_negcap = neg_top <= NEG_ACCEPT_CAP
    accepted = cond_thr and cond_margin and cond_negcap

    make, model = split_label(top1_lbl)

    debug_obj = None
    if dbg:
        debug_obj = {
            "top1_label": top1_lbl,
            "top1_sim": top1_sim,
            "top2_sim": top2_sim,
            "neg_top": neg_top,
            "thr": thr,
            "margin": margin,
            "neg_cap": NEG_ACCEPT_CAP,
            "checks": {
                "threshold": cond_thr,
                "margin": cond_margin,
                "neg_cap": cond_negcap
            },
            "prototype_mode": PROTOTYPE_MODE,
            "pos_index_size": 0 if index_pos is None else index_pos.ntotal,
            "neg_index_size": 0 if index_neg is None else index_neg.ntotal
        }

    cropped_b64 = None
    if req.return_cropped_b64:
        buf = io.BytesIO()
        pil.save(buf, "JPEG", quality=90)
        cropped_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    if not accepted:
        logger.info(f"Classification rejected - label: {top1_lbl}, score: {top1_sim:.3f}, conditions: thr={cond_thr}, margin={cond_margin}, neg_cap={cond_negcap}")
        return ClassifyResp(
            label=None, make=None, model=None, score=float(top1_sim),
            accepted=False, topk=ranked[:k], debug=debug_obj, cropped_b64=cropped_b64
        )

    logger.info(f"Classification accepted - label: {top1_lbl}, make: {make}, model: {model}, score: {top1_sim:.3f}")
    return ClassifyResp(
        label=top1_lbl, make=make, model=model, score=float(top1_sim),
        accepted=True, topk=ranked[:k], debug=debug_obj, cropped_b64=cropped_b64
    )
