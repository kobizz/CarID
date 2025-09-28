import uuid
import base64
import io
import logging
import re
import sys

from fastapi import HTTPException, FastAPI
from PIL import Image

from config import (
    NEG_PREFIX, TOPK, ACCEPT_THRESHOLD,
    MARGIN_THRESHOLD, NEG_ACCEPT_CAP, PROTOTYPE_MODE, JPEG_QUALITY
)
from models import ClassifyReq, ClassifyResp, AddReq
from image_utils import pil_from_b64, download_image, embed_image, split_label, save_image
from ml_models import embed_multimodal, unload_model
from index_manager import (
    index_pos, labels_pos, index_neg,
    rebuild_index, add_to_index, get_index_stats, search_indexes, load_all
)
from storage import (
    force_backup_now, get_backup_status, cleanup_old_backups_all,
    list_index_versions, list_prototype_versions
)

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


def validate_label(label: str) -> str:
    """Validate and sanitize label input to prevent path traversal and injection attacks."""
    if not label or not isinstance(label, str):
        raise ValueError("Label must be a non-empty string")

    # Check length
    if len(label) > 100:
        raise ValueError("Label too long (max 100 characters)")

    # Remove dangerous characters and normalize
    label = label.strip()

    # Check for path traversal attempts
    if '..' in label or '/' in label or '\\' in label:
        raise ValueError("Label contains invalid path characters")

    # Allow only alphanumeric, underscore, hyphen, and spaces (will be converted to underscore)
    if not re.match(r'^[a-zA-Z0-9_\-\s]+$', label):
        raise ValueError(
            "Label contains invalid characters "
            "(only alphanumeric, underscore, hyphen, and spaces allowed)"
        )

    # Convert to safe format
    safe_label = label.lower().replace(' ', '_').replace('-', '_')

    # Remove multiple consecutive underscores
    safe_label = re.sub(r'_{2,}', '_', safe_label)

    # Remove leading/trailing underscores
    safe_label = safe_label.strip('_')

    if not safe_label:
        raise ValueError("Label becomes empty after sanitization")

    return safe_label


def validate_image_input(req) -> None:
    """Validate image input for security and format compliance."""
    # Check that exactly one image source is provided
    if not req.image_b64 and not req.image_url:
        raise ValueError("Must provide either image_b64 or image_url")

    if req.image_b64 and req.image_url:
        raise ValueError("Provide only one of image_b64 or image_url, not both")

    # Validate base64 image if provided
    if req.image_b64:
        # Check for reasonable size limit (10MB in base64)
        if len(req.image_b64) > 14_000_000:  # ~10MB in base64
            raise ValueError("Image too large (max 10MB)")

        # Basic format validation for base64 images
        if req.image_b64.startswith('data:'):
            # Extract MIME type for data URLs
            try:
                header, _ = req.image_b64.split(',', 1)
                if 'image/' not in header.lower():
                    raise ValueError("Invalid image format in data URL")

                # Check for supported image types
                allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
                if not any(img_type in header.lower() for img_type in allowed_types):
                    raise ValueError("Unsupported image format (allowed: JPEG, PNG, WebP)")
            except ValueError as exc:
                raise ValueError("Invalid data URL format") from exc

    # Validate URL if provided
    if req.image_url:
        # Basic URL validation
        if not req.image_url.startswith(('http://', 'https://')):
            raise ValueError("Image URL must use HTTP or HTTPS protocol")

        if len(req.image_url) > 2000:
            raise ValueError("Image URL too long (max 2000 characters)")


@app.get("/healthz")
def healthz():
    logger.info("Health check requested")
    from ml_models import CLIP_MODEL_INSTANCE

    return {
        "ok": True,
        "prototype_mode": PROTOTYPE_MODE,
        "model_loaded": CLIP_MODEL_INSTANCE is not None,
        "pos_count": 0 if index_pos is None else index_pos.ntotal,
        "neg_count": 0 if index_neg is None else index_neg.ntotal,
        "classes": labels_pos
    }


@app.get("/index/basic-stats")
def basic_index_stats():
    return get_index_stats()


@app.post("/system/memory/free")
def free_memory():
    """Free model memory (useful for maintenance or memory pressure)"""
    logger.info("Memory cleanup requested")
    unload_model()

    # Force garbage collection
    import gc  # pylint: disable=import-outside-toplevel
    gc.collect()

    return {"ok": True, "message": "Model unloaded and memory freed"}


@app.post("/index/rebuild")
def index_rebuild():
    """Recompute positive index (and prototypes if enabled) + negative index from disk."""
    logger.info("Index rebuild requested")
    result = rebuild_index()
    logger.info("Index rebuild completed")
    return result


@app.post("/index/reload")
def index_reload():
    """Manually reload indexes from local/GCS storage without rebuilding from images"""
    logger.info("Index reload requested")
    try:
        load_all()
        logger.info("Index reload completed successfully")
        return {"ok": True, "message": "Indexes reloaded from storage"}
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Index reload failed: {e}")
        return {"ok": False, "error": str(e)}


@app.post("/index/add")
def index_add(req: AddReq):
    """Append a single image to positives/negatives WITHOUT full rebuild.
       - In PROTOTYPE_MODE: updates that class prototype & rebuilds small pos index (fast).
       - In per-image mode: appends to pos index or neg index directly.
       Also saves the image to the gallery on disk.
    """
    logger.info(f"Adding image to index - label: {req.label}, is_negative: {req.is_negative}")

    # Validate inputs
    try:
        validate_image_input(req)
        label = validate_label(req.label)
    except ValueError as e:
        logger.warning(f"Input validation failed: {e}")
        raise HTTPException(status_code=400, detail=f"Input validation error: {e}") from e

    # Load image
    if req.image_b64:
        try:
            pil = pil_from_b64(req.image_b64)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"Failed to parse base64 image: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image data: {e}") from e
    elif req.image_url:
        try:
            pil = download_image(req.image_url)
            logger.info(f"Downloaded image from URL: {req.image_url}")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"Failed to download image from URL: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to download image: {e}") from e
    else:
        raise HTTPException(status_code=400, detail="Provide image_b64 or image_url")

    # Save to storage
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
def classify(req: ClassifyReq):
    logger.info("Classification request received")

    if index_pos is None or (index_pos.ntotal == 0):
        # Get current status for better error message
        from index_manager import get_index_stats
        stats = get_index_stats()

        if PROTOTYPE_MODE:
            error_msg = (
                "No index available for classification. In PROTOTYPE_MODE, you need to either:\n"
                "1. Add images via POST /index/add (recommended for individual images)\n"
                "2. Upload images to your GCS bucket and run POST /index/rebuild\n"
                f"Current status: {stats['class_counts']} prototype classes"
            )
        else:
            error_msg = (
                "No index available for classification. In PER-IMAGE mode, you need to either:\n"
                "1. Add images via POST /index/add (recommended for individual images)\n"
                "2. Upload images to your GCS bucket and run POST /index/rebuild\n"
                f"Current status: {stats.get('pos_count', 0)} positive images indexed"
            )

        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Validate and load image
    try:
        validate_image_input(req)
    except ValueError as e:
        logger.warning(f"Classification input validation failed: {e}")
        raise HTTPException(status_code=400, detail=f"Input validation error: {e}") from e

    if req.image_b64:
        try:
            pil = pil_from_b64(req.image_b64)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"Failed to parse base64 image: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image data: {e}") from e
    elif req.image_url:
        try:
            pil = download_image(req.image_url)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"Failed to download image from URL: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to download image: {e}") from e
    else:
        raise HTTPException(status_code=400, detail="Provide image_b64 or image_url")

    if req.crop_norm:
        x0 = max(0.0, min(1.0, float(req.crop_norm.get("x0", 0.0))))
        y0 = max(0.0, min(1.0, float(req.crop_norm.get("y0", 0.0))))
        x1 = max(0.0, min(1.0, float(req.crop_norm.get("x1", 1.0))))
        y1 = max(0.0, min(1.0, float(req.crop_norm.get("y1", 1.0))))
        if x1 <= x0:
            x1 = min(1.0, x0 + 1e-3)
        if y1 <= y0:
            y1 = min(1.0, y0 + 1e-3)
        width, height = pil.width, pil.height
        box = (int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height))
        pil = pil.crop(box)

    # Convert to JPEG format before classification for consistent preprocessing
    # Use request-specific quality or fall back to config default
    jpeg_quality = req.jpeg_quality if req.jpeg_quality is not None else JPEG_QUALITY

    # Validate JPEG quality parameter
    if jpeg_quality < 1 or jpeg_quality > 100:
        logger.warning(f"Invalid JPEG quality: {jpeg_quality}, using default: {JPEG_QUALITY}")
        jpeg_quality = JPEG_QUALITY

    jpeg_buf = io.BytesIO()
    pil.save(jpeg_buf, "JPEG", quality=jpeg_quality)
    jpeg_bytes = jpeg_buf.getvalue()  # Save JPEG bytes for potential reuse
    jpeg_buf.seek(0)
    pil = Image.open(jpeg_buf).convert("RGB")

    k = req.topk or TOPK

    # Compute embedding and search based on whether text query is provided
    if req.text_query:
        # For debug comparison, we need both image-only and multimodal results
        if req.debug:
            q_image = embed_image(pil)
            ranked_image_only, neg_top_image_only = search_indexes(q_image, k)
        else:
            ranked_image_only, neg_top_image_only = None, None

        # Compute multimodal embedding and search
        q = embed_multimodal(pil, req.text_query)
        logger.info(f"Using multimodal embedding with text: '{req.text_query}'")
        ranked, neg_top = search_indexes(q, k)
    else:
        # No text query - use standard image-only embedding
        q = embed_image(pil)
        ranked, neg_top = search_indexes(q, k)
        ranked_image_only, neg_top_image_only = None, None

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
    if req.debug:
        logger.info("Debug mode enabled for this classification request")

        # Embedding comparison analysis
        embedding_analysis = None
        if req.text_query and ranked_image_only:
            # Compare image-only vs multimodal results
            img_top1_lbl, img_top1_sim = ranked_image_only[0]
            img_top2_sim = ranked_image_only[1][1] if len(ranked_image_only) > 1 else 0.0

            # Calculate ranking changes
            img_labels = [item[0] for item in ranked_image_only]
            multimodal_labels = [item[0] for item in ranked]

            # Find rank changes for top results
            rank_changes = {}
            for i, label in enumerate(multimodal_labels[:3]):  # Top 3
                img_rank = img_labels.index(label) if label in img_labels else -1
                if img_rank != -1:
                    rank_changes[label] = {
                        "image_only_rank": img_rank + 1,
                        "multimodal_rank": i + 1,
                        "rank_change": img_rank - i  # positive = improved with text
                    }

            embedding_analysis = {
                "image_only_results": {
                    "top1_label": img_top1_lbl,
                    "top1_sim": img_top1_sim,
                    "top2_sim": img_top2_sim,
                    "neg_top": neg_top_image_only,
                    "topk": ranked_image_only[:k]
                },
                "multimodal_results": {
                    "top1_label": top1_lbl,
                    "top1_sim": top1_sim,
                    "top2_sim": top2_sim,
                    "neg_top": neg_top,
                    "topk": ranked[:k]
                },
                "text_influence": {
                    "winner_changed": img_top1_lbl != top1_lbl,
                    "score_delta": top1_sim - img_top1_sim,
                    "neg_score_delta": neg_top - neg_top_image_only,
                    "rank_changes": rank_changes
                }
            }

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
            "neg_index_size": 0 if index_neg is None else index_neg.ntotal,
            "jpeg_quality_used": jpeg_quality,
            "jpeg_bytes_size": len(jpeg_bytes),
            "text_query": req.text_query,
            "embedding_analysis": embedding_analysis
        }

    cropped_b64 = None
    if req.return_cropped_b64:
        cropped_b64 = base64.b64encode(jpeg_bytes).decode("ascii")

    if not accepted:
        logger.info(
            f"Classification rejected - label: {top1_lbl}, score: {top1_sim:.3f}, "
            f"conditions: thr={cond_thr}, margin={cond_margin}, neg_cap={cond_negcap}"
        )
        return ClassifyResp(
            label=None, make=None, model=None, score=float(top1_sim),
            accepted=False, topk=ranked[:k], debug=debug_obj, cropped_b64=cropped_b64
        )

    logger.info(
        f"Classification accepted - label: {top1_lbl}, make: {make}, "
        f"model: {model}, score: {top1_sim:.3f}"
    )
    return ClassifyResp(
        label=top1_lbl, make=make, model=model, score=float(top1_sim),
        accepted=True, topk=ranked[:k], debug=debug_obj, cropped_b64=cropped_b64
    )


@app.post("/backup/force")
def force_backup():
    """Force an immediate backup of current indexes to GCS"""
    logger.info("Manual backup requested")
    result = force_backup_now()
    logger.info(f"Manual backup completed: {result}")
    return result


@app.get("/backup/status")
def backup_status():
    """Get current backup status and configuration"""
    return get_backup_status()


@app.post("/backup/cleanup")
def cleanup_backups():
    """Manually cleanup old backups, keeping only MAX_BACKUP_VERSIONS latest"""
    logger.info("Manual backup cleanup requested")
    result = cleanup_old_backups_all()
    logger.info(f"Manual backup cleanup completed: {result}")
    return result


@app.get("/backup/versions")
def list_backup_versions():
    """List all available backup versions for all backup types"""
    return {
        "positive": list_index_versions("positive"),
        "negative": list_index_versions("negative"),
        "prototypes": list_prototype_versions()
    }


@app.get("/index/status")
def index_status():
    """Get current index status - useful for debugging classification issues"""
    from index_manager import index_pos, index_neg, labels_pos, prototypes

    status = {
        "ready_for_classification": index_pos is not None and index_pos.ntotal > 0,
        "prototype_mode": PROTOTYPE_MODE,
        "positive_index": {
            "exists": index_pos is not None,
            "vector_count": 0 if index_pos is None else index_pos.ntotal,
            "label_count": len(labels_pos)
        },
        "negative_index": {
            "exists": index_neg is not None,
            "vector_count": 0 if index_neg is None else index_neg.ntotal
        }
    }

    if PROTOTYPE_MODE:
        status["prototypes"] = {
            "class_count": len(prototypes),
            "classes": list(prototypes.keys()),
            "total_samples": sum(proto.get("count", 0) for proto in prototypes.values())
        }
    else:
        status["labels"] = {
            "unique_labels": len(set(labels_pos)),
            "total_samples": len(labels_pos),
            "label_distribution": {label: labels_pos.count(label) for label in set(labels_pos)}
        }

    return status


@app.get("/index/stats")
def index_stats():
    """Get detailed index statistics including backup status"""
    basic_stats = get_index_stats()
    backup_info = get_backup_status()

    return {
        **basic_stats,
        "backup_status": backup_info
    }
