from typing import List, Optional, Dict, Tuple
import logging

import numpy as np
import faiss

logger = logging.getLogger(__name__)

from config import PROTOTYPE_MODE
from storage import (
    save_meta, save_pos_index, save_neg_index, save_prototypes,
    load_indexes, load_prototypes
)
from image_utils import list_gallery, embed_image, load_image
from ml_models import EMBED_DIM


# ---------- Globals (indexes) ----------
index_pos: Optional[faiss.Index] = None
labels_pos: List[str] = []
index_neg: Optional[faiss.Index] = None
prototypes: Dict[str, Dict[str, List[float] | int]] = {}  # {label: {"sum":[...], "count": int}}


def load_all():
    """Load persisted artifacts; build pos index from prototypes if needed."""
    global index_pos, labels_pos, index_neg, prototypes

    if PROTOTYPE_MODE:
        prototypes = load_prototypes()
        if prototypes:
            # Build index from prototypes
            labels_pos = sorted(prototypes.keys())
            vecs = []
            for lbl in labels_pos:
                rec = prototypes[lbl]
                s = np.array(rec["sum"], dtype="float32")
                c = max(1, int(rec["count"]))
                mu = s / float(c)
                mu = mu / max(1e-12, np.linalg.norm(mu))
                vecs.append(mu[None, :])
            if vecs:
                X = np.vstack(vecs).astype("float32")
                index_pos = faiss.IndexFlatIP(X.shape[1])
                index_pos.add(X)
            else:
                index_pos = None
                labels_pos = []
        else:
            index_pos = None
            labels_pos = []
    else:
        index_pos, labels_pos, _ = load_indexes()

    # Load negative index
    _, _, index_neg = load_indexes()


def rebuild_index():
    """Recompute positive index (and prototypes if enabled) + negative index from GCS storage."""
    items = list_gallery()
    if not items:
        return {"ok": False, "error": "Gallery empty (no images found in GCS bucket)"}

    pos_feats, pos_labels = [], []
    neg_feats = []

    proto_accum: Dict[str, Dict[str, np.ndarray | int]] = {}

    for label, path_or_blob, is_neg in items:
        try:
            pil = load_image(path_or_blob)
            vec = embed_image(pil)  # (1, D)
            if is_neg:
                neg_feats.append(vec)
            else:
                if PROTOTYPE_MODE:
                    rec = proto_accum.get(label)
                    if rec is None:
                        proto_accum[label] = {"sum": vec.copy(), "count": 1}
                    else:
                        rec["sum"] += vec
                        rec["count"] += 1
                else:
                    pos_feats.append(vec)
                    pos_labels.append(label)
        except Exception as e:
            logger.warning(f"Skipping {path_or_blob}: {e}")

    global index_pos, labels_pos, index_neg, prototypes

    if PROTOTYPE_MODE:
        prototypes = {}
        labels_pos = sorted(proto_accum.keys())
        vecs = []
        for lbl in labels_pos:
            s = proto_accum[lbl]["sum"].astype("float32")[0]
            c = int(proto_accum[lbl]["count"])
            mu = s / max(1, c)
            mu = mu / max(1e-12, np.linalg.norm(mu))
            vecs.append(mu[None, :])
            prototypes[lbl] = {"sum": s.tolist(), "count": c}
        if vecs:
            X = np.vstack(vecs).astype("float32")
            index_pos = faiss.IndexFlatIP(X.shape[1])
            index_pos.add(X)
        else:
            index_pos = None
            labels_pos = []
        save_prototypes(prototypes)
        if index_pos is not None:
            save_pos_index(index_pos, labels_pos)
    else:
        if not pos_feats:
            index_pos = None
            labels_pos = []
        else:
            X = np.vstack(pos_feats).astype("float32")
            index_pos = faiss.IndexFlatIP(X.shape[1])
            index_pos.add(X)
            labels_pos = pos_labels
            save_pos_index(index_pos, labels_pos)

    if neg_feats:
        Xn = np.vstack(neg_feats).astype("float32")
        index_neg = faiss.IndexFlatIP(Xn.shape[1])
        index_neg.add(Xn)
        save_neg_index(index_neg)
    else:
        index_neg = None
        save_neg_index(None)

    save_meta()
    return {
        "ok": True,
        "prototype_mode": PROTOTYPE_MODE,
        "positives": 0 if index_pos is None else index_pos.ntotal,
        "negatives": 0 if index_neg is None else index_neg.ntotal,
        "classes": labels_pos
    }


def add_to_index(label: str, vec: np.ndarray, is_negative: bool) -> Dict:
    """Add a single vector to the appropriate index"""
    global index_pos, labels_pos, index_neg

    if is_negative:
        if index_neg is None:
            index_neg = faiss.IndexFlatIP(EMBED_DIM)
        index_neg.add(vec)
        save_neg_index(index_neg)
        return {"neg_count": index_neg.ntotal}

    # Positive
    if PROTOTYPE_MODE:
        # Update prototype sums/counts
        if label not in prototypes:
            prototypes[label] = {"sum": vec[0].tolist(), "count": 1}
        else:
            s = np.array(prototypes[label]["sum"], dtype="float32")
            c = int(prototypes[label]["count"])
            s += vec[0]
            c += 1
            prototypes[label] = {"sum": s.tolist(), "count": c}
        save_prototypes(prototypes)

        # Rebuild small pos index from prototypes (quick; num_classes is small)
        labels_pos = sorted(prototypes.keys())
        vecs = []
        for lbl in labels_pos:
            s = np.array(prototypes[lbl]["sum"], dtype="float32")
            c = max(1, int(prototypes[lbl]["count"]))
            mu = s / float(c)
            mu = mu / max(1e-12, np.linalg.norm(mu))
            vecs.append(mu[None, :])
        if vecs:
            X = np.vstack(vecs).astype("float32")
            index_pos = faiss.IndexFlatIP(X.shape[1])
            index_pos.add(X)
            save_pos_index(index_pos, labels_pos)
        else:
            index_pos = None
            labels_pos = []
        save_meta()
        return {"classes": labels_pos, "count": index_pos.ntotal if index_pos else 0}
    else:
        # Per-image mode: append vector + label
        if index_pos is None:
            index_pos = faiss.IndexFlatIP(EMBED_DIM)
        index_pos.add(vec)
        labels_pos.append(label)
        save_pos_index(index_pos, labels_pos)
        save_meta()
        return {"pos_count": index_pos.ntotal}


def get_index_stats():
    """Get statistics about the current indexes"""
    if PROTOTYPE_MODE:
        counts = {k: int(v["count"]) for k, v in prototypes.items()}
    else:
        # per-image mode: count by scanning GCS gallery
        counts = {}
        try:
            items = list_gallery()
            for label, _, is_neg in items:
                if not is_neg:  # Only count positive samples
                    counts[label] = counts.get(label, 0) + 1
        except Exception:
            # If gallery scanning fails, return zeros
            for lbl in labels_pos:
                counts[lbl] = 0
    return {
        "prototype_mode": PROTOTYPE_MODE,
        "class_counts": counts,
        "neg_count": 0 if index_neg is None else index_neg.ntotal
    }


def search_indexes(query_vec: np.ndarray, k: int) -> Tuple[List[Tuple[str, float]], float]:
    """Search both positive and negative indexes, return ranked results and best negative score"""
    # Search positives
    sims_pos, ids_pos = index_pos.search(query_vec, k)
    sims_pos = sims_pos[0].tolist()
    ids_pos = ids_pos[0].tolist()
    ranked = [(labels_pos[i], float(s)) for i, s in zip(ids_pos, sims_pos) if i != -1]

    # Best negative similarity
    neg_top = 0.0
    if index_neg is not None and index_neg.ntotal > 0:
        sims_neg, _ = index_neg.search(query_vec, 1)
        neg_top = float(sims_neg[0][0])

    return ranked, neg_top


# Initialize on import
save_meta()
load_all()
