import json
from typing import List, Dict, Optional

import faiss

from config import (
    INDEX_PATH, LABELS_PATH, NEG_INDEX_PATH, PROTO_PATH, META_PATH,
    PROTOTYPE_MODE
)
from ml_models import EMBED_DIM


def save_meta():
    META_PATH.write_text(json.dumps({
        "embed_dim": EMBED_DIM,
        "prototype_mode": PROTOTYPE_MODE
    }))


def save_pos_index(ix: faiss.Index, labels: List[str]):
    faiss.write_index(ix, str(INDEX_PATH))
    LABELS_PATH.write_text(json.dumps(labels, ensure_ascii=False))


def save_neg_index(ix: Optional[faiss.Index]):
    if ix is None:
        if NEG_INDEX_PATH.exists():
            NEG_INDEX_PATH.unlink()
        return
    faiss.write_index(ix, str(NEG_INDEX_PATH))


def save_prototypes(proto: Dict[str, Dict[str, List[float] | int]]):
    PROTO_PATH.write_text(json.dumps(proto))  # raw sums+counts


def load_indexes():
    """Load persisted indexes. Returns (index_pos, labels_pos, index_neg)"""
    index_pos = None
    labels_pos = []
    index_neg = None
    
    if INDEX_PATH.exists() and LABELS_PATH.exists():
        index_pos = faiss.read_index(str(INDEX_PATH))
        labels_pos = json.loads(LABELS_PATH.read_text())
    
    if NEG_INDEX_PATH.exists():
        index_neg = faiss.read_index(str(NEG_INDEX_PATH))
        
    return index_pos, labels_pos, index_neg


def load_prototypes():
    """Load prototypes if they exist"""
    if PROTO_PATH.exists():
        return json.loads(PROTO_PATH.read_text())
    return {}
