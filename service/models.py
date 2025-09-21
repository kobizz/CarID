from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel


class ClassifyReq(BaseModel):
    image_b64: Optional[str] = None
    image_url: Optional[str] = None
    topk: Optional[int] = None
    accept_threshold: Optional[float] = None
    margin_threshold: Optional[float] = None
    debug: Optional[bool] = False
    crop_norm: Optional[Dict[str, float]] = None  # {"x0":0.80,"y0":0.40,"x1":1.0,"y1":0.65}
    return_cropped_b64: Optional[bool] = False


class ClassifyResp(BaseModel):
    label: Optional[str]
    make: Optional[str]
    model: Optional[str]
    score: float
    accepted: bool
    topk: List[Tuple[str, float]]
    debug: Optional[Dict[str, Any]] = None
    cropped_b64: Optional[str] = None


class AddReq(BaseModel):
    label: str                     # e.g., "mazda_cx5" or for negatives just "camera"
    is_negative: bool = False
    image_b64: Optional[str] = None
    image_url: Optional[str] = None
