import logging
from config import DEVICE, CLIP_MODEL, CLIP_PRETRAIN

logger = logging.getLogger(__name__)

# Global variables for lazy loading
CLIP_MODEL_INSTANCE = None
PREPROCESSOR = None
EMBED_DIM = None

# Known embedding dimensions for different models (avoids loading model just for dimension)
KNOWN_EMBED_DIMS = {
    "ViT-B-32": 512,
    "ViT-B-16": 512,
    "ViT-L-14": 768,
    "RN50": 1024,
    "RN101": 512
}


def _load_model():
    """Lazy load the OpenCLIP model only when needed"""
    global CLIP_MODEL_INSTANCE, PREPROCESSOR, EMBED_DIM

    if CLIP_MODEL_INSTANCE is not None:
        return  # Already loaded

    logger.info(f"Loading OpenCLIP {CLIP_MODEL}/{CLIP_PRETRAIN} on {DEVICE}...")

    # Import heavy dependencies only when needed
    import open_clip
    from torchvision import transforms
    import torch

    # Memory optimization: set thread count for CPU
    if DEVICE == "cpu":
        try:
            torch.set_num_threads(2)  # Conservative threading for stability
            torch.set_num_interop_threads(1)  # Reduce inter-op parallelism  
        except RuntimeError as e:
            logger.warning(f"Could not set PyTorch threading: {e}")
            # Continue anyway - threading settings are optimization, not critical

    CLIP_MODEL_INSTANCE, _, _ = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAIN, device=DEVICE
    )
    CLIP_MODEL_INSTANCE.eval()
    EMBED_DIM = CLIP_MODEL_INSTANCE.visual.output_dim

    # Fast preprocessing (OpenCLIP normalization)
    PREPROCESSOR = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    # Force garbage collection after loading
    import gc
    gc.collect()

    logger.info(f"Model loaded successfully. Embedding dimension: {EMBED_DIM}")


def get_model():
    """Get the model, loading it if necessary"""
    _load_model()
    return CLIP_MODEL_INSTANCE


def get_preprocessor():
    """Get the preprocessor, loading it if necessary"""
    _load_model()
    return PREPROCESSOR


def get_embed_dim():
    """Get the embedding dimension without loading the full model if possible"""
    # Try to get known dimension first to avoid loading model
    if CLIP_MODEL in KNOWN_EMBED_DIMS:
        return KNOWN_EMBED_DIMS[CLIP_MODEL]

    # If unknown model, fall back to loading model
    _load_model()
    return EMBED_DIM


def unload_model():
    """Unload the model to free memory (if needed for maintenance)"""
    global CLIP_MODEL_INSTANCE, PREPROCESSOR, EMBED_DIM

    if CLIP_MODEL_INSTANCE is not None:
        logger.info("Unloading OpenCLIP model to free memory")
        del CLIP_MODEL_INSTANCE
        del PREPROCESSOR
        CLIP_MODEL_INSTANCE = None
        PREPROCESSOR = None
        EMBED_DIM = None

        # Force garbage collection
        import gc
        gc.collect()
        logger.info("Model unloaded successfully")
