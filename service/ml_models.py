import logging
from config import DEVICE, CLIP_MODEL, CLIP_PRETRAIN

logger = logging.getLogger(__name__)

# Global variables for lazy loading
clip_model = None
preproc = None
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
    global clip_model, preproc, EMBED_DIM
    
    if clip_model is not None:
        return  # Already loaded
    
    logger.info(f"Loading OpenCLIP {CLIP_MODEL}/{CLIP_PRETRAIN} on {DEVICE}...")
    
    # Import heavy dependencies only when needed
    import open_clip
    from torchvision import transforms
    import torch
    
    # Memory optimization: set thread count for CPU
    if DEVICE == "cpu":
        torch.set_num_threads(2)  # Reduce threading on RPi
    
    clip_model, _, _ = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAIN, device=DEVICE
    )
    clip_model.eval()
    EMBED_DIM = clip_model.visual.output_dim

    # Fast preprocessing (OpenCLIP normalization)
    preproc = transforms.Compose([
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
    return clip_model

def get_preprocessor():
    """Get the preprocessor, loading it if necessary"""
    _load_model()
    return preproc

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
    global clip_model, preproc, EMBED_DIM
    
    if clip_model is not None:
        logger.info("Unloading OpenCLIP model to free memory")
        del clip_model
        del preproc
        clip_model = None
        preproc = None
        EMBED_DIM = None
        
        # Force garbage collection
        import gc
        gc.collect()
        logger.info("Model unloaded successfully")
