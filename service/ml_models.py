import torch
import open_clip
from torchvision import transforms
from config import DEVICE, CLIP_MODEL, CLIP_PRETRAIN

# ---------- Models ----------
print(f"Loading OpenCLIP {CLIP_MODEL}/{CLIP_PRETRAIN} on {DEVICE}...")
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
