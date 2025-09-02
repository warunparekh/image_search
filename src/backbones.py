"""Backbone model factory for global tile descriptors.

Supported (tries strongest variants first):
 - dinov2 (vit_large_patch14_dinov2, vit_base_patch14_dinov2, vit_small_patch14_dinov2)

Each backbone exposes an `encode(image: PIL.Image) -> np.ndarray` returning an L2-normalized 1D float32 vector.
Use `get_backbone('best')` to attempt loading the strongest available backbone (DINOv2 L/14 -> B/14 -> S/14).
"""
from __future__ import annotations

from dataclasses import dataclass
import os
# Optional: allow disabling flash / efficient SDP attention kernels to suppress warnings on unsupported builds
if os.environ.get('DISABLE_FLASH_ATTN', '1') == '1':
    os.environ.setdefault('TORCH_FORCE_DISABLE_SDP_KERNELS', '1')  # PyTorch 2.x
    os.environ.setdefault('TORCH_BACKEND_FLASH_ATTENTION', '0')
from typing import Callable, Optional
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

# Redirect all framework caches (torch / timm / huggingface / generic XDG) into the project so
# DINO weights are downloaded inside the repository instead of the user home .cache directory.
# Can be customized via PROJECT_MODEL_ROOT env var.
PROJECT_MODEL_ROOT = Path(os.environ.get('PROJECT_MODEL_ROOT', 'data/models')).resolve()
try:
    PROJECT_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
except Exception:
    pass
for _ev in ('TORCH_HOME', 'HF_HOME', 'XDG_CACHE_HOME', 'TIMM_HOME'):
    os.environ.setdefault(_ev, str(PROJECT_MODEL_ROOT))


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    # Explicitly disable flash/efficient attention kernels at runtime (PyTorch API)
    if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'sdp_kernel'):  # type: ignore
        torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)  # type: ignore
except Exception:
    pass

IMAGENET_PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@dataclass
class Backbone:
    name: str
    _forward: Callable[[torch.Tensor], torch.Tensor]
    preprocess: Callable[[Image.Image], torch.Tensor]

    @torch.no_grad()
    def encode(self, img: Image.Image) -> np.ndarray:
        t = self.preprocess(img).unsqueeze(0).to(DEVICE)
        feat = self._forward(t).squeeze().detach().cpu().float().numpy()
        n = np.linalg.norm(feat)
        if n > 0:
            feat /= n
        return feat.astype(np.float32)


def _dinov2_vits14() -> Optional[Backbone]:
    """Load ONLY the largest DINOv2 variant (vit/14 large) using local weights or torch.hub.

    Logic:
      1. If local file exists (any of: vit_large_patch14_dinov2.pth or dinov2_vitl14.pth) load it.
      2. Else if DINO_FORCE_LOCAL=1 -> abort (no download).
      3. Else download hub model 'dinov2_vitl14', save to data/models/dinov2/dinov2_vitl14.pth and return.
    """
    weights_dir = Path(os.environ.get('DINO_WEIGHTS_DIR', 'data/models/dinov2')).resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)
    force_local = os.environ.get('DINO_FORCE_LOCAL', '0') == '1'

    candidates = [
        weights_dir / 'vit_large_patch14_dinov2.pth',
        weights_dir / 'dinov2_vitl14.pth'
    ]
    for c in candidates:
        if c.is_file():
            try:
                model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')  # type: ignore
                state = torch.load(str(c), map_location='cpu')
                if isinstance(state, dict) and 'state_dict' in state:
                    state = state['state_dict']
                missing, unexpected = model.load_state_dict(state, strict=False)
                if missing:
                    print(f'[backbones] Warning: {len(missing)} missing keys in {c.name}')
                if unexpected:
                    print(f'[backbones] Warning: {len(unexpected)} unexpected keys in {c.name}')
                model.eval().to(DEVICE)
                print(f'[backbones] Loaded local DINOv2 large weights from {c.name}')
                return Backbone('dinov2::dinov2_vitl14', lambda x: model(x), IMAGENET_PREPROCESS)
            except Exception as e:
                print(f'[backbones] Failed loading local large weights {c}: {e}')
                if force_local:
                    return None
                break  # try download

    if force_local:
        print('[backbones] DINO_FORCE_LOCAL=1 and no local vitl14 weights found.')
        return None

    # Download hub model and save
    try:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')  # type: ignore
        model.eval().to(DEVICE)
        save_path = weights_dir / 'dinov2_vitl14.pth'
        if not save_path.exists():
            try:
                torch.save(model.state_dict(), str(save_path))
                print(f'[backbones] Downloaded + saved dinov2_vitl14 to {save_path}')
            except Exception as se:
                print(f'[backbones] Warning: could not save vitl14 weights: {se}')
        return Backbone('dinov2::dinov2_vitl14', lambda x: model(x), IMAGENET_PREPROCESS)
    except Exception as e:
        print('[backbones] Failed to download dinov2_vitl14 via torch.hub:', e)
        return None


def get_backbone(name: str, strict: bool = False, verbose: bool = True) -> Backbone:
    """Return backbone; optionally raise if requested unavailable.

    strict=True: raise RuntimeError instead of silent fallback.
    verbose=True: print fallback reason.
    """
    req = name
    # Normalize recorded variant names like 'dinov2::vit_large_patch14_dinov2' -> 'dinov2'
    if isinstance(name, str) and name.startswith('dinov2::'):
        name = 'dinov2'
    name = name.lower()
    # Accept 'best' as an alias for the strongest available DINOv2 variant.
    if name in ('dinov2_vits14', 'dinov2', 'best'):
        m = _dinov2_vits14()
        if m is not None:
            return m
        msg = 'DINOv2 unavailable (timm/torch.hub failed)'
        if strict:
            raise RuntimeError(msg)
        if verbose:
            print('[backbones]', msg)
        raise RuntimeError(msg)
    if strict:
        raise RuntimeError(f'Unsupported backbone "{req}". Only DINOv2 (use "dinov2" or "best") is supported.')
    raise RuntimeError(f'Unsupported backbone "{req}". Only DINOv2 (use "dinov2" or "best") is supported.')
