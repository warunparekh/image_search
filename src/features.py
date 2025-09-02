"""Feature extraction utilities for tile similarity.

Produces many complementary descriptors used for robust tile retrieval:
 - Global backbone embedding (already L2-normalized)
 - Lab color histogram
 - Color moments (mean, std, skew) per channel
 - Dominant colors (k-means) compact descriptor
 - HOG descriptor (texture / gradient layout)
 - LBP histogram
 - Gabor energy vector
 - GLCM statistics (Haralick-style)

All components are normalized (probability or L2) before concatenation. Optional
dependencies (scikit-image, sklearn) are used when available with safe fallbacks.
"""
from __future__ import annotations
from typing import Tuple, Dict, Any, List
import numpy as np
from PIL import Image
import cv2
from skimage.feature import local_binary_pattern
try:  # handle version differences of scikit-image
    from skimage.feature import greycomatrix, greycoprops  # older API spelling
except Exception:
    try:
        from skimage.feature.texture import graycomatrix as greycomatrix, graycoprops as greycoprops  # newer location/spelling
    except Exception:
        greycomatrix = None  # type: ignore
        greycoprops = None  # type: ignore

try:
    from skimage.feature import hog as sk_hog
except Exception:
    sk_hog = None

try:
    from sklearn.cluster import KMeans  # type: ignore
except Exception:
    KMeans = None  # type: ignore

import torch


def lab_hist(img: Image.Image, bins=(32, 16, 16)) -> np.ndarray:
    arr = np.array(img.convert('RGB'))
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = lab[..., 0].ravel(), lab[..., 1].ravel(), lab[..., 2].ravel()
    hist, _ = np.histogramdd((l, a, b), bins=bins, range=((0, 255), (0, 255), (0, 255)))
    hist = hist.astype(np.float32).ravel()
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def lbp_hist(img: Image.Image, P: int = 8, R: int = 1) -> np.ndarray:
    gray = np.array(img.convert('L'))
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    bins = P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins))
    hist = hist.astype(np.float32)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def gabor_energy(img: Image.Image, scales=(3, 5), thetas=(0, np.pi/4, np.pi/2, 3*np.pi/4)) -> np.ndarray:
    gray = np.array(img.convert('L'), dtype=np.float32) / 255.0
    feats: List[float] = []
    for k in scales:
        for theta in thetas:
            kernel = cv2.getGaborKernel((9, 9), 3.0, theta, k, gamma=0.5, psi=0)
            f = cv2.filter2D(gray, cv2.CV_32F, kernel)
            feats.append(float(np.mean(np.abs(f))))
            feats.append(float(np.var(f)))
    v = np.array(feats, dtype=np.float32)
    n = np.linalg.norm(v)
    if n > 0:
        v /= n
    return v


def color_moments(img: Image.Image) -> np.ndarray:
    """Return first three moments (mean, std, skew) per RGB channel -> 9 dims."""
    arr = np.array(img.convert('RGB')).astype(np.float32)
    feats = []
    for c in range(3):
        vals = arr[..., c].ravel()
        mean = float(vals.mean())
        std = float(vals.std())
        # skewness: robust via scipy.stats if available else simple moment
        m3 = float(((vals - mean) ** 3).mean())
        skew = m3 / (std ** 3 + 1e-12)
        feats.extend([mean, std, skew])
    v = np.array(feats, dtype=np.float32)
    # small-scale normalization to keep values in range
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v


def dominant_colors(img: Image.Image, k: int = 3, sample: int = 1000) -> np.ndarray:
    """Return a compact descriptor of dominant colors (k * 3 dims).

    Uses KMeans if sklearn is available; otherwise uses coarse histogram quantization.
    """
    arr = np.array(img.convert('RGB'))
    h, w, _ = arr.shape
    pixels = arr.reshape(-1, 3).astype(np.float32)
    if KMeans is not None and pixels.shape[0] >= k:
        # sample for speed
        if pixels.shape[0] > sample:
            idx = np.random.choice(pixels.shape[0], sample, replace=False)
            sample_pixels = pixels[idx]
        else:
            sample_pixels = pixels
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=4)
            km.fit(sample_pixels)
            centers = km.cluster_centers_.astype(np.float32)
            # flatten centers
            v = centers.ravel()
            # normalize
            nrm = np.linalg.norm(v)
            if nrm > 0:
                v = v / nrm
            return v
        except Exception:
            pass
    # fallback: simple coarse quantization by averaging blocks
    pixels = pixels.reshape(h, w, 3)
    block_h = max(1, h // k)
    block_w = max(1, w // k)
    centers = []
    for i in range(k):
        y0 = min(i * block_h, h - block_h)
        block = pixels[y0:y0 + block_h, 0:block_w]
        if block.size == 0:
            centers.append([0, 0, 0])
        else:
            centers.append(list(block.reshape(-1, 3).mean(axis=0)))
    v = np.array(centers, dtype=np.float32).ravel()
    nrm = np.linalg.norm(v)
    if nrm > 0:
        v = v / nrm
    return v


def hog_descriptor(img: Image.Image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), orientations=9) -> np.ndarray:
    """HOG descriptor (fallback if scikit-image missing uses gradient histogram)."""
    # Resize to fixed spatial size to ensure deterministic HOG length
    target_size = (128, 128)
    gray_pil = img.convert('L').resize(target_size)
    gray = np.array(gray_pil)
    if sk_hog is not None:
        try:
            h = sk_hog(gray, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, feature_vector=True)
            v = np.array(h, dtype=np.float32)
            n = np.linalg.norm(v)
            if n > 0:
                v /= n
            return v
        except Exception:
            pass
    # fallback: gradient magnitude histogram
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy).ravel()
    hist, _ = np.histogram(mag, bins=32, range=(0.0, mag.max() if mag.max() > 0 else 1.0))
    hist = hist.astype(np.float32)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def glcm_stats(img: Image.Image, distances=(1, 2, 4), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4)) -> np.ndarray:
    if greycomatrix is None or greycoprops is None:
        return np.zeros(16, dtype=np.float32)  # fallback zero vector (4 props * len(distances)*len(angles) simplified)
    gray = np.array(img.convert('L'))
    gray_q = (gray / 8).astype(np.uint8)  # quantize to 32 levels
    glcm = greycomatrix(gray_q, distances=distances, angles=angles, levels=32, symmetric=True, normed=True)
    props = ['contrast', 'homogeneity', 'energy', 'correlation']
    feats = []
    for p in props:
        vals = greycoprops(glcm, p).ravel()
        feats.extend(vals.tolist())
    v = np.array(feats, dtype=np.float32)
    n = np.linalg.norm(v)
    if n > 0:
        v /= n
    return v


def extract_all(backbone, img: Image.Image) -> Dict[str, np.ndarray]:
    feats: Dict[str, np.ndarray] = {}
    # Global DINO embedding (L2-normalized by Backbone.encode)
    feats['global'] = backbone.encode(img)

    # Try to obtain richer transformer-level features (patch tokens, cls token, pooled)
    try:
        t = backbone.preprocess(img).unsqueeze(0).to(next(iter(backbone._forward.__closure__)).cell_contents.device) if hasattr(backbone._forward, '__closure__') and backbone._forward.__closure__ else backbone.preprocess(img).unsqueeze(0)
    except Exception:
        t = backbone.preprocess(img).unsqueeze(0)
    t = t.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    try:
        with torch.no_grad():
            out = backbone._forward(t)
        # Normalize targets: out may be tensor or dict
        if isinstance(out, dict):
            # prefer patch token keys used by various DINO implementations
            for k in ('x_norm_patchtokens', 'x_norm_clstoken', 'pooled', 'last_token', 'cls_token', 'last_hidden_state'):
                if k in out:
                    o = out[k]
                    break
            else:
                o = list(out.values())[0]
        else:
            o = out

        # Interpret o
        if isinstance(o, torch.Tensor):
            if o.dim() == 3:
                # (B, N, D) ; mean over tokens -> patch-level summary
                patch_mean = o.mean(1).squeeze(0)
                # treat first token as cls if present
                cls_token = o[:, 0, :].squeeze(0) if o.size(1) > 0 else patch_mean
            elif o.dim() == 2:
                # (B, D) pooled vector
                patch_mean = o.squeeze(0)
                cls_token = o.squeeze(0)
            else:
                patch_mean = o.view(o.size(0), -1).mean(1).squeeze(0)
                cls_token = patch_mean
        else:
            # fallback to zeros if unexpected
            patch_mean = None
            cls_token = None

        def to_np(v):
            if v is None:
                return None
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            v = v.astype(np.float32)
            n = np.linalg.norm(v)
            if n > 0:
                v /= n
            return v

        feats['patch_mean'] = to_np(patch_mean) if to_np(patch_mean) is not None else np.zeros_like(feats['global'])
        feats['cls'] = to_np(cls_token) if to_np(cls_token) is not None else np.zeros_like(feats['global'])
    except Exception:
        # conservative fallback: ensure same dims
        feats['patch_mean'] = np.zeros_like(feats['global'])
        feats['cls'] = np.zeros_like(feats['global'])

    # Complementary classical descriptors
    feats['lab_hist'] = lab_hist(img)
    feats['color_moments'] = color_moments(img)
    feats['dominant_colors'] = dominant_colors(img, k=3)
    feats['hog'] = hog_descriptor(img)
    feats['lbp'] = lbp_hist(img)
    feats['gabor'] = gabor_energy(img)
    feats['glcm'] = glcm_stats(img)
    return feats


def concat_and_normalize(feat_dict: Dict[str, np.ndarray], order=('global', 'cls', 'patch_mean', 'lab_hist', 'color_moments', 'dominant_colors', 'hog', 'lbp', 'gabor', 'glcm')) -> Tuple[np.ndarray, Dict[str, slice]]:
    parts = []
    spans: Dict[str, slice] = {}
    start = 0
    for k in order:
        v = feat_dict.get(k)
        if v is None:
            continue
        v = v.astype(np.float32)
        # Normalize per-component: if histogram-like (sums to 1) leave as is, else L2 normalize
        if v.ndim == 1:
            s = v.sum()
            if s > 0 and np.all(v >= 0) and s <= 1.0001:
                # probability vector already normalized or small-range histogram
                v = v.astype(np.float32)
            else:
                n = np.linalg.norm(v)
                if n > 0:
                    v = v / n
        parts.append(v)
        end = start + v.shape[0]
        spans[k] = slice(start, end)
        start = end
    full = np.concatenate(parts).astype(np.float32)
    n = np.linalg.norm(full)
    if n > 0:
        full /= n
    return full, spans
