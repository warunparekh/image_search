import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import shutil
import uuid
import time

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Form
from fastapi.responses import JSONResponse

from contextlib import asynccontextmanager
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import json
import cv2

try:
    import faiss  # type: ignore
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

# optional advanced feature pipeline (v2)
try:
    from backbones import get_backbone  # type: ignore
    from features import extract_all, concat_and_normalize  # type: ignore
except Exception:
    try:
        from .backbones import get_backbone  # type: ignore
        from .features import extract_all, concat_and_normalize  # type: ignore
    except Exception:
        get_backbone = None  # type: ignore
        extract_all = None  # type: ignore
        concat_and_normalize = None  # type: ignore

if os.environ.get('ALLOW_OMP_DUPLICATE','1') == '1':
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')



DEVICE = torch.device('xpu' if torch.xpu.is_available() else 'cpu')

try:
    # Reduce thread usage to lessen contention
    torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS','4')))
    torch.set_num_interop_threads(1)
except Exception:
    pass

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

INDEX_V2_JSON = Path('data/index/index_v2.json')
INDEX_V2_FAISS = Path('data/index/index_v2.faiss')
EMBED_V2 = None
FAISS_INDEX = None
BACKBONE_NAME_V2 = None
BACKBONE_V2 = None
EMBED_DIM_V2 = None
SPANS_V2 = {}
IMAGE_PATHS: List[str] = []
IMAGE_IDS: List[str] = []
IMAGES_DIR = None
ERROR_MSG = None


USE_AQE = os.environ.get('USE_AQE','1') == '1'
MULTISCALE_QUERY = os.environ.get('MULTISCALE_QUERY','1') == '1'
GEOM_RERANK_K = int(os.environ.get('GEOM_RERANK_K','24'))
AQE_TOP_K = int(os.environ.get('AQE_TOP_K','5'))
AQE_ALPHA = float(os.environ.get('AQE_ALPHA','1.0'))
RETRIEVAL_MODE = 'v2'
GEOM_WEIGHT = float(os.environ.get('GEOM_WEIGHT','0.5'))

DESCR_CACHE: Dict[str, Tuple[List[cv2.KeyPoint], np.ndarray, bool]] = {}





import gc, time, threading 

EMBED_FILE_LOCK = threading.Lock()

def _close_memmap(arr):
    """Safely close a numpy.memmap so Windows will allow file replacement."""
    try:
        if isinstance(arr, np.memmap):
            try:
                arr.flush()
            except Exception:
                pass
            mm = getattr(arr, '_mmap', None)
            if mm is not None:
                try:
                    mm.close()
                except Exception:
                    pass
    except Exception:
        pass


def update_product_ids_json(product_ids_path, new_item):
    """Add a new item to the product_ids.json file properly."""
    try:
        with open(product_ids_path, 'r') as f:
            data = json.load(f)
        
        data.append(new_item)
        
        with open(product_ids_path, 'w') as f:
            json.dump(data, f, indent=4)
            
        return True
        
    except Exception as e:
        print(f"Error updating product_ids.json: {e}")
        return False



def hex_to_int(hex_str):
    """Convert hexadecimal string to integer."""
    return int(hex_str, 16)

def int_to_hex(integer, length=24):
    """Convert integer to hexadecimal string with specified length."""
    return format(integer, f'0{length}x')


def generate_next_object_id(product_ids_path):
    """Generate the next ObjectId based on the last one in product_ids.json."""
    try:
        with open(product_ids_path, 'r') as f:
            data = json.load(f)  # Parse the entire JSON array
            
            if not data or len(data) == 0:
                # If file is empty or no items, start with a base ObjectId
                return "66224b6e139bf9e8e5792d17"
            
            # Get the last item in the array
            last_item = data[-1]
            last_id = last_item["_id"]
            
            # Convert to integer, increment, and convert back to hex
            last_int = hex_to_int(last_id)
            next_int = last_int + 1
            next_id = int_to_hex(next_int)
            
            return next_id
            
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error reading product_ids.json: {e}")
        # Return a default starting ObjectId if there's an error
        return "66224b6e139bf9e8e5792d17"





def _get_descriptors(path: Path) -> Tuple[List[cv2.KeyPoint], np.ndarray, bool]:
    """Return (keypoints, descriptors, is_sift) using SIFT if available, else ORB. Cached."""
    spath = str(path.resolve())
    cached = DESCR_CACHE.get(spath)
    if cached is not None:
        return cached
    img = cv2.imdecode(np.fromfile(spath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        DESCR_CACHE[spath] = ([], np.empty((0,)), False)
        return DESCR_CACHE[spath]
    # Try SIFT
    kp: List[cv2.KeyPoint] = []
    des = None
    is_sift = False
    try:
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
        is_sift = True
    except Exception:
        try:
            orb = cv2.ORB_create(nfeatures=1000)
            kp, des = orb.detectAndCompute(img, None)
        except Exception:
            kp, des = [], None
    if des is None:
        des = np.empty((0,), dtype=np.float32)
    DESCR_CACHE[spath] = (kp, des, is_sift)
    return DESCR_CACHE[spath]

def apply_aqe(query_vec: np.ndarray, candidate_idxs: np.ndarray, embeddings: np.ndarray, k: int = 5, alpha: float = 1.0) -> np.ndarray:
    """Average Query Expansion: blend query with top-k candidate embeddings."""
    if embeddings is None or len(candidate_idxs) == 0:
        return query_vec
    k = min(k, len(candidate_idxs))
    if k <= 0:
        return query_vec
    top = embeddings[candidate_idxs[:k]]
    top = top / (np.linalg.norm(top, axis=1, keepdims=True) + 1e-12)
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    new_q = (alpha * q + top.sum(axis=0)) / (alpha + k)
    new_q = new_q / (np.linalg.norm(new_q) + 1e-12)
    return new_q.astype(np.float32)

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_lab(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """RGB (0-255) -> CIE L*a*b* with L∈[0,100], a,b≈[-128,127]."""
    rgb_array = np.array([[list(rgb)]], dtype=np.float32) / 255.0
    lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
    L, a, b = lab[0, 0]
    return (float(L), float(a), float(b))

def color_distance_lab(color1: Tuple[float, float, float], color2: Tuple[float, float, float]) -> float:
    """Calculate color distance in LAB space (Delta E)."""
    l1, a1, b1 = color1
    l2, a2, b2 = color2
    return np.sqrt((l1 - l2)**2 + (a1 - a2)**2 + (b1 - b2)**2)

def extract_mean_lab(img_path: Path) -> Tuple[float, float, float]:
    """Mean Lab over the image in true Lab ranges."""
    img = Image.open(img_path).convert('RGB').resize((100, 100))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    L = float(lab[..., 0].mean())
    a = float(lab[..., 1].mean())
    b = float(lab[..., 2].mean())
    return (L, a, b)

def neutral_aware_color_similarity(cand_lab: Tuple[float, float, float], target_lab: Tuple[float, float, float]) -> float:
    """Calculate color similarity with neutral-aware weighting."""
    Ct = np.hypot(target_lab[1], target_lab[2])
    Cc = np.hypot(cand_lab[1], cand_lab[2])

    dL = cand_lab[0] - target_lab[0]
    da = cand_lab[1] - target_lab[1]
    db = cand_lab[2] - target_lab[2]

    if Ct < 7.5:
        wL, wA, wB = 0.4, 1.7, 1.7
        sigma = 7.5
    else:
        wL, wA, wB = 1.0, 1.0, 1.0
        sigma = 12.0

    de2 = (wL*dL)**2 + (wA*da)**2 + (wB*db)**2
    return float(np.exp(-de2 / (2.0 * sigma * sigma)))

def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """Convert scores to ranks for rank-fusion."""
    order = np.argsort(-scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    return ranks.astype(np.float32)


def calculate_adaptive_color_weight(query_mean_lab: Tuple[float, float, float], target_lab: Tuple[float, float, float]) -> float:
    """Calculate adaptive color weight based on how different target color is from query's original color."""
    distance = color_distance_lab(query_mean_lab, target_lab)
    weight = 0.35 + (min(distance, 20.0) / 20.0) * (0.8 - 0.35)
    return weight

def _geom_score_cached(query_path: Path, cand_path: Path) -> float:
    """Geometric verification using cached local descriptors + FLANN/BF + RANSAC."""
    try:
        kp1, des1, sift1 = _get_descriptors(query_path)
        kp2, des2, sift2 = _get_descriptors(cand_path)
        if des1.size == 0 or des2.size == 0 or len(kp1) < 4 or len(kp2) < 4:
            return 0.0
        use_sift = sift1 and sift2 and des1.ndim == 2 and des2.ndim == 2 and des1.shape[1] == des2.shape[1] and des1.shape[1] in (64, 128)
        if use_sift:
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=40)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            matches = matcher.knnMatch(des1, des2, k=2)
            ratio = 0.7
            good = []
            for m_n in matches:
                if len(m_n) < 2:
                    continue
                m, n = m_n
                if m.distance < ratio * n.distance:
                    good.append(m)
        else:
            if des1.dtype != np.uint8:
                des1_u8 = des1.astype(np.uint8)
            else:
                des1_u8 = des1
            if des2.dtype != np.uint8:
                des2_u8 = des2.astype(np.uint8)
            else:
                des2_u8 = des2
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(des1_u8, des2_u8, k=2)
            ratio = 0.75
            good = []
            for m_n in matches:
                if len(m_n) < 2:
                    continue
                m, n = m_n
                if m.distance < ratio * n.distance:
                    good.append(m)
        if len(good) < 4:
            return float(len(good)) / max(1.0, min(200.0, len(kp1)))
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is None:
            return 0.0
        inliers = float(mask.sum())
        return float(min(1.0, inliers / max(1.0, len(good))))
    except Exception:
        return 0.0

def search(query_embed: np.ndarray, top_k: int = 8):
    if EMBED_V2 is None:
        raise RuntimeError('V2 embedding index not loaded')
    if query_embed.shape[0] != EMBED_V2.shape[1]:
        raise ValueError(f'Query dim {query_embed.shape[0]} != index dim {EMBED_V2.shape[1]}')
    if FAISS_INDEX is None:
        raise RuntimeError('FAISS index not available')

    D, I = FAISS_INDEX.search(query_embed.reshape(1, -1).astype(np.float32), top_k * 5)
    return I[0], D[0]

def compute_fusion_similarities(query_embed, candidate_pool: np.ndarray, comp_weights=None):
    if comp_weights is None:
        comp_weights = {}
    if candidate_pool is None or len(candidate_pool) == 0:
        raise RuntimeError("candidate_pool is required to avoid full-matrix ops")

    sims_pool = np.zeros(len(candidate_pool), dtype=np.float32)
    total_w = 0.0
    for comp, rng in SPANS_V2.items():
        w = float(comp_weights.get(comp, 0.0))
        if w == 0.0:
            continue
        si, sj = int(rng[0]), int(rng[1])
        q_sub = query_embed[si:sj]

        bs = 2048
        for s in range(0, len(candidate_pool), bs):
            e = min(s+bs, len(candidate_pool))
            rows = EMBED_V2[candidate_pool[s:e], si:sj]
            if rows.dtype != np.float32:
                rows = rows.astype(np.float32, copy=False)
            sims_pool[s:e] += w * (rows @ q_sub)
        total_w += w

    if total_w > 0:
        sims_pool /= total_w
    order_pool = np.argsort(-sims_pool)
    return candidate_pool[order_pool], sims_pool[order_pool]


def load_index():
    """Load v2 index (DINOv2 pipeline)."""
    global IMAGE_PATHS, IMAGES_DIR, EMBED_V2, FAISS_INDEX, BACKBONE_NAME_V2, BACKBONE_V2, EMBED_DIM_V2, RETRIEVAL_MODE, GEOM_WEIGHT, SPANS_V2, ERROR_MSG, IMAGE_IDS
    if INDEX_V2_JSON.exists():
        with open(INDEX_V2_JSON,'r') as f:
            data = json.load(f)
        try:
            SPANS_V2 = data.get('spans', {})
            IMAGE_PATHS = data['paths']
            IMAGES_DIR = Path(data['images_dir'])
            IMAGE_IDS = data.get('ids', [])
            BACKBONE_NAME_V2 = data.get('backbone', 'dinov2')

            # Support multiple schema variants:
            # 1. embedding_file (npy path relative or absolute)
            # 2. embedding_path (same as UI file)
            # 3. inline 'embedding' list (legacy, heavy JSON)
            emb_file = data.get('embedding_file') or data.get('embedding_path')
            if emb_file:
                emb_path = Path(emb_file)
                if not emb_path.is_absolute():
                    emb_path = INDEX_V2_JSON.parent / emb_path
                if not emb_path.exists():
                    raise FileNotFoundError(f"Embedding file not found: {emb_path}")
                EMBED_V2 = np.load(str(emb_path), mmap_mode='r')

            else:
                raise RuntimeError("Index JSON missing embeddings (embedding_file / embedding_path / embedding)")

            EMBED_DIM_V2 = EMBED_V2.shape[1]
        except Exception as e:
            ERROR_MSG = f"Index load error: {e}"
            print('[ERROR]', ERROR_MSG)
            raise

        if get_backbone is not None:
            try:
                BACKBONE_V2 = get_backbone('dinov2', strict=True, verbose=True)
            except Exception as e:
                ERROR_MSG = f"Backbone load failed: {e}"
                print('[ERROR]', ERROR_MSG)
        if HAVE_FAISS and INDEX_V2_FAISS.exists():
            try:
                FAISS_INDEX = faiss.read_index(str(INDEX_V2_FAISS)) 
            except Exception:
                FAISS_INDEX = None
        RETRIEVAL_MODE = 'v2'
        print(f"[INFO] Loaded v2 index: dim={EMBED_DIM_V2} backbone={BACKBONE_NAME_V2} size={len(IMAGE_PATHS)} faiss={'yes' if FAISS_INDEX else 'no'}")
        return
    else:
        raise FileNotFoundError(f"V2 index not found at {INDEX_V2_JSON}. Build index_v2 first using DINOv2 backbone.")

def load_model():
    """Compatibility hook for loading additional models."""
    return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load index and model on startup"""
    try:
        load_index()
        load_model()
        print("[INFO] FastAPI startup complete")
    except Exception as e:
        print(f"[ERROR] Startup failed: {e}")
    yield






    

    

# Add this new endpoint after the existing routes


app = FastAPI(
    title="Tile Image Search", 
    description="Image similarity search using DINOv2",
    lifespan=lifespan
)
from numpy.lib.format import open_memmap

def append_row_to_embeddings_npy(vec: np.ndarray, emb_path: Path, retries: int = 30, delay: float = 0.05):
    vec = np.asarray(vec, dtype=np.float32)
    old = np.load(emb_path, mmap_mode='r')
    n, d = old.shape
    assert vec.shape == (d,), f"dim mismatch: {vec.shape} vs {(d,)}"
    tmp = emb_path.with_suffix('.tmp.npy')
    new = open_memmap(tmp, mode='w+', dtype=np.float32, shape=(n+1, d))

    bs = max(1, 1_000_000 // d)
    s = 0
    while s < n:
        e = min(s+bs, n)
        new[s:e] = old[s:e]
        s = e
    new[n] = vec




    _close_memmap(new)
    _close_memmap(old)
    del new, old
    gc.collect()

    for _ in range(retries):
        try:
            os.replace(tmp, emb_path)
            return
        except PermissionError:
            time.sleep(delay)
        except OSError as e:
            if getattr(e, 'winerror', None) == 5:
                time.sleep(delay)
            else:
                raise
    os.replace(tmp, emb_path)

def add_image_to_index(image_path: Path, object_id: str = None) -> bool:
    """Add a single image to the existing index without rebuilding."""
    global EMBED_V2, FAISS_INDEX, IMAGE_PATHS, IMAGE_IDS
    
    try:
        if EMBED_V2 is None or BACKBONE_V2 is None:
            raise RuntimeError('Index not loaded')
        
        # Extract features for the new image
        img = Image.open(image_path).convert('RGB')
        feat_dict = extract_all(BACKBONE_V2, img)
        full_embed, local_spans = concat_and_normalize(feat_dict)

        # Align embedding dimension to index if mismatch (mirror search mapping logic)
        if full_embed.shape[0] == EMBED_V2.shape[1]:
            new_embed = full_embed
        else:
            mapped = np.zeros(EMBED_V2.shape[1], dtype=np.float32)
            for comp, rng in SPANS_V2.items():
                try:
                    si, sj = int(rng[0]), int(rng[1])
                    if comp not in local_spans:
                        continue
                    ls = local_spans[comp]
                    li, lj = int(ls.start), int(ls.stop)
                    sub = full_embed[li:lj]
                    target_len = sj - si
                    if sub.shape[0] == target_len:
                        mapped[si:sj] = sub
                    else:
                        L = min(target_len, sub.shape[0])
                        mapped[si:si+L] = sub[:L]
                except Exception:
                    continue
            nrm = np.linalg.norm(mapped)
            if nrm > 0:
                mapped = (mapped / nrm).astype(np.float32)
            new_embed = mapped
            if new_embed.shape[0] != EMBED_V2.shape[1]:
                raise ValueError(f'Aligned new image dim {new_embed.shape[0]} vs index dim {EMBED_V2.shape[1]} (mapping failed)')
        
        # Generate ObjectId if not provided
        if object_id is None:
            object_id = f"img_{len(IMAGE_PATHS) + 1}_{int(time.time())}"
        
        # Add to in-memory structures
        with open(INDEX_V2_JSON, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        emb_key = meta.get('embedding_file') or meta.get('embedding_path')
        if not emb_key:
            raise RuntimeError("index JSON missing embedding_path/embedding_file")


        emb_path = Path(emb_key) if Path(emb_key).is_absolute() else INDEX_V2_JSON.parent / emb_key


        with EMBED_FILE_LOCK:
            _close_memmap(EMBED_V2)
            EMBED_V2 = None
            gc.collect()

            append_row_to_embeddings_npy(new_embed, emb_path)

            # Reload global memmap
            EMBED_V2 = np.load(str(emb_path), mmap_mode='r')



        IMAGE_PATHS.append(str(image_path.relative_to(IMAGES_DIR)))
        IMAGE_IDS.append(object_id)
        
        # Add to FAISS index if available
        if FAISS_INDEX is not None:
            FAISS_INDEX.add(new_embed.reshape(1, -1).astype(np.float32))
        



        try:
            with open(INDEX_V2_JSON, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed reading index JSON: {e}")


        n, d = EMBED_V2.shape
        index_data['ids'] = IMAGE_IDS
        index_data['paths'] = IMAGE_PATHS
        index_data['embedding_shape'] = [int(n), int(d)]

        with open(INDEX_V2_JSON, 'w', encoding='utf-8') as f:
           json.dump(index_data, f, ensure_ascii=False, indent=2)


        # Update FAISS index file if available
        if FAISS_INDEX is not None and INDEX_V2_FAISS.exists():
            faiss.write_index(FAISS_INDEX, str(INDEX_V2_FAISS))
        
        print(f"[INFO] Added image to index: {image_path.name} with ObjectId: {object_id}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to add image to index: {e}")
        return False


def process_search(save_path: Path, color_filter: str = None, top_k: int = 8):
    """Process the image search - extracted from Flask logic"""
    if EMBED_V2 is None or BACKBONE_V2 is None:
        raise RuntimeError('V2 index or backbone not available. Rebuild index with DINOv2.')

    # Extract all local + global features
    img = Image.open(save_path).convert('RGB')
    feat_dict = extract_all(BACKBONE_V2, img)
    full_q, local_spans = concat_and_normalize(feat_dict)

    # Align query embedding dimension with index (mirror UI logic)
    if full_q.shape[0] == EMBED_V2.shape[1]:
        query_embed = full_q
    else:
        # Map overlapping components into zero-init vector of index dimension
        query_embed = np.zeros(EMBED_V2.shape[1], dtype=np.float32)
        for comp, rng in SPANS_V2.items():
            try:
                si, sj = int(rng[0]), int(rng[1])  # stored span in index
                if comp not in local_spans:
                    continue
                ls = local_spans[comp]
                li, lj = int(ls.start), int(ls.stop)
                sub = full_q[li:lj]
                target_len = sj - si
                if sub.shape[0] == target_len:
                    query_embed[si:sj] = sub
                else:
                    L = min(target_len, sub.shape[0])
                    query_embed[si:si+L] = sub[:L]
            except Exception:
                continue
        # L2 normalize
        nrm = np.linalg.norm(query_embed)
        if nrm > 0:
            query_embed = (query_embed / nrm).astype(np.float32)

    # Component weights (same as UI app)
    comp_weights = {
        'global': 1.0, 'cls': 0.7, 'patch_mean': 0.5, 'lab_hist': 0.4,
        'color_moments': 0.25, 'dominant_colors': 0.25, 'mean_lab': 0.6,
        'hog': 0.35, 'lbp': 0.15, 'gabor': 0.12, 'glcm': 0.12,
    }

    use_fusion = bool(SPANS_V2) and bool(comp_weights)
    
    # Skip color processing if color_filter is None
    if color_filter is not None:
        query_mean_lab = extract_mean_lab(save_path)
        target_rgb = hex_to_rgb(color_filter)
        target_lab = rgb_to_lab(target_rgb)
        user_selected_color = (color_filter != '#ffffff')
        adaptive_color_weight = calculate_adaptive_color_weight(query_mean_lab, target_lab)
        adaptive_color_weight = max(adaptive_color_weight, 0.75)
    else:
        # No color influence - pure pattern/shape matching
        user_selected_color = False
        adaptive_color_weight = 0.0  # No color weight
        target_lab = None

    pool_k = max(64, GEOM_RERANK_K * 4)
    rerank_k_local = GEOM_RERANK_K
    
    if user_selected_color:
        pool_k = max(pool_k, 512)
        rerank_k_local = max(rerank_k_local, 128)
        
    if FAISS_INDEX is not None:
        _, idxs = FAISS_INDEX.search(query_embed.reshape(1, -1).astype(np.float32), pool_k)
        candidate_pool = idxs[0]
    else:
        candidate_pool = None

    if use_fusion:
        cand_idx_all, cand_scores_all = compute_fusion_similarities(query_embed, candidate_pool=candidate_pool, comp_weights=comp_weights)
        cand_idx = cand_idx_all[: max(8, rerank_k_local)]
        cand_scores = cand_scores_all[: max(8, rerank_k_local)]
    else:
        cand_idx, cand_scores = search(query_embed, top_k=max(8, rerank_k_local))

    if USE_AQE:
        query_embed = apply_aqe(query_embed, cand_idx, EMBED_V2, k=AQE_TOP_K, alpha=AQE_ALPHA)
        if use_fusion:
            if FAISS_INDEX is not None and candidate_pool is not None:
                _, idxs = FAISS_INDEX.search(query_embed.reshape(1, -1).astype(np.float32), pool_k)
                candidate_pool = idxs[0]
            cand_idx_all, cand_scores_all = compute_fusion_similarities(query_embed, candidate_pool=candidate_pool, comp_weights=comp_weights)
            cand_idx = cand_idx_all[: max(8, rerank_k_local)]
            cand_scores = cand_scores_all[: max(8, rerank_k_local)]
        else:
            I2, D2 = search(query_embed, top_k=max(8, rerank_k_local))
            cand_idx, cand_scores = I2, D2
    
    rerank_subset = cand_idx[:rerank_k_local]
    base_subset = cand_scores[:len(rerank_subset)].astype(np.float32)
    
    # Skip color reranking if no color filter
    if color_filter is not None and target_lab is not None:
        color_subset = np.empty(len(rerank_subset), dtype=np.float32)
        for i, idx_local in enumerate(rerank_subset):
            cand_path = IMAGES_DIR / IMAGE_PATHS[idx_local]
            cand_lab = extract_mean_lab(cand_path)
            color_subset[i] = neutral_aware_color_similarity(cand_lab, target_lab)
        
        base_rank = ranks_from_scores(base_subset)
        color_rank = ranks_from_scores(color_subset)
        
        pattern_weight = 1.0 - adaptive_color_weight
        fused_rank = pattern_weight * base_rank + adaptive_color_weight * color_rank
        
        color_order = np.argsort(fused_rank)
        rerank_subset = rerank_subset[color_order]
    
    # Geometric verification
    geom_scores = []
    for idx_local in rerank_subset:
        cand_path = IMAGES_DIR / IMAGE_PATHS[idx_local]
        geom_scores.append(_geom_score_cached(save_path, cand_path))
    geom_scores = np.array(geom_scores, dtype=np.float32)
    
    geom_rank = ranks_from_scores(geom_scores)
    GEOM_WEIGHT_LOCAL = min(GEOM_WEIGHT, 0.15)
    final_rank = (1.0 - GEOM_WEIGHT_LOCAL) * np.arange(len(rerank_subset)) + GEOM_WEIGHT_LOCAL * geom_rank
    
    order = np.argsort(final_rank)[:top_k]
    
    results = []
    for rank, pos in enumerate(order, 1):
        idx = rerank_subset[pos]
        results.append(IMAGE_IDS[idx] if idx < len(IMAGE_IDS) else None)

    
    return results


@app.post("/api/search")
async def search_by_uploaded_image(
    file: UploadFile = File(...),
    color_filter: str = Query(None, description="Optional hex color filter"),
    top_k: int = Query(8, description="Number of results to return", ge=1, le=50)
):
    """
    Search for similar images by uploading an image file.
    
    Args:
        file: Image file to search with
        color_filter: Optional hex color for filtering (e.g., "#ff0000", "#ffffff") 
        top_k: Number of results to return (1-50)
    
    Returns:
        Array of product IDs ordered by similarity
    """
    try:
        if EMBED_V2 is None or BACKBONE_V2 is None:
            raise HTTPException(status_code=503, detail="Search index not loaded")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Create temporary file to save uploaded image
        temp_dir = Path("data/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        temp_filename = f"search_{uuid.uuid4()}.jpg"
        temp_path = temp_dir / temp_filename
        
        try:
            # Save uploaded file
            with temp_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Process search - this returns array of product IDs
            product_ids = process_search(temp_path, color_filter, top_k)
            
            # Return just the array of product IDs
            return product_ids
            
        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/api/add-to-index")
async def add_image_to_index_endpoint(
    file: UploadFile = File(...),
    product_id: str = Form(...),
    category_id: str = Form(...),
    image_name: str = Form(...)
):
    """Add uploaded image to the index with explicitly provided identifiers.

    All fields are REQUIRED. No auto-generation or fallback logic is applied.
    Rejects duplicate product_id.
    image_name is used exactly as provided (no automatic extension enforcement).
    """
    try:
        if IMAGES_DIR is None:
            raise HTTPException(status_code=500, detail="Images directory not initialized")

        # Basic validation (non-empty after strip)
        if not product_id.strip():
            raise HTTPException(status_code=400, detail="product_id is required")
        if not category_id.strip():
            raise HTTPException(status_code=400, detail="category_id is required")
        if not image_name.strip():
            raise HTTPException(status_code=400, detail="image_name is required")

        product_ids_path = Path("product_ids.json")
        if not product_ids_path.exists():
            product_ids_path.write_text('[]', encoding='utf-8')

        try:
            data = json.loads(product_ids_path.read_text(encoding='utf-8'))
            if not isinstance(data, list):
                raise ValueError("product_ids.json must contain a JSON array")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed reading product_ids.json: {e}")

        existing_ids = {item.get('_id') for item in data if isinstance(item, dict)}
        if product_id in existing_ids:
            raise HTTPException(status_code=400, detail="Duplicate product_id")

        final_filename = image_name  # use exactly what frontend passed
        save_path = IMAGES_DIR / final_filename
        with save_path.open('wb') as buffer:
            shutil.copyfileobj(file.file, buffer)

        if not add_image_to_index(save_path, product_id):
            save_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail="Failed to add image to index")

        new_entry = {
            "_id": product_id,
            "categoryId": category_id,
            "productId": product_id,
            "imageName": final_filename
        }
        if not update_product_ids_json(product_ids_path, new_entry):
            raise HTTPException(status_code=500, detail="Failed to update product_ids.json")

        return {
            "success": True,
            "message": "Image added to index successfully",
            "product_id": product_id,
            "category_id": category_id,
            "image_name": final_filename,
            "path": str(save_path.relative_to(IMAGES_DIR))
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _persist_index_changes():
    """
    Persist lightweight metadata (ids/paths/shape) and rebuild FAISS from the
    on-disk memmap **in chunks** to avoid huge RAM spikes. Does NOT rewrite
    the embeddings .npy file and NEVER inlines embeddings into JSON.
    """
    global EMBED_V2, IMAGE_IDS, IMAGE_PATHS, FAISS_INDEX

    # ---- 1) Load & update JSON metadata (no embedding payloads) ----
    try:
        with open(INDEX_V2_JSON, 'r') as f:
            index_data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed reading index JSON: {e}")

    index_data['ids'] = IMAGE_IDS
    index_data['paths'] = IMAGE_PATHS

    emb_file = index_data.get('embedding_file') or index_data.get('embedding_path')
    if not emb_file:
        raise RuntimeError(
            "Index JSON missing 'embedding_path'/'embedding_file'. "
            "Refuse to inline embeddings; set a file path instead."
        )

    # ensure path is stored relative to index JSON if possible (keeps repo portable)
    emb_path = Path(emb_file)
    if not emb_path.is_absolute():
        emb_path = INDEX_V2_JSON.parent / emb_path

    # reflect current shape/dtype from the memmap we already have
    n, d = int(EMBED_V2.shape[0]), int(EMBED_V2.shape[1])
    index_data['embedding_shape'] = [n, d]
    index_data['embedding_dtype'] = str(EMBED_V2.dtype)

    # normalize path storage (relative if under same dir)
    def _is_rel_to(p: Path, base: Path) -> bool:
        try:
            p.relative_to(base); return True
        except Exception:
            return False
    rel_or_abs = (
        str(emb_path.relative_to(INDEX_V2_JSON.parent))
        if _is_rel_to(emb_path, INDEX_V2_JSON.parent) else str(emb_path)
    )
    if 'embedding_file' in index_data:
        index_data['embedding_file'] = rel_or_abs
    elif 'embedding_path' in index_data:
        index_data['embedding_path'] = rel_or_abs

    with open(INDEX_V2_JSON, 'w') as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)

    # ---- 2) Rebuild FAISS from memmap in blocks (no giant cast/copy) ----
    if HAVE_FAISS:
        try:
            metric = (index_data.get('metric') or 'IP').upper()
            if metric == 'L2':
                FAISS_INDEX_NEW = faiss.IndexFlatL2(d)  # type: ignore
            else:
                FAISS_INDEX_NEW = faiss.IndexFlatIP(d)  # type: ignore

            bs = 8192  # rows per block; tune for your I/O
            # Add in chunks; upcast each chunk only (keeps RAM flat)
            for s in range(0, n, bs):
                e = min(s + bs, n)
                block = EMBED_V2[s:e]
                if block.dtype != np.float32:
                    block = block.astype(np.float32, copy=False)
                FAISS_INDEX_NEW.add(block)

            # persist and swap in
            faiss.write_index(FAISS_INDEX_NEW, str(INDEX_V2_FAISS))  # type: ignore
            FAISS_INDEX = FAISS_INDEX_NEW

        except Exception as e:
            print(f"[WARN] Failed to rebuild FAISS index: {e}")

def rewrite_without_rows(emb_path: Path, remove_idx_sorted_desc: List[int], retries: int = 30, delay: float = 0.05):
    old = np.load(emb_path, mmap_mode='r')
    n, d = old.shape
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[remove_idx_sorted_desc] = False
    new_n = int(keep_mask.sum())

    tmp = emb_path.with_suffix('.tmp.npy')
    new = open_memmap(tmp, mode='w+', dtype=old.dtype, shape=(new_n, d))

    bs = max(1, 1_000_000 // d)
    w = 0
    for s in range(0, n, bs):
        e = min(s+bs, n)
        block = old[s:e]
        km = keep_mask[s:e]
        kc = int(km.sum())
        if kc:
            new[w:w+kc] = block[km]
            w += kc

    _close_memmap(new)
    _close_memmap(old)
    del new, old
    gc.collect()

    for _ in range(retries):
        try:
            os.replace(tmp, emb_path)
            return
        except PermissionError:
            time.sleep(delay)
        except OSError as e:
            if getattr(e, 'winerror', None) == 5:
                time.sleep(delay)
            else:
                raise
    os.replace(tmp, emb_path)

@app.post("/api/remove-from-index")
async def remove_from_index(
    product_ids: Optional[str] = Form(None),  # may be JSON array string, comma/space separated, or single id
    product_id: Optional[str] = Form(None)
):
    """Remove one or multiple product_ids and their embeddings from the index.

    Accepts either:
      - multiple form fields named product_ids
      - or a single form field product_id (backward compatible)

    Steps per id:
      1. Find all occurrences of that id in IMAGE_IDS
      2. Remove rows from EMBED_V2 and entries in IMAGE_PATHS / IMAGE_IDS
    After processing all ids:
      - Update product_ids.json (dropping matching _id entries)
      - Persist updated embedding file + index JSON
      - Rebuild FAISS index
    """
    global EMBED_V2, IMAGE_IDS, IMAGE_PATHS, FAISS_INDEX
    try:
        if EMBED_V2 is None:
            raise HTTPException(status_code=503, detail="Index not loaded")

        # Consolidate input ids
        ids_input: List[str] = []
        if product_ids:
            txt = product_ids.strip()
            parsed = False
            # Try JSON array
            if txt.startswith('[') and txt.endswith(']'):
                try:
                    arr = json.loads(txt)
                    if isinstance(arr, list):
                        ids_input.extend([str(x) for x in arr])
                        parsed = True
                except Exception:
                    pass
            if not parsed:
                # Allow comma or whitespace separated list
                sep = ',' if ',' in txt else None
                parts = [p for p in (txt.split(sep) if sep else txt.split())]
                ids_input.extend(parts)
        if product_id:
            ids_input.append(product_id)
        ids_input = [pid.strip() for pid in ids_input if pid and pid.strip()]
        if not ids_input:
            raise HTTPException(status_code=400, detail="No product_ids provided")

        # Deduplicate while preserving order
        seen = set()
        target_ids: List[str] = []
        for pid in ids_input:
            if pid not in seen:
                seen.add(pid)
                target_ids.append(pid)

        # Map id -> indices list
        id_to_indices: Dict[str, List[int]] = {}
        for i, pid in enumerate(IMAGE_IDS):
            if pid in seen:
                id_to_indices.setdefault(pid, []).append(i)

        if not any(id_to_indices.values()):
            raise HTTPException(status_code=404, detail="None of the product_ids found")

        # Collect all indices to remove
        all_remove_indices: List[int] = []
        per_id_counts: Dict[str, int] = {}
        for pid in target_ids:
            idxs = id_to_indices.get(pid, [])
            per_id_counts[pid] = len(idxs)
            all_remove_indices.extend(idxs)

        # Remove from embeddings / metadata
        if all_remove_indices:
            all_remove_indices_sorted = sorted(all_remove_indices, reverse=True)

            emb_file = json.load(open(INDEX_V2_JSON)).get('embedding_file') \
                    or json.load(open(INDEX_V2_JSON)).get('embedding_path')
            emb_path = Path(emb_file) if Path(emb_file).is_absolute() else INDEX_V2_JSON.parent / emb_file


            with EMBED_FILE_LOCK:
                _close_memmap(EMBED_V2)
                EMBED_V2 = None
                gc.collect()

                rewrite_without_rows(emb_path, all_remove_indices_sorted)

                EMBED_V2 = np.load(str(emb_path), mmap_mode='r')

            for idx in all_remove_indices_sorted:
                del IMAGE_PATHS[idx]
                del IMAGE_IDS[idx]

        # Update product_ids.json
        pid_path = Path('product_ids.json')
        if pid_path.exists():
            try:
                existing = json.loads(pid_path.read_text(encoding='utf-8'))
                if isinstance(existing, list):
                    to_remove = set(target_ids)
                    filtered = [row for row in existing if row.get('_id') not in to_remove]
                    if len(filtered) != len(existing):
                        pid_path.write_text(json.dumps(filtered, indent=4), encoding='utf-8')
                else:
                    print('[WARN] product_ids.json not a list; skipping removal there')
            except Exception as e:
                print(f"[WARN] Failed updating product_ids.json: {e}")

        # Persist & rebuild FAISS
        _persist_index_changes()

        return {
            "success": True,
            "removed_total": sum(per_id_counts.values()),
            "details": per_id_counts,
            "not_found": [pid for pid in target_ids if per_id_counts.get(pid, 0) == 0]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=4000)
    
