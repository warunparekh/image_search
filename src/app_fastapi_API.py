import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import shutil
import uuid
import time

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Form, Request
from fastapi.responses import JSONResponse

from contextlib import asynccontextmanager
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import json
import cv2
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except Exception:
    psutil = None
    _PSUTIL_AVAILABLE = False

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
STARTUP_MEM: Dict[str, float] = {}


USE_AQE = os.environ.get('USE_AQE','1') == '1'
MULTISCALE_QUERY = os.environ.get('MULTISCALE_QUERY','1') == '1'
GEOM_RERANK_K = int(os.environ.get('GEOM_RERANK_K','24'))
AQE_TOP_K = int(os.environ.get('AQE_TOP_K','5'))
AQE_ALPHA = float(os.environ.get('AQE_ALPHA','1.0'))
RETRIEVAL_MODE = 'v2'
GEOM_WEIGHT = float(os.environ.get('GEOM_WEIGHT','0.5'))

DESCR_CACHE: Dict[str, Tuple[List[cv2.KeyPoint], np.ndarray, bool]] = {}


def _format_mem(bytes_val: int) -> str:
    try:
        return f"{bytes_val / 1024 / 1024:.1f} MB"
    except Exception:
        return f"{bytes_val} B"


def get_memory_info() -> Dict[str, float]:
    """Return memory usage info for the current process.

    Returns a dict with rss (bytes), vms (bytes) and percent (float).
    """
    pid = os.getpid()
    if _PSUTIL_AVAILABLE:
        p = psutil.Process(pid)
        mi = p.memory_info()
        return {"rss": float(mi.rss), "vms": float(getattr(mi, 'vms', 0)), "percent": float(p.memory_percent())}
    else:
        # Best-effort fallback using resource (Unix) or zeros on Windows without psutil
        try:
            import resource  # type: ignore
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # ru_maxrss is kilobytes on many Unixes; convert to bytes
            return {"rss": float(rss) * 1024.0, "vms": 0.0, "percent": 0.0}
        except Exception:
            return {"rss": 0.0, "vms": 0.0, "percent": 0.0}


def log_memory(label: str = "") -> None:
    mi = get_memory_info()
    print(f"[MEM] {label} RSS={_format_mem(int(mi.get('rss',0)))} VMS={_format_mem(int(mi.get('vms',0)))} Pct={mi.get('percent',0):.2f}%")








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
    """V2-only search against precomputed embedding matrix"""
    if EMBED_V2 is None:
        raise RuntimeError('V2 embedding index not loaded')
    if query_embed.shape[0] != EMBED_V2.shape[1]:
        raise ValueError(f'Query dim {query_embed.shape[0]} != index dim {EMBED_V2.shape[1]}')
    if FAISS_INDEX is not None:
        sims, idxs = FAISS_INDEX.search(query_embed.reshape(1, -1).astype(np.float32), top_k * 5)
        return idxs[0], sims[0]
    # If EMBED_V2 is a memmap or very large, avoid creating full sims in memory
    try:
        is_memmap = hasattr(EMBED_V2, 'filename') or (hasattr(EMBED_V2, 'mode') and getattr(EMBED_V2, 'mode') == 'r')
    except Exception:
        is_memmap = False

    # threshold in rows above which we prefer chunking
    rows_threshold = int(os.environ.get('EMBED_CHUNK_ROWS', '200000'))
    use_chunk = is_memmap or (hasattr(EMBED_V2, 'shape') and EMBED_V2.shape[0] > rows_threshold)
    if use_chunk:
        return _topk_by_chunks(EMBED_V2, query_embed, top_k * 5)
    else:
        sims = EMBED_V2 @ query_embed
        order = np.argsort(-sims)[: top_k * 5]
        return order, sims[order]


def _topk_by_chunks(emb_matrix, query_vec: np.ndarray, k: int, chunk_size: int = 8192):
    """Compute top-k indices and scores by processing the embedding matrix in chunks.

    emb_matrix may be a memmap or an ndarray. Returns (indices, scores) for top-k.
    """
    import heapq
    n_rows = int(emb_matrix.shape[0])
    best = []  # min-heap of (score, idx)
    q = query_vec.astype(np.float32)
    for start in range(0, n_rows, chunk_size):
        stop = min(n_rows, start + chunk_size)
        try:
            block = np.asarray(emb_matrix[start:stop], dtype=np.float32)
        except Exception:
            block = emb_matrix[start:stop].astype(np.float32)
        # compute block scores
        block_scores = block @ q
        for i, s in enumerate(block_scores):
            if len(best) < k:
                heapq.heappush(best, (float(s), start + i))
            else:
                if s > best[0][0]:
                    heapq.heapreplace(best, (float(s), start + i))
    if not best:
        return np.array([], dtype=int), np.array([], dtype=np.float32)
    best_sorted = sorted(best, key=lambda x: -x[0])
    scores = np.array([b[0] for b in best_sorted], dtype=np.float32)
    idxs = np.array([b[1] for b in best_sorted], dtype=int)
    return idxs, scores

def compute_fusion_similarities(query_embed: np.ndarray, candidate_pool: np.ndarray = None, comp_weights: Dict[str, float] = None):
    """Compute fusion similarities for either the full index or a candidate pool."""
    if comp_weights is None:
        comp_weights = {}
    
    if candidate_pool is None:
        n = EMBED_V2.shape[0]
        sims = np.zeros(n, dtype=np.float32)
        total_w = 0.0
        # decide whether to chunk rows
        is_memmap = hasattr(EMBED_V2, 'filename') or (hasattr(EMBED_V2, 'mode') and getattr(EMBED_V2, 'mode') == 'r')
        rows_threshold = int(os.environ.get('EMBED_CHUNK_ROWS', '200000'))
        use_chunk = is_memmap or n > rows_threshold
        if use_chunk:
            # chunk across rows
            chunk_size = int(os.environ.get('EMBED_CHUNK_SIZE', '8192'))
            for comp, rng in SPANS_V2.items():
                w = float(comp_weights.get(comp, 0.0))
                if w == 0.0:
                    continue
                start, stop = int(rng[0]), int(rng[1])
                q_sub = query_embed[start:stop]
                total_w += w
                for rs in range(0, n, chunk_size):
                    re = min(n, rs + chunk_size)
                    block = np.asarray(EMBED_V2[rs:re, start:stop], dtype=np.float32)
                    sims[rs:re] += w * (block @ q_sub)
        else:
            for comp, rng in SPANS_V2.items():
                w = float(comp_weights.get(comp, 0.0))
                if w == 0.0:
                    continue
                start, stop = int(rng[0]), int(rng[1])
                q_sub = query_embed[start:stop]
                emb_sub = EMBED_V2[:, start:stop]
                sims += w * (emb_sub @ q_sub)
                total_w += w
        if total_w > 0:
            sims /= total_w
        cand_idx = np.argsort(-sims)
        cand_scores = sims[cand_idx]
        return cand_idx, cand_scores
    else:
        sims_pool = np.zeros(len(candidate_pool), dtype=np.float32)
        total_w = 0.0
        for comp, rng in SPANS_V2.items():
            w = float(comp_weights.get(comp, 0.0))
            if w == 0.0:
                continue
            start, stop = int(rng[0]), int(rng[1])
            q_sub = query_embed[start:stop]
            emb_sub = EMBED_V2[candidate_pool, start:stop]
            sims_pool += w * (emb_sub @ q_sub)
            total_w += w
        if total_w > 0:
            sims_pool /= total_w
        order_pool = np.argsort(-sims_pool)
        cand_idx = candidate_pool[order_pool]
        cand_scores = sims_pool[order_pool]
        return cand_idx, cand_scores

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
                # Prefer memory-mapped load to reduce resident memory if requested
                use_mmap = os.environ.get('EMBED_MMAP', '1') == '1'
                if use_mmap:
                    try:
                        EMBED_V2 = np.load(str(emb_path), mmap_mode='r')
                        print(f"[INFO] Loaded embeddings as memmap from {emb_path}")
                    except Exception:
                        # fallback to full load
                        EMBED_V2 = np.load(str(emb_path)).astype(np.float32)
                        print(f"[WARN] Failed memmap, loaded embeddings into memory from {emb_path}")
                else:
                    EMBED_V2 = np.load(str(emb_path)).astype(np.float32)
            elif 'embedding' in data:
                EMBED_V2 = np.array(data['embedding'], dtype=np.float32)
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
        log_memory("startup: before load_index")
        load_index()
        log_memory("startup: after load_index")
        load_model()
        log_memory("startup: after load_model")
        # Record baseline memory snapshot after models/index loaded
        try:
            global STARTUP_MEM
            STARTUP_MEM = get_memory_info()
            log_memory("startup: baseline recorded")
        except Exception:
            pass
        print("[INFO] FastAPI startup complete")
    except Exception as e:
        print(f"[ERROR] Startup failed: {e}")
    try:
        yield
    finally:
        # shutdown: persist and report
        try:
            log_memory("shutdown: before persist")
            try:
                _persist_index_changes()
            except Exception:
                pass
            log_memory("shutdown: after persist")
        except Exception:
            pass






    

    

# Add this new endpoint after the existing routes


app = FastAPI(
    title="Tile Image Search", 
    description="Image similarity search using DINOv2",
    lifespan=lifespan
)


@app.middleware("http")
async def memory_middleware(request: Request, call_next):
    """Log memory before and after each request and print delta."""
    try:
        log_memory(f"request: start {request.method} {request.url.path}")
        before = get_memory_info()
    except Exception:
        before = {"rss": 0.0}
    response = await call_next(request)
    try:
        after = get_memory_info()
        delta = after.get('rss', 0.0) - before.get('rss', 0.0)
        print(f"[MEM] request: end {request.method} {request.url.path} delta={_format_mem(int(delta))}")
    except Exception:
        pass
    return response


@app.get("/api/memory")
def api_memory_info() -> Dict[str, object]:
    """Return memory snapshot and delta vs startup baseline (if available)."""
    mi = get_memory_info()
    baseline = STARTUP_MEM if isinstance(STARTUP_MEM, dict) else {}
    delta = float(mi.get('rss', 0.0) - baseline.get('rss', 0.0)) if baseline else 0.0
    return {
        "memory": mi,
        "rss_human": _format_mem(int(mi.get('rss', 0))),
        "vms_human": _format_mem(int(mi.get('vms', 0))),
        "percent": mi.get('percent', 0.0),
        "baseline_rss": baseline.get('rss', 0.0),
        "baseline_rss_human": _format_mem(int(baseline.get('rss', 0))) if baseline else "0 B",
        "delta_rss": delta,
        "delta_rss_human": _format_mem(int(delta))
    }



def add_image_to_index(image_path: Path, object_id: str = None) -> bool:
    """Add a single image to the existing index without rebuilding."""
    global EMBED_V2, FAISS_INDEX, IMAGE_PATHS, IMAGE_IDS
    
    try:
        log_memory("add_image_to_index: start")
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
        EMBED_V2 = np.vstack([EMBED_V2, new_embed.reshape(1, -1)])
        IMAGE_PATHS.append(str(image_path.relative_to(IMAGES_DIR)))
        IMAGE_IDS.append(object_id)
        
        # Add to FAISS index if available
        if FAISS_INDEX is not None:
            FAISS_INDEX.add(new_embed.reshape(1, -1).astype(np.float32))
            log_memory("add_image_to_index: after faiss.add")
        



        with open(INDEX_V2_JSON, 'r') as f:
            index_data = json.load(f)

        # Update the JSON index file
        index_data['ids'] = IMAGE_IDS
        index_data['paths'] = IMAGE_PATHS

        # Determine how embeddings are stored and update accordingly
        emb_file = index_data.get('embedding_file') or index_data.get('embedding_path')
        try:
            def _is_rel_to(p: Path, base: Path) -> bool:
                try:
                    p.relative_to(base)
                    return True
                except Exception:
                    return False
            if emb_file:
                emb_path = Path(emb_file)
                if not emb_path.is_absolute():
                    emb_path = INDEX_V2_JSON.parent / emb_path
                emb_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = emb_path.with_suffix('.tmp.npy')
                np.save(str(tmp_path), EMBED_V2.astype(np.float32))
                os.replace(str(tmp_path), str(emb_path))
                # Normalize key usage: preserve whichever key existed
                rel_or_abs = str(emb_path.relative_to(INDEX_V2_JSON.parent)) if _is_rel_to(emb_path, INDEX_V2_JSON.parent) else str(emb_path)
                if 'embedding_file' in index_data:
                    index_data['embedding_file'] = rel_or_abs
                elif 'embedding_path' in index_data:
                    index_data['embedding_path'] = rel_or_abs
            else:
                index_data['embedding'] = EMBED_V2.tolist()
        except Exception as e:
            print(f"[ERROR] Failed writing embedding file: {e}. Falling back to inline.")
            index_data['embedding'] = EMBED_V2.tolist()
        
        with open(INDEX_V2_JSON, 'w') as f:
            json.dump(index_data, f)
        



        # Update FAISS index file if available
        if FAISS_INDEX is not None and INDEX_V2_FAISS.exists():
            faiss.write_index(FAISS_INDEX, str(INDEX_V2_FAISS))
        log_memory("add_image_to_index: done")
        
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
    log_memory("process_search: before extract_all")
    img = Image.open(save_path).convert('RGB')
    feat_dict = extract_all(BACKBONE_V2, img)
    log_memory("process_search: after extract_all")
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
            sims = EMBED_V2 @ query_embed
            cand_idx = np.argsort(-sims)[: max(8, rerank_k_local)]
            cand_scores = sims[cand_idx]
    
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
    log_memory("process_search: before geom verification")
    for idx_local in rerank_subset:
        cand_path = IMAGES_DIR / IMAGE_PATHS[idx_local]
        geom_scores.append(_geom_score_cached(save_path, cand_path))
    geom_scores = np.array(geom_scores, dtype=np.float32)
    log_memory("process_search: after geom verification")
    
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
    """Persist current in-memory index state (EMBED_V2, IMAGE_IDS, IMAGE_PATHS) to disk."""
    global EMBED_V2, IMAGE_IDS, IMAGE_PATHS, FAISS_INDEX
    try:
        with open(INDEX_V2_JSON, 'r') as f:
            index_data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed reading index JSON: {e}")

    index_data['ids'] = IMAGE_IDS
    index_data['paths'] = IMAGE_PATHS

    emb_file = index_data.get('embedding_file') or index_data.get('embedding_path')
    try:
        def _is_rel_to(p: Path, base: Path) -> bool:
            try:
                p.relative_to(base)
                return True
            except Exception:
                return False
        if emb_file:
            emb_path = Path(emb_file)
            if not emb_path.is_absolute():
                emb_path = INDEX_V2_JSON.parent / emb_path
            emb_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = emb_path.with_suffix('.tmp.npy')
            np.save(str(tmp_path), EMBED_V2.astype(np.float32))
            os.replace(str(tmp_path), str(emb_path))
            rel_or_abs = str(emb_path.relative_to(INDEX_V2_JSON.parent)) if _is_rel_to(emb_path, INDEX_V2_JSON.parent) else str(emb_path)
            if 'embedding_file' in index_data:
                index_data['embedding_file'] = rel_or_abs
            elif 'embedding_path' in index_data:
                index_data['embedding_path'] = rel_or_abs
        else:
            index_data['embedding'] = EMBED_V2.tolist()
    except Exception as e:
        # Fall back to inline
        index_data['embedding'] = EMBED_V2.tolist()
        print(f"[WARN] Persist fallback to inline embeddings: {e}")

    with open(INDEX_V2_JSON, 'w') as f:
        json.dump(index_data, f)

    # Rebuild FAISS index (simple FlatIP) if faiss available
    if HAVE_FAISS:
        try:
            dim = EMBED_V2.shape[1]
            # Use inner product if metric=IP present, else default IP
            if FAISS_INDEX is not None and hasattr(FAISS_INDEX, 'ntotal') and FAISS_INDEX.ntotal == 0:
                pass
            FAISS_INDEX_NEW = faiss.IndexFlatIP(dim)  # type: ignore
            FAISS_INDEX_NEW.add(EMBED_V2.astype(np.float32))
            faiss.write_index(FAISS_INDEX_NEW, str(INDEX_V2_FAISS))  # type: ignore
            FAISS_INDEX = FAISS_INDEX_NEW  # reassign
        except Exception as e:
            print(f"[WARN] Failed to rebuild FAISS index: {e}")


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
            EMBED_V2 = np.delete(EMBED_V2, all_remove_indices_sorted, axis=0)
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
    
