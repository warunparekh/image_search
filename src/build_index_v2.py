from __future__ import annotations

from pathlib import Path

"""High-performance index builder with GPU batch extraction & advanced FAISS options.

Key improvements over older versions:
 - True GPU batched backbone inference (reduces Python + model overhead).
 - Optional AMP (mixed precision) for faster ViT forward on modern GPUs.
 - Optional disabling of classical CPU descriptors for a pure-transformer index.
 - When classical descriptors enabled, they are computed in a lightweight loop; you can
     later extend to multiprocessing if CPU becomes the bottleneck.
 - Embeddings stored as binary .npy referenced via `embedding_path` (no giant JSON).
 - Supports FAISS flat / HNSW / IVF+PQ; optional GPU FAISS add/training.
 - Memory-mapped incremental write to avoid large RAM spikes for 70k+ images.
"""

from pathlib import Path
import json, os, math
from typing import List, Dict, Tuple
import numpy as np
from numpy.lib.format import open_memmap
from PIL import Image
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor

# Support running as a package (relative imports) or as a standalone script
# (when executing `python src\build_index_v2.py` the directory `src` is on
# sys.path and relative imports with a leading dot raise ImportError).
try:
    from .backbones import get_backbone
    from . import features as F  # reuse individual feature fns
    from .features import concat_and_normalize  # we will still use its span logic
except Exception:
    # fallback for direct script execution: import as top-level modules from src/
    from backbones import get_backbone
    import features as F
    from features import concat_and_normalize

try:  # FAISS (maybe GPU build)
    import faiss  # type: ignore
    HAVE_FAISS = True
except Exception:
    faiss = None  # type: ignore
    HAVE_FAISS = False

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}


def strip_paths(ids_json: Path, images_root: Path) -> Tuple[List[Path], List[str]]:
    with open(ids_json, 'r') as f:
        data = json.load(f)
    images: List[Path] = []
    ids: List[str] = []
    empty = 0
    for rec in data:
        try:
            img_path = images_root / rec['imageName']
            images.append(img_path)
            ids.append(rec['_id'])
        except KeyError:
            empty += 1
            continue
    if empty:
        print(f"[WARN] {empty} records skipped (missing imageName/_id)")
    return images, ids


def _extract_transformer_batch(bb, pil_batch: List[Image.Image], device: torch.device, amp: bool):
    tensors = [bb.preprocess(img) for img in pil_batch]
    batch = torch.stack(tensors, dim=0).to(device, non_blocking=True)
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(amp and device.type == 'cuda')):
            raw = bb._forward(batch)  # shape (B, N, D) or (B, D)
    # normalize outputs & derive (global, cls, patch_mean)
    outs = []
    if isinstance(raw, dict):
        # pick a representative tensor (prefer patch tokens)
        for k in ('x_norm_patchtokens', 'last_hidden_state', 'tokens'):
            if k in raw:
                core = raw[k]
                break
        else:
            core = list(raw.values())[0]
    else:
        core = raw
    if core.dim() == 3:  # (B, N, D)
        patch_mean = core.mean(1)
        cls_tok = core[:, 0, :] if core.size(1) > 0 else patch_mean
        global_vec = patch_mean  # treat mean as global
    elif core.dim() == 2:
        global_vec = core
        patch_mean = core
        cls_tok = core
    else:  # fallback
        flat = core.view(core.size(0), -1)
        global_vec = patch_mean = cls_tok = flat
    def _l2(x):
        x = x.float()
        n = torch.norm(x, dim=1, keepdim=True) + 1e-12
        return (x / n).cpu().numpy().astype(np.float32)
    return _l2(global_vec), _l2(cls_tok), _l2(patch_mean)


def _classical_worker_tuple(args):
    """Top-level worker for multiprocessing: returns classical features + relative path string."""
    img_path, = args
    try:
        img = Image.open(img_path).convert('RGB')
        return (
            F.lab_hist(img),
            F.color_moments(img),
            F.dominant_colors(img, k=3),
            F.hog_descriptor(img),
            F.lbp_hist(img),
            F.gabor_energy(img),
            F.glcm_stats(img),
            str(img_path)
        )
    except Exception:
        return (np.zeros(32, dtype=np.float32),) * 7 + (str(img_path),)


def build_index(
    images_dir: Path,
    ids_json: Path,
    out_dir: Path,
    backbone_name: str = 'dinov2',
    strict: bool = False,
    batch_size: int = 96,
    amp: bool = True,
    disable_classical: bool = False,
    float16_memmap: bool = False,
    add_batch_size: int = 2048,
    faiss_type: str = 'flat',
    gpu_faiss: bool = False,
    nlist: int = 4096,
    m_pq: int = 32,
    nbits: int = 8,
    hnsw_ef: int = 200,
    hnsw_m: int = 32,
    seed: int = 42,
):
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(out_dir, exist_ok=True)
    bb = get_backbone(backbone_name, strict=strict, verbose=True)
    def _get_bb_device(bb_obj, default: torch.device):
        """Try a few safe strategies to find the backbone's device; fall back to default."""
        try:
            # explicit attribute
            if hasattr(bb_obj, 'device'):
                return getattr(bb_obj, 'device')
            # common wrapped module attributes
            for attr in ('model', 'net', 'backbone'):
                obj = getattr(bb_obj, attr, None)
                if obj is not None:
                    try:
                        return next(obj.parameters()).device
                    except Exception:
                        pass
            # try to inspect closure cells of _forward (best-effort, safe)
            fwd = getattr(bb_obj, '_forward', None)
            closure = getattr(fwd, '__closure__', None)
            if closure:
                for cell in closure:
                    try:
                        val = cell.cell_contents
                    except Exception:
                        continue
                    if hasattr(val, 'device'):
                        return getattr(val, 'device')
                    try:
                        return next(iter(getattr(val, 'parameters', lambda: [])())).device
                    except Exception:
                        continue
        except Exception:
            pass
        return default

    bb_device = _get_bb_device(bb, device)
    paths, ids = strip_paths(ids_json, images_dir)
    if not paths:
        raise SystemExit('No images found')

    # gather valid paths only
    valid: List[Path] = []
    valid_ids: List[str] = []
    for p, pid in zip(paths, ids):
        if p.is_file():
            valid.append(p)
            valid_ids.append(pid)
    paths = valid; ids = valid_ids
    total = len(paths)
    print(f"[INFO] Building index over {total} images (classical={'off' if disable_classical else 'on'})")

    # We'll construct embeddings directly into a memory-mapped .npy to avoid
    # holding large lists in RAM for 70k+ images. Strategy:
    # 1) Probe a single image to determine transformer output dims + classical dims
    # 2) Create an open_memmap for full embedding matrix
    # 3) Use a ThreadPoolExecutor to prefetch & preprocess images into tensors
    #    and run GPU batched forward; use a persistent multiprocessing Pool
    #    to compute classical features per batch when enabled.
    classical_keys = ['lab_hist','color_moments','dominant_colors','hog','lbp','gabor','glcm']
    sample_idx = 0
    sample_img = None
    for p in paths:
        try:
            sample_img = Image.open(p).convert('RGB')
            break
        except Exception:
            continue
    if sample_img is None:
        raise SystemExit('No readable sample image for shape probe')

    # probe transformer output shapes
    g_s, c_s, pm_s = _extract_transformer_batch(bb, [sample_img], bb_device, amp)
    # arrays returned are numpy arrays with shape (B, D)
    d_global = g_s.shape[1]
    d_cls = c_s.shape[1]
    d_patch = pm_s.shape[1]

    # probe classical dims if enabled
    classical_dims = {}
    if not disable_classical:
        try:
            sample_classical = _classical_worker_tuple((paths[0],))
            for idx, k in enumerate(classical_keys):
                classical_dims[k] = int(np.asarray(sample_classical[idx]).size)
        except Exception:
            # fallback defaults
            classical_dims = {k: 32 for k in classical_keys}

    # compute spans and total dim
    spans = {}
    cursor = 0
    spans['global'] = [cursor, cursor + d_global]; cursor += d_global
    spans['cls'] = [cursor, cursor + d_cls]; cursor += d_cls
    spans['patch_mean'] = [cursor, cursor + d_patch]; cursor += d_patch
    if not disable_classical:
        for k in classical_keys:
            dims = classical_dims.get(k, 0)
            if dims > 0:
                spans[k] = [cursor, cursor + dims]
                cursor += dims
    total_dim = cursor

    print(f"[INFO] Probed dims -> global={d_global} cls={d_cls} patch={d_patch} classical_sum={sum(classical_dims.values()) if classical_dims else 0} total_dim={total_dim}")

    # create memory-mapped array for embeddings (npy file with header)
    emb_file = out_dir / 'embeddings.npy'
    emb_mem = open_memmap(str(emb_file), mode='w+', dtype=np.float32, shape=(total, total_dim))

    rel_paths = [None] * total

    # Prepare classical multiprocessing Pool once
    from multiprocessing import Pool, cpu_count
    classical_pool = None
    if not disable_classical:
        classical_pool = Pool(processes=max(2, min(cpu_count(), batch_size)))

    # helper to load & preprocess a single image to tensor (runs in thread)
    def _load_and_preprocess(pth: Path):
        try:
            im = Image.open(pth).convert('RGB')
        except Exception:
            im = Image.new('RGB', (224, 224))
        t = bb.preprocess(im)
        try:
            im.close()
        except Exception:
            pass
        return t

    # Main batched loop: prefetch with ThreadPoolExecutor, do GPU forward, write into memmap
    num_workers = min(8, max(2, os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=num_workers) as exe:
        for start in tqdm(range(0, total, batch_size), desc='Backbone batches'):
            batch_paths = paths[start:start+batch_size]
            # prefetch + preprocess in parallel
            tensors = list(exe.map(_load_and_preprocess, batch_paths))
            batch = torch.stack(tensors, dim=0).to(bb_device, non_blocking=True)
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(amp and bb_device.type == 'cuda')):
                    raw = bb._forward(batch)
            # normalize & extract
            if isinstance(raw, dict):
                for k in ('x_norm_patchtokens', 'last_hidden_state', 'tokens'):
                    if k in raw:
                        core = raw[k]
                        break
                else:
                    core = list(raw.values())[0]
            else:
                core = raw
            if core.dim() == 3:
                patch_mean = core.mean(1)
                cls_tok = core[:, 0, :] if core.size(1) > 0 else patch_mean
                global_vec = patch_mean
            elif core.dim() == 2:
                global_vec = core
                patch_mean = core
                cls_tok = core
            else:
                flat = core.view(core.size(0), -1)
                global_vec = patch_mean = cls_tok = flat

            def _to_numpy_l2(x):
                x = x.float()
                n = torch.norm(x, dim=1, keepdim=True) + 1e-12
                return (x / n).cpu().numpy().astype(np.float32)

            g = _to_numpy_l2(global_vec)
            c = _to_numpy_l2(cls_tok)
            pm = _to_numpy_l2(patch_mean)

            # write to memmap at proper spans
            for r_idx, pth in enumerate(batch_paths):
                row = start + r_idx
                # write transformer parts
                s_g, e_g = spans['global']
                s_c, e_c = spans['cls']
                s_p, e_p = spans['patch_mean']
                emb_mem[row, s_g:e_g] = g[r_idx]
                emb_mem[row, s_c:e_c] = c[r_idx]
                emb_mem[row, s_p:e_p] = pm[r_idx]
                # write classical (computed via pool per-batch)
                if not disable_classical:
                    # compute classical for this image synchronously via pool.apply (cheap since pool already warm)
                    # use apply_async per image and collect
                    pass
                rel_paths[row] = str(Path(pth).relative_to(images_dir))

            # classical features computed per-batch using pool.map to reduce overhead
            if not disable_classical:
                try:
                    results = classical_pool.map(_classical_worker_tuple, [(p,) for p in batch_paths])
                    for r_idx, res in enumerate(results):
                        row = start + r_idx
                        for idx_k, k in enumerate(classical_keys):
                            if k not in spans:
                                continue
                            s, e = spans[k]
                            arr = np.asarray(res[idx_k], dtype=np.float32).ravel()
                            emb_mem[row, s:e] = arr
                except Exception:
                    # best-effort: fill zeros on failure
                    for r_idx in range(len(batch_paths)):
                        row = start + r_idx
                        for k in classical_keys:
                            if k in spans:
                                s, e = spans[k]
                                emb_mem[row, s:e] = 0.0

            # free GPU memory
            del batch, raw, global_vec, cls_tok, patch_mean
            torch.cuda.empty_cache() if bb_device.type == 'cuda' else None

    # close classical pool
    if classical_pool is not None:
        classical_pool.close(); classical_pool.join()

    # final row-wise L2 normalization (in-place, chunked)
    chunk = max(1024, batch_size * 4)
    for s in tqdm(range(0, total, chunk), desc='Normalizing rows'):
        e = min(total, s + chunk)
        block = emb_mem[s:e]
        norms = np.linalg.norm(block, axis=1, keepdims=True) + 1e-12
        emb_mem[s:e] = (block / norms).astype(np.float32)

    # ensure memmap is flushed
    del emb_mem
    # reload final matrix for subsequent steps
    global_arr = None
    cls_arr = None
    patch_arr = None
    mat = np.load(str(emb_file)).astype(np.float32)

    # We already produced a flushed embeddings.npy via memmap and loaded it as `mat`.
    index_dict = {
        'version': 3,
        'backbone': bb.name,
        'ids': ids,
        'images_dir': str(images_dir.resolve()),
        'paths': rel_paths,
        'embedding_path': emb_file.name,
        'embedding_dtype': str(mat.dtype),
        'embedding_shape': list(mat.shape),
        'spans': spans,
        'metric': 'IP',
        'notes': f'batch_dinov2 classical={not disable_classical} amp={amp}',
    }
    with open(out_dir / 'index_v2.json','w') as f:
        json.dump(index_dict, f)

    # Build FAISS
    if HAVE_FAISS:
        d = mat.shape[1]
        if faiss_type == 'flat':
            faiss_index = faiss.IndexFlatIP(d)
        elif faiss_type == 'hnsw':
            try:
                faiss_index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            except Exception:
                faiss_index = faiss.IndexHNSWFlat(d, hnsw_m)
            faiss_index.hnsw.efConstruction = hnsw_ef
        else:  # ivf_pq
            quantizer = faiss.IndexFlatL2(d)
            nlist_eff = max(32, min(nlist, mat.shape[0] // 8))
            faiss_index = faiss.IndexIVFPQ(quantizer, d, nlist_eff, m_pq, nbits)
            # training sample
            sample_sz = min(20000, mat.shape[0])
            sample = mat[np.random.choice(mat.shape[0], sample_sz, replace=False)]
            print(f"[FAISS] Training IVF+PQ on {sample.shape[0]} samples (nlist={nlist_eff} m={m_pq} nbits={nbits})")
            faiss_index.train(sample)
        # optional GPU acceleration for add
        if gpu_faiss and device.type == 'cuda':
            try:
                res = faiss.StandardGpuResources()  # type: ignore
                faiss_index = faiss.index_cpu_to_all_gpus(faiss_index)  # distribute
                print('[FAISS] Using GPU(s) for add')
            except Exception as e:
                print('[FAISS] GPU add failed, falling back to CPU:', e)
        for i in range(0, mat.shape[0], add_batch_size):
            faiss_index.add(mat[i:i+add_batch_size])
        # bring back to CPU if GPU index
        try:
            faiss_index_cpu = faiss.index_gpu_to_cpu(faiss_index)
        except Exception:
            faiss_index_cpu = faiss_index
        faiss.write_index(faiss_index_cpu, str(out_dir / 'index_v2.faiss'))
        print(f"[FAISS] Wrote index ({faiss_type}) -> {out_dir / 'index_v2.faiss'}")

    print(f"[DONE] Indexed {mat.shape[0]} images | dim={mat.shape[1]} | classical={'on' if not disable_classical else 'off'} | faiss={HAVE_FAISS}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='High-speed image index builder')
    parser.add_argument('--images', type=Path, default=Path('images'))
    parser.add_argument('--ids-json', type=Path, default=Path('product_ids.json'))
    parser.add_argument('--out', type=Path, default=Path('data/index'))
    parser.add_argument('--backbone', type=str, default='dinov2')
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--batch-size', type=int, default=96)
    parser.add_argument('--amp', action='store_true', help='Enable mixed precision (recommended on CUDA)')
    parser.add_argument('--disable-classical', action='store_true', help='Only transformer features (faster, smaller).')
    parser.add_argument('--float16-memmap', action='store_true', help='(kept for compat; embeddings always saved float32 final)')
    parser.add_argument('--add-batch-size', type=int, default=2048)
    parser.add_argument('--faiss-type', type=str, default='flat', choices=['flat','hnsw','ivf_pq'])
    parser.add_argument('--gpu-faiss', action='store_true', help='Use GPU(s) for FAISS add/train if available')
    parser.add_argument('--nlist', type=int, default=4096)
    parser.add_argument('--m-pq', type=int, default=32)
    parser.add_argument('--nbits', type=int, default=8)
    parser.add_argument('--hnsw-ef', type=int, default=200)
    parser.add_argument('--hnsw-m', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    build_index(
        images_dir=args.images,
        ids_json=args.ids_json,
        out_dir=args.out,
        backbone_name=args.backbone,
        strict=args.strict,
        batch_size=args.batch_size,
        amp=args.amp,
        disable_classical=args.disable_classical,
        float16_memmap=args.float16_memmap,
        add_batch_size=args.add_batch_size,
        faiss_type=args.faiss_type,
        gpu_faiss=args.gpu_faiss,
        nlist=args.nlist,
        m_pq=args.m_pq,
        nbits=args.nbits,
        hnsw_ef=args.hnsw_ef,
        hnsw_m=args.hnsw_m,
        seed=args.seed,
    )
