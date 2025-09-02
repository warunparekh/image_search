"""
Convert a large .npy embedding file to float16 memmap safely in chunks.

Usage (PowerShell):
  python .\scripts\convert_embeddings_to_float16.py --index data/index/index_v2.json

This will:
 - read the index JSON to locate the embedding_file or embedding_path
 - create a new file next to it named <orig>-f16.npy as a memmap in float16
 - copy data in row-chunks to avoid a full memory spike
 - print the suggested JSON edit to point to the new file (don't modify JSON automatically)

Notes:
 - Keep a backup of the original embedding file.
 - The FastAPI app supports loading memmap float16 files; the code already casts blocks to float32 for dot-products.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import sys

CHUNK_ROWS = 65536


def convert_to_f16(orig_path: Path, out_path: Path, chunk_rows: int = CHUNK_ROWS):
    print(f"Converting {orig_path} -> {out_path} using chunk size {chunk_rows}")
    if not orig_path.exists():
        raise FileNotFoundError(orig_path)
    # load header info without forcing full array in memory
    try:
        src = np.load(str(orig_path), mmap_mode='r')
    except Exception:
        src = np.load(str(orig_path))
    shape = src.shape
    dtype = src.dtype
    print(f"Source shape={shape} dtype={dtype}")
    # create memmap destination
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dst = np.lib.format.open_memmap(str(out_path), mode='w+', dtype=np.float16, shape=shape)
    rows = shape[0]
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        print(f"  copying rows {start}:{stop}")
        block = np.asarray(src[start:stop], dtype=np.float32)
        # convert to float16 and write
        dst[start:stop] = block.astype(np.float16)
    # flush
    del dst
    print("Conversion complete")
    return out_path


def locate_embedding(index_json: Path) -> Path:
    if not index_json.exists():
        raise FileNotFoundError(index_json)
    data = json.loads(index_json.read_text(encoding='utf-8'))
    emb_file = data.get('embedding_file') or data.get('embedding_path')
    if not emb_file:
        raise RuntimeError('Index JSON does not reference embedding_file or embedding_path')
    emb_path = Path(emb_file)
    if not emb_path.is_absolute():
        emb_path = index_json.parent / emb_path
    return emb_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', '-i', type=str, default='data/index/index_v2.json', help='Path to index JSON')
    p.add_argument('--out', '-o', type=str, help='Optional output path for float16 npy')
    p.add_argument('--chunk', type=int, default=CHUNK_ROWS, help='Rows per chunk')
    args = p.parse_args()
    idx = Path(args.index)
    try:
        emb = locate_embedding(idx)
    except Exception as e:
        print(f"Failed to locate embedding: {e}")
        sys.exit(2)
    if args.out:
        out = Path(args.out)
    else:
        out = emb.with_name(emb.stem + '-f16.npy')
    # avoid overwriting same file
    if out.exists():
        print(f"Output already exists: {out}")
        resp = input("Overwrite? (y/N): ")
        if resp.strip().lower() != 'y':
            print("Aborting")
            sys.exit(0)
    convert_to_f16(emb, out, chunk_rows=args.chunk)
    rel = out.relative_to(idx.parent) if out.exists() and out.is_relative_to(idx.parent) else str(out)
    print('\nConversion done. Suggested change to index JSON:')
    print(f"  set 'embedding_file' (or 'embedding_path') to: {rel}")
    print("Keep the original file as a backup until you verify searches behave the same.")


if __name__ == '__main__':
    main()
