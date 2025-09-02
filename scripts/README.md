Usage notes for embedding conversion and memory-reduction tips

1) Convert embeddings to float16 memmap (recommended for large indexes):

   python .\scripts\convert_embeddings_to_float16.py --index data/index/index_v2.json

   This produces a file next to the existing embeddings named like `embedding-f16.npy`.
   Update `data/index/index_v2.json` to point to the new file (key `embedding_file` or `embedding_path`).

2) Runtime flags for the API (PowerShell):

   # prefer memmap loading
   $env:EMBED_MMAP = "1"

   # lower threshold to force chunking (if you want to test chunking on small indexes)
   $env:EMBED_CHUNK_ROWS = "1000"

   # change chunk size
   $env:EMBED_CHUNK_SIZE = "4096"

3) Next steps:
 - Consider building a FAISS IVF+PQ index for very large datasets (trade memory for accuracy/speed).
 - Consider quantizing embeddings (float16 or 8-bit) for even lower memory.
 - Keep a backup copy of the original embeddings until validated.
