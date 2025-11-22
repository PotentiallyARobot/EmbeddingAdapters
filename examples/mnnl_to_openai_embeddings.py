import os
import sys
from dotenv import load_dotenv
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
load_dotenv()
import os
import sys
import time
import os
import sys)
import numpy as np
import torch

# --- 1) Install + import google-genai ---
os.system(f'"{sys.executable}" -m pip install -q sentence-transformers')
import numpy as np
from sentence_transformers import SentenceTransformer
from embedding_adapters import EmbeddingAdapter

# 1) Compute source embeddings with a local / open-source model
src_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2) Load a pre-trained adapter from the registry
adapter = EmbeddingAdapter.from_registry(
    source="sentence-transformers/all-MiniLM-L6-v2",
    target="gemini/text-embedding-004",
    flavor="generic",
    device="gpu",
    huggingface_token=os.environ['HUGGINGFACE_TOKEN']
)

texts = [
    "NASA announces discovery of Earth-like exoplanet."
]

# --- 7) Run embeddings through the adapter on GPU ---
start = time.time()
src_embs = src_model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True,  # important: matches adapter training setup
)
translated = adapter.encode_embeddings(src_embs)  # (N, out_dim)
elapsed_ms = (time.time() - start) * 1000.0
device = 'cpu'
print(f"[Device: {device}]")
print(f"Elapsed time for {len(texts)} embeddings in batch: {elapsed_ms:.2f} ms")
print(f"Average per embedding: {(elapsed_ms / len(texts)):.2f} ms")
print("Translated embeddings shape:", translated.shape)
print("First 8 dims of first translated emb:", translated[0][:8])
