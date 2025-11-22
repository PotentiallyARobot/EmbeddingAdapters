import os
import sys
from dotenv import load_dotenv
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
load_dotenv()
import time

import numpy as np
import torch

# --- 1) Install + import google-genai ---
os.system(f'"{sys.executable}" -m pip install -q google-genai')

from google import genai
from google.genai.types import EmbedContentConfig


from embedding_adapters import EmbeddingAdapter, list_adapters

# --- 3) Device selection ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- 4) Gemini client setup ---
# Prefer environment variable for security
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "*************")
if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
    raise RuntimeError("Please set GEMINI_API_KEY env var or replace the placeholder.")

client = genai.Client(api_key=GEMINI_API_KEY)

# Some example texts to embed
texts = [
    "What is the capital of France?",
    "Explain contrastive learning in simple terms.",
    "How do I fine-tune an embedding adapter?",
    "Best practices for semantic search indexing.",
    "What are retrieval-augmented generation systems?",
    "Describe cosine similarity vs dot product.",
    "Tips for improving query embeddings.",
    "Explain vector databases to a beginner.",
    "How does Mahalanobis distance work?",
    "What is the difference between Gemini and OpenAI embeddings?",
]

print(f"Requesting {len(texts)} embeddings from Gemini...")

# --- 5) Call Gemini embeddings API ---
start = time.time()
response = client.models.embed_content(
    model="text-embedding-004",
    contents=texts,
    config=EmbedContentConfig(
        output_dimensionality=768,  # matches your adapter's in_dim
    ),
)
elapsed_ms = (time.time() - start) * 1000.0
print(f"Time to get gemini embeddings")
print(f"Elapsed time for {len(texts)} embeddings in batch: {elapsed_ms:.2f} ms")
print(f"Average per embedding: {(elapsed_ms / len(texts)):.2f} ms")


emb_list = [emb.values for emb in response.embeddings]
source_emb_np = np.asarray(emb_list, dtype="float32")
print("Gemini embeddings array shape:", source_emb_np.shape)  # (N, 768)

# --- 6) Load your new adapter from the registry ---
# NOTE: make sure your registry has this entry:
#   source="intfloat/e5-base-v2", target="text-embedding-3-small", flavor="generic"
print("Available adapters:", list_adapters())
adapter = EmbeddingAdapter.from_pair(
    source="gemini-text-embedding-004",
    target="text-embedding-3-small",
    flavor="generic",
    device='cpu',
    huggingface_token=os.environ['HUGGINGFACE_TOKEN']
)

print(f"Adapter in_dim: {adapter.in_dim}, out_dim: {adapter.out_dim}")

if adapter.in_dim != source_emb_np.shape[1]:
    raise ValueError(
        f"Dimension mismatch: adapter.in_dim={adapter.in_dim}, "
        f"but Gemini embeddings have dim={source_emb_np.shape[1]}"
    )

# --- 7) Run embeddings through the adapter on GPU ---
start = time.time()
translated = adapter.encode_embeddings(source_emb_np)  # (N, out_dim)
elapsed_ms = (time.time() - start) * 1000.0

print(f"[Device: {device}]")
print(f"Elapsed time for {len(texts)} embeddings in batch: {elapsed_ms:.2f} ms")
print(f"Average per embedding: {(elapsed_ms / len(texts)):.2f} ms")
print("Translated embeddings shape:", translated.shape)
print("First 8 dims of first translated emb:", translated[0][:8])
