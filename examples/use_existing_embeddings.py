import os
import sys
from dotenv import load_dotenv
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
load_dotenv()

import numpy as np
import torch
from embedding_adapters import EmbeddingAdapter, list_adapters
device = "cuda" if torch.cuda.is_available() else "cpu"
adapter = EmbeddingAdapter.from_pair(
    source="intfloat/e5-base-v2",
    target="openai/text_embedding_3_small",
    flavor="linear",
    device=device,
    huggingface_token=os.environ['HUGGINGFACE_TOKEN']
)

# Example: pretend this came from a Gemini-like API
gemini_response = {
    "embeddings": [
        {"values": [0.01] * adapter.in_dim},  # THESE COULD BE YOUR EMBEDDINGS YOU GET FROM SOMETHING LIKE GEMINI
        {"values": [0.01] * adapter.in_dim},  # Each row here would represent the embedding vector of a question
        {"values": [0.01] * adapter.in_dim},
        {"values": [0.01] * adapter.in_dim},
        {"values": [0.01] * adapter.in_dim},
        {"values": [0.01] * adapter.in_dim},
        {"values": [0.01] * adapter.in_dim},
        {"values": [0.01] * adapter.in_dim},
        {"values": [0.01] * adapter.in_dim},
        {"values": [0.01] * adapter.in_dim}

    ]
}

# Collect embeddings from the JSON into a Python list
source_vectors = []
for emb_obj in gemini_response["embeddings"]:
    vec = emb_obj["values"]
    source_vectors.append(vec)

# Convert to a (N, in_dim) float32 array
source_emb_np = np.array(source_vectors, dtype="float32")
print("Source embeddings array shape:", source_emb_np.shape)  # (N, in_dim)

import time
# Now run them through the adapter
start = time.time()
translated = adapter.encode_embeddings(source_emb_np)  # (N, out_dim)
elapsed_ms = ( time.time() - start ) * 1000
print(f"[Device: {device}]")
print(f"Elapsed time for 10 calls in batch mode: {elapsed_ms:.2f} ms")
print(f"Average time for 1 call: {(elapsed_ms/10):.2f} ms")
print("Translated embeddings shape:", translated.shape)
print("First 8 dims of first translated emb:", translated[0])