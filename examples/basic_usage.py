
import torch
from embedding_adapters import EmbeddingAdapter, list_adapters

print("Available adapters:")
for a in list_adapters():
    print(" -", a["slug"], "from", a["source"], "to", a["target"])

device = "cuda" if torch.cuda.is_available() else "cpu"

adapter = EmbeddingAdapter.from_pair(
    source="intfloat/e5-base-v2",
    target="text-embedding-3-small",
    device=device,
)

emb = adapter.encode("hello from examples")
print("Embedding shape:", emb.shape)
