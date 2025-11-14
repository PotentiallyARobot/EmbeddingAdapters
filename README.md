
# embedding_adapters

Lightweight **embedding-to-embedding** adapters.

First shipped adapter:

- `emb_adapter_e5-base-v2_to_text-embedding-3-small`  
  Maps **intfloat/e5-base-v2** → **text-embedding-3-small**.

## Install (from source)

```bash
pip install -e .
```

## Basic usage

```python
from embedding_adapters import EmbeddingAdapter, list_adapters

print(list_adapters())

adapter = EmbeddingAdapter.from_pair(
    source="intfloat/e5-base-v2",
    target="text-embedding-3-small",
    device="cuda",   # or "cpu"
)

v = adapter.encode("hello from embedding adapters")
print(v.shape)  # (1, 1536)
```
