"""EmbeddingAdapters: neural adapters for mapping between embedding spaces.

Typical usage:

    from embedding_adapters import EmbeddingAdapter

    adapter = EmbeddingAdapter.from_local(
        "/path/to/adapter_dir",  # contains adapter.pt + adapter_config.json
        device="cuda"
    )

    mapped = adapter(some_embeddings)  # numpy array or torch tensor

This v0 focuses on local adapters. Future versions may also load from
registries, Hugging Face Hub, or your own service.
"""

from .api import EmbeddingAdapter, list_adapters

__all__ = ["EmbeddingAdapter", "list_adapters"]

__version__ = "0.0.1"
