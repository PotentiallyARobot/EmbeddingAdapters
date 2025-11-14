from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch

from .adapter_core import ResidualMLPAdapterDeep
from .loader import load_registry, find_adapter, AdapterEntry


ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass
class AdapterMetadata:
    in_dim: int
    out_dim: int
    arch: str
    normalize: bool
    source_model: Optional[str] = None
    target_model: Optional[str] = None
    extra: dict | None = None


class EmbeddingAdapter:
    """High-level wrapper around a trained embedding-space adapter.

    v0 focuses on local adapters, loaded from a directory that contains:

        adapter_config.json
        adapter.pt

    Example:

        from embedding_adapters import EmbeddingAdapter

        adapter = EmbeddingAdapter.from_local(
            "/content/drive/MyDrive/query-adapter",
            device="cuda"
        )

        mapped = adapter(source_embeddings)  # np.ndarray or torch.Tensor
    """

    def __init__(
        self,
        model: torch.nn.Module,
        metadata: AdapterMetadata,
        device: Optional[str] = None,
    ):
        self.model = model
        self.metadata = metadata

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    # ----------------------------
    # Constructors
    # ----------------------------
    @classmethod
    def from_local(cls, adapter_dir: str | os.PathLike, device: Optional[str] = None) -> "EmbeddingAdapter":
        """Load an adapter from a local directory.

        Expected files:

        - adapter_config.json
        - adapter.pt  (state_dict)
        """
        adapter_dir = Path(adapter_dir)
        cfg_path = adapter_dir / "adapter_config.json"
        weights_path = adapter_dir / "adapter.pt"

        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        in_dim = int(cfg["in_dim"])
        out_dim = int(cfg["out_dim"])
        arch = cfg.get("arch", "resmlp")
        normalize = bool(cfg.get("normalize", True))

        # For now we only support ResidualMLPAdapterDeep. We infer width/depth
        # from the arch string if present, otherwise we fall back to safe defaults.
        width = out_dim
        depth = 8
        if arch.startswith("resmlp_w") and "_d" in arch:
            try:
                # e.g. resmlp_w1536_d8_dp0.1_dpr0.1_ls0.0005
                parts = arch.split("_")
                for p in parts:
                    if p.startswith("w") and p[1:].isdigit():
                        width = int(p[1:])
                    if p.startswith("d") and p[1:].isdigit():
                        depth = int(p[1:])
            except Exception:
                pass

        model = ResidualMLPAdapterDeep(
            in_dim=in_dim,
            out_dim=out_dim,
            width=width,
            depth=depth,
            dropout=0.1,
            ls_init=5e-4,
            drop_path_rate=0.10,
        )

        # Load weights
        state = torch.load(weights_path, map_location="cpu")

        # Handle AveragedModel / EMA-style checkpoints where parameters
        # are under "module." and there may be an "n_averaged" key.
        if any(k.startswith("module.") for k in state.keys()):
            cleaned = {}
            for k, v in state.items():
                if k == "n_averaged":
                    continue
                if k.startswith("module."):
                    k = k[len("module."):]
                cleaned[k] = v
            state = cleaned

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[EmbeddingAdapter] Warning: missing keys in state_dict: {sorted(missing)}")
        if unexpected:
            print(f"[EmbeddingAdapter] Warning: unexpected keys in state_dict: {sorted(unexpected)}")

        metadata = AdapterMetadata(
            in_dim=in_dim,
            out_dim=out_dim,
            arch=arch,
            normalize=normalize,
            source_model=cfg.get("source_model"),
            target_model=cfg.get("target_model"),
            extra={k: v for k, v in cfg.items()
                   if k not in {"in_dim", "out_dim", "arch", "normalize", "source_model", "target_model"}},
        )

        return cls(model=model, metadata=metadata, device=device)

    @classmethod
    def from_registry(
        cls,
        source: str,
        target: str,
        flavor: str = "generic",
        device: Optional[str] = None,
    ) -> "EmbeddingAdapter":
        """(Future) Load adapter using a registry entry.

        v0 implementation assumes local adapters only, using entries like:

            {
              "slug": "e5-base-v2_to_text-embedding-3-small",
              "source": "intfloat/e5-base-v2",
              "target": "text-embedding-3-small",
              "flavor": "generic",
              "mode": "local",
              "primary": {
                "type": "local_path",
                "path": "/path/to/adapter/dir"
              }
            }

        You can edit registry.json manually for now.
        """
        entry = find_adapter(source, target, flavor=flavor)

        if entry.mode != "local":
            raise NotImplementedError(
                f"Adapter mode '{entry.mode}' is not implemented in v0. "
                "Use EmbeddingAdapter.from_local(...) or extend the loader."
            )

        if not entry.primary or entry.primary.get("type") != "local_path":
            raise ValueError(
                "Registry entry must have primary.type='local_path' with a 'path' field for v0."
            )

        adapter_path = entry.primary["path"]
        return cls.from_local(adapter_path, device=device)

    # ----------------------------
    # Call / map
    # ----------------------------
    def __call__(self, x: ArrayLike) -> np.ndarray:
        """Map embeddings from source space → target space.

        Accepts:
            - numpy array of shape (N, in_dim)
            - torch tensor of shape (N, in_dim)

        Returns:
            numpy array (N, out_dim)
        """
        self.model.eval()

        if isinstance(x, np.ndarray):
            arr = torch.from_numpy(x.astype("float32", copy=False))
        elif torch.is_tensor(x):
            arr = x
            if arr.dtype != torch.float32:
                arr = arr.float()
        else:
            raise TypeError("Expected numpy.ndarray or torch.Tensor for x")

        arr = arr.to(self.device)
        with torch.no_grad():
            out = self.model(arr)
        out = out.detach().cpu().numpy().astype("float32", copy=False)
        return out

    # ----------------------------
    # Introspection helpers
    # ----------------------------
    @property
    def in_dim(self) -> int:
        return self.metadata.in_dim

    @property
    def out_dim(self) -> int:
        return self.metadata.out_dim

    @property
    def source_model(self) -> Optional[str]:
        return self.metadata.source_model

    @property
    def target_model(self) -> Optional[str]:
        return self.metadata.target_model


def list_adapters() -> list[dict]:
    """Return the current registry as a list of dicts.

    For v0 this just mirrors registry.json; in the future it can aggregate
    local + remote registries, service-only adapters, etc.
    """
    from .loader import list_adapter_entries
    return list_adapter_entries()
