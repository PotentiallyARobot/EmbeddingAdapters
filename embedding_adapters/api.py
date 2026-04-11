from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
from huggingface_hub import snapshot_download

from .utils.decrypt_helper import decrypt_if_needed
from .adapter_core import ResidualMLPAdapterDeep
from .model_v2 import EmbeddingAdapter as V2EmbeddingAdapter
from .loader import (
    load_registry,
    find_adapter,
    AdapterEntry,
    ensure_local_adapter_dir,
    list_adapter_entries,
    PROJECT_ROOT,
)
from .quality import QualityModel  # quality scoring support

def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    """
    Resolve a user-specified device into a torch.device, with safe handling
    for environments where Torch is not compiled with CUDA.

    Rules:
      - If device is 'cuda' or startswith('cuda'):
          * if torch.cuda.is_available() -> use cuda
          * else -> raise a clear RuntimeError
      - If device is 'cpu' -> use CPU
      - If device is None -> auto:
          * try cuda if available
          * else CPU
    """
    if isinstance(device, torch.device):
        return device

    if isinstance(device, str):
        d = device.lower()
        if d.startswith("cuda"):
            # User is explicitly asking for CUDA
            try:
                if torch.cuda.is_available():
                    return torch.device("cuda")
            except AssertionError:
                # Torch build without CUDA can sometimes raise here
                pass
            raise RuntimeError(
                "device='cuda' requested but CUDA is not available or Torch "
                "was not compiled with CUDA support."
            )
        # e.g. "cpu" or "mps" or other backends
        return torch.device(d)

    # Auto device selection
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
    except AssertionError:
        # Torch not compiled with CUDA; fall back to CPU
        pass

    return torch.device("cpu")

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
        [optional] adapter_quality_stats.npz

    Example
    -------
    from embedding_adapters import EmbeddingAdapter

    # Just the adapter (no text encoder)
    adapter = EmbeddingAdapter.from_pair(
        "intfloat/e5-base-v2",
        "text-embedding-3-small",
    )

    # Adapter + HF text encoder loaded together:
    adapter = EmbeddingAdapter.from_pair(
        "intfloat/e5-base-v2",
        "text-embedding-3-small",
        load_source_encoder=True
    )

    mapped = adapter(source_embeddings)        # np.ndarray or torch.Tensor
    mapped_text = adapter.encode("some text")  # text → HF encoder → adapter

    If quality stats are present:
        scores = adapter.score_source(source_embeddings)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        metadata: AdapterMetadata,
        device: Optional[str] = None,
        quality: Optional[QualityModel] = None,
        *,
        load_source_encoder: bool = False,
        huggingface_token: str = None
    ):
        self.model = model
        self.metadata = metadata
        self._quality = quality  # may be None if no stats file

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        # Lazy HF text encoder (generic, based on metadata.source_model)
        self._hf_tokenizer = None
        self._hf_model = None
        self._hf_token = huggingface_token

        # If requested and source_model is set, load the HF encoder immediately
        if load_source_encoder and self.metadata.source_model is not None:
            self._ensure_hf_encoder()

    # Helper function
    def _maybe_normalize(
            self,
            arr: np.ndarray,
            normalize: Optional[bool],
    ) -> np.ndarray:
        """
        Normalize adapted embeddings if requested.

        normalize:
          - True  → force L2-normalize
          - False → leave as-is
          - None  → follow metadata.normalize from adapter_config.json
        """
        do_norm = self.metadata.normalize if normalize is None else normalize
        if not do_norm:
            return arr

        # L2-normalize rows
        norms = np.linalg.norm(arr, ord=2, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return arr / norms

    # ----------------------------
    # Constructors
    # ----------------------------
    @classmethod
    def from_local(
        cls,
        adapter_dir: str | os.PathLike,
        device: Optional[str] = None,
        *,
        load_source_encoder: bool = False,
        huggingface_token: str = None,
        entry: AdapterEntry = None
    ) -> "EmbeddingAdapter":
        """Load an adapter from a local directory.

        Auto-detects architecture:
          - v2: config has 'source_dim' + 'quality_hidden' → model_v2.EmbeddingAdapter
          - v1: config has 'in_dim' + 'arch' → adapter_core.ResidualMLPAdapterDeep
        """
        adapter_dir = Path(adapter_dir)

        # Resolve file names from entry or defaults
        config_name = "config.json"
        weights_name = "adapter.pt"
        stats_name = "adapter_quality_stats.npz"
        if entry is not None and isinstance(entry.primary, dict):
            config_name = entry.primary.get("config_file", config_name)
            weights_name = entry.primary.get("weights_file", weights_name)
            stats_name = entry.primary.get("scoring_file", stats_name)

        cfg_path = adapter_dir / config_name
        weights_path = adapter_dir / weights_name
        stats_path = adapter_dir / stats_name

        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # ── Detect architecture version ──
        is_v2 = "source_dim" in cfg and "quality_hidden" in cfg

        if is_v2:
            return cls._load_v2(
                cfg, weights_path, stats_path, device,
                load_source_encoder=load_source_encoder,
                huggingface_token=huggingface_token,
                entry=entry,
            )
        else:
            return cls._load_v1(
                cfg, weights_path, stats_path, device,
                load_source_encoder=load_source_encoder,
                huggingface_token=huggingface_token,
                entry=entry,
            )

    @classmethod
    def _load_v2(
        cls, cfg, weights_path, stats_path, device, *,
        load_source_encoder, huggingface_token, entry
    ):
        """Load a v2 adapter (model_v2.py — MLP + skip + SELU + quality head)."""
        in_dim = int(cfg["source_dim"])
        out_dim = int(cfg["target_dim"])
        normalize = bool(cfg.get("normalize", True))

        raw_model = V2EmbeddingAdapter(
            source_dim=in_dim,
            target_dim=out_dim,
            hidden_dim=int(cfg.get("hidden_dim", 1024)),
            num_layers=int(cfg.get("num_layers", 3)),
            dropout=float(cfg.get("dropout", 0.1)),
            use_skip=bool(cfg.get("use_skip", True)),
            quality_hidden=int(cfg.get("quality_hidden", 256)),
            activation=str(cfg.get("activation", "gelu")),
        )

        # Load weights
        state = torch.load(weights_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v
                     for k, v in state.items() if k != "n_averaged"}

        missing, unexpected = raw_model.load_state_dict(state, strict=False)
        if missing:
            print(f"[EmbeddingAdapter] v2 missing keys: {sorted(missing)}")
        if unexpected:
            print(f"[EmbeddingAdapter] v2 unexpected keys: {sorted(unexpected)}")

        # Wrap v2 so forward() returns just embedding (not the quality tuple).
        # This keeps __call__, encode(), encode_embeddings() working unchanged.
        class _V2Wrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.inner = m
            def forward(self, x):
                emb, _ = self.inner(x)
                return emb

        model = _V2Wrapper(raw_model)

        metadata = AdapterMetadata(
            in_dim=in_dim,
            out_dim=out_dim,
            arch=f"v2_mlp_h{cfg.get('hidden_dim', 1024)}_l{cfg.get('num_layers', 3)}_{cfg.get('activation', 'gelu')}",
            normalize=normalize,
            source_model=cfg.get("source_model"),
            target_model=cfg.get("target_model"),
            extra={k: v for k, v in cfg.items()
                   if k not in {"source_dim", "target_dim", "normalize",
                                "source_model", "target_model"}},
        )

        quality = None
        if stats_path.exists():
            try:
                quality = QualityModel(stats_path, device=device or "cpu")
                print(f"[EmbeddingAdapter] Loaded quality stats from {stats_path}")
            except Exception as e:
                print(f"[EmbeddingAdapter] Quality stats failed ({stats_path}): {e}")

        adapter = cls(
            model=model, metadata=metadata, device=device, quality=quality,
            load_source_encoder=load_source_encoder,
            huggingface_token=huggingface_token,
        )
        # Attach raw v2 model for direct quality head access via score_v2()
        adapter._v2_model = raw_model
        print(f"[EmbeddingAdapter] Loaded v2 adapter: {in_dim}d → {out_dim}d "
              f"({cfg.get('activation', 'gelu')}, {cfg.get('num_layers', 3)} layers, "
              f"skip={cfg.get('use_skip', True)})")
        return adapter

    @classmethod
    def _load_v1(
        cls, cfg, weights_path, stats_path, device, *,
        load_source_encoder, huggingface_token, entry
    ):
        """Load a v1 adapter (adapter_core.py — ResidualMLPAdapterDeep / GEGLU)."""
        in_dim = int(cfg["in_dim"])
        out_dim = int(cfg["out_dim"])
        arch = cfg.get("arch", "resmlp")
        normalize = bool(cfg.get("normalize", True))

        width = out_dim
        depth = 8
        if arch.startswith("resmlp_w") and "_d" in arch:
            try:
                parts = arch.split("_")
                for p in parts:
                    if p.startswith("w") and p[1:].isdigit():
                        width = int(p[1:])
                    if p.startswith("d") and p[1:].isdigit():
                        depth = int(p[1:])
            except Exception:
                pass

        model = ResidualMLPAdapterDeep(
            in_dim=in_dim, out_dim=out_dim, width=width, depth=depth,
            dropout=0.1, ls_init=5e-4, drop_path_rate=0.10,
        )

        # Load weights
        if entry is not None and isinstance(entry.primary, dict) \
                and "state_dict" in entry.primary:
            state = entry.primary["state_dict"]
        else:
            state = torch.load(
                weights_path, map_location=device, weights_only=False)

        if isinstance(state, dict) and not any(
            k.startswith("module.") for k in state.keys()
        ) and "model" in state:
            state = state["model"]

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
            print(f"[EmbeddingAdapter] v1 missing keys: {sorted(missing)}")
        if unexpected:
            print(f"[EmbeddingAdapter] v1 unexpected keys: {sorted(unexpected)}")

        metadata = AdapterMetadata(
            in_dim=in_dim, out_dim=out_dim, arch=arch, normalize=normalize,
            source_model=cfg.get("source_model"),
            target_model=cfg.get("target_model"),
            extra={k: v for k, v in cfg.items()
                   if k not in {"in_dim", "out_dim", "arch", "normalize",
                                "source_model", "target_model"}},
        )

        quality = None
        if stats_path.exists():
            try:
                quality = QualityModel(stats_path, device=device or "cpu")
                print(f"[EmbeddingAdapter] Loaded quality stats from {stats_path}")
            except Exception as e:
                print(f"[EmbeddingAdapter] Quality stats failed: {e}")

        return cls(
            model=model, metadata=metadata, device=device, quality=quality,
            load_source_encoder=load_source_encoder,
            huggingface_token=huggingface_token,
        )

    @classmethod
    def from_registry(
        cls,
        source: str,
        target: str,
        flavor: str = "generic",
        device: Optional[str] = None,
        *,
        load_source_encoder: bool = False,
        huggingface_token: str = None,
        slug: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> "EmbeddingAdapter":
        """Load an adapter by (source, target, flavor) from the registry.

        Adapters tagged with 'subscription' require an active license
        ($10/month). The SDK validates once, then caches for 24 hours.

        Parameters
        ----------
        api_key : str, optional
            API key for license validation. If not provided, uses saved key
            from ``embedding-adapters login``.
        """
        entry = find_adapter(source, target, flavor=flavor, slug=slug)

        if entry.mode != "local":
            raise NotImplementedError(
                f"Adapter mode '{entry.mode}' is not implemented yet. "
                "Use mode='local' for now."
            )

        # ── DRM: license validation for subscription adapters ──
        if "subscription" in (entry.tags or []):
            from .auth import validate_license, _load_saved_key
            license_key = api_key or _load_saved_key()
            print(f"[embedding_adapters] Validating license for '{entry.slug}'...")
            validate_license(entry.slug, api_key=license_key)
            print(f"[embedding_adapters] ✓ License valid for '{entry.slug}'")

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # For encrypted HF, this will download + decrypt to adapter.pt.
        # For plain HF/local_path, it just ensures files exist.
        adapter_dir = decrypt_if_needed(entry, device=device, hf_token=huggingface_token)

        return cls.from_local(
            adapter_dir,
            device=device,
            load_source_encoder=load_source_encoder,
            huggingface_token=huggingface_token,
            entry=entry
        )

    # --- Convenience constructors ---
    @classmethod
    def from_pair(
        cls,
        source: str,
        target: str,
        flavor: str = "generic",
        device: Optional[str] = None,
        *,
        load_source_encoder: bool = False,
        huggingface_token: str=None
    ) -> "EmbeddingAdapter":
        """Convenience alias for from_registry(source=..., target=...)."""
        return cls.from_registry(
            source=source,
            target=target,
            flavor=flavor,
            device=device,
            load_source_encoder=load_source_encoder,
            huggingface_token=huggingface_token
        )

    # ----------------------------
    # Call / map (vector → vector)
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
            try:
                arr = torch.from_numpy(x.astype("float32", copy=False))
            except Exception as e:
                print(e)
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
    # Generic text → adapted embedding
    # ----------------------------
    def encode(
            self,
            texts: Union[str, Sequence[str]],
            *,
            as_numpy: bool = True,
            normalize: Optional[bool] = None,
            return_source: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        High-level convenience: text → source embeddings → adapted embeddings.

        This is *generic*:
          - Uses metadata.source_model (if set) as a Hugging Face model id.
          - Downloads the model into PROJECT_ROOT/models/hf_models/<source_model>/.
          - Reuses the on-disk copy when available.
          - Applies simple mean pooling over last_hidden_state.
          - Then (by default) applies the adapter mapping (self.__call__).

        Requirements:
          - metadata.source_model must be a valid HF model id, e.g. "intfloat/e5-base-v2"
          - `transformers` and `huggingface_hub` must be installed.

        Parameters
        ----------
        texts : str or Sequence[str]
            Input text(s).
        as_numpy : bool
            If True (default), returns np.ndarray (N, dim) on CPU.
            If False, returns torch.Tensor (N, dim) on the adapter's device.
        normalize : Optional[bool]
            If True, L2-normalize the *adapted* embeddings.
            If False, leave them as-is.
            If None (default), use metadata.normalize from adapter_config.json.
            (Ignored when return_source=True.)
        return_source : bool
            If False (default), return *adapted* embeddings (after the adapter).
            If True, return the *source model's* embeddings (before the adapter).

        Returns
        -------
        np.ndarray or torch.Tensor
            If return_source=False: adapted embeddings in the target space (N, out_dim).
            If return_source=True: source embeddings from the HF model (N, in_dim).
        """
        if isinstance(texts, str):
            texts = [texts]

        if self.source_model is None:
            raise RuntimeError(
                "Cannot encode text generically: metadata.source_model is not set. "
                "Ensure your adapter_config.json includes a 'source_model' field "
                "with a valid Hugging Face model id."
            )

        # Get source-space embeddings via HF (on self.device)
        src_emb = self._encode_texts_via_hf(texts)  # torch.Tensor (N, in_dim)

        if src_emb.shape[1] != self.in_dim:
            raise RuntimeError(
                f"Encoded source embeddings have dim={src_emb.shape[1]}, "
                f"but adapter expects in_dim={self.in_dim}."
            )

        # If caller only wants source embeddings, return them here
        if return_source:
            if as_numpy:
                # NumPy must live on CPU
                return (
                    src_emb.detach()
                    .cpu()
                    .numpy()
                    .astype("float32", copy=False)
                )
            else:
                # Keep it on whatever device the adapter / HF model is using (CPU or GPU)
                return src_emb.detach()

        # Otherwise: map through the adapter (returns np.ndarray on CPU)
        mapped = self(src_emb)  # (N, out_dim), np.ndarray float32 (CPU)

        # Optional normalization in target space
        mapped = self._maybe_normalize(mapped, normalize)

        if as_numpy:
            return mapped
        else:
            # If user wants a tensor, we’ll give them a CPU tensor by default.
            # If you prefer to move back to self.device, use .to(self.device).
            return torch.from_numpy(mapped)


    # ----------------------------
    # Existing embedding → adapted embedding
    # ----------------------------
    def encode_embeddings(
            self,
            x: ArrayLike,
            *,
            as_numpy: bool = True,
            normalize: Optional[bool] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Translate *source* embeddings → target-space embeddings using the adapter.

        This is a convenience endpoint for when you already have embeddings
        from your source model (e.g. e5-base-v2) and just want to run the
        adapter on them.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            Source-space embeddings of shape (N, in_dim) or (in_dim,) for a single one.
        as_numpy : bool
            If True (default), returns np.ndarray with shape (N, out_dim).
            If False, returns torch.Tensor on CPU with the same shape.
        normalize : Optional[bool]
            If True, L2-normalize the *adapted* embeddings.
            If False, leave them as-is.
            If None (default), use metadata.normalize from adapter_config.json.

        Returns
        -------
        np.ndarray or torch.Tensor
            Adapted embeddings in the target space.
        """
        # Normalize x to a numpy array so we can inspect the shape
        if isinstance(x, np.ndarray):
            arr = x
        elif torch.is_tensor(x):
            arr = x.detach().cpu().numpy()
        else:
            raise TypeError(
                f"encode_embeddings expected np.ndarray or torch.Tensor, got {type(x)}"
            )

        # Allow a single embedding as 1D vector: (in_dim,) → (1, in_dim)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if arr.ndim != 2:
            raise ValueError(
                f"Expected embeddings of shape (N, in_dim) or (in_dim,), "
                f"got array with shape {arr.shape}"
            )

        if arr.shape[1] != self.in_dim:
            raise ValueError(
                f"Got embeddings with dim={arr.shape[1]}, "
                f"but adapter.in_dim={self.in_dim}"
            )

        # Reuse all the dtype/device logic in __call__
        mapped = self(arr)  # __call__ returns np.ndarray

        # Optional normalization in target space
        mapped = self._maybe_normalize(mapped, normalize)

        if as_numpy:
            return mapped
        else:
            return torch.from_numpy(mapped)

    # ----------------------------
    # Internal HF helpers
    # ----------------------------
    def _ensure_hf_encoder(self):
        """Initialize a generic HF text encoder from metadata.source_model.

        Downloads the model into PROJECT_ROOT/models/hf_models/<sanitized-source-model>
        using huggingface_hub.snapshot_download (with a progress bar), then
        loads AutoTokenizer/AutoModel from that local directory.
        """
        if self._hf_tokenizer is not None and self._hf_model is not None:
            return

        if self.source_model is None:
            raise RuntimeError(
                "metadata.source_model is not set; cannot initialize HF encoder."
            )

        try:
            from transformers import AutoTokenizer, AutoModel  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "`transformers` is required for EmbeddingAdapter.encode(). "
                "Install it with: pip install transformers"
            ) from e

        # Where to keep HF models on disk:
        hf_root = PROJECT_ROOT / "models" / "hf_models"
        hf_root.mkdir(parents=True, exist_ok=True)

        # Sanitize model id into a directory name
        sanitized = self.source_model.replace("/", "__")
        model_dir = hf_root / sanitized
        model_dir.mkdir(parents=True, exist_ok=True)

        # If there's no config.json yet, snapshot_download; otherwise assume it's there.
        config_path = model_dir / "config.json"
        if not config_path.exists():
            print(
                f"[EmbeddingAdapter] Downloading HF model '{self.source_model}' "
                f"to {model_dir} ..."
            )
            start = time.perf_counter()
            snapshot_download(
                repo_id=self.source_model,
                repo_type="model",
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
            )
            elapsed = time.perf_counter() - start
            print(
                f"[EmbeddingAdapter] Finished download of '{self.source_model}' "
                f"in {elapsed:.2f}s"
            )
        else:
            print(
                f"[EmbeddingAdapter] Using existing HF model at {model_dir} "
                f"for '{self.source_model}'"
            )

        # Now load tokenizer and model from the local directory
        print(
            f"[EmbeddingAdapter] Loading HF encoder for source_model='{self.source_model}' "
            f"from disk ({model_dir})"
        )
        self._hf_tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self._hf_model = AutoModel.from_pretrained(str(model_dir)).to(self.device)
        self._hf_model.eval()

    def _encode_texts_via_hf(self, texts: Sequence[str]) -> torch.Tensor:
        """
        Generic HF encoding:
          - tokenizes with AutoTokenizer
          - runs AutoModel
          - mean-pools over last_hidden_state using attention_mask

        Returns:
            torch.Tensor of shape (N, hidden_size) on self.device
        """
        self._ensure_hf_encoder()
        tokenizer = self._hf_tokenizer
        model = self._hf_model

        enc = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)
            last_hidden = out.last_hidden_state  # (batch, seq_len, hidden)

            # Mean pooling with attention mask
            mask = enc["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
            masked_embeddings = last_hidden * mask
            sum_embeddings = masked_embeddings.sum(dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            pooled = sum_embeddings / sum_mask

        # Optional: normalize for unit-norm source embeddings
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        return pooled

    # ----------------------------
    # v2 quality head access
    # ----------------------------
    @property
    def is_v2(self) -> bool:
        """True if this adapter uses the v2 architecture with built-in quality head."""
        return hasattr(self, '_v2_model') and self._v2_model is not None

    def score_v2(self, x: ArrayLike):
        """Get quality scores from the v2 model's built-in quality head.

        Returns (embeddings_np, quality_np) where quality is (B,) in [0,1].
        Only works if this adapter was loaded from a v2 config.

        Usage:
            embs, quality = adapter.score_v2(source_embeddings)
            # quality[i] = confidence that embs[i] faithfully replicates target
        """
        if not self.is_v2:
            raise RuntimeError(
                "score_v2() is only available for v2 adapters "
                "(config must have 'source_dim' and 'quality_hidden')"
            )

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype("float32"))
        x = x.to(self.device)

        self._v2_model.eval()
        with torch.no_grad():
            emb, quality = self._v2_model(x)
        return emb.cpu().numpy(), quality.cpu().numpy()

    # ----------------------------
    # Quality scoring (external stats file)
    # ----------------------------
    @property
    def has_quality(self) -> bool:
        """Return True if this adapter has quality stats available."""
        return self._quality is not None

    def score_source(self, x: ArrayLike) -> dict:
        """Score *source-space* embeddings (e.g. e5-base-v2) against the
        adapter's training distribution.

        Returns a dict of numpy arrays with keys such as:
            - "mahalanobis"
            - "knn_distance"
            - "conf_maha"
            - "conf_knn"
            - "confidence"  (combined score in [0,1])

        Raises:
            RuntimeError if no quality stats are available.
        """
        if self._quality is None:
            raise RuntimeError("No quality stats available for this adapter.")
        return self._quality.score_source(x)

    def score_target(self, y: ArrayLike) -> dict:
        """Score *target-space* embeddings (e.g. text-embedding-3-small)
        against the adapter's teacher distribution.

        Same output format as score_source()."""
        if self._quality is None:
            raise RuntimeError("No quality stats available for this adapter.")
        return self._quality.score_target(y)

    def score(self, x: ArrayLike) -> dict:
        """Alias for score_source(x) for convenience / backward compatibility."""
        return self.score_source(x)

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
    return list_adapter_entries()
