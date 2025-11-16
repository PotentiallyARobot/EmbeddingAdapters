# embedding_adapters/loader.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from huggingface_hub import snapshot_download

# ----------------------------
# Paths: package, registry, cache
# ----------------------------

# Folder containing this file: .../embedding_adapters
PACKAGE_ROOT = Path(__file__).resolve().parent

# Project root: parent of embedding_adapters
PROJECT_ROOT = PACKAGE_ROOT.parent

# Registry location: embedding_adapters/data/registry.json
DEFAULT_REGISTRY_PATH = PACKAGE_ROOT / "data" / "registry.json"

# Cache root for downloaded adapters:
# By default: <project_root>/models
# Can be overridden with EMBEDDING_ADAPTERS_CACHE env var if desired.
_CACHE_ROOT = Path(
    os.getenv("EMBEDDING_ADAPTERS_CACHE", PROJECT_ROOT / "models")
).expanduser().resolve()


# ----------------------------
# Dataclasses
# ----------------------------

@dataclass
class AdapterEntry:
    slug: str
    source: str
    target: str
    flavor: str
    description: str
    version: str
    tags: List[str]
    mode: str  # informational: "local", "remote", etc.

    # primary / fallback / service are raw dicts mirroring registry.json
    primary: Dict[str, Any]
    fallback: Optional[Dict[str, Any]] = None
    service: Optional[Dict[str, Any]] = None

    # Filled by ensure_local_adapter_dir / ensure_local_adapter_files
    local_dir: Optional[Path] = None


# ----------------------------
# Registry loading
# ----------------------------

def load_registry(path: Optional[str | os.PathLike] = None) -> List[AdapterEntry]:
    """Load registry.json and return a list of AdapterEntry.

    Registry shape (example):

    [
      {
        "slug": "...",
        "source": "intfloat/e5-base-v2",
        "target": "text-embedding-3-small",
        "flavor": "generic",
        "description": "...",
        "version": "0.0.1",
        "tags": [],
        "mode": "local",
        "primary": {
          "type": "huggingface",
          "repo_id": "TylerF/emb_adapter_e5-base-v2_to_text-embedding-3-small-v_0_1_fp16",
          "weights_file": "adapter.pt",
          "config_file": "adapter_config.json",
          "scoring_file": "adapter_quality_stats.npz"
        },
        "fallback": null,
        "service": null
      }
    ]
    """
    if path is None:
        env_path = os.getenv("EMBEDDING_ADAPTERS_REGISTRY")
        path = env_path if env_path else DEFAULT_REGISTRY_PATH

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Registry file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Registry JSON must be a list of adapter entries.")

    entries: List[AdapterEntry] = []
    for obj in raw:
        tags = obj.get("tags") or []
        if not isinstance(tags, list):
            tags = [str(tags)]

        entry = AdapterEntry(
            slug=obj["slug"],
            source=obj["source"],
            target=obj["target"],
            flavor=obj.get("flavor", "generic"),
            description=obj.get("description", ""),
            version=obj.get("version", "0.0.1"),
            tags=[str(t) for t in tags],
            mode=obj.get("mode", "local"),
            primary=obj.get("primary", {}) or {},
            fallback=obj.get("fallback"),
            service=obj.get("service"),
        )
        entries.append(entry)

    return entries


def list_adapter_entries() -> List[dict]:
    """User-facing view of registry (used by list_adapters())."""
    entries = load_registry()
    out: List[dict] = []
    for e in entries:
        out.append(
            {
                "slug": e.slug,
                "source": e.source,
                "target": e.target,
                "flavor": e.flavor,
                "description": e.description,
                "version": e.version,
                "tags": e.tags,
                "mode": e.mode,
                "primary_type": e.primary.get("type"),
            }
        )
    return out


def find_adapter(
    source: str,
    target: str,
    flavor: str = "generic",
    registry_path: Optional[str | os.PathLike] = None,
) -> AdapterEntry:
    """Find a single adapter matching (source, target, flavor)."""
    entries = load_registry(registry_path)

    matches: List[AdapterEntry] = []
    for e in entries:
        if e.source != source:
            continue
        if e.target != target:
            continue
        if flavor is not None and e.flavor != flavor:
            continue
        matches.append(e)

    if not matches:
        raise LookupError(
            f"No adapter found for source={source!r}, target={target!r}, flavor={flavor!r}"
        )

    if len(matches) > 1:
        # You can make this smarter (pick highest version, etc.)
        print(
            f"[embedding_adapters] Warning: multiple adapters match "
            f"(source={source!r}, target={target!r}, flavor={flavor!r}); "
            f"using slug={matches[0].slug}"
        )

    return matches[0]


# ----------------------------
# Local materialization
# ----------------------------

def ensure_local_adapter_dir(entry: AdapterEntry, hf_token:str) -> Path:
    """Ensure the adapter exists as a local directory and return it.

    Uses primary['type']:

      - 'local_path'       → use primary['local_path'] or ['path']
      - 'huggingface'      → download from HF into cache
      - 'huggingface_encrypted' → same, but decrypt_helper will handle weights

    The returned directory will contain at least the files referenced by:
      primary['config_file'], primary['weights_file'] (or encrypted file).
    """
    # If we've already materialized a local directory, reuse it.
    if entry.local_dir is not None and entry.local_dir.exists():
        return entry.local_dir

    primary_type = (entry.primary or {}).get("type")

    # Ensure cache root exists (this is your <project_root>/models)
    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    # ----- local_path mode -----
    if primary_type == "local_path":
        path_str = entry.primary.get("local_path") or entry.primary.get("path")
        if not path_str:
            raise ValueError(
                f"Adapter '{entry.slug}' primary.type='local_path' but no 'local_path'/'path' set."
            )
        p = Path(path_str).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Local adapter path does not exist: {p}")
        entry.local_dir = p
        return p

    # ----- Hugging Face modes -----
    if primary_type in {"huggingface", "huggingface_encrypted"}:
        repo_id = entry.primary.get("repo_id")
        if not repo_id:
            raise ValueError(
                f"Adapter '{entry.slug}' primary.type='{primary_type}' but no 'repo_id' set."
            )

        # Download under <project_root>/models/<slug>
        cache_dir = _CACHE_ROOT / entry.slug
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[embedding_adapters] Downloading adapter '{entry.slug}' "
            f"from repo '{repo_id}' to: {cache_dir}"
        )

        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
            token=hf_token,  # <-- important for private/gated repos
        )

        entry.local_dir = cache_dir
        return cache_dir

    # Fallback: if type is missing but mode is 'local' and primary has 'path'
    if primary_type is None and entry.mode == "local" and "path" in (entry.primary or {}):
        path_str = entry.primary["path"]
        p = Path(path_str).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Local adapter path does not exist: {p}")
        entry.local_dir = p
        return p

    raise NotImplementedError(
        f"ensure_local_adapter_dir: unsupported primary.type={primary_type!r} "
        f"for entry '{entry.slug}' (mode={entry.mode!r})"
    )


def ensure_local_adapter_files(entry: AdapterEntry, hf_token:str) -> Path:
    """Ensure adapter files are present locally and return the directory.

    This is what decrypt_if_needed() calls. It guarantees that after returning,
    the directory contains at least:

      - config_file (e.g. adapter_config.json)
      - weights_file (may be encrypted in the encrypted case)
      - scoring_file is optional
    """
    target_dir = ensure_local_adapter_dir(entry, hf_token)

    primary = entry.primary or {}
    cfg_name = primary.get("config_file", "adapter_config.json")
    weights_name = primary.get("weights_file", "adapter.pt")
    stats_name = primary.get("scoring_file", "adapter_quality_stats.npz")

    cfg_path = target_dir / cfg_name
    weights_path = target_dir / weights_name

    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Adapter config not found for '{entry.slug}': {cfg_path}"
        )

    # For encrypted HF case, weights_name might be the encrypted file;
    # decrypt_if_needed will turn that into a plain adapter.pt.
    if not weights_path.exists():
        if primary.get("type") not in {"huggingface_encrypted"}:
            raise FileNotFoundError(
                f"Adapter weights not found for '{entry.slug}': {weights_path}"
            )

    # scoring file is optional; don't error if missing
    _ = target_dir / stats_name  # noqa: F841

    return target_dir
