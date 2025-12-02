# embedding_adapters/loader.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import time

from huggingface_hub import snapshot_download
from .utils.version_helper import _get_current_package_version_str, _entry_supports_version
# ----------------------------
# Paths: package, registry, cache
# ----------------------------

# Folder containing this file: .../embedding_adapters
PACKAGE_ROOT = Path(__file__).resolve().parent

# Project root: parent of embedding_adapters
PROJECT_ROOT = PACKAGE_ROOT.parent

# Registry location: embedding_adapters/data/registry.json
DEFAULT_REGISTRY_PATH = PACKAGE_ROOT / "data" / "registry.json"

# Public raw URL for your registry.json
DEFAULT_REMOTE_REGISTRY_URL = os.getenv(
    "EMBEDDING_ADAPTERS_REMOTE_REGISTRY_URL",
    "https://raw.githubusercontent.com/PotentiallyARobot/embedding-adapters-registry/main/registry.json",
)

# Cache root for downloaded adapters:
# By default: <project_root>/data
_CACHE_ROOT = Path(
    os.getenv("EMBEDDING_ADAPTERS_CACHE", PROJECT_ROOT / "models")
).expanduser().resolve()

# Simple cache dir for the downloaded registry
_REGISTRY_CACHE_DIR = _CACHE_ROOT / "registry_cache"
_REGISTRY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

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
def _fetch_remote_registry(url: str) -> Optional[Path]:
    """
    Try to download registry.json from the given URL (public GitHub in your case)
    and cache it locally.

    Returns:
        Path to the cached file on success, or None on any failure.
    """
    if not url:
        return None

    headers = {
        "User-Agent": "embedding-adapters/1.0",
        "Accept": "application/json",
    }

    req = Request(url, headers=headers)

    try:
        print(f"[embedding_adapters] Fetching remote registry from {url}")
        with urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                print(
                    f"[embedding_adapters] Remote registry fetch failed with "
                    f"status {resp.status} from {url}"
                )
                return None
            data = resp.read()
    except (HTTPError, URLError) as e:
        print(
            f"[embedding_adapters] Warning: could not fetch remote registry from "
            f"{url}: {e}"
        )
        return None

    cache_path = _REGISTRY_CACHE_DIR / "registry.json"
    try:
        with open(cache_path, "wb") as f:
            f.write(data)
    except OSError as e:
        print(
            f"[embedding_adapters] Warning: failed to write cached registry: {e}"
        )
        return None

    print(
        f"[embedding_adapters] Successfully loaded remote registry and cached at {cache_path}"
    )
    return cache_path


def load_registry(path: Optional[str | os.PathLike] = None) -> List[AdapterEntry]:
    """Load registry.json and return a list of AdapterEntry, filtered by package version.

    Resolution order:

      1. If `path` is provided → use exactly that (no remote logic).
      2. Else if EMBEDDING_ADAPTERS_REGISTRY is set → use that local path.
      3. Else:
           - If EMBEDDING_ADAPTERS_DISABLE_REMOTE=1:
               → use bundled local registry only.
           - Else:
               → Prefer in order:
                    a) Fresh remote download (if not OFFLINE)
                    b) Cached remote registry.remote.json (if it exists)
                    c) Bundled local embedding_adapters/data/registry.json

    Version behavior:
      - Each entry's "version" field is treated as a version spec string.
      - Examples:
          "0.0.1"                → only visible to package version 0.0.1
          ">=0.2.0,<0.4.0"       → visible when 0.2.0 <= pkg_version < 0.4.0
          "" or missing          → visible for all versions
    """

    # 1) Determine which registry file to load (your existing logic)
    if path is not None:
        registry_path = Path(path)
    else:
        env_path = os.getenv("EMBEDDING_ADAPTERS_REGISTRY")
        if env_path:
            registry_path = Path(env_path)
        else:
            disable_remote = os.getenv("EMBEDDING_ADAPTERS_DISABLE_REMOTE") == "1"
            offline = os.getenv("EMBEDDING_ADAPTERS_OFFLINE") == "1"

            if disable_remote:
                print(
                    "[embedding_adapters] Remote registry disabled via "
                    "EMBEDDING_ADAPTERS_DISABLE_REMOTE=1; using bundled local registry."
                )
                registry_path = DEFAULT_REGISTRY_PATH
            else:
                remote_path: Optional[Path] = None

                if not offline and DEFAULT_REMOTE_REGISTRY_URL:
                    remote_path = _fetch_remote_registry(DEFAULT_REMOTE_REGISTRY_URL)

                if remote_path is not None:
                    registry_path = remote_path
                else:
                    if REMOTE_REGISTRY_PATH.exists():
                        if offline:
                            print(
                                "[embedding_adapters] Offline mode enabled "
                                "(EMBEDDING_ADAPTERS_OFFLINE=1); using cached remote "
                                f"registry at {REMOTE_REGISTRY_PATH}."
                            )
                        else:
                            print(
                                "[embedding_adapters] Warning: using cached remote "
                                f"registry at {REMOTE_REGISTRY_PATH} because fetching "
                                f"remote registry from {DEFAULT_REMOTE_REGISTRY_URL} failed."
                            )
                        registry_path = REMOTE_REGISTRY_PATH
                    else:
                        if DEFAULT_REMOTE_REGISTRY_URL and not offline:
                            print(
                                "[embedding_adapters] Warning: using bundled local "
                                f"registry at {DEFAULT_REGISTRY_PATH} because fetching "
                                f"remote registry from {DEFAULT_REMOTE_REGISTRY_URL} failed "
                                "and no cached remote registry is available."
                            )
                        elif offline:
                            print(
                                "[embedding_adapters] Offline mode enabled but no "
                                f"cached remote registry at {REMOTE_REGISTRY_PATH}; "
                                f"using bundled local registry at {DEFAULT_REGISTRY_PATH}."
                            )
                        registry_path = DEFAULT_REGISTRY_PATH

    registry_path = registry_path.expanduser().resolve()
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry file not found: {registry_path}")

    with open(registry_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Registry JSON must be a list of adapter entries.")

    current_version_str = _get_current_package_version_str()
    entries: List[AdapterEntry] = []

    for obj in raw:
        # Filter by version spec
        if not _entry_supports_version(obj, current_version_str):
            continue

        tags = obj.get("tags") or []
        if not isinstance(tags, list):
            tags = [str(tags)]

        entry = AdapterEntry(
            slug=obj["slug"],
            source=obj["source"],
            target=obj["target"],
            flavor=obj.get("flavor", "generic"),
            description=obj.get("description", ""),
            version=str(obj.get("version", "")) or ">=0.0.1",
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
    slug: Optional[str] = None,
) -> AdapterEntry:
    """Find a single adapter.

    Priority:
      - If `slug` is provided, return the entry with that slug (if found).
      - Otherwise, match on (source, target, flavor) as before.
    """
    entries = load_registry(registry_path)

    # --- 1. Prefer slug lookup if provided ---
    if slug is not None:
        slug_matches: List[AdapterEntry] = [e for e in entries if e.slug == slug]

        if not slug_matches:
            raise LookupError(f"No adapter found with slug={slug!r}")

        if len(slug_matches) > 1:
            print(
                f"[embedding_adapters] Warning: multiple adapters share slug={slug!r}; "
                f"using first match with source={slug_matches[0].source!r}, "
                f"target={slug_matches[0].target!r}, flavor={slug_matches[0].flavor!r}"
            )

        entry = slug_matches[0]

        # Optional sanity checks (non-fatal, just warn if user also passed source/target/flavor)
        if entry.source != source or entry.target != target or (
            flavor is not None and entry.flavor != flavor
        ):
            print(
                "[embedding_adapters] Warning: slug lookup returned an entry whose "
                f"(source={entry.source!r}, target={entry.target!r}, flavor={entry.flavor!r}) "
                f"does not exactly match requested "
                f"(source={source!r}, target={target!r}, flavor={flavor!r})."
            )

        return entry

    # --- 2. Original behavior: match by (source, target, flavor) ---
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

# Cache location for remote registry (under your existing models cache)
REGISTRY_CACHE_DIR = _CACHE_ROOT / "registry"
REGISTRY_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _download_registry_from_hf(
    repo_id: str,
    filename: str = "registry.json",
    *,
    repo_type: str = "dataset",         # or "model", configurable
    revision: Optional[str] = None,     # e.g. "main"
    token: Optional[str] = None,        # HF token if needed
) -> Path:
    """
    Download a registry.json file from a Hugging Face repo and return the
    local Path to it. Uses huggingface_hub.snapshot_download so that updates
    on the repo are reflected locally over time.

    Typical usage:
        _download_registry_from_hf("your-user/your-registry-repo")
    """
    # Put each repo in its own subdir
    safe_repo_name = repo_id.replace("/", "__")
    target_dir = REGISTRY_CACHE_DIR / safe_repo_name
    target_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[embedding_adapters] Downloading registry from HF repo='{repo_id}', "
        f"filename='{filename}', repo_type='{repo_type}', revision='{revision or 'latest'}'"
    )

    local_repo_dir = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        token=token,
    )

    registry_path = Path(local_repo_dir) / filename
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Downloaded HF repo '{repo_id}' but did not find '{filename}' in it. "
            f"Looked under: {registry_path}"
        )

    return registry_path
