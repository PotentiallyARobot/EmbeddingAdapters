# embedding_adapters/utils/decrypt_helper.py

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Optional

import torch
from cryptography.fernet import Fernet, InvalidToken
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from ..loader import AdapterEntry, ensure_local_adapter_files

# Where the CLI stores config (created by `embedding-adapters login`)
DEFAULT_CONFIG_PATH = Path(
    os.getenv("EMBEDDING_ADAPTERS_CONFIG", Path.home() / ".embedding_adapters" / "config.json")
)


def _load_cli_config() -> dict:
    try:
        with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _get_api_client_info() -> tuple[str, str]:
    """
    Returns (base_url, api_key) to talk to your Worker.

    Priority:
      1. EMBEDDING_ADAPTERS_API_BASE / EMBEDDING_ADAPTERS_API_KEY envs
      2. config.json written by `embedding-adapters login`
    """
    cfg = _load_cli_config()

    api_key = (
        os.getenv("EMBEDDING_ADAPTERS_API_KEY")
        or cfg.get("api_key")
    )
    base_url = (
        os.getenv("EMBEDDING_ADAPTERS_API_BASE")
        or cfg.get("api_base_url")
        or "https://embeddingadapters-api.embedding-adapters.workers.dev"
    )

    if not api_key:
        raise RuntimeError(
            "No Embedding Adapters API key found.\n"
            "Run `embedding-adapters login` after purchasing a key."
        )

    return base_url.rstrip("/"), api_key


def _fetch_decrypt_key_for_slug(slug: str) -> str:
    """
    Call your Worker to get the Fernet key for this slug.

    GET /adapters/{slug}/decrypt-key
      Authorization: Bearer <api_key>
    → { "key": "<base64-fernnet-key>" }
    """
    base_url, api_key = _get_api_client_info()
    url = f"{base_url}/adapters/{slug}/decrypt-key"

    req = Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )

    try:
        with urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            if resp.status != 200:
                raise RuntimeError(
                    f"decrypt-key failed for slug={slug!r}: HTTP {resp.status} {body}"
                )
    except (HTTPError, URLError) as e:
        raise RuntimeError(
            f"Error calling decrypt-key for slug={slug!r}: {e}"
        ) from e

    try:
        payload = json.loads(body)
    except Exception as e:
        raise RuntimeError(
            f"Could not parse decrypt-key JSON for slug={slug!r}: {e}. Body={body!r}"
        ) from e

    key_b64 = payload.get("key")
    if not key_b64:
        raise RuntimeError(
            f"decrypt-key response missing 'key' field for slug={slug!r}"
        )

    return key_b64


def _build_fernet(key_b64: str) -> Fernet:
    """
    key_b64 should be a standard Fernet key (urlsafe base64, 32 raw bytes).
    """
    try:
        # Fernet expects the key as base64-encoded bytes.
        return Fernet(key_b64.encode("ascii"))
    except Exception as e:
        raise RuntimeError(
            f"Invalid Fernet key: {e}. Expected a base64-encoded 32-byte key."
        ) from e


def decrypt_if_needed(entry: AdapterEntry, device: str, hf_token: str):
    """
    For pro + encrypted adapters:
      - ensure files are downloaded
      - fetch a per-slug key from your Worker
      - decrypt encrypted weights (e.g. adapter.enc → state_dict in memory)
      - stash state_dict on the AdapterEntry (entry.primary["state_dict"])

    For free / non-encrypted adapters:
      - just ensure files exist, return the directory.

    Returns:
        Path to directory containing adapter_config.json and the encrypted
        weights file; the decrypted weights live only in memory.
    """
    target_dir = ensure_local_adapter_files(entry, hf_token)

    primary_type = (entry.primary or {}).get("type")
    cfg_name = entry.primary.get("config_file", "adapter_config.json")
    weights_name = entry.primary.get("weights_file", "adapter.pt")

    cfg_path = target_dir / cfg_name

    tags = entry.tags or []
    is_pro = "pro" in tags

    # --- Non-pro or non-encrypted adapters: just use as-is ---
    if not is_pro or primary_type != "huggingface_encrypted":
        return target_dir

    # --- Pro + encrypted path ---
    enc_path = target_dir / weights_name  # e.g. adapter.enc

    if not enc_path.exists():
        raise FileNotFoundError(
            f"Encrypted weights not found for '{entry.slug}': {enc_path}"
        )

    # If we've already decrypted once in this process, reuse in-memory state_dict
    if "state_dict" in entry.primary:
        print(
            f"[embedding_adapters] Using existing in-memory decrypted state_dict "
            f"for '{entry.slug}'"
        )
        return target_dir

    # Get per-slug key from your Worker
    key_b64 = _fetch_decrypt_key_for_slug(entry.slug)
    fernet = _build_fernet(key_b64)

    enc_bytes = enc_path.read_bytes()
    try:
        dec_bytes = fernet.decrypt(enc_bytes)
    except InvalidToken as e:
        raise RuntimeError(
            f"Failed to decrypt adapter weights for '{entry.slug}'. "
            "Are you sure this API key is entitled to this adapter, "
            "and the Worker has the correct key configured for this slug?"
        ) from e

    # Load the checkpoint directly from in-memory bytes.
    # NOTE: weights_only=False because this is your trusted checkpoint.
    try:
        state = torch.load(
            io.BytesIO(dec_bytes),
            map_location=device,
            weights_only=False,
        )
    except Exception as e:
        raise RuntimeError(
            f"Decrypted bytes are not a valid torch checkpoint for '{entry.slug}'. "
            f"Check that the encryption key used during upload matches the one "
            f"configured in your Worker. Original error: {e}"
        ) from e

    # Stash state_dict in memory on the entry; from_local() will use this.
    entry.primary["state_dict"] = state
    print(
        f"[embedding_adapters] Decrypted adapter for '{entry.slug}' into memory "
        "(state_dict attached to entry.primary['state_dict'])"
    )

    # We no longer patch adapter_config.json or write adapter.pt to disk.
    # cfg_path is still used for metadata (in_dim, out_dim, etc.) by from_local().
    return target_dir
