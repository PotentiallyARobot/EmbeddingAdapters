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
      - FIRST: call your Worker to check entitlement and get the Fernet key
               (if this fails, we abort before downloading anything)
      - THEN: ensure files are downloaded
      - decrypt encrypted weights (e.g. adapter.enc → in-memory bytes)
      - sanity-check with torch.load
      - attach state_dict to entry.primary["state_dict"]

    For free / non-encrypted adapters:
      - just ensure files exist, return the directory.

    Returns:
        Path to directory containing adapter_config.json and an attached
        in-memory state_dict for EmbeddingAdapter.from_local().
    """
    primary_type = (entry.primary or {}).get("type")
    tags = entry.tags or []
    is_pro = "pro" in tags

    # --- Non-pro OR non-encrypted: just download/use as-is ---
    if (not is_pro) or primary_type != "huggingface_encrypted":
        return ensure_local_adapter_files(entry, hf_token)

    # --- Pro + encrypted: check entitlement BEFORE downloading ---
    try:
        key_b64 = _fetch_decrypt_key_for_slug(entry.slug)
    except RuntimeError as e:
        from ..auth import PAYMENT_URL

        # At this point we have NOT downloaded anything from HF yet.
        # Surface a clear message to the user.
        raise RuntimeError(
            f"Access denied: The adapter '{entry.slug}' is a PRO model and your API key "
            "does not include entitlement for it.\n\n"
            "👉 To purchase PRO access, visit:\n"
            f"    {PAYMENT_URL}\n\n"
            "After purchasing, run:\n"
            "    embedding-adapters login\n"
            "and enter your API key when prompted.\n\n"
            "If you believe this is an error or need additional help, please contact:\n"
            "    embeddingadapters@gmail.com\n"
        ) from e

    # If we got here, the API key is entitled and we have a Fernet key.
    fernet = _build_fernet(key_b64)

    # Now we are allowed to download the encrypted weights from HF.
    target_dir = ensure_local_adapter_files(entry, hf_token)

    primary = entry.primary or {}
    enc_name = primary.get("weights_file", "adapter.enc")
    enc_path = target_dir / enc_name

    if not enc_path.exists():
        raise FileNotFoundError(
            f"Encrypted weights not found for pro adapter '{entry.slug}': {enc_path}"
        )

    enc_bytes = enc_path.read_bytes()

    try:
        dec_bytes = fernet.decrypt(enc_bytes)
    except InvalidToken as e:
        raise RuntimeError(
            f"Failed to decrypt adapter weights for '{entry.slug}'. "
            "This usually means the Worker is using a different Fernet key "
            "than the one used to encrypt the weights on Hugging Face."
        ) from e

    # Sanity check: make sure the decrypted bytes are a valid torch checkpoint
    try:
        state = torch.load(io.BytesIO(dec_bytes), map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(
            f"Decrypted bytes are not a valid torch checkpoint for '{entry.slug}'. "
            "Check that the encryption key used during upload matches the one "
            f"configured in your Worker. Original error: {e}"
        ) from e

    # Attach the in-memory state_dict so EmbeddingAdapter.from_local can use it
    if not isinstance(entry.primary, dict):
        entry.primary = dict(entry.primary or {})
    entry.primary["state_dict"] = state

    print(
        f"[embedding_adapters] Loaded encrypted adapter for '{entry.slug}'"
    )

    return target_dir
