import io
import os
import base64
import json

import torch
from cryptography.fernet import Fernet

from ..loader import AdapterEntry, ensure_local_adapter_files

# -------------------------------------------------------------------
# Fernet key setup
# -------------------------------------------------------------------

_FERNET = None
RAW_KEY_HEX = "9c345b75c6459c41543731cd50a3ec04d9f660d760fcab8fd1bd15d089cd80e5"

if RAW_KEY_HEX:
    raw_key = bytes.fromhex(RAW_KEY_HEX)
    if len(raw_key) != 32:
        raise ValueError("ADAPTER_SECRET_KEY_HEX / RAW_KEY_HEX must decode to 32 bytes")
    fernet_key = base64.urlsafe_b64encode(raw_key)
    _FERNET = Fernet(fernet_key)

# Option 2: base64 Fernet key directly
FERNET_KEY_B64 = os.getenv("ADAPTER_FERNET_KEY_B64")
if FERNET_KEY_B64 and _FERNET is None:
    _FERNET = Fernet(FERNET_KEY_B64.encode("ascii"))


# -------------------------------------------------------------------
# Decrypt helper
# -------------------------------------------------------------------

def decrypt_if_needed(entry: AdapterEntry, device: str, hf_token:str):
    """
    For an encrypted HF model:
      - ensure files are downloaded
      - if weights are encrypted, decrypt to a local adapter.pt so
        your normal from_local loader can use it.

    Returns the directory containing adapter_config.json and adapter.pt.
    """
    target_dir = ensure_local_adapter_files(entry, hf_token)

    primary_type = entry.primary.get("type")
    cfg_name = entry.primary.get("config_file", "adapter_config.json")
    weights_name = entry.primary.get("weights_file", "adapter.pt")

    cfg_path = target_dir / cfg_name

    if primary_type == "huggingface_encrypted":
        if _FERNET is None:
            raise RuntimeError(
                "cryptography is required for encrypted adapters and the Fernet key is not initialized."
            )

        # Encrypted file as downloaded from HF (e.g. adapter_fp16.enc)
        enc_path = target_dir / weights_name

        # Decrypted file we will create (plain PyTorch weights)
        plain_pt_path = target_dir / "adapter.pt"

        if not enc_path.exists():
            raise FileNotFoundError(f"Encrypted weights not found: {enc_path}")

        # If we've already decrypted once, reuse
        if plain_pt_path.exists():
            print(f"[embedding_adapters] Using existing decrypted adapter at {plain_pt_path}")
        else:
            enc_bytes = enc_path.read_bytes()
            dec_bytes = _FERNET.decrypt(enc_bytes)

            # Optional: sanity check the decrypted bytes are a valid torch checkpoint
            try:
                _ = torch.load(io.BytesIO(dec_bytes), map_location="cpu")
            except Exception as e:
                raise RuntimeError(
                    f"Decrypted bytes are not a valid torch checkpoint. "
                    f"Check that the Fernet key matches the one used for encryption. "
                    f"Original error: {e}"
                ) from e

            # Persist as adapter.pt for normal from_local loading
            plain_pt_path.write_bytes(dec_bytes)
            print(f"[embedding_adapters] Decrypted adapter to {plain_pt_path}")

        # --- Patch adapter_config.json so from_local loads adapter.pt ---
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                cfg = {}

            # Many configs store this under "weights_file"; update it to adapter.pt
            if isinstance(cfg, dict):
                if cfg.get("weights_file") != "adapter.pt":
                    cfg["weights_file"] = "adapter.pt"
                    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
                    print(
                        f"[embedding_adapters] Updated {cfg_path.name} weights_file to 'adapter.pt'"
                    )

        return target_dir

    # Non-encrypted: assume already plain adapter.pt present or weights_name == adapter.pt
    return target_dir
