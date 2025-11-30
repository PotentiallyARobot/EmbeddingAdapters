import json
from pathlib import Path
from typing import Optional
from getpass import getpass

import requests

# Where we store the config locally
CONFIG_DIR = Path.home() / ".embedding_adapters"
CONFIG_PATH = CONFIG_DIR / "config.json"

# Base URL for your Worker API
API_BASE = "https://embeddingadapters-api.embedding-adapters.workers.dev"

# 🔗 Your Stripe Payment Link (test or live)
PAYMENT_URL = "https://buy.stripe.com/test_eVq28s7Kk4i737G5U8eUU01"  # replace with live later


def _save_and_confirm_key(api_key: str) -> bool:
    """
    Validate the key with the API, then save it to config.json if valid.
    Returns True if successful, False otherwise.
    """
    try:
        resp = requests.get(
            f"{API_BASE}/me",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
    except Exception as e:
        print(f"❌ Error talking to license server: {e}")
        return False

    if resp.status_code != 200:
        print(f"❌ Invalid API key (status {resp.status_code})")
        try:
            print(resp.text)
        except Exception:
            pass
        return False

    data = resp.json()
    email = data.get("email", "<unknown>")
    entitlements = data.get("entitlements", [])

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps({"api_key": api_key}, indent=2))

    print(f"\n✅ Logged in as {email}")
    print(f"   Entitlements: {', '.join(entitlements) or '(none)'}")
    print(f"   Config saved at: {CONFIG_PATH}")
    return True


def _load_saved_key() -> Optional[str]:
    """
    Load an API key from disk if it exists, otherwise return None.
    """
    if not CONFIG_PATH.exists():
        return None
    try:
        data = json.loads(CONFIG_PATH.read_text())
        key = data.get("api_key")
        if isinstance(key, str) and key.strip():
            return key.strip()
        return None
    except Exception:
        return None


def login() -> None:
    """
    CLI entrypoint for `embedding-adapters login`.

    Workflow:
      - If a saved key exists, verify it via /me.
      - If valid, done.
      - Otherwise, explain how to buy a key and prompt for a pasted key.
    """
    print("")
    print("Embedding Adapters Developer API login")
    print("--------------------------------------")

    # 1) If a key is already saved, try to use it
    saved_key = _load_saved_key()
    if saved_key:
        print("Found an existing API key in your config. Verifying...")
        if _save_and_confirm_key(saved_key):
            # Already logged in and confirmed; nothing else to do.
            return
        else:
            print("Saved key is invalid or revoked. You’ll need to paste a new one.\n")

    # 2) Explain the flow in a single, intuitive path
    print("To use the Embedding Adapters Developer API you need an API key.\n")
    print("If you already have a key (from a previous purchase or from an email),")
    print("paste it below when prompted.\n")
    print("If you don’t have one yet:")
    print("  1. Open this link in your browser:")
    print(f"     {PAYMENT_URL}")
    print("  2. Complete checkout.")
    print("  3. Your API key will be emailed to the email you used in the purchase.")
    print("  4. Come back here and paste the key when it arrives.\n")
    print("For support contact embeddingadapters@gmail.com\n")

    api_key = getpass(
        "Paste your Embedding Adapters API key (or leave blank to cancel): "
    ).strip()

    if not api_key:
        print("No key entered. You can re-run `embedding-adapters login` after you receive your key.")
        return

    _save_and_confirm_key(api_key)
