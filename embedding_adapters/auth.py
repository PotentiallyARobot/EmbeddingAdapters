import json
from pathlib import Path
from typing import Optional
from getpass import getpass

import requests

# Remote system config (will override defaults if available)
SYSTEM_CONFIG_URL = (
    "https://raw.githubusercontent.com/"
    "PotentiallyARobot/embedding-adapters-registry/main/system_config.json"
)

# Where we store the config locally
CONFIG_DIR = Path.home() / ".embedding_adapters"
CONFIG_PATH = CONFIG_DIR / "config.json"

# Base URL for your Worker API (defaults, can be overridden by system_config.json)
API_BASE = "https://embeddingadapters-api.embedding-adapters.workers.dev"

# 🔗 Your Stripe Payment Link (default, can be overridden)
PAYMENT_URL = "https://buy.stripe.com/eVq28s7Kk4i737G5U8eUU01"  # REAL

# Default support email (can be overridden)
SUPPORT_EMAIL = "embeddingadapters@gmail.com"

# Default login flow text block. Can be overridden by system_config.json.
# You can also let the config file redefine this completely.
DEFAULT_LOGIN_FLOW_TEXT = """
To use the Embedding Adapters Developer API you need an API key.

If you already have a key (from a previous purchase or from an email),
paste it below when prompted.

If you don’t have one yet:
  1. Open this link in your browser:
     {payment_url}
  2. Complete checkout.
  3. Your API key will be emailed to the email you used in the purchase.
  4. Come back here and paste the key when it arrives.

For support contact {support_email}
"""

# This will be used by login(). May be overridden by remote config.
LOGIN_FLOW_TEXT = DEFAULT_LOGIN_FLOW_TEXT


def _load_remote_system_config() -> None:
    """
    Try to pull down system_config.json from GitHub and override
    API_BASE, PAYMENT_URL, SUPPORT_EMAIL, and LOGIN_FLOW_TEXT if present.

    On any failure (network error, non-200, invalid JSON, missing keys),
    we silently fall back to the existing defaults.
    """
    global API_BASE, PAYMENT_URL, SUPPORT_EMAIL, LOGIN_FLOW_TEXT

    try:
        resp = requests.get(SYSTEM_CONFIG_URL, timeout=5)
    except Exception:
        # Network or DNS error – keep defaults
        return

    if resp.status_code != 200:
        # File missing or other HTTP error – keep defaults
        return

    try:
        data = resp.json()
    except Exception:
        # Invalid JSON – keep defaults
        return

    # Override only if keys exist and are non-empty strings
    api_base = data.get("API_BASE")
    if isinstance(api_base, str) and api_base.strip():
        API_BASE = api_base.strip()

    payment_url = data.get("PAYMENT_URL")
    if isinstance(payment_url, str) and payment_url.strip():
        PAYMENT_URL = payment_url.strip()

    support_email = data.get("SUPPORT_EMAIL")
    if isinstance(support_email, str) and support_email.strip():
        SUPPORT_EMAIL = support_email.strip()

    login_flow_text = data.get("LOGIN_FLOW_TEXT")
    if isinstance(login_flow_text, str) and login_flow_text.strip():
        # Allow full override of the text block
        LOGIN_FLOW_TEXT = login_flow_text


# Apply any remote overrides at import time
_load_remote_system_config()


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

    # 2) Single block of explanatory text (possibly overridden by remote config)
    # We allow the text to reference {payment_url} and {support_email}
    print(
        LOGIN_FLOW_TEXT.format(
            payment_url=PAYMENT_URL,
            support_email=SUPPORT_EMAIL,
        )
    )

    api_key = getpass(
        "Paste your Embedding Adapters API key (or leave blank to cancel): "
    ).strip()

    if not api_key:
        print(
            "No key entered. You can re-run `embedding-adapters login` after you "
            "receive your key."
        )
        return

    _save_and_confirm_key(api_key)
