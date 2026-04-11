import json
import time
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
LICENSE_CACHE_PATH = CONFIG_DIR / "license_cache.json"

# Base URL for your Worker API (defaults, can be overridden by system_config.json)
API_BASE = "https://embeddingadapters-api.embedding-adapters.workers.dev"

# Stripe Payment Link (default, can be overridden)
PAYMENT_URL = "https://buy.stripe.com/eVq28s7Kk4i737G5U8eUU01"

# Default support email (can be overridden)
SUPPORT_EMAIL = "embeddingadapters@gmail.com"

# License cache TTL: 24 hours
LICENSE_CACHE_TTL = 86400

# Default login flow text block
DEFAULT_LOGIN_FLOW_TEXT = """
To use the Embedding Adapters Developer API you need an API key.

If you already have a key (from a previous purchase or from an email),
paste it below when prompted.

If you don't have one yet:
  1. Open this link in your browser:
     {payment_url}
  2. Complete checkout.
  3. Your API key will be emailed to the email you used in the purchase.
  4. Come back here and paste the key when it arrives.

For support contact {support_email}
"""

LOGIN_FLOW_TEXT = DEFAULT_LOGIN_FLOW_TEXT


def _load_remote_system_config() -> None:
    global API_BASE, PAYMENT_URL, SUPPORT_EMAIL, LOGIN_FLOW_TEXT

    try:
        resp = requests.get(SYSTEM_CONFIG_URL, timeout=5)
    except Exception:
        return

    if resp.status_code != 200:
        return

    try:
        data = resp.json()
    except Exception:
        return

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
        LOGIN_FLOW_TEXT = login_flow_text


# Apply any remote overrides at import time
_load_remote_system_config()


# ─── License Validation (DRM) ───────────────────────────────────────────────

def _load_license_cache() -> dict:
    """Load the license cache from disk."""
    if not LICENSE_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(LICENSE_CACHE_PATH.read_text())
    except Exception:
        return {}


def _save_license_cache(cache: dict) -> None:
    """Save the license cache to disk."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LICENSE_CACHE_PATH.write_text(json.dumps(cache, indent=2))


def validate_license(model_id: str, api_key: Optional[str] = None) -> bool:
    """
    Validate that the user has an active subscription for the given model.

    Checks:
      1. Local cache (valid for 24 hours)
      2. Remote API call to /v1/license/validate

    Returns True if licensed, raises RuntimeError if not.
    """
    if api_key is None:
        api_key = _load_saved_key()

    if not api_key:
        raise RuntimeError(
            f"No API key found. Run `embedding-adapters login` first.\n"
            f"The adapter '{model_id}' requires an active subscription ($10/month).\n"
            f"Subscribe at: {PAYMENT_URL}"
        )

    # 1. Check local cache
    cache = _load_license_cache()
    cache_key = f"{api_key[:8]}:{model_id}"
    cached = cache.get(cache_key)

    if cached and isinstance(cached, dict):
        expires_at = cached.get("expires_at", 0)
        if time.time() < expires_at and cached.get("valid", False):
            return True

    # 2. Call remote API
    try:
        resp = requests.post(
            f"{API_BASE}/v1/license/validate",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model_id": model_id},
            timeout=10,
        )
    except Exception as e:
        # Network error — allow if we have any cached result (even stale)
        if cached and cached.get("valid", False):
            print(
                f"[embedding_adapters] Warning: license server unreachable, "
                f"using cached license for '{model_id}'"
            )
            return True
        raise RuntimeError(
            f"Cannot validate license for '{model_id}': {e}\n"
            f"Check your internet connection or run in offline mode."
        ) from e

    if resp.status_code == 200:
        data = resp.json()
        if data.get("valid"):
            # Cache the valid license for 24 hours
            cache[cache_key] = {
                "valid": True,
                "expires_at": time.time() + LICENSE_CACHE_TTL,
                "subscription_id": data.get("subscription_id"),
                "model_id": model_id,
            }
            _save_license_cache(cache)
            return True

    # License invalid or expired
    error_msg = "Unknown error"
    try:
        data = resp.json()
        error_msg = data.get("error", {}).get("message", error_msg)
    except Exception:
        pass

    # Clear cached license
    if cache_key in cache:
        del cache[cache_key]
        _save_license_cache(cache)

    raise RuntimeError(
        f"License validation failed for '{model_id}': {error_msg}\n"
        f"Subscribe at the dashboard: https://embedding-adapters.com/dashboard.html\n"
        f"$10/month for the MiniLM → openai/text-embedding-3-large adapter."
    )


# ─── Existing auth functions ─────────────────────────────────────────────────

def _save_and_confirm_key(api_key: str) -> bool:
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
    print("")
    print("Embedding Adapters Developer API login")
    print("--------------------------------------")

    saved_key = _load_saved_key()
    if saved_key:
        print("Found an existing API key in your config. Verifying...")
        if _save_and_confirm_key(saved_key):
            return
        else:
            print("Saved key is invalid or revoked. You'll need to paste a new one.\n")

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
