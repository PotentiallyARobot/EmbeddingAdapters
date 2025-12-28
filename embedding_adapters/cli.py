# embedding_adapters/cli.py

import sys
import json
import argparse
import os
import time
import threading
from importlib import metadata as importlib_metadata
import urllib.request
import urllib.error

from .auth import login
from .loader import (
    load_registry,
    list_adapter_entries,
    find_adapter,
    AdapterEntry,
)

ANNOUNCEMENTS_URL = (
    "https://raw.githubusercontent.com/"
    "PotentiallyARobot/embedding-adapters-registry/main/announcements.txt"
)

SYSTEM_CONFIG_URL = (
    "https://raw.githubusercontent.com/"
    "PotentiallyARobot/embedding-adapters-registry/main/system_config.json"
)


def _get_version() -> str:
    """
    Try to get the package version from importlib.metadata,
    fall back to embedding_adapters.__version__ if present,
    or 'unknown' as a last resort.
    """
    try:
        return importlib_metadata.version("embedding-adapters")
    except Exception:
        try:
            from . import __version__  # type: ignore
            return __version__
        except Exception:
            return "unknown"


def _adapter_entry_to_dict(entry: AdapterEntry) -> dict:
    """
    Convert an AdapterEntry into a JSON-serializable dict that matches the
    registry.json structure as closely as possible.
    """
    return {
        "slug": getattr(entry, "slug", None),
        "source": getattr(entry, "source", None),
        "target": getattr(entry, "target", None),
        "flavor": getattr(entry, "flavor", None),
        "description": getattr(entry, "description", None),
        "version": getattr(entry, "version", None),
        "tags": list(getattr(entry, "tags", []) or []),
        "mode": getattr(entry, "mode", None),
        "primary": getattr(entry, "primary", {}) or {},
        "fallback": getattr(entry, "fallback", None),
        "service": getattr(entry, "service", None),
    }


# ---------------------------------------------------------------------
# small cross-platform spinner (stderr)
# ---------------------------------------------------------------------
def _start_spinner(message: str):
    """
    Start a tiny spinner that renders to stderr.
    Returns a stop() function you must call.

    Works on Windows/macOS/Linux terminals. If stderr is not a TTY,
    it becomes a no-op.
    """
    if not sys.stderr.isatty():
        def _noop():
            return None
        return _noop

    stop_event = threading.Event()
    chars = "|/-\\"

    def _run():
        i = 0
        while not stop_event.is_set():
            sys.stderr.write("\r" + message + " " + chars[i % len(chars)])
            sys.stderr.flush()
            i += 1
            time.sleep(0.12)
        sys.stderr.write("\r" + message + " ✓\n")
        sys.stderr.flush()

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    def _stop():
        stop_event.set()
        try:
            t.join(timeout=0.5)
        except Exception:
            pass

    return _stop


# ---------------------------------------------------------------------
# helpers for remote config
# ---------------------------------------------------------------------
def _fetch_text(url: str) -> str:
    """Fetch a text resource over HTTP and return it decoded as str."""
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            encoding = resp.headers.get_content_charset() or "utf-8"
            return resp.read().decode(encoding, errors="replace")
    except urllib.error.URLError as exc:
        print(f"Error: could not fetch {url}: {exc}")
        raise SystemExit(1)
    except Exception as exc:
        print(f"Unexpected error while fetching {url}: {exc}")
        raise SystemExit(1)


def _fetch_system_config() -> dict:
    """Fetch and parse system_config.json from the registry repo."""
    text = _fetch_text(SYSTEM_CONFIG_URL)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"Error: system_config.json is not valid JSON: {exc}")
        raise SystemExit(1)
    if not isinstance(data, dict):
        print("Error: system_config.json is not a JSON object.")
        raise SystemExit(1)
    return data


def _get_registry_add_url() -> str:
    """
    Resolve the URL to which we send registry-add requests.
    This is stored in system_config.json so it can be changed
    without shipping a new client.
    """
    config = _fetch_system_config()
    url = (
        config.get("REGISTRY_ADD_URL")
        or config.get("REGISTRY_ADD_ENDPOINT")
        or config.get("REGISTRY_ADD")
    )
    if not url:
        print(
            "Error: no REGISTRY_ADD_URL (or REGISTRY_ADD_ENDPOINT / REGISTRY_ADD) "
            "configured in system_config.json."
        )
        raise SystemExit(1)
    return url


def _post_json(url: str, payload: dict) -> tuple[int, str]:
    """
    Send a JSON POST to `url` and return (status_code, response_text).
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            status = resp.getcode()
            body = resp.read().decode(
                resp.headers.get_content_charset() or "utf-8",
                errors="replace",
            )
            return status, body
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = str(exc)
        print(f"Error: HTTP {exc.code} from {url}: {body}")
        raise SystemExit(1)
    except urllib.error.URLError as exc:
        print(f"Error: could not POST to {url}: {exc}")
        raise SystemExit(1)
    except Exception as exc:
        print(f"Unexpected error while POSTing to {url}: {exc}")
        raise SystemExit(1)


# ---------------------------------------------------------------------
# login
# ---------------------------------------------------------------------
def _cmd_login(_args: argparse.Namespace) -> None:
    """Handle `embedding-adapters login`."""
    login()


# ---------------------------------------------------------------------
# announcements
# ---------------------------------------------------------------------
def _cmd_announcements(_args: argparse.Namespace) -> None:
    """
    Handle `embedding-adapters news`.

    Fetch and display project-wide announcements from the central registry
    repository. This allows the project maintainers to broadcast important
    messages (breaking changes, deprecations, etc.) to all CLI users.
    """
    text = _fetch_text(ANNOUNCEMENTS_URL).strip()
    if not text:
        print("No announcements at this time.")
        return
    print(text)


# ---------------------------------------------------------------------
# donate
# ---------------------------------------------------------------------
def _cmd_donate(_args: argparse.Namespace) -> None:
    """
    Handle `embedding-adapters donate`.

    Fetch DONATE_URL from system_config.json and print it, so users can
    easily support the project financially.
    """
    config = _fetch_system_config()
    url = config.get("DONATE_URL")
    if not url:
        print("No DONATE_URL configured in system_config.json.")
        return
    print(url)


# ---------------------------------------------------------------------
# docs
# ---------------------------------------------------------------------
_DOC_KEYS = [
    "DOCS_URL",
    "DOCUMENTATION_URL",
    "DOC_URL",
]


def _cmd_docs(_args: argparse.Namespace) -> None:
    """
    Handle `embedding-adapters docs`.

    Fetch a documentation URL from system_config.json and print it.
    """
    config = _fetch_system_config()

    url = None
    for key in _DOC_KEYS:
        value = config.get(key)
        if value:
            url = value
            break

    if not url:
        print("No documentation URL configured in system_config.json.")
        return

    print(url)


# ---------------------------------------------------------------------
# add (propose new registry entry -> worker -> PR)
# ---------------------------------------------------------------------
def _cmd_add(args: argparse.Namespace) -> None:
    """
    Handle `embedding-adapters add`.

    This command proposes a new registry entry by sending it to the
    remote worker, which will validate it and open a PR against the
    central registry repo.
    """
    path = args.file
    email = args.email

    if not email:
        print("Error: --email is required.")
        raise SystemExit(1)

    # Load the JSON payload from the given file (or stdin if "-")
    try:
        if path == "-":
            raw_text = sys.stdin.read()
            entry = json.loads(raw_text)
        else:
            with open(path, "r", encoding="utf-8") as f:
                entry = json.load(f)
    except FileNotFoundError:
        print(f"Error: file not found: {path}")
        raise SystemExit(1)
    except json.JSONDecodeError as exc:
        print(f"Error: {path} does not contain valid JSON: {exc}")
        raise SystemExit(1)
    except Exception as exc:
        print(f"Error: could not read {path}: {exc}")
        raise SystemExit(1)

    if not isinstance(entry, dict):
        print("Error: registry entry JSON must be a single JSON object.")
        raise SystemExit(1)

    # Build payload for the worker
    payload = {
        "entry": entry,
        "email": email,
        "client_version": _get_version(),
    }

    url = _get_registry_add_url()
    status, body = _post_json(url, payload)

    # Try to parse JSON response; if that fails, just print the body
    try:
        resp_json = json.loads(body)
        print(json.dumps(resp_json, indent=2, sort_keys=False))
    except Exception:
        print(body)

    if status < 200 or status >= 300:
        raise SystemExit(1)


# ---------------------------------------------------------------------
# embed (transformers + torch, with optional per-text quality score)
# ---------------------------------------------------------------------
def _mean_pool(last_hidden_state, attention_mask):
    # last_hidden_state: (B, T, H), attention_mask: (B, T)
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _l2_normalize(x, eps=1e-12):
    return x / x.norm(dim=1, keepdim=True).clamp(min=eps)


def _extract_confidence(scores):
    """
    Best-effort extraction of a per-text "confidence" array/list from score_source().
    Supports:
      - np.ndarray / list of floats
      - dict with 'confidence' key
      - object with .confidence attribute
    Returns: list[float] | None
    """
    try:
        if isinstance(scores, dict) and "confidence" in scores:
            conf = scores["confidence"]
        elif hasattr(scores, "confidence"):
            conf = scores.confidence
        else:
            conf = scores

        if conf is None:
            return None

        if hasattr(conf, "tolist"):
            conf_list = conf.tolist()
        else:
            conf_list = list(conf)

        return [float(x) for x in conf_list]
    except Exception:
        return None


def _cmd_embed(args: argparse.Namespace) -> None:
    """
    Handle `embedding-adapters embed`.

    - Encodes text with a source HF model via transformers (no sentence-transformers).
    - Translates embeddings using a registry adapter.
    - Optionally computes per-text quality/confidence score using adapter.score_source()
      if quality stats are available for the adapter.

    Input:
      - --text (repeatable)
      - OR stdin lines (if not a TTY)

    Output:
      - JSON by default (includes embeddings + confidence if available)
      - NDJSON with --ndjson (each row includes embedding + confidence if available)
    """
    try:
        import torch  # type: ignore
        import numpy as np  # noqa: F401  # type: ignore
        from transformers import AutoTokenizer, AutoModel  # type: ignore
    except Exception as exc:
        print("Error: missing required dependency for `embed`.")
        print("This command requires: torch, numpy, transformers")
        print(f"Details: {exc}")
        raise SystemExit(1)

    from embedding_adapters import EmbeddingAdapter  # local import

    source = args.source
    target = args.target
    flavor = args.flavor

    # texts: from --text (repeatable) and/or stdin (lines)
    texts: list[str] = list(args.text or [])
    if not sys.stdin.isatty():
        stdin_blob = sys.stdin.read()
        texts.extend([ln.strip() for ln in stdin_blob.splitlines() if ln.strip()])

    if not texts:
        print("Error: provide at least one --text, or pipe text on stdin.")
        raise SystemExit(1)

    # Cache dir (optional): set HF cache vars (both transformers & hf_hub respect HF_HOME)
    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        os.environ["TRANSFORMERS_CACHE"] = args.cache_dir

    # device selection
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.verbose:
        cache_home = os.environ.get("HF_HOME") or "~/.cache/huggingface (default)"
        print("ℹ️ First run may download model weights; later runs use cache.", file=sys.stderr)
        print(f"HF cache: {cache_home}", file=sys.stderr)

    # Load tokenizer/model
    if args.verbose:
        print(f"Loading tokenizer: {source}", file=sys.stderr)
    stop = _start_spinner("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(source)
    stop()

    if args.verbose:
        print(f"Loading model: {source} (device={device})", file=sys.stderr)
    stop = _start_spinner("Loading model")
    model = AutoModel.from_pretrained(source).to(device)
    model.eval()
    stop()

    # Load adapter
    if args.verbose:
        print(f"Loading adapter: source={source}, target={target}, flavor={flavor}", file=sys.stderr)
    stop = _start_spinner("Loading adapter")
    adapter = EmbeddingAdapter.from_registry(
        source=source,
        target=target,
        flavor=flavor,
        device=device,
    )
    stop()

    # Encode in batches
    if args.verbose:
        print(f"Encoding {len(texts)} texts (batch_size={args.batch_size})...", file=sys.stderr)

    all_embs = []
    t0 = time.time()

    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            batch_texts = texts[i:i + args.batch_size]
            batch = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(**batch)

            if args.pooling == "cls":
                embs = out.last_hidden_state[:, 0, :]
            else:
                embs = _mean_pool(out.last_hidden_state, batch["attention_mask"])

            if not args.no_normalize:
                embs = _l2_normalize(embs)

            all_embs.append(embs.detach().cpu())

    src_embs_torch = torch.cat(all_embs, dim=0)
    src_embs = src_embs_torch.numpy()

    # Translate embeddings
    if args.verbose:
        print("Translating embeddings with adapter...", file=sys.stderr)
    stop = _start_spinner("Translating embeddings")
    translated = adapter.encode_embeddings(src_embs)
    stop()

    # Quality scoring (optional; default ON, can be skipped with --no-quality)
    confidence = None
    if not args.no_quality:
        try:
            if args.verbose:
                print("Scoring quality/confidence (if available)...", file=sys.stderr)
            stop = _start_spinner("Scoring quality")
            scores = adapter.score_source(src_embs)
            stop()
            confidence = _extract_confidence(scores)
            if confidence is None and args.verbose:
                print("⚠️ Could not extract 'confidence' from score_source() output.", file=sys.stderr)
        except Exception as exc:
            confidence = None
            if args.verbose:
                print(f"⚠️ Quality scoring unavailable: {exc}", file=sys.stderr)
    else:
        if args.verbose:
            print("Skipping quality scoring (--no-quality).", file=sys.stderr)

    elapsed_ms = (time.time() - t0) * 1000.0

    translated_list = translated.tolist() if hasattr(translated, "tolist") else translated

    if args.ndjson:
        for idx, txt in enumerate(texts):
            row = {
                "text": txt,
                "embedding": translated_list[idx],
                "source": source,
                "target": target,
                "flavor": flavor,
                "device": device,
            }
            if confidence is not None and idx < len(confidence):
                row["confidence"] = confidence[idx]
            print(json.dumps(row))
        return

    out_json = {
        "source": source,
        "target": target,
        "flavor": flavor,
        "device": device,
        "pooling": args.pooling,
        "max_length": args.max_length,
        "count": len(texts),
        "elapsed_ms": elapsed_ms,
        "embeddings": translated_list,
    }
    if confidence is not None:
        out_json["confidence"] = confidence

    print(json.dumps(out_json, indent=2, sort_keys=False))


# ---------------------------------------------------------------------
# list (human-friendly summary)
# ---------------------------------------------------------------------
def _cmd_list(args: argparse.Namespace) -> None:
    """
    Handle `embedding-adapters list`.

    Show a human-friendly summary grouped by (source -> target).
    """
    registry = load_registry() or []

    if registry and not isinstance(registry[0], dict):
        entries = [_adapter_entry_to_dict(e) for e in registry]
    else:
        entries = list(registry)

    if args.pro_only:
        entries = [e for e in entries if "pro" in (e.get("tags") or [])]

    if not entries:
        msg = "No adapters found in the registry."
        if args.pro_only:
            msg = "No pro adapters found in the registry."
        print(msg)
        return

    grouped = {}
    for e in entries:
        source = e.get("source")
        target = e.get("target")
        slug = e.get("slug")
        flavor = e.get("flavor", "generic")
        tags = e.get("tags") or []
        pro = "pro" in tags

        key = (source, target)
        if key not in grouped:
            grouped[key] = {"source": source, "target": target, "count": 0, "adapters": []}

        grouped[key]["count"] += 1
        grouped[key]["adapters"].append({"slug": slug, "flavor": flavor, "pro": pro})

    summary_list = list(grouped.values())
    summary_list.sort(key=lambda p: (p["source"] or "", p["target"] or ""))

    print(json.dumps(summary_list, indent=2, sort_keys=False))


# ---------------------------------------------------------------------
# registry (raw registry.json)
# ---------------------------------------------------------------------
def _cmd_registry(_args: argparse.Namespace) -> None:
    """Handle `embedding-adapters registry`."""
    registry = load_registry() or []
    if registry and not isinstance(registry[0], dict):
        data = [_adapter_entry_to_dict(e) for e in registry]
    else:
        data = list(registry)
    print(json.dumps(data, indent=2, sort_keys=False))


# ---------------------------------------------------------------------
# info / paths / help / version
# ---------------------------------------------------------------------
def _cmd_info(args: argparse.Namespace) -> None:
    """Handle `embedding-adapters info <slug>`."""
    slug = args.slug
    try:
        entry = find_adapter(source=None, target=None, flavor=None, slug=slug)
    except Exception as exc:
        print(f"Error: could not find adapter with slug '{slug}': {exc}")
        raise SystemExit(1)

    data = _adapter_entry_to_dict(entry)
    print(json.dumps(data, indent=2, sort_keys=False))


def _cmd_paths(_args: argparse.Namespace) -> None:
    """Handle `embedding-adapters paths`."""
    registry = load_registry() or []

    if registry and not isinstance(registry[0], dict):
        entries = [_adapter_entry_to_dict(e) for e in registry]
    else:
        entries = list(registry)

    if not entries:
        print("No adapters found in the registry.")
        return

    paths: dict[tuple[str | None, str | None], dict] = {}

    for e in entries:
        source = e.get("source")
        target = e.get("target")
        slug = e.get("slug")

        key = (source, target)
        if key not in paths:
            paths[key] = {"source": source, "target": target, "count": 0, "slugs": []}

        paths[key]["count"] += 1
        if slug is not None:
            paths[key]["slugs"].append(slug)

    path_list = list(paths.values())
    path_list.sort(key=lambda p: (p["source"] or "", p["target"] or ""))

    print(json.dumps(path_list, indent=2, sort_keys=False))


def _cmd_help(args: argparse.Namespace) -> None:
    """Handle `embedding-adapters help`."""
    parser = getattr(args, "_parser", None)
    if parser is None:
        print("No help available.")
        return
    parser.print_help()


def _cmd_version(_args: argparse.Namespace) -> None:
    """Handle `embedding-adapters version`."""
    print(_get_version())


# ---------------------------------------------------------------------
# parser / main
# ---------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="embedding-adapters",
        description=(
            "Command-line tools for working with embedding adapters. "
            "Run `embedding-adapters login` to purchase an API key, and "
            "`embedding-adapters news` to check for project updates."
        )
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>", required=True)

    # login
    p_login = subparsers.add_parser(
        "login",
        help="Log in with your API key (This command prints a link to purchase this if you need to)",
    )
    p_login.set_defaults(func=_cmd_login)

    # announcements
    p_announcements = subparsers.add_parser("news", help="Show the latest project-wide announcements / news")
    p_announcements.set_defaults(func=_cmd_announcements)

    # donate
    p_donate = subparsers.add_parser("donate", help="Print a link where you can financially support this project")
    p_donate.set_defaults(func=_cmd_donate)

    # docs
    p_docs = subparsers.add_parser("docs", help="Print a link to the project documentation")
    p_docs.set_defaults(func=_cmd_docs)

    # add
    p_add = subparsers.add_parser(
        "add",
        help=("Propose a new adapter registry entry. Submit your JSON entry and email, this will be validated and a PR opened."),
    )
    p_add.add_argument("file", help=("Path to a JSON file containing a single registry entry (or '-' to read from stdin)."))
    p_add.add_argument("--email", required=True, help="Your contact email to attach to the registry proposal.")
    p_add.set_defaults(func=_cmd_add)

    # embed
    p_embed = subparsers.add_parser(
        "embed",
        help="Encode text with a source model, then translate embeddings into a target model space using a registry adapter",
    )
    p_embed.add_argument("--source", required=True, help="Source HF model id")
    p_embed.add_argument("--target", required=True, help="Target model space id")
    p_embed.add_argument("--flavor", default="generic", help="Adapter flavor")
    p_embed.add_argument("--text", action="append", help="Text to embed (repeatable). If omitted, reads stdin lines.")
    p_embed.add_argument("--device", help="cpu|cuda|mps (default: auto)")
    p_embed.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    p_embed.add_argument("--max-length", type=int, default=256, help="Tokenizer max_length (default: 256)")
    p_embed.add_argument("--pooling", choices=["mean", "cls"], default="mean", help="Pooling strategy (default: mean)")
    p_embed.add_argument("--no-normalize", action="store_true", help="Disable L2 normalization of source embeddings")
    p_embed.add_argument("--ndjson", action="store_true", help="Emit newline-delimited JSON (one record per text)")
    p_embed.add_argument("--cache-dir", help="Override Hugging Face cache dir (sets HF_HOME and TRANSFORMERS_CACHE)")
    p_embed.add_argument("--verbose", action="store_true", help="Print status messages to stderr")
    p_embed.add_argument(
        "--no-quality",
        action="store_true",
        help="Skip per-text quality/confidence scoring (faster)",
    )
    p_embed.set_defaults(func=_cmd_embed)

    # list
    p_list = subparsers.add_parser("list", help="List adapters in a summarized, grouped form")
    p_list.add_argument("--pro-only", action="store_true", help="Only show adapters tagged with 'pro'")
    p_list.set_defaults(func=_cmd_list)

    # registry
    p_registry = subparsers.add_parser("registry", help="Print the full registry JSON as-is")
    p_registry.set_defaults(func=_cmd_registry)

    # info
    p_info = subparsers.add_parser("info", help="Show detailed info for a single adapter by slug (JSON)")
    p_info.add_argument("slug", help="Adapter slug")
    p_info.set_defaults(func=_cmd_info)

    # paths
    p_paths = subparsers.add_parser("paths", help="Show unique (source -> target) adapter paths and counts")
    p_paths.set_defaults(func=_cmd_paths)

    # help
    p_help = subparsers.add_parser("help", help="Show this help message and exit")
    p_help.set_defaults(func=_cmd_help, _parser=parser)

    # version
    p_version = subparsers.add_parser("version", help="Show the embedding-adapters package version")
    p_version.set_defaults(func=_cmd_version)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:])

    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        raise SystemExit(1)
    func(args)
