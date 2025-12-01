# embedding_adapters/cli.py

import sys
import json
import argparse
from importlib import metadata as importlib_metadata

from .auth import login
from .loader import (
    load_registry,
    list_adapter_entries,
    find_adapter,
    AdapterEntry,
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
# login
# ---------------------------------------------------------------------
def _cmd_login(_args: argparse.Namespace) -> None:
    """Handle `embedding-adapters login`."""
    login()


# ---------------------------------------------------------------------
# list (human-friendly summary)
# ---------------------------------------------------------------------
def _cmd_list(args: argparse.Namespace) -> None:
    """
    Handle `embedding-adapters list`.

    Show a human-friendly summary grouped by (source -> target):

      [
        {
          "source": "intfloat/e5-base-v2",
          "target": "openai/text-embedding-3-small",
          "count": 2,
          "adapters": [
            {"slug": "...", "flavor": "generic", "pro": true},
            {"slug": "...", "flavor": "large", "pro": false}
          ]
        },
        ...
      ]

    Use `embedding-adapters registry` to see the full raw registry JSON.
    """
    registry = load_registry() or []

    # If the loader returns AdapterEntry objects, convert them to dicts first
    if registry and not isinstance(registry[0], dict):
        entries = [_adapter_entry_to_dict(e) for e in registry]
    else:
        entries = list(registry)

    # Filter by pro-only if requested
    if args.pro_only:
        entries = [
            e for e in entries
            if "pro" in (e.get("tags") or [])
        ]

    if not entries:
        msg = "No adapters found in the registry."
        if args.pro_only:
            msg = "No pro adapters found in the registry."
        print(msg)
        return

    # Build a grouped summary: (source, target) -> {source, target, count, adapters}
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
            grouped[key] = {
                "source": source,
                "target": target,
                "count": 0,
                "adapters": [],
            }

        grouped[key]["count"] += 1
        grouped[key]["adapters"].append(
            {
                "slug": slug,
                "flavor": flavor,
                "pro": pro,
            }
        )

    summary_list = list(grouped.values())
    summary_list.sort(key=lambda p: (p["source"] or "", p["target"] or ""))

    print(json.dumps(summary_list, indent=2, sort_keys=False))


# ---------------------------------------------------------------------
# registry (raw registry.json)
# ---------------------------------------------------------------------
def _cmd_registry(_args: argparse.Namespace) -> None:
    """
    Handle `embedding-adapters registry`.

    Print the full registry JSON as-is (no summarization). The source is:
      - remote registry.json if load_registry is configured that way, or
      - the bundled embedding_adapters/data/registry.json otherwise.
    """
    registry = load_registry() or []

    # If load_registry returned AdapterEntry objects, convert to plain dicts
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
    """
    Handle `embedding-adapters paths`.

    Build paths directly from registry.json (via load_registry):

      [
        {
          "source": "intfloat/e5-base-v2",
          "target": "openai/text-embedding-3-small",
          "count": 2,
          "slugs": [
            "emb_adapter_e5-base-v2-to-openai_text_embedding_3_large_v2",
            "emb_adapter_e5-base-v2_to_text-embedding-3-small-v_0_1_fp16"
          ]
        },
        ...
      ]
    """
    registry = load_registry() or []

    # If load_registry returned AdapterEntry objects, convert them to plain dicts
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
            paths[key] = {
                "source": source,
                "target": target,
                "count": 0,
                "slugs": [],
            }

        paths[key]["count"] += 1
        if slug is not None:
            paths[key]["slugs"].append(slug)

    path_list = list(paths.values())
    path_list.sort(key=lambda p: (p["source"] or "", p["target"] or ""))

    print(json.dumps(path_list, indent=2, sort_keys=False))



def _cmd_help(args: argparse.Namespace) -> None:
    """Handle `embedding-adapters help` – print the main help."""
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
        description="CLI for managing and inspecting embedding adapters.",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        metavar="<command>",
        required=True,
    )

    # login
    p_login = subparsers.add_parser(
        "login",
        help="Log in with your API key (This command prints a link to purchase this if you need to)",
    )
    p_login.set_defaults(func=_cmd_login)

    # list (summary)
    p_list = subparsers.add_parser(
        "list",
        help="List adapters in a summarized, grouped form",
    )
    p_list.add_argument(
        "--pro-only",
        action="store_true",
        help="Only show adapters tagged with 'pro'",
    )
    p_list.set_defaults(func=_cmd_list)

    # registry (raw JSON)
    p_registry = subparsers.add_parser(
        "registry",
        help="Print the full registry JSON (remote/local registry.json) as-is",
    )
    p_registry.set_defaults(func=_cmd_registry)

    # info
    p_info = subparsers.add_parser(
        "info",
        help="Show detailed info for a single adapter by slug (JSON)",
    )
    p_info.add_argument(
        "slug",
        help="Adapter slug (see `embedding-adapters registry` or registry.json)",
    )
    p_info.set_defaults(func=_cmd_info)

    # paths
    p_paths = subparsers.add_parser(
        "paths",
        help="Show unique (source -> target) adapter paths and how many adapters each has",
    )
    p_paths.set_defaults(func=_cmd_paths)

    # help
    p_help = subparsers.add_parser(
        "help",
        help="Show this help message and exit",
    )
    p_help.set_defaults(func=_cmd_help, _parser=parser)

    # version
    p_version = subparsers.add_parser(
        "version",
        help="Show the embedding-adapters package version",
    )
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
