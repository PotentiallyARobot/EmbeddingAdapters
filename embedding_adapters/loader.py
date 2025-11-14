import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


_LOCAL_REGISTRY = Path(__file__).with_name("registry.json")


@dataclass
class AdapterEntry:
    slug: str
    source: str
    target: str
    flavor: str = "generic"
    mode: str = "local"          # future: "service", "remote"
    primary: dict | None = None   # e.g. {"type": "local_path" / "huggingface" / "url", ...}
    fallback: dict | None = None
    service: dict | None = None   # e.g. {"base_url": "https://api...", "endpoint": "/..."}


def _ensure_registry_file():
    if not _LOCAL_REGISTRY.exists():
        with open(_LOCAL_REGISTRY, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)


def load_registry() -> List[AdapterEntry]:
    _ensure_registry_file()
    with open(_LOCAL_REGISTRY, "r", encoding="utf-8") as f:
        raw = json.load(f)
    entries = []
    for item in raw:
        entries.append(AdapterEntry(
            slug=item["slug"],
            source=item["source"],
            target=item["target"],
            flavor=item.get("flavor", "generic"),
            mode=item.get("mode", "local"),
            primary=item.get("primary"),
            fallback=item.get("fallback"),
            service=item.get("service"),
        ))
    return entries


def find_adapter(source: str, target: str, flavor: str = "generic") -> AdapterEntry:
    for entry in load_registry():
        if entry.source == source and entry.target == target and entry.flavor == flavor:
            return entry
    raise ValueError(f"No adapter registered for {source} → {target} (flavor={flavor})")


def list_adapter_entries() -> list[dict]:
    """Return registry as plain dicts (for printing / debugging / future CLI)."""
    _ensure_registry_file()
    with open(_LOCAL_REGISTRY, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw
