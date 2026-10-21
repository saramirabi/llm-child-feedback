"""Load rendered items. Generation, coding, and analysis use the usable rows."""

from __future__ import annotations

import json
from pathlib import Path


def load_items(path: Path) -> list[dict]:
    items: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
    return items


def usable_items(items: list[dict]) -> list[dict]:
    """Rows later steps may use. Excluded items stay in the file and are skipped."""
    return [item for item in items if not item.get("exclude")]
