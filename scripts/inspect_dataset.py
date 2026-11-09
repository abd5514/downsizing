"""Print what actually landed in data/fl3d.

Written because the exact FL3D layout has to be confirmed before converting it,
rather than assumed. Run this once after downloading, then prepare_yolo.py.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path("data/fl3d")


def tree(path: Path, prefix: str = "", depth: int = 0, max_depth: int = 3) -> None:
    if depth > max_depth:
        return
    entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))[:12]
    for e in entries:
        if e.is_dir():
            n = sum(1 for _ in e.iterdir())
            print(f"{prefix}{e.name}/  ({n} entries)")
            tree(e, prefix + "  ", depth + 1, max_depth)
        else:
            print(f"{prefix}{e.name}  ({e.stat().st_size:,} bytes)")


def main() -> int:
    if not ROOT.exists():
        print(f"{ROOT} not found - run scripts/get_dataset.py first")
        return 1

    print(f"=== tree of {ROOT} ===")
    tree(ROOT)

    exts = Counter(p.suffix.lower() for p in ROOT.rglob("*") if p.is_file())
    print(f"\n=== file types ===\n{dict(exts)}")

    for manifest in list(ROOT.rglob("*.json"))[:3] + list(ROOT.rglob("*.csv"))[:3]:
        print(f"\n=== {manifest.relative_to(ROOT)} ===")
        if manifest.suffix == ".json":
            data = json.loads(manifest.read_text())
            if isinstance(data, dict):
                print("top-level keys:", list(data.keys())[:15])
                first = next(iter(data.values()))
                print("first value:", json.dumps(first, indent=2)[:600])
            elif isinstance(data, list):
                print(f"list of {len(data)}; first entry:")
                print(json.dumps(data[0], indent=2)[:600])
        else:
            lines = manifest.read_text().splitlines()[:4]
            print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
