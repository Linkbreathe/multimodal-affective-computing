"""Create a read-only inventory of the pre-migration artifact layout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ARTIFACTS / "legacy" / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    records = []
    for path in sorted(ARTIFACTS.rglob("*")):
        relative = path.relative_to(ARTIFACTS)
        if not path.is_file() or relative.parts[0] in {"runs", "legacy"}:
            continue
        records.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    payload = {
        "schema_version": "1.0.0",
        "scope": "legacy artifacts only; artifacts/runs is excluded",
        "files": records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} legacy records to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
