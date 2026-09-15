#!/usr/bin/env python
"""Initialize the versioned corrected Relax cache from immutable legacy tensors."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.relax_foundation.extract_relax_foundation_embeddings import copy_non_eeg_window_cache  # noqa: E402
from src.data.relax_foundation import RELAX_EEG_RUN_TAG, RelaxHardFailure  # noqa: E402


DEFAULT_MODALITIES = ["ecg", "eye", "head", "video", "attention_video"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--modalities", nargs="+", default=DEFAULT_MODALITIES)
    parser.add_argument("--run-tag", default=RELAX_EEG_RUN_TAG)
    args = parser.parse_args()
    try:
        manifest = copy_non_eeg_window_cache(
            source_root=args.source,
            destination_root=args.destination,
            modalities=args.modalities,
            run_tag=args.run_tag,
        )
    except RelaxHardFailure as error:
        print(str(error), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "run_tag": manifest["run_tag"],
                "destination_root": manifest["destination_root"],
                "file_counts": {
                    name: payload["file_count"]
                    for name, payload in manifest["modalities"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
