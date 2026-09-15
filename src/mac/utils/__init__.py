"""Shared helpers.

``io`` holds the atomic-write and hashing helpers that were ``real_time_ml.utils``;
they are re-exported here so ``from mac.utils import write_json`` keeps working.
"""

from mac.utils.io import atomic_write_text, file_sha256, write_json, write_jsonl

__all__ = ["atomic_write_text", "file_sha256", "write_json", "write_jsonl"]
