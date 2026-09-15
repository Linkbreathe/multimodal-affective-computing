"""Fail-closed WSL preflight for the RQ2 cross-environment experiment.

The RQ2 hand-off is deliberately treated as an external, read-only contract.
This module validates that hand-off and the local saved representation cache
before any downstream model can be started.  It never regenerates labels,
folds, anchors, manifests, or encoder representations.

The downstream runner is intentionally gated behind :func:`preflight`.  A
missing or ambiguous key-bearing cache is an alignment failure, not a reason
to infer ordering from an array, filename, or directory listing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
SHARED_ROOT = Path("/mnt/c/Users/linki/Wei/Models/rq2_shared")
CACHE_ROOT = PROJECT_ROOT / "artifacts/relax/aligned_20260716"
RESULTS_ROOT = PROJECT_ROOT / "wsl_results"

REQUIRED_HANDOFF = (
    "contract/rq2_contract.json",
    "contract/rq2_contract.sha256",
    "contract/condition_manifest.csv",
    "contract/window_manifest.csv",
    "contract/folds.csv",
    "contract/condition_anchors.csv",
    "windows_results/windows_representation_index.csv",
    "windows_results/WINDOWS_DONE.json",
)

MODALITIES = ("eeg", "ecg", "eye", "head", "video")
EXPECTED_DIMENSIONS = {"eeg": 1024, "ecg": 1024, "eye": 128, "head": 18, "video": 768}
EXPECTED_OBSERVATIONS = 81
EXPECTED_WINDOWS = 545
EXPECTED_FOLDS = 9
EXPECTED_SEEDS = (20260705, 20260706, 20260707)
EXPECTED_FALLBACK = ("P004", "C6")
EXPECTED_CONTRACT_HASH = "9838ca9ce5e2f6105393f3b8d9116021e81b738adf2653dfb6f957c1dae15853"


class AlignmentError(RuntimeError):
    """Raised when the immutable hand-off or saved cache is not auditable."""


@dataclass
class AuditState:
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def check(self, name: str, passed: bool, detail: Any) -> None:
        self.checks[name] = {"passed": bool(passed), "detail": detail}
        if not passed:
            self.failures.append(f"{name}: {detail}")

    def require(self) -> None:
        if self.failures:
            raise AlignmentError("\n".join(self.failures))


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - message is user-facing
        raise AlignmentError(f"Cannot parse JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AlignmentError(f"Expected JSON object in {path}")
    return payload


def _normalise_path(value: str) -> str:
    value = value.replace("\\", "/")
    value = re.sub(r"^[A-Za-z]:/", "", value)
    return value.lstrip("./").lower()


def _checksum_file(path: Path) -> str:
    """Return the composite ``contract_hash`` from the hand-off manifest."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise AlignmentError(f"Empty checksum file: {path}")
    for line in text.splitlines():
        if line.strip().lower().startswith("contract_hash"):
            parts = line.split()
            if len(parts) == 2 and re.fullmatch(r"[a-fA-F0-9]{64}", parts[1]):
                return parts[1].lower()
    matches = re.findall(r"\b[a-fA-F0-9]{64}\b", text)
    if len(matches) == 1:
        return matches[0].lower()
    raise AlignmentError(f"{path} lacks a contract_hash declaration")


def _individual_checksums(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 2 or not re.fullmatch(r"[a-fA-F0-9]{64}", parts[1]):
            continue
        if parts[0].lower() != "contract_hash":
            values[Path(parts[0]).name] = parts[1].lower()
    if len(values) != 5:
        raise AlignmentError(f"Expected five individual contract hashes in {path}, found {len(values)}")
    return values


def _composite_contract_hash(contract_dir: Path, individual: Mapping[str, str]) -> str:
    lines = "".join(f"{name}\t{individual[name]}\n" for name in sorted(individual))
    return sha256(lines.encode("utf-8")).hexdigest()


def _walk_hash_declarations(value: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], str]]:
    """Collect hash-like fields from an arbitrary Windows_DONE schema."""
    found: list[tuple[tuple[str, ...], str]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower()
            child_path = path + (str(key),)
            if isinstance(child, str) and re.fullmatch(r"[a-fA-F0-9]{64}", child.strip()):
                if "hash" in key_text or "sha" in key_text:
                    found.append((child_path, child.lower()))
            found.extend(_walk_hash_declarations(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk_hash_declarations(child, path + (str(index),)))
    return found


def _declared_contract_hashes(done: Mapping[str, Any]) -> list[str]:
    hashes: list[str] = []
    for path, value in _walk_hash_declarations(done):
        joined = "/".join(path).lower()
        if "contract" in joined or "rq2" in joined:
            hashes.append(value)
    return sorted(set(hashes))


def _first_column(frame: pd.DataFrame, names: Iterable[str], label: str) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise AlignmentError(f"{label} lacks one of required columns: {tuple(names)}")


def _key_frame(frame: pd.DataFrame, *, include_window: bool = False) -> pd.DataFrame:
    participant = _first_column(frame, ("participant", "participant_id"), "key frame")
    condition = _first_column(frame, ("condition", "condition_id"), "key frame")
    output = pd.DataFrame(
        {
            "participant": frame[participant].astype(str),
            "condition": frame[condition].astype(str),
        }
    )
    if include_window:
        window = _first_column(frame, ("window_id",), "window key frame")
        output["window_id"] = frame[window].astype(str)
    return output


def _key_set(frame: pd.DataFrame, *, include_window: bool = False) -> set[tuple[str, ...]]:
    keys = _key_frame(frame, include_window=include_window)
    return set(map(tuple, keys.to_numpy(dtype=str)))


def _load_table(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - message is user-facing
        raise AlignmentError(f"Cannot read CSV {path}: {exc}") from exc
    if frame.empty:
        raise AlignmentError(f"CSV is empty: {path}")
    return frame


def _check_contract_files(shared_root: Path, state: AuditState) -> dict[str, Path]:
    paths = {relative: shared_root / relative for relative in REQUIRED_HANDOFF}
    missing = [relative for relative, path in paths.items() if not path.is_file()]
    state.check("required_handoff_files", not missing, {"missing": missing})
    if missing:
        raise AlignmentError("Missing required Windows hand-off files: " + ", ".join(missing))
    return paths


def _check_contract_hashes(paths: Mapping[str, Path], state: AuditState) -> str:
    contract_dir = paths["contract/rq2_contract.json"].parent
    checksum_path = paths["contract/rq2_contract.sha256"]
    individual = _individual_checksums(checksum_path)
    observed_individual = {
        name: file_sha256(contract_dir / name) for name in individual
    }
    state.check(
        "contract_individual_sha256",
        observed_individual == individual,
        {"observed": observed_individual, "declared": individual},
    )
    observed = _composite_contract_hash(contract_dir, observed_individual)
    declared = _checksum_file(checksum_path)
    state.check(
        "rq2_contract_composite_hash",
        observed == declared,
        {"observed": observed, "declared": declared},
    )
    state.check(
        "rq2_contract_expected_hash",
        observed == EXPECTED_CONTRACT_HASH,
        {"observed": observed, "expected": EXPECTED_CONTRACT_HASH},
    )
    done = _read_json(paths["windows_results/WINDOWS_DONE.json"])
    done_hashes = [str(done.get("contract_hash", "")).lower()]
    state.check(
        "windows_done_contract_hash",
        observed in done_hashes,
        {"observed": observed, "declared_contract_hashes": done_hashes},
    )
    state.evidence["contract_hash"] = observed
    state.evidence["windows_done_hashes"] = done_hashes
    if observed_individual != individual or observed != declared or observed not in done_hashes:
        raise AlignmentError("Contract hash does not match rq2_contract.sha256 and WINDOWS_DONE.json")
    return observed


def _check_condition_manifest(path: Path, state: AuditState) -> tuple[pd.DataFrame, set[tuple[str, str]]]:
    frame = _load_table(path)
    keys = _key_frame(frame)
    duplicate = bool(keys.duplicated().any())
    key_set = set(map(tuple, keys.to_numpy(dtype=str)))
    counts = keys.groupby("participant").size()
    fallback_present = EXPECTED_FALLBACK in key_set
    state.check("condition_manifest_unique", not duplicate, int(keys.duplicated().sum()))
    state.check("condition_manifest_81_rows", len(keys) == EXPECTED_OBSERVATIONS, len(keys))
    state.check("condition_manifest_nine_per_participant", set(counts.astype(int)) == {9}, counts.to_dict())
    state.check("condition_manifest_p004_c6", fallback_present, EXPECTED_FALLBACK)
    if state.failures:
        raise AlignmentError("Condition manifest validation failed")
    return frame, key_set


def _check_window_manifest(path: Path, condition_keys: set[tuple[str, str]], state: AuditState) -> pd.DataFrame:
    frame = _load_table(path)
    keys = _key_frame(frame, include_window=True)
    duplicate = bool(keys.duplicated().any())
    pair_keys = set(map(tuple, keys[["participant", "condition"]].drop_duplicates().to_numpy(dtype=str)))
    unknown = sorted(pair_keys - condition_keys)
    window_ids = keys["window_id"]
    state.check("window_manifest_unique_keys", not duplicate, int(keys.duplicated().sum()))
    state.check("window_manifest_545_rows", len(keys) == EXPECTED_WINDOWS, len(keys))
    state.check("window_manifest_condition_membership", not unknown, unknown[:10])
    state.check("window_manifest_nonempty_window_ids", bool((window_ids.str.len() > 0).all()), "empty window_id")
    if state.failures:
        raise AlignmentError("Window manifest validation failed")
    return frame


def _check_folds(path: Path, participants: set[str], state: AuditState) -> None:
    frame = _load_table(path)
    fold = _first_column(frame, ("fold", "fold_index"), "fold manifest")
    participant = _first_column(frame, ("participant", "participant_id"), "fold manifest")
    role = _first_column(frame, ("role",), "fold manifest")
    groups = []
    failures: list[str] = []
    for fold_id, group in frame.groupby(fold, sort=True):
        observed_participants = set(group[participant].astype(str))
        roles = group[role].astype(str).str.lower().value_counts().to_dict()
        groups.append(str(fold_id))
        if observed_participants != participants:
            failures.append(f"fold {fold_id} participant membership")
        if roles.get("train", 0) != 7 or roles.get("validation", 0) != 1 or roles.get("test", 0) != 1:
            failures.append(f"fold {fold_id} role counts {roles}")
    state.check("fold_manifest_nine_7_1_1", len(groups) == EXPECTED_FOLDS and not failures, failures or groups)
    if failures or len(groups) != EXPECTED_FOLDS:
        raise AlignmentError("Fold manifest validation failed")


def _check_anchors(path: Path, condition_keys: set[tuple[str, str]], state: AuditState) -> None:
    frame = _load_table(path)
    required = {"fold_index", "condition", "target", "condition_anchor"}
    missing = sorted(required - set(frame.columns))
    keys = frame[["fold_index", "condition", "target"]] if not missing else pd.DataFrame()
    expected = {
        (fold, condition, target)
        for fold in range(1, EXPECTED_FOLDS + 1)
        for condition in sorted({condition for _, condition in condition_keys})
        for target in ("relaxation", "discomfort")
    }
    observed = set(map(tuple, keys.to_numpy())) if not missing else set()
    values = pd.to_numeric(frame["condition_anchor"], errors="coerce") if not missing else pd.Series(dtype=float)
    state.check("condition_anchors_schema", not missing, missing)
    state.check("condition_anchors_unique", len(observed) == len(frame), int(len(frame) - len(observed)) if not missing else 0)
    state.check("condition_anchors_coverage", observed == expected, {"missing": sorted(expected - observed)[:10], "extra": sorted(observed - expected)[:10]})
    state.check("condition_anchors_finite", bool(np.isfinite(values.to_numpy(dtype=float)).all()) if not missing else False, int(values.isna().sum()) if not missing else 0)
    if state.failures:
        raise AlignmentError("Condition anchor validation failed")


def _check_windows_representation_index(
    path: Path,
    condition_keys: set[tuple[str, str]],
    window_keys: set[tuple[str, str, str]],
    state: AuditState,
) -> pd.DataFrame:
    frame = _load_table(path)
    keys = _key_frame(frame)
    modality = _first_column(frame, ("modality",), "Windows representation index")
    family = _first_column(frame, ("representation_family",), "Windows representation index")
    fold = _first_column(frame, ("fold",), "Windows representation index")
    seed = _first_column(frame, ("seed",), "Windows representation index")
    modality_values = frame[modality].astype(str).str.lower()
    family_values = frame[family].astype(str).str.lower()
    group_keys = pd.DataFrame(
        {
            "family": family_values,
            "modality": modality_values,
            "fold": frame[fold].astype(str),
            "seed": frame[seed].astype(str),
            "participant": keys["participant"],
            "condition": keys["condition"],
        }
    )
    observed_keys = set(map(tuple, keys.to_numpy(dtype=str)))
    allowed_condition_keys = {key[:2] for key in window_keys} | {EXPECTED_FALLBACK}
    unknown = sorted(observed_keys - allowed_condition_keys)
    duplicates = int(group_keys.duplicated().sum())
    counts = group_keys.groupby(["family", "fold", "seed"]).size().to_dict()
    expected_groups = {("handcrafted", "global", "deterministic")} | {
        ("temporal_1dcnn", f"{fold_id:02d}", str(seed_id))
        for fold_id in range(1, EXPECTED_FOLDS + 1)
        for seed_id in EXPECTED_SEEDS
    }
    observed_groups = set(counts)
    state.check("windows_representation_index_key_membership", not unknown, unknown[:10])
    state.check("windows_representation_index_duplicates", duplicates == 0, duplicates)
    state.check("windows_representation_index_modalities", set(modality_values) == set(MODALITIES), counts)
    state.check("windows_representation_index_groups", observed_groups == expected_groups, {"missing": sorted(expected_groups - observed_groups), "extra": sorted(observed_groups - expected_groups)})
    state.check("windows_representation_index_group_rows", all(value == EXPECTED_OBSERVATIONS * len(MODALITIES) for value in counts.values()), counts)
    if "contract_hash" in frame:
        state.check("windows_representation_index_contract_hash", set(frame["contract_hash"].astype(str)) == {state.evidence["contract_hash"]}, frame["contract_hash"].astype(str).value_counts().to_dict())
    if "fallback_used" in frame:
        fallback = frame[frame["participant"].astype(str).eq(EXPECTED_FALLBACK[0]) & frame["condition"].astype(str).eq(EXPECTED_FALLBACK[1])]
        state.check("windows_representation_index_fallback", len(fallback) > 0 and (~fallback["representation_available"].astype(bool)).all(), len(fallback))
    state.evidence["windows_representation_groups"] = {str(key): int(value) for key, value in counts.items()}
    if state.failures:
        raise AlignmentError("Windows representation index validation failed")
    return frame


def _candidate_cache_indexes(cache_root: Path) -> list[Path]:
    patterns = (
        "*representation*index*.csv",
        "*window*index*.csv",
        "window_manifest.csv",
        "representation_index.csv",
    )
    found: set[Path] = set()
    for pattern in patterns:
        found.update(path for path in cache_root.rglob(pattern) if path.is_file())
    return sorted(found)


def _check_key_bearing_cache(cache_root: Path, window_keys: set[tuple[str, str, str]], state: AuditState) -> pd.DataFrame:
    indexes = _candidate_cache_indexes(cache_root)
    if not indexes:
        cache_path = cache_root / "condition_embeddings.pt"
        if not cache_path.is_file():
            detail = {
                "cache_root": str(cache_root),
                "required": "a local representation index or condition_embeddings.pt",
                "found": [],
            }
            state.check("wsl_cache_explicit_window_keys", False, detail)
            raise AlignmentError("No saved WSL representation cache was found")
        try:
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            state.check("wsl_cache_explicit_window_keys", False, {"cache_path": str(cache_path), "error": str(exc)})
            raise AlignmentError(f"Cannot load local condition cache {cache_path}: {exc}") from exc
        required = {"participant_ids", "conditions", "embeddings", "masks"}
        missing = sorted(required - set(cache))
        if missing:
            raise AlignmentError(f"Local condition cache lacks fields: {missing}")
        participants = [str(value) for value in cache["participant_ids"]]
        conditions = [str(value) for value in cache["conditions"]]
        pairs = list(zip(participants, conditions, strict=True))
        if len(pairs) != len(set(pairs)):
            raise AlignmentError("Local condition cache contains duplicate participant-condition keys")
        valid_keys = {key[:2] for key in window_keys}
        contract_pairs = sorted(valid_keys | {EXPECTED_FALLBACK})
        observed_pairs = set(pairs)
        missing_pairs = sorted(set(contract_pairs) - observed_pairs)
        if missing_pairs:
            raise AlignmentError(f"Local condition cache lacks contract keys: {missing_pairs[:10]}")
        rows: list[dict[str, Any]] = []
        for modality in MODALITIES:
            key = modality.lower()
            if key not in cache["embeddings"] or key not in cache["masks"]:
                raise AlignmentError(f"Local condition cache lacks modality {modality}")
            values = np.asarray(cache["embeddings"][key])
            masks = np.asarray(cache["masks"][key], dtype=bool)
            if values.ndim != 3 or masks.shape != values.shape[:2]:
                raise AlignmentError(f"Invalid local cache shape for {modality}: {values.shape}/{masks.shape}")
            if values.shape[2] != EXPECTED_DIMENSIONS[key]:
                raise AlignmentError(f"Unexpected local cache dimension for {modality}: {values.shape[2]}")
            if not np.isfinite(values[masks]).all():
                raise AlignmentError(f"Non-finite valid values in local cache for {modality}")
            for participant, condition in contract_pairs:
                index = pairs.index((participant, condition))
                row_window_keys = [key for key in window_keys if key[:2] == (participant, condition)]
                indices = [int(key[2].rsplit("_W", 1)[-1]) for key in row_window_keys]
                available = bool((masks[index, indices]).any()) if indices else False
                if (participant, condition) == EXPECTED_FALLBACK:
                    available = False
                rows.append({
                    "participant": participant,
                    "condition": condition,
                    "window_id": "|".join(sorted(key[2] for key in row_window_keys)),
                    "modality": modality,
                    "representation_dimension": int(values.shape[2]),
                    "representation_available": available,
                    "valid_window_count": int(masks[index, indices].sum()) if indices else 0,
                    "artifact_path": str(cache_path),
                })
        frame = pd.DataFrame(rows)
        state.check("wsl_cache_explicit_window_keys", True, {"derived_from": str(cache_path), "rows": len(frame)})
        state.check("wsl_cache_dimensions", set(frame["representation_dimension"]) == {EXPECTED_DIMENSIONS[m.lower()] for m in MODALITIES}, frame.groupby("modality").representation_dimension.unique().to_dict())
        state.check("wsl_cache_fallback", not frame.loc[(frame.participant == EXPECTED_FALLBACK[0]) & (frame.condition == EXPECTED_FALLBACK[1]), "representation_available"].any(), "P004/C6")
        state.evidence["wsl_representation_index"] = str(cache_path)
        return frame
    valid: list[tuple[Path, pd.DataFrame]] = []
    errors: list[str] = []
    for path in indexes:
        try:
            frame = _load_table(path)
            _key_frame(frame, include_window=True)
            _first_column(frame, ("modality",), "local representation index")
            valid.append((path, frame))
        except AlignmentError as exc:
            errors.append(f"{path}: {exc}")
    state.check("wsl_cache_explicit_window_keys", bool(valid), {"indexes": [str(p) for p, _ in valid], "errors": errors})
    if not valid:
        raise AlignmentError("No local representation index has the required explicit keys")
    if len(valid) > 1:
        state.warnings.append("Multiple local representation indexes found; using the shortest path")
    path, frame = min(valid, key=lambda item: len(str(item[0])))
    keys = _key_frame(frame, include_window=True)
    observed = set(map(tuple, keys.to_numpy(dtype=str)))
    unknown = sorted(observed - window_keys)
    state.check("wsl_cache_key_membership", not unknown, unknown[:10])
    if unknown:
        raise AlignmentError(f"Local representation index contains unknown window keys: {unknown[:10]}")
    state.evidence["wsl_representation_index"] = str(path)
    return frame


def preflight(
    shared_root: Path = SHARED_ROOT,
    cache_root: Path = CACHE_ROOT,
    state: AuditState | None = None,
) -> AuditState:
    """Validate the immutable hand-off and explicit local cache keys."""
    state = state or AuditState()
    paths = _check_contract_files(shared_root, state)
    contract_hash = _check_contract_hashes(paths, state)
    contract = _read_json(paths["contract/rq2_contract.json"])
    state.evidence["contract_schema_version"] = contract.get("schema_version")
    state.check("contract_hash_recorded", bool(contract_hash), contract_hash)

    condition_manifest, condition_keys = _check_condition_manifest(
        paths["contract/condition_manifest.csv"], state
    )
    window_manifest = _check_window_manifest(
        paths["contract/window_manifest.csv"], condition_keys, state
    )
    window_keys = _key_set(window_manifest, include_window=True)
    participants = set(_key_frame(condition_manifest)["participant"])
    _check_folds(paths["contract/folds.csv"], participants, state)
    _check_anchors(paths["contract/condition_anchors.csv"], condition_keys, state)
    windows_index = _check_windows_representation_index(
        paths["windows_results/windows_representation_index.csv"],
        condition_keys,
        window_keys,
        state,
    )
    local_index = _check_key_bearing_cache(cache_root, window_keys, state)

    state.evidence.update(
        {
            "condition_rows": int(len(condition_manifest)),
            "window_rows": int(len(window_manifest)),
            "window_representation_rows": int(len(windows_index)),
            "local_representation_rows": int(len(local_index)),
            "participants": sorted(participants),
            "seeds": list(EXPECTED_SEEDS),
            "fallback": {"participant": EXPECTED_FALLBACK[0], "condition": EXPECTED_FALLBACK[1]},
            "cache_root": str(cache_root),
        }
    )
    state.require()
    return state


def _format_error(shared_root: Path, cache_root: Path, error: Exception, state: AuditState | None) -> str:
    lines = [
        "# WSL Alignment Error",
        "",
        "The RQ2 WSL pipeline stopped before representation aggregation or model training.",
        "",
        f"- Shared hand-off: `{shared_root}`",
        f"- Local cache root: `{cache_root}`",
        f"- Error: `{error}`",
        "",
        "## Required behavior",
        "",
        "No contract file was regenerated, no encoder was run, no representation was imputed, and no downstream model was trained.",
        "",
        "## Checks",
        "",
    ]
    if state is None or not state.checks:
        lines.append("- preflight: FAILED")
    else:
        for name, record in state.checks.items():
            status = "PASS" if record["passed"] else "FAIL"
            lines.append(f"- `{name}`: **{status}** — `{record['detail']}`")
    if state and state.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in state.warnings)
    return "\n".join(lines) + "\n"


def write_alignment_error(output_root: Path, shared_root: Path, cache_root: Path, error: Exception, state: AuditState | None = None) -> Path:
    path = output_root / "WSL_ALIGNMENT_ERROR.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_format_error(shared_root, cache_root, error, state), encoding="utf-8")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, default=SHARED_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--audit-only", action="store_true", help="Run only the immutable preflight gate")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    state: AuditState | None = AuditState()
    try:
        state = preflight(args.shared_root, args.cache_root, state)
    except (AlignmentError, OSError, ValueError) as error:
        path = write_alignment_error(args.output_root, args.shared_root, args.cache_root, error, state)
        print(f"RQ2 WSL run stopped: {error}", file=sys.stderr)
        print(f"Wrote {path}", file=sys.stderr)
        return 2
    if args.audit_only:
        print(json.dumps({"status": "pass", "evidence": state.evidence}, indent=2, sort_keys=True))
        return 0
    try:
        from scripts.rq2_pipeline import run_pipeline

        payload = run_pipeline(args.shared_root, args.cache_root, args.output_root, PROJECT_ROOT / "combined", state)
        print(json.dumps({"status": "complete", "contract_hash": payload["contract_hash"], "outputs": len(payload["output_files"])}, indent=2, sort_keys=True))
        return 0
    except Exception as error:  # pragma: no cover - user-facing run boundary
        print(f"RQ2 WSL downstream run stopped: {error}", file=sys.stderr)
        print(f"Inspect {PROJECT_ROOT / 'combined' / 'SHARED_FUSION_ERROR.md'}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
