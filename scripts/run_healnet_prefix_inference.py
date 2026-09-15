"""Run frozen three-seed HealNet inference on P009 causal prefixes and baseline."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import torch

from mac.adaptive.offline.baseline import extract_baseline_embeddings
from mac.adaptive.offline.healnet_prefix import EXPECTED_MODALITIES, FrozenHealNetEnsemble, file_sha256


EXTERNAL_ROOT = Path(
    "/mnt/c/Users/linki/amaster/data_collection_v3/analysis/real_time_inference/"
    "artifacts/cross_project_alignment_2026-07-16"
)
SESSION_DIR = Path(
    "/mnt/c/Users/linki/amaster/data_collection_v3/relaxdata/P009/P009_20260615_160828_449"
)
XDF_PATH = Path(
    "/mnt/c/Users/linki/amaster/data_collection_v3/relaxdata/P009/sub-P009/"
    "ses-S001/eeg/sub-P009_ses-S002_task-Default_run-001_eeg.xdf"
)
DEFAULT_OUTPUT = ROOT / "artifacts/healnet_adaptive_replay/p009_healnet_frozen_prefix_v1"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _check_split(split_manifest: Path) -> dict[str, Any]:
    frame = pd.read_csv(split_manifest, dtype=str)
    fold = frame.loc[pd.to_numeric(frame["fold_index"]).eq(5)].copy()
    if len(fold) != 9:
        raise ValueError("EEG-eligible split fold 05 must contain exactly nine participants")
    role = dict(zip(fold["participant_id"], fold["role"], strict=True))
    if role.get("P009") != "test" or role.get("P011") != "validation":
        raise ValueError("Fold 05 no longer has P009=test and P011=validation")
    return {
        "path": str(split_manifest.resolve()),
        "sha256": file_sha256(split_manifest),
        "fold_index": 5,
        "test_participant": "P009",
        "validation_participant": "P011",
        "train_participants": sorted(participant for participant, value in role.items() if value == "train"),
    }


def _load_or_extract_baseline(args: argparse.Namespace, windows_path: Path) -> dict[str, Any]:
    cache_path = args.output_dir / "baseline_embeddings.pt"
    if cache_path.is_file() and not args.force_baseline:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        required = {"windows", "embeddings", "masks", "metadata"}
        if required - set(payload):
            raise ValueError(f"Existing baseline cache is malformed: {cache_path}")
        return payload
    payload = extract_baseline_embeddings(
        events_path=args.session_dir / "events.csv",
        samples_path=args.session_dir / "samples.csv",
        eye_tracking_path=args.session_dir / "eye_tracking.csv",
        video_frames_path=args.session_dir / "video_frames.csv",
        session_dir=args.session_dir,
        xdf_path=args.xdf_path,
        formal_windows_path=windows_path,
        device=args.device,
    )
    torch.save(payload, cache_path)
    return payload


def _formal_predictions(
    ensemble: FrozenHealNetEnsemble,
    cache: dict[str, Any],
    windows: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    participant_windows = windows.loc[windows["participant_id"].astype(str).eq("P009")].copy()
    participant_windows = participant_windows.sort_values("window_start_unix_ms").reset_index(drop=True)
    if len(participant_windows) != 63:
        raise ValueError(f"P009 should have 63 formal windows; received {len(participant_windows)}")
    selected = participant_windows.iloc[:50].copy()
    sample_lookup = {
        (str(participant), str(condition)): index
        for index, (participant, condition) in enumerate(
            zip(cache["participant_ids"], cache["conditions"], strict=True)
        )
    }
    records = []
    prefix_audit = []
    for replay_index, row in selected.iterrows():
        key = ("P009", str(row["condition"]))
        if key not in sample_lookup:
            raise ValueError(f"Embedding cache lacks {key}")
        sample_index = sample_lookup[key]
        prefix_length = int(row["condition_window_index"]) + 1
        embeddings = {
            modality: torch.as_tensor(cache["embeddings"][modality][sample_index], dtype=torch.float32)
            for modality in EXPECTED_MODALITIES
        }
        masks = {
            modality: torch.as_tensor(cache["masks"][modality][sample_index], dtype=torch.bool)
            for modality in EXPECTED_MODALITIES
        }
        prediction = ensemble.predict_prefix(embeddings, masks, prefix_length)
        current_masks = {modality: bool(masks[modality][prefix_length - 1]) for modality in EXPECTED_MODALITIES}
        record = {
            "participant_id": "P009",
            "replay_index": int(replay_index),
            "window_id": str(row["window_id"]),
            "window_start_xdf": float(row["window_start_xdf"]),
            "window_end_xdf": float(row["window_end_xdf"]),
            "window_start_unix_ms": int(row["window_start_unix_ms"]),
            "window_end_unix_ms": int(row["window_end_unix_ms"]),
            "actual_condition": str(row["condition"]),
            "presentation_position": int(row["presentation_position"]),
            "condition_window_index": int(row["condition_window_index"]),
            "condition_window_count": int(row["condition_window_count"]),
            "prefix_length": prefix_length,
            "true_relaxation": float(row["relaxation"]),
            "true_discomfort": float(row["discomfort"]),
            "actual_intensity_index": int(row["intensity_index"]),
            "actual_frequency_index": int(row["frequency_index"]),
            "actual_intensity_value": float(row["intensity"]),
            "actual_frequency_value": float(row["frequency"]),
            "signal_valid": all(current_masks.values()),
            "modalities_used": "+".join(modality for modality, valid in current_masks.items() if valid),
            "missing_modalities": "+".join(modality for modality, valid in current_masks.items() if not valid),
            "model_version": "healnet_full_fold05_three_seed_frozen_causal_prefix_v1",
            "model_received_condition_or_rating": False,
            **{f"{modality}_valid": valid for modality, valid in current_masks.items()},
            **{
                f"{modality}_valid_prefix_windows": int(masks[modality][:prefix_length].sum())
                for modality in EXPECTED_MODALITIES
            },
            **prediction,
        }
        records.append(record)
        prefix_audit.append(
            {
                "window_id": record["window_id"],
                "condition": record["actual_condition"],
                "prefix_length": prefix_length,
                "maximum_embedding_index_read": prefix_length - 1,
                "future_embedding_indexes_read": [],
            }
        )
    output = pd.DataFrame(records)
    if len(output) != 50 or not output["signal_valid"].all():
        raise ValueError("Frozen first-50 window segment must have 50 common-valid windows")
    return output, {
        "rows": prefix_audit,
        "future_embedding_access_count": 0,
        "causal_prefix_contract": "Only indices 0..current condition_window_index are sliced before model forward.",
    }


def _baseline_predictions(
    ensemble: FrozenHealNetEnsemble,
    payload: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    windows = payload["windows"].reset_index(drop=True)
    embeddings = {modality: payload["embeddings"][modality] for modality in EXPECTED_MODALITIES}
    masks = {modality: payload["masks"][modality] for modality in EXPECTED_MODALITIES}
    records = []
    for index, row in windows.iterrows():
        prefix_length = index + 1
        current_masks = {modality: bool(masks[modality][index]) for modality in EXPECTED_MODALITIES}
        records.append(
            {
                "participant_id": "P009",
                "window_id": str(row["window_id"]),
                "baseline_window_index": index,
                "prefix_length": prefix_length,
                "window_start_unix_ms": int(row["window_start_unix_ms"]),
                "window_end_unix_ms": int(row["window_end_unix_ms"]),
                "signal_valid": all(current_masks.values()),
                **{f"{modality}_valid": value for modality, value in current_masks.items()},
                **ensemble.predict_prefix(embeddings, masks, prefix_length),
            }
        )
    frame = pd.DataFrame(records)
    usable = frame.loc[frame["signal_valid"]].copy()
    if len(usable) < 3:
        raise RuntimeError(
            f"Fewer than three all-modality-valid independent baseline windows are available ({len(usable)})"
        )
    relaxation0 = float(usable["pred_relaxation"].median())
    discomfort0 = float(usable["pred_discomfort"].median())
    epsilon_relaxation = float(np.percentile(np.abs(usable["pred_relaxation"] - relaxation0), 90))
    epsilon_discomfort = float(np.percentile(np.abs(usable["pred_discomfort"] - discomfort0), 90))
    calibration = {
        "source": payload["metadata"]["source"],
        "participant_id": "P009",
        "n_windows": int(len(usable)),
        "n_raw_non_overlapping_windows": int(len(frame)),
        "duration_seconds_used": float(len(usable) * 10),
        "baseline_event_duration_seconds": float(payload["metadata"]["baseline_event_duration_seconds"]),
        "relaxation_baseline_median": relaxation0,
        "discomfort_baseline_median": discomfort0,
        "epsilon_relaxation_p90_absolute_deviation": epsilon_relaxation,
        "epsilon_discomfort_p90_absolute_deviation": epsilon_discomfort,
        "median_seed_std_relaxation": float(usable["pred_relaxation_seed_std"].median()),
        "median_seed_std_discomfort": float(usable["pred_discomfort_seed_std"].median()),
        "formula": "epsilon = percentile_90(abs(causal_baseline_output - baseline_median))",
        "c1_safety_condition_baseline": False,
        "c1_formal_test_windows_used_for_calibration": False,
        "requested_physiological_baseline_seconds": 60,
        "requested_c1_safety_baseline_seconds": 120,
        "calibration_adequate_for_deployment": False,
        "limitation": (
            "Only four independent 10-second initial no-condition windows were available. "
            "The later C1 test condition was not used, preventing test leakage but leaving the C1 safety baseline absent."
        ),
        "all_modality_valid_windows": int(frame["signal_valid"].sum()),
        "embedding_extraction": payload["metadata"]["extraction"],
    }
    return frame, calibration


def _full_sequence_equivalence(
    predictions: pd.DataFrame,
    ensemble: FrozenHealNetEnsemble,
    tolerance: float = 5e-5,
) -> dict[str, Any]:
    completed = predictions.loc[
        predictions["prefix_length"].eq(predictions["condition_window_count"])
    ]
    comparisons = []
    for checkpoint in ensemble.records:
        seed = int(checkpoint["seed"])
        original_path = Path(checkpoint["path"]).parent.parent / f"eeg9_healnet_full_s{seed}_predictions.csv"
        original = pd.read_csv(original_path)
        original = original.loc[original["participant_id"].astype(str).eq("P009")].set_index("condition")
        for row in completed.itertuples(index=False):
            expected = original.loc[row.actual_condition]
            for target in ("relaxation", "discomfort"):
                actual_value = float(getattr(row, f"pred_{target}_seed_{seed}"))
                expected_value = float(expected[f"{target}_pred"])
                comparisons.append(
                    {
                        "seed": seed,
                        "condition": row.actual_condition,
                        "target": target,
                        "causal_prefix_full_value": actual_value,
                        "original_frozen_full_sequence_value": expected_value,
                        "absolute_difference": abs(actual_value - expected_value),
                    }
                )
    maximum = max(record["absolute_difference"] for record in comparisons)
    if maximum > tolerance:
        raise AssertionError(f"Prefix/full model equivalence failed: max difference {maximum} > {tolerance}")
    return {
        "n_comparisons": len(comparisons),
        "completed_conditions_in_segment": completed["actual_condition"].astype(str).tolist(),
        "absolute_tolerance": tolerance,
        "maximum_absolute_difference": maximum,
        "pass": True,
        "comparisons": comparisons,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.require_cuda and (not str(args.device).startswith("cuda") or not torch.cuda.is_available()):
        raise RuntimeError("--require-cuda was set but a CUDA device is not active")
    cache = torch.load(args.embedding_cache, map_location="cpu", weights_only=False)
    windows_path = args.windows or Path(cache["metadata"]["inputs"]["windows"]["path"])
    labels_path = args.labels or Path(cache["metadata"]["inputs"]["labels"]["path"])
    windows = pd.read_csv(windows_path)
    labels = pd.read_csv(labels_path)
    split = _check_split(args.split_manifest)

    baseline_payload = _load_or_extract_baseline(args, windows_path)
    ensemble = FrozenHealNetEnsemble(args.checkpoints, device=args.device)
    predictions, causal_audit = _formal_predictions(ensemble, cache, windows)
    baseline_predictions, calibration = _baseline_predictions(ensemble, baseline_payload)
    equivalence = _full_sequence_equivalence(predictions, ensemble)

    canonical_predictions = args.output_dir / "foundation_predictions_50windows.csv"
    detailed_predictions = args.output_dir / "healnet_prefix_predictions_50windows.csv"
    predictions.to_csv(canonical_predictions, index=False)
    shutil.copyfile(canonical_predictions, detailed_predictions)
    baseline_predictions.to_csv(args.output_dir / "baseline_predictions.csv", index=False)
    selected_ids = set(predictions["window_id"])
    window_manifest = windows.loc[windows["window_id"].astype(str).isin(selected_ids)].copy()
    window_manifest = window_manifest.sort_values("window_start_unix_ms")
    window_manifest.to_csv(args.output_dir / "replay_window_manifest.csv", index=False)

    created_at = datetime.now(timezone.utc).isoformat()
    participant_cache_indexes = [
        index for index, participant in enumerate(cache["participant_ids"]) if str(participant) == "P009"
    ]
    participant_masks = {
        modality: torch.as_tensor(cache["masks"][modality], dtype=torch.bool)[participant_cache_indexes]
        for modality in EXPECTED_MODALITIES
    }
    participant_selection = {
        "participant_id": "P009",
        "selection_record_created_at_utc": created_at,
        "selection_time": None,
        "selection_time_note": "The exact earlier freeze timestamp is unavailable; this run did not reselect a participant.",
        "selection_rule": "pre-existing quality-only frozen choice from the experiment plan",
        "selection_used_outcomes": False,
        "selection_recomputed_from_predictions": False,
        "modalities_available": list(EXPECTED_MODALITIES),
        "n_formal_windows": 63,
        "n_valid_common_windows": int(torch.stack(list(participant_masks.values())).all(dim=0).sum()),
        "labels_complete": bool(labels.loc[labels["participant_id"].eq("P009"), "condition"].nunique() == 9),
        "replacement_participant_used": False,
    }
    model_manifest = {
        "created_at_utc": created_at,
        "model": "HealNet full five-modality ensemble",
        "checkpoint_count": 3,
        "checkpoints": ensemble.records,
        "split": split,
        "inference_only": True,
        "training_performed": False,
        "p009_tuning_performed": False,
        "ensemble_rule": "arithmetic mean of three frozen-seed sigmoid predictions",
        "model_input_modalities": list(EXPECTED_MODALITIES),
        "condition_intensity_frequency_or_rating_used_as_model_input": False,
        "device": str(ensemble.device),
        "torch_version": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda) if torch.version.cuda else None,
        "embedding_cache": {
            "path": str(args.embedding_cache.resolve()),
            "sha256": file_sha256(args.embedding_cache),
        },
    }
    time_gaps = predictions["window_start_unix_ms"].diff().fillna(10_000) / 1000.0 - 10.0
    inference_audit = {
        "created_at_utc": created_at,
        "participant": "P009",
        "n_predictions": len(predictions),
        "observed_signal_seconds": len(predictions) * 10,
        "wall_clock_span_seconds": (
            int(predictions.iloc[-1]["window_end_unix_ms"])
            - int(predictions.iloc[0]["window_start_unix_ms"])
        ) / 1000.0,
        "large_gap_count": int((time_gaps > 5).sum()),
        "large_gap_seconds": [float(value) for value in time_gaps[time_gaps > 5]],
        "independent_condition_labels": int(predictions["actual_condition"].nunique()),
        "causal_prefix": causal_audit,
        "full_sequence_equivalence": equivalence,
        "model_input_fields": ["fold-scaled multimodal embeddings", "modality validity masks"],
        "forbidden_model_inputs": ["condition", "intensity", "frequency", "current/future ratings"],
        "future_window_embedding_access_count": 0,
        "baseline_precedes_first_formal_window": bool(
            baseline_payload["metadata"]["baseline_event_end_unix_ms"]
            < predictions.iloc[0]["window_start_unix_ms"]
        ),
        "baseline_embedding_cache": {
            "path": str((args.output_dir / "baseline_embeddings.pt").resolve()),
            "sha256": file_sha256(args.output_dir / "baseline_embeddings.pt"),
        },
        "input_files": {
            "windows": {"path": str(windows_path.resolve()), "sha256": file_sha256(windows_path)},
            "labels": {"path": str(labels_path.resolve()), "sha256": file_sha256(labels_path)},
            "split_manifest": {"path": str(args.split_manifest.resolve()), "sha256": file_sha256(args.split_manifest)},
            "events": {"path": str((args.session_dir / "events.csv").resolve()), "sha256": file_sha256(args.session_dir / "events.csv")},
        },
        "run_command": " ".join(sys.argv),
    }
    _write_json(args.output_dir / "participant_selection.json", participant_selection)
    _write_json(args.output_dir / "healnet_frozen_ensemble_manifest.json", model_manifest)
    _write_json(args.output_dir / "prefix_inference_audit.json", inference_audit)
    _write_json(args.output_dir / "calibration.json", calibration)
    result = {
        "output_dir": str(args.output_dir.resolve()),
        "predictions": str(canonical_predictions.resolve()),
        "n_predictions": len(predictions),
        "baseline_windows": len(baseline_predictions),
        "calibration": calibration,
        "full_sequence_equivalence_max_abs_difference": equivalence["maximum_absolute_difference"],
    }
    _write_json(args.output_dir / "prefix_stage_result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--embedding-cache", type=Path,
        default=ROOT / "artifacts/relax/aligned_20260716/condition_embeddings.pt",
    )
    parser.add_argument("--windows", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument(
        "--split-manifest", type=Path,
        default=EXTERNAL_ROOT / "eeg_eligible_ablation/contract/split_manifest.csv",
    )
    parser.add_argument(
        "--checkpoints", type=Path, nargs=3,
        default=[
            EXTERNAL_ROOT / f"eeg_eligible_ablation/runs/healnet/full/seed_{seed}/checkpoints/fold_05.pt"
            for seed in (20260705, 20260706, 20260707)
        ],
    )
    parser.add_argument("--session-dir", type=Path, default=SESSION_DIR)
    parser.add_argument("--xdf-path", type=Path, default=XDF_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--force-baseline", action="store_true")
    return parser.parse_args()


def main() -> int:
    result = run(parse_args())
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
