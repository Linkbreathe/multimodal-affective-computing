"""Orchestration and recoverable per-participant extraction caches."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import traceback

import cv2
import numpy as np
import pandas as pd

from mac.data.video import VideoIndex, load_video_index
from .behavior import behavioral_features
from .common import feature_result, read_logged, verify_lock, write_csv, write_json
from .physiology import ecg_features, eeg_features, load_physiology, prepare_ecg
from .specification import FEATURES, PARTICIPANTS, file_hash, spec_hash
from .video import optical_flow_features


def source_integrity(output: Path) -> list[dict]:
    records = []
    for row in pd.read_csv(output / "input_files.csv").fillna("").to_dict("records"):
        path = Path(row["path"])
        actual = file_hash(path) if path.is_file() else ""
        records.append(
            dict(
                path=str(path),
                expected_sha256=row["sha256"],
                actual_sha256=actual,
                unchanged=actual == row["sha256"],
            )
        )
    return records


def _participant(output_string: str, pid: str) -> dict:
    cv2.setNumThreads(1)
    output = Path(output_string)
    verify_lock(output)
    cache = output / "cache" / pid
    cache.mkdir(parents=True, exist_ok=True)
    source = (
        pd.read_csv(output / "recording_sources.csv")
        .fillna("")
        .set_index("participant_id")
        .loc[pid]
        .to_dict()
    )
    conditions = pd.read_csv(output / "condition_manifest.csv").fillna({"alignment_reason": ""})
    conditions = conditions.loc[conditions.participant_id.eq(pid)].sort_values("condition")
    errors = []
    frames = {}
    for name, key in [("head", "samples_csv"), ("eye", "eye_tracking_csv")]:
        try:
            frames[name] = read_logged(Path(source[key])) if source[key] else pd.DataFrame()
        except Exception as error:
            frames[name] = pd.DataFrame()
            errors.append(dict(scope=name, reason=f"{type(error).__name__}: {error}"))
    try:
        video = load_video_index(
            Path(source["video_frames_csv"]) if source["video_frames_csv"] else None,
            Path(source["session_dir"]) if source["session_dir"] else None,
            pid,
        )
    except Exception as error:
        video = VideoIndex(pid, "none", "video_index_failed", ())
        errors.append(dict(scope="video", reason=f"{type(error).__name__}: {error}"))
    physio = None
    rr = []
    peaks = []
    spans = []
    info = {}
    ecg_error = ""
    try:
        physio = load_physiology(Path(source["xdf_path"]))
        samples, times, fs, edges, info = physio
        try:
            rr, peaks, spans = prepare_ecg(samples, times, fs, edges)
        except Exception as error:
            ecg_error = f"{type(error).__name__}: {error}"
            errors.append(dict(scope="ecg", reason=ecg_error, traceback=traceback.format_exc()))
    except Exception as error:
        errors.append(
            dict(
                scope="physio",
                reason=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
            )
        )
    all_results = []
    all_channels = []
    all_rr = []
    all_frames = []
    diagnostics = []
    for row in conditions.to_dict("records"):
        c = row["condition"]
        results = []
        alignment_good = row["alignment_status"] == "available"
        if physio is not None and alignment_good:
            try:
                eeg, channels = eeg_features(
                    samples, times, fs, edges, row["start_xdf"], row["end_xdf"]
                )
                results.extend(eeg)
                all_channels.extend(dict(participant_id=pid, condition=c, **r) for r in channels)
            except Exception as error:
                errors.append(
                    dict(
                        scope="eeg",
                        condition=c,
                        reason=f"{type(error).__name__}: {error}",
                        traceback=traceback.format_exc(),
                    )
                )
            if not ecg_error:
                try:
                    ecg, rr_audit = ecg_features(
                        rr, times, fs, spans, row["start_xdf"], row["end_xdf"]
                    )
                    results.extend(ecg)
                    all_rr.extend(dict(participant_id=pid, condition=c, **r) for r in rr_audit)
                except Exception as error:
                    errors.append(
                        dict(
                            scope="ecg",
                            condition=c,
                            reason=f"{type(error).__name__}: {error}",
                            traceback=traceback.format_exc(),
                        )
                    )
        unity_start = row.get("unity_start_unix_ms", np.nan)
        unity_end = row.get("unity_end_unix_ms", np.nan)
        # A genuinely inconsistent boundary is not silently repaired. Missing XDF alone
        # does not make independently timestamped Unity measurements unavailable.
        unity_good = np.isfinite(unity_start) and np.isfinite(unity_end) and unity_end > unity_start
        if row["alignment_reason"] == "boundary_discrepancy_over_250ms":
            unity_good = False
        if unity_good:
            for modality in ["eye", "head"]:
                try:
                    values, diagnostic = behavioral_features(
                        frames[modality], modality, unity_start, unity_end, frames["head"]
                    )
                    results.extend(values)
                    diagnostics.append(dict(participant_id=pid, condition=c, **diagnostic))
                except Exception as error:
                    errors.append(
                        dict(
                            scope=modality,
                            condition=c,
                            reason=f"{type(error).__name__}: {error}",
                            traceback=traceback.format_exc(),
                        )
                    )
            try:
                values, files, diagnostic = optical_flow_features(video, unity_start, unity_end)
                results.extend(values)
                all_frames.extend(dict(participant_id=pid, condition=c, **r) for r in files)
                diagnostics.append(
                    dict(participant_id=pid, condition=c, modality="video", **diagnostic)
                )
            except Exception as error:
                errors.append(
                    dict(
                        scope="video",
                        condition=c,
                        reason=f"{type(error).__name__}: {error}",
                        traceback=traceback.format_exc(),
                    )
                )
        found = {r["feature"] for r in results}
        for feature in FEATURES:
            if feature not in found:
                reason = (
                    "physio_source_or_extraction_failed"
                    if feature.startswith(("eeg_", "ecg_")) and physio is None
                    else "unavailable_alignment"
                    if not alignment_good and feature.startswith(("eeg_", "ecg_"))
                    else "missing_or_inconsistent_unity_boundaries"
                    if not unity_good and not feature.startswith(("eeg_", "ecg_"))
                    else "extraction_error_see_audit"
                )
                results.append(feature_result(feature, reason=reason))
        all_results.extend(dict(participant_id=pid, condition=c, **r) for r in results)
        print(
            f"extract {pid}/{c}: {sum(r['analysis_available'] for r in results)}/11 available",
            flush=True,
        )
    for filename, records in [
        ("feature_quality", all_results),
        ("eeg_channel_quality", all_channels),
        ("ecg_rr_audit", all_rr),
        ("ecg_peaks", peaks),
        ("consumed_video_frames", all_frames),
        ("modality_diagnostics", diagnostics),
    ]:
        write_csv(cache / (filename + ".csv"), pd.DataFrame(records))
    write_json(cache / "source_metadata.json", info)
    write_json(cache / "errors.json", errors)
    write_json(
        cache / "complete.json",
        dict(
            participant_id=pid,
            specification_sha256=spec_hash(),
            quality_rows=len(all_results),
            errors=len(errors),
        ),
    )
    return dict(participant_id=pid, quality_rows=len(all_results), errors=len(errors))


def extract(output: Path, participants: list[str] | None = None) -> None:
    config = verify_lock(output)
    if (output / "COMPLETE.json").exists():
        raise FileExistsError("Completed output is immutable; choose a new run directory.")
    checks = source_integrity(output)
    if not all(r["unchanged"] for r in checks):
        raise ValueError("Source changed since audit; use a new run.")
    selected = participants or list(PARTICIPANTS)
    if not set(selected) <= set(PARTICIPANTS):
        raise ValueError("Participant outside fixed cohort")
    tasks = []
    for pid in selected:
        done = output / "cache" / pid / "complete.json"
        if done.exists():
            if json.loads(done.read_text())["specification_sha256"] != spec_hash():
                raise ValueError("Cache specification mismatch")
            print(f"reuse completed extraction: {pid}", flush=True)
        else:
            tasks.append(pid)
    if tasks:
        with ProcessPoolExecutor(max_workers=min(config["workers"], len(tasks))) as pool:
            futures = {pool.submit(_participant, str(output), pid): pid for pid in tasks}
            for future in as_completed(futures):
                print(f"participant complete: {future.result()}", flush=True)
    if not all((output / "cache" / p / "complete.json").exists() for p in PARTICIPANTS):
        print("Partial extraction saved; analysis awaits the remaining cohort.", flush=True)
        return
    qualities = []
    for pid in PARTICIPANTS:
        q = pd.read_csv(output / "cache" / pid / "feature_quality.csv")
        qualities.append(q)
    quality = pd.concat(qualities, ignore_index=True)
    quality["_feature_order"] = quality.feature.map({f: i for i, f in enumerate(FEATURES)})
    quality = quality.sort_values(["participant_id", "condition", "_feature_order"]).drop(
        columns="_feature_order"
    )
    write_csv(output / "feature_quality.csv", quality)
    table = pd.read_csv(output / "condition_manifest.csv")
    for feature in FEATURES:
        selected = quality.loc[
            quality.feature.eq(feature),
            ["participant_id", "condition", "raw_value", "analysis_value"],
        ]
        table = table.merge(
            selected.rename(
                columns={"raw_value": feature + "_raw", "analysis_value": feature + "_analysis"}
            ),
            on=["participant_id", "condition"],
            validate="one_to_one",
        )
    write_csv(
        output / "participant_condition_features.csv",
        table.sort_values(["participant_id", "condition"]),
    )
    for name in [
        "eeg_channel_quality",
        "ecg_rr_audit",
        "consumed_video_frames",
        "modality_diagnostics",
    ]:
        parts = []
        for pid in PARTICIPANTS:
            try:
                parts.append(pd.read_csv(output / "cache" / pid / (name + ".csv")))
            except pd.errors.EmptyDataError:
                continue
        write_csv(
            output / (name + ".csv"),
            pd.concat(parts, ignore_index=True)
            if parts
            else pd.DataFrame(columns=["participant_id", "condition"]),
        )
    write_json(
        output / "extraction_status.json",
        dict(
            status="complete",
            participants=list(PARTICIPANTS),
            specification_sha256=spec_hash(),
            conditions=len(table),
            quality_records=len(quality),
            hardware=dict(
                opencv_version=cv2.__version__,
                opencv_cuda_devices=cv2.cuda.getCudaEnabledDeviceCount(),
                backend="CPU: installed OpenCV does not provide CUDA Farneback; classical statistics remain on CPU",
            ),
        ),
    )
