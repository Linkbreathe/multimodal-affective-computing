"""Build the frozen Relax EEG/ECG embedding replacement ladder.

Only raw physiological samples below ``data/datasets/relaxdata`` are read.
The existing aligned cache supplies labels, masks, and the unchanged
eye/head/video embeddings.  Each stage replaces only the declared embedding
tensor, writes a complete cache, and verifies byte-level tensor identity for
every untouched modality.
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
from contextlib import nullcontext
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import torch

from mac.preprocessing.relax_physio import (
    LINKED_EEG_CHANNELS,
    RAW_EEG_CHANNELS,
    TARGET_SAMPLING_RATE,
    basic_signal_qc,
    crop_indexes_with_context,
    exact_resample,
    linked_mastoid_reference,
    neurorvq_eeg_filter,
    reve_pretraining_filter,
    select_lead_i_like_ecg_uv,
    select_raw_eeg,
    session_zscore_clip,
    window_zscore,
)
from mac.encoders.neurorvq import NeuroRVQFoundationEncoder


KEYS = ("participant_id", "condition", "condition_window_index")
MODALITIES = ("eeg", "ecg", "eye", "head", "video")
TARGET_SAMPLES = 2000
DEFAULT_DATA_ROOT = ROOT / "data/datasets/relaxdata"
DEFAULT_BASE_CACHE = ROOT / "artifacts/relax/aligned_20260716/condition_embeddings.pt"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/relax/neurorvq_embedding_ladder_20260718"
DEFAULT_ALIGNMENT_ROOT = Path(
    "/mnt/c/Users/linki/amaster/data_collection_v3/analysis/real_time_inference/"
    "artifacts/cross_project_alignment_2026-07-16/alignment_contract"
)
DEFAULT_SOURCE_MANIFEST = Path(
    "/mnt/c/Users/linki/amaster/data_collection_v3/analysis/real_time_inference/"
    "artifacts/manifests/source_manifest.csv"
)
DEFAULT_NEURORVQ_REPO = Path("/tmp/neurorvq-code.7jh6Y8/NeuroRVQ")
DEFAULT_NEURORVQ_WEIGHTS = Path("/tmp/neurorvq-weights/pretrained_models/foundation_models")


STAGE_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "s0_legacy": {
        "eeg": "REVE-large raw [M2,TP9,TP10,M1], 200 Hz FFT resample, window/channel z-score, native 1216 adaptively pooled to 1024",
        "ecg": "legacy ECGFounder",
        "role": "fixed baseline",
    },
    "s1_reve_native_raw4": {
        "eeg": "S0 signal and positions, REVE native 1216-d attention pool",
        "ecg": "legacy ECGFounder",
        "role": "remove arbitrary 1216-to-1024 feature-axis pooling",
    },
    "s2_reve_native_linked2": {
        "eeg": "linked-mastoid TP9/TP10 before z-score, no filter, REVE native 1216-d",
        "ecg": "legacy ECGFounder",
        "role": "isolate correct M1/M2 referencing",
    },
    "s3_reve_clean_linked2": {
        "eeg": "linked TP9/TP10, official NeuroRVQ EEG filter/clip, 200 Hz, window/channel z-score, REVE native 1216-d",
        "ecg": "legacy ECGFounder",
        "role": "clean corrected REVE comparator",
    },
    "s4_neurorvq_clean_z": {
        "eeg": "exact S3 cleaned and z-scored signal, NeuroRVQ EEG four-branch token mean pool (800-d)",
        "ecg": "legacy ECGFounder",
        "role": "encoder-only REVE-to-NeuroRVQ comparison",
    },
    "s5_neurorvq_clean_native": {
        "eeg": "linked TP9/TP10, official NeuroRVQ EEG filter/clip, 200 Hz, physical microvolts without external z-score, NeuroRVQ 800-d",
        "ecg": "legacy ECGFounder",
        "role": "NeuroRVQ official native-scaling main candidate",
    },
    "s6_neurorvq_eeg_ecg": {
        "eeg": "S5 NeuroRVQ EEG",
        "ecg": "LA-RA Lead-I-like signal converted from microvolts to millivolts, 200 Hz, NeuroRVQ ECG 160-d",
        "role": "joint NeuroRVQ EEG+ECG replacement",
    },
    "r3_reve_official_session": {
        "eeg": "linked TP9/TP10, 0.5-99.5 Hz, session/channel z-score, ±15 SD, 200 Hz, REVE native 1216-d",
        "ecg": "legacy ECGFounder",
        "role": "offline/transductive REVE-pretraining preprocessing sensitivity",
    },
    "ecg_only_neurorvq": {
        "eeg": "S3 clean corrected REVE",
        "ecg": "same NeuroRVQ ECG embedding as S6",
        "role": "isolate ECG encoder replacement",
    },
}


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_array(values: np.ndarray | torch.Tensor) -> str:
    array = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    contiguous = np.ascontiguousarray(array)
    digest = sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(contiguous.shape).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _git_value(repo: Path, *args: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repo), *args),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _local_relax_path(value: str | Path, data_root: Path) -> Path:
    """Map a manifest path to the user-scoped local Relax dataset root."""

    text = str(value).replace("\\", "/")
    marker = "/relaxdata/"
    if marker in text.lower():
        offset = text.lower().index(marker) + len(marker)
        relative = Path(text[offset:])
        candidate = data_root / relative
    else:
        source = Path(text)
        candidate = source if source.is_absolute() else data_root / source
    root = data_root.resolve()
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Resolved source escapes the local Relax root: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Local Relax source is missing: {resolved}")
    return resolved


def _load_physio(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    import pyxdf

    streams, _ = pyxdf.load_xdf(
        str(path),
        select_streams=[{"type": "eeg"}],
        verbose=False,
    )
    if len(streams) != 1:
        raise ValueError(f"Expected exactly one physiological stream in {path}; got {len(streams)}")
    stream = streams[0]
    samples = np.asarray(stream["time_series"], dtype=np.float32)
    timestamps = np.asarray(stream["time_stamps"], dtype=np.float64)
    sampling_rate = float(stream["info"]["nominal_srate"][0])
    if samples.ndim != 2 or samples.shape[1] != 9:
        raise ValueError(f"Expected the Relax 9-column stream, got {samples.shape} in {path}")
    if len(samples) != len(timestamps):
        raise ValueError(f"Samples/timestamps mismatch in {path}")
    return samples, timestamps, sampling_rate


def _window_bounds(timestamps: np.ndarray, row: pd.Series) -> tuple[int, int]:
    left, right = np.searchsorted(
        timestamps,
        [float(row["window_start_xdf"]), float(row["window_end_xdf"])],
        side="left",
    )
    if int(right - left) < 100:
        raise ValueError(f"Unexpectedly short window: {tuple(row[key] for key in KEYS)}")
    return int(left), int(right)


def _autocast(device: torch.device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )


def _batch_encode(
    inputs: np.ndarray,
    *,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    autocast: bool,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, len(inputs), batch_size):
        batch = torch.as_tensor(
            inputs[start : start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        context = _autocast(device) if autocast else nullcontext()
        with torch.inference_mode(), context:
            result = model(batch)
        result = result.detach().float().cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError(f"Encoder returned non-finite values for batch starting at {start}")
        outputs.append(result)
    if not outputs:
        raise ValueError("No windows were passed to the encoder")
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def _extract_reve(
    inputs: Mapping[str, tuple[np.ndarray, Sequence[str]]],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel

    started = time.perf_counter()
    model = AutoModel.from_pretrained(
        "brain-bzh/reve-large",
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()
    # The official position-bank class has zero Parameters but one essential
    # 543x3 coordinate buffer. Transformers 4.49 raises StopIteration while
    # checking the device of this parameterless model in from_pretrained, so
    # instantiate the cached config and then strictly load the official buffer.
    position_config = AutoConfig.from_pretrained(
        "brain-bzh/reve-positions",
        trust_remote_code=True,
        local_files_only=True,
    )
    position_bank = AutoModel.from_config(
        position_config,
        trust_remote_code=True,
    )
    position_checkpoint = Path(
        hf_hub_download(
            "brain-bzh/reve-positions",
            "model.safetensors",
            local_files_only=True,
        )
    )
    position_state = load_file(str(position_checkpoint), device="cpu")
    position_load_result = position_bank.load_state_dict(position_state, strict=True)
    if position_load_result.missing_keys or position_load_result.unexpected_keys:
        raise RuntimeError(f"REVE position buffer did not load strictly: {position_load_result}")
    if set(position_state) != {"embedding"} or tuple(position_state["embedding"].shape) != (543, 3):
        raise RuntimeError("Unexpected official REVE position-bank checkpoint structure")
    position_bank = position_bank.to(device).eval()
    dimension = int(model.config.embed_dim)
    if dimension != 1216:
        raise ValueError(f"Expected REVE-large native dimension 1216, got {dimension}")

    outputs: dict[str, np.ndarray] = {}
    stage_times: dict[str, float] = {}
    for stage, (values, channels) in inputs.items():
        stage_started = time.perf_counter()
        positions = position_bank(list(channels)).detach().to(device)

        def forward(batch: torch.Tensor) -> torch.Tensor:
            pos = positions.unsqueeze(0).expand(batch.shape[0], -1, -1)
            pooled = model.attention_pooling(model(batch, pos))
            if pooled.shape != (batch.shape[0], dimension):
                raise RuntimeError(f"Unexpected REVE output shape: {tuple(pooled.shape)}")
            return pooled

        outputs[stage] = _batch_encode(
            values,
            model=forward,  # type: ignore[arg-type]
            device=device,
            batch_size=batch_size,
            autocast=True,
        )
        stage_times[stage] = time.perf_counter() - stage_started
        print(f"REVE {stage}: {len(values)} windows in {stage_times[stage]:.1f}s", flush=True)

    metadata = {
        "model": "brain-bzh/reve-large",
        "position_bank": "brain-bzh/reve-positions",
        "native_dimension": dimension,
        "model_commit_hash": getattr(model.config, "_commit_hash", None),
        "position_bank_commit_hash": getattr(position_config, "_commit_hash", None),
        "position_bank_parameter_count": sum(p.numel() for p in position_bank.parameters()),
        "position_bank_buffer_count": sum(b.numel() for b in position_bank.buffers()),
        "position_bank_checkpoint": str(position_checkpoint),
        "position_bank_checkpoint_sha256": _sha256(position_checkpoint),
        "position_bank_load_workaround": "AutoModel.from_config followed by strict official buffer load because Transformers 4.49 from_pretrained raises StopIteration for the zero-Parameter class",
        "stage_runtime_seconds": stage_times,
        "runtime_seconds": time.perf_counter() - started,
    }
    del model, position_bank
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return outputs, metadata


def _extract_neurorvq(
    inputs: Mapping[str, np.ndarray],
    *,
    modality: str,
    channels: Sequence[str],
    repo_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    started = time.perf_counter()
    encoder = NeuroRVQFoundationEncoder(
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        modality=modality,
        channels=channels,
    ).to(device).eval()
    outputs: dict[str, np.ndarray] = {}
    stage_times: dict[str, float] = {}
    for stage, values in inputs.items():
        stage_started = time.perf_counter()
        outputs[stage] = _batch_encode(
            values,
            model=encoder,
            device=device,
            batch_size=batch_size,
            autocast=False,
        )
        stage_times[stage] = time.perf_counter() - stage_started
        print(
            f"NeuroRVQ-{modality.upper()} {stage}: {len(values)} windows in {stage_times[stage]:.1f}s",
            flush=True,
        )
    metadata = {
        "model": f"NeuroRVQ-{modality.upper()} foundation v1",
        "channels": list(channels),
        "embedding_dimension": encoder.embedding_dimension,
        "patch_size": encoder.patch_size,
        "maximum_patches": encoder.maximum_patches,
        "checkpoint": str(checkpoint_path.resolve()),
        "load_diagnostics": encoder.load_diagnostics,
        "stage_runtime_seconds": stage_times,
        "runtime_seconds": time.perf_counter() - started,
    }
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return outputs, metadata


def _slots_from_windows(
    windows: pd.DataFrame,
    base: Mapping[str, Any],
    modality: str,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    lookup = {
        (str(participant), str(condition)): index
        for index, (participant, condition) in enumerate(
            zip(base["participant_ids"], base["conditions"], strict=True)
        )
    }
    mask = torch.as_tensor(base["masks"][modality], dtype=torch.bool).numpy()
    valid_rows: list[int] = []
    slots: list[tuple[int, int]] = []
    for row_index, row in windows.iterrows():
        key = (str(row["participant_id"]), str(row["condition"]))
        condition_index = lookup.get(key)
        if condition_index is None:
            raise ValueError(f"Window key is absent from the base cache: {key}")
        sequence_index = int(row["condition_window_index"])
        if sequence_index < 0 or sequence_index >= mask.shape[1]:
            raise ValueError(f"Window index is outside the base sequence: {tuple(row[key] for key in KEYS)}")
        if bool(mask[condition_index, sequence_index]):
            valid_rows.append(int(row_index))
            slots.append((condition_index, sequence_index))
    if len(valid_rows) != int(mask.sum()):
        raise ValueError(
            f"{modality} valid-row mapping mismatch: rows={len(valid_rows)}, cache={int(mask.sum())}"
        )
    return np.asarray(valid_rows, dtype=int), slots


def _preprocess_windows(
    *,
    windows: pd.DataFrame,
    sources: pd.DataFrame,
    data_root: Path,
    eeg_rows: np.ndarray,
    ecg_rows: np.ndarray,
    output_root: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    eeg_position = {int(row): index for index, row in enumerate(eeg_rows)}
    ecg_position = {int(row): index for index, row in enumerate(ecg_rows)}
    eeg_inputs = {
        "raw4_z": np.empty((len(eeg_rows), 4, TARGET_SAMPLES), dtype=np.float32),
        "linked2_z": np.empty((len(eeg_rows), 2, TARGET_SAMPLES), dtype=np.float32),
        "clean2_z": np.empty((len(eeg_rows), 2, TARGET_SAMPLES), dtype=np.float32),
        "clean2_native_uv": np.empty((len(eeg_rows), 2, TARGET_SAMPLES), dtype=np.float32),
        "reve_session2": np.empty((len(eeg_rows), 2, TARGET_SAMPLES), dtype=np.float32),
    }
    ecg_input = np.empty((len(ecg_rows), 1, TARGET_SAMPLES), dtype=np.float32)
    qc_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    started = time.perf_counter()

    eeg_row_set = set(int(value) for value in eeg_rows)
    ecg_row_set = set(int(value) for value in ecg_rows)
    for participant, group in windows.groupby("participant_id", sort=True):
        participant_rows = [int(value) for value in group.index]
        needed_eeg = [value for value in participant_rows if value in eeg_row_set]
        needed_ecg = [value for value in participant_rows if value in ecg_row_set]
        if not needed_eeg and not needed_ecg:
            continue
        if str(participant) not in sources.index:
            raise ValueError(f"Source manifest lacks participant {participant}")
        xdf_path = _local_relax_path(sources.loc[str(participant), "xdf_path"], data_root)
        samples, timestamps, sampling_rate = _load_physio(xdf_path)
        source_records.append(
            {
                "participant_id": str(participant),
                "relative_xdf_path": str(xdf_path.relative_to(data_root.resolve())),
                "xdf_size_bytes": xdf_path.stat().st_size,
                "xdf_sha256": _sha256(xdf_path),
                "nominal_sampling_rate": sampling_rate,
                "sample_count": len(samples),
            }
        )

        raw_session = select_raw_eeg(samples) if needed_eeg else None
        linked_session = linked_mastoid_reference(raw_session) if raw_session is not None else None
        reve_session = None
        if linked_session is not None:
            filtered_session = reve_pretraining_filter(linked_session, sampling_rate)
            reve_session, session_mean, session_scale = session_zscore_clip(
                filtered_session,
                clip_standard_deviations=15.0,
            )
            for channel_index, channel in enumerate(LINKED_EEG_CHANNELS):
                session_rows.append(
                    {
                        "participant_id": str(participant),
                        "channel": channel,
                        "mean_after_0p5_99p5_uv": float(session_mean[channel_index]),
                        "std_after_0p5_99p5_uv": float(session_scale[channel_index]),
                        "fraction_at_15sd_clip": float(
                            np.mean(np.abs(reve_session[channel_index]) >= 15.0)
                        ),
                    }
                )

        for row_index in needed_eeg:
            row = windows.loc[row_index]
            assert raw_session is not None and linked_session is not None and reve_session is not None
            left, right = _window_bounds(timestamps, row)
            output_index = eeg_position[row_index]
            raw_resampled = exact_resample(raw_session[:, left:right], TARGET_SAMPLES)
            linked_resampled = exact_resample(linked_session[:, left:right], TARGET_SAMPLES)
            outer, inner = crop_indexes_with_context(
                timestamps,
                float(row["window_start_xdf"]),
                float(row["window_end_xdf"]),
                context_seconds=2.0,
            )
            cleaned_padded = neurorvq_eeg_filter(
                linked_session[:, outer],
                sampling_rate,
                clip_uv=500.0,
            )
            clean_native = exact_resample(cleaned_padded[:, inner], TARGET_SAMPLES)
            revealike = exact_resample(reve_session[:, left:right], TARGET_SAMPLES)
            eeg_inputs["raw4_z"][output_index] = window_zscore(raw_resampled)
            eeg_inputs["linked2_z"][output_index] = window_zscore(linked_resampled)
            eeg_inputs["clean2_native_uv"][output_index] = clean_native
            eeg_inputs["clean2_z"][output_index] = window_zscore(clean_native)
            eeg_inputs["reve_session2"][output_index] = revealike

            key_record = {
                "participant_id": str(row["participant_id"]),
                "condition": str(row["condition"]),
                "condition_window_index": int(row["condition_window_index"]),
            }
            for stage_name, values in (
                ("raw4_z", eeg_inputs["raw4_z"][output_index]),
                ("linked2_z", eeg_inputs["linked2_z"][output_index]),
                ("clean2_z", eeg_inputs["clean2_z"][output_index]),
                ("clean2_native_uv", eeg_inputs["clean2_native_uv"][output_index]),
                ("reve_session2", eeg_inputs["reve_session2"][output_index]),
            ):
                qc_rows.append(
                    {
                        **key_record,
                        "stage_signal": stage_name,
                        "channel_count": values.shape[0],
                        "sampling_rate": TARGET_SAMPLING_RATE,
                        "fraction_absolute_ge_500": float(np.mean(np.abs(values) >= 500.0)),
                        **basic_signal_qc(values, TARGET_SAMPLING_RATE),
                    }
                )

        for row_index in needed_ecg:
            row = windows.loc[row_index]
            left, right = _window_bounds(timestamps, row)
            ecg_uv = select_lead_i_like_ecg_uv(samples[left:right])
            # WFDB/PTB-XL physical values consumed by the official example are mV.
            ecg_mv = exact_resample(ecg_uv, TARGET_SAMPLES) / 1000.0
            ecg_input[ecg_position[row_index]] = ecg_mv
            qc_rows.append(
                {
                    "participant_id": str(row["participant_id"]),
                    "condition": str(row["condition"]),
                    "condition_window_index": int(row["condition_window_index"]),
                    "stage_signal": "ecg_lead_i_like_native_mv",
                    "channel_count": 1,
                    "sampling_rate": TARGET_SAMPLING_RATE,
                    "fraction_absolute_ge_500": float(np.mean(np.abs(ecg_mv) >= 500.0)),
                    **basic_signal_qc(ecg_mv, TARGET_SAMPLING_RATE),
                }
            )
        print(
            f"Preprocessed {participant}: EEG={len(needed_eeg)}, ECG={len(needed_ecg)}",
            flush=True,
        )
        del samples, raw_session, linked_session, reve_session

    for name, values in eeg_inputs.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values remain in preprocessed EEG input {name}")
    if not np.isfinite(ecg_input).all():
        raise ValueError("Non-finite values remain in preprocessed ECG input")

    qc_dir = output_root / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(qc_rows).to_csv(qc_dir / "preprocessed_window_qc.csv", index=False)
    pd.DataFrame(session_rows).to_csv(qc_dir / "reve_session_statistics.csv", index=False)
    pd.DataFrame(source_records).to_csv(qc_dir / "local_xdf_sources.csv", index=False)
    metadata = {
        "runtime_seconds": time.perf_counter() - started,
        "eeg_window_count": len(eeg_rows),
        "ecg_window_count": len(ecg_rows),
        "source_files": source_records,
        "window_qc_path": str((qc_dir / "preprocessed_window_qc.csv").resolve()),
        "session_statistics_path": str((qc_dir / "reve_session_statistics.csv").resolve()),
    }
    return eeg_inputs, ecg_input, metadata


def _pack_replacement(
    embeddings: np.ndarray,
    slots: Sequence[tuple[int, int]],
    *,
    condition_count: int,
    sequence_length: int,
) -> torch.Tensor:
    if len(embeddings) != len(slots):
        raise ValueError("Embedding/slot count mismatch")
    output = np.zeros(
        (condition_count, sequence_length, embeddings.shape[1]),
        dtype=np.float32,
    )
    for embedding, (condition_index, sequence_index) in zip(embeddings, slots, strict=True):
        output[condition_index, sequence_index] = embedding
    return torch.from_numpy(output)


def _cache_for_stage(
    *,
    base: Mapping[str, Any],
    stage: str,
    output_path: Path,
    eeg: torch.Tensor | None,
    ecg: torch.Tensor | None,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    payload = deepcopy(base)
    if eeg is not None:
        payload["embeddings"]["eeg"] = eeg
    if ecg is not None:
        payload["embeddings"]["ecg"] = ecg
    payload["metadata"] = deepcopy(dict(base.get("metadata", {})))
    payload["metadata"].update(
        {
            "schema_version": "relax_aligned_condition_cache_embedding_ladder_v1",
            "ladder_stage": stage,
            "ladder_stage_description": STAGE_DESCRIPTIONS[stage],
            "embedding_dimensions": {
                modality: int(torch.as_tensor(payload["embeddings"][modality]).shape[-1])
                for modality in MODALITIES
            },
            "embedding_ladder": dict(metadata),
        }
    )
    if eeg is not None:
        payload["metadata"]["eeg_model"] = STAGE_DESCRIPTIONS[stage]["eeg"]
        payload["metadata"]["eeg_output_reduction"] = (
            "four_branch_patch_token_concat_then_token_mean"
            if stage.startswith("s4_") or stage.startswith("s5_") or stage.startswith("s6_")
            else "native_attention_pool_1216"
        )
    if ecg is not None:
        payload["metadata"]["ecg_model"] = STAGE_DESCRIPTIONS[stage]["ecg"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    observed = torch.load(output_path, map_location="cpu", weights_only=False)
    for modality in MODALITIES:
        expected_tensor = payload["embeddings"][modality]
        observed_tensor = observed["embeddings"][modality]
        if _hash_array(expected_tensor) != _hash_array(observed_tensor):
            raise ValueError(f"Serialized {modality} tensor changed for stage {stage}")
        if not torch.equal(
            torch.as_tensor(base["masks"][modality]),
            torch.as_tensor(observed["masks"][modality]),
        ):
            raise ValueError(f"Mask changed for {modality} at stage {stage}")
        if modality not in {"eeg" if eeg is not None else "", "ecg" if ecg is not None else ""}:
            if _hash_array(base["embeddings"][modality]) != _hash_array(observed_tensor):
                raise ValueError(f"Untouched modality {modality} changed at stage {stage}")
    return {
        "stage": stage,
        "description": STAGE_DESCRIPTIONS[stage],
        "cache_path": str(output_path.resolve()),
        "cache_sha256": _sha256(output_path),
        "embedding_dimensions": observed["metadata"]["embedding_dimensions"],
        "tensor_sha256": {
            modality: _hash_array(observed["embeddings"][modality]) for modality in MODALITIES
        },
        "mask_sha256": {
            modality: _hash_array(observed["masks"][modality]) for modality in MODALITIES
        },
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    existing_nonprotocol = (
        [path for path in args.output_root.iterdir() if path.name != "preregistration"]
        if args.output_root.exists()
        else []
    )
    if existing_nonprotocol and not args.resume:
        raise FileExistsError(
            f"Output root is non-empty; pass --resume only after verifying provenance: {args.output_root}"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This formal frozen-encoder extraction requires CUDA")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    base = torch.load(args.base_cache, map_location="cpu", weights_only=False)
    windows = pd.read_csv(args.windows).reset_index(drop=True)
    sources = pd.read_csv(args.source_manifest, dtype=str).set_index("participant_id")
    if len(windows) != 946 or len(base["participant_ids"]) != 135:
        raise ValueError("The embedding ladder requires the fixed 946-window/135-condition alignment")
    if windows.duplicated(list(KEYS)).any():
        raise ValueError("Alignment windows contain duplicate keys")
    if tuple(base["embeddings"]) != MODALITIES:
        raise ValueError(f"Unexpected base modality order: {tuple(base['embeddings'])}")

    eeg_rows, eeg_slots = _slots_from_windows(windows, base, "eeg")
    ecg_rows, ecg_slots = _slots_from_windows(windows, base, "ecg")
    if (len(eeg_rows), len(ecg_rows)) != (545, 945):
        raise ValueError(f"Expected 545 EEG and 945 ECG internal-valid windows; got {len(eeg_rows)}, {len(ecg_rows)}")

    eeg_inputs, ecg_input, preprocessing_metadata = _preprocess_windows(
        windows=windows,
        sources=sources,
        data_root=args.data_root.resolve(),
        eeg_rows=eeg_rows,
        ecg_rows=ecg_rows,
        output_root=args.output_root,
    )

    reve_outputs, reve_metadata = _extract_reve(
        {
            "s1_reve_native_raw4": (eeg_inputs["raw4_z"], RAW_EEG_CHANNELS),
            "s2_reve_native_linked2": (eeg_inputs["linked2_z"], LINKED_EEG_CHANNELS),
            "s3_reve_clean_linked2": (eeg_inputs["clean2_z"], LINKED_EEG_CHANNELS),
            "r3_reve_official_session": (eeg_inputs["reve_session2"], LINKED_EEG_CHANNELS),
        },
        device=device,
        batch_size=args.reve_batch_size,
    )
    eeg_neurorvq_outputs, eeg_neurorvq_metadata = _extract_neurorvq(
        {
            "s4_neurorvq_clean_z": eeg_inputs["clean2_z"],
            "s5_neurorvq_clean_native": eeg_inputs["clean2_native_uv"],
        },
        modality="EEG",
        channels=LINKED_EEG_CHANNELS,
        repo_path=args.neurorvq_repo,
        checkpoint_path=args.neurorvq_eeg_checkpoint,
        device=device,
        batch_size=args.neurorvq_eeg_batch_size,
    )
    ecg_neurorvq_outputs, ecg_neurorvq_metadata = _extract_neurorvq(
        {"ecg_neurorvq": ecg_input},
        modality="ECG",
        channels=("i",),
        repo_path=args.neurorvq_repo,
        checkpoint_path=args.neurorvq_ecg_checkpoint,
        device=device,
        batch_size=args.neurorvq_ecg_batch_size,
    )

    condition_count = len(base["participant_ids"])
    sequence_length = int(torch.as_tensor(base["masks"]["eeg"]).shape[1])
    packed_eeg = {
        stage: _pack_replacement(
            values,
            eeg_slots,
            condition_count=condition_count,
            sequence_length=sequence_length,
        )
        for stage, values in {**reve_outputs, **eeg_neurorvq_outputs}.items()
    }
    packed_ecg = _pack_replacement(
        ecg_neurorvq_outputs["ecg_neurorvq"],
        ecg_slots,
        condition_count=condition_count,
        sequence_length=sequence_length,
    )

    neurorvq_commit = _git_value(args.neurorvq_repo, "rev-parse", "HEAD")
    neurorvq_dirty = bool(_git_value(args.neurorvq_repo, "status", "--porcelain"))
    provenance = {
        "created_at_local": pd.Timestamp.now(tz="Europe/Berlin").isoformat(),
        "seed": args.seed,
        "device": str(device),
        "cuda_used": True,
        "cuda_device_name": torch.cuda.get_device_name(device),
        "torch_version": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "base_cache": {
            "path": str(args.base_cache.resolve()),
            "sha256": _sha256(args.base_cache),
        },
        "local_data_root": str(args.data_root.resolve()),
        "local_data_only_enforced": True,
        "inputs": {
            "windows": {"path": str(args.windows.resolve()), "sha256": _sha256(args.windows)},
            "source_manifest": {
                "path": str(args.source_manifest.resolve()),
                "sha256": _sha256(args.source_manifest),
                "note": "paths are remapped below local_data_root; no raw /mnt/c signal file is read",
            },
        },
        "preprocessing": preprocessing_metadata,
        "reve": reve_metadata,
        "neurorvq": {
            "repo": str(args.neurorvq_repo.resolve()),
            "git_commit": neurorvq_commit,
            "git_dirty": neurorvq_dirty,
            "eeg": eeg_neurorvq_metadata,
            "ecg": ecg_neurorvq_metadata,
        },
        "extractor_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "preprocessing_source": {
            "path": str((ROOT / "src/data/relax_physio_preprocessing.py").resolve()),
            "sha256": _sha256(ROOT / "src/data/relax_physio_preprocessing.py"),
        },
        "adapter_source": {
            "path": str((ROOT / "src/encoders/neurorvq.py").resolve()),
            "sha256": _sha256(ROOT / "src/encoders/neurorvq.py"),
        },
    }

    stage_manifests: dict[str, Any] = {
        "s0_legacy": {
            "stage": "s0_legacy",
            "description": STAGE_DESCRIPTIONS["s0_legacy"],
            "cache_path": str(args.base_cache.resolve()),
            "cache_sha256": _sha256(args.base_cache),
            "embedding_dimensions": dict(base["metadata"]["embedding_dimensions"]),
            "tensor_sha256": {
                modality: _hash_array(base["embeddings"][modality]) for modality in MODALITIES
            },
            "mask_sha256": {
                modality: _hash_array(base["masks"][modality]) for modality in MODALITIES
            },
        }
    }
    cache_inputs: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
        "s1_reve_native_raw4": (packed_eeg["s1_reve_native_raw4"], None),
        "s2_reve_native_linked2": (packed_eeg["s2_reve_native_linked2"], None),
        "s3_reve_clean_linked2": (packed_eeg["s3_reve_clean_linked2"], None),
        "s4_neurorvq_clean_z": (packed_eeg["s4_neurorvq_clean_z"], None),
        "s5_neurorvq_clean_native": (packed_eeg["s5_neurorvq_clean_native"], None),
        "s6_neurorvq_eeg_ecg": (packed_eeg["s5_neurorvq_clean_native"], packed_ecg),
        "r3_reve_official_session": (packed_eeg["r3_reve_official_session"], None),
        "ecg_only_neurorvq": (packed_eeg["s3_reve_clean_linked2"], packed_ecg),
    }
    for stage, (eeg, ecg) in cache_inputs.items():
        output_path = args.output_root / "caches" / stage / "condition_embeddings.pt"
        stage_manifests[stage] = _cache_for_stage(
            base=base,
            stage=stage,
            output_path=output_path,
            eeg=eeg,
            ecg=ecg,
            metadata=provenance,
        )
        _write_json(output_path.parent / "cache_manifest.json", stage_manifests[stage])
        print(f"Wrote {stage}: {stage_manifests[stage]['cache_sha256']}", flush=True)

    untouched_reference = {
        modality: _hash_array(base["embeddings"][modality])
        for modality in ("eye", "head", "video")
    }
    if any(
        manifest["tensor_sha256"][modality] != reference
        for manifest in stage_manifests.values()
        for modality, reference in untouched_reference.items()
    ):
        raise ValueError("At least one stage changed an untouched eye/head/video tensor")

    result = {
        "schema_version": "relax_neurorvq_embedding_ladder_v1",
        "output_root": str(args.output_root.resolve()),
        "stages": stage_manifests,
        "provenance": provenance,
        "untouched_eye_head_video_tensor_sha256": untouched_reference,
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(args.output_root / "embedding_ladder_manifest.json", result)
    print(json.dumps({
        "manifest": str((args.output_root / "embedding_ladder_manifest.json").resolve()),
        "stages": list(stage_manifests),
        "runtime_seconds": result["runtime_seconds"],
    }, ensure_ascii=False, indent=2))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--base-cache", type=Path, default=DEFAULT_BASE_CACHE)
    parser.add_argument("--windows", type=Path, default=DEFAULT_ALIGNMENT_ROOT / "windows.csv")
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--neurorvq-repo", type=Path, default=DEFAULT_NEURORVQ_REPO)
    parser.add_argument(
        "--neurorvq-eeg-checkpoint",
        type=Path,
        default=DEFAULT_NEURORVQ_WEIGHTS / "NeuroRVQ_EEG_foundation_model_v1.pt",
    )
    parser.add_argument(
        "--neurorvq-ecg-checkpoint",
        type=Path,
        default=DEFAULT_NEURORVQ_WEIGHTS / "NeuroRVQ_ECG_foundation_model_v1.pt",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--reve-batch-size", type=int, default=8)
    parser.add_argument("--neurorvq-eeg-batch-size", type=int, default=16)
    parser.add_argument("--neurorvq-ecg-batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    build(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
