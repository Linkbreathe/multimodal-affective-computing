"""Build corrected multimodal Relax embeddings inside the authoritative Project B checkout."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
from scipy.signal import resample
import torch

from src.data.video_transforms import normalize_clip, preprocess_frame
from src.encoders.ecgfounder import ECGFounderEncoder
from src.encoders.inceptiontime import InceptionTimeGazeEncoder
from src.encoders.video_mae import VideoMAEV2Encoder


KEYS = ["participant_id", "condition", "condition_window_index"]
MODALITIES = ("eeg", "ecg", "eye", "head", "video")
EEG_CHANNELS = ("M2", "TP9", "TP10", "M1")
EEG_COLUMNS = (1, 2, 3, 4)
ECG_COLUMNS = (7, 8)


def _local_path(value: str | Path) -> Path:
    """Map a Windows source-manifest path to the same file under WSL."""

    text = str(value)
    if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", text):
        drive = text[0].lower()
        relative = text[2:].replace("\\", "/").lstrip("/")
        return Path(f"/mnt/{drive}/{relative}")
    return Path(text)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device(name: str, require_cuda: bool) -> torch.device:
    if require_cuda and not str(name).startswith("cuda"):
        raise RuntimeError("--require-cuda requires --device cuda")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for neural embedding extraction but is unavailable")
    device = torch.device(name)
    if require_cuda and device.type != "cuda":
        raise RuntimeError("Neural embedding extraction may not fall back to CPU")
    return device


def _autocast(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def _mask_values(path: Path, windows: pd.DataFrame) -> dict[str, np.ndarray]:
    masks = pd.read_csv(path)
    merged = windows[KEYS].merge(masks, on=KEYS, how="left", validate="one_to_one")
    output = {}
    for modality in MODALITIES:
        values = merged[f"{modality}_valid"]
        if values.isna().any():
            raise ValueError(f"Shared mask lacks {modality} rows")
        output[modality] = values.astype(str).str.lower().isin(["true", "1", "1.0"]).to_numpy()
    return output


def _load_physio(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    import pyxdf

    streams, _ = pyxdf.load_xdf(str(path), select_streams=[{"type": "eeg"}], verbose=False)
    if len(streams) != 1:
        raise ValueError(f"Expected one eeg stream in {path}; found {len(streams)}")
    stream = streams[0]
    return (
        np.asarray(stream["time_series"], dtype=np.float32),
        np.asarray(stream["time_stamps"], dtype=float),
        float(stream["info"]["nominal_srate"][0]),
    )


def _window_slice(samples: np.ndarray, timestamps: np.ndarray, row: pd.Series) -> np.ndarray:
    left, right = np.searchsorted(
        timestamps,
        [float(row["window_start_xdf"]), float(row["window_end_xdf"])],
        side="left",
    )
    return samples[int(left) : int(right)]


def _fixed_length(values: np.ndarray, length: int) -> np.ndarray:
    if values.shape[-1] == length:
        return values.astype(np.float32, copy=False)
    return resample(values, length, axis=-1).astype(np.float32)


def _zscore(values: np.ndarray) -> np.ndarray:
    mean = np.nanmean(values, axis=-1, keepdims=True)
    scale = np.nanstd(values, axis=-1, keepdims=True)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    return np.nan_to_num((values - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _run_batches(
    inputs: list[np.ndarray],
    indexes: list[int],
    *,
    batch_size: int,
    device: torch.device,
    forward: Callable[[torch.Tensor], torch.Tensor],
    output: np.ndarray,
) -> None:
    for start in range(0, len(inputs), batch_size):
        batch = torch.as_tensor(np.stack(inputs[start : start + batch_size]), dtype=torch.float32, device=device)
        with torch.no_grad(), _autocast(device):
            embedding = forward(batch)
        values = embedding.detach().float().cpu().numpy()
        for row_index, value in zip(indexes[start : start + batch_size], values, strict=True):
            output[row_index] = value


def _extract_eeg(
    windows: pd.DataFrame,
    sources: pd.DataFrame,
    valid: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        "brain-bzh/reve-large",
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()
    position_bank = AutoModel.from_pretrained(
        "brain-bzh/reve-positions",
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()
    positions = position_bank(list(EEG_CHANNELS)).detach().to(device)
    dimension = int(model.config.embed_dim)
    output = np.zeros((len(windows), 1024), dtype=np.float32)
    for participant, row_indexes in windows.groupby("participant_id", sort=True).groups.items():
        source = sources.loc[str(participant)]
        samples, timestamps, _ = _load_physio(_local_path(source["xdf_path"]))
        inputs, indexes = [], []
        for row_index in row_indexes:
            if not valid[row_index]:
                continue
            window = _window_slice(samples, timestamps, windows.loc[row_index])
            if len(window) < 100:
                raise ValueError(f"EEG window {row_index} is unexpectedly short")
            eeg = _fixed_length(window[:, EEG_COLUMNS].T, 2000)
            inputs.append(_zscore(eeg))
            indexes.append(int(row_index))

        def forward(batch: torch.Tensor) -> torch.Tensor:
            pos = positions.unsqueeze(0).expand(batch.shape[0], -1, -1)
            pooled = model.attention_pooling(model(batch, pos))
            if pooled.shape[-1] != dimension:
                raise RuntimeError(f"Unexpected REVE output shape {tuple(pooled.shape)}")
            return torch.nn.functional.adaptive_avg_pool1d(pooled.unsqueeze(1), 1024).squeeze(1)

        _run_batches(inputs, indexes, batch_size=batch_size, device=device, forward=forward, output=output)
        print(f"EEG embeddings: {participant} ({len(indexes)} windows)", flush=True)
    del model, position_bank
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def _extract_ecg(
    windows: pd.DataFrame,
    sources: pd.DataFrame,
    valid: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model = ECGFounderEncoder(
        weights_path=str(ROOT / "weights/ecgfounder/1_lead_ECGFounder.pth")
    ).to(device).eval()
    output = np.zeros((len(windows), 1024), dtype=np.float32)
    for participant, row_indexes in windows.groupby("participant_id", sort=True).groups.items():
        source = sources.loc[str(participant)]
        samples, timestamps, _ = _load_physio(_local_path(source["xdf_path"]))
        inputs, indexes = [], []
        for row_index in row_indexes:
            if not valid[row_index]:
                continue
            window = _window_slice(samples, timestamps, windows.loc[row_index])
            if len(window) < 100:
                raise ValueError(f"ECG window {row_index} is unexpectedly short")
            ecg = (window[:, ECG_COLUMNS[0]] - window[:, ECG_COLUMNS[1]])[None, :]
            inputs.append(_zscore(_fixed_length(ecg, 5000)))
            indexes.append(int(row_index))

        def forward(batch: torch.Tensor) -> torch.Tensor:
            return model(batch)

        _run_batches(inputs, indexes, batch_size=batch_size, device=device, forward=forward, output=output)
        print(f"ECG embeddings: {participant} ({len(indexes)} windows)", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def _read_logged_csv(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        first = handle.readline().rstrip("\r\n")
        second = handle.readline()
    skip = 1 if first.lower().startswith("sep=") else 0
    sample = second if skip else f"{first}\n{second}"
    delimiter = (first[4:5] or ",") if skip else (";" if sample.count(";") > sample.count(",") else ",")
    decimal = "," if delimiter == ";" and re.search(r"\d,\d", sample) else "."
    return pd.read_csv(
        path,
        sep=delimiter,
        decimal=decimal,
        skiprows=skip,
        low_memory=False,
    )


def _gaze_window(frame: pd.DataFrame, start_ms: float, end_ms: float) -> np.ndarray | None:
    timestamp = pd.to_numeric(frame["unix_time_ms"], errors="coerce")
    selected = frame.loc[(timestamp >= start_ms) & (timestamp < end_ms)].copy()
    if "gaze_available" in selected:
        available = selected["gaze_available"].astype(str).str.lower().isin(["true", "1", "1.0"])
        selected = selected.loc[available]
    xyz = selected[["gaze_direction_x", "gaze_direction_y", "gaze_direction_z"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna().to_numpy(dtype=float)
    if len(xyz) < 10:
        return None
    norm = np.linalg.norm(xyz, axis=1).clip(min=1e-8)
    yaw = np.arctan2(xyz[:, 0], xyz[:, 2])
    pitch = np.arcsin(np.clip(xyz[:, 1] / norm, -1.0, 1.0))
    return _zscore(_fixed_length(np.stack([yaw, pitch]), 900))


def _extract_eye(
    windows: pd.DataFrame,
    sources: pd.DataFrame,
    valid: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model = InceptionTimeGazeEncoder().to(device)
    state = torch.load(ROOT / "checkpoints/inceptiontime_gaze_pretrained.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model.freeze()
    output = np.zeros((len(windows), 128), dtype=np.float32)
    for participant, row_indexes in windows.groupby("participant_id", sort=True).groups.items():
        source = sources.loc[str(participant)]
        log = _read_logged_csv(_local_path(source["eye_tracking_csv"]))
        inputs, indexes = [], []
        for row_index in row_indexes:
            if not valid[row_index]:
                continue
            row = windows.loc[row_index]
            gaze = _gaze_window(log, float(row["window_start_unix_ms"]), float(row["window_end_unix_ms"]))
            if gaze is None:
                raise ValueError(f"Shared eye mask marks {row_index} valid but raw gaze is unavailable")
            inputs.append(gaze)
            indexes.append(int(row_index))
        _run_batches(inputs, indexes, batch_size=batch_size, device=device, forward=model, output=output)
        print(f"Eye embeddings: {participant} ({len(indexes)} windows)", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def _extract_head(
    windows: pd.DataFrame,
    features_path: Path,
    valid: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    features = pd.read_csv(features_path)
    columns = sorted(
        column
        for column in features.columns
        if column.startswith("head_") and pd.to_numeric(features[column], errors="coerce").notna().any()
    )
    merged = windows[KEYS].merge(features[KEYS + columns], on=KEYS, how="left", validate="one_to_one")
    output = merged[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    output[~valid] = 0.0
    return output, columns


def _video_index(path: Path, session_dir: Path) -> tuple[np.ndarray, list[Path]]:
    frame = _read_logged_csv(path)
    direct = pd.to_numeric(frame["unix_time_ms"], errors="coerce")
    deltas = np.diff(direct.dropna().to_numpy(dtype=float))
    positive = deltas[deltas > 0]
    direct_ok = (
        direct.notna().all()
        and len(positive) >= max(2, int(0.9 * max(1, len(deltas))))
        and len(positive)
        and 50.0 <= float(np.median(positive)) <= 200.0
    )
    if direct_ok:
        timestamps = direct.to_numpy(dtype=float)
    else:
        timestamps = pd.to_datetime(frame["utc_timestamp_iso"], errors="coerce", utc=True).astype("int64").to_numpy(dtype=float) / 1e6
    paths = [
        (session_dir / str(value).replace("\\", "/")).resolve()
        for value in frame["relative_path"].astype(str)
    ]
    order = np.argsort(timestamps)
    return timestamps[order], [paths[int(index)] for index in order]


def _video_window(
    timestamps: np.ndarray,
    paths: list[Path],
    start_ms: float,
    end_ms: float,
) -> np.ndarray | None:
    left, right = np.searchsorted(timestamps, [start_ms, end_ms], side="left")
    candidates = list(range(int(left), int(right)))
    if len(candidates) < 16:
        return None
    targets = np.linspace(start_ms, end_ms, 16, endpoint=False) + (end_ms - start_ms) / 32.0
    chosen, used = [], set()
    local_times = timestamps[candidates]
    for target in targets:
        order = np.argsort(np.abs(local_times - target))
        selected = next((candidates[int(position)] for position in order if candidates[int(position)] not in used), None)
        if selected is None:
            return None
        used.add(selected)
        chosen.append(selected)
    frames = []
    for index in chosen:
        image = cv2.imread(str(paths[index]), cv2.IMREAD_COLOR)
        if image is None:
            return None
        frames.append(preprocess_frame(image))
    return np.stack(frames)


def _extract_video(
    windows: pd.DataFrame,
    sources: pd.DataFrame,
    valid: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model = VideoMAEV2Encoder().to(device).eval()
    output = np.zeros((len(windows), 768), dtype=np.float32)
    for participant, row_indexes in windows.groupby("participant_id", sort=True).groups.items():
        source = sources.loc[str(participant)]
        timestamps, paths = _video_index(
            _local_path(source["video_frames_csv"]),
            _local_path(source["session_dir"]),
        )
        inputs, indexes = [], []
        for row_index in row_indexes:
            if not valid[row_index]:
                continue
            row = windows.loc[row_index]
            clip = _video_window(
                timestamps,
                paths,
                float(row["window_start_unix_ms"]),
                float(row["window_end_unix_ms"]),
            )
            if clip is None:
                raise ValueError(f"Shared video mask marks {row_index} valid but 16 frames cannot be loaded")
            inputs.append(normalize_clip(clip).numpy())
            indexes.append(int(row_index))

        def forward(batch: torch.Tensor) -> torch.Tensor:
            return model(batch)

        _run_batches(inputs, indexes, batch_size=batch_size, device=device, forward=forward, output=output)
        print(f"Video embeddings: {participant} ({len(indexes)} windows)", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def _partial_path(output: Path, modality: str) -> Path:
    return output.parent / "window_embeddings" / f"{modality}.pt"


def _load_or_extract(
    output: Path,
    modality: str,
    valid: np.ndarray,
    extract: Callable[[], Any],
    force: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    path = _partial_path(output, modality)
    if path.is_file() and not force:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        values = np.asarray(payload["values"], dtype=np.float32)
        if len(values) != len(valid) or not np.array_equal(np.asarray(payload["valid"], dtype=bool), valid):
            raise ValueError(f"Partial {modality} cache does not match the shared mask")
        print(f"Reusing {path}", flush=True)
        return values, dict(payload.get("metadata", {}))
    started = time.perf_counter()
    result = extract()
    if isinstance(result, tuple):
        values, extra = result
    else:
        values, extra = result, {}
    metadata = {"runtime_seconds": time.perf_counter() - started, **extra}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"values": values, "valid": valid, "metadata": metadata}, path)
    return np.asarray(values, dtype=np.float32), metadata


def _pack_conditions(
    labels: pd.DataFrame,
    windows: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    labels = labels.sort_values(["participant_id", "presentation_position"]).reset_index(drop=True)
    sequence_length = int(pd.to_numeric(windows["condition_window_index"]).max()) + 1
    by_key = {
        (str(participant), str(condition)): group.index.to_numpy(dtype=int)
        for (participant, condition), group in windows.groupby(["participant_id", "condition"], sort=False)
    }
    packed_embeddings = {
        modality: np.zeros((len(labels), sequence_length, values.shape[1]), dtype=np.float32)
        for modality, values in embeddings.items()
    }
    packed_masks = {
        modality: np.zeros((len(labels), sequence_length), dtype=bool) for modality in embeddings
    }
    for condition_index, row in labels.iterrows():
        key = (str(row["participant_id"]), str(row["condition"]))
        if key not in by_key:
            raise ValueError(f"Labels contain a condition without windows: {key}")
        for window_row in by_key[key]:
            sequence_index = int(windows.loc[window_row, "condition_window_index"])
            for modality, values in embeddings.items():
                packed_embeddings[modality][condition_index, sequence_index] = values[window_row]
                packed_masks[modality][condition_index, sequence_index] = bool(masks[modality][window_row])
    return {
        "participant_ids": labels["participant_id"].astype(str).tolist(),
        "conditions": labels["condition"].astype(str).tolist(),
        "presentation_positions": labels["presentation_position"].to_numpy(dtype=np.float32),
        "targets": labels[["relaxation", "discomfort"]].to_numpy(dtype=np.float32),
        "embeddings": {key: torch.from_numpy(value) for key, value in packed_embeddings.items()},
        "masks": {key: torch.from_numpy(value) for key, value in packed_masks.items()},
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    device = _device(args.device, args.require_cuda)
    windows = pd.read_csv(args.windows).reset_index(drop=True)
    labels = pd.read_csv(args.labels)
    sources = pd.read_csv(args.source_manifest, dtype=str).set_index("participant_id")
    if len(windows) != 946 or len(labels) != 135:
        raise ValueError("Formal Relax cache requires 946 windows and 135 labels")
    if windows.duplicated(KEYS).any() or labels.duplicated(["participant_id", "condition"]).any():
        raise ValueError("Shared inputs contain duplicate keys")
    masks = _mask_values(args.mask_manifest, windows)
    embeddings: dict[str, np.ndarray] = {}
    extraction: dict[str, Any] = {}
    requested = tuple(args.modalities)
    if "eeg" in requested:
        embeddings["eeg"], extraction["eeg"] = _load_or_extract(
            args.output, "eeg", masks["eeg"],
            lambda: _extract_eeg(windows, sources, masks["eeg"], device, args.eeg_batch_size),
            args.force,
        )
    if "ecg" in requested:
        embeddings["ecg"], extraction["ecg"] = _load_or_extract(
            args.output, "ecg", masks["ecg"],
            lambda: _extract_ecg(windows, sources, masks["ecg"], device, args.ecg_batch_size),
            args.force,
        )
    if "eye" in requested:
        embeddings["eye"], extraction["eye"] = _load_or_extract(
            args.output, "eye", masks["eye"],
            lambda: _extract_eye(windows, sources, masks["eye"], device, args.eye_batch_size),
            args.force,
        )
    if "head" in requested:
        def extract_head() -> tuple[np.ndarray, dict[str, Any]]:
            values, columns = _extract_head(
                windows, args.project_a_window_features, masks["head"]
            )
            return values, {"feature_columns": columns}

        embeddings["head"], extraction["head"] = _load_or_extract(
            args.output, "head", masks["head"],
            extract_head,
            args.force,
        )
    if "video" in requested:
        embeddings["video"], extraction["video"] = _load_or_extract(
            args.output, "video", masks["video"],
            lambda: _extract_video(windows, sources, masks["video"], device, args.video_batch_size),
            args.force,
        )
    if set(embeddings) != set(requested):
        raise ValueError("Not every requested modality was extracted")
    payload = _pack_conditions(labels, windows, embeddings, masks)
    payload["metadata"] = {
        "schema_version": "relax_aligned_condition_cache_v1",
        "eeg_input_order": list(EEG_CHANNELS),
        "eeg_model": "brain-bzh/reve-large",
        "eeg_output_reduction": "adaptive_avg_pool_1216_to_1024",
        "ecg_model": "ECGFounder 1-lead",
        "eye_model": "pretrained InceptionTime gaze",
        "head_representation": "shared-mask Project A window head features",
        "video_model": "OpenGVLab/VideoMAEv2-Base",
        "device": str(device),
        "cuda_used": device.type == "cuda",
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda) if torch.version.cuda else None,
        "valid_windows": {modality: int(masks[modality].sum()) for modality in requested},
        "embedding_dimensions": {modality: int(values.shape[1]) for modality, values in embeddings.items()},
        "extraction": extraction,
        "inputs": {
            "labels": {"path": str(args.labels.resolve()), "sha256": _sha256(args.labels)},
            "windows": {"path": str(args.windows.resolve()), "sha256": _sha256(args.windows)},
            "mask_manifest": {"path": str(args.mask_manifest.resolve()), "sha256": _sha256(args.mask_manifest)},
            "source_manifest": {"path": str(args.source_manifest.resolve()), "sha256": _sha256(args.source_manifest)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    result = {
        "output": str(args.output.resolve()),
        "sha256": _sha256(args.output),
        "conditions": len(payload["participant_ids"]),
        "modalities": list(requested),
        "valid_windows": payload["metadata"]["valid_windows"],
        "embedding_dimensions": payload["metadata"]["embedding_dimensions"],
        "cuda_used": payload["metadata"]["cuda_used"],
    }
    (args.output.parent / "cache_manifest.json").write_text(
        json.dumps(result | {"metadata": payload["metadata"]}, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--project-a-window-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--eeg-batch-size", type=int, default=8)
    parser.add_argument("--ecg-batch-size", type=int, default=16)
    parser.add_argument("--eye-batch-size", type=int, default=32)
    parser.add_argument("--video-batch-size", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
