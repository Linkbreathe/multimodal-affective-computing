"""Research-only dynamic-texture descriptors for aligned egocentric windows.

The extractor deliberately emits only twelve scalar descriptors per window.
RGB pixels are decoded only inside this offline feature boundary and are never
accepted by the training interface.  The frozen five-way common-valid mask is
applied after frame selection; masked source windows stay present as rows but
their descriptors remain missing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from real_time_ml.config import ProjectConfig
from real_time_ml.data.io import discover_session_dir
from real_time_ml.data.video import VideoFrame, load_video_index, uniform_clip_frames
from real_time_ml.evaluation.alignment import file_sha256, validate_alignment_contract
from real_time_ml.modeling.condition_data import aggregate_window_frame
from real_time_ml.utils import write_json


DYNAMIC_TEXTURE_VERSION = "dynamic_texture_v1"
DYNAMIC_TEXTURE_COLUMNS = (
    "video_dynamic_motion_mean_diag_s",
    "video_dynamic_motion_p90_diag_s",
    "video_dynamic_temporal_dominant_hz",
    "video_dynamic_temporal_centroid_hz",
    "video_dynamic_spatial_dominant_cycles_diag",
    "video_dynamic_spatial_centroid_cycles_diag",
    "video_dynamic_direction_cos",
    "video_dynamic_direction_sin",
    "video_dynamic_direction_resultant",
    "video_dynamic_structure_coherence_3d",
    "video_dynamic_spectral_entropy_3d",
    "video_dynamic_motion_energy_stability",
)
KEY_COLUMNS = ("participant_id", "condition", "condition_window_index")


@dataclass(frozen=True)
class DynamicTextureSettings:
    frames_per_window: int = 16
    resize_short_side: int = 224
    center_crop_size: int = 224
    analysis_size: int = 112
    farneback_pyr_scale: float = 0.5
    farneback_levels: int = 3
    farneback_window_size: int = 15
    farneback_iterations: int = 3
    farneback_poly_n: int = 5
    farneback_poly_sigma: float = 1.2


def _stable_array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _resize_center_crop(image: np.ndarray, settings: DynamicTextureSettings) -> np.ndarray:
    """Resize by the short side and return an RGB center crop."""
    import cv2

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Dynamic-texture extraction expects decoded BGR images")
    height, width = image.shape[:2]
    if min(height, width) < 1:
        raise ValueError("Decoded video frame has an empty spatial dimension")
    scale = float(settings.resize_short_side) / float(min(height, width))
    resized_width = max(settings.center_crop_size, int(round(width * scale)))
    resized_height = max(settings.center_crop_size, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    left = (resized_width - settings.center_crop_size) // 2
    top = (resized_height - settings.center_crop_size) // 2
    crop = resized[
        top : top + settings.center_crop_size,
        left : left + settings.center_crop_size,
    ]
    if crop.shape[:2] != (settings.center_crop_size, settings.center_crop_size):
        raise ValueError("Center crop did not produce the configured spatial size")
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def load_analysis_clip(
    frames: Iterable[VideoFrame],
    settings: DynamicTextureSettings | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Decode one selected clip into grayscale analysis frames.

    RGB crops exist only during this function.  The returned tensor is
    ``[time, height, width]`` float32 grayscale in ``[0, 1]``.
    """
    import cv2

    cfg = settings or DynamicTextureSettings()
    selected = tuple(frames)
    if len(selected) != cfg.frames_per_window:
        raise ValueError(
            f"Expected {cfg.frames_per_window} distinct selected frames; found {len(selected)}"
        )
    analysis: list[np.ndarray] = []
    audit: list[dict[str, Any]] = []
    for slot, frame in enumerate(selected):
        bgr = cv2.imread(str(frame.path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not decode selected JPEG: {frame.path}")
        source_height, source_width = bgr.shape[:2]
        rgb = _resize_center_crop(bgr, cfg)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(
            gray,
            (cfg.analysis_size, cfg.analysis_size),
            interpolation=cv2.INTER_AREA,
        )
        analysis.append(gray.astype(np.float32) / 255.0)
        audit.append(
            {
                "slot": int(slot),
                "frame_index": int(frame.frame_index),
                "frame_unix_ms": int(frame.unix_time_ms),
                "path": str(frame.path.resolve()),
                "source_width": int(source_width),
                "source_height": int(source_height),
            }
        )
        del rgb
    timestamps = np.asarray([frame.unix_time_ms for frame in selected], dtype=np.float64)
    tensor = np.stack(analysis).astype(np.float32)
    return tensor, timestamps, audit


def _positive_spectrum_frequency(
    values: np.ndarray,
    times_seconds: np.ndarray,
) -> tuple[float, float]:
    """Return dominant and centroid temporal frequencies using actual elapsed time."""
    series = np.asarray(values, dtype=float)
    times = np.asarray(times_seconds, dtype=float)
    if len(series) < 3 or len(times) != len(series) or not np.all(np.diff(times) > 0):
        raise ValueError("Temporal spectrum requires at least three strictly ordered samples")
    uniform_times = np.linspace(times[0], times[-1], len(times), dtype=float)
    uniform = np.interp(uniform_times, times, series)
    x = np.arange(len(uniform), dtype=float)
    coefficients = np.polyfit(x, uniform, 1)
    detrended = uniform - np.polyval(coefficients, x)
    detrended *= np.hanning(len(detrended))
    step = float(np.mean(np.diff(uniform_times)))
    frequencies = np.fft.rfftfreq(len(detrended), d=step)
    power = np.abs(np.fft.rfft(detrended)) ** 2
    frequencies, power = frequencies[1:], power[1:]
    total = float(np.sum(power))
    if not len(power) or total <= np.finfo(float).eps:
        return 0.0, 0.0
    return (
        float(frequencies[int(np.argmax(power))]),
        float(np.sum(frequencies * power) / total),
    )


def _spatial_spectrum_features(frames: np.ndarray) -> tuple[float, float]:
    """Return radial dominant and centroid frequencies in cycles per image diagonal."""
    count, height, width = frames.shape
    diagonal = float(np.hypot(height, width))
    window = np.hanning(height)[:, None] * np.hanning(width)[None, :]
    power = np.zeros((height, width), dtype=np.float64)
    for frame in frames:
        centered = (frame.astype(np.float64) - float(np.mean(frame))) * window
        power += np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    power /= float(max(1, count))
    fy = np.fft.fftshift(np.fft.fftfreq(height))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(width))[None, :]
    radial = np.sqrt(fx**2 + fy**2) * diagonal
    keep = radial > 0
    radial_values = radial[keep]
    weights = power[keep]
    total = float(np.sum(weights))
    if total <= np.finfo(float).eps:
        return 0.0, 0.0
    centroid = float(np.sum(radial_values * weights) / total)
    edges = np.linspace(0.0, float(np.max(radial_values)), 65)
    radial_power, _ = np.histogram(radial_values, bins=edges, weights=weights)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dominant = float(centers[int(np.argmax(radial_power))])
    return dominant, centroid


def _structure_coherence(frames: np.ndarray, times_seconds: np.ndarray) -> float:
    """Compute dimensionless coherence of the normalized 3-D gradient tensor."""
    _, height, width = frames.shape
    diagonal = float(np.hypot(height, width))
    gx = np.gradient(frames.astype(np.float64), axis=2) * diagonal
    gy = np.gradient(frames.astype(np.float64), axis=1) * diagonal
    gt = np.gradient(frames.astype(np.float64), times_seconds, axis=0)
    gradients = np.stack((gx, gy, gt), axis=-1).reshape(-1, 3)
    scales = np.sqrt(np.mean(gradients**2, axis=0))
    scales = np.where(scales > 1e-12, scales, 1.0)
    normalized = gradients / scales
    tensor = normalized.T @ normalized / float(max(1, len(normalized)))
    eigenvalues = np.sort(np.linalg.eigvalsh(tensor))[::-1]
    total = float(np.sum(eigenvalues))
    if total <= np.finfo(float).eps:
        return 0.0
    coherence = (float(eigenvalues[0]) - float(np.mean(eigenvalues[1:]))) / total
    return float(np.clip(coherence, 0.0, 1.0))


def _spectral_entropy_3d(frames: np.ndarray) -> float:
    """Normalized entropy of the windowed spatiotemporal power spectrum."""
    count, height, width = frames.shape
    centered = frames.astype(np.float64) - np.mean(frames, axis=0, keepdims=True)
    window = (
        np.hanning(count)[:, None, None]
        * np.hanning(height)[None, :, None]
        * np.hanning(width)[None, None, :]
    )
    power = np.abs(np.fft.rfftn(centered * window)) ** 2
    flat = power.reshape(-1)
    total = float(np.sum(flat))
    if total <= np.finfo(float).eps:
        return 0.0
    probabilities = flat[flat > 0] / total
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    maximum = float(np.log(len(flat))) if len(flat) > 1 else 1.0
    return float(np.clip(entropy / maximum, 0.0, 1.0))


def dynamic_texture_descriptor(
    analysis_frames: np.ndarray,
    timestamps_ms: np.ndarray,
    settings: DynamicTextureSettings | None = None,
) -> dict[str, float]:
    """Calculate the twelve ``dynamic_texture_v1`` values.

    The training boundary is intentionally grayscale-only.  Passing an RGB
    tensor raises instead of silently creating a new representation.
    """
    import cv2

    cfg = settings or DynamicTextureSettings()
    frames = np.asarray(analysis_frames)
    if frames.ndim == 4:
        raise ValueError("RGB tensors are rejected by the dynamic-texture training interface")
    if frames.ndim != 3 or frames.shape[0] != cfg.frames_per_window:
        raise ValueError(
            "Dynamic-texture descriptors require a grayscale [16, height, width] tensor"
        )
    if frames.shape[1:] != (cfg.analysis_size, cfg.analysis_size):
        raise ValueError(
            f"Dynamic-texture analysis frames must be {cfg.analysis_size}x{cfg.analysis_size}"
        )
    times = np.asarray(timestamps_ms, dtype=np.float64) / 1000.0
    if times.shape != (cfg.frames_per_window,) or not np.all(np.diff(times) > 0):
        raise ValueError("Selected frame timestamps must be distinct and strictly increasing")
    diagonal = float(np.hypot(frames.shape[1], frames.shape[2]))
    all_speed: list[np.ndarray] = []
    energy: list[float] = []
    midpoints: list[float] = []
    direction_x = 0.0
    direction_y = 0.0
    direction_weight = 0.0
    for index in range(1, len(frames)):
        dt = float(times[index] - times[index - 1])
        flow = cv2.calcOpticalFlowFarneback(
            (frames[index - 1] * 255.0).astype(np.float32),
            (frames[index] * 255.0).astype(np.float32),
            None,
            cfg.farneback_pyr_scale,
            cfg.farneback_levels,
            cfg.farneback_window_size,
            cfg.farneback_iterations,
            cfg.farneback_poly_n,
            cfg.farneback_poly_sigma,
            0,
        )
        velocity = flow.astype(np.float64) / (dt * diagonal)
        speed = np.linalg.norm(velocity, axis=2)
        all_speed.append(speed.reshape(-1))
        energy.append(float(np.mean(speed)))
        midpoints.append(0.5 * (times[index] + times[index - 1]))
        direction_x += float(np.sum(velocity[:, :, 0]))
        direction_y += float(np.sum(velocity[:, :, 1]))
        direction_weight += float(np.sum(speed))
    speed_values = np.concatenate(all_speed)
    mean_speed = float(np.mean(speed_values))
    p90_speed = float(np.quantile(speed_values, 0.90))
    if direction_weight > np.finfo(float).eps:
        direction_cos = direction_x / direction_weight
        direction_sin = direction_y / direction_weight
    else:
        direction_cos = 0.0
        direction_sin = 0.0
    resultant = float(np.clip(np.hypot(direction_cos, direction_sin), 0.0, 1.0))
    temporal_dominant, temporal_centroid = _positive_spectrum_frequency(
        np.asarray(energy), np.asarray(midpoints)
    )
    spatial_dominant, spatial_centroid = _spatial_spectrum_features(frames)
    energy_array = np.asarray(energy, dtype=float)
    energy_mean = float(np.mean(energy_array))
    energy_cv = float(np.std(energy_array) / max(energy_mean, np.finfo(float).eps))
    stability = float(1.0 / (1.0 + energy_cv))
    values = (
        mean_speed,
        p90_speed,
        temporal_dominant,
        temporal_centroid,
        spatial_dominant,
        spatial_centroid,
        float(direction_cos),
        float(direction_sin),
        resultant,
        _structure_coherence(frames, times),
        _spectral_entropy_3d(frames),
        stability,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("dynamic_texture_v1 produced a non-finite descriptor")
    return dict(zip(DYNAMIC_TEXTURE_COLUMNS, (float(value) for value in values), strict=True))


def _frame_targets(start_ms: float, end_ms: float, count: int) -> np.ndarray:
    targets = np.linspace(start_ms, end_ms, num=count, endpoint=False, dtype=float)
    return targets + (end_ms - start_ms) / (2.0 * count)


def _condition_zero_windows(mask_frame: Any) -> list[dict[str, str]]:
    valid = mask_frame.groupby(["participant_id", "condition"], sort=True)["common_valid"].sum()
    return [
        {"participant_id": str(participant), "condition": str(condition)}
        for (participant, condition), count in valid.items()
        if int(count) == 0
    ]


def _default_contract_dir() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "artifacts"
        / "cross_project_alignment_2026-07-16"
        / "eeg_eligible_ablation"
        / "contract"
    )


def extract_dynamic_texture_features(
    config: ProjectConfig,
    participants: list[str] | None = None,
    *,
    contract_dir: str | Path | None = None,
    base_window_features: str | Path | None = None,
    output_dir: str | Path | None = None,
    force: bool = False,
    settings: DynamicTextureSettings | None = None,
) -> dict[str, Any]:
    """Extract, mask, merge, aggregate, and audit the frozen 9-person cohort."""
    import pandas as pd

    cfg = settings or DynamicTextureSettings()
    contract_root = Path(
        contract_dir or config.get("dynamic_texture.contract_dir") or _default_contract_dir()
    ).resolve()
    contract = validate_alignment_contract(contract_root)
    cohort = list(participants or contract.get("participants", ()))
    if sorted(cohort) != sorted(contract.get("participants", ())):
        raise ValueError("Formal dynamic-texture extraction requires the complete frozen cohort")
    files = contract["files"]
    windows_path = contract_root / files["windows"]["path"]
    masks_path = contract_root / files["common_masks"]["path"]
    labels_path = contract_root / files["labels"]["path"]
    if base_window_features is None:
        base_window_features = config.get("dynamic_texture.base_window_features")
    if base_window_features is None:
        base_window_features = (
            contract_root.parent / "project_a" / "inputs" / "project_a_window_features_common.csv"
        )
    base_path = Path(base_window_features).resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"Project A common-window features are missing: {base_path}")
    destination = Path(
        output_dir
        or config.get("dynamic_texture.output_dir")
        or (config.path("features") / DYNAMIC_TEXTURE_VERSION)
    ).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "window_descriptors": destination / "window_dynamic_texture_v1.csv",
        "frame_selection_audit": destination / "frame_selection_audit.csv",
        "selected_frame_hashes": destination / "selected_frame_hashes.csv",
        "common_mask_merge": destination / "common_mask_merged.csv",
        "merged_window_features": destination / "project_a_window_features_dynamic_texture.csv",
        "condition_features": destination / "condition_dynamic_texture_v1.csv",
        "manifest": destination / "dynamic_texture_manifest.json",
        "config": destination / "dynamic_texture_config.json",
    }
    if not force and output_paths["manifest"].is_file():
        manifest = json.loads(output_paths["manifest"].read_text(encoding="utf-8"))
        expected_sources = {
            "contract": file_sha256(contract_root / "contract.json"),
            "windows": file_sha256(windows_path),
            "masks": file_sha256(masks_path),
            "labels": file_sha256(labels_path),
            "base_window_features": file_sha256(base_path),
        }
        if manifest.get("source_sha256") == expected_sources and all(
            path.is_file() for name, path in output_paths.items() if name != "manifest"
        ):
            return {
                "reused": True,
                **manifest,
                "outputs": {key: str(value) for key, value in output_paths.items()},
            }

    windows = pd.read_csv(windows_path)
    masks = pd.read_csv(masks_path)
    labels = pd.read_csv(labels_path)
    base = pd.read_csv(base_path)
    for frame, name in ((windows, "windows"), (masks, "masks"), (base, "base features")):
        if frame.duplicated(list(KEY_COLUMNS)).any():
            raise ValueError(f"Frozen {name} contains duplicate window keys")
    if len(windows) != 567 or len(masks) != 567 or len(base) != 567:
        raise ValueError("Frozen source tables must each contain exactly 567 windows")
    if len(labels) != 81 or labels.duplicated(["participant_id", "condition"]).any():
        raise ValueError("Frozen label table must contain 81 unique participant-condition rows")
    masks["common_valid"] = masks["common_valid"].astype(str).str.lower().isin({"true", "1", "1.0"})
    if int(masks["common_valid"].sum()) != 545 or int((~masks["common_valid"]).sum()) != 22:
        raise ValueError("Frozen common mask must contain 545 valid and 22 masked windows")
    zero_windows = _condition_zero_windows(masks)
    if zero_windows != [{"participant_id": "P004", "condition": "C6"}]:
        raise ValueError(f"Unexpected zero-common-window observations: {zero_windows}")
    merged_keys = windows.merge(
        masks[[*KEY_COLUMNS, "common_valid"]], on=list(KEY_COLUMNS), validate="one_to_one"
    )
    if len(merged_keys) != 567:
        raise ValueError("Window and common-mask keys do not align one-to-one")

    raw_root = config.path("raw_root")
    video_indexes = {}
    video_logs: dict[str, Path] = {}
    for participant in cohort:
        participant_dir = raw_root / participant
        session_dir = discover_session_dir(participant_dir) if participant_dir.is_dir() else None
        video_log = session_dir / "video_frames.csv" if session_dir else None
        index = load_video_index(video_log, session_dir, participant)
        if index.reason != "ok":
            raise ValueError(f"{participant} video index is unavailable: {index.reason}")
        video_indexes[participant] = index
        assert video_log is not None
        video_logs[participant] = video_log

    descriptor_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    frame_hash_rows: list[dict[str, Any]] = []
    hash_cache: dict[Path, str] = {}
    for row in merged_keys.sort_values(
        ["participant_id", "presentation_position", "condition_window_index"]
    ).itertuples(index=False):
        participant = str(row.participant_id)
        condition = str(row.condition)
        window_index = int(row.condition_window_index)
        start_ms = float(row.window_start_unix_ms)
        end_ms = float(row.window_end_unix_ms)
        common_valid = bool(row.common_valid)
        selected = uniform_clip_frames(
            video_indexes[participant], start_ms, end_ms, cfg.frames_per_window
        )
        targets = _frame_targets(start_ms, end_ms, cfg.frames_per_window)
        selected_times = np.asarray([frame.unix_time_ms for frame in selected], dtype=float)
        base_record: dict[str, Any] = {
            "participant_id": participant,
            "condition": condition,
            "condition_window_index": window_index,
            "window_start_unix_ms": int(round(start_ms)),
            "window_end_unix_ms": int(round(end_ms)),
            "common_valid": common_valid,
            "video_timestamp_source": video_indexes[participant].timestamp_source,
            "selected_frame_count": int(len(selected)),
            "selected_frame_indexes": json.dumps([item.frame_index for item in selected]),
            "selected_frame_timestamps_ms": json.dumps([item.unix_time_ms for item in selected]),
        }
        if len(selected) == cfg.frames_per_window:
            offsets = np.abs(selected_times - targets)
            intervals = np.diff(selected_times)
            base_record.update(
                {
                    "nearest_target_offset_ms_mean": float(np.mean(offsets)),
                    "nearest_target_offset_ms_max": float(np.max(offsets)),
                    "frame_interval_ms_mean": float(np.mean(intervals)),
                    "frame_interval_ms_min": float(np.min(intervals)),
                    "frame_interval_ms_max": float(np.max(intervals)),
                    "selected_frame_indexes_unique": len({item.frame_index for item in selected})
                    == cfg.frames_per_window,
                }
            )
        else:
            base_record.update(
                {
                    "nearest_target_offset_ms_mean": float("nan"),
                    "nearest_target_offset_ms_max": float("nan"),
                    "frame_interval_ms_mean": float("nan"),
                    "frame_interval_ms_min": float("nan"),
                    "frame_interval_ms_max": float("nan"),
                    "selected_frame_indexes_unique": False,
                }
            )
        descriptor = {name: float("nan") for name in DYNAMIC_TEXTURE_COLUMNS}
        descriptor_record = {
            **{name: getattr(row, name) for name in KEY_COLUMNS},
            "common_valid": common_valid,
            "video_dynamic_validity": 0.0,
            **descriptor,
        }
        if not common_valid:
            base_record["descriptor_status"] = "masked_by_frozen_common_mask"
        else:
            if len(selected) != cfg.frames_per_window:
                raise ValueError(
                    f"Common-valid window {participant}/{condition}/W{window_index:02d} "
                    f"has {len(selected)} selected frames, expected {cfg.frames_per_window}"
                )
            analysis, timestamps, decoded_audit = load_analysis_clip(selected, cfg)
            descriptor = dynamic_texture_descriptor(analysis, timestamps, cfg)
            descriptor_record.update(descriptor)
            descriptor_record["video_dynamic_validity"] = 1.0
            base_record["descriptor_status"] = "ok"
            base_record["analysis_tensor_sha256"] = _stable_array_hash(analysis)
            base_record["analysis_width"] = int(analysis.shape[2])
            base_record["analysis_height"] = int(analysis.shape[1])
            base_record["decoded_source_shapes"] = json.dumps(
                [[item["source_height"], item["source_width"]] for item in decoded_audit]
            )
        for slot, selected_frame in enumerate(selected):
            source_path = selected_frame.path.resolve()
            if source_path not in hash_cache:
                hash_cache[source_path] = file_sha256(source_path)
            frame_hash_rows.append(
                {
                    "participant_id": participant,
                    "condition": condition,
                    "condition_window_index": window_index,
                    "slot": int(slot),
                    "frame_index": int(selected_frame.frame_index),
                    "frame_unix_ms": int(selected_frame.unix_time_ms),
                    "path": str(source_path),
                    "sha256": hash_cache[source_path],
                }
            )
        selection_rows.append(base_record)
        descriptor_rows.append(descriptor_record)

    descriptors = pd.DataFrame(descriptor_rows)
    finite = np.isfinite(descriptors[list(DYNAMIC_TEXTURE_COLUMNS)].to_numpy(dtype=float))
    valid_rows = descriptors["common_valid"].astype(bool).to_numpy()
    if not finite[valid_rows].all() or finite[~valid_rows].any():
        raise ValueError("Descriptor finiteness does not exactly match the frozen common mask")
    if int(descriptors["video_dynamic_validity"].sum()) != 545:
        raise ValueError("Dynamic-texture validity stream must contain exactly 545 valid windows")

    merged_window = base.merge(
        descriptors[[*KEY_COLUMNS, "video_dynamic_validity", *DYNAMIC_TEXTURE_COLUMNS]],
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    if len(merged_window) != 567:
        raise ValueError("Dynamic-texture merge changed the frozen source-window count")
    condition_source = merged_window[
        [
            *[
                name
                for name in merged_window.columns
                if name
                in {
                    "participant_id",
                    "condition",
                    "condition_window_index",
                    "condition_window_count",
                    "presentation_position",
                    "intensity",
                    "frequency",
                    "intensity_index",
                    "frequency_index",
                    "condition_index",
                    "relaxation",
                    "discomfort",
                    "calm",
                    "pleasantness",
                    "monotony",
                    "visual_fit",
                    "relaxation_raw",
                    "discomfort_raw",
                    "arousal_raw",
                    "pleasantness_raw",
                    "monotony_raw",
                    "label_source_row",
                }
            ],
            *DYNAMIC_TEXTURE_COLUMNS,
        ]
    ].copy()
    condition_features = aggregate_window_frame(condition_source)
    video_aggregates = [
        name for name in condition_features.columns if name.startswith("video_dynamic_")
    ]
    if len(condition_features) != 81 or len(video_aggregates) != 108:
        raise ValueError(
            "Condition aggregation must produce 81 rows and exactly 108 dynamic-video candidates"
        )

    descriptors.to_csv(output_paths["window_descriptors"], index=False)
    pd.DataFrame(selection_rows).to_csv(output_paths["frame_selection_audit"], index=False)
    pd.DataFrame(frame_hash_rows).to_csv(output_paths["selected_frame_hashes"], index=False)
    merged_keys[[*KEY_COLUMNS, "common_valid"]].merge(
        descriptors[[*KEY_COLUMNS, "video_dynamic_validity"]],
        on=list(KEY_COLUMNS),
        validate="one_to_one",
    ).to_csv(output_paths["common_mask_merge"], index=False)
    merged_window.to_csv(output_paths["merged_window_features"], index=False)
    condition_features.to_csv(output_paths["condition_features"], index=False)
    for name in ("window_descriptors", "merged_window_features", "condition_features"):
        try:
            pd.read_csv(output_paths[name]).to_parquet(
                output_paths[name].with_suffix(".parquet"), index=False
            )
        except (ImportError, ValueError):
            pass
    config_payload = {
        "schema_version": DYNAMIC_TEXTURE_VERSION,
        "settings": asdict(cfg),
        "descriptor_columns": list(DYNAMIC_TEXTURE_COLUMNS),
        "normalization": {
            "motion": "Farneback displacement / actual frame interval / 112x112 image diagonal",
            "temporal_frequency": "Hz from actual selected-frame timestamps",
            "spatial_frequency": "cycles per 112x112 image diagonal",
            "direction": "magnitude-weighted unit-vector components",
            "structure_coherence": "eigenvalue coherence of RMS-normalized x/y/time gradients",
            "spectral_entropy": "Shannon entropy / log(number of 3-D Fourier bins)",
            "motion_stability": "1 / (1 + coefficient of variation of interval motion energy)",
        },
        "training_contract": {
            "grayscale_shape": [cfg.frames_per_window, cfg.analysis_size, cfg.analysis_size],
            "rgb_tensor_rejected": True,
            "frozen_mask_unchanged": True,
        },
    }
    write_json(output_paths["config"], config_payload)
    source_hashes = {
        "contract": file_sha256(contract_root / "contract.json"),
        "windows": file_sha256(windows_path),
        "masks": file_sha256(masks_path),
        "labels": file_sha256(labels_path),
        "base_window_features": file_sha256(base_path),
    }
    manifest = {
        "schema_version": "dynamic_texture_extraction_manifest_v1",
        "feature_version": DYNAMIC_TEXTURE_VERSION,
        "source_sha256": source_hashes,
        "video_log_sha256": {
            participant: file_sha256(path) for participant, path in video_logs.items()
        },
        "participants": cohort,
        "source_windows": int(len(windows)),
        "common_valid_windows": int(valid_rows.sum()),
        "masked_windows": int((~valid_rows).sum()),
        "participant_condition_labels": int(len(condition_features)),
        "descriptor_count": len(DYNAMIC_TEXTURE_COLUMNS),
        "condition_video_candidate_count": len(video_aggregates),
        "zero_common_window_observations": zero_windows,
        "selected_frame_rows": len(frame_hash_rows),
        "unique_selected_frames": len(hash_cache),
        "outputs": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for name, path in output_paths.items()
            if name != "manifest"
        },
    }
    write_json(output_paths["manifest"], manifest)
    return {
        "reused": False,
        **manifest,
        "outputs": {key: str(value) for key, value in output_paths.items()},
    }


__all__ = [
    "DYNAMIC_TEXTURE_COLUMNS",
    "DYNAMIC_TEXTURE_VERSION",
    "DynamicTextureSettings",
    "dynamic_texture_descriptor",
    "extract_dynamic_texture_features",
    "load_analysis_clip",
]
