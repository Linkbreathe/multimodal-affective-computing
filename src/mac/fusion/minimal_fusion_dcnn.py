"""Research-only minimal multimodal LOPO 1DCNN fusion benchmark.

This benchmark intentionally has one input boundary: the handcrafted
participant--Condition window table.  It reuses the temporal CNN architecture
but is isolated from runtime model selection, saved checkpoints, VideoMAE2,
Unity, and the Shadow/hold policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2s
from pathlib import Path
from typing import Any

import numpy as np

from real_time_ml.config import ProjectConfig
from real_time_ml.modeling.dcnn import (
    _architecture,
    _device,
    _make_model,
    _torch,
    _validation_indexes,
)
from real_time_ml.modeling.minimal_fusion import (
    COMBINATIONS,
    EXPECTED_LABELS,
    FEATURES_PER_MODALITY,
    HIGH_DISCOMFORT_PREDICTION_THRESHOLD,
    HIGH_DISCOMFORT_TRUTH_THRESHOLD,
    MODALITY_ORDER,
    MODALITY_PREFIXES,
    RANDOM_SIMULATIONS,
    TARGETS,
    _condition_baseline,
    _dependencies,
    _format_number,
    _history_baseline,
    _improvements,
    _point_metrics,
    _selection_summary,
    _wide_metrics,
    _wide_random,
    random_uniform_baseline,
)
from real_time_ml.utils import atomic_write_text


SOURCE_RELATIVE_PATH = Path("video_ml") / "window_features.csv"
OUTPUT_DIRECTORY = "fusion_minimal_dcnn"
METRICS_FILENAME = "metrics.csv"
OOF_FILENAME = "oof_predictions.csv"
REPORT_FILENAME = "minimal_multimodal_fusion_dcnn_zh.md"

REQUIRED_COLUMNS = (
    "participant_id",
    "condition",
    "presentation_position",
    "condition_window_index",
    *TARGETS,
)


@dataclass(frozen=True)
class MinimalFusionSequences:
    """One padded temporal sample per participant--Condition label."""

    values: np.ndarray
    targets: np.ndarray
    participant_ids: np.ndarray
    conditions: np.ndarray
    presentation_positions: np.ndarray
    lengths: np.ndarray
    feature_columns: tuple[str, ...]


def _window_modal_columns(frame: Any, modality: str) -> list[str]:
    prefixes = MODALITY_PREFIXES[modality]
    return sorted(name for name in frame.columns if name.startswith(prefixes))


def _validate_window_input(frame: Any, *, sequence_length: int, expected_labels: int | None) -> Any:
    pd, *_ = _dependencies()
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"Minimal fusion 1DCNN input is missing required columns: {missing}")
    if sequence_length < 1:
        raise ValueError("Minimal fusion 1DCNN sequence_length must be positive")

    frame = frame.copy()
    frame["participant_id"] = frame["participant_id"].astype(str)
    frame["condition"] = frame["condition"].astype(str)
    numeric = ("presentation_position", "condition_window_index", *TARGETS)
    for name in numeric:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Minimal fusion 1DCNN input has missing identifiers, ordering, windows, or labels")

    for modality in MODALITY_ORDER:
        if not _window_modal_columns(frame, modality):
            raise ValueError(f"Minimal fusion 1DCNN input has no {modality} prefixed features")

    label_count = 0
    for (participant_id, condition), group in frame.groupby(["participant_id", "condition"], sort=True):
        label_count += 1
        if group[list(TARGETS)].nunique(dropna=False).gt(1).any():
            raise ValueError(
                f"Inconsistent inherited questionnaire labels for {participant_id}/{condition}"
            )
        if group["presentation_position"].nunique(dropna=False) != 1:
            raise ValueError(f"Inconsistent presentation_position for {participant_id}/{condition}")
        indexes = group["condition_window_index"].to_numpy(dtype=float)
        if not np.allclose(indexes, np.rint(indexes)):
            raise ValueError(f"Non-integer condition_window_index for {participant_id}/{condition}")
        indexes = indexes.astype(int)
        if (
            indexes.min() < 0
            or indexes.max() >= sequence_length
            or len(np.unique(indexes)) != len(indexes)
        ):
            raise ValueError(
                f"Invalid 0..{sequence_length - 1} window indexes for {participant_id}/{condition}"
            )
    if expected_labels is not None and label_count != expected_labels:
        raise ValueError(
            f"Expected exactly {expected_labels} participant--Condition labels; found {label_count}"
        )
    if frame["participant_id"].nunique() < 2:
        raise ValueError("Minimal fusion 1DCNN LOPO requires at least two participants")
    return frame.reset_index(drop=True)


def build_minimal_fusion_sequences(
    frame: Any, *, sequence_length: int, expected_labels: int | None = EXPECTED_LABELS
) -> MinimalFusionSequences:
    """Convert the permitted P/H/E/V window features to padded Condition sequences."""
    pd, *_ = _dependencies()
    frame = _validate_window_input(
        frame, sequence_length=sequence_length, expected_labels=expected_labels
    )
    feature_columns = tuple(
        sorted(
            name
            for modality in MODALITY_ORDER
            for name in _window_modal_columns(frame, modality)
        )
    )
    values: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    participants: list[str] = []
    conditions: list[str] = []
    positions: list[float] = []
    lengths: list[int] = []
    for (participant_id, condition), group in frame.groupby(["participant_id", "condition"], sort=True):
        ordered = group.sort_values("condition_window_index", kind="stable")
        indexes = ordered["condition_window_index"].to_numpy(dtype=int)
        matrix = np.full((len(feature_columns), sequence_length), np.nan, dtype=float)
        numeric = ordered.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
        matrix[:, indexes] = numeric.to_numpy(dtype=float).T
        values.append(matrix)
        targets.append(ordered.iloc[0][list(TARGETS)].to_numpy(dtype=float))
        participants.append(str(participant_id))
        conditions.append(str(condition))
        positions.append(float(ordered.iloc[0]["presentation_position"]))
        lengths.append(int(indexes.max() + 1))
    return MinimalFusionSequences(
        values=np.stack(values),
        targets=np.stack(targets),
        participant_ids=np.asarray(participants, dtype=str),
        conditions=np.asarray(conditions, dtype=str),
        presentation_positions=np.asarray(positions, dtype=float),
        lengths=np.asarray(lengths, dtype=int),
        feature_columns=feature_columns,
    )


def _labels_frame(sequences: MinimalFusionSequences) -> Any:
    pd, *_ = _dependencies()
    return pd.DataFrame(
        {
            "participant_id": sequences.participant_ids,
            "condition": sequences.conditions,
            "presentation_position": sequences.presentation_positions,
            "relaxation": sequences.targets[:, 0],
            "discomfort": sequences.targets[:, 1],
        }
    )


def _modal_feature_indexes(sequences: MinimalFusionSequences, modality: str) -> list[int]:
    prefixes = MODALITY_PREFIXES[modality]
    return [
        index
        for index, name in enumerate(sequences.feature_columns)
        if name.startswith(prefixes)
    ]


def _rank_window_features(
    sequences: MinimalFusionSequences,
    train_indexes: np.ndarray,
    feature_indexes: list[int],
    residual: np.ndarray,
    *,
    limit: int,
) -> list[str]:
    """Rank only outer-fold windows by absolute Pearson correlation to residual labels."""
    ranked: list[tuple[float, str]] = []
    repeated_residual = np.repeat(np.asarray(residual, dtype=float), sequences.values.shape[2])
    for feature_index in feature_indexes:
        values = sequences.values[train_indexes, feature_index, :].reshape(-1)
        valid = np.isfinite(values) & np.isfinite(repeated_residual)
        if valid.sum() < 2:
            continue
        feature_values = values[valid]
        residual_values = repeated_residual[valid]
        if np.std(feature_values) <= 0.0 or np.std(residual_values) <= 0.0:
            continue
        correlation = float(np.corrcoef(feature_values, residual_values)[0, 1])
        if np.isfinite(correlation):
            ranked.append((abs(correlation), sequences.feature_columns[feature_index]))
    return [name for _, name in sorted(ranked, key=lambda item: (-item[0], item[1]))[:limit]]


def _selected_by_modality(
    sequences: MinimalFusionSequences,
    train_indexes: np.ndarray,
    residual: np.ndarray,
) -> dict[str, list[str]]:
    return {
        modality: _rank_window_features(
            sequences,
            train_indexes,
            _modal_feature_indexes(sequences, modality),
            residual,
            limit=FEATURES_PER_MODALITY,
        )
        for modality in MODALITY_ORDER
    }


def _combine_selected_features(
    selected_by_modality: dict[str, list[str]], combination: str
) -> tuple[list[str], dict[str, int]]:
    selected: list[str] = []
    counts: dict[str, int] = {}
    for modality in MODALITY_ORDER:
        retained = selected_by_modality[modality] if modality in combination else []
        selected.extend(retained)
        counts[modality] = len(retained)
    return sorted(selected), counts


def _fit_sequence_scaler(
    sequences: MinimalFusionSequences, train_indexes: np.ndarray, feature_indexes: np.ndarray
) -> dict[str, np.ndarray]:
    """Fit median imputation and standardization strictly on model-training windows."""
    raw = sequences.values[train_indexes][:, feature_indexes, :]
    medians = np.zeros(len(feature_indexes), dtype=float)
    scales = np.ones(len(feature_indexes), dtype=float)
    for position in range(len(feature_indexes)):
        finite = raw[:, position, :][np.isfinite(raw[:, position, :])]
        if finite.size:
            median = float(np.median(finite))
            imputed = np.where(np.isfinite(raw[:, position, :]), raw[:, position, :], median)
            scale = float(np.std(imputed))
            medians[position] = median
            scales[position] = scale if np.isfinite(scale) and scale > 1e-8 else 1.0
    return {"feature_median": medians, "feature_scale": scales}


def _transform_sequences(
    sequences: MinimalFusionSequences,
    indexes: np.ndarray,
    feature_indexes: np.ndarray,
    scaler: dict[str, np.ndarray],
    prefix_lengths: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = sequences.values[indexes][:, feature_indexes, :].copy()
    if prefix_lengths is not None:
        for position, prefix in enumerate(np.asarray(prefix_lengths, dtype=int)):
            values[position, :, max(0, int(prefix)) :] = np.nan
    medians = np.asarray(scaler["feature_median"], dtype=float)[None, :, None]
    scales = np.asarray(scaler["feature_scale"], dtype=float)[None, :, None]
    values = np.where(np.isfinite(values), values, medians)
    standardized = ((values - medians) / scales).astype(np.float32)
    return standardized, np.zeros((len(indexes), 0), dtype=np.float32)


def _fold_seed(random_seed: int, target: str, combination: str, fold: int) -> int:
    payload = f"{int(random_seed)}|{target}|{combination}|{int(fold)}".encode("utf-8")
    return int.from_bytes(blake2s(payload, digest_size=4).digest(), "little")


def _set_torch_seed(seed: int, device) -> None:
    torch = _torch()
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, RuntimeError):  # pragma: no cover - version-specific torch behavior
        pass


def _train_residual_model(
    sequences: MinimalFusionSequences,
    train_indexes: np.ndarray,
    validation_indexes: np.ndarray,
    feature_indexes: np.ndarray,
    residual: np.ndarray,
    architecture: dict[str, Any],
    config: ProjectConfig,
    seed: int,
    device,
):
    """Use the standard DCNN optimizer, random prefixes, and early stopping."""
    torch = _torch()
    node = dict(config.get("modeling.dcnn", {}))
    _set_torch_seed(seed, device)
    scaler = _fit_sequence_scaler(sequences, train_indexes, feature_indexes)
    model = _make_model(
        len(feature_indexes),
        architecture,
        context_size=0,
        output_size=1,
        output_activation="tanh",
        stream_execution="grouped",
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(node.get("learning_rate", 1e-4)),
        weight_decay=float(node.get("weight_decay", 5e-5)),
    )
    loss_fn = torch.nn.MSELoss()
    # A LOPO core fold has only 117 Condition sequences.  Keep its complete
    # random-prefix draw together so the fixed research protocol remains
    # tractable across 450 independently seeded models; architecture, AdamW,
    # learning rate, random prefixes, and early stopping are unchanged.
    batch_size = max(int(node.get("batch_size", 16)), len(train_indexes))
    epochs = int(node.get("max_epochs", 100))
    patience = int(node.get("early_stopping_patience", 15))
    rng = np.random.default_rng(seed)
    validation_indexes = validation_indexes if len(validation_indexes) else train_indexes
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    for _ in range(epochs):
        prefixes = np.asarray(
            [rng.integers(1, int(sequences.lengths[index]) + 1) for index in train_indexes], dtype=int
        )
        train_values, train_context = _transform_sequences(
            sequences, train_indexes, feature_indexes, scaler, prefixes
        )
        order = rng.permutation(len(train_indexes))
        model.train()
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            values = torch.as_tensor(train_values[batch], device=device)
            context = torch.as_tensor(train_context[batch], device=device)
            targets = torch.as_tensor(
                residual[train_indexes][batch, None], dtype=torch.float32, device=device
            )
            optimizer.zero_grad()
            loss = loss_fn(model(values, context), targets)
            loss.backward()
            optimizer.step()
        validation_values, validation_context = _transform_sequences(
            sequences, validation_indexes, feature_indexes, scaler
        )
        with torch.no_grad():
            model.eval()
            prediction = model(
                torch.as_tensor(validation_values, device=device),
                torch.as_tensor(validation_context, device=device),
            )
            validation_loss = float(
                loss_fn(
                    prediction,
                    torch.as_tensor(
                        residual[validation_indexes, None], dtype=torch.float32, device=device
                    ),
                ).item()
            )
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("Minimal fusion 1DCNN training did not produce a model state")
    model.load_state_dict(best_state)
    model.eval()
    return model, scaler, best_loss


def _predict_residual_model(
    model,
    sequences: MinimalFusionSequences,
    indexes: np.ndarray,
    feature_indexes: np.ndarray,
    scaler: dict[str, np.ndarray],
    device,
) -> np.ndarray:
    torch = _torch()
    values, context = _transform_sequences(sequences, indexes, feature_indexes, scaler)
    with torch.no_grad():
        return (
            model(
                torch.as_tensor(values, device=device),
                torch.as_tensor(context, device=device),
            )
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(float)
        )


def evaluate_minimal_fusion_dcnn_frame(
    frame: Any,
    config: ProjectConfig,
    *,
    random_simulations: int = RANDOM_SIMULATIONS,
    expected_labels: int | None = EXPECTED_LABELS,
) -> dict[str, Any]:
    """Evaluate the fixed 15-combination temporal residual protocol without file I/O."""
    pd, *_ = _dependencies()
    architecture = _architecture(config)
    sequences = build_minimal_fusion_sequences(
        frame,
        sequence_length=int(architecture["sequence_length"]),
        expected_labels=expected_labels,
    )
    labels = _labels_frame(sequences)
    participants = sorted(labels["participant_id"].unique())
    device = _device(config.get("modeling.dcnn.device", "cuda"))
    random_seed = int(config.get("modeling.random_seed"))
    truth_by_target = {target: labels[target].to_numpy(dtype=float) for target in TARGETS}
    predictions = {
        combination: {target: np.full(len(labels), np.nan, dtype=float) for target in TARGETS}
        for combination in COMBINATIONS
    }
    condition_only = {target: np.full(len(labels), np.nan, dtype=float) for target in TARGETS}
    history = {target: np.full(len(labels), np.nan, dtype=float) for target in TARGETS}
    selections = {
        combination: {target: {modality: [] for modality in MODALITY_ORDER} for target in TARGETS}
        for combination in COMBINATIONS
    }

    for fold, participant in enumerate(participants, start=1):
        test_mask = labels["participant_id"].eq(participant).to_numpy()
        test_indexes = np.flatnonzero(test_mask)
        train_indexes = np.flatnonzero(~test_mask)
        train = labels.iloc[train_indexes].reset_index(drop=True)
        test = labels.iloc[test_indexes].reset_index(drop=True)
        for target_index, target in enumerate(TARGETS):
            baseline, baseline_map, fallback = _condition_baseline(train, test, target)
            condition_only[target][test_indexes] = baseline
            history[target][test_indexes] = _history_baseline(test, fallback, target)
            train_baseline = np.asarray(
                [float(baseline_map.get(condition, fallback)) for condition in train["condition"]],
                dtype=float,
            )
            residual = np.full(len(labels), np.nan, dtype=float)
            residual[train_indexes] = sequences.targets[train_indexes, target_index] - train_baseline
            selected_by_modality = _selected_by_modality(
                sequences, train_indexes, residual[train_indexes]
            )
            for combination in COMBINATIONS:
                columns, counts = _combine_selected_features(selected_by_modality, combination)
                for modality, count in counts.items():
                    selections[combination][target][modality].append(count)
                if not columns:
                    held_out_residual = np.zeros(len(test_indexes), dtype=float)
                else:
                    feature_lookup = {name: index for index, name in enumerate(sequences.feature_columns)}
                    feature_indexes = np.asarray([feature_lookup[name] for name in columns], dtype=int)
                    seed = _fold_seed(random_seed, target, combination, fold)
                    core_indexes, validation_indexes = _validation_indexes(
                        sequences.participant_ids, train_indexes, seed
                    )
                    model, scaler, _ = _train_residual_model(
                        sequences,
                        core_indexes,
                        validation_indexes,
                        feature_indexes,
                        residual,
                        architecture,
                        config,
                        seed,
                        device,
                    )
                    held_out_residual = _predict_residual_model(
                        model, sequences, test_indexes, feature_indexes, scaler, device
                    )
                predictions[combination][target][test_indexes] = np.clip(
                    baseline + held_out_residual, 0.0, 1.0
                )
            print(
                f"Minimal fusion 1DCNN {target} LOPO fold {fold}/{len(participants)}: {participant}",
                flush=True,
            )

    for target in TARGETS:
        if np.isnan(condition_only[target]).any() or np.isnan(history[target]).any():
            raise AssertionError("LOPO baseline predictions must cover every participant--Condition row")
    for combination in COMBINATIONS:
        for target in TARGETS:
            if np.isnan(predictions[combination][target]).any():
                raise AssertionError("LOPO 1DCNN predictions must cover every participant--Condition row")

    baseline_metrics = {
        "condition_only": {
            target: _point_metrics(
                truth_by_target[target], condition_only[target], discomfort=target == "discomfort"
            )
            for target in TARGETS
        },
        "history": {
            target: _point_metrics(
                truth_by_target[target], history[target], discomfort=target == "discomfort"
            )
            for target in TARGETS
        },
    }
    baseline_wide = {name: _wide_metrics(value) for name, value in baseline_metrics.items()}
    random_summary = random_uniform_baseline(
        truth_by_target, random_seed=random_seed, simulations=random_simulations
    )
    random_wide = _wide_random(random_summary)
    random_point = {
        name: value
        for name, value in random_wide.items()
        if not name.endswith(("_mean", "_p025", "_p975"))
    }

    combination_metrics: dict[str, dict[str, float]] = {}
    metric_rows: list[dict[str, Any]] = []
    for baseline_name in ("condition_only", "history"):
        metric_rows.append(
            {
                "record_type": "baseline",
                "baseline": baseline_name,
                "combination": "",
                "research_only": True,
                **baseline_wide[baseline_name],
            }
        )
    metric_rows.append(
        {
            "record_type": "baseline",
            "baseline": "random_uniform",
            "combination": "",
            "research_only": True,
            "random_simulations": int(random_simulations),
            **random_wide,
        }
    )
    for combination in COMBINATIONS:
        point = _wide_metrics(
            {
                target: _point_metrics(
                    truth_by_target[target],
                    predictions[combination][target],
                    discomfort=target == "discomfort",
                )
                for target in TARGETS
            }
        )
        combination_metrics[combination] = point
        metric_rows.append(
            {
                "record_type": "combination",
                "baseline": "",
                "combination": combination,
                "research_only": True,
                "n_labels": len(labels),
                "model_family": "1dcnn_residual",
                "residual_head_activation": "tanh",
                "stream_execution": "grouped",
                "context_size": 0,
                "sequence_length": int(architecture["sequence_length"]),
                "feature_limit_per_modality": FEATURES_PER_MODALITY,
                "high_discomfort_truth_threshold": HIGH_DISCOMFORT_TRUTH_THRESHOLD,
                "high_discomfort_prediction_threshold": HIGH_DISCOMFORT_PREDICTION_THRESHOLD,
                **point,
                **{
                    f"condition_only_{name}": value
                    for name, value in baseline_wide["condition_only"].items()
                },
                **{f"history_{name}": value for name, value in baseline_wide["history"].items()},
                **{f"random_uniform_{name}": value for name, value in random_wide.items()},
                **_improvements(point, random_point, "random_uniform_mean"),
                **_improvements(point, baseline_wide["condition_only"], "condition_only"),
                **_improvements(point, baseline_wide["history"], "history"),
                **_selection_summary(selections[combination]),
            }
        )

    for full_combination in COMBINATIONS:
        if len(full_combination) < 2:
            continue
        for removed_modality in full_combination:
            remaining = "".join(
                modality for modality in full_combination if modality != removed_modality
            )
            full, reduced = combination_metrics[full_combination], combination_metrics[remaining]
            changes = {
                metric: (-1.0 if metric.endswith(("_mae", "_false_negatives")) else 1.0)
                * (full[metric] - reduced[metric])
                for metric in full
            }
            metric_rows.append(
                {
                    "record_type": "ablation",
                    "baseline": "",
                    "combination": full_combination,
                    "full_combination": full_combination,
                    "removed_modality": removed_modality,
                    "remaining_combination": remaining,
                    "research_only": True,
                    **{f"full_{name}": value for name, value in full.items()},
                    **{f"remaining_{name}": value for name, value in reduced.items()},
                    **{f"{name}_improvement": value for name, value in changes.items()},
                }
            )

    oof_rows: list[dict[str, Any]] = []
    for combination in COMBINATIONS:
        for row_index, source_row in labels.iterrows():
            for target in TARGETS:
                truth = float(source_row[target])
                prediction = float(predictions[combination][target][row_index])
                condition_prediction = float(condition_only[target][row_index])
                history_prediction = float(history[target][row_index])
                is_discomfort = target == "discomfort"
                oof_rows.append(
                    {
                        "combination": combination,
                        "participant_id": source_row["participant_id"],
                        "condition": source_row["condition"],
                        "presentation_position": int(source_row["presentation_position"]),
                        "target": target,
                        "truth": truth,
                        "prediction": prediction,
                        "absolute_error": abs(truth - prediction),
                        "condition_only_prediction": condition_prediction,
                        "condition_only_absolute_error": abs(truth - condition_prediction),
                        "history_prediction": history_prediction,
                        "history_absolute_error": abs(truth - history_prediction),
                        "high_discomfort_truth": (
                            int(truth >= HIGH_DISCOMFORT_TRUTH_THRESHOLD)
                            if is_discomfort
                            else np.nan
                        ),
                        "high_discomfort_prediction": (
                            int(prediction >= HIGH_DISCOMFORT_PREDICTION_THRESHOLD)
                            if is_discomfort
                            else np.nan
                        ),
                    }
                )
    return {
        "frame": labels,
        "sequences": sequences,
        "metric_rows": metric_rows,
        "metrics": pd.DataFrame(metric_rows),
        "oof_predictions": pd.DataFrame(oof_rows),
        "combination_metrics": combination_metrics,
        "baseline_metrics": baseline_wide,
        "random_summary": random_summary,
        "random_wide": random_wide,
        "selection_counts": selections,
    }


def _combination_table(
    metrics: dict[str, dict[str, float]], *, key: str, descending: bool = False, percent: bool = False
) -> list[str]:
    ordered = sorted(metrics.items(), key=lambda item: item[1][key], reverse=descending)
    return [
        f"| {rank} | {combination} | {_format_number(values[key], percent=percent)} |"
        for rank, (combination, values) in enumerate(ordered, start=1)
    ]


def _write_report(result: dict[str, Any], output: Path, *, random_simulations: int) -> None:
    baseline, random = result["baseline_metrics"], result["random_summary"]
    combination_metrics, metrics = result["combination_metrics"], result["metrics"]
    ablations = metrics.loc[metrics["record_type"].eq("ablation")]
    baseline_rows = [
        "| 基线 | Relaxation MAE | Relaxation Spearman | Discomfort MAE | Discomfort Spearman | 高 discomfort recall | precision | false negatives |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in baseline.items():
        baseline_rows.append(
            "| {name} | {rel_mae} | {rel_rho} | {dis_mae} | {dis_rho} | {recall} | {precision} | {fn} |".format(
                name="Condition-only" if name == "condition_only" else "history",
                rel_mae=_format_number(values["relaxation_mae"]),
                rel_rho=_format_number(values["relaxation_spearman"]),
                dis_mae=_format_number(values["discomfort_mae"]),
                dis_rho=_format_number(values["discomfort_spearman"]),
                recall=_format_number(values["discomfort_high_recall"], percent=True),
                precision=_format_number(values["discomfort_high_precision"], percent=True),
                fn=int(values["discomfort_high_false_negatives"]),
            )
        )
    random_rows = ["| 指标 | 随机 Uniform(0,1) 均值 [2.5%, 97.5%] |", "|---|---|"]
    for target, metric in (
        ("relaxation", "mae"),
        ("relaxation", "spearman"),
        ("discomfort", "mae"),
        ("discomfort", "spearman"),
        ("discomfort", "high_recall"),
        ("discomfort", "high_precision"),
        ("discomfort", "high_false_negatives"),
    ):
        values = random[target][metric]
        random_rows.append(
            f"| {target} {metric} | {values['mean']:.4f} [{values['p025']:.4f}, {values['p975']:.4f}] |"
        )
    ablation_rows = [
        "| 完整组合 | 移除 | 剩余组合 | Relaxation MAE 改善 | Discomfort MAE 改善 | Recall 改善 | Precision 改善 | 假阴性减少 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in ablations.iterrows():
        ablation_rows.append(
            "| {full} | {removed} | {remaining} | {relaxation:+.4f} | {discomfort:+.4f} | {recall:+.4f} | {precision:+.4f} | {fn:+.0f} |".format(
                full=row["full_combination"],
                removed=row["removed_modality"],
                remaining=row["remaining_combination"],
                relaxation=row["relaxation_mae_improvement"],
                discomfort=row["discomfort_mae_improvement"],
                recall=row["discomfort_high_recall_improvement"],
                precision=row["discomfort_high_precision_improvement"],
                fn=row["discomfort_high_false_negatives_improvement"],
            )
        )
    lines = [
        "# 最小多模态 1DCNN 融合基准（研究比较，不能部署）",
        "",
        "## 输入边界与固定协议",
        "",
        "- 唯一输入是 `artifacts/features/video_ml/window_features.csv` 的手工 P/H/E/V 十秒窗口特征；每个 participant--Condition 最多 8 个窗口。不会读取 `condition_features.csv`、VideoMAE2、其他深度模型、实时或 Unity 产物。",
        "- 模态固定为 P=`eeg_`+`ecg_`、H=`head_`、E=`eye_`、V=`video_`；仅比较 15 个非空 P/H/E/V 组合。标签、问卷原始列、标识、Condition、刺激上下文及 QC 列均不进入模型。",
        "- 外层验证为 LOPO。每折先以其 14 名训练参与者的同 Condition 标签均值建立 Condition-only 基线；每个目标、组合、折单独训练 1DCNN 拟合 `target - baseline`，预测以 `clip(baseline + residual, 0, 1)` 还原。",
        "- 每目标、每模态仅按外层训练折窗口和残差的绝对 Pearson 相关保留最多 20 列；中位数插补与标准化只由该模型训练参与者拟合。模型沿用逐特征 CNN stream、卷积/池化、AdamW、随机前缀训练与早停；头部为单目标 `tanh` 残差输出，且上下文输入为零维。",
        "- 早停验证参与者按既有确定性逻辑选择；种子由配置随机种子、目标、组合和折号确定。不搜索架构、学习率、风险阈值或特征上限。history、10,000 次随机 Uniform、风险阈值和七项指标沿用 Ridge 基准定义：真实 discomfort `>=0.50`，预测报警 `>=0.20`。",
        f"- random_uniform 以配置随机种子生成 {random_simulations:,} 次独立 Uniform(0,1) 预测。所有结论仅限这 135 条标签的离线 LOPO 比较，不是部署、实时接入、自动推荐或安全放行结论。",
        "",
        "## 三类基线",
        "",
        *baseline_rows,
        "",
        *random_rows,
        "",
        "## 15 个组合：按 Relaxation MAE（低者优先）",
        "",
        "| 排名 | 组合 | Relaxation MAE |",
        "|---:|---|---:|",
        *_combination_table(combination_metrics, key="relaxation_mae"),
        "",
        "## 15 个组合：按 Discomfort MAE（低者优先）",
        "",
        "| 排名 | 组合 | Discomfort MAE |",
        "|---:|---|---:|",
        *_combination_table(combination_metrics, key="discomfort_mae"),
        "",
        "## 15 个组合：按高 discomfort recall（高者优先）",
        "",
        "| 排名 | 组合 | 高 discomfort recall |",
        "|---:|---|---:|",
        *_combination_table(
            combination_metrics, key="discomfort_high_recall", descending=True, percent=True
        ),
        "",
        "## 全部 28 条移除单模态消融",
        "",
        "正值始终表示完整组合更好：MAE 为“移除后 MAE − 完整组合 MAE”；Spearman、recall、precision 为“完整组合 − 移除后”；假阴性为“移除后 − 完整组合”。",
        "",
        *ablation_rows,
        "",
        "## 结论边界",
        "",
        "本报告最多只能说明该 135 条数据上的离线 LOPO 基准中哪些组合观察上更有价值。它不构成模型选择、部署、实时接入、自动 Condition 推荐或安全门通过的证据；现有运行时以及 Shadow/hold 策略不因本基准而改变。",
    ]
    atomic_write_text(output, "\n".join(lines) + "\n")


def benchmark_minimal_fusion_dcnn(
    config: ProjectConfig, *, random_simulations: int = RANDOM_SIMULATIONS
) -> dict[str, Any]:
    """Run and persist the fixed research-only temporal 1DCNN benchmark."""
    pd, *_ = _dependencies()
    source = config.path("features") / SOURCE_RELATIVE_PATH
    if not source.exists():
        raise FileNotFoundError(f"Minimal fusion 1DCNN input not found: {source}")
    result = evaluate_minimal_fusion_dcnn_frame(
        pd.read_csv(source), config, random_simulations=random_simulations, expected_labels=EXPECTED_LABELS
    )
    if config.is_legacy:
        output_directory = config.path("artifacts") / OUTPUT_DIRECTORY
        metrics_output = output_directory / METRICS_FILENAME
        oof_output = output_directory / OOF_FILENAME
        report_output: Path | None = config.path("reports") / REPORT_FILENAME
    else:
        metrics_output = config.path("metrics") / "minimal_fusion_dcnn_metrics.csv"
        oof_output = config.path("predictions") / "minimal_fusion_dcnn_oof_predictions.csv"
        report_output = None
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    oof_output.parent.mkdir(parents=True, exist_ok=True)
    result["metrics"].to_csv(metrics_output, index=False)
    result["oof_predictions"].to_csv(oof_output, index=False)
    if report_output:
        _write_report(result, report_output, random_simulations=random_simulations)
    return {
        "research_only": True,
        "source": str(source),
        "n_labels": int(len(result["frame"])),
        "n_combinations": len(COMBINATIONS),
        "n_ablations": int(sum(len(name) for name in COMBINATIONS if len(name) > 1)),
        "random_simulations": int(random_simulations),
        "metrics": str(metrics_output),
        "oof_predictions": str(oof_output),
        "report": str(report_output) if report_output else None,
    }
