#!/usr/bin/env python
"""Run LOPO Relax foundation probe experiments from condition embedding caches."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.relax_foundation_dataset import RelaxConditionEmbeddingDataset, relax_condition_collate  # noqa: E402
from src.data.relax_foundation import (  # noqa: E402
    CONDITIONS,
    MODALITIES,
    RELAX_EEG_RUN_TAG,
    RelaxHardFailure,
    assert_relax_modalities,
    write_json,
)
from src.tasks.relaxation import (  # noqa: E402
    TARGETS,
    RelaxFusionRegressor,
    compute_relax_regression_metrics,
    random_9_condition_baseline,
)
from src.tasks.relax_condition_control import FoldConditionResidualizer  # noqa: E402


HANDCRAFTED_RIDGE_ALPHAS: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
HANDCRAFTED_CLIP_QUANTILES: tuple[float, float] = (1.0, 99.0)


def _load_cohort(path: str | Path, cohort: str) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise RelaxHardFailure(f"Locked cohorts file is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if cohort not in data:
        raise RelaxHardFailure(f"Unknown cohort {cohort!r}; available={sorted(data)}")
    participants = data[cohort].get("participants")
    if not participants:
        raise RelaxHardFailure(f"Cohort {cohort!r} has no participants")
    return [str(p) for p in participants]


def _dataset_frame(dataset: RelaxConditionEmbeddingDataset) -> pd.DataFrame:
    rows = []
    for sample in dataset:
        rows.append(
            {
                "participant_id": sample["participant_id"],
                "condition": sample["condition"],
                "condition_index": sample["condition_index"],
                "relaxation": sample["labels"]["relaxation"],
                "discomfort": sample["labels"]["discomfort"],
            }
        )
    return pd.DataFrame(rows)


def _participant_indices(dataset: RelaxConditionEmbeddingDataset) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for idx, sample in enumerate(dataset):
        out.setdefault(sample["participant_id"], []).append(idx)
    return out


def _lopo_splits(participants: list[str]) -> list[dict[str, Any]]:
    if len(participants) < 3:
        raise RelaxHardFailure("LOPO requires at least 3 participants for train/val/test separation")
    ordered = sorted(participants)
    splits = []
    for i, test_participant in enumerate(ordered):
        val_participant = ordered[(i + 1) % len(ordered)]
        train_participants = [p for p in ordered if p not in {test_participant, val_participant}]
        splits.append(
            {
                "test_participant": test_participant,
                "val_participant": val_participant,
                "train_participants": train_participants,
            }
        )
    return splits


def _condition_baseline(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    means = train.groupby("condition", sort=False)[list(TARGETS)].mean()
    missing = [condition for condition in CONDITIONS if condition not in means.index]
    if missing:
        raise RelaxHardFailure(f"Condition baseline training fold missing Condition means: {missing}")
    pred = means.loc[test["condition"], list(TARGETS)].to_numpy(dtype=float)
    frame = test[["participant_id", "condition"]].copy()
    frame["relaxation_pred"] = pred[:, 0]
    frame["discomfort_pred"] = pred[:, 1]
    frame["relaxation_true"] = test["relaxation"].to_numpy(dtype=float)
    frame["discomfort_true"] = test["discomfort"].to_numpy(dtype=float)
    return pred, frame


def _run_handcrafted_baseline(
    *,
    features_path: str | Path,
    split: dict[str, Any],
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    path = Path(features_path)
    if not path.exists():
        raise RelaxHardFailure(f"Relax handcrafted feature table is missing: {path}")
    frame = pd.read_csv(path)
    required = {"participant_id", "condition", *TARGETS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RelaxHardFailure(f"Relax handcrafted feature table missing columns: {missing}")
    frame["participant_id"] = frame["participant_id"].astype(str)
    frame["condition"] = frame["condition"].astype(str)

    metadata = {
        "participant_id",
        "condition",
        "presentation_position",
        "intensity",
        "frequency",
        "intensity_index",
        "frequency_index",
        "condition_index",
        "window_count",
        "window_count_expected",
        *TARGETS,
        "calm",
        "pleasantness",
        "monotony",
        "visual_fit",
    }
    numeric = frame.drop(columns=[col for col in metadata if col in frame.columns], errors="ignore").apply(
        pd.to_numeric,
        errors="coerce",
    )
    train_mask = frame["participant_id"].isin(split["train_participants"])
    test_mask = frame["participant_id"] == split["test_participant"]
    train_x = numeric.loc[train_mask]
    test_x = numeric.loc[test_mask]
    train_y = frame.loc[train_mask, list(TARGETS)].to_numpy(dtype=float)
    test_y = frame.loc[test_mask, list(TARGETS)].to_numpy(dtype=float)
    if train_x.empty or test_x.empty:
        raise RelaxHardFailure(f"Handcrafted baseline has empty train/test fold for {split['test_participant']}")

    non_missing_fraction = train_x.notna().mean(axis=0)
    keep = non_missing_fraction[non_missing_fraction >= 0.40].index.tolist()
    if not keep:
        raise RelaxHardFailure("No Relax handcrafted features pass train-fold non-missing threshold")
    train_x = train_x[keep]
    test_x = test_x[keep]

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_arr = scaler.fit_transform(imputer.fit_transform(train_x))
    test_arr = scaler.transform(imputer.transform(test_x))
    model = Ridge(alpha=1.0)
    model.fit(train_arr, train_y)
    pred = model.predict(test_arr)
    pred_frame = frame.loc[test_mask, ["participant_id", "condition"]].copy()
    pred_frame["relaxation_pred"] = pred[:, 0]
    pred_frame["discomfort_pred"] = pred[:, 1]
    pred_frame["relaxation_true"] = test_y[:, 0]
    pred_frame["discomfort_true"] = test_y[:, 1]
    return pred, pred_frame, test_y


def _numeric_handcrafted_frame(features_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(features_path)
    if not path.exists():
        raise RelaxHardFailure(f"Relax handcrafted feature table is missing: {path}")
    frame = pd.read_csv(path)
    required = {"participant_id", "condition", *TARGETS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RelaxHardFailure(f"Relax handcrafted feature table missing columns: {missing}")
    frame["participant_id"] = frame["participant_id"].astype(str)
    frame["condition"] = frame["condition"].astype(str)

    metadata = {
        "participant_id",
        "condition",
        "presentation_position",
        "intensity",
        "frequency",
        "intensity_index",
        "frequency_index",
        "condition_index",
        "window_count",
        "window_count_expected",
        *TARGETS,
        "calm",
        "pleasantness",
        "monotony",
        "visual_fit",
    }
    numeric = frame.drop(columns=[col for col in metadata if col in frame.columns], errors="ignore").apply(
        pd.to_numeric,
        errors="coerce",
    )
    return frame, numeric


def _fit_handcrafted_transform(
    train_x: pd.DataFrame,
    *eval_frames: pd.DataFrame,
) -> tuple[np.ndarray, list[np.ndarray]]:
    imputer = SimpleImputer(strategy="median")
    train_imputed = imputer.fit_transform(train_x)
    eval_imputed = [imputer.transform(frame) for frame in eval_frames]
    lower = np.nanpercentile(train_imputed, HANDCRAFTED_CLIP_QUANTILES[0], axis=0)
    upper = np.nanpercentile(train_imputed, HANDCRAFTED_CLIP_QUANTILES[1], axis=0)
    finite_bounds = np.isfinite(lower) & np.isfinite(upper) & (lower <= upper)
    lower = np.where(finite_bounds, lower, -np.inf)
    upper = np.where(finite_bounds, upper, np.inf)
    train_clipped = np.clip(train_imputed, lower, upper)
    eval_clipped = [np.clip(arr, lower, upper) for arr in eval_imputed]
    scaler = StandardScaler()
    train_arr = scaler.fit_transform(train_clipped)
    eval_arr = [scaler.transform(arr) for arr in eval_clipped]
    return train_arr, eval_arr


def _run_handcrafted_baseline_cv(
    *,
    features_path: str | Path,
    split: dict[str, Any],
    alphas: tuple[float, ...] = HANDCRAFTED_RIDGE_ALPHAS,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Run a train-only Ridge baseline with alpha selected on the validation participant."""
    frame, numeric = _numeric_handcrafted_frame(features_path)
    train_mask = frame["participant_id"].isin(split["train_participants"])
    val_mask = frame["participant_id"] == split["val_participant"]
    test_mask = frame["participant_id"] == split["test_participant"]
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise RelaxHardFailure(
            "Handcrafted CV baseline has an empty train/val/test fold for "
            f"test={split['test_participant']} val={split['val_participant']}"
        )

    train_x = numeric.loc[train_mask]
    val_x = numeric.loc[val_mask]
    test_x = numeric.loc[test_mask]
    train_y = frame.loc[train_mask, list(TARGETS)].to_numpy(dtype=float)
    val_y = frame.loc[val_mask, list(TARGETS)].to_numpy(dtype=float)
    test_y = frame.loc[test_mask, list(TARGETS)].to_numpy(dtype=float)

    non_missing_fraction = train_x.notna().mean(axis=0)
    keep = non_missing_fraction[non_missing_fraction >= 0.40].index.tolist()
    if not keep:
        raise RelaxHardFailure("No Relax handcrafted features pass train-fold non-missing threshold")
    train_x = train_x[keep]
    val_x = val_x[keep]
    test_x = test_x[keep]

    imputer = SimpleImputer(strategy="median")
    train_imputed = imputer.fit_transform(train_x)
    if train_imputed.shape[1] == 0:
        raise RelaxHardFailure("No Relax handcrafted features remain after train-fold imputation")
    train_variance = np.nanvar(train_imputed, axis=0)
    keep_after_variance = [col for col, var in zip(keep, train_variance) if np.isfinite(var) and var > 0.0]
    if not keep_after_variance:
        raise RelaxHardFailure("No Relax handcrafted features have non-zero train-fold variance")
    train_x = train_x[keep_after_variance]
    val_x = val_x[keep_after_variance]
    test_x = test_x[keep_after_variance]

    selected_alpha: float | None = None
    selected_val = float("inf")
    val_metrics_by_alpha: dict[str, float] = {}
    for alpha in alphas:
        train_arr, (val_arr,) = _fit_handcrafted_transform(train_x, val_x)
        model = Ridge(alpha=float(alpha))
        model.fit(train_arr, train_y)
        val_pred = model.predict(val_arr)
        val_mae = compute_relax_regression_metrics(val_y, val_pred)["macro_mae"]
        val_metrics_by_alpha[str(float(alpha))] = float(val_mae)
        if val_mae < selected_val:
            selected_val = float(val_mae)
            selected_alpha = float(alpha)
    if selected_alpha is None:
        raise RelaxHardFailure("Handcrafted CV baseline could not select a Ridge alpha")

    train_arr, (test_arr,) = _fit_handcrafted_transform(train_x, test_x)
    model = Ridge(alpha=selected_alpha)
    model.fit(train_arr, train_y)
    pred = model.predict(test_arr)
    pred_frame = frame.loc[test_mask, ["participant_id", "condition"]].copy()
    pred_frame["relaxation_pred"] = pred[:, 0]
    pred_frame["discomfort_pred"] = pred[:, 1]
    pred_frame["relaxation_true"] = test_y[:, 0]
    pred_frame["discomfort_true"] = test_y[:, 1]
    metadata = {
        "selected_alpha": selected_alpha,
        "selected_val_macro_mae": selected_val,
        "val_macro_mae_by_alpha": val_metrics_by_alpha,
        "num_features": len(keep_after_variance),
    }
    return pred, pred_frame, test_y, metadata


def run_baseline(
    dataset: RelaxConditionEmbeddingDataset,
    *,
    participants: list[str],
    baseline: str,
    random_seeds: range,
    handcrafted_features: str | Path | None = None,
) -> dict[str, Any]:
    frame = _dataset_frame(dataset)
    fold_payloads: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for split in _lopo_splits(participants):
        train = frame[frame["participant_id"].isin(split["train_participants"])].reset_index(drop=True)
        test = frame[frame["participant_id"] == split["test_participant"]].reset_index(drop=True)
        if baseline == "condition":
            pred, pred_frame = _condition_baseline(train, test)
            metrics = compute_relax_regression_metrics(test[list(TARGETS)].to_numpy(dtype=float), pred)
            pred_frame["fold"] = split["test_participant"]
            prediction_frames.append(pred_frame)
        elif baseline == "random_9_condition":
            out = random_9_condition_baseline(train, test, seeds=random_seeds)
            metrics = out["summary"]
            pred_frame = out["predictions"].copy()
            pred_frame["fold"] = split["test_participant"]
            prediction_frames.append(pred_frame)
        elif baseline == "relax_handcrafted":
            if handcrafted_features is None:
                raise RelaxHardFailure("--handcrafted-features is required for relax_handcrafted baseline")
            pred, pred_frame, truth = _run_handcrafted_baseline(
                features_path=handcrafted_features,
                split=split,
            )
            metrics = compute_relax_regression_metrics(truth, pred)
            pred_frame["fold"] = split["test_participant"]
            prediction_frames.append(pred_frame)
        elif baseline == "relax_handcrafted_ridge_cv":
            if handcrafted_features is None:
                raise RelaxHardFailure("--handcrafted-features is required for relax_handcrafted_ridge_cv baseline")
            pred, pred_frame, truth, metadata = _run_handcrafted_baseline_cv(
                features_path=handcrafted_features,
                split=split,
            )
            metrics = compute_relax_regression_metrics(truth, pred)
            pred_frame["fold"] = split["test_participant"]
            prediction_frames.append(pred_frame)
        else:
            raise RelaxHardFailure(f"Unsupported baseline in this runner: {baseline}")
        fold_metadata = metadata if baseline == "relax_handcrafted_ridge_cv" else {}
        fold_payloads.append({**split, "metrics": metrics, **fold_metadata})
    return {
        "folds": fold_payloads,
        "predictions": pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame(),
    }


def _infer_embed_dims(dataset: RelaxConditionEmbeddingDataset, modalities: list[str]) -> dict[str, int | tuple[int, int]]:
    dims: dict[str, int | tuple[int, int]] = {}
    for modality in modalities:
        raw = dataset.embedding_dims.get(modality)
        if isinstance(raw, int):
            dims[modality] = raw
        elif isinstance(raw, list) and len(raw) == 2:
            dims[modality] = (int(raw[0]), int(raw[1]))
        else:
            sample = dataset[0]["embeddings"][modality]
            dims[modality] = int(sample.shape[-1]) if sample.ndim == 2 else (int(sample.shape[1]), int(sample.shape[2]))
    return dims


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return out


def _evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    residual_means: pd.DataFrame | None = None,
    condition_control: FoldConditionResidualizer | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    participants: list[str] = []
    conditions: list[str] = []
    with torch.no_grad():
        for batch in loader:
            if condition_control is not None:
                batch = condition_control.transform_batch(batch)
            batch = _batch_to_device(batch, device)
            pred = model(batch).cpu().numpy()
            if residual_means is not None:
                means = residual_means.loc[batch["conditions"], list(TARGETS)].to_numpy(dtype=float)
                pred = pred + means
            preds.append(pred)
            targets.append(batch["targets"].cpu().numpy())
            participants.extend(batch["participant_ids"])
            conditions.extend(batch["conditions"])
    return np.concatenate(targets), np.concatenate(preds), participants, conditions


def _train_fold(
    dataset: RelaxConditionEmbeddingDataset,
    *,
    split: dict[str, Any],
    participant_to_indices: dict[str, list[int]],
    modalities: list[str],
    fusion: str,
    d_common: int,
    pool: str,
    device: torch.device,
    batch_size: int,
    max_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    seed: int,
    target_mode: str,
    fusion_kwargs: dict[str, Any],
    condition_control_name: str,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_indices = [idx for p in split["train_participants"] for idx in participant_to_indices[p]]
    val_indices = participant_to_indices[split["val_participant"]]
    test_indices = participant_to_indices[split["test_participant"]]
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=relax_condition_collate)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=relax_condition_collate)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=relax_condition_collate)

    embed_dims = _infer_embed_dims(dataset, modalities)
    model = RelaxFusionRegressor(
        embed_dims=embed_dims,
        modalities=modalities,
        fusion_name=fusion,
        d_common=d_common,
        pool=pool,
        fusion_kwargs=fusion_kwargs,
    ).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    frame = _dataset_frame(dataset)
    residual_means: pd.DataFrame | None = None
    if target_mode == "residual_to_condition_baseline":
        train_frame = frame[frame["participant_id"].isin(split["train_participants"])]
        residual_means = train_frame.groupby("condition", sort=False)[list(TARGETS)].mean()
        missing = [condition for condition in CONDITIONS if condition not in residual_means.index]
        if missing:
            raise RelaxHardFailure(f"Residual baseline training fold missing Condition means: {missing}")

    condition_control: FoldConditionResidualizer | None = None
    if condition_control_name == "video_condition_residualized":
        if "video" not in modalities:
            raise RelaxHardFailure("--condition-control video_condition_residualized requires the video modality")
        condition_control = FoldConditionResidualizer.fit(dataset, train_indices=train_indices, modality="video")
    elif condition_control_name != "none":
        raise RelaxHardFailure(f"Unsupported Condition-control mode: {condition_control_name}")

    best_state = None
    best_val = float("inf")
    stale = 0
    for _epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            if condition_control is not None:
                batch = condition_control.transform_batch(batch)
            batch = _batch_to_device(batch, device)
            target = batch["targets"]
            if residual_means is not None:
                means = torch.as_tensor(
                    residual_means.loc[batch["conditions"], list(TARGETS)].to_numpy(dtype=float),
                    dtype=target.dtype,
                    device=device,
                )
                target = target - means
            pred = model(batch)
            loss = loss_fn(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        y_val, p_val, _, _ = _evaluate_model(
            model,
            val_loader,
            device,
            residual_means=residual_means,
            condition_control=condition_control,
        )
        val_mae = compute_relax_regression_metrics(y_val, p_val)["macro_mae"]
        if val_mae < best_val:
            best_val = val_mae
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_test, p_test, participants, conditions = _evaluate_model(
        model,
        test_loader,
        device,
        residual_means=residual_means,
        condition_control=condition_control,
    )
    metrics = compute_relax_regression_metrics(y_test, p_test)
    predictions = pd.DataFrame(
        {
            "participant_id": participants,
            "condition": conditions,
            "relaxation_true": y_test[:, 0],
            "discomfort_true": y_test[:, 1],
            "relaxation_pred": p_test[:, 0],
            "discomfort_pred": p_test[:, 1],
            "fold": split["test_participant"],
        }
    )
    return {**split, "metrics": metrics, "predictions": predictions, "best_val_macro_mae": best_val}


def run_model(args: argparse.Namespace, dataset: RelaxConditionEmbeddingDataset, participants: list[str]) -> dict[str, Any]:
    participant_to_indices = _participant_indices(dataset)
    missing = [p for p in participants if p not in participant_to_indices]
    if missing:
        raise RelaxHardFailure(f"Cohort participants missing from embedding cache: {missing}")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    fusion_kwargs: dict[str, Any] = {}
    if args.fusion == "mm_lego":
        fusion_kwargs = {
            "latent_dim": args.latent_dim,
            "latent_channels": args.latent_channels,
            "depth": args.fusion_depth,
            "heads": args.fusion_heads,
            "dim_head": args.dim_head,
            "mode": args.mm_lego_mode,
        }
    elif args.fusion == "qformer":
        fusion_kwargs = {"n_layers": args.fusion_depth, "n_heads": args.fusion_heads}
    elif args.fusion == "healnet":
        fusion_kwargs = {"n_layers": args.fusion_depth, "n_heads": args.fusion_heads}

    folds = []
    predictions = []
    max_epochs = 2 if args.smoke else args.max_epochs
    for fold_idx, split in enumerate(_lopo_splits(participants)):
        result = _train_fold(
            dataset,
            split=split,
            participant_to_indices=participant_to_indices,
            modalities=args.modalities,
            fusion=args.fusion,
            d_common=args.d_common,
            pool=args.pool,
            device=device,
            batch_size=args.batch_size,
            max_epochs=max_epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed + fold_idx,
            target_mode=args.target,
            fusion_kwargs=fusion_kwargs,
            condition_control_name=args.condition_control,
        )
        predictions.append(result.pop("predictions"))
        folds.append(result)
    return {"folds": folds, "predictions": pd.concat(predictions, ignore_index=True)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-cache", default="artifacts/relax/condition_embeddings.pt")
    parser.add_argument("--cohorts", default="artifacts/relax/cohorts.json")
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--modalities", nargs="+", default=list(MODALITIES))
    parser.add_argument("--fusion", default="early", choices=["early", "mid", "late", "qformer", "healnet", "mm_lego"])
    parser.add_argument(
        "--baseline",
        default="none",
        choices=["none", "condition", "random_9_condition", "relax_handcrafted", "relax_handcrafted_ridge_cv"],
    )
    parser.add_argument("--handcrafted-features", default=None, help="Relax condition_features.csv for relax_handcrafted baseline.")
    parser.add_argument("--target", default="original", choices=["original", "residual_to_condition_baseline"])
    parser.add_argument("--condition-control", default="none", choices=["none", "video_condition_residualized"])
    parser.add_argument("--pool", default="sequence", choices=["mean", "attention", "sequence"])
    parser.add_argument("--output-dir", default="logs/relax_foundation_probe")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-common", type=int, default=128)
    parser.add_argument("--fusion-depth", type=int, default=2)
    parser.add_argument("--fusion-heads", type=int, default=4)
    parser.add_argument("--dim-head", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--latent-channels", type=int, default=32)
    parser.add_argument("--mm-lego-mode", default="merge-sum")
    parser.add_argument("--random-baseline-seeds", type=int, default=1000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Kept for CLI parity; runner is strict by default.")
    parser.add_argument("--run-tag", default=RELAX_EEG_RUN_TAG)
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        args.modalities = assert_relax_modalities(args.modalities)
        participants = _load_cohort(args.cohorts, args.cohort)
        dataset = RelaxConditionEmbeddingDataset(
            args.embedding_cache,
            modalities=args.modalities,
            participants=participants,
        )
        if args.baseline != "none":
            result = run_baseline(
                dataset,
                participants=participants,
                baseline=args.baseline,
                random_seeds=range(args.random_baseline_seeds),
                handcrafted_features=args.handcrafted_features,
            )
            run_name = f"{args.cohort}_{args.baseline}"
        else:
            result = run_model(args, dataset, participants)
            run_name = f"{args.cohort}_{args.fusion}_{args.target}_{args.pool}"

        pred_path = out / f"{run_name}_predictions.csv"
        result["predictions"].to_csv(pred_path, index=False)
        serializable = {
            "args": vars(args),
            "folds": result["folds"],
            "predictions_csv": str(pred_path),
        }
        write_json(out / f"{run_name}_results.json", serializable)
        return 0
    except RelaxHardFailure as error:
        write_json(out / "hard_failure.json", {"status": "failed", "error": str(error), "args": vars(args)})
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
