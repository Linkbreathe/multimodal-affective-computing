from __future__ import annotations

import argparse
import json
import sys
import warnings
from hashlib import blake2s
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path("src").resolve()))

from mac.config import load_config  # noqa: E402
from mac.models.dcnn import _architecture, _device, _torch, _validation_indexes  # noqa: E402
from mac.fusion.minimal_fusion import (  # noqa: E402
    FEATURES_PER_MODALITY,
    HIGH_DISCOMFORT_PREDICTION_THRESHOLD,
    HIGH_DISCOMFORT_TRUTH_THRESHOLD,
    MODALITY_ORDER,
    TARGETS,
    _condition_baseline,
    _history_baseline,
)
from mac.fusion.minimal_fusion_dcnn import (  # noqa: E402
    SOURCE_RELATIVE_PATH,
    _combine_selected_features,
    _fit_sequence_scaler,
    _selected_by_modality,
    _transform_sequences,
    build_minimal_fusion_sequences,
)


ROOT = Path(".")
OUTPUT_DIR = ROOT / "artifacts" / "fusion_minimal_lstm"
REPORT_DIR = ROOT / "artifacts" / "reports"
ROUND2_DIR = REPORT_DIR / "supplementary_round2"
DEFAULT_COMBINATIONS = ("P", "H", "E", "V", "PHEV")


class LstmResidualModel:
    def __init__(self, input_size: int, hidden_size: int, dropout: float, device) -> None:
        torch = _torch()

        class _Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = torch.nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                )
                self.head = torch.nn.Sequential(
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden_size, hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden_size, 1),
                    torch.nn.Tanh(),
                )

            def forward(self, values):
                # Incoming tensor follows the existing 1D-CNN convention:
                # batch x features x time. LSTM expects batch x time x features.
                sequence = values.transpose(1, 2)
                output, _ = self.lstm(sequence)
                last = output[:, -1, :]
                return self.head(last)

        self.model = _Model().to(device)


def set_torch_seed(seed: int, device) -> None:
    torch = _torch()
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def fold_seed(random_seed: int, target: str, combination: str, fold: int) -> int:
    payload = f"{int(random_seed)}|lstm|{target}|{combination}|{int(fold)}".encode("utf-8")
    return int.from_bytes(blake2s(payload, digest_size=4).digest(), "little")


def train_lstm_residual(
    sequences,
    train_indexes: np.ndarray,
    validation_indexes: np.ndarray,
    feature_indexes: np.ndarray,
    residual: np.ndarray,
    config,
    seed: int,
    device,
):
    torch = _torch()
    node = dict(config.get("modeling.dcnn", {}))
    set_torch_seed(seed, device)
    scaler = _fit_sequence_scaler(sequences, train_indexes, feature_indexes)
    model = LstmResidualModel(
        input_size=len(feature_indexes),
        hidden_size=int(node.get("mlp_hidden", 64)),
        dropout=float(node.get("dropout", 0.30)),
        device=device,
    ).model
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(node.get("learning_rate", 1e-4)),
        weight_decay=float(node.get("weight_decay", 5e-5)),
    )
    loss_fn = torch.nn.MSELoss()
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
            [rng.integers(1, int(sequences.lengths[index]) + 1) for index in train_indexes],
            dtype=int,
        )
        train_values, _ = _transform_sequences(sequences, train_indexes, feature_indexes, scaler, prefixes)
        order = rng.permutation(len(train_indexes))
        model.train()
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            values = torch.as_tensor(train_values[batch], device=device)
            targets = torch.as_tensor(residual[train_indexes][batch, None], dtype=torch.float32, device=device)
            optimizer.zero_grad()
            loss = loss_fn(model(values), targets)
            loss.backward()
            optimizer.step()
        validation_values, _ = _transform_sequences(sequences, validation_indexes, feature_indexes, scaler)
        with torch.no_grad():
            model.eval()
            prediction = model(torch.as_tensor(validation_values, device=device))
            validation_loss = float(
                loss_fn(
                    prediction,
                    torch.as_tensor(residual[validation_indexes, None], dtype=torch.float32, device=device),
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
        raise RuntimeError("LSTM training did not produce a model state")
    model.load_state_dict(best_state)
    model.eval()
    return model, scaler, best_loss


def predict_lstm(model, sequences, indexes: np.ndarray, feature_indexes: np.ndarray, scaler: dict[str, np.ndarray], device) -> np.ndarray:
    torch = _torch()
    values, _ = _transform_sequences(sequences, indexes, feature_indexes, scaler)
    with torch.no_grad():
        return (
            model(torch.as_tensor(values, device=device))
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(float)
        )


def safe_spearman(truth: np.ndarray, prediction: np.ndarray) -> float:
    value = float(stats.spearmanr(truth, prediction).statistic)
    return value if np.isfinite(value) else 0.0


def metric_block(truth: np.ndarray, prediction: np.ndarray, *, discomfort: bool) -> dict[str, float]:
    residual = truth - prediction
    sst = float(np.sum((truth - np.mean(truth)) ** 2))
    output = {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1.0 - np.sum(residual**2) / sst) if sst > 1e-12 else float("nan"),
        "spearman": safe_spearman(truth, prediction),
    }
    if discomfort:
        high_truth = truth >= HIGH_DISCOMFORT_TRUTH_THRESHOLD
        high_prediction = prediction >= HIGH_DISCOMFORT_PREDICTION_THRESHOLD
        output.update(
            {
                "high_recall": float(np.mean(high_prediction[high_truth])) if high_truth.any() else float("nan"),
                "high_precision": float(np.mean(high_truth[high_prediction])) if high_prediction.any() else 0.0,
                "high_false_negatives": float(np.sum(high_truth & ~high_prediction)),
            }
        )
    return output


def wide_metrics(oof: pd.DataFrame, model_family: str) -> pd.DataFrame:
    rows = []
    for combination, group in oof.groupby("combination", sort=False):
        row: dict[str, Any] = {"model_family": model_family, "combination": combination}
        for target in TARGETS:
            sub = group[group["target"].eq(target)]
            block = metric_block(
                sub["truth"].to_numpy(dtype=float),
                sub["prediction"].to_numpy(dtype=float),
                discomfort=target == "discomfort",
            )
            for key, value in block.items():
                row[f"{target}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_lstm(frame: pd.DataFrame, config, combinations: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    architecture = _architecture(config)
    sequences = build_minimal_fusion_sequences(
        frame,
        sequence_length=int(architecture["sequence_length"]),
        expected_labels=135,
    )
    labels = pd.DataFrame(
        {
            "participant_id": sequences.participant_ids,
            "condition": sequences.conditions,
            "presentation_position": sequences.presentation_positions,
            "relaxation": sequences.targets[:, 0],
            "discomfort": sequences.targets[:, 1],
        }
    )
    participants = sorted(labels["participant_id"].unique())
    device = _device(config.get("modeling.dcnn.device", "cuda"))
    random_seed = int(config.get("modeling.random_seed"))
    predictions = {
        combination: {target: np.full(len(labels), np.nan, dtype=float) for target in TARGETS}
        for combination in combinations
    }
    condition_only = {target: np.full(len(labels), np.nan, dtype=float) for target in TARGETS}
    history = {target: np.full(len(labels), np.nan, dtype=float) for target in TARGETS}
    selection_rows: list[dict[str, Any]] = []
    for fold, participant in enumerate(participants, start=1):
        test_indexes = np.flatnonzero(labels["participant_id"].eq(participant).to_numpy())
        train_indexes = np.flatnonzero(~labels["participant_id"].eq(participant).to_numpy())
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
            selected_by_modality = _selected_by_modality(sequences, train_indexes, residual[train_indexes])
            for combination in combinations:
                columns, counts = _combine_selected_features(selected_by_modality, combination)
                selection_rows.append(
                    {
                        "fold": fold,
                        "participant_id": participant,
                        "target": target,
                        "combination": combination,
                        **{f"selected_{modality}": counts.get(modality, 0) for modality in MODALITY_ORDER},
                    }
                )
                feature_lookup = {name: index for index, name in enumerate(sequences.feature_columns)}
                feature_indexes = np.asarray([feature_lookup[name] for name in columns], dtype=int)
                seed = fold_seed(random_seed, target, combination, fold)
                core_indexes, validation_indexes = _validation_indexes(sequences.participant_ids, train_indexes, seed)
                model, scaler, _ = train_lstm_residual(
                    sequences,
                    core_indexes,
                    validation_indexes,
                    feature_indexes,
                    residual,
                    config,
                    seed,
                    device,
                )
                held_out_residual = predict_lstm(model, sequences, test_indexes, feature_indexes, scaler, device)
                predictions[combination][target][test_indexes] = np.clip(baseline + held_out_residual, 0.0, 1.0)
            print(f"Minimal fusion LSTM {target} LOPO fold {fold}/{len(participants)}: {participant}", flush=True)

    oof_rows = []
    for combination in combinations:
        for index, label_row in labels.iterrows():
            for target in TARGETS:
                truth = float(label_row[target])
                prediction = float(predictions[combination][target][index])
                row = {
                    "combination": combination,
                    "participant_id": label_row["participant_id"],
                    "condition": label_row["condition"],
                    "presentation_position": label_row["presentation_position"],
                    "target": target,
                    "truth": truth,
                    "prediction": prediction,
                    "absolute_error": abs(truth - prediction),
                    "condition_only_prediction": float(condition_only[target][index]),
                    "condition_only_absolute_error": abs(truth - float(condition_only[target][index])),
                    "history_prediction": float(history[target][index]),
                    "history_absolute_error": abs(truth - float(history[target][index])),
                }
                if target == "discomfort":
                    row["high_discomfort_truth"] = float(truth >= HIGH_DISCOMFORT_TRUTH_THRESHOLD)
                    row["high_discomfort_prediction"] = float(prediction >= HIGH_DISCOMFORT_PREDICTION_THRESHOLD)
                else:
                    row["high_discomfort_truth"] = np.nan
                    row["high_discomfort_prediction"] = np.nan
                oof_rows.append(row)
    oof = pd.DataFrame(oof_rows)
    metrics = wide_metrics(oof, "lstm_residual")
    selections = pd.DataFrame(selection_rows)
    return {"oof": oof, "metrics": metrics, "selections": selections}


def write_report(comparison: pd.DataFrame, combinations: tuple[str, ...]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table = comparison[
        [
            "model_family",
            "combination",
            "relaxation_mae",
            "relaxation_spearman",
            "relaxation_rmse",
            "relaxation_r2",
            "discomfort_mae",
            "discomfort_spearman",
            "discomfort_rmse",
            "discomfort_r2",
            "discomfort_high_recall",
            "discomfort_high_precision",
            "discomfort_high_false_negatives",
        ]
    ].copy()
    for column in table.columns:
        if pd.api.types.is_numeric_dtype(table[column]):
            table[column] = table[column].map(lambda value: "NA" if not np.isfinite(value) else f"{float(value):.4f}")
    lines = [
        "# LSTM 深度模型补充报告",
        "",
        "生成日期：2026-07-01。",
        "",
        "本轮 LSTM 是 research-only 对比：输入仍为 `artifacts/features/video_ml/window_features.csv` 的 participant-condition 窗口序列，外层 leave-one-participant-out，目标为 relaxation/discomfort，特征选择、Condition-only residual 目标、AdamW、early stopping、随机种子、阈值和高 discomfort 判定沿用现有 1D-CNN 协议；仅将 residual 网络结构替换为单层 LSTM + tanh residual head。",
        "",
        f"实际运行组合：{', '.join(combinations)}。",
        "",
        "## 1D-CNN vs LSTM",
        "",
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in table.columns) + " |")
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "LSTM 结果仅用于补充深度模型对比，不进入 `classical`、Shadow-only 或 `hold` 部署结论。若与 1D-CNN 不一致，应解释为当前 N=135 participant-condition 数据下的离线模型差异，而不是部署优先级变化。",
        ]
    )
    (REPORT_DIR / "lstm_model_report_zh.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combinations", nargs="*", default=list(DEFAULT_COMBINATIONS))
    args = parser.parse_args()
    combinations = tuple(args.combinations)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ROUND2_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "configs" / "project.yaml")
    source = config.path("features") / SOURCE_RELATIVE_PATH
    frame = pd.read_csv(source)
    result = evaluate_lstm(frame, config, combinations)
    result["oof"].to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
    result["metrics"].to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    result["selections"].to_csv(OUTPUT_DIR / "selection_audit.csv", index=False)
    dcnn_oof_path = ROOT / "artifacts" / "fusion_minimal_dcnn" / "oof_predictions.csv"
    if dcnn_oof_path.exists():
        dcnn_oof = pd.read_csv(dcnn_oof_path)
        dcnn_oof = dcnn_oof[dcnn_oof["combination"].isin(combinations)].copy()
        dcnn_metrics = wide_metrics(dcnn_oof, "1dcnn_residual")
        comparison = pd.concat([dcnn_metrics, result["metrics"]], ignore_index=True)
    else:
        comparison = result["metrics"].copy()
    comparison.to_csv(ROUND2_DIR / "lstm_vs_1dcnn_comparison.csv", index=False)
    write_report(comparison, combinations)
    print(json.dumps({"metrics": str(OUTPUT_DIR / "metrics.csv"), "report": str(REPORT_DIR / "lstm_model_report_zh.md")}, indent=2))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
