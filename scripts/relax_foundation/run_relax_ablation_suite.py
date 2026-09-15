#!/usr/bin/env python
"""Generate or execute the registered Relax foundation ablation suite."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from scripts.relax_foundation.relax_suite_common import result_paths, suite_registry, validate_job_output  # noqa: E402
from mac.data.relax_foundation import RELAX_EEG_RUN_TAG, assert_relax_modalities  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]


COHORTS = [
    "high_quality_pilot",
    "all_135",
    "foundation_complete",
    "video_complete_sensitivity",
    "eeg_eligible",
]
FUSIONS = ["early", "mid", "late", "qformer", "healnet", "mm_lego"]
FULL_MODALITIES = ["ecg", "eeg", "eye", "head", "video"]
PROGRESSIVE_MODALITIES = [
    ["ecg"],
    ["ecg", "eeg"],
    ["ecg", "eeg", "eye"],
    ["ecg", "eeg", "eye", "head"],
    ["ecg", "eeg", "eye", "head", "video"],
]


def _locked_args(args: argparse.Namespace) -> list[str]:
    values = [
        "--batch-size", str(args.batch_size),
        "--max-epochs", str(args.max_epochs),
        "--patience", str(args.patience),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--d-common", str(args.d_common),
        "--fusion-depth", str(args.fusion_depth),
        "--fusion-heads", str(args.fusion_heads),
        "--dim-head", str(args.dim_head),
        "--latent-dim", str(args.latent_dim),
        "--latent-channels", str(args.latent_channels),
        "--seed", str(args.seed),
        "--run-tag", str(args.run_tag),
        "--torch-threads", str(args.torch_threads),
        "--strict",
    ]
    if args.device:
        values.extend(["--device", str(args.device)])
    return values


def _probe_command(
    args: argparse.Namespace,
    *,
    cohort: str,
    modalities: list[str],
    fusion: str,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "relax_foundation" / "run_relax_foundation_probe.py"),
        "--embedding-cache",
        args.embedding_cache,
        "--cohorts",
        args.cohorts,
        "--cohort",
        cohort,
        "--modalities",
        *modalities,
        "--fusion",
        fusion,
        "--target",
        args.target,
        "--pool",
        args.pool,
        "--output-dir",
        str(output_dir),
        *_locked_args(args),
    ]


def build_suite(args: argparse.Namespace) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    root = Path(args.output_dir)

    jobs.append(
        {
            "name": "end_to_end_smoke",
            "phase": "smoke",
            "command": [
                *_probe_command(
                    args,
                    cohort="high_quality_pilot",
                    modalities=FULL_MODALITIES,
                    fusion="mm_lego",
                    output_dir=root / "smoke",
                ),
                "--smoke",
            ],
        }
    )

    # Baselines.
    for cohort in COHORTS:
        for baseline in ("condition", "random_9_condition", "relax_handcrafted"):
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "relax_foundation" / "run_relax_foundation_probe.py"),
                "--embedding-cache",
                args.embedding_cache,
                "--cohorts",
                args.cohorts,
                "--cohort",
                cohort,
                "--modalities",
                *FULL_MODALITIES,
                "--baseline",
                baseline,
                "--output-dir",
                str(root / "base" / "baselines" / cohort / baseline),
                *_locked_args(args),
            ]
            if baseline == "relax_handcrafted":
                command.extend(["--handcrafted-features", args.handcrafted_features])
            jobs.append(
                {
                    "name": f"{cohort}_baseline_{baseline}",
                    "phase": "baseline",
                    "command": command,
                }
            )

    # Formal fusion comparison.
    for cohort in COHORTS:
        for fusion in FUSIONS:
            jobs.append(
                {
                    "name": f"{cohort}_fusion_{fusion}",
                    "phase": "formal_fusion",
                    "command": _probe_command(
                        args,
                        cohort=cohort,
                        modalities=FULL_MODALITIES,
                        fusion=fusion,
                        output_dir=root / "base" / "formal_fusion" / cohort / fusion,
                    ),
                }
            )

    # Leave-one-modality-out.
    for removed in FULL_MODALITIES:
        modalities = [mod for mod in FULL_MODALITIES if mod != removed]
        jobs.append(
            {
                "name": f"leave_one_out_without_{removed}",
                "phase": "modality_ablation",
                "removed_modality": removed,
                "command": _probe_command(
                    args,
                    cohort="all_135",
                    modalities=modalities,
                    fusion=args.ablation_fusion,
                    output_dir=root / "base" / "leave_one_modality_out" / f"without_{removed}",
                ),
            }
        )

    # Progressive additions with fixed pre-registered order.
    for idx, modalities in enumerate(PROGRESSIVE_MODALITIES, start=1):
        jobs.append(
            {
                "name": f"progressive_{idx}_{'+'.join(modalities)}",
                "phase": "progressive_modalities",
                "command": _probe_command(
                    args,
                    cohort="all_135",
                    modalities=modalities,
                    fusion=args.ablation_fusion,
                    output_dir=root / "base" / "progressive_modalities" / f"step_{idx}_{'+'.join(modalities)}",
                ),
            }
        )

    # Representation/loss sensitivity hooks.
    for target in ("original", "residual_to_condition_baseline"):
        for pool in ("mean", "attention", "sequence"):
            jobs.append(
                {
                    "name": f"representation_{target}_{pool}",
                    "phase": "representation_loss_ablation",
                    "command": [
                        * _probe_command(
                            args,
                            cohort="all_135",
                            modalities=FULL_MODALITIES,
                            fusion=args.ablation_fusion,
                            output_dir=root / "base" / "representation_loss" / f"{target}_{pool}",
                        ),
                    ],
                }
            )
            jobs[-1]["command"][jobs[-1]["command"].index("--target") + 1] = target
            jobs[-1]["command"][jobs[-1]["command"].index("--pool") + 1] = pool

    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-cache",
        default=f"artifacts/relax/runs/{RELAX_EEG_RUN_TAG}/condition_embeddings.pt",
    )
    parser.add_argument("--cohorts", default=f"artifacts/relax/runs/{RELAX_EEG_RUN_TAG}/cohorts.json")
    parser.add_argument(
        "--handcrafted-features",
        default=f"artifacts/relax_model/runs/{RELAX_EEG_RUN_TAG}/features/condition_features.csv",
    )
    parser.add_argument("--output-dir", default=f"logs/relax_{RELAX_EEG_RUN_TAG}/foundation")
    parser.add_argument("--suite-json", default=None)
    parser.add_argument("--target", default="original", choices=["original", "residual_to_condition_baseline"])
    parser.add_argument("--pool", default="sequence", choices=["mean", "attention", "sequence"])
    parser.add_argument("--ablation-fusion", default="mm_lego", choices=FUSIONS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-common", type=int, default=64)
    parser.add_argument("--fusion-depth", type=int, default=1)
    parser.add_argument("--fusion-heads", type=int, default=2)
    parser.add_argument("--dim-head", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--latent-channels", type=int, default=8)
    parser.add_argument("--run-tag", default=RELAX_EEG_RUN_TAG)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--execute", action="store_true", help="Execute jobs after writing suite_jobs.json.")
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N job limit for smoke execution.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assert_relax_modalities(FULL_MODALITIES)
    for modalities in PROGRESSIVE_MODALITIES:
        assert_relax_modalities(modalities)
    jobs = build_suite(args)
    if args.limit is not None:
        jobs = jobs[: args.limit]
    suite_path = Path(args.suite_json) if args.suite_json else Path(args.output_dir) / "suite_jobs.json"
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    suite_path.write_text(json.dumps(suite_registry(jobs), indent=2), encoding="utf-8")
    print(f"Wrote {len(jobs)} jobs to {suite_path}")
    if not args.execute:
        return 0
    for index, job in enumerate(jobs, start=1):
        complete, reason = validate_job_output(job["command"])
        if complete:
            print(f"[{index}/{len(jobs)}] skip complete: {job['name']}")
            continue
        print(f"Running {job['name']}")
        result_path, _ = result_paths(job["command"])
        result_path.parent.mkdir(parents=True, exist_ok=True)
        (result_path.parent / "hard_failure.json").unlink(missing_ok=True)
        subprocess.run(job["command"], cwd=REPO_ROOT, check=True)
        complete, reason = validate_job_output(job["command"])
        if not complete:
            raise RuntimeError(f"Job finished but failed integrity validation: {reason}")
        suite_path.write_text(json.dumps(suite_registry(jobs), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
