#!/usr/bin/env python
"""Run the Relax claim-validation experiment matrix."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.relax_foundation.relax_suite_common import suite_registry, validate_job_output
from src.data.relax_foundation import RELAX_EEG_RUN_TAG

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "relax_foundation" / "run_relax_foundation_probe.py"
FULL_MODALITIES = ("eeg", "ecg", "eye", "head", "video")
NO_EEG_MODALITIES = ("ecg", "eye", "head", "video")
NO_VIDEO_MODALITIES = ("eeg", "ecg", "eye", "head")


@dataclass(frozen=True)
class ClaimValidationJob:
    phase: str
    cohort: str
    output_dir: str
    seed: int | None = None
    fusion: str | None = None
    modalities: tuple[str, ...] = FULL_MODALITIES
    condition_control: str = "none"
    baseline: str = "none"

    def command(self, args: argparse.Namespace) -> list[str]:
        cmd = [
            sys.executable,
            str(RUNNER),
            "--embedding-cache",
            str(args.embedding_cache),
            "--cohorts",
            str(args.cohorts),
            "--cohort",
            self.cohort,
            "--output-dir",
            self.output_dir,
            "--modalities",
            *self.modalities,
            "--batch-size",
            str(args.batch_size),
            "--max-epochs",
            str(args.max_epochs),
            "--patience",
            str(args.patience),
            "--d-common",
            str(args.d_common),
            "--fusion-depth",
            str(args.fusion_depth),
            "--fusion-heads",
            str(args.fusion_heads),
            "--lr",
            str(getattr(args, "lr", 1e-3)),
            "--weight-decay",
            str(getattr(args, "weight_decay", 1e-4)),
            "--dim-head",
            str(args.dim_head),
            "--latent-dim",
            str(args.latent_dim),
            "--latent-channels",
            str(args.latent_channels),
            "--condition-control",
            self.condition_control,
            "--strict",
            "--run-tag",
            str(getattr(args, "run_tag", RELAX_EEG_RUN_TAG)),
            "--torch-threads",
            str(getattr(args, "torch_threads", 1)),
        ]
        if args.device:
            cmd.extend(["--device", str(args.device)])
        if args.smoke:
            cmd.append("--smoke")
        if self.seed is not None:
            cmd.extend(["--seed", str(self.seed)])
        if self.baseline != "none":
            cmd.extend(["--baseline", self.baseline])
            if self.baseline.startswith("relax_handcrafted"):
                cmd.extend(["--handcrafted-features", str(args.handcrafted_features)])
        else:
            if self.fusion is None:
                raise ValueError("Model jobs require a fusion method")
            cmd.extend(["--fusion", self.fusion, "--target", "original", "--pool", "sequence"])
        return cmd


def _out(root: Path, *parts: str) -> str:
    return str(root.joinpath(*parts))


def build_claim_validation_jobs(args: argparse.Namespace) -> list[ClaimValidationJob]:
    root = Path(args.output_dir)
    jobs: list[ClaimValidationJob] = []

    for cohort in ("all_135", "eeg_eligible"):
        jobs.append(
            ClaimValidationJob(
                phase="baseline_condition",
                cohort=cohort,
                baseline="condition",
                output_dir=_out(root, "baselines", cohort, "condition"),
            )
        )
        jobs.append(
            ClaimValidationJob(
                phase="baseline_handcrafted_cv",
                cohort=cohort,
                baseline="relax_handcrafted_ridge_cv",
                output_dir=_out(root, "baselines", cohort, "relax_handcrafted_ridge_cv"),
            )
        )

    for seed in args.seeds:
        for fusion in args.fusions:
            jobs.append(
                ClaimValidationJob(
                    phase="seed_stability",
                    cohort="all_135",
                    seed=int(seed),
                    fusion=fusion,
                    modalities=FULL_MODALITIES,
                    output_dir=_out(root, "seed_stability", "all_135", fusion, f"seed_{seed}", "full"),
                )
            )
            jobs.append(
                ClaimValidationJob(
                    phase="eeg_paired",
                    cohort="eeg_eligible",
                    seed=int(seed),
                    fusion=fusion,
                    modalities=FULL_MODALITIES,
                    output_dir=_out(root, "eeg_paired", "eeg_eligible", fusion, f"seed_{seed}", "with_eeg"),
                )
            )
            jobs.append(
                ClaimValidationJob(
                    phase="eeg_paired",
                    cohort="eeg_eligible",
                    seed=int(seed),
                    fusion=fusion,
                    modalities=NO_EEG_MODALITIES,
                    output_dir=_out(root, "eeg_paired", "eeg_eligible", fusion, f"seed_{seed}", "without_eeg"),
                )
            )
            jobs.append(
                ClaimValidationJob(
                    phase="video_condition_control",
                    cohort="all_135",
                    seed=int(seed),
                    fusion=fusion,
                    modalities=FULL_MODALITIES,
                    condition_control="video_condition_residualized",
                    output_dir=_out(root, "video_control", "all_135", fusion, f"seed_{seed}", "video_residualized"),
                )
            )
            jobs.append(
                ClaimValidationJob(
                    phase="video_condition_control",
                    cohort="all_135",
                    seed=int(seed),
                    fusion=fusion,
                    modalities=NO_VIDEO_MODALITIES,
                    output_dir=_out(root, "video_control", "all_135", fusion, f"seed_{seed}", "without_video"),
                )
            )

    output_dirs = [job.output_dir for job in jobs]
    duplicates = sorted({path for path in output_dirs if output_dirs.count(path) > 1})
    if duplicates:
        raise ValueError(f"Duplicate claim-validation output dirs: {duplicates}")
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
    parser.add_argument(
        "--output-dir",
        default=f"logs/relax_{RELAX_EEG_RUN_TAG}/claim_validation",
    )
    parser.add_argument("--suite-json", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260705, 20260706, 20260707])
    parser.add_argument(
        "--fusions",
        nargs="+",
        default=["early", "mid", "late", "qformer", "healnet", "mm_lego"],
        choices=["early", "mid", "late", "qformer", "healnet", "mm_lego"],
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--d-common", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--fusion-depth", type=int, default=1)
    parser.add_argument("--fusion-heads", type=int, default=2)
    parser.add_argument("--dim-head", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--latent-channels", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-tag", default=RELAX_EEG_RUN_TAG)
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    jobs = build_claim_validation_jobs(args)
    if args.limit is not None:
        jobs = jobs[: args.limit]
    suite_path = Path(args.suite_json) if args.suite_json else Path(args.output_dir) / "suite_jobs.json"
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    registered = [{"job": asdict(job), "command": job.command(args)} for job in jobs]
    suite_path.write_text(
        json.dumps(suite_registry(registered), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote {len(jobs)} jobs to {suite_path}")
    if not args.execute:
        print("Dry run only. Re-run with --execute to launch jobs.")
        return 0

    for index, job in enumerate(jobs, start=1):
        command = job.command(args)
        complete, reason = validate_job_output(command)
        if complete:
            print(f"[{index}/{len(jobs)}] skip complete: {job.phase} {job.cohort} {job.fusion or job.baseline}")
            continue
        print(f"[{index}/{len(jobs)}] {job.phase} {job.cohort} {job.fusion or job.baseline} seed={job.seed}")
        Path(job.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(job.output_dir) / "hard_failure.json").unlink(missing_ok=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        complete, reason = validate_job_output(command)
        if not complete:
            raise RuntimeError(f"Job finished but failed integrity validation: {reason}")
        suite_path.write_text(json.dumps(suite_registry(registered), indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
