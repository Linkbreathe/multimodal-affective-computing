from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from real_time_ml.config import load_config, load_config_layers
from real_time_ml.reporting import write_run_summary


def _participants(value: str | None) -> list[str] | None:
    return [item.strip().upper() for item in value.split(",") if item.strip()] if value else None


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rtml", description="P002-P016 multimodal 10-second shadow inference")
    parser.add_argument("--config", default=None, help="legacy project.yaml path")
    parser.add_argument("--base-config", default=None, help="optional layered base.yaml override")
    parser.add_argument("--experiment", default=None, help="layered experiment YAML (new run-scoped entry point)")
    parser.add_argument("--local-config", default=None, help="untracked local.yaml with data roots/device")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("index", "preprocess"):
        command = commands.add_parser(name)
        command.add_argument("--participants", help="comma-separated participant ids")
    extract = commands.add_parser("extract-features")
    extract.add_argument("--participants", help="comma-separated participant ids")
    extract.add_argument("--no-video", action="store_true", help="fast extraction without OpenCV frames")
    mp4 = commands.add_parser("build-video-mp4", help="retain 10 fps egocentric MP4 cache under artifacts/video")
    mp4.add_argument("--participants", help="comma-separated participant ids")
    mp4.add_argument("--force", action="store_true", help="re-encode existing participant MP4 files")
    mp4.add_argument("--ffmpeg", help="explicit ffmpeg executable path")
    visual = commands.add_parser("extract-handcrafted-video", help="create isolated handcrafted egocentric feature table")
    visual.add_argument("--participants", help="comma-separated participant ids")
    mae = commands.add_parser("extract-videomae2", help="extract frozen official VideoMAE V2 embeddings from retained MP4")
    mae.add_argument("--participants", help="comma-separated participant ids")
    mae.add_argument("--force", action="store_true", help="discard matching embedding cache")
    dynamic = commands.add_parser(
        "extract-dynamic-texture",
        help="research-only 12-value dynamic_texture_v1 extraction on the frozen common mask",
    )
    dynamic.add_argument("--participants", help="comma-separated participant ids")
    dynamic.add_argument("--contract-dir", type=Path)
    dynamic.add_argument("--base-window-features", type=Path)
    dynamic.add_argument("--output-dir", type=Path)
    dynamic.add_argument("--force", action="store_true", help="discard a matching descriptor cache")
    commands.add_parser("train-video-ml", help="LOPO handcrafted-video ML and same-cohort no-video comparisons")
    commands.add_parser("train-videomae2-dcnn", help="LOPO frozen VideoMAE2 + 1DCNN fusion and fallback")
    commands.add_parser("report-video-fusion", help="write evidence-backed Chinese ML/DL egocentric-video report")
    commands.add_parser("train-video-relaxation-ml", help="research-only relaxation Ridge video/no-video comparison")
    commands.add_parser("train-videomae2-relaxation", help="research-only relaxation VideoMAE2 + 1DCNN comparison")
    commands.add_parser("report-video-relaxation", help="write Chinese relaxation-only video research report")
    commands.add_parser(
        "train-videomae2-video-encoder-ablation",
        help="uniform no-video/direct-MLP/temporal-1DCNN VideoMAE2 ablation",
    )
    commands.add_parser(
        "report-videomae2-video-encoder-ablation",
        help="write Chinese VideoMAE2 direct-MLP versus temporal-1DCNN report",
    )
    commands.add_parser(
        "benchmark-minimal-fusion",
        help="research-only 15-combination LOPO Ridge minimal multimodal benchmark",
    )
    commands.add_parser(
        "benchmark-minimal-fusion-dcnn",
        help="research-only 15-combination LOPO temporal 1DCNN minimal multimodal benchmark",
    )
    commands.add_parser(
        "analyze-minimal-fusion-dcnn-hp",
        help="research-only H/P fold-selection audit and feature-family 1DCNN ablations",
    )
    commands.add_parser(
        "report-latest-multimodal",
        help="write Chinese evidence-indexed latest multimodal research report",
    )
    commands.add_parser(
        "report",
        help="write the single reports/<run_id>_summary_zh.md from this run's metrics and predictions",
    )
    commands.add_parser("train-state")
    commands.add_parser(
        "train-realtime-multimodal-window",
        help="train the adaptive-control realtime EEG/ECG/eye/HMD-motion window model",
    )
    commands.add_parser("train-dcnn-state")
    commands.add_parser("train-policy")
    commands.add_parser("evaluate")
    replay_parser = commands.add_parser("replay")
    replay_parser.add_argument("--participants", help="comma-separated participant ids")
    replay_parser.add_argument("--output", type=Path)
    video_replay = commands.add_parser("replay-video", help="recorded-only visual Shadow replay")
    video_replay.add_argument("--backend", choices=("handcrafted", "videomae2"), required=True)
    video_replay.add_argument("--participants", help="comma-separated participant ids")
    video_replay.add_argument("--output", type=Path)
    serve_parser = commands.add_parser("serve")
    serve_parser.add_argument("--max-cycles", type=int, help="test-only finite cycle count")
    adaptive_control = commands.add_parser("adaptive-control", help="start the local adaptive-control service")
    adaptive_control.add_argument("--control-config", help="adaptive-control.yaml path")
    adaptive_control.add_argument("--bundle", help="registered adaptive-control model bundle id")
    adaptive_control.add_argument("--max-cycles", type=int, help="test-only finite cycle count")
    adaptive_model = commands.add_parser("adaptive-model", help="inspect or verify registered adaptive-control model bundles")
    adaptive_model.add_argument("--control-config", help="adaptive-control.yaml path")
    adaptive_model.add_argument("action", choices=("list", "verify"))
    adaptive_model.add_argument("--bundle", help="bundle id; required for verify")
    run_all = commands.add_parser("run-all")
    run_all.add_argument("--participants", help="comma-separated participant ids")
    run_all.add_argument("--no-video", action="store_true")
    run_all.add_argument("--skip-training", action="store_true", help="data/QC smoke run")
    run_all.add_argument("--force", action="store_true", help="ignore cached stage artifacts")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.experiment or args.local_config or args.base_config:
        if not args.experiment:
            parser.error("--experiment is required when using --base-config or --local-config")
        if args.config:
            parser.error("--config is the legacy entry point and cannot be combined with --experiment")
        config = load_config_layers(args.base_config, args.experiment, args.local_config)
        config.write_run_manifest()
    else:
        config = load_config(args.config)
    selected = _participants(getattr(args, "participants", None))
    if args.command == "index":
        from real_time_ml.data.index import build_index

        result = {"participants": build_index(config, selected)}
    elif args.command == "preprocess":
        from real_time_ml.preprocessing.pipeline import preprocess

        result = preprocess(config, selected)
    elif args.command == "extract-features":
        from real_time_ml.features.extract import extract_features

        result = extract_features(config, selected, include_video=not args.no_video)
    elif args.command == "build-video-mp4":
        from real_time_ml.data.video import build_video_mp4s

        result = build_video_mp4s(config, selected, force=args.force, ffmpeg=args.ffmpeg)
    elif args.command == "extract-handcrafted-video":
        from real_time_ml.features.egocentric import extract_handcrafted_egocentric_features

        result = extract_handcrafted_egocentric_features(config, selected)
    elif args.command == "extract-videomae2":
        from real_time_ml.features.videomae2 import extract_videomae2_embeddings

        result = extract_videomae2_embeddings(config, selected, force=args.force)
    elif args.command == "extract-dynamic-texture":
        from real_time_ml.features.dynamic_texture import extract_dynamic_texture_features

        result = extract_dynamic_texture_features(
            config,
            selected,
            contract_dir=args.contract_dir,
            base_window_features=args.base_window_features,
            output_dir=args.output_dir,
            force=args.force,
        )
    elif args.command == "train-video-ml":
        from real_time_ml.modeling.video_train import train_handcrafted_video_ml

        result = train_handcrafted_video_ml(config)
    elif args.command == "train-videomae2-dcnn":
        from real_time_ml.modeling.video_dcnn import train_videomae2_dcnn

        result = train_videomae2_dcnn(config)
    elif args.command == "report-video-fusion":
        from real_time_ml.modeling.video_report import write_egocentric_video_report

        result = write_egocentric_video_report(config)
    elif args.command == "train-video-relaxation-ml":
        from real_time_ml.modeling.video_train import train_handcrafted_video_relaxation_ml

        result = train_handcrafted_video_relaxation_ml(config)
    elif args.command == "train-videomae2-relaxation":
        from real_time_ml.modeling.video_dcnn import train_videomae2_relaxation

        result = train_videomae2_relaxation(config)
    elif args.command == "report-video-relaxation":
        from real_time_ml.modeling.video_report import write_video_relaxation_report

        result = write_video_relaxation_report(config)
    elif args.command == "train-videomae2-video-encoder-ablation":
        from real_time_ml.modeling.video_dcnn import train_videomae2_video_encoder_ablation

        result = train_videomae2_video_encoder_ablation(config)
    elif args.command == "report-videomae2-video-encoder-ablation":
        from real_time_ml.modeling.video_encoder_report import write_videomae2_video_encoder_ablation_report

        result = write_videomae2_video_encoder_ablation_report(config)
    elif args.command == "benchmark-minimal-fusion":
        from real_time_ml.experiments import benchmark_minimal_fusion

        result = benchmark_minimal_fusion(config)
    elif args.command == "benchmark-minimal-fusion-dcnn":
        from real_time_ml.experiments import benchmark_minimal_fusion_dcnn

        result = benchmark_minimal_fusion_dcnn(config)
    elif args.command == "analyze-minimal-fusion-dcnn-hp":
        from real_time_ml.experiments import analyze_minimal_fusion_dcnn_hp

        audit = analyze_minimal_fusion_dcnn_hp(config)
        result = {
            key: value
            for key, value in audit.items()
            if key not in {
                "selection_audit",
                "selection_stability",
                "family_ablation_metrics",
                "subgroup_metrics",
                "oof_predictions",
            }
        }
    elif args.command == "report-latest-multimodal":
        from real_time_ml.modeling.latest_multimodal_report import write_latest_multimodal_report

        result = write_latest_multimodal_report(config)
    elif args.command == "report":
        result = write_run_summary(config)
    elif args.command == "train-state":
        from real_time_ml.training import train_state

        result = train_state(config)
    elif args.command == "train-realtime-multimodal-window":
        from real_time_ml.training import train_realtime_multimodal_window_model

        result = train_realtime_multimodal_window_model(config)
    elif args.command == "train-dcnn-state":
        from real_time_ml.training import train_dcnn_state

        result = train_dcnn_state(config)
    elif args.command == "train-policy":
        from real_time_ml.training import train_policy

        result = train_policy(config)
    elif args.command == "evaluate":
        from real_time_ml.evaluation import evaluate

        result = evaluate(config)
    elif args.command == "replay":
        from real_time_ml.realtime.replay import replay

        result = replay(config, selected, args.output)
    elif args.command == "replay-video":
        from real_time_ml.realtime.video_replay import replay_visual_model

        result = replay_visual_model(config, args.backend, selected, args.output)
    elif args.command == "serve":
        from real_time_ml.realtime.serve import serve

        result = serve(config, args.max_cycles)
    elif args.command == "adaptive-control":
        from real_time_ml.adaptive_control.service import serve_adaptive_control
        from real_time_ml.adaptive_control.settings import load_adaptive_control_settings

        control_settings = load_adaptive_control_settings(args.control_config)
        try:
            result = serve_adaptive_control(
                config,
                control_settings,
                bundle_id=args.bundle,
                max_cycles=args.max_cycles,
            )
        except OSError as exc:
            _print({
                "ok": False,
                "error": "adaptive_control_startup_failed",
                "reason": str(exc),
                "listen_host": control_settings.listen_host,
                "unity_to_python_port": control_settings.unity_to_python_port,
                "python_send_host": control_settings.python_send_host,
                "python_to_unity_port": control_settings.python_to_unity_port,
            })
            return 2
    elif args.command == "adaptive-model":
        from real_time_ml.adaptive_control.service import list_models, verify_model
        from real_time_ml.adaptive_control.settings import load_adaptive_control_settings

        settings = load_adaptive_control_settings(args.control_config)
        if args.action == "list":
            result = {"models": list_models(settings)}
        else:
            if not args.bundle:
                parser.error(f"{args.command} verify requires --bundle")
            report = verify_model(settings, args.bundle)
            result = {
                "compatible": report.compatible,
                "reasons": report.reasons,
                "descriptor": report.descriptor.__dict__,
            }
    elif args.command == "run-all":
        from real_time_ml.data.index import build_index
        from real_time_ml.features.extract import extract_features
        from real_time_ml.preprocessing.pipeline import preprocess
        from real_time_ml.realtime.replay import replay
        from real_time_ml.training import train_policy, train_state
        from real_time_ml.evaluation import evaluate

        steps: dict[str, Any] = {}
        steps["index"] = {"participants": len(build_index(config, selected))}
        windows_cache = config.path("preprocessed") / "windows.csv"
        feature_cache = config.path("features") / "window_features.csv"
        if args.force or not windows_cache.exists() or selected:
            steps["preprocess"] = preprocess(config, selected)
        else:
            steps["preprocess"] = {"cached": True, "path": str(windows_cache)}
        if args.force or not feature_cache.exists() or selected:
            steps["extract_features"] = extract_features(config, selected, include_video=not args.no_video)
        else:
            steps["extract_features"] = {"cached": True, "path": str(feature_cache)}
        if not args.skip_training:
            steps["train_state"] = train_state(config)
            steps["train_policy"] = train_policy(config)
            steps["evaluate"] = evaluate(config)
            steps["replay"] = replay(config, selected)
        result = steps
    else:
        raise AssertionError(args.command)
    _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
