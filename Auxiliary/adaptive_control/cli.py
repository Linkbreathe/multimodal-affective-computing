"""Command line entry point for the archived Adaptive Control runtime.

This module intentionally lives outside the installable thesis CLI. It remains
available for reproducing the Unity/UDP experiment, but it is not part of the
paper's active Shadow or offline-replay workflow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mac.config import load_config

from .settings import load_adaptive_control_settings


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adaptive-control", description="Archived Unity/UDP Adaptive Control runtime")
    parser.add_argument(
        "--control-config",
        type=Path,
        help="path to the archived adaptive-control.yaml configuration",
    )
    parser.add_argument(
        "--project-config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "project.yaml",
        help="research project.yaml used to construct the shared feature pipeline",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    control = commands.add_parser("adaptive-control", help="start the archived local adaptive-control service")
    control.add_argument("--bundle", help="registered adaptive-control model bundle id")
    control.add_argument("--max-cycles", type=int, help="test-only finite cycle count")
    model = commands.add_parser("adaptive-model", help="inspect archived adaptive-control model bundles")
    model.add_argument("action", choices=("list", "verify"))
    model.add_argument("--bundle", help="bundle id; required for verify")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_adaptive_control_settings(args.control_config)
    if args.command == "adaptive-model":
        from .service import list_models, verify_model

        if args.action == "list":
            _print({"models": list_models(settings)})
            return 0
        if not args.bundle:
            raise SystemExit("adaptive-model verify requires --bundle")
        report = verify_model(settings, args.bundle)
        _print({
            "compatible": report.compatible,
            "reasons": report.reasons,
            "descriptor": report.descriptor.__dict__,
        })
        return 0 if report.compatible else 2

    config = load_config(args.project_config)
    from .service import serve_adaptive_control

    try:
        result = serve_adaptive_control(
            config,
            settings,
            bundle_id=args.bundle,
            max_cycles=args.max_cycles,
        )
    except OSError as exc:
        _print({
            "ok": False,
            "error": "adaptive_control_startup_failed",
            "reason": str(exc),
            "listen_host": settings.listen_host,
            "unity_to_python_port": settings.unity_to_python_port,
            "python_send_host": settings.python_send_host,
            "python_to_unity_port": settings.python_to_unity_port,
        })
        return 2
    _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
