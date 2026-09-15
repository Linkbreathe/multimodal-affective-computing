"""Live UDP/LSL service for the isolated adaptive-control runtime."""

from __future__ import annotations

import json
import socket
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from real_time_ml.config import ProjectConfig
from real_time_ml.features.eye import eye_features
from real_time_ml.features.head import head_features
from real_time_ml.features.physio import EEGMontage, StreamingPhysioProcessor
from real_time_ml.realtime.cycle import TenSecondCycleClock, TimeBuffer
from real_time_ml.utils import write_json, write_jsonl

from .contracts import (
    ControlProfile,
    AdaptiveControlCommand,
    AdaptiveReadinessSnapshot,
    AdaptiveStatusSnapshot,
    json_bytes,
    unix_milliseconds,
)
from .models import CompatibilityReport, ControlModelAdapter, load_adapter
from .physio_monitor import build_physio_snapshot
from .policy import ControlDecision, AdaptiveControlPolicy
from .settings import AdaptiveControlSettings


def _make_processor(config: ProjectConfig, participant_id: str | None) -> StreamingPhysioProcessor:
    return StreamingPhysioProcessor(
        sample_rate=float(config.get("streams.sample_rate_hz")),
        eeg_columns=list(config.get("streams.eeg_columns")),
        ecg_columns=list(config.get("streams.ecg_columns")),
        counter_column=int(config.get("streams.counter_column")),
        bands=dict(config.get("features.eeg.bands")),
        montage=EEGMontage.from_config(config),
        eeg_disabled=participant_id in set(config.get("participants.eeg_disabled")),
        strict_coverage_min=float(config.get("quality.eeg_strict_coverage_min")),
        eeg_abs_uV_max=float(config.get("quality.eeg_abs_uV_max")),
        eeg_flat_std_uV_min=float(config.get("quality.eeg_flat_std_uV_min")),
    )


def _expected_physio_channel_count(config: ProjectConfig) -> int:
    required_columns = [int(config.get("streams.counter_column"))]
    required_columns.extend(int(column) for column in config.get("streams.eeg_columns"))
    required_columns.extend(int(column) for column in config.get("streams.ecg_columns"))
    return max(required_columns) + 1


def verify_model(settings: AdaptiveControlSettings, bundle_id: str | None = None) -> CompatibilityReport:
    profile = ControlProfile.from_path(settings.profile_path)
    adapter = load_adapter(settings.bundle_manifest(bundle_id))
    return adapter.preflight(profile)


def list_models(settings: AdaptiveControlSettings) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if not settings.registry_path.exists():
        return output
    for manifest in sorted(settings.registry_path.glob("*/manifest.json")):
        try:
            adapter = load_adapter(manifest)
            report = adapter.preflight(ControlProfile.from_path(settings.profile_path))
            output.append({
                "bundle_id": report.descriptor.bundle_id,
                "version": report.descriptor.version,
                "adapter_id": report.descriptor.adapter_id,
                "compatible": report.compatible,
                "reasons": report.reasons,
            })
        except (OSError, ValueError, KeyError) as exc:
            output.append({"manifest": str(manifest), "compatible": False, "reasons": [str(exc)]})
    return output


class AdaptiveControlRuntime:
    """Owns one adaptive-control session and exposes status/log snapshots."""

    def __init__(self, config: ProjectConfig, settings: AdaptiveControlSettings, bundle_id: str | None = None) -> None:
        self.config = config
        self.settings = settings
        self.profile = ControlProfile.from_path(settings.profile_path)
        self.adapter: ControlModelAdapter = load_adapter(settings.bundle_manifest(bundle_id))
        self.compatibility = self.adapter.preflight(self.profile)
        if not self.compatibility.compatible:
            raise ValueError("Adaptive control model preflight failed: " + "; ".join(self.compatibility.reasons))
        self.policy = AdaptiveControlPolicy(
            self.profile,
            relaxation_weight=settings.utility_relaxation_weight,
            discomfort_weight=settings.utility_discomfort_weight,
            hysteresis=settings.utility_hysteresis,
            extreme_discomfort_limit=settings.extreme_discomfort_limit,
            calm_exploration_dwell_windows=settings.calm_exploration_dwell_windows,
            calm_exploration_penalty_per_window=settings.calm_exploration_penalty_per_window,
            calm_exploration_penalty_max=settings.calm_exploration_penalty_max,
            calm_exploration_relaxation_min=settings.calm_exploration_relaxation_min,
            calm_exploration_discomfort_max=settings.calm_exploration_discomfort_max,
            stochastic_exploration_enabled=settings.stochastic_exploration_enabled,
            exploration_candidate_scope=settings.exploration_candidate_scope,
            exploration_random_seed=settings.exploration_random_seed,
            exploration_temperature=settings.exploration_temperature,
            exploration_random_floor=settings.exploration_random_floor,
            sensor_conditioning_enabled=settings.sensor_conditioning_enabled,
            sensor_conditioning_weight=settings.sensor_conditioning_weight,
            switch_probability_enabled=settings.switch_probability_enabled,
            switch_probability_after_windows=settings.switch_probability_after_windows,
            switch_probability_force_after_windows=settings.switch_probability_force_after_windows,
            switch_probability_boredom_weight=settings.switch_probability_boredom_weight,
            switch_probability_arousal_weight=settings.switch_probability_arousal_weight,
            switch_probability_discomfort_weight=settings.switch_probability_discomfort_weight,
            switch_probability_stable_calm_weight=settings.switch_probability_stable_calm_weight,
            safety_discomfort_min=settings.safety_discomfort_min,
            safety_conditions=settings.safety_conditions,
            min_condition_dwell_windows=settings.min_condition_dwell_windows,
            max_condition_dwell_windows=settings.max_condition_dwell_windows,
            recent_history_window=settings.recent_history_window,
            recent_history_penalty=settings.recent_history_penalty,
            high_load_conditions=settings.high_load_conditions,
            high_load_penalty=settings.high_load_penalty,
            high_load_cooldown_windows=settings.high_load_cooldown_windows,
        )
        self.session_id: str | None = None
        self.participant_id: str | None = None
        self.current_condition = settings.initial_condition
        self.runtime_state = "Idle"
        self.last_command: AdaptiveControlCommand | None = None
        self.last_decision: ControlDecision | None = None
        self.last_coverage = {name: 0.0 for name in ("eeg", "ecg", "head", "eye")}
        self.last_seen: dict[str, int | None] = {
            "lsl": None, "unity": None, "ack": None, "eeg": None, "ecg": None, "head": None, "eye": None
        }
        self.low_modality_windows = 0
        self.condition_dwell_windows = 0
        self.condition_history = [self.current_condition]
        self.condition_cooldowns: dict[str, int] = {}
        self.locked_failsafe = False
        self.headset_presence_available = False
        self.headset_worn = False
        self.lsl_eeg_stream_found = False
        self._logs_dir: Path | None = None

    @property
    def model_descriptor(self):
        return self.adapter.descriptor

    def begin_session(self, session_id: str, participant_id: str | None) -> bool:
        if not session_id:
            raise ValueError("UnitySensorFrame must include session_id")
        if self.session_id and session_id != self.session_id and self.runtime_state not in {"Stopped", "Failsafe"}:
            raise ValueError("Finish or fail-safe the active Adaptive control session before starting another")
        if self.session_id != session_id:
            self.session_id = session_id
            self.participant_id = participant_id
            self.adapter.reset_session(session_id)
            self.runtime_state = "Warmup"
            self.current_condition = self.settings.initial_condition
            self.last_command = None
            self.last_decision = None
            self.last_coverage = {name: 0.0 for name in ("eeg", "ecg", "head", "eye")}
            self.last_seen = {
                "lsl": None, "unity": None, "ack": None, "eeg": None, "ecg": None, "head": None, "eye": None
            }
            self.low_modality_windows = 0
            self.condition_dwell_windows = 0
            self.condition_history = [self.current_condition]
            self.condition_cooldowns = {}
            self.locked_failsafe = False
            self.headset_presence_available = False
            self.headset_worn = False
            self._logs_dir = self.config.path("realtime_logs") / "adaptive_control" / session_id
            self._logs_dir.mkdir(parents=True, exist_ok=True)
            write_json(self._logs_dir / "session_manifest.json", {
                "session_id": session_id,
                "participant_id": participant_id,
                "profile_id": self.profile.profile_id,
                "profile_sha256": self.profile.sha256,
                "model": asdict(self.model_descriptor),
                "adaptive_control_only": True,
            })
            return True
        return False

    def remember_condition(self, condition: str) -> None:
        self.condition_history.append(condition)
        keep = max(
            16,
            self.settings.recent_history_window * 4,
            self.settings.high_load_cooldown_windows + 4,
        )
        self.condition_history = self.condition_history[-keep:]

    def update_condition_memory(self, decision: ControlDecision) -> None:
        for condition, remaining in list(self.condition_cooldowns.items()):
            next_remaining = max(0, remaining - 1)
            if next_remaining:
                self.condition_cooldowns[condition] = next_remaining
            else:
                self.condition_cooldowns.pop(condition, None)
        observed = decision.target_condition if decision.action == "apply" else decision.current_condition
        self.remember_condition(observed)
        if decision.action == "apply" and observed in set(self.settings.high_load_conditions):
            self.condition_cooldowns[observed] = max(
                self.condition_cooldowns.get(observed, 0),
                self.settings.high_load_cooldown_windows,
            )

    def log(self, name: str, payload: dict[str, Any]) -> None:
        if self._logs_dir is not None:
            write_jsonl(self._logs_dir / name, [payload], append=True)

    def make_command(self, cycle_index: int, now_ms: int, decision: ControlDecision) -> AdaptiveControlCommand:
        values = self.profile.baseline if decision.action == "failsafe" else self.profile.values_for(decision.target_condition)
        return AdaptiveControlCommand(
            session_id=self.session_id or "",
            command_id=f"{self.session_id}:{cycle_index}",
            cycle_index=cycle_index,
            issued_unix_ms=now_ms,
            expires_unix_ms=now_ms + self.settings.command_expiry_ms,
            action=decision.action,
            current_condition=decision.current_condition,
            target_condition=decision.target_condition,
            intensity=values.intensity,
            frequency=values.frequency,
            transition_seconds=self.settings.transition_seconds,
            utility_delta=decision.utility_delta,
            predicted_relaxation=decision.estimate.relaxation if decision.estimate else None,
            predicted_discomfort=decision.estimate.discomfort if decision.estimate else None,
            model_variant=decision.estimate.model_variant if decision.estimate else "none",
            active_modalities=decision.estimate.active_modalities if decision.estimate else [],
            profile_id=self.profile.profile_id,
            profile_sha256=self.profile.sha256,
            reasons=decision.reasons,
        )

    def status(self, now_ms: int, next_decision_in_seconds: float, reasons: list[str] | None = None) -> AdaptiveStatusSnapshot:
        ages = {
            name: None if timestamp is None else float(max(0, now_ms - timestamp))
            for name, timestamp in self.last_seen.items()
        }
        estimate = self.last_decision.estimate if self.last_decision else None
        raw = estimate.raw if estimate else {}
        modality_names = sorted(self.last_coverage)
        candidate_utilities = dict(self.last_decision.candidate_utilities) if self.last_decision else {}
        candidate_conditions = sorted(candidate_utilities)
        modalities_used = raw.get("modalities_used") if isinstance(raw.get("modalities_used"), list) else (estimate.active_modalities if estimate else [])
        return AdaptiveStatusSnapshot(
            session_id=self.session_id,
            unix_time_ms=now_ms,
            runtime_state=self.runtime_state,
            next_decision_in_seconds=max(0.0, next_decision_in_seconds),
            current_condition=self.current_condition if self.session_id else None,
            target_condition=self.last_command.target_condition if self.last_command else None,
            modality_coverage=dict(self.last_coverage),
            modality_age_ms=ages,
            model_bundle_id=self.model_descriptor.bundle_id,
            model_version=self.model_descriptor.version,
            model_variant=estimate.model_variant if estimate else None,
            predictions={
                "relaxation": estimate.relaxation if estimate else None,
                "discomfort": estimate.discomfort if estimate else None,
            },
            candidate_utilities=candidate_utilities,
            last_command_id=self.last_command.command_id if self.last_command else None,
            modality_names=modality_names,
            modality_coverage_values=[float(self.last_coverage[name]) for name in modality_names],
            modality_age_values=[float(ages.get(name) or -1.0) for name in modality_names],
            candidate_conditions=candidate_conditions,
            candidate_utility_values=[float(candidate_utilities[name]) for name in candidate_conditions],
            relaxation=estimate.relaxation if estimate else None,
            discomfort=estimate.discomfort if estimate else None,
            prediction_source=str(raw.get("prediction_source", "")) or None,
            model_supervision=str(raw.get("supervision", "")) or None,
            motion_source=str(raw.get("motion_source", "")) or None,
            model_input_feature_count=int(float(raw.get("input_feature_count", 0) or 0)),
            model_available_feature_count=int(float(raw.get("available_feature_count", 0) or 0)),
            model_missing_feature_count=int(float(raw.get("missing_feature_count", 0) or 0)),
            model_modalities_used=[str(name) for name in modalities_used],
            headset_presence_available=self.headset_presence_available,
            headset_worn=self.headset_worn,
            lsl_eeg_stream_found=self.lsl_eeg_stream_found,
            lsl_eeg_sample_received=self.last_seen["eeg"] is not None,
            warmup_complete=estimate is not None,
            reasons=list(reasons or (self.last_decision.reasons if self.last_decision else [])),
        )


def _sensor_rows(message: dict[str, Any], timestamp: int) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    head = message.get("head_pose") or {}
    eye = message.get("eye") or {}
    head_row = None
    eye_row = None
    if all(name in head for name in ("head_position_x", "head_position_y", "head_position_z")):
        head_row = {
            "unix_time_ms": timestamp,
            "head_position_x": float(head["head_position_x"]),
            "head_position_y": float(head["head_position_y"]),
            "head_position_z": float(head["head_position_z"]),
            "head_angular_velocity_deg_s": float(head.get("head_angular_velocity_deg_s", float("nan"))),
        }
    if all(name in eye for name in ("gaze_direction_x", "gaze_direction_y", "gaze_direction_z")):
        eye_row = {
            "unix_time_ms": timestamp,
            "gaze_direction_x": float(eye["gaze_direction_x"]),
            "gaze_direction_y": float(eye["gaze_direction_y"]),
            "gaze_direction_z": float(eye["gaze_direction_z"]),
            "gaze_on_painting": float(eye.get("gaze_on_painting", float("nan"))),
        }
    return head_row, eye_row


def serve_adaptive_control(
    config: ProjectConfig,
    settings: AdaptiveControlSettings,
    *,
    bundle_id: str | None = None,
    max_cycles: int | None = None,
) -> dict[str, Any]:
    runtime = AdaptiveControlRuntime(config, settings, bundle_id)
    listen = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listen.bind((settings.listen_host, settings.unity_to_python_port))
    listen.setblocking(False)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    destination = (settings.python_send_host, settings.python_to_unity_port)
    lsl_inlet = None
    lsl_unix_offset = 0.0
    lsl_time_correction = 0.0
    lsl_stream_name = ""
    lsl_stream_type = ""
    lsl_channel_count = 0
    lsl_nominal_srate = 0.0
    pylsl_module = None
    last_lsl_resolve_ms = 0
    expected_physio_channels = _expected_physio_channel_count(config)
    last_physio_sample_channel_count = 0
    monitor_interval_ms = 200
    monitor_window_ms = 10_000
    monitor_max_points = 240
    last_physio_monitor_ms = 0

    def try_open_lsl_inlet(now_ms: int) -> None:
        """Discover EEG without requiring the operator to start it before this service."""
        nonlocal lsl_inlet, lsl_unix_offset, lsl_time_correction, lsl_stream_name, lsl_stream_type
        nonlocal lsl_channel_count, lsl_nominal_srate, pylsl_module, last_lsl_resolve_ms
        last_lsl_resolve_ms = now_ms
        try:
            if pylsl_module is None:
                import pylsl

                pylsl_module = pylsl
            streams = pylsl_module.resolve_byprop("type", config.get("streams.physio_type"), timeout=1.0)
            if streams:
                stream = streams[0]
                lsl_inlet = pylsl_module.StreamInlet(stream, max_buflen=30)
                lsl_stream_name = str(stream.name())
                lsl_stream_type = str(stream.type())
                lsl_channel_count = int(stream.channel_count())
                lsl_nominal_srate = float(stream.nominal_srate())
                lsl_unix_offset = time.time() - pylsl_module.local_clock()
                try:
                    lsl_time_correction = float(lsl_inlet.time_correction(timeout=1.0))
                except Exception:
                    lsl_time_correction = 0.0
                runtime.lsl_eeg_stream_found = True
                return
        except Exception:
            pass
        runtime.lsl_eeg_stream_found = False

    def record_physio_sample(sample: Any, timestamp: float | None, now_ms: int) -> None:
        nonlocal last_physio_sample_channel_count
        if sample is None:
            return
        try:
            last_physio_sample_channel_count = len(sample)
        except TypeError:
            last_physio_sample_channel_count = 0
        unix_ms = now_ms if timestamp is None else int((float(timestamp) + lsl_time_correction + lsl_unix_offset) * 1000)
        physio_buffer.add(unix_ms, sample)
        runtime.last_seen["lsl"] = unix_ms
        runtime.last_seen["eeg"] = unix_ms
        runtime.last_seen["ecg"] = unix_ms

    def pull_one_physio_sample(now_ms: int) -> None:
        if lsl_inlet is None:
            return
        try:
            sample, timestamp = lsl_inlet.pull_sample(timeout=0.05)
        except Exception:
            return
        record_physio_sample(sample, timestamp, now_ms)

    def readiness_snapshot(request_id: str | None, now_ms: int) -> AdaptiveReadinessSnapshot:
        model_ready = runtime.compatibility.compatible
        stream_found = lsl_inlet is not None
        sample_received = runtime.last_seen["lsl"] is not None
        sample_shape_valid = last_physio_sample_channel_count >= expected_physio_channels
        physio_ready = stream_found and sample_received and sample_shape_valid
        reasons: list[str] = []
        if not stream_found:
            reasons.append("no_type_eeg_lsl_stream")
        elif not sample_received:
            reasons.append("lsl_stream_found_but_no_sample")
        elif not sample_shape_valid:
            reasons.append(
                f"physio_sample_too_short:{last_physio_sample_channel_count}<{expected_physio_channels}"
            )
        if not model_ready:
            reasons.extend(runtime.compatibility.reasons)
        if not reasons:
            reasons.append("ready")
        return AdaptiveReadinessSnapshot(
            request_id=request_id,
            unix_time_ms=now_ms,
            python_ready=True,
            model_ready=model_ready,
            model_bundle_id=runtime.model_descriptor.bundle_id,
            model_version=runtime.model_descriptor.version,
            lsl_eeg_stream_found=stream_found,
            lsl_eeg_sample_received=sample_received,
            eeg_ready=physio_ready,
            ecg_ready=physio_ready,
            physio_sample_channel_count=last_physio_sample_channel_count,
            expected_min_channel_count=expected_physio_channels,
            reasons=reasons,
        )

    def send_physio_snapshot(now_ms: int) -> None:
        nonlocal last_physio_monitor_ms
        if now_ms - last_physio_monitor_ms < monitor_interval_ms:
            return
        start_ms = now_ms - monitor_window_ms
        rows = [value for timestamp, value in physio_buffer.rows if start_ms <= timestamp <= now_ms]
        snapshot = build_physio_snapshot(
            config,
            rows,
            now_ms,
            stream_found=lsl_inlet is not None,
            sample_received=runtime.last_seen["lsl"] is not None,
            last_sample_ms=runtime.last_seen["lsl"],
            stream_name=lsl_stream_name,
            stream_type=lsl_stream_type,
            channel_count=lsl_channel_count,
            nominal_srate=lsl_nominal_srate,
            window_seconds=monitor_window_ms / 1000.0,
            max_points=monitor_max_points,
        )
        sender.sendto(json_bytes(snapshot.to_dict()), destination)
        last_physio_monitor_ms = now_ms
        prune_before = now_ms - 30_000
        physio_buffer.rows = [(timestamp, value) for timestamp, value in physio_buffer.rows if timestamp >= prune_before]

    now_ms = int(time.time() * 1000)
    try_open_lsl_inlet(now_ms)
    clock = TenSecondCycleClock()
    clock.initialize(now_ms, settings.initial_condition)
    runtime.lsl_eeg_stream_found = lsl_inlet is not None
    processor = _make_processor(config, None)
    physio_buffer, head_buffer, eye_buffer = TimeBuffer(), TimeBuffer(), TimeBuffer()
    produced = 0
    last_status_ms = 0
    try:
        while max_cycles is None or produced < max_cycles:
            now_ms = int(time.time() * 1000)
            if lsl_inlet is None and now_ms - last_lsl_resolve_ms >= 3_000:
                try_open_lsl_inlet(now_ms)
            # Drain every available Unity message; processing just one datagram per loop lets 10 Hz frames go stale.
            while True:
                try:
                    raw, _ = listen.recvfrom(1_000_000)
                except BlockingIOError:
                    break
                try:
                    message = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                message_type = message.get("message_type")
                if message_type == "AdaptiveReadinessRequest":
                    if message.get("protocol_version") != "adaptive-control-v1":
                        continue
                    if lsl_inlet is None:
                        try_open_lsl_inlet(now_ms)
                    pull_one_physio_sample(now_ms)
                    snapshot = readiness_snapshot(str(message.get("request_id", "")) or None, now_ms)
                    sender.sendto(json_bytes(snapshot.to_dict()), destination)
                    continue
                if message_type == "AdaptiveControlAck":
                    runtime.last_seen["ack"] = now_ms
                    runtime.log("acks.jsonl", message)
                    if message.get("status") == "stopped":
                        runtime.runtime_state = "Stopped"
                    continue
                if message_type != "UnitySensorFrame" or message.get("protocol_version") != "adaptive-control-v1":
                    continue
                timestamp = unix_milliseconds(message.get("unix_time_ms"), now_ms)
                try:
                    new_session = runtime.begin_session(str(message.get("session_id", "")), message.get("participant_id"))
                    if new_session:
                        clock.initialize(timestamp, settings.initial_condition)
                        clock.origin_ms = timestamp
                        physio_buffer.clear()
                        head_buffer.clear()
                        eye_buffer.clear()
                        processor = _make_processor(config, runtime.participant_id)
                    runtime.headset_presence_available = bool(message.get("headset_presence_available", False))
                    runtime.headset_worn = bool(message.get("headset_worn", False))
                    incoming_condition = str(message.get("condition_id", runtime.current_condition))
                    if incoming_condition != runtime.current_condition:
                        if clock.condition_event(incoming_condition, timestamp):
                            runtime.current_condition = incoming_condition
                            runtime.condition_dwell_windows = 0
                            runtime.remember_condition(incoming_condition)
                            physio_buffer.clear()
                            head_buffer.clear()
                            eye_buffer.clear()
                    head_row, eye_row = _sensor_rows(message, timestamp)
                except (TypeError, ValueError) as exc:
                    runtime.log("rejected_sensor_frames.jsonl", {"unix_time_ms": now_ms, "reason": str(exc), "payload": message})
                    continue
                runtime.last_seen["unity"] = timestamp
                if head_row:
                    head_buffer.add(timestamp, head_row)
                    runtime.last_seen["head"] = timestamp
                if eye_row:
                    eye_buffer.add(timestamp, eye_row)
                    runtime.last_seen["eye"] = timestamp
                runtime.log("sensor_frames.jsonl", message)
            if lsl_inlet is not None:
                chunk, timestamps = lsl_inlet.pull_chunk(timeout=0.0, max_samples=1024)
                for sample, timestamp in zip(chunk, timestamps):
                    record_physio_sample(sample, timestamp, now_ms)
            send_physio_snapshot(now_ms)
            if runtime.session_id:
                for cycle_index, start_ms, end_ms in clock.ready_windows(now_ms):
                    # The condition clock resets when Unity finishes a transition, but
                    # Unity command validation requires a session-wide monotonic id.
                    command_cycle_index = produced
                    physio_rows = physio_buffer.window(start_ms, end_ms)
                    features: dict[str, float] = {}
                    qc: dict[str, Any] = {}
                    if physio_rows:
                        physio_features, physio_qc = processor.process_window(np.asarray(physio_rows, dtype=float))
                        features.update(physio_features)
                        qc.update(physio_qc)
                    head, head_qc = head_features(head_buffer.window(start_ms, end_ms))
                    eye, eye_qc = eye_features(eye_buffer.window(start_ms, end_ms), float(config.get("features.eye.ivt_velocity_threshold_deg_s")))
                    features.update(head)
                    features.update(eye)
                    qc.update(head_qc)
                    qc.update(eye_qc)
                    coverage = {
                        "eeg": float(qc.get("eeg_strict_coverage", 0.0)),
                        "ecg": float(qc.get("ecg_quality", 0.0)),
                        "head": float(qc.get("head_coverage", 0.0)),
                        "eye": float(qc.get("eye_coverage", 0.0)),
                    }
                    runtime.last_coverage = coverage
                    active = [name for name, value in coverage.items() if value >= settings.min_modality_coverage]
                    if runtime.locked_failsafe:
                        decision = ControlDecision("failsafe", runtime.current_condition, runtime.current_condition, None, {}, None, ["failsafe_locked"])
                    elif runtime.headset_presence_available and not runtime.headset_worn:
                        runtime.low_modality_windows = 0
                        runtime.runtime_state = "Degraded"
                        decision = ControlDecision("hold", runtime.current_condition, runtime.current_condition, None, {}, None, ["headset_not_worn"])
                    elif len(active) < settings.min_active_modalities:
                        runtime.low_modality_windows += 1
                        if runtime.low_modality_windows >= settings.consecutive_low_modality_windows:
                            runtime.runtime_state = "Failsafe"
                            runtime.locked_failsafe = True
                            decision = ControlDecision("failsafe", runtime.current_condition, runtime.current_condition, None, {}, None, ["low_modality_coverage_timeout"])
                        else:
                            runtime.runtime_state = "Degraded"
                            decision = ControlDecision("hold", runtime.current_condition, runtime.current_condition, None, {}, None, ["insufficient_active_modalities"])
                    else:
                        runtime.low_modality_windows = 0
                        try:
                            decision = runtime.policy.decide(
                                runtime.adapter,
                                features,
                                runtime.current_condition,
                                coverage,
                                dwell_windows=runtime.condition_dwell_windows,
                                condition_history=runtime.condition_history,
                                condition_cooldowns=runtime.condition_cooldowns,
                            )
                            runtime.runtime_state = "Warmup" if cycle_index == 0 else "Running"
                            if decision.action == "failsafe":
                                runtime.runtime_state = "Failsafe"
                                runtime.locked_failsafe = True
                        except (RuntimeError, ValueError, KeyError) as exc:
                            runtime.runtime_state = "Failsafe"
                            runtime.locked_failsafe = True
                            decision = ControlDecision("failsafe", runtime.current_condition, runtime.current_condition, None, {}, None, [f"model_error:{exc}"])
                    command = runtime.make_command(command_cycle_index, now_ms, decision)
                    runtime.last_decision = decision
                    runtime.last_command = command
                    if decision.estimate is not None and decision.target_condition == runtime.current_condition:
                        runtime.condition_dwell_windows += 1
                    else:
                        runtime.condition_dwell_windows = 0
                    runtime.update_condition_memory(decision)
                    payload = command.to_dict()
                    sender.sendto(json_bytes(payload), destination)
                    runtime.log("decisions.jsonl", {
                        "window_start_ms": start_ms, "window_end_ms": end_ms, "coverage": coverage,
                        "qc": qc, "features": features, "decision": payload,
                        "condition_dwell_windows": runtime.condition_dwell_windows,
                        "condition_history": runtime.condition_history,
                        "condition_cooldowns": runtime.condition_cooldowns,
                        "prediction_source": decision.estimate.raw.get("prediction_source") if decision.estimate else None,
                        "modalities_used": decision.estimate.active_modalities if decision.estimate else [],
                        "model_diagnostics": decision.estimate.raw if decision.estimate else {},
                        "candidate_utilities": decision.candidate_utilities,
                    })
                    produced += 1
            if runtime.session_id and now_ms - last_status_ms >= int(1000 / settings.status_hz):
                next_due = (clock.origin_ms or now_ms) + (clock.cycle_index + 1) * clock.cycle_ms
                status = runtime.status(now_ms, (next_due - now_ms) / 1000.0)
                sender.sendto(json_bytes(status.to_dict()), destination)
                runtime.log("status.jsonl", status.to_dict())
                last_status_ms = now_ms
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        listen.close()
        sender.close()
        if runtime._logs_dir is not None:
            write_json(runtime._logs_dir / "summary.json", {
                "session_id": runtime.session_id,
                "cycles": produced,
                "runtime_state": runtime.runtime_state,
                "last_command_id": runtime.last_command.command_id if runtime.last_command else None,
            })
    return {
        "session_id": runtime.session_id,
        "cycles": produced,
        "model": runtime.model_descriptor.bundle_id,
        "runtime_state": runtime.runtime_state,
        "adaptive_control_only": True,
    }
