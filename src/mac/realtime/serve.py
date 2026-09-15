from __future__ import annotations

import json
import socket
import time
from typing import Any

import numpy as np

from mac.config import ProjectConfig
from mac.features.eye import eye_features
from mac.features.head import head_features
from mac.features.physio import EEGMontage, StreamingPhysioProcessor
from mac.realtime.cycle import TenSecondCycleClock, TimeBuffer
from mac.realtime.engine import InferenceEngine
from mac.utils import write_jsonl


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


def serve(config: ProjectConfig, max_cycles: int | None = None) -> dict[str, Any]:
    listen = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listen.bind((config.get("realtime.unity_listen_host"), int(config.get("realtime.unity_to_python_port"))))
    listen.setblocking(False)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    destination = (config.get("realtime.python_send_host"), int(config.get("realtime.python_to_unity_port")))
    lsl_inlet = None
    lsl_unix_offset = 0.0
    lsl_time_correction = 0.0
    try:
        import pylsl

        streams = pylsl.resolve_byprop("type", config.get("streams.physio_type"), timeout=1.0)
        if streams:
            lsl_inlet = pylsl.StreamInlet(streams[0], max_buflen=30)
            lsl_unix_offset = time.time() - pylsl.local_clock()
            try:
                lsl_time_correction = float(lsl_inlet.time_correction(timeout=1.0))
            except Exception:
                lsl_time_correction = 0.0
    except Exception:
        lsl_inlet = None
    clock = TenSecondCycleClock()
    clock.initialize(int(time.time() * 1000))
    physio_buffer, head_buffer, eye_buffer = TimeBuffer(), TimeBuffer(), TimeBuffer()
    engine = InferenceEngine(config)
    participant_id = None
    condition = None
    processor = _make_processor(config, participant_id)
    last_seen = {"lsl": None, "unity": None}
    output_messages: list[dict[str, Any]] = []
    produced = 0
    try:
        while max_cycles is None or produced < max_cycles:
            now_ms = int(time.time() * 1000)
            try:
                raw, _ = listen.recvfrom(1_000_000)
                message = json.loads(raw.decode("utf-8"))
                timestamp = int(message.get("unix_time_ms", now_ms))
                last_seen["unity"] = timestamp
                incoming_condition = message.get("condition") or message.get("condition_id")
                if incoming_condition:
                    changed = clock.condition_event(str(incoming_condition), timestamp)
                    if changed:
                        condition = str(incoming_condition)
                        physio_buffer.clear()
                        head_buffer.clear()
                        eye_buffer.clear()
                participant_id = message.get("participant_id", participant_id)
                if message.get("head_pose"):
                    head_buffer.add(timestamp, {"unix_time_ms": timestamp, **message["head_pose"]})
                if message.get("eye"):
                    eye_buffer.add(timestamp, {"unix_time_ms": timestamp, **message["eye"]})
            except (BlockingIOError, json.JSONDecodeError):
                pass
            if lsl_inlet is not None:
                chunk, timestamps = lsl_inlet.pull_chunk(timeout=0.0, max_samples=1024)
                for sample, timestamp in zip(chunk, timestamps):
                    unix_ms = int((timestamp + lsl_time_correction + lsl_unix_offset) * 1000)
                    physio_buffer.add(unix_ms, sample)
                    last_seen["lsl"] = unix_ms
            for cycle_index, start_ms, end_ms in clock.ready_windows(now_ms):
                physio_rows = physio_buffer.window(start_ms, end_ms)
                features: dict[str, Any] = {}
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
                    "video": 0.0,
                }
                reasons = []
                timeout_ms = int(float(config.get("quality.source_timeout_seconds")) * 1000)
                if last_seen["lsl"] is None or end_ms - last_seen["lsl"] > timeout_ms:
                    reasons.append("lsl_timeout")
                if last_seen["unity"] is None or end_ms - last_seen["unity"] > timeout_ms:
                    reasons.append("unity_timeout")
                state, recommendation = engine.infer(
                    participant_id=participant_id, condition=condition, cycle_index=cycle_index,
                    start_ms=start_ms, end_ms=end_ms, features=features, qc=qc, coverage=coverage,
                    force_hold_reasons=reasons,
                )
                for payload in (state.to_dict(), recommendation.to_dict()):
                    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                    sender.sendto(encoded, destination)
                    output_messages.append(payload)
                produced += 1
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        listen.close()
        sender.close()
        if output_messages:
            write_jsonl(config.path("realtime_logs") / "serve_shadow.jsonl", output_messages, append=True)
    return {"cycles": produced, "messages": len(output_messages), "shadow": True}
