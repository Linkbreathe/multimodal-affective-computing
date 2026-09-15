"""Pluggable adapters for adaptive-control models.

The adapter boundary deliberately keeps model-specific loading and feature-history
handling out of the UDP/runtime and Unity control paths.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from real_time_ml.data.io import condition_parameters
from real_time_ml.modeling.condition_data import aggregate_realtime_history
from real_time_ml.modeling.realtime_multimodal import (
    ADAPTER_ID as REALTIME_MULTIMODAL_ADAPTER_ID,
    MODEL_KIND as REALTIME_MULTIMODAL_MODEL_KIND,
    SUPERVISION as REALTIME_MULTIMODAL_SUPERVISION,
    feature_modalities,
    validate_realtime_feature_columns,
)
from real_time_ml.modeling.train import load_state_model, predict_state

from .contracts import ControlProfile
from .policy import ControlEstimate


@dataclass(frozen=True)
class ModelDescriptor:
    bundle_id: str
    version: str
    adapter_id: str
    input_schema_version: str
    window_seconds: float
    required_modalities: list[str]
    optional_modalities: list[str]
    target_schema: list[str]
    candidate_prediction_supported: bool
    manifest_path: Path
    manifest_sha256: str


@dataclass(frozen=True)
class CompatibilityReport:
    compatible: bool
    reasons: list[str]
    descriptor: ModelDescriptor


class ControlModelAdapter(ABC):
    """Stable interface that every future Adaptive control model adapter must implement."""

    @property
    @abstractmethod
    def descriptor(self) -> ModelDescriptor: ...

    @abstractmethod
    def preflight(self, profile: ControlProfile) -> CompatibilityReport: ...

    @abstractmethod
    def reset_session(self, session_id: str) -> None: ...

    @abstractmethod
    def predict_current(self, features: dict[str, float], condition: str, coverage: dict[str, float]) -> ControlEstimate: ...

    @abstractmethod
    def predict_candidates(
        self,
        features: dict[str, float],
        current_condition: str,
        candidates: list[str],
        current: ControlEstimate,
    ) -> dict[str, ControlEstimate]: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ClassicalConditionAdapter(ControlModelAdapter):
    """Adapter around the existing classical condition and policy joblib bundles."""

    adapter_id = "classical_condition_v1"

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        raw = self.manifest_path.read_bytes()
        self.payload: dict[str, Any] = json.loads(raw.decode("utf-8"))
        self._descriptor = ModelDescriptor(
            bundle_id=str(self.payload["bundle_id"]),
            version=str(self.payload["version"]),
            adapter_id=str(self.payload["adapter_id"]),
            input_schema_version=str(self.payload["input_schema_version"]),
            window_seconds=float(self.payload["window_seconds"]),
            required_modalities=list(self.payload.get("required_modalities", [])),
            optional_modalities=list(self.payload.get("optional_modalities", [])),
            target_schema=list(self.payload["target_schema"]),
            candidate_prediction_supported=bool(self.payload.get("candidate_prediction_supported", False)),
            manifest_path=self.manifest_path,
            manifest_sha256=hashlib.sha256(raw).hexdigest(),
        )
        if self._descriptor.adapter_id != self.adapter_id:
            raise ValueError(f"Expected {self.adapter_id}; got {self._descriptor.adapter_id}")
        self._state_models: dict[str, dict[str, Any]] = {}
        self._policy_model: dict[str, Any] | None = None
        self._profile: ControlProfile | None = None
        self._session_id: str | None = None
        self._history: list[dict[str, float]] = []
        self._history_condition: str | None = None

    @property
    def descriptor(self) -> ModelDescriptor:
        return self._descriptor

    def _files(self) -> dict[str, Path]:
        result: dict[str, Path] = {}
        for name, spec in dict(self.payload.get("model_files", {})).items():
            if not isinstance(spec, dict) or "path" not in spec or "sha256" not in spec:
                raise ValueError(f"Invalid model file specification for {name}")
            path = (self.manifest_path.parent / str(spec["path"])).resolve()
            if not path.exists():
                raise FileNotFoundError(f"Model file missing: {path}")
            actual = _sha256(path)
            if actual.lower() != str(spec["sha256"]).lower():
                raise ValueError(f"SHA-256 mismatch for model file {path.name}")
            result[name] = path
        return result

    def preflight(self, profile: ControlProfile) -> CompatibilityReport:
        reasons: list[str] = []
        if self.descriptor.window_seconds != 10.0:
            reasons.append("model_window_seconds_must_be_10")
        if self.descriptor.target_schema != ["relaxation", "discomfort"]:
            reasons.append("target_schema_must_be_relaxation_discomfort")
        if not self.descriptor.candidate_prediction_supported:
            reasons.append("candidate_prediction_not_supported")
        try:
            files = self._files()
            required = {"full", "no_eeg", "behavior_only", "policy"}
            missing = sorted(required - set(files))
            if missing:
                reasons.append(f"missing_model_files:{','.join(missing)}")
            else:
                models = {name: load_state_model(files[name]) for name in required}
                for name in ("full", "no_eeg", "behavior_only"):
                    if list(models[name].get("targets", [])) != ["relaxation", "discomfort"]:
                        reasons.append(f"{name}_targets_incompatible")
                if list(models["policy"].get("targets", [])) != ["relaxation", "discomfort"]:
                    reasons.append("policy_targets_incompatible")
                if not reasons:
                    self._state_models = {name: models[name] for name in ("full", "no_eeg", "behavior_only")}
                    self._policy_model = models["policy"]
                    self._profile = profile
        except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
            reasons.append(str(exc))
        return CompatibilityReport(compatible=not reasons, reasons=reasons, descriptor=self.descriptor)

    def reset_session(self, session_id: str) -> None:
        if not self._state_models or self._policy_model is None or self._profile is None:
            raise RuntimeError("Adaptive control model adapter must pass preflight before starting a session")
        self._session_id = session_id
        self._history = []
        self._history_condition = None

    @staticmethod
    def _active_modalities(coverage: dict[str, float]) -> list[str]:
        return sorted(name for name, value in coverage.items() if float(value) >= 0.2)

    def _select_variant(self, coverage: dict[str, float]) -> str:
        active = set(self._active_modalities(coverage))
        if {"eeg", "ecg", "head", "eye"}.issubset(active):
            return "full"
        if {"ecg", "head", "eye"}.issubset(active):
            return "no_eeg"
        if {"head", "eye"}.issubset(active):
            return "behavior_only"
        raise ValueError("fewer_than_two_usable_modalities")

    def _aggregate(self, bundle: dict[str, Any], features: dict[str, float], condition: str) -> dict[str, float]:
        if self._profile is None:
            raise RuntimeError("Adaptive control model adapter profile is not loaded")
        if self._history_condition != condition:
            self._history = []
            self._history_condition = condition
        self._history.append(dict(features))
        if bundle.get("model_kind") != "condition_residual_ensemble_v1":
            return features
        values = [self._profile.values_for(f"C{row * 3 + 1}").intensity for row in range(3)]
        frequencies = [self._profile.values_for(f"C{column + 1}").frequency for column in range(3)]
        static = {"condition": condition}
        static.update(condition_parameters(condition, values, frequencies))
        return aggregate_realtime_history(self._history, bundle["feature_columns"], static)

    def predict_current(self, features: dict[str, float], condition: str, coverage: dict[str, float]) -> ControlEstimate:
        variant = self._select_variant(coverage)
        bundle = self._state_models[variant]
        active_features = self._aggregate(bundle, features, condition)
        prediction = predict_state(bundle, active_features, condition)
        return ControlEstimate(
            relaxation=float(prediction["relaxation"]),
            discomfort=float(prediction["discomfort"]),
            model_variant=variant,
            active_modalities=self._active_modalities(coverage),
            raw={name: float(value) for name, value in prediction.items()},
        )

    def predict_candidates(
        self,
        features: dict[str, float],
        current_condition: str,
        candidates: list[str],
        current: ControlEstimate,
    ) -> dict[str, ControlEstimate]:
        if self._policy_model is None or self._profile is None:
            raise RuntimeError("Adaptive control model adapter is not loaded")
        intensities = [self._profile.values_for(f"C{row * 3 + 1}").intensity for row in range(3)]
        frequencies = [self._profile.values_for(f"C{column + 1}").frequency for column in range(3)]
        output: dict[str, ControlEstimate] = {}
        for candidate in candidates:
            context = condition_parameters(candidate, intensities, frequencies)
            context.update({
                "previous_relaxation": current.relaxation,
                "previous_discomfort": current.discomfort,
                "participant_baseline_relaxation": current.relaxation,
                "participant_baseline_discomfort": current.discomfort,
                "presentation_position": int(candidate[1:]),
            })
            prediction = predict_state(self._policy_model, context)
            output[candidate] = ControlEstimate(
                relaxation=float(prediction["relaxation"]),
                discomfort=float(prediction["discomfort"]),
                model_variant=current.model_variant,
                active_modalities=current.active_modalities,
                raw={name: float(value) for name, value in prediction.items()},
            )
        return output


class RealtimeMultimodalWindowAdapter(ControlModelAdapter):
    """Adapter for direct 10-second EEG/ECG/eye/HMD-motion window inference."""

    adapter_id = REALTIME_MULTIMODAL_ADAPTER_ID

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        raw = self.manifest_path.read_bytes()
        self.payload: dict[str, Any] = json.loads(raw.decode("utf-8"))
        self._descriptor = ModelDescriptor(
            bundle_id=str(self.payload["bundle_id"]),
            version=str(self.payload["version"]),
            adapter_id=str(self.payload["adapter_id"]),
            input_schema_version=str(self.payload["input_schema_version"]),
            window_seconds=float(self.payload["window_seconds"]),
            required_modalities=list(self.payload.get("required_modalities", [])),
            optional_modalities=list(self.payload.get("optional_modalities", [])),
            target_schema=list(self.payload["target_schema"]),
            candidate_prediction_supported=bool(self.payload.get("candidate_prediction_supported", False)),
            manifest_path=self.manifest_path,
            manifest_sha256=hashlib.sha256(raw).hexdigest(),
        )
        if self._descriptor.adapter_id != self.adapter_id:
            raise ValueError(f"Expected {self.adapter_id}; got {self._descriptor.adapter_id}")
        self._bundle: dict[str, Any] | None = None
        self._profile: ControlProfile | None = None
        self._session_id: str | None = None

    @property
    def descriptor(self) -> ModelDescriptor:
        return self._descriptor

    def _model_path(self) -> Path:
        files = dict(self.payload.get("model_files", {}))
        spec = files.get("full")
        if not isinstance(spec, dict) or "path" not in spec or "sha256" not in spec:
            raise ValueError("Realtime multimodal bundle requires model_files.full path and sha256")
        path = (self.manifest_path.parent / str(spec["path"])).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Model file missing: {path}")
        actual = _sha256(path)
        if actual.lower() != str(spec["sha256"]).lower():
            raise ValueError(f"SHA-256 mismatch for model file {path.name}")
        return path

    def preflight(self, profile: ControlProfile) -> CompatibilityReport:
        reasons: list[str] = []
        if self.descriptor.window_seconds != 10.0:
            reasons.append("model_window_seconds_must_be_10")
        if self.descriptor.target_schema != ["relaxation", "discomfort"]:
            reasons.append("target_schema_must_be_relaxation_discomfort")
        try:
            bundle = load_state_model(self._model_path())
            if bundle.get("model_kind") != REALTIME_MULTIMODAL_MODEL_KIND:
                reasons.append("model_kind_must_be_realtime_multimodal_window_v1")
            if list(bundle.get("targets", [])) != ["relaxation", "discomfort"]:
                reasons.append("model_targets_incompatible")
            validate_realtime_feature_columns(list(bundle.get("feature_columns", [])))
            if not reasons:
                self._bundle = bundle
                self._profile = profile
        except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
            reasons.append(str(exc))
        return CompatibilityReport(compatible=not reasons, reasons=reasons, descriptor=self.descriptor)

    def reset_session(self, session_id: str) -> None:
        if self._bundle is None or self._profile is None:
            raise RuntimeError("Realtime multimodal adapter must pass preflight before starting a session")
        self._session_id = session_id

    @staticmethod
    def _finite_feature_names(features: dict[str, float], columns: list[str]) -> list[str]:
        present: list[str] = []
        for name in columns:
            value = features.get(name)
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric == numeric and numeric not in (float("inf"), float("-inf")):
                present.append(name)
        return present

    @staticmethod
    def _modalities_from_features(columns: list[str]) -> list[str]:
        return feature_modalities(columns)

    def predict_current(self, features: dict[str, float], condition: str, coverage: dict[str, float]) -> ControlEstimate:
        if self._bundle is None:
            raise RuntimeError("Realtime multimodal model is not loaded")
        columns = list(self._bundle["feature_columns"])
        present = self._finite_feature_names(features, columns)
        missing = len(columns) - len(present)
        prediction = predict_state(self._bundle, features, None)
        modalities_used = self._modalities_from_features(present)
        confidence = float(len(present) / len(columns)) if columns else 0.0
        raw = {
            "prediction_source": "window_multimodal_model",
            "supervision": str(self._bundle.get("supervision", REALTIME_MULTIMODAL_SUPERVISION)),
            "motion_source": "HMD Motion",
            "input_feature_count": float(len(columns)),
            "available_feature_count": float(len(present)),
            "missing_feature_count": float(missing),
            "modalities_used": modalities_used,
        }
        return ControlEstimate(
            relaxation=float(prediction["relaxation"]),
            discomfort=float(prediction["discomfort"]),
            model_variant="full",
            active_modalities=modalities_used,
            raw=raw,
            confidence=confidence,
        )

    def _condition_load(self, condition: str) -> float:
        if self._profile is None:
            raise RuntimeError("Realtime multimodal adapter profile is not loaded")
        values = self._profile.values_for(condition)
        max_intensity = max(item.intensity for item in self._profile.conditions.values()) or 1.0
        max_frequency = max(item.frequency for item in self._profile.conditions.values()) or 1.0
        return 0.5 * (values.intensity / max_intensity) + 0.5 * (values.frequency / max_frequency)

    def predict_candidates(
        self,
        features: dict[str, float],
        current_condition: str,
        candidates: list[str],
        current: ControlEstimate,
    ) -> dict[str, ControlEstimate]:
        current_load = self._condition_load(current_condition)
        high_stress = current.discomfort >= 0.50 or current.relaxation <= 0.35
        calm = current.relaxation >= 0.55 and current.discomfort <= 0.40
        steady = current.relaxation >= 0.45 and current.discomfort <= 0.50
        output: dict[str, ControlEstimate] = {}
        for candidate in candidates:
            load_delta = self._condition_load(candidate) - current_load
            if high_stress:
                desirability = -load_delta
                mode = "stress_recovery"
            elif calm:
                desirability = load_delta
                mode = "calm_exploration"
            elif steady:
                desirability = 0.35 * load_delta if load_delta >= 0.0 else 0.10 * load_delta
                mode = "stable_probe"
            else:
                desirability = -abs(load_delta)
                mode = "conservative_stay"
            relaxation = max(0.0, min(1.0, current.relaxation + 0.10 * desirability))
            discomfort = max(0.0, min(1.0, current.discomfort - 0.08 * desirability))
            raw = dict(current.raw)
            raw["candidate_utility_source"] = "reactive_policy_from_current_model_output"
            raw["candidate_policy_mode"] = mode
            output[candidate] = ControlEstimate(
                relaxation=relaxation,
                discomfort=discomfort,
                model_variant=current.model_variant,
                active_modalities=current.active_modalities,
                raw=raw,
                confidence=current.confidence,
            )
        return output


class DcnnSequenceAdapter(ControlModelAdapter):
    """Reserved adapter identifier with a clear preflight result until a DCNN bundle is supplied."""

    def __init__(self, manifest_path: str | Path) -> None:
        raw = Path(manifest_path).read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        self._descriptor = ModelDescriptor(
            bundle_id=str(payload["bundle_id"]), version=str(payload["version"]), adapter_id="dcnn_sequence_v1",
            input_schema_version=str(payload.get("input_schema_version", "")), window_seconds=float(payload.get("window_seconds", 0)),
            required_modalities=list(payload.get("required_modalities", [])), optional_modalities=list(payload.get("optional_modalities", [])),
            target_schema=list(payload.get("target_schema", [])), candidate_prediction_supported=False,
            manifest_path=Path(manifest_path).resolve(), manifest_sha256=hashlib.sha256(raw).hexdigest(),
        )

    @property
    def descriptor(self) -> ModelDescriptor:
        return self._descriptor

    def preflight(self, profile: ControlProfile) -> CompatibilityReport:
        return CompatibilityReport(False, ["dcnn_sequence_v1_adapter_not_implemented_for_this_adaptive_control_release"], self.descriptor)

    def reset_session(self, session_id: str) -> None:
        raise RuntimeError("DCNN adaptive-control adapter is unavailable")

    def predict_current(self, features: dict[str, float], condition: str, coverage: dict[str, float]) -> ControlEstimate:
        raise RuntimeError("DCNN adaptive-control adapter is unavailable")

    def predict_candidates(self, features: dict[str, float], current_condition: str, candidates: list[str], current: ControlEstimate) -> dict[str, ControlEstimate]:
        raise RuntimeError("DCNN adaptive-control adapter is unavailable")


def load_adapter(manifest_path: str | Path) -> ControlModelAdapter:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    adapter_id = str(payload.get("adapter_id", ""))
    if adapter_id == ClassicalConditionAdapter.adapter_id:
        return ClassicalConditionAdapter(manifest_path)
    if adapter_id == RealtimeMultimodalWindowAdapter.adapter_id:
        return RealtimeMultimodalWindowAdapter(manifest_path)
    if adapter_id == "dcnn_sequence_v1":
        return DcnnSequenceAdapter(manifest_path)
    raise ValueError(f"Unsupported adaptive-control model adapter: {adapter_id}")
