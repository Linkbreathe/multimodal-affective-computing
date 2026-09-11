"""Leakage-safe compression primitives for frozen multimodal window embeddings.

This module is the July 18, 2026 revision of the Project B compression path.
Unlike :mod:`src.fusion.frozen_compression`, every unsupervised reducer here is
fitted on *outer-training windows* before condition pooling.  Each
participant-condition contributes total weight one, irrespective of how many
common-valid windows it contains.  All public estimators retain the exact
observation indexes and participant IDs used to fit their state.

The module deliberately contains no target predictor and no split discovery.
Callers must supply the already-fixed outer-training observation indexes.  The
only label-conditioned object, :class:`TargetwiseModalityPLS1`, accepts a full
target array but indexes it exclusively by ``fit_indexes``.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.covariance import LedoitWolf


FIVE_MODALITIES = ("eeg", "ecg", "eye", "head", "video")
DEFAULT_TOTAL_COMPONENTS = 12


def array_sha256(values: np.ndarray) -> str:
    """Return a shape- and dtype-aware hash for fitted numerical state."""

    array = np.ascontiguousarray(values)
    digest = sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _indexes(indexes: np.ndarray | Iterable[int], length: int) -> np.ndarray:
    result = np.asarray(
        list(indexes) if not isinstance(indexes, np.ndarray) else indexes,
        dtype=int,
    )
    if result.ndim != 1 or len(result) == 0:
        raise ValueError("indexes must be a nonempty one-dimensional array")
    if result.min() < 0 or result.max() >= length:
        raise ValueError("indexes contain an out-of-range observation")
    if len(np.unique(result)) != len(result):
        raise ValueError("indexes contain duplicate observations")
    return result


def _modalities(modalities: Iterable[str]) -> tuple[str, ...]:
    result = tuple(str(value) for value in modalities)
    if not result:
        raise ValueError("At least one modality is required")
    if len(set(result)) != len(result):
        raise ValueError("Modalities must be unique")
    unknown = set(result).difference(FIVE_MODALITIES)
    if unknown:
        raise ValueError(f"Unknown modalities: {sorted(unknown)}")
    expected = tuple(value for value in FIVE_MODALITIES if value in result)
    if result != expected:
        raise ValueError(f"Modalities must follow canonical order {FIVE_MODALITIES}")
    return result


def _participant_evidence(
    indexes: np.ndarray, participant_ids: Sequence[str] | np.ndarray | None, length: int
) -> list[str]:
    if participant_ids is None:
        return []
    values = np.asarray(participant_ids, dtype=str)
    if values.ndim != 1 or len(values) != length:
        raise ValueError("participant_ids must have one entry per observation")
    return sorted(np.unique(values[indexes]).tolist())


def _validate_window_block(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if array.ndim != 3:
        raise ValueError("Window embeddings must have shape [observation, window, dimension]")
    if valid.shape != array.shape[:2]:
        raise ValueError("Window mask does not match embedding observations/windows")
    if array.shape[2] == 0:
        raise ValueError("Embedding dimension must be positive")
    if not np.isfinite(array[valid]).all():
        raise ValueError("Valid embedding windows contain non-finite values")
    return array, valid


def validate_window_inputs(
    blocks: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    modalities: Iterable[str],
) -> int:
    """Validate a multimodal window dictionary and return observation count."""

    names = _modalities(modalities)
    shapes: set[tuple[int, int]] = set()
    for modality in names:
        if modality not in blocks or modality not in masks:
            raise ValueError(f"Missing block or mask for {modality}")
        array, valid = _validate_window_block(blocks[modality], masks[modality])
        shapes.add((len(array), valid.shape[1]))
    if len(shapes) != 1:
        raise ValueError("All modality blocks must share observation and window axes")
    return next(iter(shapes))[0]


@dataclass(frozen=True)
class WindowCollection:
    """Flattened windows with equal total weight per nonempty observation."""

    values: np.ndarray
    weights: np.ndarray
    observation_indexes: np.ndarray
    window_indexes: np.ndarray
    fit_observation_indexes: np.ndarray
    nonempty_observation_indexes: np.ndarray

    def evidence(self) -> dict[str, Any]:
        totals = np.bincount(
            self.observation_indexes,
            weights=self.weights,
            minlength=int(self.fit_observation_indexes.max()) + 1,
        )
        nonempty_totals = totals[self.nonempty_observation_indexes]
        return {
            "fit_observation_indexes": self.fit_observation_indexes.tolist(),
            "nonempty_observation_indexes": self.nonempty_observation_indexes.tolist(),
            "window_count": int(len(self.values)),
            "nonempty_observation_count": int(len(self.nonempty_observation_indexes)),
            "per_nonempty_observation_weight_min": float(nonempty_totals.min()),
            "per_nonempty_observation_weight_max": float(nonempty_totals.max()),
            "weight_sum": float(self.weights.sum()),
        }


def collect_condition_balanced_windows(
    values: np.ndarray,
    mask: np.ndarray,
    fit_indexes: np.ndarray | Iterable[int],
) -> WindowCollection:
    """Collect train windows with each nonempty condition receiving weight one."""

    array, valid = _validate_window_block(values, mask)
    indexes = _indexes(fit_indexes, len(array))
    pieces: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    observations: list[np.ndarray] = []
    windows: list[np.ndarray] = []
    nonempty: list[int] = []
    for observation in indexes:
        selected = np.flatnonzero(valid[observation])
        if len(selected) == 0:
            continue
        nonempty.append(int(observation))
        pieces.append(array[observation, selected])
        weights.append(np.full(len(selected), 1.0 / len(selected), dtype=np.float64))
        observations.append(np.full(len(selected), observation, dtype=int))
        windows.append(selected.astype(int, copy=False))
    if not pieces:
        raise ValueError("No valid outer-training windows were supplied")
    return WindowCollection(
        values=np.concatenate(pieces, axis=0),
        weights=np.concatenate(weights),
        observation_indexes=np.concatenate(observations),
        window_indexes=np.concatenate(windows),
        fit_observation_indexes=indexes.copy(),
        nonempty_observation_indexes=np.asarray(nonempty, dtype=int),
    )


def common_window_mask(
    masks: Mapping[str, np.ndarray], modalities: Iterable[str]
) -> np.ndarray:
    names = _modalities(modalities)
    arrays = [np.asarray(masks[modality], dtype=bool) for modality in names]
    if len({value.shape for value in arrays}) != 1:
        raise ValueError("Modality masks do not share a common shape")
    return np.logical_and.reduce(arrays)


class WeightedStandardizer:
    """Coordinate standardization with explicit nonnegative sample weights."""

    def __init__(self, variance_floor: float = 1e-12) -> None:
        self.variance_floor = float(variance_floor)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray, weights: np.ndarray | None = None) -> "WeightedStandardizer":
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or len(array) == 0 or not np.isfinite(array).all():
            raise ValueError("Standardizer values must be a finite nonempty matrix")
        sample_weights = (
            np.ones(len(array), dtype=np.float64)
            if weights is None
            else np.asarray(weights, dtype=np.float64)
        )
        if sample_weights.shape != (len(array),) or (sample_weights < 0).any():
            raise ValueError("Invalid standardizer weights")
        total = float(sample_weights.sum())
        if total <= 0:
            raise ValueError("Standardizer weights must have positive sum")
        self.mean_ = np.sum(array * sample_weights[:, None], axis=0) / total
        variance = np.sum(
            (array - self.mean_) ** 2 * sample_weights[:, None], axis=0
        ) / total
        self.scale_ = np.sqrt(np.maximum(variance, self.variance_floor))
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("WeightedStandardizer has not been fitted")
        array = np.asarray(values, dtype=np.float64)
        return (array - self.mean_) / self.scale_

    def provenance(self) -> dict[str, Any]:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("WeightedStandardizer has not been fitted")
        return {
            "mean_sha256": array_sha256(self.mean_),
            "scale_sha256": array_sha256(self.scale_),
            "coordinate_count": int(len(self.mean_)),
        }


def _canonicalize_component_rows(components: np.ndarray) -> np.ndarray:
    result = np.asarray(components, dtype=np.float64).copy()
    for row in result:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1.0
    return result


def _canonicalize_component_columns(components: np.ndarray) -> np.ndarray:
    return _canonicalize_component_rows(np.asarray(components).T).T


@dataclass(frozen=True)
class Spectrum:
    eigenvalues: np.ndarray
    entropy_effective_rank: float
    numeric_rank: int

    def as_dict(self, include_eigenvalues: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "entropy_effective_rank": float(self.entropy_effective_rank),
            "numeric_rank": int(self.numeric_rank),
            "positive_eigenvalue_sum": float(self.eigenvalues.sum()),
        }
        if include_eigenvalues:
            result["eigenvalues"] = self.eigenvalues.tolist()
        return result


def weighted_spectrum(values: np.ndarray, weights: np.ndarray) -> Spectrum:
    """Return covariance spectrum, entropy rank and numerical matrix rank."""

    array = np.asarray(values, dtype=np.float64)
    sample_weights = np.asarray(weights, dtype=np.float64)
    if array.ndim != 2 or sample_weights.shape != (len(array),):
        raise ValueError("Spectrum input/weight shapes are invalid")
    total = float(sample_weights.sum())
    if total <= 0:
        raise ValueError("Spectrum weights must have positive sum")
    mean = np.sum(array * sample_weights[:, None], axis=0) / total
    weighted = (array - mean) * np.sqrt(sample_weights[:, None] / total)
    singular_values = np.linalg.svd(weighted, full_matrices=False, compute_uv=False)
    eigenvalues = singular_values**2
    if len(singular_values) == 0 or singular_values[0] == 0:
        numeric_rank = 0
    else:
        tolerance = max(weighted.shape) * np.finfo(np.float64).eps * singular_values[0]
        numeric_rank = int(np.sum(singular_values > tolerance))
    positive = eigenvalues[eigenvalues > 0]
    if len(positive) == 0:
        effective = 0.0
    else:
        probabilities = positive / positive.sum()
        effective = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    return Spectrum(eigenvalues=eigenvalues, entropy_effective_rank=effective, numeric_rank=numeric_rank)


class WeightedPCA:
    """Deterministic weighted PCA implemented by SVD of weighted observations."""

    def __init__(self, n_components: int, clip_to_rank: bool = False) -> None:
        self.n_components = int(n_components)
        self.clip_to_rank = bool(clip_to_rank)
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.numeric_rank_: int | None = None
        self.spectrum_: Spectrum | None = None

    def fit(self, values: np.ndarray, weights: np.ndarray | None = None) -> "WeightedPCA":
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or len(array) == 0 or not np.isfinite(array).all():
            raise ValueError("PCA values must be a finite nonempty matrix")
        sample_weights = (
            np.ones(len(array), dtype=np.float64)
            if weights is None
            else np.asarray(weights, dtype=np.float64)
        )
        if sample_weights.shape != (len(array),) or (sample_weights < 0).any():
            raise ValueError("Invalid PCA weights")
        total = float(sample_weights.sum())
        if total <= 0:
            raise ValueError("PCA weights must have positive sum")
        self.mean_ = np.sum(array * sample_weights[:, None], axis=0) / total
        weighted = (array - self.mean_) * np.sqrt(sample_weights[:, None] / total)
        _, singular_values, components = np.linalg.svd(weighted, full_matrices=False)
        tolerance = (
            max(weighted.shape) * np.finfo(np.float64).eps * singular_values[0]
            if len(singular_values) and singular_values[0] > 0
            else 0.0
        )
        self.numeric_rank_ = int(np.sum(singular_values > tolerance))
        eigenvalues = singular_values**2
        positive = eigenvalues
        positive = positive[positive > 0]
        if len(positive):
            probabilities = positive / positive.sum()
            effective = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
        else:
            effective = 0.0
        self.spectrum_ = Spectrum(
            eigenvalues=eigenvalues,
            entropy_effective_rank=effective,
            numeric_rank=self.numeric_rank_,
        )
        if self.clip_to_rank and self.n_components > self.numeric_rank_:
            self.n_components = self.numeric_rank_
        if self.n_components < 1 or self.n_components > self.numeric_rank_:
            raise ValueError(
                f"Requested PCA dimension {self.n_components} exceeds numeric rank {self.numeric_rank_}"
            )
        self.components_ = _canonicalize_component_rows(components[: self.n_components])
        self.explained_variance_ = eigenvalues[: self.n_components]
        denominator = float(eigenvalues.sum())
        self.explained_variance_ratio_ = (
            self.explained_variance_ / denominator
            if denominator > 0
            else np.zeros_like(self.explained_variance_)
        )
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("WeightedPCA has not been fitted")
        array = np.asarray(values, dtype=np.float64)
        return (array - self.mean_) @ self.components_.T

    def provenance(self) -> dict[str, Any]:
        if self.components_ is None or self.explained_variance_ratio_ is None:
            raise RuntimeError("WeightedPCA has not been fitted")
        return {
            "n_components": self.n_components,
            "numeric_rank": int(self.numeric_rank_),
            "entropy_effective_rank": float(self.spectrum_.entropy_effective_rank),
            "components_sha256": array_sha256(self.components_),
            "explained_variance_ratio": self.explained_variance_ratio_.tolist(),
            "explained_variance_sum": float(self.explained_variance_ratio_.sum()),
        }


def hamilton_allocate(
    scores: Mapping[str, float],
    total: int = DEFAULT_TOTAL_COMPONENTS,
    caps: Mapping[str, int] | None = None,
    minimum: int = 1,
) -> dict[str, int]:
    """Allocate an exact integer budget by capped largest remainders.

    Insertion order is the deterministic tie breaker.  Seats that hit a rank
    cap are redistributed by repeating Hamilton allocation among modalities
    with remaining capacity.
    """

    names = tuple(scores)
    if not names:
        raise ValueError("No allocation scores supplied")
    if minimum < 0:
        raise ValueError("minimum must be nonnegative")
    capacities = {
        name: int(caps[name]) if caps is not None else int(total) for name in names
    }
    if any(capacities[name] < minimum for name in names):
        raise ValueError("A rank cap is below the required per-modality minimum")
    if total < minimum * len(names) or total > sum(capacities.values()):
        raise ValueError("Total component budget is infeasible under the supplied caps")
    allocation = {name: minimum for name in names}
    remaining = int(total - minimum * len(names))
    base_scores = {name: max(float(scores[name]), 0.0) for name in names}
    while remaining:
        active = [name for name in names if allocation[name] < capacities[name]]
        if not active:
            raise RuntimeError("Component redistribution exhausted all rank capacity")
        active_scores = np.asarray([base_scores[name] for name in active], dtype=np.float64)
        if active_scores.sum() <= 0:
            active_scores[:] = 1.0
        quotas = remaining * active_scores / active_scores.sum()
        floors = np.floor(quotas).astype(int)
        added = 0
        for index, name in enumerate(active):
            seats = min(int(floors[index]), capacities[name] - allocation[name])
            allocation[name] += seats
            added += seats
        remaining -= added
        if remaining == 0:
            break
        fractions = quotas - floors
        order = sorted(range(len(active)), key=lambda index: (-fractions[index], index))
        awarded = 0
        for index in order:
            name = active[index]
            if allocation[name] >= capacities[name]:
                continue
            allocation[name] += 1
            remaining -= 1
            awarded += 1
            if remaining == 0:
                break
        if added == 0 and awarded == 0:
            raise RuntimeError("Hamilton allocation made no progress")
    return allocation


@dataclass(frozen=True)
class RankAllocation:
    components: dict[str, int]
    effective_ranks: dict[str, float]
    numeric_ranks: dict[str, int]
    spectra: dict[str, Spectrum]
    collection_evidence: dict[str, dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "components": dict(self.components),
            "effective_ranks": dict(self.effective_ranks),
            "numeric_ranks": dict(self.numeric_ranks),
            "spectra": {name: value.as_dict() for name, value in self.spectra.items()},
            "collection_evidence": self.collection_evidence,
        }


def fit_window_rank_allocation(
    blocks: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    fit_indexes: np.ndarray | Iterable[int],
    modalities: Iterable[str] = FIVE_MODALITIES,
    total_components: int = DEFAULT_TOTAL_COMPONENTS,
) -> RankAllocation:
    """Compute train-only entropy ranks and the preregistered PCA12 allocation."""

    names = _modalities(modalities)
    length = validate_window_inputs(blocks, masks, names)
    indexes = _indexes(fit_indexes, length)
    spectra: dict[str, Spectrum] = {}
    evidence: dict[str, dict[str, Any]] = {}
    for modality in names:
        collection = collect_condition_balanced_windows(
            blocks[modality], masks[modality], indexes
        )
        scaler = WeightedStandardizer().fit(collection.values, collection.weights)
        spectrum = weighted_spectrum(
            scaler.transform(collection.values), collection.weights
        )
        if spectrum.numeric_rank < 1:
            raise ValueError(f"{modality} has zero numerical rank in outer training")
        spectra[modality] = spectrum
        evidence[modality] = collection.evidence()
    effective = {name: spectra[name].entropy_effective_rank for name in names}
    numeric = {name: spectra[name].numeric_rank for name in names}
    components = hamilton_allocate(
        effective, total=total_components, caps=numeric, minimum=1
    )
    return RankAllocation(components, effective, numeric, spectra, evidence)


def _pool_condition_scores(
    values: np.ndarray,
    mask: np.ndarray,
    indexes: np.ndarray,
    scaler: WeightedStandardizer,
    pca: WeightedPCA,
) -> np.ndarray:
    array, valid = _validate_window_block(values, mask)
    output = np.zeros((len(indexes), pca.n_components), dtype=np.float64)
    for output_index, observation in enumerate(indexes):
        selected = valid[observation]
        if selected.any():
            transformed = pca.transform(scaler.transform(array[observation, selected]))
            output[output_index] = transformed.mean(axis=0)
    return output


class WindowPCAReducer:
    """One train-only, condition-balanced window PCA for a modality expert."""

    def __init__(self, modality: str, n_components: int, clip_to_rank: bool = False) -> None:
        self.modality = _modalities((modality,))[0]
        self.n_components = int(n_components)
        self.scaler = WeightedStandardizer()
        self.pca = WeightedPCA(self.n_components, clip_to_rank=clip_to_rank)
        self.fit_indexes_: np.ndarray | None = None
        self.fit_participants_: list[str] = []
        self.collection_evidence_: dict[str, Any] = {}

    def fit(
        self,
        values: np.ndarray,
        mask: np.ndarray,
        fit_indexes: np.ndarray | Iterable[int],
        participant_ids: Sequence[str] | np.ndarray | None = None,
    ) -> "WindowPCAReducer":
        array, _ = _validate_window_block(values, mask)
        indexes = _indexes(fit_indexes, len(array))
        collection = collect_condition_balanced_windows(array, mask, indexes)
        self.scaler.fit(collection.values, collection.weights)
        standardized = self.scaler.transform(collection.values)
        self.pca.fit(standardized, collection.weights)
        self.n_components = self.pca.n_components
        self.fit_indexes_ = indexes.copy()
        self.fit_participants_ = _participant_evidence(indexes, participant_ids, len(array))
        self.collection_evidence_ = collection.evidence()
        return self

    def transform(
        self,
        values: np.ndarray,
        mask: np.ndarray,
        indexes: np.ndarray | Iterable[int] | None = None,
        n_components: int | None = None,
    ) -> np.ndarray:
        if self.fit_indexes_ is None:
            raise RuntimeError("WindowPCAReducer has not been fitted")
        array, _ = _validate_window_block(values, mask)
        rows = np.arange(len(array), dtype=int) if indexes is None else _indexes(indexes, len(array))
        result = _pool_condition_scores(array, mask, rows, self.scaler, self.pca)
        selected = self.n_components if n_components is None else int(n_components)
        if selected < 1 or selected > self.n_components:
            raise ValueError(
                f"Requested prefix {selected} is outside fitted PCA1..{self.n_components}"
            )
        return result[:, :selected]

    def transform_conditions(
        self,
        values: np.ndarray,
        mask: np.ndarray,
        indexes: np.ndarray | Iterable[int] | None = None,
        n_components: int | None = None,
    ) -> np.ndarray:
        """Alias emphasizing that window scores are pooled to condition rows."""

        return self.transform(values, mask, indexes, n_components=n_components)

    def provenance(self) -> dict[str, Any]:
        if self.fit_indexes_ is None:
            raise RuntimeError("WindowPCAReducer has not been fitted")
        return {
            "modality": self.modality,
            "fit_indexes": self.fit_indexes_.tolist(),
            "fit_participants": self.fit_participants_,
            "collection": self.collection_evidence_,
            "coordinate_scaler": self.scaler.provenance(),
            "pca": self.pca.provenance(),
            "unsupervised_loading_parameters": int(
                self.n_components * self.scaler.provenance()["coordinate_count"]
            ),
        }


class ModalityRankAllocationPCA:
    """Independent modality PCA with a fold-specific entropy-rank budget."""

    method = "modality_rank_alloc_pca12"

    def __init__(
        self,
        modalities: Iterable[str] = FIVE_MODALITIES,
        total_components: int = DEFAULT_TOTAL_COMPONENTS,
    ) -> None:
        self.modalities = _modalities(modalities)
        self.total_components = int(total_components)
        self.allocation_: RankAllocation | None = None
        self.reducers_: dict[str, WindowPCAReducer] = {}
        self.fit_indexes_: np.ndarray | None = None
        self.fit_participants_: list[str] = []

    def fit(
        self,
        blocks: Mapping[str, np.ndarray],
        masks: Mapping[str, np.ndarray],
        fit_indexes: np.ndarray | Iterable[int],
        participant_ids: Sequence[str] | np.ndarray | None = None,
    ) -> "ModalityRankAllocationPCA":
        length = validate_window_inputs(blocks, masks, self.modalities)
        indexes = _indexes(fit_indexes, length)
        self.allocation_ = fit_window_rank_allocation(
            blocks,
            masks,
            indexes,
            modalities=self.modalities,
            total_components=self.total_components,
        )
        self.reducers_ = {
            modality: WindowPCAReducer(
                modality, self.allocation_.components[modality]
            ).fit(
                blocks[modality],
                masks[modality],
                indexes,
                participant_ids=participant_ids,
            )
            for modality in self.modalities
        }
        self.fit_indexes_ = indexes.copy()
        self.fit_participants_ = _participant_evidence(indexes, participant_ids, length)
        return self

    def transform_by_modality(
        self,
        blocks: Mapping[str, np.ndarray],
        masks: Mapping[str, np.ndarray],
        indexes: np.ndarray | Iterable[int] | None = None,
    ) -> dict[str, np.ndarray]:
        if self.fit_indexes_ is None:
            raise RuntimeError("ModalityRankAllocationPCA has not been fitted")
        length = validate_window_inputs(blocks, masks, self.modalities)
        rows = np.arange(length, dtype=int) if indexes is None else _indexes(indexes, length)
        return {
            modality: self.reducers_[modality].transform(
                blocks[modality], masks[modality], rows
            )
            for modality in self.modalities
        }

    def transform_condition_blocks(
        self,
        blocks: Mapping[str, np.ndarray],
        masks: Mapping[str, np.ndarray],
        indexes: np.ndarray | Iterable[int] | None = None,
    ) -> dict[str, np.ndarray]:
        return self.transform_by_modality(blocks, masks, indexes)

    def transform(
        self,
        blocks: Mapping[str, np.ndarray],
        masks: Mapping[str, np.ndarray],
        indexes: np.ndarray | Iterable[int] | None = None,
    ) -> np.ndarray:
        scores = self.transform_by_modality(blocks, masks, indexes)
        return np.concatenate([scores[name] for name in self.modalities], axis=1)

    def provenance(self) -> dict[str, Any]:
        if self.fit_indexes_ is None or self.allocation_ is None:
            raise RuntimeError("ModalityRankAllocationPCA has not been fitted")
        return {
            "method": self.method,
            "modalities": list(self.modalities),
            "total_components": self.total_components,
            "fit_indexes": self.fit_indexes_.tolist(),
            "fit_participants": self.fit_participants_,
            "rank_allocation": self.allocation_.as_dict(),
            "reducers": {
                modality: self.reducers_[modality].provenance()
                for modality in self.modalities
            },
            "unsupervised_loading_parameters": int(
                sum(
                    self.reducers_[modality].n_components
                    * self.reducers_[modality].scaler.provenance()["coordinate_count"]
                    for modality in self.modalities
                )
            ),
        }


class JointBlockBalancedWindowPCA:
    """Joint PCA after coordinate scaling and inverse-sqrt block balancing."""

    method = "joint_block_balanced_pca12"

    def __init__(
        self,
        modalities: Iterable[str] = FIVE_MODALITIES,
        n_components: int = DEFAULT_TOTAL_COMPONENTS,
    ) -> None:
        self.modalities = _modalities(modalities)
        self.n_components = int(n_components)
        self.scalers_: dict[str, WeightedStandardizer] = {}
        self.pca_ = WeightedPCA(self.n_components)
        self.fit_indexes_: np.ndarray | None = None
        self.fit_participants_: list[str] = []
        self.collection_evidence_: dict[str, Any] = {}
        self.input_dimensions_: dict[str, int] = {}

    def fit(
        self,
        blocks: Mapping[str, np.ndarray],
        masks: Mapping[str, np.ndarray],
        fit_indexes: np.ndarray | Iterable[int],
        participant_ids: Sequence[str] | np.ndarray | None = None,
    ) -> "JointBlockBalancedWindowPCA":
        length = validate_window_inputs(blocks, masks, self.modalities)
        indexes = _indexes(fit_indexes, length)
        shared_mask = common_window_mask(masks, self.modalities)
        first = collect_condition_balanced_windows(
            blocks[self.modalities[0]], shared_mask, indexes
        )
        pieces: list[np.ndarray] = []
        self.scalers_ = {}
        self.input_dimensions_ = {}
        for modality in self.modalities:
            collection = collect_condition_balanced_windows(
                blocks[modality], shared_mask, indexes
            )
            if not np.array_equal(collection.observation_indexes, first.observation_indexes):
                raise RuntimeError("Aligned modality collections changed observation order")
            if not np.array_equal(collection.window_indexes, first.window_indexes):
                raise RuntimeError("Aligned modality collections changed window order")
            scaler = WeightedStandardizer().fit(collection.values, collection.weights)
            self.scalers_[modality] = scaler
            self.input_dimensions_[modality] = int(collection.values.shape[1])
            pieces.append(
                scaler.transform(collection.values) / np.sqrt(collection.values.shape[1])
            )
        balanced = np.concatenate(pieces, axis=1)
        self.pca_.fit(balanced, first.weights)
        self.fit_indexes_ = indexes.copy()
        self.fit_participants_ = _participant_evidence(indexes, participant_ids, length)
        self.collection_evidence_ = first.evidence()
        return self

    def transform(
        self,
        blocks: Mapping[str, np.ndarray],
        masks: Mapping[str, np.ndarray],
        indexes: np.ndarray | Iterable[int] | None = None,
    ) -> np.ndarray:
        if self.fit_indexes_ is None:
            raise RuntimeError("JointBlockBalancedWindowPCA has not been fitted")
        length = validate_window_inputs(blocks, masks, self.modalities)
        rows = np.arange(length, dtype=int) if indexes is None else _indexes(indexes, length)
        shared_mask = common_window_mask(masks, self.modalities)
        output = np.zeros((len(rows), self.n_components), dtype=np.float64)
        for output_index, observation in enumerate(rows):
            selected = shared_mask[observation]
            if not selected.any():
                continue
            pieces = []
            for modality in self.modalities:
                array = np.asarray(blocks[modality], dtype=np.float64)
                pieces.append(
                    self.scalers_[modality].transform(array[observation, selected])
                    / np.sqrt(array.shape[2])
                )
            output[output_index] = self.pca_.transform(
                np.concatenate(pieces, axis=1)
            ).mean(axis=0)
        return output

    def provenance(self) -> dict[str, Any]:
        if self.fit_indexes_ is None:
            raise RuntimeError("JointBlockBalancedWindowPCA has not been fitted")
        return {
            "method": self.method,
            "modalities": list(self.modalities),
            "fit_indexes": self.fit_indexes_.tolist(),
            "fit_participants": self.fit_participants_,
            "input_dimensions": self.input_dimensions_,
            "block_multiplier": {
                name: float(1.0 / np.sqrt(self.input_dimensions_[name]))
                for name in self.modalities
            },
            "collection": self.collection_evidence_,
            "coordinate_scalers": {
                name: self.scalers_[name].provenance() for name in self.modalities
            },
            "pca": self.pca_.provenance(),
            "unsupervised_loading_parameters": int(
                sum(self.input_dimensions_.values()) * self.n_components
            ),
        }

    def transform_conditions(
        self,
        blocks: Mapping[str, np.ndarray],
        masks: Mapping[str, np.ndarray],
        indexes: np.ndarray | Iterable[int] | None = None,
    ) -> np.ndarray:
        return self.transform(blocks, masks, indexes)


class ModalityWindowPCABank:
    """Reusable train-only PCA prefixes for full and leave-one-out experts.

    Every modality is fitted once to ``max_components_per_modality`` (or its
    train-only numeric rank).  A variant subsequently selects deterministic PCA
    prefixes via :meth:`allocation_for`; no transform is refitted when an
    ablation changes the active modality subset.
    """

    method = "modality_window_pca_prefix_bank"

    def __init__(
        self,
        modalities: Iterable[str] = FIVE_MODALITIES,
        max_components_per_modality: int = DEFAULT_TOTAL_COMPONENTS,
    ) -> None:
        self.modalities = _modalities(modalities)
        self.max_components_per_modality = int(max_components_per_modality)
        if self.max_components_per_modality < 1:
            raise ValueError("max_components_per_modality must be positive")
        self.rank_allocation_: RankAllocation | None = None
        self.reducers_: dict[str, WindowPCAReducer] = {}
        self.fit_indexes_: np.ndarray | None = None
        self.fit_participants_: list[str] = []
        self.allocation_history_: dict[str, dict[str, int]] = {}

    def fit(
        self,
        blocks: Mapping[str, np.ndarray],
        masks: Mapping[str, np.ndarray],
        fit_indexes: np.ndarray | Iterable[int],
        participant_ids: Sequence[str] | np.ndarray | None = None,
    ) -> "ModalityWindowPCABank":
        length = validate_window_inputs(blocks, masks, self.modalities)
        indexes = _indexes(fit_indexes, length)
        self.reducers_ = {}
        spectra: dict[str, Spectrum] = {}
        collection_evidence: dict[str, dict[str, Any]] = {}
        for modality in self.modalities:
            reducer = WindowPCAReducer(
                modality,
                self.max_components_per_modality,
                clip_to_rank=True,
            ).fit(
                blocks[modality],
                masks[modality],
                indexes,
                participant_ids=participant_ids,
            )
            assert reducer.pca.spectrum_ is not None
            self.reducers_[modality] = reducer
            spectra[modality] = reducer.pca.spectrum_
            collection_evidence[modality] = dict(reducer.collection_evidence_)
        effective = {
            name: spectra[name].entropy_effective_rank for name in self.modalities
        }
        numeric = {name: spectra[name].numeric_rank for name in self.modalities}
        full_components = hamilton_allocate(
            effective,
            total=DEFAULT_TOTAL_COMPONENTS,
            caps={name: self.reducers_[name].n_components for name in self.modalities},
            minimum=1,
        )
        self.rank_allocation_ = RankAllocation(
            components=full_components,
            effective_ranks=effective,
            numeric_ranks=numeric,
            spectra=spectra,
            collection_evidence=collection_evidence,
        )
        self.fit_indexes_ = indexes.copy()
        self.fit_participants_ = _participant_evidence(indexes, participant_ids, length)
        self.allocation_history_ = {}
        self.allocation_for(
            self.modalities,
            total_components=DEFAULT_TOTAL_COMPONENTS,
            variant_name="full_five",
        )
        return self

    def allocation_for(
        self,
        modalities: Iterable[str],
        total_components: int = DEFAULT_TOTAL_COMPONENTS,
        variant_name: str | None = None,
    ) -> dict[str, int]:
        if self.fit_indexes_ is None or self.rank_allocation_ is None:
            raise RuntimeError("ModalityWindowPCABank has not been fitted")
        names = _modalities(modalities)
        if not set(names).issubset(self.modalities):
            raise ValueError("Variant contains a modality not fitted by this bank")
        scores = {
            name: self.rank_allocation_.effective_ranks[name] for name in names
        }
        caps = {name: self.reducers_[name].n_components for name in names}
        allocation = hamilton_allocate(
            scores, total=int(total_components), caps=caps, minimum=1
        )
        key = variant_name or "+".join(names)
        existing = self.allocation_history_.get(key)
        if existing is not None and existing != allocation:
            raise ValueError(f"Variant name {key!r} was reused with a different allocation")
        self.allocation_history_[key] = dict(allocation)
        return allocation

    def transform_condition_blocks(
        self,
        blocks: Mapping[str, np.ndarray],
        masks: Mapping[str, np.ndarray],
        indexes: np.ndarray | Iterable[int] | None = None,
        allocation: Mapping[str, int] | None = None,
        modalities: Iterable[str] | None = None,
    ) -> dict[str, np.ndarray]:
        if self.fit_indexes_ is None or self.rank_allocation_ is None:
            raise RuntimeError("ModalityWindowPCABank has not been fitted")
        names = self.modalities if modalities is None else _modalities(modalities)
        if allocation is None:
            selected = {
                name: self.rank_allocation_.components[name] for name in names
            }
        else:
            if set(allocation) != set(names):
                raise ValueError("Allocation keys must exactly match active modalities")
            selected = {name: int(allocation[name]) for name in names}
        return {
            name: self.reducers_[name].transform(
                blocks[name], masks[name], indexes, n_components=selected[name]
            )
            for name in names
        }

    def provenance(self) -> dict[str, Any]:
        if self.fit_indexes_ is None or self.rank_allocation_ is None:
            raise RuntimeError("ModalityWindowPCABank has not been fitted")
        return {
            "method": self.method,
            "modalities": list(self.modalities),
            "fit_indexes": self.fit_indexes_.tolist(),
            "fit_participants": self.fit_participants_,
            "max_components_per_modality": self.max_components_per_modality,
            "rank_allocation": self.rank_allocation_.as_dict(),
            "fitted_components": {
                name: self.reducers_[name].n_components for name in self.modalities
            },
            "selected_variant_allocations": {
                name: dict(value) for name, value in self.allocation_history_.items()
            },
            "reducers": {
                name: self.reducers_[name].provenance() for name in self.modalities
            },
        }


class TargetwiseModalityPLS1:
    """One supervised PLS1 direction per target and modality.

    This is the first PLS1 weight, proportional to ``X.T @ centered_y`` after
    train-only standardization.  The caller is responsible for providing
    cross-fitted Condition residuals as ``targets``.
    """

    method = "targetwise_modality_pls1"

    def __init__(
        self,
        modalities: Iterable[str] = FIVE_MODALITIES,
        target_names: Sequence[str] = ("relaxation", "discomfort"),
    ) -> None:
        self.modalities = _modalities(modalities)
        self.target_names = tuple(str(value) for value in target_names)
        if len(self.target_names) != 2:
            raise ValueError("The formal pipeline requires exactly two targets")
        self.scalers_: dict[str, WeightedStandardizer] = {}
        self.weights_: dict[str, dict[str, np.ndarray]] = {}
        self.score_means_: dict[str, dict[str, float]] = {}
        self.score_scales_: dict[str, dict[str, float]] = {}
        self.fit_indexes_: np.ndarray | None = None
        self.fit_participants_: list[str] = []
        self.target_description_: str = ""

    def fit(
        self,
        scores: Mapping[str, np.ndarray],
        targets: np.ndarray,
        fit_indexes: np.ndarray | Iterable[int],
        availability: Mapping[str, np.ndarray] | None = None,
        participant_ids: Sequence[str] | np.ndarray | None = None,
        target_description: str = "cross_fitted_condition_residual",
    ) -> "TargetwiseModalityPLS1":
        lengths = {len(np.asarray(scores[name])) for name in self.modalities}
        if len(lengths) != 1:
            raise ValueError("Condition score blocks have different row counts")
        length = lengths.pop()
        indexes = _indexes(fit_indexes, length)
        target_array = np.asarray(targets, dtype=np.float64)
        if target_array.shape != (length, len(self.target_names)):
            raise ValueError("targets must have shape [observation, 2]")
        if not np.isfinite(target_array[indexes]).all():
            raise ValueError("Train targets contain non-finite values")
        present = {
            name: np.ones(length, dtype=bool)
            if availability is None
            else np.asarray(availability[name], dtype=bool)
            for name in self.modalities
        }
        self.scalers_ = {}
        standardized: dict[str, np.ndarray] = {}
        for modality in self.modalities:
            values = np.asarray(scores[modality], dtype=np.float64)
            if values.ndim != 2 or len(values) != length:
                raise ValueError(f"Invalid condition score block for {modality}")
            selected = indexes[present[modality][indexes]]
            if len(selected) < 2:
                raise ValueError(f"Too few available training rows for {modality}")
            scaler = WeightedStandardizer().fit(values[selected])
            self.scalers_[modality] = scaler
            standardized[modality] = scaler.transform(values)
        self.weights_ = {target: {} for target in self.target_names}
        self.score_means_ = {target: {} for target in self.target_names}
        self.score_scales_ = {target: {} for target in self.target_names}
        for target_index, target in enumerate(self.target_names):
            for modality in self.modalities:
                selected = indexes[present[modality][indexes]]
                x_values = standardized[modality][selected]
                y_values = target_array[selected, target_index]
                direction = x_values.T @ (y_values - y_values.mean())
                norm = float(np.linalg.norm(direction))
                if norm <= 1e-14:
                    direction = np.zeros(x_values.shape[1], dtype=np.float64)
                else:
                    direction = direction / norm
                    pivot = int(np.argmax(np.abs(direction)))
                    if direction[pivot] < 0:
                        direction *= -1.0
                projected = x_values @ direction
                mean = float(projected.mean())
                scale = float(projected.std())
                if scale < 1e-12:
                    scale = 1.0
                self.weights_[target][modality] = direction
                self.score_means_[target][modality] = mean
                self.score_scales_[target][modality] = scale
        self.fit_indexes_ = indexes.copy()
        self.fit_participants_ = _participant_evidence(indexes, participant_ids, length)
        self.target_description_ = str(target_description)
        return self

    def transform(
        self,
        scores: Mapping[str, np.ndarray],
        indexes: np.ndarray | Iterable[int] | None = None,
        availability: Mapping[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        if self.fit_indexes_ is None:
            raise RuntimeError("TargetwiseModalityPLS1 has not been fitted")
        length = len(np.asarray(scores[self.modalities[0]]))
        rows = np.arange(length, dtype=int) if indexes is None else _indexes(indexes, length)
        outputs = {
            target: np.zeros((len(rows), len(self.modalities)), dtype=np.float64)
            for target in self.target_names
        }
        for modality_index, modality in enumerate(self.modalities):
            values = self.scalers_[modality].transform(np.asarray(scores[modality])[rows])
            present = (
                np.ones(len(rows), dtype=bool)
                if availability is None
                else np.asarray(availability[modality], dtype=bool)[rows]
            )
            for target in self.target_names:
                projected = values @ self.weights_[target][modality]
                projected = (
                    projected - self.score_means_[target][modality]
                ) / self.score_scales_[target][modality]
                projected[~present] = 0.0
                outputs[target][:, modality_index] = projected
        return outputs

    def provenance(self) -> dict[str, Any]:
        if self.fit_indexes_ is None:
            raise RuntimeError("TargetwiseModalityPLS1 has not been fitted")
        return {
            "method": self.method,
            "modalities": list(self.modalities),
            "target_names": list(self.target_names),
            "target_description": self.target_description_,
            "fit_indexes": self.fit_indexes_.tolist(),
            "fit_participants": self.fit_participants_,
            "weights_sha256": {
                target: {
                    modality: array_sha256(self.weights_[target][modality])
                    for modality in self.modalities
                }
                for target in self.target_names
            },
            "label_conditioned_projection_parameters": int(
                len(self.target_names)
                * sum(len(self.weights_[self.target_names[0]][name]) for name in self.modalities)
            ),
        }

    def transform_targets(
        self,
        scores: Mapping[str, np.ndarray],
        indexes: np.ndarray | Iterable[int] | None = None,
        availability: Mapping[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        return self.transform(scores, indexes, availability)


class LinearGCCASharedPrivate:
    """Regularized MAXVAR-style linear GCCA plus one private PC per modality."""

    method = "linear_gcca_shared_private"

    def __init__(
        self,
        modalities: Iterable[str] = FIVE_MODALITIES,
        shared_components: int = 2,
    ) -> None:
        self.modalities = _modalities(modalities)
        self.shared_components = int(shared_components)
        if self.shared_components < 1:
            raise ValueError("shared_components must be positive")
        self.scalers_: dict[str, WeightedStandardizer] = {}
        self.shared_loadings_: dict[str, np.ndarray] = {}
        self.reconstruction_: dict[str, np.ndarray] = {}
        self.private_pcas_: dict[str, WeightedPCA] = {}
        self.output_scaler_ = WeightedStandardizer()
        self.fit_indexes_: np.ndarray | None = None
        self.fit_participants_: list[str] = []
        self.shared_fit_indexes_: np.ndarray | None = None
        self.ledoit_wolf_shrinkage_: dict[str, float] = {}

    @staticmethod
    def _availability(
        availability: Mapping[str, np.ndarray] | None,
        modalities: tuple[str, ...],
        length: int,
    ) -> dict[str, np.ndarray]:
        result = {}
        for modality in modalities:
            values = (
                np.ones(length, dtype=bool)
                if availability is None
                else np.asarray(availability[modality], dtype=bool)
            )
            if values.shape != (length,):
                raise ValueError(f"Invalid availability for {modality}")
            result[modality] = values
        return result

    def fit(
        self,
        scores: Mapping[str, np.ndarray],
        fit_indexes: np.ndarray | Iterable[int],
        availability: Mapping[str, np.ndarray] | None = None,
        participant_ids: Sequence[str] | np.ndarray | None = None,
    ) -> "LinearGCCASharedPrivate":
        lengths = {len(np.asarray(scores[name])) for name in self.modalities}
        if len(lengths) != 1:
            raise ValueError("Condition score blocks have different row counts")
        length = lengths.pop()
        indexes = _indexes(fit_indexes, length)
        present = self._availability(availability, self.modalities, length)
        shared_present = np.logical_and.reduce([present[name] for name in self.modalities])
        shared_indexes = indexes[shared_present[indexes]]
        if len(shared_indexes) <= self.shared_components:
            raise ValueError("Too few jointly available observations for GCCA")
        self.scalers_ = {}
        standardized: dict[str, np.ndarray] = {}
        projection_sum = np.zeros((len(shared_indexes), len(shared_indexes)), dtype=np.float64)
        inverse_covariances: dict[str, np.ndarray] = {}
        self.ledoit_wolf_shrinkage_ = {}
        for modality in self.modalities:
            values = np.asarray(scores[modality], dtype=np.float64)
            if values.ndim != 2 or len(values) != length or not np.isfinite(values).all():
                raise ValueError(f"Invalid condition score block for {modality}")
            scaler = WeightedStandardizer().fit(values[indexes[present[modality][indexes]]])
            self.scalers_[modality] = scaler
            standardized[modality] = scaler.transform(values)
            x_values = standardized[modality][shared_indexes]
            covariance_model = LedoitWolf(assume_centered=False).fit(x_values)
            inverse = np.linalg.pinv(covariance_model.covariance_, hermitian=True)
            inverse_covariances[modality] = inverse
            self.ledoit_wolf_shrinkage_[modality] = float(covariance_model.shrinkage_)
            projection = x_values @ inverse @ x_values.T / len(shared_indexes)
            projection_sum += (projection + projection.T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(projection_sum)
        order = np.argsort(eigenvalues)[::-1][: self.shared_components]
        consensus_target = _canonicalize_component_columns(eigenvectors[:, order])
        consensus_target *= np.sqrt(len(shared_indexes))
        self.shared_loadings_ = {}
        modality_shared: list[np.ndarray] = []
        for modality in self.modalities:
            x_values = standardized[modality][shared_indexes]
            loading = inverse_covariances[modality] @ (
                x_values.T @ consensus_target / len(shared_indexes)
            )
            self.shared_loadings_[modality] = loading
            modality_shared.append(x_values @ loading)
        consensus = np.mean(modality_shared, axis=0)
        self.reconstruction_ = {}
        self.private_pcas_ = {}
        for modality in self.modalities:
            x_values = standardized[modality][shared_indexes]
            reconstruction, _, _, _ = np.linalg.lstsq(consensus, x_values, rcond=None)
            self.reconstruction_[modality] = reconstruction
            residual = x_values - consensus @ reconstruction
            self.private_pcas_[modality] = WeightedPCA(1).fit(residual)
        self.fit_indexes_ = indexes.copy()
        self.shared_fit_indexes_ = shared_indexes.copy()
        self.fit_participants_ = _participant_evidence(indexes, participant_ids, length)
        raw_train = self._raw_transform(scores, indexes, present)
        train_shared_present = shared_present[indexes]
        self.output_scaler_.fit(raw_train[train_shared_present])
        return self

    def _raw_transform(
        self,
        scores: Mapping[str, np.ndarray],
        rows: np.ndarray,
        present: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        standardized = {
            modality: self.scalers_[modality].transform(np.asarray(scores[modality])[rows])
            for modality in self.modalities
        }
        availability_matrix = np.column_stack(
            [present[modality][rows] for modality in self.modalities]
        )
        shared_predictions = [
            standardized[modality] @ self.shared_loadings_[modality]
            for modality in self.modalities
        ]
        denominator = np.maximum(availability_matrix.sum(axis=1, keepdims=True), 1)
        shared = sum(
            shared_predictions[index] * availability_matrix[:, index : index + 1]
            for index in range(len(self.modalities))
        ) / denominator
        shared[availability_matrix.sum(axis=1) == 0] = 0.0
        private: list[np.ndarray] = []
        for index, modality in enumerate(self.modalities):
            residual = standardized[modality] - shared @ self.reconstruction_[modality]
            private_score = self.private_pcas_[modality].transform(residual)
            private_score[~availability_matrix[:, index]] = 0.0
            private.append(private_score)
        return np.column_stack([shared, *private])

    def transform(
        self,
        scores: Mapping[str, np.ndarray],
        indexes: np.ndarray | Iterable[int] | None = None,
        availability: Mapping[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        if self.fit_indexes_ is None:
            raise RuntimeError("LinearGCCASharedPrivate has not been fitted")
        length = len(np.asarray(scores[self.modalities[0]]))
        rows = np.arange(length, dtype=int) if indexes is None else _indexes(indexes, length)
        present = self._availability(availability, self.modalities, length)
        raw = self._raw_transform(scores, rows, present)
        output = self.output_scaler_.transform(raw)
        all_missing = ~np.logical_or.reduce([present[name][rows] for name in self.modalities])
        output[all_missing] = 0.0
        return output

    def provenance(self) -> dict[str, Any]:
        if self.fit_indexes_ is None or self.shared_fit_indexes_ is None:
            raise RuntimeError("LinearGCCASharedPrivate has not been fitted")
        return {
            "method": self.method,
            "modalities": list(self.modalities),
            "shared_components": self.shared_components,
            "private_components_per_modality": {
                modality: 1 for modality in self.modalities
            },
            "fit_indexes": self.fit_indexes_.tolist(),
            "shared_fit_indexes": self.shared_fit_indexes_.tolist(),
            "fit_participants": self.fit_participants_,
            "ledoit_wolf_shrinkage": self.ledoit_wolf_shrinkage_,
            "shared_loadings_sha256": {
                modality: array_sha256(self.shared_loadings_[modality])
                for modality in self.modalities
            },
            "private_components_sha256": {
                modality: array_sha256(self.private_pcas_[modality].components_)
                for modality in self.modalities
            },
            "reconstruction_sha256": {
                modality: array_sha256(self.reconstruction_[modality])
                for modality in self.modalities
            },
            "reduced_state_parameters": int(
                sum(
                    self.shared_loadings_[name].size
                    + self.reconstruction_[name].size
                    + self.private_pcas_[name].components_.size
                    for name in self.modalities
                )
            ),
            "output_dimension": self.shared_components + len(self.modalities),
        }

    def transform_conditions(
        self,
        scores: Mapping[str, np.ndarray],
        indexes: np.ndarray | Iterable[int] | None = None,
        availability: Mapping[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        return self.transform(scores, indexes, availability)


def fit_modality_window_reducers(
    blocks: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    fit_indexes: np.ndarray | Iterable[int],
    modalities: Iterable[str] = FIVE_MODALITIES,
    total_components: int = DEFAULT_TOTAL_COMPONENTS,
    participant_ids: Sequence[str] | np.ndarray | None = None,
) -> ModalityRankAllocationPCA:
    """Fit the allocated independent PCA bank used by modality experts."""

    return ModalityRankAllocationPCA(
        modalities=modalities, total_components=total_components
    ).fit(blocks, masks, fit_indexes, participant_ids=participant_ids)


def fit_modality_bank(
    blocks: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    fit_indexes: np.ndarray | Iterable[int],
    modalities: Iterable[str] = FIVE_MODALITIES,
    max_components_per_modality: int = DEFAULT_TOTAL_COMPONENTS,
    participant_ids: Sequence[str] | np.ndarray | None = None,
) -> ModalityWindowPCABank:
    """Compact constructor for the reusable full/ablation PCA-prefix bank."""

    return ModalityWindowPCABank(
        modalities=modalities,
        max_components_per_modality=max_components_per_modality,
    ).fit(blocks, masks, fit_indexes, participant_ids=participant_ids)


def fit_joint_bank(
    blocks: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    fit_indexes: np.ndarray | Iterable[int],
    modalities: Iterable[str] = FIVE_MODALITIES,
    n_components: int = DEFAULT_TOTAL_COMPONENTS,
    participant_ids: Sequence[str] | np.ndarray | None = None,
) -> JointBlockBalancedWindowPCA:
    """Compact constructor for joint block-balanced window PCA."""

    return JointBlockBalancedWindowPCA(
        modalities=modalities, n_components=n_components
    ).fit(blocks, masks, fit_indexes, participant_ids=participant_ids)


def fit_shared_private(
    scores: Mapping[str, np.ndarray],
    fit_indexes: np.ndarray | Iterable[int],
    modalities: Iterable[str] = FIVE_MODALITIES,
    availability: Mapping[str, np.ndarray] | None = None,
    participant_ids: Sequence[str] | np.ndarray | None = None,
    shared_components: int = 2,
) -> LinearGCCASharedPrivate:
    """Compact constructor for the condition-level shared/private reducer."""

    return LinearGCCASharedPrivate(
        modalities=modalities, shared_components=shared_components
    ).fit(
        scores,
        fit_indexes,
        availability=availability,
        participant_ids=participant_ids,
    )


def fit_targetwise_pls1(
    scores: Mapping[str, np.ndarray],
    targets: np.ndarray,
    fit_indexes: np.ndarray | Iterable[int],
    modalities: Iterable[str] = FIVE_MODALITIES,
    availability: Mapping[str, np.ndarray] | None = None,
    participant_ids: Sequence[str] | np.ndarray | None = None,
    target_names: Sequence[str] = ("relaxation", "discomfort"),
    target_description: str = "cross_fitted_condition_residual",
) -> TargetwiseModalityPLS1:
    """Compact constructor for train-only targetwise modality PLS1 scores."""

    return TargetwiseModalityPLS1(
        modalities=modalities, target_names=target_names
    ).fit(
        scores,
        targets,
        fit_indexes,
        availability=availability,
        participant_ids=participant_ids,
        target_description=target_description,
    )


__all__ = [
    "DEFAULT_TOTAL_COMPONENTS",
    "FIVE_MODALITIES",
    "JointBlockBalancedWindowPCA",
    "LinearGCCASharedPrivate",
    "ModalityRankAllocationPCA",
    "ModalityWindowPCABank",
    "RankAllocation",
    "Spectrum",
    "TargetwiseModalityPLS1",
    "WeightedPCA",
    "WeightedStandardizer",
    "WindowCollection",
    "WindowPCAReducer",
    "array_sha256",
    "collect_condition_balanced_windows",
    "common_window_mask",
    "fit_modality_window_reducers",
    "fit_modality_bank",
    "fit_joint_bank",
    "fit_shared_private",
    "fit_targetwise_pls1",
    "fit_window_rank_allocation",
    "hamilton_allocate",
    "validate_window_inputs",
    "weighted_spectrum",
]
