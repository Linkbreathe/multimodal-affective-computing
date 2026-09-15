"""Low-capacity compressors for frozen condition-level multimodal embeddings.

The classes in this module are deliberately sklearn-like, deterministic and
index-explicit.  ``fit`` receives the full arrays plus the exact rows it may use;
``transform`` never refits state.  This makes outer-participant leakage tests
straightforward and keeps the formal runner independent of neural code.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Iterable

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


FORMAL_METHODS = (
    "joint_block_balanced_pca8",
    "modality_pca2_additive",
    "supervised_pls2",
    "linear_shared_private",
)


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _indexes(indexes: np.ndarray | Iterable[int], length: int) -> np.ndarray:
    result = np.asarray(list(indexes) if not isinstance(indexes, np.ndarray) else indexes, dtype=int)
    if result.ndim != 1 or len(result) == 0:
        raise ValueError("fit/transform indexes must be a nonempty one-dimensional array")
    if result.min() < 0 or result.max() >= length or len(np.unique(result)) != len(result):
        raise ValueError("fit/transform indexes are invalid or duplicated")
    return result


def _validate_blocks(
    blocks: dict[str, np.ndarray], counts: dict[str, np.ndarray], modalities: tuple[str, ...]
) -> int:
    if not modalities:
        raise ValueError("At least one modality is required")
    lengths = set()
    for modality in modalities:
        if modality not in blocks or modality not in counts:
            raise ValueError(f"Missing block/count for {modality}")
        values = np.asarray(blocks[modality], dtype=np.float64)
        count = np.asarray(counts[modality])
        if values.ndim != 2 or count.ndim != 1 or len(values) != len(count):
            raise ValueError(f"Invalid block/count shape for {modality}")
        if not np.isfinite(values).all() or (count < 0).any():
            raise ValueError(f"Non-finite values or negative counts for {modality}")
        lengths.add(len(values))
    if len(lengths) != 1:
        raise ValueError("Modality blocks have different row counts")
    return lengths.pop()


def _shared_presence(counts: dict[str, np.ndarray], modalities: tuple[str, ...], indexes: np.ndarray) -> np.ndarray:
    present = np.logical_and.reduce([np.asarray(counts[m])[indexes] > 0 for m in modalities])
    return present.astype(np.float64)[:, None]


class _BaseCompressor:
    method: str

    def __init__(self, modalities: Iterable[str]) -> None:
        self.modalities = tuple(str(value) for value in modalities)
        self.fit_indexes_: np.ndarray | None = None
        self.input_dimensions_: dict[str, int] = {}
        self.output_dimension_: int | None = None

    def _start_fit(
        self, blocks: dict[str, np.ndarray], counts: dict[str, np.ndarray], fit_indexes: np.ndarray | Iterable[int]
    ) -> np.ndarray:
        length = _validate_blocks(blocks, counts, self.modalities)
        indexes = _indexes(fit_indexes, length)
        self.fit_indexes_ = indexes.copy()
        self.input_dimensions_ = {m: int(np.asarray(blocks[m]).shape[1]) for m in self.modalities}
        return indexes

    def _transform_indexes(
        self,
        blocks: dict[str, np.ndarray],
        counts: dict[str, np.ndarray],
        indexes: np.ndarray | Iterable[int] | None,
    ) -> np.ndarray:
        if self.fit_indexes_ is None:
            raise RuntimeError("Compressor has not been fitted")
        length = _validate_blocks(blocks, counts, self.modalities)
        return np.arange(length, dtype=int) if indexes is None else _indexes(indexes, length)

    def provenance(self) -> dict[str, Any]:
        if self.fit_indexes_ is None or self.output_dimension_ is None:
            raise RuntimeError("Compressor has not been fitted")
        return {
            "method": self.method,
            "modalities": list(self.modalities),
            "fit_indexes": self.fit_indexes_.tolist(),
            "fit_observation_count": int(len(self.fit_indexes_)),
            "input_dimensions": dict(self.input_dimensions_),
            "output_dimension": int(self.output_dimension_),
        }


class JointBlockBalancedPCA(_BaseCompressor):
    method = "joint_block_balanced_pca8"

    def __init__(self, modalities: Iterable[str], n_components: int = 8) -> None:
        super().__init__(modalities)
        self.n_components = int(n_components)
        self.block_scalers: dict[str, StandardScaler] = {}
        self.pca: PCA | None = None
        self.latent_scaler: StandardScaler | None = None

    def _balanced(self, blocks: dict[str, np.ndarray], indexes: np.ndarray) -> np.ndarray:
        pieces = []
        for modality in self.modalities:
            scaled = self.block_scalers[modality].transform(np.asarray(blocks[modality])[indexes])
            pieces.append(scaled / np.sqrt(scaled.shape[1]))
        return np.concatenate(pieces, axis=1)

    def fit(
        self,
        blocks: dict[str, np.ndarray],
        counts: dict[str, np.ndarray],
        fit_indexes: np.ndarray | Iterable[int],
        targets: np.ndarray | None = None,
    ) -> "JointBlockBalancedPCA":
        del targets
        indexes = self._start_fit(blocks, counts, fit_indexes)
        self.block_scalers = {
            m: StandardScaler().fit(np.asarray(blocks[m], dtype=np.float64)[indexes]) for m in self.modalities
        }
        balanced = self._balanced(blocks, indexes)
        components = min(self.n_components, len(indexes) - 1, balanced.shape[1])
        if components != self.n_components:
            raise ValueError("Training matrix cannot support the preregistered joint PCA dimension")
        self.pca = PCA(n_components=components, svd_solver="full").fit(balanced)
        latent = np.column_stack([self.pca.transform(balanced), _shared_presence(counts, self.modalities, indexes)])
        self.latent_scaler = StandardScaler().fit(latent)
        self.output_dimension_ = int(latent.shape[1])
        return self

    def transform(
        self,
        blocks: dict[str, np.ndarray],
        counts: dict[str, np.ndarray],
        indexes: np.ndarray | Iterable[int] | None = None,
    ) -> np.ndarray:
        rows = self._transform_indexes(blocks, counts, indexes)
        assert self.pca is not None and self.latent_scaler is not None
        latent = np.column_stack(
            [self.pca.transform(self._balanced(blocks, rows)), _shared_presence(counts, self.modalities, rows)]
        )
        return self.latent_scaler.transform(latent)

    def provenance(self) -> dict[str, Any]:
        result = super().provenance()
        assert self.pca is not None and self.latent_scaler is not None
        result.update(
            {
                "pca_components": self.n_components,
                "pca_explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
                "pca_explained_variance_sum": float(self.pca.explained_variance_ratio_.sum()),
                "pca_components_sha256": array_sha256(self.pca.components_),
                "latent_scaler_mean_sha256": array_sha256(self.latent_scaler.mean_),
                "unsupervised_state_parameters": int(self.pca.components_.size),
            }
        )
        return result


class ModalityPCACompressor(_BaseCompressor):
    method = "modality_pca2_additive"

    def __init__(self, modalities: Iterable[str], n_components: int = 2) -> None:
        super().__init__(modalities)
        self.n_components = int(n_components)
        self.block_scalers: dict[str, StandardScaler] = {}
        self.pcas: dict[str, PCA] = {}
        self.latent_scaler: StandardScaler | None = None

    def _scores(
        self, blocks: dict[str, np.ndarray], counts: dict[str, np.ndarray], indexes: np.ndarray
    ) -> np.ndarray:
        pieces = []
        for modality in self.modalities:
            scaled = self.block_scalers[modality].transform(np.asarray(blocks[modality])[indexes])
            score = self.pcas[modality].transform(scaled)
            score[np.asarray(counts[modality])[indexes] == 0] = 0.0
            pieces.append(score)
        return np.concatenate(pieces, axis=1)

    def fit(
        self,
        blocks: dict[str, np.ndarray],
        counts: dict[str, np.ndarray],
        fit_indexes: np.ndarray | Iterable[int],
        targets: np.ndarray | None = None,
    ) -> "ModalityPCACompressor":
        del targets
        indexes = self._start_fit(blocks, counts, fit_indexes)
        self.block_scalers = {}
        self.pcas = {}
        for modality in self.modalities:
            scaler = StandardScaler().fit(np.asarray(blocks[modality], dtype=np.float64)[indexes])
            scaled = scaler.transform(np.asarray(blocks[modality], dtype=np.float64)[indexes])
            components = min(self.n_components, len(indexes) - 1, scaled.shape[1])
            if components != self.n_components:
                raise ValueError(f"{modality} cannot support PCA{self.n_components}")
            self.block_scalers[modality] = scaler
            self.pcas[modality] = PCA(n_components=components, svd_solver="full").fit(scaled)
        scores = self._scores(blocks, counts, indexes)
        latent = np.column_stack([scores, _shared_presence(counts, self.modalities, indexes)])
        self.latent_scaler = StandardScaler().fit(latent)
        self.output_dimension_ = int(latent.shape[1])
        return self

    def transform(
        self,
        blocks: dict[str, np.ndarray],
        counts: dict[str, np.ndarray],
        indexes: np.ndarray | Iterable[int] | None = None,
    ) -> np.ndarray:
        rows = self._transform_indexes(blocks, counts, indexes)
        assert self.latent_scaler is not None
        latent = np.column_stack(
            [self._scores(blocks, counts, rows), _shared_presence(counts, self.modalities, rows)]
        )
        return self.latent_scaler.transform(latent)

    def provenance(self) -> dict[str, Any]:
        result = super().provenance()
        assert self.latent_scaler is not None
        result.update(
            {
                "components_per_modality": {m: self.n_components for m in self.modalities},
                "explained_variance_ratio": {
                    m: self.pcas[m].explained_variance_ratio_.tolist() for m in self.modalities
                },
                "pca_components_sha256": {
                    m: array_sha256(self.pcas[m].components_) for m in self.modalities
                },
                "latent_scaler_mean_sha256": array_sha256(self.latent_scaler.mean_),
                "unsupervised_state_parameters": int(sum(pca.components_.size for pca in self.pcas.values())),
            }
        )
        return result


class SupervisedPLSCompressor(_BaseCompressor):
    method = "supervised_pls2"

    def __init__(self, modalities: Iterable[str], block_components: int = 2, pls_components: int = 2) -> None:
        super().__init__(modalities)
        self.block_components = int(block_components)
        self.pls_components = int(pls_components)
        self.block_scalers: dict[str, StandardScaler] = {}
        self.pcas: dict[str, PCA] = {}
        self.block_score_scaler: StandardScaler | None = None
        self.pls: PLSRegression | None = None
        self.latent_scaler: StandardScaler | None = None

    def _block_scores(self, blocks: dict[str, np.ndarray], counts: dict[str, np.ndarray], indexes: np.ndarray) -> np.ndarray:
        pieces = []
        for modality in self.modalities:
            scaled = self.block_scalers[modality].transform(np.asarray(blocks[modality])[indexes])
            score = self.pcas[modality].transform(scaled)
            score[np.asarray(counts[modality])[indexes] == 0] = 0.0
            pieces.append(score)
        result = np.concatenate(pieces, axis=1)
        assert self.block_score_scaler is not None
        return self.block_score_scaler.transform(result)

    def fit(
        self,
        blocks: dict[str, np.ndarray],
        counts: dict[str, np.ndarray],
        fit_indexes: np.ndarray | Iterable[int],
        targets: np.ndarray | None = None,
    ) -> "SupervisedPLSCompressor":
        if targets is None:
            raise ValueError("supervised_pls2 requires the two train residual targets")
        indexes = self._start_fit(blocks, counts, fit_indexes)
        target_array = np.asarray(targets, dtype=np.float64)
        if target_array.ndim != 2 or target_array.shape[1] != 2 or len(target_array) != len(next(iter(blocks.values()))):
            raise ValueError("PLS targets must be a full [observation, 2] array")
        raw_scores = []
        for modality in self.modalities:
            scaler = StandardScaler().fit(np.asarray(blocks[modality], dtype=np.float64)[indexes])
            scaled = scaler.transform(np.asarray(blocks[modality], dtype=np.float64)[indexes])
            pca = PCA(n_components=self.block_components, svd_solver="full").fit(scaled)
            score = pca.transform(scaled)
            score[np.asarray(counts[modality])[indexes] == 0] = 0.0
            self.block_scalers[modality] = scaler
            self.pcas[modality] = pca
            raw_scores.append(score)
        concatenated = np.concatenate(raw_scores, axis=1)
        self.block_score_scaler = StandardScaler().fit(concatenated)
        standardized = self.block_score_scaler.transform(concatenated)
        self.pls = PLSRegression(n_components=self.pls_components, scale=False, max_iter=1000, tol=1e-8)
        self.pls.fit(standardized, target_array[indexes])
        pls_scores = self.pls.transform(standardized)
        latent = np.column_stack([pls_scores, _shared_presence(counts, self.modalities, indexes)])
        self.latent_scaler = StandardScaler().fit(latent)
        self.output_dimension_ = int(latent.shape[1])
        return self

    def transform(
        self,
        blocks: dict[str, np.ndarray],
        counts: dict[str, np.ndarray],
        indexes: np.ndarray | Iterable[int] | None = None,
    ) -> np.ndarray:
        rows = self._transform_indexes(blocks, counts, indexes)
        assert self.pls is not None and self.latent_scaler is not None
        scores = self.pls.transform(self._block_scores(blocks, counts, rows))
        latent = np.column_stack([scores, _shared_presence(counts, self.modalities, rows)])
        return self.latent_scaler.transform(latent)

    def provenance(self) -> dict[str, Any]:
        result = super().provenance()
        assert self.pls is not None and self.latent_scaler is not None
        result.update(
            {
                "pre_pca_components_per_modality": {m: self.block_components for m in self.modalities},
                "pls_components": self.pls_components,
                "pls_x_weights_sha256": array_sha256(self.pls.x_weights_),
                "pls_y_weights_sha256": array_sha256(self.pls.y_weights_),
                "label_conditioned_projection_parameters": int(
                    self.pls.x_weights_.size + self.pls.y_weights_.size
                ),
                "unsupervised_state_parameters": int(sum(pca.components_.size for pca in self.pcas.values())),
            }
        )
        return result


class LinearSharedPrivateCompressor(_BaseCompressor):
    method = "linear_shared_private"

    def __init__(self, modalities: Iterable[str], shared_components: int = 4) -> None:
        super().__init__(modalities)
        self.shared_components = int(shared_components)
        self.block_scalers: dict[str, StandardScaler] = {}
        self.pcas: dict[str, PCA] = {}
        self.score_scalers: dict[str, StandardScaler] = {}
        self.rotations: dict[str, np.ndarray] = {}
        self.private_pcas: dict[str, PCA] = {}
        self.latent_scaler: StandardScaler | None = None

    def _scores(self, blocks: dict[str, np.ndarray], counts: dict[str, np.ndarray], indexes: np.ndarray) -> dict[str, np.ndarray]:
        result = {}
        for modality in self.modalities:
            raw = self.block_scalers[modality].transform(np.asarray(blocks[modality])[indexes])
            score = self.score_scalers[modality].transform(self.pcas[modality].transform(raw))
            score[np.asarray(counts[modality])[indexes] == 0] = 0.0
            result[modality] = score
        return result

    def _representation(
        self, blocks: dict[str, np.ndarray], counts: dict[str, np.ndarray], indexes: np.ndarray
    ) -> np.ndarray:
        scores = self._scores(blocks, counts, indexes)
        aligned = {m: scores[m] @ self.rotations[m] for m in self.modalities}
        available = np.column_stack([np.asarray(counts[m])[indexes] > 0 for m in self.modalities])
        denominator = np.maximum(available.sum(axis=1, keepdims=True), 1)
        shared = sum(aligned[m] * available[:, i : i + 1] for i, m in enumerate(self.modalities)) / denominator
        shared[available.sum(axis=1) == 0] = 0.0
        private = []
        for i, modality in enumerate(self.modalities):
            residual = aligned[modality] - shared
            residual[~available[:, i]] = 0.0
            score = self.private_pcas[modality].transform(residual)
            score[~available[:, i]] = 0.0
            private.append(score)
        return np.column_stack([shared, *private, _shared_presence(counts, self.modalities, indexes)])

    def fit(
        self,
        blocks: dict[str, np.ndarray],
        counts: dict[str, np.ndarray],
        fit_indexes: np.ndarray | Iterable[int],
        targets: np.ndarray | None = None,
    ) -> "LinearSharedPrivateCompressor":
        del targets
        indexes = self._start_fit(blocks, counts, fit_indexes)
        train_scores = {}
        for modality in self.modalities:
            scaler = StandardScaler().fit(np.asarray(blocks[modality], dtype=np.float64)[indexes])
            scaled = scaler.transform(np.asarray(blocks[modality], dtype=np.float64)[indexes])
            pca = PCA(n_components=self.shared_components, svd_solver="full").fit(scaled)
            raw_score = pca.transform(scaled)
            raw_score[np.asarray(counts[modality])[indexes] == 0] = 0.0
            score_scaler = StandardScaler().fit(raw_score)
            score = score_scaler.transform(raw_score)
            score[np.asarray(counts[modality])[indexes] == 0] = 0.0
            self.block_scalers[modality] = scaler
            self.pcas[modality] = pca
            self.score_scalers[modality] = score_scaler
            train_scores[modality] = score

        consensus = train_scores[self.modalities[0]].copy()
        rotations = {m: np.eye(self.shared_components) for m in self.modalities}
        reference_norm = max(float(np.linalg.norm(consensus)), 1e-12)
        for _ in range(10):
            aligned = []
            for modality in self.modalities:
                rotation, _ = orthogonal_procrustes(train_scores[modality], consensus)
                rotations[modality] = rotation
                aligned.append(train_scores[modality] @ rotation)
            updated = np.mean(aligned, axis=0)
            norm = float(np.linalg.norm(updated))
            consensus = updated * (reference_norm / norm) if norm else updated
        self.rotations = rotations

        aligned = {m: train_scores[m] @ self.rotations[m] for m in self.modalities}
        available = np.column_stack([np.asarray(counts[m])[indexes] > 0 for m in self.modalities])
        denominator = np.maximum(available.sum(axis=1, keepdims=True), 1)
        shared = sum(aligned[m] * available[:, i : i + 1] for i, m in enumerate(self.modalities)) / denominator
        shared[available.sum(axis=1) == 0] = 0.0
        self.private_pcas = {}
        for i, modality in enumerate(self.modalities):
            residual = aligned[modality] - shared
            residual[~available[:, i]] = 0.0
            self.private_pcas[modality] = PCA(n_components=1, svd_solver="full").fit(residual)
        representation = self._representation(blocks, counts, indexes)
        self.latent_scaler = StandardScaler().fit(representation)
        self.output_dimension_ = int(representation.shape[1])
        return self

    def transform(
        self,
        blocks: dict[str, np.ndarray],
        counts: dict[str, np.ndarray],
        indexes: np.ndarray | Iterable[int] | None = None,
    ) -> np.ndarray:
        rows = self._transform_indexes(blocks, counts, indexes)
        assert self.latent_scaler is not None
        return self.latent_scaler.transform(self._representation(blocks, counts, rows))

    def provenance(self) -> dict[str, Any]:
        result = super().provenance()
        result.update(
            {
                "shared_components": self.shared_components,
                "private_components_per_modality": {m: 1 for m in self.modalities},
                "rotation_sha256": {m: array_sha256(self.rotations[m]) for m in self.modalities},
                "private_components_sha256": {
                    m: array_sha256(self.private_pcas[m].components_) for m in self.modalities
                },
                "unsupervised_state_parameters": int(
                    sum(pca.components_.size for pca in self.pcas.values())
                    + sum(rotation.size for rotation in self.rotations.values())
                    + sum(pca.components_.size for pca in self.private_pcas.values())
                ),
            }
        )
        return result


class SingleModalityPCA:
    """Two-dimensional train-only docking transform for one modality expert."""

    def __init__(self, modality: str, n_components: int = 2) -> None:
        self.modality = str(modality)
        self.n_components = int(n_components)
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None
        self.latent_scaler: StandardScaler | None = None
        self.fit_indexes_: np.ndarray | None = None

    def fit(
        self,
        blocks: dict[str, np.ndarray],
        counts: dict[str, np.ndarray],
        fit_indexes: np.ndarray | Iterable[int],
    ) -> "SingleModalityPCA":
        length = _validate_blocks(blocks, counts, (self.modality,))
        indexes = _indexes(fit_indexes, length)
        values = np.asarray(blocks[self.modality], dtype=np.float64)
        self.scaler = StandardScaler().fit(values[indexes])
        scaled = self.scaler.transform(values[indexes])
        self.pca = PCA(n_components=self.n_components, svd_solver="full").fit(scaled)
        score = self.pca.transform(scaled)
        score[np.asarray(counts[self.modality])[indexes] == 0] = 0.0
        self.latent_scaler = StandardScaler().fit(score)
        self.fit_indexes_ = indexes.copy()
        return self

    def transform(
        self,
        blocks: dict[str, np.ndarray],
        counts: dict[str, np.ndarray],
        indexes: np.ndarray | Iterable[int],
    ) -> np.ndarray:
        if self.fit_indexes_ is None or self.scaler is None or self.pca is None or self.latent_scaler is None:
            raise RuntimeError("SingleModalityPCA has not been fitted")
        length = _validate_blocks(blocks, counts, (self.modality,))
        rows = _indexes(indexes, length)
        score = self.pca.transform(self.scaler.transform(np.asarray(blocks[self.modality])[rows]))
        score[np.asarray(counts[self.modality])[rows] == 0] = 0.0
        return self.latent_scaler.transform(score)

    def provenance(self) -> dict[str, Any]:
        if self.fit_indexes_ is None or self.pca is None:
            raise RuntimeError("SingleModalityPCA has not been fitted")
        return {
            "modality": self.modality,
            "fit_indexes": self.fit_indexes_.tolist(),
            "fit_observation_count": int(len(self.fit_indexes_)),
            "input_dimension": int(self.pca.components_.shape[1]),
            "output_dimension": self.n_components,
            "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
            "pca_components_sha256": array_sha256(self.pca.components_),
            "unsupervised_state_parameters": int(self.pca.components_.size),
        }


def fit_modality_reducers(
    blocks: dict[str, np.ndarray],
    counts: dict[str, np.ndarray],
    fit_indexes: np.ndarray | Iterable[int],
    modalities: Iterable[str],
    n_components: int = 2,
) -> dict[str, SingleModalityPCA]:
    return {
        modality: SingleModalityPCA(modality, n_components=n_components).fit(blocks, counts, fit_indexes)
        for modality in tuple(modalities)
    }


def make_compressor(method: str, modalities: Iterable[str]) -> _BaseCompressor:
    mapping = {
        "joint_block_balanced_pca8": JointBlockBalancedPCA,
        "modality_pca2_additive": ModalityPCACompressor,
        "supervised_pls2": SupervisedPLSCompressor,
        "linear_shared_private": LinearSharedPrivateCompressor,
    }
    if method not in mapping:
        raise ValueError(f"No shared compressor for {method!r}; expert methods use fit_modality_reducers")
    return mapping[method](modalities)


__all__ = [
    "FORMAL_METHODS",
    "JointBlockBalancedPCA",
    "LinearSharedPrivateCompressor",
    "ModalityPCACompressor",
    "SingleModalityPCA",
    "SupervisedPLSCompressor",
    "array_sha256",
    "fit_modality_reducers",
    "make_compressor",
]
