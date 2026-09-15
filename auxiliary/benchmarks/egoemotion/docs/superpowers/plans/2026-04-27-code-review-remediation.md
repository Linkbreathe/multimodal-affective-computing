# Code Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the reported egoEMOTION and SEED-V results auditable by preventing silent data corruption, silent fold shrinkage, LOSO leakage, and unstable fusion behavior found in the code review.

**Architecture:** Add a checked egoEMOTION manifest loader that validates cached embedding metadata and returns missingness/report metadata. Thread that metadata through training reports, add LOSO split assertions, fix confirmed leakage/logic bugs, then do small targeted refactors to reduce pipeline confusion without redesigning the whole repo.

**Tech Stack:** Python, PyTorch, pandas, NumPy, scikit-learn, pytest, YAML configs.

---

## File Structure

- Create `src/data/egoemotion/__init__.py`: package marker for egoEMOTION-specific data utilities.
- Create `src/data/egoemotion/manifest.py`: manifest schema validation, soft-label parsing, embedding metadata checks, missingness reporting, and checked 10s loader.
- Modify `scripts/run_experiment_10s.py`: replace local loader with the checked loader; add missingness CLI threshold and report metadata.
- Modify `scripts/segment_and_extract_10s.py`: write a manifest hash/schema version into new embedding files and avoid reusing stale files when requested.
- Modify `src/trainer/fusion_trainer.py`: assert disjoint train/val/test subjects and attach fold metadata to results.
- Modify `src/utils/reporting.py`: include run metadata, fold counts, fold subject table, and missing modality summary.
- Modify `src/fusion/early.py`, `src/fusion/mid.py`, `src/fusion/late.py`, `src/fusion/distill_late.py`, `src/fusion/tmc.py`, `src/tasks/heads.py`, `src/tasks/losses.py`: fix BatchNorm batch-size failure, modality-order hazards, and TMC probability semantics.
- Modify `scripts/run_seedv_eeg_eye_fusion.py`: move PCA/scaling inside LOSO folds to remove test-subject leakage.
- Modify `scripts/pretrain_lego_blocks.py`, `scripts/run_lego_experiment.py`: mark globally supervised MM-Lego blocks unsafe unless explicitly allowed; prepare fold-local pretraining path.
- Modify `scripts/verify_pipeline.py`: replace stale 6663/28 constants with manifest-derived checks.
- Modify tests: `tests/test_run_experiment_10s.py`, `tests/test_fusion_modules.py`; create `tests/test_egoemotion_manifest.py`, `tests/test_seedv_eeg_eye_fusion.py`, and `tests/test_trainer_loso_splits.py`.

---

### Task 1: Checked egoEMOTION 10s Manifest Loader

**Files:**
- Create: `src/data/egoemotion/__init__.py`
- Create: `src/data/egoemotion/manifest.py`
- Modify: `scripts/run_experiment_10s.py`
- Test: `tests/test_egoemotion_manifest.py`
- Test: `tests/test_run_experiment_10s.py`

- [ ] **Step 1: Write failing loader tests**

Add tests for these exact behaviors:

```python
def test_loader_rejects_label_mismatch(tmp_path):
    # manifest emotion_label=4, embedding file label=3
    # expect ValueError containing "label mismatch"

def test_loader_reports_and_limits_missing_modalities(tmp_path):
    # manifest has two rows; video exists for one row only
    # max_missing_fraction=0.0 raises RuntimeError
    # max_missing_fraction=0.6 loads one sample and reports missing video

def test_loader_preserves_sequence_shapes_and_pooling(tmp_path):
    # video [6,768], eye [39,128], ppg [1,512]
    # with pool_clips=True: video -> [768], eye remains [39,128], ppg -> [512]
```

Run:

```bash
pytest -q tests/test_egoemotion_manifest.py tests/test_run_experiment_10s.py
```

Expected now: fail because `src.data.egoemotion.manifest` does not exist and current tests still use old manifest schema.

- [ ] **Step 2: Implement manifest validation module**

Implement `src/data/egoemotion/manifest.py` with:

```python
REQUIRED_MANIFEST_COLUMNS = {
    "subject", "task_name", "chunk_idx_in_task", "global_seq",
    "emotion_label", "emotion_name", "soft_label",
    "valence", "arousal", "dominance",
}

@dataclass
class Ego10sLoadReport:
    manifest_rows: int
    loaded_rows: int
    dropped_rows: int
    manifest_hash: str
    missing_by_encoder: dict[str, int]
    missing_by_subject: dict[str, dict[str, int]]
    loaded_by_subject: dict[str, int]
```

Core functions:

```python
def compute_manifest_hash(manifest: pd.DataFrame) -> str:
    cols = sorted(REQUIRED_MANIFEST_COLUMNS & set(manifest.columns))
    canonical = manifest[cols].copy()
    canonical["subject"] = canonical["subject"].astype(str).str.zfill(3)
    canonical = canonical.sort_values(["subject", "global_seq"]).reset_index(drop=True)
    payload = canonical.to_csv(index=False, float_format="%.8g")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

def parse_soft_label(value: object, num_classes: int = 9) -> torch.Tensor:
    vals = [float(v) for v in str(value).split(",")]
    if len(vals) != num_classes:
        raise ValueError(f"Expected {num_classes} soft-label values, got {len(vals)}")
    x = torch.tensor(vals, dtype=torch.float32)
    if not torch.isfinite(x).all() or x.sum() <= 0:
        raise ValueError(f"Invalid soft label: {value}")
    return x / x.sum()
```

`load_egoemotion_10s_by_subject(...)` must:

- normalize subject IDs with `zfill(3)`;
- validate required columns;
- validate `len(encoder_dirs) == len(modality_names)`;
- load only rows where all enabled embeddings exist;
- validate embedding object has `"embedding"`;
- validate saved `"subject"`, `"segment_idx"`, `"label"`, and `"emotion"` when present;
- validate `"manifest_hash"` if present, and warn if absent unless `require_manifest_hash=True`;
- normalize embedding shape through the existing `normalize_loaded_embedding` callback;
- return `(data_by_subject, report)`;
- raise `RuntimeError` if missing fraction exceeds `max_missing_fraction`.

- [ ] **Step 3: Replace local loader wrapper**

In `scripts/run_experiment_10s.py`, replace the current body of `load_10s_data_by_subject` with a compatibility wrapper around `load_egoemotion_10s_by_subject`. Preserve the public function name so existing imports continue working.

Add CLI:

```python
parser.add_argument("--max-missing-fraction", type=float, default=0.01)
parser.add_argument("--require-manifest-hash", action="store_true")
```

Default `0.01` allows the known 21 missing VideoMAE rows in the 40-subject Papagei run, but fails the current PulsePPG 28-subject truncation.

- [ ] **Step 4: Thread load report into run metadata**

After loading:

```python
data_by_subject, load_report = ...
logger.info("Loaded: %d/%d usable rows across %d subjects",
            load_report.loaded_rows, load_report.manifest_rows, len(data_by_subject))
logger.info("Missing by encoder: %s", load_report.missing_by_encoder)
```

Pass `load_report.__dict__` to `generate_report(...)` through a new optional `run_metadata` argument.

- [ ] **Step 5: Run tests**

```bash
pytest -q tests/test_egoemotion_manifest.py tests/test_run_experiment_10s.py
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/data/egoemotion scripts/run_experiment_10s.py tests/test_egoemotion_manifest.py tests/test_run_experiment_10s.py
git commit -m "fix: validate egoemotion manifest and embedding alignment"
```

---

### Task 2: Extraction Manifest Hash and Stale Cache Protection

**Files:**
- Modify: `scripts/segment_and_extract_10s.py`
- Test: `tests/test_egoemotion_manifest.py`

- [ ] **Step 1: Write failing extraction metadata test**

Add a unit-level test around a small helper function, not a full encoder run:

```python
def test_build_embedding_payload_includes_manifest_hash():
    payload = build_embedding_payload(
        embedding=torch.zeros(1, 512),
        chunk={"global_seq": 5, "emotion_label": 2, "emotion_name": "Excited"},
        subject="005",
        manifest_hash="abc123",
    )
    assert payload["manifest_hash"] == "abc123"
    assert payload["schema_version"] == "egoemotion_10s_v1"
```

- [ ] **Step 2: Add helper and hash use**

In `scripts/segment_and_extract_10s.py`, import `compute_manifest_hash` and add:

```python
EMBEDDING_SCHEMA_VERSION = "egoemotion_10s_v1"

def build_embedding_payload(embedding, chunk, subject, manifest_hash):
    return {
        "embedding": embedding.cpu(),
        "segment_idx": int(chunk["global_seq"]),
        "subject": str(subject).zfill(3),
        "label": int(chunk["emotion_label"]),
        "emotion": str(chunk["emotion_name"]),
        "manifest_hash": manifest_hash,
        "schema_version": EMBEDDING_SCHEMA_VERSION,
    }
```

After writing `manifest.csv`, compute `manifest_hash = compute_manifest_hash(manifest_df)` and pass it into extraction functions.

- [ ] **Step 3: Add controlled overwrite behavior**

Add CLI:

```python
parser.add_argument("--overwrite-stale", action="store_true")
```

When an existing embedding file exists:

- if it has matching `manifest_hash`, skip;
- if it lacks hash or has different hash and `--overwrite-stale` is false, log a warning and skip for backward compatibility;
- if `--overwrite-stale` is true, recompute and overwrite.

- [ ] **Step 4: Run tests**

```bash
pytest -q tests/test_egoemotion_manifest.py
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/segment_and_extract_10s.py tests/test_egoemotion_manifest.py
git commit -m "fix: stamp egoemotion embeddings with manifest hash"
```

---

### Task 3: LOSO Split Assertions and Fold Metadata

**Files:**
- Modify: `src/trainer/fusion_trainer.py`
- Modify: `src/utils/reporting.py`
- Test: `tests/test_trainer_loso_splits.py`

- [ ] **Step 1: Write failing split tests**

Add tests:

```python
def test_assert_disjoint_subject_split_passes_clean_split():
    train = [{"meta": {"subject": "001"}}]
    val = [{"meta": {"subject": "002"}}]
    test = [{"meta": {"subject": "003"}}]
    assert_disjoint_subject_split(train, val, test)

def test_assert_disjoint_subject_split_rejects_overlap():
    train = [{"meta": {"subject": "001"}}]
    test = [{"meta": {"subject": "001"}}]
    with pytest.raises(AssertionError, match="Train/test subject overlap"):
        assert_disjoint_subject_split(train, [], test)
```

- [ ] **Step 2: Implement split assertion helper**

In `src/trainer/fusion_trainer.py`:

```python
def _subjects_from_samples(samples: list[dict]) -> set[str]:
    subjects = set()
    for sample in samples:
        meta = sample.get("meta", {})
        subj = meta.get("subject")
        if subj is not None:
            subjects.add(str(subj).zfill(3))
    return subjects

def assert_disjoint_subject_split(train_data, val_data, test_data) -> None:
    train = _subjects_from_samples(train_data)
    val = _subjects_from_samples(val_data)
    test = _subjects_from_samples(test_data)
    if train and val:
        assert train.isdisjoint(val), f"Train/val subject overlap: {train & val}"
    if train and test:
        assert train.isdisjoint(test), f"Train/test subject overlap: {train & test}"
    if val and test:
        assert val.isdisjoint(test), f"Val/test subject overlap: {val & test}"
```

Call it in `run_loso` after `train_data`, `val_data`, and `test_data` are built.

- [ ] **Step 3: Attach fold metadata**

Each test fold result must include:

```python
metrics.update({
    "test_subject": test_subj,
    "val_subject": val_subj,
    "n_train": len(train_data),
    "n_val": len(val_data),
    "n_test": len(test_data),
    "n_train_subjects": len(train_subjects),
    "protocol": "train_N_minus_2_val_1_test_1",
})
```

- [ ] **Step 4: Expand reporting**

Modify `generate_report(..., run_metadata: dict | None = None)` to add:

- `## Run Metadata`
- `manifest_rows`, `loaded_rows`, `dropped_rows`, `manifest_hash`
- `missing_by_encoder`
- `n_folds`
- `## Fold Details` table with test subject, val subject, train size, val size, test size, F1, CCC.

Keep the aggregate metric table unchanged for backward readability.

- [ ] **Step 5: Run tests**

```bash
pytest -q tests/test_trainer_loso_splits.py tests/test_trainer.py
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/trainer/fusion_trainer.py src/utils/reporting.py tests/test_trainer_loso_splits.py tests/test_trainer.py
git commit -m "fix: assert loso split integrity and report fold metadata"
```

---

### Task 4: Fusion Correctness and Batch-Size Stability

**Files:**
- Modify: `src/fusion/early.py`
- Modify: `src/fusion/mid.py`
- Modify: `src/fusion/late.py`
- Modify: `src/fusion/distill_late.py`
- Modify: `src/fusion/tmc.py`
- Modify: `src/tasks/heads.py`
- Modify: `src/tasks/losses.py`
- Modify: `scripts/run_experiment.py`
- Test: `tests/test_fusion_modules.py`
- Test: `tests/test_losses.py`

- [ ] **Step 1: Write failing fusion tests**

Add tests:

```python
def test_early_and_mid_train_batch_size_one():
    for fusion in [
        EarlyFusion(d_common=16, num_modalities=3),
        MidFusion(d_common=16, modality_ids=["video", "eye_tracking", "ppg"]),
    ]:
        fusion.train()
        out = fusion([torch.randn(1, 16) for _ in range(3)], ["video", "eye_tracking", "ppg"])
        assert out.shape == (1, 16)

def test_late_fusion_rejects_unexpected_order():
    fusion = LateFusion(d_common=16, modality_ids=["video", "eye_tracking", "ppg"])
    with pytest.raises(ValueError, match="Expected modality order"):
        fusion([torch.randn(2, 16) for _ in range(3)], ["ppg", "video", "eye_tracking"])

def test_tmc_task_head_normalizes_belief_for_logits():
    head = TMCTaskHead(num_emotions=3)
    belief = torch.tensor([[0.2, 0.3, 0.1]])
    out = head(belief)
    probs = out["emotion_logits"].softmax(dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1))
```

- [ ] **Step 2: Replace BatchNorm**

In `EarlyFusion` and `MidFusion`, replace `nn.BatchNorm1d(...)` with `nn.LayerNorm(...)`.

- [ ] **Step 3: Add modality-order guards**

Update `LateFusion`, `DistillLateFusion`, and `TMCFusion` constructors to accept `modality_ids: list[str] | None = None`. Store:

```python
self.modality_ids = list(modality_ids) if modality_ids is not None else None
```

Add in `forward`:

```python
if self.modality_ids is not None and list(modality_ids) != self.modality_ids:
    raise ValueError(f"Expected modality order {self.modality_ids}, got {modality_ids}")
```

Modify `scripts/run_experiment.py` factory calls so positional fusion modules receive `modality_ids=enabled`.

- [ ] **Step 4: Normalize TMC belief for CE/KL**

In `TMCTaskHead.forward`:

```python
prob = belief / belief.sum(dim=-1, keepdim=True).clamp_min(1e-8)
log_prob = torch.log(prob.clamp_min(1e-8))
return {
    "emotion_logits": log_prob,
    "soft_logits": log_prob,
    "vad_pred": self.vad_head(prob),
}
```

Update the class docstring to say belief mass is normalized for CE/KL while uncertainty remains tracked in `TMCFusion`.

- [ ] **Step 5: Fix misleading Dirichlet KL comments**

In `src/tasks/losses.py`, correct the comment around `alpha_tilde` so it says the target class is set to 1 and non-target alphas are retained for KL-to-uniform penalty, matching the implementation.

- [ ] **Step 6: Run tests**

```bash
pytest -q tests/test_fusion_modules.py tests/test_losses.py
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add src/fusion src/tasks scripts/run_experiment.py tests/test_fusion_modules.py tests/test_losses.py
git commit -m "fix: stabilize fusion modules and tmc probability handling"
```

---

### Task 5: SEED-V EEG+Eye PCA Leakage Fix

**Files:**
- Modify: `scripts/run_seedv_eeg_eye_fusion.py`
- Test: `tests/test_seedv_eeg_eye_fusion.py`

- [ ] **Step 1: Refactor fold feature construction into testable helper**

Add:

```python
def build_fold_feature_sets(all_eye, all_eeg, all_labels, all_subjects, test_sub, pca_components=66):
    train_mask = all_subjects != test_sub
    test_mask = all_subjects == test_sub

    scaler_eye = StandardScaler().fit(all_eye[train_mask])
    eye_train = scaler_eye.transform(all_eye[train_mask])
    eye_test = scaler_eye.transform(all_eye[test_mask])

    scaler_eeg = StandardScaler().fit(all_eeg[train_mask])
    eeg_train_scaled = scaler_eeg.transform(all_eeg[train_mask])
    eeg_test_scaled = scaler_eeg.transform(all_eeg[test_mask])

    pca = PCA(n_components=min(pca_components, eeg_train_scaled.shape[1]))
    eeg_train_pca = pca.fit_transform(eeg_train_scaled)
    eeg_test_pca = pca.transform(eeg_test_scaled)

    return {
        "eye": (eye_train, eye_test),
        "eeg": (eeg_train_scaled, eeg_test_scaled),
        "concat": (np.hstack([eye_train, eeg_train_scaled]), np.hstack([eye_test, eeg_test_scaled])),
        "pca_concat": (np.hstack([eye_train, eeg_train_pca]), np.hstack([eye_test, eeg_test_pca])),
    }
```

- [ ] **Step 2: Write leakage regression test**

In `tests/test_seedv_eeg_eye_fusion.py`, monkeypatch/inspect that PCA is fit on train rows only. Use a synthetic test subject with extreme values and verify train-transformed PCA shape is independent of the held-out rows.

- [ ] **Step 3: Update `run_loso` path**

Remove global:

```python
eeg_scaled = scaler_pca.fit_transform(all_eeg)
pca = PCA(n_components=66)
all_eeg_pca = pca.fit_transform(eeg_scaled)
```

Compute PCA-concat features inside each fold using `build_fold_feature_sets`.

- [ ] **Step 4: Run tests**

```bash
pytest -q tests/test_seedv_eeg_eye_fusion.py
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_seedv_eeg_eye_fusion.py tests/test_seedv_eeg_eye_fusion.py
git commit -m "fix: fit seedv eeg-eye pca inside loso folds"
```

---

### Task 6: MM-Lego Leakage Guard

**Files:**
- Modify: `scripts/pretrain_lego_blocks.py`
- Modify: `scripts/run_lego_experiment.py`
- Test: add lightweight tests if imports are stable; otherwise run script-level smoke checks.

- [ ] **Step 1: Add checkpoint metadata**

When saving a pretrained block in `pretrain_lego_blocks.py`, include:

```python
"pretraining_scope": "global_supervised_all_subjects",
"subjects": list(all_subject_ids),
"uses_labels": True,
```

- [ ] **Step 2: Require explicit unsafe flag in runner**

In `run_lego_experiment.py`, add CLI:

```python
parser.add_argument("--allow-leaky-global-blocks", action="store_true")
```

After loading block checkpoint metadata, if `pretraining_scope == "global_supervised_all_subjects"` and flag is absent, raise:

```python
RuntimeError(
    "MM-Lego blocks were supervised-pretrained on all subjects. "
    "This leaks LOSO test subjects. Repretrain per fold or pass "
    "--allow-leaky-global-blocks for diagnostic-only runs."
)
```

- [ ] **Step 3: Document old results as unsafe**

Add a short warning to `reports/final_comprehensive_report.md` and `reports/final_comprehensive_report_zh.md` near any MM-Lego result: globally supervised MM-Lego block results are diagnostic-only unless fold-local pretraining is used.

- [ ] **Step 4: Smoke test**

```bash
python scripts/run_lego_experiment.py --mode merge-sum --device cpu
```

Expected: fails fast with the explicit leakage error if old global blocks are present.

- [ ] **Step 5: Commit**

```bash
git add scripts/pretrain_lego_blocks.py scripts/run_lego_experiment.py reports/final_comprehensive_report.md reports/final_comprehensive_report_zh.md
git commit -m "fix: guard against leaky mm-lego loso evaluation"
```

---

### Task 7: Verification and Report Drift Cleanup

**Files:**
- Modify: `scripts/verify_pipeline.py`
- Modify: `reports/final_comprehensive_report_zh.md`
- Modify: `reports/final_comprehensive_report.md`
- Test: run verifier manually against current data if available.

- [ ] **Step 1: Replace stale constants**

In `scripts/verify_pipeline.py`, replace checks like:

```python
n_files == 6663
manifest_count == 6663
len(manifest_subjects) == 28
```

with:

```python
expected_count = manifest_count
expected_subjects = len(manifest_subjects)
```

For each encoder, report:

```text
encoder files: N
manifest rows: M
coverage: N / M
missing: M - N
```

Do not fail solely because video has the known 9472/9493 coverage; fail only if coverage is below a configurable threshold.

- [ ] **Step 2: Add CLI threshold**

```python
parser.add_argument("--min-coverage", type=float, default=0.99)
```

- [ ] **Step 3: Correct current final report claims**

In both final reports, correct `final40_enriched_late_pulseppg` wording:

- It is currently a 28-subject / 6663-sample run, not a 40-subject run.
- Do not compare it directly against 40-subject Papagei runs until PulsePPG extraction covers all 40 subjects.

- [ ] **Step 4: Run verifier**

```bash
python scripts/verify_pipeline.py --min-coverage 0.99
```

Expected: pass for Papagei/InceptionTime/VideoMAE current 40-subject data; fail or warn clearly for PulsePPG if included.

- [ ] **Step 5: Commit**

```bash
git add scripts/verify_pipeline.py reports/final_comprehensive_report.md reports/final_comprehensive_report_zh.md
git commit -m "fix: align verification and reports with current data coverage"
```

---

### Task 8: Minimal Structural Refactor After Correctness Fixes

**Files:**
- Create: `src/fusion/wrappers.py`
- Create: `src/fusion/builders.py`
- Modify: `scripts/run_experiment.py`
- Modify: `scripts/run_experiment_10s.py`
- Modify: `scripts/run_finetune_ppg.py`
- Test: `tests/test_fusion_modules.py`, existing runner tests.

- [ ] **Step 1: Move `ProjectedFusion`**

Create `src/fusion/wrappers.py` with one shared `ProjectedFusion` implementation matching current behavior:

- builds `proj_dict` keyed by modality;
- projects through `ModalityProjector`;
- pools sequence tensors only if fusion module does not support sequence input;
- forwards optional masks;
- leaves `HybridProjectedFusion` in `run_finetune_ppg.py` for now because it handles raw encoders.

- [ ] **Step 2: Move fusion factory**

Create `src/fusion/builders.py`:

```python
def build_fusion_model(cfg: dict, registry: ModalityRegistry) -> tuple[ModalityProjector, BaseFusionModule]:
    ...
```

Move the current `scripts/run_experiment.py:61-207` logic there. Update all callers to import from `src.fusion.builders`.

- [ ] **Step 3: Keep script wrappers backward compatible**

In `scripts/run_experiment.py`, leave:

```python
from src.fusion.builders import build_fusion_model
from src.fusion.wrappers import ProjectedFusion
```

so older imports do not break.

- [ ] **Step 4: Run tests**

```bash
pytest -q tests/test_fusion_modules.py tests/test_run_experiment_10s.py
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/fusion/wrappers.py src/fusion/builders.py scripts/run_experiment.py scripts/run_experiment_10s.py scripts/run_finetune_ppg.py tests
git commit -m "refactor: share fusion builder and projected wrapper"
```

---

## Acceptance Criteria

- `pytest -q tests/test_egoemotion_manifest.py tests/test_run_experiment_10s.py tests/test_trainer_loso_splits.py tests/test_fusion_modules.py tests/test_seedv_eeg_eye_fusion.py` passes.
- Running `run_experiment_10s.py` with current Papagei/InceptionTime/VideoMAE data reports 40 subjects, 40 folds, and explicit 9472/9493 usable rows.
- Running `run_experiment_10s.py` with current PulsePPG config fails by default because only 28 subjects / 6663 rows are available.
- SEED-V EEG+Eye fusion no longer fits scaler/PCA on held-out test subjects.
- Reports include enough metadata to detect accidental 28-fold vs 40-fold comparisons.
- MM-Lego globally supervised block runs cannot be accidentally reported as valid LOSO.

## Execution Order

1. Task 1: Checked loader.
2. Task 3: LOSO/report metadata.
3. Task 4: fusion correctness.
4. Task 5: SEED-V PCA leakage.
5. Task 7: stale verifier/report correction.
6. Task 2: extraction hash stamping.
7. Task 6: MM-Lego guard.
8. Task 8: structural refactor.

This order makes existing results auditable before doing cleanup that could obscure behavior changes.

