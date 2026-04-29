## Overall Architecture Assessment

The codebase has a workable core, but it is currently split across **several partially overlapping pipelines** that are easy to confuse:

* **Current egoEMOTION 10s task-aware pipeline:** `scripts/segment_and_extract_10s.py` → `scripts/run_experiment_10s.py` → `src/trainer/fusion_trainer.py`
* **Legacy task-level egoEMOTION pipeline:** `scripts/run_experiment.py`, `src/data/segments.py`, `src/data/label_builder.py`
* **Fine-tuning pipeline:** `scripts/run_finetune_ppg.py`
* **MM-Lego pipeline:** `scripts/pretrain_lego_blocks.py`, `scripts/run_lego_experiment.py`
* **Audit/verification scripts:** `scripts/audit_clip_leakage_ego.py`, `scripts/verify_pipeline.py`
* **SEED-V configs and legacy dataset flows** mixed into the same config/script structure

The main 10s egoEMOTION path is identifiable, but it is **not isolated enough**. Critical assumptions are spread across scripts: embedding directory naming in `src/encoders/registry.py`, segment IDs in `segment_and_extract_10s.py`, manifest parsing in `run_experiment_10s.py`, model construction in `scripts/run_experiment.py`, and LOSO splitting in `FusionTrainer`.

I did **not** find direct subject overlap in the main 10s LOSO split: `src/trainer/fusion_trainer.py:513-522` excludes the test subject and validation subject from training. Evaluation is also correctly wrapped with `@torch.no_grad()` at `src/trainer/fusion_trainer.py:308`. However, I found several serious issues that can still invalidate results or make runs non-reproducible:

1. The main “LOSO” protocol is actually **test-subject plus held-out validation-subject** per fold, so training uses 38 subjects in a 40-subject run, not 39.
2. Same stimulus/task appears across subjects, while labels are task-derived. This is not subject leakage, but it is a major **task/stimulus confound**.
3. The 10s manifest and embedding files are tightly coupled only by path and `global_seq`, with no metadata validation.
4. Several fusion modules are order-dependent and can silently break if modality order changes.
5. `EarlyFusion` and `MidFusion` can fail on training batches of size 1 because they use `BatchNorm1d`.
6. The MM-Lego pretraining script leaks all subjects into pretrained modality blocks before LOSO evaluation.
7. Some verification/tests are stale and contradict the current 9493/40-subject pipeline.

---

## High Priority Issues — Critical

### 1. Main LOSO split avoids direct subject overlap, but the protocol is not classic 40-fold LOSO

**Location:** `src/trainer/fusion_trainer.py:513-522`

```python
val_subj = subject_ids[(i + 1) % len(subject_ids)]
train_subjects = [
    s for s in subject_ids if s not in (test_subj, val_subj)
]

train_data = [
    s for subj in train_subjects for s in all_data.get(subj, [])
]
val_data = all_data.get(val_subj, [])
test_data = all_data.get(test_subj, [])
```

This does correctly prevent the test subject from appearing in training. However, it is not standard “train on N-1 subjects, test on 1 subject” LOSO. It is closer to:

> train on N-2 subjects, validate on 1 subject, test on 1 subject.

That is acceptable if documented, but it should not be reported ambiguously as plain LOSO. More importantly, there are no runtime assertions that the split is clean.

**Actionable fix:**

Add explicit split metadata and assertions before each fold:

```python
def assert_disjoint_subject_split(train_data, val_data, test_data):
    def subjects(samples):
        return {s["meta"]["subject"] for s in samples}

    train_subjects = subjects(train_data)
    val_subjects = subjects(val_data)
    test_subjects = subjects(test_data)

    assert train_subjects.isdisjoint(val_subjects), (
        f"Train/val subject overlap: {train_subjects & val_subjects}"
    )
    assert train_subjects.isdisjoint(test_subjects), (
        f"Train/test subject overlap: {train_subjects & test_subjects}"
    )
    assert val_subjects.isdisjoint(test_subjects), (
        f"Val/test subject overlap: {val_subjects & test_subjects}"
    )
```

Then store metadata in each sample in `load_10s_data_by_subject`, e.g. `subject`, `global_seq`, `task_name`, `chunk_idx_in_task`.

---

### 2. Strong task/stimulus leakage confound under subject-LOSO

The project is doing subject-LOSO, but the same egoEMOTION tasks/stimuli appear across subjects, and labels are derived at the task level. That means the model can learn:

> “this task/video/activity usually has this emotion label”

rather than subject-general emotion recognition.

Your own audit script recognizes this risk:

**Location:** `scripts/audit_clip_leakage_ego.py:3-11`

```python
Answers: is task classification disguised as emotion recognition?

Tests:
  A) Task-ID linear probe under subject-LOSO ...
  B) Emotion probe under three protocols:
       B1 subject-LOSO (leakage-permissive baseline)
       B2 task-LOSO (stimulus never seen during training)
       B3 subject x task-LOSO (honest)
```

This is not a strict train/test subject leak, but it is a **protocol-level confound**. A high F1 under subject-LOSO may partly reflect task recognition.

**Actionable fix:**

Report at least three protocols side-by-side:

1. **Subject-LOSO:** current protocol.
2. **Task-LOSO:** hold out task/stimulus across all subjects.
3. **Subject × Task holdout:** test on subject/task pairs while excluding both the subject and the task from training.

Do not rely only on subject-LOSO for claims of generalizable emotion recognition.

---

### 3. The “honest” subject × task audit is currently not honest

**Location:** `scripts/audit_clip_leakage_ego.py:193-212`

```python
te = (data["subject"] == s) & (data["task"] == t)
if te.sum() == 0:
    continue
tr = ~te
```

This only removes the exact `(subject, task)` pair from training. Training still contains:

* the same subject on other tasks
* the same task from other subjects

That contradicts the script comment that B3 is “subject x task-LOSO (honest)”.

**Correct fix:**

```python
te = (data["subject"] == s) & (data["task"] == t)

# Honest holdout: training sees neither this subject nor this task.
tr = (data["subject"] != s) & (data["task"] != t)
```

You should also log skipped folds where this leaves too few classes in training.

---

### 4. Manifest/embedding coupling can silently corrupt labels or modalities

**Location:** `scripts/run_experiment_10s.py:127-147`, `149-169`

```python
global_seq = int(row["global_seq"])

all_exist = all(
    (emb_path / enc / subj / f"segment_{global_seq:04d}.pt").exists()
    for enc in encoder_dir_names
)
if not all_exist:
    continue

for enc in encoder_dir_names:
    data = torch.load(
        emb_path / enc / subj / f"segment_{global_seq:04d}.pt",
        weights_only=False,
    )
    emb = normalize_loaded_embedding(data["embedding"], enc)
```

The loader trusts that:

```text
manifest row subject/global_seq == embedding file subject/segment_idx/label
```

But it never checks:

```python
data["subject"]
data["segment_idx"]
data["label"]
data["emotion"]
```

The extraction script does save those fields:

**Location:** `scripts/segment_and_extract_10s.py:251-257`, `343-349`, `407-413`

```python
torch.save({
    "embedding": emb.cpu(),
    "segment_idx": c["global_seq"],
    "subject": subj,
    "label": c["emotion_label"],
    "emotion": c["emotion_name"],
}, save_path)
```

So the validation data exists, but the training loader ignores it.

This is a major silent bug risk because `scripts/segment_and_extract_10s.py` rewrites `manifest.csv` every time before extraction:

**Location:** `scripts/segment_and_extract_10s.py:488-493`

```python
manifest_df = pd.DataFrame(manifest_rows)
manifest_df.to_csv(manifest_path, index=False)
```

If you regenerate the manifest after changing task enumeration, task labels, subject filtering, or raw data, previously extracted embeddings can still sit on disk under the same `segment_XXXX.pt` names but refer to different chunks.

**Actionable fix:**

Fail fast when metadata does not match:

```python
expected_label = int(row["emotion_label"])

if str(data.get("subject")).zfill(3) != subj:
    raise ValueError(f"{path}: subject mismatch")

if int(data.get("segment_idx")) != global_seq:
    raise ValueError(f"{path}: segment_idx mismatch")

if "label" in data and int(data["label"]) != expected_label:
    raise ValueError(
        f"{path}: label mismatch. file={data['label']} manifest={expected_label}"
    )
```

Also save a `manifest_hash` into every embedding file at extraction time and refuse to load embeddings from a different manifest hash.

---

### 5. Missing modality files are silently dropped, changing fold composition

**Location:** `scripts/run_experiment_10s.py:130-136`

```python
all_exist = all(
    (emb_path / enc / subj / f"segment_{global_seq:04d}.pt").exists()
    for enc in encoder_dir_names
)
if not all_exist:
    continue
```

This silently removes samples if any modality is missing. Since video embeddings are described as “nearly complete,” this can alter:

* per-subject sample counts
* per-class class distribution
* which chunks survive in each fold
* comparability between Papagei vs PulsePPG configs
* comparability between early/mid/late/TMC/enriched late

There is no per-modality missingness report.

**Actionable fix:**

Track and log missingness by subject and modality, then fail if missingness exceeds a configured threshold:

```python
missing_by_modality[enc] += 1
missing_by_subject[subj][enc] += 1
```

At minimum, the experiment report should include:

```text
manifest rows: 9493
usable multimodal rows: N
dropped rows: 9493 - N
dropped by modality:
  video_mae_v2: ...
  inceptiontime: ...
  papagei_ppg: ...
usable rows per subject:
  001: ...
  ...
```

---

### 6. `EarlyFusion` and `MidFusion` can crash with batch size 1

**Locations:**

`src/fusion/early.py:22-31`

```python
self.mlp = nn.Sequential(
    nn.Linear(d_in, d_common * 2),
    nn.BatchNorm1d(d_common * 2),
    ...
    nn.Linear(d_common * 2, d_common),
    nn.BatchNorm1d(d_common),
    ...
)
```

`src/fusion/mid.py:31-37`

```python
self.cross_modal = nn.Sequential(
    nn.Linear(d_in, d_common * 2),
    nn.BatchNorm1d(d_common * 2),
    ...
)
```

The training loader does not use `drop_last=True`:

**Location:** `src/trainer/fusion_trainer.py:492-498`

```python
return DataLoader(
    _ListDS(data),
    batch_size=batch_size,
    shuffle=shuffle,
    collate_fn=collate,
    num_workers=0,
)
```

With `batch_size: 64` in `configs/base.yaml:24-29`, the last training batch can be size 1 depending on fold sample count. `BatchNorm1d` in training mode can fail with:

```text
Expected more than 1 value per channel when training
```

This is especially dangerous because it is fold-size dependent.

**Best fix:**

Use `LayerNorm` instead of `BatchNorm1d` in fusion MLPs. For subject-level affective data, `LayerNorm` is usually safer than batch statistics anyway.

```python
nn.Linear(d_in, d_common * 2),
nn.LayerNorm(d_common * 2),
nn.ReLU(),
```

Alternative fix:

```python
drop_last=shuffle
```

for training only, but that discards data and does not fix instability from small batches.

---

### 7. TMC output is treated as logits, but it is a belief mass vector

**Locations:**

`src/fusion/tmc.py:145-171`

```python
evidence = self.evidence_branches[i](emb)
alpha = evidence + 1.0
S = alpha.sum(dim=-1, keepdim=True)
belief = evidence / S
uncertainty = K / S
...
return combined_b
```

`src/tasks/heads.py:36-41`

```python
log_belief = torch.log(belief + 1e-8)
return {
    "emotion_logits": log_belief,
    "soft_logits": log_belief,
    "vad_pred": self.vad_head(belief),
}
```

`combined_b` is a belief mass. It sums to `1 - uncertainty`, not necessarily 1. Then the task head logs it and feeds it to `CrossEntropyLoss` and `KL`. `CrossEntropyLoss` applies `log_softmax` internally, so it will renormalize the belief masses and effectively discard the explicit uncertainty semantics.

This does not necessarily crash, but it makes the TMC objective mathematically inconsistent with the evidential representation.

**Actionable fix options:**

Option A — use expected class probabilities from Dirichlet:

```python
prob = alpha / alpha.sum(dim=-1, keepdim=True)
logits = torch.log(prob.clamp_min(1e-8))
```

Option B — if using Dempster-combined belief, normalize for CE/KL but keep uncertainty separately:

```python
prob = combined_b / combined_b.sum(dim=-1, keepdim=True).clamp_min(1e-8)
emotion_logits = torch.log(prob.clamp_min(1e-8))
```

Option C — implement a proper evidential loss and do not call the belief vector “logits.”

Also fix the misleading comment in `src/tasks/losses.py:41-48`. The comment says the correct-class alpha is kept, but the implementation at `src/tasks/losses.py:63-65` actually sets the target class to 1 and keeps non-target alphas:

```python
alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha
```

That implementation is consistent with penalizing non-target evidence, but the comment is wrong.

---

### 8. Fusion modules are order-dependent and can silently bind the wrong branch to a modality

**Locations:**

`src/fusion/late.py:23-33`, `41-48`

```python
self.branches = nn.ModuleList([... for _ in range(num_modalities)])
...
branch_outs = [branch(emb) for branch, emb in zip(self.branches, embeddings)]
```

`src/fusion/tmc.py:116-120`, `145-147`

```python
self.evidence_branches = nn.ModuleList([
    EvidenceBranch(d_common, num_classes, dropout)
    for _ in range(num_modalities)
])
...
for i, (emb, mod_id) in enumerate(zip(embeddings, modality_ids)):
    evidence = self.evidence_branches[i](emb)
```

These implementations assume that the modality order from YAML never changes. But `ModalityRegistry.get_enabled_modalities()` returns modalities in config dictionary order:

**Location:** `src/encoders/registry.py:37-38`

```python
return [k for k, v in self._config.items() if v.get("enabled", False)]
```

If YAML order changes, or if egoEMOTION/SEED-V configs differ, the meaning of branch 0/1/2 changes silently.

**Actionable fix:**

Use `nn.ModuleDict` keyed by modality name:

```python
self.branches = nn.ModuleDict({
    mod: Branch(d_common, d_out, dropout)
    for mod in modality_ids
})

branch_outs = [self.branches[mod](emb) for mod, emb in zip(modality_ids, embeddings)]
```

For weighted late fusion:

```python
self.weights = nn.ParameterDict({
    mod: nn.Parameter(torch.tensor(1.0))
    for mod in modality_ids
})
```

Or store a fixed `self.modality_ids` and assert exact equality in forward:

```python
if modality_ids != self.modality_ids:
    raise ValueError(f"Expected modality order {self.modality_ids}, got {modality_ids}")
```

---

### 9. MM-Lego pretraining leaks test subjects into LOSO evaluation

**Locations:**

`pretrain_lego_blocks.py:97-113`

```python
dataset = UnimodalDataset(
    cfg["embeddings_dir"], encoder_dir_name, all_subject_ids, labels
)
...
train_ds, val_ds = torch.utils.data.random_split(
    dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(cfg.get("seed", 42))
)
```

`pretrain_lego_blocks.py:215-230`

```python
extractor = SegmentExtractor(cfg["data_dir"], f"{cfg['data_dir']}/task_times.npy")
all_subjects = extractor.get_subject_ids()
...
pretrain_one_modality(
    mod_name, enc_dir, input_dim,
    all_subjects, labels, cfg, device, args.save_dir
)
```

`run_lego_experiment.py:437-459`

```python
blocks, heads = load_pretrained_blocks(args.blocks_dir, enabled, cfg, device)
...
fold_results = run_merge_experiment(...)
# or
fold_results = run_fuse_experiment(...)
```

The Lego blocks are pretrained once using **all subjects**, with labels, then reused for every LOSO fold. That means the test subject has influenced the pretrained modality blocks before the fold evaluation.

This invalidates MM-Lego LOSO results if they are reported alongside the main frozen-embedding fusion results.

**Actionable fix:**

Pretrain Lego blocks **inside each fold using only training subjects**:

```python
for test_subj in subject_ids:
    train_subjects = [s for s in subject_ids if s != test_subj]
    blocks = pretrain_blocks(subjects=train_subjects)
    evaluate_on(test_subj)
```

If pretraining is meant to be external/self-supervised, remove labels from pretraining and document that no test-subject labels are used.

---

### 10. Fine-tuning PPG pipeline can append samples with missing raw PPG

**Location:** `scripts/run_finetune_ppg.py:199-207`, `220-255`, `267-276`

```python
ppg_signal = None
if "ppg" in raw_set:
    ppg_path = raw_path / subj / "ppg_ear_125hz.npy"
    if ppg_path.exists():
        ppg_signal = np.load(ppg_path)
    else:
        log.warning(f"PPG not found for {subj}: {ppg_path}")
```

Later:

```python
if mod_name in raw_set:
    emb_list.append(torch.zeros(1))
```

And the sample is still appended even if `raw_signals` is empty:

```python
samples.append({
    "embeddings": emb_list,
    "modality_ids": config_mod_names,
    "raw_signals": raw_signals,
    ...
})
```

If raw PPG is missing, the placeholder `torch.zeros(1)` can reach the projector for PPG, which expects a 512-dimensional embedding. Depending on the path, this either crashes or creates a batch-size mismatch.

The trainer’s raw signal collation also permits partial raw batches:

**Location:** `src/trainer/fusion_trainer.py:473-480`

```python
tensors = [b["raw_signals"][sig_name] for b in batch
           if sig_name in b.get("raw_signals", {})]
if tensors:
    raw_signals[sig_name] = torch.stack(tensors)
```

This can produce `raw_signals["ppg"]` with fewer rows than the main embedding batch.

**Actionable fix:**

Skip samples if required raw modality is missing:

```python
if "ppg" in raw_set and ppg_signal is None:
    continue
```

And assert raw batch size:

```python
if sig_name in required_raw_modalities:
    if len(tensors) != len(batch):
        raise ValueError(
            f"Raw signal {sig_name} missing for "
            f"{len(batch) - len(tensors)} samples in batch"
        )
```

---

### 11. Verification and tests are stale relative to the current 9493/40-subject pipeline

**Locations:**

`verify_pipeline.py:162-165`

```python
n_files == 6663
...
"expected 6663 files"
```

`verify_pipeline.py:236-239`

```python
manifest_count == 6663
...
len(manifest_subjects) == 28
```

But your project background says the current run has about **9493 manifest chunks across 40 subjects**. This verifier will now produce false failures or encourage people to “fix” the pipeline toward stale numbers.

`extract_embeddings_10s.py:3-5` is also stale:

```python
paper's manifest ... exact 2,678 non-overlapping 10-second segments across 40 subjects.
```

The tests in `tests/test_run_experiment_10s.py` are also incompatible with the current loader signature and manifest schema. Example:

**Location:** `tests/test_run_experiment_10s.py:22-32`

```python
manifest = pd.DataFrame([
    {"subject": 5, "segment_path": "005/ppg_segments/5.p", "label": 4},
])

loaded = load_10s_data_by_subject(
    str(embeddings_dir),
    ["video_mae_v2", "patchtst_eye", "papagei_ppg"],
    ["video", "eye_tracking", "ppg"],
    manifest,
    data_dir,
)
```

Current `load_10s_data_by_subject` expects `global_seq`, `soft_label`, `valence`, `arousal`, `dominance`, and `emotion_label`, and it does not take `data_dir`.

**Actionable fix:**

Update the verifier to compare against the actual manifest dynamically:

```python
expected_count = len(manifest)
expected_subjects = manifest["subject"].nunique()
```

Update tests to use the current manifest schema.

---

## Medium / Low Priority Optimizations

### 1. Centralize egoEMOTION schema constants

`EMOTIONS`, session task lists, label maps, sampling rates, and chunk lengths are repeated across:

* `scripts/run_experiment_10s.py:36-39`
* `scripts/segment_and_extract_10s.py`
* `scripts/segment_only_10s.py:26-33`
* `src/data/label_builder.py`

Create one module:

```text
src/data/egoemotion/schema.py
```

containing:

```python
EMOTIONS
EMOTION_TO_LABEL
SESSION_B_TASKS
SESSION_A_EMOTION_MAP
FS_EYE = 90
FS_PPG = 125
CHUNK_SECONDS = 10
```

Then import it everywhere.

---

### 2. Separate egoEMOTION, SEED-V, legacy, and 10s pipelines

Right now the repo structure encourages accidental cross-use. A cleaner structure would be:

```text
src/
  data/
    egoemotion/
      schema.py
      manifest.py
      datasets.py
      segmentation.py
      labels.py
    seedv/
      schema.py
      datasets.py
  fusion/
    modules/
    builders.py
  training/
    loso.py
    metrics.py
scripts/
  egoemotion/
    extract_10s_embeddings.py
    train_fusion_10s.py
    audit_task_leakage.py
  seedv/
    train_seedv.py
  legacy/
    run_experiment_legacy.py
```

This makes it harder to accidentally run SEED-V configs through egoEMOTION logic or legacy task-level data through the 10s manifest pipeline.

---

### 3. Move `ProjectedFusion` out of scripts

`ProjectedFusion` is duplicated in:

* `scripts/run_experiment.py:210-256`
* `scripts/run_experiment_10s.py:48-92`
* similar logic in `scripts/run_finetune_ppg.py:50-117`

Move it to:

```text
src/fusion/wrappers.py
```

with one implementation of:

```python
project()
pool_if_needed()
forward()
```

Then the scripts only build configs and call shared library code.

---

### 4. Replace the large fusion factory with a registry

Current factory:

**Location:** `scripts/run_experiment.py:61-207`

This is a long `if/elif` chain with hardcoded imports and config logic.

Use a registry:

```python
FUSION_BUILDERS = {
    "early": build_early,
    "mid": build_mid,
    "late": build_late,
    "bottleneck": build_bottleneck,
    "tmc": build_tmc,
    "healnet": build_healnet,
}
```

Move this to:

```text
src/fusion/builders.py
```

This makes fusion variants easier to add without editing a monolithic script.

---

### 5. Main loader ignores `training.num_workers`

`configs/base.yaml:24-29` says:

```yaml
training:
  batch_size: 64
  lr: 0.0001
  max_epochs: 100
  patience: 10
  num_workers: 4
```

But the trainer hardcodes:

**Location:** `src/trainer/fusion_trainer.py:492-498`

```python
num_workers=0
```

This is likely intentional because data is already loaded in memory, but then remove `num_workers` from config or pass it through consistently.

---

### 6. Logging/reporting is too thin for reproducibility

Current report generation only aggregates numeric keys:

**Location:** `src/utils/reporting.py:21-38`

It does not save:

* full merged YAML config
* config hash plus actual config text
* fold train/val/test subjects
* fold sample counts
* per-subject usable sample counts
* per-class class distributions
* confusion matrices
* macro F1
* per-class F1
* KL/CE/VAD loss breakdown
* best epoch per fold
* embedding manifest hash
* git commit or dirty status
* encoder directory names actually used

`ResultsRegistry` appends JSON without locking:

**Location:** `src/utils/registry.py:19-28`

This is fragile if multiple experiments finish at the same time.

---

### 7. Improve metrics

Current metrics include weighted F1 and CCC:

**Location:** `src/utils/metrics.py:9-25`

Add:

```python
macro_f1
balanced_accuracy
per_class_f1
confusion_matrix
macro_ccc
ccc_valence
ccc_arousal
ccc_dominance
```

You already compute per-dimension CCC in `src/trainer/fusion_trainer.py:373-379`, but the aggregate is named simply `ccc`. Rename it to `macro_ccc` to avoid ambiguity.

---

### 8. Standardize paths in YAML

`configs/base.yaml:2-4` contains:

```yaml
data_dir: "data/datasets/egoemotion_raw"
embeddings_dir: "data/embeddings/egoemotion/papagei"
ref_data_dir: "/mnt/c/Users/Public/Data/egoEMOTION/egoEMOTION"
```

This is inconsistent with the current 10s path:

```text
data/embeddings/egoemotion/10s_task_aware
```

Move dataset-specific paths into a dataset block:

```yaml
dataset:
  name: egoemotion
  raw_dir: data/datasets/egoemotion_raw
  embeddings_dir: data/embeddings/egoemotion/10s_task_aware
  manifest: data/embeddings/egoemotion/10s_task_aware/manifest.csv
```

Avoid absolute Windows/WSL paths in base config.

---

### 9. Remove committed cache artifacts

The zip contains `__pycache__` directories under scripts and `src`. These should not be in the repository. Add to `.gitignore`:

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
```

---

## Refactoring Example 1 — Safer manifest-indexed 10s dataset loader

This directly addresses the biggest data-flow issue: manifest/embedding coupling and silent sample drops.

```python
# src/data/egoemotion/manifest_dataset.py

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import torch


REQUIRED_MANIFEST_COLUMNS = {
    "subject",
    "global_seq",
    "task_name",
    "chunk_idx_in_task",
    "emotion_label",
    "soft_label",
    "valence",
    "arousal",
    "dominance",
}


@dataclass(frozen=True)
class EgoSampleMeta:
    subject: str
    global_seq: int
    task_name: str
    chunk_idx_in_task: int


def parse_soft_label(value: str, num_classes: int = 9) -> torch.Tensor:
    values = [float(v) for v in str(value).split(",")]
    if len(values) != num_classes:
        raise ValueError(f"Expected {num_classes} soft-label values, got {len(values)}")

    tensor = torch.tensor(values, dtype=torch.float32)

    if not torch.isfinite(tensor).all():
        raise ValueError(f"Non-finite soft label: {value}")

    total = tensor.sum()
    if total <= 0:
        raise ValueError(f"Soft label has non-positive sum: {value}")

    # Normalizes away harmless CSV rounding drift.
    return tensor / total


def validate_embedding_metadata(
    *,
    path: Path,
    obj: dict,
    subject: str,
    global_seq: int,
    emotion_label: int,
) -> None:
    file_subject = str(obj.get("subject", "")).zfill(3)
    file_segment = obj.get("segment_idx", None)
    file_label = obj.get("label", None)

    if file_subject and file_subject != subject:
        raise ValueError(f"{path}: subject mismatch: file={file_subject}, manifest={subject}")

    if file_segment is not None and int(file_segment) != global_seq:
        raise ValueError(
            f"{path}: segment mismatch: file={file_segment}, manifest={global_seq}"
        )

    if file_label is not None and int(file_label) != emotion_label:
        raise ValueError(
            f"{path}: label mismatch: file={file_label}, manifest={emotion_label}"
        )


def load_egoemotion_10s_by_subject(
    *,
    embeddings_dir: str | Path,
    manifest: pd.DataFrame,
    encoder_dirs: list[str],
    modality_names: list[str],
    normalize_embedding: Callable[[torch.Tensor, str], torch.Tensor],
    pool_video_clips: bool = False,
    max_missing_fraction: float = 0.0,
) -> dict[str, list[dict]]:
    missing_columns = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing_columns:
        raise ValueError(f"Manifest missing required columns: {sorted(missing_columns)}")

    if len(encoder_dirs) != len(modality_names):
        raise ValueError("encoder_dirs and modality_names must have the same length")

    embeddings_dir = Path(embeddings_dir)
    data_by_subject: dict[str, list[dict]] = {}
    missing_by_encoder: Counter[str] = Counter()
    missing_by_subject: dict[str, Counter[str]] = defaultdict(Counter)

    manifest = manifest.copy()
    manifest["subject"] = manifest["subject"].astype(str).str.zfill(3)
    manifest = manifest.sort_values(["subject", "global_seq"]).reset_index(drop=True)

    for subject, rows in manifest.groupby("subject", sort=True):
        samples: list[dict] = []

        for _, row in rows.iterrows():
            global_seq = int(row["global_seq"])
            emotion_label = int(row["emotion_label"])

            paths = {
                enc: embeddings_dir / enc / subject / f"segment_{global_seq:04d}.pt"
                for enc in encoder_dirs
            }

            missing = [enc for enc, path in paths.items() if not path.is_file()]
            if missing:
                for enc in missing:
                    missing_by_encoder[enc] += 1
                    missing_by_subject[subject][enc] += 1
                continue

            embeddings: list[torch.Tensor] = []
            for enc in encoder_dirs:
                path = paths[enc]
                obj = torch.load(path, map_location="cpu", weights_only=False)

                if not isinstance(obj, dict) or "embedding" not in obj:
                    raise ValueError(f"{path}: expected dict with key 'embedding'")

                validate_embedding_metadata(
                    path=path,
                    obj=obj,
                    subject=subject,
                    global_seq=global_seq,
                    emotion_label=emotion_label,
                )

                emb = normalize_embedding(obj["embedding"], enc)

                if pool_video_clips and enc == "video_mae_v2" and emb.dim() == 2:
                    emb = emb.mean(dim=0)

                if not torch.isfinite(emb).all():
                    raise ValueError(f"{path}: embedding contains NaN or Inf")

                embeddings.append(emb)

            meta = EgoSampleMeta(
                subject=subject,
                global_seq=global_seq,
                task_name=str(row["task_name"]),
                chunk_idx_in_task=int(row["chunk_idx_in_task"]),
            )

            samples.append({
                "embeddings": embeddings,
                "modality_ids": list(modality_names),
                "labels": {
                    "emotion_label": torch.tensor(emotion_label, dtype=torch.long),
                    "soft_label": parse_soft_label(row["soft_label"]),
                    "vad": torch.tensor(
                        [
                            float(row["valence"]),
                            float(row["arousal"]),
                            float(row["dominance"]),
                        ],
                        dtype=torch.float32,
                    ),
                },
                "meta": {
                    "subject": meta.subject,
                    "global_seq": meta.global_seq,
                    "task_name": meta.task_name,
                    "chunk_idx_in_task": meta.chunk_idx_in_task,
                },
            })

        if samples:
            data_by_subject[subject] = samples

    total_rows = len(manifest)
    total_missing = sum(missing_by_encoder.values())
    missing_fraction = total_missing / max(total_rows * len(encoder_dirs), 1)

    if missing_fraction > max_missing_fraction:
        raise RuntimeError(
            "Too many missing embeddings: "
            f"{missing_fraction:.2%}. Missing by encoder: {dict(missing_by_encoder)}"
        )

    return data_by_subject
```

This turns silent assumptions into explicit validation.

---

## Refactoring Example 2 — Safer modality-keyed late fusion

This fixes the branch-order dependency in `LateFusion`.

```python
# src/fusion/late_keyed.py

from __future__ import annotations

import torch
import torch.nn as nn

from src.fusion.base import BaseFusionModule


class KeyedLateFusion(BaseFusionModule):
    """Late fusion with branches keyed by modality name, not list position."""

    supports_sequence_input = False

    def __init__(
        self,
        *,
        d_common: int,
        modality_ids: list[str],
        mode: str = "weighted",
        dropout: float = 0.1,
        d_out: int | None = None,
    ) -> None:
        super().__init__(d_common=d_common, d_out=d_out or d_common)

        if not modality_ids:
            raise ValueError("modality_ids must be non-empty")

        if mode not in {"average", "weighted"}:
            raise ValueError(f"Unsupported late-fusion mode: {mode}")

        self.modality_ids = list(modality_ids)
        self.mode = mode

        self.branches = nn.ModuleDict({
            mod: nn.Sequential(
                nn.Linear(d_common, d_common),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_common, self.d_out),
            )
            for mod in self.modality_ids
        })

        if self.mode == "weighted":
            self.logits = nn.ParameterDict({
                mod: nn.Parameter(torch.zeros(()))
                for mod in self.modality_ids
            })

    def forward(
        self,
        embeddings: list[torch.Tensor],
        modality_ids: list[str],
        masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if len(embeddings) != len(modality_ids):
            raise ValueError(
                f"Got {len(embeddings)} embeddings for {len(modality_ids)} modalities"
            )

        unknown = set(modality_ids) - set(self.modality_ids)
        if unknown:
            raise ValueError(f"Unknown modalities in forward pass: {sorted(unknown)}")

        branch_outputs = {
            mod: self.branches[mod](emb)
            for mod, emb in zip(modality_ids, embeddings)
        }

        ordered_outputs = [branch_outputs[mod] for mod in modality_ids]
        stacked = torch.stack(ordered_outputs, dim=0)

        if self.mode == "average":
            return stacked.mean(dim=0)

        weight_logits = torch.stack([self.logits[mod] for mod in modality_ids])
        weights = torch.softmax(weight_logits, dim=0).view(-1, 1, 1)

        return (stacked * weights).sum(dim=0)
```

This prevents a YAML order change from silently swapping learned branches.

---

## Most Important Immediate Fixes

If you want the shortest actionable priority list, I would fix these first:

1. Add metadata validation in `load_10s_data_by_subject`.
2. Add split assertions and sample metadata to `FusionTrainer.run_loso`.
3. Replace `BatchNorm1d` with `LayerNorm` in `EarlyFusion` and `MidFusion`.
4. Fix `audit_clip_leakage_ego.py` subject × task split.
5. Treat MM-Lego pretrained blocks as leaky unless pretrained inside each fold.
6. Update stale verifier/tests to the 9493/40-subject manifest.
7. Log usable sample counts, missing modality counts, class distributions, and fold subject IDs in every report.

The main 10s fusion trainer is not fundamentally broken, but right now the codebase makes it too easy to obtain plausible-looking results from subtly different or contaminated pipelines.
