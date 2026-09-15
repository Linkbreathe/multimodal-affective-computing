# Video Encoder Ablation Study — Design Spec

## Goal

Systematically test the frozen VideoMAE V2 encoder in isolation to determine
whether its standalone classification performance improves when:

1. The loss function is simplified to CE-only (no KL, no CCC/VAD).
2. The per-segment clip sequence `[T, 768]` is mean-pooled to a single
   global vector `[768]` before entering fusion.

These two ablations are run sequentially (Phase 1, then Phase 2), each
producing a LOSO weighted-F1 result comparable to the existing multi-modal
runs and to the paper's 0.46 classical baseline.

---

## Phase 1 — CE-Only Video Isolation

### What changes

A single new config file. **No code changes for this phase.**

```yaml
# configs/ablation/video_ce_only.yaml
fusion_type: early
modalities:
  video:         { encoder: VideoMAEV2, embed_dim: 768, enabled: true }
  eye_tracking:  { encoder: PatchTST,   embed_dim: 128, enabled: false }
  ppg:           { encoder: Papagei,    embed_dim: 512, enabled: false }
loss_weights:
  ce: 1.0
  kl: 0.0
  vad: 0.0
```

### Why it works

- `MultiTaskLoss` skips KL and CCC computation when their lambda is 0
  (see NaN-safety fix below).
- `emotion_head` is trained by CE. Evaluation uses
  `emotion_logits.argmax()` — correct.
- `soft_head` and `vad_head` still exist but receive no gradient (benign).
- `EarlyFusion` with 1 modality receives `[B, 256]` after
  `ProjectedFusion` mean-pools the video sequence (since
  `EarlyFusion.supports_sequence_input = False`).

### Run command

```bash
conda run -n visphy python scripts/run_experiment_10s.py \
  --fusion_config configs/ablation/video_ce_only.yaml \
  --name ablation_video_ce_only
```

---

## Phase 2 — Global-Pooled Video + CE-Only

### What changes

Add `--pool-clips` CLI flag to `run_experiment_10s.py`.
When set, mean-pool video embeddings at load time: `[T, 768]` → `[768]`.

### Implementation detail

In `load_10s_data_by_subject`, after loading each embedding:

```python
# Step 1: normalize (strips batch-dim artifact)
emb = normalize_loaded_embedding(data["embedding"], enc)

# Step 2: pool (video-only, after normalization)
if pool_clips and enc == "video_mae_v2" and emb.dim() == 2:
    emb = emb.mean(dim=0)  # [T, 768] → [768]
```

**Ordering contract — normalize then pool:**

| Step | Input | Output | Purpose |
|------|-------|--------|---------|
| 1. normalize | `[1, T, 768]` or `[T, 768]` | `[T, 768]` | Strip batch-dim artifact from extraction |
| 2. pool | `[T, 768]` | `[768]` | Collapse clip sequence to global vector |

Reversing the order is incorrect: `mean(dim=0)` on `[1, T, 768]` collapses
the batch dim, not the sequence dim.

**Video-only gate:**

The pool condition checks `enc == "video_mae_v2"` explicitly.
Eye-tracking (`patchtst_eye`, shape `[39, 128]`) is **never** pooled, even
though it is also 2D. This is not a generic "pool any sequence" flag.

### Run command

```bash
conda run -n visphy python scripts/run_experiment_10s.py \
  --fusion_config configs/ablation/video_ce_only.yaml \
  --pool-clips \
  --name ablation_video_ce_only_pooled
```

---

## Required Code Fix — NaN-Safe MultiTaskLoss

### Problem

`0.0 * NaN = NaN` in IEEE 754 floating point. The current
`MultiTaskLoss.forward` computes all three losses unconditionally:

```python
# CURRENT (unsafe)
ce  = self.ce_loss(outputs["emotion_logits"], targets["emotion_label"])
kl  = self.kl_loss(outputs["soft_logits"],    targets["soft_label"])   # may NaN
vad = self.ccc_loss(outputs["vad_pred"],      targets["vad"])          # may NaN
total = self.lambda_ce * ce + self.lambda_kl * kl + self.lambda_vad * vad
```

When `lambda_kl = 0` but `kl` is NaN (e.g. soft targets with exact zeros
where `0 * log(0) = NaN`), the total becomes NaN and gradients explode.

### Fix

Skip computation entirely when lambda is 0:

```python
def forward(self, outputs, targets):
    device = outputs["emotion_logits"].device
    total = torch.zeros(1, device=device)
    breakdown = {}

    if self.lambda_ce > 0:
        ce = self.ce_loss(outputs["emotion_logits"], targets["emotion_label"])
        total = total + self.lambda_ce * ce
        breakdown["ce"] = ce.item()
    else:
        breakdown["ce"] = 0.0

    if self.lambda_kl > 0:
        kl = self.kl_loss(outputs["soft_logits"], targets["soft_label"])
        total = total + self.lambda_kl * kl
        breakdown["kl"] = kl.item()
    else:
        breakdown["kl"] = 0.0

    if self.lambda_vad > 0:
        vad = self.ccc_loss(outputs["vad_pred"], targets["vad"])
        total = total + self.lambda_vad * vad
        breakdown["vad"] = vad.item()
    else:
        breakdown["vad"] = 0.0

    return total.squeeze(), breakdown
```

### Test

Unit test confirming `MultiTaskLoss` with `lambda_kl=0, lambda_vad=0`
returns a finite, non-NaN loss when:
- `soft_label` contains exact zeros
- `vad` targets are all-zeros

---

## Files Modified

| File | Change |
|------|--------|
| `configs/ablation/video_ce_only.yaml` | **New.** Video-only, CE-only ablation config. |
| `src/tasks/losses.py` | Fix `MultiTaskLoss.forward` to skip 0-lambda losses (NaN safety). |
| `scripts/run_experiment_10s.py` | Add `--pool-clips` flag; pass to `load_10s_data_by_subject`; video-only pool after normalize. |
| `tests/test_losses.py` | Add NaN-safety test for 0-lambda losses. |
| `tests/test_run_experiment_10s.py` | Add test for pool-clips with video-only gate and normalize-then-pool order. |

---

## Expected Results Matrix

| Experiment | Loss | Video repr | What it measures |
|------------|------|-----------|------------------|
| Phase 1 | CE-only | `[~18, 768]` sequence | Video baseline with simplified loss. Isolates multi-task harm. |
| Phase 2 | CE-only | `[768]` global | Whether temporal clip structure helps or hurts. |

Both report LOSO weighted F1 across 40 folds.

- If Phase 1 > current multi-modal → multi-task loss and/or other modalities are hurting.
- If Phase 2 > Phase 1 → fusion model fails to learn temporal aggregation; pre-pooling is better.
- If Phase 2 < Phase 1 → clip sequence carries useful temporal signal.

---

## Out of Scope

- Fine-tuning VideoMAE on EgoEmotion (separate future experiment).
- Changing the PPG output layer (`outputs[0]` vs `outputs[3]`).
- Switching from weighted F1 to macro F1.
- Any changes to the extraction pipeline or saved embeddings.
