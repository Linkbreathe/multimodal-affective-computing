# ege/baselines Code Review Notes

## Review Scope

- Review target: `ege/baselines`
- Base for comparison: `main`
- Merge base used for inspection: `4e9f9cd11e6b7a9ee449caa35c91ea6691605a71`
- Overall verdict: patch is incorrect

This document expands the four review findings into a more detailed explanation. The main theme is not that the new scripts fail loudly, but that several paths run successfully while producing misleading experimental results. That is more dangerous for baseline work, because the branch can appear "done" while the reported numbers no longer correspond to the intended models or data flow.

## High-Level Summary

The branch introduces a substantial multimodal baseline pipeline, but there are four correctness issues that materially affect experimental validity:

1. The full-run experiment loader collapses sequence embeddings into single vectors before fusion.
2. The eye-tracking extractor silently falls back to a random frozen PatchTST when the pretrained checkpoint is missing.
3. The embedding cache invalidation path does not actually refresh stale cache entries.
4. The 10-second benchmark loader also collapses sequence embeddings into single vectors.

The first and fourth issues are especially important because they change what model is really being evaluated. A run configured as `perceiver_io`, `qformer`, `healnet`, or `multimodal_lego` is no longer operating on multi-token modality inputs once those loaders have pooled everything away.

## Finding 1: Full-Run Experiments Collapse Sequence Embeddings

**Location:** `scripts/run_experiment.py:189-190`

### What the code does

Inside `load_data_by_subject()`, each cached embedding is loaded from disk and then normalized into a single vector with:

- `emb = data["embedding"]`
- `if emb.dim() == 2: emb = emb.mean(dim=0)`

That means any cached tensor with shape `[tokens, D]` becomes `[D]` before it reaches the projector and fusion model.

### Why this is a bug

For this branch, the raw cached embeddings are not always single vectors:

- Video embeddings are stored as multiple clip-level tokens.
- Eye-tracking embeddings are stored as many PatchTST tokens/chunks.
- PPG is typically already much closer to a single-vector representation.

The advanced fusion modules added in this branch were implemented specifically to consume richer tokenized modality inputs. Once `mean(dim=0)` is applied here, those modules no longer receive temporal or token-level structure. They still run, but they are effectively being benchmarked on pooled modality summaries rather than the intended sequential embeddings.

### Why this matters in practice

This issue changes the meaning of the experiment, not just its efficiency:

- `perceiver_io` loses the multi-token key/value set it is supposed to attend over.
- `qformer` loses the token bank that its learned queries are supposed to interrogate.
- `healnet` loses modality token structure before iterative cross-attention.
- `multimodal_lego` loses the richer per-modality representation that motivates the LegoBlock design.

So the code can report results under the names of those models while actually evaluating a much weaker pooled-input variant.

### Recommended fix direction

Keep cached `[tokens, D]` tensors intact through data loading for sequence-aware fusion paths. If some baselines truly need pooled inputs, make that an explicit, opt-in mode instead of an unconditional loader behavior.

## Finding 2: Missing PatchTST Checkpoint Produces Random Frozen Eye Embeddings

**Location:** `scripts/extract_embeddings.py:119-123`

### What the code does

The eye-tracking extraction path constructs a `PatchTSTEncoder`, checks whether `checkpoints/patchtst_pretrained.pt` exists, and only loads weights if the file is present. Regardless of whether weights were loaded, the encoder is then frozen immediately with `encoder.freeze()`.

### Why this is a bug

If the checkpoint is missing, the script does not fail, warn strongly, or switch to a training path. It simply freezes a randomly initialized PatchTST and writes those embeddings to disk as if they were legitimate pretrained features.

This is especially risky because the pipeline keeps moving afterward:

- extraction completes
- caches are written successfully
- later training/evaluation scripts run successfully
- reported numbers look real

But the eye modality is then built from meaningless random features.

### Why this matters in practice

On a fresh machine, CI environment, or any checkout where the checkpoint has not already been generated locally, the branch can silently produce invalid baselines. Because the failure mode is silent degradation instead of a crash, it is very easy to trust the results by mistake.

This same fallback pattern also appears in the 10-second extraction path, so the problem is not limited to one script.

### Recommended fix direction

Fail fast when the checkpoint is missing, or make the workflow explicitly require running the PatchTST pretraining step first. At minimum, this should be a hard error for the default extraction path instead of a silent fallback to random frozen weights.

## Finding 3: Cache Invalidation Does Not Refresh Stale Embeddings

**Location:** `src/encoders/extract.py:92-93`

### What the code does

`EmbeddingExtractor.extract_all()` calls `validate_cache()` first and logs either:

- cache valid, or
- cache invalid/missing

However, inside the actual extraction loop it still does:

- `if self.is_cached(...): continue`

`is_cached()` only checks whether the `.pt` file exists. It does not check whether the stored `config_hash` matches the current one.

### Why this is a bug

Once a cache file exists, a later run with changed preprocessing or encoder settings will still skip that file, even if the stored hash is stale. In other words, the code can correctly detect "this cache is outdated" and then immediately ignore that information.

That makes cache invalidation ineffective. The user may believe a new extraction was performed for a changed configuration, but the old embeddings remain on disk and keep getting reused.

### Why this matters in practice

This is dangerous for experimental iteration:

- a preprocessing change may not actually propagate into the cache
- an encoder configuration change may not regenerate embeddings
- old and new results can become mixed without obvious symptoms

For baseline work, this can poison comparisons because the experiment metadata and the actual cached features drift apart.

### Recommended fix direction

When a hash mismatch is detected, either:

- force regeneration of those files, or
- include the hash in the cache key/path so stale and fresh caches cannot collide.

The important part is that existence alone should not be treated as cache validity.

## Finding 4: The 10-Second Benchmark Loader Also Collapses Sequence Inputs

**Location:** `scripts/run_experiment_10s.py:156-158`

### What the code does

In the 10-second benchmark loader, each cached embedding is converted as follows:

- `emb = data["embedding"]`
- if it is 2D, it is either squeezed to `[D]` or averaged across tokens

So any `[tokens, D]` representation is again reduced to a single vector before training.

### Why this is a bug

The 10-second experiment path is presented as the paper-matched benchmark path. That makes correctness here especially important. If the loader always collapses token structure, then the benchmark cannot actually evaluate sequence-aware fusion architectures, even if the upstream extractor stores tokenized embeddings.

This means the 10-second path suffers from the same conceptual problem as the full-run path: model names imply token-aware fusion, but the actual inputs are already pooled.

### Why this matters in practice

This affects the trustworthiness of benchmark claims:

- the benchmark appears to compare sophisticated fusion models
- the actual data path removes the sequence structure those models need
- the resulting numbers are not faithful evaluations of the configured methods

Because this path is specifically framed as "paper-matched," the mismatch is especially misleading.

### Recommended fix direction

Preserve tokenized embeddings through the 10-second loader as well. If a pooled-feature baseline is desired, expose that as a separate, explicit experiment mode rather than silently applying pooling in the default benchmark path.

## Suggested Fix Order

If these issues are going to be addressed, the most practical order is:

1. Fix the full-run and 10-second loaders so sequence inputs are preserved.
2. Make missing pretrained PatchTST weights a hard failure.
3. Repair cache invalidation so configuration changes really trigger regeneration.

That order reduces the risk of generating or reusing invalid embeddings while updating the experiment scripts.

## Closing Note

None of these findings are about style. They are about the pipeline reporting results that do not correspond to the intended model/data configuration. For a baseline branch, that is the critical failure mode to avoid.
