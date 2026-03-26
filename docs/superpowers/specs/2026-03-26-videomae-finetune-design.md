# VideoMAE V2 Fine-Tuning on EgoEmotion — Design Spec

## Goal

Fine-tune `OpenGVLab/VideoMAEv2-Base` on 9-class emotion classification
using the existing fixed train/val/test split, then extract globally-pooled
`[768]` embeddings and run a linear probe. This is Phase C — validate that
fine-tuning closes the feature quality gap before investing in 40-fold LOSO
fine-tuning (Phase A).

---

## Phase C Pipeline

```
1. Load VideoMAEv2-Base with num_classes=9 (classification head)
2. Fine-tune on train split (2,128 segments) from pov.mp4 frames
3. Select best checkpoint by val macro F1 (279 segments)
4. Extract [768] embeddings from fine-tuned model for all 2,678 segments
5. Train nn.Linear(768, 9) probe on fixed split
6. Report macro F1 on test split (271 segments)
```

---

## 1. Data

### Fixed split CSVs

**Location:** `data/egoemotion_raw/ce_hardlabel_manifests/splits/`

| File | Segments | Format |
|------|----------|--------|
| `train.csv` | 2,128 | `subject_id,segments,emotion` (header + rows) |
| `val.csv` | 279 | same |
| `test.csv` | 271 | same |

Example row: `005,5,Neutral` — subject 005, segment index 5, emotion label Neutral.

### Emotion-to-label mapping

```python
EMOTIONS = ["Amused", "Content", "Excited", "Awe", "Neutral",
            "Fear", "Sad", "Disgust", "Anger"]
# label = EMOTIONS.index(emotion)
```

### Video frame loading

Reuse the proven logic from `scripts/segment_and_extract_10s.py:269-303`:

```python
# Segment index → absolute time in pov.mp4
abs_start_90 = session_a_start_90 + seg_idx * 900  # 90Hz samples
start_sec = abs_start_90 / 90.0
end_sec = start_sec + 10.0

# Read frames from pov.mp4
start_frame = int(start_sec * fps)
end_frame = int(end_sec * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
# Read end_frame - start_frame frames, resize to 224x224, BGR→RGB
```

### Clip sampling for fine-tuning

The old repo uses `--num_frames 16 --sampling_rate 6` — meaning **one
16-frame clip per segment**, with temporal stride 6 (spans ~96 frames
≈ 3.2s at 30fps). For fine-tuning, sample one clip per segment:

```python
# From ~300 frames (10s at 30fps), sample 16 with stride 6
indices = list(range(0, 16 * 6, 6))  # [0, 6, 12, ..., 90]
# If segment has fewer frames, clamp indices
clip = frames_arr[indices]  # [16, 224, 224, 3]
```

Normalize with ImageNet stats:
```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
clip = (clip / 255.0 - MEAN) / STD   # float32
clip = clip.permute(3, 0, 1, 2)      # [3, 16, 224, 224]
```

### `task_times.npy`

Required for computing `session_a_start_90` per subject. Already used by
the extraction scripts. Located via `configs/base.yaml` → `data_dir`.

---

## 2. Model

### Loading (with transformers 5.x meta-tensor workaround)

```python
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

config = AutoConfig.from_pretrained(
    "OpenGVLab/VideoMAEv2-Base", trust_remote_code=True)
config.model_config["num_classes"] = 9  # add classification head

model_class = get_class_from_dynamic_module(
    "modeling_videomaev2.VideoMAEv2",
    "OpenGVLab/VideoMAEv2-Base",
    trust_remote_code=True,
)
model = model_class(config)  # CPU init, bypasses meta device bug

# Load pretrained weights (ignore missing head.weight/head.bias)
weights_path = hf_hub_download("OpenGVLab/VideoMAEv2-Base", "model.safetensors")
sd = load_file(weights_path)
model.load_state_dict(sd, strict=False)
```

The `strict=False` allows the new `head.weight` and `head.bias` (for 9
classes) to be randomly initialized while loading all pretrained backbone
weights.

### Forward paths

- **Fine-tuning:** `model(pixel_values)` → `[B, 9]` logits (goes through
  `forward_features()` → `head_dropout()` → `nn.Linear(768, 9)`)
- **Embedding extraction:** `model.extract_features(pixel_values)` → `[B, 768]`
  (mean-pooled patch tokens through LayerNorm, bypasses classification head)

---

## 3. Training Recipe

Matches the old repo's `vit_b_egoemotion_ft.sh` as closely as possible,
adapted for single-GPU PyTorch.

| Setting | Value | Source |
|---------|-------|--------|
| Optimizer | AdamW | `ft.sh:38` |
| Learning rate | 2e-4 | `ft.sh:39` |
| Weight decay | 0.05 | `ft.sh:44` |
| Layer decay | 0.75 | `ft.sh:42` |
| Drop path | 0.2 | `ft.sh:40` |
| Grad clip | 5.0 | `ft.sh:41` |
| Warmup epochs | 5 | `ft.sh:45` |
| Total epochs | 40 | `ft.sh:46` |
| Batch size | 4 (gradient accumulation 4 = effective 16) | `ft.sh:29-30` |
| Mixup/cutmix/smoothing | 0 / 0 / 0 | `ft.sh:49-51` |
| Class weights | Yes (inverse frequency) | `ft.sh:53` |
| Best checkpoint metric | Macro F1 | `ft.sh:52` |
| Frames per clip | 16, stride 6 | `ft.sh:34-35` |

### Layer decay implementation

Layer decay assigns different learning rates to different transformer
layers. With `layer_decay=0.75` and `depth=12`:

```python
# Layer i (0-indexed from input) gets lr * (layer_decay ^ (depth - i))
# Layer 0 (closest to input):  lr * 0.75^12 ≈ lr * 0.032
# Layer 11 (closest to output): lr * 0.75^1  = lr * 0.75
# Head:                          lr * 1.0
```

This is implemented by creating parameter groups with per-group `lr`.

### Warmup schedule

Linear warmup over first 5 epochs, then cosine decay to 0.

---

## 4. Embedding Extraction (Post Fine-Tuning)

After fine-tuning, extract embeddings for ALL 2,678 segments (all splits):

```python
model.eval()
with torch.no_grad():
    emb = model.extract_features(clip.unsqueeze(0))  # [1, 768]
```

Save to `data/embeddings_10s_finetuned/video_mae_v2/{subject}/segment_{idx:04d}.pt`
with the same format as existing embeddings but using a SINGLE `[768]`
vector (not `[T, 768]` clip sequence).

---

## 5. Linear Probe (Post Extraction)

Reuse `scripts/run_linear_probe_10s.py` from the previous experiment,
pointed at the new embeddings directory:

```bash
conda run -n visphy python scripts/run_linear_probe_10s.py \
    --embeddings-dir data/embeddings_10s_finetuned/video_mae_v2 \
    --name linear_probe_finetuned_video
```

This runs LOSO and reports both macro F1 and weighted F1.

Additionally, run a **fixed-split probe** for direct comparison with the
old repo (add a `--fixed-split` flag to the linear probe script).

---

## 6. Files

| File | Change |
|------|--------|
| `scripts/finetune_videomae_10s.py` | **New.** Fine-tuning script. |
| `scripts/run_linear_probe_10s.py` | **Modify.** Add `--fixed-split` option. |
| `tests/test_finetune.py` | **New.** Test dataset loading and model forward. |

---

## 7. Expected GPU Requirements

- Model: 86M params × 4 bytes = ~344 MB
- Batch size 4 × `[3, 16, 224, 224]` = ~37 MB per batch
- Gradient + optimizer states: ~1.4 GB
- **Total: ~2-3 GB VRAM** — well within RTX 5080 16 GB
- **Time estimate:** ~2,128 segments × 40 epochs = ~85K forward+backward passes.
  At ~50ms each ≈ ~70 minutes.

---

## 8. Controlled vs Uncontrolled (Phase C)

### Controlled

| Factor | Phase C value | Old repo value | Match? |
|--------|---------------|----------------|--------|
| Architecture | VideoMAEv2 ViT-B | VideoMAEv2 ViT-B | Yes |
| Fine-tune objective | 9-class CE with class weights | Same | Yes |
| Optimizer | AdamW lr=2e-4, wd=0.05 | Same | Yes |
| Epochs | 40 | 40 | Yes |
| Layer decay | 0.75 | 0.75 | Yes |
| Augmentation | None | None | Yes |
| Probe | nn.Linear(768, 9) | nn.Linear(768, 9) | Yes |
| Probe optimizer | AdamW lr=1e-3 wd=1e-4 | Same | Yes |
| Model selection | Val macro F1 | Val macro F1 | Yes |

### Uncontrolled

| Factor | Phase C | Old repo | Impact |
|--------|---------|----------|--------|
| Starting checkpoint | HF VideoMAEv2-Base (self-supervised) | distilled-from-giant + K710 | Unknown — tests fine-tuning value of HF checkpoint |
| Clip sampling | stride 6, 1 clip/segment | Same config but via VideoClsDataset | Should be equivalent |
| Frame decoding | OpenCV from pov.mp4 | Same | Equivalent |

---

## 9. Executable Checklist

```bash
# 1. Fine-tune (writes best checkpoint to checkpoints/)
conda run -n visphy python scripts/finetune_videomae_10s.py \
    --device cuda --name videomae_ft_fixedsplit

# 2. Extract embeddings from fine-tuned model
#    (built into the fine-tuning script as a post-training step)

# 3. Run LOSO linear probe on fine-tuned embeddings
conda run -n visphy python scripts/run_linear_probe_10s.py \
    --embeddings-dir data/embeddings_10s_finetuned/video_mae_v2 \
    --name linear_probe_finetuned_video

# 4. Compare results:
#    - Fine-tuned linear probe macro F1 vs frozen linear probe (0.1706)
#    - Fine-tuned linear probe weighted F1 vs frozen MLP ablation (0.2279)
#    - Fine-tuned fixed-split macro F1 vs old repo's reported number
```

---

## Out of Scope

- Phase A (LOSO fine-tuning, 40 runs) — separate spec after Phase C results
- Loading the distilled-from-giant checkpoint — test HF Base first
- Multi-task loss, soft labels, VAD
- Other modalities (eye tracking, PPG)
