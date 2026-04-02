# Depression Detection & Analysis — Implementation Plan (v5.1)

## Background

This project builds a **multimodal AI system** for estimating depression risk from facial expressions and acoustic features. Two datasets are available locally:

1. **E-DAIC** — ✅ 275 clinical interview participants, fully downloaded, all labeled with PHQ-8 continuous + binary scores
2. **D-Vlog** — ✅ 961 YouTube vlog subjects, pre-extracted acoustic + facial landmark features

**Hardware**: NVIDIA RTX 5060 (8 GB VRAM, CUDA)

**Storage**: SSD at `D:\DL-Datasets\` for large dataset processing. Internal drive at `c:\Users\arjun\Desktop\DLP\` for code and small files. No disk space constraints — maximize dataset utilization.

**Reference SOTA** (cosmaadrian/multimodal-depression-from-video — ECIR 2024, [arXiv:2401.02746](https://arxiv.org/abs/2401.02746)):

| Dataset | Their F1 | Architecture | Window |
|---------|----------|-------------|--------|
| D-Vlog | **0.78** | Perceiver (8L, 8×32d heads, cross+self attn) | 9s |
| DAIC-WOZ | **0.67** | Same | 9s |
| E-DAIC | **0.56** | Same | 6s |

Our target: **match or exceed** these benchmarks via multi-task learning, PHQ regression, and quality-aware gating.

> **Scope clarification**: The system is a **depression risk estimation / behavioral screening support** tool. It is not a clinical diagnostic instrument. Body-movement prediction is a second-phase research track — not part of the initial trained model.

---

## Resolved Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| PHQ Threshold | **PHQ ≥ 10** (standard) | Best sensitivity/specificity balance per meta-analytic evidence; matches `Depression_label` column |
| Naming | **E-DAIC** consistently (not DAIC-WOZ) | Avoids confusion between original 189-session DAIC-WOZ and extended 275-session release |
| Text Modality | **Deferred until transcript cleaning pipeline is proven** | Transcripts have no speaker column — raw use would re-introduce interviewer bias |
| Body Movement | **Visual overlay only in v1; second-phase research track** | Neither dataset contains body-joint supervision |
| Live Inference | **Prototype risk estimation** — requires feature-space bridge | Training uses OpenFace/eGeMAPS; live uses MediaPipe — different feature spaces |
| D-Vlog Raw Videos | ✅ **773/961 downloaded** to `D:\DL-Datasets\dvlog_videos\` (~54.6 GB). 188 unavailable on YouTube. | Enables feature-space bridging for 80%+ of subjects |

---

## Phase 1: Dataset Analysis ✅ COMPLETE

### 1.1 E-DAIC

**Status**: ✅ All 275 archives downloaded, all 275 labeled

**Splits & Class Distribution**:

| Split | Total | Depressed | Non-depressed | Rate |
|-------|-------|-----------|---------------|------|
| Train | 163 | 37 | 126 | 23% |
| Dev | 56 | 12 | 44 | 21% |
| Test | 56 | 17 | 39 | 30% |
| **Total** | **275** | **66** | **209** | **24%** |

> **⚠️ Test Split Domain Shift**: The test set consists entirely of AI-controlled interviews (sessions 600–718), while train and dev are primarily Wizard-of-Oz sessions. Test performance measures generalization across interview modalities, not just unseen subjects.

**PHQ-8 Severity Distribution**:

| PHQ Range | Severity | Count | Percentage |
|-----------|----------|-------|------------|
| 0–4 | Minimal | 122 | 44% |
| 5–9 | Mild | 67 | 24% |
| 10–14 | Moderate | 43 | 16% |
| 15–19 | Moderately Severe | 33 | 12% |
| 20–27 | Severe | 10 | 4% |

**Demographics**: 169 male, 105 female, 1 unknown. Age 18–69, mean 41.

**Per-Archive Contents** (12 files each):

| File | Description | Usable Dims |
|------|-------------|-------------|
| `XXX_AUDIO.wav` | Raw audio recording | — |
| `XXX_Transcript.csv` | Time-aligned dialogue (no speaker column!) | Deferred |
| `OpenFace2.1.0_Pose_gaze_AUs.csv` | 6 pose + 8 gaze + 17 AU_r + 18 AU_c = **49 dims** | **49** |
| `OpenSMILE2.3.0_egemaps.csv` | Semicolon-delimited, 25 cols (2 metadata + **23 acoustic features**) | **23** |
| `OpenSMILE2.3.0_mfcc.csv` | MFCC features | Alt. acoustic |
| `CNN_ResNet.mat` / `CNN_VGG.mat` | Deep visual embeddings (~152/304 MB) | Secondary |
| `densenet201.csv` / `vgg16.csv` | DenseNet/VGG16 features | Secondary |
| `BoAW_*.csv` / `BoVW_*.csv` | Bag-of-words (audio & visual) | Secondary |

**Label Files** (in `wwwedaic/labels/`):

| File | Content |
|------|---------|
| `detailed_lables.csv` | Master: 275 rows, PHQ-8 subscores, PCL-C subscores, age, gender, Depression_label, PTSD_label, split |
| `Detailed_PHQ8_Labels.csv` | 219 rows, PHQ8 subscores + totals only |
| `train_split.csv` / `dev_split.csv` / `test_split.csv` | Split participant IDs |

**Transcript Format Warning**:
The transcript CSV contains only `Start_Time, End_Time, Text, Confidence` — **no speaker column**. The archived format does not expose reliable speaker identity, and sampled sessions show non-target content (setup chatter, interviewer prompts). Text is unsafe for model training until a full 275-session corpus audit and rule-based cleaning pipeline prove that participant-only turns can be reliably isolated.

### 1.2 D-Vlog

**Pre-extracted Features** (✅ complete, 961 subjects):

| Feature | File | Shape | Dtype | Description |
|---------|------|-------|-------|-------------|
| Acoustic | `<id>_acoustic.npy` | `(T, 25)` | float64 | 25 LLDs per second |
| Visual | `<id>_visual.npy` | `(T, 136)` | float64 | 68 facial landmarks × 2 coords |

**Splits** (from `labels.csv`): train=647, valid=102, test=212. Labels: 555 depression / 406 normal (58%/42%).

**Raw Videos** (✅ downloaded, 773/961 available):

- Location: `D:\DL-Datasets\dvlog_videos\`
- Format: `<youtube_key>.mp4` (e.g., `2s3EFyjUmfs.mp4`)
- Size: ~54.6 GB total, ~72.3 MB average per file, ~124.7 hours total duration
- 188 videos unavailable on YouTube (deleted/private/region-blocked)
- 1 incomplete download (`pKLVKOpxNe4` — partial fragments only)

**Raw Video Coverage by Fold**:

| Fold | Available | Total | Coverage |
|------|-----------|-------|----------|
| Train | 517 | 647 | 79.9% |
| Valid | 84 | 102 | 82.4% |
| Test | 172 | 212 | 81.1% |

**Raw Video Coverage by Label**:

- Depression: 408/555 available (73.5%)
- Normal: 365/406 available (89.9%)

> **Note**: All 961 subjects have complete pre-extracted features regardless of video availability. Raw videos are only needed for the feature-space bridge (Phase 6). 80%+ coverage per fold is sufficient for bridge training.

**Known Data Issues**:

- **14 subjects** have acoustic/visual length mismatch (e.g., sid=454: acoustic=137 vs visual=163). DataLoader must truncate to `min(len_a, len_v)`.
- Features stored as **float64** — must cast to float32 in DataLoader.
- `normalization_stats.npz` exists with precomputed mean/std vectors. **Normalization protocol**: for strict final experiments, compare provided stats vs train-fold-only recomputed stats and report which protocol is used. Provided stats may leak test distribution if computed over the full dataset.
- Visual features are **facial landmarks only** — no body/pose data.

### 1.3 Dataset Comparison

| Aspect | E-DAIC | D-Vlog |
|--------|--------|--------|
| Setting | Clinical interviews (controlled) | YouTube vlogs (in-the-wild) |
| Participants | 275 | 961 |
| Visual features | OpenFace: pose+gaze+AUs+AU_c (**49 dim**) | 68 landmarks (**136 dim**) |
| Acoustic features | eGeMAPS (**23 dim**) | 25 LLDs |
| Deep CNN features | ResNet, VGG, DenseNet | — |
| Labels | PHQ-8 continuous (0–27) + binary | Binary only |
| Class balance | 24% depressed | 58% depressed |
| Body data | None | None |

> **Visual dim upgrade (v5.1)**: Changed from 31 to **49 dims** after studying the reference repo, which includes 18 binary AU_c activation features alongside the 17 continuous AU_r intensities. Binary activations complement continuous intensities (e.g., "is AU12 active at all?" vs "how intense is AU12?").

---

## Phase 2: Data Pipeline ✅ IN PROGRESS

### 2.1 Dataset Audit ✅ DONE

#### `src/data/dataset_audit.py`

**Completed.** Output saved to `data/audit_report.json`. Results:

- D-Vlog: 961/961 subjects — 0 NaN/Inf, 0 corrupt, 14 length mismatches (12 off-by-1, 2 at ~23-26 frames)
- E-DAIC: 275/275 archives matched to labels, 14 files each, 33 label columns verified
- Sequence lengths: min=23, max=3968, mean=596, median=472
- All splits verified: train=647, valid=102, test=212 (D-Vlog); train=163, dev=56, test=56 (E-DAIC)

### 2.2 Manifest-First Data System ✅ DONE

#### `src/data/manifest_generator.py`

**Completed.** Manifest written to `D:\DL-Datasets\processed\manifest.jsonl` — 1236 entries (961 D-Vlog + 275 E-DAIC). Windowing is deferred to DataLoader time (manifest is subject-level, not window-level).

Each manifest line contains:

```json
{
  "subject_id": "300_P",
  "dataset_source": "edaic",
  "label_binary": 0,
  "label_phq": 4,
  "fold": "train",
  "visual_path": "...",
  "acoustic_path": "...",
  "aligned_len": 1200,
  "quality_flags": [],
  "session_type": "woz"
}
```

### 2.3 E-DAIC Pipeline 🔄 IN PROGRESS

#### `src/data/edaic_extractor.py` ✅ DONE (extraction running)

- Extracts from `.tar.gz` → `D:\DL-Datasets\processed\edaic\<pid>_P\`
- **Visual**: 6 pose + 8 gaze + 17 AU_r + 18 AU_c = **49 dims** (upgraded from 31)
- **Acoustic**: eGeMAPS → strip 2 metadata columns → **23 dims**
- Also extracts frame-level OpenFace confidence for quality gating
- Incremental: re-running skips already-extracted archives
- Extraction running at ~0.2 archives/sec, ETD ~20 min for 275 archives

#### `src/data/edaic_dataset.py` ⬜ NEXT

- Similar to `dvlog_dataset.py` but adapted for E-DAIC:
- Resample to 1 FPS (OpenFace ~30fps, eGeMAPS ~100fps)
- Quality filter: drop frames with OpenFace confidence < 0.5
- 30-second windows, 15-second stride (baseline; **ablate 6s, 9s, 15s, 30s** — reference repo uses 6-9s)
- Trim leading/trailing invalid tracking segments per session
- Drop windows below minimum valid-frame ratio (>=50% valid frames)
- Compute and cache normalization stats from **train split only**

### 2.4 D-Vlog Pipeline ✅ DONE

#### `src/data/dvlog_dataset.py`

**Completed and verified.** Results:

| Split | Subjects | Windows | Depression | Normal |
|-------|----------|---------|------------|--------|
| Train | 647 | 25,738 | 375 | 272 |
| Valid | 102 | 3,746 | 57 | 45 |
| Test | 212 | 8,139 | 123 | 89 |

- float64 → float32 casting ✅
- min-truncation length alignment ✅
- Train-fold-only normalization (Welford's algorithm) ✅
- WeightedRandomSampler for class balance ✅
- Subject overlap across splits: 0 ✅
- Shapes: visual `(30, 136)`, acoustic `(30, 25)`, dtype `float32`

### 2.5 Unified Format

```python
{
    "visual": (num_windows, 30, visual_dim),
    "acoustic": (num_windows, 30, acoustic_dim),
    "label_binary": int,                    # 0 or 1
    "label_phq": float,                     # PHQ score (E-DAIC) or NaN (D-Vlog)
    "subject_id": str,
    "dataset_source": str,                  # "edaic" or "dvlog"
    "fold": str,                            # "train" / "dev" / "test"
    "quality_flags": list[str],
    "modality_mask": dict,                  # {"visual": True, "acoustic": True}
}
```

Feature dims inferred at load time. Dataset-specific input projections handle the difference (E-DAIC 49/23 vs D-Vlog 136/25).

### 2.6 Class Imbalance Handling

| Method | Layer | Effect |
|--------|-------|--------|
| **Focal Loss** (γ=2.0, α=0.75) | Loss function | Down-weight easy negatives |
| **Weighted Random Sampling** | DataLoader | 50/50 class balance per batch |
| **Temporal augmentation** | Window level | Random jitter, speed perturbation |
| Cross-dataset pretraining | Strategy | Only if ablation proves it helps E-DAIC dev F1 |

---

## Phase 3: Model Architecture

### 3.1 Architecture: Source-Aware Multimodal Transformer

```
┌─────────────────┐     ┌──────────────────┐
│  Visual Stream   │     │  Acoustic Stream  │
│  (T, V_dim)      │     │  (T, A_dim)       │
└────────┬────────┘     └────────┬─────────┘
         │                        │
    ┌────▼────┐            ┌─────▼─────┐
    │ Source-  │            │ Source-    │
    │ Specific │            │ Specific   │
    │ Proj     │            │ Proj       │
    │ (→128)   │            │ (→128)     │
    └────┬────┘            └─────┬─────┘
         │                        │
    ┌────▼────┐            ┌─────▼─────┐
    │ Visual  │            │ Acoustic   │
    │ Encoder │            │ Encoder    │
    │(CNN+GRU)│            │(TCN+GRU)   │
    └────┬────┘            └─────┬─────┘
         │                        │
    ┌────▼────┐            ┌─────▼─────┐
    │Modality │            │ Modality   │
    │  Gate   │            │   Gate     │
    │(quality)│            │ (quality)  │
    └────┬────┘            └─────┬─────┘
         │                        │
         └──────────┬─────────────┘
                    │
           ┌────────▼────────┐
           │  Cross-Modal     │
           │  Attention       │
           │  Fusion (4-head) │
           │  + missing-mod   │
           │    masking       │
           └────────┬────────┘
                    │
       ┌────────────┼────────────┐
       │            │            │
  ┌────▼────┐ ┌────▼────┐ ┌────▼────┐
  │ Head A  │ │ Head B  │ │ Head C  │
  │ E-DAIC  │ │ E-DAIC  │ │ D-Vlog  │
  │ Binary  │ │ PHQ Reg │ │ Binary  │
  └─────────┘ └─────────┘ └─────────┘
```

### 3.2 Key Design Changes from v3

| Change | Before (v3) | After (v5.1) | Why |
|--------|-------------|-------------|-----|
| Visual dims (E-DAIC) | 27 → 31 | **49** (6 pose + 8 gaze + 17 AU_r + 18 AU_c) | Verified from CSV headers + reference repo uses AU_c |
| Acoustic dims | 88 | **23** (25 cols - 2 metadata) | Verified from actual archive file |
| Task heads | Single binary head | **Multi-task: 3 heads** | PHQ regression exploits continuous labels |
| Modality gates | None | **Quality-aware sigmoid gates** | Down-weight unreliable streams |
| Missing-mod masking | None | **Explicit modality mask** | Handles face-only, audio-only, both |
| Dim inference | Hardcoded | **Inferred from data** | Prevents dimension errors |
| Text stream | Optional 3rd modality | **Removed from v1** | No speaker column = unsafe to use |
| Window size | Fixed 30s | **Ablate 6s, 9s, 15s, 30s** | Reference repo achieves SOTA with 9s windows |

### 3.3 Component Specifications

| Component | Architecture | Output Dim | Parameters |
|-----------|-------------|-----------|------------|
| Input Projection | Linear (per-dataset, per-modality) | 128 | ~20K |
| Visual Encoder | 3× 1D-CNN (k=3,5,7) → BiGRU (2L, h=128) | 256 | ~800K |
| Acoustic Encoder | TCN (dilations 1,2,4,8) → BiGRU (2L, h=128) | 256 | ~800K |
| Modality Gates | σ(Linear(confidence_features)) per modality | scalar | ~1K |
| Source-Conditioned Norm | Dataset-specific LayerNorm/BatchNorm + source embedding | — | ~2K |
| Cross-Modal Fusion | 4-head cross-attn + 2× self-attn + modality masking | 512 | ~1M |
| Head A (E-DAIC binary) | Attn pool → 512→256→1, Dropout(0.3) | 1 | ~130K |
| Head B (E-DAIC PHQ reg) | Attn pool → 512→256→1 | 1 | ~130K |
| Head C (D-Vlog binary) | Attn pool → 512→256→1, Dropout(0.3) | 1 | ~130K |
| **Total** | | | **~3M** |

### 3.4 Multi-Task Loss

```
L = α · L_binary_edaic + β · L_phq_regression + γ · L_binary_dvlog

Where:
  L_binary = FocalLoss(γ=2.0, α=0.75)
  L_phq_regression = SmoothL1Loss
  α, β, γ tuned via validation (start: 1.0, 0.5, 1.0)

Loss masking rule:
  E-DAIC samples → activate Head A (binary) + Head B (PHQ regression) only
  D-Vlog samples → activate Head C (binary) only
  Inactive heads produce zero loss for that sample.
```

---

## Phase 4: Training Strategy

### 4.1 Benchmark Ladder (Build Baselines First)

Before multimodal training, establish baselines:

```
Level 1: Acoustic-only baseline (each dataset separately)
Level 2: Visual-only baseline (each dataset separately)
Level 3: Acoustic + Visual bimodal (each dataset)
Level 4: Cross-dataset pretraining (D-Vlog → E-DAIC)
Level 5: Full multi-task multimodal

Success criterion: Each level must beat the best single-modality baseline.
If Level 3 doesn't beat Level 1 or 2, fusion is broken — debug before proceeding.
```

### 4.2 Training Pipeline

```
Stage 0: Baselines
    ├── Acoustic-only on D-Vlog
    ├── Visual-only on D-Vlog
    ├── Acoustic-only on E-DAIC
    ├── Visual-only on E-DAIC
    └── Record all baseline metrics

Stage 1: Bimodal training on D-Vlog (961 subjects)
    ├── 1a: Bimodal (acoustic+visual) training (25 epochs)
    ├── 1b: Must beat best unimodal D-Vlog baseline
    └── Target: F1 ≥ 0.65

Stage 2: Fine-tune on E-DAIC with E-DAIC heads
    ├── 2a: Freeze encoders, train fusion + E-DAIC heads only (5 epochs)
    ├── 2b: Unfreeze all, low LR (15 epochs)
    ├── 2c: Ablation: with vs without D-Vlog pretraining
    └── Stretch target: F1 ≥ 0.75 on dev (not a success/fail gate — see Reporting Protocol)

Stage 3 (if pretrain helps): Joint multi-task training
    └── Mixed batches, dataset-specific heads active per sample
```

### 4.3 Hyperparameters (RTX 5060)

| Parameter | Value |
|-----------|-------|
| Batch Size | 16 (effective 32 via gradient accumulation ×2) |
| Learning Rate | 1e-4 (baselines/Stage 1), 3e-5 (Stage 2) |
| Optimizer | AdamW (weight_decay=1e-3) |
| Scheduler | CosineAnnealingWarmRestarts (T₀=5) |
| Early Stopping | patience=7, monitor=val_F1 |
| Gradient Clipping | max_norm=1.0 |
| Mixed Precision | torch.amp (fp16) |
| Window / Stride | 30s / 15s (baseline default; **ablate 6s/9s/15s/30s** on dev — ref repo uses 9s) |
| Seeds | Fixed per experiment, logged in config |

### 4.4 Evaluation Suite

**Metrics tracked per experiment**:

| Metric | Purpose |
|--------|---------|
| Macro F1 | Reveals minority-class failures |
| Weighted F1 | Overall performance |
| AUROC | Threshold-independent discrimination |
| PR-AUC | Esp. important for imbalanced E-DAIC |
| Expected Calibration Error (ECE) | Probability calibration |
| Per-domain results | E-DAIC vs D-Vlog separately |

**Slice Metrics** (E-DAIC):

- By gender (male / female)
- By age bucket (18-30 / 31-45 / 46-60 / 61+)
- By session type (WoZ / AI-controlled)
- By PHQ severity band

**Evaluation Artifacts** (saved per run):

- Confusion matrices
- ROC / PR curves
- Calibration curves
- Per-subject prediction CSVs
- Training loss / metric curves

### 4.5 Subject-Level Integrity

```
RULE: No window from subject X in train may appear in dev or test.
VALIDATION: assert len(set(train_subjects) & set(test_subjects)) == 0
```

### 4.6 Bias Prevention

> **Interviewer Prompt Bias**: The model must NOT learn from Ellie's question patterns.
> - Use only participant visual + acoustic features (no text until speaker identification is solved)
> - E-DAIC test split is domain-shifted (AI vs WoZ) — report this in all evaluation
> - Validate attention maps focus on participant behaviour
> - Track slice metrics by gender, age, session type to detect demographic bias

### 4.7 Primary Reporting Protocol

**Subject-level metrics are the primary reported result.** Window-level metrics are secondary.

| Rule | Detail |
|------|--------|
| Primary metrics | Subject/session-level (aggregate windows → one prediction per subject) |
| Secondary metrics | Window-level (for debugging and analysis only) |
| Model selection | Dev-only. All threshold tuning, calibration fitting, aggregation choice on train/dev |
| Final test | Touched **exactly once** for final reporting. No test-set threshold tuning, calibration fitting, or architectural decisions |
| Multi-seed | Minimum 3–5 seeds for baselines + main models on dev. Report mean ± std |
| Confidence intervals | Subject-level bootstrap intervals for headline metrics |
| Cross-dataset reporting | E-DAIC and D-Vlog results side by side, never merged into one headline number |
| Success criteria | Reproducible improvement over strong baselines + stable across seeds + calibrated outputs — **not** a single F1 threshold |

### 4.8 Window-to-Subject Aggregation

All windows for a subject → single subject-level prediction. Compare on dev:

| Method | Description |
|--------|-------------|
| Mean pooling | Average window probabilities |
| Top-k pooling | Average top-k most confident windows |
| Attention pooling | Learned attention weights over windows |
| MIL-style | Multiple Instance Learning aggregation |

Choose on dev only. Report which method is used in final results.

### 4.9 Quality-Aware Window Filtering

Each window carries a quality score that flows through the full pipeline:

| Signal | Metric |
|--------|--------|
| Face tracking validity | Ratio of frames with OpenFace confidence ≥ 0.5 |
| Voiced-frame ratio | Ratio of frames with detected speech (for audio) |
| Minimum valid ratio | ≥ 50% valid frames required to keep window |
| Leading/trailing trim | Remove invalid tracking at session start/end |
| Transcript token count | If text is used: minimum token count per window |

Quality scores available to model (via modality gates), sampler (weight high-quality windows), and evaluator (stratify by quality).

### 4.10 Abstention-Aware Evaluation

Since the live system has a "not enough evidence" state, offline validation should also measure:

| Metric | Description |
|--------|-------------|
| Coverage | Fraction of subjects the model makes a prediction for |
| Risk-coverage tradeoff | Performance at different abstention thresholds |
| Abstention rate | Fraction of windows/subjects where model abstains |

### 4.11 Cross-Dataset Label Caveat

E-DAIC and D-Vlog do **not** share identical label semantics:

| Dataset | Label basis | Context |
|---------|-------------|---------|
| E-DAIC | PHQ-8 clinical screening score | Controlled clinical interview |
| D-Vlog | Behavior/content-derived from public self-disclosure | YouTube vlogs |

These are related but non-identical supervision sources. Joint training is **transfer learning across adjacent targets**, not pure label unification. Code and reports must treat them accordingly.

### 4.12 Error Taxonomy Review

After each major experiment, conduct a structured error review:

- Review false positives, false negatives, low-confidence abstentions
- Check domain-specific failures (E-DAIC WoZ vs AI, D-Vlog short vs long)
- Log findings in experiment report
- Feed findings back into data quality flags and model design

> Learning more from 20 reviewed bad cases than from chasing one more F1 point.

---

## Phase 5: Real-Time Inference

### 5.1 Honest Capability Statement

The live system is a **prototype depression-risk inference** system. It performs:

- Real-time face mesh visualization (MediaPipe Face Mesh)
- Real-time upper-body pose overlay (MediaPipe Pose) — **visual telemetry only, not a trained depression signal**
- Rolling behavioural signal charts (head movement, facial expression dynamics)
- Model confidence with uncertainty indicator
- **Visible disclaimer**: "This is a behavioral screening support tool, not a clinical diagnosis"

> **No hand-crafted AU mapping**: Approximate AU visualization from MediaPipe is acceptable for UI overlays. But MediaPipe features must **not** be fed into the depression model through a manually invented AU conversion layer. If MediaPipe is used for inference, it must feed a properly trained student/bridge model.

### 5.2 Feature Space Bridge (Required for Accurate Live Inference)

The offline model trains on OpenFace AUs + eGeMAPS features. The live system captures MediaPipe Face Mesh landmarks. These are **different feature spaces**.

**Bridge Pipeline** (Stage 2, after raw video download):

```
Teacher: best offline model trained on OpenFace/eGeMAPS features
Bridge:  run both OpenFace & MediaPipe on same raw videos
Student: train projection: MediaPipe features → teacher's embedding space
```

Until the bridge is built, the live system should:

1. Display raw behavioural signals (valid and interesting on their own)
2. Run a **prototype** model with honest confidence bounds
3. Show "model currently trained on clinical features; live projection is approximate"

### 5.3 Asynchronous Pipeline Design

Capture, rendering, feature extraction, and inference must be **decoupled** so the UI stays smooth even when inference is slower than video rendering:

```
Thread/Process 1: Video capture + rendering (60 FPS target)
Thread/Process 2: Feature extraction (face mesh, pose) → buffer
Thread/Process 3: Audio capture + VAD → buffer
Thread/Process 4: Model inference on assembled windows (runs on cadence, not every frame)
```

"Real-time" means: streaming capture works smoothly, overlays update fluidly, risk estimate updates on stable cadence, and latency is low enough for live demo. It does **not** require frame-by-frame depression predictions.

### 5.4 Streaming Audio Stack

The audio path requires more than `librosa` alone:

| Component | Purpose |
|-----------|---------|
| Voice Activity Detection (VAD) | Identify speech vs silence segments |
| Buffering / window assembly | Accumulate audio into analysis windows |
| Denoising / noise-robust features | Handle real-world mic environments |
| Dropped-chunk handling | Graceful fallback when audio stream drops |
| Noisy-environment fallback | Reduce weight of acoustic modality when SNR is low |

### 5.5 Components

| Component | Technology | Description |
|-----------|-----------|-------------|
| Face tracking | MediaPipe Face Mesh | 468 landmarks → visual overlay + bridge model input |
| Body tracking | MediaPipe Pose | 33 landmarks → visual overlay only (not model input) |
| Audio | VAD + librosa + streaming buffer | Real-time acoustic feature extraction from mic |
| Server | Flask + WebSocket | Feature extraction, inference, prediction streaming |
| Dashboard | HTML/CSS/JS | Live overlay, risk gauge, signal charts, disclaimers |

### 5.6 Safety Features

| Feature | Implementation |
|---------|---------------|
| Minimum evidence | ≥ 3 consecutive windows (90s) before any risk indication |
| Confidence threshold | Display prediction only when confidence > threshold |
| Temporal smoothing | EMA of predictions, no single-window alarms |
| **Cooldown / hysteresis** | Require persistence for both escalation AND de-escalation of risk state. Prevents flickering between states |
| Graceful degradation | Low light / occlusion / multiple faces → "tracking quality insufficient" |
| Privacy | Log derived features only, no raw video unless explicit consent |
| Low-confidence mode | "Insufficient evidence" output when tracking quality is poor |

### 5.7 Demo Failure-Case Test Set

Before claiming the live interface is robust, test explicitly against:

- Low light conditions
- Glasses / sunglasses
- Head turns / partial face
- No speech / silence
- Background noise
- Multiple people entering frame
- Missing microphone

Document results for each case.

---

## Phase 6: Feature-Space Bridge (D-Vlog Raw Videos ✅ Available)

**Video Download Status**: ✅ COMPLETE

- 773/961 videos downloaded to `D:\DL-Datasets\dvlog_videos\` (~54.6 GB)
- 188 videos unavailable on YouTube — no further download attempts
- Coverage: ~80% per fold, sufficient for bridge training

**Bridge Pipeline** (uses downloaded raw videos):

1. Run OpenFace 2.0 on 773 downloaded videos → extract AUs, pose, gaze
2. Run MediaPipe Face Mesh on same 773 videos → extract 468 landmarks
3. Align OpenFace and MediaPipe frame-by-frame on same video
4. Train projection model: MediaPipe features → OpenFace feature space
5. Validate bridge accuracy against teacher model predictions
6. Output: `D:\DL-Datasets\dvlog_bridge_features\` (paired feature files)

> **Note**: Bridge only covers 773/961 subjects. The remaining 188 subjects still contribute to model training via pre-extracted features — they just can't participate in bridge training.

---

## Phase 7: Text Modality (Future — Requires Corpus Audit + Speaker Identification)

Text is **not abandoned**, just deferred until safe:

1. **Full 275-session transcript corpus audit** (prerequisite before any text work):
   - Percentage of very short utterances
   - Percentage of likely interviewer/setup utterances
   - Percentage of scrubbed or anomalous entries
   - Token count per session
   - This may show text is salvageable with rule-based cleaning, even without diarization
2. Build speaker identification for E-DAIC transcripts (rule-based: Ellie questions follow interviewer patterns)
3. Validate speaker identification accuracy manually on 10+ transcripts
4. If ≥95% speaker accuracy: add Sentence-BERT text encoder as 3rd modality
5. Ablate: does text improve dev F1 over bimodal-only?
6. Only keep text if ablation proves value
7. If text enters: it must **not** be fused at equal weight — text is the most leakage-prone modality, so it enters behind strong cleaning + quality gating

---

## Phase 8: Body Movement Research (Second-Phase Track)

Body movement as a depression signal is a legitimate hypothesis, but requires its own validation:

1. Extract MediaPipe Pose features from 773 downloaded D-Vlog videos at `D:\DL-Datasets\dvlog_videos\`
2. Correlate extracted body features with depression labels
3. Train body-only baseline → does it predict above chance?
4. If yes: add body as auxiliary modality with ablation + bias checks + failure analysis
5. If no: body remains visual overlay only

Even if body features are extracted from D-Vlog videos, that still does not instantly validate body movement as a robust depression cue — it only creates an exploratory signal.

---

## Phase 9: Documentation & Responsible Deployment

### 9.1 Model Card (per checkpoint)

Every serious checkpoint gets a model-card-style summary:

- Training datasets used
- Modalities used
- Normalization protocol
- Evaluation split and results
- Known limitations
- Intended use
- Non-intended use

### 9.2 Project Honesty Section (README + Demo UI)

A dedicated section in the final README and visible in the demo UI:

- What the model is trained on
- What the live demo currently uses
- Which parts are validated
- Which parts are exploratory
- Known failure modes

> This makes the project look **more** mature, not less.

### 9.3 Privacy & Retention Policy

Even for the demo build, document:

- What is stored (derived features, predictions, experiment logs)
- What is NOT stored (raw webcam frames, unless explicit consent)
- How long logs remain
- How a user can disable logging

---

## File Structure

```
c:\Users\arjun\Desktop\DLP\              (internal drive — code + small files)
├── src/
│   ├── data/
│   │   ├── dataset_audit.py          # Data quality audit
│   │   ├── manifest_generator.py     # Manifest-first pipeline
│   │   ├── dvlog_dataset.py          # D-Vlog PyTorch Dataset
│   │   ├── edaic_extractor.py        # E-DAIC archive extraction
│   │   ├── edaic_preprocessor.py     # E-DAIC feature processing
│   │   ├── bridge_extractor.py       # OpenFace + MediaPipe paired extraction
│   │   ├── dataset.py                # Unified Dataset
│   │   └── augmentations.py          # Temporal augmentations
│   ├── model/
│   │   ├── encoders.py               # Visual & Acoustic encoders
│   │   ├── fusion.py                 # Cross-attention + modality gates + source norms
│   │   ├── aggregation.py            # Window-to-subject aggregation strategies
│   │   ├── classifier.py             # Multi-task model (3 heads)
│   │   └── losses.py                 # Focal loss + multi-task loss (dataset-masked)
│   ├── training/
│   │   ├── trainer.py                # Training loop + experiment logging
│   │   ├── evaluate.py               # Metrics, slicing, calibration, abstention
│   │   ├── baselines.py              # Unimodal baseline runners
│   │   └── config.py                 # Experiment configs (YAML-backed)
│   ├── inference/
│   │   ├── feature_extractor.py      # MediaPipe real-time
│   │   ├── audio_stream.py           # VAD + buffering + streaming audio pipeline
│   │   ├── server.py                 # Flask inference server (async pipeline)
│   │   └── dashboard/                # Web UI + disclaimers + honesty section
│   └── utils/
│       ├── logging_utils.py          # Structured JSON experiment logs
│       ├── visualization.py          # ROC, PR, calibration curves
│       ├── model_card.py             # Model card generator
│       └── seeds.py                  # Deterministic seeding
├── data/                             # D-Vlog pre-extracted features (existing, ~416 MB)
├── wwwedaic/                         # E-DAIC archives + labels (existing, ~105 GB)
├── models/                           # Checkpoints + model cards
├── configs/                          # Experiment config YAMLs
├── History.md
├── requirements.txt
└── README.md                         # Includes project honesty + privacy sections

D:\DL-Datasets\                        (external SSD — large data processing)
├── dvlog_videos/                     # ✅ 773 raw D-Vlog MP4s (~54.6 GB)
├── dvlog_bridge_features/            # Paired OpenFace + MediaPipe features (future)
├── processed/                        # Pre-normalized arrays + normalized caches
│   ├── edaic/                        # Extracted E-DAIC features + windows
│   ├── dvlog/                        # Windowed D-Vlog features
│   └── manifest.jsonl                # Master manifest
└── results/                          # Per-run evaluation artifacts + error taxonomy
```

---

## Execution Order

| Step | Phase | What | Depends On | Status |
|------|-------|------|------------|--------|
| 1 | Audit | `dataset_audit.py` + manifest generator for both datasets | — | ✅ DONE |
| 2 | Data | D-Vlog Dataset + DataLoader (float32 cast, length alignment, normalization) | Step 1 | ✅ DONE |
| 3 | Data | E-DAIC extraction to `D:\DL-Datasets\processed\edaic\` (**49** visual, 23 acoustic dims) | Step 1 | 🔄 Extraction running |
| 3b | Data | E-DAIC DataLoader (resample, quality filter, windowing) | Step 3 | ⬜ Next |
| 4 | Baselines | Unimodal baselines: acoustic-only + visual-only, multi-seed | Steps 2+3b | ⬜ Queued |
| 5 | Model | Multi-task model: encoders + source norms + gates + fusion + 3 heads + aggregation | Step 4 | ⬜ Queued |
| 6 | Train | Bimodal D-Vlog training (must beat unimodal baselines) | Step 5 | ⬜ Queued |
| 7 | Train | Fine-tune on E-DAIC (E-DAIC heads only) + ablate pretraining + window sizes | Steps 5+6 | ⬜ Queued |
| 8 | Eval | Full evaluation: subject-level primary, slice metrics, calibration, abstention, error taxonomy | Step 7 | ⬜ Queued |
| 9 | Inference | Live UI with async pipeline, streaming audio, honest boundaries, failure-case testing | Step 7 | ⬜ Queued |
| 10 | Bridge | Feature-space bridge: OpenFace + MediaPipe on 773 D-Vlog videos → projection model | Step 7 | ✅ Videos Ready |
| 11 | Text | Transcript corpus audit → speaker ID → text modality ablation (gated fusion) | Step 7 | Future |
| 12 | Body | Body-movement research: MediaPipe Pose on 773 D-Vlog videos → ablation | Step 10 | Future |
| 13 | Docs | Model cards, project honesty section, privacy policy | Step 8 | Future |

---

## Project Progress Tracker

### Step 1: Dataset Audit + Manifest Generator — ✅ DONE
- [x] Built `src/data/dataset_audit.py` — D-Vlog audit
- [x] Built `src/data/dataset_audit.py` — E-DAIC audit
- [x] Ran audit, generated `data/audit_report.json`
  - D-Vlog: 961/961 features, 0 NaN/Inf, 0 corrupt, 14 length mismatches
  - E-DAIC: 275/275 archives matched labels, 14 files each, 33 columns verified
- [x] Built `src/data/manifest_generator.py`
- [x] Generated `D:\DL-Datasets\processed\manifest.jsonl` (1236 entries: 961 D-Vlog + 275 E-DAIC)

### Step 2: D-Vlog DataLoader — ✅ DONE
- [x] Built `src/data/dvlog_dataset.py`
- [x] Validated float32 cast, min-truncation alignment, windowing
  - Train: 647 subjects, 25,738 windows (dep=375, norm=272)
  - Valid: 102 subjects, 3,746 windows (dep=57, norm=45)
  - Test: 212 subjects, 8,139 windows (dep=123, norm=89)
  - Shapes: visual (30,136), acoustic (30,25), dtype float32
- [x] Verified train/valid/test split integrity — 0 subject overlap

### Step 3: E-DAIC Extraction + Preprocessing — 🔄 IN PROGRESS
- [x] Built `src/data/edaic_extractor.py`
- [x] Started extraction of 275 archives (visual: 49 dims, acoustic: 23 dims)
- [ ] Verify extraction complete → check `D:\DL-Datasets\processed\edaic\extraction_report.json`
  - If incomplete, re-run: `python -m src.data.edaic_extractor` (incremental)
- [ ] Build `src/data/edaic_dataset.py` (PyTorch Dataset)
  - Resample to 1 FPS, quality filter, windowing
  - Splits from `wwwedaic/labels/detailed_lables.csv`
- [ ] Re-run manifest generator to update E-DAIC entries with extracted paths

### Step 4: Unimodal Baselines — ⬜ QUEUED
- [ ] Build `src/models/encoders.py` (GRU/Transformer encoders)
- [ ] Build `src/training/trainer.py` (training loop + metrics)
- [ ] Acoustic-only D-Vlog baseline
- [ ] Visual-only D-Vlog baseline
- [ ] Acoustic-only E-DAIC baseline
- [ ] Visual-only E-DAIC baseline
- [ ] Record metrics — **benchmarks to beat**: D-Vlog F1=0.78, E-DAIC F1=0.56

### Step 5+: Model, Training, Evaluation, Inference — ⬜ QUEUED
- [ ] (blocked on Step 4)

---

## Reference Repo Comparison

**Paper**: "Reading Between the Frames" (Gimeno-Gómez et al., ECIR 2024)
**Repo**: `github.com/cosmaadrian/multimodal-depression-from-video`
**License**: CC BY-NC-ND 4.0 (reference only, no code reuse)

| Aspect | Their Approach | Our Approach | Advantage |
|--------|---------------|-------------|----------|
| Architecture | Perceiver (cross-attn + self-attn) | Source-aware multimodal transformer (CNN/GRU + cross-attn fusion) | Ours has stronger inductive biases for small data |
| Modalities | Audio, EmoNet face, landmarks, gaze, blink | Acoustic, visual (landmarks/AUs), quality-gated | Comparable; theirs has more modalities |
| Task heads | Single binary classification | **Multi-task: 3 heads (2 binary + PHQ regression)** | Ours exploits PHQ continuous labels |
| Window size | 9s (D-Vlog), 6s (E-DAIC) | 30s default, **ablate 6/9/15/30s** | We'll test their optimal sizes |
| Quality gating | None | **Quality-aware sigmoid gates per modality** | Ours handles noisy streams |
| Abstention | None | **Abstention-aware evaluation** | Ours handles low-confidence |
| PHQ regression | None | **SmoothL1 regression head** | Ours provides severity estimation |
| D-Vlog evaluation | Subject-level | **Subject-level + bootstrap CIs** | Ours has stronger statistical reporting |
| Feature-space bridge | None | **MediaPipe→OpenFace projection** | Enables live inference |
| E-DAIC visual dims | 49 | **49** (matched) | Same |
| Loss | XE with class weights | **Focal loss + multi-task masking** | Ours targets imbalance harder |
| Parameters | 8.4–15M | ~3M | Ours fits 8GB VRAM comfortably |
