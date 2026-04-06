# Vision V3 Milestone Plan

## Summary

This milestone starts from the locked winners and the new project rule:

- we are building a **vision-first depression model**
- audio is an **auxiliary modality**
- if a vision or fusion model is within `0.05` test macro F1 of the best acoustic model, we prefer the vision/fusion path

Locked references:

- `D-Vlog` benchmark winner: `dvlog_acoustic`
- `D-Vlog` preferred direction: `dvlog_fusion_v2`
- `E-DAIC` benchmark winner and preferred direction: `edaic_bimodal (Fusion V1)`

`Vision V3` is therefore **not** “Fusion V2 but larger.”
It is a new milestone with a different center of gravity:

- stronger pretrained **visual** features on D-Vlog
- better temporal preservation for **visual** E-DAIC streams
- smaller, safer fusion on top
- audio added as support, not as the main path

---

## Main Goal

Build a vision-first multimodal stack that can become the preferred architecture on both datasets, while staying honest about benchmark winners.

Success looks like:

- `D-Vlog`: a visual/fusion model that stays within `0.05` of `dvlog_acoustic`, ideally beats it
- `E-DAIC`: a visual/fusion model that beats locked `Fusion V1` test macro F1 `0.5563`

---

## Why This Is The Right Next Milestone

The previous milestone taught us:

- `Fusion V2` did not fail because fusion is useless
- it failed because the current visual representation is still too weak relative to the acoustic anchor
- on `D-Vlog`, corrected `Fusion V2` already got close enough to matter under the project’s vision-first rule
- on `E-DAIC`, the likely structural issue is not just model choice, but aggressive temporal compression and poor transfer

So the next gain should come from **better visual priors**, not from more acoustic tuning.

---

## Pretrained Additions To Prioritize

Only add pretrained components that directly strengthen the visual side.

### Tier 1: Definitely Worth Adding

1. **Face emotion / affect embeddings for D-Vlog**
   - Purpose: stronger face representation than raw 136-d landmarks alone
   - Best fit: EmoNet-style or equivalent affect-pretrained face embedding extractor
   - Why: this is the closest low-risk move toward the reference repo’s strongest visual stack

2. **Body pose landmarks for D-Vlog raw videos**
   - Purpose: capture posture, self-touch, slumping, movement energy, gesture sparsity
   - Best fit: MediaPipe Pose
   - Why: body cues are vision-first and complement face cues well

3. **Hand landmarks for D-Vlog raw videos**
   - Purpose: capture fidgeting, self-contact, gesture suppression, hand visibility patterns
   - Best fit: MediaPipe Hands
   - Why: cheap relative to full video encoders and useful for non-verbal behavior

4. **Gaze + blink features for D-Vlog raw videos**
   - Purpose: add eye-contact and fatigue-like visual cues
   - Best fit: gaze estimator + blink heuristic pipeline
   - Why: reference-style visual enrichment without shifting toward audio

### Tier 2: Worth Adding If Tier 1 Lands Cleanly

5. **Stronger pretrained frame embeddings**
   - Candidates: DINOv2, CLIP image encoder, facial expression backbone
   - Purpose: replace or augment weak hand-engineered visual streams
   - Why: potentially high upside, but heavier implementation and storage cost

6. **Higher-fidelity E-DAIC visual temporal handling**
   - This is not an external pretrained model, but it is equally important
   - Preserve richer temporal structure instead of collapsing everything to 1 Hz too early
   - Use the already available `cnn_resnet` + `pose_gaze_aus` streams more faithfully

### Not Worth Prioritizing First

- more acoustic descriptors
- acoustic self-supervised models as the next main step
- another complex latent-fusion redesign before visual features improve

---

## Dataset Strategy

### D-Vlog: Main V3 Target

Why:

- raw videos are already available for `773/961` subjects
- corrected `Fusion V2` is already close enough to matter
- this dataset is the best place to make the system genuinely vision-first

Vision V3 D-Vlog bundle:

- `visual_landmarks_existing`
- `visual_face_affect_embed`
- `visual_body_pose`
- `visual_hand_pose`
- `visual_gaze_blink`
- `audio_aux_existing`

### E-DAIC: Visual-Transfer Repair Track

Why:

- E-DAIC is still important, but a raw-video-heavy milestone is less natural here
- the bigger likely gains are from **temporal handling** and **simpler fusion**

Vision V3 E-DAIC bundle:

- `video_pose_gaze_aus`
- `video_cnn_resnet`
- optional `audio_egemaps_aux`
- optional `audio_mfcc_aux`

Key change:

- stop assuming 1 Hz visual compression is good enough for the main V3 path

---

## Model Strategy

Use a **visual-core architecture** with light auxiliary fusion.

### D-Vlog Model

- visual branch is the primary trunk
- audio branch is a small auxiliary encoder
- fusion happens late and cheaply
- final output prioritizes visual or fused heads, not acoustic dominance

Recommended form:

- one visual-composite encoder over concatenated or tokenized visual modalities
- one small audio auxiliary encoder
- subject-level transformer or attention pooling
- 3 heads:
  - `visual_logit`
  - `fused_logit`
  - `audio_aux_logit`
- final decision:
  - `mixture = 0.60 * visual + 0.30 * fused + 0.10 * audio_aux` as the default starting rule
  - learnable mixture only after the fixed-prior version is benchmarked

### E-DAIC Model

Do **not** start with another big fusion stack.

Start with:

- visual-only strong baseline over `pose_gaze_aus + cnn_resnet`
- then simple late fusion:
  - `p = 0.70 * visual + 0.30 * audio`

Why:

- E-DAIC punishes over-parameterized fusion
- we need to know whether stronger visual signal alone is enough before adding more learned fusion

---

## Implementation Surfaces

### New Data Surfaces

- `src/data/dvlog_video_extractor.py`
  - extracts body pose, hand pose, gaze, blink, and face-affect embeddings from raw videos

- `src/data/dvlog_video_dataset.py`
  - vision-first D-Vlog dataset using extracted visual modalities plus existing auxiliary audio

- extend `src/data/fusion_v2_datasets.py`
  - or create `src/data/vision_v3_datasets.py`
  - support named vision-first modality bundles

- extend `src/data/edaic_extractor.py`
  - only if needed for better temporal caching or chunked visual views

### New Model Surfaces

- `src/model/vision_v3.py`
  - visual-core multimodal model
  - lighter than V2 on the fusion side
  - fixed-prior and learnable-prior variants

### New Training Surfaces

- `src/training/vision_v3.py`
  - train/eval loop
  - vision-priority mixture handling
  - visual-only and visual+audio auxiliary modes

### New Config Surfaces

- `configs/vision_v3_dvlog_smoke.json`
- `configs/vision_v3_dvlog_benchmark.json`
- `configs/vision_v3_dvlog_showdown.json`
- `configs/vision_v3_edaic_smoke.json`
- `configs/vision_v3_edaic_benchmark.json`
- `configs/vision_v3_edaic_showdown.json`

### Result Roots

- `results/benchmark_quality/vision_v3_dvlog_smoke/`
- `results/benchmark_quality/vision_v3_dvlog_benchmark/`
- `results/benchmark_quality/vision_v3_dvlog_showdown/`
- `results/benchmark_quality/vision_v3_edaic_smoke/`
- `results/benchmark_quality/vision_v3_edaic_benchmark/`
- `results/benchmark_quality/vision_v3_edaic_showdown/`

---

## Benchmark Plan

### Phase 1: D-Vlog Visual Feature Build

Goal:

- create the new visual feature set for the `773` available raw videos

What to benchmark first:

- `existing_visual_only`
- `existing_visual + affect`
- `existing_visual + affect + body`
- `existing_visual + affect + body + hands + gaze/blink`
- then add `audio_aux`

Selection metric:

- primary: `macro_f1`
- track also:
  - `binary_f1`
  - `precision`
  - `recall`
  - visual-only vs fused gap

### Phase 2: D-Vlog Vision V3 Benchmark

Goal:

- choose the strongest vision-first D-Vlog model

Promotion logic:

- if best vision/fusion model is within `0.05` of acoustic, it becomes the preferred D-Vlog architecture
- if it beats acoustic, it becomes both benchmark and preferred winner

### Phase 3: E-DAIC Visual-Core Repair

Goal:

- test whether stronger visual use plus simpler fusion can beat locked `Fusion V1`

Order:

1. stronger visual-only baseline
2. fixed-weight late fusion
3. only then consider learnable fusion

### Phase 4: Locked Showdowns

D-Vlog showdown against:

- `dvlog_acoustic`
- `dvlog_bimodal`
- `dvlog_fusion_v2`
- selected `vision_v3_dvlog`

E-DAIC showdown against:

- `edaic_visual`
- `edaic_acoustic`
- `edaic_bimodal`
- selected `vision_v3_edaic`

---

## Recommended Run Order

1. Build D-Vlog raw-video visual extractors
2. Run D-Vlog visual-feature smoke
3. Run D-Vlog vision-first benchmark
4. Lock preferred D-Vlog Vision V3 candidate
5. Build E-DAIC visual-core benchmark path
6. Run E-DAIC visual-only then late-fusion benchmark
7. Lock E-DAIC candidate only if it beats `edaic_bimodal`

---

## Commands To Run Later

Not for now. These should only be run after the code/config implementation is complete.

Expected sequence:

```powershell
python -m src.data.dvlog_video_extractor
python -m src.training.benchmark_suite --suite configs/vision_v3_dvlog_smoke.json --stage dev
python -m src.training.benchmark_suite --suite configs/vision_v3_dvlog_benchmark.json --stage dev
python -m src.training.benchmark_suite --suite configs/vision_v3_dvlog_showdown.json --stage finalize
python -m src.training.benchmark_suite --suite configs/vision_v3_edaic_smoke.json --stage dev
python -m src.training.benchmark_suite --suite configs/vision_v3_edaic_benchmark.json --stage dev
python -m src.training.benchmark_suite --suite configs/vision_v3_edaic_showdown.json --stage finalize
```

---

## Acceptance Criteria

### D-Vlog

- preferred direction must be visual or fusion
- benchmark winner may remain acoustic, but only if the visual/fusion gap is > `0.05`
- otherwise the visual/fusion model becomes the preferred production/research path

### E-DAIC

- new candidate must beat locked `Fusion V1` test macro F1 `0.5563`
- if it does not, `edaic_bimodal` remains the winner

---

## Final Recommendation

Yes, we should add more pretrained things.

But we should add:

- pretrained **visual** things first
- cheap auxiliary audio second
- more acoustic complexity only if it directly improves the vision-first model

The best next milestone is therefore:

- **Vision V3**
- **D-Vlog-first**
- **pretrained visual enrichment**
- **audio as support**
