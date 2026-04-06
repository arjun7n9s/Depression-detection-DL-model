# Making E-DAIC a Flagship Killer

## The Diagnosis — Why E-DAIC Is Hard

After digging through the actual data, error reviews, and per-seed metrics, there are **3 root causes** why every model struggles on E-DAIC test. These aren't vague guesses — they're backed by the numbers.

---

### Root Cause 1: Domain Shift (The Big One)

```
Train PIDs: 302-707   (Wizard-of-Oz interviews)
Dev PIDs:   300-713   (Wizard-of-Oz interviews)
Test PIDs:  600-718   (ALL autonomous AI interviews)
```

The test split is **entirely** AI-controlled sessions. The model trains/validates on human-operated interviews, then gets evaluated on a fundamentally different interaction style. This is confirmed by the suggestion_response.md accepting this as the single most critical evaluation caveat.

**Evidence:** Dev-test gap is enormous across every model:
| Model | Dev F1 | Test F1 | Gap |
|-------|--------|---------|-----|
| Unimodal acoustic | 0.592 | 0.513 | **-0.079** |
| Fusion V2 | 0.597 | 0.493 | **-0.104** |

More complex models = *bigger* gaps. This is classic domain-shift overfitting.

---

### Root Cause 2: Tiny, Imbalanced Dataset

```
Train: 163 subjects (37 depressed = 22.7%)
Dev:   56 subjects  (12 depressed = 21.4%)
Test:  56 subjects  (17 depressed = 30.4%)  ← higher prevalence!
```

- Only **37 positive training examples** for depression
- Test has **significantly higher depression prevalence** (30.4% vs 22.7% train)
- With 56 test subjects, each wrong prediction moves F1 by ~2 points
- Dev selection on 56 subjects is **extremely noisy** — a 12-subject positive class means ±1 subject flips F1 by ~5%

---

### Root Cause 3: Gender Confound Across Splits

```
Train: female=71(19dep), male=92(18dep)
Dev:   female=20(4dep),  male=35(8dep)
Test:  female=14(8dep),  male=42(9dep)
```

Test has **57% female depression rate** (8/14) vs train's **27%** (19/71). A model that leans on gender-correlated features will fail on test because the gender-depression relationship shifts dramatically.

---

## The Strategy — 7 Interventions, Ranked by Expected Impact

### 1. Train on Train+Dev, Validate with Leave-One-Subject-Out CV

**Impact: HIGH** | **Effort: MEDIUM**

This is the single highest-impact change. Right now we:
- Train on 163 subjects, validate on 56
- Select hyperparameters on 56 subjects with 12 positive examples (!!!!)
- This dev set is so small that F1 estimates have ±5% noise

**What to do instead:**
- Merge train+dev (219 subjects) 
- Run **5-fold stratified subject-level cross-validation** for hyperparameter selection
- Final model trains on all 219 subjects before test evaluation
- CV on 219 subjects gives **far more reliable** F1 estimates than holdout on 56

> [!IMPORTANT]  
> This alone probably accounts for 3-5% test F1 improvement. The current dev selection is picking configs that got lucky on 12 positive subjects.

---

### 2. PHQ Regression as Primary Task (Not Binary Classification)

**Impact: HIGH** | **Effort: LOW**

The E-DAIC data already has continuous PHQ-8 scores (0-24). Right now the model throws away this rich information and reduces everything to binary (depressed/not).

**What to do:**
- Train primary task as PHQ-8 score regression (MSE loss)
- At evaluation time, threshold the predicted score at PHQ ≥ 10 for binary classification
- Multi-task: `L = α·L_regression + β·L_binary`

**Why this works:**
- The model gets gradient signal from the difference between PHQ=2 and PHQ=7, not just "both are non-depressed"
- Regression is naturally more robust on small datasets — no sharp decision boundary to overfit
- Test PHQ distribution is shifted (mean 8.1 vs train 6.7), and a regression model adapts to this more gracefully than a binary classifier

---

### 3. Domain-Robust Training (Test-Time Robustness)

**Impact: HIGH** | **Effort: MEDIUM**

Since the test split is a known domain shift (WoZ → AI interviews), we should actively train for robustness:

**a) Feature-level augmentation:**
- Add Gaussian noise (σ=0.1–0.3) to input features during training
- Random temporal jitter: shift windows by ±1-2 seconds
- Random channel dropout: zero out 10-30% of feature dimensions per window
- This forces the model to not rely on WoZ-specific patterns

**b) Domain-invariant features:**
- The acoustic features (eGeMAPS) should be more domain-stable than visual (OpenFace)
- Consider **feature selection**: drop features with high variance between early PIDs (WoZ) and late PIDs (overlap zone 600-707 that appears in both train and test PID ranges)
- Compute feature-wise domain divergence between train PIDs < 600 and train PIDs ≥ 600, drop features with high divergence

**c) Mixup / CutMix on feature level:**
- Interpolate between subjects: `x_new = λ·x_i + (1-λ)·x_j`, `y_new = λ·y_i + (1-λ)·y_j`
- This smooths the decision boundary and regularizes heavily — crucial for 37 positive examples

---

### 4. Calibration-First Model Selection

**Impact: MEDIUM-HIGH** | **Effort: LOW**

Look at the actual test metrics from the Fusion V2 locked run:
```
seed_17: macro F1=0.411, AUROC=0.434, recall=0.0 (!)  ← predicts ALL negative
seed_7:  macro F1=0.514, AUROC=0.564, recall=0.706
```

The model is **wildly uncalibrated** — the threshold that works on dev completely fails on test because the prevalence shifts (21.4% → 30.4%).

**What to do:**
- Stop using 0.5 as the threshold
- Use Platt scaling on the dev logits
- Or better: select threshold on dev that maximizes macro F1, but also evaluate at multiple thresholds and pick the one that's most *stable* across validation folds
- Apply **temperature scaling** post-hoc before thresholding

---

### 5. Simpler Fusion That Actually Works

**Impact: MEDIUM** | **Effort: MEDIUM**

Fusion V2 with 328 lines, Perceiver latents, multi-task heads, quality gates, and transformer aggregators is massively over-parameterized for 163 training subjects. The evidence is clear: **simpler models transfer better**.

**What to do for E-DAIC specifically:**
- **Score-level fusion**: Train separate acoustic and visual models, average their predicted probabilities
- No shared parameters, no learnable gate, no cross-attention
- Each modality model is small and independently regularized
- The averaging itself acts as regularization

**Why this should beat everything tried so far:**
- Fusion V1 failed because the gate/concatenation added parameters that overfit
- Fusion V2 failed worse because it added even more parameters
- Score-level fusion adds **zero extra parameters** for fusion

---

### 6. Smart Window Aggregation

**Impact: MEDIUM** | **Effort: LOW**

The current subject-level prediction averages window probabilities. But not all windows carry equal signal.

**What to do:**
- **Top-K trimmed mean**: Average top K% of window probabilities (highest confidence windows)
- **Quality-weighted aggregation**: Weight each window's contribution by its `valid_ratio` quality score
- **Temporal attention**: Learn to attend to windows that matter, but with heavy regularization (single-head, weight decay)
- Use the **median** instead of mean — it's more robust to outlier windows

---

### 7. Leverage the PID Overlap Zone

**Impact: MEDIUM** | **Effort: MEDIUM**

Train PIDs go up to 707, test PIDs start at 600. **PIDs 600-707 appear in BOTH ranges**, meaning some training subjects were already from the AI-interview domain.

**What to do:**
- Identify which training subjects are in the 600-707 overlap zone
- Use these as a **domain bridge** — they're your best proxy for test behavior
- During training, give these subjects slightly higher sampling weight
- During validation (if not using CV), use these subjects as a secondary validation signal

> [!NOTE]
> This doesn't guarantee they're AI sessions (the split is by Participant, not interview type), but higher-numbered PIDs are more likely to be. Checking the actual interview metadata would confirm this.

---

## Recommended Execution Order

```
Phase 1 — Quick wins (1-2 days):
  ├── [2] PHQ regression as primary task
  ├── [4] Calibration-first threshold selection
  └── [6] Better window aggregation (top-k, median)

Phase 2 — Architecture change (2-3 days):
  ├── [1] Train+Dev with 5-fold CV
  ├── [5] Score-level late fusion (zero extra params)
  └── [3a] Feature augmentation (noise, dropout, mixup)

Phase 3 — Domain analysis (1-2 days):
  ├── [7] PID overlap zone analysis
  └── [3b] Domain-divergent feature selection
```

## Target

Current best E-DAIC test F1: **0.556** (Fusion V1)

With these changes, realistic target: **0.62–0.68 test macro F1**

The biggest single jump will come from **not selecting hyperparameters on 12 positive dev subjects** (intervention 1) and **using PHQ regression** (intervention 2). Those two alone should close most of the gap.

---

## Open Questions

> [!IMPORTANT]
> 1. Should we implement all 7 interventions as a new "Fusion V3" milestone, or start with the quick wins first and evaluate?
> 2. Do you want to keep the benchmark_suite framework for this, or build a separate experiment runner?
> 3. The train+dev CV approach means we lose the fixed dev set — are you okay with that tradeoff?
