# MindSense

> Multimodal depression-risk estimation research project using facial and acoustic signals from E-DAIC and D-Vlog.

![Status](https://img.shields.io/badge/status-active%20research-0f766e)
![Public Snapshot](https://img.shields.io/badge/public%20snapshot-curated%20stable-1d4ed8)
![Unimodal](https://img.shields.io/badge/unimodal%20benchmark-finalized-15803d)
![Bimodal](https://img.shields.io/badge/bimodal%20v1-dev--stage%20complete-f59e0b)
![Next Step](https://img.shields.io/badge/next-bridge%20%2B%20inference%20prep-7c3aed)
![License](https://img.shields.io/badge/license-MIT-eab308)

This repo is the public research log for the project as it actually exists today. It includes the verified data-foundation layer, finalized unimodal benchmark results, implemented bimodal `Fusion V1` code and curated benchmark artifacts, plus the running progress log in `team_progress`.

## At A Glance

| Area | Exact status |
|---|---|
| Dataset audit | Complete and verified for D-Vlog and E-DAIC |
| Manifest + extraction tracking | Complete, including explicit partial-recovery handling |
| D-Vlog loader | Complete and verified |
| E-DAIC loader | Complete and verified |
| Milestone unimodal baselines | Complete |
| Benchmark-quality unimodal search | Complete |
| Locked final unimodal benchmark runs | Complete |
| Bimodal acoustic+visual `Fusion V1` | Implemented and locked benchmark complete |
| Bimodal `Fusion V2` | Implemented, benchmarked, not promoted |
| D-Vlog `Vision V3` | Implemented, benchmarked, promoted |
| Strategic model lock | Complete and downstream-ready |
| Live inference / dashboard | Prototype scaffolding started |

## What Is Public In This Repo

- Source code for dataset auditing, manifest generation, unimodal and bimodal loaders, encoders, aggregation, training, and evaluation.
- Benchmark configs under `configs/`.
- Curated experiment artifacts under:
  - `results/baselines/`
  - `results/benchmark_quality/unimodal_benchmark_v1/`
  - `results/benchmark_quality/bimodal_benchmark_smoke/`
  - `results/benchmark_quality/bimodal_benchmark_v1/`
- The running research log in `team_progress`.

Intentionally not published:
- Raw datasets, extracted heavy arrays, downloaded videos, `_reference_repo/`, installers, live logs, and other bulky or scratch-only files.

## Foundation Snapshot

The current data layer is not a mockup. It was verified end to end before model work was expanded.

| Checkpoint | Recorded result |
|---|---|
| Manifest coverage | `1236` total entries |
| E-DAIC manifest entries | `275` |
| E-DAIC extraction state | `274` complete, `1` partial (`383_P`, acoustic-only) |
| D-Vlog loader verification | train `647` subjects / `25738` windows, valid `102` / `3746`, test `212` / `8139` |
| E-DAIC loader verification | visual train `162` subjects / `10369` windows, acoustic train `163` / `10499` windows |

## Final Unimodal Benchmark Snapshot

These are the locked 5-seed benchmark results from `results/benchmark_quality/unimodal_benchmark_v1/final/benchmark_summary.csv`. This is the current unimodal source of truth for the repo.

| Track | Window | Loss | Capacity | Dev macro F1 | Test macro F1 |
|---|---|---|---|---:|---:|
| `dvlog_acoustic` | `9s` | `bce_balanced` | `hidden128_layers1` | `0.6680 +/- 0.0415` | `0.6630 +/- 0.0100` |
| `dvlog_visual` | `9s` | `bce_balanced` | `hidden64_layers1` | `0.6028 +/- 0.0189` | `0.5943 +/- 0.0412` |
| `edaic_acoustic` | `9s` | `focal_balanced` | `hidden128_layers2` | `0.5922 +/- 0.0202` | `0.5134 +/- 0.0257` |
| `edaic_visual` | `30s` | `bce_balanced` | `hidden128_layers2` | `0.5220 +/- 0.0292` | `0.5355 +/- 0.0686` |

Key readouts:
- D-Vlog acoustic is the strongest finalized unimodal branch so far.
- E-DAIC acoustic was the strongest unimodal branch on dev, but E-DAIC visual slightly edged it on final test with higher variance.
- Final locked runs are complete, so the repo has moved beyond dev-stage-only reporting.

## Bimodal Fusion Snapshot

The first multimodal architecture is implemented as `Fusion V1`, and `Fusion V2` was later benchmarked in a corrected locked showdown. The source-of-truth multimodal winners are now frozen by dataset.

| Track | Selected window | Selected policy | Selected capacity | Frozen aggregation | Best dev macro F1 |
|---|---|---|---|---|---:|
| `edaic_bimodal` | `15s` | `focal_balanced` | `hidden128_layers2` | `attention` | `0.5352` |
| `dvlog_bimodal` | `9s` | `bce_balanced` | `hidden128_layers1` | `mean` | `0.7024` |

Evidence-backed interpretation:
- `Fusion V1` is a valid multimodal baseline.
- On `D-Vlog`, `Fusion V1` already beats both finalized unimodal dev baselines.
- On `E-DAIC`, `Fusion V1` does not yet beat the stronger acoustic unimodal baseline.

## Locked Winners

These are the current best verified models after the locked `Vision V3` D-Vlog showdown in `results/benchmark_quality/vision_v3_dvlog_showdown/final/benchmark_summary.csv` and the corrected synced `Fusion V2` showdown for E-DAIC.

| Dataset | Locked winner | Test macro F1 | Why it stays locked |
|---|---|---:|---|
| `D-Vlog` | `dvlog_vision_v3` | `0.6666 +/- 0.0310` | `Vision V3` slightly beat the locked acoustic benchmark on final test while matching the project’s vision-first goal |
| `E-DAIC` | `edaic_bimodal` (`Fusion V1`) | `0.5563 +/- 0.0342` | corrected `Fusion V2` still failed to transfer to test despite clearing the dev bar |

Evidence-backed interpretation:
- `Fusion V2` was a useful research step, but it is not promoted on either dataset.
- `Vision V3` is now the promoted D-Vlog direction and the current best verified D-Vlog model.
- `E-DAIC` currently favors the locked `Fusion V1` bimodal model.
- Any future multimodal push should be treated as a new research milestone rather than a continuation of the current promotion path.

## Preferred Architecture Direction

This project is **vision-first**. Audio is an auxiliary signal, not the primary identity of the system.

Decision rule:
- if a `vision` or `fusion` model is within about `0.05` test macro F1 of the best acoustic model, prefer the `vision` or `fusion` path for product/research direction
- acoustic-only remains the strict metric reference, not automatically the preferred architecture

Current interpretation under that rule:
- `D-Vlog` benchmark winner: `dvlog_vision_v3`
- `D-Vlog` preferred direction: `dvlog_vision_v3`
- `E-DAIC` benchmark winner: `edaic_bimodal` (`Fusion V1`)
- `E-DAIC` preferred direction: `edaic_bimodal` (`Fusion V1`)

## Public Repro Notes

- Large processed artifacts stay outside Git by design.
- External storage paths are environment-configurable through:
  - `MINDSENSE_EXTERNAL_DATA_ROOT`
  - `MINDSENSE_PROCESSED_ROOT`
  - `MINDSENSE_DVLOG_VIDEOS_DIR`
- Result paths written by the benchmark suite are repo-relative so curated public artifacts stay portable.

## Clinical And Privacy Note

This project is a behavioral screening support system for research use. It is not a clinical diagnostic instrument.

By default, the public repo avoids raw webcam, microphone, archive, and video dumps. The published snapshot focuses on code, configs, summaries, and curated evaluation artifacts rather than sensitive or heavyweight source media.

## Live Progress Of Project

This section is the repo's public heartbeat. It shows what has been finished, why those steps mattered, what evidence we have recorded, and what comes next.

```mermaid
flowchart LR
    A[Dataset audit] --> B[Manifest + recovery-aware extraction]
    B --> C[Verified dataset loaders]
    C --> D[Milestone unimodal baselines]
    D --> E[Locked unimodal benchmark pack]
    E --> F[First bimodal Fusion V1 benchmark]
    F --> G[Fusion V2 architecture milestone]
    G --> H[Inference / bridge / dashboard]
```

### Current State

| Workstream | Status | Verified artifact(s) | Why this step existed |
|---|---|---|---|
| Data audit | Complete | `data/audit_report.json` | We needed to prove the datasets were structurally usable before trusting any training result |
| Manifest generation | Complete | `manifest.jsonl` generation + extraction-state tracking | This creates one clean interface for training code instead of hand-written split logic scattered across scripts |
| E-DAIC extraction recovery | Complete for milestone use | `274` complete subjects and `1` partial subject | Recovery logic mattered because silent corruption would have produced misleading availability counts and unreliable loader behavior |
| D-Vlog loader | Complete | Verified subject and window counts across train/valid/test | This step converts raw feature files into repeatable model-ready windows |
| E-DAIC loader | Complete | Verified 1 Hz resampling, quality filtering, and modality-aware window creation | E-DAIC is messy enough that loader correctness directly affects every downstream metric |
| Locked unimodal benchmark | Complete | Final milestone report + benchmark summary CSV | This gives us the benchmark numbers we can cite publicly today |
| Bimodal `Fusion V1` implementation | Complete | Bimodal model code, runner support, configs, smoke outputs | This proved the repo supports real multimodal training rather than only unimodal benchmarking |
| Bimodal `Fusion V1` locked benchmark | Complete | `results/benchmark_quality/fusion_v1_locked/final/benchmark_summary.csv` | This froze the first multimodal reference honestly against final test metrics |
| Bimodal `Fusion V2` benchmark + synced showdown | Complete | `results/benchmark_quality/fusion_v2_showdown_synced/final/benchmark_summary.csv` | This tested the stronger fusion idea and showed it was not yet promotion-worthy |
| D-Vlog `Vision V3` benchmark + showdown | Complete | `results/benchmark_quality/vision_v3_dvlog_showdown/final/benchmark_summary.csv` | This is the first vision-first architecture to beat the locked D-Vlog acoustic benchmark on final test |
| Next architecture milestone | Replan from stronger position | Vision V3 D-Vlog locked showdown + E-DAIC gap | D-Vlog now has a promoted vision-first winner; the next question is how to extend or adapt that progress |

### Recorded Results We Can Stand Behind

| Category | Recorded value | Interpretation |
|---|---|---|
| E-DAIC recovery | `274` success + `1` partial | The data layer is usable, but still honest about the one remaining damaged archive |
| Manifest size | `1236` entries | Both datasets are now represented in one consistent subject-level format |
| Final D-Vlog acoustic benchmark | test macro F1 `0.6630 +/- 0.0100` | Current strongest finalized unimodal result in the repo |
| Final E-DAIC acoustic benchmark | dev macro F1 `0.5922 +/- 0.0202` | Strongest unimodal E-DAIC dev reference that multimodal models must beat |
| Final E-DAIC visual benchmark | test macro F1 `0.5355 +/- 0.0686` | Shows longer-context visual modeling can stay competitive on final test |
| `Fusion V1` D-Vlog bimodal | test macro F1 `0.6131 +/- 0.0111` | Useful multimodal baseline, but not the final D-Vlog winner |
| `Fusion V1` E-DAIC bimodal | test macro F1 `0.5563 +/- 0.0342` | Current best verified E-DAIC model in the repo |
| Corrected `Fusion V2` D-Vlog showdown | test macro F1 `0.6279 +/- 0.0142` | Better than `Fusion V1` on D-Vlog test, but still below acoustic-only |
| Corrected `Fusion V2` E-DAIC showdown | test macro F1 `0.4871 +/- 0.0658` | Cleared the dev bar but failed to hold up on final test |
| `Vision V3` D-Vlog showdown | test macro F1 `0.6666 +/- 0.0310` | First promoted vision-first D-Vlog winner; slightly above acoustic on final test |

### Why These Steps Matter

| Step | Why we did it before the next one | What it unlocked |
|---|---|---|
| Audit before training | Training on unknown corruption would make every score suspect | Trustworthy dataset assumptions |
| Manifest before loaders | We needed one shared subject-level contract across datasets | Cleaner training and benchmarking code |
| Recovery-aware extraction before E-DAIC modeling | Partial failure had to be explicit, not silently dropped | Honest availability counts and safer modality handling |
| Milestone baselines before benchmark search | We first needed to verify the stack could train, evaluate, save, and aggregate correctly | A working end-to-end baseline pipeline |
| Locked unimodal runs before multimodal promotion | Fusion should beat strong unimodal references, not weak placeholders | A real bar for multimodal progress |
| `Fusion V1` before `Fusion V2` | We needed a first multimodal baseline to reveal where simple fusion helps and where it fails | Evidence-driven architecture upgrades instead of guesswork |

### Exact Status Right Now

- The repo is past the foundation phase.
- The repo is past the toy-baseline phase.
- The repo already contains a finalized unimodal benchmark pack.
- The repo already contains an implemented bimodal baseline with a completed dev-stage benchmark.
- The current architectural decision is evidence-backed:
  - lock `dvlog_vision_v3` as the D-Vlog winner
  - lock `edaic_bimodal` (`Fusion V1`) as the E-DAIC winner
  - treat `Fusion V2` as an explored but not promoted branch
  - treat `Vision V3` as the first promoted D-Vlog vision-first architecture

### What We're Doing Next

1. Freeze the current winners as the milestone outcome: `dvlog_vision_v3` for D-Vlog and `edaic_bimodal` for E-DAIC.
2. Keep `Fusion V2` as benchmark evidence, not as a promoted architecture.
3. Build the downstream bridge and honest live-inference scaffolding around the locked winners.
4. Continue reporting dataset-specific winners rather than forcing one universal champion.

For a narrative log of the work as it happened, see `team_progress`.
