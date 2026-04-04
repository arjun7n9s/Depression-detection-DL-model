# Bimodal Benchmark Report

- Generated: 2026-04-04 05:49:42
- Suite: bimodal_benchmark_smoke

## Final Baselines

| track | dataset | modality | selected_aggregation | window_size | loss_name | balanced_sampling | use_pos_weight | hidden_dim | num_layers | normalization_source | dev_macro_f1_mean | dev_macro_f1_std | test_macro_f1_mean | test_macro_f1_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| edaic_bimodal | edaic | both | mean | 9 | bce | True | False | 64 | 1 | train | 0.5492260061919505 | 0.0 | 0.4490161001788909 | 0.0 |

## Modality Ranking

### edaic
- both: test macro F1 0.4490 +/- 0.0000

## Conclusions

- Compare each bimodal result against the finalized unimodal baseline for the same dataset before increasing architecture complexity.
- This report is the source of truth for benchmark-quality bimodal performance.