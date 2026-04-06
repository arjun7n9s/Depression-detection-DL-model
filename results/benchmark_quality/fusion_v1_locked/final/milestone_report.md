# Bimodal Benchmark Report

- Generated: 2026-04-04 22:44:17
- Suite: fusion_v1_locked

## Final Baselines

| track | dataset | modality | selected_aggregation | window_size | loss_name | balanced_sampling | use_pos_weight | hidden_dim | num_layers | normalization_source | dev_macro_f1_mean | dev_macro_f1_std | test_macro_f1_mean | test_macro_f1_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dvlog_bimodal | dvlog | both | mean | 9 | bce | True | False | 128 | 1 | train | 0.682041223843626 | 0.04041589290677001 | 0.6131049553931067 | 0.011077399854771491 |
| edaic_bimodal | edaic | both | attention | 15 | focal | True | False | 128 | 2 | train | 0.511208238707225 | 0.06700016255953072 | 0.5562721155655937 | 0.03417248478734069 |

## Modality Ranking

### dvlog
- both: test macro F1 0.6131 +/- 0.0111

### edaic
- both: test macro F1 0.5563 +/- 0.0342

## Conclusions

- Compare each multimodal result against the strongest unimodal baseline for the same dataset before promotion.
- This report is the source of truth for the finalized benchmark run.