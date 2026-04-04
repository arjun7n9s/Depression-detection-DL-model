# Unimodal Benchmark Report

- Generated: 2026-04-04 05:21:39
- Suite: unimodal_benchmark_v1

## Final Baselines

| track | dataset | modality | selected_aggregation | window_size | loss_name | balanced_sampling | use_pos_weight | hidden_dim | num_layers | normalization_source | dev_macro_f1_mean | dev_macro_f1_std | test_macro_f1_mean | test_macro_f1_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dvlog_acoustic | dvlog | acoustic | mean | 9 | bce | True | False | 128 | 1 | train | 0.6679754842277625 | 0.04150849620901992 | 0.6629850130213909 | 0.009991912647991987 |
| dvlog_visual | dvlog | visual | mean | 9 | bce | True | False | 64 | 1 | train | 0.6027700214367113 | 0.01887840509554474 | 0.5943358021499155 | 0.041245903901470746 |
| edaic_visual | edaic | visual | mean | 30 | bce | True | False | 128 | 2 | train | 0.5219894383520909 | 0.029248393721841993 | 0.5355378511959381 | 0.06863785195059706 |
| edaic_acoustic | edaic | acoustic | mean | 9 | focal | True | False | 128 | 2 | train | 0.5921515277996858 | 0.02021397121917693 | 0.5134208046333028 | 0.025737388640781973 |

## Modality Ranking

### dvlog
- acoustic: test macro F1 0.6630 +/- 0.0100
- visual: test macro F1 0.5943 +/- 0.0412

### edaic
- visual: test macro F1 0.5355 +/- 0.0686
- acoustic: test macro F1 0.5134 +/- 0.0257

## Conclusions

- D-Vlog normalization conclusion: final selected normalization sources = train.
- Bimodal work is justified only if the next model is required to beat the stronger unimodal result on each dataset.
- This report is the source of truth for benchmark-quality unimodal performance.