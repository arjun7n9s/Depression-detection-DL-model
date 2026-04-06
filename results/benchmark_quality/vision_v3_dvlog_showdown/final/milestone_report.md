# Benchmark Benchmark Report

- Generated: 2026-04-06 15:07:24
- Suite: vision_v3_dvlog_showdown

## Final Baselines

| track | dataset | modality | selected_aggregation | window_size | loss_name | balanced_sampling | use_pos_weight | hidden_dim | num_layers | normalization_source | dev_macro_f1_mean | dev_macro_f1_std | test_macro_f1_mean | test_macro_f1_std | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dvlog_vision_v3 | dvlog | visual_full_aux_audio | mean | 9 | vision_v3 | True | False | 128 | 0 | train | 0.661605979293358 | 0.03202493264055907 | 0.6666043397680148 | 0.03103698485988923 | vision_v3_dvlog_showdown |
| dvlog_acoustic | dvlog | acoustic | mean | 9 | bce | True | False | 128 | 1 | train | 0.6679754842277625 | 0.0415084962090199 | 0.6629850130213909 | 0.0099919126479919 | unimodal_benchmark_v1 |
| dvlog_acoustic | dvlog | acoustic | mean | 9 | bce | True | False | 128 | 1 | train | 0.6679754842277625 | 0.0415084962090199 | 0.6629850130213909 | 0.0099919126479919 | fusion_v2_showdown_synced |
| dvlog_fusion_v2 | dvlog | current_av | transformer | 9 | fusion_v2 | True | False | 128 | 4 | train | 0.6274260541625133 | 0.0210176696429593 | 0.6278906909800781 | 0.0142474467121806 | fusion_v2_showdown_synced |
| dvlog_bimodal | dvlog | both | mean | 9 | bce | True | False | 128 | 1 | train | 0.682041223843626 | 0.04041589290677 | 0.6131049553931067 | 0.0110773998547714 | fusion_v1_locked |
| dvlog_bimodal | dvlog | both | mean | 9 | bce | True | False | 128 | 1 | train | 0.682041223843626 | 0.04041589290677 | 0.6131049553931067 | 0.0110773998547714 | fusion_v2_showdown_synced |
| dvlog_visual | dvlog | visual | mean | 9 | bce | True | False | 64 | 1 | train | 0.6027700214367113 | 0.0188784050955447 | 0.5943358021499155 | 0.0412459039014707 | unimodal_benchmark_v1 |
| dvlog_visual | dvlog | visual | mean | 9 | bce | True | False | 64 | 1 | train | 0.6027700214367113 | 0.0188784050955447 | 0.5943358021499155 | 0.0412459039014707 | fusion_v2_showdown_synced |
| edaic_bimodal | edaic | both | attention | 15 | focal | True | False | 128 | 2 | train | 0.511208238707225 | 0.0670001625595307 | 0.5562721155655937 | 0.0341724847873406 | fusion_v1_locked |
| edaic_bimodal | edaic | both | attention | 15 | focal | True | False | 128 | 2 | train | 0.511208238707225 | 0.0670001625595307 | 0.5562721155655937 | 0.0341724847873406 | fusion_v2_showdown_synced |
| edaic_visual | edaic | visual | mean | 30 | bce | True | False | 128 | 2 | train | 0.5219894383520909 | 0.0292483937218419 | 0.5355378511959381 | 0.068637851950597 | unimodal_benchmark_v1 |
| edaic_visual | edaic | visual | mean | 30 | bce | True | False | 128 | 2 | train | 0.5219894383520909 | 0.0292483937218419 | 0.5355378511959381 | 0.068637851950597 | fusion_v2_showdown_synced |
| edaic_acoustic | edaic | acoustic | mean | 9 | focal | True | False | 128 | 2 | train | 0.5921515277996858 | 0.0202139712191769 | 0.5134208046333028 | 0.0257373886407819 | unimodal_benchmark_v1 |
| edaic_acoustic | edaic | acoustic | mean | 9 | focal | True | False | 128 | 2 | train | 0.5921515277996858 | 0.0202139712191769 | 0.5134208046333028 | 0.0257373886407819 | fusion_v2_showdown_synced |
| edaic_fusion_v2 | edaic | rich_av | transformer | 9 | fusion_v2 | True | False | 128 | 4 | train | 0.5959742967302861 | 0.052363806169051 | 0.4870739562241949 | 0.0657929435594468 | fusion_v2_showdown_synced |

## Modality Ranking

### dvlog
- visual_full_aux_audio [vision_v3_dvlog_showdown]: test macro F1 0.6666 +/- 0.0310
- acoustic [unimodal_benchmark_v1]: test macro F1 0.6630 +/- 0.0100
- acoustic [fusion_v2_showdown_synced]: test macro F1 0.6630 +/- 0.0100
- current_av [fusion_v2_showdown_synced]: test macro F1 0.6279 +/- 0.0142
- both [fusion_v1_locked]: test macro F1 0.6131 +/- 0.0111
- both [fusion_v2_showdown_synced]: test macro F1 0.6131 +/- 0.0111
- visual [unimodal_benchmark_v1]: test macro F1 0.5943 +/- 0.0412
- visual [fusion_v2_showdown_synced]: test macro F1 0.5943 +/- 0.0412

### edaic
- both [fusion_v1_locked]: test macro F1 0.5563 +/- 0.0342
- both [fusion_v2_showdown_synced]: test macro F1 0.5563 +/- 0.0342
- visual [unimodal_benchmark_v1]: test macro F1 0.5355 +/- 0.0686
- visual [fusion_v2_showdown_synced]: test macro F1 0.5355 +/- 0.0686
- acoustic [unimodal_benchmark_v1]: test macro F1 0.5134 +/- 0.0257
- acoustic [fusion_v2_showdown_synced]: test macro F1 0.5134 +/- 0.0257
- rich_av [fusion_v2_showdown_synced]: test macro F1 0.4871 +/- 0.0658

## Conclusions

- Vision V3 reports prioritize visual-first bundles and track mixture weights, quality slices, and teacher alignment.
- Promotion should follow the vision-first rule: prefer vision/fusion when the gap to acoustic stays within the accepted margin.
- This report is the source of truth for the finalized benchmark run.