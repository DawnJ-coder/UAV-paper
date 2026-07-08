一、预测结果诊断
----------------------------------------------------------------------------------------------------
144226 样本数: 38
  FALSE_LEAK: 19
  TRUE_LEAK: 19
模型准确率: 0.5789
模型误判数量: 16
TRUE_LEAK 平均概率: 0.5618
FALSE_LEAK 平均概率: 0.4443
TRUE-FALSE 平均概率差: 0.1175
概率灰区 [0.4, 0.6] 样本数: 10
冲突/SUSPECT样本数量: 10
预测明细: C:\Users\jiangxinru6\Desktop\wurenji\leak_v9_144226_diagnosis_results\v9_144226_predictions_sorted.csv
误判样本: C:\Users\jiangxinru6\Desktop\wurenji\leak_v9_144226_diagnosis_results\v9_144226_wrong_samples.csv
冲突/SUSPECT样本: C:\Users\jiangxinru6\Desktop\wurenji\leak_v9_144226_diagnosis_results\v9_144226_suspect_or_conflict_samples.csv
按center配对预测: C:\Users\jiangxinru6\Desktop\wurenji\leak_v9_144226_diagnosis_results\v9_144226_center_pair_predictions.csv

按 center 配对概率排序:
  TRUE 概率 > FALSE 概率 的 center 比例: 0.7368
  配对排序失败 center 数量: 5
  配对排序失败 center: 18, 17, 16, 15, 14

二、特征区分度诊断
----------------------------------------------------------------------------------------------------
[v7_feature]
对比表: C:\Users\jiangxinru6\Desktop\wurenji\leak_v9_144226_diagnosis_results\v9_144226_v7_feature_compare.csv
  区分力前10特征:
    best_direction_combined_score__time_robust_z: AUC=1.000, |d|=7.286, overlap=0.000, true_mean=-0.486357, false_mean=0.556104
    best_direction_combined_score: AUC=1.000, |d|=7.286, overlap=0.000, true_mean=0.770592, false_mean=0.852768
    direction_contrast__time_robust_z: AUC=1.000, |d|=5.402, overlap=0.000, true_mean=0.585012, false_mean=-0.367863
    direction_contrast: AUC=1.000, |d|=5.402, overlap=0.000, true_mean=1.27288, false_mean=1.07355
    spec_slope__time_robust_z: AUC=1.000, |d|=5.190, overlap=0.000, true_mean=0.629372, false_mean=-0.373148
    spec_slope: AUC=1.000, |d|=5.190, overlap=0.000, true_mean=-0.000193195, false_mean=-0.000205389
    direction_contrast__time_rank_pct: AUC=1.000, |d|=3.379, overlap=0.000, true_mean=0.763158, false_mean=0.263158
    spec_slope__time_rank_pct: AUC=1.000, |d|=3.379, overlap=0.000, true_mean=0.763158, false_mean=0.263158
    best_direction_combined_score__time_rank_pct: AUC=1.000, |d|=3.379, overlap=0.000, true_mean=0.263158, false_mean=0.763158
    decay_R2__time_robust_z: AUC=0.823, |d|=1.389, overlap=0.316, true_mean=0.246309, false_mean=-0.562114
  单特征最好AUC: 1.0000
  特征分布中位重叠度: 0.2105
  诊断: 存在较明显区分特征，可以进一步重点分析。

[raw_feature]
对比表: C:\Users\jiangxinru6\Desktop\wurenji\leak_v9_144226_diagnosis_results\v9_144226_raw_feature_compare.csv
  区分力前10特征:
    best_direction_combined_score: AUC=1.000, |d|=7.286, overlap=0.000, true_mean=0.770592, false_mean=0.852768
    direction_contrast: AUC=1.000, |d|=5.402, overlap=0.000, true_mean=1.27288, false_mean=1.07355
    spec_slope: AUC=1.000, |d|=5.190, overlap=0.000, true_mean=-0.000193195, false_mean=-0.000205389
    high_freq_ratio: AUC=0.950, |d|=2.236, overlap=0.158, true_mean=0.0096807, false_mean=0.0108996
    decay_R2: AUC=0.823, |d|=1.389, overlap=0.316, true_mean=0.878475, false_mean=0.843452
    near_far_ratio: AUC=0.789, |d|=1.152, overlap=0.526, true_mean=1.67801, false_mean=1.7465
    ratio_60_70k: AUC=0.784, |d|=1.120, overlap=0.211, true_mean=0.000149452, false_mean=0.000130545
    time_energy_cv: AUC=0.643, |d|=0.315, overlap=0.211, true_mean=0.116509, false_mean=0.115439
    spec_flatness: AUC=0.518, |d|=0.304, overlap=0.474, true_mean=0.0320446, false_mean=0.0315262
  单特征最好AUC: 1.0000
  特征分布中位重叠度: 0.2105
  诊断: 存在较明显区分特征，可以进一步重点分析。

[heatmap_feature]
对比表: C:\Users\jiangxinru6\Desktop\wurenji\leak_v9_144226_diagnosis_results\v9_144226_heatmap_feature_compare.csv
  区分力前10特征:
    hm_core_to_outer_energy_ratio: AUC=0.986, |d|=1.377, overlap=0.211, true_mean=0.998405, false_mean=0.561856
    hm_radial_spread_norm: AUC=0.978, |d|=2.534, overlap=0.158, true_mean=0.183294, false_mean=0.216957
    hm_diffuse_score: AUC=0.978, |d|=2.298, overlap=0.053, true_mean=0.611463, false_mean=0.673118
    hm_entropy_2d: AUC=0.978, |d|=2.064, overlap=0.105, true_mean=0.949549, false_mean=0.9664
    hm_energy_concentration_top10: AUC=0.970, |d|=2.223, overlap=0.211, true_mean=0.457129, false_mean=0.387143
    hm_weighted_eccentricity: AUC=0.898, |d|=1.906, overlap=0.000, true_mean=0.170964, false_mean=0.304482
    hm_weighted_elongation: AUC=0.898, |d|=1.393, overlap=0.000, true_mean=1.02026, false_mean=1.05071
    hm_energy_concentration_top5: AUC=0.892, |d|=1.845, overlap=0.263, true_mean=0.251478, false_mean=0.220164
    hm_shape_leak_like_score: AUC=0.820, |d|=1.326, overlap=0.211, true_mean=0.35683, false_mean=0.409802
    hm_largest_component_ratio_p95: AUC=0.601, |d|=0.393, overlap=0.158, true_mean=0.998028, false_mean=0.99648
  单特征最好AUC: 0.9861
  特征分布中位重叠度: 0.2105
  诊断: 存在较明显区分特征，可以进一步重点分析。




