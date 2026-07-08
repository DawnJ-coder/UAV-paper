PS C:\Users\jiangxinru6\Desktop\wurenji> python .\leak_v8.py
====================================================================================================
v8.1 热力图核心形态特征消融版
====================================================================================================
样本数量: 146
label
TRUE_LEAK     73
FALSE_LEAK    73
Name: count, dtype: int64

开始提取 v8.1 核心 heatmap 形态特征...
  已处理 20/146 张/行
  已处理 40/146 张/行
  已处理 60/146 张/行
  已处理 80/146 张/行
  已处理 100/146 张/行
  已处理 120/146 张/行
  已处理 140/146 张/行
  已处理 146/146 张/行
核心 heatmap 特征表: C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_1_heatmap_shape_ablation_results\v8_1_heatmap_core_shape_features.csv
缺失/失败 heatmap 数量: 0

开始构造 v8.1 消融特征集...
初始 v7 数值特征数: 41
v7 稳健基础特征数: 26
heatmap 核心形态基础特征数: 26
  A_v7_only: 模型特征数=78, 文件=C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_1_heatmap_shape_ablation_results\v8_1_features_A_v7_only.csv
  B_heatmap_shape_only: 模型特征数=78, 文件=C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_1_heatmap_shape_ablation_results\v8_1_features_B_heatmap_shape_only.csv
  C_v7_plus_heatmap_shape: 模型特征数=156, 文件=C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_1_heatmap_shape_ablation_results\v8_1_features_C_v7_plus_heatmap_shape.csv

开始实验 A_v7_only 按时间点整组验证...
特征数: 78
  HM20260626_142938.ld: best_t=0.440, default_acc=1.000, model_acc=0.933, rank_acc=1.000, final_decisive_acc=1.000, suspect=2, auc=1.0
  HM20260626_143034.ld: best_t=0.450, default_acc=0.895, model_acc=0.921, rank_acc=0.947, final_decisive_acc=0.971, suspect=3, auc=0.96398891966759
  HM20260626_144226.ld: best_t=0.440, default_acc=0.711, model_acc=0.579, rank_acc=0.684, final_decisive_acc=0.647, suspect=4, auc=0.7008310249307479
  HM20260626_144325.ld: best_t=0.480, default_acc=1.000, model_acc=1.000, rank_acc=1.000, final_decisive_acc=1.000, suspect=0, auc=1.0

开始实验 B_heatmap_shape_only 按时间点整组验证...
特征数: 78
  HM20260626_142938.ld: best_t=0.310, default_acc=0.600, model_acc=0.467, rank_acc=0.533, final_decisive_acc=0.500, suspect=8, auc=0.64
  HM20260626_143034.ld: best_t=0.470, default_acc=0.711, model_acc=0.737, rank_acc=0.684, final_decisive_acc=0.735, suspect=4, auc=0.7423822714681441
  HM20260626_144226.ld: best_t=0.380, default_acc=0.553, model_acc=0.579, rank_acc=0.579, final_decisive_acc=0.594, suspect=6, auc=0.5373961218836565
  HM20260626_144325.ld: best_t=0.420, default_acc=0.950, model_acc=0.825, rank_acc=0.950, final_decisive_acc=0.970, suspect=7, auc=0.9675

开始实验 C_v7_plus_heatmap_shape 按时间点整组验证...
特征数: 156
  HM20260626_142938.ld: best_t=0.460, default_acc=1.000, model_acc=1.000, rank_acc=1.000, final_decisive_acc=1.000, suspect=0, auc=1.0
  HM20260626_143034.ld: best_t=0.280, default_acc=0.842, model_acc=0.842, rank_acc=0.895, final_decisive_acc=0.912, suspect=4, auc=0.9529085872576177
  HM20260626_144226.ld: best_t=0.440, default_acc=0.605, model_acc=0.579, rank_acc=0.658, final_decisive_acc=0.629, suspect=3, auc=0.6343490304709141
  HM20260626_144325.ld: best_t=0.520, default_acc=0.900, model_acc=0.900, rank_acc=1.000, final_decisive_acc=1.000, suspect=4, auc=1.0

=========================================================================\===========================
v8.1 消融实验完成
====================================================================================================
总报告: C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_1_heatmap_shape_ablation_results\v8_1_report.txt
总汇总表: C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_1_heatmap_shape_ablation_results\v8_1_ablation_overall_summary.csv
heatmap核心特征表: C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_1_heatmap_shape_ablation_results\v8_1_heatmap_core_shape_features.csv
144226 heatmap形态真假对比: C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_1_heatmap_shape_ablation_results\diagnosis_144226\HM20260626_144226_heatmap_shape_true_false_compare.csv

消融实验核心结果:

实验 A_v7_only: 特征数=78
  平均AUC=0.916, 平均model_acc=0.858, 平均rank_acc=0.908
  144226: AUC=0.701, default_acc=0.711, model_acc=0.579, rank_acc=0.684, final_decisive_acc=0.647, suspect=4
  重要特征前10:
    spec_slope__time_robust_z: 0.131072
    spec_slope__time_rank_pct: 0.115445
    ratio_60_70k__time_robust_z: 0.081877
    best_direction_combined_score__time_robust_z: 0.081713
    ratio_60_70k__time_rank_pct: 0.064652
    best_direction_combined_score__time_rank_pct: 0.062435
    best_direction_combined_score: 0.060873
    spec_slope: 0.044274
    direction_contrast__time_robust_z: 0.031571
    ratio_60_70k: 0.030385


实验 B_heatmap_shape_only: 特征数=78
  平均AUC=0.722, 平均model_acc=0.652, 平均rank_acc=0.687
  144226: AUC=0.537, default_acc=0.553, model_acc=0.579, rank_acc=0.579, final_decisive_acc=0.594, suspect=6
  重要特征前10:
    hm_energy_concentration_top5: 0.078261
    hm_energy_concentration_top5__time_robust_z: 0.064916
    hm_energy_concentration_top10__time_robust_z: 0.048842
    hm_energy_concentration_top10: 0.046192
    hm_energy_concentration_top5__time_rank_pct: 0.037031
    hm_p90_elongation: 0.035298
    hm_diffuse_score: 0.027792
    hm_core_to_outer_energy_ratio: 0.027629
    hm_p90_elongation__time_robust_z: 0.025472
    hm_energy_concentration_top10__time_rank_pct: 0.025145

实验 C_v7_plus_heatmap_shape: 特征数=156
  平均AUC=0.897, 平均model_acc=0.830, 平均rank_acc=0.888
  144226: AUC=0.634, default_acc=0.605, model_acc=0.579, rank_acc=0.658, final_decisive_acc=0.629, suspect=3
  重要特征前10:
    spec_slope__time_robust_z: 0.105873
    spec_slope__time_rank_pct: 0.073402
    ratio_60_70k__time_robust_z: 0.063853
    spec_slope: 0.063053
    best_direction_combined_score__time_robust_z: 0.059533
    best_direction_combined_score__time_rank_pct: 0.056513
    ratio_60_70k__time_rank_pct: 0.052200
    best_direction_combined_score: 0.049120
    ratio_60_70k: 0.033411
    direction_contrast__time_robust_z: 0.031924

图片输出:
  C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_1_heatmap_shape_ablation_results\figures\v8_1_ablation_auc.png
  C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_1_heatmap_shape_ablation_results\figures\v8_1_ablation_model_accuracy.png
  C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_1_heatmap_shape_ablation_results\figures\v8_1_ablation_rank_accuracy.png

输出文件夹: C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_1_heatmap_shape_ablation_results
PS C:\Users\jiangxinru6\Desktop\wurenji> 

