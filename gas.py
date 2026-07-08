PS C:\Users\jiangxinru6\Desktop\wurenji> python .\leak_v8.py   
====================================================================================================
v8 热力图形态特征 + v7 稳健特征 分类程序
====================================================================================================
样本数量: 146
label
TRUE_LEAK     73
FALSE_LEAK    73
Name: count, dtype: int64

开始提取热力图形态特征...
  已处理 20/146 张/行
  已处理 40/146 张/行
  已处理 60/146 张/行
  已处理 80/146 张/行
  已处理 100/146 张/行
  已处理 120/146 张/行
  已处理 140/146 张/行
  已处理 146/146 张/行
热力图特征表: C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_heatmap_shape_results\v8_heatmap_features.csv
缺失/失败热力图数量: 0

开始构造 v8 特征矩阵：v7稳健特征 + heatmap形态特征 + time内部z/rank...
v7初始数值特征数: 41
v7稳健基础特征数: 26
heatmap基础形态特征数: 48
最终模型特征数: 222
v8稳健特征数据表: C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_heatmap_shape_results\v8_robust_feature_dataset.csv

开始 v8 按时间点整组验证...
分组数量: 4
  测试组 HM20260626_142938.ld: n=30, best_t=0.500, default_acc=1.000, model_acc=1.000, rank_acc=1.000, final_decisive_acc=1.000, suspect=0, auc=1.0
  测试组 HM20260626_143034.ld: n=38, best_t=0.430, default_acc=0.895, model_acc=0.842, rank_acc=0.895, final_decisive_acc=0.889, suspect=2, auc=0.9639889196675899
  测试组 HM20260626_144226.ld: n=38, best_t=0.460, default_acc=0.605, model_acc=0.553, rank_acc=0.526, final_decisive_acc=0.543, suspect=3, auc=0.6980609418282548
  测试组 HM20260626_144325.ld: n=40, best_t=0.660, default_acc=0.975, model_acc=0.875, rank_acc=1.000, final_decisive_acc=1.000, suspect=5, auc=1.0

分组验证汇总: C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_heatmap_shape_results\v8_group_validation_summary.csv
预测明细: C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_heatmap_shape_results\v8_predictions.csv
v8模型误判样本: C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_heatmap_shape_results\v8_model_misclassified_samples.csv
v8最终SUSPECT样本: C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_heatmap_shape_results\v8_suspect_samples.csv

最终模型: C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_heatmap_shape_results\v8_final_heatmap_robust_classifier.pkl
最终配置: C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_heatmap_shape_results\v8_final_model_config.json
最终特征重要性: C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_heatmap_shape_results\v8_final_feature_importance.csv
全局OOF阈值曲线: C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_heatmap_shape_results\v8_global_group_oof_threshold_curve.csv
v8推荐阈值: 0.480
v8 OOF阈值优化得分: 0.8699
报告: C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_heatmap_shape_results\v8_report.txt

图片输出:
  C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_heatmap_shape_results\figures\v8_group_accuracy_comparison.png
  C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_heatmap_shape_results\figures\v8_top_feature_importance.png
  C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_heatmap_shape_results\figures\v8_global_threshold_curve.png
  C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_heatmap_shape_results\figures\v8_heatmap_feature_hm_hot_area_p95_ratio.png
  C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_heatmap_shape_results\figures\v8_heatmap_feature_hm_weighted_elongation.png
  C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_heatmap_shape_results\figures\v8_heatmap_feature_hm_entropy_2d.png
  
C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_heatmap_shape_results\figures\v8_heatmap_feature_hm_largest_component_ratio_to_hot_p95.png
  C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_heatmap_shape_results\figures\v8_heatmap_feature_hm_directed_core_score.png
  C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_heatmap_shape_results\figures\v8_heatmap_feature_hm_diffuse_score.png

最终模型重要特征前15:
  spec_slope__time_rank_pct: 0.081030
  spec_slope__time_robust_z: 0.073795
  hm_weighted_cy__time_robust_z: 0.053763
  best_direction_combined_score__time_robust_z: 0.052509
  best_direction_combined_score: 0.046677
  ratio_60_70k__time_robust_z: 0.045633
  spec_slope: 0.038405
  ratio_60_70k__time_rank_pct: 0.036889
  best_direction_combined_score__time_rank_pct: 0.035576
  hm_weighted_cy__time_rank_pct: 0.034481
  hm_weighted_cy: 0.029639
  hm_weighted_cx__time_robust_z: 0.029261
  hm_asymmetry_ud__time_rank_pct: 0.024236
  hm_weighted_cx__time_rank_pct: 0.024232
  hm_asymmetry_ud__time_robust_z: 0.023494

各时间点核心结果:
  HM20260626_142938.ld: default_acc=1.000, model_acc=1.000, rank_acc=1.000, final_decisive_acc=1.000, suspect=0, auc=1.0
  HM20260626_143034.ld: default_acc=0.895, model_acc=0.842, rank_acc=0.895, final_decisive_acc=0.889, suspect=2, auc=0.9639889196675899
  HM20260626_144226.ld: default_acc=0.605, model_acc=0.553, rank_acc=0.526, final_decisive_acc=0.543, suspect=3, auc=0.6980609418282548
  HM20260626_144325.ld: default_acc=0.975, model_acc=0.875, rank_acc=1.000, final_decisive_acc=1.000, suspect=5, auc=1.0

====================================================================================================
全部完成
输出文件夹: C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_heatmap_shape_results
====================================================================================================
PS C:\Users\jiangxinru6\Desktop\wurenji> 
