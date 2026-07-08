PS C:\Users\jiangxinru6\Desktop\wurenji> python leak_v7_robust_feature_rank_classifier.py
==========================================================================================
v7 稳健特征重构 + 时间点内部归一化/排序 分类程序
==========================================================================================
样本数量: 146
label
TRUE_LEAK     73
FALSE_LEAK    73
Name: count, dtype: int64

开始筛选稳健特征并构造 time 内部归一化/排名特征...
初始数值特征数: 41
删除不稳定特征后基础特征数: 26
最终模型特征数: 78
被删除特征列表: C:\Users\jiangxinru6\Desktop\wurenji\leak_v7_robust_feature_results\v7_removed_features.csv
v7稳健特征数据表: C:\Users\jiangxinru6\Desktop\wurenji\leak_v7_robust_feature_results\v7_robust_feature_dataset.csv

开始 v7 按时间点整组验证...
分组数量: 4
  测试组 HM20260626_142938.ld: n=30, best_t=0.440, default_acc=1.000, model_acc=0.967, rank_acc=1.000, final_decisive_acc=1.000, suspect=1, auc=1.0
  测试组 HM20260626_143034.ld: n=38, best_t=0.420, default_acc=0.895, model_acc=0.947, rank_acc=0.947, final_decisive_acc=0.972, suspect=2, auc=0.96398891966759
  测试组 HM20260626_144226.ld: n=38, best_t=0.440, default_acc=0.711, model_acc=0.579, rank_acc=0.684, final_decisive_acc=0.647, suspect=4, auc=0.6869806094182825
  测试组 HM20260626_144325.ld: n=40, best_t=0.490, default_acc=1.000, model_acc=1.000, rank_acc=1.000, final_decisive_acc=1.000, suspect=0, auc=1.0

分组验证汇总: C:\Users\jiangxinru6\Desktop\wurenji\leak_v7_robust_feature_results\v7_group_validation_summary.csv
预测明细: C:\Users\jiangxinru6\Desktop\wurenji\leak_v7_robust_feature_results\v7_predictions.csv
v7模型误判样本: C:\Users\jiangxinru6\Desktop\wurenji\leak_v7_robust_feature_results\v7_model_misclassified_samples.csv
v7最终SUSPECT样本: C:\Users\jiangxinru6\Desktop\wurenji\leak_v7_robust_feature_results\v7_suspect_samples.csv

最终模型: C:\Users\jiangxinru6\Desktop\wurenji\leak_v7_robust_feature_results\v7_final_robust_classifier.pkl
最终配置: C:\Users\jiangxinru6\Desktop\wurenji\leak_v7_robust_feature_results\v7_final_model_config.json
最终特征重要性: C:\Users\jiangxinru6\Desktop\wurenji\leak_v7_robust_feature_results\v7_final_feature_importance.csv
全局OOF阈值曲线: C:\Users\jiangxinru6\Desktop\wurenji\leak_v7_robust_feature_results\v7_global_oof_threshold_curve.csv
v7推荐阈值: 0.500
v7 OOF阈值优化得分: 0.8973
报告: C:\Users\jiangxinru6\Desktop\wurenji\leak_v7_robust_feature_results\v7_report.txt

图片输出:
  C:\Users\jiangxinru6\Desktop\wurenji\leak_v7_robust_feature_results\figures\v7_group_accuracy_comparison.png
  C:\Users\jiangxinru6\Desktop\wurenji\leak_v7_robust_feature_results\figures\v7_top_feature_importance.png
  C:\Users\jiangxinru6\Desktop\wurenji\leak_v7_robust_feature_results\figures\v7_global_threshold_curve.png


最终模型重要特征前10:
  spec_slope__time_robust_z: 0.134353
  spec_slope__time_rank_pct: 0.111990
  best_direction_combined_score__time_robust_z: 0.082974
  ratio_60_70k__time_robust_z: 0.078560
  ratio_60_70k__time_rank_pct: 0.061660
  best_direction_combined_score: 0.061315
  best_direction_combined_score__time_rank_pct: 0.060558
  spec_slope: 0.045432
  direction_contrast__time_robust_z: 0.034279
  ratio_60_70k: 0.031547

各时间点核心结果:
  HM20260626_142938.ld: default_acc=1.000, model_acc=0.967, rank_acc=1.000, final_decisive_acc=1.000, suspect=1, auc=1.0
  HM20260626_143034.ld: default_acc=0.895, model_acc=0.947, rank_acc=0.947, final_decisive_acc=0.972, suspect=2, auc=0.96398891966759
  HM20260626_144226.ld: default_acc=0.711, model_acc=0.579, rank_acc=0.684, final_decisive_acc=0.647, suspect=4, auc=0.6869806094182825
  HM20260626_144325.ld: default_acc=1.000, model_acc=1.000, rank_acc=1.000, final_decisive_acc=1.000, suspect=0, auc=1.0

==========================================================================================
全部完成
输出文件夹: C:\Users\jiangxinru6\Desktop\wurenji\leak_v7_robust_feature_results
==========================================================================================
PS C:\Users\jiangxinru6\Desktop\wurenji> 
