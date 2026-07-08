PS C:\Users\jiangxinru6\Desktop\wurenji> python leak_v5_group_validate_and_build_model.py
================================================================================
v5 按时间点分组验证 + 最终模型训练
================================================================================
样本数: 146
label
TRUE_LEAK     73
FALSE_LEAK    73
Name: count, dtype: int64
可用特征数: 41
数据质量报告: C:\Users\jiangxinru6\Desktop\wurenji\leak_v5_group_validation_results\data_quality_report.csv
频率特征有效性报告: C:\Users\jiangxinru6\Desktop\wurenji\leak_v5_group_validation_results\frequency_feature_validity_report.csv

开始按时间点整组留出验证...
分组数量: 4
  测试组 HM20260626_142938.ld: 样本=30, 真=15, 假=15, acc=0.533, auc=1.0
  测试组 HM20260626_143034.ld: 样本=38, 真=19, 假=19, acc=0.868, auc=0.8919667590027701
  测试组 HM20260626_144226.ld: 样本=38, 真=19, 假=19, acc=0.500, auc=0.6606648199445984
  测试组 HM20260626_144325.ld: 样本=40, 真=20, 假=20, acc=1.000, auc=1.0
分组验证汇总: C:\Users\jiangxinru6\Desktop\wurenji\leak_v5_group_validation_results\group_validation_summary.csv
分组验证预测明细: C:\Users\jiangxinru6\Desktop\wurenji\leak_v5_group_validation_results\group_validation_predictions.csv
误判样本: C:\Users\jiangxinru6\Desktop\wurenji\leak_v5_group_validation_results\misclassified_samples.csv
分组验证报告: C:\Users\jiangxinru6\Desktop\wurenji\leak_v5_group_validation_results\group_validation_report.txt
最终模型: C:\Users\jiangxinru6\Desktop\wurenji\leak_v5_group_validation_results\final_leak_classifier_random_forest.pkl
模型特征配置: C:\Users\jiangxinru6\Desktop\wurenji\leak_v5_group_validation_results\final_feature_config.json
最终模型特征重要性: C:\Users\jiangxinru6\Desktop\wurenji\leak_v5_group_validation_results\final_model_feature_importance.csv
分组准确率图: C:\Users\jiangxinru6\Desktop\wurenji\leak_v5_group_validation_results\figures\group_validation_accuracy.png
重要特征图: C:\Users\jiangxinru6\Desktop\wurenji\leak_v5_group_validation_results\figures\top_feature_importance.png
重要特征前10:
  energy_30_40k: 0.109391
  spec_slope: 0.109181
  time_energy_std: 0.089470
  ratio_60_70k: 0.064194
  direction_contrast: 0.059183
  spec_flatness: 0.036863
  spec_centroid_hz: 0.034444
  energy_40_50k: 0.033397
  ratio_30_40k: 0.028967

================================================================================
全部完成
输出文件夹: C:\Users\jiangxinru6\Desktop\wurenji\leak_v5_group_validation_results
================================================================================
PS C:\Users\jiangxinru6\Desktop\wurenji> python leak_v6_threshold_calibrated_classifier.py
================================================================================
v6 自动阈值校准版真假泄漏分类程序
================================================================================
样本数量: 146
label
TRUE_LEAK     73
FALSE_LEAK    73
Name: count, dtype: int64
可用特征数量: 41
开始 v6 按时间点留出验证 + 自动阈值校准...
分组数量: 4
  测试组 HM20260626_142938.ld: n=30, best_t=0.450, default_acc=0.533, calib_acc=0.533, calib_bal_acc=0.533, auc=1.0
  测试组 HM20260626_143034.ld: n=38, best_t=0.420, default_acc=0.868, calib_acc=0.842, calib_bal_acc=0.842, auc=0.8933518005540166
  测试组 HM20260626_144226.ld: n=38, best_t=0.350, default_acc=0.500, calib_acc=0.500, calib_bal_acc=0.500, auc=0.670360110803324
  测试组 HM20260626_144325.ld: n=40, best_t=0.390, default_acc=1.000, calib_acc=1.000, calib_bal_acc=1.000, auc=1.0

分组阈值验证汇总: C:\Users\jiangxinru6\Desktop\wurenji\leak_v6_threshold_calibrated_results\v6_group_threshold_validation_summary.csv
预测明细: C:\Users\jiangxinru6\Desktop\wurenji\leak_v6_threshold_calibrated_results\v6_group_threshold_predictions.csv
校准阈值后的误判样本: C:\Users\jiangxinru6\Desktop\wurenji\leak_v6_threshold_calibrated_results\v6_calibrated_misclassified_samples.csv
报告: C:\Users\jiangxinru6\Desktop\wurenji\leak_v6_threshold_calibrated_results\v6_threshold_calibration_report.txt

最终模型: C:\Users\jiangxinru6\Desktop\wurenji\leak_v6_threshold_calibrated_results\v6_final_leak_classifier.pkl
最终模型配置: C:\Users\jiangxinru6\Desktop\wurenji\leak_v6_threshold_calibrated_results\v6_final_model_config.json
全局阈值曲线: C:\Users\jiangxinru6\Desktop\wurenji\leak_v6_threshold_calibrated_results\v6_global_threshold_curve.csv
最终模型特征重要性: C:\Users\jiangxinru6\Desktop\wurenji\leak_v6_threshold_calibrated_results\v6_final_feature_importance.csv
推荐全局阈值: 0.390
OOF阈值优化得分: 0.986

图片输出:
  C:\Users\jiangxinru6\Desktop\wurenji\leak_v6_threshold_calibrated_results\figures\v6_default_vs_calibrated_accuracy.png
  C:\Users\jiangxinru6\Desktop\wurenji\leak_v6_threshold_calibrated_results\figures\v6_calibrated_thresholds.png
  C:\Users\jiangxinru6\Desktop\wurenji\leak_v6_threshold_calibrated_results\figures\v6_top_feature_importance.png
  C:\Users\jiangxinru6\Desktop\wurenji\leak_v6_threshold_calibrated_results\figures\v6_global_threshold_curve.png
最终模型重要特征前10:
  best_direction_combined_score: 0.137865
  energy_30_40k: 0.109391
  spec_slope: 0.109181
  time_energy_std: 0.089470
  ratio_60_70k: 0.064194
  direction_contrast: 0.059183
  spec_flatness: 0.036863
  spec_centroid_hz: 0.034444
  energy_40_50k: 0.033397
  ratio_30_40k: 0.028967

================================================================================
全部完成
输出文件夹: C:\Users\jiangxinru6\Desktop\wurenji\leak_v6_threshold_calibrated_results
================================================================================
PS C:\Users\jiangxinru6\Desktop\wurenji> 





