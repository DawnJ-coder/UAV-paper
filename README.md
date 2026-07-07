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
  测试组 HM20260626_142938.ld: n=30, best_t=0.330, default_acc=0.533, calib_acc=0.500, calib_bal_acc=0.500, auc=1.0
  测试组 HM20260626_143034.ld: n=38, best_t=0.300, default_acc=0.868, calib_acc=0.895, calib_bal_acc=0.895, auc=0.9722991689750693
  测试组 HM20260626_144226.ld: n=38, best_t=0.320, default_acc=0.500, calib_acc=0.789, calib_bal_acc=0.789, auc=1.0
  测试组 HM20260626_144325.ld: n=40, best_t=0.340, default_acc=0.825, calib_acc=1.000, calib_bal_acc=1.000, auc=1.0

分组阈值验证汇总: C:\Users\jiangxinru6\Desktop\wurenji\leak_v6_threshold_calibrated_results\v6_group_threshold_validation_summary.csv
预测明细: C:\Users\jiangxinru6\Desktop\wurenji\leak_v6_threshold_calibrated_results\v6_group_threshold_predictions.csv
校准阈值后的误判样本: C:\Users\jiangxinru6\Desktop\wurenji\leak_v6_threshold_calibrated_results\v6_calibrated_misclassified_samples.csv
报告: C:\Users\jiangxinru6\Desktop\wurenji\leak_v6_threshold_calibrated_results\v6_threshold_calibration_report.txt

最终模型: C:\Users\jiangxinru6\Desktop\wurenji\leak_v6_threshold_calibrated_results\v6_final_leak_classifier.pkl
最终模型配置: C:\Users\jiangxinru6\Desktop\wurenji\leak_v6_threshold_calibrated_results\v6_final_model_config.json
全局阈值曲线: C:\Users\jiangxinru6\Desktop\wurenji\leak_v6_threshold_calibrated_results\v6_global_threshold_curve.csv
最终模型特征重要性: C:\Users\jiangxinru6\Desktop\wurenji\leak_v6_threshold_calibrated_results\v6_final_feature_importance.csv
推荐全局阈值: 0.330
OOF阈值优化得分: 0.986



图片输出:
  C:\Users\jiangxinru6\Desktop\wurenji\leak_v6_threshold_calibrated_results\figures\v6_default_vs_calibrated_accuracy.png
  C:\Users\jiangxinru6\Desktop\wurenji\leak_v6_threshold_calibrated_results\figures\v6_calibrated_thresholds.png
  C:\Users\jiangxinru6\Desktop\wurenji\leak_v6_threshold_calibrated_results\figures\v6_top_feature_importance.png
  C:\Users\jiangxinru6\Desktop\wurenji\leak_v6_threshold_calibrated_results\figures\v6_global_threshold_curve.png

最终模型重要特征前10:
  ratio_60_70k: 0.110552
  direction_contrast: 0.106970
  spec_centroid_hz: 0.104205
  spec_flatness: 0.074251
  ratio_50_60k: 0.066474
  spec_rolloff_85_hz: 0.050637
  best_direction_combined_score: 0.042580
  ratio_40_50k: 0.038380
  high_freq_ratio: 0.036808
  energy_30_40k: 0.034507

================================================================================
全部完成
输出文件夹: C:\Users\jiangxinru6\Desktop\wurenji\leak_v6_threshold_calibrated_results
================================================================================
PS C:\Users\jiangxinru6\Desktop\wurenji>
