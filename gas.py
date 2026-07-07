数据质量报告: C:\Users\jiangxinru6\Desktop\wurenji\leak_v5_group_validation_results\data_quality_report.csv
频率特征有效性报告: C:\Users\jiangxinru6\Desktop\wurenji\leak_v5_group_validation_results\frequency_feature_validity_report.csv
开始按时间点整组留出验证...
分组数量: 4
  测试组 HM20260626_142938.ld: 样本=30, 真=15, 假=15, acc=0.533, auc=1.0
  测试组 HM20260626_143034.ld: 样本=38, 真=19, 假=19, acc=0.868, auc=0.9695290858725762
  测试组 HM20260626_144226.ld: 样本=38, 真=19, 假=19, acc=0.500, auc=1.0
  测试组 HM20260626_144325.ld: 样本=40, 真=20, 假=20, acc=0.825, auc=1.0
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
全部完成
输出文件夹: C:\Users\jiangxinru6\Desktop\wurenji\leak_v5_group_validation_results
