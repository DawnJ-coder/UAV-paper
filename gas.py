实验: A_v7_like_baseline
特征数量: 87
HM20260626_142938.ld: AUC=1.000, default=0.867, model=0.600, prob_rank=1.000, physics_rank=1.000, final=0.600
HM20260626_143034.ld: AUC=0.974, default=0.947, model=0.553, prob_rank=0.947, physics_rank=0.947, final=0.553
HM20260626_144226.ld: AUC=0.000, default=0.000, model=0.026, prob_rank=0.000, physics_rank=0.000, final=0.026
HM20260626_144325.ld: AUC=0.724, default=0.650, model=0.500, prob_rank=0.750, physics_rank=0.000, final=0.500


实验: B_directed_wideband_only
特征数量: 94
HM20260626_142938.ld: AUC=1.000, default=1.000, model=0.500, prob_rank=1.000, physics_rank=1.000, final=1.000
HM20260626_143034.ld: AUC=0.338, default=0.474, model=0.500, prob_rank=0.368, physics_rank=0.947, final=0.947
HM20260626_144226.ld: AUC=0.014, default=0.053, model=0.184, prob_rank=0.053, physics_rank=0.000, final=0.000
HM20260626_144325.ld: AUC=0.095, default=0.250, model=0.500, prob_rank=0.200, physics_rank=0.000, final=0.000
实验: C_v8_v7_plus_directed_wideband
特征数量: 181
HM20260626_142938.ld: AUC=1.000, default=0.867, model=0.500, prob_rank=1.000, physics_rank=1.000, final=1.000
HM20260626_143034.ld: AUC=0.877, default=0.737, model=0.500, prob_rank=0.789, physics_rank=0.947, final=0.947
HM20260626_144226.ld: AUC=0.000, default=0.000, model=0.184, prob_rank=0.000, physics_rank=0.000, final=0.000
HM20260626_144325.ld: AUC=0.545, default=0.475, model=0.500, prob_rank=0.650, physics_rank=0.000, final=0.000

输出文件夹: C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_standalone_directed_wideband_auto_center_v2_results
报告: C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_standalone_directed_wideband_auto_center_v2_results\v8_report.txt
完整特征表: C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_standalone_directed_wideband_auto_center_v2_results\v8_feature_dataset_with_time_relative.csv
预测明细: C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_standalone_directed_wideband_auto_center_v2_results\v8_predictions.csv
分组汇总: C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_standalone_directed_wideband_auto_center_v2_results\v8_group_validation_summary.csv
144226配对检查: C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_standalone_directed_wideband_auto_center_v2_results\v8_144226_pair_check.csv
144226方向性宽频特征对比: C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_standalone_directed_wideband_auto_center_v2_results\v8_144226_directed_wideband_feature_compare.csv

核心结果摘要:

A_v7_like_baseline:
  平均AUC=0.674, 平均model_acc=0.420, 平均physics_rank_acc=0.487, 平均final_acc=0.420
  144226: AUC=0.000, model_acc=0.026, physics_rank_acc=0.000, final_acc=0.026

B_directed_wideband_only:
  平均AUC=0.362, 平均model_acc=0.421, 平均physics_rank_acc=0.487, 平均final_acc=0.487
  144226: AUC=0.014, model_acc=0.184, physics_rank_acc=0.000, final_acc=0.000

C_v8_v7_plus_directed_wideband:
  平均AUC=0.605, 平均model_acc=0.421, 平均physics_rank_acc=0.487, 平均final_acc=0.487
  144226: AUC=0.000, model_acc=0.184, physics_rank_acc=0.000, final_acc=0.000

144226 center配对检查:
  prob排序正确率: 0.000
  physics排序正确率: 0.000
  physics排序失败center: 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18
