PS C:\Users\jiangxinru6\Documents\HikLink_Files\jiangxinru6\received\leak_v11_cross_scene_validation_package> python leak_v10_fingerprint_similarity_blocked_seconds.py --mode external --v9-dir "C:\Users\jiangxinru6\Desktop\wurenji\factory_C_v9_results" --output-dir "C:\Users\jiangxinru6\Desktop\wurenji\factory_C_v10_external" --prototype-file "C:\Users\jiangxinru6\Desktop\wurenji\leak_v10_fingerprint_results\factory_A_frozen_reference.npz" 
v10.1 文件格式与秒编号解析报告
========================================================================================
输入CSV: C:\Users\jiangxinru6\Desktop\wurenji\factory_C_v9_results\v9_all_features.csv
总样本数: 38
成功解析秒编号: 38
解析成功率: 1.0000
连续时间块大小: 5 秒
规则: 00_00=秒索引0=第1秒；00_01=秒索引1=第2秒。
解析优先级: center_file > center > sample_id。

解析来源统计:
  center_file: 38

各标签秒编号/时间块统计:
  FALSE_LEAK: n=19, 秒索引范围=0~18, 时间块数=4, 时间块=['second_block_0000', 'second_block_0001', 'second_block_0002', 'second_block_0003']
  TRUE_LEAK: n=19, 秒索引范围=0~18, 时间块数=4, 时间块=['second_block_0000', 'second_block_0001', 'second_block_0002', 'second_block_0003']
==========================================================================================
v10 冻结指纹外部验证完成
参考scene: factory_A
冻结阈值: 0.0
成功样本: 38
失败样本: 0
逐样本结果: C:\Users\jiangxinru6\Desktop\wurenji\factory_C_v10_external\v10_external_similarity.csv
汇总: C:\Users\jiangxinru6\Desktop\wurenji\factory_C_v10_external\v10_external_summary.csv
报告: C:\Users\jiangxinru6\Desktop\wurenji\factory_C_v10_external\v10_external_report.txt
PS C:\Users\jiangxinru6\Documents\HikLink_Files\jiangxinru6\received\leak_v11_cross_scene_validation_package> 
