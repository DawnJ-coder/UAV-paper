PS C:\Users\jiangxinru6\Desktop\wurenji> python 0713.py
==========================================================================================
v10 泄漏残差指纹相似度验证
==========================================================================================
v9结果目录: C:\Users\jiangxinru6\Desktop\wurenji\leak_v9_local_background_results
参考scene: factory_A
参考样本数: 30
参考TRUE/FALSE: 15 / 15

内部验证方式: stratified_sample_cv_fallback
[警告] 每个类别没有至少两个独立time，无法按time整组验证；当前退回分层样本交叉验证。同一time的不同center可能进入不同折，因此结果只能作为初步检查，不能证明跨时间泛化。
OOF Margin AUC: 1.0000
自然阈值0 平衡准确率: 1.0000
冻结阈值: 0.000000
冻结阈值平衡准确率: 1.0000
TRUE Margin中位数: 0.023620
FALSE Margin中位数: -0.021056

输出目录: C:\Users\jiangxinru6\Desktop\wurenji\leak_v10_fingerprint_results
内部OOF明细: C:\Users\jiangxinru6\Desktop\wurenji\leak_v10_fingerprint_results\v10_internal_oof_similarity.csv
内部汇总: C:\Users\jiangxinru6\Desktop\wurenji\leak_v10_fingerprint_results\v10_internal_summary.csv
冻结指纹: C:\Users\jiangxinru6\Desktop\wurenji\leak_v10_fingerprint_results\v10_frozen_reference.npz
报告: C:\Users\jiangxinru6\Desktop\wurenji\leak_v10_fingerprint_results\v10_report.txt
PS C:\Users\jiangxinru6\Desktop\wurenji> 
