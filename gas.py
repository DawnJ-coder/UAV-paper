第一步：文件格式、标签、NPZ字段和v9参数预检
  factory_A: n=30, TRUE=15, FALSE=15, NPZ=30/30
  factory_B: n=38, TRUE=19, FALSE=19, NPZ=38/38
  factory_C: n=38, TRUE=19, FALSE=19, NPZ=38/38
  factory_D: n=40, TRUE=20, FALSE=20, NPZ=40/40
  factory_G: n=10, TRUE=5, FALSE=5, NPZ=10/10
  v9参数不一致项: 0
第二步：读取每个场景v9残差并构造指纹
  factory_A: 成功=30, TRUE=15, FALSE=15
  factory_B: 成功=38, TRUE=19, FALSE=19
  factory_C: 成功=38, TRUE=19, FALSE=19
  factory_D: 成功=40, TRUE=20, FALSE=20
  factory_G: 成功=10, TRUE=5, FALSE=5
第三步：每个场景独立执行TRUE-FALSE，提取本场景泄漏候选成分
  factory_A: TRUE高于FALSE的稳定频率维度=179
  factory_B: TRUE高于FALSE的稳定频率维度=215
  factory_C: TRUE高于FALSE的稳定频率维度=163
  factory_D: TRUE高于FALSE的稳定频率维度=174
  factory_G: TRUE高于FALSE的稳定频率维度=19
第四步：寻找所有场景严格公共成分和多数公共成分
  spectral: 严格公共=6, 严格评分使用=6, 多数公共=112, 多数评分使用=100
  temporal: 严格公共=1, 严格评分使用=1, 多数公共=59, 多数评分使用=40
  spatial: 严格公共=0, 严格评分使用=0, 多数公共=6, 多数评分使用=6
第五步：两个场景发现公共成分，第三场景验证
  test=factory_A: overall_AUC=0.9289, spectral=0.6311, temporal=0.8622, spatial=0.5000, bal_acc@0=0.8000
  test=factory_B: overall_AUC=0.8476, spectral=0.8670, temporal=0.8393, spatial=0.5000, bal_acc@0=0.8421
  test=factory_C: overall_AUC=0.9584, spectral=1.0000, temporal=0.7119, spatial=0.5000, bal_acc@0=0.8421
  test=factory_D: overall_AUC=1.0000, spectral=1.0000, temporal=1.0000, spatial=0.5000, bal_acc@0=1.0000
  test=factory_G: overall_AUC=0.4400, spectral=0.4800, temporal=0.4800, spatial=0.2000, bal_acc@0=0.4000

v12处理完成
输出目录: C:\Users\jiangxinru6\Desktop\wurenji\leak_v12_shared_component_results2
场景内差异: C:\Users\jiangxinru6\Desktop\wurenji\leak_v12_shared_component_results2\v12_scene_internal_contrasts.csv
公共属性: C:\Users\jiangxinru6\Desktop\wurenji\leak_v12_shared_component_results2\v12_cross_scene_common_attributes.csv
公共泄漏频谱: C:\Users\jiangxinru6\Desktop\wurenji\leak_v12_shared_component_results2\v12_shared_leak_spectrum.csv
留一场景验证: C:\Users\jiangxinru6\Desktop\wurenji\leak_v12_shared_component_results2\v12_loso_validation_summary.csv
报告: C:\Users\jiangxinru6\Desktop\wurenji\leak_v12_shared_component_results2\v12_report.txt
