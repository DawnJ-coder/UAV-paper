第一步：文件格式、标签、NPZ字段和v9参数预检
  factory_A: n=30, TRUE=15, FALSE=15, NPZ=30/30
  factory_B: n=38, TRUE=19, FALSE=19, NPZ=38/38
  factory_C: n=38, TRUE=19, FALSE=19, NPZ=38/38
  v9参数不一致项: 0

第二步：读取每个场景v9残差并构造指纹
  factory_A: 成功=30, TRUE=15, FALSE=15
  factory_B: 成功=38, TRUE=19, FALSE=19
  factory_C: 成功=38, TRUE=19, FALSE=19

第三步：每个场景独立执行TRUE-FALSE，提取本场景泄漏候选成分
  factory_A: TRUE高于FALSE的稳定频率维度=179
  factory_B: TRUE高于FALSE的稳定频率维度=215
  factory_C: TRUE高于FALSE的稳定频率维度=163

第四步：寻找所有场景严格公共成分和多数公共成分
  spectral: 严格公共=124, 严格评分使用=100, 多数公共=233, 多数评分使用=100
  temporal: 严格公共=59, 严格评分使用=40, 多数公共=64, 多数评分使用=40
  spatial: 严格公共=6, 严格评分使用=6, 多数公共=7, 多数评分使用=7

第五步：两个场景发现公共成分，第三场景验证
  test=factory_A: overall_AUC=1.0000, spectral=1.0000, temporal=1.0000, spatial=0.9111, bal_acc@0=0.9667
  test=factory_B: overall_AUC=0.9418, spectral=0.9640, temporal=0.9723, spatial=0.9003, bal_acc@0=0.8158
  test=factory_C: overall_AUC=1.0000, spectral=0.9945, temporal=1.0000, spatial=0.9917, bal_acc@0=1.0000

====================================================================================================
v12处理完成
输出目录: C:\Users\jiangxinru6\Desktop\wurenji\leak_v12_shared_component_results
场景内差异: C:\Users\jiangxinru6\Desktop\wurenji\leak_v12_shared_component_results\v12_scene_internal_contrasts.csv
公共属性: C:\Users\jiangxinru6\Desktop\wurenji\leak_v12_shared_component_results\v12_cross_scene_common_attributes.csv
公共泄漏频谱: C:\Users\jiangxinru6\Desktop\wurenji\leak_v12_shared_component_results\v12_shared_leak_spectrum.csv
留一场景验证: C:\Users\jiangxinru6\Desktop\wurenji\leak_v12_shared_component_results\v12_loso_validation_summary.csv
报告: C:\Users\jiangxinru6\Desktop\wurenji\leak_v12_shared_component_results\v12_report.txt
