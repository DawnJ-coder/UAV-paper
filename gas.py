PS C:\Users\jiangxinru6\Desktop\wurenji> python leak_v8.py --config AB_lab_config.json
====================================================================================================
AB场景共性 + 实验室真实泄漏锚定分析
====================================================================================================
场景A: C:\Users\jiangxinru6\Desktop\wurenji\factory_A_v9_results
场景B: C:\Users\jiangxinru6\Desktop\wurenji\factory_B_v9_results
实验室WAV: ['C:\\Users\\jiangxinru6\\Desktop\\0.1mm铜管泄漏_150kPa_240sccm_1.0m_null_136c_bf.wav']
v9残差: representation=excess_power, variant=median

第一步：读取A、B的v9残差
  factory_A: n=30, TRUE=15, FALSE=15
  factory_B: n=38, TRUE=19, FALSE=19

第二步：分别计算A_TRUE-A_FALSE、B_TRUE-B_FALSE
  A稳定TRUE增强频谱维度: 162
  B稳定TRUE增强频谱维度: 213

第三步：提取A与B共同部分
  AB严格共同频谱维度: 150
  A/B频谱余弦相似度: 0.4247

第四步：读取实验室真实泄漏WAV并建立原型
leak_v8.py:732: WavFileWarning: Chunk (non-data) not understood, skipping it.
  fs, raw = wavfile.read(str(path))
  实验室有效切片数: 39

第五步：寻找AB共同部分与实验室泄漏的共同部分
  ab_vs_lab_cosine_fine: 0.28230215439035883
  ab_vs_lab_overlap_fine: 0.3861313790406235
  ab_vs_lab_js_similarity_fine: 0.5242232884851563
  ab_vs_lab_shape_correlation: -0.05667746745139286
  ab_vs_lab_cosine_coarse: 0.7652458371788806
  ab_vs_lab_overlap_coarse: 0.7123652835996123
  A_vs_B_cosine: 0.42469257010324674
  A_vs_B_overlap: 0.41122971638374717
  A_vs_B_js_similarity: 0.5853634568170099
  n_AB_strict_bins: 150
  n_lab_active_bins: 114
  n_final_selected_bins: 24
  n_final_bands: 7

====================================================================================================
处理完成
输出目录: C:\Users\jiangxinru6\Desktop\wurenji\AB_lab_commonality_results
AB共同频谱: C:\Users\jiangxinru6\Desktop\wurenji\AB_lab_commonality_results\AB_common_spectrum.csv
AB与实验室共同频谱: C:\Users\jiangxinru6\Desktop\wurenji\AB_lab_commonality_results\AB_lab_common_spectrum.csv
共同频带: C:\Users\jiangxinru6\Desktop\wurenji\AB_lab_commonality_results\AB_lab_common_frequency_bands.csv
报告: C:\Users\jiangxinru6\Desktop\wurenji\AB_lab_commonality_results\AB_lab_commonality_report.txt
PS C:\Users\jiangxinru6\Desktop\wurenji> 
