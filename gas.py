结果判断顺序
================================================================================

1. 先看 03_false_background_metrics_summary.csv
   positive_residual_fraction_median越低越好；
   background_to_center_ratio_median应接近1；
   如果false_prediction在C/D的FALSE上也明显优于plane，才说明新背景模型可靠。

2. 再看 04_scene_method_TRUE_FALSE_separation.csv
   false_prediction需要在A/B/C/D中TRUE分数高于FALSE，并保持较高AUC。

3. 再看 08_lab_similarity_summary.csv
   residual_*：保留下来的残差与实验室泄漏的相似度；
   removed_background_*：被减掉部分与实验室泄漏的相似度。
   理想情况：residual相似度高于removed_background。

4. 最后看 05_band_effects_factory_lab_support.csv
   AB_selected=1：A与B同时支持；
   factory_support_count>=3：至少三个工厂场景支持；
   lab_supported=1：实验室多数WAV的中位频谱也支持；
   final_candidate=1：最终候选频带，但仍不能直接称为纯泄漏。

总体FALSE剩余比例中位数：false_prediction=0.0220，plane=0.3801。
初步判断：新FALSE背景模型比当前plane更能解释无泄漏中心背景。
最终候选频带数量：20

重要：不要只看A/B相似度。方法必须同时通过FALSE背景误差、C/D外部测试和实验室验证。
