二、预测层面差异
------------------------------------------------------------------------------------------------------------------------
预测差异表: C:\Users\jiangxinru6\Desktop\wurenji\leak_v11_144226_difference_analysis_results\v11_144226_prediction_difference.csv
  144226 TRUE平均概率: 0.5618
  144226 FALSE平均概率: 0.4443
  144226 TRUE-FALSE概率差: 0.1175
  144226 灰区比例[0.4,0.6]: 0.2632
  144226 center配对排序正确率: 0.7368421052631579
  144226 配对失败center: 14 | 15 | 16 | 17 | 18


三、整体特征分布偏移：144226 哪些特征整体和别人不一样
------------------------------------------------------------------------------------------------------------------------
分布偏移表: C:\Users\jiangxinru6\Desktop\wurenji\leak_v11_144226_difference_analysis_results\v11_144226_feature_shift_vs_others.csv
  偏移最大的前15个特征:
    time_energy_kurtosis: shift=5.614, |d|=3.713, KS=0.963, 144226均值=7.25123, 其他均值=4.08247
    time_energy_max_mean_ratio: shift=3.743, |d|=1.771, KS=1.000, 144226均值=1.42841, 其他均值=1.75866
    time_energy_cv: shift=3.636, |d|=1.830, KS=0.954, 144226均值=0.115974, 其他均值=0.20632
    spec_centroid_hz: shift=2.734, |d|=1.149, KS=0.731, 144226均值=26481.9, 其他均值=26047.2
    hm_compactness_p95: shift=2.296, |d|=1.201, KS=0.554, 144226均值=0.138446, 其他均值=0.420474
    hm_p90_elongation: shift=2.283, |d|=1.289, KS=0.495, 144226均值=1.24703, 其他均值=1.68278
    spec_peak_freq_hz: shift=2.227, |d|=0.696, KS=0.719, 144226均值=26299.3, 其他均值=25599.4
    time_rms: shift=2.051, |d|=0.255, KS=0.815, 144226均值=0.0986256, 其他均值=0.0900572
    energy_60_70k: shift=2.032, |d|=0.625, KS=0.620, 144226均值=1.38257e-06, 其他均值=1.08718e-06
    high_freq_ratio: shift=2.025, |d|=0.700, KS=0.568, 144226均值=0.0102902, 其他均值=0.0122504
    spec_flatness: shift=1.993, |d|=0.520, KS=0.574, 144226均值=0.0317854, 其他均值=0.0356383
    ratio_40_50k: shift=1.982, |d|=0.888, KS=0.505, 144226均值=0.00620388, 其他均值=0.0075185
    spec_bandwidth_hz: shift=1.940, |d|=0.931, KS=0.398, 144226均值=3343.49, 其他均值=3624.5
    hm_p95_elongation: shift=1.927, |d|=1.040, KS=0.452, 144226均值=1.33647, 其他均值=1.82067
    energy_20_30k: shift=1.842, |d|=0.080, KS=0.815, 144226均值=0.00900353, 其他均值=0.00962612
四、TRUE/FALSE关系：哪些特征在144226里方向和别人相反
------------------------------------------------------------------------------------------------------------------------
方向对比表: C:\Users\jiangxinru6\Desktop\wurenji\leak_v11_144226_difference_analysis_results\v11_144226_label_separation_vs_others.csv
方向翻转特征表: C:\Users\jiangxinru6\Desktop\wurenji\leak_v11_144226_difference_analysis_results\v11_144226_direction_flip_features.csv
  方向翻转特征数量: 48
  方向翻转前15:
    ratio_30_40k__time_robust_z: 144226=TRUE<FALSE, 其他多数=TRUE>FALSE, 144226_AUC=1.000, 144226_diff=-1.00663, 其他平均diff=0.186771
    ratio_30_40k: 144226=TRUE<FALSE, 其他多数=TRUE>FALSE, 144226_AUC=1.000, 144226_diff=-0.073355, 其他平均diff=0.0127616
    ratio_20_30k__time_robust_z: 144226=TRUE>FALSE, 其他多数=TRUE<FALSE, 144226_AUC=1.000, 144226_diff=1.00778, 其他平均diff=-0.250135
    ratio_20_30k: 144226=TRUE>FALSE, 其他多数=TRUE<FALSE, 144226_AUC=1.000, 144226_diff=0.0745673, 其他平均diff=-0.0143786
    spec_bandwidth_hz: 144226=TRUE<FALSE, 其他多数=TRUE>FALSE, 144226_AUC=1.000, 144226_diff=-461.064, 其他平均diff=357.106
    spec_bandwidth_hz__time_robust_z: 144226=TRUE<FALSE, 其他多数=TRUE>FALSE, 144226_AUC=1.000, 144226_diff=-1.03955, 其他平均diff=0.398247
    spec_entropy__time_robust_z: 144226=TRUE<FALSE, 其他多数=TRUE>FALSE, 144226_AUC=1.000, 144226_diff=-0.937137, 其他平均diff=0.981329
    spec_entropy: 144226=TRUE<FALSE, 其他多数=TRUE>FALSE, 144226_AUC=1.000, 144226_diff=-0.0218372, 其他平均diff=0.0240442
    spec_rolloff_85_hz: 144226=TRUE<FALSE, 其他多数=TRUE>FALSE, 144226_AUC=1.000, 144226_diff=-1796.05, 其他平均diff=687.706
    spec_rolloff_85_hz__time_robust_z: 144226=TRUE<FALSE, 其他多数=TRUE>FALSE, 144226_AUC=1.000, 144226_diff=-0.917743, 其他平均diff=0.176935
    spec_centroid_hz: 144226=TRUE<FALSE, 其他多数=TRUE>FALSE, 144226_AUC=1.000, 144226_diff=-300.177, 其他平均diff=-63.151
    spec_centroid_hz__time_robust_z: 144226=TRUE<FALSE, 其他多数=TRUE>FALSE, 144226_AUC=1.000, 144226_diff=-1.04773, 其他平均diff=0.132944
    spec_rolloff_85_hz__time_rank_pct: 144226=TRUE<FALSE, 其他多数=TRUE>FALSE, 144226_AUC=1.000, 144226_diff=-0.5, 其他平均diff=0.138597
    spec_entropy__time_rank_pct: 144226=TRUE<FALSE, 其他多数=TRUE>FALSE, 144226_AUC=1.000, 144226_diff=-0.5, 其他平均diff=0.499077
    spec_centroid_hz__time_rank_pct: 144226=TRUE<FALSE, 其他多数=TRUE>FALSE, 144226_AUC=1.000, 144226_diff=-0.5, 其他平均diff=0.106484

五、center层面：异常是否集中在 center_14~18
------------------------------------------------------------------------------------------------------------------------
center配对异常表: C:\Users\jiangxinru6\Desktop\wurenji\leak_v11_144226_difference_analysis_results\v11_144226_center_pair_anomaly.csv
  center异常前10:
    center_18: 14-18=1, prob_diff=-0.4185714285714286, order_correct=0, max_z=9.794
    center_17: 14-18=1, prob_diff=-0.41714285714285715, order_correct=0, max_z=9.252
    center_16: 14-18=1, prob_diff=-0.38857142857142857, order_correct=0, max_z=6.724
    center_15: 14-18=1, prob_diff=-0.32, order_correct=0, max_z=4.506
    center_14: 14-18=1, prob_diff=-0.12428571428571433, order_correct=0, max_z=4.481
    center_08: 14-18=0, prob_diff=0.49142857142857144, order_correct=1, max_z=8.896
    center_12: 14-18=0, prob_diff=0.4342857142857144, order_correct=1, max_z=7.639
    center_07: 14-18=0, prob_diff=0.23714285714285716, order_correct=1, max_z=7.223
    center_10: 14-18=0, prob_diff=0.01571428571428579, order_correct=1, max_z=4.979
    center_01: 14-18=0, prob_diff=0.2542857142857143, order_correct=1, max_z=4.918

令行摘要:
  144226 TRUE-FALSE概率差: 0.1175
  144226 灰区比例[0.4,0.6]: 0.2632
  144226 center配对排序正确率: 0.7368421052631579
  144226 配对失败center: 14 | 15 | 16 | 17 | 18


  144226整体分布偏移最大的前8个特征:
    time_energy_kurtosis: shift=5.614, |d|=3.713, 144226均值=7.25123, 其他均值=4.08247
    time_energy_max_mean_ratio: shift=3.743, |d|=1.771, 144226均值=1.42841, 其他均值=1.75866
    time_energy_cv: shift=3.636, |d|=1.830, 144226均值=0.115974, 其他均值=0.20632
    spec_centroid_hz: shift=2.734, |d|=1.149, 144226均值=26481.9, 其他均值=26047.2
    hm_compactness_p95: shift=2.296, |d|=1.201, 144226均值=0.138446, 其他均值=0.420474
    hm_p90_elongation: shift=2.283, |d|=1.289, 144226均值=1.24703, 其他均值=1.68278
    spec_peak_freq_hz: shift=2.227, |d|=0.696, 144226均值=26299.3, 其他均值=25599.4
    time_rms: shift=2.051, |d|=0.255, 144226均值=0.0986256, 其他均值=0.0900572

  方向翻转特征数量: 48
    ratio_30_40k__time_robust_z: 144226=TRUE<FALSE, 其他多数=TRUE>FALSE, 144226_AUC=1.000
    ratio_30_40k: 144226=TRUE<FALSE, 其他多数=TRUE>FALSE, 144226_AUC=1.000
    ratio_20_30k__time_robust_z: 144226=TRUE>FALSE, 其他多数=TRUE<FALSE, 144226_AUC=1.000
    ratio_20_30k: 144226=TRUE>FALSE, 其他多数=TRUE<FALSE, 144226_AUC=1.000
    spec_bandwidth_hz: 144226=TRUE<FALSE, 其他多数=TRUE>FALSE, 144226_AUC=1.000
    spec_bandwidth_hz__time_robust_z: 144226=TRUE<FALSE, 其他多数=TRUE>FALSE, 144226_AUC=1.000
    spec_entropy__time_robust_z: 144226=TRUE<FALSE, 其他多数=TRUE>FALSE, 144226_AUC=1.000
    spec_entropy: 144226=TRUE<FALSE, 其他多数=TRUE>FALSE, 144226_AUC=1.000

  center异常前8:
    center_18: 14-18=1, prob_diff=-0.4185714285714286, order_correct=0, max_z=9.794
    center_17: 14-18=1, prob_diff=-0.41714285714285715, order_correct=0, max_z=9.252
    center_16: 14-18=1, prob_diff=-0.38857142857142857, order_correct=0, max_z=6.724
    center_15: 14-18=1, prob_diff=-0.32, order_correct=0, max_z=4.506
    center_14: 14-18=1, prob_diff=-0.12428571428571433, order_correct=0, max_z=4.481
    center_08: 14-18=0, prob_diff=0.49142857142857144, order_correct=1, max_z=8.896
    center_12: 14-18=0, prob_diff=0.4342857142857144, order_correct=1, max_z=7.639
    center_07: 14-18=0, prob_diff=0.23714285714285716, order_correct=1, max_z=7.223



