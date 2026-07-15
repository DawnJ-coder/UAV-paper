{
  "scene_A": {
    "name": "factory_A",
    "v9_dir": "C:\\Users\\jiangxinru6\\Desktop\\wurenji\\factory_A_v9_results"
  },
  "scene_B": {
    "name": "factory_B",
    "v9_dir": "C:\\Users\\jiangxinru6\\Desktop\\wurenji\\factory_B_v9_results"
  },
  "lab_wav_paths": [
    "C:\\Users\\jiangxinru6\\Desktop\\wurenji\\laboratory_real_leak.wav"
  ],
  "output_dir": "C:\\Users\\jiangxinru6\\Desktop\\wurenji\\AB_lab_commonality_results",

  "residual_variant": "median",
  "factory_representation": "excess_power",

  "freq_low_hz": 20000.0,
  "freq_high_hz": 80000.0,
  "n_spectral_bins": 256,
  "n_temporal_quantiles": 64,
  "spectral_smooth_bins": 5,

  "bootstrap_iterations": 300,
  "bootstrap_positive_probability": 0.90,
  "min_robust_effect": 0.25,
  "random_state": 42,

  "lab_segment_seconds": 1.0,
  "lab_segment_hop_seconds": 0.5,
  "lab_nperseg": 4096,
  "lab_hop_length": 2048,
  "lab_nfft": 4096,
  "min_lab_segments": 3,

  "lab_active_spectrum_quantile": 0.55,
  "lab_min_segment_support": 0.60,

  "common_weight_quantile": 0.35,
  "minimum_common_bandwidth_hz": 700.0,
  "merge_gap_hz": 500.0,
  "coarse_bandwidth_hz": 5000.0,

  "envelope_window_bins": 41,
  "envelope_polyorder": 2,
  "broadband_common_weight": 0.70,
  "local_shape_common_weight": 0.30,
  "save_figures": true
}
