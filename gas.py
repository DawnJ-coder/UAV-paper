{
  "scenes": [
    {
      "name": "factory_A",
      "group": "AB",
      "v9_dir": "D:\\wurenji\\factory_A_v9_results"
    },
    {
      "name": "factory_B",
      "group": "AB",
      "v9_dir": "D:\\wurenji\\factory_B_v9_results"
    },
    {
      "name": "factory_C",
      "group": "CD",
      "v9_dir": "D:\\wurenji\\factory_C_v9_results"
    },
    {
      "name": "factory_D",
      "group": "CD",
      "v9_dir": "D:\\wurenji\\factory_D_v9_results"
    }
  ],
  "lab_wavs": [
    "D:\\wurenji\\0.1mm"
  ],
  "output_dir": "D:\\wurenji\\v14_plane_20_80kHz_results",
  "algorithm": {
    "freq_low_hz": 20000.0,
    "freq_high_hz": 80000.0,
    "nperseg": 4096,
    "hop_length": 2048,
    "nfft": 4096,
    "n_spectral_bins": 256,
    "spectral_smooth_bins": 7,
    "lab_segment_seconds": 1.0,
    "lab_segment_hop_seconds": 1.0,
    "lab_min_segment_seconds": 0.4,
    "methods": [
      "plane"
    ],
    "max_frequency_shift_hz": 1500.0,
    "wasserstein_scale_hz": 5000.0,
    "min_abs_effect": 0.35,
    "min_directional_auc": 0.58,
    "lab_anchor_floor": 0.25,
    "lab_strict_quantile": 0.6,
    "band_min_bins": 3,
    "band_relative_threshold": 0.15,
    "minimum_method_coverage": 0.8,
    "require_all_methods": true,
    "random_state": 42
  }
}
