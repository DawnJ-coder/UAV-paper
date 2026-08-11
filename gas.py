{
  "lab_groups": {
    "0.1mm": [
      "C:\\Users\\jiangxinru6\\Desktop\\0.1mm"
    ],
    "0.5mm": [
      "C:\\Users\\jiangxinru6\\Desktop\\0.5mm"
    ]
  },
  "factory_datasets": [
    {
      "name": "TRUE_center",
      "default_label": "T",
      "center_root_dir": "D:\\gas\\beamform_results",
      "offset_root_dirs": [
        "D:\\gas\\beamform_results_offset_multiple"
      ],
      "time_folders": [],
      "label_by_folder": {}
    },
    {
      "name": "FALSE_center",
      "default_label": "F",
      "center_root_dir": "D:\\gas\\beamform_results_cs",
      "offset_root_dirs": [
        "D:\\gas\\beamform_results_offset_multiple_cs"
      ],
      "time_folders": [],
      "label_by_folder": {}
    }
  ],
  "pairing_remove_tokens": [
    "_null",
    "null"
  ],
  "output_dir": "D:\\gas\\spatial_local_excess_leak_separation_v3_1_2_fixed_results_tz",
  "algorithm": {
    "freq_low_hz": 50000.0,
    "freq_high_hz": 70000.0,
    "expected_sample_rate_hz": 192000,
    "time_slice_seconds": 1.0,
    "segment_mode": "indexed",
    "strict_time_slice": true,
    "nperseg": 4096,
    "hop_length": 2048,
    "nfft": 4096,
    "minimum_frequency_bins": 50,
    "leak_components": 8,
    "lab_train_fraction": 0.7,
    "random_state": 42,
    "max_lab_seconds_per_file": 30.0,
    "max_lab_training_frames": 20000,
    "nmf_max_iter": 350,
    "lab_projection_iterations": 180,
    "lab_projection_l1_ratio": 0.002,
    "spatial_min_distance_cm": 5.0,
    "spatial_max_distance_cm": 80.0,
    "min_neighbors": 64,
    "min_complete_opposite_pairs": 20,
    "min_complete_axes": 3,
    "pair_support_threshold_db": 1.0,
    "excess_base_threshold_db": 1.0,
    "uncertainty_threshold_weight": 0.35,
    "excess_db_softness": 1.0,
    "pair_support_midpoint": 0.55,
    "axis_support_midpoint": 0.5,
    "support_softness": 0.12,
    "prediction_uncertainty_ref_db": 4.0,
    "prediction_confidence_floor": 0.35,
    "lab_frame_cosine_midpoint": 0.76,
    "lab_frame_cosine_softness": 0.06,
    "lab_gate_floor": 0.35,
    "lab_gate_exponent": 0.35,
    "mask_frequency_median_bins": 3,
    "mask_temporal_median_frames": 3,
    "evidence_score_threshold": 0.22,
    "evidence_axis_support_threshold": 0.5,
    "evidence_lab_similarity_threshold": 0.7,
    "save_plots": true,
    "save_npz": true,
    "save_wavs": true,
    "plot_limit": 100,
    "lab_split_mode": "cross_group",
    "lab_train_groups": [
      "0.5mm"
    ],
    "lab_validation_groups": [
      "0.1mm"
    ]
  },
  "program_notes": {
    "folder_scan": "递归扫描中心根目录下所有真正包含中心WAV的工况目录",
    "unicode_output": "输出目录保留快插、螺纹等中文类型名称",
    "stale_results": "使用新的输出目录，避免旧版错误目录与新结果混在一起"
  }
}
