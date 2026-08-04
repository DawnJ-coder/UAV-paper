PS D:\wurenji\gas0803\spatial_local_excess_leak_separation_v3_1_2_package> python spatial_local_excess_leak_separation_v3_1_2.py --config spatial_local_excess_config_v3_1_2.json
============================================================================================
空间局部超额 + 消声室跨孔径辅助验证：泄漏分离 v3.1.2
输出目录: D:\gas\spatial_local_excess_leak_separation_v3_1_2_results
主分析频带: 50000-70000 Hz
============================================================================================
[LAB] group=0.5mm, role=train, configured_paths=['D:\\请填写消声室路径\\0.5mm'], discovered_wavs=0
[LAB] group=0.1mm, role=validation, configured_paths=['D:\\请填写消声室路径\\0.1mm'], discovered_wavs=0

程序失败: RuntimeError: 没有可用的消声室训练WAV。训练组=['0.5mm']；配置路径=['D:\\请填写消声室路径\\0.5mm']；发现WAV=0个；成功读取=0个。首要错误：无具体读取错误（更可能是配置路径不存在、指错组或未发现WAV）。请
查看输出目录中的00_lab_path_diagnostics.csv、00_lab_input_scan.csv、01_lab_train_validation_split.csv和99_errors_lab.csv。
Traceback (most recent call last):
  File "spatial_local_excess_leak_separation_v3_1_2.py", line 2047, in <module>
    main()
  File "spatial_local_excess_leak_separation_v3_1_2.py", line 2042, in main
    run_analysis(cfg)
  File "spatial_local_excess_leak_separation_v3_1_2.py", line 1762, in run_analysis
    lab = prepare_lab_reference(cfg, output_dir)
  File "spatial_local_excess_leak_separation_v3_1_2.py", line 927, in prepare_lab_reference
    raise RuntimeError(
RuntimeError: 没有可用的消声室训练WAV。训练组=['0.5mm']；配置路径=['D:\\请填写消声室路径\\0.5mm']；发现WAV=0个；成功读取=0个。首要错误：无具体读取错误（更可能是配置路径不存在、指错组或未发现WAV）。请查看输出目
录中的00_lab_path_diagnostics.csv、00_lab_input_scan.csv、01_lab_train_validation_split.csv和99_errors_lab.csv。
PS D:\wurenji\gas0803\spatial_local_excess_leak_separation_v3_1_2_package> 


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
      "center_root_dir": "D:\\gas\\beamform_results_gc",
      "offset_root_dirs": [
        "D:\\gas\\beamform_results_offset_multiple_gc_0001",
        "D:\\gas\\beamform_results_offset_multiple_gc_gy_0001"
      ],
      "time_folders": [],
      "label_by_folder": {}
    },
    {
      "name": "FALSE_center",
      "default_label": "F",
      "center_root_dir": "D:\\gas\\beamform_results_gc_cs",
      "offset_root_dirs": [
        "D:\\gas\\beamform_results_offset_multiple_gc_cs_0001",
        "D:\\gas\\beamform_results_offset_multiple_gc_gy_cs_0001"
      ],
      "time_folders": [],
      "label_by_folder": {}
    }
  ],

  "pairing_remove_tokens": [
    "_null",
    "null"
  ],
  "output_dir": "D:\\gas\\spatial_local_excess_leak_separation_v3_1_2_results",
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
  }
}
