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
