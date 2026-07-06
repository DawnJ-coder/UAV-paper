# ===================================================
# 新方案：方向级单点差分滤波 (定向去噪)
# ===================================================
# 核心思想：既然只有 down 方向的 40cm 异常，那其他方向的 40cm 依然是干净的背景。
# 我们让每个点，只减去【自己所属方向】的 40cm 能量，实行“精准定点去噪”。

# 1. 先用一个字典存下每个方向各自 40cm 的原始能量
direction_bg = {}
for direction in direction_angles.keys():
    pattern = os.path.join(data_dir, f"*00_00d40_{direction}*.wav")
    files = glob.glob(pattern)
    if files:
        e = get_high_freq_energy(files[0])
        direction_bg[direction] = e if e is not None else 0
    else:
        direction_bg[direction] = 0

# 2. 计算所有周边的净能量
for direction, angle in direction_angles.items():
    # 获取该方向自己的背景噪声
    my_bg = direction_bg[direction]
    
    for d in distances:
        pattern = os.path.join(data_dir, f"*00_00d{d}_{direction}*.wav")
        files = glob.glob(pattern)
        
        if files:
            e = get_high_freq_energy(files[0])
            if e is not None:
                # 【新滤波器】：只减去自己方向的背景
                pure_energy = max(e - my_bg, 0)
                
                # 如果这个方向 40cm 异常高（比如 down），减完后整个方向就会变成安全的深蓝色
                x = d * np.cos(angle)
                y = d * np.sin(angle)
                plot_x.append(x)
                plot_y.append(y)
                plot_energy.append(pure_energy)

# 3. 中心点则减去【所有方向中，最干净、能量最低的那个40cm】作为最保守的背景基准
if len(plot_energy) > 0 and center_energy > 0:
    valid_bgs = [v for v in direction_bg.values() if v > 0]
    best_bg = np.min(valid_bgs) if valid_bgs else 0
    plot_energy[0] = max(center_energy - best_bg, 0)