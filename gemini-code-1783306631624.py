# ===================================================
# 核心算法：全自动空间峰值定位与指向性分析
# ===================================================

# 1. 找出方向去噪后的最大能量点
max_idx = np.argmax(plot_energy)
max_x = plot_x[max_idx]
max_y = plot_y[max_idx]
max_val = plot_energy[max_idx]

print("=" * 50)
print(f"【自动定位结果】系统检测到能量最高点坐标为: ({max_x:.1f}, {max_y:.1f}) cm")
print(f"该点去噪后的绝对高频能量为: {max_val:.6f}")

# 2. 误报过滤验证：检查最高点是否为“局部火山中心”
# 统计距离该中心最邻近的一圈（5cm内）的平均能量
near_mask = (np.sqrt((plot_x - max_x)**2 + (plot_y - max_y)**2) <= 6) & (plot_x != max_x)
if np.any(near_mask):
    near_avg_energy = np.mean(plot_energy[near_mask])
    # 计算能量陡降比（中心能量是否远大于周围临近能量）
    drop_ratio = max_val / (near_avg_energy + 1e-12)
    print(f"【空间梯度验证】中心与邻域能量比 (Drop Ratio): {drop_ratio:.2f}")
    
    if drop_ratio > 1.2: # 阈值可根据实际调整
        print(">>> 空间梯度验证通过：确认为【真实气流泄漏源】！")
    else:
        print(">>> 警告：空间梯度过平缓，可能是全局环境背景噪声抬升，判定为虚警。")

# 3. 喷流指向性自动分析
# 看看 8 个方向里，哪几个方向在 10cm-20cm 处依然留存有较多能量
print("-" * 50)
print("【气流喷射方向分析】:")
for direction, angle in direction_angles.items():
    # 找出该方向上距离在 5 到 20cm 之间的点
    dir_mask = (np.abs(np.arctan2(plot_y, plot_x) - angle) < 0.1) & (np.sqrt(plot_x**2 + plot_y**2) <= 20)
    if np.any(dir_mask):
        dir_energy = np.mean(plot_energy[dir_mask])
        # 如果该方向平均能量大于全局平均的一定比例，说明气流往这里喷
        if dir_energy > np.mean(plot_energy) * 1.5:
            print(f" -> 检测到高压气体正在向【 {direction} 】方向喷射 💨")
print("=" * 50)