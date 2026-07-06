import os
import glob
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# ==========================================
# 1. 配置参数
# ==========================================
# 你的数据所在的文件夹路径
data_dir = r"D:\gas\beamform_results_offset_multiple\HM20260626_142938.ld"

# 8个方向对应的极坐标角度（弧度）
# 上、下、左、右，左上、左下、右上、右下
direction_angles = {
    'up': np.pi / 2,         # 90度
    'down': -np.pi / 2,      # 270度
    'left': np.pi,           # 180度
    'right': 0,              # 0度
    'up_left': 3 * np.pi / 4, # 135度
    'down_left': -3 * np.pi / 4, # 225度
    'up_right': np.pi / 4,    # 45度
    'down_right': -np.pi / 4  # 315度
}

# 8个距离 (cm)
distances = [5, 10, 15, 20, 25, 30, 35, 40]

# ==========================================
# 2. 核心算法：提取高频能量的函数
# ==========================================
def get_high_freq_energy(file_path, cutoff_freq=20000):
    """读取WAV文件，对信号进行高通滤波，计算其高频能量"""
    if not os.path.exists(file_path):
        return None
    
    # 读取音频
    sr, y = wav.read(file_path)
    
    # 如果是双声道，取单通道
    if len(y.shape) > 1:
        y = y[:, 0]
        
    # 归一化
    y = y / (np.max(np.abs(y)) + 1e-6)
    
    # 设计高通滤波器（由于泄漏主要在高频/超声，默认切在20kHz）
    # 如果你的采样率SR不够两倍的20kHz，会自动调整
    if sr / 2 <= cutoff_freq:
        cutoff_freq = sr / 2 * 0.8 
        
    b, a = signal.butter(4, cutoff_freq / (sr / 2), btype='high')
    filtered_y = signal.filtfilt(b, a, y)
    
    # 计算能量 (均方根 RMS)
    energy = np.sqrt(np.mean(filtered_y**2))
    return energy

# ==========================================
# 3. 自动扫描文件夹并收集所有点的坐标和能量
# ==========================================
# 用来存放所有点的直角坐标 (X, Y) 和对应的能量值
plot_x = []
plot_y = []
plot_energy = []

# --- 3.1 首先读取中心点 (00) 的能量 ---
# 假设中心点的文件名固定包含 "00_center" 或者直接模糊匹配没有方向的中心文件
# 这里根据你提供的中心点描述，寻找中心点文件
center_pattern = os.path.join(data_dir, "*_00_beamform_result.wav") # 根据实际中心文件名可微调
center_files = glob.glob(center_pattern)
if not center_files:
    # 如果找不到，尝试匹配任意一个不带距离方向的00文件
    center_files = glob.glob(os.path.join(data_dir, "*00*.wav"))

# 如果找到了中心点
center_energy = 0
if center_files:
    # 排除掉带d5, d10等周边文件，精准定位中心
    actual_center = [f for f in center_files if not any(f"00d{d}" in f for d in distances)]
    if actual_center:
        center_energy = get_high_freq_energy(actual_center[0])
        # 中心点坐标为 (0, 0)
        plot_x.append(0)
        plot_y.append(0)
        plot_energy.append(center_energy)

# --- 3.2 循环读取 64 个周边点的能量 ---
# 用来存 40cm 处所有方向的能量，最后取平均作为背景噪声
bg_energies = []

# 第一次循环：先找出 40cm 处的背景噪声基准
for direction, angle in direction_angles.items():
    # 匹配 40cm 的文件，例如: *00_00d40_down*.wav
    pattern = os.path.join(data_dir, f"*00_00d40_{direction}*.wav")
    files = glob.glob(pattern)
    if files:
        e = get_high_freq_energy(files[0])
        if e is not None:
            bg_energies.append(e)

# 计算 40cm 处的平均背景噪声
bg_noise_baseline = np.mean(bg_energies) if bg_energies else 0
print(f"--- 成功计算背景噪声基准 (40cm平均能量): {bg_noise_baseline:.6f} ---")

# 第二次循环：计算所有点减去背景噪声后的净能量，并转换成直角坐标 (X, Y)
for direction, angle in direction_angles.items():
    for d in distances:
        pattern = os.path.join(data_dir, f"*00_00d{d}_{direction}*.wav")
        files = glob.glob(pattern)
        
        if files:
            e = get_high_freq_energy(files[0])
            if e is not None:
                # 【关键空间滤波】：减去 40cm 的背景噪声
                # 使用 max(..., 0) 确保减完不会变成负数
                pure_energy = max(e - bg_noise_baseline, 0)
                
                # 极坐标转直角坐标 (X = R*cos(theta), Y = R*sin(theta))
                x = d * np.cos(angle)
                y = d * np.sin(angle)
                
                plot_x.append(x)
                plot_y.append(y)
                plot_energy.append(pure_energy)

# 更新中心点的净能量（中心点也减去背景）
if len(plot_energy) > 0 and center_energy > 0:
    plot_energy[0] = max(center_energy - bg_noise_baseline, 0)

# ==========================================
# 4. 2D 空间插值与热力图绘制
# ==========================================
plot_x = np.array(plot_x)
plot_y = np.array(plot_y)
plot_energy = np.array(plot_energy)

# 创建更密集的网格用于平滑画图
grid_x, grid_y = np.mgrid[-45:45:200j, -45:45:200j]

# 使用立方插值法把 65 个点转换成平滑的 2D 图像
grid_z = griddata((plot_x, plot_y), plot_energy, (grid_x, grid_y), method='cubic')

# 开始画图
plt.figure(figsize=(8, 7), dpi=100)
# 画出热力图
plt.imshow(grid_z.T, extent=(-45, 45, -45, 45), origin='lower', cmap='jet')
plt.colorbar(label='Pure Leakage Energy (De-noised)')

# 散点标出实际采集点的位置
plt.scatter(plot_x, plot_y, c='white', s=15, edgecolors='black', alpha=0.6, label='Microphone Focal Points')
# 特别标注出真实泄漏中心 (0,0)
plt.scatter(0, 0, c='red', marker='*', s=200, edgecolors='yellow', label='True Leak Source (0,0)')

plt.title("2D Spatial Acoustic Feature Heatmap\n(Background Subtracted by 40cm Group)", fontsize=12)
plt.xlabel("X Distance (cm)")
plt.ylabel("Y Distance (cm)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')
plt.show()