import os
import glob
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
from scipy.optimize import curve_fit

# ==========================================
# 1. 基础配置与数据读取 (保持和你之前一致)
# ==========================================
data_dir = r"D:\gas\beamform_results_offset_multiple\HM20260626_142938.ld"

direction_angles = {
    'up': np.pi / 2, 'down': -np.pi / 2, 'left': np.pi, 'right': 0,
    'up_left': 3 * np.pi / 4, 'down_left': -3 * np.pi / 4,
    'up_right': np.pi / 4, 'down_right': -3 * np.pi / 4
}
distances = [5, 10, 15, 20, 25, 30, 35, 40]

def get_high_freq_energy(file_path, cutoff_freq=20000):
    if not os.path.exists(file_path): return None
    sr, y = wav.read(file_path)
    if len(y.shape) > 1: y = y[:, 0]
    y = y / (np.max(np.abs(y)) + 1e-6)
    if sr / 2 <= cutoff_freq: cutoff_freq = sr / 2 * 0.8 
    b, a = signal.butter(4, cutoff_freq / (sr / 2), btype='high')
    filtered_y = signal.filtfilt(b, a, y)
    return np.sqrt(np.mean(filtered_y**2))

# 收集原始数据
plot_x, plot_y, plot_energy = [], [], []

# 读取方向背景 (40cm)
direction_bg = {}
for direction in direction_angles.keys():
    pattern = os.path.join(data_dir, f"*00_00d40_{direction}*.wav")
    files = glob.glob(pattern)
    direction_bg[direction] = get_high_freq_energy(files[0]) if files else 0

# 读取所有点并进行定向去噪
for direction, angle in direction_angles.items():
    my_bg = direction_bg[direction]
    for d in distances:
        pattern = os.path.join(data_dir, f"*00_00d{d}_{direction}*.wav")
        files = glob.glob(pattern)
        if files:
            e = get_high_freq_energy(files[0])
            if e is not None:
                pure_energy = max(e - my_bg, 0)
                plot_x.append(d * np.cos(angle))
                plot_y.append(d * np.sin(angle))
                plot_energy.append(pure_energy)

# 读取并添加中心点 (0,0)
center_files = glob.glob(os.path.join(data_dir, "*00*.wav"))
actual_center = [f for f in center_files if not any(f"00d{d}" in f for d in distances)]
if actual_center:
    c_e = get_high_freq_energy(actual_center[0])
    best_bg = np.min([v for v in direction_bg.values() if v > 0]) if direction_bg else 0
    plot_x.append(0)
    plot_y.append(0)
    plot_energy.append(max(c_e - best_bg, 0))

X = np.array(plot_x)
Y = np.array(plot_y)
Z = np.array(plot_energy)

# ==========================================
# 2. 核心高级算法：二维非对称高斯曲面拟合
# ==========================================
def gaussian_2d(coords, x0, y0, sigma_x, sigma_y, amplitude, offset):
    """定义二维高斯数学模型公式"""
    x, y = coords
    # 为了防止除以0，限制sigma的下限
    sigma_x = max(sigma_x, 1.0)
    sigma_y = max(sigma_y, 1.0)
    inner = ((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2)
    return amplitude * np.exp(-inner) + offset

# 给出拟合的初始猜测值 [x0, y0, sigma_x, sigma_y, amplitude, offset]
# 我们猜测山顶在 (0,0) 附近，山体胖瘦大概 15cm
initial_guess = (0.0, 0.0, 15.0, 15.0, np.max(Z), np.min(Z))

# 设定参数边界，防止数学计算飞掉
# 限制寻找的山顶 (x0, y0) 必须在 [-40, 40] 厘米的空间矩阵内
bounds = (
    [-40, -40, 1.0, 1.0, 0, 0],  # 最小值边界
    [40, 40, 50.0, 50.0, 1.0, 1.0] # 最大值边界
)

print("\n正在进行全局空间拓扑曲面拟合寻优...")
try:
    # 利用最小二乘法，让标准高斯曲面去拟合这 65 个散点
    popt, _ = curve_fit(gaussian_2d, (X, Y), Z, p0=initial_guess, bounds=bounds)
    
    # 提取拟合出的完美山顶坐标
    calc_x, calc_y, size_x, size_y, amp, off = popt
    
    print("=" * 60)
    print("【全局数学曲面定位成功】")
    print(f"拟合出的唯一实体声源质心坐标 (X, Y): ({calc_x:.2f}, {calc_y:.2f}) cm")
    print(f"气流在 X 轴向扩散半径: {size_x:.1f} cm, 在 Y 轴向扩散半径: {size_y:.1f} cm")
    
    # 安全验证机制：如果拟合出来的山峰高度太低，说明全图都是平的，根本没漏气
    if amp < 0.0005: 
        print("【系统判定】：全局声能起伏过小，判定为【环境背景杂音】，无泄漏。")
    else:
        print("【系统判定】：空间特征符合标准点源辐射模型，确认为【真实气体泄漏】！")
    print("=" * 60)

except Exception as e:
    print(f"拟合计算失败: {e}。请检查数据完整性。")