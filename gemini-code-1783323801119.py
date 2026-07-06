# ============================================================
# 气体泄漏检测高级重构版：超声纯化 + 门控乘法融合 + 真实物理梯度
# ============================================================
import os
import glob
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# ------------------------------------------------------------
# 1. 核心物理参数配置（专治低频干扰与假归一化）
# ------------------------------------------------------------
# 关键开关一：彻底关闭背景减法，拿回真实空间物理梯度原貌
USE_BACKGROUND_SUBTRACTION = False

# 关键开关二：强行切入超声波段（20kHz - 50kHz），彻底滤除低频机械内鬼
ULTRA_FMIN = 20000    # 20 kHz
ULTRA_FMAX = 50000    # 50 kHz

direction_angles = {
    'up': np.pi / 2, 'down': -np.pi / 2, 'left': np.pi, 'right': 0,
    'up_left': 3 * np.pi / 4, 'down_left': -3 * np.pi / 4,
    'up_right': np.pi / 4, 'down_right': -np.pi / 4
}
distances = [5, 10, 15, 20, 25, 30, 35, 40]


# ============================================================
# 2. 深度纯化：声学特征超声带提取函数
# ============================================================
def load_wav_raw(file_path):
    """读取音频，拒绝单波形单独归一化，保留相对空间梯度"""
    if not os.path.exists(file_path):
        return None, None
    sr, y = wav.read(file_path)
    if len(y.shape) > 1:
        y = y[:, 0]
    return sr, y.astype(float)


def get_ultra_bounds(sr):
    """动态适配奈奎斯特频率，确保不发生频谱混叠"""
    fmax = min(ULTRA_FMAX, sr / 2 * 0.95)
    return ULTRA_FMIN, fmax


def ultra_band_energy(y, sr):
    """特征1：超声带绝对RMS能量。泄漏最直接的物理尺度"""
    fmin, fmax = get_ultra_bounds(sr)
    b, a = signal.butter(6, [fmin / (sr / 2), fmax / (sr / 2)], btype='band')
    yf = signal.filtfilt(b, a, y)
    return np.sqrt(np.mean(yf ** 2)) + 1e-12


def spectral_flatness_ultra(y, sr):
    """特征2：超声带内频谱平坦度。越接近1越符合宽带湍流摩擦"""
    fmin, fmax = get_ultra_bounds(sr)
    f, Pxx = signal.welch(y, sr, nperseg=min(2048, len(y)))
    mask = (f >= fmin) & (f <= fmax)
    if not np.any(mask):
        return 0.0
    P = Pxx[mask] + 1e-12
    return np.exp(np.mean(np.log(P))) / np.mean(P)


def spectral_centroid_ultra(y, sr):
    """特征3：超声带内频谱质心。真实泄漏喷流质心会向 30kHz 以上猛拉"""
    fmin, fmax = get_ultra_bounds(sr)
    f, Pxx = signal.welch(y, sr, nperseg=min(2048, len(y)))
    mask = (f >= fmin) & (f <= fmax)
    if not np.any(mask):
        return 0.0
    return np.sum(f[mask] * Pxx[mask]) / (np.sum(Pxx[mask]) + 1e-12)


def temporal_stationarity_ultra(y, sr):
    """特征4：超声带内时域平稳性（变异系数）。连续喷流极稳，CV趋于0"""
    fmin, fmax = get_ultra_bounds(sr)
    b, a = signal.butter(4, [fmin / (sr / 2), fmax / (sr / 2)], btype='band')
    yf = signal.filtfilt(b, a, y)
    frame = 2048
    if len(yf) < frame: return 1.0
    rms = [np.sqrt(np.mean(yf[i:i + frame] ** 2)) for i in range(0, len(yf) - frame, 1024)]
    rms = np.array(rms) + 1e-12
    return np.std(rms) / np.mean(rms)


def line_spectrum_ratio_ultra(y, sr):
    """特征5：超声带内线谱占比。机械摩擦常带窄带谐波峰(高值)，纯泄漏极低(0)"""
    fmin, fmax = get_ultra_bounds(sr)
    f, Pxx = signal.welch(y, sr, nperseg=min(4096, len(y)))
    mask = (f >= fmin) & (f <= fmax)
    if not np.any(mask): return 0.0
    f_sub, P_sub = f[mask], Pxx[mask]
    prom = np.mean(P_sub) * 4
    peaks, _ = signal.find_peaks(P_sub, prominence=prom)
    peak_e = np.sum(P_sub[peaks]) if len(peaks) > 0 else 0
    return peak_e / (np.sum(P_sub) + 1e-12)


def extract_ultra_features(file_path, global_max_energy=1.0):
    """提取单个文件的纯超声特征"""
    sr, y = load_wav_raw(file_path)
    if y is None: return None
    
    # 检查采样率门槛，硬性阻断硬件缺陷
    if sr < 100000:
        raise ValueError(f"当前采样率仅为 {sr}Hz，无法支持 50kHz 超声分析，请检查硬件！")

    raw_energy = ultra_band_energy(y, sr)
    # 使用全数据集最大能量进行全局相对缩放，保留点位之间的相对梯度
    normalized_energy = raw_energy / global_max_energy

    return {
        'ultra_energy': normalized_energy,
        'flatness':     spectral_flatness_ultra(y, sr),
        'centroid':     spectral_centroid_ultra(y, sr),
        'cv':           temporal_stationarity_ultra(y, sr),
        'line_r':       line_spectrum_ratio_ultra(y, sr)
    }


# ============================================================
# 3. 门控与非线性乘法融合（一票否决制）
# ============================================================
def leak_confidence_score_gated(feat, gates):
    """
    一票否决门控机制：
    假泄漏源在低频可能很响，但在超声带内，只要有一项不符合硬性物理门槛，总分直接归零。
    """
    if feat is None: return 0.0

    # 门槛 1：超声带能量门槛 (防止低能量环境噪声虚警)
    if feat['ultra_energy'] < gates['energy_gate']: return 0.0
    # 门槛 2：频谱平坦度门槛 (机械单频噪声平坦度极低，直接拍死)
    if feat['flatness'] < gates['flatness_gate']: return 0.0
    # 门槛 3：线谱占比上限 (含有强谐波单频尖刺的，判定为假泄漏)
    if feat['line_r'] > gates['line_gate']: return 0.0

    # 核心通过后，执行非线性相乘。任一维度表现差，都会呈指数级下拉总分
    score = (
        np.tanh(feat['ultra_energy'] / (gates['energy_gate'] + 1e-6)) * feat['flatness'] * (feat['centroid'] / ULTRA_FMAX) * (1.0 - np.tanh(feat['cv'])) * (1.0 - feat['line_r'])
    )
    return score


# ============================================================
# 4. 双路径数据集扫描机制与数据驱动门槛自标定
# ============================================================
def scan_directory_raw_energy(data_path):
    """第一遍扫描：获取整个文件夹中的超声绝对能量最大值，用于全局缩放"""
    files = glob.glob(os.path.join(data_path, "*.wav"))
    max_e = 1.0
    for f in files:
        sr, y = load_wav_raw(f)
        if y is not None:
            e = ultra_band_energy(y, sr)
            if e > max_e: max_e = e
    return max_e


def collect_folder_records(data_path, label):
    """执行精细化扫描，并打上统一的数据集标签（拒绝中心外圈盲目分类）"""
    global_max_e = scan_directory_raw_energy(data_path)
    records = []
    
    # 遍历 64 个周边点 + 1 个中心点
    all_files = glob.glob(os.path.join(data_path, "*.wav"))
    for f in all_files:
        feat = extract_ultra_features(f, global_max_e)
        if feat is None: continue
        
        # 解析坐标
        fname = os.path.basename(f)
        x, y = 0.0, 0.0
        for direction, angle in direction_angles.items():
            for d in distances:
                if f"00d{d}_{direction}" in fname:
                    x = d * np.cos(angle)
                    y = d * np.sin(angle)
                    break
                    
        rec = feat.copy()
        rec.update({'x': x, 'y': y, 'label': label, 'file': fname})
        records.append(rec)
    return records


# ============================================================
# 5. 主程序控制流
# ============================================================
if __name__ == "__main__":
    # 【请在此处填入你真实的真、假泄漏文件夹路径进行联合训练】
    # 即使当前只跑单文件夹，我们也能利用统计分位数建立安全门槛值
    current_run_dir = data_dir 

    print("🚀 正在启动第一路径：超声原始能量梯度分析...")
    # 提取当前目录全部点位特征（这里默认打为1类，后续用统计分位数自标定门槛）
    all_feature_records = collect_folder_records(current_run_dir, label=1)
    
    import pandas as pd
    df_feat = pd.DataFrame(all_feature_records)
    
    # ------------------------------------------------------------
    # 门槛自标定引擎：用稳健统计学（分位数）自动计算否决门槛
    # ------------------------------------------------------------
    print("\n⚙️ 正在启动自适应物理门槛标定...")
    # 动态设定：超声能量需大于场内前 30% 强度的水平；平坦度必须大于全场均值；线谱必须低于前 80% 水平
    gates = {
        'energy_gate':   float(df_feat['ultra_energy'].quantile(0.3)),
        'flatness_gate': float(df_feat['flatness'].mean() * 0.9),
        'line_gate':     float(df_feat['line_r'].quantile(0.8))
    }
    print(f" -> 动态自适应门槛判定输出: {gates}")

    # ------------------------------------------------------------
    # 执行精细门控评分计算
    # ------------------------------------------------------------
    plot_x, plot_y, plot_score = [], [], []
    for rec in all_feature_records:
        score = leak_confidence_score_gated(rec, gates)
        plot_x.append(rec['x'])
        plot_y.append(rec['y'])
        plot_score.append(score)
        rec['final_score'] = score

    plot_x = np.array(plot_x)
    plot_y = np.array(plot_y)
    plot_score = np.array(plot_score)

    # ------------------------------------------------------------
    # 物理断言与自动定位
    # ------------------------------------------------------------
    print("\n" + "="*60)
    max_idx = np.argmax(plot_score)
    mx, my, mval = plot_x[max_idx], plot_y[max_idx], plot_score[max_idx]
    
    print(f"【全场最高置信度坐标点】: ({mx:.1f}, {my:.1f}) cm, 置信度得分 = {mval:.4f}")
    
    # 全场最大质心审查器
    absolute_max_centroid = df_feat['centroid'].max()
    print(f"【全场物理最大超声质心】: {absolute_max_centroid:.1f} Hz")
    
    # 终极物理断言判据：如果全场最高评分接近 0，或者最大质心根本没有进入超声核心带
    if mval > 0.05 and absolute_max_centroid > 24000:
        print(">>> 物理断言：超声能量聚焦且完全符合湍流声学签名 -> 【真实气体泄漏 ✅】")
    else:
        print(">>> 物理断言：全场由于门控拦截导致分数塌陷，或无超声特征质心 -> 【虚假机械噪声 ❌】")
    print("="*60)

    # ------------------------------------------------------------
    # 6. 重新绘制置信度热力图
    # ------------------------------------------------------------
    grid_x, grid_y = np.mgrid[-45:45:200j, -45:45:200j]
    grid_z = griddata((plot_x, plot_y), plot_score, (grid_x, grid_y), method='cubic')
    grid_z = np.nan_to_num(grid_z, nan=0.0)

    plt.figure(figsize=(8, 6), dpi=100)
    plt.imshow(grid_z.T, extent=(-45, 45, -45, 45), origin='lower', cmap='jet')
    plt.colorbar(label='Ultrasonic Gated Leak Score')
    plt.scatter(plot_x, plot_y, c='white', s=15, edgecolors='black', alpha=0.5)
    plt.scatter(0, 0, c='red', marker='*', s=200, edgecolors='yellow', label='Center Point (0,0)')
    plt.title("Pure Ultrasonic Gated Heatmap\n(Fake Leak Suppressed Below 20kHz)", fontsize=11)
    plt.xlabel("X Coordinate (cm)"); plt.ylabel("Y Coordinate (cm)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()