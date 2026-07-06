import os
import glob
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal

def extract_leak_micro_features(data_dir, cutoff_freq=20000):
    """
    通过提取时频域的微观声学特征，深度鉴别是【真气流泄漏】还是【假机械杂音】
    """
    # 找到核心的中心点文件（因为中心点受泄漏信号影响最深）
    center_files = glob.glob(os.path.join(data_dir, "*00*.wav"))
    # 排除带d的周边测点
    actual_center = [f for f in center_files if not any(f"00d" in f for d in [5,10,15,20,25,30,35,40])]
    
    if not actual_center:
        return "未找到中心通道文件"
    
    # 1. 读取音频
    sr, y = wav.read(actual_center[0])
    if len(y.shape) > 1: y = y[:, 0]
    y = y / (np.max(np.abs(y)) + 1e-6)
    
    # 2. 实施高通滤波，只保留超声高频段
    b, a = signal.butter(4, cutoff_freq / (sr / 2), btype='high')
    filtered_y = signal.filtfilt(b, a, y)
    
    # ==========================================
    # 特征计算一：时域平稳度指标
    # ==========================================
    # 峭度 (Kurtosis): 标准高斯分布（如持续气流）接近 3。若有机械冲击毛刺，峭度会暴涨
    kurt = np.mean((filtered_y - np.mean(filtered_y))**4) / (np.var(filtered_y)**2 + 1e-12)
    # 峰值因子 (Crest Factor): 峰值与有效值的比值
    crest_factor = np.max(np.abs(filtered_y)) / (np.sqrt(np.mean(filtered_y**2)) + 1e-12)
    
    # ==========================================
    # 特征计算二：频域谱平坦度 (Spectral Flatness)
    # ==========================================
    # 计算功率谱密度
    freqs, psd = signal.welch(filtered_y, sr, nperseg=1024)
    # 仅提取高频段功率谱
    high_freq_mask = freqs >= cutoff_freq
    psd_high = psd[high_freq_mask] + 1e-12
    
    # 谱平坦度公式 = 功率谱的几何平均数 / 算术平均数
    geometric_mean = np.exp(np.mean(np.log(psd_high)))
    arithmetic_mean = np.mean(psd_high)
    spectral_flatness = geometric_mean / arithmetic_mean
    
    # ==========================================
    # 核心分类决策树逻辑
    # ==========================================
    print(f"\n[数据包 {os.path.basename(data_dir)} 微观特征分析结果]:")
    print(f" -> 频域谱平坦度 (Spectral Flatness): {spectral_flatness:.4f}  (越接近1越说明是宽带气流)")
    print(f" -> 时域信号峭度 (Kurtosis): {kurt:.4f}  (接近3说明平稳，远大于3说明有机械冲击)")
    print(f" -> 时域峰值因子 (Crest Factor): {crest_factor:.4f}")
    
    # 综合判定阈值（阈值可根据你的真假实验包跑出的数据进行微调）
    if spectral_flatness > 0.4 and kurt < 4.5:
        return "【分类判定】：特征符合宽带平稳湍流模型 -> [ 真实气体泄漏 YES ]"
    else:
        return "【分类判定】：检测到明显的单频谐波或机械周期冲击 -> [ 虚假机械噪声 NO ]"

# ---- 你可以用这两个文件夹路径分别跑一下，对比它们的特征差异 ----
# folder_true_leak = r"D:\...\真实泄漏数据包.ld"
# folder_fake_noise = r"D:\...\虚假干扰数据包.ld"

# print(extract_leak_micro_features(folder_true_leak))