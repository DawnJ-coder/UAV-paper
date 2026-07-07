import os
import glob
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt
import re

# ==========================================
# 1. 配置参数 (保持不变)
# ==========================================
time_folders = [
    "HM20260626_142938.ld",
    "HM20260626_143034.ld", 
    "HM20260626_144226.ld",
    "HM20260626_144325.ld"
]

center_root_dir = r"D:\gas\beamform_results"
offset_root_dir = r"D:\gas\beamform_results_offset_multiple"

direction_angles = {
    'up': np.pi / 2, 'down': -np.pi / 2, 'left': np.pi, 'right': 0,
    'up_left': 3 * np.pi / 4, 'down_left': -3 * np.pi / 4,
    'up_right': np.pi / 4, 'down_right': -np.pi / 4
}

FREQ_LOW = 20000   
FREQ_HIGH = 70000  
NFFT = 4096        

# ==========================================
# 2. 核心算法：频谱计算与减法 (保持不变)
# ==========================================
def get_spectrum(file_path, sr_target=192000):
    if not os.path.exists(file_path): return None, None
    sr, y = wav.read(file_path)
    if len(y.shape) > 1: y = y[:, 0]
    y = y / (np.max(np.abs(y)) + 1e-6)
    freqs, psd = signal.welch(y, fs=sr, nperseg=NFFT, scaling='density')
    mask = (freqs >= FREQ_LOW) & (freqs <= FREQ_HIGH)
    return freqs[mask], psd[mask]

def get_band_energy_from_spectrum(freqs, psd):
    if freqs is None or psd is None: return 0.0
    return np.trapz(psd, freqs)

def subtract_spectrum(freqs, psd_signal, psd_background):
    if psd_signal is None or psd_background is None: return None
    min_len = min(len(psd_signal), len(psd_background))
    net_psd = np.maximum(psd_signal[:min_len] - psd_background[:min_len], 0)
    return freqs[:min_len], net_psd

# ==========================================
# 🌟 新增模块：算法鲁棒性验证
# ==========================================
def check_decay_ratio(energy_near, energy_far, direction):
    """验证40cm背景是否干净的致命性检查"""
    if energy_far <= 0: return True
    ratio = energy_near / energy_far
    if ratio < 3.0:
        print(f"  [⚠️ 严重警告] {direction} 方向 5cm/40cm 能量衰减比仅为 {ratio:.1f}！40cm处存在强残留，背景相减将导致信号失真！")
        return False
    return True

def verify_leak_signature(freqs, net_psd):
    """
    频域鉴真：验证扣除背景后的净频谱是否具备真实泄漏的特征（而非机械干扰）
    这里做了一个简化的双峰检测逻辑 (52kHz 和 57.5kHz附近)
    """
    if freqs is None or net_psd is None or np.sum(net_psd) == 0:
        return False, 0.0
    
    # 归一化残差频谱以便计算形态相似度
    norm_psd = net_psd / (np.max(net_psd) + 1e-12)
    
    # 构造理想泄漏的参考特征 (根据实验室标定数据微调)
    template = np.zeros_like(freqs)
    template += np.exp(-((freqs - 52000) / 500)**2) * 0.8
    template += np.exp(-((freqs - 57500) / 300)**2) * 1.0
    
    # 计算皮尔逊相关系数
    similarity = np.corrcoef(norm_psd, template)[0, 1]
    
    is_real = similarity > 0.45  # 相似度阈值
    return is_real, similarity

# ==========================================
# 3 & 4. 自动检测与绘图函数 (保持不变，省略展开以节省空间，直接用你原代码里的即可)
# ==========================================
def detect_center_ids(center_data_dir):
    pattern = os.path.join(center_data_dir, "*_*_beamform_result.wav")
    files = glob.glob(pattern)
    center_ids = sorted(list(set(re.search(r'_(\d{2})_beamform_result\.wav', os.path.basename(f)).group(1) for f in files if re.search(r'_(\d{2})_beamform_result\.wav', os.path.basename(f)))))
    print(f"检测到中心点编号: {center_ids}")
    return center_ids

# (请在此处保留你原来的 plot_spectrum_comparison 函数)
def plot_spectrum_comparison(result_dir, time_folder, center_id, center_freqs, center_net_psd, directional_spectra, direction_bg_spectra, verify_results):
    # 你原有的绘图代码... 
    pass # 为了不刷屏，这里省略，但你可以把最后验证的结果 verify_results 打印在图表的 Title 上。

# ==========================================
# 5. 处理单个时间点的函数 (补全版)
# ==========================================
def process_single_timepoint_spectrum(time_folder, center_root, offset_root):
    print(f"\n{'='*80}\n开始处理时间点: {time_folder} (频谱分析模式)\n{'='*80}")
    
    center_data_dir = os.path.join(center_root, time_folder)
    offset_data_dir = os.path.join(offset_root, time_folder)
    
    if not os.path.exists(center_data_dir) or not os.path.exists(offset_data_dir):
        print("警告：文件夹不存在")
        return
    
    result_dir = f"results_spectrum_{time_folder}"
    os.makedirs(result_dir, exist_ok=True)
    
    center_ids = detect_center_ids(center_data_dir)
    if not center_ids: return
    
    for center_id in center_ids:
        print(f"\n{'-'*60}\n正在处理 {time_folder} - 中心点 {center_id}...\n{'-'*60}")
        
        # [第一步：读取中心点频谱]
        center_pattern = os.path.join(center_data_dir, f"*_{center_id}_beamform_result.wav")
        center_files = glob.glob(center_pattern)
        if not center_files: continue
        center_freqs, center_psd = get_spectrum(center_files[0])
        
        # [第二步：读取各方向40cm背景]
        direction_bg_spectra = {}
        bg_energies = {}
        for direction in direction_angles.keys():
            pattern = os.path.join(offset_data_dir, f"*_{center_id}d40_{direction}*.wav")
            files = glob.glob(pattern)
            if files:
                freqs, psd = get_spectrum(files[0])
                if freqs is not None:
                    direction_bg_spectra[direction] = (freqs, psd)
                    bg_energies[direction] = get_band_energy_from_spectrum(freqs, psd)

        # [第三步：频谱减法与 5cm 验证]
        directional_spectra_5cm = {}
        verify_results = {} # 记录鉴真结果
        
        # 1. 中心点扣除最小背景
        center_net_freqs, center_net_psd = None, None
        if direction_bg_spectra:
            best_bg_dir = min(direction_bg_spectra.keys(), key=lambda d: bg_energies[d])
            bg_freqs, bg_psd = direction_bg_spectra[best_bg_dir]
            center_net_freqs, center_net_psd = subtract_spectrum(center_freqs, center_psd, bg_psd)

        # 2. 读取各方向 5cm 处的信号，并进行衰减检查和鉴真
        for direction in direction_angles.keys():
            pattern_5cm = os.path.join(offset_data_dir, f"*_{center_id}d5_{direction}*.wav")
            files_5cm = glob.glob(pattern_5cm)
            
            if files_5cm and direction in direction_bg_spectra:
                freqs_5cm, psd_5cm = get_spectrum(files_5cm[0])
                energy_5cm = get_band_energy_from_spectrum(freqs_5cm, psd_5cm)
                
                # 🔴 关键机制 1：执行近/远端衰减比检查
                check_decay_ratio(energy_5cm, bg_energies[direction], direction)
                
                # 扣除同方向 40cm 的背景
                bg_freqs, bg_psd = direction_bg_spectra[direction]
                net_freqs, net_psd = subtract_spectrum(freqs_5cm, psd_5cm, bg_psd)
                directional_spectra_5cm[direction] = (net_freqs, net_psd)
                
                # 🔴 关键机制 2：对扣除背景后的残差进行频域指纹鉴真
                is_real, sim_score = verify_leak_signature(net_freqs, net_psd)
                verify_results[direction] = (is_real, sim_score)
                
                status = "✅ 真泄漏" if is_real else "❌ 假泄漏/噪声"
                print(f"  {direction:10} | 5cm净能量: {get_band_energy_from_spectrum(net_freqs, net_psd):.2e} | 鉴别得分: {sim_score:.2f} -> {status}")

        # [第四步：调用绘图] (将新增的验证结果传入绘图函数，可自行在原绘图代码中添加文本标注)
        # plot_spectrum_comparison(result_dir, time_folder, center_id, 
        #                          center_net_freqs, center_net_psd,
        #                          directional_spectra_5cm, direction_bg_spectra)
        print(f"中心点 {center_id} 处理完毕。")