# ============================================================
# 工业级气体泄漏检测系统 (双数据集对比标定 192kHz 完美无错版)
# ============================================================
import os, glob
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pandas as pd

direction_angles = {
    'up': np.pi/2, 'down': -np.pi/2, 'left': np.pi, 'right': 0,
    'up_left': 3*np.pi/4, 'down_left': -3*np.pi/4,
    'up_right': np.pi/4, 'down_right': -np.pi/4
}
distances = [5, 10, 15, 20, 25, 30, 35, 40]

# ------------------------------------------------------------
# 2. 完美的 192kHz 特征提取引擎
# ------------------------------------------------------------
def load_wav(fp):
    if not os.path.exists(fp): return None, None
    sr, y = wav.read(fp)
    if y.ndim > 1: y = y[:, 0]
    return sr, y.astype(float)

def extract_features(y, sr):
    if len(y) < 16384: return None  # 192kHz 下必须保证足够长度
    
    # 【Bug修复 1】扩充窗口至 16384 点 (约85ms)，彻底压制谱线噪声毛刺
    nperseg = 16384
    df = sr / nperseg
    f, Pxx = signal.welch(y, sr, nperseg=nperseg)
    Pxx = Pxx + 1e-12

    # 【Bug修复 2】在提取时域指标前，必须用 4 阶巴特沃斯进行 20k-80kHz 超声纯化带通
    # 彻底隔离低频机械内鬼
    b, a = signal.butter(4, [20000 / (sr/2), 80000 / (sr/2)], btype='band')
    y_ultra = signal.filtfilt(b, a, y)

    kurtosis = np.mean((y_ultra-np.mean(y_ultra))**4) / (np.var(y_ultra)**2 + 1e-6)
    # 适配 192kHz 采样率的时域 rms 分块
    rms = [np.sqrt(np.mean(y_ultra[i:i+4096]**2)) for i in range(0, len(y_ultra)-4096, 2048)]
    cv = np.std(rms)/(np.mean(rms)+1e-6) if rms else 1.0

    # 【Bug修复 3】严格引入 df 进行频域能量真积分
    mask_low = (f>=0) & (f<20000)
    mask_ultra = (f>=20000) & (f<80000)
    
    band_low   = np.sum(Pxx[mask_low]) * df
    band_ultra = np.sum(Pxx[mask_ultra]) * df
    hf_ratio = band_ultra / (band_low + band_ultra + 1e-12)

    if np.any(mask_ultra):
        P = Pxx[mask_ultra]
        flatness = np.exp(np.mean(np.log(P))) / np.mean(P)
        # 在大窗口平滑谱下，稳健寻找线谱峰
        peaks, _ = signal.find_peaks(P, prominence=np.mean(P)*4)
        line_r = (np.sum(P[peaks])*df)/(band_ultra+1e-12) if len(peaks)>0 else 0.0
    else:
        flatness, line_r = 0.0, 1.0

    return {'ultra_energy': band_ultra, 'hf_ratio': hf_ratio,
            'kurtosis': kurtosis, 'cv': cv,
            'flatness': flatness, 'line_r': line_r}

# ------------------------------------------------------------
# 3. 稳健解析与多维扫描（继承你的优秀架构）
# ------------------------------------------------------------
def parse_position(fname):
    for direction, angle in direction_angles.items():
        for d in distances:
            if f"00d{d}_{direction}" in fname:
                return d*np.cos(angle), d*np.sin(angle)
    if "_00_beamform_result" in fname or "00_00" in fname:
        return 0.0, 0.0
    return None, None

def scan_folder(target_dir):
    files = glob.glob(os.path.join(target_dir, "*.wav"))
    if not files:
        raise FileNotFoundError(f"没找到 wav：{target_dir}")
    tasks = []
    for f in files:
        sr, y = load_wav(f)
        if y is None: continue
        if sr < 100000: continue
        feat = extract_features(y, sr)
        if feat is None: continue
        coords = parse_position(os.path.basename(f))
        if coords[0] is None: continue
        tasks.append((coords[0], coords[1], feat))
    return tasks

# ------------------------------------------------------------
# 5 & 6. 对比标定模型更新（宽容度与数学对齐修复）
# ------------------------------------------------------------
def build_baseline_from_fake(fake_tasks):
    ue   = np.array([t[2]['ultra_energy'] for t in fake_tasks])
    hf   = np.array([t[2]['hf_ratio']     for t in fake_tasks])
    kurt = np.array([t[2]['kurtosis']     for t in fake_tasks])

    return {
        'energy_ref'   : np.percentile(ue, 90),
        'hf_mean'      : np.mean(hf),
        'hf_std'       : np.std(hf) + 1e-6,
        'kurt_ref'     : np.median(kurt),
    }

def leak_probability(feat, base):
    if feat is None: return 0.0

    # 维度1：能量显著度
    energy_gain = feat['ultra_energy'] / (base['energy_ref'] + 1e-12)
    p_energy = 1.0/(1.0+np.exp(-4*(energy_gain - 1.5)))

    # 维度2：超声占比偏离度
    z_hf = (feat['hf_ratio'] - base['hf_mean']) / base['hf_std']
    p_freq = 1.0/(1.0+np.exp(-2*(z_hf - 1.0)))

    # 维度3：时域超声平稳度
    p_style = 1.0/(1.0+np.exp(25*(feat['cv'] - 0.12)))

    # 维度4：谐波干扰滤除
    p_pure = 1.0/(1.0+np.exp(25*(feat['line_r'] - 0.08)))

    # 维度5：高压喷流峭度包容曲线 (修复中心点至 3.8，给强喷流适当容错)
    p_gauss = np.exp(-((feat['kurtosis'] - 3.8)**2) / (2*2.5**2))

    return p_energy * p_freq * p_style * p_pure * p_gauss

def calibrate_threshold(fake_tasks, base):
    probs = [leak_probability(t[2], base) for t in fake_tasks]
    fake_max = max(probs) if probs else 0.0
    threshold = min(0.95, fake_max * 1.2 + 0.05)
    return threshold, fake_max

def run(real_dir, fake_dir):
    print("=== 扫描假泄漏（标定基准） ===")
    fake_tasks = scan_folder(fake_dir)
    base = build_baseline_from_fake(fake_tasks)
    threshold, fake_max = calibrate_threshold(fake_tasks, base)
    
    print("=== 扫描待测数据 ===")
    real_tasks = scan_folder(real_dir)
    x, y, prob = [], [], []
    recs = []
    for xc, yc, feat in real_tasks:
        p = leak_probability(feat, base)
        x.append(xc); y.append(yc); prob.append(p)
        r = feat.copy(); r.update({'x': xc, 'y': yc, 'prob': p})
        recs.append(r)

    return (np.array(x), np.array(y), np.array(prob),
            pd.DataFrame(recs), threshold, fake_max)

# ------------------------------------------------------------
# 9. 可视化优化 (双插值无缝外推，修复边缘切边死影)
# ------------------------------------------------------------
def plot(x, y, prob, df, threshold, fake_max):
    if len(prob) == 0: return

    points = np.vstack((x, y)).T
    gx, gy = np.mgrid[-45:45:250j, -45:45:250j]
    
    # 【Bug修复 5】采用混合插值策略，彻底消灭生硬的黑边NaN
    gz_linear = griddata(points, prob, (gx, gy), method='linear')
    gz_nearest = griddata(points, prob, (gx, gy), method='nearest')
    gz = np.where(np.isnan(gz_linear), gz_nearest, gz_linear)

    plt.figure(figsize=(8.5, 6.5), dpi=100)
    plt.imshow(gz.T, extent=(-45,45,-45,45), origin='lower', cmap='hot')
    plt.colorbar(label='Leak Probability (calibrated vs FAKE)')
    plt.scatter(x, y, c='cyan', s=15, alpha=0.5, edgecolors='black', label='Nodes')
    plt.scatter(0, 0, c='red', marker='*', s=150, edgecolors='yellow', label='Center')

    mi = np.argmax(prob)
    plt.scatter(x[mi], y[mi], c='lime', marker='X', s=180, edgecolors='black',
                label=f'Peak ({x[mi]:.1f},{y[mi]:.1f})')
    plt.title("Leak Probability Map (Baseline from FAKE dataset)")
    plt.xlabel("X (cm)"); plt.ylabel("Y (cm)")
    plt.grid(True, ls=':', alpha=0.4); plt.legend(loc='upper right')
    plt.tight_layout()

    max_p = prob[mi]
    print("\n" + "="*65)
    print("【对比标定审计报告 —— 终极闭环版】")
    print(f" 假泄漏可达最高概率 : {fake_max*100:.2f}%")
    print(f" 判定阈值           : {threshold*100:.2f}%")
    print(f" 待测数据最高概率   : {max_p*100:.2f}%")
    if max_p > threshold:
        print(f"\n>>> 待测显著超过假泄漏基准")
        print(f">>> 物理断言：【真实气体泄漏 ✅】")
        print(f">>> 定位坐标：({x[mi]:.1f}, {y[mi]:.1f}) cm")
    else:
        print(f"\n>>> 待测未能超过假泄漏基准")
        print(f">>> 物理断言：【无泄漏 / 干扰噪声 ❌】")
    print("="*65 + "\n")
    plt.show()

if __name__ == "__main__":
    real_dir = r"D:\gas\...\真泄漏文件夹.ld"     
    fake_dir = r"D:\gas\...\假泄漏文件夹.ld"     

    if os.path.isdir(real_dir) and os.path.isdir(fake_dir):
        x, y, prob, df, thr, fmax = run(real_dir, fake_dir)
        plot(x, y, prob, df, thr, fmax)