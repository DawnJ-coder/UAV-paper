# -*- coding: utf-8 -*-
"""
leak_feature_classifier_v3.py

v3: 特征提取 + 真假分类准备版

用途：
    把真实泄漏/假泄漏数据都转成同一张特征表，方便后续对比、画图、训练分类器。

输出：
    leak_feature_v3_results/
        leak_feature_dataset.csv
        energy_matrix_detail.csv
        feature_description.txt
        simple_classifier_report.txt  # 只有同时有TRUE_LEAK和FALSE_LEAK且安装sklearn时生成

运行：
    python leak_feature_classifier_v3.py
"""

import os
import glob
import re
import csv
import time
import math
import warnings

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")


# ============================================================
# 1. 配置区：您主要改这里
# ============================================================

DATASETS = [
    {
        "name": "true_leak_dataset",
        "label": "TRUE_LEAK",
        "center_root_dir": r"D:\gas\beamform_results",
        "offset_root_dir": r"D:\gas\beamform_results_offset_multiple",
        "time_folders": [
            "HM20260626_142938.ld",
            "HM20260626_143034.ld",
            "HM20260626_144226.ld",
            "HM20260626_144325.ld",
        ],
    },

    # 如果要同时加入假泄漏数据，把下面这一段取消注释，并改成您的假泄漏路径
    # {
    #     "name": "false_leak_dataset",
    #     "label": "FALSE_LEAK",
    #     "center_root_dir": r"D:\gas_false\beamform_results",
    #     "offset_root_dir": r"D:\gas_false\beamform_results_offset_multiple",
    #     "time_folders": [
    #         "HM20260626_142938.ld",
    #         "HM20260626_143034.ld",
    #     ],
    # },
]

# 调试时可以设成 3；正式提取全部特征时设为 None
MAX_CENTERS_PER_TIME = None

# 默认不使用 40cm 当背景扣除，因为40cm点也可能含有泄漏声
USE_40CM_BACKGROUND_SUBTRACTION = False

# 频谱范围
FREQ_LOW = 20000
FREQ_HIGH = 70000
FREQ_SPLIT = 40000

# 分频段特征
SUB_BANDS = [
    (20000, 30000),
    (30000, 40000),
    (40000, 50000),
    (50000, 60000),
    (60000, 70000),
]

NFFT = 4096
FRAME_MS = 20
HOP_MS = 10

distances = np.array([5, 10, 15, 20, 25, 30, 35, 40], dtype=float)

directions = [
    "up", "down", "left", "right",
    "up_left", "down_left", "up_right", "down_right"
]


# ============================================================
# 2. 基础函数
# ============================================================

def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def read_wav(file_path):
    if not os.path.exists(file_path):
        return None, None, None
    try:
        fs, x = wav.read(file_path)
        if len(x.shape) > 1:
            x = x[:, 0]
        x = x.astype(np.float32)
        rms = float(np.sqrt(np.mean(x * x)) + 1e-12)
        return fs, x, rms
    except Exception as e:
        print("[读取失败]", file_path, e)
        return None, None, None


def compute_psd(file_path, norm_factor=None):
    """计算20-70kHz PSD，同时返回归一化后的时域信号。"""
    fs, x, rms = read_wav(file_path)
    if x is None:
        return None, None, None, None

    if norm_factor is None:
        norm_factor = rms
    x = x / (norm_factor + 1e-12)

    nperseg = min(NFFT, len(x))
    if nperseg < 16:
        return None, None, None, None

    freq, psd = signal.welch(x, fs=fs, nperseg=nperseg, scaling="density")
    mask = (freq >= FREQ_LOW) & (freq <= FREQ_HIGH)
    if not np.any(mask):
        return None, None, fs, x
    return freq[mask], psd[mask], fs, x


def integrate_energy(freq, psd, low=None, high=None):
    if freq is None or psd is None or len(freq) == 0:
        return 0.0
    if low is None:
        low = FREQ_LOW
    if high is None:
        high = FREQ_HIGH
    mask = (freq >= low) & (freq <= high)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freq[mask]))


def spectrum_subtract(psd, bg_psd):
    if psd is None or bg_psd is None:
        return None
    n = min(len(psd), len(bg_psd))
    if n <= 0:
        return None
    return np.maximum(psd[:n] - bg_psd[:n], 0.0)


# ============================================================
# 3. 文件查找
# ============================================================

def detect_center_ids(center_dir):
    files = glob.glob(os.path.join(center_dir, "*_beamform_result.wav"))
    ids = []
    for f in files:
        name = os.path.basename(f)
        m = re.search(r"_(\d+)_beamform_result", name)
        if m:
            ids.append(m.group(1))
    return sorted(set(ids))


def find_center_file(center_dir, center_id):
    files = glob.glob(os.path.join(center_dir, f"*_{center_id}_beamform_result.wav"))
    return files[0] if files else None


def find_offset_file(offset_dir, center_id, distance, direction):
    files = glob.glob(os.path.join(offset_dir, f"*_{center_id}d{int(distance)}_{direction}*.wav"))
    return files[0] if files else None


# ============================================================
# 4. 空间传播特征
# ============================================================

def power_decay_model(r, A, n):
    return A * np.power(r, -n)


def fit_decay(energy_list):
    """拟合 E = A * r^(-n)，返回 decay_score, n, R2。"""
    r = distances[:len(energy_list)]
    e = np.array(energy_list, dtype=float)
    mask = e > 0
    r = r[mask]
    e = e[mask]

    if len(r) < 3:
        return 0.0, 0.0, 0.0
    try:
        popt, _ = curve_fit(
            power_decay_model,
            r,
            e,
            p0=[max(e[0] * 25.0, 1e-12), 1.0],
            bounds=([0.0, 0.0], [np.inf, 5.0]),
            maxfev=5000,
        )
        A, n = popt
        pred = power_decay_model(r, A, n)
        ss_res = np.sum((e - pred) ** 2)
        ss_tot = np.sum((e - np.mean(e)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        r2 = float(np.clip(r2, 0.0, 1.0))

        # 波束形成数据里 n 不一定接近2，放宽到 n>=0.3 即可给满 n_score
        if n >= 0.3:
            n_score = 1.0
        elif n <= 0:
            n_score = 0.0
        else:
            n_score = n / 0.3
        return float(r2 * n_score), float(n), float(r2)
    except Exception:
        return 0.0, 0.0, 0.0


def near_far_features(energy_list):
    """近场/远场能量比：mean(5,10cm)/mean(30,35,40cm)。"""
    e = np.array(energy_list, dtype=float)
    if len(e) < 4 or np.max(e) <= 0:
        return 0.0, 0.0
    near = np.mean(e[:2])
    far = np.mean(e[-3:])
    ratio = float(near / (far + 1e-12))
    score = float(np.clip((ratio - 1.0) / 1.0, 0.0, 1.0))
    return score, ratio


def monotonic_decay_ratio(energy_list):
    e = np.array(energy_list, dtype=float)
    if len(e) < 2:
        return 0.0
    return float(np.mean(np.diff(e) <= 0))


def direction_contrast(direction_features):
    if not direction_features:
        return 0.0
    vals = np.array([direction_features[d]["max_energy"] for d in direction_features], dtype=float)
    if np.mean(vals) <= 0:
        return 0.0
    return float(np.max(vals) / (np.mean(vals) + 1e-12))


# ============================================================
# 5. 频谱形态特征
# ============================================================

def empty_spectral_features():
    out = {
        "spec_entropy": 0.0,
        "spec_flatness": 0.0,
        "spec_centroid_hz": 0.0,
        "spec_bandwidth_hz": 0.0,
        "spec_peakiness": 0.0,
        "spec_peak_freq_hz": 0.0,
        "spec_peak_count": 0.0,
        "spec_rolloff_85_hz": 0.0,
        "spec_slope": 0.0,
        "energy_20_70": 0.0,
        "energy_20_40": 0.0,
        "energy_40_70": 0.0,
        "high_freq_ratio": 0.0,
    }
    for low, high in SUB_BANDS:
        out[f"energy_{int(low/1000)}_{int(high/1000)}k"] = 0.0
        out[f"ratio_{int(low/1000)}_{int(high/1000)}k"] = 0.0
    return out


def spectral_features(freq, psd):
    """频谱特征：宽带性、尖峰性、分频段能量等。"""
    if freq is None or psd is None or len(freq) == 0:
        return empty_spectral_features()

    f = np.asarray(freq, dtype=float)
    p = np.maximum(np.asarray(psd, dtype=float), 0.0)
    total_sum = np.sum(p) + 1e-12
    prob = p / total_sum

    entropy = -np.sum(prob * np.log(prob + 1e-12))
    entropy_norm = entropy / (np.log(len(prob) + 1e-12))

    positive = p[p > 0]
    if len(positive) > 0:
        flatness = np.exp(np.mean(np.log(positive + 1e-20))) / (np.mean(positive) + 1e-20)
    else:
        flatness = 0.0

    centroid = np.sum(f * p) / total_sum
    bandwidth = np.sqrt(np.sum(((f - centroid) ** 2) * p) / total_sum)
    peakiness = np.max(p) / (np.mean(p) + 1e-12)
    peak_freq = f[int(np.argmax(p))]

    threshold = np.mean(p) + 3.0 * np.std(p)
    peaks, _ = signal.find_peaks(p, height=threshold)
    peak_count = len(peaks)

    cumsum = np.cumsum(p)
    roll_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
    roll_idx = min(max(roll_idx, 0), len(f) - 1)
    rolloff_85 = f[roll_idx]

    try:
        slope = np.polyfit(f, np.log(p + 1e-20), 1)[0]
    except Exception:
        slope = 0.0

    total_energy = integrate_energy(f, p, FREQ_LOW, FREQ_HIGH)
    low_energy = integrate_energy(f, p, FREQ_LOW, FREQ_SPLIT)
    high_energy = integrate_energy(f, p, FREQ_SPLIT, FREQ_HIGH)
    high_ratio = high_energy / (total_energy + 1e-12)

    out = {
        "spec_entropy": safe_float(entropy_norm),
        "spec_flatness": safe_float(flatness),
        "spec_centroid_hz": safe_float(centroid),
        "spec_bandwidth_hz": safe_float(bandwidth),
        "spec_peakiness": safe_float(peakiness),
        "spec_peak_freq_hz": safe_float(peak_freq),
        "spec_peak_count": safe_float(peak_count),
        "spec_rolloff_85_hz": safe_float(rolloff_85),
        "spec_slope": safe_float(slope),
        "energy_20_70": safe_float(total_energy),
        "energy_20_40": safe_float(low_energy),
        "energy_40_70": safe_float(high_energy),
        "high_freq_ratio": safe_float(high_ratio),
    }

    for low, high in SUB_BANDS:
        band_e = integrate_energy(f, p, low, high)
        out[f"energy_{int(low/1000)}_{int(high/1000)}k"] = safe_float(band_e)
        out[f"ratio_{int(low/1000)}_{int(high/1000)}k"] = safe_float(band_e / (total_energy + 1e-12))
    return out


# ============================================================
# 6. 时间域特征
# ============================================================

def bandpass_signal(x, fs):
    if fs is None or x is None:
        return None
    nyq = fs / 2.0
    if FREQ_HIGH >= nyq:
        return x
    try:
        sos = signal.butter(4, [FREQ_LOW / nyq, FREQ_HIGH / nyq], btype="bandpass", output="sos")
        return signal.sosfilt(sos, x)
    except Exception:
        return x


def short_time_energy_features(x, fs):
    out = {
        "time_energy_mean": 0.0,
        "time_energy_std": 0.0,
        "time_energy_cv": 0.0,
        "time_energy_max_mean_ratio": 0.0,
        "time_energy_kurtosis": 0.0,
        "time_rms": 0.0,
    }
    if x is None or fs is None or len(x) < 16:
        return out

    y = bandpass_signal(x, fs)
    if y is None:
        return out

    frame_len = max(int(fs * FRAME_MS / 1000), 16)
    hop_len = max(int(fs * HOP_MS / 1000), 1)

    if len(y) < frame_len:
        e = np.array([np.mean(y * y)], dtype=float)
    else:
        vals = []
        for start in range(0, len(y) - frame_len + 1, hop_len):
            frame = y[start:start + frame_len]
            vals.append(np.mean(frame * frame))
        e = np.array(vals, dtype=float)

    if len(e) == 0:
        return out

    mean_e = np.mean(e) + 1e-12
    std_e = np.std(e)
    cv = std_e / mean_e
    max_mean = np.max(e) / mean_e
    centered = e - np.mean(e)
    kurt = np.mean(centered ** 4) / ((np.var(e) + 1e-12) ** 2)

    out["time_energy_mean"] = safe_float(mean_e)
    out["time_energy_std"] = safe_float(std_e)
    out["time_energy_cv"] = safe_float(cv)
    out["time_energy_max_mean_ratio"] = safe_float(max_mean)
    out["time_energy_kurtosis"] = safe_float(kurt)
    out["time_rms"] = safe_float(np.sqrt(np.mean(y * y)) + 1e-12)
    return out


# ============================================================
# 7. 单中心点特征提取
# ============================================================

def analyze_one_center(dataset_name, label, time_folder, center_id, center_root_dir, offset_root_dir):
    center_dir = os.path.join(center_root_dir, time_folder)
    offset_dir = os.path.join(offset_root_dir, time_folder)
    center_file = find_center_file(center_dir, center_id)
    if center_file is None:
        return None, []

    _, _, center_rms = read_wav(center_file)
    if center_rms is None:
        return None, []

    direction_features = {}
    matrix_rows = []
    representative = {}

    for direction in directions:
        bg_psd = None
        if USE_40CM_BACKGROUND_SUBTRACTION:
            bg_file = find_offset_file(offset_dir, center_id, 40, direction)
            if bg_file is not None:
                _, bg_psd, _, _ = compute_psd(bg_file, center_rms)

        energy_list = []
        spec_by_distance = {}

        for dist in distances:
            offset_file = find_offset_file(offset_dir, center_id, dist, direction)
            if offset_file is None:
                energy_list.append(0.0)
                continue

            freq, psd, fs, x_norm = compute_psd(offset_file, center_rms)
            if freq is None or psd is None:
                energy_list.append(0.0)
                continue

            used_freq = freq
            used_psd = psd
            if USE_40CM_BACKGROUND_SUBTRACTION and bg_psd is not None:
                tmp = spectrum_subtract(psd, bg_psd)
                if tmp is not None:
                    n = min(len(freq), len(tmp))
                    used_freq = freq[:n]
                    used_psd = tmp[:n]

            total_e = integrate_energy(used_freq, used_psd, FREQ_LOW, FREQ_HIGH)
            low_e = integrate_energy(used_freq, used_psd, FREQ_LOW, FREQ_SPLIT)
            high_e = integrate_energy(used_freq, used_psd, FREQ_SPLIT, FREQ_HIGH)
            hf_ratio = high_e / (total_e + 1e-12)

            energy_list.append(total_e)
            spec_by_distance[int(dist)] = (used_freq, used_psd, fs, x_norm, offset_file)

            matrix_rows.append({
                "dataset": dataset_name,
                "label": label,
                "time": time_folder,
                "center": center_id,
                "direction": direction,
                "distance_cm": int(dist),
                "energy_20_70": safe_float(total_e),
                "energy_20_40": safe_float(low_e),
                "energy_40_70": safe_float(high_e),
                "high_freq_ratio": safe_float(hf_ratio),
                "file": offset_file,
            })

        decay_score, n_value, r2 = fit_decay(energy_list)
        near_score, near_far_ratio = near_far_features(energy_list)
        mono_ratio = monotonic_decay_ratio(energy_list)

        direction_features[direction] = {
            "max_energy": float(np.max(energy_list)) if energy_list else 0.0,
            "mean_energy": float(np.mean(energy_list)) if energy_list else 0.0,
            "energy_5cm": float(energy_list[0]) if energy_list else 0.0,
            "decay_score": decay_score,
            "attenuation_n": n_value,
            "decay_R2": r2,
            "near_score": near_score,
            "near_far_ratio": near_far_ratio,
            "monotonic_decay_ratio": mono_ratio,
        }

        if 5 in spec_by_distance:
            representative[direction] = spec_by_distance[5]

    if not direction_features:
        return None, matrix_rows

    energy_direction = max(direction_features, key=lambda d: direction_features[d]["max_energy"])
    decay_direction = max(direction_features, key=lambda d: direction_features[d]["decay_score"])
    max_energy_all = max(direction_features[d]["max_energy"] for d in direction_features) + 1e-12

    combined_scores = {}
    for d, feat in direction_features.items():
        energy_score = feat["max_energy"] / max_energy_all
        combined_scores[d] = (
            0.40 * energy_score
            + 0.35 * feat["decay_score"]
            + 0.15 * feat["near_score"]
            + 0.10 * feat["monotonic_decay_ratio"]
        )

    best_direction = max(combined_scores, key=combined_scores.get)
    best_feat = direction_features[best_direction]

    direction_agreement = 0.0
    if best_direction == energy_direction:
        direction_agreement += 0.5
    if best_direction == decay_direction:
        direction_agreement += 0.5

    if best_direction in representative:
        best_freq, best_psd, best_fs, best_x, best_file = representative[best_direction]
    elif energy_direction in representative:
        best_freq, best_psd, best_fs, best_x, best_file = representative[energy_direction]
    else:
        best_freq, best_psd, best_fs, best_x, best_file = None, None, None, None, ""

    row = {
        "dataset": dataset_name,
        "label": label,
        "time": time_folder,
        "center": center_id,
        "best_direction": best_direction,
        "energy_direction": energy_direction,
        "decay_direction": decay_direction,
        "direction_agreement": direction_agreement,
        "direction_contrast": direction_contrast(direction_features),
        "best_direction_combined_score": safe_float(combined_scores[best_direction]),
        "raw_best_energy": safe_float(best_feat["max_energy"]),
        "mean_energy_best_direction": safe_float(best_feat["mean_energy"]),
        "energy_5cm_best_direction": safe_float(best_feat["energy_5cm"]),
        "decay_score": safe_float(best_feat["decay_score"]),
        "attenuation_n": safe_float(best_feat["attenuation_n"]),
        "decay_R2": safe_float(best_feat["decay_R2"]),
        "near_score": safe_float(best_feat["near_score"]),
        "near_far_ratio": safe_float(best_feat["near_far_ratio"]),
        "monotonic_decay_ratio": safe_float(best_feat["monotonic_decay_ratio"]),
        "representative_file": best_file,
    }

    row.update(spectral_features(best_freq, best_psd))
    row.update(short_time_energy_features(best_x, best_fs))
    return row, matrix_rows


# ============================================================
# 8. 可选自动训练简单分类器
# ============================================================

def try_train_simple_classifier(feature_csv, output_dir):
    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix
    except Exception:
        print("未检测到 pandas/scikit-learn，跳过自动训练。只生成特征CSV。")
        return

    try:
        df = pd.read_csv(feature_csv)
        labels = sorted(df["label"].dropna().unique().tolist())
        if len(labels) < 2:
            print("当前只有一种label，无法训练真假分类器。")
            return
        if len(df) < 10:
            print("样本数量太少，暂不训练分类器。")
            return

        drop_cols = [
            "dataset", "label", "time", "center",
            "best_direction", "energy_direction", "decay_direction",
            "representative_file",
        ]
        feature_cols = [c for c in df.columns if c not in drop_cols]
        X = df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        clf = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        )
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        report = classification_report(y_test, pred)
        cm = confusion_matrix(y_test, pred, labels=labels)
        importances = sorted(zip(feature_cols, clf.feature_importances_), key=lambda x: x[1], reverse=True)

        report_path = os.path.join(output_dir, "simple_classifier_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("RandomForest 初步分类报告\n")
            f.write("=" * 80 + "\n\n")
            f.write("Labels:\n")
            f.write(str(labels) + "\n\n")
            f.write("Classification report:\n")
            f.write(report + "\n\n")
            f.write("Confusion matrix:\n")
            f.write(str(cm) + "\n\n")
            f.write("Top 30 important features:\n")
            for name, val in importances[:30]:
                f.write(f"{name}: {val:.6f}\n")
        print("已生成初步分类报告:", report_path)
    except Exception as e:
        print("自动训练分类器失败，已跳过:", e)


# ============================================================
# 9. 特征说明
# ============================================================

def write_feature_description(path):
    text = """leak_feature_classifier_v3 特征说明

一、空间特征
- raw_best_energy: 最优方向下20-70kHz最大能量。
- decay_R2: 距离衰减模型 E=A*r^-n 的拟合优度，越接近1说明越像稳定声源传播。
- attenuation_n: 距离衰减指数。波束形成数据里不一定接近2，主要用于比较。
- near_far_ratio: 近处5/10cm平均能量 与 远处30/35/40cm平均能量之比。
- monotonic_decay_ratio: 距离增加时能量下降的比例。
- direction_contrast: 最强方向能量 / 方向平均能量，用于判断方向性强弱。
- direction_agreement: 综合方向是否同时等于最大能量方向和最佳衰减方向。

二、频谱形态特征
- spec_entropy: 频谱熵。越高说明越宽带；真实泄漏通常更宽带。
- spec_flatness: 谱平坦度。越高说明越接近宽带噪声；窄带干扰通常较低。
- spec_peakiness: 频谱尖峰程度=max(PSD)/mean(PSD)。越高越像窄带尖峰干扰。
- spec_peak_count: 显著峰数量。窄带谐波干扰可能峰较明显。
- spec_centroid_hz: 频谱重心。
- spec_bandwidth_hz: 频谱带宽。
- spec_rolloff_85_hz: 85%能量滚降频率。
- spec_slope: log(PSD)随频率变化的斜率。
- ratio_20_30k ~ ratio_60_70k: 分频段能量比例，用于分析泄漏/干扰频带差异。

三、时间特征
- time_energy_cv: 短时能量变异系数。越大说明越不稳定、越脉冲。
- time_energy_max_mean_ratio: 最大短时能量 / 平均短时能量。越大说明越突发。
- time_energy_kurtosis: 短时能量峰度。越大说明越脉冲。
- time_rms: 20-70kHz带通后的RMS。

建议：
1. 先分别跑 TRUE_LEAK 和 FALSE_LEAK，得到 leak_feature_dataset.csv。
2. 用Excel或Python画 spec_entropy、spec_flatness、spec_peakiness、time_energy_cv 的真假分布。
3. 如果真假样本都在 DATASETS 里配置了，程序会尝试自动训练 RandomForest 并输出重要特征。
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ============================================================
# 10. 主程序
# ============================================================

def main():
    start_time = time.time()
    output_dir = os.path.join(os.getcwd(), "leak_feature_v3_results")
    os.makedirs(output_dir, exist_ok=True)

    feature_csv = os.path.join(output_dir, "leak_feature_dataset.csv")
    matrix_csv = os.path.join(output_dir, "energy_matrix_detail.csv")
    desc_path = os.path.join(output_dir, "feature_description.txt")

    all_feature_rows = []
    all_matrix_rows = []

    print("=" * 80)
    print("v3 特征提取 + 真假分类准备程序开始运行")
    print("输出文件夹:", output_dir)
    print("=" * 80)

    for ds in DATASETS:
        dataset_name = ds["name"]
        label = ds["label"]
        center_root_dir = ds["center_root_dir"]
        offset_root_dir = ds["offset_root_dir"]
        time_folders = ds["time_folders"]

        print("\n" + "#" * 80)
        print("数据集:", dataset_name)
        print("标签:", label)
        print("center_root_dir:", center_root_dir)
        print("offset_root_dir:", offset_root_dir)
        print("#" * 80)

        for time_folder in time_folders:
            center_dir = os.path.join(center_root_dir, time_folder)
            offset_dir = os.path.join(offset_root_dir, time_folder)
            print("\n处理时间点:", time_folder)

            if not os.path.exists(center_dir):
                print("  [跳过] 中心目录不存在:", center_dir)
                continue
            if not os.path.exists(offset_dir):
                print("  [跳过] 偏移目录不存在:", offset_dir)
                continue

            ids = detect_center_ids(center_dir)
            if MAX_CENTERS_PER_TIME is not None:
                ids = ids[:MAX_CENTERS_PER_TIME]
            print("  中心点数量:", len(ids))

            for i, center_id in enumerate(ids, start=1):
                print(f"  [{i}/{len(ids)}] 提取中心点 {center_id} ...")
                row, matrix_rows = analyze_one_center(
                    dataset_name=dataset_name,
                    label=label,
                    time_folder=time_folder,
                    center_id=center_id,
                    center_root_dir=center_root_dir,
                    offset_root_dir=offset_root_dir,
                )
                all_matrix_rows.extend(matrix_rows)
                if row is not None:
                    all_feature_rows.append(row)

    if all_feature_rows:
        fieldnames = []
        for row in all_feature_rows:
            for k in row.keys():
                if k not in fieldnames:
                    fieldnames.append(k)

        with open(feature_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_feature_rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        print("\n已保存特征表:", feature_csv)
    else:
        print("\n没有提取到有效特征。")

    if all_matrix_rows:
        matrix_fields = [
            "dataset", "label", "time", "center",
            "direction", "distance_cm",
            "energy_20_70", "energy_20_40", "energy_40_70",
            "high_freq_ratio", "file",
        ]
        with open(matrix_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=matrix_fields)
            writer.writeheader()
            for row in all_matrix_rows:
                writer.writerow({k: row.get(k, "") for k in matrix_fields})
        print("已保存能量矩阵明细:", matrix_csv)

    write_feature_description(desc_path)
    print("已保存特征说明:", desc_path)

    if all_feature_rows:
        try_train_simple_classifier(feature_csv, output_dir)

    cost = time.time() - start_time
    print("\n" + "=" * 80)
    print("全部完成")
    print("特征CSV:", feature_csv)
    print("能量矩阵CSV:", matrix_csv)
    print("特征说明:", desc_path)
    print(f"耗时: {cost:.1f} 秒")
    print("=" * 80)


if __name__ == "__main__":
    main()
