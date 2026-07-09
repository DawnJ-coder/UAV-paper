# -*- coding: utf-8 -*-
"""
leak_v8_standalone_directed_wideband.py

独立完整 v8：方向性宽频 + 稳健特征 分类程序
============================================================

这个版本不依赖 v3/v4/v7/v8.1 的任何中间 CSV。
它直接从原始 WAV 数据中重新计算所有需要的特征。

核心目标：
    解决 HM20260626_144226.ld 中真/假泄漏区分困难的问题。

人工观察：
    TRUE_LEAK：
        - 有一段明显宽频；
        - 指向性明显；
        - 只有少数几个方向能量高。

    FALSE_LEAK：
        - 各方向能量比较平均；
        - 更弥散；
        - 缺少“宽频集中在少数方向”的形态。

所以 v8 新增核心特征：
    directed wideband features：
        - 方向能量集中度
        - top方向宽频覆盖度
        - top方向宽频熵/平坦度
        - 宽频是否集中在少数方向
        - 弥散宽频得分

运行：
    python leak_v8_standalone_directed_wideband.py

你只需要确认下面 DATASETS 里的原始 WAV 路径是否正确：
    TRUE_LEAK 的 center_root / offset_root
    FALSE_LEAK 的 center_root / offset_root

输出：
    C:\\Users\\jiangxinru6\\Desktop\\wurenji\\leak_v8_standalone_directed_wideband_results

重点输出：
    v8_report.txt
    v8_feature_dataset.csv
    v8_group_validation_summary.csv
    v8_predictions.csv
    v8_144226_pair_check.csv
    v8_144226_directed_wideband_feature_compare.csv
"""

import os
import re
import json
import math
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy import signal
from scipy.stats import kurtosis


# ============================================================
# 1. 配置区：你主要需要检查这里
# ============================================================

BASE_DIR = r"C:\Users\jiangxinru6\Desktop\wurenji"

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "leak_v8_standalone_directed_wideband_results"
)

# 四个时间文件夹
TIME_FOLDERS = [
    "HM20260626_142938.ld",
    "HM20260626_143034.ld",
    "HM20260626_144226.ld",
    "HM20260626_144325.ld",
]

TARGET_TIME = "HM20260626_144226.ld"

# ------------------------------------------------------------
# 原始 WAV 数据路径
# ------------------------------------------------------------
# 注意：
#   这里必须是原始 wav 所在目录，不是 v3/v7/v8 的结果目录。
#
# 如果你的假泄漏原始 wav 根目录名字不一样，
# 只需要改 FALSE_LEAK 的 center_root / offset_root。
# ------------------------------------------------------------

DATASETS = [
    {
        "label": "TRUE_LEAK",
        "center_root": r"D:\gas\beamform_results",
        "offset_root": r"D:\gas\beamform_results_offset_multiple",
    },
    {
        "label": "FALSE_LEAK",
        # 如果你的假泄漏原始中心点目录不是这个名字，请改这里
        "center_root": r"D:\gas\beamform_results_cs",
        # 如果你的假泄漏原始偏移点目录不是这个名字，请改这里
        "offset_root": r"D:\gas\beamform_results_cs_offset_multiple",
    },
]

# 如果你不知道假泄漏原始目录在哪里，可以先运行程序。
# 程序会检查路径是否存在，并提示你需要改哪一项。

# 采样与频段
FREQ_LOW = 20000
FREQ_HIGH = 70000
SUBBANDS = [
    (20000, 30000),
    (30000, 40000),
    (40000, 50000),
    (50000, 60000),
    (60000, 70000),
]

# Welch 参数
NFFT = 4096
WELCH_NPERSEG = 4096
WELCH_NOVERLAP = 2048

# 近场距离：用于判断“方向性宽频”
NEAR_DISTANCE_MAX_CM = 20

# 方向和距离
DIRECTIONS = [
    "up",
    "down",
    "left",
    "right",
    "up_left",
    "up_right",
    "down_left",
    "down_right",
]

DISTANCES_CM = [5, 10, 15, 20, 25, 30, 35, 40]

# 频段激活阈值：用于判断某方向是不是有“宽频覆盖”
BAND_ACTIVE_REL_TO_MAX_IN_DIRECTION = 0.18
BAND_ACTIVE_FRAC_OF_DIRECTION_TOTAL = 0.06

# 方向激活阈值：用于判断有几个方向明显
DIR_ACTIVE_REL_TO_MAX = 0.25

# 是否使用缓存
USE_CACHE = True
CACHE_FEATURE_CSV = os.path.join(OUTPUT_DIR, "v8_feature_dataset.csv")

# 模型设置
RANDOM_STATE = 42
THRESHOLD_GRID = np.linspace(0.01, 0.99, 99)
RANK_TRUE_FRACTION = 0.50

# 运行速度控制
# None 表示处理所有 center。
# 调试时可以设为 3。
MAX_CENTERS_PER_TIME = None

# 文件搜索
WAV_EXTS = [".wav", ".WAV"]


# ============================================================
# 2. 基础工具
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_name(s):
    return str(s).replace("\\", "_").replace("/", "_").replace(":", "_").replace(".", "_")


def normalize_center_id(x):
    try:
        if pd.isna(x):
            return "00"
    except Exception:
        pass

    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]

    # 优先 center_XX
    m = re.search(r"(?:center|centre|c)[_\-\s]*(\d{1,3})", s, re.IGNORECASE)
    if m:
        return m.group(1).zfill(2)

    nums = "".join(ch for ch in s if ch.isdigit())
    if nums == "":
        return s

    # 防止从 time_folder 里抓到太多数字，只取最后两位/三位
    if len(nums) > 3:
        nums = nums[-2:]

    return nums.zfill(2)


def safe_float_series(s):
    x = pd.to_numeric(s, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    return x


def label_to_y(labels):
    return np.array([1 if str(x) == "TRUE_LEAK" else 0 for x in labels], dtype=int)


def y_to_label(y):
    return "TRUE_LEAK" if int(y) == 1 else "FALSE_LEAK"


def entropy_norm(x):
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 0)

    total = np.sum(x)
    if total <= 1e-20:
        return 0.0

    p = x / total
    p = p[p > 0]

    if len(p) <= 1:
        return 0.0

    return float(-np.sum(p * np.log(p + 1e-20)) / np.log(len(p)))


def spectral_flatness_from_values(x):
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 1e-20)

    if len(x) == 0:
        return 0.0

    return float(np.exp(np.mean(np.log(x))) / (np.mean(x) + 1e-20))


def gini_coefficient(x):
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 0)

    if len(x) == 0 or np.sum(x) <= 1e-20:
        return 0.0

    x = np.sort(x)
    n = len(x)
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def cohen_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) < 2 or len(b) < 2:
        return np.nan

    ma, mb = np.mean(a), np.mean(b)
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    pooled = math.sqrt((sa * sa + sb * sb) / 2.0) + 1e-12
    return float((ma - mb) / pooled)


def safe_auc(y_true, score):
    try:
        from sklearn.metrics import roc_auc_score
        y_true = np.asarray(y_true, dtype=int)
        score = np.asarray(score, dtype=float)

        if len(np.unique(y_true)) < 2:
            return np.nan

        return float(roc_auc_score(y_true, score))
    except Exception:
        return np.nan


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0

    recall_true = tp / (tp + fn + 1e-12)
    recall_false = tn / (tn + fp + 1e-12)
    bal_acc = 0.5 * (recall_true + recall_false)

    precision_true = tp / (tp + fp + 1e-12)
    f1_true = 2 * precision_true * recall_true / (precision_true + recall_true + 1e-12)

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "recall_TRUE": float(recall_true),
        "recall_FALSE": float(recall_false),
        "precision_TRUE": float(precision_true),
        "f1_TRUE": float(f1_true),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ============================================================
# 3. 文件发现与解析
# ============================================================

def list_wav_files(root):
    out = []

    if not os.path.exists(root):
        return out

    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if any(name.endswith(ext) for ext in WAV_EXTS):
                out.append(os.path.join(dirpath, name))

    return sorted(out)


def parse_center_from_path(path):
    """
    解析 center 编号。

    你的 offset 文件名格式是：
        HM20260626_142938.ld_00_00d5_down_beamform_result.wav
        HM20260626_142938.ld_00_00d10_up_left_beamform_result.wav

    其中：
        .ld_ 后面的第一个 00 才是 center；
        00d5 / 00d10 里的 d5 / d10 是距离，不是 center。

    之前版本错误地从文件名最后抓数字，容易把 d5 误认为 center_05，
    导致整个文件夹只识别出 8 个 offset 组合。
    """
    base = os.path.basename(str(path))

    # 最关键：匹配 .ld_00_00d5_down...
    m = re.search(r"\.ld[_\-](\d{1,3})[_\-]", base, flags=re.IGNORECASE)
    if m:
        return m.group(1).zfill(2)

    # 兜底：center_00
    patterns = [
        r"(?:center|centre)[_\-\s]*(\d{1,3})",
        r"(?:^|[\\/_.\-])c[_\-\s]*(\d{1,3})(?:[\\/_.\-]|$)",
    ]

    s = str(path)
    for pat in patterns:
        ms = re.findall(pat, s, flags=re.IGNORECASE)
        if ms:
            return str(ms[-1]).zfill(2)

    # 最后兜底：如果文件名开头就是 00_xxx
    m = re.match(r"^(\d{1,3})[_\-]", base)
    if m:
        return m.group(1).zfill(2)

    return None


def parse_direction_from_path(path):
    """
    解析方向。

    兼容：
        ..._00d5_down_beamform_result.wav
        ..._00d5_down_left_beamform_result.wav
    """
    base = os.path.basename(str(path)).lower().replace("-", "_").replace(" ", "_")
    s = str(path).lower().replace("-", "_").replace(" ", "_")

    # 优先匹配你的命名：00d5_down_left_beamform
    m = re.search(
        r"\d{0,3}d\d{1,3}_(up_left|up_right|down_left|down_right|up|down|left|right)_beamform",
        base,
        flags=re.IGNORECASE
    )
    if m:
        return m.group(1).lower()

    # 注意先匹配组合方向，避免 up 被 up_left 提前匹配
    alias = [
        ("up_left", ["up_left", "upleft", "ul", "left_up"]),
        ("up_right", ["up_right", "upright", "ur", "right_up"]),
        ("down_left", ["down_left", "downleft", "dl", "left_down"]),
        ("down_right", ["down_right", "downright", "dr", "right_down"]),
        ("up", ["_up_", "/up/", "\\up\\", "dir_up", "direction_up"]),
        ("down", ["_down_", "/down/", "\\down\\", "dir_down", "direction_down"]),
        ("left", ["_left_", "/left/", "\\left\\", "dir_left", "direction_left"]),
        ("right", ["_right_", "/right/", "\\right\\", "dir_right", "direction_right"]),
    ]

    padded = "_" + s + "_"

    for direction, keys in alias:
        for k in keys:
            kk = k.lower()
            if kk in padded or kk in s:
                return direction

    return None


def parse_distance_from_path(path):
    """
    解析偏移距离。

    兼容你的格式：
        HM20260626_142938.ld_00_00d5_down_beamform_result.wav
    这里的 00d5 表示距离 5cm；
    00d10 表示距离 10cm。
    """
    base = os.path.basename(str(path)).lower()
    s = str(path).lower()

    # 优先匹配 .ld_00_00d5_down / .ld_00_00d10_down_left
    m = re.search(r"\.ld[_\-]\d{1,3}[_\-]\d{1,3}d(\d{1,3})(?=[_\-])", base, flags=re.IGNORECASE)
    if m:
        val = int(m.group(1))
        if 0 < val <= 200:
            return val

    # 5cm / 05cm
    ms = re.findall(r"(?<!\d)(\d{1,3})\s*cm(?!\w)", s)
    if ms:
        val = int(ms[-1])
        if 0 < val <= 200:
            return val

    # dist_5 / distance_05 / d05 / 00d5
    patterns = [
        r"(?:dist|distance)[_\-\s]*(\d{1,3})",
        r"(?<![a-zA-Z])\d{0,3}d(\d{1,3})(?=[_\-\\/\.]|$)",
        r"(?<![a-zA-Z])d[_\-\s]*(\d{1,3})",
    ]

    for pat in patterns:
        ms = re.findall(pat, s)
        if ms:
            val = int(ms[-1])
            if 0 < val <= 200:
                return val

    return None


def discover_center_files(center_root, time_folder):
    root = os.path.join(center_root, time_folder)
    files = list_wav_files(root)

    out = {}

    for f in files:
        center = parse_center_from_path(f)
        if center is None:
            continue

        if center not in out:
            out[center] = f

    return out


def discover_offset_files(offset_root, time_folder):
    root = os.path.join(offset_root, time_folder)
    files = list_wav_files(root)

    out = {}

    for f in files:
        center = parse_center_from_path(f)
        direction = parse_direction_from_path(f)
        dist = parse_distance_from_path(f)

        if center is None or direction is None or dist is None:
            continue

        key = (center, direction, int(dist))

        if key not in out:
            out[key] = []
        out[key].append(f)

    return out


# ============================================================
# 4. WAV 分析：频谱、频段能量、时间特征
# ============================================================

def read_wav_float(path):
    fs, x = wavfile.read(path)

    if x.ndim > 1:
        # 多通道取平均。若你的 wav 是阵列通道，这里代表波束/文件已生成的单点信号。
        x = x.astype(np.float64).mean(axis=1)
    else:
        x = x.astype(np.float64)

    # 转 float
    if np.issubdtype(x.dtype, np.integer):
        # 这行一般不会进来，因为前面已 astype float64
        pass

    max_abs = np.max(np.abs(x)) + 1e-12

    # 如果原始是 int16 转过来的，幅值会很大；统一缩到近似 [-1,1]
    if max_abs > 10:
        x = x / max_abs

    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)

    return fs, x


def welch_psd(x, fs):
    nperseg = min(WELCH_NPERSEG, len(x))
    if nperseg < 256:
        return np.array([]), np.array([])

    noverlap = min(WELCH_NOVERLAP, max(0, nperseg // 2))

    f, pxx = signal.welch(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=max(NFFT, nperseg),
        scaling="density",
    )

    pxx = np.maximum(pxx, 0)
    return f, pxx


def band_energy_from_psd(f, pxx, lo, hi):
    mask = (f >= lo) & (f < hi)

    if not np.any(mask):
        return 0.0

    return float(np.trapz(pxx[mask], f[mask]))


def analyze_spectrum_file(path):
    fs, x = read_wav_float(path)

    f, pxx = welch_psd(x, fs)

    if len(f) == 0:
        return None

    band_total = band_energy_from_psd(f, pxx, FREQ_LOW, FREQ_HIGH)

    sub = {}
    for lo, hi in SUBBANDS:
        sub[(lo, hi)] = band_energy_from_psd(f, pxx, lo, hi)

    mask = (f >= FREQ_LOW) & (f <= FREQ_HIGH)
    fb = f[mask]
    pb = pxx[mask]

    if len(fb) == 0 or np.sum(pb) <= 1e-20:
        centroid = 0.0
        bandwidth = 0.0
        entropy = 0.0
        flat = 0.0
        peak_freq = 0.0
        rolloff = 0.0
        slope = 0.0
        peakiness = 0.0
    else:
        total_p = np.sum(pb) + 1e-20
        centroid = float(np.sum(fb * pb) / total_p)
        bandwidth = float(np.sqrt(np.sum(((fb - centroid) ** 2) * pb) / total_p))
        entropy = entropy_norm(pb)
        flat = spectral_flatness_from_values(pb)
        peak_idx = int(np.argmax(pb))
        peak_freq = float(fb[peak_idx])
        peakiness = float(np.max(pb) / (np.mean(pb) + 1e-20))

        cum = np.cumsum(pb)
        idx = int(np.searchsorted(cum, 0.85 * cum[-1]))
        idx = min(idx, len(fb) - 1)
        rolloff = float(fb[idx])

        # log PSD 对频率的斜率
        try:
            y = np.log10(pb + 1e-20)
            xfreq = (fb - fb.mean()) / (fb.std() + 1e-12)
            slope = float(np.polyfit(xfreq, y, 1)[0])
        except Exception:
            slope = 0.0

    return {
        "fs": fs,
        "n_samples": len(x),
        "band_energy_20_70": band_total,
        "subband_energy": sub,
        "freq": f,
        "psd": pxx,
        "spec_centroid_hz": centroid,
        "spec_bandwidth_hz": bandwidth,
        "spec_entropy": entropy,
        "spec_flatness": flat,
        "spec_peak_freq_hz": peak_freq,
        "spec_rolloff_85_hz": rolloff,
        "spec_slope": slope,
        "spec_peakiness": peakiness,
    }


def bandpass_signal(x, fs, lo=FREQ_LOW, hi=FREQ_HIGH):
    nyq = fs / 2

    hi2 = min(hi, nyq * 0.95)
    lo2 = min(lo, hi2 * 0.8)

    if lo2 <= 0 or hi2 <= lo2:
        return x

    try:
        sos = signal.butter(4, [lo2 / nyq, hi2 / nyq], btype="bandpass", output="sos")
        if len(x) > 3 * 4 * 2:
            return signal.sosfiltfilt(sos, x)
        return signal.sosfilt(sos, x)
    except Exception:
        return x


def time_features_from_wav(path):
    try:
        fs, x = read_wav_float(path)
        xb = bandpass_signal(x, fs)

        # 20ms窗，10ms hop
        win = max(64, int(0.020 * fs))
        hop = max(32, int(0.010 * fs))

        energies = []
        for start in range(0, max(1, len(xb) - win + 1), hop):
            seg = xb[start:start + win]
            if len(seg) < win // 2:
                continue
            energies.append(float(np.mean(seg ** 2)))

        if len(energies) == 0:
            energies = [float(np.mean(xb ** 2))]

        e = np.asarray(energies, dtype=float)
        mean = float(np.mean(e))
        std = float(np.std(e))
        cv = float(std / (mean + 1e-20))
        max_mean = float(np.max(e) / (mean + 1e-20))
        kurt = float(kurtosis(e, fisher=False, bias=False)) if len(e) >= 4 else 0.0
        rms = float(np.sqrt(np.mean(xb ** 2)))

        return {
            "time_energy_mean": mean,
            "time_energy_std": std,
            "time_energy_cv": cv,
            "time_energy_max_mean_ratio": max_mean,
            "time_energy_kurtosis": kurt,
            "time_rms": rms,
        }
    except Exception:
        return {
            "time_energy_mean": np.nan,
            "time_energy_std": np.nan,
            "time_energy_cv": np.nan,
            "time_energy_max_mean_ratio": np.nan,
            "time_energy_kurtosis": np.nan,
            "time_rms": np.nan,
        }


# ============================================================
# 5. 空间/频谱/方向性宽频特征
# ============================================================

def fit_decay(distance_values, energy_values):
    d = np.asarray(distance_values, dtype=float)
    e = np.asarray(energy_values, dtype=float)

    mask = np.isfinite(d) & np.isfinite(e) & (d > 0) & (e > 0)

    d = d[mask]
    e = e[mask]

    if len(d) < 3:
        return 0.0, 0.0

    x = np.log(d)
    y = np.log(e + 1e-20)

    try:
        coef = np.polyfit(x, y, 1)
        slope = coef[0]
        n = float(-slope)

        y_hat = np.polyval(coef, x)
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-20
        r2 = 1.0 - ss_res / ss_tot

        return max(0.0, n), float(np.clip(r2, 0.0, 1.0))
    except Exception:
        return 0.0, 0.0


def compute_direction_wideband_from_matrix(matrix):
    """
    matrix: shape [n_directions, n_bands]
    行是方向，列是频段。
    """
    matrix = np.asarray(matrix, dtype=float)
    matrix = np.maximum(matrix, 0)

    total = float(np.sum(matrix))
    n_dir, n_band = matrix.shape

    if total <= 1e-20:
        return {
            "dw_total_near_band_energy": 0.0,
            "dw_top1_direction_ratio": 0.0,
            "dw_top2_direction_ratio": 0.0,
            "dw_top3_direction_ratio": 0.0,
            "dw_direction_entropy_norm": 0.0,
            "dw_direction_cv": 0.0,
            "dw_direction_gini": 0.0,
            "dw_direction_active_count": 0,
            "dw_top1_wideband_coverage": 0.0,
            "dw_top2_wideband_coverage": 0.0,
            "dw_top1_wideband_quality": 0.0,
            "dw_top2_wideband_quality": 0.0,
            "dw_rest_wideband_quality": 0.0,
            "dw_directed_wideband_score": 0.0,
            "dw_diffuse_wideband_score": 0.0,
            "dw_directional_wideband_contrast": 0.0,
            "dw_top2_wideband_minus_rest": 0.0,
            "dw_matrix_entropy_norm": 0.0,
            "dw_matrix_top10_ratio": 0.0,
            "dw_matrix_top20_ratio": 0.0,
        }

    dir_total = matrix.sum(axis=1)
    sort_idx = np.argsort(dir_total)[::-1]

    top1 = sort_idx[0]
    top2 = sort_idx[:min(2, len(sort_idx))]
    top3 = sort_idx[:min(3, len(sort_idx))]

    top1_ratio = float(dir_total[top1] / (total + 1e-20))
    top2_ratio = float(np.sum(dir_total[top2]) / (total + 1e-20))
    top3_ratio = float(np.sum(dir_total[top3]) / (total + 1e-20))

    dir_entropy = entropy_norm(dir_total)
    dir_cv = float(np.std(dir_total) / (np.mean(dir_total) + 1e-20))
    dir_gini = gini_coefficient(dir_total)

    max_dir = float(np.max(dir_total))
    active_dir_count = int(np.sum(dir_total >= max_dir * DIR_ACTIVE_REL_TO_MAX)) if max_dir > 0 else 0

    # 每个方向的宽频覆盖度
    coverage = []
    band_entropy = []
    band_flatness = []

    for i in range(n_dir):
        b = matrix[i, :]
        dtotal = float(np.sum(b))

        if dtotal <= 1e-20:
            coverage.append(0.0)
            band_entropy.append(0.0)
            band_flatness.append(0.0)
            continue

        max_band = float(np.max(b))
        active = (
            (b >= max_band * BAND_ACTIVE_REL_TO_MAX_IN_DIRECTION) &
            (b >= dtotal * BAND_ACTIVE_FRAC_OF_DIRECTION_TOTAL)
        )

        coverage.append(float(np.sum(active) / n_band))
        band_entropy.append(entropy_norm(b))
        band_flatness.append(spectral_flatness_from_values(b))

    coverage = np.asarray(coverage, dtype=float)
    band_entropy = np.asarray(band_entropy, dtype=float)
    band_flatness = np.asarray(band_flatness, dtype=float)

    wide_quality = (
        0.50 * coverage +
        0.35 * band_entropy +
        0.15 * band_flatness
    )

    top1_wide = float(wide_quality[top1])
    top2_wide = float(np.average(wide_quality[top2], weights=dir_total[top2] + 1e-20))
    top1_cov = float(coverage[top1])
    top2_cov = float(np.average(coverage[top2], weights=dir_total[top2] + 1e-20))

    rest_idx = [i for i in range(n_dir) if i not in set(top2.tolist())]

    if len(rest_idx):
        rest_wide = float(np.average(wide_quality[rest_idx], weights=dir_total[rest_idx] + 1e-20))
        rest_wide_energy = float(np.sum(wide_quality[rest_idx] * dir_total[rest_idx]))
    else:
        rest_wide = 0.0
        rest_wide_energy = 0.0

    top2_wide_energy = float(np.sum(wide_quality[top2] * dir_total[top2]))

    direction_concentration = max(0.0, 1.0 - dir_entropy)

    # 最核心：宽频集中在少数方向
    directed_wideband_score = float(
        top2_ratio *
        top2_wide *
        (0.55 + 0.45 * direction_concentration) *
        (0.50 + 0.50 * min(dir_cv, 3.0) / 3.0)
    )

    # 弥散宽频：方向越平均、非top方向也越宽频，越高
    diffuse_wideband_score = float(
        dir_entropy *
        (0.5 * rest_wide + 0.5 * (1.0 - top2_ratio))
    )

    if len(rest_idx):
        directional_wideband_contrast = float(
            (top2_wide_energy / (len(top2) + 1e-20)) /
            (rest_wide_energy / (len(rest_idx) + 1e-20) + 1e-20)
        )
    else:
        directional_wideband_contrast = 999.0

    top2_minus_rest = float(top2_wide - rest_wide)

    flat_cells = np.sort(matrix.flatten())[::-1]
    matrix_entropy = entropy_norm(flat_cells)
    top10_n = max(1, int(len(flat_cells) * 0.10))
    top20_n = max(1, int(len(flat_cells) * 0.20))

    matrix_top10_ratio = float(np.sum(flat_cells[:top10_n]) / (total + 1e-20))
    matrix_top20_ratio = float(np.sum(flat_cells[:top20_n]) / (total + 1e-20))

    band_total = matrix.sum(axis=0)
    band_p = band_total / (np.sum(band_total) + 1e-20)

    top2_band = matrix[top2, :].sum(axis=0)
    top2_band_p = top2_band / (np.sum(top2_band) + 1e-20)

    out = {
        "dw_total_near_band_energy": total,
        "dw_top1_direction_ratio": top1_ratio,
        "dw_top2_direction_ratio": top2_ratio,
        "dw_top3_direction_ratio": top3_ratio,
        "dw_direction_entropy_norm": dir_entropy,
        "dw_direction_cv": dir_cv,
        "dw_direction_gini": dir_gini,
        "dw_direction_active_count": active_dir_count,

        "dw_top1_wideband_coverage": top1_cov,
        "dw_top2_wideband_coverage": top2_cov,
        "dw_top1_wideband_quality": top1_wide,
        "dw_top2_wideband_quality": top2_wide,
        "dw_rest_wideband_quality": rest_wide,

        "dw_directed_wideband_score": directed_wideband_score,
        "dw_diffuse_wideband_score": diffuse_wideband_score,
        "dw_directional_wideband_contrast": directional_wideband_contrast,
        "dw_top2_wideband_minus_rest": top2_minus_rest,

        "dw_matrix_entropy_norm": matrix_entropy,
        "dw_matrix_top10_ratio": matrix_top10_ratio,
        "dw_matrix_top20_ratio": matrix_top20_ratio,

        "dw_global_20_30_ratio": float(band_p[0]),
        "dw_global_30_40_ratio": float(band_p[1]),
        "dw_global_40_50_ratio": float(band_p[2]),
        "dw_global_50_60_ratio": float(band_p[3]),
        "dw_global_60_70_ratio": float(band_p[4]),

        "dw_top2_20_30_ratio": float(top2_band_p[0]),
        "dw_top2_30_40_ratio": float(top2_band_p[1]),
        "dw_top2_40_50_ratio": float(top2_band_p[2]),
        "dw_top2_50_60_ratio": float(top2_band_p[3]),
        "dw_top2_60_70_ratio": float(top2_band_p[4]),
    }

    return out


def compute_sample_features(label, time_folder, center, center_file, offset_files_for_time):
    """
    对一个样本，即某 time_folder + center，计算所有特征。
    """
    row = {
        "label": label,
        "time": time_folder,
        "center": center,
        "center_norm": normalize_center_id(center),
        "center_file": center_file if center_file else "",
    }

    # --------------------------------------------------------
    # 读取所有 offset wav，构造 direction x distance x bands
    # --------------------------------------------------------
    direction_distance_energy = {d: {} for d in DIRECTIONS}
    direction_distance_band = {d: {} for d in DIRECTIONS}
    direction_distance_spec = {d: {} for d in DIRECTIONS}

    found_count = 0

    for direction in DIRECTIONS:
        for dist in DISTANCES_CM:
            key = (center, direction, dist)
            files = offset_files_for_time.get(key, [])

            if not files:
                continue

            total_energy_list = []
            band_energy_list = []
            spec_list = []

            for f in files:
                try:
                    res = analyze_spectrum_file(f)
                    if res is None:
                        continue

                    total_energy_list.append(res["band_energy_20_70"])
                    band_energy_list.append([res["subband_energy"][(lo, hi)] for lo, hi in SUBBANDS])
                    spec_list.append(res)
                    found_count += 1
                except Exception:
                    continue

            if total_energy_list:
                direction_distance_energy[direction][dist] = float(np.mean(total_energy_list))
                direction_distance_band[direction][dist] = np.mean(np.asarray(band_energy_list), axis=0)
                direction_distance_spec[direction][dist] = spec_list[0]

    row["offset_wav_count_used"] = found_count

    if found_count == 0:
        # 没有偏移文件，返回空特征
        return row

    # --------------------------------------------------------
    # 空间方向能量特征
    # --------------------------------------------------------
    dir_near_energy = []
    dir_all_energy = []
    matrix_near = np.zeros((len(DIRECTIONS), len(SUBBANDS)), dtype=float)

    for i, direction in enumerate(DIRECTIONS):
        near_vals = []
        all_vals = []

        for dist, e in direction_distance_energy[direction].items():
            all_vals.append(e)
            if dist <= NEAR_DISTANCE_MAX_CM:
                near_vals.append(e)

        if len(near_vals) == 0:
            near_vals = all_vals

        dir_near_energy.append(float(np.sum(near_vals)) if near_vals else 0.0)
        dir_all_energy.append(float(np.sum(all_vals)) if all_vals else 0.0)

        for dist, bands in direction_distance_band[direction].items():
            if dist <= NEAR_DISTANCE_MAX_CM:
                matrix_near[i, :] += np.asarray(bands, dtype=float)

    dir_near_energy = np.asarray(dir_near_energy, dtype=float)
    dir_all_energy = np.asarray(dir_all_energy, dtype=float)

    total_near = float(np.sum(dir_near_energy))
    total_all = float(np.sum(dir_all_energy))

    sort_idx = np.argsort(dir_near_energy)[::-1]
    best_i = int(sort_idx[0])
    best_direction = DIRECTIONS[best_i]
    second_i = int(sort_idx[1]) if len(sort_idx) > 1 else best_i

    row["best_direction"] = best_direction
    row["raw_best_energy"] = float(dir_near_energy[best_i])
    row["mean_direction_energy"] = float(np.mean(dir_near_energy))
    row["direction_energy_std"] = float(np.std(dir_near_energy))
    row["direction_cv"] = float(np.std(dir_near_energy) / (np.mean(dir_near_energy) + 1e-20))
    row["direction_entropy"] = entropy_norm(dir_near_energy)
    row["direction_gini"] = gini_coefficient(dir_near_energy)
    row["direction_top1_ratio"] = float(dir_near_energy[best_i] / (total_near + 1e-20))
    row["direction_top2_ratio"] = float((dir_near_energy[best_i] + dir_near_energy[second_i]) / (total_near + 1e-20))

    others = np.delete(dir_near_energy, best_i)
    row["direction_contrast"] = float(dir_near_energy[best_i] / (np.mean(others) + 1e-20)) if len(others) else 0.0
    row["direction_active_count"] = int(np.sum(dir_near_energy >= np.max(dir_near_energy) * DIR_ACTIVE_REL_TO_MAX)) if np.max(dir_near_energy) > 0 else 0

    # --------------------------------------------------------
    # 衰减特征：沿最佳方向
    # --------------------------------------------------------
    dists = []
    energies = []
    for dist in DISTANCES_CM:
        if dist in direction_distance_energy[best_direction]:
            dists.append(dist)
            energies.append(direction_distance_energy[best_direction][dist])

    attenuation_n, decay_R2 = fit_decay(dists, energies)
    row["attenuation_n"] = attenuation_n
    row["decay_R2"] = decay_R2

    if energies:
        near_e = np.mean([e for d, e in zip(dists, energies) if d <= 20]) if any(d <= 20 for d in dists) else np.mean(energies)
        far_e = np.mean([e for d, e in zip(dists, energies) if d >= 30]) if any(d >= 30 for d in dists) else np.mean(energies)
        row["near_far_ratio"] = float(near_e / (far_e + 1e-20))
        row["energy_5cm_best_direction"] = float(direction_distance_energy[best_direction].get(5, np.nan))
    else:
        row["near_far_ratio"] = np.nan
        row["energy_5cm_best_direction"] = np.nan

    # monotonic ratio
    if len(energies) >= 2:
        decreases = 0
        pairs = 0
        for a, b in zip(energies[:-1], energies[1:]):
            pairs += 1
            if a >= b:
                decreases += 1
        row["monotonic_decay_ratio"] = decreases / pairs if pairs else 0
    else:
        row["monotonic_decay_ratio"] = 0.0

    # --------------------------------------------------------
    # 方向性宽频特征
    # --------------------------------------------------------
    dw = compute_direction_wideband_from_matrix(matrix_near)

    for k, v in dw.items():
        row[k] = v

    # --------------------------------------------------------
    # 频谱特征：最佳方向近场聚合
    # --------------------------------------------------------
    best_band = matrix_near[best_i, :]
    best_total = float(np.sum(best_band))

    if best_total <= 1e-20:
        # 如果近场矩阵没有，退回全部最佳方向距离
        best_band = np.zeros(len(SUBBANDS), dtype=float)
        for dist, bands in direction_distance_band[best_direction].items():
            best_band += np.asarray(bands, dtype=float)
        best_total = float(np.sum(best_band))

    row["energy_20_70"] = best_total

    for j, (lo, hi) in enumerate(SUBBANDS):
        name = f"energy_{lo//1000}_{hi//1000}k"
        ratio_name = f"ratio_{lo//1000}_{hi//1000}k"
        row[name] = float(best_band[j])
        row[ratio_name] = float(best_band[j] / (best_total + 1e-20))

    low_20_40 = best_band[0] + best_band[1]
    high_40_70 = best_band[2] + best_band[3] + best_band[4]
    row["energy_20_40"] = float(low_20_40)
    row["energy_40_70"] = float(high_40_70)
    row["high_freq_ratio"] = float(high_40_70 / (best_total + 1e-20))

    # 从最佳方向最近的可用距离取详细 PSD 频谱特征
    chosen_spec = None
    for dist in DISTANCES_CM:
        if dist in direction_distance_spec[best_direction]:
            chosen_spec = direction_distance_spec[best_direction][dist]
            break

    if chosen_spec is not None:
        for k in [
            "spec_centroid_hz",
            "spec_bandwidth_hz",
            "spec_entropy",
            "spec_flatness",
            "spec_peak_freq_hz",
            "spec_rolloff_85_hz",
            "spec_slope",
            "spec_peakiness",
        ]:
            row[k] = chosen_spec.get(k, np.nan)
    else:
        row["spec_centroid_hz"] = np.nan
        row["spec_bandwidth_hz"] = np.nan
        row["spec_entropy"] = np.nan
        row["spec_flatness"] = np.nan
        row["spec_peak_freq_hz"] = np.nan
        row["spec_rolloff_85_hz"] = np.nan
        row["spec_slope"] = np.nan
        row["spec_peakiness"] = np.nan

    # --------------------------------------------------------
    # 时间特征：优先 center wav，否则用最佳方向最近 offset wav
    # --------------------------------------------------------
    time_source = center_file

    if not time_source:
        for dist in DISTANCES_CM:
            key = (center, best_direction, dist)
            files = offset_files_for_time.get(key, [])
            if files:
                time_source = files[0]
                break

    tf = {}
    if time_source and os.path.exists(time_source):
        tf = time_features_from_wav(time_source)
    else:
        tf = {
            "time_energy_mean": np.nan,
            "time_energy_std": np.nan,
            "time_energy_cv": np.nan,
            "time_energy_max_mean_ratio": np.nan,
            "time_energy_kurtosis": np.nan,
            "time_rms": np.nan,
        }

    row.update(tf)

    # --------------------------------------------------------
    # 组合物理分数
    # --------------------------------------------------------
    # 值越高越像：宽频 + 少数方向集中 + 非弥散
    row["physics_directed_leak_score"] = float(
        row.get("dw_directed_wideband_score", 0.0)
        * (1.0 + min(row.get("direction_cv", 0.0), 3.0))
        * (1.0 - min(row.get("dw_diffuse_wideband_score", 0.0), 1.0) * 0.5)
    )

    row["physics_false_diffuse_score"] = float(
        row.get("dw_diffuse_wideband_score", 0.0)
        * (row.get("direction_entropy", 0.0) + 1e-20)
        / (row.get("dw_directed_wideband_score", 0.0) + 1e-20)
    )

    return row


# ============================================================
# 6. 构建完整特征表
# ============================================================

def validate_paths():
    print("\n检查原始 WAV 路径...")
    ok = True

    for ds in DATASETS:
        label = ds["label"]
        center_root = ds["center_root"]
        offset_root = ds["offset_root"]

        print(f"\n[{label}]")
        print("  center_root:", center_root)
        print("  offset_root:", offset_root)

        if not os.path.exists(center_root):
            print("  [错误] center_root 不存在")
            ok = False
        else:
            print("  center_root 存在")

        if not os.path.exists(offset_root):
            print("  [错误] offset_root 不存在")
            ok = False
        else:
            print("  offset_root 存在")

    if not ok:
        raise FileNotFoundError(
            "\n至少有一个原始 WAV 路径不存在。\n"
            "请打开本程序，修改 DATASETS 里的 center_root / offset_root。\n"
            "注意：这里要填原始 WAV 目录，不是 v3/v7/v8 的结果目录。"
        )


def extract_all_features():
    ensure_dir(OUTPUT_DIR)

    if USE_CACHE and os.path.exists(CACHE_FEATURE_CSV):
        print("发现缓存特征表，直接读取:", CACHE_FEATURE_CSV)
        return pd.read_csv(CACHE_FEATURE_CSV)

    validate_paths()

    rows = []

    for ds in DATASETS:
        label = ds["label"]
        center_root = ds["center_root"]
        offset_root = ds["offset_root"]

        print("\n" + "=" * 100)
        print("开始处理数据集:", label)
        print("=" * 100)

        for time_folder in TIME_FOLDERS:
            print(f"\n处理 {label} / {time_folder}")

            center_files = discover_center_files(center_root, time_folder)
            offset_files = discover_offset_files(offset_root, time_folder)

            centers_from_center = set(center_files.keys())
            centers_from_offset = set([k[0] for k in offset_files.keys()])
            centers = sorted(centers_from_center | centers_from_offset)

            if MAX_CENTERS_PER_TIME is not None:
                centers = centers[:MAX_CENTERS_PER_TIME]

            print("  center数量:", len(centers))
            print("  center wav数量:", len(center_files))
            print("  offset wav组合数量:", len(offset_files))

            # 诊断：正常情况下每个 center 应接近 8方向×8距离=64 个 offset组合
            if len(centers) > 0:
                offset_count_by_center = {}
                for (cc, dd, dist), files in offset_files.items():
                    offset_count_by_center.setdefault(cc, 0)
                    offset_count_by_center[cc] += 1
                shown = sorted(offset_count_by_center.items())[:5]
                avg_offset_per_center = np.mean(list(offset_count_by_center.values())) if offset_count_by_center else 0
                print(f"  平均每个center识别到offset组合数: {avg_offset_per_center:.1f} / 64")
                print("  前5个center offset组合数:", shown)

            for i, center in enumerate(centers, 1):
                if i % 10 == 0 or i == len(centers):
                    print(f"  已处理 {i}/{len(centers)}")

                center_file = center_files.get(center, "")

                try:
                    row = compute_sample_features(
                        label=label,
                        time_folder=time_folder,
                        center=center,
                        center_file=center_file,
                        offset_files_for_time=offset_files,
                    )
                    rows.append(row)
                except Exception as e:
                    rows.append({
                        "label": label,
                        "time": time_folder,
                        "center": center,
                        "center_norm": normalize_center_id(center),
                        "error": str(e),
                    })

    df = pd.DataFrame(rows)

    # 只保留有 offset 的样本
    if "offset_wav_count_used" in df.columns:
        before = len(df)
        df = df[df["offset_wav_count_used"].fillna(0) > 0].copy()
        after = len(df)
        print(f"\n去除无offset样本: {before} -> {after}")

    df.to_csv(CACHE_FEATURE_CSV, index=False, encoding="utf-8-sig")
    print("\n完整特征表已保存:", CACHE_FEATURE_CSV)

    return df


# ============================================================
# 7. v7式稳健特征：time内部 z/rank
# ============================================================

def numeric_feature_columns(df):
    ignore = {
        "label", "time", "center", "center_norm", "center_file",
        "best_direction", "error",
    }

    cols = []

    for c in df.columns:
        if c in ignore:
            continue

        vals = safe_float_series(df[c])
        if vals.notna().mean() >= 0.75:
            cols.append(c)

    return cols


def add_time_relative_features(df, cols):
    df = df.copy()

    for c in cols:
        vals = safe_float_series(df[c])

        z_col = f"{c}__time_robust_z"
        r_col = f"{c}__time_rank_pct"

        df[z_col] = np.nan
        df[r_col] = np.nan

        for t, idx in df.groupby("time").groups.items():
            v = vals.loc[idx]
            med = v.median()
            mad = (v - med).abs().median()

            if not np.isfinite(mad) or mad < 1e-12:
                mad = v.std()

            if not np.isfinite(mad) or mad < 1e-12:
                mad = 1.0

            df.loc[idx, z_col] = (v - med) / (1.4826 * mad)
            df.loc[idx, r_col] = v.rank(method="average", pct=True)

    return df


def build_model_feature_sets(df):
    base_cols = numeric_feature_columns(df)

    # v7式：保留相对稳健的频谱/空间/时间形态特征，尽量少用绝对能量
    unstable_keywords = [
        "raw_best_energy",
        "mean_direction_energy",
        "direction_energy_std",
        "energy_5cm",
        "energy_20_70",
        "energy_20_40",
        "energy_40_70",
        "energy_20_30k",
        "energy_30_40k",
        "energy_40_50k",
        "energy_50_60k",
        "energy_60_70k",
        "time_energy_mean",
        "time_energy_std",
        "time_rms",
        "dw_total_near_band_energy",
    ]

    robust_base = []
    for c in base_cols:
        if any(k in c for k in unstable_keywords):
            continue
        robust_base.append(c)

    # directed wideband
    dw_base = [c for c in base_cols if c.startswith("dw_") or c.startswith("physics_")]

    # 添加 time z/rank
    df2 = add_time_relative_features(df, robust_base)

    robust_all = []
    for c in robust_base:
        robust_all.append(c)
        z = f"{c}__time_robust_z"
        r = f"{c}__time_rank_pct"
        if z in df2.columns:
            robust_all.append(z)
        if r in df2.columns:
            robust_all.append(r)

    dw_all = []
    for c in dw_base:
        if c in df2.columns:
            dw_all.append(c)
        z = f"{c}__time_robust_z"
        r = f"{c}__time_rank_pct"
        if z in df2.columns:
            dw_all.append(z)
        if r in df2.columns:
            dw_all.append(r)

    # v7 baseline：不要 dw / physics
    v7_like = [c for c in robust_all if not (c.startswith("dw_") or c.startswith("physics_"))]

    # v8：v7 + dw
    v8_all = []
    for c in v7_like + dw_all:
        if c not in v8_all:
            v8_all.append(c)

    return df2, {
        "A_v7_like_baseline": v7_like,
        "B_directed_wideband_only": dw_all,
        "C_v8_v7_plus_directed_wideband": v8_all,
    }


# ============================================================
# 8. 训练验证
# ============================================================

def build_matrix(train_df, test_df, cols):
    X_train = pd.DataFrame(index=train_df.index)
    X_test = pd.DataFrame(index=test_df.index)

    medians = {}

    for c in cols:
        tr = safe_float_series(train_df[c]) if c in train_df.columns else pd.Series(np.nan, index=train_df.index)
        te = safe_float_series(test_df[c]) if c in test_df.columns else pd.Series(np.nan, index=test_df.index)

        med = tr.median()
        if not np.isfinite(med):
            med = 0.0

        medians[c] = float(med)

        X_train[c] = tr.fillna(med).astype(float)
        X_test[c] = te.fillna(med).astype(float)

    return X_train, X_test, medians


def build_classifier():
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=700,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=1,
    )


def get_group_oof_prob(X, y, groups):
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups).astype(str)

    prob = np.zeros(len(y), dtype=float)
    filled = np.zeros(len(y), dtype=bool)

    unique_groups = sorted(pd.unique(groups).tolist())

    if len(unique_groups) >= 2:
        for g in unique_groups:
            val_mask = groups == g
            tr_mask = ~val_mask

            if len(np.unique(y[tr_mask])) < 2:
                continue

            clf = build_classifier()
            clf.fit(X.loc[tr_mask], y[tr_mask])
            prob[val_mask] = clf.predict_proba(X.loc[val_mask])[:, 1]
            filled[val_mask] = True

    if not np.all(filled):
        from sklearn.model_selection import StratifiedKFold

        min_count = min(np.sum(y == 0), np.sum(y == 1))
        n_splits = max(2, min(5, int(min_count)))

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

        for tr_idx, val_idx in cv.split(X, y):
            clf = build_classifier()
            clf.fit(X.iloc[tr_idx], y[tr_idx])
            prob[val_idx] = clf.predict_proba(X.iloc[val_idx])[:, 1]
            filled[val_idx] = True

    return prob


def find_best_threshold(y, prob):
    best_t = 0.5
    best_score = -1
    rows = []

    for t in THRESHOLD_GRID:
        pred = (prob >= t).astype(int)
        m = metrics(y, pred)
        score = m["balanced_accuracy"]

        rows.append({
            "threshold": float(t),
            "score": float(score),
            **m,
        })

        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t, best_score, pd.DataFrame(rows)


def add_rank_pred(pred_df, score_col, prefix):
    pred_df = pred_df.copy()

    rank_col = f"{prefix}_rank_pct"
    pred_col = f"{prefix}_rank_pred"

    pred_df[rank_col] = 0.5

    for g, idx in pred_df.groupby("test_group").groups.items():
        s = safe_float_series(pred_df.loc[idx, score_col])
        pred_df.loc[idx, rank_col] = s.rank(method="average", pct=True)

    cutoff = 1.0 - RANK_TRUE_FRACTION

    pred_df[pred_col] = np.where(
        pred_df[rank_col] > cutoff,
        "TRUE_LEAK",
        "FALSE_LEAK",
    )

    return pred_df


def validate_experiment(df, feature_cols, experiment_name):
    exp_dir = os.path.join(OUTPUT_DIR, experiment_name)
    ensure_dir(exp_dir)

    groups = sorted(df["time"].astype(str).unique().tolist())

    summary_rows = []
    pred_rows = []

    print("\n" + "-" * 110)
    print("实验:", experiment_name)
    print("特征数量:", len(feature_cols))
    print("-" * 110)

    for test_group in groups:
        test_mask = df["time"].astype(str).values == str(test_group)

        train_df = df.loc[~test_mask].reset_index(drop=True)
        test_df = df.loc[test_mask].reset_index(drop=True)

        y_train = label_to_y(train_df["label"].values)
        y_test = label_to_y(test_df["label"].values)

        X_train, X_test, medians = build_matrix(train_df, test_df, feature_cols)

        oof_prob = get_group_oof_prob(X_train, y_train, train_df["time"].astype(str).values)
        best_t, best_score, curve = find_best_threshold(y_train, oof_prob)

        curve_path = os.path.join(exp_dir, f"{experiment_name}_threshold_curve_without_{safe_name(test_group)}.csv")
        curve.to_csv(curve_path, index=False, encoding="utf-8-sig")

        clf = build_classifier()
        clf.fit(X_train, y_train)

        prob = clf.predict_proba(X_test)[:, 1]

        pred_default = (prob >= 0.5).astype(int)
        pred_model = (prob >= best_t).astype(int)

        m_default = metrics(y_test, pred_default)
        m_model = metrics(y_test, pred_model)
        auc = safe_auc(y_test, prob)

        p = pd.DataFrame({
            "experiment": experiment_name,
            "test_group": test_group,
            "time": test_df["time"].astype(str).values,
            "center": test_df["center"].values,
            "center_norm": test_df["center_norm"].values,
            "true_label": test_df["label"].astype(str).values,
            "prob_TRUE_LEAK": prob,
            "best_threshold": best_t,
            "default_pred": [y_to_label(x) for x in pred_default],
            "model_pred": [y_to_label(x) for x in pred_model],
        })

        # 加方向性宽频物理分数，方便 rank
        if "physics_directed_leak_score" in test_df.columns:
            p["physics_directed_leak_score"] = safe_float_series(test_df["physics_directed_leak_score"]).values
        else:
            p["physics_directed_leak_score"] = prob

        if "dw_directed_wideband_score" in test_df.columns:
            p["dw_directed_wideband_score"] = safe_float_series(test_df["dw_directed_wideband_score"]).values

        if "dw_diffuse_wideband_score" in test_df.columns:
            p["dw_diffuse_wideband_score"] = safe_float_series(test_df["dw_diffuse_wideband_score"]).values

        p = add_rank_pred(p, "prob_TRUE_LEAK", "prob")
        p = add_rank_pred(p, "physics_directed_leak_score", "physics")

        y_prob_rank = label_to_y(p["prob_rank_pred"].values)
        y_phy_rank = label_to_y(p["physics_rank_pred"].values)

        m_prob_rank = metrics(y_test, y_prob_rank)
        m_phy_rank = metrics(y_test, y_phy_rank)

        # v8_final：如果实验含 directed wideband，就采用 physics rank 作为工程辅助判据；
        # 否则采用 model_pred。
        if "directed_wideband" in experiment_name or "v8" in experiment_name:
            p["v8_final_pred"] = p["physics_rank_pred"]
            final_y = y_phy_rank
        else:
            p["v8_final_pred"] = p["model_pred"]
            final_y = pred_model

        m_final = metrics(y_test, final_y)

        p["model_correct"] = (p["model_pred"] == p["true_label"]).astype(int)
        p["prob_rank_correct"] = (p["prob_rank_pred"] == p["true_label"]).astype(int)
        p["physics_rank_correct"] = (p["physics_rank_pred"] == p["true_label"]).astype(int)
        p["v8_final_correct"] = (p["v8_final_pred"] == p["true_label"]).astype(int)

        pred_rows.extend(p.to_dict(orient="records"))

        summary_rows.append({
            "experiment": experiment_name,
            "test_group": test_group,
            "n_test": len(test_df),
            "n_true": int(np.sum(y_test == 1)),
            "n_false": int(np.sum(y_test == 0)),
            "n_features": len(feature_cols),
            "auc": auc,
            "best_threshold": best_t,
            "train_oof_score": best_score,

            "default_acc": m_default["accuracy"],
            "model_acc": m_model["accuracy"],
            "prob_rank_acc": m_prob_rank["accuracy"],
            "physics_rank_acc": m_phy_rank["accuracy"],
            "v8_final_acc": m_final["accuracy"],

            "model_recall_TRUE": m_model["recall_TRUE"],
            "model_recall_FALSE": m_model["recall_FALSE"],
            "physics_recall_TRUE": m_phy_rank["recall_TRUE"],
            "physics_recall_FALSE": m_phy_rank["recall_FALSE"],
        })

        print(
            f"{test_group}: "
            f"AUC={auc:.3f}, "
            f"default={m_default['accuracy']:.3f}, "
            f"model={m_model['accuracy']:.3f}, "
            f"prob_rank={m_prob_rank['accuracy']:.3f}, "
            f"physics_rank={m_phy_rank['accuracy']:.3f}, "
            f"final={m_final['accuracy']:.3f}"
        )

    summary = pd.DataFrame(summary_rows)
    preds = pd.DataFrame(pred_rows)

    summary_path = os.path.join(exp_dir, f"{experiment_name}_group_summary.csv")
    pred_path = os.path.join(exp_dir, f"{experiment_name}_predictions.csv")

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    preds.to_csv(pred_path, index=False, encoding="utf-8-sig")

    return {
        "experiment": experiment_name,
        "summary": summary,
        "preds": preds,
        "summary_path": summary_path,
        "pred_path": pred_path,
    }


# ============================================================
# 9. 144226 专项诊断
# ============================================================

def compare_features_144226(df, cols):
    sub = df[df["time"].astype(str) == TARGET_TIME].copy()

    rows = []

    for c in cols:
        if c not in sub.columns:
            continue

        true_vals = safe_float_series(sub.loc[sub["label"] == "TRUE_LEAK", c]).dropna().values
        false_vals = safe_float_series(sub.loc[sub["label"] == "FALSE_LEAK", c]).dropna().values

        if len(true_vals) < 2 or len(false_vals) < 2:
            continue

        y = np.concatenate([np.ones(len(true_vals)), np.zeros(len(false_vals))])
        score = np.concatenate([true_vals, false_vals])
        auc = safe_auc(y, score)
        auc_free = max(auc, 1 - auc) if np.isfinite(auc) else np.nan
        d = cohen_d(true_vals, false_vals)

        rows.append({
            "feature": c,
            "true_mean": float(np.mean(true_vals)),
            "false_mean": float(np.mean(false_vals)),
            "diff_TRUE_minus_FALSE": float(np.mean(true_vals) - np.mean(false_vals)),
            "cohen_d": d,
            "abs_cohen_d": abs(d) if np.isfinite(d) else np.nan,
            "auc_signed_TRUE_larger": auc,
            "auc_direction_free": auc_free,
        })

    out = pd.DataFrame(rows)

    if len(out):
        out = out.sort_values(["auc_direction_free", "abs_cohen_d"], ascending=[False, False])

    path = os.path.join(OUTPUT_DIR, "v8_144226_directed_wideband_feature_compare.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")

    return out, path


def pair_check_144226(preds):
    sub = preds[preds["test_group"].astype(str) == TARGET_TIME].copy()

    rows = []

    for center, g in sub.groupby("center_norm"):
        tr = g[g["true_label"] == "TRUE_LEAK"]
        fa = g[g["true_label"] == "FALSE_LEAK"]

        if len(tr) == 0 or len(fa) == 0:
            continue

        tr = tr.iloc[0]
        fa = fa.iloc[0]

        row = {
            "center_norm": center,

            "true_prob": tr["prob_TRUE_LEAK"],
            "false_prob": fa["prob_TRUE_LEAK"],
            "prob_diff_TRUE_minus_FALSE": tr["prob_TRUE_LEAK"] - fa["prob_TRUE_LEAK"],
            "prob_order_correct": int(tr["prob_TRUE_LEAK"] > fa["prob_TRUE_LEAK"]),

            "true_physics_score": tr.get("physics_directed_leak_score", np.nan),
            "false_physics_score": fa.get("physics_directed_leak_score", np.nan),
            "physics_score_diff_TRUE_minus_FALSE": tr.get("physics_directed_leak_score", np.nan) - fa.get("physics_directed_leak_score", np.nan),
            "physics_order_correct": int(tr.get("physics_directed_leak_score", np.nan) > fa.get("physics_directed_leak_score", np.nan)),

            "true_model_pred": tr["model_pred"],
            "false_model_pred": fa["model_pred"],
            "true_physics_rank_pred": tr["physics_rank_pred"],
            "false_physics_rank_pred": fa["physics_rank_pred"],
            "true_v8_final_pred": tr["v8_final_pred"],
            "false_v8_final_pred": fa["v8_final_pred"],
        }

        rows.append(row)

    out = pd.DataFrame(rows)

    if len(out):
        out = out.sort_values("center_norm")

    path = os.path.join(OUTPUT_DIR, "v8_144226_pair_check.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")

    return out, path


# ============================================================
# 10. 图和报告
# ============================================================

def plot_results(results):
    fig_dir = os.path.join(OUTPUT_DIR, "figures")
    ensure_dir(fig_dir)

    all_rows = []
    for r in results:
        all_rows.extend(r["summary"].to_dict(orient="records"))

    df = pd.DataFrame(all_rows)

    paths = []

    if len(df) == 0:
        return paths

    for metric in ["auc", "model_acc", "physics_rank_acc", "v8_final_acc"]:
        plt.figure(figsize=(12, 5))

        experiments = df["experiment"].unique().tolist()
        groups = sorted(df["test_group"].unique().tolist())

        x = np.arange(len(groups))
        width = 0.22

        for i, exp in enumerate(experiments):
            vals = []
            for g in groups:
                sub = df[(df["experiment"] == exp) & (df["test_group"] == g)]
                vals.append(float(sub[metric].iloc[0]) if len(sub) else np.nan)

            plt.bar(x + (i - (len(experiments) - 1) / 2) * width, vals, width, label=exp)

        plt.xticks(x, groups, rotation=45, ha="right")
        plt.ylim(0, 1.05)
        plt.ylabel(metric)
        plt.title(f"v8 standalone comparison: {metric}")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()

        path = os.path.join(fig_dir, f"v8_{metric}_comparison.png")
        plt.savefig(path, dpi=150)
        plt.close()

        paths.append(path)

    return paths


def make_report(df, feature_sets, results, compare_df, compare_path, pair_df, pair_path, plot_paths):
    lines = []

    lines.append("独立完整 v8：方向性宽频 + 稳健特征 分类报告")
    lines.append("=" * 120)
    lines.append(f"生成时间: {datetime.now()}")
    lines.append("")
    lines.append("说明:")
    lines.append("  本程序不依赖 v3/v4/v7/v8.1 的中间特征文件。")
    lines.append("  所有空间、频谱、时间、方向性宽频特征都从原始 WAV 中重新计算。")
    lines.append("")
    lines.append("数据路径:")
    for ds in DATASETS:
        lines.append(f"  [{ds['label']}]")
        lines.append(f"    center_root: {ds['center_root']}")
        lines.append(f"    offset_root: {ds['offset_root']}")
    lines.append("")

    lines.append("样本统计:")
    lines.append(str(df["label"].value_counts()))
    lines.append("")
    lines.append("time统计:")
    for t, g in df.groupby("time"):
        lines.append(f"  {t}: n={len(g)}, TRUE={(g['label']=='TRUE_LEAK').sum()}, FALSE={(g['label']=='FALSE_LEAK').sum()}")
    lines.append("")

    lines.append("特征集:")
    for k, v in feature_sets.items():
        lines.append(f"  {k}: {len(v)} features")
    lines.append("")

    lines.append("实验结果:")
    lines.append("-" * 120)

    overall_rows = []

    for r in results:
        exp = r["experiment"]
        s = r["summary"]

        lines.append("")
        lines.append(f"[{exp}]")
        lines.append(f"  summary: {r['summary_path']}")
        lines.append(f"  predictions: {r['pred_path']}")

        if len(s):
            lines.append(
                f"  平均AUC={s['auc'].mean():.4f}, "
                f"平均model_acc={s['model_acc'].mean():.4f}, "
                f"平均physics_rank_acc={s['physics_rank_acc'].mean():.4f}, "
                f"平均final_acc={s['v8_final_acc'].mean():.4f}"
            )

            target = s[s["test_group"] == TARGET_TIME]
            if len(target):
                tr = target.iloc[0]
                lines.append(
                    f"  {TARGET_TIME}: "
                    f"AUC={tr['auc']:.4f}, "
                    f"model_acc={tr['model_acc']:.4f}, "
                    f"physics_rank_acc={tr['physics_rank_acc']:.4f}, "
                    f"final_acc={tr['v8_final_acc']:.4f}"
                )

            lines.append("  各time:")
            for _, row in s.iterrows():
                lines.append(
                    f"    {row['test_group']}: "
                    f"AUC={row['auc']:.3f}, "
                    f"default={row['default_acc']:.3f}, "
                    f"model={row['model_acc']:.3f}, "
                    f"prob_rank={row['prob_rank_acc']:.3f}, "
                    f"physics_rank={row['physics_rank_acc']:.3f}, "
                    f"final={row['v8_final_acc']:.3f}"
                )

            item = {
                "experiment": exp,
                "mean_auc": s["auc"].mean(),
                "mean_model_acc": s["model_acc"].mean(),
                "mean_physics_rank_acc": s["physics_rank_acc"].mean(),
                "mean_final_acc": s["v8_final_acc"].mean(),
            }

            if len(target):
                tr = target.iloc[0]
                item.update({
                    "auc_144226": tr["auc"],
                    "model_acc_144226": tr["model_acc"],
                    "physics_rank_acc_144226": tr["physics_rank_acc"],
                    "final_acc_144226": tr["v8_final_acc"],
                })

            overall_rows.append(item)

    overall = pd.DataFrame(overall_rows)
    overall_path = os.path.join(OUTPUT_DIR, "v8_group_validation_summary.csv")
    overall.to_csv(overall_path, index=False, encoding="utf-8-sig")

    lines.append("")
    lines.append("总汇总:")
    lines.append(f"  {overall_path}")

    lines.append("")
    lines.append("144226 方向性宽频特征对比:")
    lines.append(f"  {compare_path}")

    if len(compare_df):
        lines.append("  区分力前15:")
        for _, row in compare_df.head(15).iterrows():
            lines.append(
                f"    {row['feature']}: "
                f"AUC={row['auc_direction_free']:.3f}, "
                f"d={row['cohen_d']:.3f}, "
                f"TRUE_mean={row['true_mean']:.6g}, "
                f"FALSE_mean={row['false_mean']:.6g}"
            )

    lines.append("")
    lines.append("144226 center配对检查:")
    lines.append(f"  {pair_path}")

    if len(pair_df):
        lines.append(f"  prob配对排序正确率: {pair_df['prob_order_correct'].mean():.4f}")
        lines.append(f"  physics配对排序正确率: {pair_df['physics_order_correct'].mean():.4f}")

        failed = pair_df[pair_df["physics_order_correct"] == 0]["center_norm"].astype(str).tolist()
        lines.append(f"  physics排序失败center: {' | '.join(failed) if failed else '无'}")

    lines.append("")
    lines.append("图像:")
    for p in plot_paths:
        lines.append(f"  {p}")

    report_path = os.path.join(OUTPUT_DIR, "v8_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path, overall_path


# ============================================================
# 11. 主程序
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)

    print("=" * 120)
    print("独立完整 v8：方向性宽频 + 稳健特征 分类程序")
    print("=" * 120)

    df = extract_all_features()

    print("\n特征提取完成")
    print("样本数:", len(df))
    print(df["label"].value_counts())
    print("time groups:", sorted(df["time"].astype(str).unique().tolist()))

    df2, feature_sets = build_model_feature_sets(df)

    feature_dataset_path = os.path.join(OUTPUT_DIR, "v8_feature_dataset_with_time_relative.csv")
    df2.to_csv(feature_dataset_path, index=False, encoding="utf-8-sig")

    with open(os.path.join(OUTPUT_DIR, "v8_feature_sets.json"), "w", encoding="utf-8") as f:
        json.dump(feature_sets, f, ensure_ascii=False, indent=2)

    print("\n特征集:")
    for k, v in feature_sets.items():
        print(f"  {k}: {len(v)}")

    results = []

    for exp, cols in feature_sets.items():
        results.append(validate_experiment(df2, cols, exp))

    all_preds = pd.concat([r["preds"] for r in results], ignore_index=True)
    pred_path = os.path.join(OUTPUT_DIR, "v8_predictions.csv")
    all_preds.to_csv(pred_path, index=False, encoding="utf-8-sig")

    # 144226 诊断：只看 directed/physics 特征
    dw_cols = feature_sets["B_directed_wideband_only"]
    compare_df, compare_path = compare_features_144226(df2, dw_cols)

    # 配对检查：优先看最终 C_v8 的预测
    c_preds = results[-1]["preds"]
    pair_df, pair_path = pair_check_144226(c_preds)

    plot_paths = plot_results(results)

    report_path, overall_path = make_report(
        df=df2,
        feature_sets=feature_sets,
        results=results,
        compare_df=compare_df,
        compare_path=compare_path,
        pair_df=pair_df,
        pair_path=pair_path,
        plot_paths=plot_paths,
    )

    print("\n" + "=" * 120)
    print("v8 完成")
    print("=" * 120)
    print("输出文件夹:", OUTPUT_DIR)
    print("报告:", report_path)
    print("完整特征表:", feature_dataset_path)
    print("预测明细:", pred_path)
    print("分组汇总:", overall_path)
    print("144226配对检查:", pair_path)
    print("144226方向性宽频特征对比:", compare_path)

    print("\n核心结果摘要:")
    for r in results:
        s = r["summary"]
        print(f"\n{r['experiment']}:")
        print(
            f"  平均AUC={s['auc'].mean():.3f}, "
            f"平均model_acc={s['model_acc'].mean():.3f}, "
            f"平均physics_rank_acc={s['physics_rank_acc'].mean():.3f}, "
            f"平均final_acc={s['v8_final_acc'].mean():.3f}"
        )

        target = s[s["test_group"] == TARGET_TIME]
        if len(target):
            tr = target.iloc[0]
            print(
                f"  144226: "
                f"AUC={tr['auc']:.3f}, "
                f"model_acc={tr['model_acc']:.3f}, "
                f"physics_rank_acc={tr['physics_rank_acc']:.3f}, "
                f"final_acc={tr['v8_final_acc']:.3f}"
            )

    if len(pair_df):
        print("\n144226 center配对检查:")
        print(f"  prob排序正确率: {pair_df['prob_order_correct'].mean():.3f}")
        print(f"  physics排序正确率: {pair_df['physics_order_correct'].mean():.3f}")
        failed = pair_df[pair_df["physics_order_correct"] == 0]["center_norm"].astype(str).tolist()
        print("  physics排序失败center:", " | ".join(failed) if failed else "无")

    print("\n请把“核心结果摘要”和“144226 center配对检查”发给我，我帮你判断是否解决了144226。")


if __name__ == "__main__":
    main()
