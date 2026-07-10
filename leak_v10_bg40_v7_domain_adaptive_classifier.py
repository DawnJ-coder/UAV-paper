# -*- coding: utf-8 -*-
r"""
leak_v10_bg40_v7_domain_adaptive_classifier.py

v10：v7稳健特征 + 40cm背景扣除/比值 + 跨工厂逐center验证
================================================================================

为什么写这个版本？
--------------------------------------------------------------------------------
原 v7 在 A 工厂有效，但换到 B 工厂后容易出现：
    1) single 模式全部判 TRUE；
    2) pairwise 模式强制一真一假，不符合现场实际；
    3) 不同工厂背景噪声、增益、反射、假泄漏来源不同，导致特征分布偏移。

这个 v10 不废掉 v7，而是在 v7 的基础上增加 40cm 背景参照：
    - 保留 v7 的频谱、方向、衰减、时间形态特征；
    - 用每个 center 的 40cm 偏离点作为局部背景；
    - 新增 40cm 背景扣除特征；
    - 新增 40cm 背景比值特征；
    - 保留/增强 time 内部 robust_z 和 rank_pct；
    - 每个 center 独立输出 TRUE_LEAK / FALSE_LEAK / SUSPECT。

核心思想：
--------------------------------------------------------------------------------
原 v7 问的是：
    这个点像不像 A 工厂里的 TRUE_LEAK？

v10 问的是：
    这个点相对于它自己 40cm 背景，有没有泄漏型局部增强？

输入数据要求：
--------------------------------------------------------------------------------
每个 center 理想情况下应有：
    8 directions × 8 distances = 64 个 offset wav

支持的 offset 文件名示例：
    HM20260626_142938.ld_00_14d5_down_beamform_result.wav
    HM20260626_142938.ld_00_14d10_up_left_beamform_result.wav

其中：
    14d10 表示 center=14, distance=10cm

运行方式：
--------------------------------------------------------------------------------
1) 修改下面配置区路径；
2) 运行：
       python leak_v10_bg40_v7_domain_adaptive_classifier.py

输出：
--------------------------------------------------------------------------------
默认输出到：
    C:\Users\jiangxinru6\Desktop\wurenji\leak_v10_bg40_v7_domain_adaptive_results

主要文件：
    v10_train_feature_dataset.csv
    v10_model_feature_dataset.csv
    v10_bg40_v7_model.pkl
    v10_bg40_v7_model_config.json
    v10_test_per_center_predictions.csv
    v10_feature_importance.csv
    v10_data_quality_report.csv
    v10_report.txt

注意：
--------------------------------------------------------------------------------
1) 这个版本不再使用 144226 pairwise 规则；
2) 不强制同一组里面一真一假；
3) 如果模型不确定，会输出 SUSPECT；
4) 如果你必须二分类，可把 FORCE_BINARY_OUTPUT=True。
"""

import os
import re
import json
import math
import warnings
from datetime import datetime
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from scipy.io import wavfile
from scipy import signal
from scipy.stats import kurtosis


# =============================================================================
# 1. 配置区：主要改这里
# =============================================================================

BASE_DIR = r"C:\Users\jiangxinru6\Desktop\wurenji"
OUTPUT_DIR = os.path.join(BASE_DIR, "leak_v10_bg40_v7_domain_adaptive_results")

# -----------------------------------------------------------------------------
# 运行模式
# -----------------------------------------------------------------------------
# train_and_predict : 重新用 A 工厂训练，并预测 B 工厂
# train_only        : 只训练并保存模型
# predict_only      : 读取已保存模型，只预测 B 工厂
RUN_MODE = "train_and_predict"

# -----------------------------------------------------------------------------
# A 工厂训练数据
# -----------------------------------------------------------------------------
# 你现在 A 工厂数据结构是：真泄漏有 center/offset，假泄漏也有 center/offset。
# 如果你的路径不同，直接改这里。
TRAIN_TRUE_OFFSET_ROOTS = [
    r"D:\gas\beamform_results_offset_multiple",
]
TRAIN_TRUE_CENTER_ROOTS = [
    r"D:\gas\beamform_results",
]

TRAIN_FALSE_OFFSET_ROOTS = [
    r"D:\gas\beamform_results_cs_offset_multiple",
]
TRAIN_FALSE_CENTER_ROOTS = [
    r"D:\gas\beamform_results_cs",
]

# -----------------------------------------------------------------------------
# B 工厂待测数据
# -----------------------------------------------------------------------------
# B 工厂待测 offset 根目录。
# 可以是：
#   1) 包含多个 .ld 子文件夹的根目录；
#   2) 单个 .ld 文件夹；
#   3) 直接包含 wav 的文件夹。
TEST_OFFSET_ROOT = r"D:\gas\B_factory_offset_root"
TEST_CENTER_ROOT = r""
TEST_DATASET_NAME = "B_FACTORY_UNKNOWN"

# 如果 B 工厂有标签表，可填路径用于评估；没有就留空。
# CSV 至少包含：time, center, label
# label 取值：TRUE_LEAK / FALSE_LEAK
TEST_LABEL_CSV = r""

# -----------------------------------------------------------------------------
# 模型保存/读取路径
# -----------------------------------------------------------------------------
MODEL_PATH = os.path.join(OUTPUT_DIR, "v10_bg40_v7_model.pkl")
CONFIG_PATH = os.path.join(OUTPUT_DIR, "v10_bg40_v7_model_config.json")

# -----------------------------------------------------------------------------
# 判定参数
# -----------------------------------------------------------------------------
# 默认不强制二分类。概率落在中间会输出 SUSPECT。
FORCE_BINARY_OUTPUT = False

# 如果不确定区间太大/太小，可调这两个 margin。
# true_threshold = recommended_threshold + SUSPECT_MARGIN_TRUE
# false_threshold = recommended_threshold - SUSPECT_MARGIN_FALSE
SUSPECT_MARGIN_TRUE = 0.10
SUSPECT_MARGIN_FALSE = 0.10

# 如果 FORCE_BINARY_OUTPUT=True，则只用 recommended_threshold 进行二分类。
DEFAULT_THRESHOLD = 0.50

# 训练时按 time OOF 搜索阈值
THRESHOLD_GRID = np.linspace(0.05, 0.95, 91)
THRESHOLD_METRIC = "balanced_accuracy"

# -----------------------------------------------------------------------------
# 声学参数
# -----------------------------------------------------------------------------
FREQ_LOW = 20000
FREQ_HIGH = 70000

SUBBANDS = [
    (20000, 30000),
    (30000, 40000),
    (40000, 50000),
    (50000, 60000),
    (60000, 70000),
]

SUBBAND_NAMES = [f"{lo//1000}_{hi//1000}k" for lo, hi in SUBBANDS]
SUBBAND_CENTERS = np.array([(lo + hi) / 2 for lo, hi in SUBBANDS], dtype=float)

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
NEAR_DISTANCES_CM = [5, 10, 15, 20]
BACKGROUND_DISTANCE_CM = 40

WAV_EXTS = [".wav", ".WAV"]
NFFT = 4096
WELCH_NPERSEG = 4096
WELCH_NOVERLAP = 2048
RANDOM_STATE = 42

# -----------------------------------------------------------------------------
# 数据质量控制
# -----------------------------------------------------------------------------
EXPECTED_OFFSET_COMBOS_PER_CENTER = 64
MIN_VALID_OFFSET_COMBOS = 40
MIN_VALID_BG40_DIRECTIONS = 4


# =============================================================================
# 2. 基础工具
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def safe_name(s):
    return str(s).replace("\\", "_").replace("/", "_").replace(":", "_").replace(".", "_")


def normalize_center_id(x):
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    nums = "".join(ch for ch in s if ch.isdigit())
    if nums == "":
        return s
    return nums.zfill(2)


def label_to_binary(labels):
    return np.array([1 if str(x) == "TRUE_LEAK" else 0 for x in labels], dtype=int)


def binary_to_label(v):
    return "TRUE_LEAK" if int(v) == 1 else "FALSE_LEAK"


def safe_float_series(s):
    x = pd.to_numeric(s, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    return x


def safe_ratio(a, b, eps=1e-20):
    return float(a / (b + eps))


def safe_log_ratio(a, b, eps=1e-20):
    return float(np.log((a + eps) / (b + eps)))


def entropy_norm(x):
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 0)
    total = float(np.sum(x))
    if total <= 1e-20:
        return 0.0
    p = x / total
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    return float(-np.sum(p * np.log(p + 1e-20)) / np.log(len(p)))


def gini_coefficient(x):
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 0)
    if len(x) == 0 or np.sum(x) <= 1e-20:
        return 0.0
    x = np.sort(x)
    n = len(x)
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / (cum[-1] + 1e-20)) / n)


def spectral_flatness_from_values(x):
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 1e-20)
    return float(np.exp(np.mean(np.log(x))) / (np.mean(x) + 1e-20))


def metrics_from_pred(y_true, y_pred):
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
    balanced_acc = 0.5 * (recall_true + recall_false)
    precision_true = tp / (tp + fp + 1e-12)
    f1_true = 2 * precision_true * recall_true / (precision_true + recall_true + 1e-12)
    youden = recall_true + recall_false - 1.0

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_acc),
        "recall_TRUE_LEAK": float(recall_true),
        "recall_FALSE_LEAK": float(recall_false),
        "precision_TRUE_LEAK": float(precision_true),
        "f1_TRUE_LEAK": float(f1_true),
        "youden": float(youden),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def safe_auc(y_true, prob):
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, prob))
    except Exception:
        return np.nan


def threshold_predict(prob, threshold):
    return (np.asarray(prob, dtype=float) >= float(threshold)).astype(int)


def find_best_threshold(y_true, prob, metric="balanced_accuracy", grid=None):
    if grid is None:
        grid = THRESHOLD_GRID

    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)

    best_t = DEFAULT_THRESHOLD
    best_score = -1e18
    rows = []

    for t in grid:
        pred = threshold_predict(prob, t)
        m = metrics_from_pred(y_true, pred)
        if metric == "f1":
            score = m["f1_TRUE_LEAK"]
        elif metric == "youden":
            score = m["youden"]
        else:
            score = m["balanced_accuracy"]

        rows.append({"threshold": float(t), "score": float(score), **m})

        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t, float(best_score), pd.DataFrame(rows)


# =============================================================================
# 3. 文件解析
# =============================================================================

def list_wav_files(root):
    out = []
    if not root or not os.path.exists(root):
        return out
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if any(name.endswith(ext) for ext in WAV_EXTS):
                out.append(os.path.join(dirpath, name))
    return sorted(out)


def resolve_time_dirs(root):
    """
    root 可以是：
      1) 包含多个 .ld 子目录的根目录；
      2) 单个 .ld 目录；
      3) 直接包含 wav 的目录。
    """
    if not root or not os.path.exists(root):
        raise FileNotFoundError(f"输入文件夹不存在: {root}")

    root = os.path.abspath(root)
    base = os.path.basename(root)

    direct_wavs = []
    if os.path.isdir(root):
        direct_wavs = [f for f in os.listdir(root) if any(f.endswith(ext) for ext in WAV_EXTS)]

    if base.endswith(".ld") or len(direct_wavs) > 0:
        return [(base, root)]

    items = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        wavs = list_wav_files(p)
        if len(wavs) == 0:
            continue
        items.append((name, p))

    if not items:
        wavs = list_wav_files(root)
        if wavs:
            items.append((base, root))

    if not items:
        raise RuntimeError(f"没有在输入文件夹中找到 wav 文件: {root}")

    return items


def parse_offset_filename_candidates(path):
    """
    支持格式：
        HM20260626_142938.ld_00_14d5_down_beamform_result.wav

    候选解析：
        A: center = .ld 后第一个数字 00
        B: center = d 前面的数字 14
    你的数据中 B 通常才是正确的。
    程序会自动选择每个 center 更接近 64 个 offset 的解析方案。
    """
    base = os.path.basename(str(path)).lower().replace("-", "_").replace(" ", "_")
    direction_pattern = r"(up_left|up_right|down_left|down_right|up|down|left|right)"

    m = re.search(
        rf"\.ld_(\d{{1,3}})_(\d{{1,3}})d(\d{{1,3}})_({direction_pattern})_beamform",
        base,
        flags=re.IGNORECASE,
    )

    candidates = []
    if m:
        first_num = m.group(1).zfill(2)
        center_before_d = m.group(2).zfill(2)
        dist = int(m.group(3))
        direction = m.group(4).lower()
        if 0 < dist <= 200:
            candidates.append({
                "schema": "A_first_number_as_center",
                "center": first_num,
                "direction": direction,
                "distance": dist,
            })
            candidates.append({
                "schema": "B_number_before_d_as_center",
                "center": center_before_d,
                "direction": direction,
                "distance": dist,
            })
        return candidates

    # 兜底：center d distance direction
    direction = None
    for d in ["up_left", "up_right", "down_left", "down_right", "up", "down", "left", "right"]:
        if re.search(rf"(^|[_\-\\/]){d}($|[_\-\\/\.])", base):
            direction = d
            break

    md = re.search(r"(\d{1,3})d(\d{1,3})", base)
    if md and direction is not None:
        center = md.group(1).zfill(2)
        dist = int(md.group(2))
        candidates.append({
            "schema": "fallback_number_before_d_as_center",
            "center": center,
            "direction": direction,
            "distance": dist,
        })

    return candidates


def discover_offset_files_from_time_dir(time_dir, verbose=True):
    files = list_wav_files(time_dir)
    schema_maps = {}

    for f in files:
        for c in parse_offset_filename_candidates(f):
            schema = c["schema"]
            key = (c["center"], c["direction"], int(c["distance"]))
            schema_maps.setdefault(schema, {})
            schema_maps[schema].setdefault(key, [])
            schema_maps[schema][key].append(f)

    if not schema_maps:
        raise RuntimeError(f"没有识别到 offset wav: {time_dir}")

    rows = []
    for schema, mp in schema_maps.items():
        centers = sorted(set(k[0] for k in mp.keys()))
        counts = []
        for cc in centers:
            counts.append(sum(1 for k in mp.keys() if k[0] == cc))
        avg_per_center = float(np.mean(counts)) if counts else 0.0
        total_combos = len(mp)
        n_centers = len(centers)
        score = n_centers * 1000 + total_combos - abs(avg_per_center - EXPECTED_OFFSET_COMBOS_PER_CENTER) * 10
        rows.append({
            "schema": schema,
            "n_centers": n_centers,
            "total_combos": total_combos,
            "avg_per_center": avg_per_center,
            "score": score,
        })

    rows = sorted(rows, key=lambda x: x["score"], reverse=True)
    best_schema = rows[0]["schema"]

    if verbose:
        print("  offset解析候选:")
        for r in rows:
            print(
                f"    {r['schema']}: centers={r['n_centers']}, "
                f"combos={r['total_combos']}, avg/center={r['avg_per_center']:.1f}"
            )
        print("  采用offset解析方案:", best_schema)

    return schema_maps[best_schema], rows


def parse_center_file_candidates(path):
    base = os.path.basename(str(path)).lower().replace("-", "_").replace(" ", "_")
    candidates = []

    m = re.search(r"\.ld_(\d{1,3})(?=_beamform|_center|_result|\.|$)", base)
    if m:
        candidates.append(("after_ld", m.group(1).zfill(2)))

    m = re.search(r"(?:center|centre|c)_(\d{1,3})(?=_|\.|$)", base)
    if m:
        candidates.append(("center_token", m.group(1).zfill(2)))

    m = re.match(r"^(\d{1,3})(?=_|\.|$)", base)
    if m:
        candidates.append(("leading_number", m.group(1).zfill(2)))

    tokens = re.findall(r"(?<![a-zA-Z])(\d{1,3})(?![a-zA-Z])", base)
    if tokens:
        candidates.append(("last_number_token", tokens[-1].zfill(2)))

    out = []
    seen = set()
    for schema, center in candidates:
        key = (schema, center)
        if key not in seen:
            out.append((schema, center))
            seen.add(key)
    return out


def discover_center_files_from_time_dir(time_dir):
    if not time_dir or not os.path.exists(time_dir):
        return {}

    files = list_wav_files(time_dir)
    schema_maps = {}
    for f in files:
        for schema, center in parse_center_file_candidates(f):
            schema_maps.setdefault(schema, {})
            schema_maps[schema].setdefault(center, f)

    if not schema_maps:
        return {}

    rows = []
    for schema, mp in schema_maps.items():
        rows.append({"schema": schema, "n_centers": len(mp), "score": len(mp)})
    rows = sorted(rows, key=lambda x: x["score"], reverse=True)
    best_schema = rows[0]["schema"]
    return schema_maps[best_schema]


def match_center_time_dir(center_root, time_name):
    if not center_root or not os.path.exists(center_root):
        return ""
    base = os.path.basename(os.path.abspath(center_root))
    if base == time_name:
        return center_root
    p = os.path.join(center_root, time_name)
    if os.path.exists(p):
        return p
    return ""


# =============================================================================
# 4. WAV 分析
# =============================================================================

def read_wav_float(path):
    fs, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.astype(np.float64).mean(axis=1)
    else:
        x = x.astype(np.float64)

    x = x - np.mean(x)

    # 整型wav转浮点归一化；如果本来已经是小浮点，不强行改。
    max_abs = float(np.max(np.abs(x)) + 1e-12)
    if max_abs > 10:
        x = x / max_abs

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


def band_energy(f, pxx, lo, hi):
    if len(f) == 0:
        return 0.0
    hi2 = min(hi, float(np.max(f)))
    lo2 = min(lo, hi2)
    mask = (f >= lo2) & (f < hi2)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(pxx[mask], f[mask]))


def spectral_features_from_curve(freqs, values):
    freqs = np.asarray(freqs, dtype=float)
    values = np.asarray(values, dtype=float)
    values = np.maximum(values, 0)

    total = float(np.sum(values))
    if len(freqs) == 0 or total <= 1e-20:
        return {
            "spec_centroid_hz": 0.0,
            "spec_bandwidth_hz": 0.0,
            "spec_entropy": 0.0,
            "spec_flatness": 0.0,
            "spec_peak_freq_hz": 0.0,
            "spec_rolloff_85_hz": 0.0,
            "spec_slope": 0.0,
            "spec_peakiness": 0.0,
        }

    centroid = float(np.sum(freqs * values) / (total + 1e-20))
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * values) / (total + 1e-20)))
    entropy = entropy_norm(values)
    flatness = spectral_flatness_from_values(values)
    peak_idx = int(np.argmax(values))
    peak_freq = float(freqs[peak_idx])
    peakiness = float(np.max(values) / (np.mean(values) + 1e-20))

    cum = np.cumsum(values)
    idx = int(np.searchsorted(cum, 0.85 * cum[-1]))
    idx = min(idx, len(freqs) - 1)
    rolloff = float(freqs[idx])

    try:
        y = np.log10(values + 1e-20)
        xx = (freqs - freqs.mean()) / (freqs.std() + 1e-12)
        slope = float(np.polyfit(xx, y, 1)[0])
    except Exception:
        slope = 0.0

    return {
        "spec_centroid_hz": centroid,
        "spec_bandwidth_hz": bandwidth,
        "spec_entropy": entropy,
        "spec_flatness": flatness,
        "spec_peak_freq_hz": peak_freq,
        "spec_rolloff_85_hz": rolloff,
        "spec_slope": slope,
        "spec_peakiness": peakiness,
    }


def analyze_wav_spectrum(path):
    fs, x = read_wav_float(path)
    f, pxx = welch_psd(x, fs)
    if len(f) == 0:
        return None

    sub = []
    for lo, hi in SUBBANDS:
        sub.append(band_energy(f, pxx, lo, hi))
    sub = np.asarray(sub, dtype=float)

    total = float(np.sum(sub))

    mask = (f >= FREQ_LOW) & (f <= min(FREQ_HIGH, fs / 2 * 0.95))
    spec = spectral_features_from_curve(f[mask], pxx[mask]) if np.any(mask) else spectral_features_from_curve([], [])

    return {
        "fs": fs,
        "band_energy_20_70": total,
        "subband_energy": sub,
        **spec,
    }


def bandpass_signal(x, fs, lo=FREQ_LOW, hi=FREQ_HIGH):
    nyq = fs / 2
    hi2 = min(hi, nyq * 0.95)
    lo2 = min(lo, hi2 * 0.8)
    if hi2 <= lo2:
        return x
    try:
        sos = signal.butter(4, [lo2 / nyq, hi2 / nyq], btype="bandpass", output="sos")
        if len(x) > 100:
            return signal.sosfiltfilt(sos, x)
        return signal.sosfilt(sos, x)
    except Exception:
        return x


def time_features_from_wav(path):
    try:
        fs, x = read_wav_float(path)
        xb = bandpass_signal(x, fs)
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
        k = float(kurtosis(e, fisher=False, bias=False)) if len(e) >= 4 else 0.0
        rms = float(np.sqrt(np.mean(xb ** 2)))

        return {
            "time_energy_mean": mean,
            "time_energy_std": std,
            "time_energy_cv": cv,
            "time_energy_max_mean_ratio": max_mean,
            "time_energy_kurtosis": k,
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


def fit_decay(distances, energies):
    d = np.asarray(distances, dtype=float)
    e = np.asarray(energies, dtype=float)
    mask = np.isfinite(d) & np.isfinite(e) & (d > 0) & (e > 0)
    d = d[mask]
    e = e[mask]
    if len(d) < 3:
        return 0.0, 0.0
    x = np.log(d)
    y = np.log(e + 1e-20)
    try:
        coef = np.polyfit(x, y, 1)
        n = float(-coef[0])
        yhat = np.polyval(coef, x)
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-20
        r2 = 1 - ss_res / ss_tot
        return max(0.0, n), float(np.clip(r2, 0, 1))
    except Exception:
        return 0.0, 0.0


# =============================================================================
# 5. 每个 center 的 v7 + bg40 特征提取
# =============================================================================

def summarize_direction_values(values, prefix, row):
    values = np.asarray(values, dtype=float)
    values = np.maximum(values, 0)
    total = float(np.sum(values))

    if len(values) == 0:
        return None

    sorted_idx = np.argsort(values)[::-1]
    best_i = int(sorted_idx[0])
    second_i = int(sorted_idx[1]) if len(sorted_idx) > 1 else best_i
    others = np.delete(values, best_i)

    row[f"{prefix}_best_direction"] = DIRECTIONS[best_i]
    row[f"{prefix}_top1_energy"] = float(values[best_i])
    row[f"{prefix}_mean_direction_energy"] = float(np.mean(values))
    row[f"{prefix}_direction_energy_std"] = float(np.std(values))
    row[f"{prefix}_direction_cv"] = float(np.std(values) / (np.mean(values) + 1e-20))
    row[f"{prefix}_direction_entropy"] = entropy_norm(values)
    row[f"{prefix}_direction_gini"] = gini_coefficient(values)
    row[f"{prefix}_direction_top1_ratio"] = safe_ratio(values[best_i], total)
    row[f"{prefix}_direction_top2_ratio"] = safe_ratio(values[best_i] + values[second_i], total)
    row[f"{prefix}_direction_contrast"] = safe_ratio(values[best_i], np.mean(others) if len(others) else 0.0)
    row[f"{prefix}_direction_active_count_25pct"] = int(np.sum(values >= np.max(values) * 0.25)) if np.max(values) > 0 else 0
    row[f"{prefix}_best_direction_combined_score"] = float(
        row[f"{prefix}_direction_top1_ratio"]
        * np.log1p(max(row[f"{prefix}_direction_contrast"], 0.0))
        * (1.0 + row[f"{prefix}_direction_cv"])
    )

    return best_i


def add_band_ratio_features(row, band_values, prefix):
    band_values = np.asarray(band_values, dtype=float)
    band_values = np.maximum(band_values, 0)
    total = float(np.sum(band_values))

    row[f"{prefix}_energy_20_70"] = total
    for j, name in enumerate(SUBBAND_NAMES):
        row[f"{prefix}_energy_{name}"] = float(band_values[j])
        row[f"{prefix}_ratio_{name}"] = safe_ratio(band_values[j], total)

    # 兼容 v7 常用命名：原始特征不用 prefix 时，直接生成 energy_20_70/ratio_60_70k 等
    if prefix == "raw":
        row["energy_20_70"] = total
        for j, name in enumerate(SUBBAND_NAMES):
            row[f"energy_{name}"] = float(band_values[j])
            row[f"ratio_{name}"] = safe_ratio(band_values[j], total)
        row["energy_20_40"] = float(band_values[0] + band_values[1])
        row["energy_40_70"] = float(band_values[2] + band_values[3] + band_values[4])
        row["high_freq_ratio"] = safe_ratio(row["energy_40_70"], total)

    row[f"{prefix}_energy_20_40"] = float(band_values[0] + band_values[1])
    row[f"{prefix}_energy_40_70"] = float(band_values[2] + band_values[3] + band_values[4])
    row[f"{prefix}_high_freq_ratio"] = safe_ratio(row[f"{prefix}_energy_40_70"], total)

    spec = spectral_features_from_curve(SUBBAND_CENTERS, band_values)
    for k, v in spec.items():
        row[f"{prefix}_{k}"] = v

    return row


def compute_center_features(time_name, center, offset_files, center_file="", dataset_name="", label=""):
    row = {
        "dataset": dataset_name,
        "time": time_name,
        "center": center,
        "center_norm": normalize_center_id(center),
        "label": label,
        "center_file": center_file if center_file else "",
    }

    # E_total[direction][distance]，E_band[direction][distance]
    E_total = {d: {} for d in DIRECTIONS}
    E_band = {d: {} for d in DIRECTIONS}
    E_spec = {d: {} for d in DIRECTIONS}
    representative_files = {d: {} for d in DIRECTIONS}

    used_wav = 0

    for direction in DIRECTIONS:
        for dist in DISTANCES_CM:
            files = offset_files.get((center, direction, dist), [])
            if not files:
                continue

            totals = []
            bands = []
            first_spec = None

            for f in files:
                try:
                    res = analyze_wav_spectrum(f)
                    if res is None:
                        continue
                    totals.append(float(res["band_energy_20_70"]))
                    bands.append(np.asarray(res["subband_energy"], dtype=float))
                    if first_spec is None:
                        first_spec = res
                    used_wav += 1
                except Exception:
                    continue

            if totals:
                E_total[direction][dist] = float(np.mean(totals))
                E_band[direction][dist] = np.mean(np.asarray(bands), axis=0)
                E_spec[direction][dist] = first_spec
                representative_files[direction][dist] = files[0]

    row["offset_wav_count_used"] = int(used_wav)
    row["offset_combo_count"] = int(sum(1 for direction in DIRECTIONS for dist in DISTANCES_CM if dist in E_total[direction]))
    row["offset_combo_ratio"] = safe_ratio(row["offset_combo_count"], EXPECTED_OFFSET_COMBOS_PER_CENTER)

    bg40_dirs = [direction for direction in DIRECTIONS if BACKGROUND_DISTANCE_CM in E_total[direction]]
    row["bg40_direction_count"] = int(len(bg40_dirs))
    row["data_quality_score"] = float(min(1.0, row["offset_combo_count"] / EXPECTED_OFFSET_COMBOS_PER_CENTER))

    if row["offset_combo_count"] < MIN_VALID_OFFSET_COMBOS:
        row["data_quality_flag"] = "LOW_OFFSET_COUNT"
    elif row["bg40_direction_count"] < MIN_VALID_BG40_DIRECTIONS:
        row["data_quality_flag"] = "LOW_BG40_COUNT"
    else:
        row["data_quality_flag"] = "OK"

    if used_wav == 0:
        return row

    # -------------------------------------------------------------------------
    # A. 原 v7 风格特征：基于原始 E_total / E_band
    # -------------------------------------------------------------------------
    dir_near = []
    for direction in DIRECTIONS:
        vals = [E_total[direction][dist] for dist in NEAR_DISTANCES_CM if dist in E_total[direction]]
        if not vals:
            vals = [E_total[direction][dist] for dist in DISTANCES_CM if dist in E_total[direction]]
        dir_near.append(float(np.sum(vals)) if vals else 0.0)

    best_raw_i = summarize_direction_values(dir_near, "raw", row)
    if best_raw_i is None:
        return row

    best_direction = DIRECTIONS[best_raw_i]
    row["best_direction"] = best_direction
    row["raw_best_energy"] = row.get("raw_top1_energy", 0.0)
    row["mean_direction_energy"] = row.get("raw_mean_direction_energy", 0.0)
    row["direction_energy_std"] = row.get("raw_direction_energy_std", 0.0)
    row["direction_cv"] = row.get("raw_direction_cv", 0.0)
    row["direction_entropy"] = row.get("raw_direction_entropy", 0.0)
    row["direction_gini"] = row.get("raw_direction_gini", 0.0)
    row["direction_top1_ratio"] = row.get("raw_direction_top1_ratio", 0.0)
    row["direction_top2_ratio"] = row.get("raw_direction_top2_ratio", 0.0)
    row["direction_contrast"] = row.get("raw_direction_contrast", 0.0)
    row["best_direction_combined_score"] = row.get("raw_best_direction_combined_score", 0.0)

    # 原始最佳方向衰减
    dists = []
    energies = []
    for dist in DISTANCES_CM:
        if dist in E_total[best_direction]:
            dists.append(dist)
            energies.append(E_total[best_direction][dist])
    attenuation_n, decay_R2 = fit_decay(dists, energies)
    row["attenuation_n"] = attenuation_n
    row["decay_R2"] = decay_R2

    if energies:
        near_e = np.mean([e for d, e in zip(dists, energies) if d <= 20]) if any(d <= 20 for d in dists) else np.mean(energies)
        far_e = np.mean([e for d, e in zip(dists, energies) if d >= 30]) if any(d >= 30 for d in dists) else np.mean(energies)
        row["near_far_ratio"] = safe_ratio(near_e, far_e)
        row["energy_5cm_best_direction"] = float(E_total[best_direction].get(5, np.nan))
        row["mean_energy_best_direction"] = float(np.mean(energies))
    else:
        row["near_far_ratio"] = np.nan
        row["energy_5cm_best_direction"] = np.nan
        row["mean_energy_best_direction"] = np.nan

    if len(energies) >= 2:
        decreases = sum(1 for a, b in zip(energies[:-1], energies[1:]) if a >= b)
        row["monotonic_decay_ratio"] = decreases / (len(energies) - 1)
    else:
        row["monotonic_decay_ratio"] = 0.0

    row["decay_score"] = float(
        row["monotonic_decay_ratio"] * row["decay_R2"] * min(row["near_far_ratio"], 10.0) / 10.0
    )
    row["direction_agreement"] = 1.0

    # 原始最佳方向近场频带
    raw_best_band = np.zeros(len(SUBBANDS), dtype=float)
    for dist in NEAR_DISTANCES_CM:
        if dist in E_band[best_direction]:
            raw_best_band += np.asarray(E_band[best_direction][dist], dtype=float)
    if np.sum(raw_best_band) <= 1e-20:
        for dist in DISTANCES_CM:
            if dist in E_band[best_direction]:
                raw_best_band += np.asarray(E_band[best_direction][dist], dtype=float)

    add_band_ratio_features(row, raw_best_band, "raw")

    # 用原始PSD详细频谱覆盖 raw_ 近似，并保留 v7 原始命名
    chosen_spec = None
    for dist in NEAR_DISTANCES_CM + [25, 30, 35, 40]:
        if dist in E_spec[best_direction]:
            chosen_spec = E_spec[best_direction][dist]
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

    # -------------------------------------------------------------------------
    # B. 40cm 背景特征
    # -------------------------------------------------------------------------
    bg40_total_by_dir = np.array([
        E_total[d].get(BACKGROUND_DISTANCE_CM, np.nan) for d in DIRECTIONS
    ], dtype=float)

    valid_bg_totals = bg40_total_by_dir[np.isfinite(bg40_total_by_dir)]
    center_bg_total_median = float(np.median(valid_bg_totals)) if len(valid_bg_totals) else 0.0
    center_bg_total_mean = float(np.mean(valid_bg_totals)) if len(valid_bg_totals) else 0.0

    bg40_band_by_dir = []
    for d in DIRECTIONS:
        if BACKGROUND_DISTANCE_CM in E_band[d]:
            bg40_band_by_dir.append(np.asarray(E_band[d][BACKGROUND_DISTANCE_CM], dtype=float))
    if bg40_band_by_dir:
        center_bg_band_median = np.median(np.asarray(bg40_band_by_dir), axis=0)
        center_bg_band_mean = np.mean(np.asarray(bg40_band_by_dir), axis=0)
    else:
        center_bg_band_median = np.zeros(len(SUBBANDS), dtype=float)
        center_bg_band_mean = np.zeros(len(SUBBANDS), dtype=float)

    row["bg40_center_total_median"] = center_bg_total_median
    row["bg40_center_total_mean"] = center_bg_total_mean
    add_band_ratio_features(row, center_bg_band_median, "bg40_center_median")

    # 近场相对40cm比值：同方向40cm、center中位40cm两种
    same_dir_ratios = []
    same_dir_log_ratios = []
    center_median_ratios = []
    center_median_log_ratios = []

    for direction in DIRECTIONS:
        b_same = E_total[direction].get(BACKGROUND_DISTANCE_CM, np.nan)
        for dist in NEAR_DISTANCES_CM:
            if dist not in E_total[direction]:
                continue
            e = E_total[direction][dist]
            if np.isfinite(b_same):
                same_dir_ratios.append(safe_ratio(e, b_same))
                same_dir_log_ratios.append(safe_log_ratio(e, b_same))
            center_median_ratios.append(safe_ratio(e, center_bg_total_median))
            center_median_log_ratios.append(safe_log_ratio(e, center_bg_total_median))

    for name, arr in [
        ("bg40_same_dir_near_to_40_ratio", same_dir_ratios),
        ("bg40_same_dir_log_near_to_40_ratio", same_dir_log_ratios),
        ("bg40_center_median_near_to_40_ratio", center_median_ratios),
        ("bg40_center_median_log_near_to_40_ratio", center_median_log_ratios),
    ]:
        arr = np.asarray(arr, dtype=float)
        if len(arr):
            row[f"{name}_mean"] = float(np.mean(arr))
            row[f"{name}_median"] = float(np.median(arr))
            row[f"{name}_max"] = float(np.max(arr))
            row[f"{name}_std"] = float(np.std(arr))
        else:
            row[f"{name}_mean"] = np.nan
            row[f"{name}_median"] = np.nan
            row[f"{name}_max"] = np.nan
            row[f"{name}_std"] = np.nan

    # 单独的 5/10/15/20cm 比值，按方向求中位数
    for dist in NEAR_DISTANCES_CM:
        vals_same = []
        vals_center = []
        log_same = []
        log_center = []
        for direction in DIRECTIONS:
            if dist not in E_total[direction]:
                continue
            e = E_total[direction][dist]
            b_same = E_total[direction].get(BACKGROUND_DISTANCE_CM, np.nan)
            if np.isfinite(b_same):
                vals_same.append(safe_ratio(e, b_same))
                log_same.append(safe_log_ratio(e, b_same))
            vals_center.append(safe_ratio(e, center_bg_total_median))
            log_center.append(safe_log_ratio(e, center_bg_total_median))

        row[f"bg40_same_dir_d{dist}_to_40_ratio_median"] = float(np.median(vals_same)) if vals_same else np.nan
        row[f"bg40_same_dir_d{dist}_to_40_log_ratio_median"] = float(np.median(log_same)) if log_same else np.nan
        row[f"bg40_center_median_d{dist}_to_40_ratio_median"] = float(np.median(vals_center)) if vals_center else np.nan
        row[f"bg40_center_median_d{dist}_to_40_log_ratio_median"] = float(np.median(log_center)) if log_center else np.nan

    # 背景扣除矩阵：same-dir 40cm 扣除
    bg_net_dir_near = []
    bg_center_net_dir_near = []

    for direction in DIRECTIONS:
        b_same_total = E_total[direction].get(BACKGROUND_DISTANCE_CM, np.nan)
        v_same = []
        v_center = []
        for dist in NEAR_DISTANCES_CM:
            if dist not in E_total[direction]:
                continue
            e = E_total[direction][dist]
            if np.isfinite(b_same_total):
                v_same.append(max(e - b_same_total, 0.0))
            v_center.append(max(e - center_bg_total_median, 0.0))
        bg_net_dir_near.append(float(np.sum(v_same)) if v_same else 0.0)
        bg_center_net_dir_near.append(float(np.sum(v_center)) if v_center else 0.0)

    best_bg_i = summarize_direction_values(bg_net_dir_near, "bg40_net", row)
    best_center_bg_i = summarize_direction_values(bg_center_net_dir_near, "bg40_center_net", row)

    if best_bg_i is None:
        best_bg_i = best_raw_i
    best_bg_direction = DIRECTIONS[best_bg_i]
    row["bg40_best_direction"] = best_bg_direction

    # 背景扣除后的最佳方向频带
    bg_net_best_band = np.zeros(len(SUBBANDS), dtype=float)
    bg_center_net_best_band = np.zeros(len(SUBBANDS), dtype=float)

    same_bg_band = E_band[best_bg_direction].get(BACKGROUND_DISTANCE_CM, np.zeros(len(SUBBANDS)))

    for dist in NEAR_DISTANCES_CM:
        if dist in E_band[best_bg_direction]:
            band = np.asarray(E_band[best_bg_direction][dist], dtype=float)
            bg_net_best_band += np.maximum(band - same_bg_band, 0.0)
            bg_center_net_best_band += np.maximum(band - center_bg_band_median, 0.0)

    add_band_ratio_features(row, bg_net_best_band, "bg40_net")
    add_band_ratio_features(row, bg_center_net_best_band, "bg40_center_net")

    # 背景扣除后的衰减特征：same-dir net
    d_bg = []
    e_bg = []
    d_center_bg = []
    e_center_bg = []

    for dist in DISTANCES_CM:
        if dist not in E_total[best_bg_direction]:
            continue
        e = E_total[best_bg_direction][dist]
        b_same = E_total[best_bg_direction].get(BACKGROUND_DISTANCE_CM, np.nan)
        if np.isfinite(b_same):
            d_bg.append(dist)
            e_bg.append(max(e - b_same, 0.0))
        d_center_bg.append(dist)
        e_center_bg.append(max(e - center_bg_total_median, 0.0))

    bg_n, bg_r2 = fit_decay(d_bg, e_bg)
    row["bg40_net_attenuation_n"] = bg_n
    row["bg40_net_decay_R2"] = bg_r2

    center_bg_n, center_bg_r2 = fit_decay(d_center_bg, e_center_bg)
    row["bg40_center_net_attenuation_n"] = center_bg_n
    row["bg40_center_net_decay_R2"] = center_bg_r2

    if e_bg:
        near_e = np.mean([e for d, e in zip(d_bg, e_bg) if d <= 20]) if any(d <= 20 for d in d_bg) else np.mean(e_bg)
        far_e = np.mean([e for d, e in zip(d_bg, e_bg) if d >= 30]) if any(d >= 30 for d in d_bg) else np.mean(e_bg)
        row["bg40_net_near_far_ratio"] = safe_ratio(near_e, far_e)
        if len(e_bg) >= 2:
            decreases = sum(1 for a, b in zip(e_bg[:-1], e_bg[1:]) if a >= b)
            row["bg40_net_monotonic_decay_ratio"] = decreases / (len(e_bg) - 1)
        else:
            row["bg40_net_monotonic_decay_ratio"] = 0.0
    else:
        row["bg40_net_near_far_ratio"] = np.nan
        row["bg40_net_monotonic_decay_ratio"] = 0.0

    if e_center_bg:
        near_e = np.mean([e for d, e in zip(d_center_bg, e_center_bg) if d <= 20]) if any(d <= 20 for d in d_center_bg) else np.mean(e_center_bg)
        far_e = np.mean([e for d, e in zip(d_center_bg, e_center_bg) if d >= 30]) if any(d >= 30 for d in d_center_bg) else np.mean(e_center_bg)
        row["bg40_center_net_near_far_ratio"] = safe_ratio(near_e, far_e)
        if len(e_center_bg) >= 2:
            decreases = sum(1 for a, b in zip(e_center_bg[:-1], e_center_bg[1:]) if a >= b)
            row["bg40_center_net_monotonic_decay_ratio"] = decreases / (len(e_center_bg) - 1)
        else:
            row["bg40_center_net_monotonic_decay_ratio"] = 0.0
    else:
        row["bg40_center_net_near_far_ratio"] = np.nan
        row["bg40_center_net_monotonic_decay_ratio"] = 0.0

    # 多频段方向一致性：每个频段扣40cm后，最佳方向是否一致
    band_best_dirs = []
    for j in range(len(SUBBANDS)):
        vals = []
        for direction in DIRECTIONS:
            b = E_band[direction].get(BACKGROUND_DISTANCE_CM, np.zeros(len(SUBBANDS)))[j]
            v = 0.0
            for dist in NEAR_DISTANCES_CM:
                if dist in E_band[direction]:
                    v += max(float(E_band[direction][dist][j] - b), 0.0)
            vals.append(v)
        if np.max(vals) > 0:
            band_best_dirs.append(DIRECTIONS[int(np.argmax(vals))])

    if band_best_dirs:
        cnt = Counter(band_best_dirs)
        most_common_dir, most_common_count = cnt.most_common(1)[0]
        row["bg40_band_best_direction_mode"] = most_common_dir
        row["bg40_band_direction_consistency"] = float(most_common_count / len(band_best_dirs))
        row["bg40_band_direction_unique_count"] = int(len(cnt))
        row["bg40_band_direction_agrees_with_best"] = float(most_common_dir == best_bg_direction)
    else:
        row["bg40_band_best_direction_mode"] = ""
        row["bg40_band_direction_consistency"] = 0.0
        row["bg40_band_direction_unique_count"] = 0
        row["bg40_band_direction_agrees_with_best"] = 0.0

    # -------------------------------------------------------------------------
    # C. 时间特征
    # -------------------------------------------------------------------------
    time_source = center_file
    if not time_source:
        # 优先用原始最佳方向5cm/10cm作时间特征来源
        for dist in [5, 10, 15, 20, 25, 30, 35, 40]:
            if dist in representative_files.get(best_direction, {}):
                time_source = representative_files[best_direction][dist]
                break

    if time_source and os.path.exists(time_source):
        row.update(time_features_from_wav(time_source))
    else:
        row.update({
            "time_energy_mean": np.nan,
            "time_energy_std": np.nan,
            "time_energy_cv": np.nan,
            "time_energy_max_mean_ratio": np.nan,
            "time_energy_kurtosis": np.nan,
            "time_rms": np.nan,
        })

    return row


def extract_features_from_root(offset_root, center_root="", label="", dataset_name="DATASET"):
    time_dirs = resolve_time_dirs(offset_root)
    all_rows = []
    quality_rows = []

    print("\n" + "=" * 100)
    print(f"开始提取特征: {dataset_name} | label={label if label else 'UNKNOWN'}")
    print("offset_root:", offset_root)
    print("center_root:", center_root if center_root else "(未提供)")
    print("=" * 100)

    for time_name, time_dir in time_dirs:
        print("\n" + "-" * 100)
        print(f"[{dataset_name}] time: {time_name}")
        print("offset_dir:", time_dir)

        offset_files, schema_rows = discover_offset_files_from_time_dir(time_dir, verbose=True)

        center_time_dir = match_center_time_dir(center_root, time_name) if center_root else ""
        center_files = discover_center_files_from_time_dir(center_time_dir) if center_time_dir else {}

        centers = sorted(set(k[0] for k in offset_files.keys()))

        count_by_center = {}
        bg40_by_center = {}
        for (cc, dd, dist), files in offset_files.items():
            count_by_center.setdefault(cc, 0)
            count_by_center[cc] += 1
            if dist == BACKGROUND_DISTANCE_CM:
                bg40_by_center.setdefault(cc, 0)
                bg40_by_center[cc] += 1

        avg_count = float(np.mean(list(count_by_center.values()))) if count_by_center else 0.0
        avg_bg40 = float(np.mean(list(bg40_by_center.values()))) if bg40_by_center else 0.0

        print("  center数量:", len(centers))
        print("  center wav数量:", len(center_files))
        print("  offset组合数量:", len(offset_files))
        print(f"  平均每center offset组合数: {avg_count:.1f} / 64")
        print(f"  平均每center 40cm方向数: {avg_bg40:.1f} / 8")
        print("  前10个center offset数:", sorted(count_by_center.items())[:10])

        quality_rows.append({
            "dataset": dataset_name,
            "time": time_name,
            "offset_dir": time_dir,
            "n_centers": len(centers),
            "n_center_wav": len(center_files),
            "n_offset_combos": len(offset_files),
            "avg_offset_combos_per_center": avg_count,
            "avg_bg40_dirs_per_center": avg_bg40,
            "offset_parse_schemas": json.dumps(schema_rows, ensure_ascii=False),
        })

        if avg_count < MIN_VALID_OFFSET_COMBOS:
            print("  [警告] 平均每center offset组合数明显少于64，结果可能不可靠。")

        for i, center in enumerate(centers, 1):
            if i % 10 == 0 or i == len(centers):
                print(f"  已处理 {i}/{len(centers)}")

            center_file = center_files.get(center, "")
            row = compute_center_features(
                time_name=time_name,
                center=center,
                offset_files=offset_files,
                center_file=center_file,
                dataset_name=dataset_name,
                label=label,
            )
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    qdf = pd.DataFrame(quality_rows)

    if len(df) == 0:
        raise RuntimeError(f"{dataset_name} 没有成功提取任何 center 特征。")

    return df, qdf


# =============================================================================
# 6. v7风格 + 跨工厂特征矩阵
# =============================================================================

DROP_COLS = {
    "dataset",
    "label",
    "time",
    "center",
    "center_norm",
    "center_file",
    "best_direction",
    "raw_best_direction",
    "bg40_best_direction",
    "bg40_net_best_direction",
    "bg40_center_net_best_direction",
    "bg40_band_best_direction_mode",
    "data_quality_flag",
}

ALLOW_TIME_FEATURES = {
    "time_energy_cv",
    "time_energy_kurtosis",
    "time_energy_max_mean_ratio",
}

ABSOLUTE_ENERGY_EXACT = {
    "raw_best_energy",
    "mean_energy_best_direction",
    "energy_5cm_best_direction",
    "energy_20_70",
    "energy_20_40",
    "energy_40_70",
    "energy_20_30k",
    "energy_30_40k",
    "energy_40_50k",
    "energy_50_60k",
    "energy_60_70k",
    "raw_energy_20_70",
    "raw_energy_20_40",
    "raw_energy_40_70",
    "time_energy_mean",
    "time_energy_std",
    "time_rms",
}


def is_absolute_energy_feature(col):
    """
    原始绝对能量不直接给模型，但它的 time/scene rank 和 robust_z 会保留。
    bg40 ratio/log/delta 类特征保留。
    """
    c = str(col).lower()

    if col in ALLOW_TIME_FEATURES:
        return False

    if c in ABSOLUTE_ENERGY_EXACT:
        return True

    # 原始 energy_ 开头一般是绝对能量
    if c.startswith("energy_"):
        return True

    # raw_energy_ 开头也是绝对能量
    if c.startswith("raw_energy_"):
        return True

    # bg40_center_total_median 等是背景绝对能量，不直接使用
    if c in {"bg40_center_total_median", "bg40_center_total_mean"}:
        return True

    # 其他 bg40_net_energy_xxx 是扣除后的绝对量，直接用风险较高；但其相对rank/z会保留。
    if c.startswith("bg40_net_energy_") or c.startswith("bg40_center_net_energy_"):
        return True

    if c.startswith("bg40_center_median_energy_"):
        return True

    return False


def get_numeric_columns(df):
    cols = []
    for c in df.columns:
        if c in DROP_COLS:
            continue
        vals = safe_float_series(df[c])
        if vals.notna().mean() >= 0.75:
            cols.append(c)
    return cols


def make_numeric_base(df, numeric_cols, medians=None):
    x = pd.DataFrame(index=df.index)
    medians_out = {} if medians is None else dict(medians)

    for c in numeric_cols:
        vals = safe_float_series(df[c]) if c in df.columns else pd.Series(np.nan, index=df.index)
        if medians is None:
            med = vals.median()
            if not np.isfinite(med):
                med = 0.0
            medians_out[c] = float(med)
        else:
            med = float(medians_out.get(c, 0.0))
        x[c] = vals.fillna(med).astype(float)

    return x, medians_out


def add_group_relative_features(meta_df, base_x, group_col="time"):
    """
    复用 v7 的核心：在每个 time 内做 robust_z 和 rank_pct。
    robust_z 使用 IQR，和 v7 一致。
    """
    out = pd.DataFrame(index=base_x.index)
    groups = meta_df[group_col].astype(str) if group_col in meta_df.columns else pd.Series("ALL", index=base_x.index)
    eps = 1e-12

    for c in base_x.columns:
        values = base_x[c].astype(float)

        z_col = f"{c}__time_robust_z"
        z_values = pd.Series(index=base_x.index, dtype=float)

        for g, idx in groups.groupby(groups).groups.items():
            sub = values.loc[idx]
            med = float(np.median(sub))
            q75 = float(np.percentile(sub, 75))
            q25 = float(np.percentile(sub, 25))
            iqr = q75 - q25
            if abs(iqr) < eps:
                iqr = float(np.std(sub)) + eps
            z_values.loc[idx] = (sub - med) / (iqr + eps)

        out[z_col] = z_values.fillna(0.0).astype(float)

        r_col = f"{c}__time_rank_pct"
        rank_values = values.groupby(groups).rank(method="average", pct=True)
        out[r_col] = rank_values.fillna(0.5).astype(float)

    return out


def prepare_train_feature_matrix(df):
    """
    训练阶段：
      1. 找数值特征；
      2. 原始绝对能量不作为 base 特征；
      3. 对所有数值特征生成 time robust_z/rank_pct；
      4. base稳健特征 + 所有relative特征 共同训练。
    """
    numeric_cols = get_numeric_columns(df)
    base_x_all, medians = make_numeric_base(df, numeric_cols, medians=None)

    kept_base_cols = [c for c in numeric_cols if not is_absolute_energy_feature(c)]
    dropped_base_cols = [c for c in numeric_cols if is_absolute_energy_feature(c)]

    rel_x = add_group_relative_features(df, base_x_all, group_col="time")

    X = pd.concat([base_x_all[kept_base_cols], rel_x], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    feature_info = {
        "numeric_cols": numeric_cols,
        "kept_base_cols": kept_base_cols,
        "dropped_base_cols": dropped_base_cols,
        "model_features": X.columns.tolist(),
        "feature_medians": medians,
    }

    return X, feature_info


def prepare_predict_feature_matrix(df, feature_info):
    numeric_cols = feature_info["numeric_cols"]
    medians = feature_info.get("feature_medians", {})
    kept_base_cols = feature_info["kept_base_cols"]
    model_features = feature_info["model_features"]

    base_x_all, _ = make_numeric_base(df, numeric_cols, medians=medians)
    rel_x = add_group_relative_features(df, base_x_all, group_col="time")

    X_all = pd.concat([base_x_all[[c for c in kept_base_cols if c in base_x_all.columns]], rel_x], axis=1)

    X = pd.DataFrame(index=df.index)
    missing = []
    for c in model_features:
        if c in X_all.columns:
            X[c] = safe_float_series(X_all[c]).fillna(0.0).astype(float)
        else:
            X[c] = 0.0
            missing.append(c)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, missing


# =============================================================================
# 7. 模型训练和预测
# =============================================================================

def build_classifier():
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=900,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
    )


def get_group_oof_probabilities(X_train, y_train, groups_train):
    from sklearn.model_selection import StratifiedKFold

    y_train = np.asarray(y_train, dtype=int)
    groups_train = np.asarray(groups_train).astype(str)
    unique_groups = sorted(pd.unique(groups_train).tolist())

    oof_prob = np.zeros(len(y_train), dtype=float)
    filled = np.zeros(len(y_train), dtype=bool)

    # 优先按 time 留出，复用 v7 验证方式
    if len(unique_groups) >= 2:
        for g in unique_groups:
            val_mask = groups_train == g
            tr_mask = ~val_mask
            if len(np.unique(y_train[tr_mask])) < 2:
                continue
            clf = build_classifier()
            clf.fit(X_train.loc[tr_mask], y_train[tr_mask])
            oof_prob[val_mask] = clf.predict_proba(X_train.loc[val_mask])[:, 1]
            filled[val_mask] = True

    # 兜底：分层K折
    if not np.all(filled):
        min_class_count = min(np.sum(y_train == 0), np.sum(y_train == 1))
        n_splits = max(2, min(5, int(min_class_count)))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        for tr_idx, val_idx in cv.split(X_train, y_train):
            clf = build_classifier()
            clf.fit(X_train.iloc[tr_idx], y_train[tr_idx])
            oof_prob[val_idx] = clf.predict_proba(X_train.iloc[val_idx])[:, 1]
            filled[val_idx] = True

    return oof_prob


def extract_training_dataset():
    dfs = []
    qdfs = []

    for i, offset_root in enumerate(TRAIN_TRUE_OFFSET_ROOTS):
        center_root = TRAIN_TRUE_CENTER_ROOTS[i] if i < len(TRAIN_TRUE_CENTER_ROOTS) else ""
        df, qdf = extract_features_from_root(
            offset_root=offset_root,
            center_root=center_root,
            label="TRUE_LEAK",
            dataset_name=f"A_TRUE_{i+1}",
        )
        dfs.append(df)
        qdfs.append(qdf)

    for i, offset_root in enumerate(TRAIN_FALSE_OFFSET_ROOTS):
        center_root = TRAIN_FALSE_CENTER_ROOTS[i] if i < len(TRAIN_FALSE_CENTER_ROOTS) else ""
        df, qdf = extract_features_from_root(
            offset_root=offset_root,
            center_root=center_root,
            label="FALSE_LEAK",
            dataset_name=f"A_FALSE_{i+1}",
        )
        dfs.append(df)
        qdfs.append(qdf)

    train_df = pd.concat(dfs, ignore_index=True)
    quality_df = pd.concat(qdfs, ignore_index=True)

    train_df = train_df[train_df["label"].isin(["TRUE_LEAK", "FALSE_LEAK"])].copy()
    train_df = train_df.reset_index(drop=True)

    return train_df, quality_df


def train_model():
    ensure_dir(OUTPUT_DIR)

    print("\n" + "=" * 100)
    print("v10 训练阶段：A工厂 TRUE/FALSE，提取 v7 + bg40 特征")
    print("=" * 100)

    train_df, quality_df = extract_training_dataset()

    train_feature_csv = os.path.join(OUTPUT_DIR, "v10_train_feature_dataset.csv")
    train_df.to_csv(train_feature_csv, index=False, encoding="utf-8-sig")

    quality_csv = os.path.join(OUTPUT_DIR, "v10_data_quality_report.csv")
    quality_df.to_csv(quality_csv, index=False, encoding="utf-8-sig")

    print("\n训练样本数:", len(train_df))
    print(train_df["label"].value_counts())

    X, feature_info = prepare_train_feature_matrix(train_df)
    y = label_to_binary(train_df["label"].astype(str).values)
    groups = train_df["time"].astype(str).values

    model_feature_dataset = pd.concat(
        [
            train_df[["dataset", "time", "center", "center_norm", "label", "data_quality_flag", "offset_combo_count", "bg40_direction_count"]].reset_index(drop=True),
            X.reset_index(drop=True),
        ],
        axis=1,
    )
    model_feature_csv = os.path.join(OUTPUT_DIR, "v10_model_feature_dataset.csv")
    model_feature_dataset.to_csv(model_feature_csv, index=False, encoding="utf-8-sig")

    print("模型特征数:", X.shape[1])
    print("基础稳健特征数:", len(feature_info["kept_base_cols"]))
    print("被丢弃直接输入的绝对能量特征数:", len(feature_info["dropped_base_cols"]))

    # OOF 阈值
    oof_prob = get_group_oof_probabilities(X.reset_index(drop=True), y, groups)
    best_t, best_score, threshold_curve = find_best_threshold(y, oof_prob, metric=THRESHOLD_METRIC)
    threshold_curve_csv = os.path.join(OUTPUT_DIR, "v10_train_group_oof_threshold_curve.csv")
    threshold_curve.to_csv(threshold_curve_csv, index=False, encoding="utf-8-sig")

    oof_pred = threshold_predict(oof_prob, best_t)
    oof_metrics = metrics_from_pred(y, oof_pred)
    oof_auc = safe_auc(y, oof_prob)

    print(f"OOF推荐阈值: {best_t:.3f}")
    print(f"OOF balanced_acc: {oof_metrics['balanced_accuracy']:.4f}, acc: {oof_metrics['accuracy']:.4f}, auc: {oof_auc:.4f}")

    clf = build_classifier()
    clf.fit(X, y)

    import joblib
    joblib.dump(clf, MODEL_PATH)

    importance_df = pd.DataFrame({
        "feature": X.columns.tolist(),
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance_csv = os.path.join(OUTPUT_DIR, "v10_feature_importance.csv")
    importance_df.to_csv(importance_csv, index=False, encoding="utf-8-sig")

    config = {
        "version": "v10_bg40_v7_domain_adaptive_classifier",
        "created_at": str(datetime.now()),
        "positive_label": "TRUE_LEAK",
        "label_mapping": {"FALSE_LEAK": 0, "TRUE_LEAK": 1},
        "recommended_threshold": best_t,
        "threshold_metric": THRESHOLD_METRIC,
        "threshold_score_on_train_group_oof": best_score,
        "oof_metrics": oof_metrics,
        "oof_auc": oof_auc,
        "freq_low": FREQ_LOW,
        "freq_high": FREQ_HIGH,
        "subbands": SUBBANDS,
        "background_distance_cm": BACKGROUND_DISTANCE_CM,
        "directions": DIRECTIONS,
        "distances_cm": DISTANCES_CM,
        "expected_offset_combos_per_center": EXPECTED_OFFSET_COMBOS_PER_CENTER,
        "feature_info": feature_info,
        "model_path": MODEL_PATH,
        "train_feature_csv": train_feature_csv,
        "model_feature_csv": model_feature_csv,
        "quality_csv": quality_csv,
        "importance_csv": importance_csv,
        "threshold_curve_csv": threshold_curve_csv,
        "note": (
            "v10保留v7稳健特征和time内部rank/z，同时新增40cm背景扣除/背景比值特征。"
            "预测新工厂时，每个center独立判断，不强制一真一假。"
        ),
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    return {
        "model": clf,
        "config": config,
        "train_df": train_df,
        "X": X,
        "importance_df": importance_df,
        "train_feature_csv": train_feature_csv,
        "model_feature_csv": model_feature_csv,
        "quality_csv": quality_csv,
        "importance_csv": importance_csv,
        "threshold_curve_csv": threshold_curve_csv,
    }


def load_model_and_config():
    import joblib
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"找不到模型文件: {MODEL_PATH}")
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"找不到配置文件: {CONFIG_PATH}")

    model = joblib.load(MODEL_PATH)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    return model, config


def assign_final_label(prob, threshold, force_binary=False):
    prob = float(prob)
    threshold = float(threshold)

    if force_binary:
        return "TRUE_LEAK" if prob >= threshold else "FALSE_LEAK"

    true_t = min(0.95, threshold + SUSPECT_MARGIN_TRUE)
    false_t = max(0.05, threshold - SUSPECT_MARGIN_FALSE)

    if prob >= true_t:
        return "TRUE_LEAK"
    if prob <= false_t:
        return "FALSE_LEAK"
    return "SUSPECT"


def predict_test_dataset(model=None, config=None):
    ensure_dir(OUTPUT_DIR)

    if model is None or config is None:
        model, config = load_model_and_config()

    print("\n" + "=" * 100)
    print("v10 预测阶段：B工厂未知数据逐center判断")
    print("=" * 100)

    test_df, test_quality_df = extract_features_from_root(
        offset_root=TEST_OFFSET_ROOT,
        center_root=TEST_CENTER_ROOT,
        label="",
        dataset_name=TEST_DATASET_NAME,
    )

    test_feature_csv = os.path.join(OUTPUT_DIR, "v10_test_raw_feature_dataset.csv")
    test_df.to_csv(test_feature_csv, index=False, encoding="utf-8-sig")

    test_quality_csv = os.path.join(OUTPUT_DIR, "v10_test_data_quality_report.csv")
    test_quality_df.to_csv(test_quality_csv, index=False, encoding="utf-8-sig")

    feature_info = config["feature_info"]
    X_test, missing = prepare_predict_feature_matrix(test_df, feature_info)

    prob = model.predict_proba(X_test)[:, 1]
    threshold = float(config.get("recommended_threshold", DEFAULT_THRESHOLD))

    pred_df = test_df.copy()
    pred_df["prob_TRUE_LEAK"] = prob
    pred_df["recommended_threshold"] = threshold
    pred_df["pred_label"] = [assign_final_label(p, threshold, FORCE_BINARY_OUTPUT) for p in prob]
    pred_df["binary_pred_at_threshold"] = np.where(prob >= threshold, "TRUE_LEAK", "FALSE_LEAK")

    # 每个time内部概率排名只作为诊断，不参与最终强制分类
    pred_df["prob_rank_pct_in_time"] = 0.5
    for t, idx in pred_df.groupby("time").groups.items():
        pred_df.loc[idx, "prob_rank_pct_in_time"] = pred_df.loc[idx, "prob_TRUE_LEAK"].rank(method="average", pct=True)

    # 数据质量不足则覆盖为 INVALID_DATA
    invalid_mask = pred_df["data_quality_flag"].astype(str) != "OK"
    pred_df.loc[invalid_mask, "pred_label"] = "INVALID_DATA"

    # 可解释原因
    reasons = []
    for _, r in pred_df.iterrows():
        if r["pred_label"] == "INVALID_DATA":
            reasons.append(f"数据质量不足: {r.get('data_quality_flag', '')}, offset={r.get('offset_combo_count', '')}, bg40={r.get('bg40_direction_count', '')}")
        elif r["pred_label"] == "TRUE_LEAK":
            reasons.append("概率高于TRUE阈值；模型认为v7+40cm背景增强特征更接近真泄漏")
        elif r["pred_label"] == "FALSE_LEAK":
            reasons.append("概率低于FALSE阈值；未表现出稳定真泄漏特征")
        else:
            reasons.append("概率处于不确定区间，建议人工复核或采集背景/标定样本")
    pred_df["decision_reason"] = reasons

    # 标签评估，可选
    eval_df = pd.DataFrame()
    if TEST_LABEL_CSV and os.path.exists(TEST_LABEL_CSV):
        labels = pd.read_csv(TEST_LABEL_CSV)
        labels["center_norm"] = labels["center"].apply(normalize_center_id)
        labels["time"] = labels["time"].astype(str)
        labels["label"] = labels["label"].astype(str)

        pred_df = pred_df.merge(
            labels[["time", "center_norm", "label"]].rename(columns={"label": "true_label"}),
            on=["time", "center_norm"],
            how="left",
        )

        eval_rows = []
        valid_eval = pred_df[pred_df["true_label"].isin(["TRUE_LEAK", "FALSE_LEAK"])].copy()
        decisive = valid_eval[valid_eval["pred_label"].isin(["TRUE_LEAK", "FALSE_LEAK"])]

        if len(valid_eval):
            # strict：SUSPECT/INVALID 都算错
            strict_correct = (valid_eval["pred_label"] == valid_eval["true_label"]).astype(int)
            eval_rows.append({
                "scope": "all",
                "n": len(valid_eval),
                "strict_accuracy_suspect_invalid_as_wrong": float(strict_correct.mean()),
                "decisive_n": len(decisive),
                "decisive_rate": float(len(decisive) / len(valid_eval)),
                "decisive_accuracy": float((decisive["pred_label"] == decisive["true_label"]).mean()) if len(decisive) else np.nan,
                "auc": safe_auc(label_to_binary(valid_eval["true_label"].values), valid_eval["prob_TRUE_LEAK"].values),
            })

            for t, g in valid_eval.groupby("time"):
                dg = g[g["pred_label"].isin(["TRUE_LEAK", "FALSE_LEAK"])]
                eval_rows.append({
                    "scope": str(t),
                    "n": len(g),
                    "strict_accuracy_suspect_invalid_as_wrong": float((g["pred_label"] == g["true_label"]).mean()),
                    "decisive_n": len(dg),
                    "decisive_rate": float(len(dg) / len(g)),
                    "decisive_accuracy": float((dg["pred_label"] == dg["true_label"]).mean()) if len(dg) else np.nan,
                    "auc": safe_auc(label_to_binary(g["true_label"].values), g["prob_TRUE_LEAK"].values),
                })

        eval_df = pd.DataFrame(eval_rows)
        eval_csv = os.path.join(OUTPUT_DIR, "v10_test_label_evaluation.csv")
        eval_df.to_csv(eval_csv, index=False, encoding="utf-8-sig")
    else:
        eval_csv = ""

    front_cols = [
        "dataset", "time", "center_norm", "center",
        "pred_label", "prob_TRUE_LEAK", "recommended_threshold", "binary_pred_at_threshold",
        "prob_rank_pct_in_time",
        "data_quality_flag", "offset_combo_count", "bg40_direction_count",
        "best_direction", "bg40_best_direction",
        "direction_contrast", "bg40_net_direction_contrast", "bg40_center_net_direction_contrast",
        "near_far_ratio", "bg40_net_near_far_ratio", "bg40_center_net_near_far_ratio",
        "ratio_60_70k", "bg40_net_ratio_60_70k", "bg40_center_net_ratio_60_70k",
        "spec_slope", "bg40_net_spec_slope", "bg40_center_net_spec_slope",
        "decision_reason",
    ]
    if "true_label" in pred_df.columns:
        front_cols.insert(4, "true_label")

    cols = [c for c in front_cols if c in pred_df.columns] + [c for c in pred_df.columns if c not in front_cols]
    pred_df = pred_df[cols]

    pred_csv = os.path.join(OUTPUT_DIR, "v10_test_per_center_predictions.csv")
    pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")

    summary_rows = []
    for t, g in pred_df.groupby("time"):
        summary_rows.append({
            "time": t,
            "n_centers": len(g),
            "n_TRUE_LEAK": int((g["pred_label"] == "TRUE_LEAK").sum()),
            "n_FALSE_LEAK": int((g["pred_label"] == "FALSE_LEAK").sum()),
            "n_SUSPECT": int((g["pred_label"] == "SUSPECT").sum()),
            "n_INVALID_DATA": int((g["pred_label"] == "INVALID_DATA").sum()),
            "mean_prob_TRUE": float(g["prob_TRUE_LEAK"].mean()),
            "median_prob_TRUE": float(g["prob_TRUE_LEAK"].median()),
            "mean_offset_combo_count": float(g["offset_combo_count"].mean()),
            "mean_bg40_direction_count": float(g["bg40_direction_count"].mean()),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(OUTPUT_DIR, "v10_test_scene_summary.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    missing_path = os.path.join(OUTPUT_DIR, "v10_missing_model_features_in_test.txt")
    save_text(missing_path, "\n".join(missing))

    print("\n预测完成")
    print("逐center预测:", pred_csv)
    print("time汇总:", summary_csv)
    print("测试特征:", test_feature_csv)
    print("测试数据质量:", test_quality_csv)
    print("缺失模型特征:", missing_path)
    if eval_csv:
        print("标签评估:", eval_csv)

    print("\n逐time预测汇总:")
    print(summary_df.to_string(index=False))

    print("\n前30个center预测:")
    show_cols = [c for c in ["time", "center_norm", "pred_label", "prob_TRUE_LEAK", "data_quality_flag", "offset_combo_count", "bg40_direction_count"] if c in pred_df.columns]
    print(pred_df[show_cols].head(30).to_string(index=False))

    return {
        "pred_df": pred_df,
        "summary_df": summary_df,
        "eval_df": eval_df,
        "pred_csv": pred_csv,
        "summary_csv": summary_csv,
        "test_feature_csv": test_feature_csv,
        "test_quality_csv": test_quality_csv,
        "missing_path": missing_path,
        "eval_csv": eval_csv,
    }


# =============================================================================
# 8. 报告
# =============================================================================

def make_report(train_result=None, pred_result=None):
    lines = []
    lines.append("v10：v7稳健特征 + 40cm背景扣除/比值 + 跨工厂逐center判断 报告")
    lines.append("=" * 100)
    lines.append(f"生成时间: {datetime.now()}")
    lines.append("")
    lines.append("核心方法:")
    lines.append("  1. 保留 v7 的稳健频谱/方向/衰减/时间特征；")
    lines.append("  2. 每个 center 使用 40cm 偏离点作为局部背景；")
    lines.append("  3. 新增 E(d)/E(40cm)、log(E(d)/E(40cm)、max(E(d)-E(40cm),0) 等背景校正特征；")
    lines.append("  4. 继续生成 time 内部 robust_z 和 rank_pct；")
    lines.append("  5. 每个 center 独立判断，不强制一真一假；")
    lines.append("  6. 默认输出 TRUE/FALSE/SUSPECT/INVALID_DATA。")
    lines.append("")

    if train_result is not None:
        config = train_result["config"]
        train_df = train_result["train_df"]
        lines.append("训练数据:")
        lines.append(f"  样本数: {len(train_df)}")
        for label, count in train_df["label"].value_counts().items():
            lines.append(f"  {label}: {int(count)}")
        lines.append(f"  模型特征数: {len(config['feature_info']['model_features'])}")
        lines.append(f"  推荐阈值: {config['recommended_threshold']:.3f}")
        lines.append(f"  OOF balanced_accuracy: {config['oof_metrics']['balanced_accuracy']:.4f}")
        lines.append(f"  OOF accuracy: {config['oof_metrics']['accuracy']:.4f}")
        lines.append(f"  OOF AUC: {config['oof_auc']}")
        lines.append(f"  模型文件: {MODEL_PATH}")
        lines.append(f"  配置文件: {CONFIG_PATH}")
        lines.append("")
        lines.append("重要特征前30:")
        for _, r in train_result["importance_df"].head(30).iterrows():
            lines.append(f"  {r['feature']}: {r['importance']:.6f}")
        lines.append("")

    if pred_result is not None:
        summary_df = pred_result["summary_df"]
        lines.append("预测输出:")
        lines.append(f"  逐center预测: {pred_result['pred_csv']}")
        lines.append(f"  time汇总: {pred_result['summary_csv']}")
        lines.append(f"  测试特征: {pred_result['test_feature_csv']}")
        lines.append(f"  测试数据质量: {pred_result['test_quality_csv']}")
        lines.append("")
        lines.append("逐time汇总:")
        if summary_df is not None and len(summary_df):
            for _, r in summary_df.iterrows():
                lines.append(
                    f"  {r['time']}: n={int(r['n_centers'])}, "
                    f"T={int(r['n_TRUE_LEAK'])}, F={int(r['n_FALSE_LEAK'])}, "
                    f"SUSPECT={int(r['n_SUSPECT'])}, INVALID={int(r['n_INVALID_DATA'])}, "
                    f"mean_prob={r['mean_prob_TRUE']:.4f}"
                )
        lines.append("")
        if pred_result.get("eval_csv"):
            lines.append(f"标签评估: {pred_result['eval_csv']}")
            if pred_result["eval_df"] is not None and len(pred_result["eval_df"]):
                lines.append(pred_result["eval_df"].to_string(index=False))

    report_path = os.path.join(OUTPUT_DIR, "v10_report.txt")
    save_text(report_path, "\n".join(lines))
    return report_path


# =============================================================================
# 9. 主函数
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)

    print("=" * 100)
    print("v10：v7稳健特征 + 40cm背景扣除/比值 + 跨工厂逐center判断")
    print("=" * 100)
    print("RUN_MODE:", RUN_MODE)
    print("OUTPUT_DIR:", OUTPUT_DIR)

    train_result = None
    pred_result = None

    if RUN_MODE in ["train_and_predict", "train_only"]:
        train_result = train_model()

    if RUN_MODE in ["train_and_predict", "predict_only"]:
        if train_result is not None:
            model = train_result["model"]
            config = train_result["config"]
        else:
            model, config = load_model_and_config()
        pred_result = predict_test_dataset(model=model, config=config)

    report_path = make_report(train_result=train_result, pred_result=pred_result)
    print("\n报告:", report_path)

    print("\n" + "=" * 100)
    print("全部完成")
    print("输出文件夹:", OUTPUT_DIR)
    print("=" * 100)


if __name__ == "__main__":
    main()
