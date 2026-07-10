# -*- coding: utf-8 -*-
r"""
leak_v10_scene_relative_physics_classifier.py

V10 场景自标定物理证据泄漏检测器
============================================================

设计目标
------------------------------------------------------------
1. 不再依赖 v7_final_robust_classifier.pkl。
2. 不再依赖 144226 pairwise 规则。
3. 不强制 pairwise 一真一假。
4. 对每个 time / 每个 center 独立输出 TRUE_LEAK / FALSE_LEAK / SUSPECT / INVALID_DATA。
5. 使用当前场景内部统计量做无标签自标定，适合跨工厂初步验证。

核心思想
------------------------------------------------------------
每个 center 构建：
    8方向 × 8距离 × 多频段 能量矩阵

然后计算：
    1. 当前场景内超声异常程度 scene_ultrasound_excess_score
    2. 方向集中性 direction_score
    3. 距离衰减一致性 decay_score
    4. 多频段方向一致性 band_consistency_score
    5. 弥散噪声惩罚 diffuse_penalty
    6. 数据完整度 data_quality_score

最终：
    leak_evidence_score = 物理泄漏证据综合评分

输出文件
------------------------------------------------------------
默认输出到：
    C:\Users\jiangxinru6\Desktop\wurenji\leak_v10_scene_relative_physics_results

主要文件：
    v10_per_center_predictions.csv      每个 center 的判断结果
    v10_scene_summary.csv               每个 time/scene 的汇总
    v10_suspect_centers.csv             SUSPECT/INVALID 样本
    v10_feature_matrix.csv              完整特征矩阵
    v10_report.txt                      运行报告

使用方式
------------------------------------------------------------
方式1：直接修改本文件顶部配置，然后运行：
    python leak_v10_scene_relative_physics_classifier.py

方式2：命令行覆盖路径：
    python leak_v10_scene_relative_physics_classifier.py --offset_root "D:\gas\new_factory_offset"

可选：
    python leak_v10_scene_relative_physics_classifier.py ^
        --offset_root "D:\gas\new_factory_offset" ^
        --center_root "D:\gas\new_factory_center" ^
        --output_dir "C:\Users\jiangxinru6\Desktop\wurenji\v10_test"

依赖：
    pip install numpy pandas scipy

重要说明
------------------------------------------------------------
这不是监督训练模型，而是“场景内无标签自标定 + 物理证据评分”。
它比 v7/pairwise 更适合其他工厂初步验证，但仍建议公司后续用多工厂标签训练校准器。
"""

from __future__ import annotations

import os
import re
import math
import argparse
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy import signal
from scipy.stats import kurtosis


# ============================================================
# 1. 默认配置区：你主要改这里
# ============================================================

BASE_DIR = r"C:\Users\jiangxinru6\Desktop\wurenji"

# 新工厂 / 新场景 offset 文件夹
# 可以是：
#   1. 包含多个 .ld 子文件夹的根目录
#   2. 单个 .ld 文件夹
#   3. 直接包含 wav 的文件夹
INPUT_OFFSET_ROOT = r"D:\gas\your_new_factory_offset_folder"

# 可选：center wav 文件夹。
# 如果没有，留空即可，程序会用 offset 的最佳方向近距离 wav 代替提取时间波动特征。
INPUT_CENTER_ROOT = r""

OUTPUT_DIR = os.path.join(BASE_DIR, "leak_v10_scene_relative_physics_results")

# 可选：如果你有标签 csv，可填路径用于评估。
# 标签文件建议包含列：time, center 或 center_norm, label
# label 支持 TRUE_LEAK / FALSE_LEAK / TRUE / FALSE / T / F / 1 / 0
LABEL_CSV = r""

# 频段设置
FREQ_LOW = 20000
FREQ_HIGH = 80000
SUBBANDS = [
    (20000, 30000),
    (30000, 40000),
    (40000, 50000),
    (50000, 60000),
    (60000, 70000),
    (70000, 80000),
]

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
NEAR_DISTANCE_MAX_CM = 20
FAR_DISTANCE_MIN_CM = 30

# 每个 center 标准 offset 点数量：8方向 × 8距离 = 64
EXPECTED_OFFSET_COMBOS = 64

# 判定阈值
# TRUE 要求证据比较强；FALSE 要求证据明显不足或弥散严重；中间输出 SUSPECT。
TRUE_SCORE_THRESHOLD = 0.64
FALSE_SCORE_THRESHOLD = 0.38

# 如果你必须每个 center 都二分类，可以设 True。
# 但工业应用不推荐，因为未知场景下中间样本强行二分类风险很大。
FORCE_BINARY_OUTPUT = False
FORCE_BINARY_THRESHOLD = 0.50

# 数据质量阈值
MIN_VALID_DATA_QUALITY = 0.58
MIN_TRUE_DATA_QUALITY = 0.72

# WAV 分析参数
WELCH_NPERSEG = 4096
WELCH_NOVERLAP = 2048
NFFT = 4096

WAV_EXTS = (".wav", ".WAV")


# ============================================================
# 2. 基础工具函数
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if x is None or not np.isfinite(x):
        return lo
    return float(max(lo, min(hi, x)))


def safe_ratio(a: float, b: float, eps: float = 1e-20) -> float:
    try:
        return float(a) / (float(b) + eps)
    except Exception:
        return 0.0


def normalize_center_id(x: Any) -> str:
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    nums = "".join(ch for ch in s if ch.isdigit())
    if nums == "":
        return s
    return nums.zfill(2)


def safe_numeric(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    return x


def robust_median_mad(values: pd.Series) -> Tuple[float, float]:
    v = safe_numeric(values).dropna()
    if len(v) == 0:
        return 0.0, 1.0
    med = float(v.median())
    mad = float((v - med).abs().median())
    if not np.isfinite(mad) or mad < 1e-12:
        mad = float(v.std())
    if not np.isfinite(mad) or mad < 1e-12:
        mad = 1.0
    return med, mad


def robust_z(value: float, med: float, mad: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float((value - med) / (1.4826 * mad + 1e-12))


def sigmoid(x: float) -> float:
    x = float(np.clip(x, -30, 30))
    return float(1.0 / (1.0 + np.exp(-x)))


def score_from_z(z: float, center: float = 0.5, scale: float = 1.0) -> float:
    """把 robust z 转成 0~1 分数。z 越高说明越异常突出。"""
    return sigmoid((float(z) - center) / (scale + 1e-12))


def entropy_norm(x: np.ndarray) -> float:
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


def gini_coefficient(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 0)
    if len(x) == 0 or float(np.sum(x)) <= 1e-20:
        return 0.0
    x = np.sort(x)
    n = len(x)
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / (cum[-1] + 1e-20)) / n)


def spectral_flatness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 1e-20)
    return float(np.exp(np.mean(np.log(x))) / (np.mean(x) + 1e-20))


def list_wav_files(root: str) -> List[str]:
    if not root or not os.path.exists(root):
        return []
    out = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(WAV_EXTS):
                out.append(os.path.join(dirpath, name))
    return sorted(out)


# ============================================================
# 3. 文件夹与文件名解析
# ============================================================

def resolve_time_dirs(root: str) -> List[Tuple[str, str]]:
    """
    输入可以是：
        1. 包含多个 .ld 子文件夹的根目录
        2. 单个 .ld 文件夹
        3. 直接包含 wav 的文件夹
    返回：
        [(time_name, time_dir), ...]
    """
    if not root or not os.path.exists(root):
        raise FileNotFoundError(f"输入文件夹不存在: {root}")

    root = os.path.abspath(root)
    base = os.path.basename(root)

    direct_wavs = []
    if os.path.isdir(root):
        direct_wavs = [f for f in os.listdir(root) if f.endswith(WAV_EXTS)]

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


def parse_offset_filename_candidates(path: str) -> List[Dict[str, Any]]:
    """
    支持实际 offset 命名格式，例如：
        HM20260626_142938.ld_00_14d5_down_beamform_result.wav

    正确解析通常是：
        center   = d 前面的数字，也就是 14
        distance = d 后面的数字，也就是 5
        direction = down
    """
    base = os.path.basename(path).lower().replace("-", "_").replace(" ", "_")
    direction_pattern = r"(up_left|up_right|down_left|down_right|up|down|left|right)"

    candidates: List[Dict[str, Any]] = []

    # 标准格式：.ld_00_14d5_down_beamform
    m = re.search(
        rf"\.ld_(\d{{1,3}})_(\d{{1,3}})d(\d{{1,3}})_({direction_pattern})_beamform",
        base,
        flags=re.IGNORECASE,
    )
    if m:
        first_num = m.group(1).zfill(2)
        center_before_d = m.group(2).zfill(2)
        dist = int(m.group(3))
        direction = m.group(4).lower()
        if 0 < dist <= 200:
            candidates.append({
                "schema": "A_first_number_as_center",
                "center": first_num,
                "distance": dist,
                "direction": direction,
            })
            candidates.append({
                "schema": "B_number_before_d_as_center",
                "center": center_before_d,
                "distance": dist,
                "direction": direction,
            })
        return candidates

    # 兜底格式：xx_14d5_down_xxx.wav
    direction = None
    for d in ["up_left", "up_right", "down_left", "down_right", "up", "down", "left", "right"]:
        if re.search(rf"(^|[_\-\/]){d}($|[_\-\/\.])", base):
            direction = d
            break

    md = re.search(r"(\d{1,3})d(\d{1,3})", base)
    if md and direction is not None:
        center = md.group(1).zfill(2)
        dist = int(md.group(2))
        if 0 < dist <= 200:
            candidates.append({
                "schema": "fallback_number_before_d_as_center",
                "center": center,
                "distance": dist,
                "direction": direction,
            })

    return candidates


def discover_offset_map(time_dir: str, verbose: bool = True) -> Dict[Tuple[str, str, int], List[str]]:
    files = list_wav_files(time_dir)
    schema_maps: Dict[str, Dict[Tuple[str, str, int], List[str]]] = {}

    for f in files:
        for c in parse_offset_filename_candidates(f):
            schema = c["schema"]
            key = (normalize_center_id(c["center"]), c["direction"], int(c["distance"]))
            schema_maps.setdefault(schema, {})
            schema_maps[schema].setdefault(key, [])
            schema_maps[schema][key].append(f)

    if not schema_maps:
        raise RuntimeError(f"没有识别到 offset wav，请检查命名格式: {time_dir}")

    rows = []
    for schema, mp in schema_maps.items():
        centers = sorted(set(k[0] for k in mp.keys()))
        counts = []
        for cc in centers:
            counts.append(sum(1 for k in mp.keys() if k[0] == cc))
        avg_per_center = float(np.mean(counts)) if counts else 0.0
        total_combos = len(mp)
        n_centers = len(centers)

        # 选择能识别更多 center、且每个 center 更接近 64 组合的方案。
        closeness = 1.0 / (1.0 + abs(avg_per_center - EXPECTED_OFFSET_COMBOS))
        score = n_centers * 10000 + total_combos * 10 + closeness
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

    return schema_maps[best_schema]


def parse_center_file_candidates(path: str) -> List[Tuple[str, str]]:
    base = os.path.basename(path).lower().replace("-", "_").replace(" ", "_")
    out: List[Tuple[str, str]] = []

    m = re.search(r"\.ld_(\d{1,3})(?=_beamform|_center|_result|\.|$)", base)
    if m:
        out.append(("after_ld", m.group(1).zfill(2)))

    m = re.search(r"(?:center|centre|c)_(\d{1,3})(?=_|\.|$)", base)
    if m:
        out.append(("center_token", m.group(1).zfill(2)))

    m = re.match(r"^(\d{1,3})(?=_|\.|$)", base)
    if m:
        out.append(("leading_number", m.group(1).zfill(2)))

    tokens = re.findall(r"(?<![a-zA-Z])(\d{1,3})(?![a-zA-Z])", base)
    if tokens:
        out.append(("last_number_token", tokens[-1].zfill(2)))

    seen = set()
    final = []
    for schema, center in out:
        key = (schema, center)
        if key not in seen:
            final.append((schema, center))
            seen.add(key)
    return final


def discover_center_files(time_dir: str) -> Dict[str, str]:
    if not time_dir or not os.path.exists(time_dir):
        return {}

    files = list_wav_files(time_dir)
    schema_maps: Dict[str, Dict[str, str]] = {}

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
    return schema_maps[rows[0]["schema"]]


def match_center_time_dir(center_root: str, time_name: str) -> str:
    if not center_root or not os.path.exists(center_root):
        return ""
    center_root = os.path.abspath(center_root)
    if os.path.basename(center_root) == time_name:
        return center_root
    p = os.path.join(center_root, time_name)
    if os.path.exists(p):
        return p
    return ""


# ============================================================
# 4. WAV 频谱与时间特征
# ============================================================

def read_wav_float(path: str) -> Tuple[int, np.ndarray]:
    fs, x = wavfile.read(path)

    # 多通道取平均
    if x.ndim > 1:
        x = x.mean(axis=1)

    # 保留相对幅值：整数按 dtype 最大值归一化，不按每个文件最大值归一化。
    if np.issubdtype(x.dtype, np.integer):
        max_possible = float(np.iinfo(x.dtype).max)
        x = x.astype(np.float64) / (max_possible + 1e-12)
    else:
        x = x.astype(np.float64)
        # 如果 float wav 数值异常大，做一次温和缩放保护。
        if np.nanmax(np.abs(x)) > 20:
            x = x / (np.nanmax(np.abs(x)) + 1e-12)

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - np.mean(x)
    return int(fs), x


def welch_psd(x: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) < 256:
        return np.array([]), np.array([])
    nperseg = min(WELCH_NPERSEG, len(x))
    noverlap = min(WELCH_NOVERLAP, max(0, nperseg // 2))
    nfft = max(NFFT, nperseg)
    f, pxx = signal.welch(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        scaling="density",
    )
    pxx = np.maximum(np.nan_to_num(pxx, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    return f, pxx


def band_energy(f: np.ndarray, pxx: np.ndarray, lo: float, hi: float) -> float:
    if len(f) == 0:
        return 0.0
    mask = (f >= lo) & (f < hi)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(pxx[mask], f[mask]))


def analyze_wav(path: str) -> Optional[Dict[str, Any]]:
    try:
        fs, x = read_wav_float(path)
        f, pxx = welch_psd(x, fs)
        if len(f) == 0:
            return None

        nyq = fs / 2.0
        effective_high = min(FREQ_HIGH, nyq * 0.95)

        sub_energies = []
        for lo, hi in SUBBANDS:
            if lo >= effective_high:
                sub_energies.append(0.0)
            else:
                sub_energies.append(band_energy(f, pxx, lo, min(hi, effective_high)))

        total_energy = float(np.sum(sub_energies))

        mask = (f >= FREQ_LOW) & (f <= effective_high)
        fb = f[mask]
        pb = pxx[mask]

        spec = {
            "spec_centroid_hz": 0.0,
            "spec_bandwidth_hz": 0.0,
            "spec_entropy": 0.0,
            "spec_flatness": 0.0,
            "spec_peak_freq_hz": 0.0,
            "spec_rolloff_85_hz": 0.0,
            "spec_slope": 0.0,
            "spec_peakiness": 0.0,
        }

        if len(fb) > 0 and float(np.sum(pb)) > 1e-20:
            total_p = float(np.sum(pb)) + 1e-20
            centroid = float(np.sum(fb * pb) / total_p)
            bandwidth = float(np.sqrt(np.sum(((fb - centroid) ** 2) * pb) / total_p))
            ent = entropy_norm(pb)
            flat = spectral_flatness(pb)
            peak_idx = int(np.argmax(pb))
            peak_freq = float(fb[peak_idx])
            peakiness = safe_ratio(float(np.max(pb)), float(np.mean(pb)))

            cum = np.cumsum(pb)
            idx = int(np.searchsorted(cum, 0.85 * cum[-1]))
            idx = min(idx, len(fb) - 1)
            rolloff = float(fb[idx])

            try:
                y = np.log10(pb + 1e-20)
                xx = (fb - fb.mean()) / (fb.std() + 1e-12)
                slope = float(np.polyfit(xx, y, 1)[0])
            except Exception:
                slope = 0.0

            spec.update({
                "spec_centroid_hz": centroid,
                "spec_bandwidth_hz": bandwidth,
                "spec_entropy": ent,
                "spec_flatness": flat,
                "spec_peak_freq_hz": peak_freq,
                "spec_rolloff_85_hz": rolloff,
                "spec_slope": slope,
                "spec_peakiness": peakiness,
            })

        return {
            "fs": fs,
            "total_energy_20_80": total_energy,
            "subband_energies": np.asarray(sub_energies, dtype=float),
            **spec,
        }
    except Exception:
        return None


def bandpass_signal(x: np.ndarray, fs: int) -> np.ndarray:
    nyq = fs / 2.0
    hi = min(FREQ_HIGH, nyq * 0.95)
    lo = min(FREQ_LOW, hi * 0.8)
    if hi <= lo or nyq <= 0:
        return x
    try:
        sos = signal.butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
        if len(x) > 100:
            return signal.sosfiltfilt(sos, x)
        return signal.sosfilt(sos, x)
    except Exception:
        return x


def time_features_from_wav(path: str) -> Dict[str, float]:
    try:
        fs, x = read_wav_float(path)
        xb = bandpass_signal(x, fs)
        if len(xb) == 0:
            raise ValueError("empty wav")

        win = max(128, int(0.020 * fs))
        hop = max(64, int(0.010 * fs))
        energies = []
        for start in range(0, max(1, len(xb) - win + 1), hop):
            seg = xb[start:start + win]
            if len(seg) < win // 2:
                continue
            energies.append(float(np.mean(seg ** 2)))
        if not energies:
            energies = [float(np.mean(xb ** 2))]
        e = np.asarray(energies, dtype=float)
        mean = float(np.mean(e))
        std = float(np.std(e))
        cv = safe_ratio(std, mean)
        max_mean = safe_ratio(float(np.max(e)), mean)
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


# ============================================================
# 5. 每个 center 构建 8×8×频段矩阵并提取物理特征
# ============================================================

def fit_power_decay(distances: List[int], energies: List[float]) -> Tuple[float, float]:
    d = np.asarray(distances, dtype=float)
    e = np.asarray(energies, dtype=float)
    mask = np.isfinite(d) & np.isfinite(e) & (d > 0) & (e > 0)
    d = d[mask]
    e = e[mask]
    if len(d) < 3:
        return 0.0, 0.0
    try:
        x = np.log(d)
        y = np.log(e + 1e-20)
        coef = np.polyfit(x, y, 1)
        n = float(-coef[0])
        yhat = np.polyval(coef, x)
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-20
        r2 = float(1.0 - ss_res / ss_tot)
        return max(0.0, n), clamp(r2, 0.0, 1.0)
    except Exception:
        return 0.0, 0.0


def make_empty_matrix() -> Tuple[np.ndarray, np.ndarray]:
    energy = np.zeros((len(DIRECTIONS), len(DISTANCES_CM), len(SUBBANDS)), dtype=float)
    valid = np.zeros((len(DIRECTIONS), len(DISTANCES_CM)), dtype=bool)
    return energy, valid


def extract_center_physics_features(
    time_name: str,
    center: str,
    offset_map: Dict[Tuple[str, str, int], List[str]],
    center_file: str = "",
) -> Dict[str, Any]:
    center = normalize_center_id(center)
    row: Dict[str, Any] = {
        "time": time_name,
        "center_norm": center,
        "center": center,
        "center_file": center_file if center_file else "",
    }

    energy, valid = make_empty_matrix()
    spec_records: List[Dict[str, float]] = []
    used_wavs = 0

    for i, direction in enumerate(DIRECTIONS):
        for j, dist in enumerate(DISTANCES_CM):
            files = offset_map.get((center, direction, dist), [])
            if not files:
                continue
            sub_list = []
            for f in files:
                res = analyze_wav(f)
                if res is None:
                    continue
                sub_list.append(res["subband_energies"])
                # 只记录有效 PSD 特征，后面用最佳方向近距离的平均值
                spec_records.append({
                    "direction": direction,
                    "distance_cm": dist,
                    "spec_centroid_hz": res["spec_centroid_hz"],
                    "spec_bandwidth_hz": res["spec_bandwidth_hz"],
                    "spec_entropy": res["spec_entropy"],
                    "spec_flatness": res["spec_flatness"],
                    "spec_peak_freq_hz": res["spec_peak_freq_hz"],
                    "spec_rolloff_85_hz": res["spec_rolloff_85_hz"],
                    "spec_slope": res["spec_slope"],
                    "spec_peakiness": res["spec_peakiness"],
                })
                used_wavs += 1
            if sub_list:
                energy[i, j, :] = np.mean(np.asarray(sub_list, dtype=float), axis=0)
                valid[i, j] = True

    row["offset_wav_count_used"] = int(used_wavs)
    combo_count = int(np.sum(valid))
    row["offset_combo_count"] = combo_count
    row["offset_combo_ratio"] = combo_count / EXPECTED_OFFSET_COMBOS
    row["direction_coverage"] = float(np.sum(np.any(valid, axis=1)) / len(DIRECTIONS))
    row["distance_coverage"] = float(np.sum(np.any(valid, axis=0)) / len(DISTANCES_CM))
    row["data_quality_score"] = clamp(
        0.50 * row["offset_combo_ratio"] +
        0.25 * row["direction_coverage"] +
        0.25 * row["distance_coverage"]
    )

    if combo_count == 0:
        row["extract_status"] = "NO_VALID_OFFSET_WAV"
        return row

    row["extract_status"] = "OK"

    near_mask = np.asarray([d <= NEAR_DISTANCE_MAX_CM for d in DISTANCES_CM], dtype=bool)
    far_mask = np.asarray([d >= FAR_DISTANCE_MIN_CM for d in DISTANCES_CM], dtype=bool)

    near_energy_dir_band = energy[:, near_mask, :].sum(axis=1)       # direction × band
    far_energy_dir_band = energy[:, far_mask, :].sum(axis=1)
    all_energy_dir_band = energy.sum(axis=1)

    dir_near = near_energy_dir_band.sum(axis=1)
    dir_far = far_energy_dir_band.sum(axis=1)
    dir_all = all_energy_dir_band.sum(axis=1)

    total_near = float(np.sum(dir_near))
    total_far = float(np.sum(dir_far))
    total_all = float(np.sum(dir_all))

    row["total_near_energy"] = total_near
    row["total_far_energy"] = total_far
    row["total_all_energy"] = total_all
    row["far_to_total_ratio"] = safe_ratio(total_far, total_near + total_far)

    sorted_idx = np.argsort(dir_near)[::-1]
    best_i = int(sorted_idx[0])
    second_i = int(sorted_idx[1]) if len(sorted_idx) > 1 else best_i
    best_direction = DIRECTIONS[best_i]
    row["best_direction"] = best_direction
    row["best_direction_energy"] = float(dir_near[best_i])
    row["second_direction_energy"] = float(dir_near[second_i])

    row["direction_top1_ratio"] = safe_ratio(dir_near[best_i], total_near)
    row["direction_top2_ratio"] = safe_ratio(dir_near[best_i] + dir_near[second_i], total_near)
    others = np.delete(dir_near, best_i)
    row["direction_contrast"] = safe_ratio(dir_near[best_i], float(np.mean(others)) if len(others) else 0.0)
    row["direction_entropy"] = entropy_norm(dir_near)
    row["direction_gini"] = gini_coefficient(dir_near)
    row["direction_cv"] = safe_ratio(float(np.std(dir_near)), float(np.mean(dir_near)))

    if np.max(dir_near) > 0:
        row["direction_active_count_25pct"] = int(np.sum(dir_near >= np.max(dir_near) * 0.25))
        row["direction_active_count_40pct"] = int(np.sum(dir_near >= np.max(dir_near) * 0.40))
    else:
        row["direction_active_count_25pct"] = 0
        row["direction_active_count_40pct"] = 0

    # 距离衰减：最佳方向，每个距离的 20~80k 总能量
    best_by_dist = energy[best_i, :, :].sum(axis=1)
    valid_best = valid[best_i, :]
    dists = [d for d, ok in zip(DISTANCES_CM, valid_best) if ok]
    vals = [float(e) for e, ok in zip(best_by_dist, valid_best) if ok]

    attenuation_n, decay_r2 = fit_power_decay(dists, vals)
    row["attenuation_n"] = attenuation_n
    row["decay_R2"] = decay_r2

    if vals:
        near_vals = [v for d, v in zip(dists, vals) if d <= NEAR_DISTANCE_MAX_CM]
        far_vals = [v for d, v in zip(dists, vals) if d >= FAR_DISTANCE_MIN_CM]
        near_mean = float(np.mean(near_vals)) if near_vals else float(np.mean(vals))
        far_mean = float(np.mean(far_vals)) if far_vals else float(np.mean(vals))
        row["best_near_far_ratio"] = safe_ratio(near_mean, far_mean)
    else:
        row["best_near_far_ratio"] = 0.0

    if len(vals) >= 2:
        decreases = sum(1 for a, b in zip(vals[:-1], vals[1:]) if a >= b)
        row["monotonic_decay_ratio"] = float(decreases / (len(vals) - 1))
    else:
        row["monotonic_decay_ratio"] = 0.0

    # 多频段方向一致性
    band_best_dirs = []
    band_top1_ratios = []
    band_best_energies = []
    band_total_energies = []

    for b in range(len(SUBBANDS)):
        e_dir = near_energy_dir_band[:, b]
        total_b = float(np.sum(e_dir))
        band_total_energies.append(total_b)
        if total_b <= 1e-20:
            band_best_dirs.append("none")
            band_top1_ratios.append(0.0)
            band_best_energies.append(0.0)
            continue
        bi = int(np.argmax(e_dir))
        band_best_dirs.append(DIRECTIONS[bi])
        band_top1_ratios.append(safe_ratio(float(e_dir[bi]), total_b))
        band_best_energies.append(float(e_dir[bi]))

    valid_band_dirs = [d for d, e in zip(band_best_dirs, band_total_energies) if d != "none" and e > 0]
    if valid_band_dirs:
        same_best = sum(1 for d in valid_band_dirs if d == best_direction)
        row["band_same_best_direction_ratio"] = float(same_best / len(valid_band_dirs))
        # 众数方向占比
        mode_count = max(valid_band_dirs.count(d) for d in set(valid_band_dirs))
        row["band_mode_direction_ratio"] = float(mode_count / len(valid_band_dirs))
    else:
        row["band_same_best_direction_ratio"] = 0.0
        row["band_mode_direction_ratio"] = 0.0

    row["band_mean_top1_ratio"] = float(np.mean(band_top1_ratios)) if band_top1_ratios else 0.0
    row["band_energy_entropy"] = entropy_norm(np.asarray(band_total_energies, dtype=float))
    row["band_energy_flatness"] = spectral_flatness(np.asarray(band_total_energies, dtype=float)) if np.sum(band_total_energies) > 0 else 0.0

    best_band_vec = near_energy_dir_band[best_i, :]
    best_band_total = float(np.sum(best_band_vec))
    row["best_band_total_energy"] = best_band_total
    for b, (lo, hi) in enumerate(SUBBANDS):
        row[f"best_energy_{lo//1000}_{hi//1000}k"] = float(best_band_vec[b])
        row[f"best_ratio_{lo//1000}_{hi//1000}k"] = safe_ratio(float(best_band_vec[b]), best_band_total)

    row["best_high_ratio_50_80k"] = safe_ratio(float(np.sum(best_band_vec[3:])), best_band_total)
    row["best_high_ratio_60_80k"] = safe_ratio(float(np.sum(best_band_vec[4:])), best_band_total)

    # 最佳方向近距离 PSD 特征均值
    if spec_records:
        spec_df = pd.DataFrame(spec_records)
        near_spec = spec_df[(spec_df["direction"] == best_direction) & (spec_df["distance_cm"] <= NEAR_DISTANCE_MAX_CM)]
        if len(near_spec) == 0:
            near_spec = spec_df[spec_df["direction"] == best_direction]
        if len(near_spec) == 0:
            near_spec = spec_df
        for c in [
            "spec_centroid_hz", "spec_bandwidth_hz", "spec_entropy", "spec_flatness",
            "spec_peak_freq_hz", "spec_rolloff_85_hz", "spec_slope", "spec_peakiness",
        ]:
            row[c] = float(pd.to_numeric(near_spec[c], errors="coerce").median())
    else:
        for c in [
            "spec_centroid_hz", "spec_bandwidth_hz", "spec_entropy", "spec_flatness",
            "spec_peak_freq_hz", "spec_rolloff_85_hz", "spec_slope", "spec_peakiness",
        ]:
            row[c] = np.nan

    # 时间特征：优先用 center_file，否则用最佳方向最近距离的 offset wav
    time_source = center_file
    if not time_source:
        for d in DISTANCES_CM:
            files = offset_map.get((center, best_direction, d), [])
            if files:
                time_source = files[0]
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


def extract_dataset_features(offset_root: str, center_root: str = "") -> pd.DataFrame:
    time_dirs = resolve_time_dirs(offset_root)
    rows = []

    print("\n开始提取 V10 场景物理特征")
    print("offset_root:", offset_root)
    print("center_root:", center_root if center_root else "(未提供)")

    for time_name, time_dir in time_dirs:
        print("\n" + "=" * 100)
        print("time/scene:", time_name)
        print("offset_dir:", time_dir)

        offset_map = discover_offset_map(time_dir, verbose=True)

        center_time_dir = match_center_time_dir(center_root, time_name) if center_root else ""
        center_files = discover_center_files(center_time_dir) if center_time_dir else {}

        centers = sorted(set(k[0] for k in offset_map.keys()))
        count_by_center = {cc: 0 for cc in centers}
        for key in offset_map.keys():
            count_by_center[key[0]] += 1

        print("  center数量:", len(centers))
        print("  center wav数量:", len(center_files))
        print("  offset组合数量:", len(offset_map))
        if centers:
            print(f"  平均每center offset组合数: {np.mean(list(count_by_center.values())):.1f} / {EXPECTED_OFFSET_COMBOS}")
            print("  前10个center offset组合数:", sorted(count_by_center.items())[:10])

        for i, center in enumerate(centers, 1):
            if i % 10 == 0 or i == len(centers):
                print(f"  已处理 {i}/{len(centers)}")
            center_file = center_files.get(center, "")
            row = extract_center_physics_features(
                time_name=time_name,
                center=center,
                offset_map=offset_map,
                center_file=center_file,
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError("没有提取到任何 center 特征。")
    return df


# ============================================================
# 6. 场景内自标定与证据评分
# ============================================================

def add_scene_relative_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    relative_cols = [
        "total_near_energy",
        "total_all_energy",
        "best_direction_energy",
        "best_band_total_energy",
        "time_energy_mean",
        "time_rms",
    ]

    for c in relative_cols:
        if c not in df.columns:
            continue
        df[f"{c}__scene_robust_z"] = np.nan
        df[f"{c}__scene_rank_pct"] = np.nan
        for t, idx in df.groupby("time").groups.items():
            vals = safe_numeric(df.loc[idx, c])
            med, mad = robust_median_mad(vals)
            z = vals.apply(lambda x: robust_z(float(x), med, mad) if pd.notna(x) else 0.0)
            df.loc[idx, f"{c}__scene_robust_z"] = z
            df.loc[idx, f"{c}__scene_rank_pct"] = vals.rank(method="average", pct=True)

    return df


def compute_evidence_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    scores = []

    for _, r in df.iterrows():
        status = str(r.get("extract_status", "OK"))
        dq = clamp(float(r.get("data_quality_score", 0.0)))

        if status != "OK" or dq < MIN_VALID_DATA_QUALITY:
            scores.append({
                "scene_ultrasound_excess_score": 0.0,
                "direction_score": 0.0,
                "decay_score": 0.0,
                "band_consistency_score": 0.0,
                "diffuse_penalty": 1.0,
                "leak_evidence_score": 0.0,
                "pred_label": "INVALID_DATA",
                "pred_label_binary": "FALSE_LEAK",
                "decision_reason": "数据质量不足或无有效offset wav",
            })
            continue

        # 1) 场景内超声异常分数
        z_best = float(r.get("best_direction_energy__scene_robust_z", 0.0) or 0.0)
        z_near = float(r.get("total_near_energy__scene_robust_z", 0.0) or 0.0)
        rank_best = float(r.get("best_direction_energy__scene_rank_pct", 0.5) or 0.5)
        rank_near = float(r.get("total_near_energy__scene_rank_pct", 0.5) or 0.5)

        score_z = 0.60 * score_from_z(z_best, center=0.35, scale=1.00) + 0.40 * score_from_z(z_near, center=0.35, scale=1.00)
        score_rank = 0.60 * rank_best + 0.40 * rank_near
        scene_ultra = clamp(0.70 * score_z + 0.30 * score_rank)

        # 2) 方向集中性分数
        top1 = float(r.get("direction_top1_ratio", 0.0) or 0.0)
        contrast = float(r.get("direction_contrast", 0.0) or 0.0)
        ent = float(r.get("direction_entropy", 1.0) or 1.0)
        active25 = float(r.get("direction_active_count_25pct", 8.0) or 8.0)
        gini = float(r.get("direction_gini", 0.0) or 0.0)

        s_top1 = clamp((top1 - 0.16) / (0.48 - 0.16))
        s_contrast = clamp(math.log1p(max(contrast, 0.0)) / math.log1p(8.0))
        s_entropy = clamp(1.0 - ent)
        s_active = clamp(1.0 - (active25 - 2.0) / 6.0)
        s_gini = clamp(gini / 0.65)

        direction_score = clamp(
            0.28 * s_top1 +
            0.27 * s_contrast +
            0.20 * s_entropy +
            0.15 * s_active +
            0.10 * s_gini
        )

        # 3) 距离衰减一致性分数
        near_far = float(r.get("best_near_far_ratio", 0.0) or 0.0)
        decay_r2 = float(r.get("decay_R2", 0.0) or 0.0)
        atten_n = float(r.get("attenuation_n", 0.0) or 0.0)
        mono = float(r.get("monotonic_decay_ratio", 0.0) or 0.0)

        s_near_far = clamp(math.log1p(max(near_far, 0.0)) / math.log1p(8.0))
        s_r2 = clamp(decay_r2)
        s_n = clamp((atten_n - 0.15) / (1.80 - 0.15))
        s_mono = clamp(mono)

        decay_score = clamp(
            0.35 * s_near_far +
            0.30 * s_r2 +
            0.20 * s_mono +
            0.15 * s_n
        )

        # 4) 多频段方向一致性分数
        same_best = float(r.get("band_same_best_direction_ratio", 0.0) or 0.0)
        mode_ratio = float(r.get("band_mode_direction_ratio", 0.0) or 0.0)
        band_top1 = float(r.get("band_mean_top1_ratio", 0.0) or 0.0)
        band_entropy = float(r.get("band_energy_entropy", 0.0) or 0.0)
        band_flat = float(r.get("band_energy_flatness", 0.0) or 0.0)

        s_band_top1 = clamp((band_top1 - 0.16) / (0.45 - 0.16))
        # 真泄漏可能宽频，也可能集中在某些超声频段，因此 band_entropy 不作为强门槛，只给温和加分。
        s_band_energy_shape = clamp(0.70 * band_entropy + 0.30 * band_flat)

        band_consistency = clamp(
            0.38 * same_best +
            0.27 * mode_ratio +
            0.25 * s_band_top1 +
            0.10 * s_band_energy_shape
        )

        # 5) 弥散惩罚
        far_ratio = float(r.get("far_to_total_ratio", 0.0) or 0.0)
        active_norm = clamp((active25 - 2.0) / 6.0)
        diffuse_penalty = clamp(
            0.30 * ent +
            0.24 * active_norm +
            0.22 * far_ratio +
            0.14 * (1.0 - direction_score) +
            0.10 * (1.0 - decay_score)
        )

        # 6) 总泄漏证据
        positive = clamp(
            0.28 * scene_ultra +
            0.25 * direction_score +
            0.24 * decay_score +
            0.15 * band_consistency +
            0.08 * dq
        )

        # 弥散噪声越强，总分越低。
        leak_score = clamp(positive * (1.0 - 0.48 * diffuse_penalty) * (0.60 + 0.40 * dq))

        # 判定逻辑：TRUE 必须满足多个物理条件，避免其他工厂全部判 TRUE。
        reasons = []
        if scene_ultra >= 0.52:
            reasons.append("场景内超声异常较突出")
        else:
            reasons.append("场景内超声异常不突出")

        if direction_score >= 0.50:
            reasons.append("方向较集中")
        else:
            reasons.append("方向集中性不足")

        if decay_score >= 0.42:
            reasons.append("距离衰减较合理")
        else:
            reasons.append("距离衰减证据不足")

        if band_consistency >= 0.48:
            reasons.append("多频段方向较一致")
        else:
            reasons.append("多频段一致性不足")

        if diffuse_penalty >= 0.68:
            reasons.append("弥散噪声惩罚较高")
        else:
            reasons.append("弥散惩罚可接受")

        # INVALID 已在前面处理
        strong_true_conditions = (
            leak_score >= TRUE_SCORE_THRESHOLD and
            dq >= MIN_TRUE_DATA_QUALITY and
            scene_ultra >= 0.48 and
            direction_score >= 0.45 and
            decay_score >= 0.36 and
            band_consistency >= 0.38 and
            diffuse_penalty <= 0.66
        )

        strong_false_conditions = (
            leak_score <= FALSE_SCORE_THRESHOLD or
            diffuse_penalty >= 0.76 or
            (direction_score < 0.30 and decay_score < 0.30) or
            (scene_ultra < 0.25 and direction_score < 0.38)
        )

        if FORCE_BINARY_OUTPUT:
            pred = "TRUE_LEAK" if leak_score >= FORCE_BINARY_THRESHOLD else "FALSE_LEAK"
        else:
            if strong_true_conditions:
                pred = "TRUE_LEAK"
            elif strong_false_conditions:
                pred = "FALSE_LEAK"
            else:
                pred = "SUSPECT"

        pred_binary = "TRUE_LEAK" if leak_score >= FORCE_BINARY_THRESHOLD else "FALSE_LEAK"

        scores.append({
            "scene_ultrasound_excess_score": scene_ultra,
            "direction_score": direction_score,
            "decay_score": decay_score,
            "band_consistency_score": band_consistency,
            "diffuse_penalty": diffuse_penalty,
            "positive_evidence_score": positive,
            "leak_evidence_score": leak_score,
            "pred_label": pred,
            "pred_label_binary": pred_binary,
            "decision_reason": "；".join(reasons),
        })

    score_df = pd.DataFrame(scores, index=df.index)
    out = pd.concat([df, score_df], axis=1)
    return out


# ============================================================
# 7. 可选标签评估
# ============================================================

def normalize_label(x: Any) -> Optional[str]:
    s = str(x).strip().upper()
    if s in {"TRUE_LEAK", "TRUE", "T", "1", "LEAK", "Y", "YES"}:
        return "TRUE_LEAK"
    if s in {"FALSE_LEAK", "FALSE", "F", "0", "NO_LEAK", "NO", "N", "CS"}:
        return "FALSE_LEAK"
    return None


def attach_labels_and_evaluate(pred: pd.DataFrame, label_csv: str, output_dir: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if not label_csv or not os.path.exists(label_csv):
        return pred, None

    lab = pd.read_csv(label_csv)
    cols_lower = {c.lower(): c for c in lab.columns}

    if "label" not in cols_lower:
        print("[警告] label_csv 中没有 label 列，跳过评估。")
        return pred, None

    label_col = cols_lower["label"]

    if "time" not in cols_lower:
        print("[警告] label_csv 中没有 time 列，跳过评估。")
        return pred, None

    time_col = cols_lower["time"]

    center_col = None
    for name in ["center_norm", "center", "center_id"]:
        if name in cols_lower:
            center_col = cols_lower[name]
            break

    if center_col is None:
        print("[警告] label_csv 中没有 center/center_norm 列，跳过评估。")
        return pred, None

    lab2 = lab[[time_col, center_col, label_col]].copy()
    lab2.columns = ["time", "center_norm", "true_label"]
    lab2["center_norm"] = lab2["center_norm"].apply(normalize_center_id)
    lab2["true_label"] = lab2["true_label"].apply(normalize_label)
    lab2 = lab2.dropna(subset=["true_label"])

    out = pred.merge(lab2, on=["time", "center_norm"], how="left")

    eval_rows = []
    valid = out[out["true_label"].isin(["TRUE_LEAK", "FALSE_LEAK"])].copy()

    if len(valid) > 0:
        # 二分类硬输出评估
        valid["correct_binary"] = valid["pred_label_binary"] == valid["true_label"]

        # 三分类中，SUSPECT/INVALID 不计为正确，但另外统计覆盖率
        valid["is_decisive"] = valid["pred_label"].isin(["TRUE_LEAK", "FALSE_LEAK"])
        valid["correct_decisive_as_wrong_if_suspect"] = valid["pred_label"] == valid["true_label"]

        for t, g in valid.groupby("time"):
            decisive = g[g["is_decisive"]]
            eval_rows.append({
                "time": t,
                "n_labeled": len(g),
                "binary_acc": float(g["correct_binary"].mean()),
                "decisive_coverage": float(g["is_decisive"].mean()),
                "decisive_acc_on_decisive_only": float((decisive["pred_label"] == decisive["true_label"]).mean()) if len(decisive) else np.nan,
                "three_state_acc_suspect_as_wrong": float(g["correct_decisive_as_wrong_if_suspect"].mean()),
                "n_TRUE": int((g["true_label"] == "TRUE_LEAK").sum()),
                "n_FALSE": int((g["true_label"] == "FALSE_LEAK").sum()),
                "n_pred_TRUE": int((g["pred_label"] == "TRUE_LEAK").sum()),
                "n_pred_FALSE": int((g["pred_label"] == "FALSE_LEAK").sum()),
                "n_suspect_or_invalid": int((~g["is_decisive"]).sum()),
            })

        # overall
        g = valid
        decisive = g[g["is_decisive"]]
        eval_rows.append({
            "time": "OVERALL",
            "n_labeled": len(g),
            "binary_acc": float(g["correct_binary"].mean()),
            "decisive_coverage": float(g["is_decisive"].mean()),
            "decisive_acc_on_decisive_only": float((decisive["pred_label"] == decisive["true_label"]).mean()) if len(decisive) else np.nan,
            "three_state_acc_suspect_as_wrong": float(g["correct_decisive_as_wrong_if_suspect"].mean()),
            "n_TRUE": int((g["true_label"] == "TRUE_LEAK").sum()),
            "n_FALSE": int((g["true_label"] == "FALSE_LEAK").sum()),
            "n_pred_TRUE": int((g["pred_label"] == "TRUE_LEAK").sum()),
            "n_pred_FALSE": int((g["pred_label"] == "FALSE_LEAK").sum()),
            "n_suspect_or_invalid": int((~g["is_decisive"]).sum()),
        })

    eval_df = pd.DataFrame(eval_rows)
    eval_path = os.path.join(output_dir, "v10_label_evaluation.csv")
    eval_df.to_csv(eval_path, index=False, encoding="utf-8-sig")
    return out, eval_df


# ============================================================
# 8. 输出汇总
# ============================================================

def create_scene_summary(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t, g in pred.groupby("time"):
        rows.append({
            "time": t,
            "n_centers": len(g),
            "n_TRUE_LEAK": int((g["pred_label"] == "TRUE_LEAK").sum()),
            "n_FALSE_LEAK": int((g["pred_label"] == "FALSE_LEAK").sum()),
            "n_SUSPECT": int((g["pred_label"] == "SUSPECT").sum()),
            "n_INVALID_DATA": int((g["pred_label"] == "INVALID_DATA").sum()),
            "true_ratio_decisive": safe_ratio(
                int((g["pred_label"] == "TRUE_LEAK").sum()),
                int(g["pred_label"].isin(["TRUE_LEAK", "FALSE_LEAK"]).sum())
            ),
            "mean_leak_evidence_score": float(g["leak_evidence_score"].mean()),
            "median_leak_evidence_score": float(g["leak_evidence_score"].median()),
            "max_leak_evidence_score": float(g["leak_evidence_score"].max()),
            "mean_scene_ultrasound_excess_score": float(g["scene_ultrasound_excess_score"].mean()),
            "mean_direction_score": float(g["direction_score"].mean()),
            "mean_decay_score": float(g["decay_score"].mean()),
            "mean_diffuse_penalty": float(g["diffuse_penalty"].mean()),
            "mean_data_quality_score": float(g["data_quality_score"].mean()),
            "min_offset_combo_count": int(g["offset_combo_count"].min()),
            "median_offset_combo_count": float(g["offset_combo_count"].median()),
        })
    return pd.DataFrame(rows)


def write_outputs(pred: pd.DataFrame, output_dir: str, label_eval: Optional[pd.DataFrame] = None) -> None:
    ensure_dir(output_dir)

    front_cols = [
        "time", "center_norm", "center", "pred_label", "pred_label_binary",
        "leak_evidence_score", "scene_ultrasound_excess_score", "direction_score",
        "decay_score", "band_consistency_score", "diffuse_penalty",
        "data_quality_score", "offset_combo_count", "offset_wav_count_used",
        "best_direction", "decision_reason",
    ]
    if "true_label" in pred.columns:
        front_cols.insert(3, "true_label")

    cols = [c for c in front_cols if c in pred.columns] + [c for c in pred.columns if c not in front_cols]
    pred_out = pred[cols].copy()

    pred_path = os.path.join(output_dir, "v10_per_center_predictions.csv")
    pred_out.to_csv(pred_path, index=False, encoding="utf-8-sig")

    feature_path = os.path.join(output_dir, "v10_feature_matrix.csv")
    pred.to_csv(feature_path, index=False, encoding="utf-8-sig")

    summary = create_scene_summary(pred)
    summary_path = os.path.join(output_dir, "v10_scene_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    sus = pred[pred["pred_label"].isin(["SUSPECT", "INVALID_DATA"])].copy()
    sus_path = os.path.join(output_dir, "v10_suspect_centers.csv")
    sus.to_csv(sus_path, index=False, encoding="utf-8-sig")

    top_path = os.path.join(output_dir, "v10_top_evidence_centers.csv")
    pred.sort_values("leak_evidence_score", ascending=False).head(50).to_csv(top_path, index=False, encoding="utf-8-sig")

    report_path = os.path.join(output_dir, "v10_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("V10 场景自标定物理证据泄漏检测报告\n")
        f.write("=" * 100 + "\n")
        f.write(f"生成时间: {datetime.now()}\n")
        f.write(f"TRUE_SCORE_THRESHOLD: {TRUE_SCORE_THRESHOLD}\n")
        f.write(f"FALSE_SCORE_THRESHOLD: {FALSE_SCORE_THRESHOLD}\n")
        f.write(f"FORCE_BINARY_OUTPUT: {FORCE_BINARY_OUTPUT}\n")
        f.write(f"FREQ_LOW: {FREQ_LOW}\n")
        f.write(f"FREQ_HIGH: {FREQ_HIGH}\n")
        f.write(f"SUBBANDS: {SUBBANDS}\n")
        f.write("\n核心说明:\n")
        f.write("  本程序不使用 v7 模型，不使用 144226 pairwise 规则。\n")
        f.write("  它基于当前场景内部统计量进行无标签自标定，并根据方向集中性、距离衰减、多频段一致性和弥散惩罚逐 center 判断。\n")
        f.write("  SUSPECT 不是错误，而是工业场景下必要的拒判结果。\n")
        f.write("\n输出文件:\n")
        f.write(f"  每center预测: {pred_path}\n")
        f.write(f"  场景汇总: {summary_path}\n")
        f.write(f"  SUSPECT/INVALID: {sus_path}\n")
        f.write(f"  完整特征: {feature_path}\n")
        if label_eval is not None:
            f.write(f"  标签评估: {os.path.join(output_dir, 'v10_label_evaluation.csv')}\n")
        f.write("\n场景汇总:\n")
        f.write(summary.to_string(index=False))
        f.write("\n")
        if label_eval is not None and len(label_eval):
            f.write("\n标签评估:\n")
            f.write(label_eval.to_string(index=False))
            f.write("\n")

    print("\n" + "=" * 100)
    print("V10 完成")
    print("=" * 100)
    print("每center预测:", pred_path)
    print("场景汇总:", summary_path)
    print("SUSPECT/INVALID:", sus_path)
    print("完整特征:", feature_path)
    print("报告:", report_path)
    if label_eval is not None:
        print("标签评估:", os.path.join(output_dir, "v10_label_evaluation.csv"))

    print("\n场景汇总:")
    print(summary.to_string(index=False))

    print("\n得分最高的前20个center:")
    show_cols = [
        "time", "center_norm", "pred_label", "leak_evidence_score",
        "scene_ultrasound_excess_score", "direction_score", "decay_score",
        "band_consistency_score", "diffuse_penalty", "data_quality_score", "best_direction",
    ]
    show_cols = [c for c in show_cols if c in pred.columns]
    print(pred.sort_values("leak_evidence_score", ascending=False)[show_cols].head(20).to_string(index=False))


# ============================================================
# 9. 主流程
# ============================================================

def run_v10(offset_root: str, center_root: str, output_dir: str, label_csv: str = "") -> pd.DataFrame:
    ensure_dir(output_dir)

    raw_df = extract_dataset_features(offset_root, center_root)
    raw_path = os.path.join(output_dir, "v10_raw_center_features_before_scoring.csv")
    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")

    rel_df = add_scene_relative_columns(raw_df)
    pred = compute_evidence_scores(rel_df)

    label_eval = None
    if label_csv:
        pred, label_eval = attach_labels_and_evaluate(pred, label_csv, output_dir)

    write_outputs(pred, output_dir, label_eval=label_eval)
    return pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V10 场景自标定物理证据泄漏检测器")
    parser.add_argument("--offset_root", type=str, default=INPUT_OFFSET_ROOT, help="新工厂/新场景 offset wav 根目录")
    parser.add_argument("--center_root", type=str, default=INPUT_CENTER_ROOT, help="可选 center wav 根目录")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--label_csv", type=str, default=LABEL_CSV, help="可选标签CSV路径，用于评估")
    parser.add_argument("--force_binary", action="store_true", help="强制输出 TRUE/FALSE，不输出 SUSPECT。工业应用不推荐。")
    parser.add_argument("--true_threshold", type=float, default=TRUE_SCORE_THRESHOLD, help="TRUE 判定阈值")
    parser.add_argument("--false_threshold", type=float, default=FALSE_SCORE_THRESHOLD, help="FALSE 判定阈值")
    return parser.parse_args()


def main() -> None:
    global FORCE_BINARY_OUTPUT, TRUE_SCORE_THRESHOLD, FALSE_SCORE_THRESHOLD

    args = parse_args()
    FORCE_BINARY_OUTPUT = bool(args.force_binary)
    TRUE_SCORE_THRESHOLD = float(args.true_threshold)
    FALSE_SCORE_THRESHOLD = float(args.false_threshold)

    print("=" * 100)
    print("V10 场景自标定物理证据泄漏检测器")
    print("=" * 100)
    print("offset_root:", args.offset_root)
    print("center_root:", args.center_root if args.center_root else "(未提供)")
    print("output_dir:", args.output_dir)
    print("label_csv:", args.label_csv if args.label_csv else "(未提供)")
    print("FORCE_BINARY_OUTPUT:", FORCE_BINARY_OUTPUT)

    run_v10(
        offset_root=args.offset_root,
        center_root=args.center_root,
        output_dir=args.output_dir,
        label_csv=args.label_csv,
    )


if __name__ == "__main__":
    main()
