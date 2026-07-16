# -*- coding: utf-8 -*-
"""
v9_ABCD_raw_plane.py

v9-ABCD：四场景 × raw/plane 双方法残差输出（固定50-70 kHz）
================================================================

适配当前目录结构：
    center_root_dir / time_folder / *_<center>_beamform_result.wav
    offset_root_dir / time_folder / *_<center>d<distance>_<direction>*.wav

支持的方向：
    up, down, left, right,
    up_left, down_left, up_right, down_right

核心逻辑：
    1. 中心点和周围偏移点使用完全相同的 STFT 参数，只保留50-70 kHz；
    2. 自动排除靠近中心的一部分点，优先用外圈点估计环境背景；
    3. 计算鲁棒加权空间平面背景；中位数只作为内部安全基准；
    4. raw保留中心原始功率，plane保存中心减空间平面背景后的正残差；
    5. 输出中心相对背景的正残差、宽带程度、时间持续性和空间局部性；
    6. 不在第一阶段强行训练复杂分类器；有多个 scene 时仅做整场景留出单特征检查。

安装：
    pip install numpy pandas scipy matplotlib scikit-learn

运行：
    1. 先修改下方 DATASETS。
    2. 自检：
       python leak_v9_local_background_residual_wav.py --self-test
    3. 正式运行：
       python leak_v9_local_background_residual_wav.py
    4. 不画诊断图：
       python leak_v9_local_background_residual_wav.py --no-plots
    5. 保存完整残差矩阵：
       python leak_v9_local_background_residual_wav.py --save-residual-npz

重要说明：
    - 这里做的是“波束图局部背景消除”，不是把 40 路独立麦克风波形直接相减。
    - 程序支持任意数量的周围点，不要求必须正好 40 个；你现有 8方向×8距离=64点也可直接使用。
    - WAV 文件之间必须保留可比较的幅值标度。程序不会对每个文件单独峰值归一化。
    - 本版本固定只计算50-70 kHz；其他频率在读取STFT后直接丢弃。
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import traceback
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    from scipy import signal
    from scipy.io import wavfile
    from scipy.spatial import Delaunay, QhullError
    from scipy.stats import mannwhitneyu, spearmanr
except Exception as exc:
    raise RuntimeError("缺少 scipy，请先运行: pip install scipy") from exc


# ============================================================================
# 1. 只需要优先修改这里
# ============================================================================

# scene 表示真正希望跨越的工厂/声学场景。
# 同一工厂的 TRUE/FALSE 数据应填写相同 scene，不要把每个 time 当成 scene。
DATASETS: List[Dict[str, Any]] = [
    {
        "name": "factory_A_true",
        "scene": "factory_A",
        "label": "TRUE_LEAK",
        "center_root_dir": r"D:\gas\beamform_results",
        "offset_root_dir": r"D:\gas\beamform_results_offset_multiple",
        # 空列表表示自动读取 center_root 和 offset_root 共有的全部子文件夹。
        "time_folders": [
            "HM20260626_142938.ld",
            "HM20260626_143034.ld",
            "HM20260626_144226.ld",
            "HM20260626_144325.ld",
        ],
    },

    # 假泄漏数据示例。请取消注释并修改成你的实际路径。
    # {
    #     "name": "factory_A_false",
    #     "scene": "factory_A",
    #     "label": "FALSE_LEAK",
    #     "center_root_dir": r"D:\gas_false\beamform_results",
    #     "offset_root_dir": r"D:\gas_false\beamform_results_offset_multiple",
    #     "time_folders": [],
    # },

    # 新工厂示例。跨场景验证至少需要多个 scene，并且每个测试 scene 最好同时有真假标签。
    # {
    #     "name": "factory_B_true",
    #     "scene": "factory_B",
    #     "label": "TRUE_LEAK",
    #     "center_root_dir": r"D:\factory_B_true\beamform_results",
    #     "offset_root_dir": r"D:\factory_B_true\beamform_results_offset_multiple",
    #     "time_folders": [],
    # },
]

OUTPUT_DIR = r"C:\Users\jiangxinru6\Desktop\wurenji\leak_v9_local_background_results"

# 调试时可设成 1、3、5；正式运行设为 None。
MAX_CENTERS_PER_TIME: Optional[int] = None


# ============================================================================
# 2. 算法配置
# ============================================================================


@dataclass(frozen=True)
class V9Config:
    # 分析频带。192 kHz 采样率下最高有效频率小于 96 kHz。
    # 固定只分析 50-70 kHz；该频带以外的数据在 STFT 后立即丢弃，不参与任何特征、拟合或残差计算。
    freq_low_hz: float = 50_000.0
    freq_high_hz: float = 70_000.0

    # STFT。4096 点在 192 kHz 下频率分辨率约 46.9 Hz。
    nperseg: int = 4096
    hop_length: int = 2048
    nfft: int = 4096
    min_frames: int = 2
    min_frequency_bins: int = 30

    # 背景点选择：排除最近 35% 邻居，优先使用外圈。
    background_min_distance_quantile: float = 0.35
    min_background_points: int = 12

    # 鲁棒空间平面。
    point_huber_delta: float = 1.5
    plane_improvement_ratio: float = 0.95
    plane_clip_margin_db: float = 3.0
    max_geometry_condition: float = 1.0e6

    # 残差描述阈值，不作为最终工业阈值。
    residual_thresholds_db: Tuple[float, ...] = (1.0, 3.0, 6.0)

    # 质量要求。
    minimum_required_neighbors: int = 12
    minimum_complete_combo_ratio: float = 0.30
    epsilon_power: float = 1.0e-20

    # 诊断输出。
    save_diagnostic_plots: bool = True
    diagnostic_plot_limit: int = 30
    save_residual_npz: bool = False

    # 跨场景检查。
    group_column: str = "scene"
    max_threshold_candidates: int = 201

    random_state: int = 42


DIRECTIONS = [
    "up_left",
    "down_left",
    "up_right",
    "down_right",
    "up",
    "down",
    "left",
    "right",
]

DIRECTION_ANGLES = {
    "right": 0.0,
    "up_right": math.pi / 4.0,
    "up": math.pi / 2.0,
    "up_left": 3.0 * math.pi / 4.0,
    "left": math.pi,
    "down_left": -3.0 * math.pi / 4.0,
    "down": -math.pi / 2.0,
    "down_right": -math.pi / 4.0,
}

VALID_LABELS = {"TRUE_LEAK", "FALSE_LEAK", ""}

# 文件名示例：xxx_01_beamform_result.wav
CENTER_ID_RE = re.compile(r"_(?P<center>\d+)_beamform_result\.wav$", re.IGNORECASE)

# 文件名示例：xxx_01d40_up_left_xxx.wav
OFFSET_RE = re.compile(
    r"_(?P<center>\d+)d(?P<distance>\d+(?:\.\d+)?)_"
    r"(?P<direction>up_left|down_left|up_right|down_right|up|down|left|right)"
    r"(?:_|\.|$)",
    re.IGNORECASE,
)

FEATURE_PREFIXES = ("residual_", "spectral_", "temporal_", "spatial_")
QUALITY_PREFIXES = ("qc_", "fit_")


# ============================================================================
# 3. 通用工具
# ============================================================================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_slug(value: Any, max_len: int = 150) -> str:
    text = str(value).strip() or "sample"
    text = re.sub(r"[^0-9A-Za-z_\-.]+", "_", text)
    return text.strip("._")[:max_len] or "sample"


def normalize_center_id(value: Any) -> str:
    text = str(value).strip()
    if text.isdigit():
        return str(int(text))
    return text


def db_from_power(power: np.ndarray, eps: float) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(power, dtype=float), eps))


def power_from_db(power_db: np.ndarray) -> np.ndarray:
    return np.power(10.0, np.clip(np.asarray(power_db, dtype=float), -300.0, 300.0) / 10.0)


def safe_ratio(a: float, b: float, eps: float = 1.0e-20) -> float:
    return float(a / (b + eps))


def robust_scale(values: np.ndarray, floor: float = 1.0e-6) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return floor
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return max(1.4826 * mad, floor)


def longest_true_run(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool).ravel()
    if mask.size == 0:
        return 0
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return int(np.max(ends - starts)) if starts.size else 0


def normalized_entropy(weights: np.ndarray, eps: float = 1.0e-20) -> float:
    w = np.maximum(np.asarray(weights, dtype=float), 0.0)
    if w.size <= 1 or float(np.sum(w)) <= eps:
        return 0.0
    p = w / (np.sum(w) + eps)
    return float(-np.sum(p * np.log(p + eps)) / np.log(w.size))


def spectral_flatness(weights: np.ndarray, eps: float = 1.0e-20) -> float:
    w = np.maximum(np.asarray(weights, dtype=float), 0.0)
    arithmetic = float(np.mean(w))
    if arithmetic <= eps:
        return 0.0
    geometric = float(np.exp(np.mean(np.log(w + eps))))
    return float(geometric / (arithmetic + eps))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    recalls: List[float] = []
    for cls in (0, 1):
        mask = y_true == cls
        if np.any(mask):
            recalls.append(float(np.mean(y_pred[mask] == cls)))
    return float(np.mean(recalls)) if recalls else np.nan


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        y_true = np.asarray(y_true, dtype=int)
        score = np.asarray(score, dtype=float)
        valid = np.isfinite(score)
        if np.sum(valid) < 2 or np.unique(y_true[valid]).size < 2:
            return np.nan
        return float(roc_auc_score(y_true[valid], score[valid]))
    except Exception:
        return np.nan


def point_inside_hull(point: np.ndarray, xy: np.ndarray) -> bool:
    try:
        if len(xy) < 3:
            return False
        tri = Delaunay(np.asarray(xy, dtype=float))
        return bool(tri.find_simplex(np.asarray(point, dtype=float).reshape(1, 2))[0] >= 0)
    except (QhullError, ValueError):
        return False


# ============================================================================
# 4. WAV 与 STFT
# ============================================================================


def read_wav_preserve_scale(path: Path) -> Tuple[int, np.ndarray, Dict[str, float]]:
    fs, raw = wavfile.read(str(path))
    original_dtype = raw.dtype

    if raw.ndim > 1:
        raw = np.mean(raw.astype(np.float64), axis=1)
    elif np.issubdtype(raw.dtype, np.integer):
        raw = raw.astype(np.float64)
    else:
        raw = raw.astype(np.float64, copy=False)

    # 整数 WAV 按数据类型的固定满量程换算，不能按每个文件自身峰值归一化。
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        scale = float(max(abs(info.min), abs(info.max)))
        raw = raw / max(scale, 1.0)

    finite = np.isfinite(raw)
    finite_ratio = float(np.mean(finite)) if raw.size else 0.0
    if not np.any(finite):
        raise ValueError(f"WAV 全部无效: {path}")
    fill = float(np.median(raw[finite]))
    raw = np.where(finite, raw, fill)
    raw = raw - float(np.mean(raw))

    quality = {
        "finite_ratio": finite_ratio,
        "peak_abs": float(np.max(np.abs(raw))) if raw.size else 0.0,
        "rms": float(np.sqrt(np.mean(raw ** 2))) if raw.size else 0.0,
        "n_samples": int(raw.size),
    }
    return int(fs), raw.astype(np.float64, copy=False), quality


def wav_to_stft_power(path: Path, config: V9Config) -> Dict[str, Any]:
    fs, x, quality = read_wav_preserve_scale(path)

    if x.size < 256:
        raise ValueError(f"WAV 太短: {path}, n={x.size}")

    nperseg = min(int(config.nperseg), int(x.size))
    if nperseg < 256:
        raise ValueError(f"有效 nperseg 太小: {nperseg}")

    hop = min(int(config.hop_length), nperseg)
    noverlap = max(0, nperseg - hop)
    nfft = max(int(config.nfft), nperseg)

    freq_hz, time_s, z = signal.stft(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=False,
        return_onesided=True,
        boundary=None,
        padded=False,
    )

    mask = (freq_hz >= config.freq_low_hz) & (freq_hz <= config.freq_high_hz)
    if np.sum(mask) < config.min_frequency_bins:
        raise ValueError(
            f"{path.name} 在 {config.freq_low_hz:g}-{config.freq_high_hz:g} Hz "
            f"只有 {int(np.sum(mask))} 个频点；采样率={fs}"
        )

    power = np.abs(z[mask]) ** 2
    if power.shape[1] < config.min_frames:
        # 单个很短文件仍允许处理，但明确标记质量。
        pass

    return {
        "fs": fs,
        "freq_hz": freq_hz[mask].astype(float),
        "time_s": time_s.astype(float),
        "power": np.maximum(power, config.epsilon_power).astype(np.float64),
        "quality": quality,
    }


def average_duplicate_wavs(paths: Sequence[Path], config: V9Config) -> Dict[str, Any]:
    results = [wav_to_stft_power(p, config) for p in paths]
    if not results:
        raise ValueError("没有 WAV 可平均")

    fs_values = {int(r["fs"]) for r in results}
    if len(fs_values) != 1:
        raise ValueError(f"重复 WAV 采样率不一致: {sorted(fs_values)}")

    ref_freq = results[0]["freq_hz"]
    for r in results[1:]:
        if r["freq_hz"].shape != ref_freq.shape or not np.allclose(r["freq_hz"], ref_freq):
            raise ValueError("重复 WAV 的频率轴不一致")

    n_frames = min(r["power"].shape[1] for r in results)
    stack = np.stack([r["power"][:, :n_frames] for r in results], axis=0)
    power = np.mean(stack, axis=0)

    return {
        "fs": results[0]["fs"],
        "freq_hz": ref_freq,
        "time_s": results[0]["time_s"][:n_frames],
        "power": power,
        "quality": {
            "finite_ratio": float(np.mean([r["quality"]["finite_ratio"] for r in results])),
            "peak_abs": float(np.mean([r["quality"]["peak_abs"] for r in results])),
            "rms": float(np.mean([r["quality"]["rms"] for r in results])),
            "n_samples": int(min(r["quality"]["n_samples"] for r in results)),
            "n_duplicate_files": int(len(results)),
        },
    }


# ============================================================================
# 5. 文件发现与坐标构造
# ============================================================================


def detect_center_files(center_dir: Path) -> Dict[str, List[Path]]:
    mapping: Dict[str, List[Path]] = {}
    for raw in glob.glob(str(center_dir / "*_beamform_result.wav")):
        path = Path(raw)
        match = CENTER_ID_RE.search(path.name)
        if not match:
            continue
        center = normalize_center_id(match.group("center"))
        mapping.setdefault(center, []).append(path)
    for paths in mapping.values():
        paths.sort()
    return mapping


def parse_offset_files(offset_dir: Path) -> Dict[Tuple[str, str, float], List[Path]]:
    mapping: Dict[Tuple[str, str, float], List[Path]] = {}
    for raw in glob.glob(str(offset_dir / "*.wav")):
        path = Path(raw)
        match = OFFSET_RE.search(path.name)
        if not match:
            continue
        center = normalize_center_id(match.group("center"))
        direction = match.group("direction").lower()
        distance = float(match.group("distance"))
        key = (center, direction, distance)
        mapping.setdefault(key, []).append(path)
    for paths in mapping.values():
        paths.sort()
    return mapping


def coordinate_from_direction(direction: str, distance_cm: float) -> Tuple[float, float]:
    angle = DIRECTION_ANGLES[direction]
    return float(distance_cm * math.cos(angle)), float(distance_cm * math.sin(angle))


def common_time_folders(center_root: Path, offset_root: Path) -> List[str]:
    if not center_root.exists() or not offset_root.exists():
        return []
    a = {p.name for p in center_root.iterdir() if p.is_dir()}
    b = {p.name for p in offset_root.iterdir() if p.is_dir()}
    return sorted(a & b)


def load_one_center_cube(
    center_paths: Sequence[Path],
    offset_mapping: Dict[Tuple[str, str, float], List[Path]],
    center_id: str,
    config: V9Config,
) -> Dict[str, Any]:
    center_result = average_duplicate_wavs(center_paths, config)

    offset_items: List[Tuple[str, float, List[Path]]] = []
    for (cid, direction, distance), paths in offset_mapping.items():
        if cid == center_id and direction in DIRECTION_ANGLES:
            offset_items.append((direction, float(distance), paths))
    offset_items.sort(key=lambda x: (x[1], x[0]))

    if len(offset_items) < config.minimum_required_neighbors:
        raise ValueError(
            f"center={center_id} 只找到 {len(offset_items)} 个周围点，"
            f"至少需要 {config.minimum_required_neighbors} 个"
        )

    point_results: List[Dict[str, Any]] = [center_result]
    xy: List[Tuple[float, float]] = [(0.0, 0.0)]
    point_ids: List[str] = [f"center_{center_id}"]
    failed_offsets: List[Dict[str, str]] = []

    for direction, distance, paths in offset_items:
        try:
            result = average_duplicate_wavs(paths, config)
            if int(result["fs"]) != int(center_result["fs"]):
                raise ValueError(
                    f"采样率 {result['fs']} 与中心 {center_result['fs']} 不一致"
                )
            if result["freq_hz"].shape != center_result["freq_hz"].shape or not np.allclose(
                result["freq_hz"], center_result["freq_hz"]
            ):
                raise ValueError("频率轴与中心不一致")
            point_results.append(result)
            xy.append(coordinate_from_direction(direction, distance))
            point_ids.append(f"{distance:g}cm_{direction}")
        except Exception as exc:
            failed_offsets.append(
                {
                    "point": f"{distance:g}cm_{direction}",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if len(point_results) - 1 < config.minimum_required_neighbors:
        raise ValueError(
            f"center={center_id} 成功读取的周围点只有 {len(point_results)-1} 个"
        )

    # 全部点截取到共同帧数，保证同一时频单元可比较。
    n_frames = min(r["power"].shape[1] for r in point_results)
    power_cube = np.stack([r["power"][:, :n_frames] for r in point_results], axis=0)
    peaks = np.array([r["quality"]["peak_abs"] for r in point_results], dtype=float)
    rms_values = np.array([r["quality"]["rms"] for r in point_results], dtype=float)

    return {
        "power": power_cube,
        "freq_hz": center_result["freq_hz"],
        "time_s": center_result["time_s"][:n_frames],
        "xy": np.asarray(xy, dtype=float),
        "point_ids": np.asarray(point_ids, dtype=str),
        "center_index": 0,
        "fs": int(center_result["fs"]),
        "peak_abs": peaks,
        "rms": rms_values,
        "failed_offsets": failed_offsets,
        "n_offset_discovered": int(len(offset_items)),
    }


# ============================================================================
# 6. 局部背景估计
# ============================================================================


def select_background_indices(
    xy: np.ndarray,
    center_index: int,
    config: V9Config,
    quantile: Optional[float] = None,
) -> Dict[str, Any]:
    quantile = (
        config.background_min_distance_quantile if quantile is None else float(quantile)
    )
    distances = np.linalg.norm(xy - xy[center_index], axis=1)
    neighbors = np.array([i for i in range(len(xy)) if i != center_index], dtype=int)
    neighbor_dist = distances[neighbors]

    if neighbors.size < config.min_background_points:
        raise ValueError(
            f"邻居只有 {neighbors.size} 个，少于 min_background_points="
            f"{config.min_background_points}"
        )

    positive = neighbor_dist[neighbor_dist > 0]
    if positive.size < config.min_background_points:
        raise ValueError("过多周围点与中心坐标重合")

    cutoff = float(np.quantile(positive, quantile))
    bg = neighbors[neighbor_dist >= cutoff]
    if bg.size < config.min_background_points:
        order = np.argsort(neighbor_dist)[::-1]
        bg = neighbors[order[: config.min_background_points]]

    bg = np.unique(bg)
    inner = np.array([i for i in neighbors if i not in set(bg.tolist())], dtype=int)
    return {
        "distances": distances,
        "neighbors": neighbors,
        "background": bg,
        "inner": inner,
        "distance_cutoff": cutoff,
        "quantile": quantile,
    }


def robust_point_weights(y: np.ndarray, delta: float) -> np.ndarray:
    """
    y: (n_background_points, n_time_frequency_bins), dB。
    权重按每个空间点相对邻域中位数的整体偏离程度计算，避免少数局部声源污染背景。
    """
    median_bin = np.median(y, axis=0, keepdims=True)
    point_bias = np.median(y - median_bin, axis=1)
    point_abs_dev = np.median(np.abs(y - median_bin), axis=1)

    score = np.maximum(point_bias, 0.0) + 0.5 * point_abs_dev
    center = float(np.median(score))
    scale = robust_scale(score, floor=0.25)
    u = np.abs(score - center) / max(delta * scale, 1.0e-12)
    weights = np.ones_like(u)
    mask = u > 1.0
    weights[mask] = 1.0 / u[mask]
    return np.clip(weights, 0.05, 1.0)


def estimate_local_background(
    power_cube: np.ndarray,
    xy: np.ndarray,
    center_index: int,
    config: V9Config,
) -> Dict[str, Any]:
    selection = select_background_indices(xy, center_index, config)
    bg_idx = selection["background"]

    power_db = db_from_power(power_cube, config.epsilon_power)
    bg_db = power_db[bg_idx]
    n_bg, n_freq, n_frames = bg_db.shape
    y = bg_db.reshape(n_bg, -1)

    median_flat = np.median(y, axis=0)
    median_prediction = median_flat.reshape(n_freq, n_frames)

    bg_xy = xy[bg_idx] - xy[center_index]
    max_radius = float(np.max(np.linalg.norm(bg_xy, axis=1)))
    scale_xy = max(max_radius, 1.0)
    xy_scaled = bg_xy / scale_xy
    X = np.column_stack([np.ones(n_bg), xy_scaled[:, 0], xy_scaled[:, 1]])

    weights = robust_point_weights(y, config.point_huber_delta)
    XtWX = X.T @ (weights[:, None] * X)
    condition = float(np.linalg.cond(XtWX))
    inside_hull = point_inside_hull(np.array([0.0, 0.0]), bg_xy)

    geometry_ok = bool(
        np.isfinite(condition)
        and condition <= config.max_geometry_condition
        and np.linalg.matrix_rank(X) >= 3
        and inside_hull
    )

    plane_prediction = median_prediction.copy()
    plane_mae = np.inf
    median_mae = float(np.median(np.abs(y - median_flat[None, :])))

    if geometry_ok:
        try:
            beta = np.linalg.solve(XtWX, X.T @ (weights[:, None] * y))
            plane_flat = beta[0]
            fitted = X @ beta
            plane_mae = float(np.median(np.abs(y - fitted)))

            lower = median_flat - config.plane_clip_margin_db
            upper = median_flat + config.plane_clip_margin_db
            plane_flat = np.clip(plane_flat, lower, upper)
            plane_prediction = plane_flat.reshape(n_freq, n_frames)
        except np.linalg.LinAlgError:
            geometry_ok = False

    use_plane = bool(
        geometry_ok
        and np.isfinite(plane_mae)
        and plane_mae < config.plane_improvement_ratio * max(median_mae, 1.0e-12)
    )

    chosen_db = plane_prediction if use_plane else median_prediction
    chosen_power = power_from_db(chosen_db)

    return {
        "background_power": chosen_power,
        "background_db": chosen_db,
        "median_background_db": median_prediction,
        "plane_background_db": plane_prediction,
        "selection": selection,
        "point_weights": weights,
        "fit_geometry_ok": int(geometry_ok),
        "fit_center_inside_background_hull": int(inside_hull),
        "fit_condition_number": condition,
        "fit_median_mae_db": median_mae,
        "fit_plane_mae_db": plane_mae if np.isfinite(plane_mae) else np.nan,
        "fit_use_plane": int(use_plane),
        "fit_plane_improvement": float(median_mae - plane_mae)
        if np.isfinite(plane_mae)
        else np.nan,
    }


# ============================================================================
# 7. 残差特征
# ============================================================================


def band_feature_name(low: float, high: float) -> str:
    return f"{int(round(low/1000))}_{int(round(high/1000))}k"


def extract_features_from_cube(
    sample: Dict[str, Any],
    config: V9Config,
) -> Dict[str, Any]:
    power = np.asarray(sample["power"], dtype=float)
    freq_hz = np.asarray(sample["freq_hz"], dtype=float)
    time_s = np.asarray(sample["time_s"], dtype=float)
    xy = np.asarray(sample["xy"], dtype=float)
    center_index = int(sample.get("center_index", 0))

    if power.ndim != 3:
        raise ValueError(f"power 应为 (n_points,n_freq,n_frames)，当前 {power.shape}")
    if power.shape[0] != len(xy) or power.shape[1] != len(freq_hz):
        raise ValueError("power、xy、freq_hz 维度不匹配")

    estimate = estimate_local_background(power, xy, center_index, config)
    center_power = power[center_index]
    background_power = estimate["background_power"]
    center_db = db_from_power(center_power, config.epsilon_power)
    background_db = estimate["background_db"]
    residual_db = center_db - background_db
    excess_power = np.maximum(center_power - background_power, 0.0)

    features: Dict[str, Any] = {}

    total_center = float(np.sum(center_power))
    total_bg = float(np.sum(background_power))
    total_excess = float(np.sum(excess_power))

    features["residual_integrated_snr_db"] = float(
        10.0 * np.log10((total_center + config.epsilon_power) / (total_bg + config.epsilon_power))
    )
    features["residual_integrated_excess_ratio"] = safe_ratio(total_excess, total_bg)
    features["residual_positive_power_fraction"] = safe_ratio(total_excess, total_center)
    features["residual_db_mean"] = float(np.mean(residual_db))
    features["residual_db_median"] = float(np.median(residual_db))
    features["residual_db_std"] = float(np.std(residual_db))
    features["residual_db_p75"] = float(np.percentile(residual_db, 75))
    features["residual_db_p90"] = float(np.percentile(residual_db, 90))
    features["residual_db_p95"] = float(np.percentile(residual_db, 95))
    features["residual_db_max"] = float(np.max(residual_db))

    for threshold in config.residual_thresholds_db:
        tag = str(threshold).replace(".", "p")
        active = residual_db >= threshold
        features[f"residual_positive_bin_ratio_{tag}db"] = float(np.mean(active))
        features[f"temporal_active_frame_ratio_{tag}db"] = float(
            np.mean(np.mean(active, axis=0) >= 0.10)
        )

    # 频谱特征：先沿时间聚合正残差功率。
    excess_spectrum = np.mean(excess_power, axis=1)
    residual_spectrum_db = np.median(residual_db, axis=1)
    spectrum_total = float(np.sum(excess_spectrum))

    features["spectral_excess_entropy"] = normalized_entropy(excess_spectrum)
    features["spectral_excess_flatness"] = spectral_flatness(excess_spectrum)
    features["spectral_residual_db_mean"] = float(np.mean(residual_spectrum_db))
    features["spectral_residual_db_p90"] = float(np.percentile(residual_spectrum_db, 90))

    if spectrum_total > config.epsilon_power:
        centroid = float(np.sum(freq_hz * excess_spectrum) / spectrum_total)
        bandwidth = float(
            np.sqrt(np.sum(((freq_hz - centroid) ** 2) * excess_spectrum) / spectrum_total)
        )
    else:
        centroid = 0.0
        bandwidth = 0.0
    features["spectral_excess_centroid_hz"] = centroid
    features["spectral_excess_bandwidth_hz"] = bandwidth

    df_hz = float(np.median(np.diff(freq_hz))) if len(freq_hz) > 1 else 0.0
    for threshold in config.residual_thresholds_db:
        tag = str(threshold).replace(".", "p")
        active_freq = residual_spectrum_db >= threshold
        features[f"spectral_active_freq_ratio_{tag}db"] = float(np.mean(active_freq))
        features[f"spectral_active_bandwidth_{tag}db_hz"] = float(np.sum(active_freq) * df_hz)
        features[f"spectral_longest_active_band_{tag}db_hz"] = float(
            longest_true_run(active_freq) * df_hz
        )

    # 分频段：只保留 50-60 kHz 和 60-70 kHz。
    # 50 kHz 以下及 70 kHz 以上已在 STFT 读取阶段被丢弃。
    subbands = [
        (50_000.0, 60_000.0),
        (60_000.0, 70_000.0),
    ]
    band_excess_values: List[float] = []
    for low, high in subbands:
        mask = (freq_hz >= low) & (freq_hz < high)
        if not np.any(mask):
            continue
        tag = band_feature_name(low, high)
        band_center = float(np.sum(center_power[mask]))
        band_bg = float(np.sum(background_power[mask]))
        band_excess = float(np.sum(excess_power[mask]))
        band_excess_values.append(band_excess)
        features[f"residual_snr_db_{tag}"] = float(
            10.0 * np.log10(
                (band_center + config.epsilon_power) / (band_bg + config.epsilon_power)
            )
        )
        features[f"residual_excess_ratio_{tag}"] = safe_ratio(band_excess, band_bg)
        features[f"residual_positive_fraction_{tag}"] = safe_ratio(band_excess, band_center)

    if band_excess_values:
        band_arr = np.asarray(band_excess_values, dtype=float)
        features["spectral_subband_entropy"] = normalized_entropy(band_arr)
        features["spectral_active_subband_count_20pct"] = int(
            np.sum(band_arr >= 0.20 * np.max(band_arr))
        ) if np.max(band_arr) > 0 else 0

    # 时间特征。
    center_frame = np.sum(center_power, axis=0)
    bg_frame = np.sum(background_power, axis=0)
    excess_frame = np.sum(excess_power, axis=0)
    frame_snr_db = 10.0 * np.log10(
        (center_frame + config.epsilon_power) / (bg_frame + config.epsilon_power)
    )
    frame_excess_ratio = excess_frame / (bg_frame + config.epsilon_power)

    features["temporal_frame_snr_mean_db"] = float(np.mean(frame_snr_db))
    features["temporal_frame_snr_std_db"] = float(np.std(frame_snr_db))
    features["temporal_frame_snr_p90_db"] = float(np.percentile(frame_snr_db, 90))
    features["temporal_excess_ratio_mean"] = float(np.mean(frame_excess_ratio))
    features["temporal_excess_ratio_std"] = float(np.std(frame_excess_ratio))
    features["temporal_excess_ratio_cv"] = float(
        np.std(frame_excess_ratio) / (np.mean(frame_excess_ratio) + config.epsilon_power)
    )
    features["temporal_excess_active_ratio"] = float(np.mean(frame_excess_ratio > 0.10))
    features["temporal_excess_persistent_ratio"] = float(
        np.mean(frame_excess_ratio >= np.median(frame_excess_ratio))
    )

    if len(frame_excess_ratio) >= 3:
        x = np.arange(len(frame_excess_ratio), dtype=float)
        corr, _ = spearmanr(x, frame_excess_ratio)
        features["temporal_excess_drift_spearman"] = float(corr) if np.isfinite(corr) else 0.0
    else:
        features["temporal_excess_drift_spearman"] = 0.0

    # 空间局部性。
    point_band_power = np.mean(power, axis=(1, 2))
    point_band_db = db_from_power(point_band_power, config.epsilon_power)
    neighbor_idx = estimate["selection"]["neighbors"]
    bg_idx = estimate["selection"]["background"]
    distances = estimate["selection"]["distances"]

    center_value = float(point_band_db[center_index])
    neighbor_values = point_band_db[neighbor_idx]
    bg_values = point_band_db[bg_idx]
    sorted_neighbors = np.sort(neighbor_values)[::-1]

    features["spatial_center_minus_neighbor_median_db"] = float(
        center_value - np.median(neighbor_values)
    )
    features["spatial_center_minus_background_median_db"] = float(
        center_value - np.median(bg_values)
    )
    features["spatial_center_minus_top_neighbor_db"] = float(
        center_value - sorted_neighbors[0]
    )
    features["spatial_center_rank_pct"] = float(
        np.mean(point_band_db <= center_value)
    )
    features["spatial_center_robust_z"] = float(
        (center_value - np.median(neighbor_values)) / robust_scale(neighbor_values, floor=0.25)
    )

    radial_mask = np.arange(len(xy)) != center_index
    if np.sum(radial_mask) >= 3:
        corr, _ = spearmanr(distances[radial_mask], point_band_db[radial_mask])
        features["spatial_radial_spearman"] = float(corr) if np.isfinite(corr) else 0.0
    else:
        features["spatial_radial_spearman"] = 0.0

    # 背景点选择敏感性：外圈范围变化后，核心指标变化多大。
    sensitivity_snr: List[float] = []
    sensitivity_excess: List[float] = []
    for q in (0.20, 0.35, 0.50):
        try:
            sel = select_background_indices(xy, center_index, config, quantile=q)
            bg_q = np.median(power[sel["background"]], axis=0)
            snr_q = float(
                10.0
                * np.log10(
                    (np.sum(center_power) + config.epsilon_power)
                    / (np.sum(bg_q) + config.epsilon_power)
                )
            )
            excess_q = safe_ratio(
                float(np.sum(np.maximum(center_power - bg_q, 0.0))),
                float(np.sum(bg_q)),
            )
            sensitivity_snr.append(snr_q)
            sensitivity_excess.append(excess_q)
        except Exception:
            continue

    features["qc_background_selection_snr_range_db"] = float(
        np.ptp(sensitivity_snr)
    ) if sensitivity_snr else np.nan
    features["qc_background_selection_excess_ratio_range"] = float(
        np.ptp(sensitivity_excess)
    ) if sensitivity_excess else np.nan

    features["qc_n_points"] = int(power.shape[0])
    features["qc_n_neighbors"] = int(len(neighbor_idx))
    features["qc_n_background_points"] = int(len(bg_idx))
    features["qc_n_frequency_bins"] = int(power.shape[1])
    features["qc_n_frames"] = int(power.shape[2])
    features["qc_background_distance_cutoff_cm"] = float(
        estimate["selection"]["distance_cutoff"]
    )
    features["qc_point_weight_min"] = float(np.min(estimate["point_weights"]))
    features["qc_point_weight_median"] = float(np.median(estimate["point_weights"]))

    features["fit_geometry_ok"] = estimate["fit_geometry_ok"]
    features["fit_center_inside_background_hull"] = estimate[
        "fit_center_inside_background_hull"
    ]
    features["fit_condition_number"] = estimate["fit_condition_number"]
    features["fit_median_mae_db"] = estimate["fit_median_mae_db"]
    features["fit_plane_mae_db"] = estimate["fit_plane_mae_db"]
    features["fit_use_plane"] = estimate["fit_use_plane"]
    features["fit_plane_improvement_db"] = estimate["fit_plane_improvement"]

    diagnostics = {
        "freq_hz": freq_hz,
        "time_s": time_s,
        "center_power_db": center_db,
        "background_power_db": background_db,
        "residual_db": residual_db,
        "excess_power": excess_power,
        "residual_spectrum_db": residual_spectrum_db,
        "excess_spectrum": excess_spectrum,
        "frame_snr_db": frame_snr_db,
        "frame_excess_ratio": frame_excess_ratio,
        "xy": xy,
        "point_band_db": point_band_db,
        "background_indices": bg_idx,
        "point_weights": estimate["point_weights"],
        "fit_use_plane": estimate["fit_use_plane"],
    }

    return {"features": features, "diagnostics": diagnostics}


# ============================================================================
# 8. 诊断图与残差文件
# ============================================================================


def save_diagnostic_outputs(
    sample_id: str,
    diagnostics: Dict[str, Any],
    output_dir: Path,
    save_npz: bool,
) -> Tuple[Optional[Path], Optional[Path]]:
    fig_path: Optional[Path] = None
    npz_path: Optional[Path] = None

    fig_dir = output_dir / "diagnostic_figures"
    ensure_dir(fig_dir)
    slug = safe_slug(sample_id)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        freq_khz = diagnostics["freq_hz"] / 1000.0
        center_spec = np.median(diagnostics["center_power_db"], axis=1)
        bg_spec = np.median(diagnostics["background_power_db"], axis=1)
        residual_spec = diagnostics["residual_spectrum_db"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        axes[0, 0].plot(freq_khz, center_spec, label="Center")
        axes[0, 0].plot(freq_khz, bg_spec, label="Estimated background")
        axes[0, 0].set_title("Center vs local background")
        axes[0, 0].set_xlabel("Frequency (kHz)")
        axes[0, 0].set_ylabel("Power (dB)")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        axes[0, 1].plot(freq_khz, residual_spec)
        axes[0, 1].axhline(0.0, linewidth=1)
        axes[0, 1].axhline(3.0, linewidth=1, linestyle="--")
        axes[0, 1].set_title("Median residual spectrum")
        axes[0, 1].set_xlabel("Frequency (kHz)")
        axes[0, 1].set_ylabel("Center - background (dB)")
        axes[0, 1].grid(True, alpha=0.3)

        t = diagnostics["time_s"]
        axes[1, 0].plot(t, diagnostics["frame_snr_db"])
        axes[1, 0].axhline(0.0, linewidth=1)
        axes[1, 0].set_title("Frame-wise local SNR")
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_ylabel("dB")
        axes[1, 0].grid(True, alpha=0.3)

        xy = diagnostics["xy"]
        values = diagnostics["point_band_db"]
        scatter = axes[1, 1].scatter(xy[:, 0], xy[:, 1], c=values, s=55)
        axes[1, 1].scatter([0.0], [0.0], marker="x", s=120, label="Center")
        bg = diagnostics["background_indices"]
        axes[1, 1].scatter(
            xy[bg, 0], xy[bg, 1], facecolors="none", s=90, label="Background points"
        )
        axes[1, 1].set_aspect("equal", adjustable="box")
        axes[1, 1].set_title(
            "Spatial band power | method="
            + ("plane" if diagnostics["fit_use_plane"] else "median")
        )
        axes[1, 1].set_xlabel("x (cm)")
        axes[1, 1].set_ylabel("y (cm)")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend(fontsize=8)
        fig.colorbar(scatter, ax=axes[1, 1], label="Mean band power (dB)")

        fig.suptitle(sample_id)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig_path = fig_dir / f"{slug}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"  [图失败] {sample_id}: {exc}")

    if save_npz:
        residual_dir = output_dir / "residual_npz"
        ensure_dir(residual_dir)
        npz_path = residual_dir / f"{slug}.npz"
        np.savez_compressed(
            npz_path,
            freq_hz=diagnostics["freq_hz"],
            time_s=diagnostics["time_s"],
            center_power_db=diagnostics["center_power_db"],
            background_power_db=diagnostics["background_power_db"],
            residual_db=diagnostics["residual_db"],
            excess_power=diagnostics["excess_power"],
            xy=diagnostics["xy"],
            point_band_db=diagnostics["point_band_db"],
            background_indices=diagnostics["background_indices"],
            point_weights=diagnostics["point_weights"],
        )

    return fig_path, npz_path


# ============================================================================
# 9. 特征分离度与跨场景检查
# ============================================================================


def model_feature_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for col in df.columns:
        if col.startswith(FEATURE_PREFIXES) and not col.startswith(QUALITY_PREFIXES):
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().mean() >= 0.80 and values.nunique(dropna=True) > 1:
                cols.append(col)
    return cols


def feature_separation_analysis(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    columns = [
        "feature",
        "n_true",
        "n_false",
        "true_median",
        "false_median",
        "median_difference_true_minus_false",
        "auc",
        "auc_oriented",
        "direction",
        "mannwhitney_p",
    ]
    labeled = df[df["label"].isin(["TRUE_LEAK", "FALSE_LEAK"])].copy()
    if labeled.empty or labeled["label"].nunique() < 2:
        return pd.DataFrame(columns=columns)

    y = (labeled["label"] == "TRUE_LEAK").astype(int).to_numpy()
    rows: List[Dict[str, Any]] = []
    for feature in feature_cols:
        values = pd.to_numeric(labeled[feature], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values)
        true_vals = values[valid & (y == 1)]
        false_vals = values[valid & (y == 0)]
        if true_vals.size < 2 or false_vals.size < 2:
            continue
        auc = safe_auc(y[valid], values[valid])
        try:
            _, p = mannwhitneyu(true_vals, false_vals, alternative="two-sided")
            p = float(p)
        except Exception:
            p = np.nan
        rows.append(
            {
                "feature": feature,
                "n_true": int(true_vals.size),
                "n_false": int(false_vals.size),
                "true_median": float(np.median(true_vals)),
                "false_median": float(np.median(false_vals)),
                "median_difference_true_minus_false": float(
                    np.median(true_vals) - np.median(false_vals)
                ),
                "auc": auc,
                "auc_oriented": max(auc, 1.0 - auc) if np.isfinite(auc) else np.nan,
                "direction": "higher_is_true" if np.isfinite(auc) and auc >= 0.5 else "lower_is_true",
                "mannwhitney_p": p,
            }
        )

    out = pd.DataFrame(rows, columns=columns)
    if not out.empty:
        out = out.sort_values(["auc_oriented", "mannwhitney_p"], ascending=[False, True])
    return out.reset_index(drop=True)


def fit_univariate_threshold(
    y: np.ndarray,
    values: np.ndarray,
    max_candidates: int,
) -> Optional[Dict[str, float]]:
    y = np.asarray(y, dtype=int)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    y = y[valid]
    values = values[valid]
    if values.size < 4 or np.unique(y).size < 2 or np.unique(values).size < 2:
        return None

    auc = safe_auc(y, values)
    direction = 1.0 if not np.isfinite(auc) or auc >= 0.5 else -1.0
    oriented = values * direction

    unique = np.unique(oriented)
    if unique.size <= max_candidates:
        candidates = np.concatenate(
            ([unique[0] - 1.0e-12], (unique[:-1] + unique[1:]) / 2.0, [unique[-1] + 1.0e-12])
        )
    else:
        candidates = np.unique(
            np.quantile(oriented, np.linspace(0.0, 1.0, max_candidates))
        )

    best_threshold = float(np.median(oriented))
    best_score = -np.inf
    for threshold in candidates:
        pred = (oriented >= threshold).astype(int)
        score = balanced_accuracy(y, pred)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return {
        "direction": direction,
        "threshold_oriented": best_threshold,
        "threshold_original": best_threshold * direction,
        "train_balanced_accuracy": float(best_score),
    }


def leave_one_scene_out_univariate(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    group_col: str,
    max_threshold_candidates: int,
) -> pd.DataFrame:
    columns = [
        "test_group",
        "feature",
        "n_train",
        "n_test",
        "direction",
        "threshold_original_scale",
        "train_balanced_accuracy",
        "test_balanced_accuracy",
        "test_auc_oriented",
    ]
    if group_col not in df.columns:
        return pd.DataFrame(columns=columns)

    labeled = df[df["label"].isin(["TRUE_LEAK", "FALSE_LEAK"])].copy()
    groups = labeled[group_col].fillna("").astype(str).to_numpy()
    unique_groups = sorted(g for g in np.unique(groups) if g)
    if len(unique_groups) < 2:
        return pd.DataFrame(columns=columns)

    y_all = (labeled["label"] == "TRUE_LEAK").astype(int).to_numpy()
    rows: List[Dict[str, Any]] = []

    for test_group in unique_groups:
        test_mask = groups == test_group
        train_mask = ~test_mask
        y_train = y_all[train_mask]
        y_test = y_all[test_mask]
        if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
            continue

        for feature in feature_cols:
            values_all = pd.to_numeric(labeled[feature], errors="coerce").to_numpy(dtype=float)
            fit = fit_univariate_threshold(
                y_train,
                values_all[train_mask],
                max_candidates=max_threshold_candidates,
            )
            if fit is None:
                continue

            test_values = values_all[test_mask]
            valid = np.isfinite(test_values)
            if np.sum(valid) < 2:
                continue
            oriented = test_values[valid] * fit["direction"]
            pred = (oriented >= fit["threshold_oriented"]).astype(int)
            rows.append(
                {
                    "test_group": test_group,
                    "feature": feature,
                    "n_train": int(np.sum(train_mask)),
                    "n_test": int(np.sum(valid)),
                    "direction": "higher_is_true" if fit["direction"] == 1 else "lower_is_true",
                    "threshold_original_scale": fit["threshold_original"],
                    "train_balanced_accuracy": fit["train_balanced_accuracy"],
                    "test_balanced_accuracy": balanced_accuracy(y_test[valid], pred),
                    "test_auc_oriented": safe_auc(y_test[valid], oriented),
                }
            )

    return pd.DataFrame(rows, columns=columns)


def summarize_cross_scene(cross_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "feature",
        "n_test_groups",
        "mean_test_balanced_accuracy",
        "min_test_balanced_accuracy",
        "median_test_balanced_accuracy",
        "mean_test_auc_oriented",
        "min_test_auc_oriented",
    ]
    if cross_df is None or cross_df.empty:
        return pd.DataFrame(columns=columns)
    return (
        cross_df.groupby("feature", as_index=False)
        .agg(
            n_test_groups=("test_group", "nunique"),
            mean_test_balanced_accuracy=("test_balanced_accuracy", "mean"),
            min_test_balanced_accuracy=("test_balanced_accuracy", "min"),
            median_test_balanced_accuracy=("test_balanced_accuracy", "median"),
            mean_test_auc_oriented=("test_auc_oriented", "mean"),
            min_test_auc_oriented=("test_auc_oriented", "min"),
        )
        .sort_values(
            ["mean_test_balanced_accuracy", "min_test_balanced_accuracy"],
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================================
# 10. 主流程
# ============================================================================


def validate_dataset_config(dataset: Dict[str, Any]) -> Dict[str, Any]:
    required = ["name", "label", "center_root_dir", "offset_root_dir"]
    missing = [k for k in required if k not in dataset]
    if missing:
        raise ValueError(f"DATASETS 项缺少字段: {missing}")

    out = dict(dataset)
    out["name"] = str(out["name"])
    out["scene"] = str(out.get("scene", out["name"]))
    out["label"] = str(out.get("label", "")).upper().strip()
    if out["label"] not in VALID_LABELS:
        raise ValueError(f"非法 label: {out['label']}")
    out["center_root_dir"] = str(out["center_root_dir"])
    out["offset_root_dir"] = str(out["offset_root_dir"])
    out["time_folders"] = list(out.get("time_folders", []))
    return out


def process_all_datasets(
    datasets: Sequence[Dict[str, Any]],
    output_dir: Path,
    config: V9Config,
) -> Dict[str, Path]:
    ensure_dir(output_dir)
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    plotted = 0

    print("=" * 100)
    print("v9 局部背景鲁棒估计 + 中心残差特征提取（WAV目录版）")
    print("=" * 100)
    print("输出目录:", output_dir)
    print("频带:", f"{config.freq_low_hz:g}-{config.freq_high_hz:g} Hz")

    for raw_dataset in datasets:
        dataset = validate_dataset_config(raw_dataset)
        center_root = Path(dataset["center_root_dir"])
        offset_root = Path(dataset["offset_root_dir"])

        print("\n" + "#" * 100)
        print(
            f"数据集: {dataset['name']} | scene={dataset['scene']} | "
            f"label={dataset['label']}"
        )
        print("中心目录:", center_root)
        print("偏移目录:", offset_root)

        if not center_root.exists() or not offset_root.exists():
            failures.append(
                {
                    "dataset": dataset["name"],
                    "scene": dataset["scene"],
                    "time": "",
                    "center": "",
                    "error_type": "MissingRootDirectory",
                    "error": f"中心或偏移根目录不存在: {center_root} | {offset_root}",
                    "traceback": "",
                }
            )
            print("[跳过] 根目录不存在")
            continue

        time_folders = dataset["time_folders"] or common_time_folders(center_root, offset_root)
        print("时间文件夹数量:", len(time_folders))

        for time_name in time_folders:
            center_dir = center_root / time_name
            offset_dir = offset_root / time_name
            print("\n" + "-" * 90)
            print("处理:", time_name)

            if not center_dir.exists() or not offset_dir.exists():
                failures.append(
                    {
                        "dataset": dataset["name"],
                        "scene": dataset["scene"],
                        "time": time_name,
                        "center": "",
                        "error_type": "MissingTimeDirectory",
                        "error": f"时间目录不存在: {center_dir} | {offset_dir}",
                        "traceback": "",
                    }
                )
                print("[跳过] 时间目录不存在")
                continue

            center_files = detect_center_files(center_dir)
            offset_files = parse_offset_files(offset_dir)
            centers = sorted(center_files.keys(), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
            if MAX_CENTERS_PER_TIME is not None:
                centers = centers[:MAX_CENTERS_PER_TIME]

            print("检测到中心数:", len(centers))
            print("检测到偏移组合数:", len(offset_files))

            for index, center_id in enumerate(centers, start=1):
                sample_id = f"{dataset['name']}__{time_name}__center_{center_id}"
                try:
                    sample = load_one_center_cube(
                        center_paths=center_files[center_id],
                        offset_mapping=offset_files,
                        center_id=center_id,
                        config=config,
                    )
                    result = extract_features_from_cube(sample, config)

                    n_neighbors = int(sample["power"].shape[0] - 1)
                    expected = max(int(sample["n_offset_discovered"]), 1)
                    combo_ratio = n_neighbors / expected
                    peak_values = sample["peak_abs"]
                    rms_values = sample["rms"]

                    row: Dict[str, Any] = {
                        "sample_id": sample_id,
                        "dataset": dataset["name"],
                        "scene": dataset["scene"],
                        "time": time_name,
                        "center": center_id,
                        "label": dataset["label"],
                        "center_file": str(center_files[center_id][0]),
                        "offset_dir": str(offset_dir),
                        "sample_rate_hz": int(sample["fs"]),
                        "offset_combo_discovered": int(sample["n_offset_discovered"]),
                        "offset_combo_used": n_neighbors,
                        "offset_combo_used_ratio": float(combo_ratio),
                        "failed_offset_count": int(len(sample["failed_offsets"])),
                        "wav_peak_center": float(peak_values[0]),
                        "wav_peak_neighbor_median": float(np.median(peak_values[1:])),
                        "wav_peak_center_neighbor_ratio": safe_ratio(
                            float(peak_values[0]), float(np.median(peak_values[1:]))
                        ),
                        "wav_rms_center": float(rms_values[0]),
                        "wav_rms_neighbor_median": float(np.median(rms_values[1:])),
                    }
                    row.update(result["features"])
                    rows.append(row)

                    if config.save_diagnostic_plots and plotted < config.diagnostic_plot_limit:
                        save_diagnostic_outputs(
                            sample_id,
                            result["diagnostics"],
                            output_dir,
                            save_npz=config.save_residual_npz,
                        )
                        plotted += 1
                    elif config.save_residual_npz:
                        save_diagnostic_outputs(
                            sample_id,
                            result["diagnostics"],
                            output_dir,
                            save_npz=True,
                        )

                    print(
                        f"  [{index:>3}/{len(centers)}] OK center={center_id} | "
                        f"points={n_neighbors} | "
                        f"SNR={row['residual_integrated_snr_db']:.3f} dB | "
                        f"excess={row['residual_integrated_excess_ratio']:.4f} | "
                        f"method={'plane' if row['fit_use_plane'] else 'median'}"
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "dataset": dataset["name"],
                            "scene": dataset["scene"],
                            "time": time_name,
                            "center": center_id,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    print(f"  [{index:>3}/{len(centers)}] FAIL center={center_id}: {exc}")

    feature_df = pd.DataFrame(rows)
    failure_columns = [
        "dataset",
        "scene",
        "time",
        "center",
        "error_type",
        "error",
        "traceback",
    ]
    failure_df = pd.DataFrame(failures, columns=failure_columns)

    all_features_path = output_dir / "v9_all_features.csv"
    model_features_path = output_dir / "v9_model_ready_features.csv"
    failures_path = output_dir / "v9_failures.csv"
    separation_path = output_dir / "v9_feature_separation.csv"
    cross_path = output_dir / "v9_leave_one_scene_out_univariate.csv"
    cross_summary_path = output_dir / "v9_cross_scene_feature_summary.csv"
    config_path = output_dir / "v9_run_config.json"
    report_path = output_dir / "v9_report.txt"

    feature_df.to_csv(all_features_path, index=False, encoding="utf-8-sig")
    failure_df.to_csv(failures_path, index=False, encoding="utf-8-sig")

    if feature_df.empty:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(
                {"config": asdict(config), "datasets": list(datasets)},
                f,
                ensure_ascii=False,
                indent=2,
            )
        raise RuntimeError(
            f"没有任何样本成功处理。请先查看 {failures_path}。"
        )

    feature_cols = model_feature_columns(feature_df)
    metadata_cols = [
        c
        for c in [
            "sample_id",
            "dataset",
            "scene",
            "time",
            "center",
            "label",
            "center_file",
            "offset_dir",
        ]
        if c in feature_df.columns
    ]
    feature_df[metadata_cols + feature_cols].to_csv(
        model_features_path, index=False, encoding="utf-8-sig"
    )

    separation_df = feature_separation_analysis(feature_df, feature_cols)
    separation_df.to_csv(separation_path, index=False, encoding="utf-8-sig")

    cross_df = leave_one_scene_out_univariate(
        feature_df,
        feature_cols,
        group_col=config.group_column,
        max_threshold_candidates=config.max_threshold_candidates,
    )
    cross_df.to_csv(cross_path, index=False, encoding="utf-8-sig")
    cross_summary = summarize_cross_scene(cross_df)
    cross_summary.to_csv(cross_summary_path, index=False, encoding="utf-8-sig")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {"config": asdict(config), "datasets": list(datasets)},
            f,
            ensure_ascii=False,
            indent=2,
        )

    lines: List[str] = []
    lines.append("v9 局部背景残差提取报告")
    lines.append("=" * 90)
    lines.append(f"成功样本数: {len(feature_df)}")
    lines.append(f"失败样本数: {len(failure_df)}")
    lines.append(f"候选模型特征数: {len(feature_cols)}")
    lines.append(f"scene 数量: {feature_df['scene'].nunique() if 'scene' in feature_df else 0}")
    lines.append("")
    lines.append("重要原则:")
    lines.append("  1. v9 第一阶段先验证残差是否有效，不直接堆复杂分类器。")
    lines.append("  2. 周围点不是独立麦克风，本程序执行的是波束图局部背景消除。")
    lines.append("  3. WAV 不做逐文件峰值归一化；文件之间必须保留可比较幅值标度。")
    lines.append("  4. 跨场景结果以留出整个 scene 为准，而不是只留出 time。")
    lines.append("")
    lines.append("质量概览:")
    lines.append(
        f"  平均周围点数: {feature_df['qc_n_neighbors'].mean():.2f}"
        if "qc_n_neighbors" in feature_df
        else "  平均周围点数: NA"
    )
    lines.append(
        f"  空间平面采用比例: {feature_df['fit_use_plane'].mean():.3f}"
        if "fit_use_plane" in feature_df
        else "  空间平面采用比例: NA"
    )
    lines.append(
        f"  中心位于背景点凸包内比例: "
        f"{feature_df['fit_center_inside_background_hull'].mean():.3f}"
        if "fit_center_inside_background_hull" in feature_df
        else "  中心位于背景点凸包内比例: NA"
    )
    lines.append("")

    if not separation_df.empty:
        lines.append("全体样本单特征分离度前15（仅探索，不能证明跨场景有效）:")
        for _, row in separation_df.head(15).iterrows():
            lines.append(
                f"  {row['feature']}: AUC_oriented={row['auc_oriented']:.4f}, "
                f"TRUE_median={row['true_median']:.6g}, "
                f"FALSE_median={row['false_median']:.6g}, p={row['mannwhitney_p']:.3g}"
            )
        lines.append("")
    else:
        lines.append("没有足够的 TRUE/FALSE 标签，未计算特征分离度。")
        lines.append("")

    if not cross_summary.empty:
        lines.append("跨 scene 单特征留出结果前15:")
        for _, row in cross_summary.head(15).iterrows():
            lines.append(
                f"  {row['feature']}: mean_bal_acc={row['mean_test_balanced_accuracy']:.4f}, "
                f"min_bal_acc={row['min_test_balanced_accuracy']:.4f}, "
                f"mean_auc={row['mean_test_auc_oriented']:.4f}, "
                f"min_auc={row['min_test_auc_oriented']:.4f}"
            )
    else:
        lines.append(
            "未生成跨 scene 结果。需要至少两个 scene，且训练和测试 scene 中都要有 TRUE/FALSE。"
        )

    lines.append("")
    lines.append("先查看顺序:")
    lines.append(f"  1. {failures_path.name}")
    lines.append(f"  2. {all_features_path.name}")
    lines.append(f"  3. {separation_path.name}")
    lines.append(f"  4. {cross_summary_path.name}")
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 100)
    print("处理完成")
    print("成功样本:", len(feature_df))
    print("失败样本:", len(failure_df))
    print("全部特征:", all_features_path)
    print("失败明细:", failures_path)
    print("报告:", report_path)
    print("=" * 100)

    return {
        "all_features": all_features_path,
        "model_features": model_features_path,
        "failures": failures_path,
        "separation": separation_path,
        "cross_scene": cross_path,
        "cross_scene_summary": cross_summary_path,
        "config": config_path,
        "report": report_path,
    }


# ============================================================================
# 11. 合成数据自检
# ============================================================================


def make_synthetic_cube(
    rng: np.random.Generator,
    leak: bool,
    scene_index: int,
    config: V9Config,
) -> Dict[str, Any]:
    distances = [5, 10, 15, 20, 25]
    xy = [(0.0, 0.0)]
    for distance in distances:
        for direction in DIRECTIONS:
            xy.append(coordinate_from_direction(direction, distance))
    xy_arr = np.asarray(xy, dtype=float)

    freq_hz = np.linspace(config.freq_low_hz, config.freq_high_hz, 256)
    time_s = np.linspace(0.0, 1.0, 40)
    n_points = len(xy_arr)
    n_freq = len(freq_hz)
    n_frames = len(time_s)

    # 场景相关背景：不同整体频谱、空间梯度、机械窄带。
    base_db = -72.0 + 2.5 * scene_index
    spectral_shape = (
        2.0 * np.sin((freq_hz - 20_000.0) / 9_000.0 + scene_index)
        - 0.000025 * (freq_hz - 45_000.0)
    )
    mechanical = 10.0 * np.exp(-0.5 * ((freq_hz - (30_000 + 2_000 * scene_index)) / 350.0) ** 2)
    temporal = 1.5 * np.sin(2.0 * np.pi * (2.0 + 0.2 * scene_index) * time_s)

    power_db = np.zeros((n_points, n_freq, n_frames), dtype=float)
    for i, (x, y) in enumerate(xy_arr):
        gradient = 0.06 * x - 0.04 * y
        local = rng.normal(0.0, 0.7, size=(n_freq, n_frames))
        power_db[i] = (
            base_db
            + spectral_shape[:, None]
            + mechanical[:, None]
            + temporal[None, :]
            + gradient
            + local
        )

    if leak:
        leak_shape = (
            8.0 * np.exp(-0.5 * ((freq_hz - 48_000.0) / 8_000.0) ** 2)
            + 5.0 * np.exp(-0.5 * ((freq_hz - 65_000.0) / 7_000.0) ** 2)
        )
        leak_temporal = 0.8 + 0.4 * (np.sin(2.0 * np.pi * 3.0 * time_s) ** 2)
        distances_arr = np.linalg.norm(xy_arr, axis=1)
        for i, distance in enumerate(distances_arr):
            # 中心最强，近邻含少量主瓣/旁瓣污染，外圈迅速衰减。
            spatial_gain = math.exp(-distance / 8.0)
            linear_bg = power_from_db(power_db[i])
            leak_power = power_from_db(
                -70.0 + leak_shape[:, None] + 10.0 * np.log10(leak_temporal[None, :])
            )
            power_db[i] = db_from_power(linear_bg + spatial_gain * leak_power, config.epsilon_power)
    else:
        # 假泄漏：增加一个全场共同窄带，不只中心出现。
        false_line = 5.0 * np.exp(-0.5 * ((freq_hz - 54_000.0) / 220.0) ** 2)
        power_db += false_line[None, :, None]

    return {
        "power": power_from_db(power_db),
        "freq_hz": freq_hz,
        "time_s": time_s,
        "xy": xy_arr,
        "center_index": 0,
    }


def run_self_test(config: V9Config) -> None:
    print("开始 v9 合成数据自检...")
    rng = np.random.default_rng(config.random_state)
    true_values: List[float] = []
    false_values: List[float] = []

    for scene_index in range(3):
        for _ in range(3):
            false_sample = make_synthetic_cube(rng, False, scene_index, config)
            true_sample = make_synthetic_cube(rng, True, scene_index, config)
            false_result = extract_features_from_cube(false_sample, config)["features"]
            true_result = extract_features_from_cube(true_sample, config)["features"]
            false_values.append(false_result["residual_integrated_excess_ratio"])
            true_values.append(true_result["residual_integrated_excess_ratio"])

    true_median = float(np.median(true_values))
    false_median = float(np.median(false_values))
    if not true_median > false_median:
        raise AssertionError(
            "自检失败：TRUE 残差没有高于 FALSE。"
            f" TRUE={true_median}, FALSE={false_median}"
        )

    print("自检通过。")
    print(f"TRUE 残差中位数:  {true_median:.6f}")
    print(f"FALSE 残差中位数: {false_median:.6f}")
    print("已覆盖：场景频谱变化、空间梯度、全场机械窄带、中心宽带泄漏。")


# ============================================================================
# 12. ABCD raw/plane extension
# ============================================================================

# Keep references to the proven v9 processing functions.  The extension below
# changes only the background products/output layout and the command-line
# configuration; the original feature extraction remains the compatibility
# baseline for the selected method.
_legacy_extract_features_from_cube = extract_features_from_cube
_legacy_process_all_datasets = process_all_datasets

ALL_METHODS: Tuple[str, ...] = ("raw", "plane")


def estimate_local_background(
    power_cube: np.ndarray,
    xy: np.ndarray,
    center_index: int,
    config: V9Config,
) -> Dict[str, Any]:
    """Calculate the spatial-plane background for every sample.

    A median background is calculated internally only as a robust safety
    reference for clipping and fit diagnostics. It is not exported as a method.
    The explicit plane estimate is saved whenever finite; geometry and fit
    quality fields are retained for validation.
    """
    selection = select_background_indices(xy, center_index, config)
    bg_idx = selection["background"]

    power_db = db_from_power(power_cube, config.epsilon_power)
    bg_db = power_db[bg_idx]
    n_bg, n_freq, n_frames = bg_db.shape
    y = bg_db.reshape(n_bg, -1)

    median_flat = np.median(y, axis=0)
    median_prediction = median_flat.reshape(n_freq, n_frames)
    median_mae = float(np.median(np.abs(y - median_flat[None, :])))

    bg_xy = xy[bg_idx] - xy[center_index]
    max_radius = float(np.max(np.linalg.norm(bg_xy, axis=1))) if len(bg_xy) else 1.0
    scale_xy = max(max_radius, 1.0)
    xy_scaled = bg_xy / scale_xy
    X = np.column_stack([np.ones(n_bg), xy_scaled[:, 0], xy_scaled[:, 1]])

    weights = robust_point_weights(y, config.point_huber_delta)
    XtWX = X.T @ (weights[:, None] * X)
    condition = float(np.linalg.cond(XtWX)) if XtWX.size else np.inf
    inside_hull = point_inside_hull(np.array([0.0, 0.0]), bg_xy)
    rank_ok = bool(np.linalg.matrix_rank(X) >= 3)
    geometry_ok = bool(
        rank_ok
        and inside_hull
        and np.isfinite(condition)
        and condition <= config.max_geometry_condition
    )

    plane_prediction = median_prediction.copy()
    plane_mae = np.inf
    plane_valid = False
    plane_solver = "none"
    plane_error = ""

    try:
        # A very small scale-aware ridge makes the explicit plane available for
        # comparison even when a few offsets are missing.  It does not make the
        # plane eligible for automatic selection unless geometry_ok is true.
        ridge = max(float(np.trace(XtWX)) / 3.0, 1.0) * 1.0e-10
        lhs = XtWX + ridge * np.eye(3)
        rhs = X.T @ (weights[:, None] * y)
        try:
            beta = np.linalg.solve(lhs, rhs)
            plane_solver = "ridge_solve"
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(lhs, rcond=1.0e-12) @ rhs
            plane_solver = "ridge_pinv"

        plane_flat = beta[0]
        fitted = X @ beta
        plane_mae = float(np.median(np.abs(y - fitted)))
        lower = median_flat - config.plane_clip_margin_db
        upper = median_flat + config.plane_clip_margin_db
        plane_flat = np.clip(plane_flat, lower, upper)
        plane_prediction = plane_flat.reshape(n_freq, n_frames)
        plane_valid = bool(np.all(np.isfinite(plane_prediction)))
    except Exception as exc:
        plane_error = f"{type(exc).__name__}: {exc}"
        plane_prediction = median_prediction.copy()
        plane_mae = np.inf
        plane_valid = False

    improvement_db = (
        float(median_mae - plane_mae) if np.isfinite(plane_mae) else np.nan
    )
    improvement_ratio = (
        float(plane_mae / max(median_mae, 1.0e-12))
        if np.isfinite(plane_mae)
        else np.nan
    )
    use_plane = bool(
        plane_valid
        and geometry_ok
        and np.isfinite(plane_mae)
        and plane_mae < config.plane_improvement_ratio * max(median_mae, 1.0e-12)
    )

    reject_reasons: List[str] = []
    if not plane_valid:
        reject_reasons.append("plane_not_calculable")
    if not rank_ok:
        reject_reasons.append("geometry_rank_below_3")
    if not inside_hull:
        reject_reasons.append("center_outside_background_hull")
    if not np.isfinite(condition) or condition > config.max_geometry_condition:
        reject_reasons.append("condition_number_too_large")
    if plane_valid and np.isfinite(plane_mae) and not (
        plane_mae < config.plane_improvement_ratio * max(median_mae, 1.0e-12)
    ):
        reject_reasons.append("plane_mae_improvement_insufficient")
    if plane_error:
        reject_reasons.append(plane_error)
    reject_reason = "OK" if use_plane else "|".join(reject_reasons or ["not_selected"])

    median_power = power_from_db(median_prediction)
    plane_power = power_from_db(plane_prediction)
    # The two-method edition always exposes plane as the primary residual.
    selected_db = plane_prediction
    selected_power = plane_power

    return {
        # Legacy unsuffixed fields point to plane in this edition.
        "background_power": plane_power,
        "background_db": plane_prediction,
        "median_background_db": median_prediction,  # internal diagnostic only
        "plane_background_db": plane_prediction,
        "background_power_plane": plane_power,
        "background_db_plane": plane_prediction,
        "selected_method": "plane",
        "plane_valid": int(plane_valid),
        "plane_geometry_ok": int(geometry_ok),
        "plane_solver": plane_solver,
        "plane_reject_reason": reject_reason,
        "selection": selection,
        "point_weights": weights,
        "fit_geometry_ok": int(geometry_ok),
        "fit_center_inside_background_hull": int(inside_hull),
        "fit_condition_number": condition,
        "fit_median_mae_db": median_mae,
        "fit_plane_mae_db": plane_mae if np.isfinite(plane_mae) else np.nan,
        "fit_use_plane": int(use_plane),
        "fit_plane_improvement": improvement_db,
        "fit_plane_improvement_ratio": improvement_ratio,
    }


def _method_summary_features(
    method: str,
    component_power: np.ndarray,
    center_power: np.ndarray,
    background_power: Optional[np.ndarray],
    config: V9Config,
) -> Dict[str, float]:
    component = np.maximum(np.asarray(component_power, dtype=float), 0.0)
    center = np.maximum(np.asarray(center_power, dtype=float), 0.0)
    total_component = float(np.sum(component))
    total_center = float(np.sum(center))
    spectrum = np.mean(component, axis=1)
    frame = np.sum(component, axis=0)
    result: Dict[str, float] = {
        f"{method}_component_total_power": total_component,
        f"{method}_component_fraction_of_center": safe_ratio(total_component, total_center),
        f"{method}_spectral_entropy": normalized_entropy(spectrum),
        f"{method}_spectral_flatness": spectral_flatness(spectrum),
        f"{method}_temporal_cv": float(np.std(frame) / (np.mean(frame) + config.epsilon_power)),
        f"{method}_nonzero_bin_ratio": float(np.mean(component > config.epsilon_power)),
    }
    if background_power is not None:
        bg = np.maximum(np.asarray(background_power, dtype=float), 0.0)
        result[f"{method}_integrated_snr_db"] = float(
            10.0 * np.log10((total_center + config.epsilon_power) / (float(np.sum(bg)) + config.epsilon_power))
        )
        result[f"{method}_excess_to_background_ratio"] = safe_ratio(total_component, float(np.sum(bg)))
    return result


def extract_features_from_cube(sample: Dict[str, Any], config: V9Config) -> Dict[str, Any]:
    # Legacy features use unsuffixed fields, which point to plane here.
    result = _legacy_extract_features_from_cube(sample, config)

    power = np.asarray(sample["power"], dtype=float)
    xy = np.asarray(sample["xy"], dtype=float)
    center_index = int(sample.get("center_index", 0))
    center_power = power[center_index]
    center_db = db_from_power(center_power, config.epsilon_power)
    estimate = estimate_local_background(power, xy, center_index, config)

    plane_bg_power = np.asarray(estimate["background_power_plane"], dtype=float)
    plane_bg_db = np.asarray(estimate["background_db_plane"], dtype=float)
    plane_residual_db = center_db - plane_bg_db
    plane_excess_power = np.maximum(center_power - plane_bg_power, 0.0)

    products: Dict[str, Dict[str, np.ndarray]] = {
        "raw": {
            "background_power": np.zeros_like(center_power),
            "background_db": np.full_like(center_db, -300.0),
            "residual_db": center_db.copy(),
            "component_power": center_power.copy(),
        },
        "plane": {
            "background_power": plane_bg_power,
            "background_db": plane_bg_db,
            "residual_db": plane_residual_db,
            "component_power": plane_excess_power,
        },
    }

    features = result["features"]
    for method in ALL_METHODS:
        bg = None if method == "raw" else products[method]["background_power"]
        features.update(_method_summary_features(
            method, products[method]["component_power"], center_power, bg, config
        ))
    features.update({
        "selected_method": "plane",  # compatibility label
        "primary_method": "plane",
        "plane_valid": estimate["plane_valid"],
        "plane_geometry_ok": estimate["plane_geometry_ok"],
        "plane_solver": estimate["plane_solver"],
        "plane_reject_reason": estimate["plane_reject_reason"],
        "fit_plane_improvement_ratio": estimate["fit_plane_improvement_ratio"],
    })

    diagnostics = result["diagnostics"]
    diagnostics.update({
        "center_power_db": center_db,
        "center_power": center_power,
        "products": products,
        "background_power_db_plane": plane_bg_db,
        "residual_db_plane": plane_residual_db,
        "excess_power_plane": plane_excess_power,
        "selected_method": "plane",
        "primary_method": "plane",
        "plane_valid": estimate["plane_valid"],
        "plane_geometry_ok": estimate["plane_geometry_ok"],
        "plane_solver": estimate["plane_solver"],
        "plane_reject_reason": estimate["plane_reject_reason"],
        "fit_condition_number": estimate["fit_condition_number"],
        "fit_median_mae_db": estimate["fit_median_mae_db"],
        "fit_plane_mae_db": estimate["fit_plane_mae_db"],
        "fit_plane_improvement_db": estimate["fit_plane_improvement"],
        "fit_plane_improvement_ratio": estimate["fit_plane_improvement_ratio"],
        "plane_used_by_auto": estimate["fit_use_plane"],
    })
    diagnostics["background_power_db"] = plane_bg_db
    diagnostics["residual_db"] = plane_residual_db
    diagnostics["excess_power"] = plane_excess_power
    return {"features": features, "diagnostics": diagnostics}

def _save_one_method_npz(
    path: Path,
    sample_id: str,
    method: str,
    diagnostics: Dict[str, Any],
) -> None:
    ensure_dir(path.parent)
    product = diagnostics["products"][method]
    payload: Dict[str, Any] = {
        "sample_id": np.asarray(sample_id),
        "method": np.asarray(method),
        "freq_hz": diagnostics["freq_hz"],
        "time_s": diagnostics["time_s"],
        "center_power_db": diagnostics["center_power_db"],
        "component_power": product["component_power"],
        "background_power_db": product["background_db"],
        "residual_db": product["residual_db"],
        "selected_method": np.asarray("plane"),
        "primary_method": np.asarray("plane"),
        "plane_valid": np.asarray(diagnostics["plane_valid"]),
        "plane_geometry_ok": np.asarray(diagnostics["plane_geometry_ok"]),
        "plane_used_by_auto": np.asarray(diagnostics["plane_used_by_auto"]),
    }
    if method == "raw":
        payload["raw_power"] = product["component_power"]
    else:
        payload[f"excess_power_{method}"] = product["component_power"]
        payload[f"background_power_db_{method}"] = product["background_db"]
        payload[f"residual_db_{method}"] = product["residual_db"]
    np.savez_compressed(path, **payload)


def save_diagnostic_outputs(
    sample_id: str,
    diagnostics: Dict[str, Any],
    output_dir: Path,
    save_npz: bool,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Save one consolidated NPZ plus raw/plane method folders."""
    ensure_dir(output_dir)
    slug = safe_slug(sample_id)
    fig_path: Optional[Path] = None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = output_dir / "diagnostic_figures"
        ensure_dir(fig_dir)
        freq_khz = np.asarray(diagnostics["freq_hz"], dtype=float) / 1000.0
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
        for ax, method in zip(np.atleast_1d(axes).ravel(), ALL_METHODS):
            component = diagnostics["products"][method]["component_power"]
            spectrum = np.sqrt(np.maximum(np.mean(component, axis=1), 0.0))
            ax.plot(freq_khz, spectrum)
            ax.set_title(method)
            ax.set_xlabel("Frequency (kHz)")
            ax.set_ylabel("sqrt(mean power)")
            ax.grid(True, alpha=0.3)
        fig.suptitle(
            f"{sample_id} | plane_valid={diagnostics['plane_valid']} | "
            f"geometry_ok={diagnostics['plane_geometry_ok']}"
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig_path = fig_dir / f"{slug}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"  [图失败] {sample_id}: {exc}")

    residual_dir = output_dir / "residual_npz"
    ensure_dir(residual_dir)
    npz_path = residual_dir / f"{slug}.npz"
    np.savez_compressed(
        npz_path,
        sample_id=np.asarray(sample_id),
        freq_hz=diagnostics["freq_hz"],
        time_s=diagnostics["time_s"],
        center_power_db=diagnostics["center_power_db"],
        raw_power=diagnostics["products"]["raw"]["component_power"],
        background_power_db_plane=diagnostics["background_power_db_plane"],
        residual_db_plane=diagnostics["residual_db_plane"],
        excess_power_plane=diagnostics["excess_power_plane"],
        background_power_db=diagnostics["background_power_db_plane"],
        residual_db=diagnostics["residual_db_plane"],
        excess_power=diagnostics["excess_power_plane"],
        xy=diagnostics["xy"],
        point_band_db=diagnostics["point_band_db"],
        background_indices=diagnostics["background_indices"],
        point_weights=diagnostics["point_weights"],
        selected_method=np.asarray("plane"),
        primary_method=np.asarray("plane"),
        plane_valid=np.asarray(diagnostics["plane_valid"]),
        fit_plane_available=np.asarray(diagnostics["plane_valid"]),
        plane_geometry_ok=np.asarray(diagnostics["plane_geometry_ok"]),
        plane_solver=np.asarray(diagnostics["plane_solver"]),
        plane_reject_reason=np.asarray(diagnostics["plane_reject_reason"]),
        plane_used_by_auto=np.asarray(diagnostics["plane_used_by_auto"]),
        fit_condition_number=np.asarray(diagnostics["fit_condition_number"]),
        fit_median_mae_db=np.asarray(diagnostics["fit_median_mae_db"]),
        fit_plane_mae_db=np.asarray(diagnostics["fit_plane_mae_db"]),
        fit_plane_improvement_db=np.asarray(diagnostics["fit_plane_improvement_db"]),
        fit_plane_improvement_ratio=np.asarray(diagnostics["fit_plane_improvement_ratio"]),
        available_methods=np.asarray(ALL_METHODS),
    )

    for method in ALL_METHODS:
        method_path = output_dir / "methods" / method / "residual_npz" / f"{slug}.npz"
        _save_one_method_npz(method_path, sample_id, method, diagnostics)

    return fig_path, npz_path

def _write_method_feature_tables(scene_dir: Path) -> None:
    all_path = scene_dir / "v9_all_features.csv"
    if not all_path.exists():
        return
    df = pd.read_csv(all_path, dtype={"sample_id": str, "center": str, "time": str})
    metadata = [
        c for c in [
            "sample_id", "dataset", "scene", "time", "center", "label",
            "center_file", "offset_dir", "sample_rate_hz", "selected_method",
            "primary_method", "plane_valid", "plane_geometry_ok", "plane_solver",
            "plane_reject_reason", "fit_condition_number", "fit_median_mae_db",
            "fit_plane_mae_db", "fit_plane_improvement_db",
            "fit_plane_improvement_ratio",
        ] if c in df.columns
    ]
    for method in ALL_METHODS:
        method_dir = scene_dir / "methods" / method
        ensure_dir(method_dir)
        method_cols = [c for c in df.columns if c.startswith(method + "_")]
        table = df[metadata + method_cols].copy()
        table.insert(len(metadata), "method", method)
        table.to_csv(method_dir / "v9_method_features.csv", index=False, encoding="utf-8-sig")


def validate_abcd_datasets(datasets: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    validated = [validate_dataset_config(x) for x in datasets]
    scenes = sorted({x["scene"] for x in validated})
    if len(scenes) != 4:
        raise ValueError(f"必须配置正好4个场景A/B/C/D，当前为: {scenes}")
    for scene in scenes:
        labels = {x["label"] for x in validated if x["scene"] == scene}
        missing = {"TRUE_LEAK", "FALSE_LEAK"} - labels
        if missing:
            raise ValueError(f"场景 {scene} 缺少标签数据: {sorted(missing)}")
    return validated


def process_abcd_all_methods(
    datasets: Sequence[Dict[str, Any]],
    output_root: Path,
    config: V9Config,
) -> Dict[str, Path]:
    validated = validate_abcd_datasets(datasets)
    ensure_dir(output_root)
    scene_names = sorted({x["scene"] for x in validated})
    scene_manifest: List[Dict[str, Any]] = []
    all_features: List[pd.DataFrame] = []
    all_failures: List[pd.DataFrame] = []

    print("=" * 100)
    print("v9-ABCD：四场景 × raw/plane 双方法输出")
    print("=" * 100)
    print("输出根目录:", output_root)

    for scene in scene_names:
        scene_datasets = [x for x in validated if x["scene"] == scene]
        scene_dir = output_root / safe_slug(scene)
        print("\n" + "#" * 100)
        print(f"开始场景 {scene} → {scene_dir}")
        result_paths = _legacy_process_all_datasets(scene_datasets, scene_dir, config)
        _write_method_feature_tables(scene_dir)

        feature_path = scene_dir / "v9_all_features.csv"
        failure_path = scene_dir / "v9_failures.csv"
        feature_df = pd.read_csv(feature_path) if feature_path.exists() else pd.DataFrame()
        failure_df = pd.read_csv(failure_path) if failure_path.exists() else pd.DataFrame()
        if len(feature_df):
            all_features.append(feature_df)
        if len(failure_df):
            all_failures.append(failure_df)

        method_counts: Dict[str, int] = {}
        for method in ALL_METHODS:
            method_counts[method] = len(list((scene_dir / "methods" / method / "residual_npz").glob("*.npz")))
        scene_manifest.append({
            "scene": scene,
            "scene_dir": str(scene_dir),
            "n_samples": int(len(feature_df)),
            "method_npz_counts": method_counts,
            "result_paths": {k: str(v) for k, v in result_paths.items()},
        })

    global_features = pd.concat(all_features, ignore_index=True) if all_features else pd.DataFrame()
    global_failures = pd.concat(all_failures, ignore_index=True) if all_failures else pd.DataFrame()
    global_features.to_csv(output_root / "v9_ABCD_all_features.csv", index=False, encoding="utf-8-sig")
    global_failures.to_csv(output_root / "v9_ABCD_failures.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "version": "v9_ABCD_raw_plane",
        "available_methods": list(ALL_METHODS),
        "output_layout": {
            "consolidated_npz": "<scene>/residual_npz/<sample>.npz",
            "method_npz": "<scene>/methods/<raw|plane>/residual_npz/<sample>.npz",
            "method_features": "<scene>/methods/<method>/v9_method_features.csv",
        },
        "scenes": scene_manifest,
        "config": asdict(config),
    }
    manifest_path = output_root / "v9_ABCD_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 100)
    print("全部四场景处理完成")
    print("总清单:", manifest_path)
    print("全场景特征:", output_root / "v9_ABCD_all_features.csv")
    print("=" * 100)
    return {
        "manifest": manifest_path,
        "all_features": output_root / "v9_ABCD_all_features.csv",
        "failures": output_root / "v9_ABCD_failures.csv",
    }


def load_abcd_config(path: str) -> Tuple[List[Dict[str, Any]], Path, V9Config]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    datasets = payload.get("datasets", [])
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("配置文件必须包含非空 datasets 列表")
    output_dir = Path(str(payload.get("output_dir", OUTPUT_DIR)))
    base = V9Config(save_residual_npz=True)
    overrides = dict(payload.get("algorithm", {}))
    valid_fields = set(asdict(base))
    unknown = sorted(set(overrides) - valid_fields)
    if unknown:
        raise ValueError(f"未知algorithm参数: {unknown}")

    # 本版本固定只计算 50-70 kHz。即使旧配置文件仍写着 20-80 kHz，
    # 也会在这里强制改为 50-70 kHz，避免误把其他频率算进去。
    overrides["freq_low_hz"] = 50_000.0
    overrides["freq_high_hz"] = 70_000.0
    config = V9Config(**{**asdict(base), **overrides, "save_residual_npz": True})
    return datasets, output_dir, config


def run_self_test(config: V9Config) -> None:
    import tempfile
    print("开始 v9-ABCD raw/plane 合成自检...")
    rng = np.random.default_rng(config.random_state)
    sample = make_synthetic_cube(rng, True, 0, config)
    result = extract_features_from_cube(sample, config)
    required = {
        "center_power_db", "excess_power_plane", "plane_valid",
    }
    missing = sorted(required - set(result["diagnostics"]))
    if missing:
        raise AssertionError(f"自检缺少诊断字段: {missing}")
    with tempfile.TemporaryDirectory(prefix="v9_all_methods_") as td:
        root = Path(td)
        _, consolidated = save_diagnostic_outputs("self_test", result["diagnostics"], root, True)
        if consolidated is None or not consolidated.exists():
            raise AssertionError("未生成合并NPZ")
        with np.load(consolidated, allow_pickle=False) as npz:
            needed_keys = {
                "center_power_db", "excess_power_plane", "available_methods",
            }
            if not needed_keys.issubset(set(npz.files)):
                raise AssertionError(f"合并NPZ字段不完整: {needed_keys - set(npz.files)}")
        for method in ALL_METHODS:
            p = root / "methods" / method / "residual_npz" / "self_test.npz"
            if not p.exists():
                raise AssertionError(f"未生成方法文件: {p}")
    print("v9-ABCD raw/plane自检通过：raw和plane均已计算并保存。")


# ============================================================================
# 13. CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="v9-ABCD：A/B/C/D四场景同时计算raw和plane（固定50-70 kHz）"
    )
    parser.add_argument("--config", type=str, default="v9_ABCD_raw_plane_config.json")
    parser.add_argument("--self-test", action="store_true", help="运行合成数据自检")
    parser.add_argument("--no-plots", action="store_true", help="不保存诊断图")
    parser.add_argument("--max-plots", type=int, default=None, help="每个场景最多保存多少张图")
    parser.add_argument("--max-centers", type=int, default=None, help="调试：每个time最多处理多少中心")
    return parser.parse_args()


def main() -> None:
    global MAX_CENTERS_PER_TIME
    args = parse_args()
    if args.self_test:
        cfg = V9Config(save_residual_npz=True, save_diagnostic_plots=False)
        run_self_test(cfg)
        return

    datasets, output_dir, config = load_abcd_config(args.config)
    updates: Dict[str, Any] = {}
    if args.no_plots:
        updates["save_diagnostic_plots"] = False
    if args.max_plots is not None:
        updates["diagnostic_plot_limit"] = max(0, int(args.max_plots))
    if updates:
        config = replace(config, **updates)
    if args.max_centers is not None:
        MAX_CENTERS_PER_TIME = max(1, int(args.max_centers))
    process_abcd_all_methods(datasets, output_dir, config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中止。")
        sys.exit(130)
    except Exception as exc:
        print("\n程序失败:", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)
