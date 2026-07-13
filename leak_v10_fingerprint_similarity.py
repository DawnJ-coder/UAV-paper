# -*- coding: utf-8 -*-
"""
leak_v10_fingerprint_similarity.py

v10：泄漏残差指纹建立与相似度验证
====================================

用途
----
读取 v9 输出的：
    v9_all_features.csv
    residual_npz/*.npz

不再直接使用绝对能量训练复杂分类器，而是为每个样本构造：
    1. 频谱残差指纹：去除整体幅值影响，比较残差频谱形状；
    2. 时间持续性指纹：比较稳定泄漏与瞬态/脉冲噪声；
    3. 空间衰减指纹：比较中心局部峰值和径向衰减结构。

随后同时建立：
    TRUE_LEAK 指纹原型
    FALSE_LEAK 指纹原型

对每个待测样本计算：
    similarity_TRUE
    similarity_FALSE
    margin = similarity_TRUE - similarity_FALSE

主要运行方式
------------
1. A场景内部验证并建立最终冻结指纹：
       python leak_v10_fingerprint_similarity.py

2. 自检：
       python leak_v10_fingerprint_similarity.py --self-test

3. 指定 v9 结果目录：
       python leak_v10_fingerprint_similarity.py --v9-dir "D:\\xxx\\leak_v9_local_background_results"

4. 使用已冻结的A场景指纹，验证另一个v9结果目录：
       python leak_v10_fingerprint_similarity.py ^
           --mode external ^
           --v9-dir "D:\\factory_B\\leak_v9_local_background_results" ^
           --prototype-file "D:\\factory_A\\leak_v10_fingerprint_results\\v10_frozen_reference.npz"

重要说明
--------
- 默认优先按完整 time 分组交叉验证，避免同一次采集的不同 center 同时进入建模和验证。
- 如果每个类别没有至少两个独立 time，程序只能退回分层样本交叉验证，并在报告中明确警告。
- v10 的目标是验证“残差指纹是否可重复”，不是证明唯一、绝对的气体泄漏本质。
- 外部场景验证时不会重建原型、不会重新选阈值。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr, wasserstein_distance
except Exception as exc:
    raise RuntimeError("缺少 scipy，请运行: pip install scipy") from exc

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        roc_auc_score,
    )
    from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
except Exception as exc:
    raise RuntimeError("缺少 scikit-learn，请运行: pip install scikit-learn") from exc


# =============================================================================
# 1. 优先修改这里
# =============================================================================

V9_RESULT_DIR = r"C:\Users\jiangxinru6\Desktop\wurenji\leak_v9_local_background_results"
V10_OUTPUT_DIR = r"C:\Users\jiangxinru6\Desktop\wurenji\leak_v10_fingerprint_results"

# None：自动选择同时含 TRUE/FALSE 且样本最多的 scene。
# 明确知道A场景名称时，也可以写成 "factory_A"。
REFERENCE_SCENE: Optional[str] = None

# 频谱指纹统一到固定频率网格，方便不同采集批次比较。
FREQ_LOW_HZ = 20_000.0
FREQ_HIGH_HZ = 80_000.0
N_SPECTRAL_BINS = 256

# 时间指纹采用“帧残差分布的分位数曲线”，不要求不同文件时间严格对齐。
N_TEMPORAL_QUANTILES = 64

# 空间指纹：中心 + 4个归一化距离环 + 两个结构指标。
N_SPATIAL_RINGS = 4
SPATIAL_DB_SCALE = 6.0

# 三类指纹在总相似度中的权重。
SPECTRAL_WEIGHT = 1.0
TEMPORAL_WEIGHT = 1.0
SPATIAL_WEIGHT = 1.0

# 类内参考集合相似度：取最相似的前几个参考样本求均值。
TOP_K_REFERENCE_SIMILARITY = 3

# 最终类别相似度 = 原型相似度与类内参考集合相似度的加权平均。
PROTOTYPE_SIMILARITY_WEIGHT = 0.5
REFERENCE_SET_SIMILARITY_WEIGHT = 0.5

# 频谱轻微平滑，避免单个FFT频点抖动。
SPECTRAL_SMOOTH_BINS = 5

# 数据量太大时，成对相似度矩阵最多展示多少个样本。
MAX_PAIRWISE_SAMPLES = 250

RANDOM_STATE = 42
VALID_LABELS = {"TRUE_LEAK", "FALSE_LEAK"}


# =============================================================================
# 2. 数据结构
# =============================================================================


@dataclass(frozen=True)
class V10Config:
    freq_low_hz: float = FREQ_LOW_HZ
    freq_high_hz: float = FREQ_HIGH_HZ
    n_spectral_bins: int = N_SPECTRAL_BINS
    n_temporal_quantiles: int = N_TEMPORAL_QUANTILES
    n_spatial_rings: int = N_SPATIAL_RINGS
    spatial_db_scale: float = SPATIAL_DB_SCALE
    spectral_weight: float = SPECTRAL_WEIGHT
    temporal_weight: float = TEMPORAL_WEIGHT
    spatial_weight: float = SPATIAL_WEIGHT
    top_k_reference_similarity: int = TOP_K_REFERENCE_SIMILARITY
    prototype_similarity_weight: float = PROTOTYPE_SIMILARITY_WEIGHT
    reference_set_similarity_weight: float = REFERENCE_SET_SIMILARITY_WEIGHT
    spectral_smooth_bins: int = SPECTRAL_SMOOTH_BINS
    max_pairwise_samples: int = MAX_PAIRWISE_SAMPLES
    random_state: int = RANDOM_STATE


@dataclass
class Fingerprint:
    sample_id: str
    dataset: str
    scene: str
    time: str
    center: str
    label: str
    npz_path: str
    spectral: np.ndarray
    temporal: np.ndarray
    spatial: np.ndarray


@dataclass
class ReferenceBundle:
    reference_scene: str
    threshold: float
    frequency_grid_hz: np.ndarray
    true_spectral_prototype: np.ndarray
    false_spectral_prototype: np.ndarray
    true_temporal_prototype: np.ndarray
    false_temporal_prototype: np.ndarray
    true_spatial_prototype: np.ndarray
    false_spatial_prototype: np.ndarray
    reference_spectral: np.ndarray
    reference_temporal: np.ndarray
    reference_spatial: np.ndarray
    reference_labels: np.ndarray
    reference_sample_ids: np.ndarray
    config: V10Config


# =============================================================================
# 3. 通用函数
# =============================================================================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_slug(value: Any, max_len: int = 150) -> str:
    text = str(value).strip() or "sample"
    text = re.sub(r"[^0-9A-Za-z_\-.]+", "_", text)
    return text.strip("._")[:max_len] or "sample"


def safe_numeric_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_probability(values: np.ndarray, eps: float = 1.0e-15) -> np.ndarray:
    arr = np.maximum(safe_numeric_array(values).ravel(), 0.0)
    total = float(np.sum(arr))
    if total <= eps:
        if arr.size == 0:
            return arr
        return np.full(arr.shape, 1.0 / arr.size, dtype=float)
    return arr / total


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    arr = safe_numeric_array(values).ravel()
    if window <= 1 or arr.size < 3:
        return arr
    window = int(max(1, min(window, arr.size)))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(arr, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def power_from_db(power_db: np.ndarray) -> np.ndarray:
    arr = np.clip(safe_numeric_array(power_db), -300.0, 300.0)
    return np.power(10.0, arr / 10.0)


def cosine_similarity_01(a: np.ndarray, b: np.ndarray) -> float:
    x = safe_numeric_array(a).ravel()
    y = safe_numeric_array(b).ravel()
    if x.shape != y.shape or x.size == 0:
        return 0.0
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1.0e-15:
        return 1.0 if np.linalg.norm(x - y) <= 1.0e-15 else 0.0
    value = float(np.dot(x, y) / denom)
    return float(np.clip(value, 0.0, 1.0))


def l1_similarity_01(a: np.ndarray, b: np.ndarray) -> float:
    x = safe_numeric_array(a).ravel()
    y = safe_numeric_array(b).ravel()
    if x.shape != y.shape or x.size == 0:
        return 0.0
    return float(np.clip(1.0 - np.mean(np.abs(x - y)), 0.0, 1.0))


def jensen_shannon_similarity(p: np.ndarray, q: np.ndarray) -> float:
    p = normalize_probability(p)
    q = normalize_probability(q)
    if p.shape != q.shape or p.size == 0:
        return 0.0
    m = 0.5 * (p + q)
    eps = 1.0e-15
    kl_pm = np.sum(np.where(p > 0, p * np.log2((p + eps) / (m + eps)), 0.0))
    kl_qm = np.sum(np.where(q > 0, q * np.log2((q + eps) / (m + eps)), 0.0))
    js_divergence = float(np.clip(0.5 * (kl_pm + kl_qm), 0.0, 1.0))
    return float(np.clip(1.0 - math.sqrt(js_divergence), 0.0, 1.0))


def spectral_similarity(
    a: np.ndarray,
    b: np.ndarray,
    frequency_grid_hz: np.ndarray,
) -> Dict[str, float]:
    p = normalize_probability(a)
    q = normalize_probability(b)
    cosine = cosine_similarity_01(p, q)
    js = jensen_shannon_similarity(p, q)

    x = safe_numeric_array(frequency_grid_hz).ravel()
    if x.size != p.size or x.size < 2:
        transport = 0.0
    else:
        x_norm = (x - x.min()) / max(float(x.max() - x.min()), 1.0e-15)
        distance = float(wasserstein_distance(x_norm, x_norm, u_weights=p, v_weights=q))
        transport = float(np.clip(1.0 - distance, 0.0, 1.0))

    overall = float(np.mean([cosine, js, transport]))
    return {
        "cosine": cosine,
        "js": js,
        "transport": transport,
        "overall": overall,
    }


def bounded_shape_similarity(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    cosine = cosine_similarity_01(a, b)
    l1 = l1_similarity_01(a, b)
    return {"cosine": cosine, "l1": l1, "overall": float(0.5 * (cosine + l1))}


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    vals = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(vals) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return 0.0
    return float(np.sum(vals[mask] * w[mask]) / np.sum(w[mask]))


def median_prototype(vectors: Sequence[np.ndarray], probability: bool = False) -> np.ndarray:
    if not vectors:
        raise ValueError("无法从空样本集合建立指纹原型")
    matrix = np.vstack([safe_numeric_array(x).ravel() for x in vectors])
    proto = np.median(matrix, axis=0)
    return normalize_probability(proto) if probability else np.clip(proto, 0.0, 1.0)


def label_to_binary(labels: Iterable[str]) -> np.ndarray:
    return np.asarray([1 if str(x) == "TRUE_LEAK" else 0 for x in labels], dtype=int)


def confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.size == 0:
        return {
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = float((tp + tn) / max(tp + tn + fp + fn, 1))
    # 单个time测试折有时只含一个类别，此时平衡准确率没有完整二分类含义，记为NA。
    if np.any(y_true == 0) and np.any(y_true == 1):
        recall_false = tn / max(tn + fp, 1)
        recall_true = tp / max(tp + fn, 1)
        balanced = float(0.5 * (recall_false + recall_true))
    else:
        balanced = np.nan
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, score))
    except Exception:
        return np.nan


def find_best_margin_threshold(y_true: np.ndarray, margins: np.ndarray) -> Tuple[float, pd.DataFrame]:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(margins, dtype=float)
    unique_scores = np.unique(score[np.isfinite(score)])
    if unique_scores.size == 0:
        return 0.0, pd.DataFrame()

    if unique_scores.size == 1:
        candidates = np.asarray([0.0, unique_scores[0]], dtype=float)
    else:
        mids = 0.5 * (unique_scores[:-1] + unique_scores[1:])
        candidates = np.unique(
            np.concatenate(
                [
                    [unique_scores[0] - 1.0e-9],
                    mids,
                    [unique_scores[-1] + 1.0e-9, 0.0],
                ]
            )
        )

    rows: List[Dict[str, Any]] = []
    best_threshold = 0.0
    best_key = (-np.inf, -np.inf, -np.inf)
    for threshold in candidates:
        pred = (score >= threshold).astype(int)
        metrics = confusion_metrics(y, pred)
        # 先最大化平衡准确率，再最大化普通准确率；并优先选择更接近自然阈值0的值。
        key = (
            metrics["balanced_accuracy"],
            metrics["accuracy"],
            -abs(float(threshold)),
        )
        rows.append({"threshold": float(threshold), **metrics})
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)

    return best_threshold, pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


# =============================================================================
# 4. 读取 v9 数据并构造指纹
# =============================================================================


def locate_npz_file(residual_dir: Path, sample_id: str) -> Optional[Path]:
    direct = residual_dir / f"{safe_slug(sample_id)}.npz"
    if direct.exists():
        return direct

    # 容错：文件可能因较旧版本命名略有不同。
    matches = list(residual_dir.glob(f"*{safe_slug(sample_id)}*.npz"))
    if len(matches) == 1:
        return matches[0]
    return None


def build_spectral_fingerprint(
    freq_hz: np.ndarray,
    excess_power: np.ndarray,
    config: V10Config,
) -> Tuple[np.ndarray, np.ndarray]:
    freq = safe_numeric_array(freq_hz).ravel()
    excess = safe_numeric_array(excess_power)
    if excess.ndim != 2 or excess.shape[0] != freq.size:
        raise ValueError(f"excess_power维度异常: freq={freq.shape}, excess={excess.shape}")

    spectrum = np.mean(np.maximum(excess, 0.0), axis=1)
    # 开平方压缩极强窄带峰的支配作用，同时保留非负物理意义。
    spectrum = np.sqrt(np.maximum(spectrum, 0.0))
    spectrum = moving_average(spectrum, config.spectral_smooth_bins)

    grid = np.linspace(config.freq_low_hz, config.freq_high_hz, config.n_spectral_bins)
    valid = np.isfinite(freq) & np.isfinite(spectrum)
    if np.sum(valid) < 2:
        raise ValueError("有效频率点不足，无法建立频谱指纹")

    order = np.argsort(freq[valid])
    interp = np.interp(
        grid,
        freq[valid][order],
        spectrum[valid][order],
        left=0.0,
        right=0.0,
    )
    return normalize_probability(interp), grid


def build_temporal_fingerprint(
    excess_power: np.ndarray,
    background_power_db: np.ndarray,
    config: V10Config,
) -> np.ndarray:
    excess = np.maximum(safe_numeric_array(excess_power), 0.0)
    bg_db = safe_numeric_array(background_power_db)
    if excess.ndim != 2 or bg_db.shape != excess.shape:
        raise ValueError(
            f"时间指纹维度异常: excess={excess.shape}, background={bg_db.shape}"
        )

    bg_power = power_from_db(bg_db)
    excess_frame = np.sum(excess, axis=0)
    bg_frame = np.sum(bg_power, axis=0)
    ratio = excess_frame / (bg_frame + 1.0e-20)
    ratio = np.maximum(np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

    log_ratio = np.log1p(ratio)
    q_grid = np.linspace(0.0, 1.0, config.n_temporal_quantiles)
    quantile_curve = np.quantile(log_ratio, q_grid)

    # 每个样本内部按高分位缩放，重点比较“稳定/瞬态形态”，弱化整体幅值。
    scale = float(np.quantile(quantile_curve, 0.95))
    if scale <= 1.0e-12:
        scaled_curve = np.zeros_like(quantile_curve)
    else:
        scaled_curve = np.clip(quantile_curve / scale, 0.0, 1.5) / 1.5

    # 添加少量相对时间结构指标，全部限制在[0,1]。
    q90_raw = float(np.quantile(ratio, 0.90))
    relative_scale = max(q90_raw, 1.0e-20)
    active_10 = float(np.mean(ratio >= 0.10 * relative_scale))
    active_30 = float(np.mean(ratio >= 0.30 * relative_scale))
    active_50 = float(np.mean(ratio >= 0.50 * relative_scale))
    cv = float(np.std(ratio) / (np.mean(ratio) + 1.0e-20))
    cv_bounded = float(cv / (1.0 + cv))

    return np.concatenate(
        [scaled_curve, np.asarray([active_10, active_30, active_50, cv_bounded])]
    )


def _fill_missing_ring_values(values: np.ndarray) -> np.ndarray:
    arr = safe_numeric_array(values).ravel()
    valid = np.isfinite(arr)
    if np.all(valid):
        return arr
    if not np.any(valid):
        return np.zeros_like(arr)
    idx = np.arange(arr.size)
    return np.interp(idx, idx[valid], arr[valid])


def build_spatial_fingerprint(
    xy: np.ndarray,
    point_band_db: np.ndarray,
    background_indices: np.ndarray,
    config: V10Config,
) -> np.ndarray:
    coords = safe_numeric_array(xy)
    values = safe_numeric_array(point_band_db).ravel()
    bg_idx = np.asarray(background_indices, dtype=int).ravel()

    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] != values.size:
        raise ValueError(f"空间指纹维度异常: xy={coords.shape}, point={values.shape}")

    distances = np.linalg.norm(coords, axis=1)
    center_index = int(np.argmin(distances))
    neighbor_mask = np.arange(values.size) != center_index
    if np.sum(neighbor_mask) < config.n_spatial_rings:
        raise ValueError("周围空间点太少，无法建立空间指纹")

    bg_idx = bg_idx[(bg_idx >= 0) & (bg_idx < values.size)]
    if bg_idx.size == 0:
        baseline = float(np.median(values[neighbor_mask]))
    else:
        baseline = float(np.median(values[bg_idx]))

    neighbor_dist = distances[neighbor_mask]
    max_radius = max(float(np.max(neighbor_dist)), 1.0e-12)
    normalized_r = neighbor_dist / max_radius
    neighbor_values = values[neighbor_mask]

    edges = np.linspace(0.0, 1.0, config.n_spatial_rings + 1)
    ring_values: List[float] = []
    for i in range(config.n_spatial_rings):
        left = edges[i]
        right = edges[i + 1]
        if i == config.n_spatial_rings - 1:
            mask = (normalized_r >= left) & (normalized_r <= right + 1.0e-12)
        else:
            mask = (normalized_r >= left) & (normalized_r < right)
        ring_values.append(float(np.median(neighbor_values[mask])) if np.any(mask) else np.nan)

    ring_arr = _fill_missing_ring_values(np.asarray(ring_values, dtype=float))
    profile_db = np.concatenate([[values[center_index]], ring_arr]) - baseline

    # 将相对dB平滑映射到[0,1]，不依赖原始绝对声压。
    mapped_profile = 0.5 + 0.5 * np.tanh(profile_db / max(config.spatial_db_scale, 1.0e-6))

    center_rank = float(np.mean(values <= values[center_index]))
    try:
        corr, _ = spearmanr(neighbor_dist, neighbor_values)
        corr = float(corr) if np.isfinite(corr) else 0.0
    except Exception:
        corr = 0.0
    # 负相关代表向外衰减，将其映射为较大的“衰减度”。
    radial_decay = float(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))

    return np.clip(
        np.concatenate([mapped_profile, np.asarray([center_rank, radial_decay])]),
        0.0,
        1.0,
    )


def load_fingerprints(
    v9_dir: Path,
    config: V10Config,
) -> Tuple[List[Fingerprint], pd.DataFrame, pd.DataFrame]:
    csv_path = v9_dir / "v9_all_features.csv"
    residual_dir = v9_dir / "residual_npz"

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到: {csv_path}")
    if not residual_dir.exists():
        raise FileNotFoundError(
            f"找不到 residual_npz 文件夹: {residual_dir}\n"
            "请先运行：python leak_v9_local_background_residual_wav.py --save-residual-npz"
        )

    metadata = pd.read_csv(csv_path, dtype={"center": str})
    required_cols = ["sample_id", "dataset", "scene", "time", "center", "label"]
    missing_cols = [c for c in required_cols if c not in metadata.columns]
    if missing_cols:
        raise ValueError(f"v9_all_features.csv 缺少列: {missing_cols}")

    fingerprints: List[Fingerprint] = []
    failures: List[Dict[str, Any]] = []
    frequency_grid: Optional[np.ndarray] = None

    for _, row in metadata.iterrows():
        sample_id = str(row["sample_id"])
        npz_path = locate_npz_file(residual_dir, sample_id)
        if npz_path is None:
            failures.append(
                {
                    "sample_id": sample_id,
                    "error_type": "MissingNPZ",
                    "error": "找不到与sample_id对应的NPZ文件",
                    "traceback": "",
                }
            )
            continue

        try:
            with np.load(npz_path, allow_pickle=False) as data:
                required_keys = [
                    "freq_hz",
                    "background_power_db",
                    "excess_power",
                    "xy",
                    "point_band_db",
                    "background_indices",
                ]
                missing_keys = [k for k in required_keys if k not in data.files]
                if missing_keys:
                    raise KeyError(f"NPZ缺少字段: {missing_keys}")

                spectral, grid = build_spectral_fingerprint(
                    data["freq_hz"], data["excess_power"], config
                )
                temporal = build_temporal_fingerprint(
                    data["excess_power"], data["background_power_db"], config
                )
                spatial = build_spatial_fingerprint(
                    data["xy"],
                    data["point_band_db"],
                    data["background_indices"],
                    config,
                )

            if frequency_grid is None:
                frequency_grid = grid
            elif not np.allclose(frequency_grid, grid):
                raise ValueError("不同样本生成的统一频率网格不一致")

            fingerprints.append(
                Fingerprint(
                    sample_id=sample_id,
                    dataset=str(row["dataset"]),
                    scene=str(row["scene"]),
                    time=str(row["time"]),
                    center=str(row["center"]),
                    label=str(row["label"]),
                    npz_path=str(npz_path),
                    spectral=spectral,
                    temporal=temporal,
                    spatial=spatial,
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "sample_id": sample_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    failure_df = pd.DataFrame(
        failures, columns=["sample_id", "error_type", "error", "traceback"]
    )

    if not fingerprints:
        raise RuntimeError("没有成功读取任何v9残差NPZ，请检查v10_fingerprint_failures.csv")

    summary_rows = []
    for fp in fingerprints:
        summary_rows.append(
            {
                "sample_id": fp.sample_id,
                "dataset": fp.dataset,
                "scene": fp.scene,
                "time": fp.time,
                "center": fp.center,
                "label": fp.label,
                "npz_path": fp.npz_path,
                "spectral_length": len(fp.spectral),
                "temporal_length": len(fp.temporal),
                "spatial_length": len(fp.spatial),
            }
        )
    fingerprint_index_df = pd.DataFrame(summary_rows)
    return fingerprints, fingerprint_index_df, failure_df


# =============================================================================
# 5. 指纹原型与相似度
# =============================================================================


def build_class_prototypes(
    samples: Sequence[Fingerprint],
) -> Dict[str, Dict[str, np.ndarray]]:
    output: Dict[str, Dict[str, np.ndarray]] = {}
    for label in ["TRUE_LEAK", "FALSE_LEAK"]:
        subset = [fp for fp in samples if fp.label == label]
        if not subset:
            raise ValueError(f"建立原型时缺少类别: {label}")
        output[label] = {
            "spectral": median_prototype([fp.spectral for fp in subset], probability=True),
            "temporal": median_prototype([fp.temporal for fp in subset], probability=False),
            "spatial": median_prototype([fp.spatial for fp in subset], probability=False),
        }
    return output


def compare_fingerprint_blocks(
    sample: Fingerprint,
    target_spectral: np.ndarray,
    target_temporal: np.ndarray,
    target_spatial: np.ndarray,
    frequency_grid_hz: np.ndarray,
    config: V10Config,
) -> Dict[str, float]:
    spec = spectral_similarity(sample.spectral, target_spectral, frequency_grid_hz)
    temporal = bounded_shape_similarity(sample.temporal, target_temporal)
    spatial = bounded_shape_similarity(sample.spatial, target_spatial)
    overall = weighted_mean(
        [spec["overall"], temporal["overall"], spatial["overall"]],
        [config.spectral_weight, config.temporal_weight, config.spatial_weight],
    )
    return {
        "spectral_cosine": spec["cosine"],
        "spectral_js": spec["js"],
        "spectral_transport": spec["transport"],
        "spectral": spec["overall"],
        "temporal_cosine": temporal["cosine"],
        "temporal_l1": temporal["l1"],
        "temporal": temporal["overall"],
        "spatial_cosine": spatial["cosine"],
        "spatial_l1": spatial["l1"],
        "spatial": spatial["overall"],
        "overall": overall,
    }


def pair_similarity(
    sample_a: Fingerprint,
    sample_b: Fingerprint,
    frequency_grid_hz: np.ndarray,
    config: V10Config,
) -> Dict[str, float]:
    return compare_fingerprint_blocks(
        sample_a,
        sample_b.spectral,
        sample_b.temporal,
        sample_b.spatial,
        frequency_grid_hz,
        config,
    )


def top_k_reference_similarity(
    sample: Fingerprint,
    references: Sequence[Fingerprint],
    frequency_grid_hz: np.ndarray,
    config: V10Config,
) -> float:
    if not references:
        return 0.0
    values = [
        pair_similarity(sample, ref, frequency_grid_hz, config)["overall"]
        for ref in references
    ]
    values = sorted(values, reverse=True)
    k = max(1, min(config.top_k_reference_similarity, len(values)))
    return float(np.mean(values[:k]))


def evaluate_sample_against_reference(
    sample: Fingerprint,
    training_samples: Sequence[Fingerprint],
    prototypes: Dict[str, Dict[str, np.ndarray]],
    frequency_grid_hz: np.ndarray,
    config: V10Config,
) -> Dict[str, Any]:
    true_proto = compare_fingerprint_blocks(
        sample,
        prototypes["TRUE_LEAK"]["spectral"],
        prototypes["TRUE_LEAK"]["temporal"],
        prototypes["TRUE_LEAK"]["spatial"],
        frequency_grid_hz,
        config,
    )
    false_proto = compare_fingerprint_blocks(
        sample,
        prototypes["FALSE_LEAK"]["spectral"],
        prototypes["FALSE_LEAK"]["temporal"],
        prototypes["FALSE_LEAK"]["spatial"],
        frequency_grid_hz,
        config,
    )

    true_refs = [x for x in training_samples if x.label == "TRUE_LEAK"]
    false_refs = [x for x in training_samples if x.label == "FALSE_LEAK"]
    true_set = top_k_reference_similarity(sample, true_refs, frequency_grid_hz, config)
    false_set = top_k_reference_similarity(sample, false_refs, frequency_grid_hz, config)

    p_weight = max(config.prototype_similarity_weight, 0.0)
    r_weight = max(config.reference_set_similarity_weight, 0.0)
    denom = max(p_weight + r_weight, 1.0e-12)
    sim_true = float((p_weight * true_proto["overall"] + r_weight * true_set) / denom)
    sim_false = float((p_weight * false_proto["overall"] + r_weight * false_set) / denom)

    return {
        "similarity_TRUE_spectral": true_proto["spectral"],
        "similarity_TRUE_spectral_cosine": true_proto["spectral_cosine"],
        "similarity_TRUE_spectral_js": true_proto["spectral_js"],
        "similarity_TRUE_spectral_transport": true_proto["spectral_transport"],
        "similarity_TRUE_temporal": true_proto["temporal"],
        "similarity_TRUE_spatial": true_proto["spatial"],
        "similarity_TRUE_prototype_overall": true_proto["overall"],
        "similarity_TRUE_reference_set": true_set,
        "similarity_TRUE": sim_true,
        "similarity_FALSE_spectral": false_proto["spectral"],
        "similarity_FALSE_spectral_cosine": false_proto["spectral_cosine"],
        "similarity_FALSE_spectral_js": false_proto["spectral_js"],
        "similarity_FALSE_spectral_transport": false_proto["spectral_transport"],
        "similarity_FALSE_temporal": false_proto["temporal"],
        "similarity_FALSE_spatial": false_proto["spatial"],
        "similarity_FALSE_prototype_overall": false_proto["overall"],
        "similarity_FALSE_reference_set": false_set,
        "similarity_FALSE": sim_false,
        "margin_spectral": true_proto["spectral"] - false_proto["spectral"],
        "margin_temporal": true_proto["temporal"] - false_proto["temporal"],
        "margin_spatial": true_proto["spatial"] - false_proto["spatial"],
        "similarity_margin": sim_true - sim_false,
    }


# =============================================================================
# 6. 内部交叉验证
# =============================================================================


def choose_reference_scene(
    fingerprints: Sequence[Fingerprint],
    requested_scene: Optional[str],
) -> str:
    scenes = sorted({fp.scene for fp in fingerprints})
    if requested_scene:
        if requested_scene not in scenes:
            raise ValueError(
                f"指定的REFERENCE_SCENE={requested_scene!r}不存在。可选scene: {scenes}"
            )
        subset = [fp for fp in fingerprints if fp.scene == requested_scene]
        labels = {fp.label for fp in subset}
        if not VALID_LABELS.issubset(labels):
            raise ValueError(f"参考scene {requested_scene}没有同时包含TRUE_LEAK和FALSE_LEAK")
        return requested_scene

    candidates: List[Tuple[int, int, str]] = []
    for scene in scenes:
        subset = [fp for fp in fingerprints if fp.scene == scene]
        n_true = sum(fp.label == "TRUE_LEAK" for fp in subset)
        n_false = sum(fp.label == "FALSE_LEAK" for fp in subset)
        if n_true > 0 and n_false > 0:
            candidates.append((min(n_true, n_false), len(subset), scene))
    if not candidates:
        raise ValueError("没有任何scene同时包含TRUE_LEAK和FALSE_LEAK，无法建立T/F指纹")
    candidates.sort(reverse=True)
    return candidates[0][2]


def make_cv_splits(
    samples: Sequence[Fingerprint],
    config: V10Config,
) -> Tuple[str, List[Tuple[np.ndarray, np.ndarray]], List[str]]:
    labels = np.asarray([fp.label for fp in samples], dtype=str)
    y = label_to_binary(labels)
    groups = np.asarray([fp.time for fp in samples], dtype=str)
    warnings: List[str] = []

    true_groups = set(groups[labels == "TRUE_LEAK"])
    false_groups = set(groups[labels == "FALSE_LEAK"])

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    if len(true_groups) >= 2 and len(false_groups) >= 2 and len(np.unique(groups)) >= 3:
        logo = LeaveOneGroupOut()
        for train_idx, test_idx in logo.split(np.arange(len(samples)), y, groups):
            if len(np.unique(y[train_idx])) < 2:
                continue
            splits.append((train_idx, test_idx))
        if splits:
            return "leave_one_time_out", splits, warnings

    min_class_count = int(min(np.sum(y == 0), np.sum(y == 1)))
    if min_class_count < 2:
        raise ValueError("TRUE_LEAK或FALSE_LEAK样本少于2个，无法做内部验证")
    n_splits = max(2, min(5, min_class_count))
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=config.random_state,
    )
    splits = list(cv.split(np.arange(len(samples)), y))
    warnings.append(
        "每个类别没有至少两个独立time，无法按time整组验证；当前退回分层样本交叉验证。"
        "同一time的不同center可能进入不同折，因此结果只能作为初步检查，不能证明跨时间泛化。"
    )
    return "stratified_sample_cv_fallback", splits, warnings


def run_internal_validation(
    reference_samples: Sequence[Fingerprint],
    frequency_grid_hz: np.ndarray,
    config: V10Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, List[str], str]:
    split_mode, splits, warnings = make_cv_splits(reference_samples, config)
    rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []

    for fold_index, (train_idx, test_idx) in enumerate(splits, start=1):
        train_samples = [reference_samples[i] for i in train_idx]
        test_samples = [reference_samples[i] for i in test_idx]
        prototypes = build_class_prototypes(train_samples)

        fold_result_rows: List[Dict[str, Any]] = []
        for sample in test_samples:
            result = evaluate_sample_against_reference(
                sample,
                train_samples,
                prototypes,
                frequency_grid_hz,
                config,
            )
            row = {
                "fold": fold_index,
                "split_mode": split_mode,
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "scene": sample.scene,
                "time": sample.time,
                "center": sample.center,
                "true_label": sample.label,
                "n_train_TRUE": sum(x.label == "TRUE_LEAK" for x in train_samples),
                "n_train_FALSE": sum(x.label == "FALSE_LEAK" for x in train_samples),
                **result,
            }
            rows.append(row)
            fold_result_rows.append(row)

        fold_df = pd.DataFrame(fold_result_rows)
        if not fold_df.empty:
            y_fold = label_to_binary(fold_df["true_label"])
            margin = fold_df["similarity_margin"].to_numpy(dtype=float)
            pred_zero = (margin >= 0.0).astype(int)
            metrics_zero = confusion_metrics(y_fold, pred_zero)
            fold_rows.append(
                {
                    "fold": fold_index,
                    "split_mode": split_mode,
                    "test_times": " | ".join(sorted(set(fold_df["time"].astype(str)))),
                    "n_test": len(fold_df),
                    "n_true": int(np.sum(y_fold == 1)),
                    "n_false": int(np.sum(y_fold == 0)),
                    "margin_auc": safe_auc(y_fold, margin),
                    "accuracy_margin_0": metrics_zero["accuracy"],
                    "balanced_accuracy_margin_0": metrics_zero["balanced_accuracy"],
                }
            )

    oof_df = pd.DataFrame(rows)
    fold_df = pd.DataFrame(fold_rows)
    if oof_df.empty:
        raise RuntimeError("内部验证没有生成任何OOF结果")

    y_true = label_to_binary(oof_df["true_label"])
    margins = oof_df["similarity_margin"].to_numpy(dtype=float)
    best_threshold, threshold_curve = find_best_margin_threshold(y_true, margins)

    oof_df["pred_margin_0"] = np.where(
        oof_df["similarity_margin"] >= 0.0, "TRUE_LEAK", "FALSE_LEAK"
    )
    oof_df["pred_calibrated"] = np.where(
        oof_df["similarity_margin"] >= best_threshold,
        "TRUE_LEAK",
        "FALSE_LEAK",
    )
    oof_df["correct_margin_0"] = (
        oof_df["pred_margin_0"] == oof_df["true_label"]
    ).astype(int)
    oof_df["correct_calibrated"] = (
        oof_df["pred_calibrated"] == oof_df["true_label"]
    ).astype(int)

    return oof_df, fold_df, threshold_curve, best_threshold, warnings, split_mode


# =============================================================================
# 7. 冻结参考指纹、保存和外部验证
# =============================================================================


def build_reference_bundle(
    reference_samples: Sequence[Fingerprint],
    frequency_grid_hz: np.ndarray,
    reference_scene: str,
    threshold: float,
    config: V10Config,
) -> ReferenceBundle:
    prototypes = build_class_prototypes(reference_samples)
    return ReferenceBundle(
        reference_scene=reference_scene,
        threshold=float(threshold),
        frequency_grid_hz=frequency_grid_hz,
        true_spectral_prototype=prototypes["TRUE_LEAK"]["spectral"],
        false_spectral_prototype=prototypes["FALSE_LEAK"]["spectral"],
        true_temporal_prototype=prototypes["TRUE_LEAK"]["temporal"],
        false_temporal_prototype=prototypes["FALSE_LEAK"]["temporal"],
        true_spatial_prototype=prototypes["TRUE_LEAK"]["spatial"],
        false_spatial_prototype=prototypes["FALSE_LEAK"]["spatial"],
        reference_spectral=np.vstack([fp.spectral for fp in reference_samples]),
        reference_temporal=np.vstack([fp.temporal for fp in reference_samples]),
        reference_spatial=np.vstack([fp.spatial for fp in reference_samples]),
        reference_labels=np.asarray([fp.label for fp in reference_samples], dtype="U32"),
        reference_sample_ids=np.asarray([fp.sample_id for fp in reference_samples], dtype="U256"),
        config=config,
    )


def save_reference_bundle(bundle: ReferenceBundle, output_dir: Path) -> Tuple[Path, Path]:
    ensure_dir(output_dir)
    npz_path = output_dir / "v10_frozen_reference.npz"
    metadata_path = output_dir / "v10_frozen_reference_metadata.json"

    np.savez_compressed(
        npz_path,
        reference_scene=np.asarray(bundle.reference_scene),
        threshold=np.asarray(bundle.threshold, dtype=float),
        frequency_grid_hz=bundle.frequency_grid_hz,
        true_spectral_prototype=bundle.true_spectral_prototype,
        false_spectral_prototype=bundle.false_spectral_prototype,
        true_temporal_prototype=bundle.true_temporal_prototype,
        false_temporal_prototype=bundle.false_temporal_prototype,
        true_spatial_prototype=bundle.true_spatial_prototype,
        false_spatial_prototype=bundle.false_spatial_prototype,
        reference_spectral=bundle.reference_spectral,
        reference_temporal=bundle.reference_temporal,
        reference_spatial=bundle.reference_spatial,
        reference_labels=bundle.reference_labels,
        reference_sample_ids=bundle.reference_sample_ids,
        config_json=np.asarray(json.dumps(asdict(bundle.config), ensure_ascii=False)),
    )

    metadata = {
        "version": "v10_leak_fingerprint_similarity",
        "reference_scene": bundle.reference_scene,
        "recommended_margin_threshold": bundle.threshold,
        "n_reference_samples": int(len(bundle.reference_labels)),
        "n_reference_true": int(np.sum(bundle.reference_labels == "TRUE_LEAK")),
        "n_reference_false": int(np.sum(bundle.reference_labels == "FALSE_LEAK")),
        "reference_sample_ids": bundle.reference_sample_ids.tolist(),
        "config": asdict(bundle.config),
        "note": (
            "该文件在A场景内部验证后冻结。外部场景验证时不得重新建立原型或重新选择阈值。"
        ),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return npz_path, metadata_path


def load_reference_bundle(path: Path) -> ReferenceBundle:
    if not path.exists():
        raise FileNotFoundError(f"找不到冻结指纹文件: {path}")
    with np.load(path, allow_pickle=False) as data:
        config_data = json.loads(str(data["config_json"].item()))
        config = V10Config(**config_data)
        return ReferenceBundle(
            reference_scene=str(data["reference_scene"].item()),
            threshold=float(data["threshold"].item()),
            frequency_grid_hz=data["frequency_grid_hz"],
            true_spectral_prototype=data["true_spectral_prototype"],
            false_spectral_prototype=data["false_spectral_prototype"],
            true_temporal_prototype=data["true_temporal_prototype"],
            false_temporal_prototype=data["false_temporal_prototype"],
            true_spatial_prototype=data["true_spatial_prototype"],
            false_spatial_prototype=data["false_spatial_prototype"],
            reference_spectral=data["reference_spectral"],
            reference_temporal=data["reference_temporal"],
            reference_spatial=data["reference_spatial"],
            reference_labels=data["reference_labels"].astype(str),
            reference_sample_ids=data["reference_sample_ids"].astype(str),
            config=config,
        )


def fingerprints_from_bundle(bundle: ReferenceBundle) -> List[Fingerprint]:
    samples: List[Fingerprint] = []
    for i in range(len(bundle.reference_labels)):
        samples.append(
            Fingerprint(
                sample_id=str(bundle.reference_sample_ids[i]),
                dataset="frozen_reference",
                scene=bundle.reference_scene,
                time="frozen",
                center="",
                label=str(bundle.reference_labels[i]),
                npz_path="",
                spectral=bundle.reference_spectral[i],
                temporal=bundle.reference_temporal[i],
                spatial=bundle.reference_spatial[i],
            )
        )
    return samples


def evaluate_external_samples(
    samples: Sequence[Fingerprint],
    bundle: ReferenceBundle,
) -> pd.DataFrame:
    prototypes = {
        "TRUE_LEAK": {
            "spectral": bundle.true_spectral_prototype,
            "temporal": bundle.true_temporal_prototype,
            "spatial": bundle.true_spatial_prototype,
        },
        "FALSE_LEAK": {
            "spectral": bundle.false_spectral_prototype,
            "temporal": bundle.false_temporal_prototype,
            "spatial": bundle.false_spatial_prototype,
        },
    }
    references = fingerprints_from_bundle(bundle)
    rows: List[Dict[str, Any]] = []
    for sample in samples:
        result = evaluate_sample_against_reference(
            sample,
            references,
            prototypes,
            bundle.frequency_grid_hz,
            bundle.config,
        )
        pred_zero = "TRUE_LEAK" if result["similarity_margin"] >= 0.0 else "FALSE_LEAK"
        pred_calibrated = (
            "TRUE_LEAK"
            if result["similarity_margin"] >= bundle.threshold
            else "FALSE_LEAK"
        )
        rows.append(
            {
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "scene": sample.scene,
                "time": sample.time,
                "center": sample.center,
                "true_label": sample.label,
                "reference_scene": bundle.reference_scene,
                "frozen_margin_threshold": bundle.threshold,
                **result,
                "pred_margin_0": pred_zero,
                "pred_frozen_threshold": pred_calibrated,
                "correct_margin_0": int(sample.label in VALID_LABELS and pred_zero == sample.label),
                "correct_frozen_threshold": int(
                    sample.label in VALID_LABELS and pred_calibrated == sample.label
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_similarity_results(
    df: pd.DataFrame,
    pred_col: str,
    group_col: str = "scene",
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if df.empty or "true_label" not in df.columns:
        return pd.DataFrame()
    labeled = df[df["true_label"].isin(VALID_LABELS)].copy()
    if labeled.empty:
        return pd.DataFrame()

    for group, sub in labeled.groupby(group_col, dropna=False):
        y = label_to_binary(sub["true_label"])
        pred = label_to_binary(sub[pred_col])
        metrics = confusion_metrics(y, pred)
        rows.append(
            {
                group_col: group,
                "n_samples": len(sub),
                "n_true": int(np.sum(y == 1)),
                "n_false": int(np.sum(y == 0)),
                "margin_auc": safe_auc(y, sub["similarity_margin"].to_numpy(dtype=float)),
                **metrics,
                "true_margin_median": float(
                    sub.loc[sub["true_label"] == "TRUE_LEAK", "similarity_margin"].median()
                )
                if np.any(sub["true_label"] == "TRUE_LEAK")
                else np.nan,
                "false_margin_median": float(
                    sub.loc[sub["true_label"] == "FALSE_LEAK", "similarity_margin"].median()
                )
                if np.any(sub["true_label"] == "FALSE_LEAK")
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# 8. 成对相似度、输出与画图
# =============================================================================


def make_pairwise_similarity_matrix(
    samples: Sequence[Fingerprint],
    frequency_grid_hz: np.ndarray,
    config: V10Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    selected = list(samples)
    if len(selected) > config.max_pairwise_samples:
        rng = np.random.default_rng(config.random_state)
        idx = np.sort(
            rng.choice(len(selected), size=config.max_pairwise_samples, replace=False)
        )
        selected = [selected[i] for i in idx]

    labels = [f"{fp.label[:1]}|{fp.scene}|{fp.time}|c{fp.center}" for fp in selected]
    matrix = np.eye(len(selected), dtype=float)
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            value = pair_similarity(
                selected[i], selected[j], frequency_grid_hz, config
            )["overall"]
            matrix[i, j] = value
            matrix[j, i] = value

    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
    index_df = pd.DataFrame(
        {
            "matrix_label": labels,
            "sample_id": [fp.sample_id for fp in selected],
            "label": [fp.label for fp in selected],
            "scene": [fp.scene for fp in selected],
            "time": [fp.time for fp in selected],
            "center": [fp.center for fp in selected],
        }
    )
    return matrix_df, index_df


def save_prototype_csvs(bundle: ReferenceBundle, output_dir: Path) -> List[Path]:
    paths: List[Path] = []

    spectral_path = output_dir / "v10_spectral_prototypes.csv"
    pd.DataFrame(
        {
            "frequency_hz": bundle.frequency_grid_hz,
            "TRUE_LEAK_prototype": bundle.true_spectral_prototype,
            "FALSE_LEAK_prototype": bundle.false_spectral_prototype,
        }
    ).to_csv(spectral_path, index=False, encoding="utf-8-sig")
    paths.append(spectral_path)

    temporal_path = output_dir / "v10_temporal_prototypes.csv"
    pd.DataFrame(
        {
            "index": np.arange(len(bundle.true_temporal_prototype)),
            "TRUE_LEAK_prototype": bundle.true_temporal_prototype,
            "FALSE_LEAK_prototype": bundle.false_temporal_prototype,
        }
    ).to_csv(temporal_path, index=False, encoding="utf-8-sig")
    paths.append(temporal_path)

    spatial_path = output_dir / "v10_spatial_prototypes.csv"
    names = ["center"] + [
        f"ring_{i + 1}" for i in range(bundle.config.n_spatial_rings)
    ] + ["center_rank", "radial_decay"]
    if len(names) != len(bundle.true_spatial_prototype):
        names = [f"component_{i}" for i in range(len(bundle.true_spatial_prototype))]
    pd.DataFrame(
        {
            "component": names,
            "TRUE_LEAK_prototype": bundle.true_spatial_prototype,
            "FALSE_LEAK_prototype": bundle.false_spatial_prototype,
        }
    ).to_csv(spatial_path, index=False, encoding="utf-8-sig")
    paths.append(spatial_path)

    return paths


def make_plots(
    output_dir: Path,
    bundle: ReferenceBundle,
    internal_df: Optional[pd.DataFrame] = None,
    pairwise_df: Optional[pd.DataFrame] = None,
    external_df: Optional[pd.DataFrame] = None,
) -> List[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[警告] 无法导入matplotlib，跳过画图: {exc}")
        return []

    fig_dir = output_dir / "figures"
    ensure_dir(fig_dir)
    paths: List[Path] = []

    # 1. TRUE/FALSE频谱原型。
    plt.figure(figsize=(10, 5))
    plt.plot(
        bundle.frequency_grid_hz / 1000.0,
        bundle.true_spectral_prototype,
        label="TRUE prototype",
    )
    plt.plot(
        bundle.frequency_grid_hz / 1000.0,
        bundle.false_spectral_prototype,
        label="FALSE prototype",
    )
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Normalized residual fingerprint")
    plt.title("V10 spectral fingerprint prototypes")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = fig_dir / "v10_spectral_prototypes.png"
    plt.savefig(path, dpi=160)
    plt.close()
    paths.append(path)

    # 2. 内部OOF Margin。
    if internal_df is not None and not internal_df.empty:
        true_margin = internal_df.loc[
            internal_df["true_label"] == "TRUE_LEAK", "similarity_margin"
        ].to_numpy(dtype=float)
        false_margin = internal_df.loc[
            internal_df["true_label"] == "FALSE_LEAK", "similarity_margin"
        ].to_numpy(dtype=float)
        plt.figure(figsize=(8, 5))
        plt.boxplot([true_margin, false_margin], tick_labels=["TRUE", "FALSE"])
        plt.axhline(0.0, linestyle="--", linewidth=1, label="Natural margin = 0")
        plt.axhline(
            bundle.threshold,
            linestyle=":",
            linewidth=1,
            label=f"Calibrated = {bundle.threshold:.4f}",
        )
        plt.ylabel("Similarity margin (TRUE - FALSE)")
        plt.title("Internal out-of-fold similarity margin")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        path = fig_dir / "v10_internal_margin_distribution.png"
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(path)

        plt.figure(figsize=(8, 6))
        for label, group in internal_df.groupby("true_label"):
            plt.scatter(
                group["margin_spectral"],
                group["margin_temporal"],
                label=label,
                alpha=0.8,
            )
        plt.axvline(0.0, linewidth=1)
        plt.axhline(0.0, linewidth=1)
        plt.xlabel("Spectral margin")
        plt.ylabel("Temporal margin")
        plt.title("Spectral vs temporal fingerprint margin")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        path = fig_dir / "v10_spectral_temporal_margin.png"
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(path)

    # 3. 成对相似度矩阵。
    if pairwise_df is not None and not pairwise_df.empty:
        plt.figure(figsize=(11, 9))
        image = plt.imshow(pairwise_df.to_numpy(dtype=float), aspect="auto", vmin=0.0, vmax=1.0)
        plt.title("Pairwise fingerprint similarity")
        plt.xlabel("Samples")
        plt.ylabel("Samples")
        plt.colorbar(image, label="Similarity")
        plt.tight_layout()
        path = fig_dir / "v10_pairwise_similarity_heatmap.png"
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(path)

    # 4. 外部场景Margin。
    if external_df is not None and not external_df.empty:
        labeled = external_df[external_df["true_label"].isin(VALID_LABELS)]
        if not labeled.empty:
            groups = []
            names = []
            for (scene, label), sub in labeled.groupby(["scene", "true_label"]):
                groups.append(sub["similarity_margin"].to_numpy(dtype=float))
                names.append(f"{scene}\n{label}")
            if groups:
                plt.figure(figsize=(max(9, len(groups) * 1.2), 5))
                plt.boxplot(groups, tick_labels=names)
                plt.axhline(bundle.threshold, linestyle="--", linewidth=1)
                plt.ylabel("Similarity margin")
                plt.title("External-scene frozen-fingerprint validation")
                plt.grid(True, axis="y", alpha=0.3)
                plt.xticks(rotation=25, ha="right")
                plt.tight_layout()
                path = fig_dir / "v10_external_margin_distribution.png"
                plt.savefig(path, dpi=160)
                plt.close()
                paths.append(path)

    return paths


def internal_overall_summary(
    oof_df: pd.DataFrame,
    threshold: float,
    split_mode: str,
) -> Dict[str, Any]:
    y = label_to_binary(oof_df["true_label"])
    margin = oof_df["similarity_margin"].to_numpy(dtype=float)
    metrics_zero = confusion_metrics(y, (margin >= 0.0).astype(int))
    metrics_cal = confusion_metrics(y, (margin >= threshold).astype(int))
    return {
        "split_mode": split_mode,
        "n_samples": len(oof_df),
        "n_true": int(np.sum(y == 1)),
        "n_false": int(np.sum(y == 0)),
        "margin_auc": safe_auc(y, margin),
        "natural_threshold": 0.0,
        "natural_accuracy": metrics_zero["accuracy"],
        "natural_balanced_accuracy": metrics_zero["balanced_accuracy"],
        "calibrated_threshold": threshold,
        "calibrated_accuracy": metrics_cal["accuracy"],
        "calibrated_balanced_accuracy": metrics_cal["balanced_accuracy"],
        "calibrated_tn": metrics_cal["tn"],
        "calibrated_fp": metrics_cal["fp"],
        "calibrated_fn": metrics_cal["fn"],
        "calibrated_tp": metrics_cal["tp"],
        "true_margin_median": float(
            oof_df.loc[oof_df["true_label"] == "TRUE_LEAK", "similarity_margin"].median()
        ),
        "false_margin_median": float(
            oof_df.loc[oof_df["true_label"] == "FALSE_LEAK", "similarity_margin"].median()
        ),
        "spectral_margin_auc": safe_auc(
            y, oof_df["margin_spectral"].to_numpy(dtype=float)
        ),
        "temporal_margin_auc": safe_auc(
            y, oof_df["margin_temporal"].to_numpy(dtype=float)
        ),
        "spatial_margin_auc": safe_auc(
            y, oof_df["margin_spatial"].to_numpy(dtype=float)
        ),
    }


def make_report(
    output_dir: Path,
    v9_dir: Path,
    reference_scene: str,
    reference_samples: Sequence[Fingerprint],
    all_samples: Sequence[Fingerprint],
    internal_summary: Optional[Dict[str, Any]],
    warnings: Sequence[str],
    bundle_path: Path,
    external_summary: Optional[pd.DataFrame],
    fingerprint_failures: pd.DataFrame,
) -> Path:
    lines: List[str] = []
    lines.append("v10 泄漏残差指纹建立与相似度验证报告")
    lines.append("=" * 88)
    lines.append(f"v9结果目录: {v9_dir}")
    lines.append(f"参考scene: {reference_scene}")
    lines.append(f"全部成功指纹样本: {len(all_samples)}")
    lines.append(f"参考scene样本: {len(reference_samples)}")
    lines.append(
        f"参考TRUE/FALSE: "
        f"{sum(x.label == 'TRUE_LEAK' for x in reference_samples)}/"
        f"{sum(x.label == 'FALSE_LEAK' for x in reference_samples)}"
    )
    lines.append(f"指纹提取失败样本: {len(fingerprint_failures)}")
    lines.append(f"冻结指纹文件: {bundle_path}")
    lines.append("")

    if warnings:
        lines.append("重要警告:")
        for warning in warnings:
            lines.append(f"  - {warning}")
        lines.append("")

    if internal_summary:
        lines.append("A/参考场景内部无泄漏验证结果:")
        for key, value in internal_summary.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.6f}" if np.isfinite(value) else f"  {key}: NA")
            else:
                lines.append(f"  {key}: {value}")
        lines.append("")

        lines.append("结果解释:")
        lines.append("  1. similarity_margin = 与TRUE指纹相似度 - 与FALSE指纹相似度。")
        lines.append("  2. TRUE样本理想情况下margin为正，FALSE样本理想情况下margin为负。")
        lines.append("  3. margin_auc越接近1，说明参考场景内部未参与建模的样本分离越好。")
        lines.append("  4. 只有按time整组验证才较可信；样本级回退验证只能作为初步检查。")
        lines.append("")

    if external_summary is not None and not external_summary.empty:
        lines.append("冻结指纹外部scene验证:")
        for _, row in external_summary.iterrows():
            lines.append(
                f"  {row['scene']}: n={int(row['n_samples'])}, "
                f"balanced_accuracy={row['balanced_accuracy']:.4f}, "
                f"AUC={row['margin_auc'] if np.isfinite(row['margin_auc']) else 'NA'}, "
                f"TRUE_margin_median={row['true_margin_median']:.4f}, "
                f"FALSE_margin_median={row['false_margin_median']:.4f}"
            )
        lines.append("")

    lines.append("下一步判定原则:")
    lines.append("  - 先看内部OOF结果，不要看用全部A样本建立原型后再评价A自身的乐观结果。")
    lines.append("  - 外部B/C场景必须使用当前冻结指纹和冻结阈值，不得重新建立原型。")
    lines.append("  - 若频谱、时间、空间三个margin方向一致，可信度高于只靠总能量差。")
    lines.append("  - 如果只有一个time，需补充独立时间采集后再判断是否找到了稳定指纹。")

    report_path = output_dir / "v10_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


# =============================================================================
# 9. 主流程
# =============================================================================


def run_all_mode(
    v9_dir: Path,
    output_dir: Path,
    requested_scene: Optional[str],
    config: V10Config,
) -> Dict[str, Path]:
    ensure_dir(output_dir)
    fingerprints, fingerprint_index, failure_df = load_fingerprints(v9_dir, config)
    failure_path = output_dir / "v10_fingerprint_failures.csv"
    index_path = output_dir / "v10_fingerprint_index.csv"
    failure_df.to_csv(failure_path, index=False, encoding="utf-8-sig")
    fingerprint_index.to_csv(index_path, index=False, encoding="utf-8-sig")

    reference_scene = choose_reference_scene(fingerprints, requested_scene)
    reference_samples = [
        fp for fp in fingerprints if fp.scene == reference_scene and fp.label in VALID_LABELS
    ]
    frequency_grid = np.linspace(
        config.freq_low_hz, config.freq_high_hz, config.n_spectral_bins
    )

    print("=" * 90)
    print("v10 泄漏残差指纹相似度验证")
    print("=" * 90)
    print("v9结果目录:", v9_dir)
    print("参考scene:", reference_scene)
    print("参考样本数:", len(reference_samples))
    print(
        "参考TRUE/FALSE:",
        sum(x.label == "TRUE_LEAK" for x in reference_samples),
        "/",
        sum(x.label == "FALSE_LEAK" for x in reference_samples),
    )

    oof_df, fold_df, threshold_curve, threshold, warnings, split_mode = run_internal_validation(
        reference_samples, frequency_grid, config
    )
    oof_path = output_dir / "v10_internal_oof_similarity.csv"
    fold_path = output_dir / "v10_internal_fold_summary.csv"
    threshold_path = output_dir / "v10_margin_threshold_curve.csv"
    summary_path = output_dir / "v10_internal_summary.csv"
    oof_df.to_csv(oof_path, index=False, encoding="utf-8-sig")
    fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
    threshold_curve.to_csv(threshold_path, index=False, encoding="utf-8-sig")

    internal_summary = internal_overall_summary(oof_df, threshold, split_mode)
    pd.DataFrame([internal_summary]).to_csv(
        summary_path, index=False, encoding="utf-8-sig"
    )

    bundle = build_reference_bundle(
        reference_samples,
        frequency_grid,
        reference_scene,
        threshold,
        config,
    )
    bundle_path, metadata_path = save_reference_bundle(bundle, output_dir)
    save_prototype_csvs(bundle, output_dir)

    pairwise_df, matrix_index_df = make_pairwise_similarity_matrix(
        reference_samples, frequency_grid, config
    )
    pairwise_path = output_dir / "v10_reference_pairwise_similarity_matrix.csv"
    matrix_index_path = output_dir / "v10_reference_pairwise_matrix_index.csv"
    pairwise_df.to_csv(pairwise_path, encoding="utf-8-sig")
    matrix_index_df.to_csv(matrix_index_path, index=False, encoding="utf-8-sig")

    # 同一v9目录中若已有B/C等其他scene，直接使用冻结A指纹进行外部验证。
    external_samples = [fp for fp in fingerprints if fp.scene != reference_scene]
    external_df = evaluate_external_samples(external_samples, bundle) if external_samples else pd.DataFrame()
    external_path = output_dir / "v10_external_similarity.csv"
    external_summary_path = output_dir / "v10_external_summary.csv"
    external_df.to_csv(external_path, index=False, encoding="utf-8-sig")
    external_summary = summarize_similarity_results(
        external_df, pred_col="pred_frozen_threshold", group_col="scene"
    )
    external_summary.to_csv(external_summary_path, index=False, encoding="utf-8-sig")

    make_plots(
        output_dir,
        bundle,
        internal_df=oof_df,
        pairwise_df=pairwise_df,
        external_df=external_df,
    )

    report_path = make_report(
        output_dir,
        v9_dir,
        reference_scene,
        reference_samples,
        fingerprints,
        internal_summary,
        warnings,
        bundle_path,
        external_summary,
        failure_df,
    )

    print("\n内部验证方式:", split_mode)
    for warning in warnings:
        print("[警告]", warning)
    print(f"OOF Margin AUC: {internal_summary['margin_auc']:.4f}")
    print(f"自然阈值0 平衡准确率: {internal_summary['natural_balanced_accuracy']:.4f}")
    print(f"冻结阈值: {threshold:.6f}")
    print(f"冻结阈值平衡准确率: {internal_summary['calibrated_balanced_accuracy']:.4f}")
    print("TRUE Margin中位数:", f"{internal_summary['true_margin_median']:.6f}")
    print("FALSE Margin中位数:", f"{internal_summary['false_margin_median']:.6f}")
    print("\n输出目录:", output_dir)
    print("内部OOF明细:", oof_path)
    print("内部汇总:", summary_path)
    print("冻结指纹:", bundle_path)
    print("报告:", report_path)
    if not external_df.empty:
        print("外部scene验证:", external_path)

    return {
        "oof": oof_path,
        "summary": summary_path,
        "bundle": bundle_path,
        "metadata": metadata_path,
        "report": report_path,
        "failure": failure_path,
        "external": external_path,
    }


def run_external_mode(
    v9_dir: Path,
    output_dir: Path,
    prototype_file: Path,
) -> Dict[str, Path]:
    bundle = load_reference_bundle(prototype_file)
    ensure_dir(output_dir)
    fingerprints, fingerprint_index, failure_df = load_fingerprints(v9_dir, bundle.config)

    result_df = evaluate_external_samples(fingerprints, bundle)
    result_path = output_dir / "v10_external_similarity.csv"
    summary_path = output_dir / "v10_external_summary.csv"
    index_path = output_dir / "v10_fingerprint_index.csv"
    failure_path = output_dir / "v10_fingerprint_failures.csv"
    report_path = output_dir / "v10_external_report.txt"

    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    fingerprint_index.to_csv(index_path, index=False, encoding="utf-8-sig")
    failure_df.to_csv(failure_path, index=False, encoding="utf-8-sig")
    summary_df = summarize_similarity_results(
        result_df, pred_col="pred_frozen_threshold", group_col="scene"
    )
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    make_plots(output_dir, bundle, external_df=result_df)

    lines = [
        "v10 冻结指纹外部验证报告",
        "=" * 88,
        f"待测v9目录: {v9_dir}",
        f"冻结指纹: {prototype_file}",
        f"参考scene: {bundle.reference_scene}",
        f"冻结Margin阈值: {bundle.threshold:.6f}",
        f"成功样本: {len(result_df)}",
        f"失败样本: {len(failure_df)}",
        "",
    ]
    if summary_df.empty:
        lines.append("当前数据没有完整TRUE/FALSE标签，只输出逐样本相似度，不计算准确率。")
    else:
        lines.append("各scene结果:")
        for _, row in summary_df.iterrows():
            lines.append(
                f"  {row['scene']}: n={int(row['n_samples'])}, "
                f"balanced_accuracy={row['balanced_accuracy']:.4f}, "
                f"AUC={row['margin_auc'] if np.isfinite(row['margin_auc']) else 'NA'}, "
                f"TRUE_margin_median={row['true_margin_median']:.4f}, "
                f"FALSE_margin_median={row['false_margin_median']:.4f}"
            )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("=" * 90)
    print("v10 冻结指纹外部验证完成")
    print("参考scene:", bundle.reference_scene)
    print("冻结阈值:", bundle.threshold)
    print("成功样本:", len(result_df))
    print("失败样本:", len(failure_df))
    print("逐样本结果:", result_path)
    print("汇总:", summary_path)
    print("报告:", report_path)

    return {
        "result": result_path,
        "summary": summary_path,
        "report": report_path,
        "failure": failure_path,
    }


# =============================================================================
# 10. 内置自检
# =============================================================================


def _synthetic_sample_npz(
    path: Path,
    label: str,
    rng: np.random.Generator,
    scene_shift_hz: float = 0.0,
) -> None:
    freq = np.linspace(20_000.0, 80_000.0, 300)
    n_frames = 50
    n_points = 65

    background = 1.0 + 0.08 * rng.random((len(freq), n_frames))
    if label == "TRUE_LEAK":
        broad = np.exp(-0.5 * ((freq - (52_000.0 + scene_shift_hz)) / 9_000.0) ** 2)
        time_gain = 0.8 + 0.15 * rng.random(n_frames)
        excess = 0.65 * broad[:, None] * time_gain[None, :]
        center_boost = 5.0
        neighbor_shape = -np.linspace(0.5, 3.0, n_points - 1)
    else:
        narrow = np.exp(-0.5 * ((freq - (36_000.0 + 0.3 * scene_shift_hz)) / 900.0) ** 2)
        time_gain = np.zeros(n_frames)
        active = rng.choice(n_frames, size=8, replace=False)
        time_gain[active] = 2.0 + rng.random(len(active))
        excess = 0.85 * narrow[:, None] * time_gain[None, :]
        center_boost = 1.0
        neighbor_shape = rng.normal(0.0, 1.2, n_points - 1)

    excess += 0.01 * rng.random(excess.shape)
    center = background + excess
    residual_db = 10.0 * np.log10(center / background)

    angles = np.linspace(0.0, 2.0 * np.pi, n_points - 1, endpoint=False)
    radii = np.tile(np.linspace(10.0, 80.0, 8), 8)
    xy_neighbors = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    xy = np.vstack([[0.0, 0.0], xy_neighbors])

    baseline_db = -40.0
    point_band_db = np.concatenate(
        [[baseline_db + center_boost], baseline_db + neighbor_shape]
    )
    background_indices = np.arange(33, 65, dtype=int)

    np.savez_compressed(
        path,
        freq_hz=freq,
        time_s=np.arange(n_frames) * 0.01,
        center_power_db=10.0 * np.log10(center),
        background_power_db=10.0 * np.log10(background),
        residual_db=residual_db,
        excess_power=excess,
        xy=xy,
        point_band_db=point_band_db,
        background_indices=background_indices,
        point_weights=np.ones(len(background_indices)),
    )


def run_self_test() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    with tempfile.TemporaryDirectory(prefix="v10_self_test_") as tmp:
        root = Path(tmp)
        v9_dir = root / "v9"
        residual_dir = v9_dir / "residual_npz"
        output_dir = root / "v10"
        ensure_dir(residual_dir)

        rows: List[Dict[str, Any]] = []
        center = 0
        for scene, shift in [("factory_A", 0.0), ("factory_B", 1500.0)]:
            for label in ["TRUE_LEAK", "FALSE_LEAK"]:
                for time_index in range(3):
                    time_name = f"{scene}_time_{label}_{time_index}"
                    for _ in range(3):
                        center += 1
                        sample_id = f"{scene}_{label}_{time_name}_center_{center}"
                        npz_path = residual_dir / f"{safe_slug(sample_id)}.npz"
                        _synthetic_sample_npz(npz_path, label, rng, shift)
                        rows.append(
                            {
                                "sample_id": sample_id,
                                "dataset": f"{scene}_{label}",
                                "scene": scene,
                                "time": time_name,
                                "center": str(center),
                                "label": label,
                            }
                        )
        pd.DataFrame(rows).to_csv(
            v9_dir / "v9_all_features.csv", index=False, encoding="utf-8-sig"
        )

        paths = run_all_mode(v9_dir, output_dir, "factory_A", V10Config())
        summary = pd.read_csv(paths["summary"])
        auc = float(summary.loc[0, "margin_auc"])
        balanced = float(summary.loc[0, "calibrated_balanced_accuracy"])
        if auc < 0.90 or balanced < 0.85:
            raise AssertionError(
                f"自检未通过: AUC={auc:.4f}, balanced_accuracy={balanced:.4f}"
            )
        external = pd.read_csv(paths["external"])
        if external.empty:
            raise AssertionError("自检未生成外部场景结果")
        print("\n自检通过。")
        print(f"内部AUC: {auc:.4f}")
        print(f"内部平衡准确率: {balanced:.4f}")
        print("冻结指纹外部场景结果已成功生成。")


# =============================================================================
# 11. 命令行
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v10 泄漏残差指纹相似度验证")
    parser.add_argument(
        "--mode",
        choices=["all", "external"],
        default="all",
        help="all=A场景内部验证并冻结指纹；external=使用冻结指纹验证新场景",
    )
    parser.add_argument("--v9-dir", default=V9_RESULT_DIR, help="v9结果目录")
    parser.add_argument("--output-dir", default=V10_OUTPUT_DIR, help="v10输出目录")
    parser.add_argument(
        "--reference-scene",
        default=REFERENCE_SCENE,
        help="参考scene名称；不填则自动选择",
    )
    parser.add_argument(
        "--prototype-file",
        default="",
        help="external模式必填：v10_frozen_reference.npz路径",
    )
    parser.add_argument("--self-test", action="store_true", help="运行内置合成数据自检")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    v9_dir = Path(args.v9_dir)
    output_dir = Path(args.output_dir)

    if args.mode == "external":
        if not args.prototype_file:
            raise ValueError("external模式必须提供 --prototype-file")
        run_external_mode(v9_dir, output_dir, Path(args.prototype_file))
    else:
        run_all_mode(v9_dir, output_dir, args.reference_scene, V10Config())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n[程序失败]", type(exc).__name__, str(exc))
        print(traceback.format_exc())
        sys.exit(1)
