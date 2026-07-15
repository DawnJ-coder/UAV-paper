# -*- coding: utf-8 -*-
"""
AB_lab_commonality_analysis.py

目的
====
1. 读取场景A、场景B的v9结果（每个场景均包含TRUE_LEAK和FALSE_LEAK）；
2. 分别计算：A_TRUE - A_FALSE、B_TRUE - B_FALSE；
3. 提取A与B共同支持的泄漏候选频谱、时间属性和统计属性；
4. 从实验室真实泄漏WAV中提取稳定原型；
5. 寻找“AB共同候选”与“实验室泄漏原型”的共同部分；
6. 输出频带、相似度、逐频点表格、样本相似度和诊断图。

重要说明
========
- 本程序不训练分类器，也不学习TRUE/FALSE阈值。
- 实验室WAV不能直接与工厂WAV相减；程序比较统一归一化后的频谱和时间形态。
- 工厂侧使用v9已经完成空间背景消除后的残差NPZ。
- 默认推荐所有场景使用相同的v9主方法；程序也可显式读取median/plane/selected字段。
- 最终得到的是“实验室锚定的AB共同泄漏残差指纹”，不是带相位、可播放的纯泄漏WAV。

依赖
====
pip install numpy pandas scipy matplotlib

运行
====
python AB_lab_commonality_analysis.py --self-test
python AB_lab_commonality_analysis.py --config AB_lab_config.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import signal
    from scipy.io import wavfile
except Exception as exc:  # pragma: no cover
    raise RuntimeError("缺少 scipy，请运行: pip install scipy") from exc


EPS = 1.0e-20
VALID_LABELS = {"TRUE_LEAK", "FALSE_LEAK"}


# =============================================================================
# 1. 配置
# =============================================================================


@dataclass
class SceneInput:
    name: str
    v9_dir: str


@dataclass
class AnalysisConfig:
    scene_A: SceneInput
    scene_B: SceneInput
    lab_wav_paths: List[str]
    output_dir: str

    # v9残差读取方式。
    # residual_variant: primary / median / plane / selected
    # factory_representation: excess_power / positive_z
    residual_variant: str = "median"
    factory_representation: str = "excess_power"

    # 统一分析频带与指纹维度。
    freq_low_hz: float = 20_000.0
    freq_high_hz: float = 80_000.0
    n_spectral_bins: int = 256
    n_temporal_quantiles: int = 64
    spectral_smooth_bins: int = 5

    # 场景内TRUE-FALSE稳健筛选。
    bootstrap_iterations: int = 300
    bootstrap_positive_probability: float = 0.90
    min_robust_effect: float = 0.25
    random_state: int = 42

    # 实验室WAV切片与STFT。
    lab_segment_seconds: float = 1.0
    lab_segment_hop_seconds: float = 0.5
    lab_nperseg: int = 4096
    lab_hop_length: int = 2048
    lab_nfft: int = 4096
    min_lab_segments: int = 3

    # 实验室频谱活跃判定。
    lab_active_spectrum_quantile: float = 0.55
    lab_min_segment_support: float = 0.60

    # 最终共同频带。
    common_weight_quantile: float = 0.35
    minimum_common_bandwidth_hz: float = 700.0
    merge_gap_hz: float = 500.0
    coarse_bandwidth_hz: float = 5_000.0

    # 去平滑包络的局部谱形状。
    envelope_window_bins: int = 41
    envelope_polyorder: int = 2
    broadband_common_weight: float = 0.70
    local_shape_common_weight: float = 0.30

    save_figures: bool = True


# =============================================================================
# 2. 通用函数
# =============================================================================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_slug(value: Any, max_len: int = 180) -> str:
    text = str(value).strip() or "sample"
    text = re.sub(r"[^0-9A-Za-z_\-.]+", "_", text)
    return text.strip("._")[:max_len] or "sample"


def normalize_nonnegative(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.maximum(arr, 0.0)
    total = float(np.sum(arr))
    if total <= eps:
        return np.zeros_like(arr)
    return arr / total


def robust_scale(values: np.ndarray, axis: Optional[int] = None, floor: float = 1.0e-8) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    med = np.nanmedian(arr, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(arr - med), axis=axis)
    return np.maximum(1.4826 * mad, floor)


def smooth_1d(values: np.ndarray, bins: int) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if bins <= 1 or x.size < 3:
        return x.copy()
    width = max(1, int(bins))
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(x, kernel, mode="same")


def interp_vector(x_old: np.ndarray, y_old: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    x = np.asarray(x_old, dtype=float).ravel()
    y = np.asarray(y_old, dtype=float).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 2:
        return np.zeros_like(x_new, dtype=float)
    x = x[valid]
    y = y[valid]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique_x, unique_idx = np.unique(x, return_index=True)
    y = y[unique_idx]
    if unique_x.size < 2:
        return np.zeros_like(x_new, dtype=float)
    return np.interp(x_new, unique_x, y, left=0.0, right=0.0)


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = EPS) -> float:
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) == 0:
        return np.nan
    x = x[valid]
    y = y[valid]
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= eps:
        return np.nan
    return float(np.dot(x, y) / denom)


def overlap_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    x = normalize_nonnegative(a)
    y = normalize_nonnegative(b)
    if np.sum(x) <= 0 or np.sum(y) <= 0:
        return np.nan
    return float(np.sum(np.minimum(x, y)))


def js_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1.0e-12) -> float:
    p = normalize_nonnegative(a)
    q = normalize_nonnegative(b)
    if np.sum(p) <= 0 or np.sum(q) <= 0:
        return np.nan
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
    js = 0.5 * (kl_pm + kl_qm)
    return float(np.clip(1.0 - js / math.log(2.0), 0.0, 1.0))


def pearson_safe(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 3:
        return np.nan
    x = x[valid]
    y = y[valid]
    if np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def normalized_entropy(weights: np.ndarray, eps: float = EPS) -> float:
    p = normalize_nonnegative(weights, eps)
    if p.size <= 1 or np.sum(p) <= 0:
        return 0.0
    return float(-np.sum(p * np.log(p + eps)) / np.log(p.size))


def spectral_flatness(weights: np.ndarray, eps: float = EPS) -> float:
    x = np.maximum(np.asarray(weights, dtype=float), 0.0)
    if x.size == 0 or float(np.mean(x)) <= eps:
        return 0.0
    return float(np.exp(np.mean(np.log(x + eps))) / (np.mean(x) + eps))


def rank_percentile_rows(matrix: np.ndarray) -> np.ndarray:
    """每一行内部按频率排名，输出0~1百分位。"""
    arr = np.asarray(matrix, dtype=float)
    out = np.zeros_like(arr)
    for i, row in enumerate(arr):
        order = np.argsort(np.argsort(row, kind="mergesort"), kind="mergesort")
        out[i] = order / max(len(row) - 1, 1)
    return out


def remove_spectral_envelope(
    profile: np.ndarray,
    window_bins: int,
    polyorder: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.log1p(np.maximum(np.asarray(profile, dtype=float), 0.0))
    n = len(x)
    if n < 7:
        return x.copy(), np.zeros_like(x)

    window = int(window_bins)
    if window % 2 == 0:
        window += 1
    max_window = n if n % 2 == 1 else n - 1
    window = min(window, max_window)
    min_window = max(polyorder + 2, 5)
    if min_window % 2 == 0:
        min_window += 1
    window = max(window, min_window)
    if window > max_window or window <= polyorder:
        return x.copy(), np.zeros_like(x)

    envelope = signal.savgol_filter(
        x,
        window_length=window,
        polyorder=polyorder,
        mode="interp",
    )
    return x - envelope, envelope


def contiguous_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    m = np.asarray(mask, dtype=bool).ravel()
    if m.size == 0:
        return []
    padded = np.concatenate([[False], m, [False]]).astype(np.int8)
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def merge_regions_by_gap(
    regions: List[Tuple[int, int]],
    freq_hz: np.ndarray,
    max_gap_hz: float,
) -> List[Tuple[int, int]]:
    if not regions:
        return []
    merged = [regions[0]]
    for start, end in regions[1:]:
        prev_start, prev_end = merged[-1]
        gap = float(freq_hz[start] - freq_hz[prev_end])
        if gap <= max_gap_hz:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def read_json_config(path: Path) -> AnalysisConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        scene_a = SceneInput(**payload.pop("scene_A"))
        scene_b = SceneInput(**payload.pop("scene_B"))
        return AnalysisConfig(scene_A=scene_a, scene_B=scene_b, **payload)
    except TypeError as exc:
        raise ValueError(f"配置字段错误: {exc}") from exc


# =============================================================================
# 3. v9结果读取
# =============================================================================


def load_npz_index(v9_dir: Path) -> Dict[str, Path]:
    candidates = list((v9_dir / "residual_npz").glob("*.npz"))
    if not candidates:
        candidates = list(v9_dir.rglob("*.npz"))
    index: Dict[str, Path] = {}
    for p in candidates:
        index[p.stem] = p
        index[safe_slug(p.stem)] = p
    return index


def resolve_npz_path(row: pd.Series, v9_dir: Path, index: Mapping[str, Path]) -> Optional[Path]:
    for col in (
        "residual_npz_path",
        "npz_path",
        "residual_file",
        "diagnostic_npz",
    ):
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            p = Path(str(row[col]))
            if not p.is_absolute():
                p = v9_dir / p
            if p.exists():
                return p

    sample_id = str(row.get("sample_id", "")).strip()
    candidates = [sample_id, safe_slug(sample_id)]
    for name in candidates:
        if name in index:
            return index[name]

    # 兼容文件名中包含sample_id的情况。
    slug = safe_slug(sample_id)
    fuzzy = [p for stem, p in index.items() if slug and (slug in stem or stem in slug)]
    if len(fuzzy) == 1:
        return fuzzy[0]
    return None


def choose_npz_keys(
    npz: Mapping[str, Any],
    variant: str,
    representation: str,
) -> Tuple[str, Optional[str]]:
    variant = str(variant).strip().lower()
    representation = str(representation).strip().lower()
    if variant == "auto":
        variant = "selected"
    if variant not in {"primary", "median", "plane", "selected"}:
        raise ValueError(f"residual_variant不支持: {variant}")
    if representation not in {"excess_power", "positive_z"}:
        raise ValueError(f"factory_representation不支持: {representation}")

    suffix = "" if variant == "primary" else f"_{variant}"
    matrix_key = f"{representation}{suffix}"

    if matrix_key not in npz:
        # 兼容旧v9，只有无后缀excess_power。
        if representation == "excess_power" and "excess_power" in npz:
            matrix_key = "excess_power"
        elif representation == "positive_z":
            fallback = f"excess_power{suffix}"
            if fallback in npz:
                matrix_key = fallback
            elif "excess_power" in npz:
                matrix_key = "excess_power"
            else:
                raise KeyError(
                    f"NPZ中没有{representation}或excess_power字段。现有字段={list(npz.keys())[:30]}"
                )
        else:
            raise KeyError(f"NPZ缺少字段: {matrix_key}")

    bg_key: Optional[str] = None
    candidate_bg = "background_power_db" + suffix
    if candidate_bg in npz:
        bg_key = candidate_bg
    elif "background_power_db" in npz:
        bg_key = "background_power_db"
    return matrix_key, bg_key


def orient_tf_matrix(matrix: np.ndarray, n_freq: int) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim == 1:
        if arr.size != n_freq:
            raise ValueError(f"一维残差长度{arr.size}与频率长度{n_freq}不一致")
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"残差矩阵应为二维，当前shape={arr.shape}")
    if arr.shape[0] == n_freq:
        return arr
    if arr.shape[1] == n_freq:
        return arr.T
    raise ValueError(f"无法判断频率轴：matrix={arr.shape}, n_freq={n_freq}")


def sample_fingerprint_from_npz(
    npz_path: Path,
    freq_grid: np.ndarray,
    config: AnalysisConfig,
) -> Dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as data:
        if "freq_hz" not in data:
            raise KeyError("NPZ缺少freq_hz")
        freq = np.asarray(data["freq_hz"], dtype=float).ravel()
        matrix_key, bg_key = choose_npz_keys(
            data,
            config.residual_variant,
            config.factory_representation,
        )
        matrix = orient_tf_matrix(np.asarray(data[matrix_key]), len(freq))
        matrix = np.where(np.isfinite(matrix), matrix, 0.0)
        matrix = np.maximum(matrix, 0.0)

        freq_mask = (freq >= config.freq_low_hz) & (freq <= config.freq_high_hz)
        if np.sum(freq_mask) < 5:
            raise ValueError("NPZ在目标频带内频点过少")
        freq_use = freq[freq_mask]
        matrix_use = matrix[freq_mask]

        # 压缩动态范围，防止极强单点主导。
        spectral_native = np.sqrt(np.maximum(np.nanmean(matrix_use, axis=1), 0.0))
        spectral_native = smooth_1d(spectral_native, config.spectral_smooth_bins)
        spectral = interp_vector(freq_use, spectral_native, freq_grid)
        spectral = normalize_nonnegative(spectral)

        frame_signal = np.nansum(matrix_use, axis=0)
        if bg_key is not None:
            bg_db = orient_tf_matrix(np.asarray(data[bg_key]), len(freq))[freq_mask]
            bg_power = np.power(10.0, np.clip(bg_db, -300.0, 300.0) / 10.0)
            frame_bg = np.nansum(bg_power, axis=0)
            frame_measure = np.log1p(frame_signal / (frame_bg + EPS))
        else:
            scale = float(np.nanmedian(frame_signal[frame_signal > 0])) if np.any(frame_signal > 0) else 1.0
            frame_measure = np.log1p(frame_signal / max(scale, EPS))

        frame_measure = np.where(np.isfinite(frame_measure), frame_measure, 0.0)
        q_grid = np.linspace(0.0, 1.0, config.n_temporal_quantiles)
        temporal = np.quantile(frame_measure, q_grid) if frame_measure.size else np.zeros_like(q_grid)
        temporal_scale = float(np.percentile(temporal, 95)) if temporal.size else 0.0
        if temporal_scale > EPS:
            temporal = np.clip(temporal / temporal_scale, 0.0, None)

        scalar = scalar_features(spectral, temporal, freq_grid)

        method_primary = ""
        method_selected = ""
        if "primary_method" in data:
            method_primary = str(np.asarray(data["primary_method"]).item())
        if "selected_method" in data:
            method_selected = str(np.asarray(data["selected_method"]).item())

    return {
        "spectrum": spectral,
        "temporal": temporal,
        "scalar": scalar,
        "matrix_key": matrix_key,
        "background_key": bg_key or "",
        "primary_method": method_primary,
        "selected_method": method_selected,
    }


def scalar_features(spectrum: np.ndarray, temporal: np.ndarray, freq_grid: np.ndarray) -> Dict[str, float]:
    s = normalize_nonnegative(spectrum)
    total = float(np.sum(s))
    if total > 0:
        centroid = float(np.sum(freq_grid * s))
        bandwidth = float(np.sqrt(np.sum(((freq_grid - centroid) ** 2) * s)))
    else:
        centroid = 0.0
        bandwidth = 0.0
    high_mask = freq_grid >= 40_000.0
    high_ratio = float(np.sum(s[high_mask])) if np.any(high_mask) else 0.0

    t = np.asarray(temporal, dtype=float)
    t_mean = float(np.mean(t)) if t.size else 0.0
    t_std = float(np.std(t)) if t.size else 0.0
    return {
        "spectral_centroid_khz": centroid / 1000.0,
        "spectral_bandwidth_khz": bandwidth / 1000.0,
        "spectral_entropy": normalized_entropy(s),
        "spectral_flatness": spectral_flatness(s),
        "spectral_high_ratio_40k_plus": high_ratio,
        "temporal_mean": t_mean,
        "temporal_cv": t_std / (t_mean + EPS),
        "temporal_q90": float(np.percentile(t, 90)) if t.size else 0.0,
        "temporal_upper_spread": (
            float(np.percentile(t, 90) - np.percentile(t, 50)) if t.size else 0.0
        ),
    }


def load_scene(
    scene: SceneInput,
    freq_grid: np.ndarray,
    config: AnalysisConfig,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    v9_dir = Path(scene.v9_dir)
    csv_path = v9_dir / "v9_all_features.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "label" not in df.columns or "sample_id" not in df.columns:
        raise ValueError(f"{csv_path}必须包含label和sample_id列")

    df = df[df["label"].astype(str).str.upper().isin(VALID_LABELS)].copy()
    df["label"] = df["label"].astype(str).str.upper()
    index = load_npz_index(v9_dir)
    if not index:
        raise FileNotFoundError(f"{v9_dir}中没有找到residual_npz/*.npz")

    spectra: List[np.ndarray] = []
    temporals: List[np.ndarray] = []
    meta_rows: List[Dict[str, Any]] = []
    scalar_rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        npz_path = resolve_npz_path(row, v9_dir, index)
        if npz_path is None:
            meta_rows.append({
                "scene": scene.name,
                "sample_id": row.get("sample_id", ""),
                "label": row.get("label", ""),
                "status": "FAILED",
                "error": "未匹配到NPZ",
                "npz_path": "",
            })
            continue
        try:
            fp = sample_fingerprint_from_npz(npz_path, freq_grid, config)
            spectra.append(fp["spectrum"])
            temporals.append(fp["temporal"])
            scalar_row = {
                "scene": scene.name,
                "sample_id": str(row["sample_id"]),
                "label": str(row["label"]),
                **fp["scalar"],
            }
            scalar_rows.append(scalar_row)
            meta_rows.append({
                "scene": scene.name,
                "sample_id": str(row["sample_id"]),
                "label": str(row["label"]),
                "status": "OK",
                "error": "",
                "npz_path": str(npz_path),
                "matrix_key": fp["matrix_key"],
                "background_key": fp["background_key"],
                "npz_primary_method": fp["primary_method"],
                "npz_selected_method": fp["selected_method"],
            })
        except Exception as exc:
            meta_rows.append({
                "scene": scene.name,
                "sample_id": str(row.get("sample_id", "")),
                "label": str(row.get("label", "")),
                "status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
                "npz_path": str(npz_path),
            })

    meta_df = pd.DataFrame(meta_rows)
    scalar_df = pd.DataFrame(scalar_rows)
    ok_ids = set(meta_df.loc[meta_df["status"] == "OK", "sample_id"].astype(str))
    used_df = df[df["sample_id"].astype(str).isin(ok_ids)].copy().reset_index(drop=True)

    # 按used_df顺序重排矩阵，避免失败行造成错位。
    spectrum_map = {}
    temporal_map = {}
    for item, spec, temp in zip(
        [r for r in meta_rows if r.get("status") == "OK"],
        spectra,
        temporals,
    ):
        spectrum_map[str(item["sample_id"])] = spec
        temporal_map[str(item["sample_id"])] = temp
    spectra_arr = np.stack([spectrum_map[str(sid)] for sid in used_df["sample_id"]], axis=0)
    temporals_arr = np.stack([temporal_map[str(sid)] for sid in used_df["sample_id"]], axis=0)

    n_true = int(np.sum(used_df["label"] == "TRUE_LEAK"))
    n_false = int(np.sum(used_df["label"] == "FALSE_LEAK"))
    if n_true < 2 or n_false < 2:
        raise ValueError(
            f"{scene.name}成功样本不足：TRUE={n_true}, FALSE={n_false}；每类至少2个"
        )
    return used_df, spectra_arr, temporals_arr, meta_df.merge(
        scalar_df, on=["scene", "sample_id", "label"], how="left"
    )


# =============================================================================
# 4. 场景内TRUE-FALSE与AB共性
# =============================================================================


def bootstrap_positive_probability(
    true_matrix: np.ndarray,
    false_matrix: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_t = true_matrix.shape[0]
    n_f = false_matrix.shape[0]
    count = np.zeros(true_matrix.shape[1], dtype=float)
    for _ in range(max(int(iterations), 1)):
        t_idx = rng.integers(0, n_t, size=n_t)
        f_idx = rng.integers(0, n_f, size=n_f)
        delta = np.median(true_matrix[t_idx], axis=0) - np.median(false_matrix[f_idx], axis=0)
        count += delta > 0
    return count / max(int(iterations), 1)


def scene_contrast(
    matrix: np.ndarray,
    labels: np.ndarray,
    config: AnalysisConfig,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    y = np.asarray(labels).astype(str)
    true_m = matrix[y == "TRUE_LEAK"]
    false_m = matrix[y == "FALSE_LEAK"]

    true_med = np.median(true_m, axis=0)
    false_med = np.median(false_m, axis=0)
    delta = true_med - false_med

    spread_t = robust_scale(true_m, axis=0)
    spread_f = robust_scale(false_m, axis=0)
    pooled = 0.5 * (spread_t + spread_f)
    global_floor = max(float(np.nanmedian(pooled)) * 0.05, 1.0e-6)
    effect = delta / np.maximum(pooled, global_floor)

    support = bootstrap_positive_probability(
        true_m,
        false_m,
        config.bootstrap_iterations,
        rng,
    )
    stable = (
        (delta > 0)
        & (effect >= config.min_robust_effect)
        & (support >= config.bootstrap_positive_probability)
    )

    positive = np.maximum(delta, 0.0)
    weighted = positive * np.clip(effect, 0.0, 8.0) * support
    weighted = normalize_nonnegative(weighted)

    return {
        "true_median": true_med,
        "false_median": false_med,
        "delta": delta,
        "effect": effect,
        "positive_probability": support,
        "stable_positive": stable.astype(int),
        "weighted_positive": weighted,
        "true_q10": np.quantile(true_m, 0.10, axis=0),
        "true_q90": np.quantile(true_m, 0.90, axis=0),
        "false_q10": np.quantile(false_m, 0.10, axis=0),
        "false_q90": np.quantile(false_m, 0.90, axis=0),
    }


def build_ab_common(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    a_profile = normalize_nonnegative(a["weighted_positive"])
    b_profile = normalize_nonnegative(b["weighted_positive"])
    strict = (a["stable_positive"].astype(bool) & b["stable_positive"].astype(bool))
    soft_common = np.sqrt(np.maximum(a_profile, 0.0) * np.maximum(b_profile, 0.0))
    strict_common = np.where(strict, soft_common, 0.0)
    if np.sum(strict_common) <= EPS:
        # 不伪造严格共性，但保留软共性供诊断。
        common = normalize_nonnegative(soft_common)
    else:
        common = normalize_nonnegative(strict_common)
    return {
        "a_profile": a_profile,
        "b_profile": b_profile,
        "strict_mask": strict.astype(int),
        "soft_common": normalize_nonnegative(soft_common),
        "common": common,
    }


# =============================================================================
# 5. 实验室WAV原型
# =============================================================================


def discover_wavs(paths: Sequence[str]) -> List[Path]:
    found: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file() and p.suffix.lower() == ".wav":
            found.append(p)
        elif p.is_dir():
            found.extend(sorted(p.rglob("*.wav")))
    unique: Dict[str, Path] = {str(p.resolve()): p for p in found}
    return list(unique.values())


def read_wav_preserve_scale(path: Path) -> Tuple[int, np.ndarray]:
    fs, raw = wavfile.read(str(path))
    original_dtype = raw.dtype
    if raw.ndim > 1:
        x = np.mean(raw.astype(np.float64), axis=1)
    else:
        x = raw.astype(np.float64, copy=False)
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        full_scale = float(max(abs(info.min), abs(info.max)))
        x = x / max(full_scale, 1.0)
    finite = np.isfinite(x)
    if not np.any(finite):
        raise ValueError(f"WAV全部无效: {path}")
    fill = float(np.median(x[finite]))
    x = np.where(finite, x, fill)
    x = x - float(np.mean(x))
    return int(fs), np.asarray(x, dtype=np.float64)


def segment_signal(x: np.ndarray, fs: int, seconds: float, hop_seconds: float) -> List[np.ndarray]:
    seg_len = max(int(round(seconds * fs)), 256)
    hop = max(int(round(hop_seconds * fs)), 1)
    if len(x) <= seg_len:
        return [x] if len(x) >= 256 else []
    starts = list(range(0, len(x) - seg_len + 1, hop))
    if starts and starts[-1] + seg_len < len(x) and len(x) - seg_len - starts[-1] > hop // 2:
        starts.append(len(x) - seg_len)
    return [x[s:s + seg_len] for s in starts]


def lab_segment_fingerprint(
    x: np.ndarray,
    fs: int,
    freq_grid: np.ndarray,
    config: AnalysisConfig,
) -> Dict[str, Any]:
    nperseg = min(config.lab_nperseg, len(x))
    if nperseg < 256:
        raise ValueError("实验室切片太短")
    hop = min(config.lab_hop_length, nperseg)
    noverlap = max(0, nperseg - hop)
    nfft = max(config.lab_nfft, nperseg)

    freq, _, z = signal.stft(
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
    mask = (freq >= config.freq_low_hz) & (freq <= config.freq_high_hz)
    if np.sum(mask) < 5:
        raise ValueError(
            f"实验室WAV在{config.freq_low_hz:g}-{config.freq_high_hz:g}Hz内频点不足，fs={fs}"
        )
    power = np.abs(z[mask]) ** 2
    freq_use = freq[mask]

    spec_native = np.sqrt(np.maximum(np.mean(power, axis=1), 0.0))
    spec_native = smooth_1d(spec_native, config.spectral_smooth_bins)
    spectrum = normalize_nonnegative(interp_vector(freq_use, spec_native, freq_grid))

    frame = np.sum(power, axis=0)
    scale = float(np.median(frame[frame > 0])) if np.any(frame > 0) else 1.0
    frame_measure = np.log1p(frame / max(scale, EPS))
    q_grid = np.linspace(0.0, 1.0, config.n_temporal_quantiles)
    temporal = np.quantile(frame_measure, q_grid)
    t_scale = float(np.percentile(temporal, 95)) if temporal.size else 0.0
    if t_scale > EPS:
        temporal = np.clip(temporal / t_scale, 0.0, None)

    return {
        "spectrum": spectrum,
        "temporal": temporal,
        "scalar": scalar_features(spectrum, temporal, freq_grid),
    }


def load_lab_prototype(
    wav_paths: Sequence[str],
    freq_grid: np.ndarray,
    config: AnalysisConfig,
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    wavs = discover_wavs(wav_paths)
    if not wavs:
        raise FileNotFoundError("没有找到实验室真实泄漏WAV")

    spectra: List[np.ndarray] = []
    temporals: List[np.ndarray] = []
    scalar_rows: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    for wav in wavs:
        try:
            fs, x = read_wav_preserve_scale(wav)
            segments = segment_signal(
                x,
                fs,
                config.lab_segment_seconds,
                config.lab_segment_hop_seconds,
            )
            for seg_i, segment in enumerate(segments):
                try:
                    fp = lab_segment_fingerprint(segment, fs, freq_grid, config)
                    spectra.append(fp["spectrum"])
                    temporals.append(fp["temporal"])
                    scalar_rows.append({
                        "wav": str(wav),
                        "segment": seg_i,
                        **fp["scalar"],
                    })
                    rows.append({
                        "wav": str(wav),
                        "sample_rate_hz": fs,
                        "segment": seg_i,
                        "status": "OK",
                        "error": "",
                    })
                except Exception as exc:
                    rows.append({
                        "wav": str(wav),
                        "sample_rate_hz": fs,
                        "segment": seg_i,
                        "status": "FAILED",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
        except Exception as exc:
            rows.append({
                "wav": str(wav),
                "sample_rate_hz": np.nan,
                "segment": -1,
                "status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
            })

    if len(spectra) < config.min_lab_segments:
        raise ValueError(
            f"实验室有效切片只有{len(spectra)}个，少于min_lab_segments={config.min_lab_segments}"
        )

    spec_arr = np.stack(spectra, axis=0)
    temp_arr = np.stack(temporals, axis=0)
    prototype = normalize_nonnegative(np.median(spec_arr, axis=0))
    ranks = rank_percentile_rows(spec_arr)
    segment_support = np.mean(ranks >= config.lab_active_spectrum_quantile, axis=0)
    proto_rank = rank_percentile_rows(prototype[None, :])[0]
    active = (
        (proto_rank >= config.lab_active_spectrum_quantile)
        & (segment_support >= config.lab_min_segment_support)
    )

    result = {
        "spectrum": prototype,
        "spectrum_q10": np.quantile(spec_arr, 0.10, axis=0),
        "spectrum_q90": np.quantile(spec_arr, 0.90, axis=0),
        "segment_support": segment_support,
        "active_mask": active.astype(int),
        "temporal": np.median(temp_arr, axis=0),
        "temporal_q10": np.quantile(temp_arr, 0.10, axis=0),
        "temporal_q90": np.quantile(temp_arr, 0.90, axis=0),
        "scalar_df": pd.DataFrame(scalar_rows),
        "n_segments": np.array(len(spectra)),
    }
    return result, pd.DataFrame(rows)


# =============================================================================
# 6. AB与实验室共同部分
# =============================================================================


def coarse_profile(freq: np.ndarray, profile: np.ndarray, width_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    low = float(freq[0])
    high = float(freq[-1])
    edges = np.arange(low, high + width_hz, width_hz)
    if edges[-1] < high:
        edges = np.append(edges, high)
    centers: List[float] = []
    values: List[float] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (freq >= lo) & (freq < hi if hi < high else freq <= hi)
        if not np.any(mask):
            continue
        centers.append(0.5 * (lo + hi))
        values.append(float(np.sum(profile[mask])))
    return np.asarray(centers), normalize_nonnegative(np.asarray(values))


def build_ab_lab_common(
    freq_grid: np.ndarray,
    ab: Dict[str, np.ndarray],
    lab: Dict[str, np.ndarray],
    config: AnalysisConfig,
) -> Dict[str, Any]:
    ab_profile = normalize_nonnegative(ab["common"])
    lab_profile = normalize_nonnegative(lab["spectrum"])
    strict_ab = ab["strict_mask"].astype(bool)
    lab_active = lab["active_mask"].astype(bool)

    broad = np.sqrt(ab_profile * lab_profile)
    broad = normalize_nonnegative(broad)

    ab_shape, ab_envelope = remove_spectral_envelope(
        ab_profile,
        config.envelope_window_bins,
        config.envelope_polyorder,
    )
    lab_shape, lab_envelope = remove_spectral_envelope(
        lab_profile,
        config.envelope_window_bins,
        config.envelope_polyorder,
    )
    shape_common = np.sqrt(np.maximum(ab_shape, 0.0) * np.maximum(lab_shape, 0.0))
    shape_common = normalize_nonnegative(shape_common)

    final = (
        config.broadband_common_weight * broad
        + config.local_shape_common_weight * shape_common
    )
    final = np.where(strict_ab & lab_active, final, 0.0)
    final = normalize_nonnegative(final)

    positive = final[final > 0]
    if positive.size:
        threshold = float(np.quantile(positive, config.common_weight_quantile))
        selected = (final >= threshold) & strict_ab & lab_active
    else:
        threshold = np.nan
        selected = np.zeros_like(final, dtype=bool)

    # 频带提取与小间隙合并。
    regions = contiguous_regions(selected)
    regions = merge_regions_by_gap(regions, freq_grid, config.merge_gap_hz)
    kept: List[Tuple[int, int]] = []
    for s, e in regions:
        width = float(freq_grid[e] - freq_grid[s])
        if width >= config.minimum_common_bandwidth_hz or s == e:
            kept.append((s, e))

    band_rows: List[Dict[str, Any]] = []
    for band_id, (s, e) in enumerate(kept, start=1):
        sl = slice(s, e + 1)
        local = final[sl]
        peak_rel = int(np.argmax(local))
        peak_idx = s + peak_rel
        band_rows.append({
            "band_id": band_id,
            "start_hz": float(freq_grid[s]),
            "end_hz": float(freq_grid[e]),
            "bandwidth_hz": float(freq_grid[e] - freq_grid[s]),
            "peak_hz": float(freq_grid[peak_idx]),
            "final_common_mass": float(np.sum(final[sl])),
            "ab_common_mass": float(np.sum(ab_profile[sl])),
            "lab_spectrum_mass": float(np.sum(lab_profile[sl])),
            "mean_lab_segment_support": float(np.mean(lab["segment_support"][sl])),
            "mean_final_weight": float(np.mean(final[sl])),
        })

    coarse_freq, coarse_ab = coarse_profile(freq_grid, ab_profile, config.coarse_bandwidth_hz)
    _, coarse_lab = coarse_profile(freq_grid, lab_profile, config.coarse_bandwidth_hz)

    similarities = {
        "ab_vs_lab_cosine_fine": cosine_similarity(ab_profile, lab_profile),
        "ab_vs_lab_overlap_fine": overlap_coefficient(ab_profile, lab_profile),
        "ab_vs_lab_js_similarity_fine": js_similarity(ab_profile, lab_profile),
        "ab_vs_lab_shape_correlation": pearson_safe(ab_shape, lab_shape),
        "ab_vs_lab_cosine_coarse": cosine_similarity(coarse_ab, coarse_lab),
        "ab_vs_lab_overlap_coarse": overlap_coefficient(coarse_ab, coarse_lab),
        "A_vs_B_cosine": cosine_similarity(ab["a_profile"], ab["b_profile"]),
        "A_vs_B_overlap": overlap_coefficient(ab["a_profile"], ab["b_profile"]),
        "A_vs_B_js_similarity": js_similarity(ab["a_profile"], ab["b_profile"]),
        "n_AB_strict_bins": int(np.sum(strict_ab)),
        "n_lab_active_bins": int(np.sum(lab_active)),
        "n_final_selected_bins": int(np.sum(selected)),
        "n_final_bands": len(band_rows),
    }

    return {
        "ab_profile": ab_profile,
        "lab_profile": lab_profile,
        "broad_common": broad,
        "ab_shape": ab_shape,
        "lab_shape": lab_shape,
        "ab_envelope": ab_envelope,
        "lab_envelope": lab_envelope,
        "shape_common": shape_common,
        "final_common": final,
        "selected_mask": selected.astype(int),
        "threshold": threshold,
        "bands": pd.DataFrame(band_rows),
        "similarities": similarities,
        "coarse_freq": coarse_freq,
        "coarse_ab": coarse_ab,
        "coarse_lab": coarse_lab,
    }


def scalar_commonality(
    scene_a_scalar: pd.DataFrame,
    scene_b_scalar: pd.DataFrame,
    lab_scalar: pd.DataFrame,
) -> pd.DataFrame:
    feature_cols = [
        "spectral_centroid_khz",
        "spectral_bandwidth_khz",
        "spectral_entropy",
        "spectral_flatness",
        "spectral_high_ratio_40k_plus",
        "temporal_mean",
        "temporal_cv",
        "temporal_q90",
        "temporal_upper_spread",
    ]
    rows: List[Dict[str, Any]] = []
    for feature in feature_cols:
        row: Dict[str, Any] = {"feature": feature}
        scene_stats = {}
        directions = []
        lab_med = float(np.median(pd.to_numeric(lab_scalar[feature], errors="coerce").dropna()))
        row["lab_median"] = lab_med
        for name, df in (("A", scene_a_scalar), ("B", scene_b_scalar)):
            t = pd.to_numeric(
                df.loc[df["label"] == "TRUE_LEAK", feature], errors="coerce"
            ).dropna().to_numpy(float)
            f = pd.to_numeric(
                df.loc[df["label"] == "FALSE_LEAK", feature], errors="coerce"
            ).dropna().to_numpy(float)
            t_med = float(np.median(t))
            f_med = float(np.median(f))
            pooled = max(0.5 * (float(robust_scale(t)) + float(robust_scale(f))), 1.0e-6)
            effect = (t_med - f_med) / pooled
            direction = 1 if effect > 0 else (-1 if effect < 0 else 0)
            directions.append(direction)
            row[f"{name}_true_median"] = t_med
            row[f"{name}_false_median"] = f_med
            row[f"{name}_robust_effect"] = effect
            scene_stats[name] = (t_med, f_med, pooled)

        ab_consistent = directions[0] == directions[1] and directions[0] != 0
        true_center = float(np.median([scene_stats["A"][0], scene_stats["B"][0]]))
        false_center = float(np.median([scene_stats["A"][1], scene_stats["B"][1]]))
        scale = float(np.median([scene_stats["A"][2], scene_stats["B"][2]]))
        dist_true = abs(lab_med - true_center) / max(scale, 1.0e-6)
        dist_false = abs(lab_med - false_center) / max(scale, 1.0e-6)
        row["AB_direction_consistent"] = int(ab_consistent)
        row["AB_consensus_direction"] = (
            "TRUE_HIGHER" if directions[0] > 0 and ab_consistent
            else "TRUE_LOWER" if directions[0] < 0 and ab_consistent
            else "INCONSISTENT"
        )
        row["lab_distance_to_AB_true"] = dist_true
        row["lab_distance_to_AB_false"] = dist_false
        row["lab_closer_to_AB_true"] = int(dist_true < dist_false)
        row["common_supported"] = int(ab_consistent and dist_true < dist_false)
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 7. 输出
# =============================================================================


def save_figures(
    output_dir: Path,
    freq_grid: np.ndarray,
    scene_a: Dict[str, np.ndarray],
    scene_b: Dict[str, np.ndarray],
    ab: Dict[str, np.ndarray],
    lab: Dict[str, np.ndarray],
    final: Dict[str, Any],
    temporal_a: Dict[str, np.ndarray],
    temporal_b: Dict[str, np.ndarray],
    temporal_ab: Dict[str, np.ndarray],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = output_dir / "figures"
    ensure_dir(fig_dir)
    fkhz = freq_grid / 1000.0

    plt.figure(figsize=(13, 7))
    plt.plot(fkhz, normalize_nonnegative(scene_a["weighted_positive"]), label="A: TRUE-FALSE")
    plt.plot(fkhz, normalize_nonnegative(scene_b["weighted_positive"]), label="B: TRUE-FALSE")
    plt.plot(fkhz, ab["common"], label="AB common", linewidth=2.2)
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Normalized weight")
    plt.title("Scene A / B leak contrasts and AB common component")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "01_A_B_and_AB_common_spectrum.png", dpi=160)
    plt.close()

    plt.figure(figsize=(13, 7))
    plt.plot(fkhz, final["ab_profile"], label="AB common")
    plt.plot(fkhz, final["lab_profile"], label="Laboratory leak prototype")
    plt.plot(fkhz, final["final_common"], label="AB ∩ Lab final common", linewidth=2.4)
    selected = final["selected_mask"].astype(bool)
    if np.any(selected):
        plt.fill_between(fkhz, 0, final["final_common"], where=selected, alpha=0.25, label="Selected common bins")
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Normalized weight")
    plt.title("AB common component vs laboratory real-leak prototype")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "02_AB_vs_lab_common_spectrum.png", dpi=160)
    plt.close()

    plt.figure(figsize=(13, 7))
    plt.plot(fkhz, final["ab_shape"], label="AB local spectral shape")
    plt.plot(fkhz, final["lab_shape"], label="Lab local spectral shape")
    plt.plot(fkhz, final["shape_common"], label="Positive shared local shape", linewidth=2.0)
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Log spectrum minus smooth envelope")
    plt.title("Local spectral-shape commonality")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "03_AB_lab_local_shape.png", dpi=160)
    plt.close()

    q = np.linspace(0.0, 1.0, len(lab["temporal"]))
    plt.figure(figsize=(12, 7))
    plt.plot(q, normalize_nonnegative(temporal_a["weighted_positive"]), label="A temporal TRUE-FALSE")
    plt.plot(q, normalize_nonnegative(temporal_b["weighted_positive"]), label="B temporal TRUE-FALSE")
    plt.plot(q, temporal_ab["common"], label="AB temporal common", linewidth=2.0)
    plt.plot(q, normalize_nonnegative(lab["temporal"]), label="Lab temporal prototype")
    plt.xlabel("Frame-energy quantile")
    plt.ylabel("Normalized profile")
    plt.title("Temporal-distribution commonality")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "04_temporal_commonality.png", dpi=160)
    plt.close()


def sample_similarity_table(
    scene_name: str,
    df: pd.DataFrame,
    spectra: np.ndarray,
    ab_profile: np.ndarray,
    lab_profile: np.ndarray,
    final_profile: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        spec = spectra[i]
        rows.append({
            "scene": scene_name,
            "sample_id": row["sample_id"],
            "label": row["label"],
            "similarity_to_AB_cosine": cosine_similarity(spec, ab_profile),
            "similarity_to_lab_cosine": cosine_similarity(spec, lab_profile),
            "similarity_to_final_common_cosine": cosine_similarity(spec, final_profile),
            "overlap_with_AB": overlap_coefficient(spec, ab_profile),
            "overlap_with_lab": overlap_coefficient(spec, lab_profile),
            "overlap_with_final_common": overlap_coefficient(spec, final_profile),
        })
    return pd.DataFrame(rows)


def run_analysis(config: AnalysisConfig) -> Dict[str, Path]:
    output_dir = Path(config.output_dir)
    ensure_dir(output_dir)
    freq_grid = np.linspace(config.freq_low_hz, config.freq_high_hz, config.n_spectral_bins)
    rng = np.random.default_rng(config.random_state)

    print("=" * 100)
    print("AB场景共性 + 实验室真实泄漏锚定分析")
    print("=" * 100)
    print("场景A:", config.scene_A.v9_dir)
    print("场景B:", config.scene_B.v9_dir)
    print("实验室WAV:", config.lab_wav_paths)
    print("v9残差:", f"representation={config.factory_representation}, variant={config.residual_variant}")

    print("\n第一步：读取A、B的v9残差")
    a_df, a_spec, a_temp, a_meta = load_scene(config.scene_A, freq_grid, config)
    b_df, b_spec, b_temp, b_meta = load_scene(config.scene_B, freq_grid, config)
    print(
        f"  {config.scene_A.name}: n={len(a_df)}, TRUE={sum(a_df.label=='TRUE_LEAK')}, FALSE={sum(a_df.label=='FALSE_LEAK')}"
    )
    print(
        f"  {config.scene_B.name}: n={len(b_df)}, TRUE={sum(b_df.label=='TRUE_LEAK')}, FALSE={sum(b_df.label=='FALSE_LEAK')}"
    )

    print("\n第二步：分别计算A_TRUE-A_FALSE、B_TRUE-B_FALSE")
    a_contrast = scene_contrast(a_spec, a_df["label"].to_numpy(), config, rng)
    b_contrast = scene_contrast(b_spec, b_df["label"].to_numpy(), config, rng)
    ta_contrast = scene_contrast(a_temp, a_df["label"].to_numpy(), config, rng)
    tb_contrast = scene_contrast(b_temp, b_df["label"].to_numpy(), config, rng)
    print("  A稳定TRUE增强频谱维度:", int(np.sum(a_contrast["stable_positive"])))
    print("  B稳定TRUE增强频谱维度:", int(np.sum(b_contrast["stable_positive"])))

    print("\n第三步：提取A与B共同部分")
    ab = build_ab_common(a_contrast, b_contrast)
    temporal_ab = build_ab_common(ta_contrast, tb_contrast)
    print("  AB严格共同频谱维度:", int(np.sum(ab["strict_mask"])))
    print("  A/B频谱余弦相似度:", f"{cosine_similarity(ab['a_profile'], ab['b_profile']):.4f}")

    print("\n第四步：读取实验室真实泄漏WAV并建立原型")
    lab, lab_log = load_lab_prototype(config.lab_wav_paths, freq_grid, config)
    print("  实验室有效切片数:", int(np.asarray(lab["n_segments"]).item()))

    print("\n第五步：寻找AB共同部分与实验室泄漏的共同部分")
    final = build_ab_lab_common(freq_grid, ab, lab, config)
    for key, value in final["similarities"].items():
        print(f"  {key}: {value}")

    # 逐频点表格。
    scene_a_csv = output_dir / "scene_A_TRUE_minus_FALSE_spectrum.csv"
    scene_b_csv = output_dir / "scene_B_TRUE_minus_FALSE_spectrum.csv"
    ab_csv = output_dir / "AB_common_spectrum.csv"
    lab_csv = output_dir / "laboratory_real_leak_prototype_spectrum.csv"
    final_csv = output_dir / "AB_lab_common_spectrum.csv"
    bands_csv = output_dir / "AB_lab_common_frequency_bands.csv"
    temporal_csv = output_dir / "AB_lab_temporal_commonality.csv"
    scalar_csv = output_dir / "AB_lab_scalar_feature_commonality.csv"
    sample_csv = output_dir / "factory_sample_similarity_to_AB_and_lab.csv"
    mapping_csv = output_dir / "input_mapping_and_quality.csv"
    lab_log_csv = output_dir / "laboratory_wav_segment_log.csv"
    metrics_csv = output_dir / "commonality_metrics.csv"
    npz_path = output_dir / "AB_lab_common_fingerprint.npz"
    report_path = output_dir / "AB_lab_commonality_report.txt"
    config_path = output_dir / "AB_lab_run_config.json"

    def scene_spectrum_df(c: Dict[str, np.ndarray]) -> pd.DataFrame:
        return pd.DataFrame({
            "frequency_hz": freq_grid,
            "frequency_khz": freq_grid / 1000.0,
            "true_median": c["true_median"],
            "false_median": c["false_median"],
            "delta_TRUE_minus_FALSE": c["delta"],
            "robust_effect": c["effect"],
            "bootstrap_positive_probability": c["positive_probability"],
            "stable_TRUE_enhancement": c["stable_positive"],
            "weighted_positive_contrast": c["weighted_positive"],
        })

    scene_spectrum_df(a_contrast).to_csv(scene_a_csv, index=False, encoding="utf-8-sig")
    scene_spectrum_df(b_contrast).to_csv(scene_b_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "frequency_hz": freq_grid,
        "frequency_khz": freq_grid / 1000.0,
        "A_positive_profile": ab["a_profile"],
        "B_positive_profile": ab["b_profile"],
        "AB_strict_supported": ab["strict_mask"],
        "AB_soft_common": ab["soft_common"],
        "AB_common": ab["common"],
    }).to_csv(ab_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "frequency_hz": freq_grid,
        "frequency_khz": freq_grid / 1000.0,
        "lab_prototype": lab["spectrum"],
        "lab_q10": lab["spectrum_q10"],
        "lab_q90": lab["spectrum_q90"],
        "lab_segment_support": lab["segment_support"],
        "lab_active": lab["active_mask"],
    }).to_csv(lab_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "frequency_hz": freq_grid,
        "frequency_khz": freq_grid / 1000.0,
        "AB_common": final["ab_profile"],
        "lab_prototype": final["lab_profile"],
        "broadband_common": final["broad_common"],
        "AB_local_shape": final["ab_shape"],
        "lab_local_shape": final["lab_shape"],
        "local_shape_common": final["shape_common"],
        "final_AB_lab_common": final["final_common"],
        "final_selected": final["selected_mask"],
        "AB_strict_supported": ab["strict_mask"],
        "lab_active": lab["active_mask"],
        "lab_segment_support": lab["segment_support"],
    }).to_csv(final_csv, index=False, encoding="utf-8-sig")
    final["bands"].to_csv(bands_csv, index=False, encoding="utf-8-sig")

    q = np.linspace(0.0, 1.0, config.n_temporal_quantiles)
    pd.DataFrame({
        "quantile": q,
        "A_temporal_positive": ta_contrast["weighted_positive"],
        "B_temporal_positive": tb_contrast["weighted_positive"],
        "AB_temporal_strict_supported": temporal_ab["strict_mask"],
        "AB_temporal_common": temporal_ab["common"],
        "lab_temporal_prototype": lab["temporal"],
        "lab_temporal_q10": lab["temporal_q10"],
        "lab_temporal_q90": lab["temporal_q90"],
    }).to_csv(temporal_csv, index=False, encoding="utf-8-sig")

    scalar_df = scalar_commonality(
        a_meta[a_meta["status"] == "OK"].copy(),
        b_meta[b_meta["status"] == "OK"].copy(),
        lab["scalar_df"],
    )
    scalar_df.to_csv(scalar_csv, index=False, encoding="utf-8-sig")

    sample_df = pd.concat([
        sample_similarity_table(config.scene_A.name, a_df, a_spec, final["ab_profile"], final["lab_profile"], final["final_common"]),
        sample_similarity_table(config.scene_B.name, b_df, b_spec, final["ab_profile"], final["lab_profile"], final["final_common"]),
    ], ignore_index=True)
    sample_df.to_csv(sample_csv, index=False, encoding="utf-8-sig")

    pd.concat([a_meta, b_meta], ignore_index=True).to_csv(mapping_csv, index=False, encoding="utf-8-sig")
    lab_log.to_csv(lab_log_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame([final["similarities"]]).to_csv(metrics_csv, index=False, encoding="utf-8-sig")

    np.savez_compressed(
        npz_path,
        frequency_hz=freq_grid,
        scene_A_positive_profile=ab["a_profile"],
        scene_B_positive_profile=ab["b_profile"],
        AB_strict_mask=ab["strict_mask"],
        AB_common_spectrum=ab["common"],
        laboratory_leak_spectrum=lab["spectrum"],
        laboratory_active_mask=lab["active_mask"],
        AB_lab_final_common_spectrum=final["final_common"],
        AB_lab_final_selected_mask=final["selected_mask"],
        AB_temporal_common=temporal_ab["common"],
        laboratory_temporal_profile=lab["temporal"],
        residual_variant=np.asarray(config.residual_variant),
        factory_representation=np.asarray(config.factory_representation),
    )

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)

    if config.save_figures:
        save_figures(
            output_dir,
            freq_grid,
            a_contrast,
            b_contrast,
            ab,
            lab,
            final,
            ta_contrast,
            tb_contrast,
            temporal_ab,
        )

    # 报告。
    sim = final["similarities"]
    lines: List[str] = []
    lines.append("A/B场景共性与实验室真实泄漏共性分析")
    lines.append("=" * 92)
    lines.append(f"场景A: {config.scene_A.name} | n={len(a_df)}")
    lines.append(f"场景B: {config.scene_B.name} | n={len(b_df)}")
    lines.append(f"实验室有效切片: {int(np.asarray(lab['n_segments']).item())}")
    lines.append(f"工厂残差表示: {config.factory_representation}_{config.residual_variant}")
    lines.append("")
    lines.append("一、A和B的共性")
    lines.append(f"  A稳定TRUE增强频谱维度: {int(np.sum(a_contrast['stable_positive']))}")
    lines.append(f"  B稳定TRUE增强频谱维度: {int(np.sum(b_contrast['stable_positive']))}")
    lines.append(f"  AB严格共同频谱维度: {int(np.sum(ab['strict_mask']))}")
    lines.append(f"  A/B余弦相似度: {sim['A_vs_B_cosine']:.6f}")
    lines.append(f"  A/B重叠系数: {sim['A_vs_B_overlap']:.6f}")
    lines.append(f"  A/B JS相似度: {sim['A_vs_B_js_similarity']:.6f}")
    lines.append("")
    lines.append("二、AB共同部分与实验室泄漏的共性")
    lines.append(f"  细频谱余弦相似度: {sim['ab_vs_lab_cosine_fine']:.6f}")
    lines.append(f"  细频谱重叠系数: {sim['ab_vs_lab_overlap_fine']:.6f}")
    lines.append(f"  细频谱JS相似度: {sim['ab_vs_lab_js_similarity_fine']:.6f}")
    lines.append(f"  5kHz粗频带余弦相似度: {sim['ab_vs_lab_cosine_coarse']:.6f}")
    lines.append(f"  5kHz粗频带重叠系数: {sim['ab_vs_lab_overlap_coarse']:.6f}")
    lines.append(f"  去包络局部谱形状相关: {sim['ab_vs_lab_shape_correlation']:.6f}")
    lines.append(f"  最终选中频率维度: {sim['n_final_selected_bins']}")
    lines.append(f"  最终共同频带数: {sim['n_final_bands']}")
    lines.append("")
    lines.append("三、最终共同频带")
    if final["bands"].empty:
        lines.append("  未找到同时满足AB严格支持和实验室稳定活跃条件的连续频带。")
    else:
        for _, r in final["bands"].iterrows():
            lines.append(
                f"  Band {int(r['band_id'])}: {r['start_hz']/1000:.3f}-{r['end_hz']/1000:.3f} kHz, "
                f"peak={r['peak_hz']/1000:.3f} kHz, common_mass={r['final_common_mass']:.5f}"
            )
    lines.append("")
    lines.append("四、解释边界")
    lines.append("  1. 结果是频谱/时间残差指纹，不是纯净可播放泄漏WAV。")
    lines.append("  2. 实验室只有真实泄漏而没有实验室无泄漏背景时，实验室设备本底可能混入原型。")
    lines.append("  3. 频谱粗尺度相似度通常比逐频点相似度更重要，因为场景传播会造成频响变化。")
    lines.append("  4. 最终共性必须结合AB的TRUE/FALSE差异和实验室稳定出现两项条件解释。")
    lines.append("")
    lines.append("主要文件:")
    lines.append(f"  {ab_csv.name}: A与B共同候选")
    lines.append(f"  {lab_csv.name}: 实验室泄漏原型")
    lines.append(f"  {final_csv.name}: AB与实验室逐频点共同部分")
    lines.append(f"  {bands_csv.name}: 最终连续共同频带")
    lines.append(f"  {scalar_csv.name}: 统计属性共性")
    lines.append(f"  {sample_csv.name}: 每个工厂样本与AB/实验室的相似度")
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 100)
    print("处理完成")
    print("输出目录:", output_dir)
    print("AB共同频谱:", ab_csv)
    print("AB与实验室共同频谱:", final_csv)
    print("共同频带:", bands_csv)
    print("报告:", report_path)

    return {
        "scene_A": scene_a_csv,
        "scene_B": scene_b_csv,
        "AB_common": ab_csv,
        "lab_prototype": lab_csv,
        "AB_lab_common": final_csv,
        "bands": bands_csv,
        "temporal": temporal_csv,
        "scalar": scalar_csv,
        "samples": sample_csv,
        "report": report_path,
        "fingerprint": npz_path,
    }


# =============================================================================
# 8. 自检
# =============================================================================


def make_synthetic_v9_scene(
    root: Path,
    scene_name: str,
    rng: np.random.Generator,
    n_true: int,
    n_false: int,
    freq: np.ndarray,
) -> None:
    ensure_dir(root / "residual_npz")
    rows = []
    n_frames = 32
    leak_shape = (
        1.2 * np.exp(-0.5 * ((freq - 46_000.0) / 5_500.0) ** 2)
        + 0.8 * np.exp(-0.5 * ((freq - 64_000.0) / 4_500.0) ** 2)
    )
    scene_shape = 0.25 * np.sin(freq / 7000.0 + (0.4 if scene_name.endswith("B") else 0.0))
    for label, n in (("TRUE_LEAK", n_true), ("FALSE_LEAK", n_false)):
        for i in range(n):
            sid = f"{scene_name}_{label}_{i:03d}"
            base = np.maximum(0.05 + 0.02 * rng.random((len(freq), n_frames)), 0.0)
            if label == "TRUE_LEAK":
                temporal = 0.7 + 0.3 * np.sin(np.linspace(0, 3 * np.pi, n_frames)) ** 2
                excess = base + np.maximum(leak_shape + scene_shape, 0.0)[:, None] * temporal[None, :]
            else:
                false_line = 0.35 * np.exp(-0.5 * ((freq - 33_000.0) / 800.0) ** 2)
                excess = base + false_line[:, None]
            excess *= np.exp(rng.normal(0.0, 0.06, size=excess.shape))
            bg_db = -70.0 + scene_shape[:, None] + rng.normal(0.0, 0.4, size=excess.shape)
            np.savez_compressed(
                root / "residual_npz" / f"{safe_slug(sid)}.npz",
                freq_hz=freq,
                time_s=np.linspace(0, 1, n_frames),
                excess_power=excess,
                excess_power_median=excess,
                excess_power_selected=excess,
                excess_power_plane=excess,
                background_power_db=bg_db,
                background_power_db_median=bg_db,
                background_power_db_selected=bg_db,
                background_power_db_plane=bg_db,
                primary_method=np.asarray("median"),
                selected_method=np.asarray("plane"),
            )
            rows.append({
                "sample_id": sid,
                "dataset": scene_name,
                "scene": scene_name,
                "time": "synthetic",
                "center": i,
                "label": label,
            })
    pd.DataFrame(rows).to_csv(root / "v9_all_features.csv", index=False, encoding="utf-8-sig")


def make_synthetic_lab_wav(path: Path, fs: int = 192_000, seconds: float = 5.0) -> None:
    rng = np.random.default_rng(123)
    t = np.arange(int(fs * seconds)) / fs
    carrier = (
        0.05 * np.sin(2 * np.pi * 46_000.0 * t)
        + 0.035 * np.sin(2 * np.pi * 64_000.0 * t)
        + 0.025 * np.sin(2 * np.pi * 52_000.0 * t)
    )
    broadband = rng.normal(0.0, 0.02, size=t.size)
    # 高频带宽噪声。
    sos = signal.butter(4, [38_000, 72_000], btype="bandpass", fs=fs, output="sos")
    broadband = signal.sosfilt(sos, broadband)
    envelope = 0.8 + 0.2 * np.sin(2 * np.pi * 2.2 * t) ** 2
    x = envelope * (carrier + broadband)
    x = np.clip(x, -0.95, 0.95)
    wavfile.write(str(path), fs, (x * 32767).astype(np.int16))


def run_self_test() -> None:
    root = Path(__file__).resolve().parent / "_AB_lab_self_test"
    if root.exists():
        shutil.rmtree(root)
    ensure_dir(root)
    freq = np.linspace(20_000.0, 80_000.0, 256)
    rng = np.random.default_rng(42)
    make_synthetic_v9_scene(root / "A", "factory_A", rng, 12, 12, freq)
    make_synthetic_v9_scene(root / "B", "factory_B", rng, 12, 12, freq)
    make_synthetic_lab_wav(root / "lab_real_leak.wav")

    cfg = AnalysisConfig(
        scene_A=SceneInput("factory_A", str(root / "A")),
        scene_B=SceneInput("factory_B", str(root / "B")),
        lab_wav_paths=[str(root / "lab_real_leak.wav")],
        output_dir=str(root / "results"),
        residual_variant="median",
        bootstrap_iterations=80,
        min_lab_segments=3,
    )
    outputs = run_analysis(cfg)
    metrics = pd.read_csv(root / "results" / "commonality_metrics.csv")
    cosine = float(metrics.loc[0, "ab_vs_lab_cosine_coarse"])
    if not np.isfinite(cosine) or cosine < 0.45:
        raise AssertionError(f"自检失败：AB与实验室粗频带余弦相似度过低={cosine}")
    for path in outputs.values():
        if not Path(path).exists():
            raise AssertionError(f"自检输出缺失: {path}")
    print("\n自检通过：已验证A/B场景共性、实验室WAV原型和AB∩Lab共同频带输出。")
    print("自检目录:", root)


# =============================================================================
# 9. CLI
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="提取场景A/B共性，并寻找其与实验室真实泄漏WAV的共同部分"
    )
    parser.add_argument("--config", type=str, help="JSON配置文件")
    parser.add_argument("--self-test", action="store_true", help="运行合成数据自检")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.self_test:
        run_self_test()
        return
    if not args.config:
        raise SystemExit("请提供 --config AB_lab_config.json，或运行 --self-test")
    config = read_json_config(Path(args.config))
    run_analysis(config)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n[失败]", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)
