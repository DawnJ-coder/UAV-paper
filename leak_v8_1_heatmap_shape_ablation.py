# -*- coding: utf-8 -*-
"""
leak_v8_1_heatmap_shape_ablation.py

v8.1：热力图核心形态特征消融版

为什么要改 v8？
    v8 加入了很多 heatmap 特征，但效果没有明显优于 v7。
    重要特征里出现了很多 hm_weighted_cx / hm_weighted_cy / asymmetry 这类“位置特征”，
    说明模型可能学到了热点在图上的位置偏移，而不是我们真正关心的泄漏喷流形态。

v8.1 的核心改动：
    1. 只保留核心热图形态特征：
        - 红区面积是否小
        - 热点是否集中
        - 热点是否细长
        - 是否单主峰
        - 是否弥散
        - 是否具有定向核心
    2. 删除热图位置特征：
        - cx / cy / peak_x / peak_y / asymmetry_lr / asymmetry_ud
    3. 做三组消融实验：
        A: v7_only
        B: heatmap_shape_only
        C: v7_plus_heatmap_shape
    4. 每组都做按 time_folder 整组留出验证。
    5. 重点观察 HM20260626_144226.ld 是否改善。

运行:
    python leak_v8_1_heatmap_shape_ablation.py

输入:
    C:\\Users\\jiangxinru6\\Desktop\\wurenji\\leak_v4_compare_results\\merged_feature_dataset.csv

热力图路径:
    真泄漏:
        results_spectrum_HM20260626_142938.ld
        results_spectrum_HM20260626_143034.ld
        results_spectrum_HM20260626_144226.ld
        results_spectrum_HM20260626_144325.ld

    假泄漏:
        results_cs_spectrum_HM20260626_142938.ld
        results_cs_spectrum_HM20260626_143034.ld
        results_cs_spectrum_HM20260626_144226.ld
        results_cs_spectrum_HM20260626_144325.ld

输出:
    C:\\Users\\jiangxinru6\\Desktop\\wurenji\\leak_v8_1_heatmap_shape_ablation_results
"""

import os
import json
import math
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from scipy import ndimage


# ============================================================
# 1. 路径配置
# ============================================================

BASE_DIR = r"C:\Users\jiangxinru6\Desktop\wurenji"

MERGED_FEATURE_CSV = os.path.join(
    BASE_DIR,
    "leak_v4_compare_results",
    "merged_feature_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "leak_v8_1_heatmap_shape_ablation_results"
)

GROUP_COL = "time"
LABEL_COL = "label"

TIME_FOLDERS = [
    "HM20260626_142938.ld",
    "HM20260626_143034.ld",
    "HM20260626_144226.ld",
    "HM20260626_144325.ld",
]

TRUE_HEATMAP_DIRS = {
    t: os.path.join(BASE_DIR, f"results_spectrum_{t}") for t in TIME_FOLDERS
}

FALSE_HEATMAP_DIRS = {
    t: os.path.join(BASE_DIR, f"results_cs_spectrum_{t}") for t in TIME_FOLDERS
}

# 不作为特征的列
DROP_COLS = [
    "dataset",
    "label",
    "time",
    "center",
    "best_direction",
    "energy_direction",
    "decay_direction",
    "representative_file",
]

# 绝对能量/幅值特征，v8.1 默认删除
ABSOLUTE_ENERGY_EXACT_OR_PREFIX = [
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
    "time_energy_mean",
    "time_energy_std",
    "time_rms",
]

ALLOW_TIME_FEATURES = [
    "time_energy_cv",
    "time_energy_kurtosis",
    "time_energy_max_mean_ratio",
]

DROP_ABSOLUTE_ENERGY_FEATURES = True

# 频率异常过滤
DROP_BAD_FREQUENCY_FEATURES = True
FREQ_LOW_HZ = 20000
BAD_FREQ_RATIO_THRESHOLD = 0.30

# time 内部增强
ADD_TIME_ROBUST_Z = True
ADD_TIME_RANK_PCT = True

# 阈值搜索
THRESHOLD_GRID = np.linspace(0.01, 0.99, 99)
THRESHOLD_METRIC = "balanced_accuracy"

# rank 规则：测试集中每个 time 的 top 50% 判 TRUE，用于验证
RANK_TRUE_FRACTION_FOR_BINARY = 0.50

# 三档输出
RANK_TRUE_LIKE_PCT = 0.70
RANK_FALSE_LIKE_PCT = 0.30

RANDOM_STATE = 42

# v8.1 只保留这些核心 heatmap shape 特征
CORE_HEATMAP_FEATURES = [
    "hm_hot_area_p99_ratio",
    "hm_hot_area_p97_ratio",
    "hm_hot_area_p95_ratio",
    "hm_hot_area_p90_ratio",
    "hm_hot_area_p85_ratio",

    "hm_largest_component_ratio_p95",
    "hm_largest_component_ratio_p90",
    "hm_num_components_p95",
    "hm_num_components_p90",

    "hm_p95_elongation",
    "hm_p90_elongation",
    "hm_weighted_elongation",
    "hm_weighted_eccentricity",

    "hm_compactness_p95",
    "hm_compactness_p90",

    "hm_entropy_2d",
    "hm_effective_area_ratio",
    "hm_energy_concentration_top5",
    "hm_energy_concentration_top10",
    "hm_energy_concentration_top20",

    "hm_radial_spread_norm",
    "hm_core_to_outer_energy_ratio",

    "hm_directed_core_score",
    "hm_diffuse_score",
    "hm_single_core_score",
    "hm_shape_leak_like_score",
]


# ============================================================
# 2. 基础工具
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def safe_name(s):
    return str(s).replace("\\", "_").replace("/", "_").replace(":", "_").replace(".", "_")


def normalize_center_id(center):
    """
    center 可能是 0、00、'00'，统一成两位字符串。
    """
    try:
        if pd.isna(center):
            return "00"
    except Exception:
        pass

    s = str(center).strip()

    if s.endswith(".0"):
        s = s[:-2]

    digits = "".join(ch for ch in s if ch.isdigit())

    if digits == "":
        return s

    return digits.zfill(2)


def label_to_binary(labels):
    return np.array([1 if str(x) == "TRUE_LEAK" else 0 for x in labels], dtype=int)


def binary_to_label(v):
    return "TRUE_LEAK" if int(v) == 1 else "FALSE_LEAK"


def safe_float_array(s):
    arr = pd.to_numeric(s, errors="coerce")
    arr = arr.replace([np.inf, -np.inf], np.nan)
    return arr


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
    return (np.asarray(prob, dtype=float) >= threshold).astype(int)


def find_best_threshold(y_true, prob, metric="balanced_accuracy", grid=None):
    if grid is None:
        grid = THRESHOLD_GRID

    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)

    best_t = 0.5
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

        rows.append({
            "threshold": float(t),
            "score": float(score),
            **m
        })

        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t, float(best_score), pd.DataFrame(rows)


# ============================================================
# 3. heatmap 图像读取与核心形态特征
# ============================================================

def find_heatmap_path(label, time_folder, center_id):
    center_id = normalize_center_id(center_id)

    if str(label) == "TRUE_LEAK":
        d = TRUE_HEATMAP_DIRS.get(time_folder, "")
    else:
        d = FALSE_HEATMAP_DIRS.get(time_folder, "")

    filename = f"heatmap_{time_folder}_center_{center_id}.png"
    path = os.path.join(d, filename)

    return path


def rgb_to_hsv_np(rgb):
    """
    rgb: float 0-1, shape (H,W,3)
    return hsv 0-1
    """
    import colorsys

    h, w, _ = rgb.shape
    flat = rgb.reshape(-1, 3)
    hsv = np.zeros_like(flat)

    for i, (r, g, b) in enumerate(flat):
        hsv[i] = colorsys.rgb_to_hsv(float(r), float(g), float(b))

    return hsv.reshape(h, w, 3)


def crop_main_heatmap(rgb):
    """
    从 matplotlib 保存的 png 中尽量裁出主热力图区，排除标题、坐标轴、colorbar。
    原理：
        - 找饱和度较高的彩色区域；
        - 连通域中选择面积大且形状不是细长色条的区域；
        - 如果失败，则使用中心方形裁剪。
    """
    h, w, _ = rgb.shape

    hsv = rgb_to_hsv_np(rgb)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # 彩色区域：热力图和 colorbar 都会被选中，文字黑色通常不会被选中
    mask = (sat > 0.20) & (val > 0.05)

    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5)))

    labels, n = ndimage.label(mask)

    candidates = []

    for lab in range(1, n + 1):
        ys, xs = np.where(labels == lab)
        if len(xs) < 200:
            continue

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        bw = x1 - x0 + 1
        bh = y1 - y0 + 1
        area = len(xs)
        aspect = bw / (bh + 1e-12)

        # 排除很细的 colorbar
        if aspect < 0.35 or aspect > 3.0:
            continue

        # 热力图区应该相对较大
        score = area * min(aspect, 1.0 / aspect)

        candidates.append((score, x0, x1, y0, y1, area, aspect))

    if candidates:
        candidates = sorted(candidates, reverse=True)
        _, x0, x1, y0, y1, _, _ = candidates[0]

        # 适当向内/外调整，尽量只保留主图
        pad_x = int(0.02 * (x1 - x0 + 1))
        pad_y = int(0.02 * (y1 - y0 + 1))

        x0 = max(0, x0 - pad_x)
        x1 = min(w - 1, x1 + pad_x)
        y0 = max(0, y0 - pad_y)
        y1 = min(h - 1, y1 + pad_y)

        crop = rgb[y0:y1 + 1, x0:x1 + 1, :]
        return crop

    # 兜底：中心方形裁剪，避开 colorbar 和标题
    size = int(min(h, w) * 0.62)
    cy = int(h * 0.52)
    cx = int(w * 0.48)
    y0 = max(0, cy - size // 2)
    y1 = min(h, y0 + size)
    x0 = max(0, cx - size // 2)
    x1 = min(w, x0 + size)

    return rgb[y0:y1, x0:x1, :]


def heatmap_rgb_to_intensity(crop_rgb):
    """
    把伪彩色热力图近似转换成 0-1 强度图。
    适用于 jet/turbo 风格：
        红/黄强，蓝弱。
    """
    rgb = np.asarray(crop_rgb, dtype=float)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    hsv = rgb_to_hsv_np(rgb)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # 只取彩色区域，去掉可能残留的白底/黑字/坐标轴
    color_mask = (sat > 0.18) & (val > 0.05)

    # jet/turbo近似热度：红黄高，蓝低
    raw = 1.20 * r + 0.45 * g - 0.85 * b

    # 非彩色区域设为低值
    raw = np.where(color_mask, raw, np.nan)

    finite = np.isfinite(raw)

    if finite.sum() < 50:
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        raw = gray
        finite = np.isfinite(raw)

    vals = raw[finite]

    lo = np.percentile(vals, 1)
    hi = np.percentile(vals, 99)

    intensity = (raw - lo) / (hi - lo + 1e-12)
    intensity = np.clip(intensity, 0, 1)

    intensity[~np.isfinite(intensity)] = 0.0

    # 平滑一下，减小 colormap 条纹影响
    intensity = ndimage.gaussian_filter(intensity, sigma=1.0)
    intensity = np.clip(intensity, 0, 1)

    return intensity


def component_shape_features(mask):
    """
    输入二值 mask，返回核心连通域形状特征。
    不输出位置类特征，只输出面积、连通性、细长度、紧致度。
    """
    h, w = mask.shape
    total = h * w

    labels, n = ndimage.label(mask)

    if n == 0 or mask.sum() == 0:
        return {
            "area_ratio": 0.0,
            "largest_component_ratio_to_hot": 0.0,
            "largest_component_ratio_to_total": 0.0,
            "num_components": 0,
            "elongation": 1.0,
            "eccentricity": 0.0,
            "compactness": 0.0,
        }

    component_areas = ndimage.sum(mask.astype(float), labels, index=np.arange(1, n + 1))
    component_areas = np.asarray(component_areas, dtype=float)

    largest_idx = int(np.argmax(component_areas)) + 1
    largest_area = float(component_areas.max())
    hot_area = float(mask.sum())

    comp = labels == largest_idx

    ys, xs = np.where(comp)

    # 主轴/次轴：用协方差矩阵
    if len(xs) >= 3:
        coords = np.vstack([xs.astype(float), ys.astype(float)]).T
        coords = coords - coords.mean(axis=0, keepdims=True)
        cov = np.cov(coords, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(np.maximum(eigvals, 1e-12))
        minor = math.sqrt(eigvals[0])
        major = math.sqrt(eigvals[1])
        elongation = major / (minor + 1e-12)
        eccentricity = math.sqrt(max(0.0, 1.0 - eigvals[0] / (eigvals[1] + 1e-12)))
    else:
        elongation = 1.0
        eccentricity = 0.0

    # 紧致度 compactness = 4*pi*area / perimeter^2
    eroded = ndimage.binary_erosion(comp)
    boundary = comp & (~eroded)
    perimeter = float(boundary.sum())
    compactness = 4.0 * math.pi * largest_area / ((perimeter + 1e-12) ** 2)

    return {
        "area_ratio": hot_area / (total + 1e-12),
        "largest_component_ratio_to_hot": largest_area / (hot_area + 1e-12),
        "largest_component_ratio_to_total": largest_area / (total + 1e-12),
        "num_components": int(n),
        "elongation": float(elongation),
        "eccentricity": float(eccentricity),
        "compactness": float(compactness),
    }


def weighted_shape_features(intensity):
    """
    对连续强度图做加权形状特征，不输出位置坐标。
    """
    img = np.asarray(intensity, dtype=float)
    h, w = img.shape
    eps = 1e-12

    total_energy = float(img.sum())

    if total_energy <= eps:
        return {
            "hm_weighted_elongation": 1.0,
            "hm_weighted_eccentricity": 0.0,
            "hm_radial_spread_norm": 1.0,
            "hm_entropy_2d": 1.0,
            "hm_effective_area_ratio": 1.0,
            "hm_energy_concentration_top5": 0.0,
            "hm_energy_concentration_top10": 0.0,
            "hm_energy_concentration_top20": 0.0,
            "hm_core_to_outer_energy_ratio": 0.0,
        }

    yy, xx = np.mgrid[0:h, 0:w]
    weights = img / (total_energy + eps)

    cx = float((weights * xx).sum())
    cy = float((weights * yy).sum())

    dx = xx - cx
    dy = yy - cy

    cov_xx = float((weights * dx * dx).sum())
    cov_yy = float((weights * dy * dy).sum())
    cov_xy = float((weights * dx * dy).sum())

    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=float)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.maximum(eigvals, eps))

    minor = math.sqrt(eigvals[0])
    major = math.sqrt(eigvals[1])
    elongation = major / (minor + eps)
    eccentricity = math.sqrt(max(0.0, 1.0 - eigvals[0] / (eigvals[1] + eps)))

    radius = np.sqrt(dx * dx + dy * dy)
    radial_spread = float((weights * radius).sum())
    diag = math.sqrt(h * h + w * w)
    radial_spread_norm = radial_spread / (diag + eps)

    # entropy / effective area
    flat_w = weights.ravel()
    flat_w = flat_w[flat_w > eps]
    entropy = -float(np.sum(flat_w * np.log(flat_w + eps)))
    entropy_norm = entropy / (math.log(h * w + eps) + eps)
    effective_area_ratio = math.exp(entropy) / (h * w + eps)

    # energy concentration top k%
    flat = np.sort(img.ravel())[::-1]
    n = len(flat)

    def top_energy_ratio(frac):
        k = max(1, int(n * frac))
        return float(flat[:k].sum() / (total_energy + eps))

    conc5 = top_energy_ratio(0.05)
    conc10 = top_energy_ratio(0.10)
    conc20 = top_energy_ratio(0.20)

    # core/outer energy ratio
    p95 = np.percentile(img, 95)
    p70 = np.percentile(img, 70)

    core_energy = float(img[img >= p95].sum())
    outer_energy = float(img[img <= p70].sum())
    core_outer_ratio = core_energy / (outer_energy + eps)

    return {
        "hm_weighted_elongation": float(elongation),
        "hm_weighted_eccentricity": float(eccentricity),
        "hm_radial_spread_norm": float(radial_spread_norm),
        "hm_entropy_2d": float(entropy_norm),
        "hm_effective_area_ratio": float(effective_area_ratio),
        "hm_energy_concentration_top5": float(conc5),
        "hm_energy_concentration_top10": float(conc10),
        "hm_energy_concentration_top20": float(conc20),
        "hm_core_to_outer_energy_ratio": float(core_outer_ratio),
    }


def extract_heatmap_shape_features(image_path):
    """
    核心 heatmap 形态特征。
    不输出 cx/cy/peak_x/peak_y/asymmetry 等位置特征。
    """
    # 默认失败值
    features = {k: 0.0 for k in CORE_HEATMAP_FEATURES}
    features["hm_read_success"] = 0

    if not os.path.exists(image_path):
        return features

    try:
        img = Image.open(image_path).convert("RGB")
        rgb = np.asarray(img, dtype=float) / 255.0

        crop = crop_main_heatmap(rgb)
        intensity = heatmap_rgb_to_intensity(crop)

        features["hm_read_success"] = 1

        # 形态阈值区域
        percentiles = {
            "p99": 99,
            "p97": 97,
            "p95": 95,
            "p90": 90,
            "p85": 85,
        }

        comp_feats_by_p = {}

        for name, p in percentiles.items():
            thr = np.percentile(intensity, p)
            mask = intensity >= thr
            cf = component_shape_features(mask)
            comp_feats_by_p[name] = cf
            features[f"hm_hot_area_{name}_ratio"] = cf["area_ratio"]

        # 连通域核心特征
        features["hm_largest_component_ratio_p95"] = comp_feats_by_p["p95"]["largest_component_ratio_to_hot"]
        features["hm_largest_component_ratio_p90"] = comp_feats_by_p["p90"]["largest_component_ratio_to_hot"]
        features["hm_num_components_p95"] = comp_feats_by_p["p95"]["num_components"]
        features["hm_num_components_p90"] = comp_feats_by_p["p90"]["num_components"]

        features["hm_p95_elongation"] = comp_feats_by_p["p95"]["elongation"]
        features["hm_p90_elongation"] = comp_feats_by_p["p90"]["elongation"]
        features["hm_compactness_p95"] = comp_feats_by_p["p95"]["compactness"]
        features["hm_compactness_p90"] = comp_feats_by_p["p90"]["compactness"]

        # 连续加权特征
        wf = weighted_shape_features(intensity)
        features.update(wf)

        # 构造几个符合直觉的组合分数
        elong = features["hm_weighted_elongation"]
        ecc = features["hm_weighted_eccentricity"]
        largest = features["hm_largest_component_ratio_p95"]
        area95 = features["hm_hot_area_p95_ratio"]
        entropy = features["hm_entropy_2d"]
        diffuse = features["hm_effective_area_ratio"]
        conc10 = features["hm_energy_concentration_top10"]
        radial = features["hm_radial_spread_norm"]

        # 单一主核心分数：最大连通域越接近1、连通域数量越少越高
        ncomp = features["hm_num_components_p95"]
        features["hm_single_core_score"] = float(
            largest / (1.0 + 0.15 * max(0.0, ncomp - 1.0))
        )

        # 定向核心分数：细长/偏心 + 单核心 + 能量集中 + 热区不太大
        features["hm_directed_core_score"] = float(
            np.clip(
                (math.log1p(elong) / math.log(6.0)) *
                (0.5 + 0.5 * ecc) *
                largest *
                conc10 *
                (1.0 / (1.0 + 8.0 * area95)),
                0.0,
                5.0
            )
        )

        # 弥散分数：熵高、有效面积大、径向扩散大、集中度低
        features["hm_diffuse_score"] = float(
            np.clip(
                0.35 * entropy +
                0.35 * diffuse +
                0.20 * radial +
                0.10 * (1.0 - conc10),
                0.0,
                1.0
            )
        )

        # 泄漏形态分数：定向核心高、单核心高、弥散低
        features["hm_shape_leak_like_score"] = float(
            np.clip(
                0.45 * features["hm_directed_core_score"] +
                0.35 * features["hm_single_core_score"] +
                0.20 * (1.0 - features["hm_diffuse_score"]),
                0.0,
                5.0
            )
        )

        # 确保只返回核心特征
        for k in CORE_HEATMAP_FEATURES:
            if k not in features:
                features[k] = 0.0

        return features

    except Exception:
        return features


def build_heatmap_feature_table(df, output_dir):
    rows = []
    missing = 0

    print("\n开始提取 v8.1 核心 heatmap 形态特征...")

    for i, row in df.iterrows():
        label = str(row[LABEL_COL])
        time_folder = str(row[GROUP_COL])
        center_id = normalize_center_id(row["center"])

        path = find_heatmap_path(label, time_folder, center_id)
        feats = extract_heatmap_shape_features(path)

        if feats.get("hm_read_success", 0) == 0:
            missing += 1

        out = {
            "row_index": i,
            "label": label,
            "time": time_folder,
            "center": center_id,
            "heatmap_path": path,
            **feats,
        }

        rows.append(out)

        if (i + 1) % 20 == 0 or (i + 1) == len(df):
            print(f"  已处理 {i + 1}/{len(df)} 张/行")

    hdf = pd.DataFrame(rows)

    path = os.path.join(output_dir, "v8_1_heatmap_core_shape_features.csv")
    hdf.to_csv(path, index=False, encoding="utf-8-sig")

    print("核心 heatmap 特征表:", path)
    print("缺失/失败 heatmap 数量:", missing)

    return hdf, path, missing


# ============================================================
# 4. v7 稳健特征构造
# ============================================================

def is_absolute_energy_feature(col):
    c = col.lower()

    if col in ALLOW_TIME_FEATURES:
        return False

    for key in ABSOLUTE_ENERGY_EXACT_OR_PREFIX:
        if c == key.lower():
            return True

    if c.startswith("energy_"):
        return True

    return False


def get_initial_numeric_features(df):
    feature_cols = []

    for c in df.columns:
        if c in DROP_COLS:
            continue

        temp = safe_float_array(df[c])
        valid_ratio = temp.notna().mean()

        if valid_ratio > 0.8:
            feature_cols.append(c)

    return feature_cols


def remove_unstable_v7_features(df, feature_cols, output_dir):
    removed_rows = []
    kept = []

    for c in feature_cols:
        reason = None

        if DROP_ABSOLUTE_ENERGY_FEATURES and is_absolute_energy_feature(c):
            reason = "absolute_energy_or_amplitude_feature"

        if reason is None and DROP_BAD_FREQUENCY_FEATURES:
            lc = c.lower()
            if ("freq" in lc or "centroid" in lc or "rolloff" in lc):
                vals = safe_float_array(df[c]).fillna(0)
                bad_ratio = float((vals < FREQ_LOW_HZ).mean())
                if bad_ratio > BAD_FREQ_RATIO_THRESHOLD:
                    reason = f"bad_frequency_feature_below_{FREQ_LOW_HZ}_ratio_{bad_ratio:.3f}"

        if reason is None:
            kept.append(c)
        else:
            removed_rows.append({
                "feature": c,
                "reason": reason,
            })

    rdf = pd.DataFrame(removed_rows)
    path = os.path.join(output_dir, "v8_1_removed_v7_features.csv")
    rdf.to_csv(path, index=False, encoding="utf-8-sig")

    return kept, path


def make_numeric_df(df, cols):
    x = pd.DataFrame(index=df.index)

    for c in cols:
        vals = safe_float_array(df[c])
        med = vals.median()
        if not np.isfinite(med):
            med = 0.0
        x[c] = vals.fillna(med).astype(float)

    return x


def add_time_internal_features(df, base_x, group_col, add_z=True, add_rank=True):
    out = base_x.copy()
    groups = df[group_col].astype(str)
    eps = 1e-12

    for c in base_x.columns:
        values = base_x[c].astype(float)

        if add_z:
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

        if add_rank:
            r_col = f"{c}__time_rank_pct"
            rank_values = values.groupby(groups).rank(method="average", pct=True)
            out[r_col] = rank_values.fillna(0.5).astype(float)

    return out


def build_feature_sets(df, hdf, output_dir):
    """
    构造三组消融特征：
        A: v7_only
        B: heatmap_shape_only
        C: v7_plus_heatmap_shape
    """
    # v7 稳健基础特征
    init_v7 = get_initial_numeric_features(df)
    kept_v7, removed_v7_path = remove_unstable_v7_features(df, init_v7, output_dir)
    v7_base = make_numeric_df(df, kept_v7)

    # heatmap 核心形态特征
    hm_cols = [c for c in CORE_HEATMAP_FEATURES if c in hdf.columns]
    hm_base = make_numeric_df(hdf, hm_cols)

    # 不把 hm_read_success 加进去，避免模型学“有没有文件”这种无意义信息

    feature_sets = {}

    # A: v7 only
    x_v7 = add_time_internal_features(
        df,
        v7_base,
        GROUP_COL,
        add_z=ADD_TIME_ROBUST_Z,
        add_rank=ADD_TIME_RANK_PCT
    )
    feature_sets["A_v7_only"] = {
        "X": x_v7,
        "base_cols": kept_v7,
        "description": "只用 v7 稳健特征，不使用 heatmap。",
    }

    # B: heatmap only
    x_hm = add_time_internal_features(
        df,
        hm_base,
        GROUP_COL,
        add_z=ADD_TIME_ROBUST_Z,
        add_rank=ADD_TIME_RANK_PCT
    )
    feature_sets["B_heatmap_shape_only"] = {
        "X": x_hm,
        "base_cols": hm_cols,
        "description": "只用核心 heatmap 形态特征。",
    }

    # C: v7 + heatmap
    combo_base = pd.concat([v7_base, hm_base], axis=1)
    x_combo = add_time_internal_features(
        df,
        combo_base,
        GROUP_COL,
        add_z=ADD_TIME_ROBUST_Z,
        add_rank=ADD_TIME_RANK_PCT
    )
    feature_sets["C_v7_plus_heatmap_shape"] = {
        "X": x_combo,
        "base_cols": kept_v7 + hm_cols,
        "description": "v7 稳健特征 + 核心 heatmap 形态特征。",
    }

    # 保存特征表
    for name, info in feature_sets.items():
        out_df = pd.concat(
            [
                df[[c for c in ["dataset", "time", "center", "label"] if c in df.columns]].reset_index(drop=True),
                info["X"].reset_index(drop=True)
            ],
            axis=1
        )
        out_path = os.path.join(output_dir, f"v8_1_features_{name}.csv")
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        info["feature_csv"] = out_path

        cols_path = os.path.join(output_dir, f"v8_1_used_features_{name}.txt")
        save_text(cols_path, "\n".join(info["X"].columns.tolist()))
        info["used_features_txt"] = cols_path

    meta = {
        "initial_v7_numeric_feature_count": len(init_v7),
        "kept_v7_base_feature_count": len(kept_v7),
        "heatmap_core_shape_feature_count": len(hm_cols),
        "removed_v7_path": removed_v7_path,
    }

    return feature_sets, meta


# ============================================================
# 5. 模型训练与分组验证
# ============================================================

def build_classifier():
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=700,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
    )


def get_group_oof_probabilities(X_train, y_train, groups_train):
    y_train = np.asarray(y_train, dtype=int)
    groups_train = np.asarray(groups_train).astype(str)

    unique_groups = sorted(pd.unique(groups_train).tolist())

    oof_prob = np.zeros(len(y_train), dtype=float)
    filled = np.zeros(len(y_train), dtype=bool)

    # 优先按训练集内部 time 做 OOF
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

    # 兜底
    if not np.all(filled):
        from sklearn.model_selection import StratifiedKFold

        min_class_count = min(np.sum(y_train == 0), np.sum(y_train == 1))
        n_splits = max(2, min(5, int(min_class_count)))

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

        for tr_idx, val_idx in cv.split(X_train, y_train):
            clf = build_classifier()
            clf.fit(X_train.iloc[tr_idx], y_train[tr_idx])
            oof_prob[val_idx] = clf.predict_proba(X_train.iloc[val_idx])[:, 1]
            filled[val_idx] = True

    return oof_prob


def add_probability_rank_columns(pred_df):
    pred_df = pred_df.copy()

    pred_df["prob_rank_pct_in_group"] = 0.5
    pred_df["prob_relative_minmax_in_group"] = 0.5

    for g, idx in pred_df.groupby("test_group").groups.items():
        sub_prob = pred_df.loc[idx, "prob_TRUE_LEAK"].astype(float)

        pred_df.loc[idx, "prob_rank_pct_in_group"] = sub_prob.rank(method="average", pct=True)

        pmin = float(sub_prob.min())
        pmax = float(sub_prob.max())

        if abs(pmax - pmin) < 1e-12:
            rel = pd.Series(0.5, index=idx)
        else:
            rel = (sub_prob - pmin) / (pmax - pmin)

        pred_df.loc[idx, "prob_relative_minmax_in_group"] = rel

    cutoff = 1.0 - RANK_TRUE_FRACTION_FOR_BINARY

    pred_df["rank_binary_pred"] = np.where(
        pred_df["prob_rank_pct_in_group"] > cutoff,
        "TRUE_LEAK",
        "FALSE_LEAK"
    )

    pred_df["rank_level"] = np.select(
        [
            pred_df["prob_rank_pct_in_group"] >= RANK_TRUE_LIKE_PCT,
            pred_df["prob_rank_pct_in_group"] <= RANK_FALSE_LIKE_PCT,
        ],
        [
            "TRUE_LIKE",
            "FALSE_LIKE",
        ],
        default="SUSPECT"
    )

    final = []

    for _, r in pred_df.iterrows():
        model_pred = r["model_pred"]
        rank_pred = r["rank_binary_pred"]

        if model_pred == rank_pred:
            final.append(model_pred)
        else:
            final.append("SUSPECT")

    pred_df["final_decision"] = final

    return pred_df


def calc_final_decision_metrics(pred_df):
    if len(pred_df) == 0:
        return {
            "decisive_rate": 0.0,
            "decisive_accuracy": 0.0,
            "strict_accuracy_suspect_as_wrong": 0.0,
            "n_suspect": 0,
            "n_decisive": 0,
        }

    decisive_mask = pred_df["final_decision"].isin(["TRUE_LEAK", "FALSE_LEAK"])
    n_decisive = int(decisive_mask.sum())
    n_suspect = int((~decisive_mask).sum())

    decisive_rate = n_decisive / len(pred_df)

    if n_decisive > 0:
        decisive_accuracy = float(
            (pred_df.loc[decisive_mask, "final_decision"] ==
             pred_df.loc[decisive_mask, "true_label"]).mean()
        )
    else:
        decisive_accuracy = 0.0

    strict_correct = (
        (pred_df["final_decision"] == pred_df["true_label"]) &
        decisive_mask
    )
    strict_accuracy = float(strict_correct.mean())

    return {
        "decisive_rate": float(decisive_rate),
        "decisive_accuracy": float(decisive_accuracy),
        "strict_accuracy_suspect_as_wrong": float(strict_accuracy),
        "n_suspect": n_suspect,
        "n_decisive": n_decisive,
    }


def validate_feature_set(df, X, experiment_name, output_dir):
    y_all = label_to_binary(df[LABEL_COL].astype(str).values)
    groups = df[GROUP_COL].astype(str).values
    unique_groups = sorted(pd.unique(groups).tolist())

    group_rows = []
    all_pred_rows = []

    exp_dir = os.path.join(output_dir, experiment_name)
    ensure_dir(exp_dir)

    print(f"\n开始实验 {experiment_name} 按时间点整组验证...")
    print("特征数:", X.shape[1])

    for test_group in unique_groups:
        test_mask = groups == test_group
        train_mask = ~test_mask

        X_train = X.loc[train_mask].reset_index(drop=True)
        X_test = X.loc[test_mask].reset_index(drop=True)

        y_train = y_all[train_mask]
        y_test = y_all[test_mask]

        groups_train = groups[train_mask]
        test_df = df.loc[test_mask].reset_index(drop=True)

        if len(np.unique(y_train)) < 2:
            print(f"  [跳过] {test_group}: 训练集中不足两类")
            continue

        # OOF 选阈值
        oof_prob = get_group_oof_probabilities(X_train, y_train, groups_train)
        best_t, best_score, threshold_curve = find_best_threshold(
            y_train,
            oof_prob,
            metric=THRESHOLD_METRIC,
            grid=THRESHOLD_GRID
        )

        curve_path = os.path.join(
            exp_dir,
            f"threshold_curve_train_without_{safe_name(test_group)}.csv"
        )
        threshold_curve.to_csv(curve_path, index=False, encoding="utf-8-sig")

        clf = build_classifier()
        clf.fit(X_train, y_train)

        prob = clf.predict_proba(X_test)[:, 1]

        default_pred_binary = threshold_predict(prob, 0.5)
        model_pred_binary = threshold_predict(prob, best_t)

        m_default = metrics_from_pred(y_test, default_pred_binary)
        m_model = metrics_from_pred(y_test, model_pred_binary)
        auc = safe_auc(y_test, prob)

        group_pred_rows = []

        for i in range(len(test_df)):
            true_label = binary_to_label(y_test[i])
            default_pred = binary_to_label(default_pred_binary[i])
            model_pred = binary_to_label(model_pred_binary[i])

            row = {
                "experiment": experiment_name,
                "test_group": test_group,
                "dataset": test_df.loc[i, "dataset"] if "dataset" in test_df.columns else "",
                "time": test_df.loc[i, "time"] if "time" in test_df.columns else "",
                "center": test_df.loc[i, "center"] if "center" in test_df.columns else "",
                "true_label": true_label,
                "prob_TRUE_LEAK": float(prob[i]),
                "best_threshold": best_t,
                "default_pred_0p5": default_pred,
                "default_correct": int(default_pred == true_label),
                "model_pred": model_pred,
                "model_correct": int(model_pred == true_label),
            }

            group_pred_rows.append(row)

        group_pred_df = pd.DataFrame(group_pred_rows)
        group_pred_df = add_probability_rank_columns(group_pred_df)

        rank_pred_binary = label_to_binary(group_pred_df["rank_binary_pred"].values)
        m_rank = metrics_from_pred(y_test, rank_pred_binary)

        final_metrics = calc_final_decision_metrics(group_pred_df)

        print(
            f"  {test_group}: "
            f"best_t={best_t:.3f}, "
            f"default_acc={m_default['accuracy']:.3f}, "
            f"model_acc={m_model['accuracy']:.3f}, "
            f"rank_acc={m_rank['accuracy']:.3f}, "
            f"final_decisive_acc={final_metrics['decisive_accuracy']:.3f}, "
            f"suspect={final_metrics['n_suspect']}, "
            f"auc={auc if not np.isnan(auc) else 'NA'}"
        )

        group_rows.append({
            "experiment": experiment_name,
            "test_group": test_group,
            "n_test": len(y_test),
            "n_true": int(np.sum(y_test == 1)),
            "n_false": int(np.sum(y_test == 0)),
            "feature_count": X.shape[1],
            "best_threshold": best_t,
            "train_oof_best_score": best_score,
            "auc": auc,

            "default_accuracy_0p5": m_default["accuracy"],
            "default_balanced_accuracy_0p5": m_default["balanced_accuracy"],

            "model_accuracy": m_model["accuracy"],
            "model_balanced_accuracy": m_model["balanced_accuracy"],
            "model_recall_TRUE_LEAK": m_model["recall_TRUE_LEAK"],
            "model_recall_FALSE_LEAK": m_model["recall_FALSE_LEAK"],

            "rank_accuracy": m_rank["accuracy"],
            "rank_balanced_accuracy": m_rank["balanced_accuracy"],
            "rank_recall_TRUE_LEAK": m_rank["recall_TRUE_LEAK"],
            "rank_recall_FALSE_LEAK": m_rank["recall_FALSE_LEAK"],

            "final_decisive_rate": final_metrics["decisive_rate"],
            "final_decisive_accuracy": final_metrics["decisive_accuracy"],
            "final_strict_accuracy_suspect_as_wrong": final_metrics["strict_accuracy_suspect_as_wrong"],
            "final_n_suspect": final_metrics["n_suspect"],
            "final_n_decisive": final_metrics["n_decisive"],
        })

        all_pred_rows.extend(group_pred_df.to_dict(orient="records"))

    group_df = pd.DataFrame(group_rows)
    pred_df = pd.DataFrame(all_pred_rows)

    group_csv = os.path.join(exp_dir, f"{experiment_name}_group_summary.csv")
    pred_csv = os.path.join(exp_dir, f"{experiment_name}_predictions.csv")
    wrong_csv = os.path.join(exp_dir, f"{experiment_name}_model_misclassified.csv")
    suspect_csv = os.path.join(exp_dir, f"{experiment_name}_suspect_samples.csv")

    group_df.to_csv(group_csv, index=False, encoding="utf-8-sig")
    pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")

    if len(pred_df):
        pred_df[pred_df["model_correct"] == 0].to_csv(wrong_csv, index=False, encoding="utf-8-sig")
        pred_df[pred_df["final_decision"] == "SUSPECT"].to_csv(suspect_csv, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(wrong_csv, index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(suspect_csv, index=False, encoding="utf-8-sig")

    # 全数据训练重要性
    y = y_all
    clf_final = build_classifier()
    clf_final.fit(X, y)

    importance_df = pd.DataFrame({
        "feature": X.columns.tolist(),
        "importance": clf_final.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance_csv = os.path.join(exp_dir, f"{experiment_name}_feature_importance.csv")
    importance_df.to_csv(importance_csv, index=False, encoding="utf-8-sig")

    return {
        "experiment": experiment_name,
        "group_df": group_df,
        "pred_df": pred_df,
        "group_csv": group_csv,
        "pred_csv": pred_csv,
        "wrong_csv": wrong_csv,
        "suspect_csv": suspect_csv,
        "importance_csv": importance_csv,
        "importance_df": importance_df,
        "feature_count": X.shape[1],
    }


# ============================================================
# 6. 专项分析：144226
# ============================================================

def analyze_144226(feature_sets, hdf, df, all_results, output_dir):
    target = "HM20260626_144226.ld"
    out_dir = os.path.join(output_dir, "diagnosis_144226")
    ensure_dir(out_dir)

    # 保存 144226 的 heatmap 核心特征真假对比
    sub_hm = hdf[hdf["time"].astype(str) == target].copy()
    compare_rows = []

    for c in CORE_HEATMAP_FEATURES:
        if c not in sub_hm.columns:
            continue

        vals = safe_float_array(sub_hm[c])
        true_vals = vals[sub_hm["label"] == "TRUE_LEAK"].dropna()
        false_vals = vals[sub_hm["label"] == "FALSE_LEAK"].dropna()

        if len(true_vals) == 0 or len(false_vals) == 0:
            continue

        true_mean = float(true_vals.mean())
        false_mean = float(false_vals.mean())
        true_std = float(true_vals.std())
        false_std = float(false_vals.std())
        pooled = math.sqrt((true_std ** 2 + false_std ** 2) / 2.0) + 1e-12
        d = (true_mean - false_mean) / pooled

        compare_rows.append({
            "feature": c,
            "true_mean": true_mean,
            "false_mean": false_mean,
            "true_std": true_std,
            "false_std": false_std,
            "diff_true_minus_false": true_mean - false_mean,
            "abs_cohen_d": abs(d),
            "cohen_d": d,
        })

    compare_df = pd.DataFrame(compare_rows).sort_values("abs_cohen_d", ascending=False)
    compare_csv = os.path.join(out_dir, "HM20260626_144226_heatmap_shape_true_false_compare.csv")
    compare_df.to_csv(compare_csv, index=False, encoding="utf-8-sig")

    # 画几个核心特征分布
    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(fig_dir)

    plot_features = [
        "hm_shape_leak_like_score",
        "hm_directed_core_score",
        "hm_diffuse_score",
        "hm_entropy_2d",
        "hm_weighted_elongation",
        "hm_hot_area_p95_ratio",
        "hm_largest_component_ratio_p95",
        "hm_energy_concentration_top10",
    ]

    for c in plot_features:
        if c not in sub_hm.columns:
            continue

        plt.figure(figsize=(8, 5))

        true_vals = safe_float_array(sub_hm.loc[sub_hm["label"] == "TRUE_LEAK", c]).dropna()
        false_vals = safe_float_array(sub_hm.loc[sub_hm["label"] == "FALSE_LEAK", c]).dropna()

        plt.hist(true_vals, bins=12, alpha=0.6, label="TRUE_LEAK")
        plt.hist(false_vals, bins=12, alpha=0.6, label="FALSE_LEAK")
        plt.title(f"144226 heatmap feature: {c}")
        plt.xlabel(c)
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(os.path.join(fig_dir, f"144226_{c}.png"), dpi=150)
        plt.close()

    # 收集每个实验中 144226 的预测
    for res in all_results:
        pred_df = res["pred_df"]
        if len(pred_df) == 0:
            continue

        sub_pred = pred_df[pred_df["test_group"] == target].copy()
        path = os.path.join(out_dir, f"{res['experiment']}_144226_predictions.csv")
        sub_pred.to_csv(path, index=False, encoding="utf-8-sig")

    return {
        "compare_csv": compare_csv,
        "fig_dir": fig_dir,
    }


# ============================================================
# 7. 报告与画图
# ============================================================

def make_summary_report(df, hdf_path, feature_meta, all_results, diagnosis_info, output_dir):
    lines = []

    lines.append("v8.1 热力图核心形态特征消融实验报告")
    lines.append("=" * 100)
    lines.append(f"生成时间: {datetime.now()}")
    lines.append(f"输入特征表: {MERGED_FEATURE_CSV}")
    lines.append(f"heatmap核心形态特征表: {hdf_path}")
    lines.append("")
    lines.append("样本情况:")
    lines.append(f"  总样本数: {len(df)}")
    for label, count in df[LABEL_COL].value_counts().items():
        lines.append(f"  {label}: {int(count)}")
    lines.append("")

    lines.append("特征构造:")
    lines.append(f"  初始 v7 数值特征数: {feature_meta['initial_v7_numeric_feature_count']}")
    lines.append(f"  v7 稳健基础特征数: {feature_meta['kept_v7_base_feature_count']}")
    lines.append(f"  heatmap 核心形态特征数: {feature_meta['heatmap_core_shape_feature_count']}")
    lines.append(f"  删除的 v7 不稳定特征: {feature_meta['removed_v7_path']}")
    lines.append("")

    lines.append("消融实验结果:")
    lines.append("-" * 100)

    summary_rows = []

    for res in all_results:
        gdf = res["group_df"]
        exp = res["experiment"]

        lines.append("")
        lines.append(f"实验 {exp}:")
        lines.append(f"  特征数量: {res['feature_count']}")

        if len(gdf):
            lines.append(f"  平均 AUC: {gdf['auc'].mean():.4f}")
            lines.append(f"  平均 default_acc: {gdf['default_accuracy_0p5'].mean():.4f}")
            lines.append(f"  平均 model_acc: {gdf['model_accuracy'].mean():.4f}")
            lines.append(f"  平均 rank_acc: {gdf['rank_accuracy'].mean():.4f}")
            lines.append(f"  平均 final_decisive_acc: {gdf['final_decisive_accuracy'].mean():.4f}")

            # 144226
            target_row = gdf[gdf["test_group"] == "HM20260626_144226.ld"]
            if len(target_row):
                r = target_row.iloc[0]
                lines.append(
                    f"  144226: "
                    f"AUC={r['auc']:.4f}, "
                    f"default_acc={r['default_accuracy_0p5']:.4f}, "
                    f"model_acc={r['model_accuracy']:.4f}, "
                    f"rank_acc={r['rank_accuracy']:.4f}, "
                    f"final_decisive_acc={r['final_decisive_accuracy']:.4f}, "
                    f"suspect={int(r['final_n_suspect'])}"
                )

            lines.append("  各时间点:")
            for _, r in gdf.iterrows():
                lines.append(
                    f"    {r['test_group']}: "
                    f"AUC={r['auc']:.3f}, "
                    f"default={r['default_accuracy_0p5']:.3f}, "
                    f"model={r['model_accuracy']:.3f}, "
                    f"rank={r['rank_accuracy']:.3f}, "
                    f"final_decisive={r['final_decisive_accuracy']:.3f}, "
                    f"suspect={int(r['final_n_suspect'])}"
                )

            summary_rows.append({
                "experiment": exp,
                "feature_count": res["feature_count"],
                "mean_auc": gdf["auc"].mean(),
                "mean_default_acc": gdf["default_accuracy_0p5"].mean(),
                "mean_model_acc": gdf["model_accuracy"].mean(),
                "mean_rank_acc": gdf["rank_accuracy"].mean(),
                "mean_final_decisive_acc": gdf["final_decisive_accuracy"].mean(),
            })

            target_row = gdf[gdf["test_group"] == "HM20260626_144226.ld"]
            if len(target_row):
                r = target_row.iloc[0]
                summary_rows[-1].update({
                    "auc_144226": r["auc"],
                    "default_acc_144226": r["default_accuracy_0p5"],
                    "model_acc_144226": r["model_accuracy"],
                    "rank_acc_144226": r["rank_accuracy"],
                    "final_decisive_acc_144226": r["final_decisive_accuracy"],
                    "suspect_144226": r["final_n_suspect"],
                })

        lines.append("  Top 15 feature importances:")
        for _, row in res["importance_df"].head(15).iterrows():
            lines.append(f"    {row['feature']}: {row['importance']:.6f}")

    lines.append("")
    lines.append("144226 专项诊断:")
    lines.append(f"  heatmap形态真假对比: {diagnosis_info['compare_csv']}")
    lines.append(f"  分布图文件夹: {diagnosis_info['fig_dir']}")
    lines.append("")
    lines.append("解释建议:")
    lines.append("  如果 B_heatmap_shape_only 在 144226 上也很差，说明 heatmap 形态本身难以区分该时间点真/假。")
    lines.append("  如果 C_v7_plus_heatmap_shape 比 A_v7_only 没有提升，说明当前热图核心形态特征没有提供有效增益。")
    lines.append("  如果 heatmap 重要特征不明显，建议后续从原始二维能量矩阵而非 PNG 图像提取形态特征。")

    report_path = os.path.join(output_dir, "v8_1_report.txt")
    save_text(report_path, "\n".join(lines))

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, "v8_1_ablation_overall_summary.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    return report_path, summary_csv


def plot_ablation_summary(all_results, output_dir):
    fig_dir = os.path.join(output_dir, "figures")
    ensure_dir(fig_dir)

    rows = []
    for res in all_results:
        gdf = res["group_df"]
        if len(gdf) == 0:
            continue

        for _, r in gdf.iterrows():
            rows.append({
                "experiment": res["experiment"],
                "test_group": r["test_group"],
                "auc": r["auc"],
                "model_accuracy": r["model_accuracy"],
                "rank_accuracy": r["rank_accuracy"],
                "default_accuracy_0p5": r["default_accuracy_0p5"],
            })

    dfp = pd.DataFrame(rows)

    if len(dfp) == 0:
        return []

    paths = []

    metrics = ["auc", "model_accuracy", "rank_accuracy"]

    for metric in metrics:
        plt.figure(figsize=(12, 5))

        experiments = dfp["experiment"].unique().tolist()
        groups = sorted(dfp["test_group"].unique().tolist())
        x = np.arange(len(groups))
        width = 0.25

        for i, exp in enumerate(experiments):
            vals = []
            for g in groups:
                sub = dfp[(dfp["experiment"] == exp) & (dfp["test_group"] == g)]
                vals.append(float(sub[metric].iloc[0]) if len(sub) else np.nan)

            plt.bar(x + (i - 1) * width, vals, width, label=exp)

        plt.ylim(0, 1.05)
        plt.xticks(x, groups, rotation=45, ha="right")
        plt.ylabel(metric)
        plt.title(f"v8.1 ablation comparison: {metric}")
        plt.legend(fontsize=8)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()

        path = os.path.join(fig_dir, f"v8_1_ablation_{metric}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)

    return paths


# ============================================================
# 8. 主程序
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)

    print("=" * 100)
    print("v8.1 热力图核心形态特征消融版")
    print("=" * 100)

    if not os.path.exists(MERGED_FEATURE_CSV):
        print("找不到输入文件:", MERGED_FEATURE_CSV)
        print("请先运行 v4，生成 merged_feature_dataset.csv")
        return

    df = pd.read_csv(MERGED_FEATURE_CSV)

    if LABEL_COL not in df.columns:
        print("CSV中没有 label 列。")
        return

    if GROUP_COL not in df.columns:
        print(f"CSV中没有 {GROUP_COL} 列。")
        return

    if "center" not in df.columns:
        print("CSV中没有 center 列。")
        return

    df[LABEL_COL] = df[LABEL_COL].astype(str)
    df = df[df[LABEL_COL].isin(["TRUE_LEAK", "FALSE_LEAK"])].copy()
    df = df.reset_index(drop=True)

    print("样本数量:", len(df))
    print(df[LABEL_COL].value_counts())

    # 1. 提取 heatmap 核心形态特征
    hdf, hdf_path, missing = build_heatmap_feature_table(df, OUTPUT_DIR)

    # 2. 构造三组消融特征
    print("\n开始构造 v8.1 消融特征集...")
    feature_sets, feature_meta = build_feature_sets(df, hdf, OUTPUT_DIR)

    print("初始 v7 数值特征数:", feature_meta["initial_v7_numeric_feature_count"])
    print("v7 稳健基础特征数:", feature_meta["kept_v7_base_feature_count"])
    print("heatmap 核心形态基础特征数:", feature_meta["heatmap_core_shape_feature_count"])

    for name, info in feature_sets.items():
        print(f"  {name}: 模型特征数={info['X'].shape[1]}, 文件={info['feature_csv']}")

    # 3. 三组实验验证
    all_results = []

    for name, info in feature_sets.items():
        res = validate_feature_set(df, info["X"], name, OUTPUT_DIR)
        all_results.append(res)

    # 4. 144226 专项诊断
    diagnosis_info = analyze_144226(feature_sets, hdf, df, all_results, OUTPUT_DIR)

    # 5. 报告/图
    report_path, summary_csv = make_summary_report(
        df,
        hdf_path,
        feature_meta,
        all_results,
        diagnosis_info,
        OUTPUT_DIR
    )

    plot_paths = plot_ablation_summary(all_results, OUTPUT_DIR)

    print("\n" + "=" * 100)
    print("v8.1 消融实验完成")
    print("=" * 100)

    print("总报告:", report_path)
    print("总汇总表:", summary_csv)
    print("heatmap核心特征表:", hdf_path)
    print("144226 heatmap形态真假对比:", diagnosis_info["compare_csv"])

    print("\n消融实验核心结果:")
    for res in all_results:
        gdf = res["group_df"]
        if len(gdf) == 0:
            continue

        print(f"\n实验 {res['experiment']}: 特征数={res['feature_count']}")
        print(
            f"  平均AUC={gdf['auc'].mean():.3f}, "
            f"平均model_acc={gdf['model_accuracy'].mean():.3f}, "
            f"平均rank_acc={gdf['rank_accuracy'].mean():.3f}"
        )

        target_row = gdf[gdf["test_group"] == "HM20260626_144226.ld"]
        if len(target_row):
            r = target_row.iloc[0]
            print(
                f"  144226: "
                f"AUC={r['auc']:.3f}, "
                f"default_acc={r['default_accuracy_0p5']:.3f}, "
                f"model_acc={r['model_accuracy']:.3f}, "
                f"rank_acc={r['rank_accuracy']:.3f}, "
                f"final_decisive_acc={r['final_decisive_accuracy']:.3f}, "
                f"suspect={int(r['final_n_suspect'])}"
            )

        print("  重要特征前10:")
        for _, row in res["importance_df"].head(10).iterrows():
            print(f"    {row['feature']}: {row['importance']:.6f}")

    if plot_paths:
        print("\n图片输出:")
        for p in plot_paths:
            print(" ", p)

    print("\n输出文件夹:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
