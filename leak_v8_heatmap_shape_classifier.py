# -*- coding: utf-8 -*-
r"""
leak_v8_heatmap_shape_classifier.py

v8：在 v7 稳健特征的基础上，加入 v1 热力图形态特征。

核心目的：
    v7 已经说明：去掉绝对能量特征 + time 内部归一化/排序后，
    多数时间点效果很好，但 HM20260626_144226.ld 仍然难分。

    v8 增加热力图形态特征，尝试刻画：
        1. 红色热点是否小而集中；
        2. 热点是否沿某个方向拉长；
        3. 热点是否弥散；
        4. 是否存在多个散乱热点。

输入：
    1. v4 合并后的特征表：
       C:\Users\jiangxinru6\Desktop\wurenji\leak_v4_compare_results\merged_feature_dataset.csv

    2. v1 生成的热力图图片：
       真泄漏：results_spectrum_*.ld\heatmap_*.png
       假泄漏：results_cs_spectrum_*.ld\heatmap_*.png

输出：
    C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_heatmap_shape_results\
        v8_heatmap_features.csv
        v8_robust_feature_dataset.csv
        v8_group_validation_summary.csv
        v8_predictions.csv
        v8_report.txt
        v8_final_heatmap_robust_classifier.pkl
        v8_final_model_config.json
        figures\

运行：
    python leak_v8_heatmap_shape_classifier.py

依赖：
    pip install numpy pandas matplotlib scikit-learn pillow scipy joblib

说明：
    这版是从“已渲染的热力图 PNG”里提形态特征。
    如果后续能直接拿到热力图的原始二维矩阵，建议再做 v8b，效果会更干净。
"""

import os
import re
import glob
import json
import math
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 路径配置
# ============================================================

MERGED_FEATURE_CSV = r"C:\Users\jiangxinru6\Desktop\wurenji\leak_v4_compare_results\merged_feature_dataset.csv"
OUTPUT_DIR = r"C:\Users\jiangxinru6\Desktop\wurenji\leak_v8_heatmap_shape_results"

GROUP_COL = "time"
LABEL_COL = "label"

TRUE_HEATMAP_DIRS = {
    "HM20260626_142938.ld": r"C:\Users\jiangxinru6\Desktop\wurenji\results_spectrum_HM20260626_142938.ld",
    "HM20260626_143034.ld": r"C:\Users\jiangxinru6\Desktop\wurenji\results_spectrum_HM20260626_143034.ld",
    "HM20260626_144226.ld": r"C:\Users\jiangxinru6\Desktop\wurenji\results_spectrum_HM20260626_144226.ld",
    "HM20260626_144325.ld": r"C:\Users\jiangxinru6\Desktop\wurenji\results_spectrum_HM20260626_144325.ld",
}

FALSE_HEATMAP_DIRS = {
    "HM20260626_142938.ld": r"C:\Users\jiangxinru6\Desktop\wurenji\results_cs_spectrum_HM20260626_142938.ld",
    "HM20260626_143034.ld": r"C:\Users\jiangxinru6\Desktop\wurenji\results_cs_spectrum_HM20260626_143034.ld",
    "HM20260626_144226.ld": r"C:\Users\jiangxinru6\Desktop\wurenji\results_cs_spectrum_HM20260626_144226.ld",
    "HM20260626_144325.ld": r"C:\Users\jiangxinru6\Desktop\wurenji\results_cs_spectrum_HM20260626_144325.ld",
}

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

# 删除不稳定绝对能量特征，沿用 v7 思路
DROP_ABSOLUTE_ENERGY_FEATURES = True
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

DROP_BAD_FREQUENCY_FEATURES = True
FREQ_LOW_HZ = 20000
BAD_FREQ_RATIO_THRESHOLD = 0.30

# time 内部无标签归一化/排名特征
ADD_TIME_ROBUST_Z = True
ADD_TIME_RANK_PCT = True

# 阈值搜索
THRESHOLD_GRID = np.linspace(0.01, 0.99, 99)
THRESHOLD_METRIC = "balanced_accuracy"

# 同一时间点内部概率排名二分类：默认 top 50% 判 TRUE_LEAK
# 你当前验证数据真假基本各一半，所以 0.50 合理。
RANK_TRUE_FRACTION_FOR_BINARY = 0.50
RANK_TRUE_LIKE_PCT = 0.70
RANK_FALSE_LIKE_PCT = 0.30

RANDOM_STATE = 42

# 随机森林参数：比 v7 稍微更保守，减少小样本过拟合
RF_N_ESTIMATORS = 900
RF_MIN_SAMPLES_LEAF = 2
RF_MAX_FEATURES = "sqrt"


# ============================================================
# 2. 基础工具函数
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def label_to_binary(labels):
    return np.array([1 if str(x) == "TRUE_LEAK" else 0 for x in labels], dtype=int)


def binary_to_label(v):
    return "TRUE_LEAK" if int(v) == 1 else "FALSE_LEAK"


def safe_float_array(s):
    arr = pd.to_numeric(s, errors="coerce")
    arr = arr.replace([np.inf, -np.inf], np.nan)
    return arr


def normalize_center_id(x):
    """把 center 列统一成 00/01/02 这种格式。"""
    if pd.isna(x):
        return "00"
    s = str(x).strip()
    m = re.search(r"(\d+)", s)
    if not m:
        return s
    return f"{int(m.group(1)):02d}"


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
        rows.append({"threshold": float(t), "score": float(score), **m})
        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t, float(best_score), pd.DataFrame(rows)


# ============================================================
# 3. 图片读取与热力图区域自动裁剪
# ============================================================

def read_rgb_image(path):
    """读取图片为 RGB float 数组，范围 0~1。"""
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0
        return arr
    except Exception:
        arr = plt.imread(path)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        arr = arr.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        return arr


def rgb_to_hsv_np(rgb):
    """RGB[0,1] -> HSV[0,1]，避免依赖额外库。"""
    rgb = np.clip(rgb, 0, 1)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    diff = mx - mn

    h = np.zeros_like(mx)
    mask = diff > 1e-12

    idx = mask & (mx == r)
    h[idx] = ((g[idx] - b[idx]) / diff[idx]) % 6
    idx = mask & (mx == g)
    h[idx] = ((b[idx] - r[idx]) / diff[idx]) + 2
    idx = mask & (mx == b)
    h[idx] = ((r[idx] - g[idx]) / diff[idx]) + 4
    h = h / 6.0

    s = np.zeros_like(mx)
    s[mx > 1e-12] = diff[mx > 1e-12] / mx[mx > 1e-12]
    v = mx
    return np.stack([h, s, v], axis=-1)


def get_ndimage():
    try:
        from scipy import ndimage
        return ndimage
    except Exception:
        return None


def auto_crop_heatmap_area(rgb):
    """
    从完整 PNG 中自动裁剪出热力图主体区域。

    v1 保存的 PNG 一般包含：标题、坐标轴、热力图、右侧 colorbar。
    这里用高饱和度彩色区域做连通域，选择最大且近似方形的区域作为热力图主体。
    """
    h, w = rgb.shape[:2]
    hsv = rgb_to_hsv_np(rgb)
    sat = hsv[..., 1]
    val = hsv[..., 2]

    # 选择高饱和度且非白的彩色区域；蓝色背景、红色热点都会被选中
    mask = (sat > 0.18) & (val > 0.08) & (val < 0.98)

    # 排除最外边缘，减少窗口/背景干扰
    border_y = max(2, int(0.01 * h))
    border_x = max(2, int(0.01 * w))
    mask[:border_y, :] = False
    mask[-border_y:, :] = False
    mask[:, :border_x] = False
    mask[:, -border_x:] = False

    ndimage = get_ndimage()

    if ndimage is not None:
        # 轻微开运算，去掉文字、小碎片
        try:
            mask_clean = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
            mask_clean = ndimage.binary_closing(mask_clean, structure=np.ones((3, 3)))
        except Exception:
            mask_clean = mask

        lab, n = ndimage.label(mask_clean)
        candidates = []
        for i in range(1, n + 1):
            ys, xs = np.where(lab == i)
            if len(xs) < 50:
                continue
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            bw = x1 - x0
            bh = y1 - y0
            area = len(xs)
            bbox_area = bw * bh
            if bw < 0.15 * w or bh < 0.15 * h:
                continue
            aspect = bw / (bh + 1e-12)
            # 热力图主体通常接近方形；colorbar 是细长条，会被惩罚
            square_score = math.exp(-abs(math.log(max(aspect, 1e-6))))
            fill_ratio = area / (bbox_area + 1e-12)
            size_score = bbox_area / (w * h + 1e-12)
            score = bbox_area * square_score * (0.5 + fill_ratio) * (0.5 + size_score)
            candidates.append((score, x0, x1, y0, y1, area, aspect))

        if candidates:
            candidates.sort(reverse=True, key=lambda x: x[0])
            _, x0, x1, y0, y1, area, aspect = candidates[0]
            # 稍微扩展一点，但不要把 colorbar 包进来太多
            pad = max(2, int(0.01 * max(x1 - x0, y1 - y0)))
            x0 = max(0, x0 - pad)
            y0 = max(0, y0 - pad)
            x1 = min(w, x1 + pad)
            y1 = min(h, y1 + pad)
            crop = rgb[y0:y1, x0:x1]
            return crop, {
                "hm_crop_x0": x0,
                "hm_crop_y0": y0,
                "hm_crop_w": x1 - x0,
                "hm_crop_h": y1 - y0,
                "hm_crop_area_ratio": ((x1 - x0) * (y1 - y0)) / (w * h + 1e-12),
                "hm_crop_aspect": (x1 - x0) / (y1 - y0 + 1e-12),
                "hm_crop_method_ok": 1,
            }

    # 兜底：中心偏左区域裁剪，避开右侧 colorbar 和上下标题
    y0 = int(0.15 * h)
    y1 = int(0.88 * h)
    x0 = int(0.08 * w)
    x1 = int(0.82 * w)
    crop = rgb[y0:y1, x0:x1]
    return crop, {
        "hm_crop_x0": x0,
        "hm_crop_y0": y0,
        "hm_crop_w": x1 - x0,
        "hm_crop_h": y1 - y0,
        "hm_crop_area_ratio": ((x1 - x0) * (y1 - y0)) / (w * h + 1e-12),
        "hm_crop_aspect": (x1 - x0) / (y1 - y0 + 1e-12),
        "hm_crop_method_ok": 0,
    }


def heatmap_hotness_from_rgb(crop_rgb):
    """
    将 jet/类似伪彩色热力图转换成相对热度矩阵 0~1。

    近似规则：
        红/黄 -> 高
        绿 -> 中
        蓝 -> 低
    """
    hsv = rgb_to_hsv_np(crop_rgb)
    hue = hsv[..., 0]
    sat = hsv[..., 1]
    val = hsv[..., 2]

    # hue: red≈0, yellow≈0.16, green≈0.33, cyan≈0.5, blue≈0.66
    hot = 1.0 - np.clip(hue / 0.66, 0, 1)

    # 对极低饱和文字/背景降权
    hot = hot * (0.25 + 0.75 * sat)

    # 太白的背景/坐标文字不要作为热点
    hot = np.where((sat < 0.08) & (val > 0.80), 0.0, hot)

    # robust normalize
    lo = float(np.percentile(hot, 1))
    hi = float(np.percentile(hot, 99))
    if hi - lo < 1e-12:
        return np.zeros_like(hot, dtype=float)
    hot = (hot - lo) / (hi - lo)
    hot = np.clip(hot, 0, 1)
    return hot.astype(float)


# ============================================================
# 4. 热力图形态特征
# ============================================================

def gini_coefficient(x):
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    x = np.maximum(x, 0)
    if np.sum(x) <= 1e-12:
        return 0.0
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * x)) / (n * np.sum(x) + 1e-12))


def binary_component_features(binary):
    ndimage = get_ndimage()
    binary = np.asarray(binary, dtype=bool)
    total_area = int(binary.sum())
    h, w = binary.shape[:2]

    if total_area == 0:
        return {
            "num_components": 0,
            "largest_area": 0,
            "largest_area_ratio_to_hot": 0.0,
            "largest_area_ratio_to_image": 0.0,
            "compactness": 0.0,
        }

    if ndimage is None:
        # 简易兜底：不做连通域，只给总面积
        return {
            "num_components": 1,
            "largest_area": total_area,
            "largest_area_ratio_to_hot": 1.0,
            "largest_area_ratio_to_image": total_area / (h * w + 1e-12),
            "compactness": 0.0,
        }

    lab, n = ndimage.label(binary)
    areas = []
    for i in range(1, n + 1):
        a = int((lab == i).sum())
        if a >= max(5, int(0.0005 * h * w)):
            areas.append((a, i))

    if not areas:
        return {
            "num_components": 0,
            "largest_area": 0,
            "largest_area_ratio_to_hot": 0.0,
            "largest_area_ratio_to_image": 0.0,
            "compactness": 0.0,
        }

    areas.sort(reverse=True)
    largest_area, largest_label = areas[0]
    largest = lab == largest_label

    try:
        eroded = ndimage.binary_erosion(largest, structure=np.ones((3, 3)))
        perimeter = int(np.logical_xor(largest, eroded).sum())
    except Exception:
        perimeter = 0

    compactness = 0.0
    if perimeter > 0:
        compactness = float(4 * math.pi * largest_area / (perimeter * perimeter + 1e-12))

    return {
        "num_components": len(areas),
        "largest_area": largest_area,
        "largest_area_ratio_to_hot": largest_area / (total_area + 1e-12),
        "largest_area_ratio_to_image": largest_area / (h * w + 1e-12),
        "compactness": compactness,
    }


def weighted_shape_features(hot):
    hot = np.asarray(hot, dtype=float)
    h, w = hot.shape
    yy, xx = np.mgrid[0:h, 0:w]
    weights = np.maximum(hot, 0.0)
    total = float(weights.sum())

    if total <= 1e-12:
        return {
            "hm_weighted_cx": 0.5,
            "hm_weighted_cy": 0.5,
            "hm_weighted_spread": 0.0,
            "hm_weighted_elongation": 1.0,
            "hm_weighted_eccentricity": 0.0,
            "hm_weighted_orientation_rad": 0.0,
            "hm_directional_contrast_8bin": 0.0,
            "hm_asymmetry_lr": 0.0,
            "hm_asymmetry_ud": 0.0,
        }

    cx = float((xx * weights).sum() / total)
    cy = float((yy * weights).sum() / total)

    x = xx - cx
    y = yy - cy
    cov_xx = float((weights * x * x).sum() / total)
    cov_yy = float((weights * y * y).sum() / total)
    cov_xy = float((weights * x * y).sum() / total)

    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=float)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.sort(np.maximum(eigvals, 0.0))[::-1]
    l1, l2 = float(eigvals[0]), float(eigvals[1])

    elongation = math.sqrt((l1 + 1e-12) / (l2 + 1e-12))
    eccentricity = math.sqrt(max(0.0, 1.0 - (l2 + 1e-12) / (l1 + 1e-12))) if l1 > 0 else 0.0
    spread = math.sqrt((l1 + l2) / (h * h + w * w + 1e-12))

    # 主方向角
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, np.argmax(vals)]
    orientation = float(math.atan2(v[1], v[0]))

    # 方向能量对比：以重心为原点，分 8 个角度扇区
    angle = np.arctan2(y, x)
    bins = np.linspace(-math.pi, math.pi, 9)
    sector_energy = []
    for i in range(8):
        mask = (angle >= bins[i]) & (angle < bins[i + 1])
        sector_energy.append(float(weights[mask].sum()))
    sector_energy = np.array(sector_energy, dtype=float)
    directional_contrast = float(sector_energy.max() / (sector_energy.mean() + 1e-12))

    left = float(weights[:, :w // 2].sum())
    right = float(weights[:, w // 2:].sum())
    up = float(weights[:h // 2, :].sum())
    down = float(weights[h // 2:, :].sum())
    asym_lr = abs(left - right) / (left + right + 1e-12)
    asym_ud = abs(up - down) / (up + down + 1e-12)

    return {
        "hm_weighted_cx": cx / (w + 1e-12),
        "hm_weighted_cy": cy / (h + 1e-12),
        "hm_weighted_spread": spread,
        "hm_weighted_elongation": float(elongation),
        "hm_weighted_eccentricity": float(eccentricity),
        "hm_weighted_orientation_rad": orientation,
        "hm_directional_contrast_8bin": directional_contrast,
        "hm_asymmetry_lr": float(asym_lr),
        "hm_asymmetry_ud": float(asym_ud),
    }


def extract_heatmap_shape_features(image_path):
    """从单张 heatmap PNG 提取形态特征。"""
    if not image_path or not os.path.exists(image_path):
        return make_missing_heatmap_features()

    try:
        rgb = read_rgb_image(image_path)
        crop, crop_info = auto_crop_heatmap_area(rgb)
        hot = heatmap_hotness_from_rgb(crop)

        h, w = hot.shape
        n_pix = h * w
        vals = hot.ravel()
        total_energy = float(vals.sum()) + 1e-12

        # 相对阈值热点区域
        p70 = float(np.percentile(vals, 70))
        p80 = float(np.percentile(vals, 80))
        p90 = float(np.percentile(vals, 90))
        p95 = float(np.percentile(vals, 95))
        p98 = float(np.percentile(vals, 98))

        bin90 = hot >= p90
        bin95 = hot >= p95
        bin98 = hot >= p98

        comp90 = binary_component_features(bin90)
        comp95 = binary_component_features(bin95)
        comp98 = binary_component_features(bin98)
        shape = weighted_shape_features(hot)

        # 熵：越大越弥散
        prob = vals / total_energy
        entropy = float(-(prob * np.log(prob + 1e-12)).sum() / (math.log(len(prob) + 1e-12)))

        sorted_vals = np.sort(vals)[::-1]
        top1_n = max(1, int(0.01 * len(sorted_vals)))
        top5_n = max(1, int(0.05 * len(sorted_vals)))
        top10_n = max(1, int(0.10 * len(sorted_vals)))

        top1_ratio = float(sorted_vals[:top1_n].sum() / total_energy)
        top5_ratio = float(sorted_vals[:top5_n].sum() / total_energy)
        top10_ratio = float(sorted_vals[:top10_n].sum() / total_energy)

        area90 = float(bin90.mean())
        area95 = float(bin95.mean())
        area98 = float(bin98.mean())

        # 形态综合指标：
        # 小而集中、单连通、拉长、有方向 -> 更像喷流型热点
        small_core_score = float(comp95["largest_area_ratio_to_hot"] / (area95 + 1e-6))
        directed_core_score = float(shape["hm_weighted_elongation"] * comp95["largest_area_ratio_to_hot"])
        diffuse_score = float(entropy * area90 / (comp90["largest_area_ratio_to_hot"] + 1e-6))

        feats = {
            "hm_missing": 0,
            "hm_image_h": h,
            "hm_image_w": w,
            **crop_info,
            "hm_mean": float(vals.mean()),
            "hm_std": float(vals.std()),
            "hm_max": float(vals.max()),
            "hm_peak_mean_ratio": float(vals.max() / (vals.mean() + 1e-12)),
            "hm_gini": gini_coefficient(vals),
            "hm_entropy_2d": entropy,
            "hm_top1_energy_ratio": top1_ratio,
            "hm_top5_energy_ratio": top5_ratio,
            "hm_top10_energy_ratio": top10_ratio,
            "hm_hot_area_p70_ratio": float((hot >= p70).mean()),
            "hm_hot_area_p80_ratio": float((hot >= p80).mean()),
            "hm_hot_area_p90_ratio": area90,
            "hm_hot_area_p95_ratio": area95,
            "hm_hot_area_p98_ratio": area98,

            "hm_num_components_p90": comp90["num_components"],
            "hm_largest_component_ratio_to_hot_p90": comp90["largest_area_ratio_to_hot"],
            "hm_largest_component_ratio_to_image_p90": comp90["largest_area_ratio_to_image"],
            "hm_compactness_p90": comp90["compactness"],

            "hm_num_components_p95": comp95["num_components"],
            "hm_largest_component_ratio_to_hot_p95": comp95["largest_area_ratio_to_hot"],
            "hm_largest_component_ratio_to_image_p95": comp95["largest_area_ratio_to_image"],
            "hm_compactness_p95": comp95["compactness"],

            "hm_num_components_p98": comp98["num_components"],
            "hm_largest_component_ratio_to_hot_p98": comp98["largest_area_ratio_to_hot"],
            "hm_largest_component_ratio_to_image_p98": comp98["largest_area_ratio_to_image"],
            "hm_compactness_p98": comp98["compactness"],

            **shape,
            "hm_small_core_score": small_core_score,
            "hm_directed_core_score": directed_core_score,
            "hm_diffuse_score": diffuse_score,
        }
        return feats
    except Exception as e:
        feats = make_missing_heatmap_features()
        feats["hm_extract_error"] = str(e)[:120]
        return feats


def make_missing_heatmap_features():
    """缺失图片时返回全 0 特征，保留 hm_missing=1。"""
    keys = [
        "hm_image_h", "hm_image_w", "hm_crop_x0", "hm_crop_y0", "hm_crop_w", "hm_crop_h",
        "hm_crop_area_ratio", "hm_crop_aspect", "hm_crop_method_ok",
        "hm_mean", "hm_std", "hm_max", "hm_peak_mean_ratio", "hm_gini", "hm_entropy_2d",
        "hm_top1_energy_ratio", "hm_top5_energy_ratio", "hm_top10_energy_ratio",
        "hm_hot_area_p70_ratio", "hm_hot_area_p80_ratio", "hm_hot_area_p90_ratio",
        "hm_hot_area_p95_ratio", "hm_hot_area_p98_ratio",
        "hm_num_components_p90", "hm_largest_component_ratio_to_hot_p90",
        "hm_largest_component_ratio_to_image_p90", "hm_compactness_p90",
        "hm_num_components_p95", "hm_largest_component_ratio_to_hot_p95",
        "hm_largest_component_ratio_to_image_p95", "hm_compactness_p95",
        "hm_num_components_p98", "hm_largest_component_ratio_to_hot_p98",
        "hm_largest_component_ratio_to_image_p98", "hm_compactness_p98",
        "hm_weighted_cx", "hm_weighted_cy", "hm_weighted_spread", "hm_weighted_elongation",
        "hm_weighted_eccentricity", "hm_weighted_orientation_rad", "hm_directional_contrast_8bin",
        "hm_asymmetry_lr", "hm_asymmetry_ud", "hm_small_core_score", "hm_directed_core_score",
        "hm_diffuse_score",
    ]
    d = {k: 0.0 for k in keys}
    d["hm_missing"] = 1
    d["hm_extract_error"] = "missing_or_failed"
    return d


def find_heatmap_path(label, time_value, center_value):
    time_value = str(time_value)
    center_id = normalize_center_id(center_value)

    if str(label) == "TRUE_LEAK":
        root = TRUE_HEATMAP_DIRS.get(time_value, "")
    else:
        root = FALSE_HEATMAP_DIRS.get(time_value, "")

    if not root:
        return ""

    filename = f"heatmap_{time_value}_center_{center_id}.png"
    candidate = os.path.join(root, filename)
    if os.path.exists(candidate):
        return candidate

    # 兜底：模糊匹配
    patterns = [
        os.path.join(root, f"*heatmap*{time_value}*center_{center_id}*.png"),
        os.path.join(root, f"*center_{center_id}*.png"),
        os.path.join(root, f"*_{center_id}.png"),
    ]
    for p in patterns:
        files = glob.glob(p)
        if files:
            return files[0]

    return candidate  # 返回预期路径，方便排查


def build_heatmap_feature_table(df, output_dir):
    rows = []
    print("\n开始提取热力图形态特征...")

    for i, r in df.iterrows():
        label = str(r[LABEL_COL])
        time_value = str(r[GROUP_COL])
        center_id = normalize_center_id(r["center"] if "center" in df.columns else "00")
        path = find_heatmap_path(label, time_value, center_id)

        feats = extract_heatmap_shape_features(path)
        row = {
            "row_index": i,
            "label": label,
            "time": time_value,
            "center": center_id,
            "heatmap_path": path,
            **feats,
        }
        rows.append(row)

        if (i + 1) % 20 == 0 or i == len(df) - 1:
            print(f"  已处理 {i + 1}/{len(df)} 张/行")

    hm_df = pd.DataFrame(rows)
    out_csv = os.path.join(output_dir, "v8_heatmap_features.csv")
    hm_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    missing_count = int(hm_df["hm_missing"].sum()) if "hm_missing" in hm_df.columns else 0
    print("热力图特征表:", out_csv)
    print("缺失/失败热力图数量:", missing_count)

    return hm_df, out_csv


# ============================================================
# 5. v7 稳健数值特征 + v8 热图特征合并
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
        if temp.notna().mean() > 0.8:
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
            removed_rows.append({"feature": c, "reason": reason})

    removed_csv = os.path.join(output_dir, "v8_removed_v7_unstable_features.csv")
    pd.DataFrame(removed_rows).to_csv(removed_csv, index=False, encoding="utf-8-sig")
    return kept, removed_csv


def make_numeric_df(df, feature_cols):
    x = pd.DataFrame(index=df.index)
    for c in feature_cols:
        vals = safe_float_array(df[c])
        med = vals.median()
        if not np.isfinite(med):
            med = 0.0
        x[c] = vals.fillna(med).astype(float)
    return x


def add_time_internal_features(df, base_x, group_col):
    out = base_x.copy()
    groups = df[group_col].astype(str)
    eps = 1e-12

    for c in base_x.columns:
        values = base_x[c].astype(float)

        if ADD_TIME_ROBUST_Z:
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

        if ADD_TIME_RANK_PCT:
            r_col = f"{c}__time_rank_pct"
            rank_values = values.groupby(groups).rank(method="average", pct=True)
            out[r_col] = rank_values.fillna(0.5).astype(float)

    return out


def prepare_v8_feature_matrix(df, hm_df, output_dir):
    # v7 稳健基础特征
    initial_v7 = get_initial_numeric_features(df)
    kept_v7, removed_v7_csv = remove_unstable_v7_features(df, initial_v7, output_dir)
    x_v7_base = make_numeric_df(df, kept_v7)

    # heatmap 特征：去掉非数值/路径/标签列
    hm_feature_cols = []
    for c in hm_df.columns:
        if c in ["row_index", "label", "time", "center", "heatmap_path", "hm_extract_error"]:
            continue
        temp = safe_float_array(hm_df[c])
        if temp.notna().mean() > 0.8:
            hm_feature_cols.append(c)

    x_hm_base = make_numeric_df(hm_df, hm_feature_cols)

    # 合并基础特征
    x_base = pd.concat([x_v7_base.reset_index(drop=True), x_hm_base.reset_index(drop=True)], axis=1)

    # 对合并后的基础特征统一添加 time 内部 z/rank
    x_model = add_time_internal_features(df, x_base, GROUP_COL)

    # 保存数据集
    id_cols = [c for c in ["dataset", "time", "center", "label"] if c in df.columns]
    v8_dataset = pd.concat([df[id_cols].reset_index(drop=True), x_model.reset_index(drop=True)], axis=1)
    v8_dataset_csv = os.path.join(output_dir, "v8_robust_feature_dataset.csv")
    v8_dataset.to_csv(v8_dataset_csv, index=False, encoding="utf-8-sig")

    save_text(os.path.join(output_dir, "v8_used_v7_base_features.txt"), "\n".join(kept_v7))
    save_text(os.path.join(output_dir, "v8_used_heatmap_base_features.txt"), "\n".join(hm_feature_cols))
    save_text(os.path.join(output_dir, "v8_used_all_model_features.txt"), "\n".join(x_model.columns.tolist()))

    return {
        "initial_v7_features": initial_v7,
        "v7_base_features": kept_v7,
        "heatmap_base_features": hm_feature_cols,
        "model_features": x_model.columns.tolist(),
        "X": x_model,
        "removed_v7_csv": removed_v7_csv,
        "v8_dataset_csv": v8_dataset_csv,
    }


# ============================================================
# 6. 模型训练与分组验证
# ============================================================

def build_classifier():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        max_depth=None,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_features=RF_MAX_FEATURES,
        n_jobs=-1,
    )


def get_group_oof_probabilities(X_train, y_train, groups_train):
    y_train = np.asarray(y_train, dtype=int)
    groups_train = np.asarray(groups_train).astype(str)
    unique_train_groups = sorted(pd.unique(groups_train).tolist())
    oof_prob = np.zeros(len(y_train), dtype=float)
    filled = np.zeros(len(y_train), dtype=bool)

    if len(unique_train_groups) >= 2:
        for g in unique_train_groups:
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
        p_min = float(sub_prob.min())
        p_max = float(sub_prob.max())
        if abs(p_max - p_min) < 1e-12:
            rel = pd.Series(0.5, index=idx)
        else:
            rel = (sub_prob - p_min) / (p_max - p_min)
        pred_df.loc[idx, "prob_relative_minmax_in_group"] = rel

    cutoff = 1.0 - RANK_TRUE_FRACTION_FOR_BINARY
    pred_df["v8_rank_binary_pred"] = np.where(
        pred_df["prob_rank_pct_in_group"] > cutoff,
        "TRUE_LEAK",
        "FALSE_LEAK"
    )

    pred_df["v8_rank_level"] = np.select(
        [
            pred_df["prob_rank_pct_in_group"] >= RANK_TRUE_LIKE_PCT,
            pred_df["prob_rank_pct_in_group"] <= RANK_FALSE_LIKE_PCT,
        ],
        ["TRUE_LIKE", "FALSE_LIKE"],
        default="SUSPECT"
    )

    final = []
    for _, r in pred_df.iterrows():
        model_pred = r["v8_model_pred"]
        rank_pred = r["v8_rank_binary_pred"]
        if model_pred == rank_pred:
            final.append(model_pred)
        else:
            final.append("SUSPECT")
    pred_df["v8_final_decision"] = final
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
    decisive_mask = pred_df["v8_final_decision"].isin(["TRUE_LEAK", "FALSE_LEAK"])
    n_decisive = int(decisive_mask.sum())
    n_suspect = int((~decisive_mask).sum())
    decisive_rate = n_decisive / len(pred_df)
    if n_decisive > 0:
        decisive_accuracy = float((pred_df.loc[decisive_mask, "v8_final_decision"] == pred_df.loc[decisive_mask, "true_label"]).mean())
    else:
        decisive_accuracy = 0.0
    strict_correct = (pred_df["v8_final_decision"] == pred_df["true_label"]) & decisive_mask
    strict_accuracy = float(strict_correct.mean())
    return {
        "decisive_rate": float(decisive_rate),
        "decisive_accuracy": float(decisive_accuracy),
        "strict_accuracy_suspect_as_wrong": float(strict_accuracy),
        "n_suspect": n_suspect,
        "n_decisive": n_decisive,
    }


def leave_one_time_group_validation(df, X, output_dir):
    y_all = label_to_binary(df[LABEL_COL].astype(str).values)
    groups = df[GROUP_COL].astype(str).values
    unique_groups = sorted(pd.unique(groups).tolist())

    all_pred_rows = []
    group_rows = []

    print("\n开始 v8 按时间点整组验证...")
    print("分组数量:", len(unique_groups))

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

        oof_prob = get_group_oof_probabilities(X_train, y_train, groups_train)
        best_t, best_score, threshold_curve = find_best_threshold(
            y_train, oof_prob, metric=THRESHOLD_METRIC, grid=THRESHOLD_GRID
        )
        curve_path = os.path.join(output_dir, f"v8_threshold_curve_train_without_{safe_name(test_group)}.csv")
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
                "test_group": test_group,
                "dataset": test_df.loc[i, "dataset"] if "dataset" in test_df.columns else "",
                "time": test_df.loc[i, "time"] if "time" in test_df.columns else "",
                "center": normalize_center_id(test_df.loc[i, "center"] if "center" in test_df.columns else "00"),
                "true_label": true_label,
                "prob_TRUE_LEAK": float(prob[i]),
                "v8_best_threshold": best_t,
                "default_pred_0p5": default_pred,
                "default_correct": int(default_pred == true_label),
                "v8_model_pred": model_pred,
                "v8_model_correct": int(model_pred == true_label),
            }

            # 附带部分关键特征方便排查
            for k in [
                "ratio_60_70k", "spec_slope", "spec_flatness", "direction_contrast",
                "decay_R2", "hm_hot_area_p95_ratio", "hm_weighted_elongation",
                "hm_entropy_2d", "hm_directed_core_score", "hm_diffuse_score"
            ]:
                if k in test_df.columns:
                    row[k] = test_df.loc[i, k]
            group_pred_rows.append(row)

        group_pred_df = pd.DataFrame(group_pred_rows)
        group_pred_df = add_probability_rank_columns(group_pred_df)

        rank_pred_binary = label_to_binary(group_pred_df["v8_rank_binary_pred"].values)
        m_rank = metrics_from_pred(y_test, rank_pred_binary)
        final_metrics = calc_final_decision_metrics(group_pred_df)

        print(
            f"  测试组 {test_group}: "
            f"n={len(y_test)}, best_t={best_t:.3f}, "
            f"default_acc={m_default['accuracy']:.3f}, "
            f"model_acc={m_model['accuracy']:.3f}, "
            f"rank_acc={m_rank['accuracy']:.3f}, "
            f"final_decisive_acc={final_metrics['decisive_accuracy']:.3f}, "
            f"suspect={final_metrics['n_suspect']}, "
            f"auc={auc if not np.isnan(auc) else 'NA'}"
        )

        group_rows.append({
            "test_group": test_group,
            "n_test": len(y_test),
            "n_true": int(np.sum(y_test == 1)),
            "n_false": int(np.sum(y_test == 0)),
            "v8_best_threshold": best_t,
            "v8_train_oof_best_score": best_score,
            "auc": auc,
            "default_accuracy_0p5": m_default["accuracy"],
            "default_balanced_accuracy_0p5": m_default["balanced_accuracy"],
            "v8_model_accuracy": m_model["accuracy"],
            "v8_model_balanced_accuracy": m_model["balanced_accuracy"],
            "v8_model_recall_TRUE_LEAK": m_model["recall_TRUE_LEAK"],
            "v8_model_recall_FALSE_LEAK": m_model["recall_FALSE_LEAK"],
            "v8_model_tp": m_model["tp"],
            "v8_model_tn": m_model["tn"],
            "v8_model_fp": m_model["fp"],
            "v8_model_fn": m_model["fn"],
            "v8_rank_accuracy": m_rank["accuracy"],
            "v8_rank_balanced_accuracy": m_rank["balanced_accuracy"],
            "v8_rank_recall_TRUE_LEAK": m_rank["recall_TRUE_LEAK"],
            "v8_rank_recall_FALSE_LEAK": m_rank["recall_FALSE_LEAK"],
            "v8_final_decisive_rate": final_metrics["decisive_rate"],
            "v8_final_decisive_accuracy": final_metrics["decisive_accuracy"],
            "v8_final_strict_accuracy_suspect_as_wrong": final_metrics["strict_accuracy_suspect_as_wrong"],
            "v8_final_n_suspect": final_metrics["n_suspect"],
            "v8_final_n_decisive": final_metrics["n_decisive"],
        })

        all_pred_rows.extend(group_pred_df.to_dict(orient="records"))

    group_df = pd.DataFrame(group_rows)
    pred_df = pd.DataFrame(all_pred_rows)

    group_csv = os.path.join(output_dir, "v8_group_validation_summary.csv")
    pred_csv = os.path.join(output_dir, "v8_predictions.csv")
    wrong_csv = os.path.join(output_dir, "v8_model_misclassified_samples.csv")
    suspect_csv = os.path.join(output_dir, "v8_suspect_samples.csv")

    group_df.to_csv(group_csv, index=False, encoding="utf-8-sig")
    pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")

    if len(pred_df):
        pred_df[pred_df["v8_model_correct"] == 0].to_csv(wrong_csv, index=False, encoding="utf-8-sig")
        pred_df[pred_df["v8_final_decision"] == "SUSPECT"].to_csv(suspect_csv, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(wrong_csv, index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(suspect_csv, index=False, encoding="utf-8-sig")

    return group_df, pred_df, group_csv, pred_csv, wrong_csv, suspect_csv


def safe_name(s):
    return str(s).replace("\\", "_").replace("/", "_").replace(":", "_").replace(".", "_")


# ============================================================
# 7. 最终模型保存、报告、画图
# ============================================================

def train_final_model(df, X, feature_info, output_dir):
    try:
        import joblib
    except Exception as e:
        raise RuntimeError("缺少 joblib，请运行: pip install joblib") from e

    y = label_to_binary(df[LABEL_COL].astype(str).values)
    groups = df[GROUP_COL].astype(str).values

    oof_prob = get_group_oof_probabilities(X.reset_index(drop=True), y, groups)
    best_t, best_score, threshold_curve = find_best_threshold(
        y, oof_prob, metric=THRESHOLD_METRIC, grid=THRESHOLD_GRID
    )
    threshold_curve_csv = os.path.join(output_dir, "v8_global_group_oof_threshold_curve.csv")
    threshold_curve.to_csv(threshold_curve_csv, index=False, encoding="utf-8-sig")

    clf = build_classifier()
    clf.fit(X, y)

    model_path = os.path.join(output_dir, "v8_final_heatmap_robust_classifier.pkl")
    joblib.dump(clf, model_path)

    importance_df = pd.DataFrame({
        "feature": X.columns.tolist(),
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance_csv = os.path.join(output_dir, "v8_final_feature_importance.csv")
    importance_df.to_csv(importance_csv, index=False, encoding="utf-8-sig")

    config = {
        "version": "v8_heatmap_shape_classifier",
        "model_type": "RandomForestClassifier",
        "positive_label": "TRUE_LEAK",
        "label_mapping": {"FALSE_LEAK": 0, "TRUE_LEAK": 1},
        "recommended_threshold": best_t,
        "threshold_metric": THRESHOLD_METRIC,
        "threshold_score_on_group_oof": best_score,
        "v7_base_features": feature_info["v7_base_features"],
        "heatmap_base_features": feature_info["heatmap_base_features"],
        "model_features": feature_info["model_features"],
        "true_heatmap_dirs": TRUE_HEATMAP_DIRS,
        "false_heatmap_dirs": FALSE_HEATMAP_DIRS,
        "created_at": str(datetime.now()),
        "input_csv": MERGED_FEATURE_CSV,
        "note": "v8 使用 v7 稳健特征 + 从 heatmap PNG 提取的形态特征。新数据预测时应按完整 time_folder 批量输入。",
    }

    config_path = os.path.join(output_dir, "v8_final_model_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    return {
        "model_path": model_path,
        "config_path": config_path,
        "importance_csv": importance_csv,
        "importance_df": importance_df,
        "threshold_curve_csv": threshold_curve_csv,
        "recommended_threshold": best_t,
        "threshold_score": best_score,
    }


def make_report(df, hm_df, feature_info, group_df, pred_df, final_info, output_dir):
    lines = []
    lines.append("v8 热力图形态特征 + v7 稳健特征 分类验证报告")
    lines.append("=" * 100)
    lines.append(f"生成时间: {datetime.now()}")
    lines.append(f"输入特征表: {MERGED_FEATURE_CSV}")
    lines.append("")
    lines.append("样本情况:")
    lines.append(f"  总样本数: {len(df)}")
    for label, count in df[LABEL_COL].value_counts().items():
        lines.append(f"  {label}: {int(count)}")
    lines.append(f"  热力图缺失/失败数量: {int(hm_df['hm_missing'].sum()) if 'hm_missing' in hm_df.columns else 'NA'}")
    lines.append("")

    lines.append("特征情况:")
    lines.append(f"  v7 初始数值特征数: {len(feature_info['initial_v7_features'])}")
    lines.append(f"  v7 删除不稳定后基础特征数: {len(feature_info['v7_base_features'])}")
    lines.append(f"  heatmap 基础形态特征数: {len(feature_info['heatmap_base_features'])}")
    lines.append(f"  加入 time 内部 z/rank 后模型特征数: {len(feature_info['model_features'])}")
    lines.append(f"  v8 数据表: {feature_info['v8_dataset_csv']}")
    lines.append("")

    if group_df is not None and len(group_df):
        lines.append("按时间点整组验证平均结果:")
        lines.append(f"  默认阈值0.5平均准确率: {group_df['default_accuracy_0p5'].mean():.4f}")
        lines.append(f"  v8模型阈值平均准确率: {group_df['v8_model_accuracy'].mean():.4f}")
        lines.append(f"  v8概率排名平均准确率: {group_df['v8_rank_accuracy'].mean():.4f}")
        lines.append(f"  v8最终三档-平均明确判定比例: {group_df['v8_final_decisive_rate'].mean():.4f}")
        lines.append(f"  v8最终三档-明确样本平均准确率: {group_df['v8_final_decisive_accuracy'].mean():.4f}")
        lines.append(f"  平均AUC: {group_df['auc'].mean():.4f}")
        lines.append("")
        lines.append("各时间点结果:")
        for _, r in group_df.iterrows():
            lines.append(
                f"  {r['test_group']}: "
                f"n={int(r['n_test'])}, "
                f"best_t={r['v8_best_threshold']:.3f}, "
                f"default_acc={r['default_accuracy_0p5']:.3f}, "
                f"model_acc={r['v8_model_accuracy']:.3f}, "
                f"rank_acc={r['v8_rank_accuracy']:.3f}, "
                f"final_decisive_rate={r['v8_final_decisive_rate']:.3f}, "
                f"final_decisive_acc={r['v8_final_decisive_accuracy']:.3f}, "
                f"suspect={int(r['v8_final_n_suspect'])}, "
                f"AUC={r['auc']}"
            )
        lines.append("")

    lines.append("最终模型:")
    lines.append(f"  模型文件: {final_info['model_path']}")
    lines.append(f"  配置文件: {final_info['config_path']}")
    lines.append(f"  推荐阈值: {final_info['recommended_threshold']:.3f}")
    lines.append(f"  OOF阈值优化得分: {final_info['threshold_score']:.4f}")
    lines.append("")
    lines.append("最终模型重要特征前30:")
    for _, row in final_info["importance_df"].head(30).iterrows():
        lines.append(f"  {row['feature']}: {row['importance']:.6f}")

    report_path = os.path.join(output_dir, "v8_report.txt")
    save_text(report_path, "\n".join(lines))
    return report_path


def plot_group_metrics(group_df, output_dir):
    if group_df is None or len(group_df) == 0:
        return None
    fig_dir = os.path.join(output_dir, "figures")
    ensure_dir(fig_dir)
    labels = group_df["test_group"].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(12, 5))
    plt.bar(x - width, group_df["default_accuracy_0p5"], width, label="Default 0.5")
    plt.bar(x, group_df["v8_model_accuracy"], width, label="V8 model")
    plt.bar(x + width, group_df["v8_rank_accuracy"], width, label="V8 rank")
    plt.ylim(0, 1.05)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("V8 group validation accuracy comparison")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(fig_dir, "v8_group_accuracy_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_feature_importance(importance_df, output_dir, top_n=30):
    if importance_df is None or len(importance_df) == 0:
        return None
    fig_dir = os.path.join(output_dir, "figures")
    ensure_dir(fig_dir)
    top = importance_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(12, 10))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance")
    plt.title(f"V8 top {top_n} feature importance")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(fig_dir, "v8_top_feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_threshold_curve(threshold_curve_csv, output_dir):
    if not os.path.exists(threshold_curve_csv):
        return None
    fig_dir = os.path.join(output_dir, "figures")
    ensure_dir(fig_dir)
    df = pd.read_csv(threshold_curve_csv)
    plt.figure(figsize=(9, 5))
    plt.plot(df["threshold"], df["balanced_accuracy"], label="Balanced accuracy")
    plt.plot(df["threshold"], df["accuracy"], label="Accuracy")
    plt.plot(df["threshold"], df["f1_TRUE_LEAK"], label="F1 TRUE_LEAK")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("V8 global group-OOF threshold curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(fig_dir, "v8_global_threshold_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_heatmap_feature_examples(hm_df, output_dir):
    """画几个热力图形态特征的真假分布，帮助判断热图特征是否有效。"""
    fig_dir = os.path.join(output_dir, "figures")
    ensure_dir(fig_dir)
    features = [
        "hm_hot_area_p95_ratio",
        "hm_weighted_elongation",
        "hm_entropy_2d",
        "hm_largest_component_ratio_to_hot_p95",
        "hm_directed_core_score",
        "hm_diffuse_score",
    ]
    paths = []
    for f in features:
        if f not in hm_df.columns:
            continue
        plt.figure(figsize=(8, 5))
        for label in ["TRUE_LEAK", "FALSE_LEAK"]:
            vals = pd.to_numeric(hm_df.loc[hm_df["label"] == label, f], errors="coerce").dropna().values
            plt.hist(vals, bins=20, alpha=0.6, label=label)
        plt.title(f"Heatmap feature distribution: {f}")
        plt.xlabel(f)
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(fig_dir, f"v8_heatmap_feature_{f}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)
    return paths


# ============================================================
# 8. 主函数
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)
    print("=" * 100)
    print("v8 热力图形态特征 + v7 稳健特征 分类程序")
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
        print("CSV中没有 center 列，无法匹配 heatmap 文件。")
        return

    df[LABEL_COL] = df[LABEL_COL].astype(str)
    df = df[df[LABEL_COL].isin(["TRUE_LEAK", "FALSE_LEAK"])].copy().reset_index(drop=True)
    df["center"] = df["center"].apply(normalize_center_id)

    print("样本数量:", len(df))
    print(df[LABEL_COL].value_counts())

    # 1. 提取热力图形态特征
    hm_df, hm_csv = build_heatmap_feature_table(df, OUTPUT_DIR)

    # 2. 构造 v8 特征矩阵
    print("\n开始构造 v8 特征矩阵：v7稳健特征 + heatmap形态特征 + time内部z/rank...")
    feature_info = prepare_v8_feature_matrix(df, hm_df, OUTPUT_DIR)
    X = feature_info["X"]

    print("v7初始数值特征数:", len(feature_info["initial_v7_features"]))
    print("v7稳健基础特征数:", len(feature_info["v7_base_features"]))
    print("heatmap基础形态特征数:", len(feature_info["heatmap_base_features"]))
    print("最终模型特征数:", len(feature_info["model_features"]))
    print("v8稳健特征数据表:", feature_info["v8_dataset_csv"])

    # 3. 分组验证
    group_df, pred_df, group_csv, pred_csv, wrong_csv, suspect_csv = leave_one_time_group_validation(df, X, OUTPUT_DIR)
    print("\n分组验证汇总:", group_csv)
    print("预测明细:", pred_csv)
    print("v8模型误判样本:", wrong_csv)
    print("v8最终SUSPECT样本:", suspect_csv)

    # 4. 训练最终模型
    final_info = train_final_model(df, X, feature_info, OUTPUT_DIR)
    print("\n最终模型:", final_info["model_path"])
    print("最终配置:", final_info["config_path"])
    print("最终特征重要性:", final_info["importance_csv"])
    print("全局OOF阈值曲线:", final_info["threshold_curve_csv"])
    print(f"v8推荐阈值: {final_info['recommended_threshold']:.3f}")
    print(f"v8 OOF阈值优化得分: {final_info['threshold_score']:.4f}")

    # 5. 报告和图
    report_path = make_report(df, hm_df, feature_info, group_df, pred_df, final_info, OUTPUT_DIR)
    fig1 = plot_group_metrics(group_df, OUTPUT_DIR)
    fig2 = plot_feature_importance(final_info["importance_df"], OUTPUT_DIR)
    fig3 = plot_threshold_curve(final_info["threshold_curve_csv"], OUTPUT_DIR)
    heatmap_figs = plot_heatmap_feature_examples(hm_df, OUTPUT_DIR)

    print("报告:", report_path)
    print("\n图片输出:")
    for p in [fig1, fig2, fig3] + heatmap_figs:
        if p:
            print(" ", p)

    print("\n最终模型重要特征前15:")
    for _, row in final_info["importance_df"].head(15).iterrows():
        print(f"  {row['feature']}: {row['importance']:.6f}")

    if group_df is not None and len(group_df):
        print("\n各时间点核心结果:")
        for _, r in group_df.iterrows():
            print(
                f"  {r['test_group']}: "
                f"default_acc={r['default_accuracy_0p5']:.3f}, "
                f"model_acc={r['v8_model_accuracy']:.3f}, "
                f"rank_acc={r['v8_rank_accuracy']:.3f}, "
                f"final_decisive_acc={r['v8_final_decisive_accuracy']:.3f}, "
                f"suspect={int(r['v8_final_n_suspect'])}, "
                f"auc={r['auc'] if not np.isnan(r['auc']) else 'NA'}"
            )

    print("\n" + "=" * 100)
    print("全部完成")
    print("输出文件夹:", OUTPUT_DIR)
    print("=" * 100)


if __name__ == "__main__":
    main()
