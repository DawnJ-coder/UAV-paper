# -*- coding: utf-8 -*-
"""
leak_v8_fixed_global_matrix.py

v8 修正版：全局方向-频段矩阵版 directed wideband

用途
----
修正上一版 v8 的核心问题：
    旧 physics_directed_leak_score 在 144226 上完全反向，
    原因是“宽频覆盖”按每个方向内部归一化，容易把弥散假泄漏误认为宽频。

这一版改成：
    1. 对每个样本构造 8方向 × 5频段 的全局能量矩阵；
    2. 用全局最大能量阈值判断强频段，而不是每个方向自己内部判断；
    3. 自动从训练组里选择/校正物理分数方向；
    4. 优先复用上一版 v8 已经输出的特征表，不重复计算已有的空间/频谱/时间特征；
    5. 只补算上一版没有保存的 “方向×频段矩阵特征”。

如果已存在：
    C:\\Users\\jiangxinru6\\Desktop\\wurenji\\leak_v8_standalone_directed_wideband_auto_center_v2_results\\v8_feature_dataset_with_time_relative.csv

程序会直接读取它，然后只重新扫描 offset wav，补算全局矩阵特征。
如果本程序已经生成过：
    v8_fixed_global_matrix_features.csv

再次运行会直接读取缓存，不再重新读 wav。

运行：
    python leak_v8_fixed_global_matrix.py
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


# ============================================================
# 1. 路径配置
# ============================================================

BASE_DIR = r"C:\Users\jiangxinru6\Desktop\wurenji"

OLD_V8_DIR = os.path.join(
    BASE_DIR,
    "leak_v8_standalone_directed_wideband_auto_center_v2_results"
)

OLD_FEATURE_WITH_RELATIVE = os.path.join(
    OLD_V8_DIR,
    "v8_feature_dataset_with_time_relative.csv"
)

OLD_FEATURE_RAW = os.path.join(
    OLD_V8_DIR,
    "v8_feature_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "leak_v8_fixed_global_matrix_results"
)

GLOBAL_MATRIX_CACHE = os.path.join(
    OUTPUT_DIR,
    "v8_fixed_global_matrix_features.csv"
)

FIXED_DATASET_CSV = os.path.join(
    OUTPUT_DIR,
    "v8_fixed_dataset.csv"
)

TIME_FOLDERS = [
    "HM20260626_142938.ld",
    "HM20260626_143034.ld",
    "HM20260626_144226.ld",
    "HM20260626_144325.ld",
]

TARGET_TIME = "HM20260626_144226.ld"

DATASETS = [
    {
        "label": "TRUE_LEAK",
        "offset_root": r"D:\gas\beamform_results_offset_multiple",
    },
    {
        "label": "FALSE_LEAK",
        "offset_root": r"D:\gas\beamform_results_cs_offset_multiple",
    },
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

SUBBANDS = [
    (20000, 30000),
    (30000, 40000),
    (40000, 50000),
    (50000, 60000),
    (60000, 70000),
]

WAV_EXTS = [".wav", ".WAV"]

NFFT = 4096
WELCH_NPERSEG = 4096
WELCH_NOVERLAP = 2048

RANDOM_STATE = 42
RANK_TRUE_FRACTION = 0.50
THRESHOLD_GRID = np.linspace(0.01, 0.99, 99)

# 全局强能量阈值。不是每个方向内部阈值。
GLOBAL_STRONG_THRESHOLDS = [0.08, 0.12, 0.18, 0.25]

META_COLS = {
    "label", "true_label", "time", "test_group", "center", "center_norm",
    "center_file", "best_direction", "error", "experiment"
}


# ============================================================
# 2. 基础函数
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


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


def gini_coefficient(x):
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 0)

    if len(x) == 0 or np.sum(x) <= 1e-20:
        return 0.0

    x = np.sort(x)
    n = len(x)
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def spectral_flatness_from_values(x):
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 1e-20)
    return float(np.exp(np.mean(np.log(x))) / (np.mean(x) + 1e-20))


def safe_auc(y_true, score):
    try:
        from sklearn.metrics import roc_auc_score
        y_true = np.asarray(y_true, dtype=int)
        score = np.asarray(score, dtype=float)

        mask = np.isfinite(score)
        y_true = y_true[mask]
        score = score[mask]

        if len(np.unique(y_true)) < 2:
            return np.nan

        return float(roc_auc_score(y_true, score))
    except Exception:
        return np.nan


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

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "recall_TRUE": float(recall_true),
        "recall_FALSE": float(recall_false),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ============================================================
# 3. 文件解析：固定支持你的 offset 命名
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


def parse_offset_filename_candidates(path):
    """
    支持：
        HM20260626_142938.ld_00_14d5_down_beamform_result.wav

    两个候选：
        A: 第一个数字 00 是 center
        B: 14d5 里的 14 是 center，5 是距离

    根据 center数量和每center是否接近64自动选择。
    """
    base = os.path.basename(str(path)).lower().replace("-", "_").replace(" ", "_")

    direction_pattern = r"(up_left|up_right|down_left|down_right|up|down|left|right)"

    m = re.search(
        rf"\.ld_(\d{{1,3}})_(\d{{1,3}})d(\d{{1,3}})_({direction_pattern})_beamform",
        base,
        flags=re.IGNORECASE
    )

    candidates = []

    if m:
        first_num = m.group(1).zfill(2)
        second_num = m.group(2).zfill(2)
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
                "center": second_num,
                "direction": direction,
                "distance": dist,
            })

        return candidates

    # 兜底解析
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


def discover_offset_files(offset_root, time_folder):
    root = os.path.join(offset_root, time_folder)
    files = list_wav_files(root)

    schema_maps = {}

    for f in files:
        for c in parse_offset_filename_candidates(f):
            schema = c["schema"]
            key = (c["center"], c["direction"], int(c["distance"]))
            schema_maps.setdefault(schema, {})
            schema_maps[schema].setdefault(key, [])
            schema_maps[schema][key].append(f)

    if not schema_maps:
        print("  [错误] 没有识别到 offset wav:", root)
        return {}

    rows = []

    for schema, mp in schema_maps.items():
        centers = sorted(set(k[0] for k in mp.keys()))
        counts = []
        for cc in centers:
            counts.append(sum(1 for k in mp.keys() if k[0] == cc))

        avg_per_center = float(np.mean(counts)) if counts else 0.0
        med_per_center = float(np.median(counts)) if counts else 0.0
        total = len(mp)
        n_centers = len(centers)

        # center数量优先，其次组合数，其次接近64
        score = n_centers * 1000.0 + total + 100.0 / (1.0 + abs(avg_per_center - 64.0))

        rows.append({
            "schema": schema,
            "n_centers": n_centers,
            "total_combos": total,
            "avg_per_center": avg_per_center,
            "median_per_center": med_per_center,
            "score": score,
        })

    rows = sorted(rows, key=lambda r: r["score"], reverse=True)
    best = rows[0]["schema"]

    print("  offset解析方案候选:")
    for r in rows:
        print(
            f"    {r['schema']}: centers={r['n_centers']}, "
            f"combos={r['total_combos']}, avg/center={r['avg_per_center']:.1f}, "
            f"median/center={r['median_per_center']:.1f}"
        )
    print("  采用offset解析方案:", best)

    return schema_maps[best]


# ============================================================
# 4. WAV 频段能量
# ============================================================

def read_wav_float(path):
    fs, x = wavfile.read(path)

    if x.ndim > 1:
        x = x.astype(np.float64).mean(axis=1)
    else:
        x = x.astype(np.float64)

    x = x - np.mean(x)

    max_abs = np.max(np.abs(x)) + 1e-12
    if max_abs > 10:
        x = x / max_abs

    return fs, x


def wav_subband_energy(path):
    fs, x = read_wav_float(path)

    nperseg = min(WELCH_NPERSEG, len(x))
    if nperseg < 256:
        return np.zeros(len(SUBBANDS), dtype=float)

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

    vals = []

    for lo, hi in SUBBANDS:
        mask = (f >= lo) & (f < hi)
        if np.any(mask):
            vals.append(float(np.trapz(pxx[mask], f[mask])))
        else:
            vals.append(0.0)

    return np.asarray(vals, dtype=float)


# ============================================================
# 5. 全局矩阵特征
# ============================================================

def compute_features_from_matrix(M):
    """
    M: 8方向 × 5频段，近场距离聚合。
    新逻辑重点：
        - 强频段按全局 max 阈值判断；
        - 不再按每个方向内部归一化判断宽频。
    """
    M = np.asarray(M, dtype=float)
    M = np.maximum(M, 0)

    out = {}

    total = float(np.sum(M))
    max_cell = float(np.max(M)) if M.size else 0.0

    dir_total = M.sum(axis=1)
    band_total = M.sum(axis=0)

    sorted_dir = np.argsort(dir_total)[::-1]
    top1 = sorted_dir[0]
    top2 = sorted_dir[:2]
    rest = [i for i in range(M.shape[0]) if i not in set(top2.tolist())]

    top1_total = float(dir_total[top1])
    top2_total = float(np.sum(dir_total[top2]))
    rest_total = float(np.sum(dir_total[rest])) if rest else 0.0

    out["gm_total_energy"] = total
    out["gm_max_cell_energy"] = max_cell
    out["gm_direction_entropy"] = entropy_norm(dir_total)
    out["gm_direction_gini"] = gini_coefficient(dir_total)
    out["gm_direction_cv"] = float(np.std(dir_total) / (np.mean(dir_total) + 1e-20))
    out["gm_top1_total_ratio"] = float(top1_total / (total + 1e-20))
    out["gm_top2_total_ratio"] = float(top2_total / (total + 1e-20))
    out["gm_rest_total_ratio"] = float(rest_total / (total + 1e-20))
    out["gm_top2_to_rest_total_ratio"] = float(top2_total / (rest_total + 1e-20))

    # 频段比例
    band_p = band_total / (np.sum(band_total) + 1e-20)
    for j, (lo, hi) in enumerate(SUBBANDS):
        out[f"gm_global_{lo//1000}_{hi//1000}k_ratio"] = float(band_p[j])

    top2_band = M[top2, :].sum(axis=0)
    top2_band_p = top2_band / (np.sum(top2_band) + 1e-20)

    for j, (lo, hi) in enumerate(SUBBANDS):
        out[f"gm_top2_{lo//1000}_{hi//1000}k_ratio"] = float(top2_band_p[j])

    out["gm_top2_band_entropy"] = entropy_norm(top2_band)
    out["gm_top2_band_flatness"] = spectral_flatness_from_values(top2_band)

    # 多阈值强格子特征
    for thr in GLOBAL_STRONG_THRESHOLDS:
        tag = f"{int(thr * 100):02d}"

        if max_cell <= 1e-20:
            strong = np.zeros_like(M, dtype=bool)
        else:
            strong = M >= (max_cell * thr)

        strong_energy_matrix = M * strong
        strong_total = float(np.sum(strong_energy_matrix))

        dir_strong = strong_energy_matrix.sum(axis=1)
        band_strong = strong_energy_matrix.sum(axis=0)

        top2_strong = float(np.sum(dir_strong[top2]))
        rest_strong = float(np.sum(dir_strong[rest])) if rest else 0.0

        active_bands_top1 = int(np.sum(strong[top1, :]))
        active_bands_top2 = int(np.sum(np.any(strong[top2, :], axis=0)))
        active_dirs = int(np.sum(np.any(strong, axis=1)))
        active_cells = int(np.sum(strong))

        top2_coverage = active_bands_top2 / len(SUBBANDS)
        top1_coverage = active_bands_top1 / len(SUBBANDS)

        # 核心：强宽频是否集中在 top2方向
        top2_strong_ratio = float(top2_strong / (strong_total + 1e-20))
        rest_strong_ratio = float(rest_strong / (strong_total + 1e-20))

        # 连续宽频：top2方向是否覆盖多个强频段
        strong_band_entropy = entropy_norm(band_strong)
        strong_dir_entropy = entropy_norm(dir_strong)

        out[f"gm_thr{tag}_strong_total_ratio"] = float(strong_total / (total + 1e-20))
        out[f"gm_thr{tag}_active_cells"] = active_cells
        out[f"gm_thr{tag}_active_dirs"] = active_dirs
        out[f"gm_thr{tag}_top1_active_band_count"] = active_bands_top1
        out[f"gm_thr{tag}_top2_active_band_count"] = active_bands_top2
        out[f"gm_thr{tag}_top1_band_coverage"] = top1_coverage
        out[f"gm_thr{tag}_top2_band_coverage"] = top2_coverage
        out[f"gm_thr{tag}_top2_strong_ratio"] = top2_strong_ratio
        out[f"gm_thr{tag}_rest_strong_ratio"] = rest_strong_ratio
        out[f"gm_thr{tag}_top2_to_rest_strong_ratio"] = float(top2_strong / (rest_strong + 1e-20))
        out[f"gm_thr{tag}_strong_band_entropy"] = strong_band_entropy
        out[f"gm_thr{tag}_strong_dir_entropy"] = strong_dir_entropy

        # 正向泄漏分数：少数方向强 + top2方向覆盖多频段 + 其他方向弱
        directed = (
            top2_strong_ratio
            * top2_coverage
            * (1.0 - out["gm_direction_entropy"] + 1e-6)
            * (1.0 - rest_strong_ratio + 1e-6)
            * (0.5 + 0.5 * min(out["gm_direction_cv"], 3.0) / 3.0)
        )

        # 弥散分数：强格子分散在更多方向 + rest方向强 + 方向熵高
        diffuse = (
            out["gm_direction_entropy"]
            * (active_dirs / len(DIRECTIONS))
            * (rest_strong_ratio + 1e-6)
            * (1.0 - top2_coverage + 0.2)
        )

        out[f"gm_score_directed_thr{tag}"] = float(directed)
        out[f"gm_score_diffuse_thr{tag}"] = float(diffuse)
        out[f"gm_score_directed_minus_diffuse_thr{tag}"] = float(directed - diffuse)
        out[f"gm_score_directed_over_diffuse_thr{tag}"] = float(directed / (diffuse + 1e-20))

    # 综合分数：多个阈值平均，避免单一阈值过敏
    directed_cols = [f"gm_score_directed_thr{int(t * 100):02d}" for t in GLOBAL_STRONG_THRESHOLDS]
    diffuse_cols = [f"gm_score_diffuse_thr{int(t * 100):02d}" for t in GLOBAL_STRONG_THRESHOLDS]
    minus_cols = [f"gm_score_directed_minus_diffuse_thr{int(t * 100):02d}" for t in GLOBAL_STRONG_THRESHOLDS]
    ratio_cols = [f"gm_score_directed_over_diffuse_thr{int(t * 100):02d}" for t in GLOBAL_STRONG_THRESHOLDS]

    out["gm_score_directed_mean"] = float(np.mean([out[c] for c in directed_cols]))
    out["gm_score_diffuse_mean"] = float(np.mean([out[c] for c in diffuse_cols]))
    out["gm_score_directed_minus_diffuse_mean"] = float(np.mean([out[c] for c in minus_cols]))
    out["gm_score_directed_over_diffuse_mean"] = float(np.median([out[c] for c in ratio_cols]))

    return out


def compute_global_matrix_feature_table():
    ensure_dir(OUTPUT_DIR)

    if os.path.exists(GLOBAL_MATRIX_CACHE):
        print("发现全局矩阵特征缓存，直接读取:", GLOBAL_MATRIX_CACHE)
        return pd.read_csv(GLOBAL_MATRIX_CACHE)

    rows = []

    print("\n开始补算全局 8方向×5频段矩阵特征...")
    print("说明：这里只补算上一版没有保存的矩阵特征，不重复计算全部旧特征。")

    for ds in DATASETS:
        label = ds["label"]
        offset_root = ds["offset_root"]

        if not os.path.exists(offset_root):
            raise FileNotFoundError(
                f"{label} offset_root 不存在: {offset_root}\n"
                "请在脚本顶部 DATASETS 中修改原始 offset wav 路径。"
            )

        print("\n" + "=" * 100)
        print("数据集:", label)
        print("=" * 100)

        for time_folder in TIME_FOLDERS:
            print(f"\n处理 {label} / {time_folder}")

            offset_files = discover_offset_files(offset_root, time_folder)
            centers = sorted(set(k[0] for k in offset_files.keys()))

            count_by_center = {}
            for (cc, dd, dist), files in offset_files.items():
                count_by_center.setdefault(cc, 0)
                count_by_center[cc] += 1

            avg_count = float(np.mean(list(count_by_center.values()))) if count_by_center else 0.0
            print("  center数量:", len(centers))
            print("  offset组合数量:", len(offset_files))
            print(f"  平均每center offset组合数: {avg_count:.1f} / 64")
            print("  前10个center:", sorted(count_by_center.items())[:10])

            if len(centers) == 0:
                raise RuntimeError(f"{label}/{time_folder} 没有识别到 center。")

            if avg_count < 40:
                raise RuntimeError(
                    f"{label}/{time_folder} 平均每center offset组合数只有 {avg_count:.1f}，明显不对，停止。"
                )

            for i, center in enumerate(centers, 1):
                if i % 10 == 0 or i == len(centers):
                    print(f"  已处理 {i}/{len(centers)}")

                M = np.zeros((len(DIRECTIONS), len(SUBBANDS)), dtype=float)
                wav_count = 0

                for di, direction in enumerate(DIRECTIONS):
                    for dist in DISTANCES_CM:
                        if dist > NEAR_DISTANCE_MAX_CM:
                            continue

                        files = offset_files.get((center, direction, dist), [])
                        if not files:
                            continue

                        vals = []
                        for f in files:
                            try:
                                vals.append(wav_subband_energy(f))
                                wav_count += 1
                            except Exception:
                                pass

                        if vals:
                            M[di, :] += np.mean(np.asarray(vals), axis=0)

                row = {
                    "label": label,
                    "time": time_folder,
                    "center_norm": normalize_center_id(center),
                    "gm_near_offset_wav_count": wav_count,
                }

                # 保存矩阵格子，方便以后不用再读wav
                for di, direction in enumerate(DIRECTIONS):
                    for bj, (lo, hi) in enumerate(SUBBANDS):
                        row[f"gm_cell_{direction}_{lo//1000}_{hi//1000}k"] = float(M[di, bj])

                row.update(compute_features_from_matrix(M))
                rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(GLOBAL_MATRIX_CACHE, index=False, encoding="utf-8-sig")
    print("\n全局矩阵特征已保存:", GLOBAL_MATRIX_CACHE)

    return out


# ============================================================
# 6. 复用旧 v8 输出并合并新特征
# ============================================================

def load_or_build_base_dataset(gm_df):
    """
    优先复用旧 v8 输出。
    如果旧输出不存在，则至少用 gm_df 建一个基础数据集。
    """
    if os.path.exists(OLD_FEATURE_WITH_RELATIVE):
        print("\n复用上一版 v8 特征表:", OLD_FEATURE_WITH_RELATIVE)
        base = pd.read_csv(OLD_FEATURE_WITH_RELATIVE)
    elif os.path.exists(OLD_FEATURE_RAW):
        print("\n复用上一版 v8 原始特征表:", OLD_FEATURE_RAW)
        base = pd.read_csv(OLD_FEATURE_RAW)
    else:
        print("\n未找到上一版 v8 特征表，将只使用全局矩阵特征。")
        base = gm_df[["label", "time", "center_norm"]].copy()
        base["center"] = base["center_norm"]

    if "true_label" in base.columns and "label" not in base.columns:
        base = base.rename(columns={"true_label": "label"})

    if "center_norm" not in base.columns:
        base["center_norm"] = base["center"].apply(normalize_center_id)

    base["label"] = base["label"].astype(str)
    base["time"] = base["time"].astype(str)
    base["center_norm"] = base["center_norm"].apply(normalize_center_id)

    gm_df = gm_df.copy()
    gm_df["label"] = gm_df["label"].astype(str)
    gm_df["time"] = gm_df["time"].astype(str)
    gm_df["center_norm"] = gm_df["center_norm"].apply(normalize_center_id)

    # 删除旧 gm 列避免重复
    old_gm_cols = [c for c in base.columns if c.startswith("gm_")]
    if old_gm_cols:
        base = base.drop(columns=old_gm_cols)

    df = base.merge(
        gm_df,
        on=["label", "time", "center_norm"],
        how="inner",
        validate="one_to_one"
    )

    if len(df) == 0:
        raise RuntimeError("旧 v8 特征表和新 gm 特征无法合并，请检查 label/time/center 是否一致。")

    return df


def add_time_relative_features(df, cols):
    df = df.copy()

    for c in cols:
        if c not in df.columns:
            continue

        vals = safe_float_series(df[c])

        z_col = f"{c}__time_robust_z"
        r_col = f"{c}__time_rank_pct"

        if z_col in df.columns and r_col in df.columns:
            continue

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


def prepare_dataset():
    gm_df = compute_global_matrix_feature_table()
    df = load_or_build_base_dataset(gm_df)

    gm_cols_raw = [
        c for c in df.columns
        if c.startswith("gm_")
        and not c.startswith("gm_cell_")
        and safe_float_series(df[c]).notna().mean() >= 0.75
    ]

    df = add_time_relative_features(df, gm_cols_raw)

    df.to_csv(FIXED_DATASET_CSV, index=False, encoding="utf-8-sig")
    print("\n修正后合并数据集已保存:", FIXED_DATASET_CSV)

    return df


# ============================================================
# 7. 特征集
# ============================================================

def numeric_cols(df):
    cols = []

    for c in df.columns:
        if c in META_COLS:
            continue
        if c.startswith("gm_cell_"):
            # 原始矩阵格子不直接给模型，避免绝对能量干扰；用派生特征即可
            continue

        v = safe_float_series(df[c])
        if v.notna().mean() >= 0.75:
            cols.append(c)

    return cols


def build_feature_sets(df):
    all_num = numeric_cols(df)

    # 旧 bad dw/physics 先排除，避免继续污染 v8_fixed
    old_bad_prefixes = ("dw_", "physics_")
    base_no_old_dw = [
        c for c in all_num
        if not c.startswith(old_bad_prefixes)
        and not c.startswith("gm_")
    ]

    # 排除明显绝对能量，保留比例/形态/rank/z
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
        "total_energy",
        "max_cell_energy",
    ]

    base_robust = []
    for c in base_no_old_dw:
        if any(k in c for k in unstable_keywords):
            continue
        base_robust.append(c)

    gm_cols = [
        c for c in all_num
        if c.startswith("gm_")
        and not c.startswith("gm_cell_")
        and "total_energy" not in c
        and "max_cell_energy" not in c
    ]

    # 只挑候选物理分数，用于自动方向校正
    gm_score_cols = [
        c for c in gm_cols
        if c.startswith("gm_score_")
        or c in [
            "gm_top2_total_ratio",
            "gm_top2_to_rest_total_ratio",
            "gm_direction_entropy",
            "gm_direction_gini",
            "gm_direction_cv",
            "gm_rest_total_ratio",
        ]
    ]

    combined = []
    for c in base_robust + gm_cols:
        if c not in combined:
            combined.append(c)

    feature_sets = {
        "A_base_without_old_bad_dw": base_robust,
        "B_global_matrix_only": gm_cols,
        "C_v8_fixed_base_plus_global_matrix": combined,
    }

    return feature_sets, gm_score_cols


# ============================================================
# 8. 验证逻辑
# ============================================================

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


def build_matrix(train_df, test_df, cols):
    X_train = pd.DataFrame(index=train_df.index)
    X_test = pd.DataFrame(index=test_df.index)

    for c in cols:
        tr = safe_float_series(train_df[c]) if c in train_df.columns else pd.Series(np.nan, index=train_df.index)
        te = safe_float_series(test_df[c]) if c in test_df.columns else pd.Series(np.nan, index=test_df.index)

        med = tr.median()
        if not np.isfinite(med):
            med = 0.0

        X_train[c] = tr.fillna(med).astype(float)
        X_test[c] = te.fillna(med).astype(float)

    return X_train, X_test


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
        prob[~filled] = 0.5

    return prob


def best_threshold(y, prob):
    best_t = 0.5
    best_score = -1

    for t in THRESHOLD_GRID:
        pred = (prob >= t).astype(int)
        m = metrics(y, pred)
        score = m["balanced_accuracy"]

        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t, best_score


def add_rank_within_group(pred_df, score_col, out_prefix):
    pred_df = pred_df.copy()
    rank_col = f"{out_prefix}_rank_pct"
    pred_col = f"{out_prefix}_rank_pred"

    pred_df[rank_col] = 0.5

    for g, idx in pred_df.groupby("test_group").groups.items():
        s = safe_float_series(pred_df.loc[idx, score_col])
        pred_df.loc[idx, rank_col] = s.rank(method="average", pct=True)

    cutoff = 1.0 - RANK_TRUE_FRACTION
    pred_df[pred_col] = np.where(pred_df[rank_col] > cutoff, "TRUE_LEAK", "FALSE_LEAK")

    return pred_df


def select_physics_score_on_train(train_df, gm_score_cols):
    """
    只用训练组选择最靠谱的物理分数，并自动校正方向。
    如果某个分数在训练组里 AUC < 0.5，说明方向反了，预测时乘 -1。
    """
    y = label_to_y(train_df["label"].values)

    best = {
        "feature": None,
        "auc_train": np.nan,
        "auc_free_train": -1,
        "sign": 1,
    }

    for c in gm_score_cols:
        if c not in train_df.columns:
            continue

        s = safe_float_series(train_df[c]).fillna(safe_float_series(train_df[c]).median()).values
        auc = safe_auc(y, s)

        if not np.isfinite(auc):
            continue

        auc_free = max(auc, 1.0 - auc)
        sign = 1 if auc >= 0.5 else -1

        if auc_free > best["auc_free_train"]:
            best = {
                "feature": c,
                "auc_train": auc,
                "auc_free_train": auc_free,
                "sign": sign,
            }

    return best


def validate_experiment(df, feature_cols, gm_score_cols, experiment_name):
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

        X_train, X_test = build_matrix(train_df, test_df, feature_cols)

        oof_prob = get_group_oof_prob(X_train, y_train, train_df["time"].astype(str).values)
        t_best, train_oof_score = best_threshold(y_train, oof_prob)

        clf = build_classifier()
        clf.fit(X_train, y_train)

        prob = clf.predict_proba(X_test)[:, 1]

        default_pred = (prob >= 0.5).astype(int)
        model_pred = (prob >= t_best).astype(int)

        auc = safe_auc(y_test, prob)
        m_default = metrics(y_test, default_pred)
        m_model = metrics(y_test, model_pred)

        # 物理分数自动选择 + 自动校正方向
        selected = select_physics_score_on_train(train_df, gm_score_cols)

        if selected["feature"] is not None:
            raw_test = safe_float_series(test_df[selected["feature"]])
            raw_test = raw_test.fillna(safe_float_series(train_df[selected["feature"]]).median())
            physics_score = selected["sign"] * raw_test.values.astype(float)
        else:
            physics_score = prob
            selected = {
                "feature": "prob_TRUE_LEAK",
                "auc_train": np.nan,
                "auc_free_train": np.nan,
                "sign": 1,
            }

        pred_df = pd.DataFrame({
            "experiment": experiment_name,
            "test_group": test_group,
            "time": test_df["time"].astype(str).values,
            "center": test_df["center"].values if "center" in test_df.columns else test_df["center_norm"].values,
            "center_norm": test_df["center_norm"].astype(str).values,
            "true_label": test_df["label"].astype(str).values,
            "prob_TRUE_LEAK": prob,
            "best_threshold": t_best,
            "default_pred": [y_to_label(x) for x in default_pred],
            "model_pred": [y_to_label(x) for x in model_pred],
            "selected_physics_feature": selected["feature"],
            "selected_physics_train_auc": selected["auc_train"],
            "selected_physics_train_auc_free": selected["auc_free_train"],
            "selected_physics_sign": selected["sign"],
            "selected_physics_score": physics_score,
        })

        pred_df = add_rank_within_group(pred_df, "prob_TRUE_LEAK", "prob")
        pred_df = add_rank_within_group(pred_df, "selected_physics_score", "physics")

        # 融合：模型概率rank + 校正后的物理rank
        # 注意这里不再让物理分数直接覆盖一切。
        pred_df["ensemble_score"] = (
            0.55 * safe_float_series(pred_df["prob_rank_pct"]).values
            + 0.45 * safe_float_series(pred_df["physics_rank_pct"]).values
        )

        pred_df = add_rank_within_group(pred_df, "ensemble_score", "ensemble")

        pred_df["v8_fixed_final_pred"] = pred_df["ensemble_rank_pred"]

        y_prob_rank = label_to_y(pred_df["prob_rank_pred"].values)
        y_physics_rank = label_to_y(pred_df["physics_rank_pred"].values)
        y_ensemble = label_to_y(pred_df["ensemble_rank_pred"].values)
        y_final = label_to_y(pred_df["v8_fixed_final_pred"].values)

        m_prob_rank = metrics(y_test, y_prob_rank)
        m_physics_rank = metrics(y_test, y_physics_rank)
        m_ensemble = metrics(y_test, y_ensemble)
        m_final = metrics(y_test, y_final)

        pred_df["model_correct"] = (pred_df["model_pred"] == pred_df["true_label"]).astype(int)
        pred_df["prob_rank_correct"] = (pred_df["prob_rank_pred"] == pred_df["true_label"]).astype(int)
        pred_df["physics_rank_correct"] = (pred_df["physics_rank_pred"] == pred_df["true_label"]).astype(int)
        pred_df["ensemble_correct"] = (pred_df["ensemble_rank_pred"] == pred_df["true_label"]).astype(int)
        pred_df["v8_fixed_final_correct"] = (pred_df["v8_fixed_final_pred"] == pred_df["true_label"]).astype(int)

        pred_rows.extend(pred_df.to_dict(orient="records"))

        summary_rows.append({
            "experiment": experiment_name,
            "test_group": test_group,
            "n_test": len(test_df),
            "n_true": int(np.sum(y_test == 1)),
            "n_false": int(np.sum(y_test == 0)),
            "n_features": len(feature_cols),
            "auc": auc,
            "best_threshold": t_best,
            "train_oof_score": train_oof_score,
            "selected_physics_feature": selected["feature"],
            "selected_physics_train_auc": selected["auc_train"],
            "selected_physics_train_auc_free": selected["auc_free_train"],
            "selected_physics_sign": selected["sign"],

            "default_acc": m_default["accuracy"],
            "model_acc": m_model["accuracy"],
            "prob_rank_acc": m_prob_rank["accuracy"],
            "physics_rank_acc": m_physics_rank["accuracy"],
            "ensemble_rank_acc": m_ensemble["accuracy"],
            "final_acc": m_final["accuracy"],

            "model_recall_TRUE": m_model["recall_TRUE"],
            "model_recall_FALSE": m_model["recall_FALSE"],
            "physics_recall_TRUE": m_physics_rank["recall_TRUE"],
            "physics_recall_FALSE": m_physics_rank["recall_FALSE"],
            "final_recall_TRUE": m_final["recall_TRUE"],
            "final_recall_FALSE": m_final["recall_FALSE"],
        })

        print(
            f"{test_group}: "
            f"AUC={auc:.3f}, "
            f"default={m_default['accuracy']:.3f}, "
            f"model={m_model['accuracy']:.3f}, "
            f"prob_rank={m_prob_rank['accuracy']:.3f}, "
            f"physics_rank={m_physics_rank['accuracy']:.3f}, "
            f"ensemble={m_ensemble['accuracy']:.3f}, "
            f"final={m_final['accuracy']:.3f}, "
            f"physics={selected['feature']}, sign={selected['sign']}"
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
# 9. 144226 诊断
# ============================================================

def compare_144226_features(df, gm_score_cols):
    sub = df[df["time"].astype(str) == TARGET_TIME].copy()
    rows = []

    for c in gm_score_cols:
        if c not in sub.columns:
            continue

        tv = safe_float_series(sub.loc[sub["label"] == "TRUE_LEAK", c]).dropna().values
        fv = safe_float_series(sub.loc[sub["label"] == "FALSE_LEAK", c]).dropna().values

        if len(tv) < 2 or len(fv) < 2:
            continue

        y = np.concatenate([np.ones(len(tv)), np.zeros(len(fv))])
        s = np.concatenate([tv, fv])

        auc = safe_auc(y, s)
        d = cohen_d(tv, fv)

        rows.append({
            "feature": c,
            "true_mean": float(np.mean(tv)),
            "false_mean": float(np.mean(fv)),
            "diff_TRUE_minus_FALSE": float(np.mean(tv) - np.mean(fv)),
            "cohen_d": d,
            "abs_cohen_d": abs(d) if np.isfinite(d) else np.nan,
            "auc_signed_TRUE_larger": auc,
            "auc_direction_free": max(auc, 1 - auc) if np.isfinite(auc) else np.nan,
        })

    out = pd.DataFrame(rows)

    if len(out):
        out = out.sort_values(["auc_direction_free", "abs_cohen_d"], ascending=[False, False])

    path = os.path.join(OUTPUT_DIR, "v8_fixed_144226_global_matrix_feature_compare.csv")
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

        rows.append({
            "center_norm": center,

            "true_prob": tr["prob_TRUE_LEAK"],
            "false_prob": fa["prob_TRUE_LEAK"],
            "prob_diff_TRUE_minus_FALSE": tr["prob_TRUE_LEAK"] - fa["prob_TRUE_LEAK"],
            "prob_order_correct": int(tr["prob_TRUE_LEAK"] > fa["prob_TRUE_LEAK"]),

            "true_physics_score": tr["selected_physics_score"],
            "false_physics_score": fa["selected_physics_score"],
            "physics_score_diff_TRUE_minus_FALSE": tr["selected_physics_score"] - fa["selected_physics_score"],
            "physics_order_correct": int(tr["selected_physics_score"] > fa["selected_physics_score"]),

            "true_ensemble_score": tr["ensemble_score"],
            "false_ensemble_score": fa["ensemble_score"],
            "ensemble_score_diff_TRUE_minus_FALSE": tr["ensemble_score"] - fa["ensemble_score"],
            "ensemble_order_correct": int(tr["ensemble_score"] > fa["ensemble_score"]),

            "selected_physics_feature": tr["selected_physics_feature"],
            "selected_physics_sign": tr["selected_physics_sign"],

            "true_final_pred": tr["v8_fixed_final_pred"],
            "false_final_pred": fa["v8_fixed_final_pred"],
        })

    out = pd.DataFrame(rows)

    if len(out):
        out = out.sort_values("center_norm")

    path = os.path.join(OUTPUT_DIR, "v8_fixed_144226_pair_check.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")

    return out, path


# ============================================================
# 10. 报告
# ============================================================

def plot_results(results):
    fig_dir = os.path.join(OUTPUT_DIR, "figures")
    ensure_dir(fig_dir)

    rows = []
    for r in results:
        rows.extend(r["summary"].to_dict(orient="records"))

    df = pd.DataFrame(rows)
    paths = []

    if len(df) == 0:
        return paths

    for metric in ["auc", "model_acc", "physics_rank_acc", "ensemble_rank_acc", "final_acc"]:
        plt.figure(figsize=(12, 5))

        experiments = df["experiment"].unique().tolist()
        groups = sorted(df["test_group"].unique().tolist())

        x = np.arange(len(groups))
        width = 0.18

        for i, exp in enumerate(experiments):
            vals = []
            for g in groups:
                sub = df[(df["experiment"] == exp) & (df["test_group"] == g)]
                vals.append(float(sub[metric].iloc[0]) if len(sub) else np.nan)

            plt.bar(x + (i - (len(experiments) - 1) / 2) * width, vals, width, label=exp)

        plt.xticks(x, groups, rotation=45, ha="right")
        plt.ylim(0, 1.05)
        plt.ylabel(metric)
        plt.title(f"v8 fixed global matrix: {metric}")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()

        path = os.path.join(fig_dir, f"v8_fixed_{metric}_comparison.png")
        plt.savefig(path, dpi=150)
        plt.close()

        paths.append(path)

    return paths


def make_report(df, feature_sets, results, compare_df, compare_path, pair_df, pair_path, plot_paths):
    lines = []

    lines.append("v8 修正版：全局方向-频段矩阵 directed wideband 报告")
    lines.append("=" * 120)
    lines.append(f"生成时间: {datetime.now()}")
    lines.append("")
    lines.append("本版修正点:")
    lines.append("  1. 强宽频按 8方向×5频段 全局最大能量阈值判断。")
    lines.append("  2. 不再按每个方向内部归一化判断宽频，避免把弥散假泄漏误认为宽频。")
    lines.append("  3. 自动用训练组选择物理分数并校正方向；如果分数反向，则乘 -1。")
    lines.append("  4. final 使用 模型prob rank + 校正物理rank 的融合，不再让物理分数直接覆盖一切。")
    lines.append("")
    lines.append("复用/缓存:")
    lines.append(f"  旧v8特征: {OLD_FEATURE_WITH_RELATIVE}")
    lines.append(f"  新全局矩阵缓存: {GLOBAL_MATRIX_CACHE}")
    lines.append(f"  修正合并数据集: {FIXED_DATASET_CSV}")
    lines.append("")

    lines.append("样本统计:")
    lines.append(str(df["label"].value_counts()))
    lines.append("")

    lines.append("特征集:")
    for k, v in feature_sets.items():
        lines.append(f"  {k}: {len(v)} features")
    lines.append("")

    overall_rows = []

    lines.append("实验结果:")
    lines.append("-" * 120)

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
                f"平均ensemble_acc={s['ensemble_rank_acc'].mean():.4f}, "
                f"平均final_acc={s['final_acc'].mean():.4f}"
            )

            target = s[s["test_group"] == TARGET_TIME]
            if len(target):
                tr = target.iloc[0]
                lines.append(
                    f"  {TARGET_TIME}: "
                    f"AUC={tr['auc']:.4f}, "
                    f"model_acc={tr['model_acc']:.4f}, "
                    f"physics_rank_acc={tr['physics_rank_acc']:.4f}, "
                    f"ensemble_acc={tr['ensemble_rank_acc']:.4f}, "
                    f"final_acc={tr['final_acc']:.4f}, "
                    f"physics={tr['selected_physics_feature']}, "
                    f"sign={tr['selected_physics_sign']}"
                )

            lines.append("  各组:")
            for _, row in s.iterrows():
                lines.append(
                    f"    {row['test_group']}: "
                    f"AUC={row['auc']:.3f}, "
                    f"default={row['default_acc']:.3f}, "
                    f"model={row['model_acc']:.3f}, "
                    f"prob_rank={row['prob_rank_acc']:.3f}, "
                    f"physics_rank={row['physics_rank_acc']:.3f}, "
                    f"ensemble={row['ensemble_rank_acc']:.3f}, "
                    f"final={row['final_acc']:.3f}, "
                    f"physics={row['selected_physics_feature']}, "
                    f"sign={row['selected_physics_sign']}"
                )

            item = {
                "experiment": exp,
                "mean_auc": s["auc"].mean(),
                "mean_model_acc": s["model_acc"].mean(),
                "mean_physics_rank_acc": s["physics_rank_acc"].mean(),
                "mean_ensemble_acc": s["ensemble_rank_acc"].mean(),
                "mean_final_acc": s["final_acc"].mean(),
            }

            if len(target):
                tr = target.iloc[0]
                item.update({
                    "auc_144226": tr["auc"],
                    "model_acc_144226": tr["model_acc"],
                    "physics_rank_acc_144226": tr["physics_rank_acc"],
                    "ensemble_acc_144226": tr["ensemble_rank_acc"],
                    "final_acc_144226": tr["final_acc"],
                    "selected_physics_feature_144226": tr["selected_physics_feature"],
                    "selected_physics_sign_144226": tr["selected_physics_sign"],
                })

            overall_rows.append(item)

    overall = pd.DataFrame(overall_rows)
    overall_path = os.path.join(OUTPUT_DIR, "v8_fixed_group_validation_summary.csv")
    overall.to_csv(overall_path, index=False, encoding="utf-8-sig")

    lines.append("")
    lines.append("总汇总:")
    lines.append(f"  {overall_path}")

    lines.append("")
    lines.append("144226 全局矩阵物理特征对比:")
    lines.append(f"  {compare_path}")

    if len(compare_df):
        lines.append("  区分力前15:")
        for _, row in compare_df.head(15).iterrows():
            lines.append(
                f"    {row['feature']}: "
                f"AUC={row['auc_direction_free']:.3f}, "
                f"signedAUC={row['auc_signed_TRUE_larger']:.3f}, "
                f"d={row['cohen_d']:.3f}, "
                f"TRUE_mean={row['true_mean']:.6g}, "
                f"FALSE_mean={row['false_mean']:.6g}"
            )

    lines.append("")
    lines.append("144226 center配对检查:")
    lines.append(f"  {pair_path}")

    if len(pair_df):
        lines.append(f"  prob排序正确率: {pair_df['prob_order_correct'].mean():.4f}")
        lines.append(f"  physics排序正确率: {pair_df['physics_order_correct'].mean():.4f}")
        lines.append(f"  ensemble排序正确率: {pair_df['ensemble_order_correct'].mean():.4f}")

        failed = pair_df[pair_df["ensemble_order_correct"] == 0]["center_norm"].astype(str).tolist()
        lines.append(f"  ensemble排序失败center: {' | '.join(failed) if failed else '无'}")

    lines.append("")
    lines.append("图:")
    for p in plot_paths:
        lines.append(f"  {p}")

    report_path = os.path.join(OUTPUT_DIR, "v8_fixed_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path, overall_path


# ============================================================
# 11. 主程序
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)

    print("=" * 120)
    print("v8 修正版：全局方向-频段矩阵 directed wideband")
    print("=" * 120)

    df = prepare_dataset()

    print("\n合并后样本数:", len(df))
    print(df["label"].value_counts())
    print("time groups:", sorted(df["time"].astype(str).unique().tolist()))

    feature_sets, gm_score_cols = build_feature_sets(df)

    with open(os.path.join(OUTPUT_DIR, "v8_fixed_feature_sets.json"), "w", encoding="utf-8") as f:
        json.dump(feature_sets, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUTPUT_DIR, "v8_fixed_gm_score_candidates.json"), "w", encoding="utf-8") as f:
        json.dump(gm_score_cols, f, ensure_ascii=False, indent=2)

    print("\n特征集:")
    for k, v in feature_sets.items():
        print(f"  {k}: {len(v)}")

    print("\n候选物理分数数量:", len(gm_score_cols))

    results = []
    for exp, cols in feature_sets.items():
        results.append(validate_experiment(df, cols, gm_score_cols, exp))

    all_preds = pd.concat([r["preds"] for r in results], ignore_index=True)
    pred_path = os.path.join(OUTPUT_DIR, "v8_fixed_predictions.csv")
    all_preds.to_csv(pred_path, index=False, encoding="utf-8-sig")

    # 144226 诊断看最终 C
    c_preds = results[-1]["preds"]
    pair_df, pair_path = pair_check_144226(c_preds)

    compare_df, compare_path = compare_144226_features(df, gm_score_cols)

    plot_paths = plot_results(results)

    report_path, overall_path = make_report(
        df=df,
        feature_sets=feature_sets,
        results=results,
        compare_df=compare_df,
        compare_path=compare_path,
        pair_df=pair_df,
        pair_path=pair_path,
        plot_paths=plot_paths,
    )

    print("\n" + "=" * 120)
    print("v8 修正版完成")
    print("=" * 120)
    print("输出文件夹:", OUTPUT_DIR)
    print("报告:", report_path)
    print("修正合并数据集:", FIXED_DATASET_CSV)
    print("全局矩阵特征:", GLOBAL_MATRIX_CACHE)
    print("预测明细:", pred_path)
    print("分组汇总:", overall_path)
    print("144226配对检查:", pair_path)
    print("144226物理特征对比:", compare_path)

    print("\n核心结果摘要:")
    for r in results:
        s = r["summary"]
        print(f"\n{r['experiment']}:")
        print(
            f"  平均AUC={s['auc'].mean():.3f}, "
            f"平均model_acc={s['model_acc'].mean():.3f}, "
            f"平均physics_rank_acc={s['physics_rank_acc'].mean():.3f}, "
            f"平均ensemble_acc={s['ensemble_rank_acc'].mean():.3f}, "
            f"平均final_acc={s['final_acc'].mean():.3f}"
        )

        target = s[s["test_group"] == TARGET_TIME]
        if len(target):
            tr = target.iloc[0]
            print(
                f"  144226: "
                f"AUC={tr['auc']:.3f}, "
                f"model_acc={tr['model_acc']:.3f}, "
                f"physics_rank_acc={tr['physics_rank_acc']:.3f}, "
                f"ensemble_acc={tr['ensemble_rank_acc']:.3f}, "
                f"final_acc={tr['final_acc']:.3f}, "
                f"physics={tr['selected_physics_feature']}, "
                f"sign={tr['selected_physics_sign']}"
            )

    if len(pair_df):
        print("\n144226 center配对检查:")
        print(f"  prob排序正确率: {pair_df['prob_order_correct'].mean():.3f}")
        print(f"  physics排序正确率: {pair_df['physics_order_correct'].mean():.3f}")
        print(f"  ensemble排序正确率: {pair_df['ensemble_order_correct'].mean():.3f}")
        failed = pair_df[pair_df["ensemble_order_correct"] == 0]["center_norm"].astype(str).tolist()
        print("  ensemble排序失败center:", " | ".join(failed) if failed else "无")

    print("\n把上面的“核心结果摘要”和“144226 center配对检查”发给我即可。")


if __name__ == "__main__":
    main()
