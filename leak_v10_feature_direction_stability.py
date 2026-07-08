# -*- coding: utf-8 -*-
"""
leak_v10_feature_direction_stability.py

v10：特征方向稳定性分析 + 稳定特征筛选分类

为什么做 v10？
    v9 诊断发现：
        HM20260626_144226.ld 内部并不是完全没有可分信息。
        例如 best_direction_combined_score、direction_contrast、spec_slope、heatmap集中度等特征，
        在 144226 内部可能很能区分 TRUE/FALSE。

    但模型仍然分错，说明问题很可能是：
        某些特征在不同 time_folder 中 TRUE/FALSE 方向不一致。

    举例：
        某个特征在大多数时间点：
            TRUE_LEAK > FALSE_LEAK
        但在 144226：
            TRUE_LEAK < FALSE_LEAK
        这种特征就会导致跨时间点泛化失败。

v10 做什么？
    1. 读取 v8.1 的 v7_only 特征表和 heatmap 核心形态特征表；
    2. 对每个特征，在每个 time_folder 内计算：
        - TRUE 均值
        - FALSE 均值
        - TRUE-FALSE 差值
        - Cohen's d
        - 单特征方向无关 AUC
        - 方向：TRUE>FALSE / TRUE<FALSE / weak
    3. 输出全局特征方向稳定性报告；
    4. 做严格 leave-one-time-folder-out 验证：
        - 每次只用训练时间点判断哪些特征方向稳定；
        - 只用这些稳定特征训练；
        - 测试留出的整个 time_folder；
    5. 对比：
        A_all_v7_baseline
        B_stable_v7_only
        C_stable_heatmap_only
        D_stable_v7_plus_heatmap
    6. 重点观察 144226 是否改善。

运行：
    python leak_v10_feature_direction_stability.py

输入默认：
    C:\\Users\\jiangxinru6\\Desktop\\wurenji\\leak_v8_1_heatmap_shape_ablation_results\\v8_1_features_A_v7_only.csv
    C:\\Users\\jiangxinru6\\Desktop\\wurenji\\leak_v8_1_heatmap_shape_ablation_results\\v8_1_heatmap_core_shape_features.csv

输出：
    C:\\Users\\jiangxinru6\\Desktop\\wurenji\\leak_v10_feature_direction_stability_results
"""

import os
import math
import json
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 路径配置
# ============================================================

BASE_DIR = r"C:\Users\jiangxinru6\Desktop\wurenji"

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "leak_v10_feature_direction_stability_results"
)

# v8.1 里的 A_v7_only 特征表，即当前最好的 v7 baseline 特征表
V7_FEATURE_CSV = os.path.join(
    BASE_DIR,
    "leak_v8_1_heatmap_shape_ablation_results",
    "v8_1_features_A_v7_only.csv"
)

# v8.1 提取的 heatmap 核心形态特征表
HEATMAP_FEATURE_CSV = os.path.join(
    BASE_DIR,
    "leak_v8_1_heatmap_shape_ablation_results",
    "v8_1_heatmap_core_shape_features.csv"
)

# 备用：如果 v8.1 文件不存在，尝试 v7 输出
V7_FEATURE_CSV_FALLBACK = os.path.join(
    BASE_DIR,
    "leak_v7_robust_feature_results",
    "v7_robust_feature_dataset.csv"
)

GROUP_COL = "time"
LABEL_COL = "label"

TARGET_TIME = "HM20260626_144226.ld"

META_COLS = [
    "dataset",
    "label",
    "true_label",
    "time",
    "test_group",
    "center",
    "center_norm",
    "experiment",
    "heatmap_path",
    "row_index",
    "best_direction",
    "energy_direction",
    "decay_direction",
    "representative_file",
]

# 稳定性判断参数
# 一个特征在某个 time 内，只有达到这个强度才认为“方向有效”，否则算 weak
MIN_ABS_D_FOR_EFFECT = 0.30
MIN_AUC_FOR_EFFECT = 0.60

# 全局报告中，至少多少个 time 方向有效，才有资格叫稳定候选
MIN_GLOBAL_EFFECT_GROUPS = 3

# 训练时，如果训练集有3个time，至少2个time方向有效；
# 如果训练集有更多，按 2/3 比例。
MIN_EFFECT_GROUP_RATIO_FOR_TRAIN = 0.67

# 稳定特征数量控制
MAX_STABLE_FEATURES_PER_EXPERIMENT = 40
MIN_STABLE_FEATURES_TO_USE = 5

# 如果稳定特征太少，是否用“无强冲突且平均AUC较高”的特征补足
ENABLE_FALLBACK_TOP_FEATURES = True

# 阈值搜索
THRESHOLD_GRID = np.linspace(0.01, 0.99, 99)
THRESHOLD_METRIC = "balanced_accuracy"

# 排名规则：每个 time 内 top 50% 判 TRUE，用于验证
RANK_TRUE_FRACTION_FOR_BINARY = 0.50

# 三档决策
RANK_TRUE_LIKE_PCT = 0.70
RANK_FALSE_LIKE_PCT = 0.30

RANDOM_STATE = 42


# ============================================================
# 2. 基础工具
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def normalize_center_id(center):
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


def safe_name(s):
    return str(s).replace("\\", "_").replace("/", "_").replace(":", "_").replace(".", "_")


def safe_filename(s):
    keep = []
    for ch in str(s):
        if ch.isalnum() or ch in ["_", "-", "."]:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)[:180]


def safe_float_array(s):
    arr = pd.to_numeric(s, errors="coerce")
    arr = arr.replace([np.inf, -np.inf], np.nan)
    return arr


def label_to_binary(labels):
    return np.array([1 if str(x) == "TRUE_LEAK" else 0 for x in labels], dtype=int)


def binary_to_label(v):
    return "TRUE_LEAK" if int(v) == 1 else "FALSE_LEAK"


def find_existing_path(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


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
            **m,
        })

        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t, float(best_score), pd.DataFrame(rows)


def cohen_d(true_vals, false_vals):
    true_vals = np.asarray(true_vals, dtype=float)
    false_vals = np.asarray(false_vals, dtype=float)

    true_vals = true_vals[np.isfinite(true_vals)]
    false_vals = false_vals[np.isfinite(false_vals)]

    if len(true_vals) < 2 or len(false_vals) < 2:
        return np.nan

    m1 = np.mean(true_vals)
    m0 = np.mean(false_vals)
    s1 = np.std(true_vals, ddof=1)
    s0 = np.std(false_vals, ddof=1)

    pooled = math.sqrt((s1 ** 2 + s0 ** 2) / 2.0) + 1e-12
    return float((m1 - m0) / pooled)


def single_feature_auc_signed(true_vals, false_vals):
    """
    TRUE=1, FALSE=0，直接按特征值算 AUC。
    如果 AUC > 0.5，说明特征越大越像 TRUE。
    如果 AUC < 0.5，说明特征越小越像 TRUE。
    """
    try:
        from sklearn.metrics import roc_auc_score

        true_vals = np.asarray(true_vals, dtype=float)
        false_vals = np.asarray(false_vals, dtype=float)

        true_vals = true_vals[np.isfinite(true_vals)]
        false_vals = false_vals[np.isfinite(false_vals)]

        if len(true_vals) == 0 or len(false_vals) == 0:
            return np.nan

        y = np.concatenate([
            np.ones(len(true_vals), dtype=int),
            np.zeros(len(false_vals), dtype=int),
        ])
        x = np.concatenate([true_vals, false_vals])

        return float(roc_auc_score(y, x))
    except Exception:
        return np.nan


def histogram_overlap(true_vals, false_vals, bins=20):
    true_vals = np.asarray(true_vals, dtype=float)
    false_vals = np.asarray(false_vals, dtype=float)

    true_vals = true_vals[np.isfinite(true_vals)]
    false_vals = false_vals[np.isfinite(false_vals)]

    if len(true_vals) == 0 or len(false_vals) == 0:
        return np.nan

    all_vals = np.concatenate([true_vals, false_vals])
    lo = np.min(all_vals)
    hi = np.max(all_vals)

    if abs(hi - lo) < 1e-12:
        return 1.0

    hist_t, edges = np.histogram(true_vals, bins=bins, range=(lo, hi), density=True)
    hist_f, _ = np.histogram(false_vals, bins=bins, range=(lo, hi), density=True)

    bin_width = edges[1] - edges[0]
    overlap = np.sum(np.minimum(hist_t, hist_f)) * bin_width

    return float(np.clip(overlap, 0.0, 1.0))


# ============================================================
# 3. 数据读取与合并
# ============================================================

def numeric_feature_columns(df):
    cols = []

    for c in df.columns:
        if c in META_COLS:
            continue

        vals = safe_float_array(df[c])
        if vals.notna().mean() > 0.8:
            cols.append(c)

    return cols


def load_v7_feature_table():
    path = find_existing_path(V7_FEATURE_CSV, V7_FEATURE_CSV_FALLBACK)

    if path is None:
        raise FileNotFoundError(
            "找不到 v7 特征表。请先运行 v8.1 或 v7。\n"
            f"优先路径: {V7_FEATURE_CSV}\n"
            f"备用路径: {V7_FEATURE_CSV_FALLBACK}"
        )

    df = pd.read_csv(path)

    if LABEL_COL not in df.columns or GROUP_COL not in df.columns or "center" not in df.columns:
        raise ValueError("v7特征表缺少 label/time/center 列。")

    df[LABEL_COL] = df[LABEL_COL].astype(str)
    df = df[df[LABEL_COL].isin(["TRUE_LEAK", "FALSE_LEAK"])].copy()
    df["center_norm"] = df["center"].apply(normalize_center_id)
    df = df.reset_index(drop=True)

    return df, path


def load_heatmap_feature_table():
    if not os.path.exists(HEATMAP_FEATURE_CSV):
        return None, None

    hm = pd.read_csv(HEATMAP_FEATURE_CSV)

    if LABEL_COL not in hm.columns or GROUP_COL not in hm.columns or "center" not in hm.columns:
        return None, None

    hm[LABEL_COL] = hm[LABEL_COL].astype(str)
    hm = hm[hm[LABEL_COL].isin(["TRUE_LEAK", "FALSE_LEAK"])].copy()
    hm["center_norm"] = hm["center"].apply(normalize_center_id)
    hm = hm.reset_index(drop=True)

    return hm, HEATMAP_FEATURE_CSV


def build_combined_table(v7_df, hm_df):
    """
    以 v7_df 为主表，按 label/time/center_norm 合并 heatmap 特征。
    """
    v7 = v7_df.copy()

    v7_cols = numeric_feature_columns(v7)
    v7_feature_cols = [c for c in v7_cols if not c.startswith("hm_")]

    if hm_df is None:
        combo = v7.copy()
        hm_feature_cols = []
        return combo, v7_feature_cols, hm_feature_cols

    hm = hm_df.copy()
    hm_cols = numeric_feature_columns(hm)
    hm_feature_cols = [
        c for c in hm_cols
        if c.startswith("hm_") and c != "hm_read_success"
    ]

    merge_cols = [LABEL_COL, GROUP_COL, "center_norm"]

    hm_small = hm[merge_cols + hm_feature_cols].copy()

    # 防止列名冲突
    for c in hm_feature_cols:
        if c in v7.columns:
            hm_small = hm_small.rename(columns={c: f"{c}__hm"})

    combo = v7.merge(
        hm_small,
        on=merge_cols,
        how="left",
        validate="one_to_one"
    )

    # 更新合并后的 hm 列名
    actual_hm_feature_cols = [
        c if c in combo.columns else f"{c}__hm"
        for c in hm_feature_cols
        if (c in combo.columns or f"{c}__hm" in combo.columns)
    ]

    return combo, v7_feature_cols, actual_hm_feature_cols


# ============================================================
# 4. 特征方向稳定性分析
# ============================================================

def group_feature_stats(df, feature, group_value):
    sub = df[df[GROUP_COL].astype(str) == str(group_value)].copy()

    label = sub[LABEL_COL].astype(str)
    vals = safe_float_array(sub[feature])

    true_vals = vals[label == "TRUE_LEAK"].dropna().values
    false_vals = vals[label == "FALSE_LEAK"].dropna().values

    if len(true_vals) == 0 or len(false_vals) == 0:
        return None

    true_mean = float(np.mean(true_vals))
    false_mean = float(np.mean(false_vals))
    diff = true_mean - false_mean

    d = cohen_d(true_vals, false_vals)
    auc_signed = single_feature_auc_signed(true_vals, false_vals)

    if np.isfinite(auc_signed):
        auc_dir_free = max(auc_signed, 1.0 - auc_signed)
    else:
        auc_dir_free = np.nan

    overlap = histogram_overlap(true_vals, false_vals)

    # 方向判断
    if not np.isfinite(d) or not np.isfinite(auc_dir_free):
        direction = "weak"
        sign = 0
        has_effect = False
    else:
        has_effect = (abs(d) >= MIN_ABS_D_FOR_EFFECT) or (auc_dir_free >= MIN_AUC_FOR_EFFECT)

        if not has_effect:
            direction = "weak"
            sign = 0
        elif diff > 0:
            direction = "TRUE>FALSE"
            sign = 1
        elif diff < 0:
            direction = "TRUE<FALSE"
            sign = -1
        else:
            direction = "weak"
            sign = 0
            has_effect = False

    return {
        "group": group_value,
        "n_true": len(true_vals),
        "n_false": len(false_vals),
        "true_mean": true_mean,
        "false_mean": false_mean,
        "diff_true_minus_false": diff,
        "cohen_d": d,
        "abs_cohen_d": abs(d) if np.isfinite(d) else np.nan,
        "auc_signed_TRUE_large_positive": auc_signed,
        "auc_direction_free": auc_dir_free,
        "hist_overlap_high_is_bad": overlap,
        "direction": direction,
        "sign": sign,
        "has_effect": bool(has_effect),
    }


def analyze_feature_stability(df, feature_cols, groups=None, min_effect_groups=None):
    """
    对给定特征列表，在指定 groups 上做方向稳定性分析。
    返回：
        report_df: 每个特征一行
        detail_df: 每个特征×每个group一行
        sign_map: 稳定特征方向，+1 表示值大更像 TRUE，-1 表示值小更像 TRUE
    """
    if groups is None:
        groups = sorted(pd.unique(df[GROUP_COL].astype(str)).tolist())
    else:
        groups = [str(g) for g in groups]

    if min_effect_groups is None:
        if len(groups) >= 4:
            min_effect_groups = MIN_GLOBAL_EFFECT_GROUPS
        else:
            min_effect_groups = max(2, int(math.ceil(len(groups) * MIN_EFFECT_GROUP_RATIO_FOR_TRAIN)))

    report_rows = []
    detail_rows = []
    sign_map = {}

    for feature in feature_cols:
        if feature not in df.columns:
            continue

        stats = []

        for g in groups:
            st = group_feature_stats(df, feature, g)
            if st is None:
                continue

            stats.append(st)

            detail_row = {
                "feature": feature,
                **st,
            }
            detail_rows.append(detail_row)

        if len(stats) == 0:
            continue

        signs = [s["sign"] for s in stats]
        effect_signs = [s["sign"] for s in stats if s["has_effect"] and s["sign"] != 0]

        pos_effect = sum(1 for s in effect_signs if s > 0)
        neg_effect = sum(1 for s in effect_signs if s < 0)
        weak_groups = len(stats) - len(effect_signs)

        n_effect = len(effect_signs)
        n_groups = len(stats)

        has_strong_conflict = (pos_effect > 0 and neg_effect > 0)

        if pos_effect > neg_effect:
            majority_sign = 1
            majority_direction = "TRUE>FALSE"
            majority_count = pos_effect
            opposite_count = neg_effect
        elif neg_effect > pos_effect:
            majority_sign = -1
            majority_direction = "TRUE<FALSE"
            majority_count = neg_effect
            opposite_count = pos_effect
        else:
            majority_sign = 0
            majority_direction = "none"
            majority_count = 0
            opposite_count = 0

        mean_abs_d = np.nanmean([s["abs_cohen_d"] for s in stats])
        mean_auc_dir_free = np.nanmean([s["auc_direction_free"] for s in stats])
        mean_overlap = np.nanmean([s["hist_overlap_high_is_bad"] for s in stats])

        min_auc_dir_free = np.nanmin([s["auc_direction_free"] for s in stats])
        max_auc_dir_free = np.nanmax([s["auc_direction_free"] for s in stats])

        is_direction_stable = (
            (not has_strong_conflict) and
            (majority_sign != 0) and
            (majority_count >= min_effect_groups)
        )

        # 分数：方向稳定优先，其次平均AUC/效应量，惩罚冲突和重叠
        stability_score = (
            1.00 * (majority_count / max(1, n_groups)) +
            0.80 * (mean_auc_dir_free - 0.5) * 2.0 +
            0.30 * min(mean_abs_d / 2.0, 2.0) -
            0.80 * opposite_count -
            0.30 * mean_overlap
        )

        row = {
            "feature": feature,
            "n_groups": n_groups,
            "min_effect_groups_required": min_effect_groups,
            "n_effect_groups": n_effect,
            "pos_effect_groups_TRUE_gt_FALSE": pos_effect,
            "neg_effect_groups_TRUE_lt_FALSE": neg_effect,
            "weak_groups": weak_groups,
            "majority_direction": majority_direction,
            "majority_sign": majority_sign,
            "majority_count": majority_count,
            "opposite_count": opposite_count,
            "has_strong_conflict": int(has_strong_conflict),
            "is_direction_stable": int(is_direction_stable),
            "mean_abs_cohen_d": mean_abs_d,
            "mean_auc_direction_free": mean_auc_dir_free,
            "min_auc_direction_free": min_auc_dir_free,
            "max_auc_direction_free": max_auc_dir_free,
            "mean_overlap_high_is_bad": mean_overlap,
            "stability_score": stability_score,
            "sign_sequence": " | ".join([f"{s['group']}:{s['direction']}" for s in stats]),
        }

        # 每个 group 的详细字段也放到主表，方便看方向翻转
        for s in stats:
            prefix = safe_name(s["group"])
            row[f"{prefix}_direction"] = s["direction"]
            row[f"{prefix}_true_mean"] = s["true_mean"]
            row[f"{prefix}_false_mean"] = s["false_mean"]
            row[f"{prefix}_diff"] = s["diff_true_minus_false"]
            row[f"{prefix}_cohen_d"] = s["cohen_d"]
            row[f"{prefix}_auc_dir_free"] = s["auc_direction_free"]
            row[f"{prefix}_overlap"] = s["hist_overlap_high_is_bad"]

        report_rows.append(row)

        if is_direction_stable:
            sign_map[feature] = majority_sign

    report_df = pd.DataFrame(report_rows)
    detail_df = pd.DataFrame(detail_rows)

    if len(report_df):
        report_df = report_df.sort_values(
            ["is_direction_stable", "stability_score", "mean_auc_direction_free", "mean_abs_cohen_d"],
            ascending=[False, False, False, False]
        )

    return report_df, detail_df, sign_map


def select_stable_features_from_train(train_df, candidate_cols, source_tag, output_dir, test_group):
    """
    只用训练 groups 做稳定特征筛选，避免测试组信息泄露。
    """
    train_groups = sorted(pd.unique(train_df[GROUP_COL].astype(str)).tolist())
    min_effect_groups = max(2, int(math.ceil(len(train_groups) * MIN_EFFECT_GROUP_RATIO_FOR_TRAIN)))

    report_df, detail_df, sign_map = analyze_feature_stability(
        train_df,
        candidate_cols,
        groups=train_groups,
        min_effect_groups=min_effect_groups
    )

    select_dir = os.path.join(output_dir, "nested_selected_features")
    ensure_dir(select_dir)

    report_path = os.path.join(
        select_dir,
        f"{source_tag}_train_without_{safe_name(test_group)}_stability_report.csv"
    )
    detail_path = os.path.join(
        select_dir,
        f"{source_tag}_train_without_{safe_name(test_group)}_stability_detail.csv"
    )

    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    selected_df = report_df[report_df["is_direction_stable"] == 1].copy()

    # 先按稳定分数取前N
    selected_df = selected_df.sort_values("stability_score", ascending=False)
    selected = selected_df["feature"].head(MAX_STABLE_FEATURES_PER_EXPERIMENT).tolist()

    selected_sign_map = {
        f: int(report_df.loc[report_df["feature"] == f, "majority_sign"].iloc[0])
        for f in selected
    }

    # 如果太少，用无冲突 + 平均AUC高的特征补足
    if ENABLE_FALLBACK_TOP_FEATURES and len(selected) < MIN_STABLE_FEATURES_TO_USE and len(report_df):
        fallback_df = report_df[
            (report_df["has_strong_conflict"] == 0) &
            (report_df["majority_sign"] != 0)
        ].copy()

        fallback_df = fallback_df.sort_values("stability_score", ascending=False)

        for _, r in fallback_df.iterrows():
            f = r["feature"]
            if f not in selected:
                selected.append(f)
                selected_sign_map[f] = int(r["majority_sign"])

            if len(selected) >= MIN_STABLE_FEATURES_TO_USE:
                break

    # 仍然为空，则兜底取训练平均AUC最高的少量特征
    if len(selected) == 0 and len(report_df):
        fallback_df = report_df.sort_values("mean_auc_direction_free", ascending=False).head(MIN_STABLE_FEATURES_TO_USE)

        for _, r in fallback_df.iterrows():
            f = r["feature"]
            selected.append(f)
            s = int(r["majority_sign"])
            selected_sign_map[f] = s if s != 0 else 1

    selected_info = pd.DataFrame({
        "feature": selected,
        "align_sign": [selected_sign_map[f] for f in selected],
    })

    selected_path = os.path.join(
        select_dir,
        f"{source_tag}_train_without_{safe_name(test_group)}_selected_features.csv"
    )
    selected_info.to_csv(selected_path, index=False, encoding="utf-8-sig")

    return selected, selected_sign_map, report_path, detail_path, selected_path


# ============================================================
# 5. 特征矩阵构造与模型
# ============================================================

def build_aligned_matrix(df, selected_features, sign_map, train_medians=None):
    """
    对稳定特征做方向对齐：
        如果训练判断该特征 TRUE>FALSE，则保留原值；
        如果训练判断该特征 TRUE<FALSE，则乘以 -1；
    这样所有特征都是“值越大越像 TRUE”。
    """
    X = pd.DataFrame(index=df.index)

    medians = {}

    for f in selected_features:
        vals = safe_float_array(df[f]) if f in df.columns else pd.Series(np.nan, index=df.index)

        if train_medians is None:
            med = vals.median()
            if not np.isfinite(med):
                med = 0.0
            medians[f] = med
        else:
            med = train_medians.get(f, 0.0)

        filled = vals.fillna(med).astype(float)

        sign = int(sign_map.get(f, 1))
        if sign == 0:
            sign = 1

        X[f"{f}__aligned"] = filled * sign

    if train_medians is not None:
        return X, train_medians

    return X, medians


def build_raw_matrix(df, feature_cols, train_medians=None):
    X = pd.DataFrame(index=df.index)
    medians = {}

    for f in feature_cols:
        vals = safe_float_array(df[f]) if f in df.columns else pd.Series(np.nan, index=df.index)

        if train_medians is None:
            med = vals.median()
            if not np.isfinite(med):
                med = 0.0
            medians[f] = med
        else:
            med = train_medians.get(f, 0.0)

        X[f] = vals.fillna(med).astype(float)

    if train_medians is not None:
        return X, train_medians

    return X, medians


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
        if r["model_pred"] == r["rank_binary_pred"]:
            final.append(r["model_pred"])
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


# ============================================================
# 6. 分组验证实验
# ============================================================

def validate_experiment(
    df,
    candidate_cols,
    experiment_name,
    output_dir,
    selection_mode="stable",
    source_tag="all"
):
    """
    selection_mode:
        - "all": 不做稳定筛选，直接用全部 candidate_cols
        - "stable": 每次只用训练组筛选稳定特征，并方向对齐
    """
    exp_dir = os.path.join(output_dir, experiment_name)
    ensure_dir(exp_dir)

    groups = sorted(pd.unique(df[GROUP_COL].astype(str)).tolist())
    y_all = label_to_binary(df[LABEL_COL].astype(str).values)

    group_rows = []
    all_pred_rows = []
    selected_summary_rows = []

    print(f"\n开始实验 {experiment_name} ...")
    print("候选特征数:", len(candidate_cols))
    print("selection_mode:", selection_mode)

    for test_group in groups:
        test_mask = df[GROUP_COL].astype(str).values == str(test_group)
        train_mask = ~test_mask

        train_df = df.loc[train_mask].reset_index(drop=True)
        test_df = df.loc[test_mask].reset_index(drop=True)

        y_train = label_to_binary(train_df[LABEL_COL].astype(str).values)
        y_test = label_to_binary(test_df[LABEL_COL].astype(str).values)

        groups_train = train_df[GROUP_COL].astype(str).values

        if selection_mode == "all":
            selected_features = list(candidate_cols)
            sign_map = {f: 1 for f in selected_features}
            selected_path = ""
            stability_report_path = ""
            stability_detail_path = ""

            X_train, medians = build_raw_matrix(train_df, selected_features)
            X_test, _ = build_raw_matrix(test_df, selected_features, medians)

        else:
            selected_features, sign_map, stability_report_path, stability_detail_path, selected_path = (
                select_stable_features_from_train(
                    train_df,
                    candidate_cols,
                    source_tag=source_tag,
                    output_dir=exp_dir,
                    test_group=test_group
                )
            )

            X_train, medians = build_aligned_matrix(train_df, selected_features, sign_map)
            X_test, _ = build_aligned_matrix(test_df, selected_features, sign_map, medians)

        selected_summary_rows.append({
            "experiment": experiment_name,
            "test_group": test_group,
            "n_selected_features": len(selected_features),
            "selected_features_path": selected_path,
            "stability_report_path": stability_report_path,
            "selected_features": " | ".join(selected_features),
        })

        if len(selected_features) == 0:
            print(f"  {test_group}: 没有可用特征，跳过")
            continue

        # OOF选择阈值
        oof_prob = get_group_oof_probabilities(X_train, y_train, groups_train)
        best_t, best_score, threshold_curve = find_best_threshold(
            y_train,
            oof_prob,
            metric=THRESHOLD_METRIC,
            grid=THRESHOLD_GRID
        )

        curve_csv = os.path.join(
            exp_dir,
            f"{experiment_name}_threshold_curve_train_without_{safe_name(test_group)}.csv"
        )
        threshold_curve.to_csv(curve_csv, index=False, encoding="utf-8-sig")

        clf = build_classifier()
        clf.fit(X_train, y_train)

        prob = clf.predict_proba(X_test)[:, 1]

        default_pred_binary = threshold_predict(prob, 0.5)
        model_pred_binary = threshold_predict(prob, best_t)

        m_default = metrics_from_pred(y_test, default_pred_binary)
        m_model = metrics_from_pred(y_test, model_pred_binary)
        auc = safe_auc(y_test, prob)

        pred_rows = []

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
                "center_norm": test_df.loc[i, "center_norm"] if "center_norm" in test_df.columns else "",
                "true_label": true_label,
                "prob_TRUE_LEAK": float(prob[i]),
                "best_threshold": best_t,
                "default_pred_0p5": default_pred,
                "default_correct": int(default_pred == true_label),
                "model_pred": model_pred,
                "model_correct": int(model_pred == true_label),
                "n_selected_features": len(selected_features),
            }

            pred_rows.append(row)

        pred_df_group = pd.DataFrame(pred_rows)
        pred_df_group = add_probability_rank_columns(pred_df_group)

        rank_pred_binary = label_to_binary(pred_df_group["rank_binary_pred"].values)
        m_rank = metrics_from_pred(y_test, rank_pred_binary)

        final_metrics = calc_final_decision_metrics(pred_df_group)

        print(
            f"  {test_group}: "
            f"selected={len(selected_features)}, "
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
            "n_selected_features": len(selected_features),
            "best_threshold": best_t,
            "train_oof_best_score": best_score,
            "auc": auc,

            "default_accuracy_0p5": m_default["accuracy"],
            "default_balanced_accuracy_0p5": m_default["balanced_accuracy"],

            "model_accuracy": m_model["accuracy"],
            "model_balanced_accuracy": m_model["balanced_accuracy"],
            "model_recall_TRUE_LEAK": m_model["recall_TRUE_LEAK"],
            "model_recall_FALSE_LEAK": m_model["recall_FALSE_LEAK"],
            "model_tp": m_model["tp"],
            "model_tn": m_model["tn"],
            "model_fp": m_model["fp"],
            "model_fn": m_model["fn"],

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

        all_pred_rows.extend(pred_df_group.to_dict(orient="records"))

    group_df = pd.DataFrame(group_rows)
    pred_df = pd.DataFrame(all_pred_rows)
    selected_summary_df = pd.DataFrame(selected_summary_rows)

    group_csv = os.path.join(exp_dir, f"{experiment_name}_group_summary.csv")
    pred_csv = os.path.join(exp_dir, f"{experiment_name}_predictions.csv")
    selected_csv = os.path.join(exp_dir, f"{experiment_name}_selected_feature_summary.csv")
    wrong_csv = os.path.join(exp_dir, f"{experiment_name}_model_misclassified.csv")
    suspect_csv = os.path.join(exp_dir, f"{experiment_name}_suspect_samples.csv")

    group_df.to_csv(group_csv, index=False, encoding="utf-8-sig")
    pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")
    selected_summary_df.to_csv(selected_csv, index=False, encoding="utf-8-sig")

    if len(pred_df):
        pred_df[pred_df["model_correct"] == 0].to_csv(wrong_csv, index=False, encoding="utf-8-sig")
        pred_df[pred_df["final_decision"] == "SUSPECT"].to_csv(suspect_csv, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(wrong_csv, index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(suspect_csv, index=False, encoding="utf-8-sig")

    return {
        "experiment": experiment_name,
        "group_df": group_df,
        "pred_df": pred_df,
        "selected_summary_df": selected_summary_df,
        "group_csv": group_csv,
        "pred_csv": pred_csv,
        "selected_csv": selected_csv,
        "wrong_csv": wrong_csv,
        "suspect_csv": suspect_csv,
    }


# ============================================================
# 7. 报告与画图
# ============================================================

def plot_experiment_summary(all_results, output_dir):
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
                "final_decisive_accuracy": r["final_decisive_accuracy"],
            })

    dfp = pd.DataFrame(rows)

    if len(dfp) == 0:
        return []

    paths = []
    metrics = ["auc", "model_accuracy", "rank_accuracy", "final_decisive_accuracy"]

    for metric in metrics:
        plt.figure(figsize=(13, 5))

        experiments = dfp["experiment"].unique().tolist()
        groups = sorted(dfp["test_group"].unique().tolist())

        x = np.arange(len(groups))
        width = 0.18

        for i, exp in enumerate(experiments):
            vals = []
            for g in groups:
                sub = dfp[(dfp["experiment"] == exp) & (dfp["test_group"] == g)]
                vals.append(float(sub[metric].iloc[0]) if len(sub) else np.nan)

            plt.bar(x + (i - (len(experiments)-1)/2) * width, vals, width, label=exp)

        plt.ylim(0, 1.05)
        plt.xticks(x, groups, rotation=45, ha="right")
        plt.ylabel(metric)
        plt.title(f"v10 experiment comparison: {metric}")
        plt.legend(fontsize=8)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()

        path = os.path.join(fig_dir, f"v10_experiment_{metric}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)

    return paths


def plot_stability_overview(stability_df, output_dir):
    fig_dir = os.path.join(output_dir, "figures")
    ensure_dir(fig_dir)

    paths = []

    if len(stability_df) == 0:
        return paths

    # 稳定/冲突/弱特征统计
    stable_count = int((stability_df["is_direction_stable"] == 1).sum())
    conflict_count = int((stability_df["has_strong_conflict"] == 1).sum())
    weak_count = int(((stability_df["is_direction_stable"] == 0) & (stability_df["has_strong_conflict"] == 0)).sum())

    plt.figure(figsize=(6, 5))
    plt.bar(["stable", "conflict", "weak/other"], [stable_count, conflict_count, weak_count])
    plt.ylabel("Feature count")
    plt.title("v10 feature direction stability overview")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(fig_dir, "v10_feature_stability_count.png")
    plt.savefig(path, dpi=150)
    plt.close()
    paths.append(path)

    # top stable feature
    top = stability_df.sort_values("stability_score", ascending=False).head(25).iloc[::-1]

    plt.figure(figsize=(11, 9))
    plt.barh(top["feature"], top["stability_score"])
    plt.xlabel("stability_score")
    plt.title("Top feature stability scores")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(fig_dir, "v10_top_stability_features.png")
    plt.savefig(path, dpi=150)
    plt.close()
    paths.append(path)

    return paths


def make_report(
    input_paths,
    v7_cols,
    hm_cols,
    global_report_csv,
    global_detail_csv,
    all_results,
    plot_paths,
    output_dir
):
    lines = []

    lines.append("v10 特征方向稳定性分析 + 稳定特征筛选验证报告")
    lines.append("=" * 110)
    lines.append(f"生成时间: {datetime.now()}")
    lines.append("")
    lines.append("输入文件:")
    for k, v in input_paths.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("特征数量:")
    lines.append(f"  v7候选特征数: {len(v7_cols)}")
    lines.append(f"  heatmap候选特征数: {len(hm_cols)}")
    lines.append(f"  总候选特征数: {len(v7_cols) + len(hm_cols)}")
    lines.append("")
    lines.append("全局稳定性报告:")
    lines.append(f"  feature方向稳定报告: {global_report_csv}")
    lines.append(f"  feature×time详细报告: {global_detail_csv}")
    lines.append("")
    lines.append("实验结果:")
    lines.append("-" * 110)

    summary_rows = []

    for res in all_results:
        exp = res["experiment"]
        gdf = res["group_df"]

        lines.append("")
        lines.append(f"[{exp}]")
        lines.append(f"  group_summary: {res['group_csv']}")
        lines.append(f"  predictions: {res['pred_csv']}")
        lines.append(f"  selected_features: {res['selected_csv']}")

        if len(gdf) == 0:
            lines.append("  没有结果。")
            continue

        lines.append(f"  平均AUC: {gdf['auc'].mean():.4f}")
        lines.append(f"  平均model_acc: {gdf['model_accuracy'].mean():.4f}")
        lines.append(f"  平均rank_acc: {gdf['rank_accuracy'].mean():.4f}")
        lines.append(f"  平均final_decisive_acc: {gdf['final_decisive_accuracy'].mean():.4f}")
        lines.append(f"  平均选择特征数: {gdf['n_selected_features'].mean():.2f}")

        target_row = gdf[gdf["test_group"] == TARGET_TIME]
        if len(target_row):
            r = target_row.iloc[0]
            lines.append(
                f"  {TARGET_TIME}: "
                f"AUC={r['auc']:.4f}, "
                f"default_acc={r['default_accuracy_0p5']:.4f}, "
                f"model_acc={r['model_accuracy']:.4f}, "
                f"rank_acc={r['rank_accuracy']:.4f}, "
                f"final_decisive_acc={r['final_decisive_accuracy']:.4f}, "
                f"suspect={int(r['final_n_suspect'])}, "
                f"selected_features={int(r['n_selected_features'])}"
            )

        lines.append("  各time结果:")
        for _, r in gdf.iterrows():
            lines.append(
                f"    {r['test_group']}: "
                f"AUC={r['auc']:.3f}, "
                f"default={r['default_accuracy_0p5']:.3f}, "
                f"model={r['model_accuracy']:.3f}, "
                f"rank={r['rank_accuracy']:.3f}, "
                f"final={r['final_decisive_accuracy']:.3f}, "
                f"suspect={int(r['final_n_suspect'])}, "
                f"features={int(r['n_selected_features'])}"
            )

        row = {
            "experiment": exp,
            "mean_auc": gdf["auc"].mean(),
            "mean_model_acc": gdf["model_accuracy"].mean(),
            "mean_rank_acc": gdf["rank_accuracy"].mean(),
            "mean_final_decisive_acc": gdf["final_decisive_accuracy"].mean(),
            "mean_selected_features": gdf["n_selected_features"].mean(),
        }

        if len(target_row):
            r = target_row.iloc[0]
            row.update({
                "auc_144226": r["auc"],
                "model_acc_144226": r["model_accuracy"],
                "rank_acc_144226": r["rank_accuracy"],
                "final_decisive_acc_144226": r["final_decisive_accuracy"],
                "suspect_144226": r["final_n_suspect"],
                "selected_features_144226": r["n_selected_features"],
            })

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, "v10_experiment_overall_summary.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    lines.append("")
    lines.append("总汇总表:")
    lines.append(f"  {summary_csv}")
    lines.append("")
    lines.append("图像输出:")
    for p in plot_paths:
        lines.append(f"  {p}")

    lines.append("")
    lines.append("怎么看结果:")
    lines.append("  1. 如果 B_stable_v7_only 或 D_stable_v7_plus_heatmap 的 144226 明显优于 A_all_v7_baseline，说明方向稳定筛选有效。")
    lines.append("  2. 如果稳定筛选后整体下降，说明目前可用稳定特征太少，或方向翻转不是唯一问题。")
    lines.append("  3. 如果 heatmap稳定特征仍不能提升 144226，说明 heatmap在跨时间点上的方向也不稳定。")
    lines.append("  4. 如果某些特征在全局报告里 has_strong_conflict=1，说明它们在不同time中方向翻转，不适合作为稳定二分类规则。")

    report_path = os.path.join(output_dir, "v10_report.txt")
    save_text(report_path, "\n".join(lines))

    return report_path, summary_csv


# ============================================================
# 8. 主程序
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)

    print("=" * 110)
    print("v10：特征方向稳定性分析 + 稳定特征筛选分类")
    print("=" * 110)

    # 1. 加载数据
    v7_df, v7_path = load_v7_feature_table()
    hm_df, hm_path = load_heatmap_feature_table()

    combo_df, v7_cols, hm_cols = build_combined_table(v7_df, hm_df)

    input_paths = {
        "v7_feature_csv": v7_path,
        "heatmap_feature_csv": hm_path if hm_path else "未找到/未使用",
    }

    print("样本数量:", len(combo_df))
    print(combo_df[LABEL_COL].value_counts())
    print("time groups:", sorted(combo_df[GROUP_COL].astype(str).unique().tolist()))
    print("v7候选特征数:", len(v7_cols))
    print("heatmap候选特征数:", len(hm_cols))

    # 2. 全局方向稳定性报告：注意这是诊断报告，不用于严格模型验证
    print("\n开始生成全局特征方向稳定性报告...")
    all_candidate_cols = v7_cols + hm_cols

    global_report_df, global_detail_df, global_sign_map = analyze_feature_stability(
        combo_df,
        all_candidate_cols,
        groups=sorted(pd.unique(combo_df[GROUP_COL].astype(str)).tolist()),
        min_effect_groups=MIN_GLOBAL_EFFECT_GROUPS
    )

    global_report_csv = os.path.join(OUTPUT_DIR, "v10_global_feature_direction_stability_report.csv")
    global_detail_csv = os.path.join(OUTPUT_DIR, "v10_global_feature_direction_stability_detail.csv")
    global_stable_csv = os.path.join(OUTPUT_DIR, "v10_global_stable_features.csv")
    global_conflict_csv = os.path.join(OUTPUT_DIR, "v10_global_conflict_features.csv")

    global_report_df.to_csv(global_report_csv, index=False, encoding="utf-8-sig")
    global_detail_df.to_csv(global_detail_csv, index=False, encoding="utf-8-sig")

    if len(global_report_df):
        global_report_df[global_report_df["is_direction_stable"] == 1].to_csv(
            global_stable_csv,
            index=False,
            encoding="utf-8-sig"
        )
        global_report_df[global_report_df["has_strong_conflict"] == 1].to_csv(
            global_conflict_csv,
            index=False,
            encoding="utf-8-sig"
        )

    print("全局稳定性报告:", global_report_csv)
    print("全局稳定特征:", global_stable_csv)
    print("全局方向冲突特征:", global_conflict_csv)

    if len(global_report_df):
        n_stable = int((global_report_df["is_direction_stable"] == 1).sum())
        n_conflict = int((global_report_df["has_strong_conflict"] == 1).sum())
        print("全局稳定特征数量:", n_stable)
        print("全局方向冲突特征数量:", n_conflict)

        print("\n全局稳定性前10特征:")
        for _, r in global_report_df.head(10).iterrows():
            print(
                f"  {r['feature']}: "
                f"stable={int(r['is_direction_stable'])}, "
                f"conflict={int(r['has_strong_conflict'])}, "
                f"dir={r['majority_direction']}, "
                f"mean_auc={r['mean_auc_direction_free']:.3f}, "
                f"score={r['stability_score']:.3f}"
            )

    # 3. 实验验证
    all_results = []

    # A: baseline，全部v7特征，不做稳定筛选
    all_results.append(
        validate_experiment(
            combo_df,
            v7_cols,
            experiment_name="A_all_v7_baseline",
            output_dir=OUTPUT_DIR,
            selection_mode="all",
            source_tag="v7_all"
        )
    )

    # B: 稳定v7特征
    all_results.append(
        validate_experiment(
            combo_df,
            v7_cols,
            experiment_name="B_stable_v7_only",
            output_dir=OUTPUT_DIR,
            selection_mode="stable",
            source_tag="v7_stable"
        )
    )

    # C: 稳定heatmap特征
    if len(hm_cols) > 0:
        all_results.append(
            validate_experiment(
                combo_df,
                hm_cols,
                experiment_name="C_stable_heatmap_only",
                output_dir=OUTPUT_DIR,
                selection_mode="stable",
                source_tag="hm_stable"
            )
        )

    # D: 稳定 v7 + heatmap 特征
    all_results.append(
        validate_experiment(
            combo_df,
            all_candidate_cols,
            experiment_name="D_stable_v7_plus_heatmap",
            output_dir=OUTPUT_DIR,
            selection_mode="stable",
            source_tag="combo_stable"
        )
    )

    # 4. 画图和报告
    plot_paths = []
    plot_paths.extend(plot_stability_overview(global_report_df, OUTPUT_DIR))
    plot_paths.extend(plot_experiment_summary(all_results, OUTPUT_DIR))

    report_path, summary_csv = make_report(
        input_paths=input_paths,
        v7_cols=v7_cols,
        hm_cols=hm_cols,
        global_report_csv=global_report_csv,
        global_detail_csv=global_detail_csv,
        all_results=all_results,
        plot_paths=plot_paths,
        output_dir=OUTPUT_DIR
    )

    print("\n" + "=" * 110)
    print("v10 完成")
    print("=" * 110)
    print("输出文件夹:", OUTPUT_DIR)
    print("总报告:", report_path)
    print("总汇总表:", summary_csv)
    print("全局方向稳定性报告:", global_report_csv)
    print("全局方向冲突特征:", global_conflict_csv)

    print("\n实验核心结果:")
    for res in all_results:
        gdf = res["group_df"]
        if len(gdf) == 0:
            continue

        print(f"\n{res['experiment']}:")
        print(
            f"  平均AUC={gdf['auc'].mean():.3f}, "
            f"平均model_acc={gdf['model_accuracy'].mean():.3f}, "
            f"平均rank_acc={gdf['rank_accuracy'].mean():.3f}, "
            f"平均final_acc={gdf['final_decisive_accuracy'].mean():.3f}"
        )

        target = gdf[gdf["test_group"] == TARGET_TIME]
        if len(target):
            r = target.iloc[0]
            print(
                f"  144226: "
                f"AUC={r['auc']:.3f}, "
                f"default_acc={r['default_accuracy_0p5']:.3f}, "
                f"model_acc={r['model_accuracy']:.3f}, "
                f"rank_acc={r['rank_accuracy']:.3f}, "
                f"final_acc={r['final_decisive_accuracy']:.3f}, "
                f"suspect={int(r['final_n_suspect'])}, "
                f"selected_features={int(r['n_selected_features'])}"
            )

    print("\n图片输出:")
    for p in plot_paths:
        print(" ", p)

    print("\n请把上面“实验核心结果”和 global_conflict_features 里前几行发给我，我帮你判断哪些特征需要删。")


if __name__ == "__main__":
    main()
