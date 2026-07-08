# -*- coding: utf-8 -*-
"""
leak_v9_diagnose_144226.py

v9：HM20260626_144226.ld 专项诊断程序

目的：
    前面 v7 / v8 / v8.1 的结果都说明：
        - v7 是当前最好的 baseline；
        - heatmap PNG 形态特征没有明显帮助；
        - 最主要的困难点集中在 HM20260626_144226.ld。

    所以 v9 不再训练新模型，而是专门诊断 144226：
        1. 哪些 center 被误判？
        2. TRUE/FALSE 的核心特征是否真的混在一起？
        3. 同一个 center 下，真泄漏和假泄漏的特征差值有多大？
        4. 哪些特征对 144226 还有区分力？
        5. heatmap 核心形态特征在 144226 上有没有区分力？
        6. 哪些样本应当被标成 SUSPECT，而不是强行 TRUE/FALSE？

运行：
    python leak_v9_diagnose_144226.py

输入默认来自：
    1. v8.1 的 A_v7_only 预测：
       C:\\Users\\jiangxinru6\\Desktop\\wurenji\\leak_v8_1_heatmap_shape_ablation_results\\A_v7_only\\A_v7_only_predictions.csv

    2. v8.1 的 A_v7_only 特征：
       C:\\Users\\jiangxinru6\\Desktop\\wurenji\\leak_v8_1_heatmap_shape_ablation_results\\v8_1_features_A_v7_only.csv

    3. v8.1 的 heatmap 核心形态特征：
       C:\\Users\\jiangxinru6\\Desktop\\wurenji\\leak_v8_1_heatmap_shape_ablation_results\\v8_1_heatmap_core_shape_features.csv

    4. v4 合并后的原始特征：
       C:\\Users\\jiangxinru6\\Desktop\\wurenji\\leak_v4_compare_results\\merged_feature_dataset.csv

输出：
    C:\\Users\\jiangxinru6\\Desktop\\wurenji\\leak_v9_144226_diagnosis_results\\

重点输出文件：
    v9_144226_predictions_sorted.csv
    v9_144226_wrong_samples.csv
    v9_144226_suspect_or_conflict_samples.csv
    v9_144226_v7_feature_compare.csv
    v9_144226_raw_feature_compare.csv
    v9_144226_heatmap_feature_compare.csv
    v9_144226_center_pair_compare.csv
    v9_144226_report.txt
"""

import os
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

BASE_DIR = r"C:\Users\jiangxinru6\Desktop\wurenji"

TARGET_TIME = "HM20260626_144226.ld"

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "leak_v9_144226_diagnosis_results"
)

# v8.1 A_v7_only 预测结果，优先使用这个，因为它代表当前最优 baseline
A_V7_PRED_CSV = os.path.join(
    BASE_DIR,
    "leak_v8_1_heatmap_shape_ablation_results",
    "A_v7_only",
    "A_v7_only_predictions.csv"
)

# v8.1 A_v7_only 特征表
A_V7_FEATURE_CSV = os.path.join(
    BASE_DIR,
    "leak_v8_1_heatmap_shape_ablation_results",
    "v8_1_features_A_v7_only.csv"
)

# heatmap 核心形态特征
HEATMAP_FEATURE_CSV = os.path.join(
    BASE_DIR,
    "leak_v8_1_heatmap_shape_ablation_results",
    "v8_1_heatmap_core_shape_features.csv"
)

# 原始 v4 合并特征
MERGED_FEATURE_CSV = os.path.join(
    BASE_DIR,
    "leak_v4_compare_results",
    "merged_feature_dataset.csv"
)

# 可选：v7 原始输出，如果 v8.1 文件不存在时可手动改这里
V7_PRED_CSV_FALLBACK = os.path.join(
    BASE_DIR,
    "leak_v7_robust_feature_results",
    "v7_predictions.csv"
)

V7_FEATURE_CSV_FALLBACK = os.path.join(
    BASE_DIR,
    "leak_v7_robust_feature_results",
    "v7_robust_feature_dataset.csv"
)

LABEL_COL_OPTIONS = ["true_label", "label"]
GROUP_COL_OPTIONS = ["test_group", "time"]

META_COLS = [
    "dataset",
    "label",
    "true_label",
    "time",
    "test_group",
    "center",
    "experiment",
    "heatmap_path",
    "row_index",
    "best_direction",
    "energy_direction",
    "decay_direction",
    "representative_file",
]

# 最需要关注的 v7 / 原始核心特征
KEY_FEATURES_CANDIDATES = [
    "spec_slope",
    "spec_slope__time_robust_z",
    "spec_slope__time_rank_pct",

    "ratio_60_70k",
    "ratio_60_70k__time_robust_z",
    "ratio_60_70k__time_rank_pct",

    "best_direction_combined_score",
    "best_direction_combined_score__time_robust_z",
    "best_direction_combined_score__time_rank_pct",

    "direction_contrast",
    "direction_contrast__time_robust_z",
    "direction_contrast__time_rank_pct",

    "spec_flatness",
    "spec_flatness__time_robust_z",
    "spec_flatness__time_rank_pct",

    "high_freq_ratio",
    "high_freq_ratio__time_robust_z",
    "high_freq_ratio__time_rank_pct",

    "decay_R2",
    "decay_R2__time_robust_z",
    "decay_R2__time_rank_pct",

    "near_far_ratio",
    "near_far_ratio__time_robust_z",
    "near_far_ratio__time_rank_pct",

    "time_energy_cv",
    "time_energy_cv__time_robust_z",
    "time_energy_cv__time_rank_pct",
]

# heatmap 核心特征
KEY_HEATMAP_FEATURES = [
    "hm_shape_leak_like_score",
    "hm_directed_core_score",
    "hm_diffuse_score",
    "hm_entropy_2d",
    "hm_weighted_elongation",
    "hm_weighted_eccentricity",
    "hm_hot_area_p95_ratio",
    "hm_hot_area_p90_ratio",
    "hm_largest_component_ratio_p95",
    "hm_energy_concentration_top5",
    "hm_energy_concentration_top10",
    "hm_core_to_outer_energy_ratio",
    "hm_radial_spread_norm",
]

# SUSPECT 判断阈值
PROB_LOW_SUSPECT = 0.40
PROB_HIGH_SUSPECT = 0.60


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


def safe_float_array(s):
    arr = pd.to_numeric(s, errors="coerce")
    arr = arr.replace([np.inf, -np.inf], np.nan)
    return arr


def find_existing_path(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def get_label_col(df):
    for c in LABEL_COL_OPTIONS:
        if c in df.columns:
            return c
    raise ValueError("找不到标签列，期望 true_label 或 label。")


def get_group_col(df):
    for c in GROUP_COL_OPTIONS:
        if c in df.columns:
            return c
    raise ValueError("找不到时间/分组列，期望 test_group 或 time。")


def filter_target_time(df):
    gcol = get_group_col(df)
    return df[df[gcol].astype(str) == TARGET_TIME].copy()


def add_center_norm(df):
    if "center" in df.columns:
        df = df.copy()
        df["center_norm"] = df["center"].apply(normalize_center_id)
    return df


def label_to_binary(labels):
    return np.array([1 if str(x) == "TRUE_LEAK" else 0 for x in labels], dtype=int)


def binary_to_label(v):
    return "TRUE_LEAK" if int(v) == 1 else "FALSE_LEAK"


def infer_pred_correct(row, pred_col, label_col):
    try:
        return int(str(row[pred_col]) == str(row[label_col]))
    except Exception:
        return 0


def numeric_feature_columns(df):
    cols = []

    for c in df.columns:
        if c in META_COLS or c.endswith("_correct"):
            continue

        if c in [
            "prob_TRUE_LEAK",
            "best_threshold",
            "prob_rank_pct_in_group",
            "prob_relative_minmax_in_group",
        ]:
            continue

        vals = safe_float_array(df[c])
        if vals.notna().mean() > 0.8:
            cols.append(c)

    return cols


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


def histogram_overlap(true_vals, false_vals, bins=20):
    """
    0~1，越大表示分布重叠越严重。
    1 = 几乎完全重叠，0 = 几乎不重叠。
    """
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


def simple_auc_for_feature(true_vals, false_vals):
    """
    单个特征的方向无关 AUC。
    返回 max(AUC, 1-AUC)，越接近1越能区分，0.5接近随机。
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

        auc = float(roc_auc_score(y, x))
        return float(max(auc, 1.0 - auc))
    except Exception:
        return np.nan


def feature_compare_table(df, label_col, feature_cols, output_csv):
    rows = []

    label = df[label_col].astype(str)

    for c in feature_cols:
        if c not in df.columns:
            continue

        vals = safe_float_array(df[c])

        true_vals = vals[label == "TRUE_LEAK"].dropna().values
        false_vals = vals[label == "FALSE_LEAK"].dropna().values

        if len(true_vals) == 0 or len(false_vals) == 0:
            continue

        d = cohen_d(true_vals, false_vals)
        overlap = histogram_overlap(true_vals, false_vals)
        auc = simple_auc_for_feature(true_vals, false_vals)

        row = {
            "feature": c,
            "n_true": len(true_vals),
            "n_false": len(false_vals),

            "true_mean": float(np.mean(true_vals)),
            "false_mean": float(np.mean(false_vals)),
            "diff_true_minus_false": float(np.mean(true_vals) - np.mean(false_vals)),

            "true_median": float(np.median(true_vals)),
            "false_median": float(np.median(false_vals)),
            "median_diff_true_minus_false": float(np.median(true_vals) - np.median(false_vals)),

            "true_std": float(np.std(true_vals, ddof=1)) if len(true_vals) > 1 else 0.0,
            "false_std": float(np.std(false_vals, ddof=1)) if len(false_vals) > 1 else 0.0,

            "true_q25": float(np.percentile(true_vals, 25)),
            "true_q75": float(np.percentile(true_vals, 75)),
            "false_q25": float(np.percentile(false_vals, 25)),
            "false_q75": float(np.percentile(false_vals, 75)),

            "cohen_d_TRUE_minus_FALSE": d,
            "abs_cohen_d": abs(d) if np.isfinite(d) else np.nan,
            "hist_overlap_0to1_high_is_bad": overlap,
            "single_feature_auc_direction_free": auc,
        }
        rows.append(row)

    out = pd.DataFrame(rows)

    if len(out):
        out = out.sort_values(
            ["single_feature_auc_direction_free", "abs_cohen_d"],
            ascending=[False, False]
        )

    out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return out


# ============================================================
# 3. 预测诊断
# ============================================================

def load_predictions():
    pred_path = find_existing_path(A_V7_PRED_CSV, V7_PRED_CSV_FALLBACK)

    if pred_path is None:
        raise FileNotFoundError(
            "找不到预测文件。请确认 v8.1 或 v7 已经运行。\n"
            f"期望路径1: {A_V7_PRED_CSV}\n"
            f"期望路径2: {V7_PRED_CSV_FALLBACK}"
        )

    pred = pd.read_csv(pred_path)
    pred = add_center_norm(pred)

    return pred, pred_path


def diagnose_predictions(pred_df, output_dir):
    sub = filter_target_time(pred_df)
    sub = add_center_norm(sub)

    label_col = get_label_col(sub)

    # 尽量兼容不同版本列名
    if "model_pred" not in sub.columns:
        if "v7_model_pred" in sub.columns:
            sub["model_pred"] = sub["v7_model_pred"]
        elif "default_pred_0p5" in sub.columns:
            sub["model_pred"] = sub["default_pred_0p5"]

    if "rank_binary_pred" not in sub.columns:
        if "v7_rank_binary_pred" in sub.columns:
            sub["rank_binary_pred"] = sub["v7_rank_binary_pred"]

    if "final_decision" not in sub.columns:
        if "v7_final_decision" in sub.columns:
            sub["final_decision"] = sub["v7_final_decision"]
        else:
            sub["final_decision"] = ""

    if "model_correct" not in sub.columns and "model_pred" in sub.columns:
        sub["model_correct"] = sub.apply(
            lambda r: infer_pred_correct(r, "model_pred", label_col),
            axis=1
        )

    if "prob_TRUE_LEAK" in sub.columns:
        sub["prob_TRUE_LEAK"] = safe_float_array(sub["prob_TRUE_LEAK"])
        sub = sub.sort_values(["prob_TRUE_LEAK", "center_norm"], ascending=[False, True])

        sub["prob_margin_to_0p5"] = (sub["prob_TRUE_LEAK"] - 0.5).abs()
        sub["prob_in_gray_zone_0p4_0p6"] = (
            (sub["prob_TRUE_LEAK"] >= PROB_LOW_SUSPECT) &
            (sub["prob_TRUE_LEAK"] <= PROB_HIGH_SUSPECT)
        ).astype(int)

    # 冲突样本：模型和rank不一致，或 final=SUSPECT，或概率灰区
    conflict_mask = pd.Series(False, index=sub.index)

    if "model_pred" in sub.columns and "rank_binary_pred" in sub.columns:
        conflict_mask |= sub["model_pred"].astype(str) != sub["rank_binary_pred"].astype(str)

    if "final_decision" in sub.columns:
        conflict_mask |= sub["final_decision"].astype(str).str.upper().eq("SUSPECT")

    if "prob_in_gray_zone_0p4_0p6" in sub.columns:
        conflict_mask |= sub["prob_in_gray_zone_0p4_0p6"].astype(bool)

    pred_sorted_csv = os.path.join(output_dir, "v9_144226_predictions_sorted.csv")
    sub.to_csv(pred_sorted_csv, index=False, encoding="utf-8-sig")

    wrong = sub.copy()
    if "model_correct" in wrong.columns:
        wrong = wrong[wrong["model_correct"].astype(int) == 0].copy()
    else:
        wrong = wrong.iloc[0:0].copy()

    wrong_csv = os.path.join(output_dir, "v9_144226_wrong_samples.csv")
    wrong.to_csv(wrong_csv, index=False, encoding="utf-8-sig")

    conflict = sub[conflict_mask].copy()
    conflict_csv = os.path.join(output_dir, "v9_144226_suspect_or_conflict_samples.csv")
    conflict.to_csv(conflict_csv, index=False, encoding="utf-8-sig")

    # 按 center 的预测配对
    pair_pred = make_center_pair_prediction_table(sub, label_col)
    pair_pred_csv = os.path.join(output_dir, "v9_144226_center_pair_predictions.csv")
    pair_pred.to_csv(pair_pred_csv, index=False, encoding="utf-8-sig")

    return {
        "pred_target_df": sub,
        "pred_sorted_csv": pred_sorted_csv,
        "wrong_df": wrong,
        "wrong_csv": wrong_csv,
        "conflict_df": conflict,
        "conflict_csv": conflict_csv,
        "pair_pred_df": pair_pred,
        "pair_pred_csv": pair_pred_csv,
        "label_col": label_col,
    }


def make_center_pair_prediction_table(sub, label_col):
    rows = []

    for center, g in sub.groupby("center_norm"):
        true_row = g[g[label_col].astype(str) == "TRUE_LEAK"]
        false_row = g[g[label_col].astype(str) == "FALSE_LEAK"]

        row = {
            "center": center,
            "has_true": int(len(true_row) > 0),
            "has_false": int(len(false_row) > 0),
        }

        if len(true_row) > 0:
            tr = true_row.iloc[0]
            row["true_prob"] = tr.get("prob_TRUE_LEAK", np.nan)
            row["true_model_pred"] = tr.get("model_pred", "")
            row["true_rank_pred"] = tr.get("rank_binary_pred", "")
            row["true_final_decision"] = tr.get("final_decision", "")
            row["true_model_correct"] = tr.get("model_correct", np.nan)

        if len(false_row) > 0:
            fr = false_row.iloc[0]
            row["false_prob"] = fr.get("prob_TRUE_LEAK", np.nan)
            row["false_model_pred"] = fr.get("model_pred", "")
            row["false_rank_pred"] = fr.get("rank_binary_pred", "")
            row["false_final_decision"] = fr.get("final_decision", "")
            row["false_model_correct"] = fr.get("model_correct", np.nan)

        if "true_prob" in row and "false_prob" in row:
            try:
                row["true_minus_false_prob"] = float(row["true_prob"]) - float(row["false_prob"])
                row["pair_order_correct"] = int(float(row["true_prob"]) > float(row["false_prob"]))
            except Exception:
                row["true_minus_false_prob"] = np.nan
                row["pair_order_correct"] = np.nan

        rows.append(row)

    out = pd.DataFrame(rows)

    if len(out) and "true_minus_false_prob" in out.columns:
        out = out.sort_values("true_minus_false_prob", ascending=True)

    return out


# ============================================================
# 4. 特征诊断
# ============================================================

def load_feature_tables():
    feature_path = find_existing_path(A_V7_FEATURE_CSV, V7_FEATURE_CSV_FALLBACK)
    raw_path = find_existing_path(MERGED_FEATURE_CSV)
    heatmap_path = find_existing_path(HEATMAP_FEATURE_CSV)

    tables = {}

    if feature_path:
        v7f = pd.read_csv(feature_path)
        v7f = add_center_norm(v7f)
        tables["v7_feature"] = (v7f, feature_path)

    if raw_path:
        raw = pd.read_csv(raw_path)
        raw = add_center_norm(raw)
        tables["raw_feature"] = (raw, raw_path)

    if heatmap_path:
        hm = pd.read_csv(heatmap_path)
        hm = add_center_norm(hm)
        tables["heatmap_feature"] = (hm, heatmap_path)

    return tables


def diagnose_feature_table(df, source_name, output_dir, key_features=None):
    sub = filter_target_time(df)
    sub = add_center_norm(sub)

    label_col = get_label_col(sub)

    if key_features is None:
        cols = numeric_feature_columns(sub)
    else:
        cols = [c for c in key_features if c in sub.columns]

        # 如果候选核心特征太少，则补充全部数值特征
        if len(cols) < 5:
            cols = numeric_feature_columns(sub)

    out_csv = os.path.join(output_dir, f"v9_144226_{source_name}_compare.csv")
    compare_df = feature_compare_table(sub, label_col, cols, out_csv)

    return {
        "source_name": source_name,
        "target_df": sub,
        "label_col": label_col,
        "feature_cols": cols,
        "compare_df": compare_df,
        "compare_csv": out_csv,
    }


def make_center_pair_feature_compare(v7_sub, raw_sub, hm_sub, output_dir):
    """
    按 center 配对 TRUE/FALSE 特征，输出同一 center 下真/假的差值。
    优先使用 v7 transformed 特征，也补充 raw 和 heatmap 关键特征。
    """
    rows = []

    # 准备不同来源
    sources = []

    if v7_sub is not None and len(v7_sub):
        sources.append(("v7", v7_sub, [c for c in KEY_FEATURES_CANDIDATES if c in v7_sub.columns]))

    if raw_sub is not None and len(raw_sub):
        sources.append(("raw", raw_sub, [c for c in KEY_FEATURES_CANDIDATES if c in raw_sub.columns]))

    if hm_sub is not None and len(hm_sub):
        sources.append(("hm", hm_sub, [c for c in KEY_HEATMAP_FEATURES if c in hm_sub.columns]))

    all_centers = sorted(set().union(*[set(sdf["center_norm"].astype(str)) for _, sdf, _ in sources]))

    for center in all_centers:
        row = {"center": center}

        for prefix, sdf, cols in sources:
            label_col = get_label_col(sdf)
            g = sdf[sdf["center_norm"].astype(str) == str(center)]

            t = g[g[label_col].astype(str) == "TRUE_LEAK"]
            f = g[g[label_col].astype(str) == "FALSE_LEAK"]

            row[f"{prefix}_has_true"] = int(len(t) > 0)
            row[f"{prefix}_has_false"] = int(len(f) > 0)

            if len(t) == 0 or len(f) == 0:
                continue

            t0 = t.iloc[0]
            f0 = f.iloc[0]

            for c in cols:
                tv = pd.to_numeric(pd.Series([t0.get(c, np.nan)]), errors="coerce").iloc[0]
                fv = pd.to_numeric(pd.Series([f0.get(c, np.nan)]), errors="coerce").iloc[0]

                row[f"{prefix}_{c}_true"] = tv
                row[f"{prefix}_{c}_false"] = fv
                row[f"{prefix}_{c}_diff_true_minus_false"] = tv - fv if np.isfinite(tv) and np.isfinite(fv) else np.nan

        rows.append(row)

    out = pd.DataFrame(rows)

    out_csv = os.path.join(output_dir, "v9_144226_center_pair_feature_compare.csv")
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    return out, out_csv


# ============================================================
# 5. 作图
# ============================================================

def plot_prediction_probability(pred_sub, label_col, output_dir):
    fig_dir = os.path.join(output_dir, "figures")
    ensure_dir(fig_dir)

    if "prob_TRUE_LEAK" not in pred_sub.columns:
        return None

    df = pred_sub.copy()
    df = df.sort_values("center_norm")

    plt.figure(figsize=(12, 5))

    true_df = df[df[label_col].astype(str) == "TRUE_LEAK"]
    false_df = df[df[label_col].astype(str) == "FALSE_LEAK"]

    plt.scatter(true_df["center_norm"], true_df["prob_TRUE_LEAK"], label="TRUE_LEAK", marker="o")
    plt.scatter(false_df["center_norm"], false_df["prob_TRUE_LEAK"], label="FALSE_LEAK", marker="x")

    plt.axhline(0.5, linestyle="--", linewidth=1, label="threshold 0.5")
    plt.axhline(PROB_LOW_SUSPECT, linestyle=":", linewidth=1, label="gray zone")
    plt.axhline(PROB_HIGH_SUSPECT, linestyle=":", linewidth=1)

    plt.xlabel("center")
    plt.ylabel("prob_TRUE_LEAK")
    plt.title(f"{TARGET_TIME} prediction probability by center")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(fig_dir, "v9_144226_prob_by_center.png")
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def plot_pair_prob_diff(pair_pred_df, output_dir):
    fig_dir = os.path.join(output_dir, "figures")
    ensure_dir(fig_dir)

    if len(pair_pred_df) == 0 or "true_minus_false_prob" not in pair_pred_df.columns:
        return None

    df = pair_pred_df.copy().sort_values("center")

    plt.figure(figsize=(12, 5))
    plt.bar(df["center"].astype(str), df["true_minus_false_prob"])
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("center")
    plt.ylabel("TRUE prob - FALSE prob")
    plt.title(f"{TARGET_TIME} paired probability difference by center")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(fig_dir, "v9_144226_pair_prob_diff.png")
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def plot_feature_distributions(feature_diag, output_dir, top_n=8):
    fig_dir = os.path.join(output_dir, "figures")
    ensure_dir(fig_dir)

    paths = []

    compare_df = feature_diag["compare_df"]
    sub = feature_diag["target_df"]
    label_col = feature_diag["label_col"]
    source = feature_diag["source_name"]

    if len(compare_df) == 0:
        return paths

    # 前 top_n 个最能区分的特征 + 关键特征
    top_features = compare_df["feature"].head(top_n).tolist()

    for c in top_features:
        if c not in sub.columns:
            continue

        true_vals = safe_float_array(sub.loc[sub[label_col].astype(str) == "TRUE_LEAK", c]).dropna()
        false_vals = safe_float_array(sub.loc[sub[label_col].astype(str) == "FALSE_LEAK", c]).dropna()

        if len(true_vals) == 0 or len(false_vals) == 0:
            continue

        plt.figure(figsize=(8, 5))
        plt.hist(true_vals, bins=12, alpha=0.6, label="TRUE_LEAK")
        plt.hist(false_vals, bins=12, alpha=0.6, label="FALSE_LEAK")
        plt.xlabel(c)
        plt.ylabel("Count")
        plt.title(f"{TARGET_TIME} {source}: {c}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        safe_c = safe_filename(c)
        path = os.path.join(fig_dir, f"v9_144226_{source}_{safe_c}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)

    return paths


def safe_filename(s):
    keep = []
    for ch in str(s):
        if ch.isalnum() or ch in ["_", "-", "."]:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)[:150]


# ============================================================
# 6. 生成报告
# ============================================================

def make_report(pred_info, feature_diags, pair_feature_csv, output_dir, input_paths, plot_paths):
    lines = []

    pred_sub = pred_info["pred_target_df"]
    wrong_df = pred_info["wrong_df"]
    conflict_df = pred_info["conflict_df"]
    pair_pred_df = pred_info["pair_pred_df"]
    label_col = pred_info["label_col"]

    lines.append("v9 144226 专项诊断报告")
    lines.append("=" * 100)
    lines.append(f"生成时间: {datetime.now()}")
    lines.append(f"目标时间点: {TARGET_TIME}")
    lines.append("")
    lines.append("输入文件:")
    for name, path in input_paths.items():
        lines.append(f"  {name}: {path}")
    lines.append("")

    lines.append("一、预测结果诊断")
    lines.append("-" * 100)
    lines.append(f"144226 样本数: {len(pred_sub)}")

    if len(pred_sub):
        label_counts = pred_sub[label_col].value_counts()
        for label, cnt in label_counts.items():
            lines.append(f"  {label}: {int(cnt)}")

    if "model_correct" in pred_sub.columns and len(pred_sub):
        acc = float(pred_sub["model_correct"].astype(int).mean())
        lines.append(f"模型准确率: {acc:.4f}")
        lines.append(f"模型误判数量: {len(wrong_df)}")

    if "prob_TRUE_LEAK" in pred_sub.columns and len(pred_sub):
        true_probs = pred_sub.loc[pred_sub[label_col].astype(str) == "TRUE_LEAK", "prob_TRUE_LEAK"]
        false_probs = pred_sub.loc[pred_sub[label_col].astype(str) == "FALSE_LEAK", "prob_TRUE_LEAK"]

        lines.append(f"TRUE_LEAK 平均概率: {true_probs.mean():.4f}")
        lines.append(f"FALSE_LEAK 平均概率: {false_probs.mean():.4f}")
        lines.append(f"TRUE-FALSE 平均概率差: {(true_probs.mean() - false_probs.mean()):.4f}")
        lines.append(f"概率灰区 [{PROB_LOW_SUSPECT}, {PROB_HIGH_SUSPECT}] 样本数: {int(pred_sub['prob_in_gray_zone_0p4_0p6'].sum()) if 'prob_in_gray_zone_0p4_0p6' in pred_sub.columns else 'NA'}")

    lines.append(f"冲突/SUSPECT样本数量: {len(conflict_df)}")
    lines.append(f"预测明细: {pred_info['pred_sorted_csv']}")
    lines.append(f"误判样本: {pred_info['wrong_csv']}")
    lines.append(f"冲突/SUSPECT样本: {pred_info['conflict_csv']}")
    lines.append(f"按center配对预测: {pred_info['pair_pred_csv']}")
    lines.append("")

    if len(pair_pred_df) and "pair_order_correct" in pair_pred_df.columns:
        pair_valid = pair_pred_df["pair_order_correct"].dropna()
        if len(pair_valid):
            lines.append("按 center 配对概率排序:")
            lines.append(f"  TRUE 概率 > FALSE 概率 的 center 比例: {pair_valid.mean():.4f}")
            bad_pairs = pair_pred_df[pair_pred_df["pair_order_correct"] == 0]
            lines.append(f"  配对排序失败 center 数量: {len(bad_pairs)}")
            if len(bad_pairs):
                bad_centers = ", ".join(bad_pairs["center"].astype(str).tolist())
                lines.append(f"  配对排序失败 center: {bad_centers}")
            lines.append("")

    lines.append("二、特征区分度诊断")
    lines.append("-" * 100)

    for diag in feature_diags:
        source = diag["source_name"]
        compare_df = diag["compare_df"]

        lines.append("")
        lines.append(f"[{source}]")
        lines.append(f"对比表: {diag['compare_csv']}")

        if len(compare_df) == 0:
            lines.append("  没有可用数值特征。")
            continue

        lines.append("  区分力前10特征:")
        for _, r in compare_df.head(10).iterrows():
            lines.append(
                f"    {r['feature']}: "
                f"AUC={r['single_feature_auc_direction_free']:.3f}, "
                f"|d|={r['abs_cohen_d']:.3f}, "
                f"overlap={r['hist_overlap_0to1_high_is_bad']:.3f}, "
                f"true_mean={r['true_mean']:.6g}, "
                f"false_mean={r['false_mean']:.6g}"
            )

        # 判断整体分离情况
        best_auc = compare_df["single_feature_auc_direction_free"].max()
        median_overlap = compare_df["hist_overlap_0to1_high_is_bad"].median()

        lines.append(f"  单特征最好AUC: {best_auc:.4f}")
        lines.append(f"  特征分布中位重叠度: {median_overlap:.4f}")

        if best_auc < 0.65:
            lines.append("  诊断: 单个特征基本没有明显区分力。")
        elif best_auc < 0.75:
            lines.append("  诊断: 有少量弱区分特征，但不足以稳定分类。")
        else:
            lines.append("  诊断: 存在较明显区分特征，可以进一步重点分析。")

    lines.append("")
    lines.append("三、同 center 真/假配对特征对比")
    lines.append("-" * 100)
    lines.append(f"配对特征对比表: {pair_feature_csv}")
    lines.append("说明: 如果同一个 center 下 TRUE 和 FALSE 的关键特征差值很小，说明该位置本身真假声学差异弱。")
    lines.append("")

    lines.append("四、图像输出")
    lines.append("-" * 100)
    for p in plot_paths:
        lines.append(f"  {p}")

    lines.append("")
    lines.append("五、建议")
    lines.append("-" * 100)
    lines.append("1. 如果 144226 的 v7_feature_compare 中大部分特征 AUC 接近 0.5~0.65，说明当前稳健特征确实分不开。")
    lines.append("2. 如果 heatmap_feature_compare 的 AUC 也接近 0.5，说明 PNG heatmap 形态在该时间点没有提供有效增益。")
    lines.append("3. 如果配对概率 true_minus_false_prob 在多个 center 上为负，说明模型在这些位置把假泄漏排得比真泄漏更像真泄漏，应重点人工检查这些 center。")
    lines.append("4. 对于概率灰区、模型/rank冲突、final=SUSPECT 的样本，工程上建议不要强行二分类，而应输出 SUSPECT。")
    lines.append("5. 如果所有诊断都显示特征重叠，下一步应检查标签、center位置对应关系，以及从原始 8方向×8距离能量矩阵提取空间形态特征。")

    report_path = os.path.join(output_dir, "v9_144226_report.txt")
    save_text(report_path, "\n".join(lines))

    return report_path


# ============================================================
# 7. 主程序
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)

    print("=" * 100)
    print("v9：HM20260626_144226.ld 专项诊断程序")
    print("=" * 100)

    # 1. 读取预测
    pred_df, pred_path = load_predictions()
    print("预测文件:", pred_path)

    pred_info = diagnose_predictions(pred_df, OUTPUT_DIR)

    print("144226 样本数:", len(pred_info["pred_target_df"]))
    print("误判样本:", pred_info["wrong_csv"])
    print("冲突/SUSPECT样本:", pred_info["conflict_csv"])
    print("按center配对预测:", pred_info["pair_pred_csv"])

    # 2. 读取特征表
    tables = load_feature_tables()

    feature_diags = []
    input_paths = {
        "prediction": pred_path,
    }

    v7_sub = None
    raw_sub = None
    hm_sub = None

    if "v7_feature" in tables:
        df_v7, path_v7 = tables["v7_feature"]
        input_paths["v7_feature"] = path_v7
        diag = diagnose_feature_table(
            df_v7,
            "v7_feature",
            OUTPUT_DIR,
            key_features=KEY_FEATURES_CANDIDATES
        )
        feature_diags.append(diag)
        v7_sub = diag["target_df"]
        print("v7特征对比:", diag["compare_csv"])

    if "raw_feature" in tables:
        df_raw, path_raw = tables["raw_feature"]
        input_paths["raw_feature"] = path_raw
        diag = diagnose_feature_table(
            df_raw,
            "raw_feature",
            OUTPUT_DIR,
            key_features=KEY_FEATURES_CANDIDATES
        )
        feature_diags.append(diag)
        raw_sub = diag["target_df"]
        print("原始特征对比:", diag["compare_csv"])

    if "heatmap_feature" in tables:
        df_hm, path_hm = tables["heatmap_feature"]
        input_paths["heatmap_feature"] = path_hm
        diag = diagnose_feature_table(
            df_hm,
            "heatmap_feature",
            OUTPUT_DIR,
            key_features=KEY_HEATMAP_FEATURES
        )
        feature_diags.append(diag)
        hm_sub = diag["target_df"]
        print("heatmap特征对比:", diag["compare_csv"])

    # 3. 同 center 配对特征
    pair_feature_df, pair_feature_csv = make_center_pair_feature_compare(
        v7_sub,
        raw_sub,
        hm_sub,
        OUTPUT_DIR
    )
    print("按center配对特征对比:", pair_feature_csv)

    # 4. 画图
    plot_paths = []

    p = plot_prediction_probability(
        pred_info["pred_target_df"],
        pred_info["label_col"],
        OUTPUT_DIR
    )
    if p:
        plot_paths.append(p)

    p = plot_pair_prob_diff(pred_info["pair_pred_df"], OUTPUT_DIR)
    if p:
        plot_paths.append(p)

    for diag in feature_diags:
        plot_paths.extend(plot_feature_distributions(diag, OUTPUT_DIR, top_n=8))

    # 5. 报告
    report_path = make_report(
        pred_info,
        feature_diags,
        pair_feature_csv,
        OUTPUT_DIR,
        input_paths,
        plot_paths
    )

    print("\n" + "=" * 100)
    print("v9 诊断完成")
    print("=" * 100)
    print("输出文件夹:", OUTPUT_DIR)
    print("诊断报告:", report_path)

    print("\n重点文件:")
    print("  预测排序:", pred_info["pred_sorted_csv"])
    print("  误判样本:", pred_info["wrong_csv"])
    print("  冲突/SUSPECT样本:", pred_info["conflict_csv"])
    print("  center配对预测:", pred_info["pair_pred_csv"])
    print("  center配对特征:", pair_feature_csv)

    for diag in feature_diags:
        print(f"  {diag['source_name']} 对比:", diag["compare_csv"])

    print("\n图像输出:")
    for p in plot_paths:
        print(" ", p)

    # 命令行简要总结
    pred_sub = pred_info["pred_target_df"]
    if "model_correct" in pred_sub.columns and len(pred_sub):
        print("\n144226 简要结果:")
        print(f"  样本数: {len(pred_sub)}")
        print(f"  模型准确率: {pred_sub['model_correct'].astype(int).mean():.3f}")
        print(f"  误判数: {len(pred_info['wrong_df'])}")
        print(f"  冲突/SUSPECT数: {len(pred_info['conflict_df'])}")

    if len(pred_info["pair_pred_df"]) and "pair_order_correct" in pred_info["pair_pred_df"].columns:
        pv = pred_info["pair_pred_df"]["pair_order_correct"].dropna()
        if len(pv):
            print(f"  center配对排序正确率: {pv.mean():.3f}")

    print("\n请优先把 v9_144226_report.txt 里的“区分力前10特征”和“配对排序失败 center”发给我。")


if __name__ == "__main__":
    main()
