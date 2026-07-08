# -*- coding: utf-8 -*-
"""
leak_v11_analyze_144226_difference.py

目的：专项分析 HM20260626_144226.ld 到底哪里和其他三个时间点不一样。

它不训练新模型，只做诊断：
1. 预测层面：144226 的概率间隔、灰区、配对排序是否异常。
2. 分布层面：144226 的哪些特征整体分布明显偏离其他时间点。
3. 真假关系层面：哪些特征在 144226 内部 TRUE/FALSE 的方向与其他时间点相反。
4. center 层面：异常是否集中在 center_14 ~ center_18。
5. 自动输出 CSV、报告和图。

运行：
    python leak_v11_analyze_144226_difference.py

输出：
    C:\\Users\\jiangxinru6\\Desktop\\wurenji\\leak_v11_144226_difference_analysis_results
"""

import os
import math
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import ks_2samp
except Exception:
    ks_2samp = None

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None


# ============================================================
# 1. 路径配置
# ============================================================

BASE_DIR = r"C:\Users\jiangxinru6\Desktop\wurenji"
TARGET_TIME = "HM20260626_144226.ld"

OUTPUT_DIR = os.path.join(BASE_DIR, "leak_v11_144226_difference_analysis_results")

V7_FEATURE_CSV = os.path.join(
    BASE_DIR,
    "leak_v8_1_heatmap_shape_ablation_results",
    "v8_1_features_A_v7_only.csv"
)
V7_FEATURE_CSV_FALLBACK = os.path.join(
    BASE_DIR,
    "leak_v7_robust_feature_results",
    "v7_robust_feature_dataset.csv"
)

RAW_FEATURE_CSV = os.path.join(
    BASE_DIR,
    "leak_v4_compare_results",
    "merged_feature_dataset.csv"
)

HEATMAP_FEATURE_CSV = os.path.join(
    BASE_DIR,
    "leak_v8_1_heatmap_shape_ablation_results",
    "v8_1_heatmap_core_shape_features.csv"
)

PRED_CSV = os.path.join(
    BASE_DIR,
    "leak_v8_1_heatmap_shape_ablation_results",
    "A_v7_only",
    "A_v7_only_predictions.csv"
)
PRED_CSV_FALLBACK = os.path.join(
    BASE_DIR,
    "leak_v7_robust_feature_results",
    "v7_predictions.csv"
)

META_COLS = {
    "dataset", "label", "true_label", "time", "test_group", "center", "center_norm",
    "experiment", "row_index", "heatmap_path", "best_direction", "energy_direction",
    "decay_direction", "representative_file", "model_pred", "rank_binary_pred",
    "default_pred_0p5", "final_decision", "v7_model_pred", "v7_rank_binary_pred",
    "v7_final_decision"
}

KEY_FEATURES = [
    "spec_slope", "spec_slope__time_robust_z", "spec_slope__time_rank_pct",
    "ratio_60_70k", "ratio_60_70k__time_robust_z", "ratio_60_70k__time_rank_pct",
    "best_direction_combined_score", "best_direction_combined_score__time_robust_z", "best_direction_combined_score__time_rank_pct",
    "direction_contrast", "direction_contrast__time_robust_z", "direction_contrast__time_rank_pct",
    "high_freq_ratio", "high_freq_ratio__time_robust_z", "high_freq_ratio__time_rank_pct",
    "decay_R2", "decay_R2__time_robust_z", "decay_R2__time_rank_pct",
    "near_far_ratio", "near_far_ratio__time_robust_z", "near_far_ratio__time_rank_pct",
    "spec_flatness", "time_energy_cv",
    "hm_core_to_outer_energy_ratio", "hm_radial_spread_norm", "hm_diffuse_score",
    "hm_entropy_2d", "hm_energy_concentration_top10", "hm_energy_concentration_top5",
    "hm_weighted_elongation", "hm_shape_leak_like_score"
]

MIN_EFFECT_AUC = 0.60
MIN_EFFECT_ABS_D = 0.30
TOP_N_PLOTS = 12


# ============================================================
# 2. 工具函数
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def find_existing_path(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


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
    return digits.zfill(2) if digits else s


def safe_num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def safe_filename(s):
    return "".join(ch if (ch.isalnum() or ch in "_-." ) else "_" for ch in str(s))[:150]


def numeric_cols(df):
    cols = []
    for c in df.columns:
        if c in META_COLS:
            continue
        vals = safe_num(df[c])
        if vals.notna().mean() > 0.8:
            cols.append(c)
    return cols


def cohen_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = math.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2.0) + 1e-12
    return float((np.mean(a) - np.mean(b)) / pooled)


def hist_overlap(a, b, bins=20):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    allv = np.concatenate([a, b])
    lo, hi = np.min(allv), np.max(allv)
    if abs(hi - lo) < 1e-12:
        return 1.0
    ha, edges = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    return float(np.clip(np.sum(np.minimum(ha, hb)) * (edges[1] - edges[0]), 0, 1))


def ks_stat(a, b):
    if ks_2samp is None:
        return np.nan, np.nan
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan
    r = ks_2samp(a, b)
    return float(r.statistic), float(r.pvalue)


def auc_signed(true_vals, false_vals):
    if roc_auc_score is None:
        return np.nan
    true_vals = np.asarray(true_vals, dtype=float)
    false_vals = np.asarray(false_vals, dtype=float)
    true_vals = true_vals[np.isfinite(true_vals)]
    false_vals = false_vals[np.isfinite(false_vals)]
    if len(true_vals) == 0 or len(false_vals) == 0:
        return np.nan
    y = np.r_[np.ones(len(true_vals)), np.zeros(len(false_vals))]
    x = np.r_[true_vals, false_vals]
    try:
        return float(roc_auc_score(y, x))
    except Exception:
        return np.nan


def direction_from_diff(diff, d, auc_free):
    effective = False
    if np.isfinite(d) and abs(d) >= MIN_EFFECT_ABS_D:
        effective = True
    if np.isfinite(auc_free) and auc_free >= MIN_EFFECT_AUC:
        effective = True
    if not effective or abs(diff) < 1e-15:
        return "weak", 0, False
    if diff > 0:
        return "TRUE>FALSE", 1, True
    return "TRUE<FALSE", -1, True


# ============================================================
# 3. 读取并合并数据
# ============================================================

def read_standard_table(path, fallback=None, name="table"):
    p = find_existing_path(path, fallback)
    if p is None:
        print(f"[警告] 找不到 {name}: {path}")
        return None, None
    df = pd.read_csv(p)
    if "test_group" in df.columns and "time" not in df.columns:
        df = df.rename(columns={"test_group": "time"})
    if "true_label" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"true_label": "label"})
    if "label" in df.columns:
        df["label"] = df["label"].astype(str)
    if "center" in df.columns:
        df["center_norm"] = df["center"].apply(normalize_center_id)
    return df, p


def build_table():
    v7, v7_path = read_standard_table(V7_FEATURE_CSV, V7_FEATURE_CSV_FALLBACK, "v7 feature")
    if v7 is None:
        raise FileNotFoundError("找不到 v7 特征表，请先运行 v8.1 或 v7。")
    required = {"time", "label", "center_norm"}
    missing = required - set(v7.columns)
    if missing:
        raise ValueError(f"v7特征表缺列: {missing}")
    v7 = v7[v7["label"].isin(["TRUE_LEAK", "FALSE_LEAK"])].copy().reset_index(drop=True)
    df = v7.copy()
    input_paths = {"v7_feature": v7_path}

    raw, raw_path = read_standard_table(RAW_FEATURE_CSV, None, "raw feature")
    if raw is not None and {"time", "label", "center_norm"}.issubset(raw.columns):
        raw = raw[raw["label"].isin(["TRUE_LEAK", "FALSE_LEAK"])].copy()
        add = [c for c in numeric_cols(raw) if c not in df.columns]
        if add:
            df = df.merge(raw[["time", "label", "center_norm"] + add], on=["time", "label", "center_norm"], how="left")
        input_paths["raw_feature"] = raw_path
    else:
        input_paths["raw_feature"] = "未找到/未使用"

    hm, hm_path = read_standard_table(HEATMAP_FEATURE_CSV, None, "heatmap feature")
    if hm is not None and {"time", "label", "center_norm"}.issubset(hm.columns):
        hm = hm[hm["label"].isin(["TRUE_LEAK", "FALSE_LEAK"])].copy()
        hm_cols = [c for c in numeric_cols(hm) if c.startswith("hm_") and c != "hm_read_success"]
        hm_small = hm[["time", "label", "center_norm"] + hm_cols].copy()
        rename = {}
        for c in hm_cols:
            if c in df.columns:
                rename[c] = c + "__hm"
        hm_small = hm_small.rename(columns=rename)
        df = df.merge(hm_small, on=["time", "label", "center_norm"], how="left")
        input_paths["heatmap_feature"] = hm_path
    else:
        input_paths["heatmap_feature"] = "未找到/未使用"

    pred, pred_path = read_standard_table(PRED_CSV, PRED_CSV_FALLBACK, "prediction")
    if pred is not None and {"time", "center_norm"}.issubset(pred.columns):
        if "label" not in pred.columns and "true_label" in pred.columns:
            pred["label"] = pred["true_label"]
        if "v7_model_pred" in pred.columns and "model_pred" not in pred.columns:
            pred["model_pred"] = pred["v7_model_pred"]
        if "v7_rank_binary_pred" in pred.columns and "rank_binary_pred" not in pred.columns:
            pred["rank_binary_pred"] = pred["v7_rank_binary_pred"]
        if "v7_final_decision" in pred.columns and "final_decision" not in pred.columns:
            pred["final_decision"] = pred["v7_final_decision"]
        keep = [c for c in [
            "time", "label", "center_norm", "prob_TRUE_LEAK", "best_threshold",
            "default_pred_0p5", "model_pred", "rank_binary_pred", "final_decision",
            "model_correct", "default_correct"
        ] if c in pred.columns]
        if "label" in keep:
            df = df.merge(pred[keep], on=["time", "label", "center_norm"], how="left")
        else:
            df = df.merge(pred[keep], on=["time", "center_norm"], how="left")
        input_paths["prediction"] = pred_path
    else:
        input_paths["prediction"] = "未找到/未使用"
    return df, input_paths


# ============================================================
# 4. 分析：预测差异
# ============================================================

def prediction_difference(df, outdir):
    rows = []
    if "prob_TRUE_LEAK" not in df.columns:
        out = pd.DataFrame()
        path = os.path.join(outdir, "v11_144226_prediction_difference.csv")
        out.to_csv(path, index=False, encoding="utf-8-sig")
        return out, path

    for t, g in df.groupby("time"):
        prob = safe_num(g["prob_TRUE_LEAK"])
        lab = g["label"].astype(str)
        tp = prob[lab == "TRUE_LEAK"].dropna()
        fp = prob[lab == "FALSE_LEAK"].dropna()
        row = {
            "time": t,
            "n": len(g),
            "n_true": int((lab == "TRUE_LEAK").sum()),
            "n_false": int((lab == "FALSE_LEAK").sum()),
            "true_prob_mean": float(tp.mean()) if len(tp) else np.nan,
            "false_prob_mean": float(fp.mean()) if len(fp) else np.nan,
            "prob_gap_TRUE_minus_FALSE": float(tp.mean() - fp.mean()) if len(tp) and len(fp) else np.nan,
            "gray_zone_0p4_0p6_count": int(((prob >= 0.4) & (prob <= 0.6)).sum()),
            "gray_zone_0p4_0p6_ratio": float(((prob >= 0.4) & (prob <= 0.6)).mean()),
        }
        if "model_correct" in g.columns:
            row["model_accuracy"] = float(safe_num(g["model_correct"]).fillna(0).mean())
        pair_rows = []
        for c, gc in g.groupby("center_norm"):
            tr = gc[gc["label"] == "TRUE_LEAK"]
            fa = gc[gc["label"] == "FALSE_LEAK"]
            if len(tr) and len(fa):
                tprob = pd.to_numeric(pd.Series([tr.iloc[0].get("prob_TRUE_LEAK", np.nan)]), errors="coerce").iloc[0]
                fprob = pd.to_numeric(pd.Series([fa.iloc[0].get("prob_TRUE_LEAK", np.nan)]), errors="coerce").iloc[0]
                if np.isfinite(tprob) and np.isfinite(fprob):
                    pair_rows.append((c, tprob - fprob, int(tprob > fprob)))
        if pair_rows:
            row["pair_prob_order_accuracy"] = float(np.mean([r[2] for r in pair_rows]))
            row["pair_prob_order_failed_centers"] = " | ".join([str(r[0]) for r in pair_rows if r[2] == 0])
        rows.append(row)
    out = pd.DataFrame(rows)
    out["is_target_144226"] = (out["time"].astype(str) == TARGET_TIME).astype(int)
    path = os.path.join(outdir, "v11_144226_prediction_difference.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out, path


# ============================================================
# 5. 分析：整体分布偏移
# ============================================================

def feature_shift(df, features, outdir):
    target = df[df["time"].astype(str) == TARGET_TIME]
    others = df[df["time"].astype(str) != TARGET_TIME]
    rows = []
    for f in features:
        a = safe_num(target[f]).dropna().values
        b = safe_num(others[f]).dropna().values
        if len(a) < 2 or len(b) < 2:
            continue
        d = cohen_d(a, b)
        ov = hist_overlap(a, b)
        ks, ksp = ks_stat(a, b)
        row = {
            "feature": f,
            "target_mean": float(np.mean(a)),
            "others_mean": float(np.mean(b)),
            "diff_target_minus_others": float(np.mean(a) - np.mean(b)),
            "target_median": float(np.median(a)),
            "others_median": float(np.median(b)),
            "target_std": float(np.std(a, ddof=1)),
            "others_std": float(np.std(b, ddof=1)),
            "cohen_d_target_vs_others": d,
            "abs_cohen_d": abs(d) if np.isfinite(d) else np.nan,
            "hist_overlap_high_is_similar": ov,
            "ks_statistic": ks,
            "ks_pvalue": ksp,
            "shift_score": (abs(d) if np.isfinite(d) else 0) + (ks if np.isfinite(ks) else 0) + (1 - ov if np.isfinite(ov) else 0),
        }
        for label in ["TRUE_LEAK", "FALSE_LEAK"]:
            ta = safe_num(target.loc[target["label"] == label, f]).dropna().values
            ob = safe_num(others.loc[others["label"] == label, f]).dropna().values
            if len(ta) >= 2 and len(ob) >= 2:
                dl = cohen_d(ta, ob)
                row[f"{label}_target_mean"] = float(np.mean(ta))
                row[f"{label}_others_mean"] = float(np.mean(ob))
                row[f"{label}_diff_target_minus_others"] = float(np.mean(ta) - np.mean(ob))
                row[f"{label}_abs_cohen_d"] = abs(dl) if np.isfinite(dl) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("shift_score", ascending=False)
    path = os.path.join(outdir, "v11_144226_feature_shift_vs_others.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out, path


# ============================================================
# 6. 分析：TRUE/FALSE 方向是否和别人相反
# ============================================================

def group_separation(df, feature, time_value):
    g = df[df["time"].astype(str) == str(time_value)]
    lab = g["label"].astype(str)
    vals = safe_num(g[feature])
    tv = vals[lab == "TRUE_LEAK"].dropna().values
    fv = vals[lab == "FALSE_LEAK"].dropna().values
    if len(tv) < 2 or len(fv) < 2:
        return None
    tm, fm = float(np.mean(tv)), float(np.mean(fv))
    diff = tm - fm
    d = cohen_d(tv, fv)
    auc = auc_signed(tv, fv)
    auc_free = max(auc, 1 - auc) if np.isfinite(auc) else np.nan
    direction, sign, effective = direction_from_diff(diff, d, auc_free)
    return {
        "time": time_value,
        "true_mean": tm,
        "false_mean": fm,
        "diff_TRUE_minus_FALSE": diff,
        "cohen_d_TRUE_minus_FALSE": d,
        "abs_cohen_d": abs(d) if np.isfinite(d) else np.nan,
        "auc_signed_TRUE_larger": auc,
        "auc_direction_free": auc_free,
        "hist_overlap": hist_overlap(tv, fv),
        "direction": direction,
        "sign": sign,
        "effective": int(effective),
    }


def direction_analysis(df, features, outdir):
    times = sorted(df["time"].astype(str).unique().tolist())
    rows = []
    detail = []
    for f in features:
        stats = {}
        for t in times:
            st = group_separation(df, f, t)
            if st is not None:
                stats[t] = st
                detail.append({"feature": f, **st})
        if TARGET_TIME not in stats:
            continue
        target = stats[TARGET_TIME]
        other = [stats[t] for t in times if t != TARGET_TIME and t in stats]
        signs = [s["sign"] for s in other if s["effective"] and s["sign"] != 0]
        pos, neg = sum(1 for s in signs if s > 0), sum(1 for s in signs if s < 0)
        if pos > neg:
            other_sign, other_dir = 1, "TRUE>FALSE"
        elif neg > pos:
            other_sign, other_dir = -1, "TRUE<FALSE"
        else:
            other_sign, other_dir = 0, "none"
        flip = int(target["effective"] and target["sign"] != 0 and other_sign != 0 and target["sign"] != other_sign)
        row = {
            "feature": f,
            "target_direction": target["direction"],
            "target_sign": target["sign"],
            "target_true_mean": target["true_mean"],
            "target_false_mean": target["false_mean"],
            "target_diff_TRUE_minus_FALSE": target["diff_TRUE_minus_FALSE"],
            "target_auc_direction_free": target["auc_direction_free"],
            "target_abs_cohen_d": target["abs_cohen_d"],
            "others_majority_direction": other_dir,
            "others_majority_sign": other_sign,
            "others_pos_count_TRUE_gt_FALSE": pos,
            "others_neg_count_TRUE_lt_FALSE": neg,
            "others_mean_diff_TRUE_minus_FALSE": float(np.nanmean([s["diff_TRUE_minus_FALSE"] for s in other])) if other else np.nan,
            "others_mean_auc_direction_free": float(np.nanmean([s["auc_direction_free"] for s in other])) if other else np.nan,
            "is_direction_flip_vs_others": flip,
            "direction_change_score": (target["auc_direction_free"] if np.isfinite(target["auc_direction_free"]) else 0.5) + (target["abs_cohen_d"] if np.isfinite(target["abs_cohen_d"]) else 0) + (1.5 if flip else 0),
            "sign_sequence": " | ".join([f"{t}:{stats[t]['direction']}" for t in times if t in stats]),
        }
        for t in times:
            if t in stats:
                p = t.replace(".", "_")
                row[f"{p}_direction"] = stats[t]["direction"]
                row[f"{p}_diff"] = stats[t]["diff_TRUE_minus_FALSE"]
                row[f"{p}_auc_free"] = stats[t]["auc_direction_free"]
        rows.append(row)
    out = pd.DataFrame(rows)
    det = pd.DataFrame(detail)
    if len(out):
        out = out.sort_values(["is_direction_flip_vs_others", "direction_change_score"], ascending=[False, False])
    sep_path = os.path.join(outdir, "v11_144226_label_separation_vs_others.csv")
    detail_path = os.path.join(outdir, "v11_feature_label_separation_detail_by_time.csv")
    flip_path = os.path.join(outdir, "v11_144226_direction_flip_features.csv")
    out.to_csv(sep_path, index=False, encoding="utf-8-sig")
    det.to_csv(detail_path, index=False, encoding="utf-8-sig")
    out[out["is_direction_flip_vs_others"] == 1].to_csv(flip_path, index=False, encoding="utf-8-sig") if len(out) else pd.DataFrame().to_csv(flip_path, index=False, encoding="utf-8-sig")
    return out, sep_path, detail_path, flip_path


# ============================================================
# 7. 分析：center 配对异常
# ============================================================

def build_pair_diff(df, features):
    rows = []
    for (t, c), g in df.groupby(["time", "center_norm"]):
        tr = g[g["label"] == "TRUE_LEAK"]
        fa = g[g["label"] == "FALSE_LEAK"]
        if len(tr) == 0 or len(fa) == 0:
            continue
        tr, fa = tr.iloc[0], fa.iloc[0]
        row = {"time": t, "center_norm": c}
        if "prob_TRUE_LEAK" in df.columns:
            tp = pd.to_numeric(pd.Series([tr.get("prob_TRUE_LEAK", np.nan)]), errors="coerce").iloc[0]
            fp = pd.to_numeric(pd.Series([fa.get("prob_TRUE_LEAK", np.nan)]), errors="coerce").iloc[0]
            row["prob_diff_TRUE_minus_FALSE"] = tp - fp if np.isfinite(tp) and np.isfinite(fp) else np.nan
            row["prob_pair_order_correct"] = int(tp > fp) if np.isfinite(tp) and np.isfinite(fp) else np.nan
        for f in features:
            if f in df.columns:
                tv = pd.to_numeric(pd.Series([tr.get(f, np.nan)]), errors="coerce").iloc[0]
                fv = pd.to_numeric(pd.Series([fa.get(f, np.nan)]), errors="coerce").iloc[0]
                if np.isfinite(tv) and np.isfinite(fv):
                    row[f"{f}__pair_diff"] = tv - fv
        rows.append(row)
    return pd.DataFrame(rows)


def center_anomaly(df, features, outdir):
    pair = build_pair_diff(df, features)
    pair_path = os.path.join(outdir, "v11_all_time_center_pair_diff_table.csv")
    pair.to_csv(pair_path, index=False, encoding="utf-8-sig")
    if len(pair) == 0:
        out = pd.DataFrame()
        path = os.path.join(outdir, "v11_144226_center_pair_anomaly.csv")
        out.to_csv(path, index=False, encoding="utf-8-sig")
        return out, path
    target = pair[pair["time"].astype(str) == TARGET_TIME]
    others = pair[pair["time"].astype(str) != TARGET_TIME]
    diff_cols = [c for c in pair.columns if c.endswith("__pair_diff")]
    if "prob_diff_TRUE_minus_FALSE" in pair.columns:
        diff_cols = ["prob_diff_TRUE_minus_FALSE"] + diff_cols
    rows = []
    for _, r in target.iterrows():
        row = {
            "center_norm": r["center_norm"],
            "is_center_14_to_18": int(str(r["center_norm"]) in ["14", "15", "16", "17", "18"]),
        }
        if "prob_diff_TRUE_minus_FALSE" in r.index:
            row["prob_diff_TRUE_minus_FALSE"] = r.get("prob_diff_TRUE_minus_FALSE", np.nan)
            row["prob_pair_order_correct"] = r.get("prob_pair_order_correct", np.nan)
        zs = []
        for col in diff_cols:
            val = r.get(col, np.nan)
            ov = safe_num(others[col]).dropna().values if col in others.columns else []
            if np.isfinite(val) and len(ov) >= 2:
                mu = float(np.mean(ov))
                sd = float(np.std(ov, ddof=1)) + 1e-12
                z = (float(val) - mu) / sd
                base = col.replace("__pair_diff", "")
                row[f"{base}__144226_pair_diff"] = float(val)
                row[f"{base}__others_pair_diff_mean"] = mu
                row[f"{base}__pair_diff_z_vs_others"] = float(z)
                zs.append(abs(z))
        row["mean_abs_pair_diff_z"] = float(np.mean(zs)) if zs else np.nan
        row["max_abs_pair_diff_z"] = float(np.max(zs)) if zs else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["is_center_14_to_18", "max_abs_pair_diff_z"], ascending=[False, False])
    path = os.path.join(outdir, "v11_144226_center_pair_anomaly.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out, path


# ============================================================
# 8. 作图
# ============================================================

def plot_prob_summary(pred_df, outdir):
    figdir = os.path.join(outdir, "figures")
    ensure_dir(figdir)
    paths = []
    if len(pred_df) == 0:
        return paths
    x = pred_df["time"].astype(str).str.replace("HM20260626_", "", regex=False).str.replace(".ld", "", regex=False)
    if "prob_gap_TRUE_minus_FALSE" in pred_df.columns:
        plt.figure(figsize=(9, 5))
        plt.bar(x, pred_df["prob_gap_TRUE_minus_FALSE"])
        plt.axhline(0, linestyle="--", linewidth=1)
        plt.ylabel("TRUE mean prob - FALSE mean prob")
        plt.title("Prediction probability gap by time")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        p = os.path.join(figdir, "v11_prediction_prob_gap_by_time.png")
        plt.savefig(p, dpi=150)
        plt.close()
        paths.append(p)
    if "gray_zone_0p4_0p6_ratio" in pred_df.columns:
        plt.figure(figsize=(9, 5))
        plt.bar(x, pred_df["gray_zone_0p4_0p6_ratio"])
        plt.ylabel("Gray zone ratio [0.4,0.6]")
        plt.title("Uncertain probability ratio by time")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        p = os.path.join(figdir, "v11_gray_zone_ratio_by_time.png")
        plt.savefig(p, dpi=150)
        plt.close()
        paths.append(p)
    return paths


def plot_feature_by_time(df, features, outdir, prefix="feature"):
    figdir = os.path.join(outdir, "figures")
    ensure_dir(figdir)
    paths = []
    times = sorted(df["time"].astype(str).unique().tolist())
    for f in features:
        if f not in df.columns:
            continue
        data = []
        labels = []
        for t in times:
            vals = safe_num(df.loc[df["time"].astype(str) == t, f]).dropna().values
            if len(vals):
                data.append(vals)
                labels.append(t.replace("HM20260626_", "").replace(".ld", ""))
        if len(data) < 2:
            continue
        plt.figure(figsize=(10, 5))
        plt.boxplot(data, labels=labels, showfliers=True)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(f)
        plt.title(f"{f} by time")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        p = os.path.join(figdir, f"{prefix}_{safe_filename(f)}_by_time.png")
        plt.savefig(p, dpi=150)
        plt.close()
        paths.append(p)
    return paths


def plot_label_by_time(df, features, outdir, prefix="label"):
    figdir = os.path.join(outdir, "figures")
    ensure_dir(figdir)
    paths = []
    times = sorted(df["time"].astype(str).unique().tolist())
    for f in features:
        if f not in df.columns:
            continue
        data, labels, positions = [], [], []
        pos = 1
        for t in times:
            short = t.replace("HM20260626_", "").replace(".ld", "")
            for lab in ["FALSE_LEAK", "TRUE_LEAK"]:
                vals = safe_num(df.loc[(df["time"].astype(str) == t) & (df["label"] == lab), f]).dropna().values
                if len(vals):
                    data.append(vals)
                    labels.append(short + "\n" + lab.replace("_LEAK", ""))
                    positions.append(pos)
                    pos += 1
            pos += 0.6
        if len(data) < 2:
            continue
        plt.figure(figsize=(12, 5))
        plt.boxplot(data, positions=positions, labels=labels, showfliers=True)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.ylabel(f)
        plt.title(f"TRUE/FALSE distribution by time - {f}")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        p = os.path.join(figdir, f"{prefix}_{safe_filename(f)}_label_by_time.png")
        plt.savefig(p, dpi=150)
        plt.close()
        paths.append(p)
    return paths


# ============================================================
# 9. 报告
# ============================================================

def make_report(df, paths, pred_df, pred_csv, shift_df, shift_csv, sep_df, sep_csv, flip_csv, center_df, center_csv, plot_paths, outdir):
    lines = []
    lines.append("v11：144226 到底哪里和其他时间点不一样 —— 诊断报告")
    lines.append("=" * 120)
    lines.append(f"生成时间: {datetime.now()}")
    lines.append(f"目标时间点: {TARGET_TIME}")
    lines.append("")
    lines.append("输入文件:")
    for k, v in paths.items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    lines.append("一、样本概况")
    lines.append("-" * 120)
    for t, g in df.groupby("time"):
        lines.append(f"  {t}: n={len(g)}, TRUE={(g['label']=='TRUE_LEAK').sum()}, FALSE={(g['label']=='FALSE_LEAK').sum()}")
    lines.append("")

    lines.append("二、预测层面差异")
    lines.append("-" * 120)
    lines.append(f"预测差异表: {pred_csv}")
    if len(pred_df):
        tr = pred_df[pred_df["time"].astype(str) == TARGET_TIME]
        if len(tr):
            r = tr.iloc[0]
            lines.append(f"  144226 TRUE平均概率: {r.get('true_prob_mean', np.nan):.4f}")
            lines.append(f"  144226 FALSE平均概率: {r.get('false_prob_mean', np.nan):.4f}")
            lines.append(f"  144226 TRUE-FALSE概率差: {r.get('prob_gap_TRUE_minus_FALSE', np.nan):.4f}")
            lines.append(f"  144226 灰区比例[0.4,0.6]: {r.get('gray_zone_0p4_0p6_ratio', np.nan):.4f}")
            lines.append(f"  144226 center配对排序正确率: {r.get('pair_prob_order_accuracy', np.nan)}")
            lines.append(f"  144226 配对失败center: {r.get('pair_prob_order_failed_centers', '')}")
    lines.append("")

    lines.append("三、整体特征分布偏移：144226 哪些特征整体和别人不一样")
    lines.append("-" * 120)
    lines.append(f"分布偏移表: {shift_csv}")
    if len(shift_df):
        lines.append("  偏移最大的前15个特征:")
        for _, r in shift_df.head(15).iterrows():
            lines.append(f"    {r['feature']}: shift={r['shift_score']:.3f}, |d|={r['abs_cohen_d']:.3f}, KS={r['ks_statistic']:.3f}, 144226均值={r['target_mean']:.6g}, 其他均值={r['others_mean']:.6g}")
    lines.append("")

    lines.append("四、TRUE/FALSE关系：哪些特征在144226里方向和别人相反")
    lines.append("-" * 120)
    lines.append(f"方向对比表: {sep_csv}")
    lines.append(f"方向翻转特征表: {flip_csv}")
    if len(sep_df):
        flip = sep_df[sep_df["is_direction_flip_vs_others"] == 1]
        lines.append(f"  方向翻转特征数量: {len(flip)}")
        if len(flip):
            lines.append("  方向翻转前15:")
            for _, r in flip.head(15).iterrows():
                lines.append(f"    {r['feature']}: 144226={r['target_direction']}, 其他多数={r['others_majority_direction']}, 144226_AUC={r['target_auc_direction_free']:.3f}, 144226_diff={r['target_diff_TRUE_minus_FALSE']:.6g}, 其他平均diff={r['others_mean_diff_TRUE_minus_FALSE']:.6g}")
    lines.append("")

    lines.append("五、center层面：异常是否集中在 center_14~18")
    lines.append("-" * 120)
    lines.append(f"center配对异常表: {center_csv}")
    if len(center_df):
        lines.append("  center异常前10:")
        for _, r in center_df.head(10).iterrows():
            lines.append(f"    center_{r['center_norm']}: 14-18={int(r.get('is_center_14_to_18',0))}, prob_diff={r.get('prob_diff_TRUE_minus_FALSE', np.nan)}, order_correct={r.get('prob_pair_order_correct', np.nan)}, max_z={r.get('max_abs_pair_diff_z', np.nan):.3f}")
    lines.append("")

    lines.append("六、如何解释")
    lines.append("-" * 120)
    lines.append("  如果预测概率差小、灰区比例高，说明 v7 对 144226 本身不确信。")
    lines.append("  如果分布偏移表中关键特征排前面，说明 144226 有整体工况/特征分布漂移。")
    lines.append("  如果方向翻转表中出现 spec_slope、direction_contrast、best_direction_combined_score 等核心特征，说明 144226 的真假规律和其他时间点不一致。")
    lines.append("  如果 center_14~18 在 center异常表中排前面，说明问题具有局部空间集中性，不是随机错误。")
    lines.append("  总结：144226 不是简单检测不到，而是特征分布、真假方向或局部center表现与其他时间点不一致，导致 v7 学到的通用规则失效。")
    lines.append("")

    lines.append("七、图像输出")
    lines.append("-" * 120)
    for p in plot_paths:
        lines.append(f"  {p}")

    report = os.path.join(outdir, "v11_report.txt")
    save_text(report, "\n".join(lines))
    return report


# ============================================================
# 10. 主程序
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)
    print("=" * 120)
    print("v11：分析 HM20260626_144226.ld 到底哪里和其他时间点不一样")
    print("=" * 120)

    df, input_paths = build_table()
    print("样本数量:", len(df))
    print(df["label"].value_counts())
    print("time groups:", sorted(df["time"].astype(str).unique().tolist()))

    exclude = {"prob_TRUE_LEAK", "best_threshold", "model_correct", "default_correct"}
    features = [c for c in numeric_cols(df) if c not in exclude]
    print("可分析数值特征数量:", len(features))

    pred_df, pred_csv = prediction_difference(df, OUTPUT_DIR)
    print("预测差异表:", pred_csv)

    shift_df, shift_csv = feature_shift(df, features, OUTPUT_DIR)
    print("特征分布偏移表:", shift_csv)

    sep_df, sep_csv, sep_detail_csv, flip_csv = direction_analysis(df, features, OUTPUT_DIR)
    print("TRUE/FALSE方向对比表:", sep_csv)
    print("方向翻转特征表:", flip_csv)

    top_shift = shift_df["feature"].head(20).tolist() if len(shift_df) else []
    top_flip = sep_df.loc[sep_df["is_direction_flip_vs_others"] == 1, "feature"].head(20).tolist() if len(sep_df) else []
    pair_features = []
    for f in KEY_FEATURES + top_shift + top_flip:
        if f in df.columns and f not in pair_features:
            pair_features.append(f)

    center_df, center_csv = center_anomaly(df, pair_features, OUTPUT_DIR)
    print("center配对异常表:", center_csv)

    plot_paths = []
    plot_paths += plot_prob_summary(pred_df, OUTPUT_DIR)

    plot_shift = shift_df["feature"].head(TOP_N_PLOTS).tolist() if len(shift_df) else []
    plot_features = []
    for f in KEY_FEATURES + plot_shift:
        if f in df.columns and f not in plot_features:
            plot_features.append(f)
    plot_paths += plot_feature_by_time(df, plot_features[:TOP_N_PLOTS], OUTPUT_DIR, prefix="v11_shift")

    plot_dir_features = []
    if len(sep_df):
        fl = sep_df.loc[sep_df["is_direction_flip_vs_others"] == 1, "feature"].head(TOP_N_PLOTS).tolist()
        for f in fl + KEY_FEATURES:
            if f in df.columns and f not in plot_dir_features:
                plot_dir_features.append(f)
    plot_paths += plot_label_by_time(df, plot_dir_features[:TOP_N_PLOTS], OUTPUT_DIR, prefix="v11_direction")

    report = make_report(
        df, input_paths, pred_df, pred_csv, shift_df, shift_csv,
        sep_df, sep_csv, flip_csv, center_df, center_csv, plot_paths, OUTPUT_DIR
    )

    print("\n" + "=" * 120)
    print("v11 分析完成")
    print("=" * 120)
    print("输出文件夹:", OUTPUT_DIR)
    print("总报告:", report)
    print("\n重点文件:")
    print("  预测差异:", pred_csv)
    print("  特征分布偏移:", shift_csv)
    print("  TRUE/FALSE方向对比:", sep_csv)
    print("  方向翻转特征:", flip_csv)
    print("  center配对异常:", center_csv)

    print("\n命令行摘要:")
    if len(pred_df):
        tr = pred_df[pred_df["time"].astype(str) == TARGET_TIME]
        if len(tr):
            r = tr.iloc[0]
            print(f"  144226 TRUE-FALSE概率差: {r.get('prob_gap_TRUE_minus_FALSE', np.nan):.4f}")
            print(f"  144226 灰区比例[0.4,0.6]: {r.get('gray_zone_0p4_0p6_ratio', np.nan):.4f}")
            print(f"  144226 center配对排序正确率: {r.get('pair_prob_order_accuracy', np.nan)}")
            print(f"  144226 配对失败center: {r.get('pair_prob_order_failed_centers', '')}")

    if len(shift_df):
        print("\n  144226整体分布偏移最大的前8个特征:")
        for _, r in shift_df.head(8).iterrows():
            print(f"    {r['feature']}: shift={r['shift_score']:.3f}, |d|={r['abs_cohen_d']:.3f}, 144226均值={r['target_mean']:.6g}, 其他均值={r['others_mean']:.6g}")

    if len(sep_df):
        flip = sep_df[sep_df["is_direction_flip_vs_others"] == 1]
        print(f"\n  方向翻转特征数量: {len(flip)}")
        for _, r in flip.head(8).iterrows():
            print(f"    {r['feature']}: 144226={r['target_direction']}, 其他多数={r['others_majority_direction']}, 144226_AUC={r['target_auc_direction_free']:.3f}")

    if len(center_df):
        print("\n  center异常前8:")
        for _, r in center_df.head(8).iterrows():
            print(f"    center_{r['center_norm']}: 14-18={int(r.get('is_center_14_to_18',0))}, prob_diff={r.get('prob_diff_TRUE_minus_FALSE', np.nan)}, order_correct={r.get('prob_pair_order_correct', np.nan)}, max_z={r.get('max_abs_pair_diff_z', np.nan):.3f}")

    print("\n图像输出:")
    for p in plot_paths[:20]:
        print(" ", p)
    if len(plot_paths) > 20:
        print(f"  ... 共 {len(plot_paths)} 张图")

    print("\n请把命令行摘要和 v11_report.txt 的第2~5部分发给我，我帮你解释144226具体异常在哪里。")


if __name__ == "__main__":
    main()
