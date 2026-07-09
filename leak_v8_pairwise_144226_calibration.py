# -*- coding: utf-8 -*-
"""
leak_v8_pairwise_144226_calibration.py

v8 144226 专用配对校准版
------------------------------------------------------------
不重新读取 WAV，不重复计算已有特征。
直接复用已有 v8 输出，针对 HM20260626_144226.ld 做 TRUE/FALSE center 配对校准。

核心：
    如果某个特征在 144226 上 TRUE 稳定小于 FALSE，说明它仍然有用，只是方向反了，程序会自动乘 -1。

输出目录：
    C:\\Users\\jiangxinru6\\Desktop\\wurenji\\leak_v8_pairwise_144226_calibration_results

重点看：
    v8_pairwise_report.txt
    v8_pairwise_summary.csv
    v8_pairwise_full_predictions.csv
    v8_pairwise_loocv_predictions.csv
    v8_pairwise_selected_features.csv
    v8_pairwise_feature_ranking.csv
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


# ============================================================
# 1. 配置
# ============================================================

BASE_DIR = r"C:\Users\jiangxinru6\Desktop\wurenji"
TARGET_TIME = "HM20260626_144226.ld"

OUTPUT_DIR = os.path.join(BASE_DIR, "leak_v8_pairwise_144226_calibration_results")

DATASET_CANDIDATES = [
    os.path.join(BASE_DIR, "leak_v8_fixed_global_matrix_results", "v8_fixed_dataset.csv"),
    os.path.join(BASE_DIR, "leak_v8_standalone_directed_wideband_auto_center_v2_results", "v8_feature_dataset_with_time_relative.csv"),
    os.path.join(BASE_DIR, "leak_v8_standalone_directed_wideband_auto_center_v2_results", "v8_feature_dataset.csv"),
]

PREDICTION_CANDIDATES = [
    os.path.join(BASE_DIR, "leak_v8_fixed_global_matrix_results", "v8_fixed_predictions.csv"),
    os.path.join(BASE_DIR, "leak_v8_standalone_directed_wideband_auto_center_v2_results", "v8_predictions.csv"),
]

MAX_SELECTED_FEATURES = 15
MIN_FEATURE_PAIR_ACC = 0.85
MAX_CORR = 0.96

META_COL_KEYWORDS = [
    "label", "true_label", "time", "test_group", "center", "center_norm",
    "file", "path", "direction", "pred", "correct", "threshold",
    "experiment", "reason", "error",
]

# 这些预测/分数字段虽然名字里有 pred/prob/score，但允许作为候选特征。
ALLOW_PRED_NUMERIC_KEYWORDS = [
    "prob_TRUE_LEAK",
    "selected_physics_score",
    "ensemble_score",
    "physics_directed_leak_score",
    "dw_directed_wideband_score",
    "dw_diffuse_wideband_score",
]


# ============================================================
# 2. 基础函数
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


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


def robust_median_mad(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0, 1.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if not np.isfinite(mad) or mad < 1e-12:
        mad = float(np.std(x))
    if not np.isfinite(mad) or mad < 1e-12:
        mad = 1.0
    return med, 1.4826 * mad


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


def binary_metrics(true_labels, pred_labels):
    true = np.array([1 if x == "TRUE_LEAK" else 0 for x in true_labels], dtype=int)
    pred = np.array([1 if x == "TRUE_LEAK" else 0 for x in pred_labels], dtype=int)
    tp = int(np.sum((true == 1) & (pred == 1)))
    tn = int(np.sum((true == 0) & (pred == 0)))
    fp = int(np.sum((true == 0) & (pred == 1)))
    fn = int(np.sum((true == 1) & (pred == 0)))
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    recall_true = tp / (tp + fn + 1e-12)
    recall_false = tn / (tn + fp + 1e-12)
    bal = 0.5 * (recall_true + recall_false)
    return {
        "acc": float(acc),
        "balanced_acc": float(bal),
        "recall_TRUE": float(recall_true),
        "recall_FALSE": float(recall_false),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def clean_experiment_name(s):
    s = str(s)
    replace_map = {
        "A_v7_like_baseline": "A",
        "A_base_without_old_bad_dw": "A2",
        "B_directed_wideband_only": "B",
        "B_global_matrix_only": "B2",
        "C_v8_v7_plus_directed_wideband": "C",
        "C_v8_fixed_base_plus_global_matrix": "C2",
    }
    for k, v in replace_map.items():
        s = s.replace(k, v)
    s = re.sub(r"[^0-9A-Za-z_]+", "_", s)
    return s[:60]


# ============================================================
# 3. 读取数据
# ============================================================

def load_base_dataset():
    path = find_first_existing(DATASET_CANDIDATES)
    if path is None:
        raise FileNotFoundError(
            "找不到已有 v8 特征表。请确认至少存在以下之一：\n" + "\n".join(DATASET_CANDIDATES)
        )
    df = pd.read_csv(path)
    if "true_label" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"true_label": "label"})
    if "test_group" in df.columns and "time" not in df.columns:
        df = df.rename(columns={"test_group": "time"})
    if "label" not in df.columns or "time" not in df.columns:
        raise ValueError(f"特征表缺少 label/time 列: {path}")
    if "center_norm" not in df.columns:
        if "center" not in df.columns:
            raise ValueError("特征表缺少 center 或 center_norm 列。")
        df["center_norm"] = df["center"].apply(normalize_center_id)
    df["label"] = df["label"].astype(str)
    df["time"] = df["time"].astype(str)
    df["center_norm"] = df["center_norm"].apply(normalize_center_id)
    df = df[df["label"].isin(["TRUE_LEAK", "FALSE_LEAK"])].copy()
    print("读取基础特征表:", path)
    print("基础特征表样本数:", len(df))
    return df, path


def merge_prediction_features(df):
    out = df.copy()
    for pred_path in PREDICTION_CANDIDATES:
        if not os.path.exists(pred_path):
            continue
        p = pd.read_csv(pred_path)
        if "true_label" in p.columns:
            p["label"] = p["true_label"].astype(str)
        elif "label" in p.columns:
            p["label"] = p["label"].astype(str)
        else:
            continue
        if "test_group" in p.columns:
            p["time"] = p["test_group"].astype(str)
        elif "time" in p.columns:
            p["time"] = p["time"].astype(str)
        else:
            continue
        if "center_norm" not in p.columns:
            if "center" not in p.columns:
                continue
            p["center_norm"] = p["center"].apply(normalize_center_id)
        else:
            p["center_norm"] = p["center_norm"].apply(normalize_center_id)
        p = p[p["time"].astype(str) == TARGET_TIME].copy()
        if len(p) == 0:
            continue
        source_tag = "fixed" if "fixed" in pred_path else "old"
        if "experiment" not in p.columns:
            p["experiment"] = "single"
        use_cols = []
        for c in p.columns:
            if c in ["label", "time", "center_norm", "experiment"]:
                continue
            if any(k in c for k in ALLOW_PRED_NUMERIC_KEYWORDS):
                vals = safe_float_series(p[c])
                if vals.notna().mean() >= 0.8:
                    use_cols.append(c)
        if not use_cols:
            continue
        for exp, g in p.groupby("experiment"):
            exp_tag = clean_experiment_name(exp)
            keep = ["label", "time", "center_norm"] + use_cols
            gg = g[keep].copy().drop_duplicates(["label", "time", "center_norm"])
            rename = {c: f"pred_{source_tag}_{exp_tag}_{c}" for c in use_cols}
            gg = gg.rename(columns=rename)
            out = out.merge(gg, on=["label", "time", "center_norm"], how="left", validate="one_to_one")
        print("合并预测特征:", pred_path)
    return out


# ============================================================
# 4. 配对结构与候选特征
# ============================================================

def get_target_pairs(df):
    sub = df[df["time"].astype(str) == TARGET_TIME].copy()
    if len(sub) == 0:
        raise ValueError(f"特征表中没有 {TARGET_TIME}")
    centers, pair_map = [], {}
    for center, g in sub.groupby("center_norm"):
        tr = g[g["label"] == "TRUE_LEAK"]
        fa = g[g["label"] == "FALSE_LEAK"]
        if len(tr) == 0 or len(fa) == 0:
            continue
        pair_map[center] = {"TRUE_LEAK": tr.iloc[0], "FALSE_LEAK": fa.iloc[0]}
        centers.append(center)
    centers = sorted(centers)
    if len(centers) == 0:
        raise ValueError("144226 中没有找到成对 TRUE/FALSE center。")
    print(f"\n{TARGET_TIME} 成对center数量:", len(centers))
    print("centers:", " | ".join(centers))
    return sub, centers, pair_map


def candidate_numeric_features(df):
    cols = []
    allow_pred_lc = [x.lower() for x in ALLOW_PRED_NUMERIC_KEYWORDS]
    for c in df.columns:
        if c in ["label", "true_label", "time", "test_group", "center", "center_norm"]:
            continue
        lc = c.lower()
        is_allowed_pred_score = any(k in lc for k in allow_pred_lc)
        if not is_allowed_pred_score:
            if any(k in lc for k in META_COL_KEYWORDS):
                continue
        vals = safe_float_series(df[c])
        if vals.notna().mean() < 0.80:
            continue
        if vals.nunique(dropna=True) <= 2:
            continue
        if vals.std(skipna=True) < 1e-12:
            continue
        cols.append(c)
    return cols


# ============================================================
# 5. 特征评分与选择
# ============================================================

def evaluate_feature_pairwise(pair_map, centers, feature):
    diffs, true_vals, false_vals, used_centers = [], [], [], []
    for center in centers:
        tr = pair_map[center]["TRUE_LEAK"]
        fa = pair_map[center]["FALSE_LEAK"]
        tv = pd.to_numeric(tr.get(feature, np.nan), errors="coerce")
        fv = pd.to_numeric(fa.get(feature, np.nan), errors="coerce")
        if not np.isfinite(tv) or not np.isfinite(fv):
            continue
        true_vals.append(float(tv))
        false_vals.append(float(fv))
        diffs.append(float(tv - fv))
        used_centers.append(center)
    diffs = np.asarray(diffs, dtype=float)
    true_vals = np.asarray(true_vals, dtype=float)
    false_vals = np.asarray(false_vals, dtype=float)
    n = len(diffs)
    if n < max(5, int(0.5 * len(centers))):
        return None
    med_diff = float(np.median(diffs))
    mean_diff = float(np.mean(diffs))
    sign = 1 if (med_diff >= 0 or (abs(med_diff) < 1e-20 and mean_diff >= 0)) else -1
    oriented = sign * diffs
    pair_acc = float(np.mean(oriented > 0))
    pair_tie = float(np.mean(np.abs(oriented) <= 1e-20))
    med_margin = float(np.median(oriented))
    min_margin = float(np.min(oriented))
    mean_margin = float(np.mean(oriented))
    _, scale_diff = robust_median_mad(diffs)
    robust_margin = float(med_margin / (scale_diff + 1e-12))
    y = np.concatenate([np.ones(len(true_vals)), np.zeros(len(false_vals))])
    s = np.concatenate([true_vals, false_vals])
    auc_signed = safe_auc(y, s)
    auc_free = max(auc_signed, 1 - auc_signed) if np.isfinite(auc_signed) else np.nan
    d = cohen_d(true_vals, false_vals)
    return {
        "feature": feature,
        "n_pairs": n,
        "sign_TRUE_larger": sign,
        "pair_acc_after_sign": pair_acc,
        "pair_tie_rate": pair_tie,
        "median_diff_TRUE_minus_FALSE": med_diff,
        "mean_diff_TRUE_minus_FALSE": mean_diff,
        "median_oriented_margin": med_margin,
        "min_oriented_margin": min_margin,
        "mean_oriented_margin": mean_margin,
        "robust_margin": robust_margin,
        "true_mean": float(np.mean(true_vals)),
        "false_mean": float(np.mean(false_vals)),
        "cohen_d": d,
        "abs_cohen_d": abs(d) if np.isfinite(d) else np.nan,
        "auc_signed_TRUE_larger": auc_signed,
        "auc_direction_free": auc_free,
        "used_centers": "|".join(used_centers),
    }


def evaluate_all_features(pair_map, centers, features):
    rows = []
    for f in features:
        r = evaluate_feature_pairwise(pair_map, centers, f)
        if r is not None:
            rows.append(r)
    ranking = pd.DataFrame(rows)
    if len(ranking) == 0:
        raise RuntimeError("没有可用的数值特征用于配对分析。")
    ranking = ranking.sort_values(
        ["pair_acc_after_sign", "auc_direction_free", "median_oriented_margin", "abs_cohen_d"],
        ascending=[False, False, False, False],
    )
    return ranking


def transform_feature_vector(df, feature, sign, med, scale):
    vals = safe_float_series(df[feature]).fillna(med).astype(float).values
    return sign * (vals - med) / (scale + 1e-12)


def select_features_from_ranking(df_sub, ranking, max_features=MAX_SELECTED_FEATURES):
    max_n_pairs = int(ranking["n_pairs"].max())
    chosen_base = ranking[
        (ranking["pair_acc_after_sign"] >= MIN_FEATURE_PAIR_ACC) &
        (ranking["n_pairs"] >= max(8, int(0.7 * max_n_pairs)))
    ].copy()
    if len(chosen_base) == 0:
        chosen_base = ranking.head(30).copy()
    selected, selected_vectors = [], []
    for _, row in chosen_base.iterrows():
        f = row["feature"]
        sign = int(row["sign_TRUE_larger"])
        vals = safe_float_series(df_sub[f])
        med, scale = robust_median_mad(vals.values)
        vec = transform_feature_vector(df_sub, f, sign, med, scale)
        if not np.all(np.isfinite(vec)):
            continue
        ok = True
        for old_vec in selected_vectors:
            if np.std(vec) < 1e-12 or np.std(old_vec) < 1e-12:
                continue
            corr = np.corrcoef(vec, old_vec)[0, 1]
            if np.isfinite(corr) and abs(corr) >= MAX_CORR:
                ok = False
                break
        if not ok:
            continue
        pair_acc = float(row["pair_acc_after_sign"])
        auc_free = float(row["auc_direction_free"]) if np.isfinite(row["auc_direction_free"]) else pair_acc
        margin = max(0.0, float(row["robust_margin"]))
        d_abs = max(0.0, float(row["abs_cohen_d"])) if np.isfinite(row["abs_cohen_d"]) else 0.0
        weight = (pair_acc ** 4) * (auc_free ** 2) * (1.0 + min(margin, 5.0)) * (1.0 + min(d_abs, 5.0) / 3.0)
        selected.append({
            "feature": f,
            "sign_TRUE_larger": sign,
            "median": med,
            "scale": scale,
            "weight": float(weight),
            "pair_acc_after_sign": pair_acc,
            "auc_direction_free": auc_free,
            "median_diff_TRUE_minus_FALSE": float(row["median_diff_TRUE_minus_FALSE"]),
            "median_oriented_margin": float(row["median_oriented_margin"]),
            "robust_margin": float(row["robust_margin"]),
            "cohen_d": float(row["cohen_d"]) if np.isfinite(row["cohen_d"]) else np.nan,
        })
        selected_vectors.append(vec)
        if len(selected) >= max_features:
            break
    if len(selected) == 0:
        row = ranking.iloc[0]
        f = row["feature"]
        sign = int(row["sign_TRUE_larger"])
        med, scale = robust_median_mad(safe_float_series(df_sub[f]).values)
        selected.append({
            "feature": f,
            "sign_TRUE_larger": sign,
            "median": med,
            "scale": scale,
            "weight": 1.0,
            "pair_acc_after_sign": float(row["pair_acc_after_sign"]),
            "auc_direction_free": float(row["auc_direction_free"]),
            "median_diff_TRUE_minus_FALSE": float(row["median_diff_TRUE_minus_FALSE"]),
            "median_oriented_margin": float(row["median_oriented_margin"]),
            "robust_margin": float(row["robust_margin"]),
            "cohen_d": float(row["cohen_d"]) if np.isfinite(row["cohen_d"]) else np.nan,
        })
    return selected


def score_rows(df_rows, selected_features, top_k=None):
    if top_k is not None:
        selected_features = selected_features[:top_k]
    score = np.zeros(len(df_rows), dtype=float)
    total_weight = 0.0
    for item in selected_features:
        f = item["feature"]
        if f not in df_rows.columns:
            continue
        sign = int(item["sign_TRUE_larger"])
        med = float(item["median"])
        scale = float(item["scale"])
        weight = float(item["weight"])
        vals = safe_float_series(df_rows[f]).fillna(med).values.astype(float)
        z = sign * (vals - med) / (scale + 1e-12)
        z = np.clip(z, -6, 6)
        score += weight * z
        total_weight += abs(weight)
    if total_weight > 0:
        score = score / total_weight
    return score


def pair_predict_from_scores(df_scored, score_col):
    rows = []
    for center, g in df_scored.groupby("center_norm"):
        if not {"TRUE_LEAK", "FALSE_LEAK"}.issubset(set(g["label"])):
            continue
        idx_max = g[score_col].astype(float).idxmax()
        for idx, r in g.iterrows():
            pred = "TRUE_LEAK" if idx == idx_max else "FALSE_LEAK"
            rows.append({
                "center_norm": center,
                "label": r["label"],
                "score": float(r[score_col]),
                "pred_label": pred,
                "correct": int(pred == r["label"]),
            })
    return pd.DataFrame(rows)


def evaluate_pair_order(df_scored, score_col):
    rows = []
    for center, g in df_scored.groupby("center_norm"):
        tr = g[g["label"] == "TRUE_LEAK"]
        fa = g[g["label"] == "FALSE_LEAK"]
        if len(tr) == 0 or len(fa) == 0:
            continue
        tr, fa = tr.iloc[0], fa.iloc[0]
        rows.append({
            "center_norm": center,
            "true_score": float(tr[score_col]),
            "false_score": float(fa[score_col]),
            "score_diff_TRUE_minus_FALSE": float(tr[score_col] - fa[score_col]),
            "order_correct": int(float(tr[score_col]) > float(fa[score_col])),
        })
    out = pd.DataFrame(rows)
    return out, float(out["order_correct"].mean()) if len(out) else 0.0


# ============================================================
# 6. full calibration 与 LOOCV
# ============================================================

def full_calibration(df_sub, pair_map, centers, features):
    ranking = evaluate_all_features(pair_map, centers, features)
    selected = select_features_from_ranking(df_sub, ranking, max_features=MAX_SELECTED_FEATURES)
    k_candidates = sorted(set([1, 2, 3, 5, 8, 10, len(selected)]))
    k_candidates = [k for k in k_candidates if k <= len(selected)]
    best = None
    for k in k_candidates:
        scored = df_sub.copy()
        scored["calibrated_score"] = score_rows(scored, selected, top_k=k)
        pair_order, order_acc = evaluate_pair_order(scored, "calibrated_score")
        pred = pair_predict_from_scores(scored, "calibrated_score")
        m = binary_metrics(pred["label"].values, pred["pred_label"].values)
        item = {"top_k": k, "order_acc": order_acc, "sample_acc": m["acc"], "scored": scored, "pair_order": pair_order, "pred": pred}
        if best is None or (item["order_acc"], item["sample_acc"]) > (best["order_acc"], best["sample_acc"]):
            best = item
    scored = best["scored"].copy()
    scored["final_pairwise_pred"] = ""
    for center, g in scored.groupby("center_norm"):
        idx_max = g["calibrated_score"].astype(float).idxmax()
        for idx in g.index:
            scored.loc[idx, "final_pairwise_pred"] = "TRUE_LEAK" if idx == idx_max else "FALSE_LEAK"
    scored["final_correct"] = (scored["final_pairwise_pred"] == scored["label"]).astype(int)
    return {"ranking": ranking, "selected": selected, "best_top_k": best["top_k"], "full_scored": scored, "full_pair_order": best["pair_order"], "full_predictions": best["pred"], "order_acc": best["order_acc"], "sample_acc": best["sample_acc"]}


def build_pair_map_from_df(df_sub, centers_subset):
    pair_map = {}
    for center in centers_subset:
        g = df_sub[df_sub["center_norm"] == center]
        tr = g[g["label"] == "TRUE_LEAK"]
        fa = g[g["label"] == "FALSE_LEAK"]
        if len(tr) == 0 or len(fa) == 0:
            continue
        pair_map[center] = {"TRUE_LEAK": tr.iloc[0], "FALSE_LEAK": fa.iloc[0]}
    return pair_map


def loocv_pairwise(df_sub, centers, features):
    pred_rows, selected_rows = [], []
    for hold_center in centers:
        train_centers = [c for c in centers if c != hold_center]
        train_pair_map = build_pair_map_from_df(df_sub, train_centers)
        train_df = df_sub[df_sub["center_norm"].isin(train_centers)].copy()
        test_df = df_sub[df_sub["center_norm"] == hold_center].copy()
        ranking = evaluate_all_features(train_pair_map, train_centers, features)
        selected = select_features_from_ranking(train_df, ranking, max_features=MAX_SELECTED_FEATURES)
        k_candidates = sorted(set([1, 2, 3, 5, 8, 10, len(selected)]))
        k_candidates = [k for k in k_candidates if k <= len(selected)]
        best_k, best_train_acc = k_candidates[0], -1
        for k in k_candidates:
            train_scored = train_df.copy()
            train_scored["score"] = score_rows(train_scored, selected, top_k=k)
            _, acc = evaluate_pair_order(train_scored, "score")
            if acc > best_train_acc:
                best_train_acc, best_k = acc, k
        test_scored = test_df.copy()
        test_scored["loocv_score"] = score_rows(test_scored, selected, top_k=best_k)
        if len(test_scored) != 2:
            continue
        idx_max = test_scored["loocv_score"].astype(float).idxmax()
        for idx, r in test_scored.iterrows():
            pred = "TRUE_LEAK" if idx == idx_max else "FALSE_LEAK"
            pred_rows.append({
                "holdout_center": hold_center,
                "center_norm": hold_center,
                "label": r["label"],
                "loocv_score": float(r["loocv_score"]),
                "pred_label": pred,
                "correct": int(pred == r["label"]),
                "best_k": best_k,
                "train_pair_acc_for_k": best_train_acc,
                "top_feature": selected[0]["feature"],
                "top_feature_sign": selected[0]["sign_TRUE_larger"],
                "top_feature_train_pair_acc": selected[0]["pair_acc_after_sign"],
            })
        for rank, item in enumerate(selected, 1):
            row = {"holdout_center": hold_center, "rank": rank}
            row.update(item)
            selected_rows.append(row)
    pred = pd.DataFrame(pred_rows)
    selected = pd.DataFrame(selected_rows)
    pair_rows = []
    for center, g in pred.groupby("center_norm"):
        tr = g[g["label"] == "TRUE_LEAK"]
        fa = g[g["label"] == "FALSE_LEAK"]
        if len(tr) == 0 or len(fa) == 0:
            continue
        tr, fa = tr.iloc[0], fa.iloc[0]
        pair_rows.append({
            "center_norm": center,
            "true_score": tr["loocv_score"],
            "false_score": fa["loocv_score"],
            "score_diff_TRUE_minus_FALSE": tr["loocv_score"] - fa["loocv_score"],
            "order_correct": int(tr["loocv_score"] > fa["loocv_score"]),
            "top_feature": tr["top_feature"],
            "top_feature_sign": tr["top_feature_sign"],
        })
    pair = pd.DataFrame(pair_rows)
    m = binary_metrics(pred["label"].values, pred["pred_label"].values) if len(pred) else {"acc": 0.0, "balanced_acc": 0.0}
    order_acc = float(pair["order_correct"].mean()) if len(pair) else 0.0
    return {"pred": pred, "pair": pair, "selected": selected, "sample_acc": m["acc"], "balanced_acc": m["balanced_acc"], "order_acc": order_acc}


# ============================================================
# 7. 输出
# ============================================================

def save_outputs(full_res, loocv_res, ranking, selected, base_path, df_sub):
    ensure_dir(OUTPUT_DIR)
    ranking_path = os.path.join(OUTPUT_DIR, "v8_pairwise_feature_ranking.csv")
    selected_path = os.path.join(OUTPUT_DIR, "v8_pairwise_selected_features.csv")
    full_pred_path = os.path.join(OUTPUT_DIR, "v8_pairwise_full_predictions.csv")
    full_pair_path = os.path.join(OUTPUT_DIR, "v8_pairwise_full_pair_order.csv")
    loocv_pred_path = os.path.join(OUTPUT_DIR, "v8_pairwise_loocv_predictions.csv")
    loocv_pair_path = os.path.join(OUTPUT_DIR, "v8_pairwise_loocv_pair_order.csv")
    loocv_selected_path = os.path.join(OUTPUT_DIR, "v8_pairwise_loocv_selected_features.csv")
    summary_path = os.path.join(OUTPUT_DIR, "v8_pairwise_summary.csv")
    rule_path = os.path.join(OUTPUT_DIR, "v8_pairwise_calibration_rule.json")
    ranking.to_csv(ranking_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(selected).to_csv(selected_path, index=False, encoding="utf-8-sig")
    full_res["full_scored"].to_csv(full_pred_path, index=False, encoding="utf-8-sig")
    full_res["full_pair_order"].to_csv(full_pair_path, index=False, encoding="utf-8-sig")
    loocv_res["pred"].to_csv(loocv_pred_path, index=False, encoding="utf-8-sig")
    loocv_res["pair"].to_csv(loocv_pair_path, index=False, encoding="utf-8-sig")
    loocv_res["selected"].to_csv(loocv_selected_path, index=False, encoding="utf-8-sig")
    summary = pd.DataFrame([
        {"mode": "full_calibration_uses_all_144226_labels", "n_centers": int(df_sub["center_norm"].nunique()), "sample_acc": full_res["sample_acc"], "pair_order_acc": full_res["order_acc"], "best_top_k": full_res["best_top_k"], "n_selected_features": len(selected), "top_feature": selected[0]["feature"] if selected else "", "top_feature_sign": selected[0]["sign_TRUE_larger"] if selected else "", "top_feature_pair_acc": selected[0]["pair_acc_after_sign"] if selected else np.nan},
        {"mode": "leave_one_center_out", "n_centers": int(df_sub["center_norm"].nunique()), "sample_acc": loocv_res["sample_acc"], "pair_order_acc": loocv_res["order_acc"], "best_top_k": np.nan, "n_selected_features": np.nan, "top_feature": "", "top_feature_sign": "", "top_feature_pair_acc": np.nan},
    ])
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    rule = {"target_time": TARGET_TIME, "created_at": str(datetime.now()), "base_dataset": base_path, "mode": "144226_pairwise_calibration", "best_top_k": int(full_res["best_top_k"]), "selected_features": selected, "prediction_rule": "For each center, compute calibrated_score for TRUE/FALSE candidates; higher score is TRUE_LEAK.", "note": "144226-specific paired calibration. Uses known 144226 labels to correct feature direction."}
    with open(rule_path, "w", encoding="utf-8") as f:
        json.dump(rule, f, ensure_ascii=False, indent=2)
    fig_dir = os.path.join(OUTPUT_DIR, "figures")
    ensure_dir(fig_dir)
    top = ranking.head(20).copy()
    plt.figure(figsize=(12, 6))
    plt.barh(np.arange(len(top)), top["pair_acc_after_sign"].values)
    plt.yticks(np.arange(len(top)), top["feature"].astype(str).values, fontsize=7)
    plt.xlim(0, 1.05)
    plt.xlabel("pair accuracy after auto sign")
    plt.title("Top pairwise features for 144226")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    fig1 = os.path.join(fig_dir, "v8_pairwise_top_feature_pair_acc.png")
    plt.savefig(fig1, dpi=150)
    plt.close()
    fig2 = ""
    if len(full_res["full_pair_order"]):
        p = full_res["full_pair_order"].copy()
        plt.figure(figsize=(12, 5))
        plt.bar(p["center_norm"].astype(str), p["score_diff_TRUE_minus_FALSE"].astype(float))
        plt.axhline(0, linestyle="--")
        plt.xlabel("center")
        plt.ylabel("TRUE score - FALSE score")
        plt.title("Full calibration pair margin: 144226")
        plt.tight_layout()
        fig2 = os.path.join(fig_dir, "v8_pairwise_full_pair_margin.png")
        plt.savefig(fig2, dpi=150)
        plt.close()
    report_path = os.path.join(OUTPUT_DIR, "v8_pairwise_report.txt")
    lines = []
    lines.append("v8 144226 专用配对校准报告")
    lines.append("=" * 110)
    lines.append(f"生成时间: {datetime.now()}")
    lines.append("")
    lines.append("输入:")
    lines.append(f"  base_dataset: {base_path}")
    for p in PREDICTION_CANDIDATES:
        lines.append(f"  optional_prediction: {p} | exists={os.path.exists(p)}")
    lines.append("")
    lines.append("核心说明:")
    lines.append("  本版不重新读 WAV。")
    lines.append("  本版专门处理 144226：自动寻找 TRUE/FALSE 成对中心点中的稳定差异特征。")
    lines.append("  如果某个特征在 144226 上 TRUE 稳定小于 FALSE，则自动翻转方向使用。")
    lines.append("")
    lines.append("结果摘要:")
    lines.append(f"  full calibration sample_acc: {full_res['sample_acc']:.4f}")
    lines.append(f"  full calibration pair_order_acc: {full_res['order_acc']:.4f}")
    lines.append(f"  full calibration best_top_k: {full_res['best_top_k']}")
    lines.append(f"  LOOCV sample_acc: {loocv_res['sample_acc']:.4f}")
    lines.append(f"  LOOCV pair_order_acc: {loocv_res['order_acc']:.4f}")
    lines.append("")
    lines.append("full calibration 选中特征前15:")
    for i, item in enumerate(selected[:15], 1):
        lines.append(f"  {i}. {item['feature']} | sign={item['sign_TRUE_larger']} | pair_acc={item['pair_acc_after_sign']:.3f} | auc_free={item['auc_direction_free']:.3f} | median_diff={item['median_diff_TRUE_minus_FALSE']:.6g} | weight={item['weight']:.4g}")
    lines.append("")
    lines.append("LOOCV 失败center:")
    if len(loocv_res["pair"]):
        failed = loocv_res["pair"][loocv_res["pair"]["order_correct"] == 0]["center_norm"].astype(str).tolist()
        lines.append("  " + (" | ".join(failed) if failed else "无"))
    else:
        lines.append("  无LOOCV结果")
    lines.append("")
    lines.append("输出文件:")
    for p in [ranking_path, selected_path, full_pred_path, full_pair_path, loocv_pred_path, loocv_pair_path, loocv_selected_path, summary_path, rule_path, fig1, fig2]:
        if p:
            lines.append(f"  {p}")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return {"ranking_path": ranking_path, "selected_path": selected_path, "full_pred_path": full_pred_path, "full_pair_path": full_pair_path, "loocv_pred_path": loocv_pred_path, "loocv_pair_path": loocv_pair_path, "summary_path": summary_path, "rule_path": rule_path, "report_path": report_path}


# ============================================================
# 8. 主程序
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)
    print("=" * 110)
    print("v8 144226 专用配对校准版")
    print("=" * 110)
    df, base_path = load_base_dataset()
    df = merge_prediction_features(df)
    df_sub, centers, pair_map = get_target_pairs(df)
    features = candidate_numeric_features(df_sub)
    print("\n候选数值特征数:", len(features))
    ranking = evaluate_all_features(pair_map, centers, features)
    print("\n144226 配对特征排名前20:")
    show_cols = ["feature", "pair_acc_after_sign", "sign_TRUE_larger", "auc_direction_free", "median_diff_TRUE_minus_FALSE", "true_mean", "false_mean"]
    print(ranking[show_cols].head(20).to_string(index=False))
    full_res = full_calibration(df_sub, pair_map, centers, features)
    selected = full_res["selected"]
    loocv_res = loocv_pairwise(df_sub, centers, features)
    paths = save_outputs(full_res, loocv_res, ranking, selected, base_path, df_sub)
    print("\n" + "=" * 110)
    print("完成")
    print("=" * 110)
    print("输出文件夹:", OUTPUT_DIR)
    print("报告:", paths["report_path"])
    print("汇总:", paths["summary_path"])
    print("full预测:", paths["full_pred_path"])
    print("LOOCV预测:", paths["loocv_pred_path"])
    print("选中特征:", paths["selected_path"])
    print("特征排名:", paths["ranking_path"])
    print("\n核心结果摘要:")
    print(f"  full calibration sample_acc: {full_res['sample_acc']:.3f}")
    print(f"  full calibration pair_order_acc: {full_res['order_acc']:.3f}")
    print(f"  full calibration best_top_k: {full_res['best_top_k']}")
    print(f"  LOOCV sample_acc: {loocv_res['sample_acc']:.3f}")
    print(f"  LOOCV pair_order_acc: {loocv_res['order_acc']:.3f}")
    print("\nfull calibration 选中特征前10:")
    for i, item in enumerate(selected[:10], 1):
        print(f"  {i}. {item['feature']} | sign={item['sign_TRUE_larger']} | pair_acc={item['pair_acc_after_sign']:.3f} | auc_free={item['auc_direction_free']:.3f} | median_diff={item['median_diff_TRUE_minus_FALSE']:.6g}")
    if len(loocv_res["pair"]):
        failed = loocv_res["pair"][loocv_res["pair"]["order_correct"] == 0]["center_norm"].astype(str).tolist()
        print("\nLOOCV失败center:", " | ".join(failed) if failed else "无")
    print("\n把这段“核心结果摘要 + 选中特征前10 + LOOCV失败center”发给我。")


if __name__ == "__main__":
    main()
