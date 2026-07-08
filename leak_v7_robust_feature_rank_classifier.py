# -*- coding: utf-8 -*-
"""
leak_v7_robust_feature_rank_classifier.py

v7：稳健特征重构 + 时间点内部归一化/排序 + 按时间点整组验证

为什么要做 v7？
    v5/v6 结果说明：
        1. 随机划分效果很好，但按时间点整组验证不稳定；
        2. 某些时间点 AUC 也不高，说明不只是阈值问题；
        3. 模型可能过度依赖绝对能量、时间点背景、采集幅值等不稳定因素。

v7 的核心改动：
    1. 去掉不稳定的绝对能量特征：
        energy_30_40k、energy_40_50k、raw_best_energy、time_energy_std 等
    2. 保留更稳健的比例类、频谱形态类、空间结构类特征：
        ratio_60_70k、high_freq_ratio、spec_flatness、spec_entropy、
        spec_bandwidth_hz、direction_contrast、decay_R2 等
    3. 对每个 time_folder 内部做稳健归一化：
        feature_time_robust_z = (x - 当前time中位数) / IQR
        feature_time_rank_pct = 当前样本在该time内部的百分位排名
    4. 按时间点整组留出验证：
        每次拿一个完整 time_folder 测试，其余 time_folder 训练。
    5. 同时输出三种判断：
        - v7_model_pred：模型 + 训练集校准阈值
        - v7_rank_pred：同一时间点内部概率排名
        - v7_final_decision：两者一致则给 TRUE/FALSE，不一致则给 SUSPECT

运行：
    python leak_v7_robust_feature_rank_classifier.py

输入：
    C:\\Users\\jiangxinru6\\Desktop\\wurenji\\leak_v4_compare_results\\merged_feature_dataset.csv

输出：
    C:\\Users\\jiangxinru6\\Desktop\\wurenji\\leak_v7_robust_feature_results\\
"""

import os
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

MERGED_FEATURE_CSV = r"C:\Users\jiangxinru6\Desktop\wurenji\leak_v4_compare_results\merged_feature_dataset.csv"

OUTPUT_DIR = r"C:\Users\jiangxinru6\Desktop\wurenji\leak_v7_robust_feature_results"

GROUP_COL = "time"
LABEL_COL = "label"

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

# 是否去掉绝对能量特征
DROP_ABSOLUTE_ENERGY_FEATURES = True

# 绝对能量/幅值相关特征关键词
# 注意：不要把 ratio_xxx、high_freq_ratio 这种比例特征删掉。
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

# 这些时间特征是相对/形态特征，允许保留
ALLOW_TIME_FEATURES = [
    "time_energy_cv",
    "time_energy_kurtosis",
    "time_energy_max_mean_ratio",
]

# 频率特征低于 20kHz 的比例太高时，是否删除该列
DROP_BAD_FREQUENCY_FEATURES = True
FREQ_LOW_HZ = 20000
BAD_FREQ_RATIO_THRESHOLD = 0.30

# 时间点内部增强特征
ADD_TIME_ROBUST_Z = True
ADD_TIME_RANK_PCT = True

# 模型概率阈值搜索
THRESHOLD_GRID = np.linspace(0.01, 0.99, 99)
THRESHOLD_METRIC = "balanced_accuracy"

# 同一时间点内部排名判断：前多少比例判 TRUE_LIKE
# 你现在验证集中真假样本基本各一半，所以默认 0.5 用来做二分类验证。
RANK_TRUE_FRACTION_FOR_BINARY = 0.50

# 三档输出阈值
RANK_TRUE_LIKE_PCT = 0.70
RANK_FALSE_LIKE_PCT = 0.30

# 随机种子
RANDOM_STATE = 42


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

        row = {
            "threshold": float(t),
            "score": float(score),
            **m,
        }
        rows.append(row)

        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t, float(best_score), pd.DataFrame(rows)


# ============================================================
# 3. 特征筛选与重构
# ============================================================

def is_absolute_energy_feature(col):
    """
    判断是否是绝对能量/幅值类特征。
    这些特征容易受到采集距离、增益、声源强度、背景噪声影响。
    """
    c = col.lower()

    # 明确保留相对时间特征
    if col in ALLOW_TIME_FEATURES:
        return False

    for key in ABSOLUTE_ENERGY_EXACT_OR_PREFIX:
        k = key.lower()
        if c == k:
            return True

    # 更宽松的规则：energy_ 开头的多数是绝对能量
    # ratio_xxx 不删，因为它不是 energy_ 开头。
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


def remove_unstable_features(df, feature_cols, output_dir):
    """
    删除：
        1. 绝对能量类特征
        2. 无效频率比例过高的频率类特征
    """
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

    removed_df = pd.DataFrame(removed_rows)
    removed_csv = os.path.join(output_dir, "v7_removed_features.csv")
    removed_df.to_csv(removed_csv, index=False, encoding="utf-8-sig")

    return kept, removed_csv


def make_base_numeric_df(df, feature_cols):
    x = pd.DataFrame(index=df.index)

    for c in feature_cols:
        vals = safe_float_array(df[c])
        # 先用全局中位数填充，避免空值
        median = vals.median()
        if not np.isfinite(median):
            median = 0.0
        x[c] = vals.fillna(median).astype(float)

    return x


def add_time_internal_features(df, base_x, group_col):
    """
    对每个 time_folder 内部做：
        1. robust z: (x - median) / IQR
        2. rank percentile: 当前值在本组内的百分位
    """
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
            # pct=True: 最小接近 1/n，最大为 1
            rank_values = values.groupby(groups).rank(method="average", pct=True)
            out[r_col] = rank_values.fillna(0.5).astype(float)

    return out


def prepare_v7_feature_matrix(df, output_dir):
    """
    完成 v7 的特征筛选和重构。
    """
    initial_features = get_initial_numeric_features(df)

    kept_base_features, removed_csv = remove_unstable_features(
        df,
        initial_features,
        output_dir
    )

    base_x = make_base_numeric_df(df, kept_base_features)
    v7_x = add_time_internal_features(df, base_x, GROUP_COL)

    used_base_path = os.path.join(output_dir, "v7_used_base_features.txt")
    used_all_path = os.path.join(output_dir, "v7_used_all_model_features.txt")

    save_text(used_base_path, "\n".join(kept_base_features))
    save_text(used_all_path, "\n".join(v7_x.columns.tolist()))

    feature_dataset = pd.concat(
        [
            df[[c for c in ["dataset", "time", "center", "label"] if c in df.columns]].reset_index(drop=True),
            v7_x.reset_index(drop=True)
        ],
        axis=1
    )

    feature_dataset_csv = os.path.join(output_dir, "v7_robust_feature_dataset.csv")
    feature_dataset.to_csv(feature_dataset_csv, index=False, encoding="utf-8-sig")

    return {
        "initial_features": initial_features,
        "base_features": kept_base_features,
        "model_features": v7_x.columns.tolist(),
        "X": v7_x,
        "removed_csv": removed_csv,
        "used_base_path": used_base_path,
        "used_all_path": used_all_path,
        "feature_dataset_csv": feature_dataset_csv,
    }


# ============================================================
# 4. 模型与 OOF 阈值
# ============================================================

def build_classifier():
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=800,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
    )


def get_group_oof_probabilities(X_train, y_train, groups_train):
    """
    在训练集内部按 group 做 OOF 概率。
    例如当前测试组是 142938，则训练集中有另外三个 time。
    内部再做 leave-one-training-time-out，得到训练样本的 OOF 概率，用来选阈值。
    """
    y_train = np.asarray(y_train, dtype=int)
    groups_train = np.asarray(groups_train).astype(str)

    unique_train_groups = sorted(pd.unique(groups_train).tolist())

    oof_prob = np.zeros(len(y_train), dtype=float)
    filled = np.zeros(len(y_train), dtype=bool)

    # 优先按 group 留出
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

    # 如果有没填上的，兜底用 StratifiedKFold
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


# ============================================================
# 5. v7 排名判断与最终决策
# ============================================================

def add_probability_rank_columns(pred_df):
    """
    在每个 test_group 内部，对 prob_TRUE_LEAK 做排序和归一化。
    """
    pred_df = pred_df.copy()

    pred_df["prob_rank_pct_in_group"] = 0.5
    pred_df["prob_relative_minmax_in_group"] = 0.5

    for g, idx in pred_df.groupby("test_group").groups.items():
        sub_prob = pred_df.loc[idx, "prob_TRUE_LEAK"].astype(float)

        # 百分位排名
        pred_df.loc[idx, "prob_rank_pct_in_group"] = sub_prob.rank(method="average", pct=True)

        # min-max
        p_min = float(sub_prob.min())
        p_max = float(sub_prob.max())
        if abs(p_max - p_min) < 1e-12:
            rel = pd.Series(0.5, index=idx)
        else:
            rel = (sub_prob - p_min) / (p_max - p_min)

        pred_df.loc[idx, "prob_relative_minmax_in_group"] = rel

    # 二分类排名规则：默认 top 50% 判 TRUE_LEAK
    cutoff = 1.0 - RANK_TRUE_FRACTION_FOR_BINARY

    pred_df["v7_rank_binary_pred"] = np.where(
        pred_df["prob_rank_pct_in_group"] > cutoff,
        "TRUE_LEAK",
        "FALSE_LEAK"
    )

    # 三档排名等级
    conditions = [
        pred_df["prob_rank_pct_in_group"] >= RANK_TRUE_LIKE_PCT,
        pred_df["prob_rank_pct_in_group"] <= RANK_FALSE_LIKE_PCT,
    ]
    choices = ["TRUE_LIKE", "FALSE_LIKE"]
    pred_df["v7_rank_level"] = np.select(conditions, choices, default="SUSPECT")

    # 最终决策：
    # 模型阈值判断和排名判断一致 -> 给 TRUE/FALSE
    # 不一致 -> SUSPECT
    final = []
    for _, r in pred_df.iterrows():
        model_pred = r["v7_model_pred"]
        rank_pred = r["v7_rank_binary_pred"]

        if model_pred == rank_pred:
            final.append(model_pred)
        else:
            final.append("SUSPECT")

    pred_df["v7_final_decision"] = final

    return pred_df


def calc_final_decision_metrics(pred_df):
    """
    对最终三档决策统计：
        - decisive_rate: 非 SUSPECT 比例
        - decisive_accuracy: 只在非 SUSPECT 样本上算准确率
        - strict_accuracy: 把 SUSPECT 当错，算严格准确率
    """
    if len(pred_df) == 0:
        return {
            "decisive_rate": 0,
            "decisive_accuracy": 0,
            "strict_accuracy_suspect_as_wrong": 0,
            "n_suspect": 0,
            "n_decisive": 0,
        }

    decisive_mask = pred_df["v7_final_decision"].isin(["TRUE_LEAK", "FALSE_LEAK"])
    n_decisive = int(decisive_mask.sum())
    n_suspect = int((~decisive_mask).sum())

    decisive_rate = n_decisive / len(pred_df)

    if n_decisive > 0:
        decisive_accuracy = float(
            (pred_df.loc[decisive_mask, "v7_final_decision"] ==
             pred_df.loc[decisive_mask, "true_label"]).mean()
        )
    else:
        decisive_accuracy = 0.0

    strict_correct = (
        (pred_df["v7_final_decision"] == pred_df["true_label"]) &
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
# 6. 按时间点整组验证
# ============================================================

def leave_one_time_group_validation(df, X, output_dir):
    y_all = label_to_binary(df[LABEL_COL].astype(str).values)
    groups = df[GROUP_COL].astype(str).values
    unique_groups = sorted(pd.unique(groups).tolist())

    all_pred_rows = []
    group_rows = []

    print("\n开始 v7 按时间点整组验证...")
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

        # 训练集内部按时间点 OOF，选阈值
        oof_prob = get_group_oof_probabilities(X_train, y_train, groups_train)

        best_t, best_score, threshold_curve = find_best_threshold(
            y_train,
            oof_prob,
            metric=THRESHOLD_METRIC,
            grid=THRESHOLD_GRID
        )

        curve_path = os.path.join(
            output_dir,
            f"v7_threshold_curve_train_without_{safe_name(test_group)}.csv"
        )
        threshold_curve.to_csv(curve_path, index=False, encoding="utf-8-sig")

        # 用完整训练集训练模型
        clf = build_classifier()
        clf.fit(X_train, y_train)

        prob = clf.predict_proba(X_test)[:, 1]

        # 模型阈值预测
        model_pred_binary = threshold_predict(prob, best_t)
        default_pred_binary = threshold_predict(prob, 0.5)

        m_default = metrics_from_pred(y_test, default_pred_binary)
        m_model = metrics_from_pred(y_test, model_pred_binary)
        auc = safe_auc(y_test, prob)

        # 先做预测明细
        group_pred_rows = []

        for i in range(len(test_df)):
            true_label = binary_to_label(y_test[i])
            default_pred = binary_to_label(default_pred_binary[i])
            model_pred = binary_to_label(model_pred_binary[i])

            row = {
                "test_group": test_group,
                "dataset": test_df.loc[i, "dataset"] if "dataset" in test_df.columns else "",
                "time": test_df.loc[i, "time"] if "time" in test_df.columns else "",
                "center": test_df.loc[i, "center"] if "center" in test_df.columns else "",
                "true_label": true_label,
                "prob_TRUE_LEAK": float(prob[i]),
                "v7_best_threshold": best_t,
                "default_pred_0p5": default_pred,
                "default_correct": int(default_pred == true_label),
                "v7_model_pred": model_pred,
                "v7_model_correct": int(model_pred == true_label),
            }

            # 附带关键特征方便排查
            key_features = [
                "ratio_60_70k",
                "ratio_50_60k",
                "ratio_40_50k",
                "high_freq_ratio",
                "spec_flatness",
                "spec_entropy",
                "spec_bandwidth_hz",
                "spec_centroid_hz",
                "spec_slope",
                "direction_contrast",
                "decay_R2",
                "near_far_ratio",
                "time_energy_cv",
            ]

            for k in key_features:
                if k in test_df.columns:
                    row[k] = test_df.loc[i, k]

            group_pred_rows.append(row)

        group_pred_df = pd.DataFrame(group_pred_rows)
        group_pred_df = add_probability_rank_columns(group_pred_df)

        # 排名二分类指标
        rank_pred_binary = label_to_binary(group_pred_df["v7_rank_binary_pred"].values)
        m_rank = metrics_from_pred(y_test, rank_pred_binary)

        # 最终三档指标
        final_metrics = calc_final_decision_metrics(group_pred_df)

        print(
            f"  测试组 {test_group}: "
            f"n={len(y_test)}, "
            f"best_t={best_t:.3f}, "
            f"default_acc={m_default['accuracy']:.3f}, "
            f"model_acc={m_model['accuracy']:.3f}, "
            f"rank_acc={m_rank['accuracy']:.3f}, "
            f"final_decisive_acc={final_metrics['decisive_accuracy']:.3f}, "
            f"suspect={final_metrics['n_suspect']}, "
            f"auc={auc if not np.isnan(auc) else 'NA'}"
        )

        group_row = {
            "test_group": test_group,
            "n_test": len(y_test),
            "n_true": int(np.sum(y_test == 1)),
            "n_false": int(np.sum(y_test == 0)),
            "v7_best_threshold": best_t,
            "v7_train_oof_best_score": best_score,
            "auc": auc,

            "default_accuracy_0p5": m_default["accuracy"],
            "default_balanced_accuracy_0p5": m_default["balanced_accuracy"],

            "v7_model_accuracy": m_model["accuracy"],
            "v7_model_balanced_accuracy": m_model["balanced_accuracy"],
            "v7_model_recall_TRUE_LEAK": m_model["recall_TRUE_LEAK"],
            "v7_model_recall_FALSE_LEAK": m_model["recall_FALSE_LEAK"],
            "v7_model_tp": m_model["tp"],
            "v7_model_tn": m_model["tn"],
            "v7_model_fp": m_model["fp"],
            "v7_model_fn": m_model["fn"],

            "v7_rank_accuracy": m_rank["accuracy"],
            "v7_rank_balanced_accuracy": m_rank["balanced_accuracy"],
            "v7_rank_recall_TRUE_LEAK": m_rank["recall_TRUE_LEAK"],
            "v7_rank_recall_FALSE_LEAK": m_rank["recall_FALSE_LEAK"],

            "v7_final_decisive_rate": final_metrics["decisive_rate"],
            "v7_final_decisive_accuracy": final_metrics["decisive_accuracy"],
            "v7_final_strict_accuracy_suspect_as_wrong": final_metrics["strict_accuracy_suspect_as_wrong"],
            "v7_final_n_suspect": final_metrics["n_suspect"],
            "v7_final_n_decisive": final_metrics["n_decisive"],
        }

        group_rows.append(group_row)

        all_pred_rows.extend(group_pred_df.to_dict(orient="records"))

    group_df = pd.DataFrame(group_rows)
    pred_df = pd.DataFrame(all_pred_rows)

    group_csv = os.path.join(output_dir, "v7_group_validation_summary.csv")
    pred_csv = os.path.join(output_dir, "v7_predictions.csv")
    wrong_csv = os.path.join(output_dir, "v7_model_misclassified_samples.csv")
    suspect_csv = os.path.join(output_dir, "v7_suspect_samples.csv")

    group_df.to_csv(group_csv, index=False, encoding="utf-8-sig")
    pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")

    if len(pred_df):
        wrong_df = pred_df[pred_df["v7_model_correct"] == 0].copy()
        wrong_df.to_csv(wrong_csv, index=False, encoding="utf-8-sig")

        suspect_df = pred_df[pred_df["v7_final_decision"] == "SUSPECT"].copy()
        suspect_df.to_csv(suspect_csv, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(wrong_csv, index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(suspect_csv, index=False, encoding="utf-8-sig")

    return group_df, pred_df, group_csv, pred_csv, wrong_csv, suspect_csv


def safe_name(s):
    return str(s).replace("\\", "_").replace("/", "_").replace(":", "_").replace(".", "_")


# ============================================================
# 7. 最终模型训练与保存
# ============================================================

def train_final_model(df, X, feature_info, output_dir):
    try:
        import joblib
    except Exception as e:
        raise RuntimeError("缺少 joblib，请运行: pip install joblib") from e

    y = label_to_binary(df[LABEL_COL].astype(str).values)
    groups = df[GROUP_COL].astype(str).values

    # 全数据内部按 group OOF 选最终阈值
    oof_prob = get_group_oof_probabilities(X.reset_index(drop=True), y, groups)

    best_t, best_score, threshold_curve = find_best_threshold(
        y,
        oof_prob,
        metric=THRESHOLD_METRIC,
        grid=THRESHOLD_GRID
    )

    threshold_curve_csv = os.path.join(output_dir, "v7_global_oof_threshold_curve.csv")
    threshold_curve.to_csv(threshold_curve_csv, index=False, encoding="utf-8-sig")

    clf = build_classifier()
    clf.fit(X, y)

    model_path = os.path.join(output_dir, "v7_final_robust_classifier.pkl")
    joblib.dump(clf, model_path)

    importance_df = pd.DataFrame({
        "feature": X.columns.tolist(),
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance_csv = os.path.join(output_dir, "v7_final_feature_importance.csv")
    importance_df.to_csv(importance_csv, index=False, encoding="utf-8-sig")

    config = {
        "model_type": "RandomForestClassifier",
        "version": "v7_robust_feature_rank_classifier",
        "positive_label": "TRUE_LEAK",
        "label_mapping": {
            "FALSE_LEAK": 0,
            "TRUE_LEAK": 1,
        },
        "recommended_threshold": best_t,
        "threshold_metric": THRESHOLD_METRIC,
        "threshold_score_on_group_oof": best_score,
        "drop_absolute_energy_features": DROP_ABSOLUTE_ENERGY_FEATURES,
        "drop_bad_frequency_features": DROP_BAD_FREQUENCY_FEATURES,
        "bad_frequency_ratio_threshold": BAD_FREQ_RATIO_THRESHOLD,
        "add_time_robust_z": ADD_TIME_ROBUST_Z,
        "add_time_rank_pct": ADD_TIME_RANK_PCT,
        "rank_true_fraction_for_binary_validation": RANK_TRUE_FRACTION_FOR_BINARY,
        "rank_true_like_pct": RANK_TRUE_LIKE_PCT,
        "rank_false_like_pct": RANK_FALSE_LIKE_PCT,
        "base_features": feature_info["base_features"],
        "model_features": feature_info["model_features"],
        "created_at": str(datetime.now()),
        "input_csv": MERGED_FEATURE_CSV,
        "note": "v7 使用同一 time_folder 内部的无标签归一化/排名特征。新数据预测时建议按一个完整 time_folder 批量输入。",
    }

    config_path = os.path.join(output_dir, "v7_final_model_config.json")
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


# ============================================================
# 8. 报告和画图
# ============================================================

def make_report(df, feature_info, group_df, pred_df, final_info, output_dir):
    lines = []

    lines.append("v7 稳健特征重构 + 时间点内部归一化/排序 验证报告")
    lines.append("=" * 90)
    lines.append(f"生成时间: {datetime.now()}")
    lines.append(f"输入文件: {MERGED_FEATURE_CSV}")
    lines.append("")
    lines.append("样本情况:")
    lines.append(f"  总样本数: {len(df)}")
    for label, count in df[LABEL_COL].value_counts().items():
        lines.append(f"  {label}: {int(count)}")
    lines.append("")

    lines.append("特征情况:")
    lines.append(f"  初始数值特征数: {len(feature_info['initial_features'])}")
    lines.append(f"  删除不稳定特征后基础特征数: {len(feature_info['base_features'])}")
    lines.append(f"  加入 time 内部归一化/排名后模型特征数: {len(feature_info['model_features'])}")
    lines.append(f"  被删除特征列表: {feature_info['removed_csv']}")
    lines.append("")

    if group_df is not None and len(group_df):
        lines.append("按时间点整组验证平均结果:")
        lines.append(f"  默认阈值0.5平均准确率: {group_df['default_accuracy_0p5'].mean():.4f}")
        lines.append(f"  v7模型阈值平均准确率: {group_df['v7_model_accuracy'].mean():.4f}")
        lines.append(f"  v7时间点内部排名平均准确率: {group_df['v7_rank_accuracy'].mean():.4f}")
        lines.append(f"  v7最终三档决策-平均明确判定比例: {group_df['v7_final_decisive_rate'].mean():.4f}")
        lines.append(f"  v7最终三档决策-明确样本平均准确率: {group_df['v7_final_decisive_accuracy'].mean():.4f}")
        lines.append(f"  平均AUC: {group_df['auc'].mean():.4f}")
        lines.append("")

        lines.append("各时间点结果:")
        for _, r in group_df.iterrows():
            lines.append(
                f"  {r['test_group']}: "
                f"n={int(r['n_test'])}, "
                f"best_t={r['v7_best_threshold']:.3f}, "
                f"default_acc={r['default_accuracy_0p5']:.3f}, "
                f"model_acc={r['v7_model_accuracy']:.3f}, "
                f"rank_acc={r['v7_rank_accuracy']:.3f}, "
                f"final_decisive_rate={r['v7_final_decisive_rate']:.3f}, "
                f"final_decisive_acc={r['v7_final_decisive_accuracy']:.3f}, "
                f"suspect={int(r['v7_final_n_suspect'])}, "
                f"AUC={r['auc']}"
            )
        lines.append("")

    if pred_df is not None and len(pred_df):
        lines.append("误判/可疑统计:")
        lines.append(f"  v7模型误判数: {int((pred_df['v7_model_correct'] == 0).sum())}")
        lines.append(f"  v7最终SUSPECT数量: {int((pred_df['v7_final_decision'] == 'SUSPECT').sum())}")
        lines.append("")

    lines.append("最终模型:")
    lines.append(f"  模型文件: {final_info['model_path']}")
    lines.append(f"  配置文件: {final_info['config_path']}")
    lines.append(f"  推荐阈值: {final_info['recommended_threshold']:.3f}")
    lines.append(f"  OOF阈值优化得分: {final_info['threshold_score']:.4f}")
    lines.append("")

    lines.append("最终模型重要特征前20:")
    for _, row in final_info["importance_df"].head(20).iterrows():
        lines.append(f"  {row['feature']}: {row['importance']:.6f}")

    report_path = os.path.join(output_dir, "v7_report.txt")
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
    plt.bar(x, group_df["v7_model_accuracy"], width, label="V7 model")
    plt.bar(x + width, group_df["v7_rank_accuracy"], width, label="V7 rank")
    plt.ylim(0, 1.05)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("V7 group validation accuracy comparison")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(fig_dir, "v7_group_accuracy_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def plot_feature_importance(importance_df, output_dir, top_n=25):
    if importance_df is None or len(importance_df) == 0:
        return None

    fig_dir = os.path.join(output_dir, "figures")
    ensure_dir(fig_dir)

    top = importance_df.head(top_n).iloc[::-1]

    plt.figure(figsize=(11, 9))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance")
    plt.title(f"V7 top {top_n} feature importance")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(fig_dir, "v7_top_feature_importance.png")
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
    plt.title("V7 global group-OOF threshold curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(fig_dir, "v7_global_threshold_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()

    return path


# ============================================================
# 9. 主函数
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)

    print("=" * 90)
    print("v7 稳健特征重构 + 时间点内部归一化/排序 分类程序")
    print("=" * 90)

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

    df[LABEL_COL] = df[LABEL_COL].astype(str)
    df = df[df[LABEL_COL].isin(["TRUE_LEAK", "FALSE_LEAK"])].copy()
    df = df.reset_index(drop=True)

    print("样本数量:", len(df))
    print(df[LABEL_COL].value_counts())

    # 1. 特征重构
    print("\n开始筛选稳健特征并构造 time 内部归一化/排名特征...")
    feature_info = prepare_v7_feature_matrix(df, OUTPUT_DIR)

    print("初始数值特征数:", len(feature_info["initial_features"]))
    print("删除不稳定特征后基础特征数:", len(feature_info["base_features"]))
    print("最终模型特征数:", len(feature_info["model_features"]))
    print("被删除特征列表:", feature_info["removed_csv"])
    print("v7稳健特征数据表:", feature_info["feature_dataset_csv"])

    X = feature_info["X"]

    # 2. 分组验证
    group_df, pred_df, group_csv, pred_csv, wrong_csv, suspect_csv = leave_one_time_group_validation(
        df,
        X,
        OUTPUT_DIR
    )

    print("\n分组验证汇总:", group_csv)
    print("预测明细:", pred_csv)
    print("v7模型误判样本:", wrong_csv)
    print("v7最终SUSPECT样本:", suspect_csv)

    # 3. 最终模型
    final_info = train_final_model(df, X, feature_info, OUTPUT_DIR)

    print("\n最终模型:", final_info["model_path"])
    print("最终配置:", final_info["config_path"])
    print("最终特征重要性:", final_info["importance_csv"])
    print("全局OOF阈值曲线:", final_info["threshold_curve_csv"])
    print(f"v7推荐阈值: {final_info['recommended_threshold']:.3f}")
    print(f"v7 OOF阈值优化得分: {final_info['threshold_score']:.4f}")

    # 4. 报告和图
    report_path = make_report(df, feature_info, group_df, pred_df, final_info, OUTPUT_DIR)
    fig1 = plot_group_metrics(group_df, OUTPUT_DIR)
    fig2 = plot_feature_importance(final_info["importance_df"], OUTPUT_DIR)
    fig3 = plot_threshold_curve(final_info["threshold_curve_csv"], OUTPUT_DIR)

    print("报告:", report_path)

    print("\n图片输出:")
    for p in [fig1, fig2, fig3]:
        if p:
            print(" ", p)

    print("\n最终模型重要特征前10:")
    for _, row in final_info["importance_df"].head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.6f}")

    if group_df is not None and len(group_df):
        print("\n各时间点核心结果:")
        for _, r in group_df.iterrows():
            print(
                f"  {r['test_group']}: "
                f"default_acc={r['default_accuracy_0p5']:.3f}, "
                f"model_acc={r['v7_model_accuracy']:.3f}, "
                f"rank_acc={r['v7_rank_accuracy']:.3f}, "
                f"final_decisive_acc={r['v7_final_decisive_accuracy']:.3f}, "
                f"suspect={int(r['v7_final_n_suspect'])}, "
                f"auc={r['auc'] if not np.isnan(r['auc']) else 'NA'}"
            )

    print("\n" + "=" * 90)
    print("全部完成")
    print("输出文件夹:", OUTPUT_DIR)
    print("=" * 90)


if __name__ == "__main__":
    main()
