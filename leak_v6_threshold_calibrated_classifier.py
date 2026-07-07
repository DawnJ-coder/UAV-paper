# -*- coding: utf-8 -*-
"""
leak_v6_threshold_calibrated_classifier.py

v6：自动阈值校准版真假泄漏分类程序

为什么需要 v6？
    v5 结果显示:
        有些时间点 AUC 很高，但 acc 较低。
    这说明模型已经能把 TRUE_LEAK 排在 FALSE_LEAK 前面，
    但默认概率阈值 0.5 不合适。

v6 做什么？
    1. 读取 v4 合并后的 merged_feature_dataset.csv
    2. 按 time 整组留出验证
    3. 在训练集内部用交叉验证得到 out-of-fold 概率
    4. 在训练集上自动寻找最佳阈值
    5. 用这个阈值测试留出的完整时间点
    6. 同时比较:
        - 默认阈值 0.5 的结果
        - 自动校准阈值的结果
    7. 训练最终模型，并保存:
        - 模型 pkl
        - 特征配置 json
        - 推荐阈值
        - 各种报告和预测明细

运行:
    python leak_v6_threshold_calibrated_classifier.py

输入:
    C:\\Users\\jiangxinru6\\Desktop\\wurenji\\leak_v4_compare_results\\merged_feature_dataset.csv

输出:
    C:\\Users\\jiangxinru6\\Desktop\\wurenji\\leak_v6_threshold_calibrated_results\\
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

OUTPUT_DIR = r"C:\Users\jiangxinru6\Desktop\wurenji\leak_v6_threshold_calibrated_results"

GROUP_COL = "time"
LABEL_COL = "label"

# 不参与训练的列
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

# 如果不同实验批次音量差异特别大，可以设为 True，删除绝对能量特征。
# 现在先设 False，保留全部有效特征。
DROP_ABSOLUTE_ENERGY_FEATURES = False

ABSOLUTE_ENERGY_KEYWORDS = [
    "raw_best_energy",
    "mean_energy",
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
]

# 自动找阈值时优化哪个指标:
# 可选:
#   "balanced_accuracy" 推荐
#   "f1"
#   "youden"
THRESHOLD_METRIC = "balanced_accuracy"

# 阈值搜索范围
THRESHOLD_GRID = np.linspace(0.01, 0.99, 99)

# 最终模型是否保存
SAVE_FINAL_MODEL = True


# ============================================================
# 2. 工具函数
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


def get_numeric_features(df):
    feature_cols = []

    for c in df.columns:
        if c in DROP_COLS:
            continue

        temp = pd.to_numeric(df[c], errors="coerce")
        valid_ratio = temp.notna().mean()

        if valid_ratio > 0.8:
            feature_cols.append(c)

    if DROP_ABSOLUTE_ENERGY_FEATURES:
        filtered = []
        for c in feature_cols:
            lower = c.lower()
            if any(k.lower() in lower for k in ABSOLUTE_ENERGY_KEYWORDS):
                continue
            filtered.append(c)
        feature_cols = filtered

    return feature_cols


def make_numeric_matrix(df, feature_cols):
    x = df[feature_cols].copy()

    for c in feature_cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(0.0)

    return x


def calc_confusion(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return tp, tn, fp, fn


def metrics_from_pred(y_true, y_pred):
    tp, tn, fp, fn = calc_confusion(y_true, y_pred)

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0

    recall_true = tp / (tp + fn + 1e-12)
    recall_false = tn / (tn + fp + 1e-12)

    balanced_acc = 0.5 * (recall_true + recall_false)

    precision_true = tp / (tp + fp + 1e-12)
    f1 = 2 * precision_true * recall_true / (precision_true + recall_true + 1e-12)

    youden = recall_true + recall_false - 1.0

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_acc),
        "recall_TRUE_LEAK": float(recall_true),
        "recall_FALSE_LEAK": float(recall_false),
        "precision_TRUE_LEAK": float(precision_true),
        "f1_TRUE_LEAK": float(f1),
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
    return (np.asarray(prob) >= threshold).astype(int)


def find_best_threshold(y_true, prob, metric="balanced_accuracy", grid=None):
    """
    在训练集 out-of-fold 概率上寻找最佳阈值。
    """
    if grid is None:
        grid = THRESHOLD_GRID

    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)

    best_t = 0.5
    best_score = -1e9
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

    curve = pd.DataFrame(rows)

    return best_t, float(best_score), curve


def build_classifier():
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(
        n_estimators=700,
        random_state=42,
        class_weight="balanced",
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
    )

    return clf


def get_oof_probabilities(X, y):
    """
    用训练集内部交叉验证得到 out-of-fold 概率。
    这个概率用于找阈值，避免在同一批训练预测上直接调阈值造成过拟合。
    """
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y, dtype=int)

    min_class_count = min(np.sum(y == 0), np.sum(y == 1))

    if min_class_count < 2:
        raise RuntimeError("训练集中某一类样本少于2个，无法做交叉验证阈值校准。")

    n_splits = min(5, int(min_class_count))

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    oof_prob = np.zeros(len(y), dtype=float)

    for train_idx, val_idx in cv.split(X, y):
        clf = build_classifier()
        clf.fit(X.iloc[train_idx], y[train_idx])
        oof_prob[val_idx] = clf.predict_proba(X.iloc[val_idx])[:, 1]

    return oof_prob


# ============================================================
# 3. 按时间点留出 + 自动阈值校准
# ============================================================

def group_validation_with_threshold(df, feature_cols, output_dir):
    X_all = make_numeric_matrix(df, feature_cols)
    y_all = label_to_binary(df[LABEL_COL].astype(str).values)
    groups = df[GROUP_COL].astype(str).values

    unique_groups = sorted(pd.unique(groups).tolist())

    all_prediction_rows = []
    group_summary_rows = []
    threshold_curve_paths = []

    print("\n开始 v6 按时间点留出验证 + 自动阈值校准...")
    print("分组数量:", len(unique_groups))

    for g in unique_groups:
        test_mask = groups == g
        train_mask = ~test_mask

        X_train = X_all.loc[train_mask].reset_index(drop=True)
        X_test = X_all.loc[test_mask].reset_index(drop=True)

        y_train = y_all[train_mask]
        y_test = y_all[test_mask]

        test_df = df.loc[test_mask].reset_index(drop=True)

        if len(np.unique(y_train)) < 2:
            print(f"  [跳过] 测试组 {g}: 训练集中只有一种类别。")
            continue

        if len(np.unique(y_test)) < 2:
            print(f"  [提醒] 测试组 {g}: 测试集中只有一种类别，AUC可能不可用。")

        # 1. 训练集内部 OOF 概率，用于找阈值
        oof_prob = get_oof_probabilities(X_train, y_train)

        best_t, best_score, threshold_curve = find_best_threshold(
            y_train,
            oof_prob,
            metric=THRESHOLD_METRIC,
            grid=THRESHOLD_GRID
        )

        # 保存这个测试组对应的阈值曲线
        curve_path = os.path.join(
            output_dir,
            f"threshold_curve_train_without_{safe_group_name(g)}.csv"
        )
        threshold_curve.to_csv(curve_path, index=False, encoding="utf-8-sig")
        threshold_curve_paths.append(curve_path)

        # 2. 用完整训练集训练模型
        clf = build_classifier()
        clf.fit(X_train, y_train)

        test_prob = clf.predict_proba(X_test)[:, 1]

        # 3. 默认阈值 0.5
        pred_default = threshold_predict(test_prob, 0.5)
        m_default = metrics_from_pred(y_test, pred_default)
        auc = safe_auc(y_test, test_prob)

        # 4. 自动阈值
        pred_calib = threshold_predict(test_prob, best_t)
        m_calib = metrics_from_pred(y_test, pred_calib)

        print(
            f"  测试组 {g}: "
            f"n={len(y_test)}, "
            f"best_t={best_t:.3f}, "
            f"default_acc={m_default['accuracy']:.3f}, "
            f"calib_acc={m_calib['accuracy']:.3f}, "
            f"calib_bal_acc={m_calib['balanced_accuracy']:.3f}, "
            f"auc={auc if not np.isnan(auc) else 'NA'}"
        )

        row = {
            "test_group": g,
            "n_test": len(y_test),
            "n_true": int(np.sum(y_test == 1)),
            "n_false": int(np.sum(y_test == 0)),
            "train_oof_best_threshold": best_t,
            "train_oof_best_score": best_score,
            "auc": auc,

            "default_accuracy": m_default["accuracy"],
            "default_balanced_accuracy": m_default["balanced_accuracy"],
            "default_recall_TRUE_LEAK": m_default["recall_TRUE_LEAK"],
            "default_recall_FALSE_LEAK": m_default["recall_FALSE_LEAK"],
            "default_tp": m_default["tp"],
            "default_tn": m_default["tn"],
            "default_fp": m_default["fp"],
            "default_fn": m_default["fn"],

            "calibrated_accuracy": m_calib["accuracy"],
            "calibrated_balanced_accuracy": m_calib["balanced_accuracy"],
            "calibrated_recall_TRUE_LEAK": m_calib["recall_TRUE_LEAK"],
            "calibrated_recall_FALSE_LEAK": m_calib["recall_FALSE_LEAK"],
            "calibrated_tp": m_calib["tp"],
            "calibrated_tn": m_calib["tn"],
            "calibrated_fp": m_calib["fp"],
            "calibrated_fn": m_calib["fn"],
        }

        group_summary_rows.append(row)

        # 预测明细
        key_features = [
            "ratio_60_70k",
            "direction_contrast",
            "spec_centroid_hz",
            "spec_flatness",
            "ratio_50_60k",
            "spec_rolloff_85_hz",
            "ratio_40_50k",
            "high_freq_ratio",
            "decay_R2",
            "spec_bandwidth_hz",
            "spec_peakiness",
            "spec_entropy",
            "time_energy_cv",
            "near_far_ratio",
        ]

        for i in range(len(test_df)):
            true_label = binary_to_label(y_test[i])
            pred_default_label = binary_to_label(pred_default[i])
            pred_calib_label = binary_to_label(pred_calib[i])

            pred_row = {
                "test_group": g,
                "dataset": test_df.loc[i, "dataset"] if "dataset" in test_df.columns else "",
                "time": test_df.loc[i, "time"] if "time" in test_df.columns else "",
                "center": test_df.loc[i, "center"] if "center" in test_df.columns else "",
                "true_label": true_label,
                "prob_TRUE_LEAK": float(test_prob[i]),

                "default_threshold": 0.5,
                "default_pred": pred_default_label,
                "default_correct": int(true_label == pred_default_label),

                "calibrated_threshold": best_t,
                "calibrated_pred": pred_calib_label,
                "calibrated_correct": int(true_label == pred_calib_label),
            }

            for k in key_features:
                if k in test_df.columns:
                    pred_row[k] = test_df.loc[i, k]

            all_prediction_rows.append(pred_row)

    group_df = pd.DataFrame(group_summary_rows)
    pred_df = pd.DataFrame(all_prediction_rows)

    group_csv = os.path.join(output_dir, "v6_group_threshold_validation_summary.csv")
    pred_csv = os.path.join(output_dir, "v6_group_threshold_predictions.csv")
    wrong_csv = os.path.join(output_dir, "v6_calibrated_misclassified_samples.csv")

    group_df.to_csv(group_csv, index=False, encoding="utf-8-sig")
    pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")

    if len(pred_df):
        wrong_df = pred_df[pred_df["calibrated_correct"] == 0].copy()
        wrong_df.to_csv(wrong_csv, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(wrong_csv, index=False, encoding="utf-8-sig")

    # 报告
    lines = []
    lines.append("v6 自动阈值校准分组验证报告")
    lines.append("=" * 80)
    lines.append(f"生成时间: {datetime.now()}")
    lines.append(f"输入文件: {MERGED_FEATURE_CSV}")
    lines.append(f"分组列: {GROUP_COL}")
    lines.append(f"阈值优化指标: {THRESHOLD_METRIC}")
    lines.append(f"特征数量: {len(feature_cols)}")
    lines.append("")

    if len(group_df):
        lines.append("整体结果:")
        lines.append(f"默认阈值平均准确率: {group_df['default_accuracy'].mean():.4f}")
        lines.append(f"校准阈值平均准确率: {group_df['calibrated_accuracy'].mean():.4f}")
        lines.append(f"默认阈值平均平衡准确率: {group_df['default_balanced_accuracy'].mean():.4f}")
        lines.append(f"校准阈值平均平衡准确率: {group_df['calibrated_balanced_accuracy'].mean():.4f}")
        lines.append(f"平均AUC: {group_df['auc'].mean():.4f}")
        lines.append("")

        lines.append("各测试组结果:")
        for _, r in group_df.iterrows():
            lines.append(
                f"{r['test_group']}: "
                f"best_t={r['train_oof_best_threshold']:.3f}, "
                f"default_acc={r['default_accuracy']:.3f}, "
                f"calib_acc={r['calibrated_accuracy']:.3f}, "
                f"calib_bal_acc={r['calibrated_balanced_accuracy']:.3f}, "
                f"AUC={r['auc']}"
            )

    if len(pred_df):
        lines.append("")
        lines.append("误判统计:")
        lines.append(f"默认阈值误判数: {int((pred_df['default_correct'] == 0).sum())}")
        lines.append(f"校准阈值误判数: {int((pred_df['calibrated_correct'] == 0).sum())}")

    report_path = os.path.join(output_dir, "v6_threshold_calibration_report.txt")
    save_text(report_path, "\n".join(lines))

    return group_df, pred_df, group_csv, pred_csv, wrong_csv, report_path


def safe_group_name(g):
    return str(g).replace("\\", "_").replace("/", "_").replace(":", "_").replace(".", "_")


# ============================================================
# 4. 全数据最终阈值 + 最终模型
# ============================================================

def train_final_model_with_global_threshold(df, feature_cols, output_dir):
    try:
        import joblib
    except Exception as e:
        raise RuntimeError("缺少 joblib。请运行: pip install joblib") from e

    X = make_numeric_matrix(df, feature_cols)
    y = label_to_binary(df[LABEL_COL].astype(str).values)

    # 用全数据 OOF 概率找全局推荐阈值
    oof_prob = get_oof_probabilities(X, y)

    best_t, best_score, threshold_curve = find_best_threshold(
        y,
        oof_prob,
        metric=THRESHOLD_METRIC,
        grid=THRESHOLD_GRID
    )

    threshold_curve_csv = os.path.join(output_dir, "v6_global_threshold_curve.csv")
    threshold_curve.to_csv(threshold_curve_csv, index=False, encoding="utf-8-sig")

    # 用全数据训练最终模型
    clf = build_classifier()
    clf.fit(X, y)

    model_path = os.path.join(output_dir, "v6_final_leak_classifier.pkl")
    joblib.dump(clf, model_path)

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance_csv = os.path.join(output_dir, "v6_final_feature_importance.csv")
    importance_df.to_csv(importance_csv, index=False, encoding="utf-8-sig")

    config = {
        "model_type": "RandomForestClassifier",
        "positive_label": "TRUE_LEAK",
        "label_mapping": {
            "FALSE_LEAK": 0,
            "TRUE_LEAK": 1,
        },
        "recommended_threshold": best_t,
        "threshold_metric": THRESHOLD_METRIC,
        "threshold_score_on_oof": best_score,
        "feature_cols": feature_cols,
        "drop_absolute_energy_features": DROP_ABSOLUTE_ENERGY_FEATURES,
        "created_at": str(datetime.now()),
        "input_csv": MERGED_FEATURE_CSV,
    }

    config_path = os.path.join(output_dir, "v6_final_model_config.json")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return {
        "model_path": model_path,
        "config_path": config_path,
        "importance_csv": importance_csv,
        "threshold_curve_csv": threshold_curve_csv,
        "recommended_threshold": best_t,
        "threshold_score": best_score,
        "importance_df": importance_df,
    }


# ============================================================
# 5. 画图
# ============================================================

def plot_group_comparison(group_df, output_dir):
    if group_df is None or len(group_df) == 0:
        return None

    fig_dir = os.path.join(output_dir, "figures")
    ensure_dir(fig_dir)

    x = np.arange(len(group_df))
    width = 0.35

    plt.figure(figsize=(11, 5))
    plt.bar(x - width / 2, group_df["default_accuracy"], width, label="Default threshold 0.5")
    plt.bar(x + width / 2, group_df["calibrated_accuracy"], width, label="Calibrated threshold")

    plt.ylim(0, 1.05)
    plt.xticks(x, group_df["test_group"].astype(str), rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Default vs calibrated threshold accuracy by time group")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(fig_dir, "v6_default_vs_calibrated_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def plot_thresholds(group_df, output_dir):
    if group_df is None or len(group_df) == 0:
        return None

    fig_dir = os.path.join(output_dir, "figures")
    ensure_dir(fig_dir)

    plt.figure(figsize=(11, 5))
    plt.plot(group_df["test_group"].astype(str), group_df["train_oof_best_threshold"], marker="o")
    plt.axhline(0.5, linestyle="--", label="Default 0.5")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Threshold")
    plt.title("Calibrated threshold for each leave-one-time-folder-out test")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(fig_dir, "v6_calibrated_thresholds.png")
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def plot_top_importance(importance_df, output_dir, top_n=20):
    if importance_df is None or len(importance_df) == 0:
        return None

    fig_dir = os.path.join(output_dir, "figures")
    ensure_dir(fig_dir)

    top = importance_df.head(top_n).iloc[::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance")
    plt.title(f"v6 final model top {top_n} feature importance")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(fig_dir, "v6_top_feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def plot_global_threshold_curve(threshold_curve_csv, output_dir):
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
    plt.title("Global OOF threshold calibration curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(fig_dir, "v6_global_threshold_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()

    return path


# ============================================================
# 6. 主程序
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)

    print("=" * 80)
    print("v6 自动阈值校准版真假泄漏分类程序")
    print("=" * 80)

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

    print("样本数量:", len(df))
    print(df[LABEL_COL].value_counts())

    feature_cols = get_numeric_features(df)

    if len(feature_cols) == 0:
        print("没有可用数值特征。")
        return

    print("可用特征数量:", len(feature_cols))

    used_feature_path = os.path.join(OUTPUT_DIR, "v6_used_feature_columns.txt")
    save_text(used_feature_path, "\n".join(feature_cols))

    # 分组验证 + 阈值校准
    group_df, pred_df, group_csv, pred_csv, wrong_csv, report_path = group_validation_with_threshold(
        df,
        feature_cols,
        OUTPUT_DIR
    )

    print("\n分组阈值验证汇总:", group_csv)
    print("预测明细:", pred_csv)
    print("校准阈值后的误判样本:", wrong_csv)
    print("报告:", report_path)

    # 训练最终模型 + 全局阈值
    final_info = train_final_model_with_global_threshold(
        df,
        feature_cols,
        OUTPUT_DIR
    )

    print("\n最终模型:", final_info["model_path"])
    print("最终模型配置:", final_info["config_path"])
    print("全局阈值曲线:", final_info["threshold_curve_csv"])
    print("最终模型特征重要性:", final_info["importance_csv"])
    print(f"推荐全局阈值: {final_info['recommended_threshold']:.3f}")
    print(f"OOF阈值优化得分: {final_info['threshold_score']:.3f}")

    # 画图
    fig1 = plot_group_comparison(group_df, OUTPUT_DIR)
    fig2 = plot_thresholds(group_df, OUTPUT_DIR)
    fig3 = plot_top_importance(final_info["importance_df"], OUTPUT_DIR)
    fig4 = plot_global_threshold_curve(final_info["threshold_curve_csv"], OUTPUT_DIR)

    print("\n图片输出:")
    for p in [fig1, fig2, fig3, fig4]:
        if p:
            print(" ", p)

    print("\n最终模型重要特征前10:")
    for _, row in final_info["importance_df"].head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.6f}")

    print("\n" + "=" * 80)
    print("全部完成")
    print("输出文件夹:", OUTPUT_DIR)
    print("=" * 80)


if __name__ == "__main__":
    main()
