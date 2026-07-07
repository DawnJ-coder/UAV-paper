# -*- coding: utf-8 -*-
"""
leak_v5_group_validate_and_build_model.py

v5：按时间点分组验证 + 数据质量检查 + 最终分类模型训练

为什么需要 v5？
    v4 的随机划分准确率很高，但随机划分可能把同一个时间点/相邻中心点的数据
    同时放进训练集和测试集，导致结果偏乐观。

v5 做什么？
    1. 读取 v4 生成的 merged_feature_dataset.csv
    2. 检查特征是否有大量 0、空值、无效频率
    3. 按 time 整组留出验证：
        例如拿 HM20260626_142938.ld 整个时间点当测试集，
        其余时间点当训练集。
    4. 输出每个时间点的测试准确率
    5. 输出所有误判样本
    6. 输出重要特征
    7. 训练最终模型并保存为 pkl
    8. 额外生成一个 final_feature_config.json，后面实时预测程序会用到

运行：
    python leak_v5_group_validate_and_build_model.py

输入：
    C:\Users\jiangxinru6\Desktop\wurenji\leak_v4_compare_results\merged_feature_dataset.csv

输出：
    C:\Users\jiangxinru6\Desktop\wurenji\leak_v5_group_validation_results\
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
# 1. 路径配置：一般只需要改这里
# ============================================================

MERGED_FEATURE_CSV = r"C:\Users\jiangxinru6\Desktop\wurenji\leak_v4_compare_results\merged_feature_dataset.csv"

OUTPUT_DIR = r"C:\Users\jiangxinru6\Desktop\wurenji\leak_v5_group_validation_results"

# 分组列。建议用 time。
# 含义：每次把一个完整 time_folder 留出来作为测试集。
GROUP_COL = "time"

# 标签列
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

# 明显容易受绝对幅值影响的特征可以保留，也可以删掉。
# 如果不同批次音量差异很大，可以把 raw_best_energy / energy_20_70 这类绝对能量列删掉。
# 默认先保留，因为当前结果显示它们可能有用。
DROP_ABSOLUTE_ENERGY_FEATURES = False

ABSOLUTE_ENERGY_KEYWORDS = [
    "energy_20_70",
    "energy_20_40",
    "energy_40_70",
    "energy_20_30k",
    "energy_30_40k",
    "energy_40_50k",
    "energy_50_60k",
    "energy_60_70k",
    "raw_best_energy",
    "mean_energy",
    "energy_5cm",
    "time_energy_mean",
    "time_energy_std",
    "time_rms",
]


# ============================================================
# 2. 工具函数
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_numeric_features(df):
    """
    自动筛选可用于训练的数值列。
    """
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
    """
    转数值矩阵，处理空值和无穷值。
    """
    x = df[feature_cols].copy()

    for c in feature_cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(0.0)

    return x


def label_to_binary(y):
    """
    TRUE_LEAK -> 1
    FALSE_LEAK -> 0
    """
    return np.array([1 if str(v) == "TRUE_LEAK" else 0 for v in y], dtype=int)


def binary_to_label(v):
    return "TRUE_LEAK" if int(v) == 1 else "FALSE_LEAK"


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        return 0.0

    return float(np.mean(y_true == y_pred))


def safe_auc(y_true, prob):
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, prob))
    except Exception:
        return np.nan


def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ============================================================
# 3. 数据质量检查
# ============================================================

def data_quality_check(df, feature_cols, output_dir):
    """
    检查：
        1. 空值比例
        2. 0值比例
        3. 按真假分别统计均值
        4. 频率类特征是否出现大量低于20kHz/等于0
    """
    rows = []

    y = df[LABEL_COL].astype(str)

    for c in feature_cols:
        vals = pd.to_numeric(df[c], errors="coerce")
        vals_numeric = vals.replace([np.inf, -np.inf], np.nan)

        nan_ratio = float(vals_numeric.isna().mean())
        zero_ratio = float((vals_numeric.fillna(0) == 0).mean())

        true_vals = vals_numeric[y == "TRUE_LEAK"].dropna()
        false_vals = vals_numeric[y == "FALSE_LEAK"].dropna()

        true_mean = float(true_vals.mean()) if len(true_vals) else np.nan
        false_mean = float(false_vals.mean()) if len(false_vals) else np.nan

        true_zero_ratio = float((true_vals.fillna(0) == 0).mean()) if len(true_vals) else np.nan
        false_zero_ratio = float((false_vals.fillna(0) == 0).mean()) if len(false_vals) else np.nan

        rows.append({
            "feature": c,
            "nan_ratio": nan_ratio,
            "zero_ratio": zero_ratio,
            "true_mean": true_mean,
            "false_mean": false_mean,
            "true_zero_ratio": true_zero_ratio,
            "false_zero_ratio": false_zero_ratio,
        })

    qdf = pd.DataFrame(rows)
    qdf = qdf.sort_values(["nan_ratio", "zero_ratio"], ascending=False)

    quality_csv = os.path.join(output_dir, "data_quality_report.csv")
    qdf.to_csv(quality_csv, index=False, encoding="utf-8-sig")

    # 额外检查频率类特征
    freq_check_rows = []

    freq_cols = [
        c for c in feature_cols
        if ("freq" in c.lower() or "rolloff" in c.lower() or "centroid" in c.lower())
    ]

    for c in freq_cols:
        vals = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
        low_ratio = float((vals < 20000).mean())
        zero_ratio = float((vals == 0).mean())

        freq_check_rows.append({
            "feature": c,
            "ratio_below_20k": low_ratio,
            "zero_ratio": zero_ratio,
            "min": float(vals.min()),
            "mean": float(vals.mean()),
            "max": float(vals.max()),
        })

    freq_df = pd.DataFrame(freq_check_rows)

    freq_csv = os.path.join(output_dir, "frequency_feature_validity_report.csv")
    freq_df.to_csv(freq_csv, index=False, encoding="utf-8-sig")

    return quality_csv, freq_csv


# ============================================================
# 4. 按时间点整组留出验证
# ============================================================

def leave_one_group_out_validation(df, feature_cols, output_dir):
    """
    每次留出一个 time_folder 作为测试集。
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix
    except Exception as e:
        raise RuntimeError(
            "缺少 scikit-learn。请先运行: pip install scikit-learn pandas matplotlib"
        ) from e

    X = make_numeric_matrix(df, feature_cols)
    y_label = df[LABEL_COL].astype(str).values
    y = label_to_binary(y_label)

    groups = df[GROUP_COL].astype(str).values
    unique_groups = sorted(pd.unique(groups).tolist())

    all_pred_rows = []
    group_rows = []

    print("\n开始按时间点整组留出验证...")
    print("分组数量:", len(unique_groups))

    for g in unique_groups:
        test_mask = groups == g
        train_mask = ~test_mask

        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]

        # 训练集必须同时有真假两类
        if len(np.unique(y_train)) < 2:
            print(f"  [跳过] 测试组 {g}: 训练集只有一种标签")
            continue

        if len(X_test) == 0:
            continue

        clf = RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced",
            max_depth=None,
            min_samples_leaf=1,
        )

        clf.fit(X_train, y_train)

        pred = clf.predict(X_test)

        if hasattr(clf, "predict_proba"):
            prob_true = clf.predict_proba(X_test)[:, 1]
        else:
            prob_true = pred.astype(float)

        acc = accuracy(y_test, pred)
        auc = safe_auc(y_test, prob_true)

        n_true = int(np.sum(y_test == 1))
        n_false = int(np.sum(y_test == 0))

        print(
            f"  测试组 {g}: "
            f"样本={len(y_test)}, 真={n_true}, 假={n_false}, "
            f"acc={acc:.3f}, auc={auc if not np.isnan(auc) else 'NA'}"
        )

        group_rows.append({
            "test_group": g,
            "n_test": len(y_test),
            "n_true": n_true,
            "n_false": n_false,
            "accuracy": acc,
            "auc": auc,
        })

        sub = df.loc[test_mask].copy().reset_index(drop=True)

        for i in range(len(sub)):
            true_label = binary_to_label(y_test[i])
            pred_label = binary_to_label(pred[i])

            row = {
                "test_group": g,
                "dataset": sub.loc[i, "dataset"] if "dataset" in sub.columns else "",
                "time": sub.loc[i, "time"] if "time" in sub.columns else "",
                "center": sub.loc[i, "center"] if "center" in sub.columns else "",
                "true_label": true_label,
                "pred_label": pred_label,
                "prob_TRUE_LEAK": float(prob_true[i]),
                "correct": int(true_label == pred_label),
            }

            # 附带关键特征，方便看误判原因
            key_features = [
                "ratio_60_70k",
                "spec_flatness",
                "direction_contrast",
                "high_freq_ratio",
                "decay_R2",
                "spec_bandwidth_hz",
                "spec_peakiness",
                "spec_entropy",
                "time_energy_cv",
                "near_far_ratio",
            ]

            for k in key_features:
                if k in sub.columns:
                    row[k] = sub.loc[i, k]

            all_pred_rows.append(row)

    group_df = pd.DataFrame(group_rows)
    pred_df = pd.DataFrame(all_pred_rows)

    group_csv = os.path.join(output_dir, "group_validation_summary.csv")
    pred_csv = os.path.join(output_dir, "group_validation_predictions.csv")
    wrong_csv = os.path.join(output_dir, "misclassified_samples.csv")

    group_df.to_csv(group_csv, index=False, encoding="utf-8-sig")
    pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")

    if len(pred_df):
        wrong_df = pred_df[pred_df["correct"] == 0].copy()
        wrong_df.to_csv(wrong_csv, index=False, encoding="utf-8-sig")
    else:
        wrong_df = pd.DataFrame()
        wrong_df.to_csv(wrong_csv, index=False, encoding="utf-8-sig")

    # 总报告
    report_lines = []
    report_lines.append("v5 按时间点整组留出验证报告")
    report_lines.append("=" * 80)
    report_lines.append(f"生成时间: {datetime.now()}")
    report_lines.append(f"总样本数: {len(df)}")
    report_lines.append(f"特征数量: {len(feature_cols)}")
    report_lines.append(f"分组列: {GROUP_COL}")
    report_lines.append("")

    if len(group_df):
        mean_acc = group_df["accuracy"].mean()
        std_acc = group_df["accuracy"].std()
        report_lines.append(f"平均分组准确率: {mean_acc:.4f}")
        report_lines.append(f"分组准确率标准差: {std_acc:.4f}")
        report_lines.append("")

        report_lines.append("每个测试组结果:")
        for _, r in group_df.iterrows():
            report_lines.append(
                f"  {r['test_group']}: "
                f"n={int(r['n_test'])}, "
                f"true={int(r['n_true'])}, "
                f"false={int(r['n_false'])}, "
                f"acc={r['accuracy']:.4f}, "
                f"auc={r['auc']}"
            )

    if len(pred_df):
        overall_acc = pred_df["correct"].mean()
        report_lines.append("")
        report_lines.append(f"所有留出预测汇总准确率: {overall_acc:.4f}")
        report_lines.append(f"误判数量: {int((pred_df['correct'] == 0).sum())}")

    report_path = os.path.join(output_dir, "group_validation_report.txt")
    save_text(report_path, "\n".join(report_lines))

    return group_df, pred_df, group_csv, pred_csv, wrong_csv, report_path


# ============================================================
# 5. 训练最终模型并保存
# ============================================================

def train_final_model(df, feature_cols, output_dir):
    """
    用全部数据训练最终模型，保存模型和特征列表。
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        import joblib
    except Exception as e:
        raise RuntimeError(
            "缺少 scikit-learn 或 joblib。请先运行: pip install scikit-learn joblib"
        ) from e

    X = make_numeric_matrix(df, feature_cols)
    y = label_to_binary(df[LABEL_COL].astype(str).values)

    clf = RandomForestClassifier(
        n_estimators=700,
        random_state=42,
        class_weight="balanced",
        max_depth=None,
        min_samples_leaf=1,
    )

    clf.fit(X, y)

    # 保存模型
    model_path = os.path.join(output_dir, "final_leak_classifier_random_forest.pkl")
    joblib.dump(clf, model_path)

    # 保存特征列表
    config = {
        "model_type": "RandomForestClassifier",
        "label_mapping": {
            "FALSE_LEAK": 0,
            "TRUE_LEAK": 1,
        },
        "positive_label": "TRUE_LEAK",
        "feature_cols": feature_cols,
        "drop_absolute_energy_features": DROP_ABSOLUTE_ENERGY_FEATURES,
        "created_at": str(datetime.now()),
        "input_csv": MERGED_FEATURE_CSV,
    }

    config_path = os.path.join(output_dir, "final_feature_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 特征重要性
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance_csv = os.path.join(output_dir, "final_model_feature_importance.csv")
    importance_df.to_csv(importance_csv, index=False, encoding="utf-8-sig")

    return model_path, config_path, importance_csv, importance_df


# ============================================================
# 6. 画图
# ============================================================

def plot_group_accuracy(group_df, output_dir):
    if group_df is None or len(group_df) == 0:
        return None

    fig_dir = os.path.join(output_dir, "figures")
    ensure_dir(fig_dir)

    plt.figure(figsize=(10, 5))
    plt.bar(group_df["test_group"].astype(str), group_df["accuracy"])
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Leave-one-time-folder-out validation accuracy")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(fig_dir, "group_validation_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def plot_top_feature_importance(importance_df, output_dir, top_n=20):
    if importance_df is None or len(importance_df) == 0:
        return None

    fig_dir = os.path.join(output_dir, "figures")
    ensure_dir(fig_dir)

    top = importance_df.head(top_n).iloc[::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} feature importance")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(fig_dir, "top_feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()

    return path


# ============================================================
# 7. 主程序
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)

    print("=" * 80)
    print("v5 按时间点分组验证 + 最终模型训练")
    print("=" * 80)

    if not os.path.exists(MERGED_FEATURE_CSV):
        print("找不到输入文件:", MERGED_FEATURE_CSV)
        print("请先运行 v4，生成 merged_feature_dataset.csv")
        return

    df = pd.read_csv(MERGED_FEATURE_CSV)

    if LABEL_COL not in df.columns:
        print("CSV 中没有 label 列，无法训练。")
        return

    if GROUP_COL not in df.columns:
        print(f"CSV 中没有 {GROUP_COL} 列，无法按组验证。")
        return

    # 只保留 TRUE_LEAK / FALSE_LEAK
    df[LABEL_COL] = df[LABEL_COL].astype(str)
    df = df[df[LABEL_COL].isin(["TRUE_LEAK", "FALSE_LEAK"])].copy()

    print("样本数:", len(df))
    print(df[LABEL_COL].value_counts())

    feature_cols = get_numeric_features(df)

    if len(feature_cols) == 0:
        print("没有可用数值特征。")
        return

    print("可用特征数:", len(feature_cols))

    feature_list_path = os.path.join(OUTPUT_DIR, "used_feature_columns.txt")
    save_text(feature_list_path, "\n".join(feature_cols))

    # 数据质量检查
    quality_csv, freq_csv = data_quality_check(df, feature_cols, OUTPUT_DIR)
    print("数据质量报告:", quality_csv)
    print("频率特征有效性报告:", freq_csv)

    # 分组验证
    group_df, pred_df, group_csv, pred_csv, wrong_csv, report_path = leave_one_group_out_validation(
        df,
        feature_cols,
        OUTPUT_DIR
    )

    print("分组验证汇总:", group_csv)
    print("分组验证预测明细:", pred_csv)
    print("误判样本:", wrong_csv)
    print("分组验证报告:", report_path)

    # 训练最终模型
    model_path, config_path, importance_csv, importance_df = train_final_model(
        df,
        feature_cols,
        OUTPUT_DIR
    )

    print("最终模型:", model_path)
    print("模型特征配置:", config_path)
    print("最终模型特征重要性:", importance_csv)

    # 画图
    acc_fig = plot_group_accuracy(group_df, OUTPUT_DIR)
    imp_fig = plot_top_feature_importance(importance_df, OUTPUT_DIR)

    if acc_fig:
        print("分组准确率图:", acc_fig)

    if imp_fig:
        print("重要特征图:", imp_fig)

    print("\n重要特征前10:")
    for _, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.6f}")

    print("\n" + "=" * 80)
    print("全部完成")
    print("输出文件夹:", OUTPUT_DIR)
    print("=" * 80)


if __name__ == "__main__":
    main()
