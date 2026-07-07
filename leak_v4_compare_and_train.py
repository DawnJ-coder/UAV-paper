# -*- coding: utf-8 -*-
"""
leak_v4_compare_and_train.py

用途:
    读取 v3 生成的两份特征CSV:
        1. 真泄漏 leak_feature_dataset.csv
        2. 假泄漏 leak_feature_dataset.csv

    然后完成:
        1. 合并真假样本
        2. 强制修正 label，避免前面配置没改导致标签错误
        3. 统计真假泄漏特征差异
        4. 训练一个 RandomForest 初步分类器
        5. 输出重要特征排序
        6. 画出最重要特征的真假分布图

运行:
    python leak_v4_compare_and_train.py

输出:
    leak_v4_compare_results/
        merged_feature_dataset.csv
        feature_compare_summary.csv
        feature_importance.csv
        classifier_report.txt
        figures/
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 路径配置：一般只需要改这里
# ============================================================

TRUE_CSV = r"C:\Users\jiangxinru6\Desktop\wurenji\leak_feature_v3_results\leak_feature_dataset.csv"

FALSE_CSV = r"C:\Users\jiangxinru6\Desktop\wurenji\leak_feature_cs_v3_results\leak_feature_dataset.csv"

OUTPUT_DIR = r"C:\Users\jiangxinru6\Desktop\wurenji\leak_v4_compare_results"


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


# ============================================================
# 2. 工具函数
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def cohen_d(a, b):
    """
    Cohen's d:
        衡量两个类别的均值差异大小。
        绝对值越大，说明越能区分真假泄漏。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) < 2 or len(b) < 2:
        return 0.0

    mean_a = np.mean(a)
    mean_b = np.mean(b)

    std_a = np.std(a, ddof=1)
    std_b = np.std(b, ddof=1)

    pooled = np.sqrt((std_a ** 2 + std_b ** 2) / 2.0) + 1e-12

    return float((mean_a - mean_b) / pooled)


def safe_numeric_df(df, feature_cols):
    x = df[feature_cols].copy()

    for c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(0.0)

    return x


# ============================================================
# 3. 主流程
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)

    fig_dir = os.path.join(OUTPUT_DIR, "figures")
    ensure_dir(fig_dir)

    print("=" * 80)
    print("v4 真假泄漏特征对比 + 初步分类器训练")
    print("=" * 80)

    if not os.path.exists(TRUE_CSV):
        print("真泄漏CSV不存在:", TRUE_CSV)
        return

    if not os.path.exists(FALSE_CSV):
        print("假泄漏CSV不存在:", FALSE_CSV)
        return

    true_df = pd.read_csv(TRUE_CSV)
    false_df = pd.read_csv(FALSE_CSV)

    # 强制修正标签，防止 v3 跑假泄漏时忘记改 label
    true_df["label"] = "TRUE_LEAK"
    false_df["label"] = "FALSE_LEAK"

    true_df["dataset"] = "true_leak"
    false_df["dataset"] = "false_leak"

    df = pd.concat([true_df, false_df], ignore_index=True)

    merged_csv = os.path.join(OUTPUT_DIR, "merged_feature_dataset.csv")
    df.to_csv(merged_csv, index=False, encoding="utf-8-sig")

    print("已合并特征表:", merged_csv)
    print("总样本数:", len(df))
    print("真泄漏样本数:", int((df["label"] == "TRUE_LEAK").sum()))
    print("假泄漏样本数:", int((df["label"] == "FALSE_LEAK").sum()))

    # ========================================================
    # 4. 选择数值特征
    # ========================================================

    feature_cols = []

    for c in df.columns:
        if c in DROP_COLS:
            continue

        # 尝试转成数字
        temp = pd.to_numeric(df[c], errors="coerce")
        valid_ratio = temp.notna().mean()

        if valid_ratio > 0.8:
            feature_cols.append(c)

    print("可用数值特征数量:", len(feature_cols))

    if len(feature_cols) == 0:
        print("没有可用数值特征，请检查CSV。")
        return

    X = safe_numeric_df(df, feature_cols)
    y = df["label"].astype(str)

    # ========================================================
    # 5. 真假特征统计对比
    # ========================================================

    compare_rows = []

    true_mask = y == "TRUE_LEAK"
    false_mask = y == "FALSE_LEAK"

    for c in feature_cols:
        true_vals = X.loc[true_mask, c].values
        false_vals = X.loc[false_mask, c].values

        row = {
            "feature": c,
            "true_mean": float(np.mean(true_vals)),
            "false_mean": float(np.mean(false_vals)),
            "true_std": float(np.std(true_vals)),
            "false_std": float(np.std(false_vals)),
            "mean_diff_true_minus_false": float(np.mean(true_vals) - np.mean(false_vals)),
            "abs_mean_diff": float(abs(np.mean(true_vals) - np.mean(false_vals))),
            "cohen_d_true_minus_false": cohen_d(true_vals, false_vals),
            "abs_cohen_d": abs(cohen_d(true_vals, false_vals)),
        }

        compare_rows.append(row)

    compare_df = pd.DataFrame(compare_rows)
    compare_df = compare_df.sort_values("abs_cohen_d", ascending=False)

    compare_csv = os.path.join(OUTPUT_DIR, "feature_compare_summary.csv")
    compare_df.to_csv(compare_csv, index=False, encoding="utf-8-sig")

    print("已保存真假特征对比:", compare_csv)

    print("\n真假差异最大的前10个特征:")
    for _, row in compare_df.head(10).iterrows():
        print(
            f"  {row['feature']}: "
            f"TRUE均值={row['true_mean']:.4g}, "
            f"FALSE均值={row['false_mean']:.4g}, "
            f"|d|={row['abs_cohen_d']:.3f}"
        )

    # ========================================================
    # 6. 训练初步分类器
    # ========================================================

    try:
        from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    except Exception as e:
        print("\n未安装 scikit-learn，跳过分类器训练。")
        print("可以在命令行安装:")
        print("pip install scikit-learn pandas matplotlib")
        print("错误信息:", e)
        return

    if len(df) < 10:
        print("\n样本数量太少，不训练分类器。建议真假样本总数至少几十个。")
        return

    if y.nunique() < 2:
        print("\n只有一种标签，无法训练真假分类器。")
        return

    clf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        max_depth=None,
    )

    report_path = os.path.join(OUTPUT_DIR, "classifier_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("v4 RandomForest 初步分类报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"总样本数: {len(df)}\n")
        f.write(f"真泄漏样本数: {int((y == 'TRUE_LEAK').sum())}\n")
        f.write(f"假泄漏样本数: {int((y == 'FALSE_LEAK').sum())}\n")
        f.write(f"特征数量: {len(feature_cols)}\n\n")

        # 小样本时，分层划分可能失败，做保护
        min_class_count = y.value_counts().min()

        if min_class_count >= 2 and len(df) >= 10:
            test_size = 0.3

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
                stratify=y
            )

            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)

            acc = accuracy_score(y_test, pred)
            report = classification_report(y_test, pred)
            cm = confusion_matrix(y_test, pred, labels=["TRUE_LEAK", "FALSE_LEAK"])

            print("\n初步测试集准确率:", round(acc, 3))
            print("分类报告已保存:", report_path)

            f.write("Train/Test split 结果\n")
            f.write("-" * 80 + "\n")
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write("Classification report:\n")
            f.write(report + "\n\n")
            f.write("Confusion matrix, labels=[TRUE_LEAK, FALSE_LEAK]:\n")
            f.write(str(cm) + "\n\n")

        # 交叉验证
        if min_class_count >= 3:
            n_splits = min(5, int(min_class_count))
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

            f.write("Cross validation accuracy:\n")
            f.write(str(scores) + "\n")
            f.write(f"Mean CV accuracy: {np.mean(scores):.4f}\n")
            f.write(f"Std CV accuracy: {np.std(scores):.4f}\n\n")

            print("交叉验证平均准确率:", round(float(np.mean(scores)), 3))

        # 用全量数据重新训练，用于重要性分析
        clf.fit(X, y)

        importances = pd.DataFrame({
            "feature": feature_cols,
            "importance": clf.feature_importances_
        }).sort_values("importance", ascending=False)

        importance_csv = os.path.join(OUTPUT_DIR, "feature_importance.csv")
        importances.to_csv(importance_csv, index=False, encoding="utf-8-sig")

        print("已保存特征重要性:", importance_csv)

        f.write("\nTop 30 feature importances:\n")
        f.write("-" * 80 + "\n")
        for _, row in importances.head(30).iterrows():
            f.write(f"{row['feature']}: {row['importance']:.6f}\n")

    # ========================================================
    # 7. 画前几个重要特征的真假分布图
    # ========================================================

    top_features = []

    # 优先使用模型重要性
    try:
        for c in importances["feature"].head(8).tolist():
            if c not in top_features:
                top_features.append(c)
    except Exception:
        pass

    # 再补充统计差异大的特征
    for c in compare_df["feature"].head(8).tolist():
        if c not in top_features:
            top_features.append(c)

    top_features = top_features[:10]

    for c in top_features:
        plt.figure(figsize=(8, 5))

        true_vals = X.loc[true_mask, c].values
        false_vals = X.loc[false_mask, c].values

        plt.hist(true_vals, bins=20, alpha=0.6, label="TRUE_LEAK")
        plt.hist(false_vals, bins=20, alpha=0.6, label="FALSE_LEAK")

        plt.title(f"Feature distribution: {c}")
        plt.xlabel(c)
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, alpha=0.3)

        fig_path = os.path.join(fig_dir, f"{c}.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()

    print("已保存特征分布图:", fig_dir)

    print("\n" + "=" * 80)
    print("全部完成")
    print("输出文件夹:", OUTPUT_DIR)
    print("=" * 80)


if __name__ == "__main__":
    main()
