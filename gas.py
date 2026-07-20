# -*- coding: utf-8 -*-
"""
读取 v13_1_AB_train_CD_test_outer40_raw_plane.py 的 raw/plane 结果，生成：

1. 每个场景中，哪些 TRUE_LEAK / FALSE_LEAK 样本符合“T分数应高于F”的趋势；
2. 哪些样本不符合趋势，以及它们偏离对立类别中位数多少；
3. 外圈40点每个样本的原始鲁棒权重、归一化权重、实际平面影响比例；
4. 每个外圈点跨样本的中位数、最小值、最大值和排名；
5. 分数分布图、异常样本图、外圈40点权重柱状图、空间分布图和热图。

推荐运行：
python plot_v13_TF_trend_and_outer40_weights.py ^
  --result-root "D:\\wurenji\\v13_AB_train_CD_test_outer40_results" ^
  --output-dir "D:\\wurenji\\v13_AB_train_CD_test_outer40_results\\TF趋势与外圈40点权重图"

结果目录结构要求：
result_root/
├─ raw/
│  ├─ train_AB/
│  │  ├─ v13_loso_oof_scores.csv
│  │  └─ v13_train_all_point_weights.csv
│  └─ test_CD/
│     ├─ v13_external_test_predictions.csv
│     └─ v13_external_test_all_point_weights.csv
└─ plane/
   ├─ train_AB/
   └─ test_CD/
"""
from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Windows优先使用微软雅黑/黑体，避免中文标题显示成方框。
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
        "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
except Exception as exc:  # pragma: no cover
    raise RuntimeError("缺少 matplotlib，请运行: pip install matplotlib") from exc


EPS = 1.0e-12
VALID_METHODS = ("raw", "plane")
TRUE_LABEL = "TRUE_LEAK"
FALSE_LABEL = "FALSE_LEAK"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_name(text: object, max_len: int = 120) -> str:
    value = str(text).strip() or "unknown"
    chars = []
    for ch in value:
        chars.append(ch if (ch.isalnum() or ch in "-_.") else "_")
    return "".join(chars)[:max_len]


def require_columns(df: pd.DataFrame, required: Sequence[str], source: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{source} 缺少列: {missing}")


def normalize_label(values: pd.Series) -> pd.Series:
    mapping = {
        "T": TRUE_LABEL,
        "TRUE": TRUE_LABEL,
        "TRUE_LEAK": TRUE_LABEL,
        "1": TRUE_LABEL,
        "F": FALSE_LABEL,
        "FALSE": FALSE_LABEL,
        "FALSE_LEAK": FALSE_LABEL,
        "0": FALSE_LABEL,
    }
    return values.astype(str).str.strip().str.upper().map(mapping).fillna(values.astype(str).str.strip())


def choose_column(df: pd.DataFrame, candidates: Sequence[str], source: Path) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"{source} 找不到候选列中的任何一个: {list(candidates)}")


def load_score_file(path: Path, method: str, stage: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到分数文件: {path}")
    df = pd.read_csv(path, dtype={"sample_id": str})
    require_columns(df, ["sample_id"], path)

    label_col = choose_column(df, ["true_label", "label", "input_label_ignored"], path)
    score_col = choose_column(
        df,
        ["shared_leak_score_oof", "shared_leak_score", "shared_leak_score_raw"],
        path,
    )
    scene_col = choose_column(df, ["test_scene", "scene"], path)

    out = pd.DataFrame({
        "method": method,
        "stage": stage,
        "scene": df[scene_col].astype(str),
        "sample_id": df["sample_id"].astype(str),
        "true_label": normalize_label(df[label_col]),
        "shared_leak_score": pd.to_numeric(df[score_col], errors="coerce"),
    })

    for col in [
        "dataset", "time", "center", "prediction", "binary_prediction",
        "prediction_3way_oof", "prediction_binary_oof", "probability_TRUE",
        "probability_TRUE_oof", "quality_warning",
    ]:
        if col in df.columns:
            out[col] = df[col]

    out = out[out["true_label"].isin([TRUE_LABEL, FALSE_LABEL])].copy()
    out = out[np.isfinite(out["shared_leak_score"])].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"{path} 没有有效的TRUE/FALSE分数")
    return out


def discover_scores(result_root: Path, methods: Sequence[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for method in methods:
        train_path = result_root / method / "train_AB" / "v13_loso_oof_scores.csv"
        test_path = result_root / method / "test_CD" / "v13_external_test_predictions.csv"
        frames.append(load_score_file(train_path, method, "AB_OOF"))
        frames.append(load_score_file(test_path, method, "CD_EXTERNAL"))
    return pd.concat(frames, ignore_index=True)


def pairwise_fraction_greater(value: float, others: np.ndarray) -> float:
    others = np.asarray(others, dtype=float)
    if len(others) == 0:
        return float("nan")
    return float(np.mean(value > others))


def analyze_tf_trend(scores: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    detail_frames: List[pd.DataFrame] = []
    summaries: List[Dict[str, object]] = []

    group_cols = ["method", "stage", "scene"]
    for keys, group in scores.groupby(group_cols, sort=True):
        method, stage, scene = keys
        g = group.copy().reset_index(drop=True)
        true_scores = g.loc[g["true_label"] == TRUE_LABEL, "shared_leak_score"].to_numpy(float)
        false_scores = g.loc[g["true_label"] == FALSE_LABEL, "shared_leak_score"].to_numpy(float)
        if len(true_scores) == 0 or len(false_scores) == 0:
            raise ValueError(f"{method}/{scene} 缺少TRUE或FALSE，无法判断T>F")

        t_median = float(np.median(true_scores))
        f_median = float(np.median(false_scores))
        t_min, t_max = float(np.min(true_scores)), float(np.max(true_scores))
        f_min, f_max = float(np.min(false_scores)), float(np.max(false_scores))

        rows: List[Dict[str, object]] = []
        for _, row in g.iterrows():
            label = str(row["true_label"])
            score = float(row["shared_leak_score"])
            if label == TRUE_LABEL:
                opposite_median = f_median
                signed_margin = score - f_median
                conformity = pairwise_fraction_greater(score, false_scores)
                conforms = bool(score > f_median)
                strict = bool(score > f_max)
                opposite_relation = "T_score > F_median"
                violating_count = int(np.sum(false_scores >= score))
                opposite_count = int(len(false_scores))
            else:
                opposite_median = t_median
                signed_margin = t_median - score
                conformity = pairwise_fraction_greater(score * -1.0, true_scores * -1.0)
                conforms = bool(score < t_median)
                strict = bool(score < t_min)
                opposite_relation = "F_score < T_median"
                violating_count = int(np.sum(true_scores <= score))
                opposite_count = int(len(true_scores))

            result = row.to_dict()
            result.update({
                "T_score_median": t_median,
                "F_score_median": f_median,
                "scene_median_gap_T_minus_F": t_median - f_median,
                "scene_T_median_greater_than_F_median": int(t_median > f_median),
                "opposite_class_median": opposite_median,
                "trend_rule": opposite_relation,
                "margin_to_opposite_median_positive_is_good": signed_margin,
                "opposite_class_pairwise_conformity_ratio": conformity,
                "opposite_class_pairwise_conformity_percent": 100.0 * conformity,
                "opposite_class_violating_count": violating_count,
                "opposite_class_sample_count": opposite_count,
                "trend_status": "符合趋势" if conforms else "不符合趋势",
                "strict_trend_status": "严格符合" if strict else "非严格符合",
            })
            rows.append(result)

        detail = pd.DataFrame(rows)
        detail_frames.append(detail)

        pairwise_total = np.array(
            [float(t > f) for t in true_scores for f in false_scores], dtype=float
        )
        summaries.append({
            "method": method,
            "stage": stage,
            "scene": scene,
            "n_TRUE": len(true_scores),
            "n_FALSE": len(false_scores),
            "T_score_median": t_median,
            "F_score_median": f_median,
            "median_gap_T_minus_F": t_median - f_median,
            "scene_T_median_greater_than_F_median": int(t_median > f_median),
            "pairwise_probability_T_greater_than_F": float(np.mean(pairwise_total)),
            "pairwise_probability_T_greater_than_F_percent": 100.0 * float(np.mean(pairwise_total)),
            "all_T_greater_than_all_F": int(t_min > f_max),
            "n_TRUE_conforming": int(np.sum((detail["true_label"] == TRUE_LABEL) & (detail["trend_status"] == "符合趋势"))),
            "n_TRUE_nonconforming": int(np.sum((detail["true_label"] == TRUE_LABEL) & (detail["trend_status"] == "不符合趋势"))),
            "n_FALSE_conforming": int(np.sum((detail["true_label"] == FALSE_LABEL) & (detail["trend_status"] == "符合趋势"))),
            "n_FALSE_nonconforming": int(np.sum((detail["true_label"] == FALSE_LABEL) & (detail["trend_status"] == "不符合趋势"))),
            "n_total_conforming": int(np.sum(detail["trend_status"] == "符合趋势")),
            "n_total_nonconforming": int(np.sum(detail["trend_status"] == "不符合趋势")),
        })

    return pd.concat(detail_frames, ignore_index=True), pd.DataFrame(summaries)


def deterministic_jitter(n: int, width: float = 0.11) -> np.ndarray:
    if n <= 1:
        return np.zeros(n)
    return np.linspace(-width, width, n)


def short_id(value: object, max_len: int = 28) -> str:
    text = str(value)
    return text if len(text) <= max_len else text[:max_len - 3] + "..."


def plot_score_scene(group: pd.DataFrame, output_path: Path) -> None:
    method = str(group["method"].iloc[0])
    stage = str(group["stage"].iloc[0])
    scene = str(group["scene"].iloc[0])
    fig, ax = plt.subplots(figsize=(11, 7))

    for x_base, label, label_short in [(0.0, FALSE_LABEL, "F"), (1.0, TRUE_LABEL, "T")]:
        part = group[group["true_label"] == label].sort_values("shared_leak_score").reset_index(drop=True)
        good = part[part["trend_status"] == "符合趋势"]
        bad = part[part["trend_status"] == "不符合趋势"]

        if len(good):
            idx = part.index[part["trend_status"] == "符合趋势"].to_numpy()
            jit = deterministic_jitter(len(part))[idx]
            ax.scatter(np.full(len(good), x_base) + jit, good["shared_leak_score"], marker="o", s=55, label=f"{label_short} 符合趋势")
        if len(bad):
            idx = part.index[part["trend_status"] == "不符合趋势"].to_numpy()
            jit = deterministic_jitter(len(part))[idx]
            ax.scatter(np.full(len(bad), x_base) + jit, bad["shared_leak_score"], marker="x", s=75, linewidths=1.8, label=f"{label_short} 不符合趋势")
            for j, (_, row) in enumerate(bad.iterrows()):
                ax.annotate(
                    short_id(row["sample_id"]),
                    (x_base + 0.13 + 0.015 * (j % 3), float(row["shared_leak_score"])),
                    fontsize=7,
                    va="center",
                )

    t_med = float(group["T_score_median"].iloc[0])
    f_med = float(group["F_score_median"].iloc[0])
    ax.hlines(f_med, -0.27, 0.27, linestyles="--", linewidth=2, label=f"F中位数={f_med:.4g}")
    ax.hlines(t_med, 0.73, 1.27, linestyles="--", linewidth=2, label=f"T中位数={t_med:.4g}")
    ax.set_xlim(-0.45, 1.65)
    ax.set_xticks([0, 1], ["FALSE_LEAK", "TRUE_LEAK"])
    ax.set_ylabel("shared_leak_score（越高越接近T）")
    ax.set_title(f"{method} | {scene} | {stage}\nT应高于F；×号为不符合趋势的样本")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_nonconforming_margin(group: pd.DataFrame, output_path: Path) -> None:
    bad = group[group["trend_status"] == "不符合趋势"].copy()
    if bad.empty:
        return
    bad = bad.sort_values("margin_to_opposite_median_positive_is_good")
    labels = [f"{str(v)[0]} | {short_id(s, 35)}" for v, s in zip(bad["true_label"], bad["sample_id"])]
    y = np.arange(len(bad))
    fig, ax = plt.subplots(figsize=(12, max(5, 0.33 * len(bad) + 2)))
    ax.barh(y, bad["margin_to_opposite_median_positive_is_good"].to_numpy(float))
    ax.axvline(0.0, linewidth=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("相对对立类别中位数的趋势余量（负数表示不符合）")
    method = str(group["method"].iloc[0])
    scene = str(group["scene"].iloc[0])
    ax.set_title(f"{method} | {scene}：不符合T>F趋势的样本")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_trend_summary(summary: pd.DataFrame, method: str, output_path: Path) -> None:
    part = summary[summary["method"] == method].copy()
    if part.empty:
        return
    part["scene_stage"] = part["scene"].astype(str) + "\n" + part["stage"].astype(str)
    x = np.arange(len(part))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, part["n_total_conforming"], width, label="符合趋势")
    ax.bar(x + width / 2, part["n_total_nonconforming"], width, label="不符合趋势")
    ax.set_xticks(x, part["scene_stage"])
    ax.set_ylabel("样本数")
    ax.set_title(f"{method}：各场景T>F趋势符合情况")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_weight_file(path: Path, method: str, stage: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到权重文件: {path}")
    df = pd.read_csv(path, dtype={"sample_id": str, "point_id": str})
    required = [
        "sample_id", "scene", "role", "used_for_background",
        "robust_fit_weight_raw", "robust_fit_weight_percent",
        "plane_influence_coefficient_signed", "plane_absolute_influence_percent",
    ]
    require_columns(df, required, path)
    if "point_id" not in df.columns:
        df["point_id"] = "point_" + pd.to_numeric(df.get("point_index", np.arange(len(df))), errors="coerce").fillna(-1).astype(int).astype(str)
    for col in ["x_cm", "y_cm", "distance_cm", "point_band_db"]:
        if col not in df.columns:
            df[col] = np.nan
    if "label" not in df.columns:
        df["label"] = ""

    df["method"] = method
    df["stage"] = stage
    df["label"] = normalize_label(df["label"])
    outer = df[(df["role"].astype(str) == "USED_OUTER_40") | (pd.to_numeric(df["used_for_background"], errors="coerce") == 1)].copy()
    for col in [
        "robust_fit_weight_raw", "robust_fit_weight_percent",
        "plane_influence_coefficient_signed", "plane_absolute_influence_percent",
        "x_cm", "y_cm", "distance_cm", "point_band_db",
    ]:
        outer[col] = pd.to_numeric(outer[col], errors="coerce")
    outer["weight_usage_note"] = np.where(
        outer["method"] == "plane",
        "plane中实际参与背景拟合",
        "raw中仅作空间质量参考，不参与raw分数",
    )
    return outer.reset_index(drop=True)


def discover_weights(result_root: Path, methods: Sequence[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for method in methods:
        frames.append(load_weight_file(
            result_root / method / "train_AB" / "v13_train_all_point_weights.csv",
            method,
            "AB_TRAIN",
        ))
        frames.append(load_weight_file(
            result_root / method / "test_CD" / "v13_external_test_all_point_weights.csv",
            method,
            "CD_TEST",
        ))
    return pd.concat(frames, ignore_index=True)


def make_point_key(df: pd.DataFrame) -> pd.Series:
    point_id = df["point_id"].astype(str)
    generic = point_id.str.match(r"^point_\d+$", na=False)
    coord_key = (
        "x=" + df["x_cm"].round(4).astype(str)
        + ",y=" + df["y_cm"].round(4).astype(str)
    )
    return point_id.where(~generic, coord_key)


def summarize_weights(weights: pd.DataFrame) -> pd.DataFrame:
    w = weights.copy()
    w["point_key"] = make_point_key(w)
    group_cols = ["method", "stage", "scene", "point_key"]
    summary = w.groupby(group_cols, dropna=False).agg(
        point_id=("point_id", "first"),
        x_cm=("x_cm", "median"),
        y_cm=("y_cm", "median"),
        distance_cm=("distance_cm", "median"),
        n_samples=("sample_id", "nunique"),
        robust_fit_weight_raw_median=("robust_fit_weight_raw", "median"),
        robust_fit_weight_raw_min=("robust_fit_weight_raw", "min"),
        robust_fit_weight_raw_max=("robust_fit_weight_raw", "max"),
        robust_fit_weight_percent_median=("robust_fit_weight_percent", "median"),
        robust_fit_weight_percent_min=("robust_fit_weight_percent", "min"),
        robust_fit_weight_percent_max=("robust_fit_weight_percent", "max"),
        plane_influence_signed_median=("plane_influence_coefficient_signed", "median"),
        plane_absolute_influence_percent_median=("plane_absolute_influence_percent", "median"),
        plane_absolute_influence_percent_min=("plane_absolute_influence_percent", "min"),
        plane_absolute_influence_percent_max=("plane_absolute_influence_percent", "max"),
    ).reset_index()
    summary["influence_rank_by_median"] = summary.groupby(
        ["method", "stage", "scene"]
    )["plane_absolute_influence_percent_median"].rank(method="first", ascending=False).astype(int)
    summary["robust_weight_rank_by_median"] = summary.groupby(
        ["method", "stage", "scene"]
    )["robust_fit_weight_percent_median"].rank(method="first", ascending=False).astype(int)
    return summary.sort_values(["method", "stage", "scene", "influence_rank_by_median"]).reset_index(drop=True)


def plot_weight_bars(summary_group: pd.DataFrame, value_col: str, title_suffix: str, output_path: Path) -> None:
    s = summary_group.sort_values(value_col, ascending=False).reset_index(drop=True)
    labels = [short_id(v, 18) for v in s["point_key"]]
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.bar(np.arange(len(s)), s[value_col].to_numpy(float))
    ax.set_xticks(np.arange(len(s)), labels, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("百分比 (%)")
    method = str(s["method"].iloc[0])
    stage = str(s["stage"].iloc[0])
    scene = str(s["scene"].iloc[0])
    ax.set_title(f"{method} | {scene} | {stage}\n{title_suffix}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_weight_spatial(summary_group: pd.DataFrame, value_col: str, title_suffix: str, output_path: Path) -> None:
    s = summary_group.copy()
    valid = np.isfinite(s["x_cm"]) & np.isfinite(s["y_cm"]) & np.isfinite(s[value_col])
    s = s[valid].copy()
    if s.empty:
        return
    values = np.maximum(s[value_col].to_numpy(float), 0.0)
    sizes = 60.0 + 900.0 * values / max(float(np.max(values)), EPS)
    fig, ax = plt.subplots(figsize=(9, 8))
    scatter = ax.scatter(s["x_cm"], s["y_cm"], s=sizes, c=values)
    for _, row in s.iterrows():
        ax.annotate(short_id(row["point_id"], 15), (row["x_cm"], row["y_cm"]), fontsize=6, ha="center", va="center")
    ax.scatter([0.0], [0.0], marker="x", s=130, label="中心")
    method = str(s["method"].iloc[0])
    stage = str(s["stage"].iloc[0])
    scene = str(s["scene"].iloc[0])
    ax.set_title(f"{method} | {scene} | {stage}\n{title_suffix}；圆越大表示权重越大")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.colorbar(scatter, ax=ax, label="百分比 (%)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_weight_heatmap(weight_group: pd.DataFrame, value_col: str, title_suffix: str, output_path: Path) -> None:
    w = weight_group.copy()
    w["point_key"] = make_point_key(w)
    pivot = w.pivot_table(index="sample_id", columns="point_key", values=value_col, aggfunc="median")
    if pivot.empty:
        return
    # 按点的中位权重从高到低排列，方便找最重要的点。
    pivot = pivot[pivot.median(axis=0).sort_values(ascending=False).index]
    fig_height = min(18.0, max(6.0, 0.18 * len(pivot) + 2.5))
    fig, ax = plt.subplots(figsize=(16, fig_height))
    image = ax.imshow(pivot.to_numpy(float), aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(pivot.columns)), [short_id(v, 14) for v in pivot.columns], rotation=75, ha="right", fontsize=6)
    if len(pivot) <= 60:
        ax.set_yticks(np.arange(len(pivot.index)), [short_id(v, 28) for v in pivot.index], fontsize=6)
    else:
        ax.set_yticks([])
    method = str(w["method"].iloc[0])
    stage = str(w["stage"].iloc[0])
    scene = str(w["scene"].iloc[0])
    ax.set_title(f"{method} | {scene} | {stage}\n{title_suffix}：每行一个样本，每列一个外圈点")
    ax.set_xlabel("外圈40点")
    ax.set_ylabel("样本")
    fig.colorbar(image, ax=ax, label="百分比 (%)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_wide_weight_tables(weights: pd.DataFrame, output_dir: Path) -> None:
    w = weights.copy()
    w["point_key"] = make_point_key(w)
    index_cols = ["method", "stage", "scene", "sample_id", "label"]
    for col, filename in [
        ("robust_fit_weight_percent", "05_outer40_robust_weight_percent_wide.csv"),
        ("plane_absolute_influence_percent", "06_outer40_plane_influence_percent_wide.csv"),
    ]:
        wide = w.pivot_table(index=index_cols, columns="point_key", values=col, aggfunc="median").reset_index()
        wide.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")


def write_readme(output_dir: Path, detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "V13 T>F趋势与外圈40点权重结果说明",
        "=" * 80,
        "",
        "一、T>F趋势如何判断",
        "1. 每个method、每个scene单独计算，不把A/B/C/D混在一起。",
        "2. TRUE样本：shared_leak_score > 该场景FALSE分数中位数，记为‘符合趋势’。",
        "3. FALSE样本：shared_leak_score < 该场景TRUE分数中位数，记为‘符合趋势’。",
        "4. opposite_class_pairwise_conformity_percent表示该样本与对立类别逐个比较时，符合T>F的比例。",
        "5. margin_to_opposite_median_positive_is_good：正数符合，负数不符合；绝对值越大，偏离越明显。",
        "",
        "最先查看：",
        "- 01_scene_TF_trend_summary.csv：每个场景T/F中位数、符合与不符合样本数量。",
        "- 02_sample_TF_trend_details.csv：每个样本的详细判断。",
        "- 03_nonconforming_samples.csv：只列不符合趋势的样本。",
        "",
        "二、外圈40点权重怎么看",
        "1. robust_fit_weight_raw：鲁棒拟合原始权重，越接近1越信任该点。",
        "2. robust_fit_weight_percent：该点占40点鲁棒权重总和的百分比。",
        "3. plane_absolute_influence_percent：该点对中心背景平面预测的实际影响百分比，最适合比较‘哪个点影响最大’。",
        "4. plane_influence_coefficient_signed：带方向的影响系数；正负表示拉高或拉低背景估计的方向。",
        "5. raw方法不减背景，因此权重只作空间质量参考；plane方法中权重实际参与背景估计。",
        "",
        "最先查看：",
        "- 04_outer40_weight_summary_by_point.csv：每个点跨样本的中位数、最小值、最大值和排名。",
        "- 07_outer40_weight_details_all_samples.csv：每个样本、每个外圈点的完整权重。",
        "- 06_outer40_plane_influence_percent_wide.csv：一行一个样本，一列一个外圈点。",
        "",
        "三、本次结果快速汇总",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['method']} / {row['scene']} / {row['stage']}: "
            f"T中位数={row['T_score_median']:.6g}, F中位数={row['F_score_median']:.6g}, "
            f"T-F={row['median_gap_T_minus_F']:.6g}, "
            f"符合={int(row['n_total_conforming'])}, 不符合={int(row['n_total_nonconforming'])}, "
            f"成对T>F比例={row['pairwise_probability_T_greater_than_F_percent']:.2f}%"
        )
    nonconforming = detail[detail["trend_status"] == "不符合趋势"]
    lines.extend(["", f"不符合趋势样本总数: {len(nonconforming)}"])
    (output_dir / "README_怎么看结果.txt").write_text("\n".join(lines), encoding="utf-8")


def run_analysis(result_root: Path, output_dir: Path, methods: Sequence[str]) -> Dict[str, Path]:
    ensure_dir(output_dir)
    figures_dir = output_dir / "figures"
    score_fig_dir = figures_dir / "TF_scores"
    weight_fig_dir = figures_dir / "outer40_weights"
    ensure_dir(score_fig_dir)
    ensure_dir(weight_fig_dir)

    print("读取raw/plane逐样本分数……")
    scores = discover_scores(result_root, methods)
    detail, summary = analyze_tf_trend(scores)

    summary_path = output_dir / "01_scene_TF_trend_summary.csv"
    detail_path = output_dir / "02_sample_TF_trend_details.csv"
    bad_path = output_dir / "03_nonconforming_samples.csv"
    good_path = output_dir / "03b_conforming_samples.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    detail[detail["trend_status"] == "不符合趋势"].sort_values(
        ["method", "stage", "scene", "margin_to_opposite_median_positive_is_good"]
    ).to_csv(bad_path, index=False, encoding="utf-8-sig")
    detail[detail["trend_status"] == "符合趋势"].sort_values(
        ["method", "stage", "scene", "true_label", "shared_leak_score"]
    ).to_csv(good_path, index=False, encoding="utf-8-sig")

    for (method, stage, scene), group in detail.groupby(["method", "stage", "scene"], sort=True):
        stem = f"{safe_name(method)}__{safe_name(stage)}__{safe_name(scene)}"
        plot_score_scene(group, score_fig_dir / f"01_scores__{stem}.png")
        plot_nonconforming_margin(group, score_fig_dir / f"02_nonconforming__{stem}.png")
    for method in methods:
        plot_trend_summary(summary, method, score_fig_dir / f"00_summary__{safe_name(method)}.png")

    print("读取外圈40点权重……")
    weights = discover_weights(result_root, methods)
    weight_summary = summarize_weights(weights)
    weight_summary_path = output_dir / "04_outer40_weight_summary_by_point.csv"
    weight_details_path = output_dir / "07_outer40_weight_details_all_samples.csv"
    weight_summary.to_csv(weight_summary_path, index=False, encoding="utf-8-sig")
    weights.sort_values(
        ["method", "stage", "scene", "sample_id", "plane_absolute_influence_percent"],
        ascending=[True, True, True, True, False],
    ).to_csv(weight_details_path, index=False, encoding="utf-8-sig")
    save_wide_weight_tables(weights, output_dir)

    for (method, stage, scene), s_group in weight_summary.groupby(["method", "stage", "scene"], sort=True):
        stem = f"{safe_name(method)}__{safe_name(stage)}__{safe_name(scene)}"
        plot_weight_bars(
            s_group,
            "robust_fit_weight_percent_median",
            "外圈40点鲁棒权重百分比中位数",
            weight_fig_dir / f"01_robust_weight_bar__{stem}.png",
        )
        plot_weight_bars(
            s_group,
            "plane_absolute_influence_percent_median",
            "外圈40点实际平面影响百分比中位数",
            weight_fig_dir / f"02_plane_influence_bar__{stem}.png",
        )
        plot_weight_spatial(
            s_group,
            "plane_absolute_influence_percent_median",
            "外圈40点实际平面影响百分比中位数",
            weight_fig_dir / f"03_plane_influence_spatial__{stem}.png",
        )
        w_group = weights[
            (weights["method"] == method)
            & (weights["stage"] == stage)
            & (weights["scene"] == scene)
        ]
        plot_weight_heatmap(
            w_group,
            "plane_absolute_influence_percent",
            "实际平面影响百分比",
            weight_fig_dir / f"04_plane_influence_heatmap__{stem}.png",
        )
        plot_weight_heatmap(
            w_group,
            "robust_fit_weight_percent",
            "鲁棒拟合权重百分比",
            weight_fig_dir / f"05_robust_weight_heatmap__{stem}.png",
        )

    write_readme(output_dir, detail, summary)
    print("完成。输出目录:", output_dir)
    return {
        "summary": summary_path,
        "details": detail_path,
        "nonconforming": bad_path,
        "weight_summary": weight_summary_path,
        "weight_details": weight_details_path,
        "figures": figures_dir,
    }


def synthetic_scores(method_dir: Path, method: str) -> None:
    train_dir = method_dir / "train_AB"
    test_dir = method_dir / "test_CD"
    ensure_dir(train_dir)
    ensure_dir(test_dir)
    rng = np.random.default_rng(42 if method == "raw" else 43)

    train_rows: List[Dict[str, object]] = []
    for scene in ["factory_A", "factory_B"]:
        for label, center in [(FALSE_LABEL, -0.7), (TRUE_LABEL, 0.7)]:
            vals = center + 0.45 * rng.standard_normal(6)
            if scene == "factory_B" and label == TRUE_LABEL:
                vals[0] = -0.9
            for i, score in enumerate(vals):
                train_rows.append({
                    "sample_id": f"{method}_{scene}_{label}_{i}",
                    "test_scene": scene,
                    "true_label": label,
                    "shared_leak_score_oof": score,
                })
    pd.DataFrame(train_rows).to_csv(train_dir / "v13_loso_oof_scores.csv", index=False)

    test_rows: List[Dict[str, object]] = []
    for scene in ["factory_C", "factory_D"]:
        for label, center in [(FALSE_LABEL, -0.5), (TRUE_LABEL, 0.5)]:
            vals = center + 0.55 * rng.standard_normal(6)
            if scene == "factory_C" and label == FALSE_LABEL:
                vals[-1] = 0.9
            for i, score in enumerate(vals):
                test_rows.append({
                    "sample_id": f"{method}_{scene}_{label}_{i}",
                    "scene": scene,
                    "true_label": label,
                    "shared_leak_score": score,
                    "binary_prediction": TRUE_LABEL if score >= 0 else FALSE_LABEL,
                })
    pd.DataFrame(test_rows).to_csv(test_dir / "v13_external_test_predictions.csv", index=False)

    def weight_rows(scenes: Sequence[str], output_path: Path, stage: str) -> None:
        rows: List[Dict[str, object]] = []
        angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        for scene in scenes:
            for sample_i in range(4):
                raw = np.clip(0.65 + 0.25 * rng.standard_normal(40), 0.05, 1.0)
                norm = raw / raw.sum()
                influence = np.abs(norm * (1.0 + 0.25 * rng.standard_normal(40)))
                influence = influence / influence.sum()
                for p in range(40):
                    rows.append({
                        "sample_id": f"{method}_{scene}_sample_{sample_i}",
                        "scene": scene,
                        "label": TRUE_LABEL if sample_i % 2 else FALSE_LABEL,
                        "point_index": p + 25,
                        "point_id": f"outer_{p + 1:02d}",
                        "x_cm": 80.0 * math.cos(angles[p]),
                        "y_cm": 80.0 * math.sin(angles[p]),
                        "distance_cm": 80.0,
                        "role": "USED_OUTER_40",
                        "used_for_background": 1,
                        "point_band_db": -40.0 + rng.standard_normal(),
                        "robust_fit_weight_raw": raw[p],
                        "robust_fit_weight_normalized": norm[p],
                        "robust_fit_weight_percent": 100.0 * norm[p],
                        "plane_influence_coefficient_signed": influence[p] * (1 if p % 2 else -1),
                        "plane_absolute_influence_normalized": influence[p],
                        "plane_absolute_influence_percent": 100.0 * influence[p],
                        "influence_rank_within_outer40": 0,
                    })
        pd.DataFrame(rows).to_csv(output_path, index=False)

    weight_rows(["factory_A", "factory_B"], train_dir / "v13_train_all_point_weights.csv", "AB_TRAIN")
    weight_rows(["factory_C", "factory_D"], test_dir / "v13_external_test_all_point_weights.csv", "CD_TEST")


def run_self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="v13_plot_selftest_") as tmp:
        root = Path(tmp)
        result_root = root / "results"
        output_dir = root / "plots"
        for method in VALID_METHODS:
            synthetic_scores(result_root / method, method)
        outputs = run_analysis(result_root, output_dir, VALID_METHODS)
        required = [
            outputs["summary"], outputs["details"], outputs["nonconforming"],
            outputs["weight_summary"], outputs["weight_details"],
            output_dir / "README_怎么看结果.txt",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise AssertionError(f"自检缺少输出: {missing}")
        details = pd.read_csv(outputs["details"])
        weights = pd.read_csv(outputs["weight_details"])
        if not {"符合趋势", "不符合趋势"}.issubset(set(details["trend_status"])):
            raise AssertionError("自检没有同时产生符合与不符合样本")
        counts = weights.groupby(["method", "stage", "scene", "sample_id"]).size()
        if not np.all(counts.to_numpy() == 40):
            raise AssertionError("自检中每个样本不是40个外圈点")
        print("SELF-TEST PASSED")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制V13 raw/plane的T>F趋势与外圈40点权重")
    parser.add_argument("--result-root", help="v13 raw/plane结果根目录")
    parser.add_argument("--output-dir", help="图表和CSV输出目录；默认=result-root/TF趋势与外圈40点权重图")
    parser.add_argument("--methods", nargs="+", choices=VALID_METHODS, default=list(VALID_METHODS))
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if not args.result_root:
        raise ValueError("正式运行必须提供 --result-root")
    result_root = Path(args.result_root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else result_root / "TF趋势与外圈40点权重图"
    )
    run_analysis(result_root, output_dir, args.methods)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[运行失败] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
