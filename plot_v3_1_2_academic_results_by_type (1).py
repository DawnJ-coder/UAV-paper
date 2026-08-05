#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.1.2 分WAV类型学术结果绘图程序
========================

用途
----
读取 ``spatial_local_excess_leak_separation_v3_1_2.py`` 的输出目录，
生成用于论文、答辩和实验报告的统计图、代表性时频图和结果说明。

主要回答三个问题：
1. 空间局部超额：真实泄漏样本是否比假点具有更强的中心局部增强和方向支持？
2. 实验室先验辅助约束：从原始中心到最终候选泄漏，实验室泄漏相似度是否发生合理变化？
3. 泛化与稳健性：上述效应是否能在不同工厂数据集中保持一致？

依赖
----
Python >= 3.9
numpy, pandas, matplotlib, scipy, scikit-learn

运行示例
--------
python plot_v3_1_2_academic_results.py \
    --result-dir "D:/results/v3_1_2" \
    --output-dir "D:/results/v3_1_2/academic_figures"

若不填写 --output-dir，则默认保存到：
    <result-dir>/academic_figures_by_wav_type

说明
----
- 本程序不会修改 v3.1.2 的原始结果。
- 统计图默认同时保存 PNG（300 dpi）和矢量 PDF。
- 缺少可选文件时会跳过对应图件，不会导致整个程序中断。
- 图中 AUC 和效应量属于描述性结果；最终泛化结论仍需冻结参数后在独立场景验证。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import roc_auc_score


EPS = np.finfo(np.float64).eps
DEFAULT_RESULT_DIR = r"请修改为_v3_1_2_结果目录"


# -----------------------------------------------------------------------------
# 图件定义
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricSpec:
    column: str
    short_name: str
    y_label: str
    domain: str
    expected_t_higher: bool = True


CORE_METRICS: Tuple[MetricSpec, ...] = (
    MetricSpec(
        "center_local_excess_power_fraction",
        "Local excess fraction",
        "Local-excess power / raw-center power",
        "Spatial local excess",
    ),
    MetricSpec(
        "center_excess_db_positive_median",
        "Positive center excess",
        "Median positive center excess (dB)",
        "Spatial local excess",
    ),
    MetricSpec(
        "opposite_axis_support_ratio_power_weighted",
        "Opposite-axis support",
        "Power-weighted opposite-axis support ratio",
        "Spatial local excess",
    ),
    MetricSpec(
        "background_prediction_uncertainty_db_power_weighted",
        "Background uncertainty",
        "Power-weighted background-prediction uncertainty (dB)",
        "Spatial local excess",
        expected_t_higher=False,
    ),
    MetricSpec(
        "background_prediction_confidence_power_weighted",
        "Background confidence",
        "Power-weighted background-prediction confidence",
        "Spatial local excess",
    ),
    MetricSpec(
        "spatial_local_lab_similarity_combined_median",
        "Lab similarity after spatial filtering",
        "Median similarity to held-out lab leakage",
        "Laboratory prior",
    ),
    MetricSpec(
        "center_final_leak_power_fraction",
        "Final candidate fraction",
        "Final candidate power / raw-center power",
        "Laboratory prior",
    ),
    MetricSpec(
        "leak_vs_remaining_similarity_gap",
        "Leak–remaining similarity gap",
        "Lab similarity: candidate minus remaining",
        "Laboratory prior",
    ),
    MetricSpec(
        "source_center_score",
        "Integrated source score",
        "Integrated local-source evidence score",
        "Integrated evidence",
    ),
)

AUC_METRICS: Tuple[MetricSpec, ...] = (
    MetricSpec(
        "raw_lab_similarity_combined_median",
        "Raw lab similarity",
        "AUC",
        "Baseline",
    ),
    *CORE_METRICS,
)

SIMILARITY_STAGES: Tuple[Tuple[str, str], ...] = (
    ("raw_lab_similarity_combined_median", "Raw center"),
    ("local_excess_lab_similarity_combined_median", "Positive local excess"),
    ("spatial_local_lab_similarity_combined_median", "Spatially gated excess"),
    ("leak_lab_similarity_combined_median", "Final candidate"),
    ("remaining_lab_similarity_combined_median", "Remaining signal"),
)

NPZ_HEATMAPS: Tuple[Tuple[str, str, str, Optional[Tuple[float, float]]], ...] = (
    ("raw_center_power", "Raw center power", "dB relative to sample maximum", None),
    (
        "spatial_predicted_background_power",
        "Spatially predicted background power",
        "dB relative to sample maximum",
        None,
    ),
    ("center_excess_db", "Center local excess", "Center minus background (dB)", None),
    ("axis_support_ratio", "Opposite-axis support", "Support ratio", (0.0, 1.0)),
    (
        "prediction_uncertainty_db",
        "Background-prediction uncertainty",
        "Prediction uncertainty (dB)",
        None,
    ),
    ("spatial_gate", "Spatial evidence gate", "Gate value", (0.0, 1.0)),
    ("spatial_local_power", "Spatially gated local-excess power", "dB relative to sample maximum", None),
    ("lab_gate", "Anechoic prior gate", "Gate value", (0.0, 1.0)),
    ("final_leak_power", "Final candidate leakage power", "dB relative to sample maximum", None),
)


# -----------------------------------------------------------------------------
# 基础工具
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for v3.1.2 outputs."
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="v3.1.2 输出目录，必须包含 10_factory_sample_summary.csv。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="图件输出目录；默认 <result-dir>/academic_figures。",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG 分辨率，默认 300。",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help="Bootstrap 重采样次数，默认 2000。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260804,
        help="随机种子。",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="仅保存 PNG，不保存 PDF。",
    )
    parser.add_argument(
        "--no-representative-samples",
        action="store_true",
        help="不绘制代表性 T/F 样本时频图。",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        default="dataset,canonical_condition",
        help=(
            "用于区分WAV类型的CSV列，多个列用逗号分隔。"
            "默认 dataset,canonical_condition；若没有canonical_condition会自动退回time_folder。"
        ),
    )
    parser.add_argument(
        "--include-overall",
        action="store_true",
        help="额外生成全部样本混合的总体图；默认不生成，避免不同WAV类型混在一起。",
    )
    parser.add_argument(
        "--minimum-per-label",
        type=int,
        default=1,
        help="每个类型至少需要的T和F样本数；默认每类至少1个。",
    )
    return parser.parse_args()


def resolve_result_dir(args: argparse.Namespace) -> Path:
    if args.result_dir is not None:
        return args.result_dir.expanduser().resolve()
    default = Path(DEFAULT_RESULT_DIR)
    if "请修改" not in DEFAULT_RESULT_DIR and default.exists():
        return default.expanduser().resolve()
    raise ValueError(
        "未指定结果目录。请使用：\n"
        "python plot_v3_1_2_academic_results.py --result-dir \"你的v3_1_2输出目录\""
    )


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def safe_slug(text: Any) -> str:
    out = re.sub(r"[^0-9A-Za-z._-]+", "_", str(text)).strip("_")
    return out[:180] or "sample"


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path)


def normalize_label(value: Any) -> str:
    text = str(value).strip().upper()
    if text in {"T", "TRUE", "1", "LEAK", "POSITIVE"}:
        return "T"
    if text in {"F", "FALSE", "0", "NONLEAK", "NEGATIVE"}:
        return "F"
    return text


def clean_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    out = summary.copy()
    if "label" not in out.columns:
        raise ValueError("10_factory_sample_summary.csv 缺少 label 列。")
    out["label_norm"] = out["label"].map(normalize_label)
    out = out[out["label_norm"].isin(["T", "F"])].copy()
    if out.empty:
        raise ValueError("样本总表中没有可识别的 T/F 标签。")
    if "dataset" not in out.columns:
        out["dataset"] = "all"
    for spec in AUC_METRICS:
        if spec.column in out.columns:
            out[spec.column] = pd.to_numeric(out[spec.column], errors="coerce")
    for column, _ in SIMILARITY_STAGES:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def finite_array(values: Iterable[Any]) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def percentile_interval(values: Sequence[float], alpha: float = 0.05) -> Tuple[float, float]:
    arr = finite_array(values)
    if arr.size == 0:
        return float("nan"), float("nan")
    return (
        float(np.quantile(arr, alpha / 2.0)),
        float(np.quantile(arr, 1.0 - alpha / 2.0)),
    )


def bootstrap_statistic(
    values: np.ndarray,
    statistic,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    estimate = float(statistic(values))
    if values.size == 1 or n_bootstrap <= 0:
        return estimate, estimate, estimate
    samples = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, values.size, size=values.size)
        samples[i] = float(statistic(values[idx]))
    low, high = percentile_interval(samples)
    return estimate, low, high


def cliffs_delta(t_values: np.ndarray, f_values: np.ndarray) -> float:
    """Cliff's delta，正值表示 T 整体高于 F。"""
    t = np.asarray(t_values, dtype=float)
    f = np.asarray(f_values, dtype=float)
    t = t[np.isfinite(t)]
    f = f[np.isfinite(f)]
    if t.size == 0 or f.size == 0:
        return float("nan")
    comparisons = t[:, None] - f[None, :]
    return float((np.sum(comparisons > 0) - np.sum(comparisons < 0)) / comparisons.size)


def bootstrap_cliffs_delta(
    t_values: np.ndarray,
    f_values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    t = np.asarray(t_values, dtype=float)
    f = np.asarray(f_values, dtype=float)
    t = t[np.isfinite(t)]
    f = f[np.isfinite(f)]
    estimate = cliffs_delta(t, f)
    if t.size == 0 or f.size == 0 or n_bootstrap <= 0:
        return estimate, float("nan"), float("nan")
    if t.size == 1 and f.size == 1:
        return estimate, estimate, estimate
    boot = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        tb = t[rng.integers(0, t.size, size=t.size)]
        fb = f[rng.integers(0, f.size, size=f.size)]
        boot[i] = cliffs_delta(tb, fb)
    low, high = percentile_interval(boot)
    return estimate, low, high


def stratified_auc_bootstrap(
    y: np.ndarray,
    score: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float, int, int]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(score) & np.isin(y, [0, 1])
    y = y[mask]
    score = score[mask]
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if pos.size == 0 or neg.size == 0:
        return float("nan"), float("nan"), float("nan"), int(pos.size), int(neg.size)
    estimate = float(roc_auc_score(y, score))
    if n_bootstrap <= 0:
        return estimate, float("nan"), float("nan"), int(pos.size), int(neg.size)
    boot = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        pos_b = pos[rng.integers(0, pos.size, size=pos.size)]
        neg_b = neg[rng.integers(0, neg.size, size=neg.size)]
        idx = np.concatenate([pos_b, neg_b])
        boot[i] = roc_auc_score(y[idx], score[idx])
    low, high = percentile_interval(boot)
    return estimate, low, high, int(pos.size), int(neg.size)


def save_figure(
    fig: plt.Figure,
    output_base: Path,
    dpi: int,
    save_pdf: bool,
) -> List[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    png = output_base.with_suffix(".png")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    paths.append(png)
    if save_pdf:
        pdf = output_base.with_suffix(".pdf")
        fig.savefig(pdf, bbox_inches="tight")
        paths.append(pdf)
    plt.close(fig)
    return paths


def annotate_sample_size(ax: plt.Axes, counts: Mapping[str, int]) -> None:
    text = ", ".join(f"{key}: n={value}" for key, value in counts.items())
    ax.text(
        0.99,
        0.01,
        text,
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=8.5,
    )


def relative_db(power: np.ndarray, reference: Optional[float] = None, dynamic_range_db: float = 60.0) -> np.ndarray:
    arr = np.maximum(np.asarray(power, dtype=float), EPS)
    if reference is None or not np.isfinite(reference) or reference <= 0:
        reference = float(np.nanquantile(arr, 0.995))
    db = 10.0 * np.log10(arr / max(reference, EPS))
    return np.clip(db, -abs(dynamic_range_db), 0.0)


def normalized_spectrum(power: np.ndarray) -> np.ndarray:
    arr = np.maximum(np.asarray(power, dtype=float), 0.0)
    if arr.ndim != 2:
        raise ValueError("功率矩阵必须为 [frequency, time]。")
    spec = np.nanmedian(arr, axis=1)
    total = float(np.nansum(spec))
    if total <= EPS:
        return np.zeros_like(spec)
    return spec / total


# -----------------------------------------------------------------------------
# 图 1：T/F 单指标分布
# -----------------------------------------------------------------------------


def plot_metric_distributions(
    summary: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    save_pdf: bool,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Tuple[List[Path], pd.DataFrame]:
    files: List[Path] = []
    stats_rows: List[Dict[str, Any]] = []

    for index, spec in enumerate(CORE_METRICS, start=1):
        if spec.column not in summary.columns:
            continue
        f_values = finite_array(summary.loc[summary["label_norm"] == "F", spec.column])
        t_values = finite_array(summary.loc[summary["label_norm"] == "T", spec.column])
        if f_values.size == 0 or t_values.size == 0:
            continue

        delta, delta_low, delta_high = bootstrap_cliffs_delta(
            t_values, f_values, n_bootstrap, rng
        )
        direction_sign = 1.0 if spec.expected_t_higher else -1.0
        aligned_delta = direction_sign * delta
        aligned_low = delta_low if spec.expected_t_higher else -delta_high
        aligned_high = delta_high if spec.expected_t_higher else -delta_low
        expected_direction = "T > F" if spec.expected_t_higher else "T < F"
        stats_rows.append(
            {
                "metric": spec.column,
                "metric_name": spec.short_name,
                "domain": spec.domain,
                "expected_direction": expected_direction,
                "n_T": int(t_values.size),
                "n_F": int(f_values.size),
                "T_median": float(np.median(t_values)),
                "F_median": float(np.median(f_values)),
                "T_minus_F_median": float(np.median(t_values) - np.median(f_values)),
                "cliffs_delta_T_higher": delta,
                "cliffs_delta_ci_low": delta_low,
                "cliffs_delta_ci_high": delta_high,
                "cliffs_delta_expected_direction": aligned_delta,
                "cliffs_delta_expected_ci_low": aligned_low,
                "cliffs_delta_expected_ci_high": aligned_high,
            }
        )

        fig, ax = plt.subplots(figsize=(6.4, 5.1))
        data = [f_values, t_values]
        ax.boxplot(
            data,
            showfliers=False,
            widths=0.52,
        )
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["F: non-leak location", "T: true leak location"])
        for x_position, values in enumerate(data, start=1):
            jitter = rng.normal(0.0, 0.045, size=values.size)
            ax.scatter(
                np.full(values.size, x_position, dtype=float) + jitter,
                values,
                s=22,
                alpha=0.68,
            )
        ax.set_ylabel(spec.y_label)
        ax.set_title(f"{spec.domain}: {spec.short_name}")
        ax.text(
            0.02,
            0.98,
            (
                f"Expected direction: {expected_direction}\n"
                f"Direction-aligned Cliff's δ = {aligned_delta:.3f}\n"
                f"95% bootstrap CI [{aligned_low:.3f}, {aligned_high:.3f}]"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        annotate_sample_size(ax, {"F": int(f_values.size), "T": int(t_values.size)})
        fig.tight_layout()
        base = output_dir / f"Fig01_{index:02d}_{safe_slug(spec.column)}"
        files.extend(save_figure(fig, base, dpi, save_pdf))

    return files, pd.DataFrame(stats_rows)


# -----------------------------------------------------------------------------
# 图 2：实验室先验作用链
# -----------------------------------------------------------------------------


def plot_similarity_evidence_chain(
    summary: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    save_pdf: bool,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Tuple[List[Path], pd.DataFrame]:
    available = [(column, label) for column, label in SIMILARITY_STAGES if column in summary.columns]
    if len(available) < 2:
        return [], pd.DataFrame()

    files: List[Path] = []
    rows: List[Dict[str, Any]] = []
    fig, ax = plt.subplots(figsize=(9.2, 5.7))
    x = np.arange(len(available), dtype=float)

    for label_norm, group_name in (("F", "F: non-leak location"), ("T", "T: true leak location")):
        group = summary[summary["label_norm"] == label_norm]
        means: List[float] = []
        lows: List[float] = []
        highs: List[float] = []
        for column, stage_name in available:
            values = finite_array(group[column])
            estimate, low, high = bootstrap_statistic(
                values, np.mean, n_bootstrap, rng
            )
            means.append(estimate)
            lows.append(low)
            highs.append(high)
            rows.append(
                {
                    "label": label_norm,
                    "stage_column": column,
                    "stage_name": stage_name,
                    "n": int(values.size),
                    "mean": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "median": float(np.median(values)) if values.size else float("nan"),
                }
            )
        means_arr = np.asarray(means, dtype=float)
        yerr = np.vstack(
            [means_arr - np.asarray(lows), np.asarray(highs) - means_arr]
        )
        ax.errorbar(
            x,
            means_arr,
            yerr=yerr,
            marker="o",
            linewidth=1.6,
            capsize=4,
            label=group_name,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([stage for _, stage in available], rotation=18, ha="right")
    ax.set_ylabel("Similarity to held-out laboratory leakage")
    ax.set_title("Evolution of laboratory-prior similarity through the separation chain")
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False)
    ax.text(
        0.01,
        0.02,
        "Error bars: 95% nonparametric bootstrap confidence intervals of the mean.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
    )
    fig.tight_layout()
    files.extend(save_figure(fig, output_dir / "Fig02_similarity_evidence_chain", dpi, save_pdf))
    return files, pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 图 3：AUC 与置信区间
# -----------------------------------------------------------------------------


def plot_auc_forest(
    summary: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    save_pdf: bool,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Tuple[List[Path], pd.DataFrame]:
    y = (summary["label_norm"] == "T").astype(int).to_numpy()
    rows: List[Dict[str, Any]] = []
    for spec in AUC_METRICS:
        if spec.column not in summary.columns:
            continue
        score = pd.to_numeric(summary[spec.column], errors="coerce").to_numpy(dtype=float)
        auc, low, high, n_t, n_f = stratified_auc_bootstrap(
            y, score, n_bootstrap, rng
        )
        if not np.isfinite(auc):
            continue
        expected_auc = auc if spec.expected_t_higher else 1.0 - auc
        expected_low = low if spec.expected_t_higher else 1.0 - high
        expected_high = high if spec.expected_t_higher else 1.0 - low
        rows.append(
            {
                "metric": spec.column,
                "metric_name": spec.short_name,
                "domain": spec.domain,
                "expected_direction": "T > F" if spec.expected_t_higher else "T < F",
                "auc_T_higher": auc,
                "auc_expected_direction": expected_auc,
                "ci_low": expected_low,
                "ci_high": expected_high,
                "n_T": n_t,
                "n_F": n_f,
            }
        )
    table = pd.DataFrame(rows).sort_values("auc_expected_direction", ascending=True)
    if table.empty:
        return [], table

    fig_height = max(4.8, 0.48 * len(table) + 1.8)
    fig, ax = plt.subplots(figsize=(8.2, fig_height))
    y_pos = np.arange(len(table), dtype=float)
    auc = table["auc_expected_direction"].to_numpy(dtype=float)
    low = table["ci_low"].to_numpy(dtype=float)
    high = table["ci_high"].to_numpy(dtype=float)
    ax.errorbar(
        auc,
        y_pos,
        xerr=np.vstack([auc - low, high - auc]),
        fmt="o",
        capsize=4,
        linewidth=1.4,
    )
    ax.axvline(0.5, linestyle="--", linewidth=1.1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(table["metric_name"])
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("ROC AUC in the prespecified physical direction")
    ax.set_title("Descriptive discrimination of spatial and laboratory-prior evidence")
    ax.text(
        0.01,
        0.01,
        "AUC = 0.5 indicates chance-level ranking. Background uncertainty is evaluated as T < F; other metrics as T > F.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
    )
    fig.tight_layout()
    files = save_figure(fig, output_dir / "Fig03_auc_forest", dpi, save_pdf)
    return files, table.sort_values("auc_expected_direction", ascending=False)


# -----------------------------------------------------------------------------
# 图 4：跨场景效应量矩阵
# -----------------------------------------------------------------------------


def plot_dataset_effect_matrix(
    summary: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    save_pdf: bool,
) -> Tuple[List[Path], pd.DataFrame]:
    datasets = sorted(summary["dataset"].dropna().astype(str).unique().tolist())
    metrics = [spec for spec in CORE_METRICS if spec.column in summary.columns]
    rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        sub = summary[summary["dataset"].astype(str) == dataset]
        for spec in metrics:
            t_values = finite_array(sub.loc[sub["label_norm"] == "T", spec.column])
            f_values = finite_array(sub.loc[sub["label_norm"] == "F", spec.column])
            raw_delta = cliffs_delta(t_values, f_values)
            aligned_delta = raw_delta if spec.expected_t_higher else -raw_delta
            rows.append(
                {
                    "dataset": dataset,
                    "metric": spec.column,
                    "metric_name": spec.short_name,
                    "expected_direction": "T > F" if spec.expected_t_higher else "T < F",
                    "n_T": int(t_values.size),
                    "n_F": int(f_values.size),
                    "cliffs_delta_T_higher": raw_delta,
                    "cliffs_delta_expected_direction": aligned_delta,
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return [], table
    pivot = table.pivot(index="dataset", columns="metric_name", values="cliffs_delta_expected_direction")
    if pivot.empty:
        return [], table

    fig_width = max(8.5, 1.12 * pivot.shape[1] + 2.5)
    fig_height = max(4.2, 0.62 * pivot.shape[0] + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    matrix = pivot.to_numpy(dtype=float)
    image = ax.imshow(matrix, aspect="auto", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Evidence metric")
    ax.set_ylabel("Factory dataset")
    ax.set_title("Cross-dataset effect consistency: positive values follow the prespecified physical direction")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            text = "NA" if not np.isfinite(value) else f"{value:.2f}"
            ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=8.5)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Direction-aligned Cliff's δ")
    ax.grid(False)
    fig.tight_layout()
    files = save_figure(fig, output_dir / "Fig04_cross_dataset_effect_matrix", dpi, save_pdf)
    return files, table


# -----------------------------------------------------------------------------
# 图 5：两个关键词的联合证据图
# -----------------------------------------------------------------------------


def plot_joint_evidence_map(
    summary: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    save_pdf: bool,
) -> List[Path]:
    x_col = "center_local_excess_power_fraction"
    y_col = "spatial_local_lab_similarity_combined_median"
    size_col = "opposite_axis_support_ratio_power_weighted"
    required = {x_col, y_col, size_col}
    if not required.issubset(summary.columns):
        return []

    fig, ax = plt.subplots(figsize=(7.3, 6.1))
    for label_norm, name, marker in (
        ("F", "F: non-leak location", "o"),
        ("T", "T: true leak location", "^"),
    ):
        sub = summary[summary["label_norm"] == label_norm].copy()
        x = pd.to_numeric(sub[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub[y_col], errors="coerce").to_numpy(dtype=float)
        support = pd.to_numeric(sub[size_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(support)
        if not np.any(mask):
            continue
        sizes = 24.0 + 150.0 * np.clip(support[mask], 0.0, 1.0)
        ax.scatter(
            x[mask],
            y[mask],
            s=sizes,
            marker=marker,
            alpha=0.72,
            label=name,
        )

    ax.set_xlabel("Spatial local-excess power fraction")
    ax.set_ylabel("Similarity of spatial excess to held-out laboratory leakage")
    ax.set_title("Joint evidence map: spatial local excess × laboratory prior")
    ax.legend(frameon=False)
    ax.text(
        0.01,
        0.01,
        "Marker area is proportional to opposite-axis support.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
    )
    fig.tight_layout()
    return save_figure(fig, output_dir / "Fig05_joint_spatial_lab_evidence", dpi, save_pdf)


# -----------------------------------------------------------------------------
# 图 6：成对 T-F 差异
# -----------------------------------------------------------------------------


def derive_paired_table(summary: pd.DataFrame, metric_columns: Sequence[str]) -> pd.DataFrame:
    if "pair_key" not in summary.columns:
        return pd.DataFrame()
    valid_metrics = [column for column in metric_columns if column in summary.columns]
    if not valid_metrics:
        return pd.DataFrame()
    grouped = (
        summary.groupby(["pair_key", "label_norm"], as_index=False)[valid_metrics]
        .mean(numeric_only=True)
    )
    t = grouped[grouped["label_norm"] == "T"].set_index("pair_key")
    f = grouped[grouped["label_norm"] == "F"].set_index("pair_key")
    common = sorted(set(t.index) & set(f.index))
    rows: List[Dict[str, Any]] = []
    for pair_key in common:
        row: Dict[str, Any] = {"pair_key": pair_key}
        for column in valid_metrics:
            t_value = float(t.loc[pair_key, column])
            f_value = float(f.loc[pair_key, column])
            row[f"T_{column}"] = t_value
            row[f"F_{column}"] = f_value
            row[f"T_minus_F_{column}"] = t_value - f_value
        rows.append(row)
    return pd.DataFrame(rows)


def plot_paired_differences(
    summary: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    save_pdf: bool,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Tuple[List[Path], pd.DataFrame]:
    selected = [
        spec
        for spec in CORE_METRICS
        if spec.column
        in {
            "center_local_excess_power_fraction",
            "opposite_axis_support_ratio_power_weighted",
            "spatial_local_lab_similarity_combined_median",
            "source_center_score",
        }
    ]
    paired = derive_paired_table(summary, [spec.column for spec in selected])
    if paired.empty:
        return [], paired

    files: List[Path] = []
    stat_rows: List[Dict[str, Any]] = []
    for index, spec in enumerate(selected, start=1):
        diff_col = f"T_minus_F_{spec.column}"
        if diff_col not in paired.columns:
            continue
        values = finite_array(paired[diff_col])
        if values.size == 0:
            continue
        mean, low, high = bootstrap_statistic(values, np.mean, n_bootstrap, rng)
        positive_fraction = float(np.mean(values > 0))
        stat_rows.append(
            {
                "metric": spec.column,
                "metric_name": spec.short_name,
                "n_pairs": int(values.size),
                "mean_T_minus_F": mean,
                "ci_low": low,
                "ci_high": high,
                "median_T_minus_F": float(np.median(values)),
                "T_greater_F_fraction": positive_fraction,
            }
        )

        fig, ax = plt.subplots(figsize=(6.4, 5.0))
        jitter = rng.normal(1.0, 0.045, size=values.size)
        ax.scatter(jitter, values, s=28, alpha=0.72)
        ax.boxplot([values], positions=[1.0], widths=0.22, showfliers=False)
        ax.axhline(0.0, linestyle="--", linewidth=1.1)
        ax.errorbar(
            [1.26],
            [mean],
            yerr=np.asarray([[mean - low], [high - mean]]),
            fmt="o",
            capsize=5,
            label="Mean and 95% bootstrap CI",
        )
        ax.set_xlim(0.65, 1.5)
        ax.set_xticks([1.0])
        ax.set_xticklabels(["Matched T − F pairs"])
        ax.set_ylabel(f"Difference in {spec.y_label}")
        ax.set_title(f"Paired contrast: {spec.short_name}")
        ax.legend(frameon=False, loc="best")
        ax.text(
            0.02,
            0.98,
            f"Pairs with T > F: {positive_fraction:.1%}\nn = {values.size}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        fig.tight_layout()
        files.extend(
            save_figure(
                fig,
                output_dir / f"Fig06_{index:02d}_paired_{safe_slug(spec.column)}",
                dpi,
                save_pdf,
            )
        )
    return files, pd.DataFrame(stat_rows)


# -----------------------------------------------------------------------------
# 图 7：相反方向点随距离的空间证据
# -----------------------------------------------------------------------------


def plot_opposite_pair_distance(
    pair_df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    save_pdf: bool,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Tuple[List[Path], pd.DataFrame]:
    required = {
        "label",
        "distance_cm",
        "center_minus_pair_prediction_db_median",
    }
    if pair_df.empty or not required.issubset(pair_df.columns):
        return [], pd.DataFrame()
    data = pair_df.copy()
    data["label_norm"] = data["label"].map(normalize_label)
    data["distance_cm"] = pd.to_numeric(data["distance_cm"], errors="coerce")
    value_col = "center_minus_pair_prediction_db_median"
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data[
        data["label_norm"].isin(["T", "F"])
        & np.isfinite(data["distance_cm"])
        & np.isfinite(data[value_col])
    ].copy()
    if data.empty:
        return [], pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    for label_norm, group_name in (("F", "F: non-leak location"), ("T", "T: true leak location")):
        group = data[data["label_norm"] == label_norm]
        x_values: List[float] = []
        means: List[float] = []
        lows: List[float] = []
        highs: List[float] = []
        for distance, sub in group.groupby("distance_cm"):
            values = finite_array(sub[value_col])
            estimate, low, high = bootstrap_statistic(values, np.mean, n_bootstrap, rng)
            x_values.append(float(distance))
            means.append(estimate)
            lows.append(low)
            highs.append(high)
            rows.append(
                {
                    "label": label_norm,
                    "distance_cm": float(distance),
                    "n": int(values.size),
                    "mean_center_minus_pair_db": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "median_center_minus_pair_db": float(np.median(values)),
                }
            )
        if not x_values:
            continue
        order = np.argsort(x_values)
        x_arr = np.asarray(x_values)[order]
        mean_arr = np.asarray(means)[order]
        low_arr = np.asarray(lows)[order]
        high_arr = np.asarray(highs)[order]
        ax.errorbar(
            x_arr,
            mean_arr,
            yerr=np.vstack([mean_arr - low_arr, high_arr - mean_arr]),
            marker="o",
            capsize=4,
            linewidth=1.5,
            label=group_name,
        )
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Radius of opposite-direction pair (cm)")
    ax.set_ylabel("Center minus opposite-pair prediction (dB)")
    ax.set_title("Spatial localization evidence across measurement radii")
    ax.legend(frameon=False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    files = save_figure(fig, output_dir / "Fig07_opposite_pair_distance", dpi, save_pdf)
    return files, pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 代表性 T/F 样本定位与时频图
# -----------------------------------------------------------------------------


def index_sample_outputs(result_dir: Path) -> Dict[Tuple[str, str, str], Dict[str, Path]]:
    index: Dict[Tuple[str, str, str], Dict[str, Path]] = {}
    sample_root = result_dir / "samples"
    if not sample_root.is_dir():
        return index
    for summary_path in sample_root.rglob("summary.json"):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            key = (
                str(payload.get("dataset", "")),
                str(payload.get("time_folder", "")),
                str(payload.get("center_id", "")),
            )
            index[key] = {
                "summary": summary_path,
                "npz": summary_path.parent / "spatial_local_excess_result.npz",
                "directory": summary_path.parent,
            }
        except Exception:
            continue
    return index


def locate_npz_for_row(
    row: pd.Series,
    result_dir: Path,
    sample_index: Mapping[Tuple[str, str, str], Dict[str, Path]],
) -> Optional[Path]:
    key = (
        str(row.get("dataset", "")),
        str(row.get("time_folder", "")),
        str(row.get("center_id", "")),
    )
    candidate = sample_index.get(key, {}).get("npz")
    if candidate is not None and candidate.is_file():
        return candidate

    constructed = (
        result_dir
        / "samples"
        / safe_slug(row.get("dataset", ""))
        / safe_slug(row.get("time_folder", "root") or "root")
        / f"center_{safe_slug(row.get('center_id', ''))}"
        / "spatial_local_excess_result.npz"
    )
    if constructed.is_file():
        return constructed
    return None


def choose_representative_rows(
    summary: pd.DataFrame,
    result_dir: Path,
    sample_index: Mapping[Tuple[str, str, str], Dict[str, Path]],
) -> List[Tuple[str, pd.Series, Path]]:
    chosen: List[Tuple[str, pd.Series, Path]] = []
    score_col = "source_center_score"
    for label_norm in ("T", "F"):
        sub = summary[summary["label_norm"] == label_norm].copy()
        if sub.empty:
            continue
        candidates: List[Tuple[pd.Series, Path]] = []
        for _, row in sub.iterrows():
            npz_path = locate_npz_for_row(row, result_dir, sample_index)
            if npz_path is not None:
                candidates.append((row, npz_path))
        if not candidates:
            continue
        scores = np.asarray(
            [pd.to_numeric(pd.Series([item[0].get(score_col)]), errors="coerce").iloc[0] for item in candidates],
            dtype=float,
        )
        finite = np.isfinite(scores)
        if np.any(finite):
            target = float(np.median(scores[finite]))
            distances = np.where(finite, np.abs(scores - target), np.inf)
            selected = int(np.argmin(distances))
        else:
            selected = 0
        row, npz_path = candidates[selected]
        chosen.append((label_norm, row, npz_path))
    return chosen


def validate_npz_axes(data: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if "freq_hz" not in data or "time_s" not in data:
        raise KeyError("NPZ 缺少 freq_hz 或 time_s。")
    freq = np.asarray(data["freq_hz"], dtype=float)
    time_s = np.asarray(data["time_s"], dtype=float)
    if freq.ndim != 1 or time_s.ndim != 1:
        raise ValueError("freq_hz 和 time_s 必须是一维数组。")
    return freq, time_s


def plot_single_tf_heatmap(
    matrix: np.ndarray,
    freq_hz: np.ndarray,
    time_s: np.ndarray,
    title: str,
    colorbar_label: str,
    output_base: Path,
    dpi: int,
    save_pdf: bool,
    fixed_range: Optional[Tuple[float, float]] = None,
    convert_power_to_db: bool = False,
    reference_power: Optional[float] = None,
) -> List[Path]:
    values = np.asarray(matrix, dtype=float)
    if values.shape != (freq_hz.size, time_s.size):
        raise ValueError(
            f"矩阵维度 {values.shape} 与频率/时间轴 {(freq_hz.size, time_s.size)} 不一致。"
        )
    if convert_power_to_db:
        values = relative_db(values, reference=reference_power)
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    kwargs: Dict[str, Any] = {"shading": "auto"}
    if fixed_range is not None:
        kwargs["vmin"], kwargs["vmax"] = fixed_range
    image = ax.pcolormesh(time_s, freq_hz / 1000.0, values, **kwargs)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(title)
    ax.grid(False)
    fig.tight_layout()
    return save_figure(fig, output_base, dpi, save_pdf)


def plot_representative_samples(
    summary: pd.DataFrame,
    result_dir: Path,
    output_dir: Path,
    dpi: int,
    save_pdf: bool,
) -> Tuple[List[Path], pd.DataFrame]:
    sample_index = index_sample_outputs(result_dir)
    chosen = choose_representative_rows(summary, result_dir, sample_index)
    if not chosen:
        return [], pd.DataFrame()

    files: List[Path] = []
    rows: List[Dict[str, Any]] = []
    for label_norm, row, npz_path in chosen:
        try:
            with np.load(npz_path, allow_pickle=False) as loaded:
                data = {key: loaded[key] for key in loaded.files}
            freq_hz, time_s = validate_npz_axes(data)
            raw_reference = None
            if "raw_center_power" in data:
                raw_reference = float(
                    np.nanquantile(np.maximum(data["raw_center_power"], EPS), 0.995)
                )
            sample_text = (
                f"{row.get('dataset', '')} | {row.get('time_folder', '')} | "
                f"center {row.get('center_id', '')}"
            )
            sample_slug = safe_slug(
                f"{label_norm}_{row.get('dataset', '')}_{row.get('time_folder', '')}_{row.get('center_id', '')}"
            )
            rows.append(
                {
                    "label": label_norm,
                    "dataset": row.get("dataset", ""),
                    "time_folder": row.get("time_folder", ""),
                    "center_id": row.get("center_id", ""),
                    "source_center_score": row.get("source_center_score", np.nan),
                    "npz_path": str(npz_path),
                }
            )

            for key, title_name, colorbar_label, fixed_range in NPZ_HEATMAPS:
                if key not in data:
                    continue
                convert_power = key.endswith("_power")
                title = f"Representative {label_norm} sample: {title_name}\n{sample_text}"
                output_base = output_dir / f"Fig08_{sample_slug}_{safe_slug(key)}"
                files.extend(
                    plot_single_tf_heatmap(
                        data[key],
                        freq_hz,
                        time_s,
                        title,
                        colorbar_label,
                        output_base,
                        dpi,
                        save_pdf,
                        fixed_range=fixed_range,
                        convert_power_to_db=convert_power,
                        reference_power=raw_reference if convert_power else None,
                    )
                )

            spectrum_keys = [
                ("raw_center_power", "Raw center"),
                ("spatial_predicted_background_power", "Predicted background"),
                ("local_excess_power", "Positive local excess"),
                ("spatial_local_power", "Spatially gated excess"),
                ("final_leak_power", "Final candidate"),
                ("remaining_power", "Remaining signal"),
            ]
            available_spectra = [(key, name) for key, name in spectrum_keys if key in data]
            if available_spectra:
                fig, ax = plt.subplots(figsize=(9.2, 5.6))
                for key, name in available_spectra:
                    ax.plot(freq_hz / 1000.0, normalized_spectrum(data[key]), label=name, linewidth=1.35)
                if "leak_dictionary" in data:
                    dictionary = np.maximum(np.asarray(data["leak_dictionary"], dtype=float), 0.0)
                    if dictionary.ndim == 2 and dictionary.shape[0] == freq_hz.size:
                        prototype = np.median(
                            dictionary / np.maximum(np.sum(dictionary, axis=0, keepdims=True), EPS),
                            axis=1,
                        )
                        prototype = prototype / max(float(np.sum(prototype)), EPS)
                        ax.plot(
                            freq_hz / 1000.0,
                            prototype,
                            linestyle="--",
                            linewidth=1.5,
                            label="Median laboratory dictionary profile",
                        )
                ax.set_xlabel("Frequency (kHz)")
                ax.set_ylabel("Normalized median power spectrum")
                ax.set_title(f"Representative {label_norm} sample: spectral separation chain\n{sample_text}")
                ax.legend(frameon=False)
                fig.tight_layout()
                files.extend(
                    save_figure(
                        fig,
                        output_dir / f"Fig09_{sample_slug}_spectral_chain",
                        dpi,
                        save_pdf,
                    )
                )
        except Exception as exc:
            rows.append(
                {
                    "label": label_norm,
                    "dataset": row.get("dataset", ""),
                    "time_folder": row.get("time_folder", ""),
                    "center_id": row.get("center_id", ""),
                    "source_center_score": row.get("source_center_score", np.nan),
                    "npz_path": str(npz_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return files, pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 结果说明与索引
# -----------------------------------------------------------------------------


def evidence_direction_text(delta: float) -> str:
    if not np.isfinite(delta):
        return "无法计算"
    magnitude = abs(delta)
    if magnitude < 0.147:
        level = "可忽略"
    elif magnitude < 0.33:
        level = "较小"
    elif magnitude < 0.474:
        level = "中等"
    else:
        level = "较大"
    direction = "T整体高于F" if delta > 0 else "T整体低于F" if delta < 0 else "无方向差异"
    return f"{direction}，效应量{level}"


def write_academic_readme(
    output_dir: Path,
    result_dir: Path,
    summary: pd.DataFrame,
    distribution_stats: pd.DataFrame,
    auc_stats: pd.DataFrame,
    effect_stats: pd.DataFrame,
    paired_stats: pd.DataFrame,
    generated_files: Sequence[Path],
    skipped: Sequence[str],
) -> Path:
    n_total = len(summary)
    n_t = int(np.sum(summary["label_norm"] == "T"))
    n_f = int(np.sum(summary["label_norm"] == "F"))
    datasets = sorted(summary["dataset"].dropna().astype(str).unique().tolist())

    lines: List[str] = [
        "v3.1.2 学术结果图：阅读说明",
        "=" * 72,
        "",
        f"原始结果目录：{result_dir}",
        f"有效样本：总计 {n_total}，T={n_t}，F={n_f}",
        f"数据集：{', '.join(datasets) if datasets else '未提供'}",
        "",
        "一、图件回答的核心问题",
        "1. 空间局部超额：中心是否显著高于相反方向、同半径点预测的背景？",
        "2. 实验室先验辅助约束：空间局部成分是否更接近独立实验室泄漏参考？",
        "3. 跨场景一致性：指标是否在不同工厂数据集中保持预设物理方向？",
        "",
        "二、主要图件",
        "- Fig01_*：每个核心指标的 T/F 分布、Cliff's delta 和 95% bootstrap CI。",
        "- Fig02_similarity_evidence_chain：原始中心→局部超额→空间门控→最终候选→剩余信号的实验室相似度变化。",
        "- Fig03_auc_forest：按预设物理方向计算描述性 ROC AUC；背景不确定度采用 T<F，其余采用 T>F。",
        "- Fig04_cross_dataset_effect_matrix：跨场景方向对齐后的 Cliff's delta；正值表示符合预设物理方向。",
        "- Fig05_joint_spatial_lab_evidence：横轴为空间局部超额，纵轴为实验室相似度，点面积表示方向支持。",
        "- Fig06_*：同工况、同中心编号的 T-F 成对差值。",
        "- Fig07_opposite_pair_distance：不同半径下中心相对相反方向背景预测的 dB 超额。",
        "- Fig08_*：代表性 T/F 样本的时频证据矩阵。",
        "- Fig09_*：代表性 T/F 样本的频谱分离链。",
        "",
        "三、自动统计摘要",
    ]

    if not distribution_stats.empty:
        for _, row in distribution_stats.iterrows():
            raw_delta = float(row["cliffs_delta_T_higher"])
            aligned_delta = float(row["cliffs_delta_expected_direction"])
            lines.append(
                f"- {row['metric_name']}（预设 {row['expected_direction']}）: "
                f"方向对齐 δ={aligned_delta:.3f}, "
                f"95% CI [{row['cliffs_delta_expected_ci_low']:.3f}, {row['cliffs_delta_expected_ci_high']:.3f}]；"
                f"原始 T-F 排序 δ={raw_delta:.3f}。"
            )
    else:
        lines.append("- 没有生成可用的 T/F 分布效应统计。")

    if not auc_stats.empty:
        best = auc_stats.iloc[0]
        lines.extend(
            [
                "",
                (
                    f"描述性 AUC 最高指标：{best['metric_name']}，"
                    f"方向对齐 AUC={best['auc_expected_direction']:.3f}，"
                    f"95% CI [{best['ci_low']:.3f}, {best['ci_high']:.3f}]。"
                ),
            ]
        )

    if not effect_stats.empty:
        effect_valid = effect_stats.dropna(subset=["cliffs_delta_expected_direction"])
        if not effect_valid.empty:
            consistent = (
                effect_valid.groupby("metric_name")["cliffs_delta_expected_direction"]
                .apply(lambda x: float(np.mean(x > 0)))
                .sort_values(ascending=False)
            )
            lines.append("")
            lines.append("跨数据集符合预设物理方向的比例：")
            for metric_name, fraction in consistent.items():
                lines.append(f"- {metric_name}: {fraction:.1%}")

    if not paired_stats.empty:
        lines.append("")
        lines.append("成对比较中 T>F 的比例：")
        for _, row in paired_stats.iterrows():
            lines.append(
                f"- {row['metric_name']}: {row['T_greater_F_fraction']:.1%} "
                f"(n={int(row['n_pairs'])})"
            )

    lines.extend(
        [
            "",
            "四、学术解释边界",
            "- estimated_local_leak 是满足空间局部性且受实验室先验支持的候选分量，不是纯泄漏真值。",
            "- AUC、效应量和当前 T/F 对比用于描述本批数据；不能替代冻结阈值后的独立场景测试。",
            "- 若 Fig04 中不同数据集效应方向相反，应优先检查场景迁移、背景预测稳定性和数据幅值归一化问题。",
            "- 若 Fig02 中 F 样本的最终候选相似度也明显升高，说明实验室先验仍可能提取到泄漏样机械噪声。",
            "- 若 Fig05 中 T/F 大量重叠，说明仅靠‘空间局部超额 × 实验室相似度’仍不足以形成稳定分类边界。",
            "",
            "五、生成状态",
            f"已生成文件数：{len(generated_files)}",
        ]
    )
    if skipped:
        lines.append("跳过项目：")
        lines.extend(f"- {item}" for item in skipped)

    readme_path = output_dir / "00_READ_ME_ACADEMIC_FIGURES.txt"
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    return readme_path


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def build_file_index(generated_files: Sequence[Path], output_dir: Path) -> Path:
    rows: List[Dict[str, str]] = []
    for path in sorted(set(generated_files)):
        if path == output_dir / "00_READ_ME_ACADEMIC_FIGURES.txt":
            purpose = "图件阅读说明与自动统计摘要"
        elif "Fig01" in path.name:
            purpose = "T/F 单指标分布及效应量"
        elif "Fig02" in path.name:
            purpose = "实验室先验作用链"
        elif "Fig03" in path.name:
            purpose = "AUC 及 bootstrap 置信区间"
        elif "Fig04" in path.name:
            purpose = "跨数据集效应一致性"
        elif "Fig05" in path.name:
            purpose = "空间局部超额与实验室先验联合证据"
        elif "Fig06" in path.name:
            purpose = "T/F 成对差异"
        elif "Fig07" in path.name:
            purpose = "相反方向点随半径的空间证据"
        elif "Fig08" in path.name:
            purpose = "代表性样本时频证据"
        elif "Fig09" in path.name:
            purpose = "代表性样本频谱分离链"
        elif path.suffix.lower() == ".csv":
            purpose = "绘图所用统计结果表"
        else:
            purpose = "结果文件"
        rows.append(
            {
                "file": path.name,
                "relative_path": str(path.relative_to(output_dir)),
                "purpose": purpose,
            }
        )
    index_path = output_dir / "00_figure_file_index.csv"
    pd.DataFrame(rows).to_csv(index_path, index=False, encoding="utf-8-sig")
    return index_path


# -----------------------------------------------------------------------------
# 分类型结果生成
# -----------------------------------------------------------------------------


def ensure_grouping_columns(summary: pd.DataFrame, requested: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
    """检查分组列；canonical_condition缺失时自动使用time_folder。"""
    out = summary.copy()
    columns: List[str] = []
    for raw in requested:
        column = str(raw).strip()
        if not column:
            continue
        if column == "canonical_condition" and column not in out.columns:
            if "time_folder" in out.columns:
                out["canonical_condition"] = out["time_folder"].astype(str)
            else:
                out["canonical_condition"] = "unknown_condition"
        if column not in out.columns:
            raise ValueError(
                f"分组列不存在：{column}。可用列包括：{', '.join(map(str, out.columns))}"
            )
        out[column] = out[column].fillna("NA").astype(str)
        columns.append(column)
    if not columns:
        raise ValueError("--group-by 至少需要一个有效列名。")
    return out, columns


def make_group_label(group_columns: Sequence[str], group_values: Sequence[Any]) -> str:
    return " | ".join(f"{column}={value}" for column, value in zip(group_columns, group_values))


def make_group_relative_dir(group_columns: Sequence[str], group_values: Sequence[Any]) -> Path:
    path = Path("01_by_wav_type")
    for column, value in zip(group_columns, group_values):
        path = path / f"{safe_slug(column)}_{safe_slug(value)}"
    return path


def subset_pair_diagnostics(
    pair_df: pd.DataFrame,
    group_columns: Sequence[str],
    group_values: Sequence[Any],
) -> pd.DataFrame:
    if pair_df.empty:
        return pair_df
    out = pair_df.copy()
    for column, value in zip(group_columns, group_values):
        if column not in out.columns:
            # pair诊断表没有该列时，不能可靠筛选；返回空表，避免混入其他类型。
            return pd.DataFrame()
        out = out[out[column].fillna("NA").astype(str) == str(value)]
    return out.copy()


def prepend_group_context(readme_path: Path, group_label: str, group_columns: Sequence[str]) -> None:
    original = readme_path.read_text(encoding="utf-8") if readme_path.is_file() else ""
    prefix = (
        "当前独立分析分组\n"
        + "=" * 72
        + f"\n分组：{group_label}\n"
        + f"分组字段：{', '.join(group_columns)}\n"
        + "本目录中的统计图只使用该分组内的样本，不与其他WAV类型混合。\n\n"
    )
    readme_path.write_text(prefix + original, encoding="utf-8")


def generate_subset_package(
    summary: pd.DataFrame,
    pair_df: pd.DataFrame,
    result_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    group_label: str,
    group_columns: Sequence[str],
    include_cross_dataset_effect: bool,
) -> Tuple[List[Path], List[str]]:
    """为一个独立WAV类型生成一整套图件和统计表。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_pdf = not args.no_pdf
    # 每个分组使用稳定但不同的随机种子，保证重复运行可复现。
    seed_offset = int(hashlib.sha256(group_label.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng((int(args.seed) + seed_offset) % (2**32 - 1))

    generated: List[Path] = []
    skipped: List[str] = []

    group_rows_path = output_dir / "00_group_sample_rows.csv"
    write_table(summary.drop(columns=["label_norm"], errors="ignore"), group_rows_path)
    generated.append(group_rows_path)

    distribution_files, distribution_stats = plot_metric_distributions(
        summary, output_dir, args.dpi, save_pdf, args.bootstrap, rng
    )
    generated.extend(distribution_files)
    if distribution_stats.empty:
        skipped.append("Fig01：该类型缺少同时可用的T/F核心指标。")
    else:
        path = output_dir / "Table01_metric_distribution_statistics.csv"
        write_table(distribution_stats, path)
        generated.append(path)

    similarity_files, similarity_stats = plot_similarity_evidence_chain(
        summary, output_dir, args.dpi, save_pdf, args.bootstrap, rng
    )
    generated.extend(similarity_files)
    if similarity_stats.empty:
        skipped.append("Fig02：实验室相似度阶段列不足。")
    else:
        path = output_dir / "Table02_similarity_chain_statistics.csv"
        write_table(similarity_stats, path)
        generated.append(path)

    auc_files, auc_stats = plot_auc_forest(
        summary, output_dir, args.dpi, save_pdf, args.bootstrap, rng
    )
    generated.extend(auc_files)
    if auc_stats.empty:
        skipped.append("Fig03：该类型无法计算T/F AUC。")
    else:
        path = output_dir / "Table03_auc_bootstrap_statistics.csv"
        write_table(auc_stats, path)
        generated.append(path)

    effect_stats = pd.DataFrame()
    if include_cross_dataset_effect and summary["dataset"].astype(str).nunique() >= 2:
        effect_files, effect_stats = plot_dataset_effect_matrix(
            summary, output_dir, args.dpi, save_pdf
        )
        generated.extend(effect_files)
        if effect_stats.empty:
            skipped.append("Fig04：无法形成跨数据集效应矩阵。")
        else:
            path = output_dir / "Table04_cross_dataset_effect_sizes.csv"
            write_table(effect_stats, path)
            generated.append(path)
    else:
        skipped.append("Fig04：单一WAV类型目录不执行跨数据集混合矩阵。")

    joint_files = plot_joint_evidence_map(summary, output_dir, args.dpi, save_pdf)
    generated.extend(joint_files)
    if not joint_files:
        skipped.append("Fig05：缺少空间超额、方向支持或实验室相似度列。")

    paired_files, paired_stats = plot_paired_differences(
        summary, output_dir, args.dpi, save_pdf, args.bootstrap, rng
    )
    generated.extend(paired_files)
    if paired_stats.empty:
        skipped.append("Fig06：该类型内没有可匹配的T/F pair_key。")
    else:
        path = output_dir / "Table06_paired_difference_statistics.csv"
        write_table(paired_stats, path)
        generated.append(path)

    distance_files, distance_stats = plot_opposite_pair_distance(
        pair_df, output_dir, args.dpi, save_pdf, args.bootstrap, rng
    )
    generated.extend(distance_files)
    if distance_stats.empty:
        skipped.append("Fig07：该类型没有独立的相反方向诊断数据，或字段不足。")
    else:
        path = output_dir / "Table07_opposite_pair_distance_statistics.csv"
        write_table(distance_stats, path)
        generated.append(path)

    if args.no_representative_samples:
        skipped.append("Fig08–Fig09：用户通过参数关闭代表性样本图。")
    else:
        representative_files, representative_stats = plot_representative_samples(
            summary, result_dir, output_dir, args.dpi, save_pdf
        )
        generated.extend(representative_files)
        if representative_stats.empty:
            skipped.append("Fig08–Fig09：该类型未找到可读取的样本NPZ文件。")
        else:
            path = output_dir / "Table08_representative_samples.csv"
            write_table(representative_stats, path)
            generated.append(path)

    readme = write_academic_readme(
        output_dir,
        result_dir,
        summary,
        distribution_stats,
        auc_stats,
        effect_stats,
        paired_stats,
        generated,
        skipped,
    )
    prepend_group_context(readme, group_label, group_columns)
    generated.append(readme)
    index_path = build_file_index(generated, output_dir)
    generated.append(index_path)
    return generated, skipped


# -----------------------------------------------------------------------------
# 主程序
# -----------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    setup_matplotlib()
    result_dir = resolve_result_dir(args)
    if not result_dir.is_dir():
        raise FileNotFoundError(f"结果目录不存在：{result_dir}")

    summary_path = result_dir / "10_factory_sample_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"缺少必要文件：{summary_path}\n"
            "请确认 --result-dir 指向 v3.1.2 的最外层输出目录。"
        )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else result_dir / "academic_figures_by_wav_type"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = clean_summary(read_csv_optional(summary_path))
    requested_columns = [item.strip() for item in args.group_by.split(",") if item.strip()]
    summary, group_columns = ensure_grouping_columns(summary, requested_columns)
    pair_df = read_csv_optional(result_dir / "11_opposite_pair_diagnostics.csv")

    manifest_rows: List[Dict[str, Any]] = []
    all_generated: List[Path] = []

    if args.include_overall:
        overall_dir = output_dir / "00_overall_mixed_optional"
        generated, skipped = generate_subset_package(
            summary=summary,
            pair_df=pair_df,
            result_dir=result_dir,
            output_dir=overall_dir,
            args=args,
            group_label="OVERALL_MIXED_ALL_TYPES",
            group_columns=["none"],
            include_cross_dataset_effect=True,
        )
        all_generated.extend(generated)
        manifest_rows.append(
            {
                "group_label": "OVERALL_MIXED_ALL_TYPES",
                "group_columns": "none",
                "n_total": len(summary),
                "n_T": int(np.sum(summary["label_norm"] == "T")),
                "n_F": int(np.sum(summary["label_norm"] == "F")),
                "status": "generated",
                "output_directory": str(overall_dir.relative_to(output_dir)),
                "notes": "；".join(skipped),
            }
        )

    groupby_key: Any = group_columns[0] if len(group_columns) == 1 else group_columns
    grouped = summary.groupby(groupby_key, dropna=False, sort=True)
    generated_group_count = 0
    skipped_group_count = 0

    for raw_values, sub in grouped:
        values = (raw_values,) if len(group_columns) == 1 else tuple(raw_values)
        label = make_group_label(group_columns, values)
        group_dir = output_dir / make_group_relative_dir(group_columns, values)
        n_t = int(np.sum(sub["label_norm"] == "T"))
        n_f = int(np.sum(sub["label_norm"] == "F"))
        pair_sub = subset_pair_diagnostics(pair_df, group_columns, values)

        if n_t < args.minimum_per_label or n_f < args.minimum_per_label:
            group_dir.mkdir(parents=True, exist_ok=True)
            reason = (
                f"跳过统计绘图：该WAV类型样本不足。T={n_t}, F={n_f}, "
                f"要求每类至少{args.minimum_per_label}个。"
            )
            (group_dir / "00_GROUP_SKIPPED.txt").write_text(
                f"分组：{label}\n{reason}\n",
                encoding="utf-8",
            )
            write_table(sub.drop(columns=["label_norm"], errors="ignore"), group_dir / "00_group_sample_rows.csv")
            manifest_rows.append(
                {
                    "group_label": label,
                    "group_columns": ",".join(group_columns),
                    "n_total": len(sub),
                    "n_T": n_t,
                    "n_F": n_f,
                    "status": "skipped_insufficient_TF",
                    "output_directory": str(group_dir.relative_to(output_dir)),
                    "notes": reason,
                }
            )
            skipped_group_count += 1
            continue

        generated, skipped = generate_subset_package(
            summary=sub.copy(),
            pair_df=pair_sub,
            result_dir=result_dir,
            output_dir=group_dir,
            args=args,
            group_label=label,
            group_columns=group_columns,
            include_cross_dataset_effect=False,
        )
        all_generated.extend(generated)
        manifest_rows.append(
            {
                "group_label": label,
                "group_columns": ",".join(group_columns),
                "n_total": len(sub),
                "n_T": n_t,
                "n_F": n_f,
                "status": "generated",
                "output_directory": str(group_dir.relative_to(output_dir)),
                "notes": "；".join(skipped),
            }
        )
        generated_group_count += 1

    manifest_path = output_dir / "00_WAV_TYPE_GROUP_MANIFEST.csv"
    write_table(pd.DataFrame(manifest_rows), manifest_path)

    top_readme = output_dir / "00_READ_ME_FIRST.txt"
    top_readme.write_text(
        "\n".join(
            [
                "v3.1.2 分WAV类型学术绘图结果",
                "=" * 72,
                f"原始结果目录：{result_dir}",
                f"分组字段：{', '.join(group_columns)}",
                f"成功生成类型数：{generated_group_count}",
                f"因T/F不足跳过类型数：{skipped_group_count}",
                "",
                "目录规则：",
                "- 01_by_wav_type/<分组字段_取值>/：每一种WAV类型的独立图件和统计表。",
                "- 00_WAV_TYPE_GROUP_MANIFEST.csv：所有类型、T/F数量、输出目录和跳过原因。",
                "- 默认不生成所有类型混合图；只有添加 --include-overall 才会生成。",
                "",
                "默认WAV类型定义：dataset + canonical_condition。",
                "如需只按工厂分组：--group-by dataset",
                "如需只按工况分组：--group-by canonical_condition",
            ]
        ),
        encoding="utf-8",
    )

    print("=" * 88)
    print("v3.1.2 分WAV类型学术结果图生成完成")
    print(f"结果目录：{result_dir}")
    print(f"输出目录：{output_dir}")
    print(f"分组字段：{', '.join(group_columns)}")
    print(f"成功生成类型：{generated_group_count}")
    print(f"T/F不足跳过：{skipped_group_count}")
    print(f"分组清单：{manifest_path}")
    print(f"先阅读：{top_readme}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print("\n程序运行失败：", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("\n详细堆栈：", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
