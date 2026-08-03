# -*- coding: utf-8 -*-
"""
GC波束结果：泄漏声源声学足迹（形态）绘制。

本程序回答的是：
    “波束形成后的声源热点，在空间中有多大、是否以候选中心为峰值、
    从中心向外怎样衰减？”

它不再尝试从同一张泄漏声场中拟合一个二维“背景曲面”。原因是：
真实泄漏的波束声斑本身可能是宽而平滑的，RBF/二次曲面会把泄漏声斑
当成背景重建并减掉。

正确处理：
1. 计算中心点和5~80 cm、8方向共129个点的50~70 kHz原始能量；
2. 用“相对实测峰值dB”定义声源形态，绘制-3 dB和-6 dB等值线；
3. 40~80、50~80、60~80 cm外圈仅用于估计统一的标量基线；
4. 所有点减去同一个标量，因此不会扭曲空间形状；
5. 输出8方向径向衰减曲线，直接检查中心是不是局部峰值；
6. 多秒取中位数并计算-6 dB声斑出现比例。

必须明确：
没有同位置的“泄漏关闭”数据时，无法唯一分离纯泄漏与环境背景。
本程序输出的是波束形成声场中的“声学足迹”，不是气体喷流的真实外形，
也不是宣称已经得到纯泄漏波形。

时间规则保持与原程序一致：
    center 00_00 -> 所有坐标读取[0, 1)秒
    center 00_01 -> 所有坐标读取[1, 2)秒
    center 00_02 -> 所有坐标读取[2, 3)秒
"""

import csv
import glob
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "gc_footprint_matplotlib_cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
from scipy.interpolate import griddata


# ============================================================
# 1. 用户配置
# ============================================================

time_folders = [
    # "HM20260626_142938.ld",
    # "HM20260626_143034.ld",
    "HM20260702_111044.ld",
]

center_root_dir = r"D:\gas\beamform_results_gc"
offset_near_root_dir = r"D:\gas\beamform_results_offset_multiple_gc_0001"
offset_far_root_dir = r"D:\gas\beamform_results_offset_multiple_gc_gy_0001"
result_root_dir = r"D:\gas\results_gc_acoustic_footprint"

# 只用于图片和CSV标注，不参与计算。
TIME_FOLDER_LABELS: Dict[str, str] = {
    # "真实泄漏中心文件夹": "TRUE",
    # "随机环境中心文件夹": "FALSE",
}

RECURSIVE_SEARCH = False
EXPECTED_SAMPLE_RATE = 192000
TIME_SLICE_SECONDS = 1.0
STRICT_TIME_SLICE = True

FREQ_LOW = 50000
FREQ_HIGH = 70000
NFFT = 4096
WELCH_OVERLAP_RATIO = 0.5

DIRECTION_ANGLES = {
    "up": np.pi / 2,
    "down": -np.pi / 2,
    "left": np.pi,
    "right": 0.0,
    "up_left": 3 * np.pi / 4,
    "down_left": -3 * np.pi / 4,
    "up_right": np.pi / 4,
    "down_right": -np.pi / 4,
}

DISTANCES_CM = list(range(5, 81, 5))
MAX_DISTANCE_CM = max(DISTANCES_CM)

# 三种统一标量基线。每一种都使用该距离范围内的全部方向。
# 它们只改变零点，不改变任何空间点之间的相对次序和形态。
OUTER_BASELINE_ANNULI_CM = (
    (40, 80),
    (50, 80),
    (60, 80),
)

# 声学足迹等值线
FOOTPRINT_LEVELS_DB = (-6.0, -3.0)
CENTER_PEAK_TOLERANCE_DB = 1.0
CENTER_PEAK_DISTANCE_CM = 10.0

# 图像
GRID_MIN_CM = -85
GRID_MAX_CM = 85
GRID_SIZE = 280
RAW_CMAP = "viridis"
RELATIVE_CMAP = "turbo"
SIGNED_CMAP = "coolwarm"
PEAK_RELATIVE_MIN_DB = -20.0
DRAW_SINGLE_SECOND_FIGURES = True
DRAW_TEMPORAL_FIGURE = True

ENERGY_FLOOR = 1e-30


# ============================================================
# 2. 数据类型
# ============================================================

OffsetKey = Tuple[str, int, str]


@dataclass
class SpatialPoint:
    point_key: str
    point_type: str
    direction: str
    distance_cm: int
    x_cm: float
    y_cm: float
    energy: float
    energy_db: float
    signal_file: str


@dataclass
class FootprintAnalysis:
    time_folder: str
    label: str
    center_id: str
    time_index: int
    segment_start_second: float
    segment_end_second: float
    points: List[SpatialPoint]
    raw_db: np.ndarray
    outer_baselines_db: Dict[str, float]
    consensus_outer_baseline_db: float
    outer_relative_db: np.ndarray
    peak_relative_db: np.ndarray
    center_relative_db: np.ndarray
    peak_index: int
    center_index: int
    center_is_peak: bool
    radial_distances_cm: np.ndarray
    radial_median_center_relative_db: np.ndarray
    radial_q25_center_relative_db: np.ndarray
    radial_q75_center_relative_db: np.ndarray
    direction_profiles: Dict[str, Tuple[np.ndarray, np.ndarray]]
    minus3_radius_cm: float
    minus6_radius_cm: float


# ============================================================
# 3. 文件解析
# ============================================================

_DIRECTION_REGEX_PART = "|".join(
    re.escape(name)
    for name in sorted(DIRECTION_ANGLES.keys(), key=len, reverse=True)
)

CENTER_FILE_REGEX = re.compile(
    r"(?P<center>\d+(?:_\d+)?)_beamform_result\.wav$",
    flags=re.IGNORECASE,
)

OFFSET_FILE_REGEX = re.compile(
    rf"(?P<center>\d+(?:_\d+)?)d(?P<distance>\d+)_"
    rf"(?P<direction>{_DIRECTION_REGEX_PART})(?P<suffix>.*?)\.wav$",
    flags=re.IGNORECASE,
)


def list_wav_files(folder: str) -> List[str]:
    if RECURSIVE_SEARCH:
        return sorted(glob.glob(os.path.join(folder, "**", "*.wav"), recursive=True))
    return sorted(glob.glob(os.path.join(folder, "*.wav")))


def resolve_time_data_dir(root_dir: str, time_folder: str) -> str:
    nested = os.path.join(root_dir, time_folder)
    if os.path.isdir(nested):
        return nested
    if os.path.isdir(root_dir) and list_wav_files(root_dir):
        print(f"提醒：未找到{nested}，直接读取：{root_dir}")
        return root_dir
    return nested


def build_center_file_index(folder: str) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    for path in list_wav_files(folder):
        match = CENTER_FILE_REGEX.search(os.path.basename(path))
        if match:
            index.setdefault(match.group("center"), []).append(path)
    for key in index:
        index[key] = sorted(index[key])
    return index


def build_offset_file_index(folder: str) -> Dict[OffsetKey, List[str]]:
    index: Dict[OffsetKey, List[str]] = {}
    for path in list_wav_files(folder):
        match = OFFSET_FILE_REGEX.search(os.path.basename(path))
        if not match:
            continue
        key = (
            match.group("center"),
            int(match.group("distance")),
            match.group("direction").lower(),
        )
        index.setdefault(key, []).append(path)
    for key in index:
        index[key] = sorted(index[key])
    return index


def merge_offset_file_indices(
    *indices: Dict[OffsetKey, List[str]],
) -> Dict[OffsetKey, List[str]]:
    merged: Dict[OffsetKey, List[str]] = {}
    for index in indices:
        for key, paths in index.items():
            merged.setdefault(key, []).extend(paths)
    for key in merged:
        merged[key] = sorted(set(merged[key]))
    return merged


def choose_first_file(paths: Optional[List[str]], description: str) -> Optional[str]:
    if not paths:
        return None
    if len(paths) > 1:
        print(f"  警告：{description}匹配到{len(paths)}个文件，使用第一个")
        for path in paths:
            print(f"    - {os.path.basename(path)}")
    return paths[0]


def center_id_sort_key(center_id: str) -> Tuple[int, ...]:
    try:
        return tuple(int(part) for part in center_id.split("_"))
    except ValueError:
        return (10**9,)


def get_time_slice_from_center_id(center_id: str) -> Tuple[int, float, float]:
    parts = center_id.split("_")
    if len(parts) < 2 or not parts[-1].isdigit():
        raise ValueError(f"center_id={center_id!r}不符合00_00格式")
    index = int(parts[-1])
    start = index * float(TIME_SLICE_SECONDS)
    return index, start, start + float(TIME_SLICE_SECONDS)


# ============================================================
# 4. WAV与频带能量
# ============================================================

def convert_wav_to_float(data: np.ndarray) -> np.ndarray:
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        full_scale = float(max(abs(info.min), abs(info.max)))
        if full_scale <= 0:
            raise ValueError(f"无效WAV整数类型：{data.dtype}")
        return data.astype(np.float64) / full_scale
    if np.issubdtype(data.dtype, np.floating):
        return data.astype(np.float64)
    raise TypeError(f"不支持的WAV类型：{data.dtype}")


def read_band_energy(
    file_path: str,
    start_second: float,
    end_second: float,
) -> Optional[float]:
    if not os.path.exists(file_path):
        print(f"  错误：文件不存在：{file_path}")
        return None

    try:
        sample_rate, data = wav.read(file_path)
    except Exception as exc:
        print(f"  错误：读取失败：{file_path}\n    {exc}")
        return None

    if data.ndim > 1:
        data = data[:, 0]

    total_samples = int(data.size)
    start_sample = int(round(start_second * sample_rate))
    end_sample = int(round(end_second * sample_rate))

    if start_sample >= total_samples:
        print(
            f"  错误：{os.path.basename(file_path)}没有"
            f"[{start_second:.3f}, {end_second:.3f})秒"
        )
        return None

    if end_sample > total_samples:
        if STRICT_TIME_SLICE:
            print(
                f"  错误：{os.path.basename(file_path)}目标秒不完整，已跳过"
            )
            return None
        end_sample = total_samples

    data = data[start_sample:end_sample]
    expected = int(round((end_second - start_second) * sample_rate))
    if STRICT_TIME_SLICE and data.size != expected:
        print(
            f"  错误：{os.path.basename(file_path)}切片点数={data.size}，"
            f"期望={expected}"
        )
        return None

    if data.size < 16:
        return None

    try:
        data = convert_wav_to_float(data)
    except (TypeError, ValueError) as exc:
        print(f"  错误：{exc}")
        return None

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = data - np.mean(data)

    if sample_rate != EXPECTED_SAMPLE_RATE:
        print(
            f"  提醒：{os.path.basename(file_path)}采样率={sample_rate}Hz，"
            f"不是{EXPECTED_SAMPLE_RATE}Hz"
        )

    nyquist = sample_rate / 2.0
    actual_high = min(FREQ_HIGH, nyquist)
    if FREQ_LOW >= actual_high:
        print(f"  错误：采样率不足，无法分析{FREQ_LOW}-{FREQ_HIGH}Hz")
        return None

    nperseg = min(NFFT, data.size)
    noverlap = min(int(nperseg * WELCH_OVERLAP_RATIO), nperseg - 1)
    try:
        frequencies, psd = signal.welch(
            data,
            fs=sample_rate,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="density",
        )
    except Exception as exc:
        print(f"  错误：Welch计算失败：{exc}")
        return None

    mask = (frequencies >= FREQ_LOW) & (frequencies <= actual_high)
    frequencies = np.asarray(frequencies[mask], dtype=np.float64)
    psd = np.maximum(
        np.nan_to_num(psd[mask], nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
    )
    if frequencies.size < 2:
        return None

    if hasattr(np, "trapezoid"):
        energy = float(np.trapezoid(psd, frequencies))
    else:
        energy = float(np.trapz(psd, frequencies))
    return max(energy, 0.0)


def energy_to_db(energy: float) -> float:
    return float(10.0 * np.log10(max(float(energy), ENERGY_FLOOR)))


# ============================================================
# 5. 收集129个实测点
# ============================================================

def collect_spatial_points(
    center_id: str,
    center_index: Dict[str, List[str]],
    offset_index: Dict[OffsetKey, List[str]],
    start_second: float,
    end_second: float,
) -> List[SpatialPoint]:
    points: List[SpatialPoint] = []
    center_file = choose_first_file(center_index.get(center_id), f"中心{center_id}")
    if center_file is None:
        return points

    center_energy = read_band_energy(center_file, start_second, end_second)
    if center_energy is None:
        return points

    points.append(
        SpatialPoint(
            point_key="center_0",
            point_type="center",
            direction="center",
            distance_cm=0,
            x_cm=0.0,
            y_cm=0.0,
            energy=center_energy,
            energy_db=energy_to_db(center_energy),
            signal_file=os.path.basename(center_file),
        )
    )

    for direction, angle in DIRECTION_ANGLES.items():
        for distance in DISTANCES_CM:
            file_path = choose_first_file(
                offset_index.get((center_id, distance, direction)),
                f"中心{center_id}、{direction}、{distance}cm",
            )
            if file_path is None:
                print(f"  缺失：{direction}方向{distance}cm")
                continue
            energy = read_band_energy(file_path, start_second, end_second)
            if energy is None:
                continue
            x = float(distance * np.cos(angle))
            y = float(distance * np.sin(angle))
            if abs(x) < 1e-10:
                x = 0.0
            if abs(y) < 1e-10:
                y = 0.0
            points.append(
                SpatialPoint(
                    point_key=f"{direction}_{distance}",
                    point_type="offset",
                    direction=direction,
                    distance_cm=distance,
                    x_cm=x,
                    y_cm=y,
                    energy=energy,
                    energy_db=energy_to_db(energy),
                    signal_file=os.path.basename(file_path),
                )
            )
    return points


# ============================================================
# 6. 形态计算（只使用实测点）
# ============================================================

def estimate_crossing_radius(
    radii: np.ndarray,
    relative_levels_db: np.ndarray,
    target_level_db: float,
) -> float:
    """估计径向中位曲线首次跌破目标dB的半径。"""
    valid = np.isfinite(radii) & np.isfinite(relative_levels_db)
    radii = np.asarray(radii[valid], dtype=np.float64)
    levels = np.asarray(relative_levels_db[valid], dtype=np.float64)
    if radii.size < 2:
        return float("nan")

    order = np.argsort(radii)
    radii = radii[order]
    levels = levels[order]
    for index in range(1, radii.size):
        previous = levels[index - 1]
        current = levels[index]
        if previous >= target_level_db and current < target_level_db:
            if abs(current - previous) < 1e-12:
                return float(radii[index])
            fraction = (target_level_db - previous) / (current - previous)
            return float(radii[index - 1] + fraction * (radii[index] - radii[index - 1]))
    if np.all(levels >= target_level_db):
        return float("nan")
    return float(radii[0])


def analyze_footprint(
    time_folder: str,
    label: str,
    center_id: str,
    time_index: int,
    start_second: float,
    end_second: float,
    points: List[SpatialPoint],
) -> FootprintAnalysis:
    raw_db = np.asarray([point.energy_db for point in points], dtype=np.float64)
    x = np.asarray([point.x_cm for point in points], dtype=np.float64)
    y = np.asarray([point.y_cm for point in points], dtype=np.float64)
    radius = np.hypot(x, y)

    center_candidates = np.flatnonzero(radius < 1e-9)
    if center_candidates.size == 0:
        raise RuntimeError("缺少中心点")
    center_index = int(center_candidates[0])
    peak_index = int(np.nanargmax(raw_db))

    baselines: Dict[str, float] = {}
    for low, high in OUTER_BASELINE_ANNULI_CM:
        mask = (radius >= low - 1e-9) & (radius <= high + 1e-9)
        values = raw_db[mask & np.isfinite(raw_db)]
        if values.size < 8:
            baselines[f"{low}_{high}"] = float("nan")
        else:
            baselines[f"{low}_{high}"] = float(np.median(values))

    finite_baselines = np.asarray(
        [value for value in baselines.values() if np.isfinite(value)],
        dtype=np.float64,
    )
    if finite_baselines.size == 0:
        raise RuntimeError("外圈有效点不足，无法计算统一标量基线")
    consensus_baseline = float(np.median(finite_baselines))

    outer_relative_db = raw_db - consensus_baseline
    peak_relative_db = raw_db - raw_db[peak_index]
    center_relative_db = raw_db - raw_db[center_index]

    peak_distance = float(radius[peak_index])
    center_below_peak = float(raw_db[peak_index] - raw_db[center_index])
    center_is_peak = (
        peak_distance <= CENTER_PEAK_DISTANCE_CM
        and center_below_peak <= CENTER_PEAK_TOLERANCE_DB
    )

    radial_distances = np.asarray([0] + DISTANCES_CM, dtype=np.float64)
    radial_median = np.full(radial_distances.shape, np.nan)
    radial_q25 = np.full(radial_distances.shape, np.nan)
    radial_q75 = np.full(radial_distances.shape, np.nan)

    for index, distance in enumerate(radial_distances):
        mask = np.isclose(radius, distance, atol=1e-6)
        values = center_relative_db[mask & np.isfinite(center_relative_db)]
        if values.size:
            radial_median[index] = float(np.median(values))
            radial_q25[index] = float(np.percentile(values, 25))
            radial_q75[index] = float(np.percentile(values, 75))

    direction_profiles: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for direction in DIRECTION_ANGLES:
        distances = [0.0]
        levels = [0.0]
        items = sorted(
            (
                (point.distance_cm, center_relative_db[index])
                for index, point in enumerate(points)
                if point.direction == direction
            ),
            key=lambda item: item[0],
        )
        for distance, level in items:
            distances.append(float(distance))
            levels.append(float(level))
        direction_profiles[direction] = (
            np.asarray(distances, dtype=np.float64),
            np.asarray(levels, dtype=np.float64),
        )

    minus3_radius = estimate_crossing_radius(
        radial_distances,
        radial_median,
        -3.0,
    )
    minus6_radius = estimate_crossing_radius(
        radial_distances,
        radial_median,
        -6.0,
    )

    return FootprintAnalysis(
        time_folder=time_folder,
        label=label,
        center_id=center_id,
        time_index=time_index,
        segment_start_second=start_second,
        segment_end_second=end_second,
        points=points,
        raw_db=raw_db,
        outer_baselines_db=baselines,
        consensus_outer_baseline_db=consensus_baseline,
        outer_relative_db=outer_relative_db,
        peak_relative_db=peak_relative_db,
        center_relative_db=center_relative_db,
        peak_index=peak_index,
        center_index=center_index,
        center_is_peak=center_is_peak,
        radial_distances_cm=radial_distances,
        radial_median_center_relative_db=radial_median,
        radial_q25_center_relative_db=radial_q25,
        radial_q75_center_relative_db=radial_q75,
        direction_profiles=direction_profiles,
        minus3_radius_cm=minus3_radius,
        minus6_radius_cm=minus6_radius,
    )


# ============================================================
# 7. CSV
# ============================================================

def write_csv(path: str, fields: Sequence[str], rows: Sequence[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def point_rows(analysis: FootprintAnalysis) -> List[dict]:
    rows: List[dict] = []
    for index, point in enumerate(analysis.points):
        rows.append({
            "time_folder": analysis.time_folder,
            "label": analysis.label,
            "center_id": analysis.center_id,
            "time_index": analysis.time_index,
            "point_key": point.point_key,
            "point_type": point.point_type,
            "direction": point.direction,
            "distance_cm": point.distance_cm,
            "x_cm": point.x_cm,
            "y_cm": point.y_cm,
            "raw_energy": point.energy,
            "raw_energy_db": point.energy_db,
            "outer_baseline_db": analysis.consensus_outer_baseline_db,
            "outer_relative_db": analysis.outer_relative_db[index],
            "peak_relative_db": analysis.peak_relative_db[index],
            "center_relative_db": analysis.center_relative_db[index],
            "inside_minus3db_measured": (
                "yes" if analysis.peak_relative_db[index] >= -3.0 else "no"
            ),
            "inside_minus6db_measured": (
                "yes" if analysis.peak_relative_db[index] >= -6.0 else "no"
            ),
            "is_measured_peak": "yes" if index == analysis.peak_index else "no",
            "signal_file": point.signal_file,
        })
    return rows


def summary_row(analysis: FootprintAnalysis) -> dict:
    peak = analysis.points[analysis.peak_index]
    center = analysis.points[analysis.center_index]
    baseline_values = np.asarray(
        [value for value in analysis.outer_baselines_db.values() if np.isfinite(value)]
    )
    baseline_spread = (
        float(np.max(baseline_values) - np.min(baseline_values))
        if baseline_values.size else float("nan")
    )
    return {
        "time_folder": analysis.time_folder,
        "label": analysis.label,
        "center_id": analysis.center_id,
        "time_index": analysis.time_index,
        "segment_start_second": analysis.segment_start_second,
        "segment_end_second": analysis.segment_end_second,
        "number_of_points": len(analysis.points),
        "center_raw_db": center.energy_db,
        "peak_raw_db": peak.energy_db,
        "center_below_peak_db": peak.energy_db - center.energy_db,
        "peak_x_cm": peak.x_cm,
        "peak_y_cm": peak.y_cm,
        "peak_distance_from_center_cm": float(np.hypot(peak.x_cm, peak.y_cm)),
        "center_is_peak": "yes" if analysis.center_is_peak else "no",
        "consensus_outer_baseline_db": analysis.consensus_outer_baseline_db,
        "center_above_outer_baseline_db": (
            center.energy_db - analysis.consensus_outer_baseline_db
        ),
        "outer_baseline_40_80_db": analysis.outer_baselines_db.get("40_80"),
        "outer_baseline_50_80_db": analysis.outer_baselines_db.get("50_80"),
        "outer_baseline_60_80_db": analysis.outer_baselines_db.get("60_80"),
        "outer_baseline_spread_db": baseline_spread,
        "radial_minus3db_radius_cm": analysis.minus3_radius_cm,
        "radial_minus6db_radius_cm": analysis.minus6_radius_cm,
        "measured_points_inside_minus3db": int(np.sum(analysis.peak_relative_db >= -3.0)),
        "measured_points_inside_minus6db": int(np.sum(analysis.peak_relative_db >= -6.0)),
    }


# ============================================================
# 8. 绘图
# ============================================================

def finite_percentile(values: np.ndarray, percentile: float, fallback: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return fallback
    return float(np.percentile(values, percentile))


def interpolate_display(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_x, grid_y = np.mgrid[
        GRID_MIN_CM:GRID_MAX_CM:complex(GRID_SIZE),
        GRID_MIN_CM:GRID_MAX_CM:complex(GRID_SIZE),
    ]
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    if np.sum(valid) < 4:
        return grid_x, grid_y, np.full_like(grid_x, np.nan)
    grid_z = griddata(
        (x[valid], y[valid]),
        values[valid],
        (grid_x, grid_y),
        method="linear",
        fill_value=np.nan,
    )
    grid_z[np.hypot(grid_x, grid_y) > MAX_DISTANCE_CM] = np.nan
    return grid_x, grid_y, grid_z


def draw_map(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    title: str,
    cmap_name: str,
    vmin: float,
    vmax: float,
    colorbar_label: str,
    peak_index: int,
    draw_footprint_contours: bool = False,
) -> None:
    grid_x, grid_y, grid_z = interpolate_display(x, y, values)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="white", alpha=0.0)
    image = axis.imshow(
        np.ma.masked_invalid(grid_z).T,
        extent=(GRID_MIN_CM, GRID_MAX_CM, GRID_MIN_CM, GRID_MAX_CM),
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    axis.scatter(
        x, y, c=values, cmap=cmap, vmin=vmin, vmax=vmax,
        s=15, edgecolors="black", linewidths=0.25, zorder=4,
    )
    if draw_footprint_contours and np.any(np.isfinite(grid_z)):
        try:
            contours = axis.contour(
                grid_x,
                grid_y,
                grid_z,
                levels=list(FOOTPRINT_LEVELS_DB),
                colors=["white", "black"],
                linewidths=[1.5, 1.5],
            )
            axis.clabel(
                contours,
                fmt={-6.0: "-6 dB", -3.0: "-3 dB"},
                fontsize=8,
            )
        except ValueError:
            pass

    axis.scatter(
        [0], [0], marker="*", s=140, c="white",
        edgecolors="red", linewidths=1.2, label="candidate center", zorder=7,
    )
    axis.scatter(
        [x[peak_index]], [y[peak_index]], marker="x", s=100,
        c="black", linewidths=2.0, label="measured peak", zorder=8,
    )
    boundary = plt.Circle(
        (0, 0), MAX_DISTANCE_CM, fill=False,
        linestyle=":", color="black", linewidth=0.9,
    )
    axis.add_patch(boundary)
    axis.set_title(title, fontsize=11)
    axis.set_xlim(GRID_MIN_CM, GRID_MAX_CM)
    axis.set_ylim(GRID_MIN_CM, GRID_MAX_CM)
    axis.set_xlabel("X Distance (cm)")
    axis.set_ylabel("Y Distance (cm)")
    axis.grid(True, linestyle="--", alpha=0.25)
    axis.legend(loc="upper right", fontsize=7)
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label=colorbar_label)


def draw_radial_profile(axis: plt.Axes, analysis: FootprintAnalysis) -> None:
    for direction, (distances, levels) in analysis.direction_profiles.items():
        axis.plot(distances, levels, linewidth=0.8, alpha=0.35, label=direction)

    distances = analysis.radial_distances_cm
    median = analysis.radial_median_center_relative_db
    q25 = analysis.radial_q25_center_relative_db
    q75 = analysis.radial_q75_center_relative_db
    axis.fill_between(distances, q25, q75, color="tab:blue", alpha=0.18, label="25%-75%")
    axis.plot(distances, median, color="black", linewidth=2.4, marker="o", label="8-dir median")
    axis.axhline(-3.0, color="tab:orange", linestyle="--", linewidth=1.2, label="-3 dB")
    axis.axhline(-6.0, color="tab:red", linestyle="--", linewidth=1.2, label="-6 dB")
    axis.axhline(0.0, color="gray", linestyle=":", linewidth=0.9)
    axis.set_xlim(0, MAX_DISTANCE_CM)
    lower = min(-20.0, finite_percentile(q25, 1.0, -20.0) - 1.0)
    upper = max(3.0, finite_percentile(q75, 99.0, 1.0) + 1.0)
    axis.set_ylim(lower, upper)
    axis.set_xlabel("Distance from candidate center (cm)")
    axis.set_ylabel("Level relative to center (dB)")
    axis.set_title("D. Eight-direction radial decay")
    axis.grid(True, alpha=0.3)
    handles, labels = axis.get_legend_handles_labels()
    wanted = [index for index, label in enumerate(labels) if label in {"25%-75%", "8-dir median", "-3 dB", "-6 dB"}]
    axis.legend([handles[i] for i in wanted], [labels[i] for i in wanted], fontsize=8)


def calculate_folder_scales(analyses: Sequence[FootprintAnalysis]) -> Dict[str, float]:
    raw = np.concatenate([analysis.raw_db for analysis in analyses])
    outer = np.concatenate([analysis.outer_relative_db for analysis in analyses])
    raw_min = finite_percentile(raw, 1.0, -120.0)
    raw_max = finite_percentile(raw, 99.0, 0.0)
    if raw_max <= raw_min:
        raw_max = raw_min + 1.0
    outer_limit = max(finite_percentile(np.abs(outer), 99.0, 10.0), 3.0)
    return {"raw_min": raw_min, "raw_max": raw_max, "outer_limit": outer_limit}


def plot_single_analysis(
    result_dir: str,
    analysis: FootprintAnalysis,
    scales: Dict[str, float],
) -> str:
    x = np.asarray([point.x_cm for point in analysis.points])
    y = np.asarray([point.y_cm for point in analysis.points])
    figure, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=110)

    draw_map(
        axes[0, 0], x, y, analysis.raw_db,
        "A. Raw 50-70 kHz beamformed energy",
        RAW_CMAP, scales["raw_min"], scales["raw_max"],
        "Energy (dB)", analysis.peak_index,
    )
    draw_map(
        axes[0, 1], x, y, analysis.peak_relative_db,
        "B. Acoustic footprint relative to measured peak",
        RELATIVE_CMAP, PEAK_RELATIVE_MIN_DB, 0.0,
        "Relative to peak (dB)", analysis.peak_index, True,
    )
    draw_map(
        axes[1, 0], x, y, analysis.outer_relative_db,
        "C. Relative to one common outer-annulus baseline",
        SIGNED_CMAP, -scales["outer_limit"], scales["outer_limit"],
        "Above outer baseline (dB)", analysis.peak_index,
    )
    draw_radial_profile(axes[1, 1], analysis)

    peak = analysis.points[analysis.peak_index]
    center = analysis.points[analysis.center_index]
    status = "CENTER IS PEAK" if analysis.center_is_peak else "CENTER IS NOT PEAK"
    baseline_values = np.asarray([
        value for value in analysis.outer_baselines_db.values() if np.isfinite(value)
    ])
    baseline_spread = float(np.ptp(baseline_values)) if baseline_values.size else float("nan")
    figure.suptitle(
        f"Acoustic footprint | {analysis.time_folder} | label={analysis.label} | "
        f"center={analysis.center_id} | {status}\n"
        f"peak=({peak.x_cm:.0f},{peak.y_cm:.0f})cm, "
        f"center below peak={peak.energy_db-center.energy_db:.2f}dB, "
        f"center above outer baseline={center.energy_db-analysis.consensus_outer_baseline_db:.2f}dB, "
        f"outer-baseline spread={baseline_spread:.2f}dB",
        fontsize=12,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(
        result_dir,
        f"acoustic_footprint_{analysis.time_folder}_center_{analysis.center_id}.png",
    )
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return path


# ============================================================
# 9. 多秒稳定形态
# ============================================================

def aggregate_temporal(
    analyses: Sequence[FootprintAnalysis],
) -> Tuple[List[SpatialPoint], Dict[str, np.ndarray]]:
    if not analyses:
        return [], {}
    maps = [
        {point.point_key: index for index, point in enumerate(analysis.points)}
        for analysis in analyses
    ]
    common = set(maps[0])
    for item in maps[1:]:
        common &= set(item)
    keys = [point.point_key for point in analyses[0].points if point.point_key in common]
    points = [analyses[0].points[maps[0][key]] for key in keys]

    raw_stack = []
    peak_relative_stack = []
    outer_relative_stack = []
    center_relative_stack = []
    for analysis, point_map in zip(analyses, maps):
        indices = np.asarray([point_map[key] for key in keys], dtype=int)
        raw_stack.append(analysis.raw_db[indices])
        peak_relative_stack.append(analysis.peak_relative_db[indices])
        outer_relative_stack.append(analysis.outer_relative_db[indices])
        center_relative_stack.append(analysis.center_relative_db[indices])

    raw_array = np.vstack(raw_stack)
    peak_relative_array = np.vstack(peak_relative_stack)
    outer_relative_array = np.vstack(outer_relative_stack)
    center_relative_array = np.vstack(center_relative_stack)

    values = {
        "median_raw_db": np.nanmedian(raw_array, axis=0),
        "median_peak_relative_db": np.nanmedian(peak_relative_array, axis=0),
        "median_outer_relative_db": np.nanmedian(outer_relative_array, axis=0),
        "median_center_relative_db": np.nanmedian(center_relative_array, axis=0),
        "minus3_time_support": np.nanmean(peak_relative_array >= -3.0, axis=0),
        "minus6_time_support": np.nanmean(peak_relative_array >= -6.0, axis=0),
    }
    return points, values


def save_temporal_csv(
    result_dir: str,
    time_folder: str,
    label: str,
    points: Sequence[SpatialPoint],
    values: Dict[str, np.ndarray],
    number_of_seconds: int,
) -> str:
    rows = []
    for index, point in enumerate(points):
        rows.append({
            "time_folder": time_folder,
            "label": label,
            "number_of_seconds": number_of_seconds,
            "point_key": point.point_key,
            "point_type": point.point_type,
            "direction": point.direction,
            "distance_cm": point.distance_cm,
            "x_cm": point.x_cm,
            "y_cm": point.y_cm,
            "median_raw_db": values["median_raw_db"][index],
            "median_peak_relative_db": values["median_peak_relative_db"][index],
            "median_outer_relative_db": values["median_outer_relative_db"][index],
            "median_center_relative_db": values["median_center_relative_db"][index],
            "minus3db_time_support": values["minus3_time_support"][index],
            "minus6db_time_support": values["minus6_time_support"][index],
        })
    path = os.path.join(result_dir, f"temporal_acoustic_footprint_{time_folder}.csv")
    if rows:
        write_csv(path, list(rows[0].keys()), rows)
    return path


def plot_temporal(
    result_dir: str,
    time_folder: str,
    label: str,
    points: Sequence[SpatialPoint],
    values: Dict[str, np.ndarray],
    analyses: Sequence[FootprintAnalysis],
    scales: Dict[str, float],
) -> str:
    x = np.asarray([point.x_cm for point in points])
    y = np.asarray([point.y_cm for point in points])
    peak_index = int(np.nanargmax(values["median_raw_db"]))
    figure, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=110)

    draw_map(
        axes[0, 0], x, y, values["median_raw_db"],
        "A. Median raw energy across seconds",
        RAW_CMAP, scales["raw_min"], scales["raw_max"],
        "Median energy (dB)", peak_index,
    )
    draw_map(
        axes[0, 1], x, y, values["median_peak_relative_db"],
        "B. Median peak-relative footprint",
        RELATIVE_CMAP, PEAK_RELATIVE_MIN_DB, 0.0,
        "Median relative level (dB)", peak_index, True,
    )
    draw_map(
        axes[1, 0], x, y, values["minus6_time_support"],
        "C. Fraction of seconds inside -6 dB footprint",
        "magma", 0.0, 1.0,
        "Time support", peak_index,
    )

    axis = axes[1, 1]
    for analysis in analyses:
        axis.plot(
            analysis.radial_distances_cm,
            analysis.radial_median_center_relative_db,
            color="tab:blue",
            alpha=0.22,
            linewidth=0.9,
        )
    radial_stack = np.vstack([
        analysis.radial_median_center_relative_db for analysis in analyses
    ])
    temporal_radial_median = np.nanmedian(radial_stack, axis=0)
    axis.plot(
        analyses[0].radial_distances_cm,
        temporal_radial_median,
        color="black",
        marker="o",
        linewidth=2.4,
        label="temporal median",
    )
    axis.axhline(-3, color="tab:orange", linestyle="--", label="-3 dB")
    axis.axhline(-6, color="tab:red", linestyle="--", label="-6 dB")
    axis.set_xlim(0, MAX_DISTANCE_CM)
    axis.set_xlabel("Distance from candidate center (cm)")
    axis.set_ylabel("Relative to center (dB)")
    axis.set_title("D. Radial decay stability across seconds")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)

    center_peak_rate = float(np.mean([analysis.center_is_peak for analysis in analyses]))
    figure.suptitle(
        f"Temporal acoustic footprint | {time_folder} | label={label} | "
        f"seconds={len(analyses)}\n"
        f"candidate center is peak in {center_peak_rate:.1%} of seconds",
        fontsize=12,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(result_dir, f"temporal_acoustic_footprint_{time_folder}.png")
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return path


# ============================================================
# 10. 主程序
# ============================================================

def process_time_folder(time_folder: str) -> List[FootprintAnalysis]:
    print("\n" + "=" * 88)
    print(f"处理：{time_folder}（原始波束声场相对声级形态）")
    print("=" * 88)

    center_dir = resolve_time_data_dir(center_root_dir, time_folder)
    near_dir = resolve_time_data_dir(offset_near_root_dir, time_folder)
    far_dir = resolve_time_data_dir(offset_far_root_dir, time_folder)
    required = [
        ("中心", center_dir),
        ("5~40cm", near_dir),
        ("45~80cm", far_dir),
    ]
    if any(not os.path.isdir(path) for _, path in required):
        for name, path in required:
            if not os.path.isdir(path):
                print(f"错误：{name}目录不存在：{path}")
        return []

    center_index = build_center_file_index(center_dir)
    offset_index = merge_offset_file_indices(
        build_offset_file_index(near_dir),
        build_offset_file_index(far_dir),
    )
    center_ids = sorted(center_index, key=center_id_sort_key)
    if not center_ids:
        print("错误：没有找到中心WAV")
        return []

    label = TIME_FOLDER_LABELS.get(time_folder, "UNLABELED")
    result_dir = os.path.join(result_root_dir, time_folder)
    os.makedirs(result_dir, exist_ok=True)
    analyses: List[FootprintAnalysis] = []
    summaries: List[dict] = []

    for center_id in center_ids:
        try:
            time_index, start, end = get_time_slice_from_center_id(center_id)
        except ValueError as exc:
            print(f"错误：{exc}")
            continue
        print(f"\n中心{center_id}：读取[{start:.0f}, {end:.0f})秒")
        points = collect_spatial_points(
            center_id, center_index, offset_index, start, end
        )
        print(f"  有效测点：{len(points)}/129")
        if len(points) < 60:
            print("  错误：有效测点少于60，跳过")
            continue
        try:
            analysis = analyze_footprint(
                time_folder, label, center_id, time_index, start, end, points
            )
        except Exception as exc:
            print(f"  错误：形态计算失败：{exc}")
            continue

        analyses.append(analysis)
        rows = point_rows(analysis)
        point_path = os.path.join(
            result_dir,
            f"points_acoustic_footprint_{time_folder}_center_{center_id}.csv",
        )
        write_csv(point_path, list(rows[0].keys()), rows)
        summary = summary_row(analysis)
        summaries.append(summary)
        print(
            f"  峰值距离中心={summary['peak_distance_from_center_cm']:.1f}cm，"
            f"中心比峰值低={summary['center_below_peak_db']:.2f}dB，"
            f"中心高于统一外圈={summary['center_above_outer_baseline_db']:.2f}dB，"
            f"center_is_peak={summary['center_is_peak']}"
        )

    if not analyses:
        return []

    write_csv(
        os.path.join(result_dir, f"summary_acoustic_footprint_{time_folder}.csv"),
        list(summaries[0].keys()),
        summaries,
    )
    scales = calculate_folder_scales(analyses)
    if DRAW_SINGLE_SECOND_FIGURES:
        for analysis in analyses:
            path = plot_single_analysis(result_dir, analysis, scales)
            print(f"  单秒形态图：{path}")

    temporal_points, temporal_values = aggregate_temporal(analyses)
    if temporal_points:
        csv_path = save_temporal_csv(
            result_dir,
            time_folder,
            label,
            temporal_points,
            temporal_values,
            len(analyses),
        )
        print(f"  多秒CSV：{csv_path}")
        if DRAW_TEMPORAL_FIGURE:
            figure_path = plot_temporal(
                result_dir,
                time_folder,
                label,
                temporal_points,
                temporal_values,
                analyses,
                scales,
            )
            print(f"  多秒形态图：{figure_path}")
    return analyses


def validate_configuration() -> None:
    if FREQ_LOW < 0 or FREQ_HIGH <= FREQ_LOW:
        raise ValueError("频率范围设置错误")
    if not OUTER_BASELINE_ANNULI_CM:
        raise ValueError("外圈基线范围不能为空")
    for low, high in OUTER_BASELINE_ANNULI_CM:
        if not (0 <= low < high <= MAX_DISTANCE_CM):
            raise ValueError(f"非法外圈范围：{low}-{high}cm")


def main() -> None:
    validate_configuration()
    os.makedirs(result_root_dir, exist_ok=True)
    print("=" * 88)
    print("GC泄漏声源声学足迹：原始波束相对声级法")
    print("=" * 88)
    print(f"频带：{FREQ_LOW/1000:.1f}-{FREQ_HIGH/1000:.1f} kHz")
    print("形态：相对实测峰值的-3 dB/-6 dB等值线")
    print(f"统一外圈基线：{list(OUTER_BASELINE_ANNULI_CM)} cm")
    print("不拟合二维背景，不会把宽泄漏声斑作为平滑背景减掉。")

    total = 0
    for folder in time_folders:
        total += len(process_time_folder(folder))
    print("\n" + "=" * 88)
    print(f"完成。成功分析秒数：{total}")
    print(f"结果目录：{result_root_dir}")
    print("=" * 88)


if __name__ == "__main__":
    main()

