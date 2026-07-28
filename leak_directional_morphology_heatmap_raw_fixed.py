# -*- coding: utf-8 -*-
"""
Directional leak morphology heatmap
===================================

目标
----
使用同一段原始阵列录音、同一秒、不同波束坐标的WAV，输出：

1. absolute_morphology_*.png
   固定统一色标的泄漏声场形态图。
   用于比较真泄漏和假泄漏，重点看这张。

2. direction_shape_*.png
   8个方向的多瓣极坐标形态图。
   不强制只有一个方向，不默认绘制方向箭头。

3. radial_profile_*.csv
   每个方向、每个距离的原始稳定分数和径向整理后的形态分数。

4. summary_all.csv
   所有样本的方向和强度摘要。

核心处理
--------
- center_id最后一段编号决定读取WAV的哪一秒；
- 每一秒拆成多个短时帧；
- 每帧使用外围点的中位数作为同一时刻背景；
- 计算测点相对外围背景的稳定dB超量；
- 去除只在少数帧出现的偶发热点；
- 每个方向沿距离做“非递增径向整理”，保留从中心附近向外衰减的声场成分；
- 所有样本先统一计算，再使用同一个颜色上限绘图。

注意
----
本程序显示的是“泄漏声场的空间形态与多方向延伸”，
不是可见光意义上的真实气体羽流。
"""

import csv
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
from scipy.interpolate import RegularGridInterpolator


# ============================================================
# 1. 用户配置
# ============================================================

# 把需要一起比较的真泄漏、假泄漏文件夹全部放进来。
# 程序会先处理全部样本，再统一色标。
time_folders = [
    "HM20260702_111044.ld",
]

center_root_dir = r"D:\gas\beamform_results_sh"
offset_root_dir = r"D:\gas\beamform_results_offset_multiple_sh"

# 总输出目录
OUTPUT_ROOT_DIR = "results_directional_morphology"

# 是否递归搜索WAV
RECURSIVE_SEARCH = False

# 采样率仅用于检查，实际计算使用每个WAV自己的采样率
EXPECTED_SAMPLE_RATE = 192000

# center00_00 -> [0,1)秒；center00_01 -> [1,2)秒
TIME_SLICE_SECONDS = 1.0
STRICT_TIME_SLICE = True

# 分析频段
FREQ_LOW = 50000
FREQ_HIGH = 70000

# 一秒拆帧：默认0.10秒一帧，步长0.05秒
FRAME_SECONDS = 0.10
HOP_SECONDS = 0.05

# 每个短时帧内部的Welch参数
WELCH_NPERSEG = 4096
WELCH_OVERLAP_RATIO = 0.5

# 8个方向
direction_angles = {
    "right": 0.0,
    "up_right": np.pi / 4,
    "up": np.pi / 2,
    "up_left": 3 * np.pi / 4,
    "left": np.pi,
    "down_left": 5 * np.pi / 4,
    "down": 3 * np.pi / 2,
    "down_right": 7 * np.pi / 4,
}

DIRECTION_ORDER = [
    "right",
    "up_right",
    "up",
    "up_left",
    "left",
    "down_left",
    "down",
    "down_right",
]

DIRECTION_LABELS = {
    "right": "Right",
    "up_right": "Up-right",
    "up": "Up",
    "up_left": "Up-left",
    "left": "Left",
    "down_left": "Down-left",
    "down": "Down",
    "down_right": "Down-right",
}

# 距离，单位cm
distances = [5, 10, 15, 20, 25, 30, 35, 40]
RADII = np.asarray([0] + distances, dtype=np.float64)

# 使用多少厘米以外的点作为外围背景
OUTER_BACKGROUND_MIN_CM = 30

# 每帧至少高于外围背景多少dB才计为有效超量
MIN_EXCESS_DB = 0.8

# 外围空间波动较大时，自动提高阈值
OUTER_MAD_FACTOR = 1.0

# 一个测点至少在多少比例的短时帧内有效，才保留
MIN_PERSISTENCE = 0.35

# 稳定分数中持续率的权重
PERSISTENCE_POWER = 0.5

# 方向形态计算使用的近场和远场
NEAR_MIN_CM = 5
NEAR_MAX_CM = 20
FAR_MIN_CM = 30
FAR_MAX_CM = 40

# 方向被认为是主要方向的相对阈值。
# 例如0.70表示：达到最强方向70%以上的方向都保留，因此允许多个方向。
DOMINANT_DIRECTION_RATIO = 0.70

# 是否输出多瓣方向极坐标图
OUTPUT_DIRECTION_SHAPE = True

# 是否输出每张图单独归一化的形态图。
# 默认False，避免假泄漏也被自动拉成红色。
OUTPUT_NORMALIZED_SHAPE = False

# 默认不画单一方向箭头，因为可能存在多个方向
DRAW_SINGLE_DIRECTION_ARROW = False

# 热力图范围
GRID_LIMIT_CM = 42
GRID_SIZE = 321
HEATMAP_CMAP = "turbo"

# 全部样本统一色标的百分位，防止极少数异常点把色标拉得过高
GLOBAL_COLOR_PERCENTILE = 99.0

# 极坐标方向图也使用统一半径
GLOBAL_DIRECTION_PERCENTILE = 99.0

# 主泄漏图只显示中心到多少厘米。
# 30~40cm仍用于外围背景诊断，不放进主泄漏区域解释。
NEAR_DISPLAY_MAX_CM = 25.0

# 外围背景点中，stable_score>0的点占比超过该阈值时，
# 标记为BACKGROUND_NOT_CLEAN。
OUTER_POSITIVE_FRACTION_WARN = 0.40

# 是否输出三类图：
# 1) 原始近场主图；2) 原始全范围背景诊断图；3) 整理后辅助图。
OUTPUT_RAW_NEAR_HEATMAP = True
OUTPUT_RAW_FULL_DIAGNOSTIC = True
OUTPUT_ADJUSTED_HEATMAP = True

# 数值保护
EPS = 1e-12
PSD_EPS = 1e-30


# ============================================================
# 2. 文件名解析
# ============================================================

OffsetKey = Tuple[str, int, str]

_DIRECTION_REGEX_PART = "|".join(
    re.escape(name)
    for name in sorted(direction_angles.keys(), key=len, reverse=True)
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
    """列出WAV文件。"""
    if RECURSIVE_SEARCH:
        pattern = os.path.join(folder, "**", "*.wav")
        return sorted(glob.glob(pattern, recursive=True))

    pattern = os.path.join(folder, "*.wav")
    return sorted(glob.glob(pattern))


def build_center_file_index(center_data_dir: str) -> Dict[str, List[str]]:
    """center_id -> 中心WAV列表。"""
    index: Dict[str, List[str]] = {}

    for file_path in list_wav_files(center_data_dir):
        match = CENTER_FILE_REGEX.search(os.path.basename(file_path))
        if not match:
            continue

        center_id = match.group("center")
        index.setdefault(center_id, []).append(file_path)

    for center_id in index:
        index[center_id] = sorted(index[center_id])

    return index


def build_offset_file_index(offset_data_dir: str) -> Dict[OffsetKey, List[str]]:
    """(center_id, distance, direction) -> 偏移WAV列表。"""
    index: Dict[OffsetKey, List[str]] = {}

    for file_path in list_wav_files(offset_data_dir):
        match = OFFSET_FILE_REGEX.search(os.path.basename(file_path))
        if not match:
            continue

        center_id = match.group("center")
        distance = int(match.group("distance"))
        direction = match.group("direction").lower()

        key = (center_id, distance, direction)
        index.setdefault(key, []).append(file_path)

    for key in index:
        index[key] = sorted(index[key])

    return index


def choose_first_file(
    files: Optional[List[str]],
    description: str,
) -> Optional[str]:
    """存在多个文件时，警告并选择排序后的第一个。"""
    if not files:
        return None

    if len(files) > 1:
        print(f"  警告：{description}匹配到{len(files)}个文件，使用第一个：")
        for file_path in files:
            print(f"    - {os.path.basename(file_path)}")

    return files[0]


def center_id_sort_key(center_id: str) -> Tuple[int, ...]:
    try:
        return tuple(int(part) for part in str(center_id).split("_"))
    except ValueError:
        return (10**9,)


def get_time_slice_from_center_id(
    center_id: str,
) -> Tuple[int, float, float]:
    """
    00_00 -> [0,1)秒
    00_01 -> [1,2)秒
    01_03 -> [3,4)秒
    """
    parts = str(center_id).split("_")

    if len(parts) < 2 or not parts[-1].isdigit():
        raise ValueError(
            f"center_id={center_id!r}不符合'00_00'格式，"
            "无法判断需要读取原始WAV的第几秒"
        )

    time_index = int(parts[-1])
    start_second = time_index * float(TIME_SLICE_SECONDS)
    end_second = start_second + float(TIME_SLICE_SECONDS)

    return time_index, start_second, end_second


# ============================================================
# 3. WAV读取和短时频带能量
# ============================================================

def convert_wav_to_float(y: np.ndarray) -> np.ndarray:
    """
    转换为float64，但不做逐文件峰值归一化。
    保留不同波束坐标之间的真实幅值关系。
    """
    if np.issubdtype(y.dtype, np.integer):
        info = np.iinfo(y.dtype)
        full_scale = float(max(abs(info.min), abs(info.max)))

        if full_scale <= 0:
            raise ValueError(f"无效的整数WAV类型：{y.dtype}")

        return y.astype(np.float64) / full_scale

    if np.issubdtype(y.dtype, np.floating):
        return y.astype(np.float64)

    raise TypeError(f"不支持的WAV类型：{y.dtype}")


def read_wav_segment(
    file_path: str,
    segment_start_second: float,
    segment_end_second: float,
) -> Tuple[Optional[int], Optional[np.ndarray]]:
    """严格读取指定的一秒，不使用整段WAV代替。"""
    if not os.path.exists(file_path):
        print(f"  警告：文件不存在：{file_path}")
        return None, None

    try:
        sample_rate, y = wav.read(file_path)
    except Exception as exc:
        print(f"  错误：读取WAV失败：{file_path}\n    {exc}")
        return None, None

    if y.ndim > 1:
        y = y[:, 0]

    try:
        y = convert_wav_to_float(y)
    except Exception as exc:
        print(f"  错误：WAV数值转换失败：{file_path}\n    {exc}")
        return None, None

    total_samples = int(y.size)
    start_sample = int(round(segment_start_second * sample_rate))
    end_sample = int(round(segment_end_second * sample_rate))

    if start_sample >= total_samples:
        print(
            f"  错误：{os.path.basename(file_path)}时长不足，"
            f"无法读取[{segment_start_second:.3f}, "
            f"{segment_end_second:.3f})秒"
        )
        return None, None

    if end_sample > total_samples:
        if STRICT_TIME_SLICE:
            print(
                f"  错误：{os.path.basename(file_path)}的目标一秒不完整，"
                "已跳过，未使用整段WAV代替"
            )
            return None, None

        end_sample = total_samples

    segment = y[start_sample:end_sample]

    expected_samples = int(round(
        (segment_end_second - segment_start_second) * sample_rate
    ))

    if STRICT_TIME_SLICE and segment.size != expected_samples:
        print(
            f"  错误：{os.path.basename(file_path)}切片长度={segment.size}，"
            f"期望={expected_samples}，已跳过"
        )
        return None, None

    segment = np.nan_to_num(
        segment,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    segment = segment - np.mean(segment)

    if sample_rate != EXPECTED_SAMPLE_RATE:
        print(
            f"  提醒：{os.path.basename(file_path)}采样率为"
            f"{sample_rate}Hz，不是期望的{EXPECTED_SAMPLE_RATE}Hz"
        )

    return sample_rate, segment


def integrate_spectrum(
    psd: np.ndarray,
    freqs: np.ndarray,
) -> float:
    """兼容不同NumPy版本的梯形积分。"""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(psd, freqs))
    return float(np.trapz(psd, freqs))


def compute_frame_band_energy_db(
    y: np.ndarray,
    sample_rate: int,
) -> Optional[np.ndarray]:
    """
    把当前一秒拆成短时帧。
    每一帧计算FREQ_LOW~FREQ_HIGH的积分能量，并转换为dB。
    """
    frame_length = int(round(FRAME_SECONDS * sample_rate))
    hop_length = int(round(HOP_SECONDS * sample_rate))

    if frame_length < 16 or hop_length < 1:
        return None

    if y.size < frame_length:
        return None

    nyquist = sample_rate / 2.0
    if FREQ_LOW >= nyquist:
        print(
            f"  错误：奈奎斯特频率仅{nyquist:.1f}Hz，"
            f"无法分析{FREQ_LOW}-{FREQ_HIGH}Hz"
        )
        return None

    actual_high = min(FREQ_HIGH, nyquist)

    starts = np.arange(
        0,
        y.size - frame_length + 1,
        hop_length,
        dtype=int,
    )

    if starts.size == 0:
        return None

    energies_db: List[float] = []

    for start in starts:
        frame = y[start:start + frame_length]
        frame = np.nan_to_num(
            frame,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        frame = frame - np.mean(frame)

        nperseg = min(WELCH_NPERSEG, frame.size)
        if nperseg < 16:
            continue

        noverlap = min(
            int(round(nperseg * WELCH_OVERLAP_RATIO)),
            nperseg - 1,
        )

        try:
            freqs, psd = signal.welch(
                frame,
                fs=sample_rate,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                detrend="constant",
                scaling="density",
            )
        except Exception as exc:
            print(f"  警告：短时Welch计算失败：{exc}")
            continue

        mask = (freqs >= FREQ_LOW) & (freqs <= actual_high)

        if np.sum(mask) < 2:
            continue

        band_freqs = freqs[mask]
        band_psd = np.maximum(
            np.nan_to_num(
                psd[mask],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            0.0,
        )

        energy = max(integrate_spectrum(band_psd, band_freqs), 0.0)
        energy_db = 10.0 * np.log10(energy + PSD_EPS)
        energies_db.append(float(energy_db))

    if not energies_db:
        return None

    return np.asarray(energies_db, dtype=np.float64)


# ============================================================
# 4. 稳健统计和径向整理
# ============================================================

def robust_mad(values: np.ndarray) -> float:
    """1.4826*MAD，与标准差量纲一致。"""
    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return 0.0

    median_value = float(np.median(values))
    mad = float(np.median(np.abs(values - median_value)))
    return 1.4826 * mad


def pava_decreasing(
    values: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    非递增等距回归（PAVA）。

    作用：
    将一个方向上随距离变化的分数整理为“整体不随距离升高”的曲线，
    只保留更符合中心附近声源向外衰减的部分。

    这不是简单排序，仍尽量接近原始数据。
    """
    y = np.asarray(values, dtype=np.float64)

    if y.ndim != 1:
        raise ValueError("pava_decreasing只接受一维数组")

    if weights is None:
        w = np.ones_like(y)
    else:
        w = np.asarray(weights, dtype=np.float64)

    if w.shape != y.shape:
        raise ValueError("weights与values形状不一致")

    block_values: List[float] = []
    block_weights: List[float] = []
    block_lengths: List[int] = []

    for value, weight in zip(y, w):
        block_values.append(float(value))
        block_weights.append(float(max(weight, EPS)))
        block_lengths.append(1)

        # 非递增要求：前一个块 >= 后一个块
        while (
            len(block_values) >= 2
            and block_values[-2] < block_values[-1]
        ):
            new_weight = block_weights[-2] + block_weights[-1]
            new_value = (
                block_values[-2] * block_weights[-2]
                + block_values[-1] * block_weights[-1]
            ) / new_weight
            new_length = block_lengths[-2] + block_lengths[-1]

            block_values[-2:] = [new_value]
            block_weights[-2:] = [new_weight]
            block_lengths[-2:] = [new_length]

    output: List[float] = []

    for value, length in zip(block_values, block_lengths):
        output.extend([value] * length)

    return np.asarray(output, dtype=np.float64)


def safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """常量数组时返回0，避免相关系数NaN。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.size < 2 or y.size < 2:
        return 0.0

    if np.std(x) <= EPS or np.std(y) <= EPS:
        return 0.0

    corr = float(np.corrcoef(x, y)[0, 1])

    if not np.isfinite(corr):
        return 0.0

    return corr


# ============================================================
# 5. 构建测点
# ============================================================

def build_point_records(
    center_id: str,
    center_index: Dict[str, List[str]],
    offset_index: Dict[OffsetKey, List[str]],
) -> List[dict]:
    """建立中心和8方向各距离的测点列表。"""
    records: List[dict] = []

    center_file = choose_first_file(
        center_index.get(center_id),
        f"中心点{center_id}",
    )

    if center_file is not None:
        records.append({
            "point_type": "center",
            "direction": "center",
            "distance_cm": 0,
            "x_cm": 0.0,
            "y_cm": 0.0,
            "file_path": center_file,
        })

    for direction in DIRECTION_ORDER:
        angle = direction_angles[direction]

        for distance in distances:
            key = (center_id, distance, direction)

            file_path = choose_first_file(
                offset_index.get(key),
                f"中心点{center_id}、{direction}方向、{distance}cm",
            )

            if file_path is None:
                continue

            x = float(distance * np.cos(angle))
            y = float(distance * np.sin(angle))

            if abs(x) < 1e-10:
                x = 0.0
            if abs(y) < 1e-10:
                y = 0.0

            records.append({
                "point_type": "offset",
                "direction": direction,
                "distance_cm": int(distance),
                "x_cm": x,
                "y_cm": y,
                "file_path": file_path,
            })

    return records


def load_all_point_frame_energies(
    point_records: List[dict],
    segment_start_second: float,
    segment_end_second: float,
) -> Tuple[List[dict], Optional[np.ndarray]]:
    """
    返回：
    valid_records
    energy_db_matrix: [测点数, 短时帧数]
    """
    valid_records: List[dict] = []
    energy_arrays: List[np.ndarray] = []

    for record in point_records:
        sample_rate, y = read_wav_segment(
            record["file_path"],
            segment_start_second,
            segment_end_second,
        )

        if sample_rate is None or y is None:
            continue

        energies_db = compute_frame_band_energy_db(y, sample_rate)

        if energies_db is None or energies_db.size < 3:
            print(
                f"  警告：有效短时帧不足："
                f"{os.path.basename(record['file_path'])}"
            )
            continue

        valid_records.append(record)
        energy_arrays.append(energies_db)

    if not energy_arrays:
        return [], None

    common_frames = min(array.size for array in energy_arrays)

    if common_frames < 3:
        return [], None

    energy_arrays = [
        array[:common_frames]
        for array in energy_arrays
    ]

    matrix = np.vstack(energy_arrays)

    return valid_records, matrix


# ============================================================
# 6. 稳定空间分数
# ============================================================

def compute_stable_point_scores(
    point_records: List[dict],
    energy_db_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    每个短时帧先使用外围点建立同时刻背景。

    excess(point, frame)
        = point_dB - outer_median_dB - dynamic_threshold_dB

    最终点分数：
        平均正超量 × 持续率权重
    """
    distances_array = np.asarray(
        [record["distance_cm"] for record in point_records],
        dtype=np.float64,
    )

    outer_mask = distances_array >= OUTER_BACKGROUND_MIN_CM

    if np.sum(outer_mask) < 8:
        raise RuntimeError(
            f"外围背景点不足：distance >= "
            f"{OUTER_BACKGROUND_MIN_CM}cm的有效点少于8个"
        )

    n_points, n_frames = energy_db_matrix.shape
    positive_excess = np.zeros((n_points, n_frames), dtype=np.float64)
    frame_thresholds = np.zeros(n_frames, dtype=np.float64)

    for frame_index in range(n_frames):
        outer_values = energy_db_matrix[outer_mask, frame_index]

        background_db = float(np.median(outer_values))
        outer_spread_db = robust_mad(outer_values)

        dynamic_threshold_db = max(
            MIN_EXCESS_DB,
            OUTER_MAD_FACTOR * outer_spread_db,
        )
        frame_thresholds[frame_index] = dynamic_threshold_db

        excess_db = (
            energy_db_matrix[:, frame_index]
            - background_db
            - dynamic_threshold_db
        )

        positive_excess[:, frame_index] = np.maximum(excess_db, 0.0)

    persistence = np.mean(
        positive_excess > 0.0,
        axis=1,
    )

    mean_positive_excess = np.mean(
        positive_excess,
        axis=1,
    )

    point_scores = (
        mean_positive_excess
        * np.power(
            np.clip(persistence, 0.0, 1.0),
            PERSISTENCE_POWER,
        )
    )

    point_scores[persistence < MIN_PERSISTENCE] = 0.0
    point_scores = np.maximum(point_scores, 0.0)

    return point_scores, persistence, frame_thresholds


# ============================================================
# 7. 多方向泄漏形态
# ============================================================

def find_record_index(
    point_records: List[dict],
    direction: str,
    distance_cm: int,
) -> Optional[int]:
    for index, record in enumerate(point_records):
        if (
            record["direction"] == direction
            and int(record["distance_cm"]) == int(distance_cm)
        ):
            return index
    return None


def build_directional_profiles(
    point_records: List[dict],
    point_scores: np.ndarray,
    persistence: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    返回：
    raw_profiles:   [8方向, 9半径]
    shape_profiles: [8方向, 9半径]
    direction_rows: 每个方向的统计结果

    shape_profiles允许同时出现多个方向，不强制单一箭头。
    """
    center_indices = [
        index
        for index, record in enumerate(point_records)
        if record["point_type"] == "center"
    ]

    if not center_indices:
        raise RuntimeError("没有有效中心点")

    center_index = center_indices[0]
    center_score = float(point_scores[center_index])
    center_persistence = float(persistence[center_index])

    raw_profiles = np.zeros(
        (len(DIRECTION_ORDER), RADII.size),
        dtype=np.float64,
    )
    shape_profiles = np.zeros_like(raw_profiles)

    direction_rows: List[dict] = []

    for direction_index, direction in enumerate(DIRECTION_ORDER):
        available_radii: List[float] = [0.0]
        available_scores: List[float] = [center_score]
        available_persistence: List[float] = [center_persistence]

        for distance in distances:
            record_index = find_record_index(
                point_records,
                direction,
                distance,
            )

            if record_index is None:
                continue

            available_radii.append(float(distance))
            available_scores.append(float(point_scores[record_index]))
            available_persistence.append(float(persistence[record_index]))

        available_radii_array = np.asarray(
            available_radii,
            dtype=np.float64,
        )
        available_scores_array = np.asarray(
            available_scores,
            dtype=np.float64,
        )

        if available_radii_array.size < 3:
            continue

        order = np.argsort(available_radii_array)
        available_radii_array = available_radii_array[order]
        available_scores_array = available_scores_array[order]

        # 缺失距离仅在本方向上按距离线性补齐。
        raw_profile = np.interp(
            RADII,
            available_radii_array,
            available_scores_array,
        )
        raw_profile = np.maximum(raw_profile, 0.0)

        # 整理为整体不随距离增加的曲线。
        decreasing_profile = pava_decreasing(raw_profile)

        near_mask = (
            (RADII >= NEAR_MIN_CM)
            & (RADII <= NEAR_MAX_CM)
        )
        far_mask = (
            (RADII >= FAR_MIN_CM)
            & (RADII <= FAR_MAX_CM)
        )

        near_score = float(np.mean(raw_profile[near_mask]))
        far_score = float(np.mean(raw_profile[far_mask]))

        if near_score > EPS:
            near_far_contrast = np.clip(
                (near_score - far_score) / (near_score + EPS),
                0.0,
                1.0,
            )
        else:
            near_far_contrast = 0.0

        radial_mask = RADII > 0
        radial_corr = safe_correlation(
            RADII[radial_mask],
            raw_profile[radial_mask],
        )
        decay_consistency = float(np.clip(-radial_corr, 0.0, 1.0))

        # 方向门控：
        # 近场必须高于远场，同时距离衰减越清楚，保留比例越高。
        direction_gate = (
            near_far_contrast
            * (0.5 + 0.5 * decay_consistency)
        )

        shape_profile = decreasing_profile.copy()

        # 中心点是所有方向共享的，不对中心乘方向门控。
        shape_profile[0] = center_score
        shape_profile[1:] *= direction_gate
        shape_profile = np.maximum(shape_profile, 0.0)

        raw_profiles[direction_index, :] = raw_profile
        shape_profiles[direction_index, :] = shape_profile

        near_shape_mask = (
            (RADII >= 5)
            & (RADII <= 25)
        )
        near_radii = RADII[near_shape_mask]
        near_values = shape_profile[near_shape_mask]

        radial_weights = np.exp(-near_radii / 25.0)
        direction_strength = float(
            np.average(
                near_values,
                weights=radial_weights,
            )
        )

        direction_rows.append({
            "direction": direction,
            "direction_label": DIRECTION_LABELS[direction],
            "angle_deg": float(np.degrees(direction_angles[direction])),
            "near_score": near_score,
            "far_score": far_score,
            "near_far_contrast": float(near_far_contrast),
            "radial_correlation": float(radial_corr),
            "decay_consistency": float(decay_consistency),
            "direction_gate": float(direction_gate),
            "direction_strength": direction_strength,
        })

    return raw_profiles, shape_profiles, direction_rows


# ============================================================
# 8. 形态插值
# ============================================================

def interpolate_directional_shape_to_xy(
    shape_profiles: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    先在“角度-距离”空间插值，再映射到X-Y平面。

    与直接对米字形测点做三角剖分不同，
    这种方式更符合当前数据本身的极坐标采样结构。
    """
    angles = np.asarray(
        [direction_angles[direction] for direction in DIRECTION_ORDER],
        dtype=np.float64,
    )

    order = np.argsort(angles)
    angles = angles[order]
    profiles = shape_profiles[order, :]

    # 周期延拓，让0度和360度连续。
    extended_angles = np.concatenate([
        angles,
        [angles[0] + 2.0 * np.pi],
    ])
    extended_profiles = np.vstack([
        profiles,
        profiles[0:1, :],
    ])

    interpolator = RegularGridInterpolator(
        (extended_angles, RADII),
        extended_profiles,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    axis = np.linspace(
        -GRID_LIMIT_CM,
        GRID_LIMIT_CM,
        GRID_SIZE,
    )
    grid_x, grid_y = np.meshgrid(axis, axis)

    grid_r = np.sqrt(grid_x**2 + grid_y**2)
    grid_theta = np.mod(
        np.arctan2(grid_y, grid_x),
        2.0 * np.pi,
    )

    query_points = np.column_stack([
        grid_theta.ravel(),
        grid_r.ravel(),
    ])

    grid_z = interpolator(query_points).reshape(grid_x.shape)
    grid_z = np.maximum(grid_z, 0.0)
    grid_z[grid_r > max(distances)] = np.nan

    return grid_x, grid_y, grid_z


# ============================================================
# 9. 绘图
# ============================================================

def draw_direction_labels(
    axis: plt.Axes,
    label_radius: Optional[float] = None,
) -> None:
    """在圆周外标出8个方向。"""
    if label_radius is None:
        label_radius = max(distances) + 3.0

    for direction in DIRECTION_ORDER:
        angle = direction_angles[direction]
        x = label_radius * np.cos(angle)
        y = label_radius * np.sin(angle)

        axis.text(
            x,
            y,
            DIRECTION_LABELS[direction],
            ha="center",
            va="center",
            fontsize=8,
        )


def _plot_profile_heatmap(
    result: dict,
    profiles: np.ndarray,
    global_vmax: float,
    display_radius_cm: float,
    title_prefix: str,
    filename_prefix: str,
    status_text: str = "",
) -> None:
    """
    通用热力图函数。

    profiles可以是：
      - raw_profiles：原始测点稳定分数，不做PAVA整理；
      - shape_profiles：经过PAVA和方向门控后的辅助形态。
    """
    grid_x, grid_y, grid_z = interpolate_directional_shape_to_xy(
        profiles
    )

    display_radius_cm = float(display_radius_cm)
    grid_r = np.sqrt(grid_x**2 + grid_y**2)
    grid_z = grid_z.copy()
    grid_z[grid_r > display_radius_cm] = np.nan

    margin = 3.0
    axis_limit = display_radius_cm + margin

    figure, axis = plt.subplots(
        figsize=(8.2, 7.2),
        dpi=140,
    )

    cmap = plt.get_cmap(HEATMAP_CMAP).copy()
    cmap.set_bad(color="white", alpha=0.0)

    image = axis.imshow(
        np.ma.masked_invalid(grid_z),
        extent=(
            -GRID_LIMIT_CM,
            GRID_LIMIT_CM,
            -GRID_LIMIT_CM,
            GRID_LIMIT_CM,
        ),
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=global_vmax,
        interpolation="bilinear",
        aspect="equal",
    )

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(
        "Stable excess score (dB-weighted, fixed scale)"
    )

    # 只绘制当前显示半径以内的实测点。
    measured_x = []
    measured_y = []
    for record in result["point_records"]:
        if float(record["distance_cm"]) <= display_radius_cm + EPS:
            measured_x.append(float(record["x_cm"]))
            measured_y.append(float(record["y_cm"]))

    if measured_x:
        axis.scatter(
            np.asarray(measured_x, dtype=np.float64),
            np.asarray(measured_y, dtype=np.float64),
            s=12,
            c="white",
            edgecolors="black",
            linewidths=0.45,
            alpha=0.80,
            zorder=3,
        )

    axis.scatter(
        0.0,
        0.0,
        marker="*",
        s=190,
        c="red",
        edgecolors="yellow",
        linewidths=1.0,
        zorder=5,
    )

    draw_direction_labels(
        axis,
        label_radius=display_radius_cm + 1.5,
    )

    title_lines = [
        title_prefix,
        (
            f"{result['time_folder']} | Center {result['center_id']} | "
            f"[{result['segment_start_second']:.0f}, "
            f"{result['segment_end_second']:.0f}) s | "
            f"{FREQ_LOW/1000:.0f}-{FREQ_HIGH/1000:.0f} kHz"
        ),
    ]

    if status_text:
        title_lines.append(status_text)

    axis.set_title(
        "\n".join(title_lines),
        fontsize=10.5,
        fontweight="bold",
    )
    axis.set_xlabel("X distance (cm)")
    axis.set_ylabel("Y distance (cm)")
    axis.set_xlim(-axis_limit, axis_limit)
    axis.set_ylim(-axis_limit, axis_limit)
    axis.set_aspect("equal")
    axis.grid(True, linestyle="--", alpha=0.30)

    save_path = os.path.join(
        result["result_dir"],
        f"{filename_prefix}_{result['time_folder']}"
        f"_center_{result['center_id']}.png",
    )

    figure.tight_layout()
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)

    print(f"  热力图已保存：{save_path}")


def plot_absolute_morphology(
    result: dict,
    global_vmax: float,
) -> None:
    """
    主图：使用raw_profiles，只显示0~NEAR_DISPLAY_MAX_CM。

    这张图不再使用PAVA整理结果，是判断真实近场形态的主图。
    为兼容旧使用习惯，文件名仍保留absolute_morphology前缀。
    """
    _plot_profile_heatmap(
        result=result,
        profiles=result["raw_profiles"],
        global_vmax=global_vmax,
        display_radius_cm=NEAR_DISPLAY_MAX_CM,
        title_prefix="RAW Near-field Morphology (Main Result)",
        filename_prefix="absolute_morphology",
        status_text=(
            f"Background: {result['background_status']} | "
            f"Outer positive fraction: "
            f"{result['outer_positive_fraction']:.2f}"
        ),
    )


def plot_raw_full_diagnostic(
    result: dict,
    global_vmax: float,
) -> None:
    """
    原始全范围诊断图：显示0~40cm，包括30~40cm外围背景点。
    这张图用于检查外围背景是否出现大片正分数。
    """
    _plot_profile_heatmap(
        result=result,
        profiles=result["raw_profiles"],
        global_vmax=global_vmax,
        display_radius_cm=float(max(distances)),
        title_prefix="RAW Full-range Background Diagnostic",
        filename_prefix="raw_full_background_diagnostic",
        status_text=(
            f"Background: {result['background_status']} | "
            f"Outer positive: {result['outer_positive_point_count']}/"
            f"{result['outer_point_count']}"
        ),
    )


def plot_adjusted_morphology(
    result: dict,
    global_vmax: float,
) -> None:
    """
    整理后辅助图：使用shape_profiles。

    该图经过PAVA径向整理和方向门控，只用于观察理想化衰减形态，
    不能替代原始近场主图。
    """
    _plot_profile_heatmap(
        result=result,
        profiles=result["shape_profiles"],
        global_vmax=global_vmax,
        display_radius_cm=float(max(distances)),
        title_prefix="ADJUSTED Morphology (Auxiliary Only)",
        filename_prefix="adjusted_morphology",
        status_text="PAVA + direction gate; do not treat as raw measurement",
    )


def plot_normalized_morphology(
    result: dict,
) -> None:
    """
    可选图：每张图归一化到0~1，只用于看形状。
    不可用于比较真泄漏和假泄漏的绝对强弱。
    """
    grid_x, grid_y, grid_z = interpolate_directional_shape_to_xy(
        result["shape_profiles"]
    )

    finite_values = grid_z[np.isfinite(grid_z)]
    sample_max = (
        float(np.max(finite_values))
        if finite_values.size > 0
        else 0.0
    )

    if sample_max > 0:
        normalized = grid_z / sample_max
    else:
        normalized = grid_z.copy()

    figure, axis = plt.subplots(
        figsize=(8.2, 7.2),
        dpi=140,
    )

    cmap = plt.get_cmap(HEATMAP_CMAP).copy()
    cmap.set_bad(color="white", alpha=0.0)

    image = axis.imshow(
        np.ma.masked_invalid(normalized),
        extent=(
            -GRID_LIMIT_CM,
            GRID_LIMIT_CM,
            -GRID_LIMIT_CM,
            GRID_LIMIT_CM,
        ),
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear",
        aspect="equal",
    )

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Normalized shape only (0-1)")

    axis.scatter(
        0.0,
        0.0,
        marker="*",
        s=190,
        c="red",
        edgecolors="yellow",
        linewidths=1.0,
        zorder=5,
    )

    draw_direction_labels(axis)

    axis.set_title(
        "Normalized Shape Only - Do Not Compare Absolute Strength\n"
        f"{result['time_folder']} | Center {result['center_id']}",
        fontsize=11,
        fontweight="bold",
    )
    axis.set_xlabel("X distance (cm)")
    axis.set_ylabel("Y distance (cm)")
    axis.set_xlim(-GRID_LIMIT_CM, GRID_LIMIT_CM)
    axis.set_ylim(-GRID_LIMIT_CM, GRID_LIMIT_CM)
    axis.set_aspect("equal")
    axis.grid(True, linestyle="--", alpha=0.30)

    save_path = os.path.join(
        result["result_dir"],
        f"normalized_shape_{result['time_folder']}"
        f"_center_{result['center_id']}.png",
    )

    figure.tight_layout()
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)

    print(f"  归一化形态图已保存：{save_path}")


def plot_direction_shape(
    result: dict,
    global_direction_max: float,
) -> None:
    """
    多瓣方向图：
    不强制一个方向，可同时出现多个方向峰。
    """
    rows_by_direction = {
        row["direction"]: row
        for row in result["direction_rows"]
    }

    angles = np.asarray(
        [direction_angles[direction] for direction in DIRECTION_ORDER],
        dtype=np.float64,
    )
    strengths = np.asarray(
        [
            rows_by_direction.get(
                direction,
                {"direction_strength": 0.0},
            )["direction_strength"]
            for direction in DIRECTION_ORDER
        ],
        dtype=np.float64,
    )

    plot_angles = np.concatenate([angles, [angles[0]]])
    plot_strengths = np.concatenate([strengths, [strengths[0]]])

    figure = plt.figure(figsize=(7.2, 7.0), dpi=140)
    axis = figure.add_subplot(111, projection="polar")

    axis.plot(
        plot_angles,
        plot_strengths,
        linewidth=2.0,
    )
    axis.fill(
        plot_angles,
        plot_strengths,
        alpha=0.30,
    )

    axis.scatter(
        angles,
        strengths,
        s=32,
        zorder=3,
    )

    axis.set_theta_zero_location("E")
    axis.set_theta_direction(1)
    axis.set_xticks(angles)
    axis.set_xticklabels(
        [DIRECTION_LABELS[d] for d in DIRECTION_ORDER],
        fontsize=8,
    )
    axis.set_ylim(0.0, global_direction_max)
    axis.grid(True, alpha=0.35)

    axis.set_title(
        "Multi-direction Leak Shape\n"
        f"{result['time_folder']} | Center {result['center_id']}\n"
        "Multiple lobes are allowed",
        fontsize=11,
        fontweight="bold",
        pad=20,
    )

    save_path = os.path.join(
        result["result_dir"],
        f"direction_shape_{result['time_folder']}"
        f"_center_{result['center_id']}.png",
    )

    figure.tight_layout()
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)

    print(f"  多方向形态图已保存：{save_path}")


# ============================================================
# 10. CSV输出
# ============================================================

def save_radial_profile_csv(
    result: dict,
) -> None:
    save_path = os.path.join(
        result["result_dir"],
        f"radial_profile_{result['time_folder']}"
        f"_center_{result['center_id']}.csv",
    )

    rows_by_direction = {
        row["direction"]: row
        for row in result["direction_rows"]
    }

    fieldnames = [
        "time_folder",
        "center_id",
        "direction",
        "direction_label",
        "angle_deg",
        "distance_cm",
        "raw_stable_score",
        "shape_score",
        "near_score",
        "far_score",
        "near_far_contrast",
        "radial_correlation",
        "decay_consistency",
        "direction_gate",
        "direction_strength",
    ]

    with open(
        save_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for direction_index, direction in enumerate(DIRECTION_ORDER):
            direction_row = rows_by_direction.get(direction, {})

            for radius_index, radius in enumerate(RADII):
                writer.writerow({
                    "time_folder": result["time_folder"],
                    "center_id": result["center_id"],
                    "direction": direction,
                    "direction_label": DIRECTION_LABELS[direction],
                    "angle_deg": float(np.degrees(direction_angles[direction])),
                    "distance_cm": float(radius),
                    "raw_stable_score": float(
                        result["raw_profiles"][
                            direction_index,
                            radius_index,
                        ]
                    ),
                    "shape_score": float(
                        result["shape_profiles"][
                            direction_index,
                            radius_index,
                        ]
                    ),
                    "near_score": direction_row.get("near_score", 0.0),
                    "far_score": direction_row.get("far_score", 0.0),
                    "near_far_contrast": direction_row.get(
                        "near_far_contrast",
                        0.0,
                    ),
                    "radial_correlation": direction_row.get(
                        "radial_correlation",
                        0.0,
                    ),
                    "decay_consistency": direction_row.get(
                        "decay_consistency",
                        0.0,
                    ),
                    "direction_gate": direction_row.get(
                        "direction_gate",
                        0.0,
                    ),
                    "direction_strength": direction_row.get(
                        "direction_strength",
                        0.0,
                    ),
                })

    print(f"  径向结果表已保存：{save_path}")


def save_point_score_csv(
    result: dict,
) -> None:
    save_path = os.path.join(
        result["result_dir"],
        f"point_score_{result['time_folder']}"
        f"_center_{result['center_id']}.csv",
    )

    fieldnames = [
        "time_folder",
        "center_id",
        "point_type",
        "direction",
        "distance_cm",
        "x_cm",
        "y_cm",
        "stable_score",
        "persistence",
        "is_outer_background",
        "file_name",
    ]

    with open(
        save_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for record, score, point_persistence in zip(
            result["point_records"],
            result["point_scores"],
            result["persistence"],
        ):
            writer.writerow({
                "time_folder": result["time_folder"],
                "center_id": result["center_id"],
                "point_type": record["point_type"],
                "direction": record["direction"],
                "distance_cm": record["distance_cm"],
                "x_cm": record["x_cm"],
                "y_cm": record["y_cm"],
                "stable_score": float(score),
                "persistence": float(point_persistence),
                "is_outer_background": bool(
                    record["point_type"] == "offset"
                    and float(record["distance_cm"])
                    >= OUTER_BACKGROUND_MIN_CM
                ),
                "file_name": os.path.basename(record["file_path"]),
            })

    print(f"  测点分数表已保存：{save_path}")


def save_summary_all(
    all_results: List[dict],
    global_vmax: float,
    global_direction_max: float,
) -> None:
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)

    save_path = os.path.join(
        OUTPUT_ROOT_DIR,
        "summary_all.csv",
    )

    fieldnames = [
        "time_folder",
        "center_id",
        "time_index",
        "segment_start_second",
        "segment_end_second",
        "num_points",
        "num_frames",
        "center_score",
        "center_persistence",
        "outer_point_count",
        "outer_positive_point_count",
        "outer_positive_fraction",
        "outer_score_median",
        "outer_score_mean",
        "background_status",
        "max_direction_strength",
        "mean_direction_strength",
        "directionality_ratio",
        "dominant_direction_count",
        "dominant_directions",
        "global_heatmap_vmax",
        "global_direction_rmax",
    ]

    with open(
        save_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in all_results:
            strengths = np.asarray(
                [
                    row["direction_strength"]
                    for row in result["direction_rows"]
                ],
                dtype=np.float64,
            )

            max_strength = (
                float(np.max(strengths))
                if strengths.size > 0
                else 0.0
            )
            mean_strength = (
                float(np.mean(strengths))
                if strengths.size > 0
                else 0.0
            )

            directionality_ratio = (
                max_strength / (mean_strength + EPS)
                if max_strength > 0
                else 0.0
            )

            writer.writerow({
                "time_folder": result["time_folder"],
                "center_id": result["center_id"],
                "time_index": result["time_index"],
                "segment_start_second": result["segment_start_second"],
                "segment_end_second": result["segment_end_second"],
                "num_points": len(result["point_records"]),
                "num_frames": result["num_frames"],
                "center_score": result["center_score"],
                "center_persistence": result["center_persistence"],
                "outer_point_count": result["outer_point_count"],
                "outer_positive_point_count": result[
                    "outer_positive_point_count"
                ],
                "outer_positive_fraction": result[
                    "outer_positive_fraction"
                ],
                "outer_score_median": result["outer_score_median"],
                "outer_score_mean": result["outer_score_mean"],
                "background_status": result["background_status"],
                "max_direction_strength": max_strength,
                "mean_direction_strength": mean_strength,
                "directionality_ratio": directionality_ratio,
                "dominant_direction_count": len(
                    result["dominant_directions"]
                ),
                "dominant_directions": result[
                    "dominant_directions_text"
                ],
                "global_heatmap_vmax": global_vmax,
                "global_direction_rmax": global_direction_max,
            })

    print(f"\n全部样本摘要已保存：{save_path}")


# ============================================================
# 11. 计算单个样本
# ============================================================

def compute_single_sample(
    time_folder: str,
    center_id: str,
    center_index: Dict[str, List[str]],
    offset_index: Dict[OffsetKey, List[str]],
    result_dir: str,
) -> Optional[dict]:
    try:
        time_index, segment_start_second, segment_end_second = (
            get_time_slice_from_center_id(center_id)
        )
    except ValueError as exc:
        print(f"  错误：{exc}")
        return None

    print(
        f"\n处理：{time_folder} | center={center_id} | "
        f"[{segment_start_second:.0f}, "
        f"{segment_end_second:.0f})秒"
    )

    point_records = build_point_records(
        center_id,
        center_index,
        offset_index,
    )

    if len(point_records) < 10:
        print("  错误：有效测点数量太少")
        return None

    valid_records, energy_db_matrix = load_all_point_frame_energies(
        point_records,
        segment_start_second,
        segment_end_second,
    )

    if energy_db_matrix is None or len(valid_records) < 10:
        print("  错误：无法建立完整短时频带能量矩阵")
        return None

    try:
        point_scores, persistence, frame_thresholds = (
            compute_stable_point_scores(
                valid_records,
                energy_db_matrix,
            )
        )

        raw_profiles, shape_profiles, direction_rows = (
            build_directional_profiles(
                valid_records,
                point_scores,
                persistence,
            )
        )
    except Exception as exc:
        print(f"  错误：形态计算失败：{exc}")
        return None

    center_indices = [
        index
        for index, record in enumerate(valid_records)
        if record["point_type"] == "center"
    ]

    if not center_indices:
        print("  错误：缺少有效中心点")
        return None

    center_index_value = center_indices[0]
    center_score = float(point_scores[center_index_value])
    center_persistence = float(persistence[center_index_value])

    # 外围背景诊断：30~40cm的点只用于判断背景是否干净。
    outer_indices = [
        index
        for index, record in enumerate(valid_records)
        if (
            record["point_type"] == "offset"
            and float(record["distance_cm"])
            >= OUTER_BACKGROUND_MIN_CM
        )
    ]

    outer_scores = np.asarray(
        [point_scores[index] for index in outer_indices],
        dtype=np.float64,
    )

    outer_point_count = int(outer_scores.size)
    outer_positive_point_count = int(np.sum(outer_scores > EPS))
    outer_positive_fraction = (
        float(outer_positive_point_count / outer_point_count)
        if outer_point_count > 0
        else 1.0
    )
    outer_score_median = (
        float(np.median(outer_scores))
        if outer_point_count > 0
        else 0.0
    )
    outer_score_mean = (
        float(np.mean(outer_scores))
        if outer_point_count > 0
        else 0.0
    )

    if outer_point_count < 8:
        background_status = "BACKGROUND_POINTS_INSUFFICIENT"
    elif outer_positive_fraction > OUTER_POSITIVE_FRACTION_WARN:
        background_status = "BACKGROUND_NOT_CLEAN"
    else:
        background_status = "CLEAN"

    strengths = np.asarray(
        [row["direction_strength"] for row in direction_rows],
        dtype=np.float64,
    )

    if strengths.size > 0 and np.max(strengths) > 0:
        dominant_threshold = (
            DOMINANT_DIRECTION_RATIO * float(np.max(strengths))
        )
        dominant_directions = [
            row["direction"]
            for row in direction_rows
            if row["direction_strength"] >= dominant_threshold
            and row["direction_strength"] > 0
        ]
    else:
        dominant_directions = []

    if dominant_directions:
        dominant_directions_text = ", ".join(
            DIRECTION_LABELS[direction]
            for direction in dominant_directions
        )
    else:
        dominant_directions_text = "None"

    result = {
        "time_folder": time_folder,
        "center_id": center_id,
        "time_index": time_index,
        "segment_start_second": segment_start_second,
        "segment_end_second": segment_end_second,
        "result_dir": result_dir,
        "point_records": valid_records,
        "energy_db_matrix": energy_db_matrix,
        "num_frames": int(energy_db_matrix.shape[1]),
        "point_scores": point_scores,
        "persistence": persistence,
        "frame_thresholds": frame_thresholds,
        "raw_profiles": raw_profiles,
        "shape_profiles": shape_profiles,
        "direction_rows": direction_rows,
        "center_score": center_score,
        "center_persistence": center_persistence,
        "outer_point_count": outer_point_count,
        "outer_positive_point_count": outer_positive_point_count,
        "outer_positive_fraction": outer_positive_fraction,
        "outer_score_median": outer_score_median,
        "outer_score_mean": outer_score_mean,
        "background_status": background_status,
        "dominant_directions": dominant_directions,
        "dominant_directions_text": dominant_directions_text,
    }

    print(f"  有效测点：{len(valid_records)}")
    print(f"  短时帧数：{energy_db_matrix.shape[1]}")
    print(f"  中心稳定分数：{center_score:.4f}")
    print(f"  中心持续率：{center_persistence:.3f}")
    print(
        f"  外围正分数点：{outer_positive_point_count}/"
        f"{outer_point_count}，比例={outer_positive_fraction:.3f}"
    )
    print(f"  外围背景状态：{background_status}")
    print(f"  主要方向：{dominant_directions_text}")

    return result


# ============================================================
# 12. 两遍处理：先算全部样本，再统一色标绘图
# ============================================================

def collect_all_results() -> List[dict]:
    all_results: List[dict] = []

    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)

    for time_folder in time_folders:
        print("\n" + "=" * 90)
        print(f"扫描文件夹：{time_folder}")
        print("=" * 90)

        center_data_dir = os.path.join(
            center_root_dir,
            time_folder,
        )
        offset_data_dir = os.path.join(
            offset_root_dir,
            time_folder,
        )

        if not os.path.isdir(center_data_dir):
            print(f"警告：中心目录不存在：{center_data_dir}")
            continue

        if not os.path.isdir(offset_data_dir):
            print(f"警告：偏移目录不存在：{offset_data_dir}")
            continue

        result_dir = os.path.join(
            OUTPUT_ROOT_DIR,
            time_folder,
        )
        os.makedirs(result_dir, exist_ok=True)

        center_index = build_center_file_index(center_data_dir)
        offset_index = build_offset_file_index(offset_data_dir)

        center_ids = sorted(
            center_index.keys(),
            key=center_id_sort_key,
        )

        print(f"检测到中心点编号：{center_ids}")
        print(f"偏移索引组数：{len(offset_index)}")

        for center_id in center_ids:
            try:
                result = compute_single_sample(
                    time_folder,
                    center_id,
                    center_index,
                    offset_index,
                    result_dir,
                )
            except Exception as exc:
                print(
                    f"  未预期错误：{time_folder} / "
                    f"{center_id}：{exc}"
                )
                import traceback
                traceback.print_exc()
                continue

            if result is not None:
                all_results.append(result)

    return all_results


def determine_global_scales(
    all_results: List[dict],
) -> Tuple[float, float]:
    """统一确定所有热力图和方向图的显示上限。"""
    heatmap_values: List[np.ndarray] = []
    direction_values: List[float] = []

    for result in all_results:
        profile_values = result["raw_profiles"].ravel()
        profile_values = profile_values[
            np.isfinite(profile_values)
            & (profile_values > 0)
        ]

        if profile_values.size > 0:
            heatmap_values.append(profile_values)

        for row in result["direction_rows"]:
            value = float(row["direction_strength"])
            if np.isfinite(value) and value > 0:
                direction_values.append(value)

    if heatmap_values:
        all_heatmap_values = np.concatenate(heatmap_values)
        global_vmax = float(np.percentile(
            all_heatmap_values,
            GLOBAL_COLOR_PERCENTILE,
        ))
    else:
        global_vmax = 1.0

    if direction_values:
        global_direction_max = float(np.percentile(
            np.asarray(direction_values, dtype=np.float64),
            GLOBAL_DIRECTION_PERCENTILE,
        ))
    else:
        global_direction_max = 1.0

    global_vmax = max(global_vmax, EPS)
    global_direction_max = max(global_direction_max, EPS)

    return global_vmax, global_direction_max


def save_scale_info(
    global_vmax: float,
    global_direction_max: float,
) -> None:
    save_path = os.path.join(
        OUTPUT_ROOT_DIR,
        "global_scale.txt",
    )

    with open(save_path, "w", encoding="utf-8") as file:
        file.write(
            "All RAW and adjusted heatmaps use the same RAW-derived scale.\n"
        )
        file.write(f"heatmap_vmin = 0\n")
        file.write(f"heatmap_vmax = {global_vmax:.12g}\n")
        file.write(f"direction_rmax = {global_direction_max:.12g}\n")
        file.write(
            f"frequency_band_hz = {FREQ_LOW}-{FREQ_HIGH}\n"
        )
        file.write(
            f"global_color_percentile = "
            f"{GLOBAL_COLOR_PERCENTILE}\n"
        )

    print(f"统一色标信息已保存：{save_path}")


def render_all_results(
    all_results: List[dict],
    global_vmax: float,
    global_direction_max: float,
) -> None:
    for result in all_results:
        save_point_score_csv(result)
        save_radial_profile_csv(result)

        if OUTPUT_RAW_NEAR_HEATMAP:
            plot_absolute_morphology(
                result,
                global_vmax,
            )

        if OUTPUT_RAW_FULL_DIAGNOSTIC:
            plot_raw_full_diagnostic(
                result,
                global_vmax,
            )

        if OUTPUT_ADJUSTED_HEATMAP:
            plot_adjusted_morphology(
                result,
                global_vmax,
            )

        if OUTPUT_NORMALIZED_SHAPE:
            plot_normalized_morphology(result)

        if OUTPUT_DIRECTION_SHAPE:
            plot_direction_shape(
                result,
                global_direction_max,
            )


# ============================================================
# 13. 配置检查和主程序
# ============================================================

def validate_config() -> bool:
    valid = True

    if not time_folders:
        print("配置错误：time_folders不能为空")
        valid = False

    if FREQ_LOW < 0 or FREQ_HIGH <= FREQ_LOW:
        print("配置错误：FREQ_HIGH必须大于FREQ_LOW")
        valid = False

    if FRAME_SECONDS <= 0 or HOP_SECONDS <= 0:
        print("配置错误：FRAME_SECONDS和HOP_SECONDS必须大于0")
        valid = False

    if WELCH_NPERSEG < 16:
        print("配置错误：WELCH_NPERSEG不能小于16")
        valid = False

    if not distances:
        print("配置错误：distances不能为空")
        valid = False

    if OUTER_BACKGROUND_MIN_CM > max(distances):
        print(
            "配置错误：OUTER_BACKGROUND_MIN_CM大于最大测量距离"
        )
        valid = False

    if not (0.0 <= MIN_PERSISTENCE <= 1.0):
        print("配置错误：MIN_PERSISTENCE必须在0~1之间")
        valid = False

    if not (0.0 < DOMINANT_DIRECTION_RATIO <= 1.0):
        print(
            "配置错误：DOMINANT_DIRECTION_RATIO必须在0~1之间"
        )
        valid = False

    if not (0.0 < NEAR_DISPLAY_MAX_CM <= max(distances)):
        print(
            "配置错误：NEAR_DISPLAY_MAX_CM必须大于0且不超过最大距离"
        )
        valid = False

    if not (0.0 <= OUTER_POSITIVE_FRACTION_WARN <= 1.0):
        print(
            "配置错误：OUTER_POSITIVE_FRACTION_WARN必须在0~1之间"
        )
        valid = False

    return valid


def main() -> None:
    print("=" * 90)
    print("Directional Acoustic Leak Morphology")
    print("=" * 90)
    print("这版输出三类热力图：")
    print("  1. absolute_morphology：原始0~25cm近场主图")
    print("  2. raw_full_background_diagnostic：原始0~40cm背景诊断图")
    print("  3. adjusted_morphology：PAVA整理后的辅助图")
    print("主图不再使用PAVA整理结果，避免把数值摊成整片绿色。")
    print(
        f"频段：{FREQ_LOW/1000:.1f}-"
        f"{FREQ_HIGH/1000:.1f} kHz"
    )
    print(
        f"短时帧：{FRAME_SECONDS:.2f}s，"
        f"步长：{HOP_SECONDS:.2f}s"
    )
    print(
        f"外围背景：distance >= "
        f"{OUTER_BACKGROUND_MIN_CM} cm"
    )
    print(
        "重要：真泄漏和假泄漏文件夹要一次性全部写入"
        "time_folders，统一色标才有比较意义。"
    )

    if not validate_config():
        print("配置检查失败，程序停止")
        return

    all_results = collect_all_results()

    if not all_results:
        print("\n没有得到任何有效结果")
        return

    global_vmax, global_direction_max = determine_global_scales(
        all_results
    )

    print("\n" + "=" * 90)
    print("统一显示尺度")
    print("=" * 90)
    print(f"热力图统一vmax：{global_vmax:.6f}")
    print(f"方向图统一rmax：{global_direction_max:.6f}")

    save_scale_info(
        global_vmax,
        global_direction_max,
    )

    render_all_results(
        all_results,
        global_vmax,
        global_direction_max,
    )

    save_summary_all(
        all_results,
        global_vmax,
        global_direction_max,
    )

    print("\n" + "=" * 90)
    print("全部处理完成")
    print(f"输出目录：{os.path.abspath(OUTPUT_ROOT_DIR)}")
    print("=" * 90)


if __name__ == "__main__":
    main()
