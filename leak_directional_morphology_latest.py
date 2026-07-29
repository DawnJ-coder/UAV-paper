# -*- coding: utf-8 -*-
"""
八方向泄漏径向形态可视化（最新稳健版）
========================================

适用数据
--------
- 1 个中心点 WAV；
- 8 个方向；
- 每个方向距离为 5, 10, ..., 40 cm；
- center_id 最后一段编号决定读取原始 WAV 的第几秒。

核心原则
--------
1. 每个方向使用本方向 35 cm、40 cm 测点的中位数作为远端背景；
2. 50~70 kHz 拆成多个 2 kHz 子频带，不再一次积分成一个总能量；
3. 用远端背景的时间波动和空间差异估计噪声尺度；
4. 保留每帧的正负差异，先在时间上取中位数，最后才进行阈值截断；
5. 只保留近场稳定增强、且连续出现的子频带；
6. 主图使用真实测量单元的扇区图，不做 PAVA，不做角度/距离连续插值；
7. 该图表示“相对于本方向远端背景的径向超量形态”，不是绝对二维声压图。

主要输出
--------
1. absolute_sector_morphology_*.png
   所有样本统一色标，用于比较稳定超量证据的强弱和空间位置。

2. shape_only_sector_morphology_*.png
   单样本归一化，只用于观察形状，不用于比较不同样本的绝对强弱。

3. radial_profiles_*.png
   8 个方向各自的径向曲线，不强制随距离下降。

4. frequency_evidence_*.png
   每个方向在各子频带上的近场证据及最终保留频带。

5. point_score_*.csv / direction_frequency_evidence_*.csv / summary_all.csv
   保存全部计算结果，便于进一步真假泄漏分析。
"""

from __future__ import annotations

import csv
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Circle, Wedge
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal


# ============================================================
# 1. 用户配置
# ============================================================

# 真泄漏和假泄漏样本应一起放进来，以便主图使用统一色标。
time_folders = [
    "HM20260702_111044.ld",
]

center_root_dir = r"D:\gas\beamform_results_sh"
offset_root_dir = r"D:\gas\beamform_results_offset_multiple_sh"

OUTPUT_ROOT_DIR = "results_directional_morphology_latest"
RECURSIVE_SEARCH = False
EXPECTED_SAMPLE_RATE = 192000

# center 00_00 读取 [0, 1) 秒；00_01 读取 [1, 2) 秒。
TIME_SLICE_SECONDS = 1.0
STRICT_TIME_SLICE = True

# 总频段与子频带宽度。
FREQ_LOW = 50000.0
FREQ_HIGH = 70000.0
SUBBAND_WIDTH_HZ = 2000.0

# 短时分析参数。
FRAME_SECONDS = 0.10
HOP_SECONDS = 0.05
WELCH_NPERSEG = 4096
WELCH_OVERLAP_RATIO = 0.50

# 8 个方向。
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

# 距离，单位 cm。
distances = [5, 10, 15, 20, 25, 30, 35, 40]
RADII = np.asarray(distances, dtype=np.float64)

# 每个方向优先使用这些远端距离作为本方向背景。
REMOTE_BACKGROUND_DISTANCES = [35, 40]
# 如果 35/40 cm 文件缺失，可退回使用不小于该距离的有效点。
REMOTE_BACKGROUND_FALLBACK_MIN_CM = 30
MIN_REMOTE_POINT_COUNT = 2

# 远端波动标准化保护，避免噪声尺度过小导致 z 值爆炸。
MIN_NOISE_SCALE_DB = 0.35

# 一帧超过背景多少个稳健波动尺度，视为该帧存在增强。
FRAME_ACTIVE_Z = 1.0

# 子频带筛选：近场范围。
NEAR_BAND_MIN_CM = 5
NEAR_BAND_MAX_CM = 20
# 一个子频带的近场时间中位数至少达到该 z 值。
MIN_BAND_NEAR_MEDIAN_Z = 0.80
# 近场点在该子频带中的有效帧比例至少达到该值。
MIN_BAND_NEAR_PERSISTENCE = 0.45
# 至少连续几个子频带满足条件才保留。
MIN_CONTIGUOUS_SUBBANDS = 2

# 单个测量单元最终显示阈值。
CELL_DISPLAY_THRESHOLD_Z = 0.80
# 持续率权重；0 表示不加权，0.5 表示乘以持续率平方根。
PERSISTENCE_POWER = 0.50

# 绘图设置。
HEATMAP_CMAP = "turbo"
NO_EVIDENCE_COLOR = "#111111"
MISSING_COLOR = "#5f6368"
SECTOR_EDGE_COLOR = "#d0d0d0"
SECTOR_EDGE_WIDTH = 0.45
ANGLE_CELL_WIDTH_DEG = 45.0
CENTER_CELL_OUTER_RADIUS_CM = 2.5
PLOT_LIMIT_CM = 43.0

# 统一色标使用全部正分数的该百分位，降低极端值影响。
GLOBAL_COLOR_PERCENTILE = 98.0
# 形状图是否输出。
OUTPUT_SHAPE_ONLY = True
OUTPUT_RADIAL_PROFILES = True
OUTPUT_FREQUENCY_EVIDENCE = True

EPS = 1e-12
PSD_EPS = 1e-30


# ============================================================
# 2. 数据结构与文件名解析
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


@dataclass(frozen=True)
class BandDefinition:
    low_hz: float
    high_hz: float

    @property
    def center_hz(self) -> float:
        return 0.5 * (self.low_hz + self.high_hz)

    @property
    def label(self) -> str:
        return f"{self.low_hz/1000:.0f}-{self.high_hz/1000:.0f} kHz"


def build_subbands() -> List[BandDefinition]:
    bands: List[BandDefinition] = []
    low = float(FREQ_LOW)
    while low < FREQ_HIGH - EPS:
        high = min(low + SUBBAND_WIDTH_HZ, FREQ_HIGH)
        if high > low + EPS:
            bands.append(BandDefinition(low, high))
        low = high
    return bands


def list_wav_files(folder: str) -> List[str]:
    pattern = os.path.join(folder, "**", "*.wav") if RECURSIVE_SEARCH else os.path.join(folder, "*.wav")
    return sorted(glob.glob(pattern, recursive=RECURSIVE_SEARCH))


def build_center_file_index(center_data_dir: str) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    for file_path in list_wav_files(center_data_dir):
        match = CENTER_FILE_REGEX.search(os.path.basename(file_path))
        if match:
            index.setdefault(match.group("center"), []).append(file_path)
    return {key: sorted(value) for key, value in index.items()}


def build_offset_file_index(offset_data_dir: str) -> Dict[OffsetKey, List[str]]:
    index: Dict[OffsetKey, List[str]] = {}
    for file_path in list_wav_files(offset_data_dir):
        match = OFFSET_FILE_REGEX.search(os.path.basename(file_path))
        if not match:
            continue
        key = (
            match.group("center"),
            int(match.group("distance")),
            match.group("direction").lower(),
        )
        index.setdefault(key, []).append(file_path)
    return {key: sorted(value) for key, value in index.items()}


def choose_first_file(files: Optional[List[str]], description: str) -> Optional[str]:
    if not files:
        return None
    if len(files) > 1:
        print(f"  警告：{description}匹配到 {len(files)} 个文件，使用排序后的第一个：")
        for file_path in files:
            print(f"    - {os.path.basename(file_path)}")
    return files[0]


def center_id_sort_key(center_id: str) -> Tuple[int, ...]:
    try:
        return tuple(int(part) for part in str(center_id).split("_"))
    except ValueError:
        return (10**9,)


def get_time_slice_from_center_id(center_id: str) -> Tuple[int, float, float]:
    parts = str(center_id).split("_")
    if len(parts) < 2 or not parts[-1].isdigit():
        raise ValueError(
            f"center_id={center_id!r} 不符合 00_00 格式，无法判断读取第几秒"
        )
    time_index = int(parts[-1])
    start_second = time_index * float(TIME_SLICE_SECONDS)
    end_second = start_second + float(TIME_SLICE_SECONDS)
    return time_index, start_second, end_second


# ============================================================
# 3. WAV 读取与短时子频带能量
# ============================================================


def convert_wav_to_float(y: np.ndarray) -> np.ndarray:
    if np.issubdtype(y.dtype, np.integer):
        info = np.iinfo(y.dtype)
        full_scale = float(max(abs(info.min), abs(info.max)))
        if full_scale <= 0:
            raise ValueError(f"无效整数 WAV 类型：{y.dtype}")
        return y.astype(np.float64) / full_scale
    if np.issubdtype(y.dtype, np.floating):
        return y.astype(np.float64)
    raise TypeError(f"不支持 WAV 类型：{y.dtype}")


def read_wav_segment(
    file_path: str,
    segment_start_second: float,
    segment_end_second: float,
) -> Tuple[Optional[int], Optional[np.ndarray]]:
    if not os.path.exists(file_path):
        print(f"  警告：文件不存在：{file_path}")
        return None, None

    try:
        sample_rate, y = wav.read(file_path)
    except Exception as exc:
        print(f"  错误：读取 WAV 失败：{file_path}\n    {exc}")
        return None, None

    if y.ndim > 1:
        y = y[:, 0]

    try:
        y = convert_wav_to_float(y)
    except Exception as exc:
        print(f"  错误：WAV 数值转换失败：{file_path}\n    {exc}")
        return None, None

    start_sample = int(round(segment_start_second * sample_rate))
    end_sample = int(round(segment_end_second * sample_rate))

    if start_sample >= y.size:
        print(f"  错误：{os.path.basename(file_path)} 时长不足，无法读取目标时间段")
        return None, None

    if end_sample > y.size:
        if STRICT_TIME_SLICE:
            print(f"  错误：{os.path.basename(file_path)} 目标一秒不完整，已跳过")
            return None, None
        end_sample = y.size

    segment = y[start_sample:end_sample]
    expected_samples = int(round((segment_end_second - segment_start_second) * sample_rate))
    if STRICT_TIME_SLICE and segment.size != expected_samples:
        print(
            f"  错误：{os.path.basename(file_path)} 切片长度={segment.size}，"
            f"期望={expected_samples}，已跳过"
        )
        return None, None

    segment = np.nan_to_num(segment, nan=0.0, posinf=0.0, neginf=0.0)
    segment = segment - np.mean(segment)

    if sample_rate != EXPECTED_SAMPLE_RATE:
        print(
            f"  提醒：{os.path.basename(file_path)} 采样率为 {sample_rate} Hz，"
            f"不是期望的 {EXPECTED_SAMPLE_RATE} Hz"
        )

    return int(sample_rate), segment


def integrate_spectrum(psd: np.ndarray, freqs: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(psd, freqs))
    return float(np.trapz(psd, freqs))


def compute_frame_subband_energy_db(
    y: np.ndarray,
    sample_rate: int,
    bands: Sequence[BandDefinition],
) -> Optional[np.ndarray]:
    """返回形状 [短时帧数, 子频带数] 的 dB 能量矩阵。"""
    frame_length = int(round(FRAME_SECONDS * sample_rate))
    hop_length = int(round(HOP_SECONDS * sample_rate))

    if frame_length < 16 or hop_length < 1 or y.size < frame_length:
        return None

    nyquist = sample_rate / 2.0
    if FREQ_LOW >= nyquist:
        print(f"  错误：奈奎斯特频率仅 {nyquist:.1f} Hz，无法分析当前频段")
        return None

    starts = np.arange(0, y.size - frame_length + 1, hop_length, dtype=int)
    if starts.size == 0:
        return None

    output_rows: List[np.ndarray] = []

    for start in starts:
        frame = y[start:start + frame_length]
        frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
        frame = frame - np.mean(frame)

        nperseg = min(WELCH_NPERSEG, frame.size)
        if nperseg < 16:
            continue
        noverlap = min(int(round(nperseg * WELCH_OVERLAP_RATIO)), nperseg - 1)

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
            print(f"  警告：短时 Welch 计算失败：{exc}")
            continue

        psd = np.maximum(np.nan_to_num(psd, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        band_values: List[float] = []

        for band in bands:
            high = min(band.high_hz, nyquist)
            mask = (freqs >= band.low_hz) & (freqs < high)
            if np.sum(mask) < 2:
                band_values.append(np.nan)
                continue
            energy = max(integrate_spectrum(psd[mask], freqs[mask]), 0.0)
            band_values.append(10.0 * np.log10(energy + PSD_EPS))

        output_rows.append(np.asarray(band_values, dtype=np.float64))

    if not output_rows:
        return None

    matrix = np.vstack(output_rows)
    if np.any(~np.isfinite(matrix)):
        return None
    return matrix


# ============================================================
# 4. 构建和读取测点
# ============================================================


def build_point_records(
    center_id: str,
    center_index: Dict[str, List[str]],
    offset_index: Dict[OffsetKey, List[str]],
) -> List[dict]:
    records: List[dict] = []

    center_file = choose_first_file(center_index.get(center_id), f"中心点 {center_id}")
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
            file_path = choose_first_file(
                offset_index.get((center_id, distance, direction)),
                f"中心点 {center_id}、{direction}、{distance} cm",
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


def load_all_point_energy_matrices(
    point_records: List[dict],
    segment_start_second: float,
    segment_end_second: float,
    bands: Sequence[BandDefinition],
) -> Tuple[List[dict], Optional[np.ndarray]]:
    valid_records: List[dict] = []
    matrices: List[np.ndarray] = []

    for record in point_records:
        sample_rate, y = read_wav_segment(
            record["file_path"],
            segment_start_second,
            segment_end_second,
        )
        if sample_rate is None or y is None:
            continue

        matrix = compute_frame_subband_energy_db(y, sample_rate, bands)
        if matrix is None or matrix.shape[0] < 3:
            print(f"  警告：有效短时帧不足：{os.path.basename(record['file_path'])}")
            continue

        valid_records.append(record)
        matrices.append(matrix)

    if not matrices:
        return [], None

    common_frames = min(matrix.shape[0] for matrix in matrices)
    if common_frames < 3:
        return [], None

    stacked = np.stack([matrix[:common_frames, :] for matrix in matrices], axis=0)
    return valid_records, stacked


# ============================================================
# 5. 稳健统计与频带连续性
# ============================================================


def robust_mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    median_value = float(np.median(values))
    mad = float(np.median(np.abs(values - median_value)))
    return 1.4826 * mad


def keep_contiguous_true_runs(mask: np.ndarray, min_length: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    output = np.zeros_like(mask)
    start: Optional[int] = None

    for index, value in enumerate(np.append(mask, False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= min_length:
                output[start:index] = True
            start = None
    return output


def find_record_index(
    point_records: Sequence[dict],
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


def choose_remote_indices(point_records: Sequence[dict], direction: str) -> List[int]:
    preferred: List[int] = []
    for distance in REMOTE_BACKGROUND_DISTANCES:
        index = find_record_index(point_records, direction, distance)
        if index is not None:
            preferred.append(index)

    if len(preferred) >= MIN_REMOTE_POINT_COUNT:
        return preferred

    fallback = [
        index
        for index, record in enumerate(point_records)
        if (
            record["direction"] == direction
            and record["point_type"] == "offset"
            and float(record["distance_cm"]) >= REMOTE_BACKGROUND_FALLBACK_MIN_CM
        )
    ]
    fallback.sort(key=lambda idx: float(point_records[idx]["distance_cm"]))
    return fallback


def calculate_noise_scale_db(
    far_data: np.ndarray,
    background_db: np.ndarray,
) -> np.ndarray:
    """
    far_data: [远端点数, 帧数, 子频带数]
    background_db: [帧数, 子频带数]

    同时考虑：
    - 远端背景随时间的波动；
    - 同一时刻远端点之间的空间差异。
    """
    n_bands = far_data.shape[2]
    scales = np.zeros(n_bands, dtype=np.float64)

    for band_index in range(n_bands):
        temporal_scale = robust_mad(background_db[:, band_index])
        spatial_residual = (
            far_data[:, :, band_index]
            - background_db[None, :, band_index]
        ).ravel()
        spatial_scale = robust_mad(spatial_residual)
        combined = float(np.sqrt(temporal_scale**2 + spatial_scale**2))
        scales[band_index] = max(combined, MIN_NOISE_SCALE_DB)

    return scales


# ============================================================
# 6. 每方向远端背景、频带筛选与测点分数
# ============================================================


def aggregate_cell_score(
    stable_z: np.ndarray,
    persistence: np.ndarray,
    valid_band_mask: np.ndarray,
) -> Tuple[float, float, float]:
    if not np.any(valid_band_mask):
        return 0.0, 0.0, 0.0

    aggregate_z = float(np.median(stable_z[valid_band_mask]))
    aggregate_persistence = float(np.median(persistence[valid_band_mask]))
    score = max(aggregate_z - CELL_DISPLAY_THRESHOLD_Z, 0.0)
    score *= float(np.power(np.clip(aggregate_persistence, 0.0, 1.0), PERSISTENCE_POWER))
    return aggregate_z, aggregate_persistence, max(score, 0.0)


def compute_relative_morphology(
    point_records: List[dict],
    energy_db: np.ndarray,
    bands: Sequence[BandDefinition],
) -> dict:
    """
    energy_db 形状：[测点数, 帧数, 子频带数]
    """
    n_points, n_frames, n_bands = energy_db.shape

    center_indices = [
        i for i, record in enumerate(point_records)
        if record["point_type"] == "center"
    ]
    if not center_indices:
        raise RuntimeError("缺少有效中心点")
    center_index = center_indices[0]

    stable_z_all = np.full((n_points, n_bands), np.nan, dtype=np.float64)
    persistence_all = np.zeros((n_points, n_bands), dtype=np.float64)
    aggregate_z_all = np.zeros(n_points, dtype=np.float64)
    aggregate_persistence_all = np.zeros(n_points, dtype=np.float64)
    cell_score_all = np.zeros(n_points, dtype=np.float64)

    direction_results: Dict[str, dict] = {}
    direction_backgrounds: List[np.ndarray] = []
    direction_noise_scales: List[np.ndarray] = []
    all_valid_band_masks: List[np.ndarray] = []

    for direction in DIRECTION_ORDER:
        direction_point_indices = [
            i for i, record in enumerate(point_records)
            if record["direction"] == direction
        ]
        remote_indices = choose_remote_indices(point_records, direction)

        if len(remote_indices) < MIN_REMOTE_POINT_COUNT:
            direction_results[direction] = {
                "status": "REMOTE_POINTS_INSUFFICIENT",
                "remote_indices": remote_indices,
                "valid_band_mask": np.zeros(n_bands, dtype=bool),
                "near_band_median_z": np.zeros(n_bands, dtype=np.float64),
                "near_band_persistence": np.zeros(n_bands, dtype=np.float64),
                "noise_scale_db": np.full(n_bands, np.nan),
            }
            all_valid_band_masks.append(np.zeros(n_bands, dtype=bool))
            continue

        far_data = energy_db[remote_indices, :, :]
        background_db = np.median(far_data, axis=0)
        noise_scale_db = calculate_noise_scale_db(far_data, background_db)

        direction_backgrounds.append(background_db)
        direction_noise_scales.append(noise_scale_db)

        for point_index in direction_point_indices:
            z_frames = (
                energy_db[point_index, :, :]
                - background_db
            ) / noise_scale_db[None, :]
            stable_z_all[point_index, :] = np.median(z_frames, axis=0)
            persistence_all[point_index, :] = np.mean(
                z_frames >= FRAME_ACTIVE_Z,
                axis=0,
            )

        near_indices = [
            point_index
            for point_index in direction_point_indices
            if (
                NEAR_BAND_MIN_CM
                <= float(point_records[point_index]["distance_cm"])
                <= NEAR_BAND_MAX_CM
            )
        ]

        if not near_indices:
            valid_band_mask = np.zeros(n_bands, dtype=bool)
            near_band_median_z = np.zeros(n_bands, dtype=np.float64)
            near_band_persistence = np.zeros(n_bands, dtype=np.float64)
            status = "NEAR_POINTS_INSUFFICIENT"
        else:
            near_band_median_z = np.nanmedian(stable_z_all[near_indices, :], axis=0)
            near_band_persistence = np.nanmedian(persistence_all[near_indices, :], axis=0)
            candidate_mask = (
                (near_band_median_z >= MIN_BAND_NEAR_MEDIAN_Z)
                & (near_band_persistence >= MIN_BAND_NEAR_PERSISTENCE)
            )
            valid_band_mask = keep_contiguous_true_runs(
                candidate_mask,
                MIN_CONTIGUOUS_SUBBANDS,
            )
            status = "OK" if np.any(valid_band_mask) else "NO_STABLE_CONTIGUOUS_BAND"

        all_valid_band_masks.append(valid_band_mask)

        for point_index in direction_point_indices:
            aggregate_z, aggregate_persistence, cell_score = aggregate_cell_score(
                stable_z_all[point_index, :],
                persistence_all[point_index, :],
                valid_band_mask,
            )
            aggregate_z_all[point_index] = aggregate_z
            aggregate_persistence_all[point_index] = aggregate_persistence
            cell_score_all[point_index] = cell_score

        direction_results[direction] = {
            "status": status,
            "remote_indices": remote_indices,
            "valid_band_mask": valid_band_mask,
            "near_band_median_z": near_band_median_z,
            "near_band_persistence": near_band_persistence,
            "noise_scale_db": noise_scale_db,
            "background_db": background_db,
        }

    # 中心点：使用 8 个方向远端背景的中位数。
    if direction_backgrounds:
        center_background_db = np.median(np.stack(direction_backgrounds, axis=0), axis=0)
        directional_background_stack = np.stack(direction_backgrounds, axis=0)
        center_noise_scale = np.zeros(n_bands, dtype=np.float64)

        for band_index in range(n_bands):
            temporal_scale = robust_mad(center_background_db[:, band_index])
            direction_residual = (
                directional_background_stack[:, :, band_index]
                - center_background_db[None, :, band_index]
            ).ravel()
            directional_scale = robust_mad(direction_residual)
            center_noise_scale[band_index] = max(
                float(np.sqrt(temporal_scale**2 + directional_scale**2)),
                MIN_NOISE_SCALE_DB,
            )

        center_z_frames = (
            energy_db[center_index, :, :] - center_background_db
        ) / center_noise_scale[None, :]
        stable_z_all[center_index, :] = np.median(center_z_frames, axis=0)
        persistence_all[center_index, :] = np.mean(
            center_z_frames >= FRAME_ACTIVE_Z,
            axis=0,
        )

        center_valid_band_mask = (
            np.any(np.stack(all_valid_band_masks, axis=0), axis=0)
            if all_valid_band_masks
            else np.zeros(n_bands, dtype=bool)
        )
        aggregate_z, aggregate_persistence, cell_score = aggregate_cell_score(
            stable_z_all[center_index, :],
            persistence_all[center_index, :],
            center_valid_band_mask,
        )
        aggregate_z_all[center_index] = aggregate_z
        aggregate_persistence_all[center_index] = aggregate_persistence
        cell_score_all[center_index] = cell_score
    else:
        center_valid_band_mask = np.zeros(n_bands, dtype=bool)

    # 8 × (中心 + 8 距离) 的绘图剖面。
    score_profiles = np.full((len(DIRECTION_ORDER), len(distances) + 1), np.nan)
    z_profiles = np.full_like(score_profiles, np.nan)
    persistence_profiles = np.full_like(score_profiles, np.nan)

    for direction_index, direction in enumerate(DIRECTION_ORDER):
        score_profiles[direction_index, 0] = cell_score_all[center_index]
        z_profiles[direction_index, 0] = aggregate_z_all[center_index]
        persistence_profiles[direction_index, 0] = aggregate_persistence_all[center_index]

        for radius_index, distance in enumerate(distances, start=1):
            point_index = find_record_index(point_records, direction, distance)
            if point_index is None:
                continue
            score_profiles[direction_index, radius_index] = cell_score_all[point_index]
            z_profiles[direction_index, radius_index] = aggregate_z_all[point_index]
            persistence_profiles[direction_index, radius_index] = aggregate_persistence_all[point_index]

    return {
        "stable_z_all": stable_z_all,
        "persistence_all": persistence_all,
        "aggregate_z_all": aggregate_z_all,
        "aggregate_persistence_all": aggregate_persistence_all,
        "cell_score_all": cell_score_all,
        "direction_results": direction_results,
        "center_valid_band_mask": center_valid_band_mask,
        "score_profiles": score_profiles,
        "z_profiles": z_profiles,
        "persistence_profiles": persistence_profiles,
        "center_index": center_index,
        "n_frames": n_frames,
    }


# ============================================================
# 7. 扇区图绘制
# ============================================================


def radius_boundaries() -> np.ndarray:
    radii = np.asarray(distances, dtype=np.float64)
    boundaries = np.empty(radii.size + 1, dtype=np.float64)
    boundaries[0] = CENTER_CELL_OUTER_RADIUS_CM
    boundaries[1:-1] = 0.5 * (radii[:-1] + radii[1:])
    boundaries[-1] = radii[-1] + 0.5 * (radii[-1] - radii[-2])
    return boundaries


def setup_sector_axis(axis: plt.Axes) -> None:
    axis.set_aspect("equal")
    axis.set_xlim(-PLOT_LIMIT_CM, PLOT_LIMIT_CM)
    axis.set_ylim(-PLOT_LIMIT_CM, PLOT_LIMIT_CM)
    axis.set_facecolor(NO_EVIDENCE_COLOR)
    axis.set_xlabel("X distance (cm)")
    axis.set_ylabel("Y distance (cm)")
    axis.grid(False)

    for radius in distances:
        axis.add_patch(Circle(
            (0.0, 0.0),
            radius,
            fill=False,
            edgecolor="white",
            linewidth=0.25,
            alpha=0.25,
            zorder=5,
        ))

    for direction in DIRECTION_ORDER:
        angle = direction_angles[direction]
        x = 41.5 * np.cos(angle)
        y = 41.5 * np.sin(angle)
        axis.text(
            x,
            y,
            DIRECTION_LABELS[direction],
            color="white",
            ha="center",
            va="center",
            fontsize=7.5,
            zorder=10,
        )


def draw_sector_cells(
    axis: plt.Axes,
    result: dict,
    vmax: float,
    normalize_sample: bool,
) -> None:
    score_profiles = result["score_profiles"]
    boundaries = radius_boundaries()

    finite_positive = score_profiles[np.isfinite(score_profiles) & (score_profiles > 0)]
    sample_max = float(np.max(finite_positive)) if finite_positive.size else 0.0

    cmap = plt.get_cmap(HEATMAP_CMAP)
    norm_vmax = 1.0 if normalize_sample else max(vmax, EPS)
    norm = colors.Normalize(vmin=0.0, vmax=norm_vmax, clip=True)

    # 中心圆只画一次。
    center_score = float(score_profiles[0, 0]) if np.isfinite(score_profiles[0, 0]) else np.nan
    if np.isnan(center_score):
        center_color = MISSING_COLOR
    elif center_score <= 0:
        center_color = NO_EVIDENCE_COLOR
    else:
        value = center_score / sample_max if normalize_sample and sample_max > 0 else center_score
        center_color = cmap(norm(value))

    axis.add_patch(Circle(
        (0.0, 0.0),
        CENTER_CELL_OUTER_RADIUS_CM,
        facecolor=center_color,
        edgecolor=SECTOR_EDGE_COLOR,
        linewidth=SECTOR_EDGE_WIDTH,
        zorder=3,
    ))

    for direction_index, direction in enumerate(DIRECTION_ORDER):
        angle_deg = float(np.degrees(direction_angles[direction]))
        theta1 = angle_deg - ANGLE_CELL_WIDTH_DEG / 2.0
        theta2 = angle_deg + ANGLE_CELL_WIDTH_DEG / 2.0

        for radius_index, _distance in enumerate(distances):
            score = score_profiles[direction_index, radius_index + 1]
            inner = boundaries[radius_index]
            outer = boundaries[radius_index + 1]

            if not np.isfinite(score):
                facecolor = MISSING_COLOR
            elif score <= 0:
                facecolor = NO_EVIDENCE_COLOR
            else:
                value = score / sample_max if normalize_sample and sample_max > 0 else score
                facecolor = cmap(norm(value))

            axis.add_patch(Wedge(
                center=(0.0, 0.0),
                r=outer,
                theta1=theta1,
                theta2=theta2,
                width=outer - inner,
                facecolor=facecolor,
                edgecolor=SECTOR_EDGE_COLOR,
                linewidth=SECTOR_EDGE_WIDTH,
                zorder=2,
            ))

            measured_radius = float(distances[radius_index])
            x = measured_radius * np.cos(direction_angles[direction])
            y = measured_radius * np.sin(direction_angles[direction])
            axis.scatter(
                [x],
                [y],
                s=6,
                c="white",
                edgecolors="black",
                linewidths=0.25,
                zorder=6,
            )

    axis.scatter(
        [0.0],
        [0.0],
        marker="*",
        s=80,
        c="white",
        edgecolors="black",
        linewidths=0.5,
        zorder=8,
    )


def plot_sector_morphology(
    result: dict,
    global_vmax: float,
    normalize_sample: bool,
) -> None:
    figure, axis = plt.subplots(figsize=(8.3, 7.4), dpi=150)
    setup_sector_axis(axis)
    draw_sector_cells(axis, result, global_vmax, normalize_sample)

    title_kind = "Shape only (sample normalized)" if normalize_sample else "Absolute relative-evidence morphology (fixed scale)"
    axis.set_title(
        f"{title_kind}\n"
        f"{result['time_folder']} | Center {result['center_id']} | "
        f"[{result['segment_start_second']:.0f}, {result['segment_end_second']:.0f}) s | "
        f"{FREQ_LOW/1000:.0f}-{FREQ_HIGH/1000:.0f} kHz",
        fontsize=10.5,
        fontweight="bold",
    )

    cmap = plt.get_cmap(HEATMAP_CMAP)
    norm = colors.Normalize(vmin=0.0, vmax=1.0 if normalize_sample else max(global_vmax, EPS))
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = figure.colorbar(scalar, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(
        "Within-sample relative shape (0-1)"
        if normalize_sample
        else "Stable relative evidence score (z-weighted, fixed scale)"
    )

    filename_prefix = "shape_only_sector_morphology" if normalize_sample else "absolute_sector_morphology"
    save_path = os.path.join(
        result["result_dir"],
        f"{filename_prefix}_{result['time_folder']}_center_{result['center_id']}.png",
    )
    figure.tight_layout()
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)
    print(f"  扇区图已保存：{save_path}")


# ============================================================
# 8. 径向曲线和频带证据图
# ============================================================


def plot_radial_profiles(result: dict) -> None:
    radii_with_center = np.asarray([0] + distances, dtype=np.float64)
    z_profiles = result["z_profiles"]

    figure, axes = plt.subplots(2, 4, figsize=(14.0, 7.2), dpi=140, sharex=True, sharey=True)
    axes_flat = axes.ravel()

    for direction_index, direction in enumerate(DIRECTION_ORDER):
        axis = axes_flat[direction_index]
        values = z_profiles[direction_index, :]
        axis.plot(radii_with_center, values, marker="o", linewidth=1.5)
        axis.axhline(CELL_DISPLAY_THRESHOLD_Z, linestyle="--", linewidth=1.0)
        axis.axhline(0.0, linewidth=0.7)
        axis.set_title(DIRECTION_LABELS[direction])
        axis.set_xlim(0, max(distances))
        axis.grid(True, linestyle="--", alpha=0.30)
        axis.set_xlabel("Distance (cm)")
        axis.set_ylabel("Stable relative z")

    figure.suptitle(
        "Measured radial profiles — no PAVA, no forced decay\n"
        f"{result['time_folder']} | Center {result['center_id']}",
        fontsize=11,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))

    save_path = os.path.join(
        result["result_dir"],
        f"radial_profiles_{result['time_folder']}_center_{result['center_id']}.png",
    )
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)
    print(f"  径向曲线图已保存：{save_path}")


def plot_frequency_evidence(result: dict, bands: Sequence[BandDefinition]) -> None:
    evidence = np.zeros((len(DIRECTION_ORDER), len(bands)), dtype=np.float64)
    valid = np.zeros_like(evidence, dtype=bool)

    for direction_index, direction in enumerate(DIRECTION_ORDER):
        direction_result = result["direction_results"].get(direction, {})
        evidence[direction_index, :] = direction_result.get(
            "near_band_median_z",
            np.zeros(len(bands)),
        )
        valid[direction_index, :] = direction_result.get(
            "valid_band_mask",
            np.zeros(len(bands), dtype=bool),
        )

    vmax = max(float(np.nanpercentile(np.maximum(evidence, 0.0), 98)), 1.0)
    figure, axis = plt.subplots(figsize=(11.0, 5.5), dpi=140)
    image = axis.imshow(
        evidence,
        aspect="auto",
        origin="upper",
        cmap=HEATMAP_CMAP,
        vmin=0.0,
        vmax=vmax,
    )

    for row in range(valid.shape[0]):
        for col in range(valid.shape[1]):
            if valid[row, col]:
                axis.scatter(col, row, marker="s", s=95, facecolors="none", edgecolors="white", linewidths=1.4)

    axis.set_yticks(np.arange(len(DIRECTION_ORDER)))
    axis.set_yticklabels([DIRECTION_LABELS[d] for d in DIRECTION_ORDER])
    axis.set_xticks(np.arange(len(bands)))
    axis.set_xticklabels([band.label for band in bands], rotation=35, ha="right")
    axis.set_title(
        "Near-field subband evidence\nWhite boxes = retained contiguous subbands",
        fontsize=11,
        fontweight="bold",
    )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Near-field median relative z")

    save_path = os.path.join(
        result["result_dir"],
        f"frequency_evidence_{result['time_folder']}_center_{result['center_id']}.png",
    )
    figure.tight_layout()
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)
    print(f"  频带证据图已保存：{save_path}")


# ============================================================
# 9. CSV 输出
# ============================================================


def band_mask_text(mask: np.ndarray, bands: Sequence[BandDefinition]) -> str:
    selected = [bands[i].label for i, flag in enumerate(mask) if flag]
    return "; ".join(selected) if selected else "None"


def save_point_score_csv(result: dict, bands: Sequence[BandDefinition]) -> None:
    save_path = os.path.join(
        result["result_dir"],
        f"point_score_{result['time_folder']}_center_{result['center_id']}.csv",
    )

    fieldnames = [
        "time_folder",
        "center_id",
        "point_type",
        "direction",
        "distance_cm",
        "x_cm",
        "y_cm",
        "aggregate_stable_z",
        "aggregate_persistence",
        "cell_score",
        "selected_subbands",
        "file_name",
    ]

    with open(save_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for index, record in enumerate(result["point_records"]):
            if record["point_type"] == "center":
                mask = result["center_valid_band_mask"]
            else:
                mask = result["direction_results"].get(record["direction"], {}).get(
                    "valid_band_mask",
                    np.zeros(len(bands), dtype=bool),
                )

            writer.writerow({
                "time_folder": result["time_folder"],
                "center_id": result["center_id"],
                "point_type": record["point_type"],
                "direction": record["direction"],
                "distance_cm": record["distance_cm"],
                "x_cm": record["x_cm"],
                "y_cm": record["y_cm"],
                "aggregate_stable_z": float(result["aggregate_z_all"][index]),
                "aggregate_persistence": float(result["aggregate_persistence_all"][index]),
                "cell_score": float(result["cell_score_all"][index]),
                "selected_subbands": band_mask_text(mask, bands),
                "file_name": os.path.basename(record["file_path"]),
            })

    print(f"  测点结果表已保存：{save_path}")


def save_frequency_evidence_csv(result: dict, bands: Sequence[BandDefinition]) -> None:
    save_path = os.path.join(
        result["result_dir"],
        f"direction_frequency_evidence_{result['time_folder']}_center_{result['center_id']}.csv",
    )

    fieldnames = [
        "time_folder",
        "center_id",
        "direction",
        "direction_label",
        "band_low_hz",
        "band_high_hz",
        "near_band_median_z",
        "near_band_persistence",
        "noise_scale_db",
        "is_selected_contiguous_band",
        "direction_status",
        "remote_distances_used_cm",
    ]

    with open(save_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for direction in DIRECTION_ORDER:
            direction_result = result["direction_results"].get(direction, {})
            remote_indices = direction_result.get("remote_indices", [])
            remote_distances = [
                str(result["point_records"][idx]["distance_cm"])
                for idx in remote_indices
            ]
            near_z = direction_result.get("near_band_median_z", np.zeros(len(bands)))
            near_p = direction_result.get("near_band_persistence", np.zeros(len(bands)))
            noise_scale = direction_result.get("noise_scale_db", np.full(len(bands), np.nan))
            valid_mask = direction_result.get("valid_band_mask", np.zeros(len(bands), dtype=bool))

            for band_index, band in enumerate(bands):
                writer.writerow({
                    "time_folder": result["time_folder"],
                    "center_id": result["center_id"],
                    "direction": direction,
                    "direction_label": DIRECTION_LABELS[direction],
                    "band_low_hz": band.low_hz,
                    "band_high_hz": band.high_hz,
                    "near_band_median_z": float(near_z[band_index]),
                    "near_band_persistence": float(near_p[band_index]),
                    "noise_scale_db": float(noise_scale[band_index]),
                    "is_selected_contiguous_band": bool(valid_mask[band_index]),
                    "direction_status": direction_result.get("status", "UNKNOWN"),
                    "remote_distances_used_cm": ";".join(remote_distances),
                })

    print(f"  方向频带结果表已保存：{save_path}")


def save_summary_all(all_results: List[dict], global_vmax: float, bands: Sequence[BandDefinition]) -> None:
    save_path = os.path.join(OUTPUT_ROOT_DIR, "summary_all.csv")
    fieldnames = [
        "time_folder",
        "center_id",
        "time_index",
        "segment_start_second",
        "segment_end_second",
        "num_points",
        "num_frames",
        "center_stable_z",
        "center_persistence",
        "center_cell_score",
        "selected_direction_count",
        "selected_direction_names",
        "total_selected_direction_subbands",
        "max_cell_score",
        "mean_positive_cell_score",
        "global_color_vmax",
    ]

    with open(save_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in all_results:
            center_index = result["center_index"]
            selected_directions = [
                direction
                for direction in DIRECTION_ORDER
                if np.any(result["direction_results"].get(direction, {}).get(
                    "valid_band_mask",
                    np.zeros(len(bands), dtype=bool),
                ))
            ]
            total_selected = sum(
                int(np.sum(result["direction_results"].get(direction, {}).get(
                    "valid_band_mask",
                    np.zeros(len(bands), dtype=bool),
                )))
                for direction in DIRECTION_ORDER
            )
            scores = result["cell_score_all"]
            positive_scores = scores[scores > 0]

            writer.writerow({
                "time_folder": result["time_folder"],
                "center_id": result["center_id"],
                "time_index": result["time_index"],
                "segment_start_second": result["segment_start_second"],
                "segment_end_second": result["segment_end_second"],
                "num_points": len(result["point_records"]),
                "num_frames": result["n_frames"],
                "center_stable_z": float(result["aggregate_z_all"][center_index]),
                "center_persistence": float(result["aggregate_persistence_all"][center_index]),
                "center_cell_score": float(result["cell_score_all"][center_index]),
                "selected_direction_count": len(selected_directions),
                "selected_direction_names": "; ".join(selected_directions) if selected_directions else "None",
                "total_selected_direction_subbands": total_selected,
                "max_cell_score": float(np.max(scores)) if scores.size else 0.0,
                "mean_positive_cell_score": float(np.mean(positive_scores)) if positive_scores.size else 0.0,
                "global_color_vmax": global_vmax,
            })

    print(f"\n全部样本摘要已保存：{save_path}")


# ============================================================
# 10. 单样本计算、全局色标和主程序
# ============================================================


def compute_single_sample(
    time_folder: str,
    center_id: str,
    center_index: Dict[str, List[str]],
    offset_index: Dict[OffsetKey, List[str]],
    result_dir: str,
    bands: Sequence[BandDefinition],
) -> Optional[dict]:
    try:
        time_index, segment_start_second, segment_end_second = get_time_slice_from_center_id(center_id)
    except ValueError as exc:
        print(f"  错误：{exc}")
        return None

    print(
        f"\n处理：{time_folder} | center={center_id} | "
        f"[{segment_start_second:.0f}, {segment_end_second:.0f}) 秒"
    )

    point_records = build_point_records(center_id, center_index, offset_index)
    if len(point_records) < 10:
        print("  错误：有效测点数量太少")
        return None

    valid_records, energy_db = load_all_point_energy_matrices(
        point_records,
        segment_start_second,
        segment_end_second,
        bands,
    )
    if energy_db is None or len(valid_records) < 10:
        print("  错误：无法建立完整的测点×帧×子频带矩阵")
        return None

    try:
        morphology = compute_relative_morphology(valid_records, energy_db, bands)
    except Exception as exc:
        print(f"  错误：形态计算失败：{exc}")
        return None

    result = {
        "time_folder": time_folder,
        "center_id": center_id,
        "time_index": time_index,
        "segment_start_second": segment_start_second,
        "segment_end_second": segment_end_second,
        "result_dir": result_dir,
        "point_records": valid_records,
        "energy_db": energy_db,
        **morphology,
    }

    selected_directions = [
        direction
        for direction in DIRECTION_ORDER
        if np.any(result["direction_results"].get(direction, {}).get(
            "valid_band_mask",
            np.zeros(len(bands), dtype=bool),
        ))
    ]

    print(f"  有效测点：{len(valid_records)}")
    print(f"  短时帧数：{result['n_frames']}")
    print(f"  子频带数：{len(bands)}")
    print(
        f"  中心稳定 z：{result['aggregate_z_all'][result['center_index']]:.3f}，"
        f"中心持续率：{result['aggregate_persistence_all'][result['center_index']]:.3f}"
    )
    print(f"  保留频带的方向：{', '.join(selected_directions) if selected_directions else 'None'}")

    for direction in DIRECTION_ORDER:
        direction_result = result["direction_results"].get(direction, {})
        mask = direction_result.get("valid_band_mask", np.zeros(len(bands), dtype=bool))
        print(
            f"    {direction:>10s}: {direction_result.get('status', 'UNKNOWN')}, "
            f"保留 {int(np.sum(mask))} 个子频带"
        )

    return result


def collect_all_results(bands: Sequence[BandDefinition]) -> List[dict]:
    all_results: List[dict] = []
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)

    for time_folder in time_folders:
        print("\n" + "=" * 92)
        print(f"扫描文件夹：{time_folder}")
        print("=" * 92)

        center_data_dir = os.path.join(center_root_dir, time_folder)
        offset_data_dir = os.path.join(offset_root_dir, time_folder)

        if not os.path.isdir(center_data_dir):
            print(f"警告：中心目录不存在：{center_data_dir}")
            continue
        if not os.path.isdir(offset_data_dir):
            print(f"警告：偏移目录不存在：{offset_data_dir}")
            continue

        result_dir = os.path.join(OUTPUT_ROOT_DIR, time_folder)
        os.makedirs(result_dir, exist_ok=True)

        center_index = build_center_file_index(center_data_dir)
        offset_index = build_offset_file_index(offset_data_dir)
        center_ids = sorted(center_index.keys(), key=center_id_sort_key)

        print(f"检测到中心编号：{center_ids}")
        print(f"偏移索引组数：{len(offset_index)}")

        for center_id in center_ids:
            result = compute_single_sample(
                time_folder,
                center_id,
                center_index,
                offset_index,
                result_dir,
                bands,
            )
            if result is not None:
                all_results.append(result)

    return all_results


def determine_global_vmax(all_results: Sequence[dict]) -> float:
    values: List[np.ndarray] = []
    for result in all_results:
        sample_values = result["cell_score_all"]
        sample_values = sample_values[np.isfinite(sample_values) & (sample_values > 0)]
        if sample_values.size:
            values.append(sample_values)

    if not values:
        return 1.0

    all_values = np.concatenate(values)
    return max(float(np.percentile(all_values, GLOBAL_COLOR_PERCENTILE)), EPS)


def save_scale_info(global_vmax: float, bands: Sequence[BandDefinition]) -> None:
    save_path = os.path.join(OUTPUT_ROOT_DIR, "global_scale_and_method.txt")
    with open(save_path, "w", encoding="utf-8") as file:
        file.write("Method: per-direction remote background + subband robust z + sector cells\n")
        file.write(f"frequency_hz = {FREQ_LOW}-{FREQ_HIGH}\n")
        file.write(f"subband_width_hz = {SUBBAND_WIDTH_HZ}\n")
        file.write(f"remote_background_distances_cm = {REMOTE_BACKGROUND_DISTANCES}\n")
        file.write(f"minimum_noise_scale_db = {MIN_NOISE_SCALE_DB}\n")
        file.write(f"frame_active_z = {FRAME_ACTIVE_Z}\n")
        file.write(f"cell_display_threshold_z = {CELL_DISPLAY_THRESHOLD_Z}\n")
        file.write(f"global_vmax = {global_vmax:.12g}\n")
        file.write("bands = " + ", ".join(band.label for band in bands) + "\n")
    print(f"统一尺度和方法说明已保存：{save_path}")


def validate_config() -> bool:
    valid = True
    if not time_folders:
        print("配置错误：time_folders 不能为空")
        valid = False
    if FREQ_LOW < 0 or FREQ_HIGH <= FREQ_LOW:
        print("配置错误：FREQ_HIGH 必须大于 FREQ_LOW")
        valid = False
    if SUBBAND_WIDTH_HZ <= 0:
        print("配置错误：SUBBAND_WIDTH_HZ 必须大于 0")
        valid = False
    if FRAME_SECONDS <= 0 or HOP_SECONDS <= 0:
        print("配置错误：FRAME_SECONDS 和 HOP_SECONDS 必须大于 0")
        valid = False
    if WELCH_NPERSEG < 16:
        print("配置错误：WELCH_NPERSEG 不能小于 16")
        valid = False
    if not distances:
        print("配置错误：distances 不能为空")
        valid = False
    if MIN_CONTIGUOUS_SUBBANDS < 1:
        print("配置错误：MIN_CONTIGUOUS_SUBBANDS 至少为 1")
        valid = False
    if MIN_REMOTE_POINT_COUNT < 1:
        print("配置错误：MIN_REMOTE_POINT_COUNT 至少为 1")
        valid = False
    return valid


def main() -> None:
    print("=" * 92)
    print("八方向泄漏径向形态可视化 — 最新稳健版")
    print("=" * 92)
    print("方法：每方向减本方向 35/40 cm 背景，2 kHz 子频带，远端波动标准化")
    print("顺序：保留正负差值 → 时间中位数 → 连续频带筛选 → 最后阈值截断")
    print("绘图：实测扇区，不做 PAVA，不做连续二维插值")
    print(f"频段：{FREQ_LOW/1000:.1f}-{FREQ_HIGH/1000:.1f} kHz")
    print(f"子频带宽：{SUBBAND_WIDTH_HZ/1000:.1f} kHz")
    print(f"远端背景距离：{REMOTE_BACKGROUND_DISTANCES} cm")
    print("重要：该图是相对远端背景的径向形态，不是绝对二维声压图。")

    if not validate_config():
        print("配置检查失败，程序停止")
        return

    bands = build_subbands()
    if not bands:
        print("没有生成有效子频带，程序停止")
        return

    all_results = collect_all_results(bands)
    if not all_results:
        print("\n没有得到任何有效结果")
        return

    global_vmax = determine_global_vmax(all_results)
    print("\n" + "=" * 92)
    print(f"统一热力图 vmax：{global_vmax:.6f}")
    print("=" * 92)

    save_scale_info(global_vmax, bands)

    for result in all_results:
        save_point_score_csv(result, bands)
        save_frequency_evidence_csv(result, bands)
        plot_sector_morphology(result, global_vmax, normalize_sample=False)
        if OUTPUT_SHAPE_ONLY:
            plot_sector_morphology(result, global_vmax, normalize_sample=True)
        if OUTPUT_RADIAL_PROFILES:
            plot_radial_profiles(result)
        if OUTPUT_FREQUENCY_EVIDENCE:
            plot_frequency_evidence(result, bands)

    save_summary_all(all_results, global_vmax, bands)

    print("\n" + "=" * 92)
    print("全部处理完成")
    print(f"输出目录：{os.path.abspath(OUTPUT_ROOT_DIR)}")
    print("重点先看 absolute_sector_morphology 和 radial_profiles。")
    print("=" * 92)


if __name__ == "__main__":
    main()
