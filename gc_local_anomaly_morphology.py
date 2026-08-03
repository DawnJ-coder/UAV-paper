# -*- coding: utf-8 -*-
"""
GC波束结果：现场自监督、多参考点局部异常声源形态分析。

适用数据：
1. beamform_results_gc：中心点波束结果；
2. beamform_results_offset_multiple_gc_0001：5~40 cm、8个方向；
3. beamform_results_offset_multiple_gc_gy_0001：45~80 cm、8个方向。

每个中心位置共使用：
    1个中心点 + 16个距离 * 8个方向 = 129个空间测点。

与旧程序的根本区别：
- 不再指定40 cm或80 cm为“真实背景”；
- 5~80 cm全部作为普通空间测点；
- 对每一个测点，排除其附近一定范围内的点，再用其余点预测该点背景；
- 同时使用15、20、25、30、40 cm五种保护半径；
- 保留有正有负的残差，不在逐频点减法后把负值置零；
- 最终输出不同保护半径下的中位结果、显著支持率和模型分歧；
- 所有科学计算只使用实测点，二维插值只用于显示。

重要解释：
当前数据没有“同一位置、泄漏关闭”的对照，因此程序得到的是
“相对于现场空间可预测背景的局部异常声源形态”，不是严格意义上
唯一可分离的纯泄漏声场。随机环境中心点也处在有泄漏的空间中，
它只能作为负对照，不能直接作为真实泄漏中心点的背景相减。

时间同步规则：
    center 00_00 -> 所有坐标统一读取原始WAV的[0, 1)秒
    center 00_01 -> 所有坐标统一读取原始WAV的[1, 2)秒
    center 00_02 -> 所有坐标统一读取原始WAV的[2, 3)秒

程序不会用整段WAV代替指定的1秒。
"""

import csv
import glob
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# 某些只读运行环境没有可写的用户配置目录；显式使用临时缓存，
# 不影响Windows本地运行，也能避免Matplotlib导入警告。
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "gc_morphology_matplotlib_cache"),
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

# 要处理的时间点文件夹。
# 同一个time_folder中的00_00、00_01……会被认为是同一候选中心的连续秒，
# 最后会额外输出该文件夹的多秒稳定形态图。
time_folders = [
    # "HM20260626_142938.ld",
    # "HM20260626_143034.ld",
    # "HM20260626_144226.ld",
    # "HM20260626_144325.ld",
    "HM20260702_111044.ld",
]

# 根目录路径
center_root_dir = r"D:\gas\beamform_results_gc"
offset_near_root_dir = r"D:\gas\beamform_results_offset_multiple_gc_0001"
offset_far_root_dir = r"D:\gas\beamform_results_offset_multiple_gc_gy_0001"
result_root_dir = r"D:\gas\results_gc_local_anomaly_morphology"

# 可选：给每个时间文件夹标注TRUE/FALSE，仅写入CSV和图片标题，不参与模型训练。
# TRUE表示中心是真实泄漏位置；FALSE表示随机环境中心位置。
TIME_FOLDER_LABELS: Dict[str, str] = {
    # "HM20260626_142938.ld": "TRUE",
    # "HM20260626_143034.ld": "FALSE",
}

# 是否递归搜索子文件夹
RECURSIVE_SEARCH = False

# 期望采样率仅用于检查；Welch仍使用每个WAV自己的实际采样率。
EXPECTED_SAMPLE_RATE = 192000

# center_id最后一个下划线后的数字表示第几秒。
TIME_SLICE_SECONDS = 1.0
STRICT_TIME_SLICE = True

# 分析频带
FREQ_LOW = 50000
FREQ_HIGH = 70000
NFFT = 4096
WELCH_OVERLAP_RATIO = 0.5

# 8个方向与极坐标角度
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

# 5~80 cm全部是普通实测点，不再把40或80 cm指定为背景。
DISTANCES_CM = list(range(5, 81, 5))
MAX_DISTANCE_CM = max(DISTANCES_CM)

# 对每个待预测测点，排除离它小于该半径的所有测点。
# 五种半径都运行，最终看它们是否得到一致结果。
GUARD_RADII_CM = (15, 20, 25, 30, 40)

# 自适应候选背景模型。程序使用空间留邻域验证自动选模型。
# constant最简单；plane是平面；quadratic是二次曲面；
# rbf_35和rbf_55是两种平滑尺度的非线性RBF模型。
CANDIDATE_MODELS = (
    "constant",
    "plane",
    "quadratic",
    "rbf_35",
    "rbf_55",
)

# 模型拟合参数
MIN_TRAINING_POINTS = 24
ROBUST_MAX_ITER = 20
ROBUST_HUBER_DELTA = 1.5
RIDGE_STRENGTH = 1e-3
MODEL_SELECTION_MAX_TARGETS = 33
MODEL_SELECTION_MIN_TOLERANCE_DB = 0.10
MIN_RESIDUAL_SCALE_DB = 0.25

# 标准化残差阈值与稳定条件
Z_THRESHOLD = 3.0
MIN_GUARD_SUPPORT = 0.60
# 不同保护半径下Z值的绝对大小会因残差尺度不同而变化；
# 3.0用于排除明显不稳定结果，同时不会误删各半径都为强阳性的中心热点。
MAX_GUARD_DISAGREEMENT_Z = 3.0

# 图像参数
GRID_MIN_CM = -85
GRID_MAX_CM = 85
GRID_SIZE = 260
DRAW_INDIVIDUAL_SECOND_FIGURES = True
DRAW_TEMPORAL_STABILITY_FIGURE = True
Z_DISPLAY_LIMIT = 6.0
RAW_CMAP = "viridis"
SIGNED_CMAP = "coolwarm"
SUPPORT_CMAP = "magma"
UNCERTAINTY_CMAP = "cividis"

# 数值安全参数
ENERGY_FLOOR = 1e-30
EPS = 1e-12


# ============================================================
# 2. 数据类型
# ============================================================

Spectrum = Tuple[np.ndarray, np.ndarray]
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
class GuardResult:
    guard_radius_cm: int
    selected_model: str
    model_cv_mae_db: float
    residual_scale_db: float
    background_db: np.ndarray
    residual_db: np.ndarray
    z_score: np.ndarray
    candidate_scores: Dict[str, float]


@dataclass
class SecondAnalysis:
    time_folder: str
    label: str
    center_id: str
    time_index: int
    segment_start_second: float
    segment_end_second: float
    points: List[SpatialPoint]
    guard_results: List[GuardResult]
    consensus_background_db: np.ndarray
    consensus_residual_db: np.ndarray
    consensus_z: np.ndarray
    guard_support: np.ndarray
    guard_disagreement_z: np.ndarray
    stable_mask: np.ndarray


# ============================================================
# 3. 文件名解析和索引
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
    """列出文件夹中的WAV文件。"""
    if RECURSIVE_SEARCH:
        pattern = os.path.join(folder, "**", "*.wav")
        return sorted(glob.glob(pattern, recursive=True))
    return sorted(glob.glob(os.path.join(folder, "*.wav")))


def resolve_time_data_dir(root_dir: str, time_folder: str) -> str:
    """兼容root/time_folder/*.wav和root/*.wav两种目录结构。"""
    nested_dir = os.path.join(root_dir, time_folder)
    if os.path.isdir(nested_dir):
        return nested_dir

    if os.path.isdir(root_dir) and list_wav_files(root_dir):
        print(f"提醒：未找到子文件夹{nested_dir}，直接读取：{root_dir}")
        return root_dir

    return nested_dir


def build_center_file_index(center_data_dir: str) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    for file_path in list_wav_files(center_data_dir):
        match = CENTER_FILE_REGEX.search(os.path.basename(file_path))
        if match:
            index.setdefault(match.group("center"), []).append(file_path)
    for center_id in index:
        index[center_id] = sorted(index[center_id])
    return index


def build_offset_file_index(offset_data_dir: str) -> Dict[OffsetKey, List[str]]:
    index: Dict[OffsetKey, List[str]] = {}
    for file_path in list_wav_files(offset_data_dir):
        match = OFFSET_FILE_REGEX.search(os.path.basename(file_path))
        if not match:
            continue
        center_id = match.group("center")
        distance = int(match.group("distance"))
        direction = match.group("direction").lower()
        index.setdefault((center_id, distance, direction), []).append(file_path)
    for key in index:
        index[key] = sorted(index[key])
    return index


def merge_offset_file_indices(
    *indices: Dict[OffsetKey, List[str]],
) -> Dict[OffsetKey, List[str]]:
    merged: Dict[OffsetKey, List[str]] = {}
    for index in indices:
        for key, files in index.items():
            merged.setdefault(key, []).extend(files)
    for key in merged:
        merged[key] = sorted(set(merged[key]))
    return merged


def choose_first_file(
    files: Optional[List[str]],
    description: str,
) -> Optional[str]:
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


def get_time_slice_from_center_id(center_id: str) -> Tuple[int, float, float]:
    parts = str(center_id).split("_")
    if len(parts) < 2 or not parts[-1].isdigit():
        raise ValueError(
            f"center_id={center_id!r}不符合'00_00'格式，无法确定秒序号"
        )
    time_index = int(parts[-1])
    start_second = time_index * float(TIME_SLICE_SECONDS)
    end_second = start_second + float(TIME_SLICE_SECONDS)
    return time_index, start_second, end_second


# ============================================================
# 4. WAV读取和频带能量
# ============================================================

def convert_wav_to_float(y: np.ndarray) -> np.ndarray:
    """转float64；不做逐文件峰值归一化，保留测点间幅值关系。"""
    if np.issubdtype(y.dtype, np.integer):
        info = np.iinfo(y.dtype)
        full_scale = float(max(abs(info.min), abs(info.max)))
        if full_scale <= 0:
            raise ValueError(f"无效整数WAV类型：{y.dtype}")
        return y.astype(np.float64) / full_scale
    if np.issubdtype(y.dtype, np.floating):
        return y.astype(np.float64)
    raise TypeError(f"不支持的WAV数据类型：{y.dtype}")


def get_spectrum(
    file_path: str,
    segment_start_second: float,
    segment_end_second: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """读取严格同步的指定秒，计算目标频带Welch PSD。"""
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

    total_samples = int(y.size)
    duration = total_samples / float(sample_rate)
    start_sample = int(round(segment_start_second * sample_rate))
    end_sample = int(round(segment_end_second * sample_rate))

    if start_sample >= total_samples:
        print(
            f"  错误：{os.path.basename(file_path)}总时长{duration:.6f}秒，"
            f"无法读取[{segment_start_second:.3f}, {segment_end_second:.3f})秒"
        )
        return None, None

    if end_sample > total_samples:
        message = (
            f"{os.path.basename(file_path)}总时长{duration:.6f}秒，"
            f"目标区间[{segment_start_second:.3f}, "
            f"{segment_end_second:.3f})秒不完整"
        )
        if STRICT_TIME_SLICE:
            print(f"  错误：{message}；已跳过")
            return None, None
        print(f"  警告：{message}；仅使用实际尾段")
        end_sample = total_samples

    y = y[start_sample:end_sample]
    expected_samples = int(round(
        (segment_end_second - segment_start_second) * sample_rate
    ))
    if STRICT_TIME_SLICE and y.size != expected_samples:
        print(
            f"  错误：{os.path.basename(file_path)}切片点数={y.size}，"
            f"期望={expected_samples}；已跳过"
        )
        return None, None

    if y.size < 16:
        print(f"  错误：目标秒数据过短：{file_path}")
        return None, None

    try:
        y = convert_wav_to_float(y)
    except (TypeError, ValueError) as exc:
        print(f"  错误：{exc}，文件：{file_path}")
        return None, None

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = y - np.mean(y)

    if sample_rate != EXPECTED_SAMPLE_RATE:
        print(
            f"  提醒：{os.path.basename(file_path)}采样率={sample_rate}Hz，"
            f"不是期望的{EXPECTED_SAMPLE_RATE}Hz"
        )

    nyquist = sample_rate / 2.0
    actual_high = min(FREQ_HIGH, nyquist)
    if FREQ_LOW >= actual_high:
        print(
            f"  错误：奈奎斯特频率={nyquist:.1f}Hz，"
            f"无法分析{FREQ_LOW}-{FREQ_HIGH}Hz"
        )
        return None, None

    nperseg = min(NFFT, y.size)
    noverlap = min(int(nperseg * WELCH_OVERLAP_RATIO), nperseg - 1)

    try:
        freqs, psd = signal.welch(
            y,
            fs=sample_rate,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="density",
        )
    except Exception as exc:
        print(f"  错误：Welch计算失败：{file_path}\n    {exc}")
        return None, None

    mask = (freqs >= FREQ_LOW) & (freqs <= actual_high)
    freqs = np.asarray(freqs[mask], dtype=np.float64)
    psd = np.asarray(psd[mask], dtype=np.float64)
    if freqs.size < 2:
        print(f"  错误：目标频带有效频点不足：{file_path}")
        return None, None

    psd = np.maximum(
        np.nan_to_num(psd, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
    )
    return freqs, psd


def trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def get_band_energy(freqs: np.ndarray, psd: np.ndarray) -> float:
    """PSD在目标频带积分。"""
    if freqs is None or psd is None or len(freqs) < 2:
        return 0.0
    return max(trapezoid_integral(psd, freqs), 0.0)


def energy_to_db(energy: float) -> float:
    """能量转相对满量程dB；不同测点保持同一参考。"""
    return float(10.0 * np.log10(max(float(energy), ENERGY_FLOOR)))


# ============================================================
# 5. 读取一个中心秒的129个空间测点
# ============================================================

def collect_spatial_points(
    center_id: str,
    center_index: Dict[str, List[str]],
    offset_index: Dict[OffsetKey, List[str]],
    segment_start_second: float,
    segment_end_second: float,
) -> List[SpatialPoint]:
    points: List[SpatialPoint] = []

    center_file = choose_first_file(
        center_index.get(center_id),
        f"中心点{center_id}",
    )
    if center_file is None:
        print(f"错误：中心点{center_id}文件不存在")
        return points

    center_freqs, center_psd = get_spectrum(
        center_file,
        segment_start_second,
        segment_end_second,
    )
    if center_freqs is None or center_psd is None:
        return points

    center_energy = get_band_energy(center_freqs, center_psd)
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
            signal_file = choose_first_file(
                offset_index.get((center_id, distance, direction)),
                f"中心{center_id}、{direction}、{distance}cm",
            )
            if signal_file is None:
                print(f"  缺失：{direction}方向{distance}cm")
                continue

            freqs, psd = get_spectrum(
                signal_file,
                segment_start_second,
                segment_end_second,
            )
            if freqs is None or psd is None:
                continue

            energy = get_band_energy(freqs, psd)
            x_cm = float(distance * np.cos(angle))
            y_cm = float(distance * np.sin(angle))
            if abs(x_cm) < 1e-10:
                x_cm = 0.0
            if abs(y_cm) < 1e-10:
                y_cm = 0.0

            points.append(
                SpatialPoint(
                    point_key=f"{direction}_{distance}",
                    point_type="offset",
                    direction=direction,
                    distance_cm=distance,
                    x_cm=x_cm,
                    y_cm=y_cm,
                    energy=energy,
                    energy_db=energy_to_db(energy),
                    signal_file=os.path.basename(signal_file),
                )
            )

    return points


# ============================================================
# 6. 稳健空间背景模型
# ============================================================

def robust_mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - median)))


def make_rbf_anchors() -> np.ndarray:
    """固定RBF锚点，避免用待预测值决定模型结构。"""
    anchors = [(0.0, 0.0)]
    for radius in (0.5, 1.0):
        for angle in DIRECTION_ANGLES.values():
            anchors.append((radius * np.cos(angle), radius * np.sin(angle)))
    return np.asarray(anchors, dtype=np.float64)


RBF_ANCHORS = make_rbf_anchors()


def model_complexity(model_name: str) -> int:
    order = {
        "constant": 1,
        "plane": 3,
        "quadratic": 6,
        "rbf_55": 20,
        "rbf_35": 20,
    }
    return order[model_name]


def build_design_matrix(
    model_name: str,
    x_cm: np.ndarray,
    y_cm: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x_cm, dtype=np.float64) / float(MAX_DISTANCE_CM)
    y = np.asarray(y_cm, dtype=np.float64) / float(MAX_DISTANCE_CM)

    if model_name == "constant":
        return np.ones((x.size, 1), dtype=np.float64)

    if model_name == "plane":
        return np.column_stack([np.ones_like(x), x, y])

    if model_name == "quadratic":
        return np.column_stack([
            np.ones_like(x), x, y, x * x, x * y, y * y,
        ])

    if model_name.startswith("rbf_"):
        length_cm = float(model_name.split("_", 1)[1])
        length_norm = length_cm / float(MAX_DISTANCE_CM)
        coordinates = np.column_stack([x, y])
        distance_sq = np.sum(
            (coordinates[:, None, :] - RBF_ANCHORS[None, :, :]) ** 2,
            axis=2,
        )
        rbf = np.exp(-0.5 * distance_sq / (length_norm ** 2))
        # 线性项负责大尺度梯度，RBF负责平滑的非线性结构。
        return np.column_stack([np.ones_like(x), x, y, rbf])

    raise ValueError(f"未知模型：{model_name}")


def fit_robust_ridge(
    design: np.ndarray,
    target: np.ndarray,
) -> Optional[np.ndarray]:
    """Huber迭代重加权岭回归；第一列截距不惩罚。"""
    design = np.asarray(design, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.all(np.isfinite(design), axis=1) & np.isfinite(target)
    design = design[valid]
    target = target[valid]

    n_samples, n_features = design.shape
    if n_samples < max(MIN_TRAINING_POINTS, n_features + 3):
        return None

    weights = np.ones(n_samples, dtype=np.float64)
    beta = np.zeros(n_features, dtype=np.float64)
    penalty = np.eye(n_features, dtype=np.float64) * RIDGE_STRENGTH
    penalty[0, 0] = 0.0

    for _ in range(ROBUST_MAX_ITER):
        weighted_design = design * weights[:, None]
        lhs = design.T @ weighted_design + penalty
        rhs = design.T @ (weights * target)
        try:
            new_beta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            try:
                new_beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            except np.linalg.LinAlgError:
                return None

        residual = target - design @ new_beta
        residual_center = float(np.median(residual))
        scale = max(robust_mad(residual), MIN_RESIDUAL_SCALE_DB)
        standardized = np.abs(residual - residual_center) / (
            ROBUST_HUBER_DELTA * scale + EPS
        )
        new_weights = np.ones_like(standardized)
        large = standardized > 1.0
        new_weights[large] = 1.0 / standardized[large]

        if np.max(np.abs(new_beta - beta)) < 1e-7:
            beta = new_beta
            break
        beta = new_beta
        weights = new_weights

    return beta


def predict_target_with_guard(
    model_name: str,
    x_cm: np.ndarray,
    y_cm: np.ndarray,
    values_db: np.ndarray,
    target_index: int,
    guard_radius_cm: float,
) -> float:
    """排除目标邻域后预测一个测点。"""
    dx = x_cm - x_cm[target_index]
    dy = y_cm - y_cm[target_index]
    distance_to_target = np.sqrt(dx * dx + dy * dy)
    train_mask = (
        (distance_to_target >= float(guard_radius_cm) - 1e-9)
        & np.isfinite(values_db)
    )

    train_indices = np.flatnonzero(train_mask)
    if train_indices.size < MIN_TRAINING_POINTS:
        return float("nan")

    train_design = build_design_matrix(
        model_name,
        x_cm[train_indices],
        y_cm[train_indices],
    )
    beta = fit_robust_ridge(train_design, values_db[train_indices])
    if beta is None:
        return float("nan")

    target_design = build_design_matrix(
        model_name,
        np.asarray([x_cm[target_index]]),
        np.asarray([y_cm[target_index]]),
    )
    return float(target_design[0] @ beta)


def choose_model_selection_indices(n_points: int) -> np.ndarray:
    """均匀抽取测点用于模型选择，选定模型后再预测全部点。"""
    if n_points <= MODEL_SELECTION_MAX_TARGETS:
        return np.arange(n_points, dtype=int)
    indices = np.linspace(
        0,
        n_points - 1,
        MODEL_SELECTION_MAX_TARGETS,
    )
    return np.unique(np.round(indices).astype(int))


def select_background_model(
    x_cm: np.ndarray,
    y_cm: np.ndarray,
    values_db: np.ndarray,
    guard_radius_cm: int,
) -> Tuple[str, float, Dict[str, float]]:
    """
    用留邻域预测误差选模型。

    采用稳健中位绝对误差；若多个模型接近最优，使用一标准误思想
    选择更简单模型，避免复杂模型追随局部泄漏热点。
    """
    selection_indices = choose_model_selection_indices(values_db.size)
    errors_by_model: Dict[str, np.ndarray] = {}
    scores: Dict[str, float] = {}

    for model_name in CANDIDATE_MODELS:
        errors: List[float] = []
        for target_index in selection_indices:
            prediction = predict_target_with_guard(
                model_name,
                x_cm,
                y_cm,
                values_db,
                int(target_index),
                guard_radius_cm,
            )
            if np.isfinite(prediction):
                errors.append(abs(values_db[target_index] - prediction))

        error_array = np.asarray(errors, dtype=np.float64)
        errors_by_model[model_name] = error_array
        if error_array.size < max(8, selection_indices.size // 3):
            scores[model_name] = float("inf")
        else:
            scores[model_name] = float(np.median(error_array))

    finite_models = [name for name in CANDIDATE_MODELS if np.isfinite(scores[name])]
    if not finite_models:
        raise RuntimeError(
            f"保护半径{guard_radius_cm}cm下没有模型得到足够的有效预测"
        )

    best_model = min(finite_models, key=lambda name: scores[name])
    best_score = scores[best_model]
    best_errors = errors_by_model[best_model]
    score_uncertainty = robust_mad(best_errors) / np.sqrt(max(best_errors.size, 1))
    tolerance = max(MODEL_SELECTION_MIN_TOLERANCE_DB, score_uncertainty)

    eligible = [
        name
        for name in finite_models
        if scores[name] <= best_score + tolerance
    ]
    selected_model = min(
        eligible,
        key=lambda name: (model_complexity(name), CANDIDATE_MODELS.index(name)),
    )
    return selected_model, scores[selected_model], scores


def run_guard_analysis(
    x_cm: np.ndarray,
    y_cm: np.ndarray,
    values_db: np.ndarray,
    guard_radius_cm: int,
) -> GuardResult:
    selected_model, cv_mae_db, candidate_scores = select_background_model(
        x_cm,
        y_cm,
        values_db,
        guard_radius_cm,
    )

    background_db = np.full(values_db.shape, np.nan, dtype=np.float64)
    for target_index in range(values_db.size):
        background_db[target_index] = predict_target_with_guard(
            selected_model,
            x_cm,
            y_cm,
            values_db,
            target_index,
            guard_radius_cm,
        )

    residual_db = values_db - background_db
    valid_residual = residual_db[np.isfinite(residual_db)]
    if valid_residual.size < 8:
        raise RuntimeError(
            f"保护半径{guard_radius_cm}cm下有效残差点不足"
        )

    # 去除整体偏移，只保留局部高于空间背景的部分；不裁掉负值。
    residual_center = float(np.median(valid_residual))
    residual_db = residual_db - residual_center
    residual_scale_db = max(
        robust_mad(residual_db[np.isfinite(residual_db)]),
        MIN_RESIDUAL_SCALE_DB,
    )
    z_score = residual_db / residual_scale_db

    return GuardResult(
        guard_radius_cm=guard_radius_cm,
        selected_model=selected_model,
        model_cv_mae_db=cv_mae_db,
        residual_scale_db=residual_scale_db,
        background_db=background_db,
        residual_db=residual_db,
        z_score=z_score,
        candidate_scores=candidate_scores,
    )


def analyze_points(
    time_folder: str,
    label: str,
    center_id: str,
    time_index: int,
    segment_start_second: float,
    segment_end_second: float,
    points: List[SpatialPoint],
) -> SecondAnalysis:
    x_cm = np.asarray([point.x_cm for point in points], dtype=np.float64)
    y_cm = np.asarray([point.y_cm for point in points], dtype=np.float64)
    values_db = np.asarray([point.energy_db for point in points], dtype=np.float64)

    guard_results: List[GuardResult] = []
    for guard_radius_cm in GUARD_RADII_CM:
        result = run_guard_analysis(
            x_cm,
            y_cm,
            values_db,
            int(guard_radius_cm),
        )
        guard_results.append(result)
        print(
            f"  保护半径{guard_radius_cm:>2d}cm："
            f"模型={result.selected_model:<9s}，"
            f"空间验证MAE={result.model_cv_mae_db:.3f}dB，"
            f"残差尺度={result.residual_scale_db:.3f}dB"
        )

    background_stack = np.vstack([item.background_db for item in guard_results])
    residual_stack = np.vstack([item.residual_db for item in guard_results])
    z_stack = np.vstack([item.z_score for item in guard_results])

    consensus_background_db = np.nanmedian(background_stack, axis=0)
    consensus_residual_db = np.nanmedian(residual_stack, axis=0)
    consensus_z = np.nanmedian(z_stack, axis=0)
    guard_support = np.nanmean(z_stack >= Z_THRESHOLD, axis=0)

    z_median = np.nanmedian(z_stack, axis=0)
    guard_disagreement_z = 1.4826 * np.nanmedian(
        np.abs(z_stack - z_median[None, :]),
        axis=0,
    )

    stable_mask = (
        np.isfinite(consensus_z)
        & (consensus_z >= Z_THRESHOLD)
        & (guard_support >= MIN_GUARD_SUPPORT)
        & (guard_disagreement_z <= MAX_GUARD_DISAGREEMENT_Z)
    )

    return SecondAnalysis(
        time_folder=time_folder,
        label=label,
        center_id=center_id,
        time_index=time_index,
        segment_start_second=segment_start_second,
        segment_end_second=segment_end_second,
        points=points,
        guard_results=guard_results,
        consensus_background_db=consensus_background_db,
        consensus_residual_db=consensus_residual_db,
        consensus_z=consensus_z,
        guard_support=guard_support,
        guard_disagreement_z=guard_disagreement_z,
        stable_mask=stable_mask,
    )


# ============================================================
# 7. CSV输出
# ============================================================

def write_csv(path: str, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def save_second_point_csv(result_dir: str, analysis: SecondAnalysis) -> str:
    rows: List[dict] = []
    for index, point in enumerate(analysis.points):
        row = {
            "time_folder": analysis.time_folder,
            "label": analysis.label,
            "center_id": analysis.center_id,
            "time_index": analysis.time_index,
            "segment_start_second": analysis.segment_start_second,
            "segment_end_second": analysis.segment_end_second,
            "point_key": point.point_key,
            "point_type": point.point_type,
            "direction": point.direction,
            "distance_cm": point.distance_cm,
            "x_cm": point.x_cm,
            "y_cm": point.y_cm,
            "raw_energy": point.energy,
            "raw_energy_db": point.energy_db,
            "consensus_background_db": analysis.consensus_background_db[index],
            "consensus_residual_db": analysis.consensus_residual_db[index],
            "consensus_z": analysis.consensus_z[index],
            "guard_support": analysis.guard_support[index],
            "guard_disagreement_z": analysis.guard_disagreement_z[index],
            "stable_anomaly": "yes" if analysis.stable_mask[index] else "no",
            "signal_file": point.signal_file,
        }
        for guard_result in analysis.guard_results:
            suffix = f"g{guard_result.guard_radius_cm}"
            row[f"background_db_{suffix}"] = guard_result.background_db[index]
            row[f"residual_db_{suffix}"] = guard_result.residual_db[index]
            row[f"z_{suffix}"] = guard_result.z_score[index]
        rows.append(row)

    base_fields = [
        "time_folder", "label", "center_id", "time_index",
        "segment_start_second", "segment_end_second", "point_key",
        "point_type", "direction", "distance_cm", "x_cm", "y_cm",
        "raw_energy", "raw_energy_db", "consensus_background_db",
        "consensus_residual_db", "consensus_z", "guard_support",
        "guard_disagreement_z", "stable_anomaly", "signal_file",
    ]
    guard_fields: List[str] = []
    for radius in GUARD_RADII_CM:
        guard_fields.extend([
            f"background_db_g{radius}",
            f"residual_db_g{radius}",
            f"z_g{radius}",
        ])

    path = os.path.join(
        result_dir,
        f"points_local_anomaly_{analysis.time_folder}_center_{analysis.center_id}.csv",
    )
    write_csv(path, base_fields + guard_fields, rows)
    return path


def model_rows_for_analysis(analysis: SecondAnalysis) -> List[dict]:
    rows: List[dict] = []
    for guard_result in analysis.guard_results:
        for model_name in CANDIDATE_MODELS:
            rows.append({
                "time_folder": analysis.time_folder,
                "label": analysis.label,
                "center_id": analysis.center_id,
                "time_index": analysis.time_index,
                "guard_radius_cm": guard_result.guard_radius_cm,
                "model": model_name,
                "selected": "yes" if model_name == guard_result.selected_model else "no",
                "cv_median_absolute_error_db": guard_result.candidate_scores.get(
                    model_name,
                    float("nan"),
                ),
                "selected_residual_scale_db": guard_result.residual_scale_db,
            })
    return rows


def summary_row_for_analysis(analysis: SecondAnalysis) -> dict:
    x_cm = np.asarray([point.x_cm for point in analysis.points], dtype=np.float64)
    y_cm = np.asarray([point.y_cm for point in analysis.points], dtype=np.float64)
    center_candidates = np.flatnonzero((np.abs(x_cm) < 1e-9) & (np.abs(y_cm) < 1e-9))
    center_index = int(center_candidates[0]) if center_candidates.size else 0

    finite_z = np.where(np.isfinite(analysis.consensus_z), analysis.consensus_z, -np.inf)
    peak_index = int(np.argmax(finite_z))
    stable_count = int(np.sum(analysis.stable_mask))

    return {
        "time_folder": analysis.time_folder,
        "label": analysis.label,
        "center_id": analysis.center_id,
        "time_index": analysis.time_index,
        "segment_start_second": analysis.segment_start_second,
        "segment_end_second": analysis.segment_end_second,
        "number_of_points": len(analysis.points),
        "center_raw_db": analysis.points[center_index].energy_db,
        "center_background_db": analysis.consensus_background_db[center_index],
        "center_residual_db": analysis.consensus_residual_db[center_index],
        "center_z": analysis.consensus_z[center_index],
        "center_guard_support": analysis.guard_support[center_index],
        "center_guard_disagreement_z": analysis.guard_disagreement_z[center_index],
        "center_stable_anomaly": "yes" if analysis.stable_mask[center_index] else "no",
        "peak_z": analysis.consensus_z[peak_index],
        "peak_x_cm": x_cm[peak_index],
        "peak_y_cm": y_cm[peak_index],
        "peak_distance_from_center_cm": float(np.hypot(x_cm[peak_index], y_cm[peak_index])),
        "stable_anomaly_point_count": stable_count,
        "selected_models": ";".join(
            f"g{item.guard_radius_cm}:{item.selected_model}"
            for item in analysis.guard_results
        ),
        "model_cv_mae_db": ";".join(
            f"g{item.guard_radius_cm}:{item.model_cv_mae_db:.4f}"
            for item in analysis.guard_results
        ),
    }


# ============================================================
# 8. 绘图
# ============================================================

def finite_percentile(values: np.ndarray, percentile: float, fallback: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(fallback)
    return float(np.percentile(values, percentile))


def interpolate_for_display(
    x_cm: np.ndarray,
    y_cm: np.ndarray,
    values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """线性插值仅用于显示，任何指标都不使用插值网格。"""
    grid_x, grid_y = np.mgrid[
        GRID_MIN_CM:GRID_MAX_CM:complex(GRID_SIZE),
        GRID_MIN_CM:GRID_MAX_CM:complex(GRID_SIZE),
    ]
    valid = np.isfinite(x_cm) & np.isfinite(y_cm) & np.isfinite(values)
    if np.sum(valid) < 4:
        grid_z = np.full_like(grid_x, np.nan, dtype=np.float64)
        return grid_x, grid_y, grid_z

    try:
        grid_z = griddata(
            (x_cm[valid], y_cm[valid]),
            values[valid],
            (grid_x, grid_y),
            method="linear",
            fill_value=np.nan,
        )
    except Exception:
        grid_z = griddata(
            (x_cm[valid], y_cm[valid]),
            values[valid],
            (grid_x, grid_y),
            method="nearest",
            fill_value=np.nan,
        )

    grid_radius = np.sqrt(grid_x * grid_x + grid_y * grid_y)
    grid_z[grid_radius > MAX_DISTANCE_CM] = np.nan
    return grid_x, grid_y, grid_z


def draw_spatial_panel(
    axis: plt.Axes,
    x_cm: np.ndarray,
    y_cm: np.ndarray,
    values: np.ndarray,
    title: str,
    cmap_name: str,
    vmin: float,
    vmax: float,
    colorbar_label: str,
    stable_mask: Optional[np.ndarray] = None,
) -> None:
    _, _, grid_z = interpolate_for_display(x_cm, y_cm, values)
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
        x_cm,
        y_cm,
        c=values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=14,
        edgecolors="black",
        linewidths=0.25,
        zorder=3,
    )
    if stable_mask is not None and np.any(stable_mask):
        axis.scatter(
            x_cm[stable_mask],
            y_cm[stable_mask],
            facecolors="none",
            edgecolors="lime",
            s=65,
            linewidths=1.2,
            zorder=5,
            label="stable anomaly",
        )
        axis.legend(loc="upper right", fontsize=7)

    axis.scatter(
        [0.0], [0.0], marker="*", s=120, c="white",
        edgecolors="black", linewidths=0.8, zorder=6,
    )
    boundary = plt.Circle(
        (0.0, 0.0),
        MAX_DISTANCE_CM,
        fill=False,
        linestyle=":",
        color="black",
        linewidth=0.8,
    )
    axis.add_patch(boundary)
    axis.set_title(title, fontsize=10)
    axis.set_xlim(GRID_MIN_CM, GRID_MAX_CM)
    axis.set_ylim(GRID_MIN_CM, GRID_MAX_CM)
    axis.set_xlabel("X Distance (cm)")
    axis.set_ylabel("Y Distance (cm)")
    axis.grid(True, linestyle="--", alpha=0.25)
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label=colorbar_label)


def calculate_folder_scales(analyses: Sequence[SecondAnalysis]) -> Dict[str, float]:
    raw = np.concatenate([
        np.asarray([point.energy_db for point in analysis.points])
        for analysis in analyses
    ])
    background = np.concatenate([
        analysis.consensus_background_db for analysis in analyses
    ])
    residual = np.concatenate([
        analysis.consensus_residual_db for analysis in analyses
    ])
    disagreement = np.concatenate([
        analysis.guard_disagreement_z for analysis in analyses
    ])

    raw_and_background = np.concatenate([raw, background])
    raw_min = finite_percentile(raw_and_background, 1.0, -120.0)
    raw_max = finite_percentile(raw_and_background, 99.0, 0.0)
    if raw_max <= raw_min:
        raw_max = raw_min + 1.0

    residual_limit = finite_percentile(np.abs(residual), 99.0, 1.0)
    residual_limit = max(residual_limit, 0.5)
    disagreement_max = finite_percentile(disagreement, 99.0, 1.0)
    disagreement_max = max(disagreement_max, 0.5)

    return {
        "raw_min": raw_min,
        "raw_max": raw_max,
        "residual_limit": residual_limit,
        "disagreement_max": disagreement_max,
    }


def plot_second_analysis(
    result_dir: str,
    analysis: SecondAnalysis,
    scales: Dict[str, float],
) -> str:
    x_cm = np.asarray([point.x_cm for point in analysis.points], dtype=np.float64)
    y_cm = np.asarray([point.y_cm for point in analysis.points], dtype=np.float64)
    raw_db = np.asarray([point.energy_db for point in analysis.points], dtype=np.float64)

    figure, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=110)
    draw_spatial_panel(
        axes[0, 0], x_cm, y_cm, raw_db,
        "A. Raw 50-70 kHz energy",
        RAW_CMAP, scales["raw_min"], scales["raw_max"], "Energy (dB)",
    )
    draw_spatial_panel(
        axes[0, 1], x_cm, y_cm, analysis.consensus_background_db,
        "B. Predicted spatial background",
        RAW_CMAP, scales["raw_min"], scales["raw_max"], "Predicted background (dB)",
    )
    draw_spatial_panel(
        axes[0, 2], x_cm, y_cm, analysis.consensus_residual_db,
        "C. Signed residual (negative values retained)",
        SIGNED_CMAP, -scales["residual_limit"], scales["residual_limit"],
        "Residual (dB)", analysis.stable_mask,
    )
    draw_spatial_panel(
        axes[1, 0], x_cm, y_cm, analysis.consensus_z,
        "D. Standardized local anomaly",
        SIGNED_CMAP, -Z_DISPLAY_LIMIT, Z_DISPLAY_LIMIT,
        "Z score", analysis.stable_mask,
    )
    draw_spatial_panel(
        axes[1, 1], x_cm, y_cm, analysis.guard_support,
        f"E. Guard support (Z >= {Z_THRESHOLD:g})",
        SUPPORT_CMAP, 0.0, 1.0, "Support fraction",
        analysis.stable_mask,
    )
    draw_spatial_panel(
        axes[1, 2], x_cm, y_cm, analysis.guard_disagreement_z,
        "F. Disagreement among guard radii",
        UNCERTAINTY_CMAP, 0.0, scales["disagreement_max"],
        "Robust disagreement (Z)",
    )

    model_text = ", ".join(
        f"g{item.guard_radius_cm}:{item.selected_model}"
        for item in analysis.guard_results
    )
    figure.suptitle(
        f"Local anomaly morphology | {analysis.time_folder} | "
        f"label={analysis.label} | center={analysis.center_id} | "
        f"[{analysis.segment_start_second:.0f}, {analysis.segment_end_second:.0f}) s\n"
        f"models: {model_text}",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))

    path = os.path.join(
        result_dir,
        f"local_anomaly_{analysis.time_folder}_center_{analysis.center_id}.png",
    )
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)
    return path


# ============================================================
# 9. 多秒稳定形态
# ============================================================

def aggregate_temporal_points(
    analyses: Sequence[SecondAnalysis],
) -> Tuple[List[SpatialPoint], Dict[str, np.ndarray]]:
    """按point_key对齐不同秒；只聚合每秒都存在的空间点。"""
    if not analyses:
        return [], {}

    point_maps = [
        {point.point_key: index for index, point in enumerate(analysis.points)}
        for analysis in analyses
    ]
    common_keys = set(point_maps[0])
    for point_map in point_maps[1:]:
        common_keys &= set(point_map)

    ordered_keys = [
        point.point_key
        for point in analyses[0].points
        if point.point_key in common_keys
    ]
    template_points = [
        analyses[0].points[point_maps[0][key]]
        for key in ordered_keys
    ]

    raw_stack = []
    background_stack = []
    residual_stack = []
    z_stack = []
    support_stack = []
    disagreement_stack = []

    for analysis, point_map in zip(analyses, point_maps):
        indices = np.asarray([point_map[key] for key in ordered_keys], dtype=int)
        raw_stack.append(np.asarray([
            analysis.points[index].energy_db for index in indices
        ]))
        background_stack.append(analysis.consensus_background_db[indices])
        residual_stack.append(analysis.consensus_residual_db[indices])
        z_stack.append(analysis.consensus_z[indices])
        support_stack.append(analysis.guard_support[indices])
        disagreement_stack.append(analysis.guard_disagreement_z[indices])

    raw_array = np.vstack(raw_stack)
    background_array = np.vstack(background_stack)
    residual_array = np.vstack(residual_stack)
    z_array = np.vstack(z_stack)
    support_array = np.vstack(support_stack)
    disagreement_array = np.vstack(disagreement_stack)

    values = {
        "raw_db": np.nanmedian(raw_array, axis=0),
        "background_db": np.nanmedian(background_array, axis=0),
        "residual_db": np.nanmedian(residual_array, axis=0),
        "z": np.nanmedian(z_array, axis=0),
        "guard_support": np.nanmean(support_array, axis=0),
        "guard_disagreement_z": np.nanmedian(disagreement_array, axis=0),
        "time_support": np.nanmean(z_array >= Z_THRESHOLD, axis=0),
        "time_disagreement_z": 1.4826 * np.nanmedian(
            np.abs(z_array - np.nanmedian(z_array, axis=0)[None, :]),
            axis=0,
        ),
    }
    values["stable_mask"] = (
        (values["z"] >= Z_THRESHOLD)
        & (values["guard_support"] >= MIN_GUARD_SUPPORT)
        & (values["time_support"] >= MIN_GUARD_SUPPORT)
        & (values["guard_disagreement_z"] <= MAX_GUARD_DISAGREEMENT_Z)
    )
    return template_points, values


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
            "median_raw_db": values["raw_db"][index],
            "median_background_db": values["background_db"][index],
            "median_residual_db": values["residual_db"][index],
            "median_z": values["z"][index],
            "mean_guard_support": values["guard_support"][index],
            "time_support": values["time_support"][index],
            "median_guard_disagreement_z": values["guard_disagreement_z"][index],
            "time_disagreement_z": values["time_disagreement_z"][index],
            "stable_across_time_and_guards": (
                "yes" if values["stable_mask"][index] else "no"
            ),
        })

    fields = list(rows[0].keys()) if rows else []
    path = os.path.join(result_dir, f"temporal_stability_{time_folder}.csv")
    if rows:
        write_csv(path, fields, rows)
    return path


def plot_temporal_stability(
    result_dir: str,
    time_folder: str,
    label: str,
    points: Sequence[SpatialPoint],
    values: Dict[str, np.ndarray],
    number_of_seconds: int,
    scales: Dict[str, float],
) -> str:
    x_cm = np.asarray([point.x_cm for point in points], dtype=np.float64)
    y_cm = np.asarray([point.y_cm for point in points], dtype=np.float64)
    stable_mask = np.asarray(values["stable_mask"], dtype=bool)

    figure, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=110)
    draw_spatial_panel(
        axes[0, 0], x_cm, y_cm, values["raw_db"],
        "A. Median raw energy across seconds",
        RAW_CMAP, scales["raw_min"], scales["raw_max"], "Energy (dB)",
    )
    draw_spatial_panel(
        axes[0, 1], x_cm, y_cm, values["background_db"],
        "B. Median predicted background",
        RAW_CMAP, scales["raw_min"], scales["raw_max"], "Background (dB)",
    )
    draw_spatial_panel(
        axes[0, 2], x_cm, y_cm, values["residual_db"],
        "C. Median signed residual",
        SIGNED_CMAP, -scales["residual_limit"], scales["residual_limit"],
        "Residual (dB)", stable_mask,
    )
    draw_spatial_panel(
        axes[1, 0], x_cm, y_cm, values["z"],
        "D. Median standardized anomaly",
        SIGNED_CMAP, -Z_DISPLAY_LIMIT, Z_DISPLAY_LIMIT,
        "Z score", stable_mask,
    )
    draw_spatial_panel(
        axes[1, 1], x_cm, y_cm, values["time_support"],
        f"E. Time support (Z >= {Z_THRESHOLD:g})",
        SUPPORT_CMAP, 0.0, 1.0, "Fraction of seconds", stable_mask,
    )
    temporal_disagreement_max = max(
        finite_percentile(values["time_disagreement_z"], 99.0, 1.0),
        0.5,
    )
    draw_spatial_panel(
        axes[1, 2], x_cm, y_cm, values["time_disagreement_z"],
        "F. Variation among seconds",
        UNCERTAINTY_CMAP, 0.0, temporal_disagreement_max,
        "Temporal disagreement (Z)",
    )

    figure.suptitle(
        f"Temporal stable local anomaly | {time_folder} | "
        f"label={label} | seconds={number_of_seconds}\n"
        "Green circles = stable across guard radii and time",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(result_dir, f"temporal_stability_{time_folder}.png")
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)
    return path


# ============================================================
# 10. 主处理流程
# ============================================================

def process_time_folder(time_folder: str) -> List[SecondAnalysis]:
    print("\n" + "=" * 88)
    print(f"处理：{time_folder}（多参考点留邻域背景预测）")
    print("=" * 88)

    center_data_dir = resolve_time_data_dir(center_root_dir, time_folder)
    near_data_dir = resolve_time_data_dir(offset_near_root_dir, time_folder)
    far_data_dir = resolve_time_data_dir(offset_far_root_dir, time_folder)

    required_dirs = [
        ("中心点", center_data_dir),
        ("5~40cm偏移点", near_data_dir),
        ("45~80cm偏移点", far_data_dir),
    ]
    missing = False
    for description, folder in required_dirs:
        if not os.path.isdir(folder):
            print(f"错误：{description}文件夹不存在：{folder}")
            missing = True
    if missing:
        return []

    center_index = build_center_file_index(center_data_dir)
    near_index = build_offset_file_index(near_data_dir)
    far_index = build_offset_file_index(far_data_dir)
    offset_index = merge_offset_file_indices(near_index, far_index)
    center_ids = sorted(center_index.keys(), key=center_id_sort_key)

    print(f"中心秒编号：{center_ids}")
    print(f"中心文件组数：{len(center_index)}")
    print(f"5~40cm测点组数：{len(near_index)}")
    print(f"45~80cm测点组数：{len(far_index)}")

    if not center_ids:
        print("错误：没有找到中心点WAV")
        return []

    label = TIME_FOLDER_LABELS.get(time_folder, "UNLABELED")
    result_dir = os.path.join(result_root_dir, time_folder)
    os.makedirs(result_dir, exist_ok=True)

    analyses: List[SecondAnalysis] = []
    model_rows: List[dict] = []
    summary_rows: List[dict] = []

    for center_id in center_ids:
        try:
            time_index, start_second, end_second = get_time_slice_from_center_id(
                center_id
            )
        except ValueError as exc:
            print(f"错误：{exc}；跳过{center_id}")
            continue

        print("\n" + "-" * 72)
        print(
            f"中心{center_id}：读取所有坐标的"
            f"[{start_second:.0f}, {end_second:.0f})秒"
        )

        points = collect_spatial_points(
            center_id,
            center_index,
            offset_index,
            start_second,
            end_second,
        )
        print(f"  有效空间测点：{len(points)}/129")

        if len(points) < 60:
            print("  错误：有效测点少于60，无法可靠建模，已跳过")
            continue

        try:
            analysis = analyze_points(
                time_folder,
                label,
                center_id,
                time_index,
                start_second,
                end_second,
                points,
            )
        except Exception as exc:
            print(f"  错误：空间背景分析失败：{exc}")
            continue

        analyses.append(analysis)
        point_csv = save_second_point_csv(result_dir, analysis)
        model_rows.extend(model_rows_for_analysis(analysis))
        summary = summary_row_for_analysis(analysis)
        summary_rows.append(summary)

        print(
            f"  中心结果：residual={summary['center_residual_db']:.3f}dB，"
            f"Z={summary['center_z']:.3f}，"
            f"guard_support={summary['center_guard_support']:.2f}，"
            f"stable={summary['center_stable_anomaly']}"
        )
        print(f"  测点CSV：{point_csv}")

    if not analyses:
        print(f"{time_folder}没有成功结果")
        return []

    model_fields = [
        "time_folder", "label", "center_id", "time_index",
        "guard_radius_cm", "model", "selected",
        "cv_median_absolute_error_db", "selected_residual_scale_db",
    ]
    write_csv(
        os.path.join(result_dir, f"model_selection_{time_folder}.csv"),
        model_fields,
        model_rows,
    )

    summary_fields = list(summary_rows[0].keys())
    write_csv(
        os.path.join(result_dir, f"summary_{time_folder}.csv"),
        summary_fields,
        summary_rows,
    )

    scales = calculate_folder_scales(analyses)
    if DRAW_INDIVIDUAL_SECOND_FIGURES:
        for analysis in analyses:
            figure_path = plot_second_analysis(result_dir, analysis, scales)
            print(f"单秒图：{figure_path}")

    temporal_points, temporal_values = aggregate_temporal_points(analyses)
    if temporal_points:
        temporal_csv = save_temporal_csv(
            result_dir,
            time_folder,
            label,
            temporal_points,
            temporal_values,
            len(analyses),
        )
        print(f"多秒稳定CSV：{temporal_csv}")
        if DRAW_TEMPORAL_STABILITY_FIGURE:
            temporal_figure = plot_temporal_stability(
                result_dir,
                time_folder,
                label,
                temporal_points,
                temporal_values,
                len(analyses),
                scales,
            )
            print(f"多秒稳定图：{temporal_figure}")

    return analyses


def validate_configuration() -> None:
    if FREQ_LOW < 0 or FREQ_HIGH <= FREQ_LOW:
        raise ValueError("FREQ_LOW/FREQ_HIGH设置错误")
    if not GUARD_RADII_CM:
        raise ValueError("GUARD_RADII_CM不能为空")
    if any(radius <= 0 for radius in GUARD_RADII_CM):
        raise ValueError("保护半径必须大于0")
    unknown_models = [
        name
        for name in CANDIDATE_MODELS
        if name not in {"constant", "plane", "quadratic", "rbf_35", "rbf_55"}
    ]
    if unknown_models:
        raise ValueError(f"未知候选模型：{unknown_models}")
    if not (0.0 <= MIN_GUARD_SUPPORT <= 1.0):
        raise ValueError("MIN_GUARD_SUPPORT必须在0~1之间")


def main() -> None:
    validate_configuration()
    os.makedirs(result_root_dir, exist_ok=True)

    print("=" * 88)
    print("GC局部异常声源形态：多参考点留邻域背景预测")
    print("=" * 88)
    print(f"频带：{FREQ_LOW / 1000:.1f}-{FREQ_HIGH / 1000:.1f} kHz")
    print(f"普通测点：中心 + 8方向 * {len(DISTANCES_CM)}距离 = 129点")
    print(f"保护半径：{list(GUARD_RADII_CM)} cm")
    print(f"候选模型：{list(CANDIDATE_MODELS)}")
    print("40cm和80cm都不再被指定为背景点。")
    print("科学计算只使用实测点；插值只负责画图。")

    total_success = 0
    for time_folder in time_folders:
        total_success += len(process_time_folder(time_folder))

    print("\n" + "=" * 88)
    print(f"全部完成。成功分析中心秒数：{total_success}")
    print(f"结果目录：{result_root_dir}")
    print("=" * 88)


if __name__ == "__main__":
    main()
