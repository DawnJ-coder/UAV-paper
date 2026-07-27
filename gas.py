# -*- coding: utf-8 -*-
"""
Leak morphology heatmap generator
--------------------------------
作用：
1. 严格按 center_id 的秒编号，从同一段原始阵列录音的同一秒读取所有坐标的 beamform WAV
2. 对每个短时帧做频谱分析
3. 用外围点做空间背景平面拟合，得到更真实的空间残差
4. 统计稳定热点，输出更清晰的泄漏形态热力图

输出：
- heatmap_final_*.png              最终形态图（推荐重点看）
- heatmap_persistence_*.png        热点持续率图
- heatmap_normalized_shape_*.png   归一化形态图
- point_scores_*.csv               每个测点分数
- summary_metrics.csv              每个中心点摘要
"""

import csv
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # 服务器/无界面环境下也能保存图片
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
from scipy import ndimage
from scipy.interpolate import griddata


# ============================================================
# 1. 配置参数
# ============================================================

# 需要处理的时间点文件夹
time_folders = [
    "HM20260702_111044.ld",
]

# 根目录路径（按你本地实际路径修改）
center_root_dir = r"D:\gas\beamform_results_sh"
offset_root_dir = r"D:\gas\beamform_results_offset_multiple_sh"

# 是否递归搜索子文件夹
RECURSIVE_SEARCH = False

# 期望采样率（只用于提醒，不强制）
EXPECTED_SAMPLE_RATE = 192000

# 时间切片：center00_00 -> [0,1)秒；center00_01 -> [1,2)秒
TIME_SLICE_SECONDS = 1.0
STRICT_TIME_SLICE = True

# 8个方向对应角度
direction_angles = {
    "up": np.pi / 2,
    "down": -np.pi / 2,
    "left": np.pi,
    "right": 0.0,
    "up_left": 3 * np.pi / 4,
    "down_left": -3 * np.pi / 4,
    "up_right": np.pi / 4,
    "down_right": -np.pi / 4,
}

# 距离（cm）
distances = [5, 10, 15, 20, 25, 30, 35, 40]

# 频段设置：如果你以后想改成20-80kHz，就改这里
FREQ_LOW = 50000
FREQ_HIGH = 70000

# 短时帧参数
FRAME_SECONDS = 0.10
HOP_SECONDS = 0.05

# Welch参数
WELCH_NPERSEG = 2048
WELCH_OVERLAP_RATIO = 0.5

# 背景拟合：使用外围这些点做平面拟合
BACKGROUND_MIN_DISTANCE_CM = 25

# 稳健拟合参数
HUBER_K = 1.5
ROBUST_EPS = 1e-12

# 频率连续性平滑
FREQ_WEIGHT_SMOOTH_BINS = 5
FREQ_WEIGHT_THRESHOLD = 0.75  # 越大越严格

# 持续率判定阈值：score > median(outer)+k*MAD(outer)
PERSISTENCE_SIGMA = 1.5

# 网格热力图参数
GRID_MIN_CM = -45
GRID_MAX_CM = 45
GRID_SIZE = 260
GRID_SMOOTH_SIGMA = 1.2
HEATMAP_CMAP = "turbo"

# 形态计算阈值
SHAPE_THRESHOLD_SIGMA = 1.0

# 数值保护
PSD_EPS = 1e-20


# ============================================================
# 2. 文件名解析和索引
# ============================================================

SpectrumFrames = Tuple[np.ndarray, np.ndarray]  # (freqs, psd_frames[n_frames, n_freq])
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
    rf"(?P<center>\d+(?:_\d+)?)d(?P<distance>\d+)_(?P<direction>{_DIRECTION_REGEX_PART})"
    rf"(?P<suffix>.*?)\.wav$",
    flags=re.IGNORECASE,
)


def list_wav_files(folder: str) -> List[str]:
    if RECURSIVE_SEARCH:
        pattern = os.path.join(folder, "**", "*.wav")
        return sorted(glob.glob(pattern, recursive=True))
    pattern = os.path.join(folder, "*.wav")
    return sorted(glob.glob(pattern))


def build_center_file_index(center_data_dir: str) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    for file_path in list_wav_files(center_data_dir):
        filename = os.path.basename(file_path)
        match = CENTER_FILE_REGEX.search(filename)
        if not match:
            continue
        center_id = match.group("center")
        index.setdefault(center_id, []).append(file_path)

    for center_id in index:
        index[center_id] = sorted(index[center_id])
    return index


def build_offset_file_index(offset_data_dir: str) -> Dict[OffsetKey, List[str]]:
    index: Dict[OffsetKey, List[str]] = {}
    for file_path in list_wav_files(offset_data_dir):
        filename = os.path.basename(file_path)
        match = OFFSET_FILE_REGEX.search(filename)
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


def choose_first_file(files: Optional[List[str]], description: str) -> Optional[str]:
    if not files:
        return None
    if len(files) > 1:
        print(f"  警告：{description}匹配到{len(files)}个文件，使用第一个：")
        for file_path in files:
            print(f"    - {os.path.basename(file_path)}")
    return files[0]


def center_id_sort_key(center_id: str) -> Tuple[int, ...]:
    parts = str(center_id).split("_")
    try:
        return tuple(int(part) for part in parts)
    except ValueError:
        return (10**9,)


def get_time_slice_from_center_id(center_id: str) -> Tuple[int, float, float]:
    parts = str(center_id).split("_")
    if len(parts) < 2 or not parts[-1].isdigit():
        raise ValueError(
            f"center_id={center_id!r} 不符合 '00_00' 格式，无法确定要读取第几秒"
        )
    time_index = int(parts[-1])
    start_second = time_index * float(TIME_SLICE_SECONDS)
    end_second = start_second + float(TIME_SLICE_SECONDS)
    return time_index, start_second, end_second


# ============================================================
# 3. WAV读取和短时频谱
# ============================================================

def convert_wav_to_float(y: np.ndarray) -> np.ndarray:
    if np.issubdtype(y.dtype, np.integer):
        info = np.iinfo(y.dtype)
        full_scale = float(max(abs(info.min), abs(info.max)))
        if full_scale <= 0:
            raise ValueError(f"无效整数WAV类型：{y.dtype}")
        return y.astype(np.float64) / full_scale

    if np.issubdtype(y.dtype, np.floating):
        return y.astype(np.float64)

    raise TypeError(f"不支持的WAV数据类型：{y.dtype}")


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
        print(f"  错误：读取WAV失败：{file_path}\n    {exc}")
        return None, None

    if y.ndim > 1:
        y = y[:, 0]

    try:
        y = convert_wav_to_float(y)
    except Exception as exc:
        print(f"  错误：WAV数据转换失败：{file_path}\n    {exc}")
        return None, None

    total_samples = int(y.size)
    start_sample = int(round(segment_start_second * sample_rate))
    end_sample = int(round(segment_end_second * sample_rate))

    if start_sample >= total_samples:
        print(
            f"  错误：{os.path.basename(file_path)} 总时长不足，"
            f"无法读取 [{segment_start_second:.3f}, {segment_end_second:.3f}) 秒"
        )
        return None, None

    if end_sample > total_samples:
        if STRICT_TIME_SLICE:
            print(
                f"  错误：{os.path.basename(file_path)} 在目标区间 "
                f"[{segment_start_second:.3f}, {segment_end_second:.3f}) 秒不完整，已跳过"
            )
            return None, None
        end_sample = total_samples

    segment = y[start_sample:end_sample]
    expected_samples = int(round((segment_end_second - segment_start_second) * sample_rate))
    if STRICT_TIME_SLICE and segment.size != expected_samples:
        print(
            f"  错误：{os.path.basename(file_path)} 切片长度不正确，"
            f"实际={segment.size}, 期望={expected_samples}"
        )
        return None, None

    segment = np.nan_to_num(segment, nan=0.0, posinf=0.0, neginf=0.0)
    segment = segment - np.mean(segment)

    if sample_rate != EXPECTED_SAMPLE_RATE:
        print(
            f"  提醒：{os.path.basename(file_path)} 采样率={sample_rate}Hz，"
            f"不是期望的 {EXPECTED_SAMPLE_RATE}Hz"
        )

    return sample_rate, segment


def compute_frame_psd_matrix(
    y: np.ndarray,
    sample_rate: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    返回：
    freqs:      [n_freq]
    psd_frames: [n_frames, n_freq]
    frame_centers_sec: [n_frames]
    """
    if y is None or y.size < 16:
        return None, None, None

    frame_len = int(round(FRAME_SECONDS * sample_rate))
    hop_len = int(round(HOP_SECONDS * sample_rate))

    if frame_len < 16 or hop_len < 1:
        return None, None, None

    if y.size < frame_len:
        return None, None, None

    starts = np.arange(0, y.size - frame_len + 1, hop_len, dtype=int)
    if starts.size == 0:
        return None, None, None

    nperseg = min(WELCH_NPERSEG, frame_len)
    noverlap = min(int(nperseg * WELCH_OVERLAP_RATIO), nperseg - 1)
    nyquist = sample_rate / 2.0

    if FREQ_LOW >= nyquist:
        print(f"  错误：奈奎斯特频率仅 {nyquist:.1f}Hz，无法分析 {FREQ_LOW}-{FREQ_HIGH}Hz")
        return None, None, None

    all_psd = []
    frame_centers_sec = []

    ref_freqs = None

    for start in starts:
        frame = y[start:start + frame_len]
        frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
        frame = frame - np.mean(frame)

        if frame.size < 16:
            continue

        freqs, psd = signal.welch(
            frame,
            fs=sample_rate,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="density",
        )

        actual_high = min(FREQ_HIGH, nyquist)
        mask = (freqs >= FREQ_LOW) & (freqs <= actual_high)
        freqs = freqs[mask]
        psd = np.maximum(np.nan_to_num(psd[mask], nan=0.0, posinf=0.0, neginf=0.0), 0.0)

        if freqs.size < 2:
            continue

        if ref_freqs is None:
            ref_freqs = freqs.copy()
            all_psd.append(psd.copy())
        else:
            if ref_freqs.shape == freqs.shape and np.allclose(ref_freqs, freqs):
                all_psd.append(psd.copy())
            else:
                psd_interp = np.interp(ref_freqs, freqs, psd)
                all_psd.append(psd_interp)

        frame_center = (start + frame_len / 2.0) / float(sample_rate)
        frame_centers_sec.append(frame_center)

    if ref_freqs is None or not all_psd:
        return None, None, None

    return ref_freqs, np.vstack(all_psd), np.asarray(frame_centers_sec, dtype=np.float64)


# ============================================================
# 4. 数学工具
# ============================================================

def robust_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)


def robust_plane_fit(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Huber IRLS 拟合平面：
    z = a + b*x + c*y
    返回 beta = [a, b, c]
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    X = np.column_stack([np.ones_like(x), x, y])

    beta, *_ = np.linalg.lstsq(X, z, rcond=None)

    for _ in range(25):
        residual = z - X @ beta
        scale = robust_mad(residual) + ROBUST_EPS

        u = np.abs(residual) / (HUBER_K * scale)
        w = np.ones_like(u)
        mask = u > 1.0
        w[mask] = 1.0 / u[mask]

        sqrt_w = np.sqrt(w)
        Xw = X * sqrt_w[:, None]
        zw = z * sqrt_w

        beta_new, *_ = np.linalg.lstsq(Xw, zw, rcond=None)

        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break

        beta = beta_new

    return beta


def moving_average_1d(x: np.ndarray, win: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or win <= 1:
        return x.copy()
    kernel = np.ones(int(win), dtype=np.float64) / float(win)
    return np.convolve(x, kernel, mode="same")


# ============================================================
# 5. 读取一个中心点的全部空间测点
# ============================================================

def build_point_records_for_center(
    center_id: str,
    center_index: Dict[str, List[str]],
    offset_index: Dict[OffsetKey, List[str]],
) -> List[dict]:
    records = []

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

    for direction, angle in direction_angles.items():
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
                "distance_cm": distance,
                "x_cm": x,
                "y_cm": y,
                "file_path": file_path,
            })

    return records


def load_all_point_frame_psd(
    point_records: List[dict],
    segment_start_second: float,
    segment_end_second: float,
) -> Tuple[List[dict], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    返回：
    valid_point_records
    freqs             [n_freq]
    log_psd_data      [n_points, n_frames, n_freq]
    frame_centers_sec [n_frames]
    """
    valid_records = []
    ref_freqs = None
    ref_frame_times = None
    log_psd_list = []

    for record in point_records:
        file_path = record["file_path"]
        sample_rate, y = read_wav_segment(file_path, segment_start_second, segment_end_second)
        if sample_rate is None or y is None:
            continue

        freqs, psd_frames, frame_centers_sec = compute_frame_psd_matrix(y, sample_rate)
        if freqs is None or psd_frames is None or frame_centers_sec is None:
            continue

        log_psd_frames = 10.0 * np.log10(psd_frames + PSD_EPS)

        if ref_freqs is None:
            ref_freqs = freqs.copy()
            ref_frame_times = frame_centers_sec.copy()
            valid_records.append(record)
            log_psd_list.append(log_psd_frames)
        else:
            # 频率轴对齐
            if not (ref_freqs.shape == freqs.shape and np.allclose(ref_freqs, freqs)):
                aligned = []
                for frame_idx in range(log_psd_frames.shape[0]):
                    aligned.append(np.interp(ref_freqs, freqs, log_psd_frames[frame_idx]))
                log_psd_frames = np.vstack(aligned)

            # 帧数对齐：取共同最小帧数
            common_frames = min(ref_frame_times.size, frame_centers_sec.size, log_psd_frames.shape[0])

            if common_frames < 3:
                continue

            if len(log_psd_list) == 0:
                continue

            # 裁剪已保存内容
            if common_frames < ref_frame_times.size:
                ref_frame_times = ref_frame_times[:common_frames]
                for i in range(len(log_psd_list)):
                    log_psd_list[i] = log_psd_list[i][:common_frames, :]

            log_psd_frames = log_psd_frames[:common_frames, :]

            valid_records.append(record)
            log_psd_list.append(log_psd_frames)

    if ref_freqs is None or ref_frame_times is None or len(log_psd_list) == 0:
        return [], None, None, None

    # 统一帧数
    min_frames = min(arr.shape[0] for arr in log_psd_list)
    if min_frames < 3:
        return [], None, None, None

    ref_frame_times = ref_frame_times[:min_frames]
    log_psd_list = [arr[:min_frames, :] for arr in log_psd_list]
    log_psd_data = np.stack(log_psd_list, axis=0)  # [n_points, n_frames, n_freq]

    return valid_records, ref_freqs, log_psd_data, ref_frame_times


# ============================================================
# 6. 逐帧空间形态评分
# ============================================================

def compute_frame_scores(
    point_records: List[dict],
    freqs: np.ndarray,
    log_psd_data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    输入：
    log_psd_data: [n_points, n_frames, n_freq]

    返回：
    frame_scores:      [n_frames, n_points]
    frame_binary_hot:  [n_frames, n_points]
    outer_mask:        [n_points]
    """
    n_points, n_frames, n_freq = log_psd_data.shape

    x = np.asarray([r["x_cm"] for r in point_records], dtype=np.float64)
    y = np.asarray([r["y_cm"] for r in point_records], dtype=np.float64)
    dist = np.asarray([r["distance_cm"] for r in point_records], dtype=np.float64)

    outer_mask = dist >= BACKGROUND_MIN_DISTANCE_CM
    if np.sum(outer_mask) < 6:
        raise RuntimeError(
            f"外围点太少（distance >= {BACKGROUND_MIN_DISTANCE_CM}cm 的点不足），无法稳定拟合背景平面"
        )

    frame_scores = np.zeros((n_frames, n_points), dtype=np.float64)
    frame_binary_hot = np.zeros((n_frames, n_points), dtype=np.float64)

    for t in range(n_frames):
        # 当前帧所有点的 log-PSD: [n_points, n_freq]
        L = log_psd_data[:, t, :]

        # 保存当前帧每个频点标准化后的 z 值
        Z = np.zeros_like(L)

        for k in range(n_freq):
            z_outer = L[outer_mask, k]

            beta = robust_plane_fit(x[outer_mask], y[outer_mask], z_outer)
            bg_all = beta[0] + beta[1] * x + beta[2] * y

            residual = L[:, k] - bg_all

            outer_res = residual[outer_mask]
            med = np.median(outer_res)
            scale = robust_mad(outer_res) + ROBUST_EPS

            Z[:, k] = (residual - med) / scale

        Z_pos = np.maximum(Z, 0.0)

        # 频率加权：优先保留“连续频带异常”，抑制窄带随机尖峰
        peak_per_freq = np.percentile(Z_pos, 90, axis=0)
        smooth_peak = moving_average_1d(peak_per_freq, FREQ_WEIGHT_SMOOTH_BINS)
        weights = np.maximum(smooth_peak - FREQ_WEIGHT_THRESHOLD, 0.0)

        if np.sum(weights) <= 0:
            weights = np.ones(n_freq, dtype=np.float64)
        weights = weights / (np.sum(weights) + ROBUST_EPS)

        score = Z_pos @ weights  # [n_points]
        frame_scores[t, :] = score

        outer_score = score[outer_mask]
        thr = np.median(outer_score) + PERSISTENCE_SIGMA * (robust_mad(outer_score) + ROBUST_EPS)
        frame_binary_hot[t, :] = (score > thr).astype(np.float64)

    return frame_scores, frame_binary_hot, outer_mask


# ============================================================
# 7. 聚合形态结果
# ============================================================

def aggregate_shape_maps(
    frame_scores: np.ndarray,
    frame_binary_hot: np.ndarray,
    outer_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回：
    median_score   [n_points]
    persistence    [n_points]  0~1
    final_score    [n_points]
    """
    median_score = np.median(frame_scores, axis=0)
    persistence = np.mean(frame_binary_hot, axis=0)

    # 最终形态图：强度 × 稳定性
    final_score = median_score * persistence

    # 再做一次外圈基线去除，增强中心结构
    outer_final = final_score[outer_mask]
    med = np.median(outer_final)
    mad = robust_mad(outer_final) + ROBUST_EPS
    final_score = np.maximum(final_score - (med + 0.5 * mad), 0.0)

    return median_score, persistence, final_score


def compute_shape_metrics(
    x: np.ndarray,
    y: np.ndarray,
    final_score: np.ndarray,
    outer_mask: np.ndarray,
) -> dict:
    """
    计算：
    - 峰值位置
    - 质心
    - 主方向
    - 各向异性
    """
    metrics = {
        "peak_x_cm": np.nan,
        "peak_y_cm": np.nan,
        "peak_score": np.nan,
        "centroid_x_cm": np.nan,
        "centroid_y_cm": np.nan,
        "major_axis_deg": np.nan,
        "anisotropy": np.nan,
        "ellipse_major_sigma_cm": np.nan,
        "ellipse_minor_sigma_cm": np.nan,
        "center_to_peak_cm": np.nan,
        "center_to_centroid_cm": np.nan,
        "shape_total_score": np.nan,
    }

    if final_score.size == 0 or np.max(final_score) <= 0:
        return metrics

    peak_idx = int(np.argmax(final_score))
    metrics["peak_x_cm"] = float(x[peak_idx])
    metrics["peak_y_cm"] = float(y[peak_idx])
    metrics["peak_score"] = float(final_score[peak_idx])
    metrics["center_to_peak_cm"] = float(np.hypot(x[peak_idx], y[peak_idx]))

    outer = final_score[outer_mask]
    med = np.median(outer)
    mad = robust_mad(outer) + ROBUST_EPS
    threshold = med + SHAPE_THRESHOLD_SIGMA * mad

    w = np.maximum(final_score - threshold, 0.0)
    sum_w = float(np.sum(w))

    metrics["shape_total_score"] = sum_w

    if sum_w <= 0:
        return metrics

    cx = float(np.sum(w * x) / sum_w)
    cy = float(np.sum(w * y) / sum_w)
    metrics["centroid_x_cm"] = cx
    metrics["centroid_y_cm"] = cy
    metrics["center_to_centroid_cm"] = float(np.hypot(cx, cy))

    dx = x - cx
    dy = y - cy

    cov_xx = float(np.sum(w * dx * dx) / sum_w)
    cov_xy = float(np.sum(w * dx * dy) / sum_w)
    cov_yy = float(np.sum(w * dy * dy) / sum_w)

    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(cov)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    l1 = max(float(eigvals[0]), 0.0)
    l2 = max(float(eigvals[1]), 0.0)

    main_vec = eigvecs[:, 0]
    angle_deg = float(np.degrees(np.arctan2(main_vec[1], main_vec[0])))

    anisotropy = (l1 - l2) / (l1 + l2 + ROBUST_EPS)

    metrics["major_axis_deg"] = angle_deg
    metrics["anisotropy"] = float(anisotropy)
    metrics["ellipse_major_sigma_cm"] = float(np.sqrt(l1))
    metrics["ellipse_minor_sigma_cm"] = float(np.sqrt(l2))

    return metrics


# ============================================================
# 8. 绘图
# ============================================================

def interpolate_to_grid(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_x, grid_y = np.mgrid[
        GRID_MIN_CM:GRID_MAX_CM:complex(GRID_SIZE),
        GRID_MIN_CM:GRID_MAX_CM:complex(GRID_SIZE),
    ]

    linear_grid = griddata(
        (x, y),
        values,
        (grid_x, grid_y),
        method="linear",
        fill_value=np.nan,
    )

    nearest_grid = griddata(
        (x, y),
        values,
        (grid_x, grid_y),
        method="nearest",
        fill_value=np.nan,
    )

    grid_z = np.where(np.isfinite(linear_grid), linear_grid, nearest_grid)

    # 限制在40cm半径内
    grid_r = np.sqrt(grid_x**2 + grid_y**2)
    grid_z[grid_r > max(distances)] = np.nan

    # 轻微平滑，让形态更清晰
    valid = np.isfinite(grid_z)
    if np.any(valid):
        filled = np.where(valid, grid_z, 0.0)
        smooth_num = ndimage.gaussian_filter(filled, sigma=GRID_SMOOTH_SIGMA)
        smooth_den = ndimage.gaussian_filter(valid.astype(np.float64), sigma=GRID_SMOOTH_SIGMA)
        good = smooth_den > 1e-6
        smoothed = np.full_like(grid_z, np.nan, dtype=np.float64)
        smoothed[good] = smooth_num[good] / smooth_den[good]
        grid_z = smoothed

    return grid_x, grid_y, grid_z


def plot_heatmap(
    save_path: str,
    title: str,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    colorbar_label: str,
    metrics: Optional[dict] = None,
    value_range: Optional[Tuple[float, float]] = None,
    draw_overlay: bool = True,
) -> None:
    grid_x, grid_y, grid_z = interpolate_to_grid(x, y, values)

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        print(f"警告：没有有效值，跳过绘图：{save_path}")
        return

    vmin, vmax = None, None
    if value_range is not None:
        vmin, vmax = value_range
    else:
        vmin = float(np.nanmin(finite_values))
        vmax = float(np.nanpercentile(finite_values, 98))
        if vmax <= vmin:
            vmax = vmin + 1.0

    fig, ax = plt.subplots(figsize=(8.6, 7.4), dpi=140)

    cmap = plt.get_cmap(HEATMAP_CMAP).copy()
    cmap.set_bad(color="white", alpha=0.0)

    im = ax.imshow(
        np.ma.masked_invalid(grid_z).T,
        extent=(GRID_MIN_CM, GRID_MAX_CM, GRID_MIN_CM, GRID_MAX_CM),
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )

    cb = fig.colorbar(im, ax=ax)
    cb.set_label(colorbar_label)

    # 实测点
    ax.scatter(
        x, y,
        c="white",
        s=18,
        edgecolors="black",
        linewidths=0.6,
        alpha=0.9,
        zorder=3,
        label="Measured points",
    )

    # 中心点
    ax.scatter(
        0.0, 0.0,
        c="red",
        marker="*",
        s=220,
        edgecolors="yellow",
        linewidths=1.0,
        zorder=5,
        label="Center",
    )

    # 峰值
    if values.size > 0 and np.max(values) > 0:
        peak_idx = int(np.argmax(values))
        ax.scatter(
            x[peak_idx],
            y[peak_idx],
            c="black",
            marker="x",
            s=120,
            linewidths=2.2,
            zorder=6,
            label="Peak point",
        )

    if draw_overlay and metrics is not None:
        cx = metrics.get("centroid_x_cm", np.nan)
        cy = metrics.get("centroid_y_cm", np.nan)
        angle_deg = metrics.get("major_axis_deg", np.nan)
        major_sigma = metrics.get("ellipse_major_sigma_cm", np.nan)
        minor_sigma = metrics.get("ellipse_minor_sigma_cm", np.nan)

        if np.isfinite(cx) and np.isfinite(cy):
            ax.scatter(
                cx, cy,
                c="lime",
                marker="o",
                s=90,
                edgecolors="black",
                linewidths=1.0,
                zorder=7,
                label="Centroid",
            )

        if (
            np.isfinite(cx) and np.isfinite(cy) and
            np.isfinite(angle_deg) and
            np.isfinite(major_sigma) and
            np.isfinite(minor_sigma) and
            major_sigma > 0 and minor_sigma >= 0
        ):
            # 2-sigma 椭圆
            ell = Ellipse(
                xy=(cx, cy),
                width=4.0 * max(minor_sigma, 0.5),
                height=4.0 * max(major_sigma, 0.5),
                angle=angle_deg,
                fill=False,
                edgecolor="cyan",
                linewidth=2.0,
                zorder=7,
                label="Shape ellipse",
            )
            ax.add_patch(ell)

            # 主方向箭头
            angle_rad = np.radians(angle_deg)
            arrow_len = max(major_sigma * 2.2, 4.0)
            dx = arrow_len * np.cos(angle_rad)
            dy = arrow_len * np.sin(angle_rad)
            ax.arrow(
                cx, cy, dx, dy,
                width=0.25,
                head_width=2.0,
                head_length=2.5,
                color="magenta",
                length_includes_head=True,
                zorder=8,
            )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("X Distance (cm)")
    ax.set_ylabel("Y Distance (cm)")
    ax.set_xlim(GRID_MIN_CM, GRID_MAX_CM)
    ax.set_ylim(GRID_MIN_CM, GRID_MAX_CM)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"热力图已保存：{save_path}")


# ============================================================
# 9. CSV输出
# ============================================================

def save_point_scores_csv(
    save_path: str,
    point_records: List[dict],
    median_score: np.ndarray,
    persistence: np.ndarray,
    final_score: np.ndarray,
) -> None:
    fieldnames = [
        "point_type", "direction", "distance_cm", "x_cm", "y_cm",
        "median_score", "persistence", "final_score", "file_name"
    ]

    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rec, med, per, fin in zip(point_records, median_score, persistence, final_score):
            writer.writerow({
                "point_type": rec["point_type"],
                "direction": rec["direction"],
                "distance_cm": rec["distance_cm"],
                "x_cm": f'{rec["x_cm"]:.6f}',
                "y_cm": f'{rec["y_cm"]:.6f}',
                "median_score": f"{float(med):.8f}",
                "persistence": f"{float(per):.8f}",
                "final_score": f"{float(fin):.8f}",
                "file_name": os.path.basename(rec["file_path"]),
            })


def save_summary_csv(save_path: str, rows: List[dict]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 10. 单个时间点处理
# ============================================================

def process_single_timepoint(time_folder: str) -> None:
    print("\n" + "=" * 90)
    print(f"开始处理：{time_folder}")
    print("=" * 90)

    center_data_dir = os.path.join(center_root_dir, time_folder)
    offset_data_dir = os.path.join(offset_root_dir, time_folder)

    if not os.path.isdir(center_data_dir):
        print(f"警告：中心点文件夹不存在：{center_data_dir}")
        return

    if not os.path.isdir(offset_data_dir):
        print(f"警告：偏移点文件夹不存在：{offset_data_dir}")
        return

    result_dir = f"results_morphology_{time_folder}"
    os.makedirs(result_dir, exist_ok=True)

    center_index = build_center_file_index(center_data_dir)
    offset_index = build_offset_file_index(offset_data_dir)

    center_ids = sorted(center_index.keys(), key=center_id_sort_key)
    print(f"检测到中心点编号：{center_ids}")
    print(f"偏移索引组数：{len(offset_index)}")

    if not center_ids:
        print("没有检测到中心点文件")
        return

    summary_rows = []

    for center_id in center_ids:
        print("\n" + "-" * 80)
        print(f"处理中心点：{center_id}")
        print("-" * 80)

        try:
            time_index, seg_start, seg_end = get_time_slice_from_center_id(center_id)
        except Exception as exc:
            print(f"  跳过：{exc}")
            continue

        print(f"  读取原始WAV区间：[{seg_start:.0f}, {seg_end:.0f}) 秒")

        point_records = build_point_records_for_center(center_id, center_index, offset_index)

        if len(point_records) < 10:
            print("  有效测点太少，跳过")
            continue

        valid_records, freqs, log_psd_data, frame_times = load_all_point_frame_psd(
            point_records,
            seg_start,
            seg_end,
        )

        if freqs is None or log_psd_data is None or frame_times is None or len(valid_records) < 10:
            print("  无法构建该中心点的完整频谱数据，跳过")
            continue

        print(f"  有效测点数：{len(valid_records)}")
        print(f"  频点数：{freqs.size}")
        print(f"  短时帧数：{frame_times.size}")

        try:
            frame_scores, frame_binary_hot, outer_mask = compute_frame_scores(
                valid_records, freqs, log_psd_data
            )
        except Exception as exc:
            print(f"  空间背景拟合失败：{exc}")
            continue

        median_score, persistence, final_score = aggregate_shape_maps(
            frame_scores, frame_binary_hot, outer_mask
        )

        x = np.asarray([r["x_cm"] for r in valid_records], dtype=np.float64)
        y = np.asarray([r["y_cm"] for r in valid_records], dtype=np.float64)

        metrics = compute_shape_metrics(x, y, final_score, outer_mask)

        # 归一化形态图
        if np.max(final_score) > 0:
            normalized_shape = final_score / (np.max(final_score) + ROBUST_EPS)
        else:
            normalized_shape = final_score.copy()

        # 保存CSV
        point_csv_path = os.path.join(
            result_dir,
            f"point_scores_{time_folder}_center_{center_id}.csv"
        )
        save_point_scores_csv(
            point_csv_path,
            valid_records,
            median_score,
            persistence,
            final_score,
        )
        print(f"  测点分数表已保存：{point_csv_path}")

        # 绘图
        title_prefix = (
            f"{time_folder} | Center {center_id} | "
            f"[{seg_start:.0f}, {seg_end:.0f}) s | "
            f"{FREQ_LOW/1000:.0f}-{FREQ_HIGH/1000:.0f} kHz"
        )

        plot_heatmap(
            save_path=os.path.join(
                result_dir,
                f"heatmap_final_{time_folder}_center_{center_id}.png"
            ),
            title=f"Final Morphology Heatmap\n{title_prefix}",
            x=x,
            y=y,
            values=final_score,
            colorbar_label="Final morphology score",
            metrics=metrics,
            draw_overlay=True,
        )

        plot_heatmap(
            save_path=os.path.join(
                result_dir,
                f"heatmap_persistence_{time_folder}_center_{center_id}.png"
            ),
            title=f"Hotspot Persistence Heatmap\n{title_prefix}",
            x=x,
            y=y,
            values=persistence,
            colorbar_label="Persistence (0~1)",
            metrics=metrics,
            value_range=(0.0, 1.0),
            draw_overlay=True,
        )

        plot_heatmap(
            save_path=os.path.join(
                result_dir,
                f"heatmap_normalized_shape_{time_folder}_center_{center_id}.png"
            ),
            title=f"Normalized Shape Heatmap\n{title_prefix}",
            x=x,
            y=y,
            values=normalized_shape,
            colorbar_label="Normalized morphology (0~1)",
            metrics=metrics,
            value_range=(0.0, 1.0),
            draw_overlay=True,
        )

        summary_rows.append({
            "time_folder": time_folder,
            "center_id": center_id,
            "time_index": time_index,
            "segment_start_second": f"{seg_start:.3f}",
            "segment_end_second": f"{seg_end:.3f}",
            "num_points": len(valid_records),
            "num_frames": frame_times.size,
            "peak_x_cm": f'{metrics["peak_x_cm"]:.6f}' if np.isfinite(metrics["peak_x_cm"]) else "",
            "peak_y_cm": f'{metrics["peak_y_cm"]:.6f}' if np.isfinite(metrics["peak_y_cm"]) else "",
            "peak_score": f'{metrics["peak_score"]:.8f}' if np.isfinite(metrics["peak_score"]) else "",
            "centroid_x_cm": f'{metrics["centroid_x_cm"]:.6f}' if np.isfinite(metrics["centroid_x_cm"]) else "",
            "centroid_y_cm": f'{metrics["centroid_y_cm"]:.6f}' if np.isfinite(metrics["centroid_y_cm"]) else "",
            "major_axis_deg": f'{metrics["major_axis_deg"]:.6f}' if np.isfinite(metrics["major_axis_deg"]) else "",
            "anisotropy": f'{metrics["anisotropy"]:.8f}' if np.isfinite(metrics["anisotropy"]) else "",
            "ellipse_major_sigma_cm": f'{metrics["ellipse_major_sigma_cm"]:.6f}' if np.isfinite(metrics["ellipse_major_sigma_cm"]) else "",
            "ellipse_minor_sigma_cm": f'{metrics["ellipse_minor_sigma_cm"]:.6f}' if np.isfinite(metrics["ellipse_minor_sigma_cm"]) else "",
            "center_to_peak_cm": f'{metrics["center_to_peak_cm"]:.6f}' if np.isfinite(metrics["center_to_peak_cm"]) else "",
            "center_to_centroid_cm": f'{metrics["center_to_centroid_cm"]:.6f}' if np.isfinite(metrics["center_to_centroid_cm"]) else "",
            "shape_total_score": f'{metrics["shape_total_score"]:.8f}' if np.isfinite(metrics["shape_total_score"]) else "",
        })

    summary_csv_path = os.path.join(result_dir, "summary_metrics.csv")
    save_summary_csv(summary_csv_path, summary_rows)
    print(f"\n摘要表已保存：{summary_csv_path}")


# ============================================================
# 11. 配置检查
# ============================================================

def validate_config() -> bool:
    ok = True

    if FREQ_LOW < 0 or FREQ_HIGH <= FREQ_LOW:
        print("配置错误：FREQ_HIGH 必须大于 FREQ_LOW")
        ok = False

    if FRAME_SECONDS <= 0 or HOP_SECONDS <= 0:
        print("配置错误：FRAME_SECONDS 和 HOP_SECONDS 必须大于0")
        ok = False

    if not distances:
        print("配置错误：distances 不能为空")
        ok = False

    if BACKGROUND_MIN_DISTANCE_CM not in distances and BACKGROUND_MIN_DISTANCE_CM > min(distances):
        print(
            f"提醒：BACKGROUND_MIN_DISTANCE_CM={BACKGROUND_MIN_DISTANCE_CM} 不在 distances 中，"
            "程序会使用 distance >= 该值 的点做背景拟合"
        )

    return ok


# ============================================================
# 12. 主程序
# ============================================================

def main() -> None:
    print("=" * 90)
    print("Leak Morphology Heatmap Generator")
    print("=" * 90)
    print("核心思路：")
    print("  1) 同一秒数据、多坐标beamform")
    print("  2) 1秒拆成多个短时帧")
    print("  3) 外围点做空间背景平面拟合")
    print("  4) 统计稳定热点，输出更清晰的泄漏形态")
    print(f"  频段：{FREQ_LOW/1000:.0f}-{FREQ_HIGH/1000:.0f} kHz")
    print(f"  短时帧：{FRAME_SECONDS:.2f}s，步长：{HOP_SECONDS:.2f}s")
    print(f"  背景拟合使用外围距离 >= {BACKGROUND_MIN_DISTANCE_CM} cm")
    print(f"  共需处理 {len(time_folders)} 个时间点文件夹")

    if not validate_config():
        print("配置检查未通过，程序停止")
        return

    for time_folder in time_folders:
        try:
            process_single_timepoint(time_folder)
        except Exception as exc:
            print(f"\n处理 {time_folder} 时出现未预期错误：{exc}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 90)
    print("全部处理完成！")
    print("=" * 90)


if __name__ == "__main__":
    main()
