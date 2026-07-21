import csv
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
from scipy.interpolate import griddata


# ============================================================
# 1. 配置参数
# ============================================================

# 要处理的时间点文件夹
time_folders = [
    # "HM20260626_142938.ld",
    # "HM20260626_143034.ld",
    # "HM20260626_144226.ld",
    # "HM20260626_144325.ld",

    "HM20260702_111044.ld",

    # "HM20260624_100936.ld",
    # "HM20260624_101130.ld",
    # "HM20260624_101209.ld",
    # "HM20260624_101256.ld",
    # "HM20260624_101448.ld",
]

# 根目录路径
center_root_dir = r"D:\gas\beamform_results_sh"
offset_root_dir = r"D:\gas\beamform_results_offset_multiple_sh"

# 是否递归搜索子文件夹
RECURSIVE_SEARCH = False

# 期望采样率，仅用于检查；实际Welch计算仍使用文件自身采样率
EXPECTED_SAMPLE_RATE = 192000

# 8个方向对应的极坐标角度（弧度）
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

# 8个距离（cm）
distances = [5, 10, 15, 20, 25, 30, 35, 40]
BACKGROUND_DISTANCE_CM = 40

# 频谱分析参数
FREQ_LOW = 50000
FREQ_HIGH = 70000
NFFT = 4096
WELCH_OVERLAP_RATIO = 0.5

# 是否减去40cm背景：
# False：本修改版默认模式。中心点及5~40cm所有测点都使用自身原始PSD，
#        不减去任何40cm频谱；40cm点本身也作为普通实测点参与热力图。
# True ：恢复原程序的40cm频谱减法逻辑。
SUBTRACT_40CM_BACKGROUND = False

# 仅当 SUBTRACT_40CM_BACKGROUND=True 时有效：
# "minimum_energy"：中心点选择8个方向中能量最低的40cm背景
# "median_spectrum"：中心点使用各方向40cm背景频谱的逐频点中位数
CENTER_BACKGROUND_MODE = "minimum_energy"

# 热力图参数
GRID_MIN_CM = -45
GRID_MAX_CM = 45
GRID_SIZE = 200
PRIMARY_INTERPOLATION = "linear"       # 推荐：linear，不产生cubic过冲
DRAW_NEAREST_REFERENCE = True           # 同时输出nearest对照图
CLIP_TO_MEASURED_RANGE = True           # 插值结果限制在实测最小值~最大值之间
HEATMAP_CMAP = "jet"

# 空间判断参数
NEAR_RADIUS_CM = 6.0
DROP_RATIO_THRESHOLD = 1.2
DIRECTION_ANALYSIS_MAX_DISTANCE_CM = 20
DIRECTION_ENERGY_FACTOR = 1.5


# ============================================================
# 2. 类型和文件名解析
# ============================================================

Spectrum = Tuple[np.ndarray, np.ndarray]
OffsetKey = Tuple[str, int, str]

# 长方向名必须排在短方向名前面，避免 down_left 被误解析成 down
_DIRECTION_REGEX_PART = "|".join(
    re.escape(name)
    for name in sorted(direction_angles.keys(), key=len, reverse=True)
)

CENTER_FILE_REGEX = re.compile(
    r"_(?P<center>\d+)_beamform_result\.wav$",
    flags=re.IGNORECASE,
)

OFFSET_FILE_REGEX = re.compile(
    rf"_(?P<center>\d+)d(?P<distance>\d+)_(?P<direction>{_DIRECTION_REGEX_PART})"
    rf"(?P<suffix>.*?)\.wav$",
    flags=re.IGNORECASE,
)


def list_wav_files(folder: str) -> List[str]:
    """列出文件夹中的WAV文件。"""
    if RECURSIVE_SEARCH:
        pattern = os.path.join(folder, "**", "*.wav")
        return sorted(glob.glob(pattern, recursive=True))

    pattern = os.path.join(folder, "*.wav")
    return sorted(glob.glob(pattern))


def build_center_file_index(center_data_dir: str) -> Dict[str, List[str]]:
    """建立中心点文件索引：center_id -> 文件列表。"""
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
    """
    建立偏移文件索引：(center_id, distance, direction) -> 文件列表。

    这里不再使用诸如 *_d40_down*.wav 的模糊glob，避免把
    down_left/down_right误当成down，或把up_left/up_right误当成up。
    """
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
    """文件不存在返回None；存在多个时给出警告并选排序后的第一个。"""
    if not files:
        return None

    if len(files) > 1:
        print(f"  警告：{description}匹配到{len(files)}个文件，使用第一个：")
        for file_path in files:
            print(f"    - {os.path.basename(file_path)}")

    return files[0]


# ============================================================
# 3. WAV读取和频谱计算
# ============================================================

def convert_wav_to_float(y: np.ndarray) -> np.ndarray:
    """
    将WAV数据转换成float64，但不做逐文件峰值归一化。

    整数WAV仅按数据类型满量程缩放；浮点WAV保留原始幅值比例。
    这样不同方向、不同距离文件之间的真实幅值关系不会被破坏。
    """
    if np.issubdtype(y.dtype, np.integer):
        info = np.iinfo(y.dtype)
        full_scale = float(max(abs(info.min), abs(info.max)))
        if full_scale <= 0:
            raise ValueError(f"无效的整数WAV数据类型：{y.dtype}")
        return y.astype(np.float64) / full_scale

    if np.issubdtype(y.dtype, np.floating):
        return y.astype(np.float64)

    raise TypeError(f"不支持的WAV数据类型：{y.dtype}")


def get_spectrum(file_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    读取WAV文件，计算50-70kHz功率谱密度。

    关键修改：不再对每个文件除以自身峰值，保留测点间幅值差异。
    """
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

    if y.size < 16:
        print(f"  警告：WAV数据太短，无法计算频谱：{file_path}")
        return None, None

    try:
        y = convert_wav_to_float(y)
    except (TypeError, ValueError) as exc:
        print(f"  错误：{exc}，文件：{file_path}")
        return None, None

    # 清理NaN/Inf并去除直流分量
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = y - np.mean(y)

    if not np.any(np.abs(y) > 0):
        print(f"  警告：WAV全为零或去均值后全为零：{file_path}")

    if sample_rate != EXPECTED_SAMPLE_RATE:
        print(
            f"  提醒：{os.path.basename(file_path)}采样率为{sample_rate}Hz，"
            f"不是期望的{EXPECTED_SAMPLE_RATE}Hz"
        )

    nyquist = sample_rate / 2.0
    if FREQ_LOW >= nyquist:
        print(
            f"  错误：文件奈奎斯特频率仅{nyquist:.1f}Hz，"
            f"无法分析{FREQ_LOW}-{FREQ_HIGH}Hz"
        )
        return None, None

    nperseg = min(NFFT, y.size)
    if nperseg < 16:
        return None, None

    noverlap = int(nperseg * WELCH_OVERLAP_RATIO)
    noverlap = min(noverlap, nperseg - 1)

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
        print(f"  错误：Welch频谱计算失败：{file_path}\n    {exc}")
        return None, None

    actual_high = min(FREQ_HIGH, nyquist)
    mask = (freqs >= FREQ_LOW) & (freqs <= actual_high)

    freqs = freqs[mask]
    psd = psd[mask]

    if freqs.size < 2:
        print(f"  错误：目标频带内有效频点不足：{file_path}")
        return None, None

    psd = np.maximum(np.nan_to_num(psd, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    return freqs, psd


def trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    """兼容不同NumPy版本的梯形积分。"""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def get_band_energy_from_spectrum(
    freqs: Optional[np.ndarray],
    psd: Optional[np.ndarray],
) -> float:
    """对PSD积分，得到目标频段总能量。"""
    if freqs is None or psd is None:
        return 0.0

    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)

    valid = np.isfinite(freqs) & np.isfinite(psd)
    freqs = freqs[valid]
    psd = psd[valid]

    if freqs.size < 2:
        return 0.0

    order = np.argsort(freqs)
    freqs = freqs[order]
    psd = np.maximum(psd[order], 0.0)

    return max(trapezoid_integral(psd, freqs), 0.0)


def subtract_spectrum(
    signal_freqs: Optional[np.ndarray],
    signal_psd: Optional[np.ndarray],
    background_freqs: Optional[np.ndarray],
    background_psd: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    频谱减法：信号PSD - 背景PSD，负值置零。

    背景频率轴会插值到信号频率轴上，避免仅按数组长度截断造成频点错位。
    如果没有背景，则返回信号原始PSD。
    """
    if signal_freqs is None or signal_psd is None:
        return None, None

    signal_freqs = np.asarray(signal_freqs, dtype=np.float64)
    signal_psd = np.asarray(signal_psd, dtype=np.float64)

    if background_freqs is None or background_psd is None:
        return signal_freqs.copy(), np.maximum(signal_psd.copy(), 0.0)

    background_freqs = np.asarray(background_freqs, dtype=np.float64)
    background_psd = np.asarray(background_psd, dtype=np.float64)

    valid_signal = np.isfinite(signal_freqs) & np.isfinite(signal_psd)
    valid_bg = np.isfinite(background_freqs) & np.isfinite(background_psd)

    signal_freqs = signal_freqs[valid_signal]
    signal_psd = signal_psd[valid_signal]
    background_freqs = background_freqs[valid_bg]
    background_psd = background_psd[valid_bg]

    if signal_freqs.size < 2 or background_freqs.size < 2:
        return None, None

    sig_order = np.argsort(signal_freqs)
    bg_order = np.argsort(background_freqs)
    signal_freqs = signal_freqs[sig_order]
    signal_psd = signal_psd[sig_order]
    background_freqs = background_freqs[bg_order]
    background_psd = background_psd[bg_order]

    overlap_low = max(signal_freqs[0], background_freqs[0])
    overlap_high = min(signal_freqs[-1], background_freqs[-1])

    overlap_mask = (signal_freqs >= overlap_low) & (signal_freqs <= overlap_high)
    out_freqs = signal_freqs[overlap_mask]
    out_signal_psd = signal_psd[overlap_mask]

    if out_freqs.size < 2:
        return None, None

    bg_on_signal_grid = np.interp(out_freqs, background_freqs, background_psd)
    net_psd = np.maximum(out_signal_psd - bg_on_signal_grid, 0.0)

    return out_freqs, net_psd


def apply_background_setting(
    signal_freqs: Optional[np.ndarray],
    signal_psd: Optional[np.ndarray],
    background_freqs: Optional[np.ndarray] = None,
    background_psd: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    根据配置决定是否做40cm频谱减法。

    SUBTRACT_40CM_BACKGROUND=False：
        原样返回当前测点PSD，不减去40cm能量。

    SUBTRACT_40CM_BACKGROUND=True：
        使用原程序的PSD频谱减法，负值置零。
    """
    if signal_freqs is None or signal_psd is None:
        return None, None

    if not SUBTRACT_40CM_BACKGROUND:
        freqs = np.asarray(signal_freqs, dtype=np.float64)
        psd = np.asarray(signal_psd, dtype=np.float64)
        valid = np.isfinite(freqs) & np.isfinite(psd)
        freqs = freqs[valid]
        psd = np.maximum(psd[valid], 0.0)

        if freqs.size < 2:
            return None, None

        order = np.argsort(freqs)
        return freqs[order].copy(), psd[order].copy()

    return subtract_spectrum(
        signal_freqs,
        signal_psd,
        background_freqs,
        background_psd,
    )


def energy_mode_name() -> str:
    """返回当前能量模式名称，用于输出标题、CSV和日志。"""
    if SUBTRACT_40CM_BACKGROUND:
        return "40cm_background_subtracted"
    return "raw_no_40cm_subtraction"


def energy_display_name() -> str:
    """返回适合图表显示的能量名称。"""
    if SUBTRACT_40CM_BACKGROUND:
        return "Net Energy (40cm Background Subtracted)"
    return "Raw Energy (No 40cm Subtraction)"


def median_background_spectrum(
    spectra: Dict[str, Spectrum],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """将多个背景频谱插值到同一频率轴后，计算逐频点中位数。"""
    valid_items = [
        (freqs, psd)
        for freqs, psd in spectra.values()
        if freqs is not None and psd is not None and len(freqs) >= 2
    ]

    if not valid_items:
        return None, None

    # 选频点最多的频率轴作为参考
    ref_freqs, _ = max(valid_items, key=lambda item: len(item[0]))
    ref_freqs = np.asarray(ref_freqs, dtype=np.float64)

    interpolated = []
    for freqs, psd in valid_items:
        freqs = np.asarray(freqs, dtype=np.float64)
        psd = np.asarray(psd, dtype=np.float64)

        if ref_freqs[0] < freqs[0] or ref_freqs[-1] > freqs[-1]:
            # 只允许在该背景真实覆盖的频率范围内插值
            continue

        interpolated.append(np.interp(ref_freqs, freqs, psd))

    if not interpolated:
        return None, None

    median_psd = np.median(np.vstack(interpolated), axis=0)
    median_psd = np.maximum(median_psd, 0.0)
    return ref_freqs.copy(), median_psd


def choose_center_background(
    direction_bg_spectra: Dict[str, Spectrum],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """按照配置选择中心点使用的背景。"""
    if not direction_bg_spectra:
        return None, None, "none"

    if CENTER_BACKGROUND_MODE == "median_spectrum":
        freqs, psd = median_background_spectrum(direction_bg_spectra)
        if freqs is not None:
            return freqs, psd, "all-directions median spectrum"

        print("  警告：中位背景构建失败，退回minimum_energy模式")

    if CENTER_BACKGROUND_MODE not in {"minimum_energy", "median_spectrum"}:
        print(
            f"  警告：未知CENTER_BACKGROUND_MODE={CENTER_BACKGROUND_MODE!r}，"
            "退回minimum_energy模式"
        )

    best_direction = min(
        direction_bg_spectra,
        key=lambda d: get_band_energy_from_spectrum(*direction_bg_spectra[d]),
    )
    freqs, psd = direction_bg_spectra[best_direction]
    return freqs, psd, f"{best_direction} direction minimum-energy background"


# ============================================================
# 4. 输出数据和绘图
# ============================================================

def save_point_energy_csv(
    result_dir: str,
    time_folder: str,
    center_id: str,
    records: List[dict],
) -> None:
    """保存所有实测点的频带能量，便于核对插值图。"""
    save_path = os.path.join(
        result_dir,
        f"point_energy_{time_folder}_center_{center_id}.csv",
    )

    fieldnames = [
        "point_type",
        "direction",
        "distance_cm",
        "x_cm",
        "y_cm",
        "energy",
        "energy_mode",
        "signal_file",
        "background_source",
    ]

    try:
        with open(save_path, "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        print(f"实测点能量表已保存：{save_path}")
    except Exception as exc:
        print(f"警告：保存实测点能量表失败：{exc}")


def plot_spectrum_comparison(
    result_dir: str,
    time_folder: str,
    center_id: str,
    center_freqs: Optional[np.ndarray],
    center_net_psd: Optional[np.ndarray],
    directional_spectra: Dict[str, Spectrum],
    direction_bg_spectra: Dict[str, Spectrum],
) -> None:
    """绘制中心点、各方向5cm以及40cm参考频谱对比图。"""
    mode_label = energy_display_name()
    spectrum_prefix = "Net" if SUBTRACT_40CM_BACKGROUND else "Raw"
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 左上：中心点净频谱
    ax1 = axes[0, 0]
    if center_freqs is not None and center_net_psd is not None:
        ax1.plot(center_freqs / 1000.0, center_net_psd, "b-", linewidth=1.5)
        ax1.fill_between(center_freqs / 1000.0, center_net_psd, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "No valid center spectrum", ha="center", va="center")
    ax1.set_title(f"Center Point {spectrum_prefix} Spectrum\n({mode_label})")
    ax1.set_xlabel("Frequency (kHz)")
    ax1.set_ylabel("Power Spectral Density")
    ax1.grid(True, alpha=0.3)

    # 右上：各方向5cm净频谱
    ax2 = axes[0, 1]
    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]

    for idx, direction in enumerate(direction_angles):
        spectrum = directional_spectra.get(direction)
        if spectrum is None:
            continue
        freqs, net_psd = spectrum
        if freqs is None or net_psd is None:
            continue

        ax2.plot(
            freqs / 1000.0,
            net_psd,
            linewidth=1.0,
            alpha=0.75,
            color=colors[idx % len(colors)],
            label=direction,
        )

    ax2.set_title(f"{spectrum_prefix} Spectrum at 5cm (All Directions)")
    ax2.set_xlabel("Frequency (kHz)")
    ax2.set_ylabel("Power Spectral Density")
    if directional_spectra:
        ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 左下：5cm净能量最大的方向，其背景和净频谱
    ax3 = axes[1, 0]
    valid_directions = [
        direction
        for direction, spectrum in directional_spectra.items()
        if spectrum[0] is not None and spectrum[1] is not None
    ]

    if valid_directions:
        best_direction = max(
            valid_directions,
            key=lambda direction: get_band_energy_from_spectrum(
                *directional_spectra[direction]
            ),
        )

        net_freqs, net_psd = directional_spectra[best_direction]

        if best_direction in direction_bg_spectra:
            bg_freqs, bg_psd = direction_bg_spectra[best_direction]
            ax3.plot(
                bg_freqs / 1000.0,
                bg_psd,
                color="gray",
                linewidth=1.0,
                alpha=0.6,
                label="40cm Reference Spectrum",
            )

        ax3.plot(
            net_freqs / 1000.0,
            net_psd,
            "r-",
            linewidth=2.0,
            label=f"{spectrum_prefix} Spectrum ({best_direction}, 5cm)",
        )
        ax3.fill_between(net_freqs / 1000.0, net_psd, alpha=0.3, color="red")
        ax3.set_title(f"Spectrum Comparison - Direction: {best_direction}")
        ax3.legend(loc="upper right")
    else:
        ax3.text(0.5, 0.5, "No valid 5cm spectrum", ha="center", va="center")
        ax3.set_title("Spectrum Comparison")

    ax3.set_xlabel("Frequency (kHz)")
    ax3.set_ylabel("Power Spectral Density")
    ax3.grid(True, alpha=0.3)

    # 右下：所有方向5cm净能量柱状图
    ax4 = axes[1, 1]
    dir_energies = {}
    for direction in direction_angles:
        spectrum = directional_spectra.get(direction)
        if spectrum is None:
            continue
        dir_energies[direction] = get_band_energy_from_spectrum(*spectrum)

    if dir_energies:
        directions = list(dir_energies.keys())
        energies = [dir_energies[direction] for direction in directions]
        bar_colors = [
            colors[list(direction_angles.keys()).index(direction) % len(colors)]
            for direction in directions
        ]
        bars = ax4.bar(directions, energies, color=bar_colors, alpha=0.7)

        for bar, energy in zip(bars, energies):
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{energy:.2e}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax4.set_title(f"{spectrum_prefix} Energy by Direction at 5cm (50-70kHz)")
    ax4.set_xlabel("Direction")
    ax4.set_ylabel("Integrated Energy")
    ax4.tick_params(axis="x", rotation=45)
    ax4.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"Spectrum Analysis (50-70kHz)\n{time_folder} - Center {center_id}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    save_path = os.path.join(
        result_dir,
        f"spectrum_{time_folder}_center_{center_id}.png",
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"频谱对比图已保存：{save_path}")


def interpolate_heatmap(
    plot_x: np.ndarray,
    plot_y: np.ndarray,
    plot_energy: np.ndarray,
    method: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成热力图网格，并限制显示范围。"""
    grid_x, grid_y = np.mgrid[
        GRID_MIN_CM:GRID_MAX_CM:complex(GRID_SIZE),
        GRID_MIN_CM:GRID_MAX_CM:complex(GRID_SIZE),
    ]

    grid_z = griddata(
        (plot_x, plot_y),
        plot_energy,
        (grid_x, grid_y),
        method=method,
        fill_value=np.nan,
    )

    measured_min = float(np.min(plot_energy))
    measured_max = float(np.max(plot_energy))

    if CLIP_TO_MEASURED_RANGE:
        grid_z = np.clip(grid_z, measured_min, measured_max)

    # 40cm外没有测量点，不显示；linear在凸包外本身也保持NaN
    grid_radius = np.sqrt(grid_x ** 2 + grid_y ** 2)
    grid_z[grid_radius > max(distances)] = np.nan

    return grid_x, grid_y, grid_z


def plot_heatmap(
    result_dir: str,
    time_folder: str,
    center_id: str,
    plot_x: np.ndarray,
    plot_y: np.ndarray,
    plot_energy: np.ndarray,
    method: str,
) -> None:
    """绘制指定插值方法的热力图。"""
    grid_x, grid_y, grid_z = interpolate_heatmap(
        plot_x,
        plot_y,
        plot_energy,
        method,
    )

    measured_min = float(np.min(plot_energy))
    measured_max = float(np.max(plot_energy))
    finite_grid = grid_z[np.isfinite(grid_z)]
    interpolated_max = float(np.max(finite_grid)) if finite_grid.size else float("nan")

    print(
        f"  {method}插值检查：实测最大值={measured_max:.6e}，"
        f"插值最大值={interpolated_max:.6e}"
    )

    figure, axis = plt.subplots(figsize=(8, 7), dpi=100)

    cmap = plt.get_cmap(HEATMAP_CMAP).copy()
    cmap.set_bad(color="white", alpha=0.0)

    vmax = measured_max if measured_max > measured_min else measured_min + 1.0

    image = axis.imshow(
        np.ma.masked_invalid(grid_z).T,
        extent=(GRID_MIN_CM, GRID_MAX_CM, GRID_MIN_CM, GRID_MAX_CM),
        origin="lower",
        cmap=cmap,
        interpolation="bilinear" if method == "linear" else "nearest",
        vmin=measured_min,
        vmax=vmax,
        aspect="equal",
    )

    figure.colorbar(
        image,
        ax=axis,
        label=f"50-70kHz {energy_display_name()}",
    )

    # 实测点仍使用原始值，不依赖插值结果
    axis.scatter(
        plot_x,
        plot_y,
        c="white",
        s=20,
        edgecolors="black",
        linewidths=0.6,
        alpha=0.85,
        label="Measured points",
        zorder=3,
    )

    max_index = int(np.argmax(plot_energy))
    axis.scatter(
        plot_x[max_index],
        plot_y[max_index],
        c="black",
        marker="x",
        s=100,
        linewidths=2.0,
        label="Measured maximum",
        zorder=5,
    )

    axis.scatter(
        0.0,
        0.0,
        c="red",
        marker="*",
        s=200,
        edgecolors="yellow",
        linewidths=1.0,
        label="Center",
        zorder=4,
    )

    axis.set_title(
        f"2D Spatial Heatmap ({method.title()} Interpolation)\n"
        f"{time_folder} - Center {center_id}"
    )
    axis.set_xlabel("X Distance (cm)")
    axis.set_ylabel("Y Distance (cm)")
    axis.set_xlim(GRID_MIN_CM, GRID_MAX_CM)
    axis.set_ylim(GRID_MIN_CM, GRID_MAX_CM)
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend(loc="upper right", fontsize=8)

    save_path = os.path.join(
        result_dir,
        f"heatmap_{energy_mode_name()}_{method}_{time_folder}_center_{center_id}.png",
    )
    figure.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    print(f"{method}热力图已保存：{save_path}")


# ============================================================
# 5. 空间分析
# ============================================================

def analyze_spatial_energy(
    plot_x: np.ndarray,
    plot_y: np.ndarray,
    plot_energy: np.ndarray,
    direction_distance_energy: Dict[str, Dict[int, float]],
) -> None:
    """使用实测点进行分析，判断逻辑不使用插值热力图。"""
    max_index = int(np.argmax(plot_energy))
    max_x = float(plot_x[max_index])
    max_y = float(plot_y[max_index])
    max_value = float(plot_energy[max_index])

    print("\n【频谱分析结果】")
    print(f"能量最高实测点坐标：({max_x:.1f}, {max_y:.1f}) cm")
    print(f"{energy_display_name()}：{max_value:.6e}")

    distance_to_max = np.sqrt((plot_x - max_x) ** 2 + (plot_y - max_y) ** 2)
    not_self = np.arange(plot_energy.size) != max_index
    near_mask = (distance_to_max <= NEAR_RADIUS_CM) & not_self

    if np.any(near_mask):
        near_average = float(np.mean(plot_energy[near_mask]))
        drop_ratio = max_value / (near_average + 1e-12)
        print(f"【空间梯度验证】Drop Ratio：{drop_ratio:.2f}")

        if drop_ratio > DROP_RATIO_THRESHOLD:
            print(">>> 判定：真实气流泄漏源！")
        else:
            print(">>> 判定：环境背景噪声（虚警）")
    else:
        print(
            f"【空间梯度验证】最高点{NEAR_RADIUS_CM:.1f}cm范围内没有其他实测点，"
            "无法计算Drop Ratio"
        )

    print("\n【喷射方向分析】")
    directional_energies: Dict[str, float] = {}

    for direction in direction_angles:
        values = [
            energy
            for distance, energy in direction_distance_energy.get(direction, {}).items()
            if 0 < distance <= DIRECTION_ANALYSIS_MAX_DISTANCE_CM
        ]

        if values:
            directional_energies[direction] = float(np.mean(values))

    if not directional_energies:
        print("没有足够的方向数据")
        return

    mean_direction_energy = float(np.mean(list(directional_energies.values())))
    detected = False

    for direction, energy in sorted(
        directional_energies.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        ratio = energy / (mean_direction_energy + 1e-12)
        print(f"  {direction:>10s}：平均能量={energy:.6e}，相对均值={ratio:.2f}")

        if energy > mean_direction_energy * DIRECTION_ENERGY_FACTOR:
            print(f"    -> 检测到较强喷射方向：【{direction}】")
            detected = True

    if not detected:
        print("  未发现明显高于其他方向的单一喷射方向")


# ============================================================
# 6. 单时间点处理
# ============================================================

def process_single_timepoint_spectrum(
    time_folder: str,
    center_root: str,
    offset_root: str,
) -> None:
    """处理单个时间点的所有中心点。"""
    print(f"\n{'=' * 80}")
    print(f"开始处理时间点：{time_folder}（频谱分析模式）")
    print(f"{'=' * 80}")

    center_data_dir = os.path.join(center_root, time_folder)
    offset_data_dir = os.path.join(offset_root, time_folder)

    if not os.path.isdir(center_data_dir):
        print(f"警告：中心点文件夹不存在：{center_data_dir}")
        return

    if not os.path.isdir(offset_data_dir):
        print(f"警告：偏移点文件夹不存在：{offset_data_dir}")
        return

    result_dir = f"results_sh_spectrum_{energy_mode_name()}_{time_folder}"
    os.makedirs(result_dir, exist_ok=True)

    center_index = build_center_file_index(center_data_dir)
    offset_index = build_offset_file_index(offset_data_dir)

    center_ids = sorted(center_index.keys())
    print(f"检测到中心点编号：{center_ids}")
    print(f"偏移文件索引中共有{len(offset_index)}组精确测点")

    if not center_ids:
        print(f"警告：在{time_folder}中没有检测到中心点文件")
        return

    for center_id in center_ids:
        print(f"\n{'=' * 60}")
        print(f"正在处理：{time_folder} - 中心点{center_id}")
        print(f"{'=' * 60}")

        center_file = choose_first_file(
            center_index.get(center_id),
            f"中心点{center_id}",
        )
        if center_file is None:
            print(f"警告：中心点{center_id}文件不存在")
            continue

        center_freqs, center_psd = get_spectrum(center_file)
        print(f"中心点文件：{os.path.basename(center_file)}")

        if center_freqs is None or center_psd is None:
            print("错误：无法读取中心点频谱")
            continue

        # ----------------------------------------------------
        # 读取各方向40cm背景
        # ----------------------------------------------------
        direction_bg_spectra: Dict[str, Spectrum] = {}
        direction_bg_files: Dict[str, str] = {}

        print("\n读取各方向40cm背景：")
        for direction in direction_angles:
            key = (center_id, BACKGROUND_DISTANCE_CM, direction)
            background_file = choose_first_file(
                offset_index.get(key),
                f"中心点{center_id}、{direction}方向、{BACKGROUND_DISTANCE_CM}cm背景",
            )

            if background_file is None:
                print(f"  警告：{direction}方向{BACKGROUND_DISTANCE_CM}cm文件不存在")
                continue

            bg_freqs, bg_psd = get_spectrum(background_file)
            if bg_freqs is None or bg_psd is None:
                continue

            direction_bg_spectra[direction] = (bg_freqs, bg_psd)
            direction_bg_files[direction] = background_file
            bg_energy = get_band_energy_from_spectrum(bg_freqs, bg_psd)
            print(
                f"  {direction:>10s}：背景能量={bg_energy:.6e}，"
                f"文件={os.path.basename(background_file)}"
            )

        fallback_bg_freqs, fallback_bg_psd = median_background_spectrum(
            direction_bg_spectra
        )

        # ----------------------------------------------------
        # 中心点频谱：
        # 本修改版默认直接使用中心点原始PSD，不减去任何40cm背景。
        # ----------------------------------------------------
        if SUBTRACT_40CM_BACKGROUND:
            center_bg_freqs, center_bg_psd, center_bg_name = choose_center_background(
                direction_bg_spectra
            )
        else:
            center_bg_freqs, center_bg_psd = None, None
            center_bg_name = "none; raw center PSD; 40cm not subtracted"

        center_net_freqs, center_net_psd = apply_background_setting(
            center_freqs,
            center_psd,
            center_bg_freqs,
            center_bg_psd,
        )

        if center_net_freqs is None or center_net_psd is None:
            print("错误：中心点频谱计算失败")
            continue

        center_net_energy = get_band_energy_from_spectrum(
            center_net_freqs,
            center_net_psd,
        )
        print(
            f"\n中心点能量：{center_net_energy:.6e}，"
            f"模式={energy_mode_name()}，背景来源={center_bg_name}"
        )

        plot_x: List[float] = [0.0]
        plot_y: List[float] = [0.0]
        plot_energy: List[float] = [center_net_energy]

        point_records: List[dict] = [
            {
                "point_type": "center",
                "direction": "center",
                "distance_cm": 0,
                "x_cm": 0.0,
                "y_cm": 0.0,
                "energy": center_net_energy,
                "energy_mode": energy_mode_name(),
                "signal_file": os.path.basename(center_file),
                "background_source": center_bg_name,
            }
        ]

        directional_spectra_5cm: Dict[str, Spectrum] = {}
        direction_distance_energy: Dict[str, Dict[int, float]] = {
            direction: {} for direction in direction_angles
        }

        # ----------------------------------------------------
        # 处理各方向各距离
        # ----------------------------------------------------
        print(f"\n各方向各距离能量（模式：{energy_mode_name()}）：")
        for direction, angle in direction_angles.items():
            exact_bg = direction_bg_spectra.get(direction)

            if not SUBTRACT_40CM_BACKGROUND:
                # 40cm文件只作为普通测点和频谱对照，不从任何测点中扣除。
                bg_freqs, bg_psd = None, None
                bg_source = "none; raw PSD; 40cm not subtracted"
            elif exact_bg is not None:
                bg_freqs, bg_psd = exact_bg
                bg_source = (
                    f"{direction} {BACKGROUND_DISTANCE_CM}cm: "
                    f"{os.path.basename(direction_bg_files[direction])}"
                )
            elif fallback_bg_freqs is not None and fallback_bg_psd is not None:
                bg_freqs, bg_psd = fallback_bg_freqs, fallback_bg_psd
                bg_source = "all-directions median fallback background"
                print(
                    f"  提醒：{direction}方向缺少精确背景，"
                    "使用所有可用方向40cm背景的中位频谱"
                )
            else:
                bg_freqs, bg_psd = None, None
                bg_source = "none"
                print(f"  提醒：{direction}方向没有可用背景，将不做背景减法")

            for distance in distances:
                key = (center_id, distance, direction)
                signal_file = choose_first_file(
                    offset_index.get(key),
                    f"中心点{center_id}、{direction}方向、{distance}cm测点",
                )

                if signal_file is None:
                    print(f"  缺失：{direction}方向{distance}cm")
                    continue

                signal_freqs, signal_psd = get_spectrum(signal_file)
                if signal_freqs is None or signal_psd is None:
                    continue

                net_freqs, net_psd = apply_background_setting(
                    signal_freqs,
                    signal_psd,
                    bg_freqs,
                    bg_psd,
                )

                if net_freqs is None or net_psd is None:
                    print(f"  警告：{direction}方向{distance}cm频谱计算失败")
                    continue

                net_energy = get_band_energy_from_spectrum(net_freqs, net_psd)

                x = float(distance * np.cos(angle))
                y = float(distance * np.sin(angle))

                # 消除cos(pi/2)一类浮点误差，便于CSV阅读
                if abs(x) < 1e-10:
                    x = 0.0
                if abs(y) < 1e-10:
                    y = 0.0

                plot_x.append(x)
                plot_y.append(y)
                plot_energy.append(net_energy)
                direction_distance_energy[direction][distance] = net_energy

                if distance == 5:
                    directional_spectra_5cm[direction] = (net_freqs, net_psd)

                point_records.append(
                    {
                        "point_type": "offset",
                        "direction": direction,
                        "distance_cm": distance,
                        "x_cm": x,
                        "y_cm": y,
                        "energy": net_energy,
                        "energy_mode": energy_mode_name(),
                        "signal_file": os.path.basename(signal_file),
                        "background_source": bg_source,
                    }
                )

                print(
                    f"  {direction:>10s}方向{distance:>2d}cm："
                    f"能量={net_energy:.6e}"
                )

        plot_x_array = np.asarray(plot_x, dtype=np.float64)
        plot_y_array = np.asarray(plot_y, dtype=np.float64)
        plot_energy_array = np.asarray(plot_energy, dtype=np.float64)

        valid_points = (
            np.isfinite(plot_x_array)
            & np.isfinite(plot_y_array)
            & np.isfinite(plot_energy_array)
        )
        plot_x_array = plot_x_array[valid_points]
        plot_y_array = plot_y_array[valid_points]
        plot_energy_array = np.maximum(plot_energy_array[valid_points], 0.0)

        if plot_energy_array.size == 0:
            print("错误：没有有效的空间能量点")
            continue

        # 分析严格使用实测点，不使用插值结果
        analyze_spatial_energy(
            plot_x_array,
            plot_y_array,
            plot_energy_array,
            direction_distance_energy,
        )

        save_point_energy_csv(
            result_dir,
            time_folder,
            center_id,
            point_records,
        )

        plot_spectrum_comparison(
            result_dir,
            time_folder,
            center_id,
            center_net_freqs,
            center_net_psd,
            directional_spectra_5cm,
            direction_bg_spectra,
        )

        # linear至少需要3个不共线点；正常8方向数据肯定满足
        try:
            plot_heatmap(
                result_dir,
                time_folder,
                center_id,
                plot_x_array,
                plot_y_array,
                plot_energy_array,
                PRIMARY_INTERPOLATION,
            )
        except Exception as exc:
            print(f"主热力图绘制失败（{PRIMARY_INTERPOLATION}）：{exc}")
            print("尝试退回nearest插值")
            try:
                plot_heatmap(
                    result_dir,
                    time_folder,
                    center_id,
                    plot_x_array,
                    plot_y_array,
                    plot_energy_array,
                    "nearest",
                )
            except Exception as fallback_exc:
                print(f"nearest热力图也绘制失败：{fallback_exc}")

        if DRAW_NEAREST_REFERENCE and PRIMARY_INTERPOLATION != "nearest":
            try:
                plot_heatmap(
                    result_dir,
                    time_folder,
                    center_id,
                    plot_x_array,
                    plot_y_array,
                    plot_energy_array,
                    "nearest",
                )
            except Exception as exc:
                print(f"nearest对照热力图绘制失败：{exc}")


# ============================================================
# 7. 主程序
# ============================================================

def validate_config() -> bool:
    """运行前检查配置。"""
    valid = True

    if FREQ_LOW < 0 or FREQ_HIGH <= FREQ_LOW:
        print("配置错误：FREQ_HIGH必须大于FREQ_LOW，且频率不能为负")
        valid = False

    if NFFT < 16:
        print("配置错误：NFFT不能小于16")
        valid = False

    if not distances:
        print("配置错误：distances不能为空")
        valid = False

    if BACKGROUND_DISTANCE_CM not in distances:
        print(
            f"提醒：背景距离{BACKGROUND_DISTANCE_CM}cm不在distances中，"
            "仍会读取该距离作为背景，但不会作为普通测点处理"
        )

    if PRIMARY_INTERPOLATION not in {"linear", "nearest"}:
        print(
            f"配置错误：PRIMARY_INTERPOLATION={PRIMARY_INTERPOLATION!r}，"
            "只建议使用'linear'或'nearest'"
        )
        valid = False

    return valid


def main() -> None:
    print("=" * 80)
    print("原始频带能量热力图 - 50-70kHz（不减40cm能量版）")
    print("=" * 80)
    print("关键修改：")
    print("  1. 中心点不减去任何40cm背景")
    print("  2. 5~40cm所有偏移点也不减去40cm背景")
    print("  3. 40cm点使用自身原始能量，作为普通实测点参与热力图")
    print("  4. 保留真实幅值比例，不做逐文件峰值归一化")
    print("  5. 主热力图使用linear插值，并限制在实测能量范围内")
    print("  6. 额外输出nearest热力图用于核对")
    print("  7. 所有判断只使用实测点，不使用插值像素")
    print(f"当前能量模式：{energy_mode_name()}")
    print(f"总共有{len(time_folders)}个时间点需要处理")

    if not validate_config():
        print("配置检查未通过，程序停止")
        return

    for time_folder in time_folders:
        try:
            process_single_timepoint_spectrum(
                time_folder,
                center_root_dir,
                offset_root_dir,
            )
        except Exception as exc:
            print(f"\n处理时间点{time_folder}时出现未预期错误：{exc}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("所有时间点频谱分析完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
