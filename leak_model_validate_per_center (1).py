# -*- coding: utf-8 -*-
r"""
leak_model_validate_dataset.py

使用已训练模型/规则验证一个文件夹下的数据集是真泄漏还是假泄漏
================================================================

支持两种模式：

1) single 模式：
   一个未知数据文件夹，判断这一批数据整体更像 TRUE_LEAK 还是 FALSE_LEAK。
   默认使用 v7 通用模型：
       leak_v7_robust_feature_results/v7_final_robust_classifier.pkl

2) pairwise 模式：
   两个候选文件夹 A/B，它们 center 一一对应。
   使用 144226 pairwise 校准规则判断：同一个 center 下 A 和 B 哪个更像 TRUE_LEAK。
   这个模式只适合类似 144226 那种“同 center 成对比较”的场景。

重要说明：
----------
- 如果你只有一个未知文件夹，请用 MODE = "single"。
- 如果你有两个成对候选文件夹，比如同一批 center 的两组数据，请用 MODE = "pairwise"。
- 144226 pairwise 规则本质是相对比较规则；单独一个文件夹没有配对对象时，不建议强行用 pairwise 判绝对真假。

运行：
    python leak_model_validate_dataset.py

输出：
    C:\Users\jiangxinru6\Desktop\wurenji\leak_inference_results

你需要修改的位置：
    CONFIG 区域里的 MODE 和输入路径。
"""

import os
import re
import json
import math
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from scipy.io import wavfile
from scipy import signal
from scipy.stats import kurtosis

try:
    import joblib
except Exception:
    joblib = None


# ============================================================
# 1. CONFIG：你主要改这里
# ============================================================

BASE_DIR = r"C:\Users\jiangxinru6\Desktop\wurenji"

# 模式：
#   "single"  ：一个文件夹，判断整批数据 TRUE/FALSE
#   "pairwise"：两个文件夹 A/B 成对比较，判断每个 center 哪个是真泄漏
MODE = "single"

# -------------------- single 模式输入 --------------------
# 填你的未知数据 offset 文件夹。
# 这个文件夹下面应包含类似：
#   HM20260626_142938.ld_00_14d5_down_beamform_result.wav
# 或者该目录下递归包含这些 wav。
SINGLE_OFFSET_FOLDER = r"D:\gas\your_unknown_offset_folder"

# 可选：如果有中心点 wav 文件夹，就填；没有可以留空。
SINGLE_CENTER_FOLDER = r""

# -------------------- pairwise 模式输入 --------------------
# 两个候选 offset 文件夹，center 应一一对应。
PAIRWISE_OFFSET_FOLDER_A = r"D:\gas\candidate_A_offset_folder"
PAIRWISE_OFFSET_FOLDER_B = r"D:\gas\candidate_B_offset_folder"

PAIRWISE_CENTER_FOLDER_A = r""
PAIRWISE_CENTER_FOLDER_B = r""

PAIRWISE_NAME_A = "CANDIDATE_A"
PAIRWISE_NAME_B = "CANDIDATE_B"

# -------------------- 模型/规则路径 --------------------
V7_MODEL_PATH = os.path.join(
    BASE_DIR,
    "leak_v7_robust_feature_results",
    "v7_final_robust_classifier.pkl",
)

V7_CONFIG_PATH = os.path.join(
    BASE_DIR,
    "leak_v7_robust_feature_results",
    "v7_final_model_config.json",
)

# 用于补充缺失特征的训练集中位数。没有也能跑，但有它更稳。
V7_TRAIN_FEATURE_CSV = os.path.join(
    BASE_DIR,
    "leak_v7_robust_feature_results",
    "v7_robust_feature_dataset.csv",
)

PAIRWISE_RULE_PATH = os.path.join(
    BASE_DIR,
    "leak_v8_pairwise_144226_calibration_results",
    "v8_pairwise_calibration_rule.json",
)

OUTPUT_DIR = os.path.join(BASE_DIR, "leak_inference_per_center_results")

# -------------------- 特征提取参数 --------------------
FREQ_LOW = 20000
FREQ_HIGH = 70000
SUBBANDS = [
    (20000, 30000),
    (30000, 40000),
    (40000, 50000),
    (50000, 60000),
    (60000, 70000),
]

DIRECTIONS = [
    "up",
    "down",
    "left",
    "right",
    "up_left",
    "up_right",
    "down_left",
    "down_right",
]

DISTANCES_CM = [5, 10, 15, 20, 25, 30, 35, 40]
NEAR_DISTANCE_MAX_CM = 20

NFFT = 4096
WELCH_NPERSEG = 4096
WELCH_NOVERLAP = 2048

WAV_EXTS = [".wav", ".WAV"]

# 数据集整体判定阈值。v7模型概率平均值 >= 0.5 判 TRUE。
DATASET_TRUE_THRESHOLD = 0.5


# ============================================================
# 2. 基础工具
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_center_id(x):
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    nums = "".join(ch for ch in s if ch.isdigit())
    if nums == "":
        return s
    return nums.zfill(2)


def safe_float_series(s):
    x = pd.to_numeric(s, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    return x


def entropy_norm(x):
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 0)
    total = np.sum(x)
    if total <= 1e-20:
        return 0.0
    p = x / total
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    return float(-np.sum(p * np.log(p + 1e-20)) / np.log(len(p)))


def spectral_flatness_from_values(x):
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 1e-20)
    if len(x) == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(x))) / (np.mean(x) + 1e-20))


def gini_coefficient(x):
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 0)
    if len(x) == 0 or np.sum(x) <= 1e-20:
        return 0.0
    x = np.sort(x)
    n = len(x)
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def safe_name(s):
    return str(s).replace("\\", "_").replace("/", "_").replace(":", "_").replace(".", "_")


# ============================================================
# 3. 文件解析
# ============================================================

def list_wav_files(root):
    out = []
    if not root or not os.path.exists(root):
        return out
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if any(name.endswith(ext) for ext in WAV_EXTS):
                out.append(os.path.join(dirpath, name))
    return sorted(out)


def parse_offset_filename_candidates(path):
    """
    支持你的 offset 命名：
        HM20260626_142938.ld_00_14d5_down_beamform_result.wav

    自动尝试：
        A: .ld_ 后第一个数字当 center
        B: d 前面的数字当 center

    通常你的数据应该选择 B：14d5 -> center=14, distance=5。
    """
    base = os.path.basename(str(path)).lower().replace("-", "_").replace(" ", "_")
    direction_pattern = r"(up_left|up_right|down_left|down_right|up|down|left|right)"

    m = re.search(
        rf"\.ld_(\d{{1,3}})_(\d{{1,3}})d(\d{{1,3}})_({direction_pattern})_beamform",
        base,
        flags=re.IGNORECASE,
    )

    candidates = []

    if m:
        first_num = m.group(1).zfill(2)
        second_num = m.group(2).zfill(2)
        dist = int(m.group(3))
        direction = m.group(4).lower()
        if 0 < dist <= 200:
            candidates.append({
                "schema": "A_first_number_as_center",
                "center": first_num,
                "direction": direction,
                "distance": dist,
            })
            candidates.append({
                "schema": "B_number_before_d_as_center",
                "center": second_num,
                "direction": direction,
                "distance": dist,
            })
        return candidates

    # 兜底：center 由 14d5 里的 14 来解析
    direction = None
    for d in ["up_left", "up_right", "down_left", "down_right", "up", "down", "left", "right"]:
        if re.search(rf"(^|[_\-\\/]){d}($|[_\-\\/\.])", base):
            direction = d
            break

    md = re.search(r"(\d{1,3})d(\d{1,3})", base)
    if md and direction is not None:
        center = md.group(1).zfill(2)
        dist = int(md.group(2))
        candidates.append({
            "schema": "fallback_number_before_d_as_center",
            "center": center,
            "direction": direction,
            "distance": dist,
        })

    return candidates


def discover_offset_files(offset_folder):
    files = list_wav_files(offset_folder)

    if not files:
        raise FileNotFoundError(f"没有在 offset_folder 中找到 wav: {offset_folder}")

    schema_maps = {}

    for f in files:
        for c in parse_offset_filename_candidates(f):
            schema = c["schema"]
            key = (c["center"], c["direction"], int(c["distance"]))
            schema_maps.setdefault(schema, {})
            schema_maps[schema].setdefault(key, [])
            schema_maps[schema][key].append(f)

    if not schema_maps:
        raise RuntimeError(
            "没有成功解析任何 offset wav。请检查文件名是否类似：\n"
            "HM20260626_142938.ld_00_14d5_down_beamform_result.wav"
        )

    rows = []
    for schema, mp in schema_maps.items():
        centers = sorted(set(k[0] for k in mp.keys()))
        counts = []
        for cc in centers:
            counts.append(sum(1 for k in mp.keys() if k[0] == cc))
        avg_per_center = float(np.mean(counts)) if counts else 0.0
        med_per_center = float(np.median(counts)) if counts else 0.0
        total = len(mp)
        n_centers = len(centers)
        score = n_centers * 1000.0 + total + 100.0 / (1.0 + abs(avg_per_center - 64.0))
        rows.append({
            "schema": schema,
            "n_centers": n_centers,
            "total_combos": total,
            "avg_per_center": avg_per_center,
            "median_per_center": med_per_center,
            "score": score,
        })

    rows = sorted(rows, key=lambda r: r["score"], reverse=True)
    best = rows[0]["schema"]

    print("offset解析方案候选:")
    for r in rows:
        print(
            f"  {r['schema']}: centers={r['n_centers']}, "
            f"combos={r['total_combos']}, avg/center={r['avg_per_center']:.1f}, "
            f"median/center={r['median_per_center']:.1f}"
        )
    print("采用offset解析方案:", best)

    return schema_maps[best]


def discover_center_files(center_folder):
    """中心点 wav 是可选的。解析不到也不影响 offset 主特征。"""
    out = {}
    files = list_wav_files(center_folder)
    if not files:
        return out

    for f in files:
        base = os.path.basename(f).lower().replace("-", "_").replace(" ", "_")
        center = None

        # 常见格式：.ld_14_beamform_result.wav
        m = re.search(r"\.ld_(\d{1,3})(?=_beamform|_center|_result|\.|$)", base)
        if m:
            center = m.group(1).zfill(2)

        # center_14
        if center is None:
            m = re.search(r"(?:center|centre|c)_(\d{1,3})(?=_|\.|$)", base)
            if m:
                center = m.group(1).zfill(2)

        # 文件开头 14_xxx
        if center is None:
            m = re.match(r"^(\d{1,3})(?=_|\.|$)", base)
            if m:
                center = m.group(1).zfill(2)

        if center is not None and center not in out:
            out[center] = f

    return out


# ============================================================
# 4. WAV 频谱/时间特征
# ============================================================

def read_wav_float(path):
    fs, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.astype(np.float64).mean(axis=1)
    else:
        x = x.astype(np.float64)
    x = x - np.mean(x)
    max_abs = np.max(np.abs(x)) + 1e-12
    if max_abs > 10:
        x = x / max_abs
    return fs, x


def welch_psd(x, fs):
    nperseg = min(WELCH_NPERSEG, len(x))
    if nperseg < 256:
        return np.array([]), np.array([])
    noverlap = min(WELCH_NOVERLAP, max(0, nperseg // 2))
    f, pxx = signal.welch(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=max(NFFT, nperseg),
        scaling="density",
    )
    return f, np.maximum(pxx, 0)


def band_energy_from_psd(f, pxx, lo, hi):
    mask = (f >= lo) & (f < hi)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(pxx[mask], f[mask]))


def analyze_spectrum_file(path):
    fs, x = read_wav_float(path)
    f, pxx = welch_psd(x, fs)
    if len(f) == 0:
        return None

    sub = []
    for lo, hi in SUBBANDS:
        sub.append(band_energy_from_psd(f, pxx, lo, hi))
    sub = np.asarray(sub, dtype=float)
    total = float(np.sum(sub))

    mask = (f >= FREQ_LOW) & (f <= FREQ_HIGH)
    fb = f[mask]
    pb = pxx[mask]

    if len(fb) == 0 or np.sum(pb) <= 1e-20:
        spec = {
            "spec_centroid_hz": 0.0,
            "spec_bandwidth_hz": 0.0,
            "spec_entropy": 0.0,
            "spec_flatness": 0.0,
            "spec_peak_freq_hz": 0.0,
            "spec_rolloff_85_hz": 0.0,
            "spec_slope": 0.0,
            "spec_peakiness": 0.0,
        }
    else:
        total_p = np.sum(pb) + 1e-20
        centroid = float(np.sum(fb * pb) / total_p)
        bandwidth = float(np.sqrt(np.sum(((fb - centroid) ** 2) * pb) / total_p))
        entropy = entropy_norm(pb)
        flat = spectral_flatness_from_values(pb)
        peak_idx = int(np.argmax(pb))
        peak_freq = float(fb[peak_idx])
        peakiness = float(np.max(pb) / (np.mean(pb) + 1e-20))
        cum = np.cumsum(pb)
        idx = int(np.searchsorted(cum, 0.85 * cum[-1]))
        idx = min(idx, len(fb) - 1)
        rolloff = float(fb[idx])
        try:
            y = np.log10(pb + 1e-20)
            xfreq = (fb - fb.mean()) / (fb.std() + 1e-12)
            slope = float(np.polyfit(xfreq, y, 1)[0])
        except Exception:
            slope = 0.0
        spec = {
            "spec_centroid_hz": centroid,
            "spec_bandwidth_hz": bandwidth,
            "spec_entropy": entropy,
            "spec_flatness": flat,
            "spec_peak_freq_hz": peak_freq,
            "spec_rolloff_85_hz": rolloff,
            "spec_slope": slope,
            "spec_peakiness": peakiness,
        }

    return {
        "band_energy_20_70": total,
        "subband_energy": sub,
        **spec,
    }


def bandpass_signal(x, fs, lo=FREQ_LOW, hi=FREQ_HIGH):
    nyq = fs / 2.0
    hi2 = min(hi, nyq * 0.95)
    lo2 = min(lo, hi2 * 0.8)
    if hi2 <= lo2 or lo2 <= 0:
        return x
    try:
        sos = signal.butter(4, [lo2 / nyq, hi2 / nyq], btype="bandpass", output="sos")
        if len(x) > 32:
            return signal.sosfiltfilt(sos, x)
        return signal.sosfilt(sos, x)
    except Exception:
        return x


def time_features_from_wav(path):
    try:
        fs, x = read_wav_float(path)
        xb = bandpass_signal(x, fs)
        win = max(64, int(0.020 * fs))
        hop = max(32, int(0.010 * fs))
        energies = []
        for start in range(0, max(1, len(xb) - win + 1), hop):
            seg = xb[start:start + win]
            if len(seg) < win // 2:
                continue
            energies.append(float(np.mean(seg ** 2)))
        if len(energies) == 0:
            energies = [float(np.mean(xb ** 2))]
        e = np.asarray(energies, dtype=float)
        mean = float(np.mean(e))
        std = float(np.std(e))
        cv = float(std / (mean + 1e-20))
        max_mean = float(np.max(e) / (mean + 1e-20))
        kurt = float(kurtosis(e, fisher=False, bias=False)) if len(e) >= 4 else 0.0
        rms = float(np.sqrt(np.mean(xb ** 2)))
        return {
            "time_energy_mean": mean,
            "time_energy_std": std,
            "time_energy_cv": cv,
            "time_energy_max_mean_ratio": max_mean,
            "time_energy_kurtosis": kurt,
            "time_rms": rms,
        }
    except Exception:
        return {
            "time_energy_mean": np.nan,
            "time_energy_std": np.nan,
            "time_energy_cv": np.nan,
            "time_energy_max_mean_ratio": np.nan,
            "time_energy_kurtosis": np.nan,
            "time_rms": np.nan,
        }


# ============================================================
# 5. 特征提取
# ============================================================

def fit_decay(distance_values, energy_values):
    d = np.asarray(distance_values, dtype=float)
    e = np.asarray(energy_values, dtype=float)
    mask = np.isfinite(d) & np.isfinite(e) & (d > 0) & (e > 0)
    d = d[mask]
    e = e[mask]
    if len(d) < 3:
        return 0.0, 0.0
    x = np.log(d)
    y = np.log(e + 1e-20)
    try:
        coef = np.polyfit(x, y, 1)
        slope = coef[0]
        n = float(-slope)
        y_hat = np.polyval(coef, x)
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-20
        r2 = 1.0 - ss_res / ss_tot
        return max(0.0, n), float(np.clip(r2, 0.0, 1.0))
    except Exception:
        return 0.0, 0.0


def extract_features_from_folder(offset_folder, center_folder="", dataset_name="UNKNOWN"):
    print("\n" + "=" * 100)
    print("开始提取特征:", dataset_name)
    print("offset_folder:", offset_folder)
    print("center_folder:", center_folder if center_folder else "未提供")
    print("=" * 100)

    offset_files = discover_offset_files(offset_folder)
    center_files = discover_center_files(center_folder) if center_folder else {}

    centers = sorted(set(k[0] for k in offset_files.keys()))

    if not centers:
        raise RuntimeError("没有识别到任何 center。")

    count_by_center = {}
    for (cc, dd, dist), files in offset_files.items():
        count_by_center.setdefault(cc, 0)
        count_by_center[cc] += 1

    avg_count = float(np.mean(list(count_by_center.values()))) if count_by_center else 0.0
    print("center数量:", len(centers))
    print("offset组合数量:", len(offset_files))
    print(f"平均每center offset组合数: {avg_count:.1f} / 64")
    print("前10个center offset组合数:", sorted(count_by_center.items())[:10])

    if avg_count < 40:
        print("[警告] 平均每center offset组合数明显小于64，可能缺文件或解析不完全。")

    rows = []

    for i, center in enumerate(centers, 1):
        if i % 10 == 0 or i == len(centers):
            print(f"  已处理 {i}/{len(centers)}")

        direction_distance_energy = {d: {} for d in DIRECTIONS}
        direction_distance_band = {d: {} for d in DIRECTIONS}
        direction_distance_spec = {d: {} for d in DIRECTIONS}

        wav_count = 0

        for direction in DIRECTIONS:
            for dist in DISTANCES_CM:
                files = offset_files.get((center, direction, dist), [])
                if not files:
                    continue

                totals = []
                bands = []
                specs = []

                for f in files:
                    try:
                        res = analyze_spectrum_file(f)
                        if res is None:
                            continue
                        totals.append(res["band_energy_20_70"])
                        bands.append(res["subband_energy"])
                        specs.append(res)
                        wav_count += 1
                    except Exception:
                        continue

                if totals:
                    direction_distance_energy[direction][dist] = float(np.mean(totals))
                    direction_distance_band[direction][dist] = np.mean(np.asarray(bands), axis=0)
                    direction_distance_spec[direction][dist] = specs[0]

        row = {
            "dataset_name": dataset_name,
            "time": os.path.basename(os.path.normpath(offset_folder)),
            "center": center,
            "center_norm": normalize_center_id(center),
            "offset_wav_count_used": wav_count,
        }

        if wav_count == 0:
            rows.append(row)
            continue

        dir_near_energy = []
        matrix_near = np.zeros((len(DIRECTIONS), len(SUBBANDS)), dtype=float)

        for di, direction in enumerate(DIRECTIONS):
            near_vals = []
            all_vals = []
            for dist, e in direction_distance_energy[direction].items():
                all_vals.append(e)
                if dist <= NEAR_DISTANCE_MAX_CM:
                    near_vals.append(e)
            if not near_vals:
                near_vals = all_vals
            dir_near_energy.append(float(np.sum(near_vals)) if near_vals else 0.0)

            for dist, b in direction_distance_band[direction].items():
                if dist <= NEAR_DISTANCE_MAX_CM:
                    matrix_near[di, :] += np.asarray(b, dtype=float)

        dir_near_energy = np.asarray(dir_near_energy, dtype=float)
        total_near = float(np.sum(dir_near_energy))
        sort_idx = np.argsort(dir_near_energy)[::-1]
        best_i = int(sort_idx[0])
        second_i = int(sort_idx[1]) if len(sort_idx) > 1 else best_i
        best_direction = DIRECTIONS[best_i]

        row["best_direction"] = best_direction
        row["raw_best_energy"] = float(dir_near_energy[best_i])
        row["mean_direction_energy"] = float(np.mean(dir_near_energy))
        row["direction_energy_std"] = float(np.std(dir_near_energy))
        row["direction_cv"] = float(np.std(dir_near_energy) / (np.mean(dir_near_energy) + 1e-20))
        row["direction_entropy"] = entropy_norm(dir_near_energy)
        row["direction_gini"] = gini_coefficient(dir_near_energy)
        row["direction_top1_ratio"] = float(dir_near_energy[best_i] / (total_near + 1e-20))
        row["direction_top2_ratio"] = float((dir_near_energy[best_i] + dir_near_energy[second_i]) / (total_near + 1e-20))
        others = np.delete(dir_near_energy, best_i)
        row["direction_contrast"] = float(dir_near_energy[best_i] / (np.mean(others) + 1e-20)) if len(others) else 0.0

        # 衰减
        dists = []
        energies = []
        for dist in DISTANCES_CM:
            if dist in direction_distance_energy[best_direction]:
                dists.append(dist)
                energies.append(direction_distance_energy[best_direction][dist])
        attenuation_n, decay_R2 = fit_decay(dists, energies)
        row["attenuation_n"] = attenuation_n
        row["decay_R2"] = decay_R2

        if energies:
            near_e = np.mean([e for d, e in zip(dists, energies) if d <= 20]) if any(d <= 20 for d in dists) else np.mean(energies)
            far_e = np.mean([e for d, e in zip(dists, energies) if d >= 30]) if any(d >= 30 for d in dists) else np.mean(energies)
            row["near_far_ratio"] = float(near_e / (far_e + 1e-20))
            row["energy_5cm_best_direction"] = float(direction_distance_energy[best_direction].get(5, np.nan))
        else:
            row["near_far_ratio"] = np.nan
            row["energy_5cm_best_direction"] = np.nan

        if len(energies) >= 2:
            decreases = sum(1 for a, b in zip(energies[:-1], energies[1:]) if a >= b)
            row["monotonic_decay_ratio"] = decreases / (len(energies) - 1)
        else:
            row["monotonic_decay_ratio"] = 0.0

        # 最佳方向频谱
        best_band = matrix_near[best_i, :]
        best_total = float(np.sum(best_band))
        if best_total <= 1e-20:
            best_band = np.zeros(len(SUBBANDS), dtype=float)
            for _, b in direction_distance_band[best_direction].items():
                best_band += np.asarray(b, dtype=float)
            best_total = float(np.sum(best_band))

        row["energy_20_70"] = best_total
        for bj, (lo, hi) in enumerate(SUBBANDS):
            row[f"energy_{lo//1000}_{hi//1000}k"] = float(best_band[bj])
            row[f"ratio_{lo//1000}_{hi//1000}k"] = float(best_band[bj] / (best_total + 1e-20))

        row["energy_20_40"] = float(best_band[0] + best_band[1])
        row["energy_40_70"] = float(best_band[2] + best_band[3] + best_band[4])
        row["high_freq_ratio"] = float(row["energy_40_70"] / (best_total + 1e-20))

        chosen_spec = None
        for dist in DISTANCES_CM:
            if dist in direction_distance_spec[best_direction]:
                chosen_spec = direction_distance_spec[best_direction][dist]
                break
        if chosen_spec is not None:
            for k in [
                "spec_centroid_hz",
                "spec_bandwidth_hz",
                "spec_entropy",
                "spec_flatness",
                "spec_peak_freq_hz",
                "spec_rolloff_85_hz",
                "spec_slope",
                "spec_peakiness",
            ]:
                row[k] = chosen_spec.get(k, np.nan)

        # 简化 gm 特征，便于 pairwise 规则可能使用
        M = matrix_near
        total_M = float(np.sum(M))
        max_cell = float(np.max(M)) if M.size else 0.0
        row["gm_direction_entropy"] = entropy_norm(M.sum(axis=1))
        row["gm_direction_gini"] = gini_coefficient(M.sum(axis=1))
        row["gm_direction_cv"] = float(np.std(M.sum(axis=1)) / (np.mean(M.sum(axis=1)) + 1e-20))
        if max_cell > 0:
            strong12 = M >= max_cell * 0.12
            row["gm_thr12_active_cells"] = int(np.sum(strong12))
        else:
            row["gm_thr12_active_cells"] = 0

        # 时间特征
        time_source = center_files.get(center, "")
        if not time_source:
            # 用最佳方向最近距离作为时间源
            for dist in DISTANCES_CM:
                files = offset_files.get((center, best_direction, dist), [])
                if files:
                    time_source = files[0]
                    break
        row.update(time_features_from_wav(time_source) if time_source else {})

        rows.append(row)

    df = pd.DataFrame(rows)
    df = add_time_relative_features(df)
    return df


def add_time_relative_features(df):
    df = df.copy()
    numeric_cols = []
    for c in df.columns:
        if c in ["dataset_name", "time", "center", "center_norm", "best_direction"]:
            continue
        v = safe_float_series(df[c])
        if v.notna().mean() >= 0.75:
            numeric_cols.append(c)

    for c in numeric_cols:
        vals = safe_float_series(df[c])
        med = vals.median()
        mad = (vals - med).abs().median()
        if not np.isfinite(mad) or mad < 1e-12:
            mad = vals.std()
        if not np.isfinite(mad) or mad < 1e-12:
            mad = 1.0
        df[f"{c}__time_robust_z"] = (vals - med) / (1.4826 * mad)
        df[f"{c}__time_rank_pct"] = vals.rank(method="average", pct=True)
    return df



# ============================================================
# 5.5 多 time 文件夹逐 center 提取封装
# ============================================================

def resolve_input_time_folders(root):
    """
    允许输入：
      1) 一个 .ld 文件夹；
      2) 一个包含多个 .ld 子文件夹的根目录；
      3) 一个直接包含 wav 的普通文件夹。
    返回 [(time_name, time_folder), ...]
    """
    if not root or not os.path.exists(root):
        raise FileNotFoundError(f"输入文件夹不存在: {root}")

    root = os.path.abspath(root)
    base = os.path.basename(os.path.normpath(root))

    direct_wavs = []
    try:
        direct_wavs = [x for x in os.listdir(root) if any(x.endswith(ext) for ext in WAV_EXTS)]
    except Exception:
        direct_wavs = []

    # 当前目录本身就是一个 time 文件夹，或者 wav 直接在当前目录下
    if base.endswith('.ld') or len(direct_wavs) > 0:
        return [(base, root)]

    items = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        if len(list_wav_files(p)) == 0:
            continue
        items.append((name, p))

    if not items:
        # 兜底：递归能找到 wav，就把 root 当成一个 time
        if len(list_wav_files(root)) > 0:
            items.append((base, root))

    if not items:
        raise RuntimeError(f"输入目录里没有找到 wav: {root}")

    return items


def match_center_time_folder(center_root, time_name):
    if not center_root or not os.path.exists(center_root):
        return ""

    center_root = os.path.abspath(center_root)
    base = os.path.basename(os.path.normpath(center_root))
    if base == time_name:
        return center_root

    p = os.path.join(center_root, time_name)
    if os.path.exists(p):
        return p

    return ""


def extract_features_from_root_per_time(offset_root, center_root="", dataset_name="UNKNOWN"):
    """
    新版入口：对输入根目录下的每个 time_folder、每个 center 都提取一行特征。
    注意：输出每一行都是一个 center，不再只做整个文件夹判断。
    """
    parts = []
    time_folders = resolve_input_time_folders(offset_root)

    print("\n" + "=" * 100)
    print(f"逐 time / 逐 center 提取特征: {dataset_name}")
    print("offset_root:", offset_root)
    print("center_root:", center_root if center_root else "未提供")
    print("识别到 time_folder 数量:", len(time_folders))
    print("=" * 100)

    for time_name, offset_folder in time_folders:
        center_folder = match_center_time_folder(center_root, time_name) if center_root else ""
        df_one = extract_features_from_folder(offset_folder, center_folder, dataset_name=dataset_name)
        # extract_features_from_folder 已经把 time 写成 basename，这里强制修正为 resolve 出来的 time_name
        df_one["time"] = time_name
        parts.append(df_one)

    if not parts:
        raise RuntimeError("没有提取到任何特征。")

    out = pd.concat(parts, ignore_index=True)
    return out

# ============================================================
# 6. v7 模型推理
# ============================================================

def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_feature_columns_in_config(obj):
    """递归查找 config 中可能的 feature columns。"""
    if isinstance(obj, dict):
        for key in [
            "feature_columns",
            "final_feature_columns",
            "selected_features",
            "model_features",
            "features",
        ]:
            if key in obj and isinstance(obj[key], list) and all(isinstance(x, str) for x in obj[key]):
                if len(obj[key]) > 0:
                    return obj[key]
        for v in obj.values():
            found = find_feature_columns_in_config(v)
            if found:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = find_feature_columns_in_config(v)
            if found:
                return found
    return []


def find_threshold_in_config(config):
    for key in ["threshold", "global_threshold", "best_threshold", "decision_threshold", "recommended_threshold"]:
        if isinstance(config, dict) and key in config:
            try:
                return float(config[key])
            except Exception:
                pass
    return 0.5


def load_training_medians(feature_cols):
    medians = {c: 0.0 for c in feature_cols}
    if not os.path.exists(V7_TRAIN_FEATURE_CSV):
        return medians
    try:
        train = pd.read_csv(V7_TRAIN_FEATURE_CSV)
        for c in feature_cols:
            if c in train.columns:
                m = safe_float_series(train[c]).median()
                if np.isfinite(m):
                    medians[c] = float(m)
    except Exception:
        pass
    return medians


def predict_single_with_v7(feature_df):
    if joblib is None:
        raise RuntimeError("缺少 joblib，请先安装：pip install joblib")
    if not os.path.exists(V7_MODEL_PATH):
        raise FileNotFoundError(f"找不到 v7 模型文件: {V7_MODEL_PATH}")

    model = joblib.load(V7_MODEL_PATH)
    config = load_json(V7_CONFIG_PATH)

    feature_cols = find_feature_columns_in_config(config)

    if not feature_cols and hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)

    if not feature_cols:
        raise RuntimeError(
            "无法确定 v7 模型需要的特征列。请检查 v7_final_model_config.json 是否包含 feature_columns。"
        )

    threshold = find_threshold_in_config(config)
    medians = load_training_medians(feature_cols)

    X = pd.DataFrame(index=feature_df.index)
    missing = []
    for c in feature_cols:
        if c in feature_df.columns:
            vals = safe_float_series(feature_df[c])
            fill = medians.get(c, 0.0)
            X[c] = vals.fillna(fill).astype(float)
        else:
            X[c] = medians.get(c, 0.0)
            missing.append(c)

    if missing:
        print(f"[提示] 当前数据缺少 {len(missing)} 个模型特征，已用训练集中位数/0补齐。")
        print("缺失特征前20个:", missing[:20])

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
    else:
        raw = model.predict(X)
        prob = np.asarray(raw, dtype=float)

    pred = np.where(prob >= threshold, "TRUE_LEAK", "FALSE_LEAK")

    out = feature_df[["dataset_name", "time", "center_norm", "offset_wav_count_used"]].copy()
    out["prob_TRUE_LEAK"] = prob
    out["threshold"] = threshold
    out["pred_label"] = pred

    dataset_mean_prob = float(np.mean(prob))
    dataset_median_prob = float(np.median(prob))
    majority_true_ratio = float(np.mean(pred == "TRUE_LEAK"))
    dataset_pred = "TRUE_LEAK" if dataset_mean_prob >= DATASET_TRUE_THRESHOLD else "FALSE_LEAK"

    summary = {
        "mode": "single_v7",
        "n_centers": int(len(out)),
        "dataset_mean_prob_TRUE": dataset_mean_prob,
        "dataset_median_prob_TRUE": dataset_median_prob,
        "majority_TRUE_ratio": majority_true_ratio,
        "dataset_pred_label": dataset_pred,
        "model_threshold": threshold,
        "n_missing_model_features": len(missing),
    }

    return out, summary


# ============================================================
# 7. pairwise 规则推理
# ============================================================

def load_pairwise_rule():
    if not os.path.exists(PAIRWISE_RULE_PATH):
        raise FileNotFoundError(f"找不到 pairwise 规则: {PAIRWISE_RULE_PATH}")
    with open(PAIRWISE_RULE_PATH, "r", encoding="utf-8") as f:
        rule = json.load(f)
    selected = rule.get("selected_features", [])
    best_top_k = int(rule.get("best_top_k", len(selected)))
    if not selected:
        raise RuntimeError("pairwise rule 中没有 selected_features。")
    return rule, selected[:best_top_k]


def score_pairwise_rule(df, selected):
    score = np.zeros(len(df), dtype=float)
    total_weight = 0.0
    used = []
    missing = []

    for item in selected:
        f = item.get("feature")
        if f not in df.columns:
            missing.append(f)
            continue
        sign = int(item.get("sign_TRUE_larger", 1))
        med = float(item.get("median", 0.0))
        scale = float(item.get("scale", 1.0))
        weight = float(item.get("weight", 1.0))
        vals = safe_float_series(df[f]).fillna(med).values.astype(float)
        z = sign * (vals - med) / (scale + 1e-12)
        z = np.clip(z, -6, 6)
        score += weight * z
        total_weight += abs(weight)
        used.append(f)

    if total_weight > 0:
        score = score / total_weight

    return score, used, missing


def predict_pairwise(df_a, df_b):
    rule, selected = load_pairwise_rule()

    df_a = df_a.copy()
    df_b = df_b.copy()
    df_a["candidate"] = PAIRWISE_NAME_A
    df_b["candidate"] = PAIRWISE_NAME_B

    both = pd.concat([df_a, df_b], ignore_index=True)
    # pairwise 规则中可能含有 time_rank_pct，所以 A/B 合并后重新做 time 内部相对特征
    both = add_time_relative_features(both)
    both["pairwise_score"], used, missing = score_pairwise_rule(both, selected)

    rows = []
    keys_a = set(zip(df_a["time"].astype(str), df_a["center_norm"].astype(str)))
    keys_b = set(zip(df_b["time"].astype(str), df_b["center_norm"].astype(str)))
    common_keys = sorted(keys_a & keys_b)

    for time_name, center in common_keys:
        g = both[(both["time"].astype(str) == str(time_name)) & (both["center_norm"].astype(str) == str(center))].copy()
        if len(g) < 2:
            continue

        idx_max = g["pairwise_score"].astype(float).idxmax()
        for idx, r in g.iterrows():
            pred = "TRUE_LEAK" if idx == idx_max else "FALSE_LEAK"
            rows.append({
                "time": time_name,
                "center_norm": center,
                "center": r.get("center", center),
                "candidate": r["candidate"],
                "dataset_name": r.get("dataset_name", r["candidate"]),
                "pairwise_score": float(r["pairwise_score"]),
                "pred_label": pred,
                "offset_wav_count_used": r.get("offset_wav_count_used", np.nan),
                "pair_status": "matched",
            })

    # 未配对 center 也输出，避免用户误以为都判过
    missing_a = sorted(keys_a - keys_b)
    missing_b = sorted(keys_b - keys_a)
    for time_name, center in missing_a:
        g = both[(both["time"].astype(str) == str(time_name)) & (both["center_norm"].astype(str) == str(center)) & (both["candidate"] == PAIRWISE_NAME_A)]
        for _, r in g.iterrows():
            rows.append({
                "time": time_name,
                "center_norm": center,
                "center": r.get("center", center),
                "candidate": r["candidate"],
                "dataset_name": r.get("dataset_name", r["candidate"]),
                "pairwise_score": float(r["pairwise_score"]),
                "pred_label": "UNPAIRED",
                "offset_wav_count_used": r.get("offset_wav_count_used", np.nan),
                "pair_status": "missing_B_counterpart",
            })
    for time_name, center in missing_b:
        g = both[(both["time"].astype(str) == str(time_name)) & (both["center_norm"].astype(str) == str(center)) & (both["candidate"] == PAIRWISE_NAME_B)]
        for _, r in g.iterrows():
            rows.append({
                "time": time_name,
                "center_norm": center,
                "center": r.get("center", center),
                "candidate": r["candidate"],
                "dataset_name": r.get("dataset_name", r["candidate"]),
                "pairwise_score": float(r["pairwise_score"]),
                "pred_label": "UNPAIRED",
                "offset_wav_count_used": r.get("offset_wav_count_used", np.nan),
                "pair_status": "missing_A_counterpart",
            })

    pred = pd.DataFrame(rows)

    summary = {
        "mode": "pairwise_144226_rule_per_center",
        "n_common_time_center_pairs": int(len(common_keys)),
        "n_unpaired_A": int(len(missing_a)),
        "n_unpaired_B": int(len(missing_b)),
        "used_rule_features": "|".join(used),
        "missing_rule_features": "|".join(missing),
        "note": "pairwise模式逐 time + center 成对比较；同一对里分数高者判 TRUE_LEAK，另一个判 FALSE_LEAK。",
    }

    return pred, summary


# ============================================================
# 8. 主程序
# ============================================================

def save_report(summary, paths):
    report_path = os.path.join(OUTPUT_DIR, "inference_report.txt")
    lines = []
    lines.append("泄漏真假验证报告")
    lines.append("=" * 80)
    lines.append(f"生成时间: {datetime.now()}")
    lines.append("")
    lines.append("配置:")
    lines.append(f"  MODE: {MODE}")
    lines.append(f"  V7_MODEL_PATH: {V7_MODEL_PATH}")
    lines.append(f"  V7_CONFIG_PATH: {V7_CONFIG_PATH}")
    lines.append(f"  PAIRWISE_RULE_PATH: {PAIRWISE_RULE_PATH}")
    lines.append("")
    lines.append("结果摘要:")
    for k, v in summary.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("输出文件:")
    for k, v in paths.items():
        lines.append(f"  {k}: {v}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


def main():
    ensure_dir(OUTPUT_DIR)

    print("=" * 100)
    print("使用已训练模型/规则验证数据集 TRUE_LEAK / FALSE_LEAK")
    print("=" * 100)
    print("MODE:", MODE)

    if MODE == "single":
        feature_df = extract_features_from_root_per_time(
            SINGLE_OFFSET_FOLDER,
            SINGLE_CENTER_FOLDER,
            dataset_name="SINGLE_UNKNOWN",
        )
        feature_path = os.path.join(OUTPUT_DIR, "single_extracted_features.csv")
        feature_df.to_csv(feature_path, index=False, encoding="utf-8-sig")

        pred_df, summary = predict_single_with_v7(feature_df)
        pred_path = os.path.join(OUTPUT_DIR, "single_per_center_predictions.csv")
        summary_path = os.path.join(OUTPUT_DIR, "single_per_center_summary.csv")
        pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
        pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding="utf-8-sig")

        paths = {
            "features": feature_path,
            "predictions": pred_path,
            "summary": summary_path,
        }
        report_path = save_report(summary, paths)

        print("\n" + "=" * 100)
        print("验证完成")
        print("=" * 100)
        print("数据集整体判断:", summary["dataset_pred_label"])
        print(f"平均 TRUE 概率: {summary['dataset_mean_prob_TRUE']:.4f}")
        print(f"中位 TRUE 概率: {summary['dataset_median_prob_TRUE']:.4f}")
        print(f"center中判TRUE比例: {summary['majority_TRUE_ratio']:.4f}")
        print("预测明细:", pred_path)
        print("报告:", report_path)

    elif MODE == "pairwise":
        df_a = extract_features_from_root_per_time(
            PAIRWISE_OFFSET_FOLDER_A,
            PAIRWISE_CENTER_FOLDER_A,
            dataset_name=PAIRWISE_NAME_A,
        )
        df_b = extract_features_from_root_per_time(
            PAIRWISE_OFFSET_FOLDER_B,
            PAIRWISE_CENTER_FOLDER_B,
            dataset_name=PAIRWISE_NAME_B,
        )

        feature_path_a = os.path.join(OUTPUT_DIR, "pairwise_A_extracted_features.csv")
        feature_path_b = os.path.join(OUTPUT_DIR, "pairwise_B_extracted_features.csv")
        df_a.to_csv(feature_path_a, index=False, encoding="utf-8-sig")
        df_b.to_csv(feature_path_b, index=False, encoding="utf-8-sig")

        pred_df, summary = predict_pairwise(df_a, df_b)
        pred_path = os.path.join(OUTPUT_DIR, "pairwise_per_center_predictions.csv")
        summary_path = os.path.join(OUTPUT_DIR, "pairwise_per_center_summary.csv")
        pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
        pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding="utf-8-sig")

        paths = {
            "features_A": feature_path_a,
            "features_B": feature_path_b,
            "predictions": pred_path,
            "summary": summary_path,
        }
        report_path = save_report(summary, paths)

        print("\n" + "=" * 100)
        print("pairwise验证完成")
        print("=" * 100)
        print("共同center数量:", summary["n_common_centers"])
        print("使用规则特征:", summary["used_rule_features"])
        if summary["missing_rule_features"]:
            print("缺失规则特征:", summary["missing_rule_features"])
        print("预测明细:", pred_path)
        print("报告:", report_path)

    else:
        raise ValueError('MODE 只能是 "single" 或 "pairwise"')


if __name__ == "__main__":
    main()
