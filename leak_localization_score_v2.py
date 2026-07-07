# -*- coding: utf-8 -*-
"""
leak_localization_score_v2.py

用途:
    适合“真实泄漏点 + 周边扫描点”的数据。
    这个程序不再强行做 TRUE/FALSE 二分类，
    而是做:
        1. 泄漏响应强弱排序
        2. 最可能泄漏方向
        3. 距离衰减物理特征
        4. 高频比例特征
        5. 每个中心点的可疑等级

为什么这样改:
    您的数据本身都是泄漏场数据，40cm点也可能含有泄漏声。
    所以默认不再使用 40cm 当背景相减。
    默认直接分析每个点的 20-70kHz 原始超声能量分布。

运行:
    python leak_localization_score_v2.py

输出:
    leak_localization_results/
        leak_localization_summary.csv        总结果
        energy_matrix_detail.csv             8方向×8距离能量矩阵明细
        figures/                             每个中心点的衰减曲线图
"""

import os
import glob
import re
import csv
import time

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt


# ============================================================
# 1. 参数配置：一般只需要改这里
# ============================================================

time_folders = [
    "HM20260626_142938.ld",
    "HM20260626_143034.ld",
    "HM20260626_144226.ld",
    "HM20260626_144325.ld",
]

# 中心点波束形成结果
center_root_dir = r"D:\gas\beamform_results"

# 周边偏移点波束形成结果
offset_root_dir = r"D:\gas\beamform_results_offset_multiple"

# 调试时可以设成 3，只跑每个时间点前3个中心点
# 正式跑全部数据时设成 None
MAX_CENTERS_PER_TIME = None

# 是否保存每个中心点的图片
SAVE_FIGURES = True

# 默认不要用40cm做背景扣除
# 因为您的40cm点也是泄漏场的一部分
USE_40CM_BACKGROUND_SUBTRACTION = False

# 频谱范围
FREQ_LOW = 20000
FREQ_HIGH = 70000
FREQ_SPLIT = 40000

# Welch频谱参数
NFFT = 4096

# 扫描距离，单位 cm
distances = np.array([5, 10, 15, 20, 25, 30, 35, 40], dtype=float)

# 八个方向
directions = [
    "up", "down", "left", "right",
    "up_left", "down_left",
    "up_right", "down_right"
]


# ============================================================
# 2. 基础工具函数
# ============================================================

def safe_name(name):
    """把文件夹名转成适合保存文件的名字"""
    return re.sub(r"[^0-9a-zA-Z_\-]+", "_", str(name))


def read_wav(file_path):
    """读取 wav，返回 fs, x, rms"""
    if not os.path.exists(file_path):
        return None, None, None

    try:
        fs, x = wav.read(file_path)

        if len(x.shape) > 1:
            x = x[:, 0]

        x = x.astype(np.float32)
        rms = float(np.sqrt(np.mean(x * x)) + 1e-12)

        return fs, x, rms

    except Exception as e:
        print("[读取失败]", file_path, e)
        return None, None, None


def compute_psd(file_path, norm_factor=None):
    """
    计算 20-70kHz PSD。
    norm_factor 使用中心点 rms，使同一中心点周边数据可比。
    """
    fs, x, rms = read_wav(file_path)

    if x is None:
        return None, None, None

    if norm_factor is None:
        norm_factor = rms

    x = x / (norm_factor + 1e-12)

    nperseg = min(NFFT, len(x))

    if nperseg < 16:
        return None, None, rms

    freq, psd = signal.welch(
        x,
        fs=fs,
        nperseg=nperseg,
        scaling="density"
    )

    mask = (freq >= FREQ_LOW) & (freq <= FREQ_HIGH)

    if not np.any(mask):
        return None, None, rms

    return freq[mask], psd[mask], rms


def integrate_energy(freq, psd, low=None, high=None):
    """积分计算指定频段能量"""
    if freq is None or psd is None or len(freq) == 0 or len(psd) == 0:
        return 0.0

    if low is None:
        low = FREQ_LOW
    if high is None:
        high = FREQ_HIGH

    mask = (freq >= low) & (freq <= high)

    if not np.any(mask):
        return 0.0

    return float(np.trapz(psd[mask], freq[mask]))


def spectrum_subtract(psd, bg_psd):
    """可选背景扣除，默认不用"""
    if psd is None or bg_psd is None:
        return None

    n = min(len(psd), len(bg_psd))

    if n <= 0:
        return None

    return np.maximum(psd[:n] - bg_psd[:n], 0.0)


# ============================================================
# 3. 文件搜索函数
# ============================================================

def detect_center_ids(center_dir):
    """自动识别中心点编号"""
    files = glob.glob(os.path.join(center_dir, "*_beamform_result.wav"))

    ids = []

    for f in files:
        name = os.path.basename(f)
        m = re.search(r"_(\d+)_beamform_result", name)
        if m:
            ids.append(m.group(1))

    return sorted(set(ids))


def find_center_file(center_dir, center_id):
    files = glob.glob(
        os.path.join(center_dir, f"*_{center_id}_beamform_result.wav")
    )
    return files[0] if files else None


def find_offset_file(offset_dir, center_id, distance, direction):
    files = glob.glob(
        os.path.join(offset_dir, f"*_{center_id}d{int(distance)}_{direction}*.wav")
    )
    return files[0] if files else None


# ============================================================
# 4. 距离衰减物理模型
# ============================================================

def power_decay_model(r, A, n):
    """
    距离衰减模型:
        E(r) = A * r^(-n)

    对波束形成后的数据，不强行要求 n 接近2。
    只要 n 为正，且 R2较高，就说明存在稳定的空间衰减趋势。
    """
    return A * np.power(r, -n)


def fit_decay(energy_list):
    """
    拟合距离衰减。
    返回:
        decay_score, n, R2
    """
    r = distances[:len(energy_list)]
    e = np.array(energy_list, dtype=float)

    mask = e > 0
    r = r[mask]
    e = e[mask]

    if len(r) < 3:
        return 0.0, 0.0, 0.0

    try:
        popt, _ = curve_fit(
            power_decay_model,
            r,
            e,
            p0=[max(e[0] * 25.0, 1e-12), 1.0],
            bounds=([0.0, 0.0], [np.inf, 5.0]),
            maxfev=5000
        )

        A, n = popt
        pred = power_decay_model(r, A, n)

        ss_res = np.sum((e - pred) ** 2)
        ss_tot = np.sum((e - np.mean(e)) ** 2)

        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        r2 = float(np.clip(r2, 0.0, 1.0))

        # 对波束形成数据放宽 n 的要求:
        # n>=0.3 就认为具有可接受的衰减趋势
        if n >= 0.3:
            n_score = 1.0
        elif n <= 0.0:
            n_score = 0.0
        else:
            n_score = n / 0.3

        decay_score = r2 * n_score

        return float(decay_score), float(n), float(r2)

    except Exception:
        return 0.0, 0.0, 0.0


def near_field_score(energy_list):
    """
    近场优势分数:
        如果 5/10cm 能量明显高于远处均值，说明更像近场泄漏源。
    """
    e = np.array(energy_list, dtype=float)

    if len(e) < 4 or np.max(e) <= 0:
        return 0.0, 0.0

    near = np.mean(e[:2])       # 5cm和10cm
    far = np.mean(e[-3:])       # 30cm、35cm、40cm

    ratio = near / (far + 1e-12)

    # 映射到 0~1
    # ratio <=1: 近处不强
    # ratio >=2: 近处明显强
    score = (ratio - 1.0) / (2.0 - 1.0)
    score = float(np.clip(score, 0.0, 1.0))

    return score, float(ratio)


def map_high_freq_score(hf_ratio):
    """
    高频比例分数。
    这里只给较低权重，不再让 40-70kHz 决定生死。
    """
    # 真实泄漏不一定全部集中在40-70kHz，
    # 所以 0.2 以上就给一定分，0.5以上满分。
    score = (hf_ratio - 0.2) / (0.5 - 0.2)
    return float(np.clip(score, 0.0, 1.0))


# ============================================================
# 5. 单中心点特征提取
# ============================================================

def analyze_one_center(time_folder, center_id):
    """
    分析一个中心点，返回:
        center_feature: dict
        energy_rows: list
        energy_matrix: dict
    """
    center_dir = os.path.join(center_root_dir, time_folder)
    offset_dir = os.path.join(offset_root_dir, time_folder)

    center_file = find_center_file(center_dir, center_id)

    if center_file is None:
        return None, [], {}

    _, _, center_rms = read_wav(center_file)

    if center_rms is None:
        return None, [], {}

    energy_matrix = {}
    high_ratio_matrix = {}
    direction_features = {}
    energy_rows = []

    for direction in directions:
        bg_freq = None
        bg_psd = None

        if USE_40CM_BACKGROUND_SUBTRACTION:
            bg_file = find_offset_file(offset_dir, center_id, 40, direction)
            if bg_file is not None:
                bg_freq, bg_psd, _ = compute_psd(bg_file, center_rms)

        energy_list = []
        high_ratio_list = []

        for dist in distances:
            offset_file = find_offset_file(offset_dir, center_id, dist, direction)

            if offset_file is None:
                energy_list.append(0.0)
                high_ratio_list.append(0.0)
                continue

            freq, psd, _ = compute_psd(offset_file, center_rms)

            if freq is None or psd is None:
                energy_list.append(0.0)
                high_ratio_list.append(0.0)
                continue

            # 默认不做背景扣除
            used_psd = psd

            if USE_40CM_BACKGROUND_SUBTRACTION and bg_psd is not None:
                tmp = spectrum_subtract(psd, bg_psd)
                if tmp is None:
                    used_psd = psd
                else:
                    n = min(len(freq), len(tmp))
                    freq = freq[:n]
                    used_psd = tmp[:n]

            total_e = integrate_energy(freq, used_psd, FREQ_LOW, FREQ_HIGH)
            low_e = integrate_energy(freq, used_psd, FREQ_LOW, FREQ_SPLIT)
            high_e = integrate_energy(freq, used_psd, FREQ_SPLIT, FREQ_HIGH)

            hf_ratio = high_e / (total_e + 1e-12)

            energy_list.append(total_e)
            high_ratio_list.append(hf_ratio)

            energy_rows.append({
                "time": time_folder,
                "center": center_id,
                "direction": direction,
                "distance_cm": int(dist),
                "energy_20_70": total_e,
                "energy_20_40": low_e,
                "energy_40_70": high_e,
                "high_freq_ratio": hf_ratio
            })

        energy_matrix[direction] = energy_list
        high_ratio_matrix[direction] = high_ratio_list

        d_score, n, r2 = fit_decay(energy_list)
        nf_score, nf_ratio = near_field_score(energy_list)

        direction_features[direction] = {
            "max_energy": float(np.max(energy_list)) if energy_list else 0.0,
            "energy_5cm": float(energy_list[0]) if len(energy_list) else 0.0,
            "mean_energy": float(np.mean(energy_list)) if len(energy_list) else 0.0,
            "decay_score": d_score,
            "attenuation_n": n,
            "decay_R2": r2,
            "near_score": nf_score,
            "near_far_ratio": nf_ratio,
            "mean_high_freq_ratio": float(np.mean(high_ratio_list)) if high_ratio_list else 0.0,
            "high_freq_ratio_5cm": float(high_ratio_list[0]) if len(high_ratio_list) else 0.0
        }

    if not direction_features:
        return None, energy_rows, energy_matrix

    # 最大能量方向
    energy_direction = max(
        direction_features.keys(),
        key=lambda d: direction_features[d]["max_energy"]
    )

    # 最好衰减方向
    decay_direction = max(
        direction_features.keys(),
        key=lambda d: direction_features[d]["decay_score"]
    )

    # 综合方向：同时考虑能量和衰减
    max_energy_all = max(
        direction_features[d]["max_energy"] for d in direction_features
    ) + 1e-12

    direction_combined_scores = {}

    for d, feat in direction_features.items():
        local_energy_score = feat["max_energy"] / max_energy_all
        local_score = (
            0.45 * local_energy_score
            + 0.40 * feat["decay_score"]
            + 0.15 * feat["near_score"]
        )
        direction_combined_scores[d] = local_score

    best_direction = max(direction_combined_scores, key=direction_combined_scores.get)
    best_feat = direction_features[best_direction]

    # 方向一致性:
    # 如果最大能量方向、最佳衰减方向、综合方向一致，说明更可靠
    agree_count = 0
    if best_direction == energy_direction:
        agree_count += 1
    if best_direction == decay_direction:
        agree_count += 1
    direction_agreement = agree_count / 2.0

    center_feature = {
        "time": time_folder,
        "center": center_id,
        "best_direction": best_direction,
        "energy_direction": energy_direction,
        "decay_direction": decay_direction,
        "raw_best_energy": best_feat["max_energy"],
        "energy_5cm": best_feat["energy_5cm"],
        "mean_energy": best_feat["mean_energy"],
        "decay_score": best_feat["decay_score"],
        "attenuation_n": best_feat["attenuation_n"],
        "decay_R2": best_feat["decay_R2"],
        "near_score": best_feat["near_score"],
        "near_far_ratio": best_feat["near_far_ratio"],
        "high_freq_ratio": best_feat["mean_high_freq_ratio"],
        "high_freq_ratio_5cm": best_feat["high_freq_ratio_5cm"],
        "high_freq_score": map_high_freq_score(best_feat["mean_high_freq_ratio"]),
        "direction_agreement": direction_agreement,
        "direction_combined_score": direction_combined_scores[best_direction],
    }

    return center_feature, energy_rows, energy_matrix


# ============================================================
# 6. 时间点内归一化评分
# ============================================================

def assign_scores_for_one_time(center_features):
    """
    对同一个时间点里的中心点做归一化排序。
    因为不同时间点整体声强可能不同，所以建议每个时间点内部排名。
    """
    if not center_features:
        return []

    energies = np.array([x["raw_best_energy"] for x in center_features], dtype=float)

    # 用95分位数做归一化，避免单个异常极大值把其他点压扁
    ref = np.percentile(energies, 95)

    if ref <= 0:
        ref = np.max(energies) + 1e-12

    for item in center_features:
        energy_score = float(np.clip(item["raw_best_energy"] / (ref + 1e-12), 0.0, 1.0))
        item["energy_score"] = energy_score

        # 最终定位分数:
        # 能量强弱 + 衰减规律 + 近场优势 + 方向一致性 + 高频比例
        localization_score = (
            0.35 * energy_score
            + 0.35 * item["decay_score"]
            + 0.15 * item["near_score"]
            + 0.10 * item["direction_agreement"]
            + 0.05 * item["high_freq_score"]
        )

        item["localization_score"] = float(localization_score)

    # 排名
    center_features = sorted(
        center_features,
        key=lambda x: x["localization_score"],
        reverse=True
    )

    for rank, item in enumerate(center_features, start=1):
        item["rank_in_time"] = rank

        s = item["localization_score"]

        if s >= 0.70:
            level = "STRONG_LEAK_ZONE"
        elif s >= 0.50:
            level = "LIKELY_LEAK_ZONE"
        elif s >= 0.35:
            level = "WEAK_LEAK_ZONE"
        else:
            level = "LOW_RESPONSE_ZONE"

        item["zone_level"] = level

    return center_features


# ============================================================
# 7. 绘图
# ============================================================

def plot_center_result(time_folder, center_id, energy_matrix, center_feature, fig_dir):
    """保存每个中心点的能量衰减曲线图"""
    os.makedirs(fig_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]

    for d in directions:
        e = energy_matrix.get(d, [])
        if len(e) == len(distances):
            ax.plot(distances, e, marker="o", label=d)

    ax.set_title(f"Energy Decay | {time_folder} | Center {center_id}")
    ax.set_xlabel("Distance (cm)")
    ax.set_ylabel("Energy 20-70 kHz")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.axis("off")

    text = (
        f"Zone level: {center_feature.get('zone_level', '')}\n"
        f"Localization score: {center_feature.get('localization_score', 0):.3f}\n"
        f"Rank in time: {center_feature.get('rank_in_time', 0)}\n\n"
        f"Best direction: {center_feature['best_direction']}\n"
        f"Energy direction: {center_feature['energy_direction']}\n"
        f"Decay direction: {center_feature['decay_direction']}\n\n"
        f"Energy score: {center_feature.get('energy_score', 0):.3f}\n"
        f"Decay score: {center_feature['decay_score']:.3f}\n"
        f"Near score: {center_feature['near_score']:.3f}\n"
        f"Direction agreement: {center_feature['direction_agreement']:.3f}\n"
        f"High freq score: {center_feature['high_freq_score']:.3f}\n\n"
        f"n: {center_feature['attenuation_n']:.3f}\n"
        f"R2: {center_feature['decay_R2']:.3f}\n"
        f"Near/Far: {center_feature['near_far_ratio']:.3f}\n"
        f"High freq ratio: {center_feature['high_freq_ratio']:.3f}"
    )

    ax.text(0.05, 0.95, text, va="top", fontsize=11)

    plt.tight_layout()

    save_path = os.path.join(
        fig_dir,
        f"{safe_name(time_folder)}_center_{center_id}.png"
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# 8. 主流程
# ============================================================

def main():
    start = time.time()

    output_root = os.path.join(os.getcwd(), "leak_localization_results")
    fig_dir = os.path.join(output_root, "figures")
    os.makedirs(output_root, exist_ok=True)

    summary_csv = os.path.join(output_root, "leak_localization_summary.csv")
    matrix_csv = os.path.join(output_root, "energy_matrix_detail.csv")

    all_summary_rows = []
    all_matrix_rows = []

    print("=" * 80)
    print("泄漏定位评分程序 v2 开始运行")
    print("默认模式: 不使用40cm背景扣除，直接分析泄漏场原始超声能量")
    print("结果文件夹:", output_root)
    print("=" * 80)

    # 保存每个中心点的能量矩阵，供绘图用
    plot_cache = []

    for time_folder in time_folders:
        print("\n" + "-" * 80)
        print("处理时间点:", time_folder)

        center_dir = os.path.join(center_root_dir, time_folder)
        offset_dir = os.path.join(offset_root_dir, time_folder)

        if not os.path.exists(center_dir):
            print("[跳过] 中心点目录不存在:", center_dir)
            continue

        if not os.path.exists(offset_dir):
            print("[跳过] 偏移点目录不存在:", offset_dir)
            continue

        ids = detect_center_ids(center_dir)

        if MAX_CENTERS_PER_TIME is not None:
            ids = ids[:MAX_CENTERS_PER_TIME]

        print("中心点数量:", len(ids))

        time_features = []

        for i, center_id in enumerate(ids, start=1):
            print(f"  [{i}/{len(ids)}] 分析中心点 {center_id} ...")

            feature, energy_rows, energy_matrix = analyze_one_center(time_folder, center_id)

            all_matrix_rows.extend(energy_rows)

            if feature is None:
                print("    无有效结果")
                continue

            time_features.append(feature)
            plot_cache.append((time_folder, center_id, energy_matrix, feature))

        # 这个时间点内部做归一化排名
        time_features = assign_scores_for_one_time(time_features)

        for item in time_features:
            all_summary_rows.append(item)

        # 更新 plot_cache 里的 feature，使其包含排名和zone_level
        feature_map = {
            (x["time"], x["center"]): x for x in time_features
        }

        for idx, (tf, cid, matrix, old_feature) in enumerate(plot_cache):
            if tf == time_folder and (tf, cid) in feature_map:
                plot_cache[idx] = (tf, cid, matrix, feature_map[(tf, cid)])

        if time_features:
            print("  当前时间点Top 5:")
            for item in time_features[:5]:
                print(
                    f"    center={item['center']}, "
                    f"level={item['zone_level']}, "
                    f"score={item['localization_score']:.3f}, "
                    f"dir={item['best_direction']}, "
                    f"R2={item['decay_R2']:.3f}, "
                    f"n={item['attenuation_n']:.3f}"
                )

    # 写总结果 CSV
    summary_fields = [
        "time",
        "center",
        "rank_in_time",
        "zone_level",
        "localization_score",
        "best_direction",
        "energy_direction",
        "decay_direction",
        "raw_best_energy",
        "energy_score",
        "energy_5cm",
        "mean_energy",
        "decay_score",
        "attenuation_n",
        "decay_R2",
        "near_score",
        "near_far_ratio",
        "direction_agreement",
        "high_freq_ratio",
        "high_freq_ratio_5cm",
        "high_freq_score",
        "direction_combined_score"
    ]

    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for row in all_summary_rows:
            writer.writerow({k: row.get(k, "") for k in summary_fields})

    # 写能量矩阵明细 CSV
    matrix_fields = [
        "time",
        "center",
        "direction",
        "distance_cm",
        "energy_20_70",
        "energy_20_40",
        "energy_40_70",
        "high_freq_ratio"
    ]

    with open(matrix_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=matrix_fields)
        writer.writeheader()
        for row in all_matrix_rows:
            writer.writerow({k: row.get(k, "") for k in matrix_fields})

    # 保存图片
    if SAVE_FIGURES:
        print("\n正在保存图片...")
        for tf, cid, matrix, feature in plot_cache:
            plot_center_result(tf, cid, matrix, feature, fig_dir)

    cost = time.time() - start

    print("\n" + "=" * 80)
    print("全部完成")
    print("总结果CSV:", summary_csv)
    print("能量矩阵CSV:", matrix_csv)
    if SAVE_FIGURES:
        print("图片文件夹:", fig_dir)
    print(f"耗时: {cost:.1f} 秒")
    print("=" * 80)


if __name__ == "__main__":
    main()
