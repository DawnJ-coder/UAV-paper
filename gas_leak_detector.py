# -*- coding: utf-8 -*-
"""
气体泄漏声学检测 —— 基线方案 (Benchmark)
=========================================

处理流程:
    1. 读取 WAV (建议 192 kHz 采样, 与你的实验一致)
    2. 18–50 kHz Butterworth 带通滤波 (零相位 sosfiltfilt)
    3. 滑动窗计算带内能量 (dB)
    4. 判决逻辑复现自文献 [26] (arXiv:2511.00348,
       "Ultralow-power standoff acoustic leak detection"):
         a. 窗口削波/过载            -> NOISE  (环境噪声事件)
         b. 能量 E <= mu_B + k*sigma_B -> QUIET  (无泄漏)
         c. 超阈值后取连续 M 个窗口:
              若 2*std > mean          -> NOISE  (能量波动过大, 环境不可靠)
              若 M 个能量全部 > 阈值    -> LEAK   (持续稳定的带内信号 = 泄漏)
              否则                     -> QUIET  (脉冲型干扰)
    5. 输出逐窗判决、统计摘要, 并绘制结果图 (PNG)

背景阈值 (mu_B, sigma_B) 的来源, 两种方式任选:
    A. --background bg.wav      用单独录制的纯环境噪声文件估计 (推荐)
    B. --calib-seconds 5        用待测文件开头 N 秒 (须确认该段无泄漏) 估计

用法示例:
    python gas_leak_detector.py --input leak.wav --background ambient.wav
    python gas_leak_detector.py --input leak.wav --calib-seconds 5
    python gas_leak_detector.py --input leak.wav --background ambient.wav \
        --lowcut 18000 --highcut 50000 --win-ms 50 --hop-ms 25 --k-sigma 3 --m-stable 5

依赖: numpy, scipy, matplotlib   (pip install numpy scipy matplotlib)
"""

import argparse
import sys
from dataclasses import dataclass, field

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, welch

import matplotlib

matplotlib.use("Agg")  # 无显示环境下也能保存图片
import matplotlib.pyplot as plt

# 中文字体兜底(若系统无中文字体, 自动退回英文标签不报错)
plt.rcParams["font.sans-serif"] = ["SimHei", "Noto Sans CJK SC",
                                   "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ----------------------------------------------------------------------
# 1. 数据读取
# ----------------------------------------------------------------------
def load_wav(path):
    """读取 WAV, 返回 (fs, 单声道 float64 信号, 是否定点格式的满量程值)。

    满量程值用于削波(ADC 过载)检测: 文献[26]中过载窗口直接判为 NOISE。
    """
    fs, x = wavfile.read(path)

    if x.ndim > 1:                       # 多声道取第一声道
        x = x[:, 0]

    if np.issubdtype(x.dtype, np.integer):
        full_scale = float(np.iinfo(x.dtype).max)
        x = x.astype(np.float64) / full_scale
        clip_level = 0.999               # 归一化后接近 1 即视为削波
    else:                                # float WAV, 约定满量程为 1.0
        x = x.astype(np.float64)
        clip_level = 0.999

    return fs, x, clip_level


# ----------------------------------------------------------------------
# 2. 带通滤波
# ----------------------------------------------------------------------
def bandpass(x, fs, lowcut, highcut, order=6):
    """18–50 kHz Butterworth 带通, 零相位滤波。

    若采样率不足以支持 highcut(需 fs/2 > highcut), 自动收缩上限并告警。
    """
    nyq = fs / 2.0
    hi = highcut
    if hi >= nyq:
        hi = 0.95 * nyq
        print(f"[警告] 采样率 {fs} Hz 不支持上限 {highcut} Hz, "
              f"已收缩到 {hi:.0f} Hz")
    if lowcut >= hi:
        raise ValueError(f"带通下限 {lowcut} Hz >= 上限 {hi:.0f} Hz, 无法滤波。"
                         f"请检查采样率(当前 {fs} Hz)。")
    sos = butter(order, [lowcut / nyq, hi / nyq], btype="bandpass",
                 output="sos")
    return sosfiltfilt(sos, x), hi


# ----------------------------------------------------------------------
# 3. 滑动窗带内能量
# ----------------------------------------------------------------------
def frame_energies(x_filt, x_raw, fs, win_ms=50.0, hop_ms=25.0,
                   clip_level=0.999, eps=1e-20):
    """逐窗计算: 带内能量(dB, 相对满量程) 与 削波标志。

    返回:
        t_centers : 各窗中心时刻 (s)
        energy_db : 各窗带内能量 10*log10(mean(x_filt^2))
        clipped   : 各窗是否存在削波样本 (基于滤波前原始信号)
    """
    win = int(round(win_ms * 1e-3 * fs))
    hop = int(round(hop_ms * 1e-3 * fs))
    n = len(x_filt)
    if n < win:
        raise ValueError("信号长度不足一个分析窗。")

    starts = np.arange(0, n - win + 1, hop)
    t_centers = (starts + win / 2.0) / fs

    energy_db = np.empty(len(starts))
    clipped = np.zeros(len(starts), dtype=bool)
    for i, s in enumerate(starts):
        seg_f = x_filt[s:s + win]
        seg_r = x_raw[s:s + win]
        energy_db[i] = 10.0 * np.log10(np.mean(seg_f ** 2) + eps)
        clipped[i] = np.max(np.abs(seg_r)) >= clip_level
    return t_centers, energy_db, clipped


# ----------------------------------------------------------------------
# 4. 背景统计 (mu_B, sigma_B)
# ----------------------------------------------------------------------
def background_stats(energy_db, clipped):
    """由背景能量序列估计 mu_B 与 sigma_B (剔除削波窗)。"""
    valid = energy_db[~clipped]
    if len(valid) < 10:
        raise ValueError("有效背景窗口太少(<10), 无法可靠估计阈值。")
    return float(np.mean(valid)), float(np.std(valid))


# ----------------------------------------------------------------------
# 5. 判决状态机 (复现文献[26]的逻辑, 推广为流式逐窗版本)
# ----------------------------------------------------------------------
QUIET, LEAK, NOISE = 0, 1, 2
LABELS = {QUIET: "QUIET", LEAK: "LEAK", NOISE: "NOISE"}


@dataclass
class DetectorConfig:
    k_sigma: float = 3.0        # 阈值 = mu_B + k_sigma * sigma_B (文献中 k=1, 实际建议 2~3 抑制虚警)
    m_stable: int = 5           # 稳定性校验所需连续窗口数 (文献为 5 次测量)
    stability_factor: float = 2.0  # 文献判据: 2*sigma > mean(线性能量) 即视为波动过大


@dataclass
class DetectionResult:
    decisions: np.ndarray = field(default=None)
    threshold_db: float = 0.0
    mu_b: float = 0.0
    sigma_b: float = 0.0


def detect(energy_db, clipped, mu_b, sigma_b, cfg: DetectorConfig):
    """逐窗三态判决: QUIET / LEAK / NOISE。

    对每个窗口 i:
      - 削波            -> NOISE
      - E <= 阈值        -> QUIET
      - E  > 阈值        -> 检查含自身在内的连续 m_stable 个窗口:
            任一窗削波                     -> NOISE
            线性能量波动 2*std > mean      -> NOISE (环境不稳定)
            m_stable 个窗能量全部 > 阈值    -> LEAK
            否则(脉冲型超阈)               -> QUIET
    注: 信号末尾不足 m_stable 个窗口时, 用已有窗口从严判决。
    """
    thr = mu_b + cfg.k_sigma * sigma_b
    n = len(energy_db)
    decisions = np.full(n, QUIET, dtype=int)

    energy_lin = 10.0 ** (energy_db / 10.0)   # 稳定性判据在线性能量域计算

    for i in range(n):
        if clipped[i]:
            decisions[i] = NOISE
            continue
        if energy_db[i] <= thr:
            decisions[i] = QUIET
            continue

        # ---- 超阈值: 进入稳定性校验 ----
        j_end = min(i + cfg.m_stable, n)
        seg_db = energy_db[i:j_end]
        seg_lin = energy_lin[i:j_end]
        seg_clip = clipped[i:j_end]

        if seg_clip.any():
            decisions[i] = NOISE
            continue

        mean_lin = np.mean(seg_lin)
        std_lin = np.std(seg_lin)
        if cfg.stability_factor * std_lin > mean_lin:
            decisions[i] = NOISE          # 波动过大 -> 环境噪声事件
        elif np.all(seg_db > thr) and (j_end - i) == cfg.m_stable:
            decisions[i] = LEAK           # 持续稳定超阈 -> 泄漏
        else:
            decisions[i] = QUIET          # 脉冲式超阈 -> 不算泄漏

    res = DetectionResult()
    res.decisions = decisions
    res.threshold_db = thr
    res.mu_b = mu_b
    res.sigma_b = sigma_b
    return res


# ----------------------------------------------------------------------
# 6. 汇总与可视化
# ----------------------------------------------------------------------
def summarize(t, energy_db, res: DetectionResult, hop_ms):
    d = res.decisions
    n = len(d)
    n_leak = int(np.sum(d == LEAK))
    n_quiet = int(np.sum(d == QUIET))
    n_noise = int(np.sum(d == NOISE))

    print("=" * 62)
    print("检测结果摘要")
    print("=" * 62)
    print(f"  背景带内能量:  mu_B = {res.mu_b:.2f} dB, "
          f"sigma_B = {res.sigma_b:.2f} dB")
    print(f"  判决阈值:      {res.threshold_db:.2f} dB")
    print(f"  窗口总数:      {n}")
    print(f"  LEAK  (泄漏):  {n_leak:5d}  ({100.0 * n_leak / n:5.1f} %)")
    print(f"  QUIET (安静):  {n_quiet:5d}  ({100.0 * n_quiet / n:5.1f} %)")
    print(f"  NOISE (干扰):  {n_noise:5d}  ({100.0 * n_noise / n:5.1f} %)")

    # 合并连续 LEAK 窗为事件段
    events = []
    in_evt = False
    for i in range(n):
        if d[i] == LEAK and not in_evt:
            in_evt, t0 = True, t[i]
        elif d[i] != LEAK and in_evt:
            in_evt = False
            events.append((t0, t[i - 1]))
    if in_evt:
        events.append((t0, t[-1]))

    if events:
        print(f"\n  共检测到 {len(events)} 段泄漏事件:")
        for k, (a, b) in enumerate(events, 1):
            print(f"    事件 {k}: {a:8.2f} s ~ {b:8.2f} s  "
                  f"(时长 {b - a + hop_ms * 1e-3:.2f} s)")
        leak_e = energy_db[d == LEAK]
        snr = float(np.mean(leak_e)) - res.mu_b
        print(f"\n  泄漏段平均带内能量高于背景 {snr:.1f} dB (带内 SNR 估计)")
    else:
        print("\n  未检测到泄漏事件。")

    verdict = "检测到泄漏" if n_leak > 0 else "未检测到泄漏"
    print("-" * 62)
    print(f"  最终结论: {verdict}")
    print("=" * 62)
    return events


def plot_results(t, energy_db, res: DetectionResult, x, fs,
                 lowcut, highcut, out_png):
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=False)

    # (1) 全带 / 带内功率谱密度
    ax = axes[0]
    f_w, p_w = welch(x, fs=fs, nperseg=8192)
    ax.semilogx(f_w, 10 * np.log10(p_w + 1e-20), color="gray", lw=0.8,
                label="full-band PSD")
    ax.axvspan(lowcut, highcut, color="tab:orange", alpha=0.15,
               label=f"detection band {lowcut/1000:.0f}-{highcut/1000:.0f} kHz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.set_title("Input signal PSD / 输入信号功率谱")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # (2) 带内能量与阈值
    ax = axes[1]
    ax.plot(t, energy_db, lw=0.8, color="tab:blue", label="in-band energy")
    ax.axhline(res.threshold_db, color="tab:red", ls="--",
               label=f"threshold = mu_B + k*sigma ({res.threshold_db:.1f} dB)")
    ax.axhline(res.mu_b, color="tab:green", ls=":",
               label=f"background mean ({res.mu_b:.1f} dB)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (dB)")
    ax.set_title("Sliding-window in-band energy / 滑动窗带内能量")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # (3) 判决序列
    ax = axes[2]
    colors = {QUIET: "tab:green", LEAK: "tab:red", NOISE: "tab:gray"}
    for state in (QUIET, LEAK, NOISE):
        m = res.decisions == state
        if m.any():
            ax.scatter(t[m], np.full(m.sum(), state), s=6,
                       color=colors[state], label=LABELS[state])
    ax.set_yticks([QUIET, LEAK, NOISE])
    ax.set_yticklabels(["QUIET", "LEAK", "NOISE"])
    ax.set_xlabel("Time (s)")
    ax.set_title("Per-window decision / 逐窗判决")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"\n[图已保存] {out_png}")


# ----------------------------------------------------------------------
# 7. 主流程
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="气体泄漏声学检测基线: 18-50kHz带通 + 滑动窗能量 "
                    "+ 自适应阈值与稳定性校验 (arXiv:2511.00348 判决逻辑)")
    ap.add_argument("--input", required=True, help="待检测 WAV 文件")
    ap.add_argument("--background", default=None,
                    help="纯环境噪声 WAV (用于估计阈值, 推荐)")
    ap.add_argument("--calib-seconds", type=float, default=None,
                    help="若无背景文件, 用待测文件开头 N 秒估计背景")
    ap.add_argument("--lowcut", type=float, default=18000.0)
    ap.add_argument("--highcut", type=float, default=50000.0)
    ap.add_argument("--win-ms", type=float, default=50.0, help="窗长 (ms)")
    ap.add_argument("--hop-ms", type=float, default=25.0, help="步进 (ms)")
    ap.add_argument("--k-sigma", type=float, default=3.0,
                    help="阈值系数 k: thr = mu_B + k*sigma_B (默认3)")
    ap.add_argument("--m-stable", type=int, default=5,
                    help="稳定性校验连续窗口数 (默认5, 同文献)")
    ap.add_argument("--out-png", default="detection_result.png")
    args = ap.parse_args()

    if args.background is None and args.calib_seconds is None:
        ap.error("必须提供 --background 或 --calib-seconds 其中之一")

    # --- 待测信号 ---
    fs, x, clip_lv = load_wav(args.input)
    print(f"[输入] {args.input}: fs = {fs} Hz, 时长 = {len(x)/fs:.2f} s")
    x_f, hi = bandpass(x, fs, args.lowcut, args.highcut)
    t, e_db, clp = frame_energies(x_f, x, fs, args.win_ms, args.hop_ms,
                                  clip_lv)

    # --- 背景统计 ---
    if args.background is not None:
        fs_b, xb, clip_b = load_wav(args.background)
        print(f"[背景] {args.background}: fs = {fs_b} Hz, "
              f"时长 = {len(xb)/fs_b:.2f} s")
        if fs_b != fs:
            print("[警告] 背景与待测采样率不同, 阈值估计可能有偏差。")
        xb_f, _ = bandpass(xb, fs_b, args.lowcut, args.highcut)
        tb, eb_db, clpb = frame_energies(xb_f, xb, fs_b, args.win_ms,
                                         args.hop_ms, clip_b)
        mu_b, sg_b = background_stats(eb_db, clpb)
    else:
        n_cal = int(np.sum(t < args.calib_seconds))
        if n_cal < 10:
            sys.exit("[错误] 校准段窗口太少, 请加长 --calib-seconds。")
        mu_b, sg_b = background_stats(e_db[:n_cal], clp[:n_cal])
        print(f"[校准] 使用开头 {args.calib_seconds:.1f} s "
              f"({n_cal} 个窗口) 估计背景。")

    # --- 判决 ---
    cfg = DetectorConfig(k_sigma=args.k_sigma, m_stable=args.m_stable)
    res = detect(e_db, clp, mu_b, sg_b, cfg)

    # --- 输出 ---
    summarize(t, e_db, res, args.hop_ms)
    plot_results(t, e_db, res, x, fs, args.lowcut, hi, args.out_png)

    # 逐窗结果存 CSV, 便于后续做 ROC / 与机器学习方案对比
    csv_path = args.out_png.rsplit(".", 1)[0] + ".csv"
    np.savetxt(csv_path,
               np.column_stack([t, e_db, res.decisions]),
               delimiter=",", header="time_s,inband_energy_db,decision"
               "(0=QUIET,1=LEAK,2=NOISE)", comments="")
    print(f"[逐窗结果已保存] {csv_path}")


if __name__ == "__main__":
    main()
