# -*- coding: utf-8 -*-
"""
气体泄漏 vs 环境噪声 检测器（跨孔径鲁棒版）
核心思想：
  1) 用尺度不变的谱形/稳定性特征 + 自适应SNR 替代绝对能量阈值，跨孔径可移植；
  2) 以"泄漏类"为锚做多维联合马氏距离判别，利用泄漏簇的紧致协方差；
  3) 加滑窗一致性门控 + 持续性 + 双阈值迟滞，区分"瞬时像"和"持续像"。

依赖: numpy, scipy, soundfile  (可选: scikit-learn 用于QDA/RF对照)
"""
import numpy as np
from scipy import signal
from scipy.stats import chi2

# ============================================================
# 1. 特征提取
# ============================================================
def stft_band(x, fs, band=(18000, 50000), nfft=2048):
    """单窗的 18-50kHz 频带功率谱。"""
    n = min(len(x), nfft)
    f, Pxx = signal.welch(x, fs, nperseg=n, noverlap=n // 2, nfft=nfft, detrend=False)
    m = (f >= band[0]) & (f <= band[1])
    return f[m], np.maximum(Pxx[m], 1e-20)


def window_features(x, fs, band=(18000, 50000)):
    """逐窗特征: [energy_dB, flatness, spread_Hz]"""
    f, P = stft_band(x, fs, band)
    energy = 10.0 * np.log10(np.sum(P) + 1e-20)
    flatness = np.exp(np.mean(np.log(P))) / np.mean(P)          # 几何均值/算术均值
    c = np.sum(f * P) / np.sum(P)                                # 谱质心
    spread = np.sqrt(np.sum(((f - c) ** 2) * P) / np.sum(P))     # 谱展宽
    return np.array([energy, flatness, spread], dtype=np.float64)


def file_to_frames(x, fs, win_ms=100, hop_ms=50, band=(18000, 50000)):
    """整条录音 -> 逐窗特征序列 (T x 3)。"""
    win = int(fs * win_ms / 1000)
    hop = int(fs * hop_ms / 1000)
    feats = []
    for s in range(0, len(x) - win + 1, hop):
        feats.append(window_features(x[s:s + win], fs, band))
    return np.asarray(feats) if feats else np.zeros((0, 3))


def add_engineered(frames, snr_win=20, consist_win=10):
    """
    在 [energy, flatness, spread] 基础上追加工程特征:
      - snr      : 能量相对本底(低分位)的自适应SNR  -> 跨孔径不变
      - cons_flat: 滑窗内 flatness 局部std (泄漏小, 噪声大)
      - cons_spr : 滑窗内 spread   局部std
    返回 (T x 6): [energy, flatness, spread, snr, cons_flat, cons_spr]
    """
    if len(frames) == 0:
        return frames
    e, fl, sp = frames[:, 0], frames[:, 1], frames[:, 2]
    floor = np.percentile(e, 10)                 # 本底估计
    snr = e - floor

    def rolling_std(v, w):
        out = np.zeros_like(v)
        for i in range(len(v)):
            a, b = max(0, i - w // 2), min(len(v), i + w // 2 + 1)
            out[i] = np.std(v[a:b])
        return out

    cons_flat = rolling_std(fl, consist_win)
    cons_spr = rolling_std(sp, consist_win)
    return np.column_stack([e, fl, sp, snr, cons_flat, cons_spr])


# 用于判别的维度（刻意排除绝对 energy，保留尺度不变量）
DISCRIM_IDX = [1, 2, 3, 4, 5]   # flatness, spread, snr, cons_flat, cons_spr


# ============================================================
# 2. 以泄漏为锚的马氏距离判别器（主方法）
# ============================================================
class LeakAnchoredMahalanobis:
    """
    只用泄漏窗拟合一个紧致高斯; 噪声因落在簇外被自然排除。
    对噪声环境漂移鲁棒（不依赖噪声分布）。
    """
    def __init__(self, idx=DISCRIM_IDX, accept=0.99):
        self.idx = idx
        self.accept = accept

    def fit(self, leak_frames_list):
        X = np.vstack([f[:, self.idx] for f in leak_frames_list if len(f)])
        self.mu_ = X.mean(0)
        self.sd_ = X.std(0) + 1e-9
        Xz = (X - self.mu_) / self.sd_              # 标准化，避免量纲主导
        cov = np.cov(Xz.T) + 1e-6 * np.eye(Xz.shape[1])
        self.inv_ = np.linalg.pinv(cov)
        self.center_ = Xz.mean(0)
        self.tau_ = np.sqrt(chi2.ppf(self.accept, df=len(self.idx)))
        return self

    def distance(self, frames):
        if len(frames) == 0:
            return np.zeros(0)
        Xz = (frames[:, self.idx] - self.mu_) / self.sd_
        d = Xz - self.center_
        return np.sqrt(np.einsum('ij,jk,ik->i', d, self.inv_, d))

    def window_scores(self, frames):
        """返回每窗的马氏距离; 距离小 => 像泄漏。"""
        return self.distance(frames)


# ============================================================
# 3. 时间后处理: 一致性门控 + 持续性 + 双阈值迟滞
# ============================================================
def temporal_decision(dist, frames, tau_in, tau_out,
                      m=10, cons_flat_max=0.05, cons_spr_max=800,
                      cons_flat_idx=4, cons_spr_idx=5):
    """
    dist     : 逐窗马氏距离 (越小越像泄漏)
    tau_in   : 进入阈 (严, < tau_out)
    tau_out  : 维持阈 (松)
    m        : 触发需连续满足的最小窗数
    一致性门控: 仅当滑窗一致性足够低(像平稳泄漏)才允许进入。
    返回: (file_is_leak: bool, frame_mask: bool[T])
    """
    T = len(dist)
    if T == 0:
        return False, np.zeros(0, bool)
    cf = frames[:, cons_flat_idx]
    cs = frames[:, cons_spr_idx]
    gate = (cf <= cons_flat_max) & (cs <= cons_spr_max)

    state = np.zeros(T, bool)
    on = False
    for i in range(T):
        if not on:
            if dist[i] < tau_in and gate[i]:
                on = True
        else:
            if dist[i] > tau_out:
                on = False
        state[i] = on

    # 持续性: 找最长连续 on 段, >= m 才算检出
    run = best = 0
    for s in state:
        run = run + 1 if s else 0
        best = max(best, run)
    return best >= m, state


# ============================================================
# 4. 端到端: 训练 + 评估
# ============================================================
def build_frames(file_list, win_ms=100, hop_ms=50, band=(18000, 50000)):
    import soundfile as sf
    out = []
    for p in file_list:
        x, fs = sf.read(p)
        if x.ndim > 1:
            x = x.mean(1)
        fr = file_to_frames(x, fs, win_ms, hop_ms, band)
        out.append(add_engineered(fr))
    return out


def train(leak_files, noise_files, **kw):
    leak = build_frames(leak_files, **kw)
    noise = build_frames(noise_files, **kw)          # 仅用于标定阈值
    det = LeakAnchoredMahalanobis().fit(leak)

    # 在训练集上用泄漏/噪声分布标定 tau_in/tau_out（取分离点）
    dl = np.concatenate([det.window_scores(f) for f in leak if len(f)])
    dn = np.concatenate([det.window_scores(f) for f in noise if len(f)])
    tau_in = np.percentile(dl, 90)                   # 接受 90% 泄漏窗
    tau_out = np.percentile(dl, 99)                  # 维持更松
    print(f"[train] leak dist  p50/p90={np.median(dl):.2f}/{tau_in:.2f}")
    print(f"[train] noise dist p10/p50={np.percentile(dn,10):.2f}/{np.median(dn):.2f}")
    return det, tau_in, tau_out


def evaluate(det, files, label, tau_in, tau_out, **kw):
    frames = build_frames(files, **kw)
    preds = []
    for fr in frames:
        d = det.window_scores(fr)
        is_leak, _ = temporal_decision(d, fr, tau_in, tau_out)
        preds.append(is_leak)
    preds = np.array(preds)
    acc = np.mean(preds == bool(label))
    print(f"[eval] {('LEAK' if label else 'NOISE')}: "
          f"{preds.sum()}/{len(preds)} judged leak | acc={acc:.3f}")
    return preds


# ============================================================
# 5. 可选: 监督对照 (QDA / RandomForest)
# ============================================================
def supervised_baselines(leak_frames, noise_frames, idx=DISCRIM_IDX):
    """对照：有标签时用 QDA / RF 验证可分性（窗级）。"""
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
    from sklearn.ensemble import RandomForestClassifier as RF
    from sklearn.model_selection import cross_val_score
    Xl = np.vstack([f[:, idx] for f in leak_frames if len(f)])
    Xn = np.vstack([f[:, idx] for f in noise_frames if len(f)])
    X = np.vstack([Xl, Xn])
    y = np.r_[np.ones(len(Xl)), np.zeros(len(Xn))]
    for name, clf in [("QDA", QDA()), ("RF", RF(n_estimators=300))]:
        s = cross_val_score(clf, X, y, cv=5)
        print(f"[{name}] 窗级 5折准确率 = {s.mean():.3f} ± {s.std():.3f}")


if __name__ == "__main__":
    # ---- 用法示例（把下面替换成你的文件清单）----
    # train_leak  = [...]   # 0.1mm 训练泄漏 wav 路径
    # train_noise = [...]
    # test_leak   = [...]
    # test_noise  = [...]
    # det, ti, to = train(train_leak, train_noise)
    # evaluate(det, test_leak,  1, ti, to)
    # evaluate(det, test_noise, 0, ti, to)
    print("import this module and call train()/evaluate().")
