# -*- coding: utf-8 -*-
"""
气体泄漏 vs 环境噪声 检测器（跨孔径鲁棒版）- 完整版
包含：5折交叉验证、QDA/RF对比、混淆矩阵、ROC曲线、完整报告

核心思想：
  1) 用尺度不变的谱形/稳定性特征 + 自适应SNR 替代绝对能量阈值，跨孔径可移植；
  2) 以"泄漏类"为锚做多维联合马氏距离判别，利用泄漏簇的紧致协方差；
  3) 文件级决策：p90(离散度)做主判据 + median(中位距离)做护栏；
  4) 5折交叉验证 + QDA/RF监督学习对照，全面评估性能。

依赖: numpy, scipy, soundfile, scikit-learn (自动安装)
"""
import os
import sys
import glob
import pickle
import json
import time
import subprocess
from datetime import datetime
import numpy as np
from scipy import signal
from scipy.stats import chi2

# ============================================================
# 自动安装依赖
# ============================================================
def install_package(package_name):
    """自动安装Python包"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except:
        return False

# 检查并安装依赖
_has_sklearn = False
_has_matplotlib = False
_has_seaborn = False

try:
    import sklearn
    _has_sklearn = True
except ImportError:
    print("Installing scikit-learn...")
    if install_package("scikit-learn"):
        import sklearn
        _has_sklearn = True
        print("  ✓ scikit-learn installed")
    else:
        print("  ✗ Failed to install scikit-learn")
try:
    import matplotlib
    _has_matplotlib = True
except ImportError:
    print("Installing matplotlib...")
    if install_package("matplotlib"):
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        _has_matplotlib = True
        print("  ✓ matplotlib installed")
    else:
        print("  ✗ Failed to install matplotlib")

try:
    import seaborn as sns
    _has_seaborn = True
except ImportError:
    if _has_matplotlib:
        print("Installing seaborn...")
        if install_package("seaborn"):
            import seaborn as sns
            _has_seaborn = True
            print("  ✓ seaborn installed")
        else:
            print("  ✗ Failed to install seaborn")

# 条件导入matplotlib.pyplot
if _has_matplotlib:
    import matplotlib.pyplot as plt
else:
    plt = None
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
    
    def to_dict(self):
        """将模型参数导出为字典，便于保存"""
        return {
            'idx': self.idx,
            'accept': self.accept,
            'mu_': self.mu_.tolist(),
            'sd_': self.sd_.tolist(),
            'inv_': self.inv_.tolist(),
            'center_': self.center_.tolist(),
            'tau_': self.tau_
        }
    
    @classmethod
    def from_dict(cls, d):
        """从字典恢复模型"""
        obj = cls(idx=d['idx'], accept=d['accept'])
        obj.mu_ = np.array(d['mu_'])
        obj.sd_ = np.array(d['sd_'])
        obj.inv_ = np.array(d['inv_'])
        obj.center_ = np.array(d['center_'])
        obj.tau_ = d['tau_']
        return obj
# ============================================================
# 3. 文件级决策：p90做主判据 + median做护栏
# ============================================================
def file_stats(det, frames, win_tau):
    """
    文件级鲁棒统计(不依赖单个最小窗口):
      median : 中位马氏距离  -> 抗离群
      frac   : 像泄漏的窗口占比 (d < win_tau)
      p90    : 距离90分位     -> 反映离散/拖尾
      std    : 距离离散度     -> 真泄漏小, 噪声大
      mean   : 平均距离
    """
    d = det.window_scores(frames)
    if len(d) == 0:
        return dict(median=np.inf, frac=0.0, p90=np.inf, std=np.inf, mean=np.inf)
    return dict(median=float(np.median(d)),
                frac=float(np.mean(d < win_tau)),
                p90=float(np.percentile(d, 90)),
                std=float(np.std(d)),
                mean=float(d.mean()))


def decide_file(st, tau_p90, tau_med):
    """
    主判据: p90(离散度/平稳性) < tau_p90  -- 真泄漏全程平稳, 距离分布紧
    护栏:   median < tau_med               -- 宽松, 仅挡病态'紧但远'的情况
    (frac 仅用于显示, 不进硬规则: 远/弱泄漏 frac 会偏低, 用它做门会误杀)
    """
    return (st['p90'] < tau_p90) and (st['median'] < tau_med)


def _gap_threshold(leak_vals, noise_vals, name):
    """在两类之间取阈值: 有干净间隔则取间隔中点, 否则取分位中点并告警。"""
    lmax, nmin = float(np.max(leak_vals)), float(np.min(noise_vals))
    if lmax < nmin:                                   # 干净间隔
        tau = (lmax + nmin) / 2
        print(f"[train] {name}: 泄漏≤{lmax:.2f} | 间隔 | 噪声≥{nmin:.2f} -> 阈值={tau:.2f} (干净分离)")
    else:                                             # 有重叠, 退化到分位
        tau = float((np.percentile(leak_vals, 95) + np.percentile(noise_vals, 5)) / 2)
        print(f"[warn]  {name}: 泄漏max={lmax:.2f} > 噪声min={nmin:.2f} 存在重叠 -> 阈值={tau:.2f}")
    return tau

# ============================================================
# 4. 端到端: 训练 + 评估
# ============================================================
def build_frames(file_list, win_ms=100, hop_ms=50, band=(18000, 50000)):
    import soundfile as sf
    out = []
    for p in file_list:
        try:
            x, fs = sf.read(p)
            if x.ndim > 1:
                x = x.mean(1)
            fr = file_to_frames(x, fs, win_ms, hop_ms, band)
            out.append(add_engineered(fr))
        except Exception as e:
            print(f"  Warning: Failed to process {p}: {e}")
    return out


def train(leak_files, noise_files, **kw):
    print(f"\n[Training] Loading {len(leak_files)} leak files and {len(noise_files)} noise files...")
    leak = build_frames(leak_files, **kw)
    noise = build_frames(noise_files, **kw)
    
    leak = [f for f in leak if len(f) > 0]
    noise = [f for f in noise if len(f) > 0]
    
    if len(leak) == 0:
        raise ValueError("No valid leak files found!")
    if len(noise) == 0:
        raise ValueError("No valid noise files found!")
    
    print(f"  Valid leak files: {len(leak)}, Valid noise files: {len(noise)}")
    
def evaluate(det, files, label, train_stats, verbose=True, **kw):
    print(f"\n[Evaluating] {len(files)} files, label={label}...")
    frames = build_frames(files, **kw)
    
    win_tau = train_stats['win_tau']
    tau_p90 = train_stats['tau_p90']
    tau_med = train_stats['tau_med']
    
    preds = []
    file_results = []
    for i, fr in enumerate(frames):
        if len(fr) == 0:
            preds.append(False)
            file_results.append({
                'file': files[i],
                'prediction': False,
                'label': bool(label),
                'correct': (False == bool(label)),
                'n_windows': 0,
                'median_distance': None,
                'mean_distance': None,
                'p90_distance': None,
                'frac': None,
                'std_distance': None,
            })
            continue
        
        st = file_stats(det, fr, win_tau)
        is_leak = decide_file(st, tau_p90, tau_med)
        preds.append(is_leak)
        file_results.append({
            'file': files[i],
            'prediction': bool(is_leak),
            'label': bool(label),
            'correct': (is_leak == bool(label)),
            'n_windows': len(fr),
            'median_distance': st['median'],
            'mean_distance': st['mean'],
            'p90_distance': st['p90'],
            'frac': st['frac'],
            'std_distance': st['std'],
        })
    
    preds = np.array(preds)
    true_labels = np.array([bool(label)] * len(preds))
    acc = np.mean(preds == true_labels)
    label_name = "LEAK" if label else "NOISE"
    
    tp = int(np.sum((preds == True) & (true_labels == True)))
    tn = int(np.sum((preds == False) & (true_labels == False)))
    fp = int(np.sum((preds == True) & (true_labels == False)))
    fn = int(np.sum((preds == False) & (true_labels == True)))
    
    eval_stats = {
        'label': label_name,
        'n_files': len(files),
        'n_predicted_leak': int(preds.sum()),
        'accuracy': float(acc),
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        'f1': float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0,
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'file_results': file_results
    }
    
    print(f"[eval] {label_name}: {preds.sum()}/{len(preds)} judged leak | acc={acc:.3f}")
    if (tp + fp) > 0:
        print(f"       Precision={eval_stats['precision']:.3f}, Recall={eval_stats['recall']:.3f}, F1={eval_stats['f1']:.3f}")
    
    if verbose:
        for r in file_results:
            name = os.path.basename(r['file'])
            flag = '泄漏' if r['prediction'] else '噪声'
            wrong = '  <== 误判' if not r['correct'] else ''
            print(f"    {name[:45]:45s} med={r['median_distance']:6.2f} "
                  f"p90={r['p90_distance']:6.2f} frac={r['frac']:.2f} -> {flag}{wrong}")
    
    return preds, eval_stats

def confusion_report(leak_preds, noise_preds):
    """打印清晰混淆矩阵。"""
    TP = int(leak_preds.sum())          # 泄漏判泄漏
    FN = int((~leak_preds).sum())       # 泄漏漏判成噪声
    FP = int(noise_preds.sum())         # 噪声误判成泄漏
    TN = int((~noise_preds).sum())      # 噪声判噪声
    P = TP / (TP + FP) if TP + FP else 0
    R = TP / (TP + FN) if TP + FN else 0
    F1 = 2 * P * R / (P + R) if P + R else 0
    acc = (TP + TN) / (TP + TN + FP + FN)
    print("\n================ 混淆矩阵 (以泄漏为正类) ================")
    print(f"                   判为泄漏   判为噪声")
    print(f"  真实泄漏 (LEAK)    TP={TP:<4}   FN={FN}")
    print(f"  真实噪声 (NOISE)   FP={FP:<4}   TN={TN}")
    print(f"  Accuracy={acc:.3f}  Precision={P:.3f}  Recall={R:.3f}  F1={F1:.3f}")
    print("  (FP=噪声误报数, 越小越好; FN=泄漏漏报数, 越小越好)")


# ============================================================
# 5. 5折交叉验证（主检测器）
# ============================================================
def cross_validate_mahalanobis(leak_files, noise_files, n_folds=5, **kw):
    """
    对马氏距离检测器做5折交叉验证
    """
    print(f"\n{'='*60}")
    print(f"5-FOLD CROSS-VALIDATION (Mahalanobis Detector - File-Level: p90 + median)")
    print(f"{'='*60}")
      np.random.seed(42)
    leak_idx = np.random.permutation(len(leak_files))
    noise_idx = np.random.permutation(len(noise_files))
    
    leak_folds = np.array_split(leak_idx, n_folds)
    noise_folds = np.array_split(noise_idx, n_folds)
    
    cv_results = []
    
    for fold in range(n_folds):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        
        # 划分训练集和验证集
        train_leak_idx = np.concatenate([leak_folds[i] for i in range(n_folds) if i != fold])
        val_leak_idx = leak_folds[fold]
        train_noise_idx = np.concatenate([noise_folds[i] for i in range(n_folds) if i != fold])
        val_noise_idx = noise_folds[fold]
        
        train_leak = [leak_files[i] for i in train_leak_idx]
        val_leak = [leak_files[i] for i in val_leak_idx]
        train_noise = [noise_files[i] for i in train_noise_idx]
        val_noise = [noise_files[i] for i in val_noise_idx]
        
        print(f"  Train: {len(train_leak)} leak + {len(train_noise)} noise")
        print(f"  Val:   {len(val_leak)} leak + {len(val_noise)} noise")
        
        # 训练
        det, train_stats = train(train_leak, train_noise, **kw)
        
        # 评估
        leak_preds, leak_eval = evaluate(det, val_leak, 1, train_stats, verbose=False, **kw)
        noise_preds, noise_eval = evaluate(det, val_noise, 0, train_stats, verbose=False, **kw)
        
        all_preds = np.concatenate([leak_preds, noise_preds])
        all_labels = np.concatenate([np.ones(len(leak_preds)), np.zeros(len(noise_preds))])
        fold_acc = float(np.mean(all_preds == all_labels))
        
        fold_result = {
            'fold': fold + 1,
            'train_leak': len(train_leak),
            'train_noise': len(train_noise),
            'val_leak': len(val_leak),
            'val_noise': len(val_noise),
            'win_tau': train_stats['win_tau'],
            'tau_p90': train_stats['tau_p90'],
            'tau_med': train_stats['tau_med'],
            'leak_accuracy': leak_eval['accuracy'],
            'leak_recall': leak_eval['recall'],
            'noise_accuracy': noise_eval['accuracy'],
            'noise_specificity': noise_eval['specificity'],
            'fold_accuracy': fold_acc,
        }
        cv_results.append(fold_result)
        print(f"  Fold {fold+1} accuracy: {fold_acc:.4f} (leak_recall={leak_eval['recall']:.3f}, noise_spec={noise_eval['specificity']:.3f})")
    
    # 汇总
    accuracies = [r['fold_accuracy'] for r in cv_results]
    leak_recalls = [r['leak_recall'] for r in cv_results]
    noise_specificities = [r['noise_specificity'] for r in cv_results]
    
    cv_summary = {
        'n_folds': n_folds,
        'fold_results': cv_results,
        'mean_accuracy': float(np.mean(accuracies)),
        'std_accuracy': float(np.std(accuracies)),
        'mean_leak_recall': float(np.mean(leak_recalls)),
        'std_leak_recall': float(np.std(leak_recalls)),
        'mean_noise_specificity': float(np.mean(noise_specificities)),
        'std_noise_specificity': float(np.std(noise_specificities)),
        'all_accuracies': [float(a) for a in accuracies],
    }
    
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Mean Accuracy: {cv_summary['mean_accuracy']:.4f} ± {cv_summary['std_accuracy']:.4f}")
    print(f"Mean Leak Recall: {cv_summary['mean_leak_recall']:.4f} ± {cv_summary['std_leak_recall']:.4f}")
    print(f"Mean Noise Specificity: {cv_summary['mean_noise_specificity']:.4f} ± {cv_summary['std_noise_specificity']:.4f}")
    
    return cv_summary
  









