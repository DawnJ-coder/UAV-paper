# -*- coding: utf-8 -*-
"""
gas_leak_localization.py
气体泄漏精准定位系统
基于已有的 LeakAnchoredMahalanobis 判别模型，实现空间多点定位

依赖: numpy, scipy, soundfile, matplotlib
      leak_detector.py (你已有的判别模型代码)
"""
import os
import pickle
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 导入你已有的判别模型模块（把上面那份代码存为 leak_detector.py）
from leak_detector import (
    LeakAnchoredMahalanobis, build_frames, file_to_frames,
    add_engineered, file_stats, decide_file, train
)


# ============================================================
# 0. 模型加载 / 保存
# ============================================================
def save_model(det, params, path="leak_model.pkl"):
    """保存训练好的判别模型和阈值参数"""
    with open(path, "wb") as f:
        pickle.dump({"det": det, "params": params}, f)
    print(f"[model] 已保存到 {path}")


def load_model(path="leak_model.pkl"):
    """加载实验室训练好的模型"""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"[model] 已从 {path} 加载模型")
    return obj["det"], obj["params"]


# ============================================================
# 1. 空间点的定义与数据组织
# ============================================================
class SpatialPoint:
    """一个空间采样点：坐标 + 增强后的单通道信号"""
    def __init__(self, name, xy, signal_data, fs):
        self.name = name
        self.xy = np.array(xy, dtype=float)   # (x, y) 像素/物理坐标
        self.signal = signal_data              # 波束成形增强后的单通道信号
        self.fs = fs
        self.frames = None                     # 特征帧，稍后计算
        self.stats = None                      # 文件级统计
        self.score = None                      # 泄漏似然分

    def extract_features(self, win_ms=100, hop_ms=50, band=(18000, 50000)):
        fr = file_to_frames(self.signal, self.fs, win_ms, hop_ms, band)
        self.frames = add_engineered(fr)
        return self.frames


def load_spatial_points(point_configs, fs=None):
    """
    point_configs: list of dict
        [{"name":"center", "xy":(x,y), "wav":"path/to/center.wav"},
         {"name":"up",     "xy":(x,y-10), "wav":"..."},
         ...]
    返回 SpatialPoint 列表
    """
    import soundfile as sf
    points = []
    for cfg in point_configs:
        x, _fs = sf.read(cfg["wav"])
        if x.ndim > 1:
            x = x.mean(1)
        pt = SpatialPoint(cfg["name"], cfg["xy"], x, _fs)
        pt.extract_features()
        points.append(pt)
    return points


# ============================================================
# 2. 泄漏似然打分
# ============================================================
def score_point(det, pt, params):
    """
    给一个空间点打泄漏似然分（越大越像泄漏源）
    综合: 中位马氏距离(主) + p90平稳性 + 占比frac
    """
    st = file_stats(det, pt.frames, params['win_tau'])
    pt.stats = st
    if not np.isfinite(st['median']):
        pt.score = -np.inf
        return pt.score
    # 距离越小越像泄漏 -> 取负；p90越小越平稳；frac越大越好
    score = (-st['median']
             - 0.5 * st['p90']
             + 2.0 * st['frac'])
    pt.score = score
    return score


# ============================================================
# 3. 空间衰减分析（物理约束核心）
# ============================================================
def gaussian_decay(r, A, sigma, C):
    """点声源衰减模型：中心最强，随距离高斯衰减"""
    return A * np.exp(-(r ** 2) / (2 * sigma ** 2)) + C


def analyze_spatial_decay(center_pt, neighbor_pts, verbose=True):
    """
    分析泄漏似然是否随距离衰减
    返回:
      corr        : 距离-分数相关系数 (真泄漏应为强负相关)
      monotonic   : 是否单调衰减
      slope       : 平均衰减斜率
    """
    dists = [0.0]  # 中心点距离为0
    scores = [center_pt.score]
    for pt in neighbor_pts:
        d = np.linalg.norm(pt.xy - center_pt.xy)
        dists.append(d)
        scores.append(pt.score)

    dists = np.array(dists)
    scores = np.array(scores)

    # 相关系数（衰减应为负相关）
    if len(dists) > 2 and np.std(scores) > 1e-9:
        corr = np.corrcoef(dists, scores)[0, 1]
    else:
        corr = 0.0

    # 中心分数是否为最大（泄漏源应该最强）
    center_is_max = center_pt.score >= max(pt.score for pt in neighbor_pts)

    # 平均衰减斜率（用中心和周边均值）
    neigh_mean = np.mean([pt.score for pt in neighbor_pts])
    mean_dist = np.mean([np.linalg.norm(pt.xy - center_pt.xy) for pt in neighbor_pts])
    slope = (neigh_mean - center_pt.score) / (mean_dist + 1e-9)

    if verbose:
        print(f"    [衰减分析] 中心分={center_pt.score:.2f}, "
              f"周边均分={neigh_mean:.2f}, 相关={corr:.2f}, "
              f"中心最强={center_is_max}, 斜率={slope:.3f}")

    return dict(corr=corr, center_is_max=center_is_max,
                slope=slope, dists=dists, scores=scores)


def fit_decay_curve(dists_all, scores_all):
    """
    用多距离数据拟合衰减曲线，返回拟合质量和峰值位置
    需要至少3个不同距离才有意义
    """
    if len(set(dists_all)) < 3:
        return None
    try:
        # 转成"似然"形式（分越高越像泄漏），归一化
        y = np.array(scores_all)
        y = y - y.min()
        A0, sigma0, C0 = y.max(), np.std(dists_all) + 1, y.min()
        popt, _ = curve_fit(gaussian_decay, dists_all, y,
                            p0=[A0, sigma0, C0], maxfev=5000)
        y_fit = gaussian_decay(np.array(dists_all), *popt)
        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
        r2 = 1 - ss_res / ss_tot
        return dict(A=popt[0], sigma=popt[1], C=popt[2], r2=r2)
    except Exception as e:
        print(f"    [拟合失败] {e}")
        return None


# ============================================================
# 4. 综合决策
# ============================================================
def decide_leak(det, center_pt, neighbor_pts, params,
                min_margin=0.5, min_corr=-0.4):
    """
    综合三层判据做最终决策
    返回决策字典
    """
    # 打分
    score_point(det, center_pt, params)
    for pt in neighbor_pts:
        score_point(det, pt, params)

    # --- 判据1: 中心点本身像泄漏吗（复用二分类护栏）---
    st = center_pt.stats
    class_ok = (st['p90'] < params['tau_p90']) and \
               (st['median'] < params['tau_med'])

    # --- 判据2: 空间对比度（中心显著优于周边）---
    neigh_max = max(pt.score for pt in neighbor_pts)
    margin = center_pt.score - neigh_max
    contrast_ok = margin >= min_margin

    # --- 判据3: 空间衰减规律 ---
    decay = analyze_spatial_decay(center_pt, neighbor_pts, verbose=False)
    decay_ok = (decay['corr'] <= min_corr) and decay['center_is_max']

    # 综合置信度
    n_ok = int(class_ok) + int(contrast_ok) + int(decay_ok)
    level = {3: '高', 2: '中', 1: '低', 0: '无泄漏/不可信'}[n_ok]

    result = dict(
        is_leak=(n_ok >= 2),               # 至少满足2个判据才判泄漏
        leak_xy=tuple(center_pt.xy) if n_ok >= 2 else None,
        confidence_level=level,
        n_criteria_passed=n_ok,
        details=dict(
            class_ok=class_ok,
            contrast_ok=contrast_ok, margin=float(margin),
            decay_ok=decay_ok, corr=float(decay['corr']),
            center_is_max=decay['center_is_max'],
            center_score=float(center_pt.score),
            center_median=float(st['median']),
            center_p90=float(st['p90']),
        ),
        decay_data=decay,
    )
    return result


# ============================================================
# 5. 可视化
# ============================================================
def visualize_result(center_pt, neighbor_pts, result,
                     save_path="leak_result.png"):
    """绘制空间分布图 + 衰减曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- 左图：空间分布（散点，颜色=泄漏分）---
    ax = axes[0]
    all_pts = [center_pt] + neighbor_pts
    xs = [p.xy[0] for p in all_pts]
    ys = [p.xy[1] for p in all_pts]
    scores = [p.score for p in all_pts]
    sc = ax.scatter(xs, ys, c=scores, s=400, cmap='hot',
                    edgecolors='k', linewidths=2)
    for p in all_pts:
        ax.annotate(f"{p.name}\n{p.score:.1f}",
                    (p.xy[0], p.xy[1]),
                    ha='center', va='center', fontsize=9)
    if result['is_leak']:
        ax.scatter(center_pt.xy[0], center_pt.xy[1], s=1000,
                   facecolors='none', edgecolors='lime',
                   linewidths=3, label='泄漏点')
        ax.legend(loc='upper right')
    plt.colorbar(sc, ax=ax, label='泄漏似然分')
    ax.set_title(f"空间分布 | 判定: {result['confidence_level']}")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.invert_yaxis()  # 图像坐标y向下
    ax.set_aspect('equal')

    # --- 右图：距离-似然衰减曲线 ---
    ax = axes[1]
    dists = result['decay_data']['dists']
    scores = result['decay_data']['scores']
    order = np.argsort(dists)
    ax.plot(dists[order], scores[order], 'o-', ms=10, lw=2)
    ax.axhline(neighbor_pts and max(p.score for p in neighbor_pts),
               color='gray', ls='--', alpha=0.5, label='周边最大')
    ax.set_xlabel("距中心距离"); ax.set_ylabel("泄漏似然分")
    ax.set_title(f"空间衰减 (相关={result['decay_data']['corr']:.2f})")
    ax.grid(alpha=0.3); ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"[viz] 结果图已保存: {save_path}")
    plt.close()


def print_report(result):
    """打印决策报告"""
    d = result['details']
    print("\n" + "=" * 55)
    print("           气体泄漏定位诊断报告")
    print("=" * 55)
    print(f"  是否泄漏      : {'★ 是 ★' if result['is_leak'] else '否'}")
    print(f"  置信度等级    : {result['confidence_level']} "
          f"({result['n_criteria_passed']}/3 判据通过)")
    if result['leak_xy']:
        print(f"  泄漏点坐标    : ({result['leak_xy'][0]:.1f}, "
              f"{result['leak_xy'][1]:.1f})")
    print("-" * 55)
    print("  判据明细:")
    print(f"    [1] 绝对相似度 : {'✓' if d['class_ok'] else '✗'}  "
          f"(median={d['center_median']:.2f}, p90={d['center_p90']:.2f})")
    print(f"    [2] 空间对比度 : {'✓' if d['contrast_ok'] else '✗'}  "
          f"(margin={d['margin']:.2f})")
    print(f"    [3] 空间衰减   : {'✓' if d['decay_ok'] else '✗'}  "
          f"(corr={d['corr']:.2f}, 中心最强={d['center_is_max']})")
    print("=" * 55 + "\n")


# ============================================================
# 6. 端到端主流程
# ============================================================
def run_localization(model_path, point_configs, multi_dist_configs=None,
                    viz=True):
    """
    完整定位流程
    参数:
      model_path         : 实验室训练好的模型 pkl 路径
      point_configs      : 5点配置 (中心+上下左右)
      multi_dist_configs : 可选，多距离配置用于拟合衰减曲线
    """
    # 1. 加载模型
    det, params = load_model(model_path)

    # 2. 加载空间点
    points = load_spatial_points(point_configs)
    center_pt = points[0]                 # 约定第一个是中心点
    neighbor_pts = points[1:]

    # 3. 综合决策
    result = decide_leak(det, center_pt, neighbor_pts, params)

    # 4. 可选：多距离衰减拟合（精定位）
    if multi_dist_configs:
        md_points = load_spatial_points(multi_dist_configs)
        for pt in md_points:
            score_point(det, pt, params)
        dists = [np.linalg.norm(pt.xy - center_pt.xy) for pt in md_points]
        scores = [pt.score for pt in md_points]
        # 加入中心点
        dists = [0.0] + dists
        scores = [center_pt.score] + scores
        fit = fit_decay_curve(dists, scores)
        if fit:
            print(f"[衰减拟合] sigma={fit['sigma']:.2f}, R²={fit['r2']:.3f}")
            result['decay_fit'] = fit
            # R²高且sigma合理 -> 更强的泄漏证据
            if fit['r2'] > 0.7:
                print("  -> 衰减曲线拟合良好，强烈支持泄漏判定")

    # 5. 输出
    print_report(result)
    if viz:
        visualize_result(center_pt, neighbor_pts, result)

    return result


# ============================================================
# 7. 使用示例
# ============================================================
if __name__ == "__main__":

    # ---------- 步骤A: 首次运行需先训练并保存模型 ----------
    # （若模型已保存为 pkl 则跳过）
    """
    train_leak  = ["lab_leak_1.wav", "lab_leak_2.wav", ...]   # 实验室纯泄漏
    train_noise = ["lab_noise_1.wav", ...]                    # 实验室噪声
    det, params = train(train_leak, train_noise)
    save_model(det, params, "leak_model.pkl")
    """

    # ---------- 步骤B: 现场定位 ----------
    # 假设泄漏疑似点在图片 (x=200, y=150)，周边距离10的4个点
    cx, cy = 200, 150
    dist = 10

    point_configs = [
        {"name": "center", "xy": (cx, cy),         "wav": "field/center.wav"},
        {"name": "up",     "xy": (cx, cy - dist),  "wav": "field/up.wav"},
        {"name": "down",   "xy": (cx, cy + dist),  "wav": "field/down.wav"},
        {"name": "left",   "xy": (cx - dist, cy),  "wav": "field/left.wav"},
        {"name": "right",  "xy": (cx + dist, cy),  "wav": "field/right.wav"},
    ]

    # 可选：多距离数据（推荐！用于拟合衰减曲线，提高可靠性）
    multi_dist_configs = [
        {"name": "up5",   "xy": (cx, cy - 5),   "wav": "field/up5.wav"},
        {"name": "up15",  "xy": (cx, cy - 15),  "wav": "field/up15.wav"},
        {"name": "up20",  "xy": (cx, cy - 20),  "wav": "field/up20.wav"},
        # ... 更多方向和距离
    ]

    result = run_localization(
        model_path="leak_model.pkl",
        point_configs=point_configs,
        multi_dist_configs=multi_dist_configs,   # 没有可传 None
        viz=True,
    )
