import os
import glob
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import re

# ==========================================
# 1. 配置参数
# ==========================================
# 定义所有要处理的时间点文件夹
time_folders = [
    # "HM20260626_142938.ld",
    # "HM20260626_143034.ld", 
    # "HM20260626_144226.ld",
    # "HM20260626_144325.ld"

    "HM20260702_111044.ld"

    # "HM20260624_100936.ld",
    # "HM20260624_101130.ld",
    # "HM20260624_101209.ld",
    # "HM20260624_101256.ld",
    # "HM20260624_101448.ld",
]

# 根目录路径
center_root_dir = r"D:\gas\beamform_results_sh"
offset_root_dir = r"D:\gas\beamform_results_offset_multiple_sh"

# 8个方向对应的极坐标角度（弧度）
direction_angles = {
    'up': np.pi / 2,         # 90度
    'down': -np.pi / 2,      # 270度
    'left': np.pi,           # 180度
    'right': 0,              # 0度
    'up_left': 3 * np.pi / 4, # 135度
    'down_left': -3 * np.pi / 4, # 225度
    'up_right': np.pi / 4,    # 45度
    'down_right': -np.pi / 4  # 315度
}

# 8个距离 (cm)
distances = [5, 10, 15, 20, 25, 30, 35, 40]

# 频谱分析参数
FREQ_LOW = 50000   # 50kHz
FREQ_HIGH = 70000  # 70kHz
NFFT = 4096        # FFT点数

# ==========================================
# 2. 核心算法：计算频谱曲线（替代原来的能量值）
# ==========================================
def get_spectrum(file_path, sr_target=192000):
    """
    读取WAV文件，计算其频谱曲线（50-70kHz频段）
    返回：freqs（频率数组）, psd（功率谱密度）
    """
    if not os.path.exists(file_path):
        return None, None
    
    # 读取音频
    sr, y = wav.read(file_path)
    
    # 如果是双声道，取单通道
    if len(y.shape) > 1:
        y = y[:, 0]
    
    # 归一化
    y = y / (np.max(np.abs(y)) + 1e-6)
    
    # 计算功率谱密度 (PSD)
    freqs, psd = signal.welch(y, fs=sr, nperseg=NFFT, scaling='density')

    
    # 只保留50-70kHz频段
    mask = (freqs >= FREQ_LOW) & (freqs <= FREQ_HIGH)
    freqs = freqs[mask]
    psd = psd[mask]
    
    return freqs, psd


def get_band_energy_from_spectrum(freqs, psd):
    """
    从频谱曲线中计算50-70kHz频段的总能量（对PSD积分）
    """
    if freqs is None or psd is None:
        return 0.0
    
    # 梯形积分计算频段总能量
    energy = np.trapz(psd, freqs)
    return energy


def subtract_spectrum(freqs, psd_signal, psd_background):
    """
    频谱减法：信号频谱 - 背景频谱
    返回净频谱（负值置零）
    """
    if psd_signal is None or psd_background is None:
        return None
    
    # 确保长度一致
    min_len = min(len(psd_signal), len(psd_background))
    net_psd = psd_signal[:min_len] - psd_background[:min_len]
    
    # 负值置零（物理上能量不能为负）
    net_psd = np.maximum(net_psd, 0)
    
    return freqs[:min_len], net_psd


# ==========================================
# 3. 自动检测中心点编号的函数
# ==========================================
def detect_center_ids(center_data_dir):
    """自动检测文件夹下所有的中心点编号"""
    pattern = os.path.join(center_data_dir, "*_*_beamform_result.wav")
    files = glob.glob(pattern)
    
    center_ids = set()
    for file_path in files:
        filename = os.path.basename(file_path)
        match = re.search(r'_(\d{2})_beamform_result\.wav', filename)
        if match:
            center_id = match.group(1)
            center_ids.add(center_id)
    
    center_ids = sorted(list(center_ids))
    print(f"检测到中心点编号: {center_ids}")
    return center_ids


# ==========================================
# 4. 绘制频谱对比图
# ==========================================
def plot_spectrum_comparison(result_dir, time_folder, center_id, 
                              center_freqs, center_net_psd,
                              directional_spectra, direction_bg_spectra):
    """
    绘制频谱对比图：
    - 左上：中心点净频谱
    - 右上：各方向5cm处净频谱对比
    - 下方：某个方向的原始/背景/净频谱对比
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    

    # ===== 左上：中心点净频谱 =====
    ax1 = axes[0, 0]
    if center_freqs is not None and center_net_psd is not None:
        ax1.plot(center_freqs/1000, center_net_psd, 'b-', linewidth=1.5)
        ax1.fill_between(center_freqs/1000, center_net_psd, alpha=0.3)
        ax1.set_title(f'Center Point Net Spectrum\n(After Background Subtraction)')
        ax1.set_xlabel('Frequency (kHz)')
        ax1.set_ylabel('Power Spectral Density')
        ax1.grid(True, alpha=0.3)
    
    # ===== 右上：各方向5cm处净频谱对比 =====
    ax2 = axes[0, 1]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    for idx, (direction, (freqs, net_psd)) in enumerate(directional_spectra.items()):
        if freqs is not None and net_psd is not None:
            color = colors[idx % len(colors)]
            ax2.plot(freqs/1000, net_psd, color=color, linewidth=1, alpha=0.7, label=direction)
    ax2.set_title(f'Net Spectrum at 5cm (All Directions)')
    ax2.set_xlabel('Frequency (kHz)')
    ax2.set_ylabel('Power Spectral Density')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # ===== 左下：某个方向的原始/背景/净频谱对比（选能量最大的方向）=====
    ax3 = axes[1, 0]
    if directional_spectra:
        # 找净能量最大的方向
        best_dir = max(directional_spectra.keys(), 
                      key=lambda d: get_band_energy_from_spectrum(*directional_spectra[d]))
        
        # 该方向的净频谱
        net_freqs, net_psd = directional_spectra[best_dir]
        
        # 该方向的背景频谱
        if best_dir in direction_bg_spectra:
            bg_freqs, bg_psd = direction_bg_spectra[best_dir]
            ax3.plot(bg_freqs/1000, bg_psd, 'gray', linewidth=1, alpha=0.5, label='Background (40cm)')
        
        ax3.plot(net_freqs/1000, net_psd, 'r-', linewidth=2, label=f'Net Spectrum ({best_dir})')
        ax3.fill_between(net_freqs/1000, net_psd, alpha=0.3, color='red')
        ax3.set_title(f'Spectrum Comparison - Direction: {best_dir}')
        ax3.set_xlabel('Frequency (kHz)')
        ax3.set_ylabel('Power Spectral Density')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
    
    # ===== 右下：所有方向的净能量对比柱状图 =====
    ax4 = axes[1, 1]
    dir_energies = {}
    for direction, (freqs, net_psd) in directional_spectra.items():
        energy = get_band_energy_from_spectrum(freqs, net_psd)
        dir_energies[direction] = energy
    
    if dir_energies:
        directions = list(dir_energies.keys())
        energies = list(dir_energies.values())
        bars = ax4.bar(directions, energies, color=colors[:len(directions)], alpha=0.7)
        ax4.set_title('Net Energy by Direction (50-70kHz)')
        ax4.set_xlabel('Direction')
        ax4.set_ylabel('Integrated Energy')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 标注数值
        for bar, energy in zip(bars, energies):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(), 
                    f'{energy:.2e}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle(f'Spectrum Analysis (50-70kHz)\n{time_folder} - Center {center_id}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存
    save_path = f'{result_dir}/spectrum_{time_folder}_center_{center_id}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"频谱对比图已保存: {save_path}")


# ==========================================
# 5. 处理单个时间点的函数（频谱版本）
# ==========================================
def process_single_timepoint_spectrum(time_folder, center_root, offset_root):
    """处理单个时间点的所有中心点数据（频谱角度）"""
    print(f"\n{'='*80}")
    print(f"开始处理时间点: {time_folder} (频谱分析模式)")
    print(f"{'='*80}")
    
    # 构建完整的文件夹路径
    center_data_dir = os.path.join(center_root, time_folder)
    offset_data_dir = os.path.join(offset_root, time_folder)
    
    # 检查文件夹是否存在
    if not os.path.exists(center_data_dir):
        print(f"警告：中心点文件夹不存在: {center_data_dir}")
        return
    if not os.path.exists(offset_data_dir):
        print(f"警告：偏移点文件夹不存在: {offset_data_dir}")
        return
    
    # 为这个时间点创建结果文件夹
    result_dir = f"results_sh_spectrum_{time_folder}"
    os.makedirs(result_dir, exist_ok=True)
    
    # 自动检测该时间点下的所有中心点编号
    center_ids = detect_center_ids(center_data_dir)
    if not center_ids:
        print(f"警告：在 {time_folder} 中没有检测到任何中心点文件")
        return
    
    print(f"共检测到 {len(center_ids)} 个中心点需要处理")
    
    # 处理该时间点的所有中心点
    for center_id in center_ids:
        print(f"\n{'='*60}")
        print(f"正在处理 {time_folder} - 中心点 {center_id}...")
        print(f"{'='*60}")
        
        # ===================================================
        # 第一步：读取中心点频谱
        # ===================================================
        center_pattern = os.path.join(center_data_dir, f"*_{center_id}_beamform_result.wav")
        center_files = glob.glob(center_pattern)
        
        if not center_files:
            print(f"警告：中心点 {center_id} 文件不存在")
            continue
        
        center_file = center_files[0]
        center_freqs, center_psd = get_spectrum(center_file)
        print(f"中心点 {center_id} 文件: {os.path.basename(center_file)}")
        
        if center_freqs is None:
            print(f"错误：无法读取中心点频谱")
            continue
        
        # ===================================================
        # 第二步：读取各方向40cm处的背景频谱
        # ===================================================
        direction_bg_spectra = {}  # {direction: (freqs, psd)}
        
        for direction in direction_angles.keys():
            pattern = os.path.join(offset_data_dir, f"*_{center_id}d40_{direction}*.wav")
            files = glob.glob(pattern)
            if files:
                freqs, psd = get_spectrum(files[0])
                if freqs is not None:
                    direction_bg_spectra[direction] = (freqs, psd)
                    bg_energy = get_band_energy_from_spectrum(freqs, psd)
                    print(f"  {direction}方向40cm背景能量: {bg_energy:.6e}")
            else:
                print(f"  警告：{direction}方向40cm文件不存在")

        
        # ===================================================
        # 第三步：频谱减法 - 计算各方向各距离的净频谱
        # ===================================================
        # 存储所有点的净能量（用于热力图）
        plot_x = []
        plot_y = []
        plot_energy = []
        
        # 存储各方向5cm处的净频谱（用于频谱对比图）
        directional_spectra_5cm = {}
        
        # 中心点频谱减法（使用所有方向中最干净的背景）
        if direction_bg_spectra:
            # 找能量最小的背景频谱作为中心点的背景
            best_bg_dir = min(direction_bg_spectra.keys(),
                            key=lambda d: get_band_energy_from_spectrum(*direction_bg_spectra[d]))
            bg_freqs, bg_psd = direction_bg_spectra[best_bg_dir]
            center_net_freqs, center_net_psd = subtract_spectrum(center_freqs, center_psd, bg_psd)
            center_net_energy = get_band_energy_from_spectrum(center_net_freqs, center_net_psd)
            
            plot_x.append(0)
            plot_y.append(0)
            plot_energy.append(center_net_energy)
            print(f"中心点净能量: {center_net_energy:.6e} (使用{best_bg_dir}方向背景)")
        else:
            center_net_freqs, center_net_psd = center_freqs, center_psd
            center_net_energy = get_band_energy_from_spectrum(center_freqs, center_psd)
            plot_x.append(0)
            plot_y.append(0)
            plot_energy.append(center_net_energy)
            print(f"中心点能量(无背景减法): {center_net_energy:.6e}")
        
        # 处理各方向各距离
        for direction, angle in direction_angles.items():
            # 获取该方向的背景频谱
            bg_freqs, bg_psd = direction_bg_spectra.get(direction, (None, None))
            
            for d in distances:
                pattern = os.path.join(offset_data_dir, f"*_{center_id}d{d}_{direction}*.wav")
                files = glob.glob(pattern)
                
                if files:
                    sig_freqs, sig_psd = get_spectrum(files[0])
                    
                    if sig_freqs is not None:
                        # 频谱减法
                        net_freqs, net_psd = subtract_spectrum(sig_freqs, sig_psd, bg_psd)
                        net_energy = get_band_energy_from_spectrum(net_freqs, net_psd)
                        
                        # 计算坐标
                        x = d * np.cos(angle)
                        y = d * np.sin(angle)
                        plot_x.append(x)
                        plot_y.append(y)
                        plot_energy.append(net_energy)
                        
                        # 保存5cm处的净频谱
                        if d == 5:
                            directional_spectra_5cm[direction] = (net_freqs, net_psd)
                        
                        print(f"  {direction}方向{d}cm: 净能量{net_energy:.6e}")
        
        # ===================================================
        # 第四步：空间分析与泄漏判断
        # ===================================================
        plot_x = np.array(plot_x)
        plot_y = np.array(plot_y)
        plot_energy = np.array(plot_energy)
        
        if len(plot_energy) > 0:
            max_idx = np.argmax(plot_energy)
            max_x = plot_x[max_idx]
            max_y = plot_y[max_idx]
            max_val = plot_energy[max_idx]
            
            print(f"\n【频谱分析结果】")
            print(f"能量最高点坐标: ({max_x:.1f}, {max_y:.1f}) cm")
            print(f"净能量: {max_val:.6e}")
            
            # 空间梯度验证
            near_mask = (np.sqrt((plot_x - max_x)**2 + (plot_y - max_y)**2) <= 6) & (plot_x != max_x)
            if np.any(near_mask):
                near_avg_energy = np.mean(plot_energy[near_mask])
                drop_ratio = max_val / (near_avg_energy + 1e-12)
                print(f"【空间梯度验证】Drop Ratio: {drop_ratio:.2f}")
                
                if drop_ratio > 1.2:
                    print(">>> 判定：真实气流泄漏源！")
                else:
                    print(">>> 判定：环境背景噪声（虚警）")
            

            # 方向性分析
            print("\n【喷射方向分析】:")
            directional_energies = {}
            for direction, angle in direction_angles.items():
                dir_mask = (np.abs(np.arctan2(plot_y, plot_x) - angle) < 0.3) & (np.sqrt(plot_x**2 + plot_y**2) <= 20)
                if np.any(dir_mask):
                    dir_energy = np.mean(plot_energy[dir_mask])
                    directional_energies[direction] = dir_energy
            
            if directional_energies:
                mean_energy = np.mean(list(directional_energies.values()))
                for direction, energy in sorted(directional_energies.items(), key=lambda x: x[1], reverse=True):
                    if energy > mean_energy * 1.5:
                        print(f" -> 检测到喷射方向: 【{direction}】💨 (能量: {energy:.6e})")
        
        # ===================================================
        # 第五步：绘制频谱对比图
        # ===================================================
        plot_spectrum_comparison(
            result_dir, time_folder, center_id,
            center_net_freqs, center_net_psd,
            directional_spectra_5cm, direction_bg_spectra
        )
        
        # ===================================================
        # 第六步：绘制空间热力图
        # ===================================================
        try:
            grid_x, grid_y = np.mgrid[-45:45:200j, -45:45:200j]
            grid_z = griddata((plot_x, plot_y), plot_energy, (grid_x, grid_y), 
                            method='cubic', fill_value=0)
            
            plt.figure(figsize=(8, 7), dpi=100)
            plt.imshow(grid_z.T, extent=(-45, 45, -45, 45), origin='lower', cmap='jet')
            plt.colorbar(label='50-70kHz Net Energy (Spectrum Subtraction)')
            plt.scatter(plot_x, plot_y, c='white', s=15, edgecolors='black', alpha=0.6)
            plt.scatter(0, 0, c='red', marker='*', s=200, edgecolors='yellow')
            plt.title(f"2D Spatial Heatmap (Spectrum Subtraction)\n{time_folder} - Center {center_id}")
            plt.xlabel("X Distance (cm)")
            plt.ylabel("Y Distance (cm)")
            plt.grid(True, linestyle='--', alpha=0.5)
            
            plt.savefig(f'{result_dir}/heatmap_{time_folder}_center_{center_id}.png', 
                       dpi=100, bbox_inches='tight')
            plt.close()
            print(f"热力图已保存: {result_dir}/heatmap_{time_folder}_center_{center_id}.png")
            
        except Exception as e:
            print(f"绘图出错: {e}")


# 6. 主程序
def main():
    print("="*80)
    print("频谱减法模式 - 50-70kHz频段分析")
    print("="*80)
    print(f"总共有 {len(time_folders)} 个时间点需要处理")
    
    for time_folder in time_folders:
        process_single_timepoint_spectrum(time_folder, center_root_dir, offset_root_dir)
    
    print("\n" + "="*80)
    print("所有时间点频谱分析完成！")
    print("="*80)


if __name__ == "__main__":
    main()







