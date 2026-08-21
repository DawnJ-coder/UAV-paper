
from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
except ImportError as e:
    raise SystemExit(
        "缺少 soundfile，请先安装：pip install soundfile"
    ) from e


# ============================================================
# 1. 配置区
# ============================================================

LEAK_ROOT = Path("/data1/data_for_jiang/leak_data")
NOISE_ROOT = Path("/data1/data_for_jiang/noise_data")

OUTPUT_ROOT = Path("/data1/data_for_jiang/snr_augmented")

LEAK_SEG_ROOT = OUTPUT_ROOT / "segments" / "leak"
NOISE_SEG_ROOT = OUTPUT_ROOT / "segments" / "noise"
MIXED_ROOT = OUTPUT_ROOT / "mixed"

EXPECTED_CHANNELS = 136
SEGMENT_SECONDS = 1.0

# 当前按两档 SNR：-5 dB 和 0 dB
SNR_LIST = [-5.0, 0.0]

# 控制变量：固定“0.1mm铜管泄漏 + 150kPa”，距离 1.0~5.0 m
LEAK_NAME_PREFIX = "0.1mm铜管泄漏_150kPa"
MIN_DISTANCE_M = 1.0
MAX_DISTANCE_M = 5.0

# 是否覆盖已经切好的 1 秒片段 / 已经生成的混合文件
OVERWRITE_SEGMENTS = False
OVERWRITE_MIXED = False

# 固定随机种子：保证每次运行时，信号片段与噪声片段的配对一致
RANDOM_SEED = 20260821

# 使用 FLOAT WAV，避免 136 通道混合后发生整数 PCM 裁剪。
# 不对每条 mixed 做单独归一化，否则会破坏不同距离之间的绝对幅值差异。
OUTPUT_SUBTYPE = "FLOAT"

# 仅使用完整的 1 秒片段；不足 1 秒的尾巴丢弃
DROP_LAST_PARTIAL_SECOND = True


# ============================================================
# 2. 基础工具
# ============================================================

DISTANCE_PATTERN = re.compile(r"_(\d+(?:\.\d+)?)m(?:_|$)")


def find_wav_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"目录不存在：{root}")
    return sorted(p for p in root.rglob("*.wav") if p.is_file())


def parse_distance_m(filename: str) -> float | None:
    """
    从类似：
    0.1mm铜管泄漏_150kPa_240sccm_1.0m_null_136c.wav
    中解析 1.0m。
    """
    match = DISTANCE_PATTERN.search(filename)
    if match is None:
        return None
    return float(match.group(1))


def is_target_leak_file(path: Path) -> bool:
    """
    筛选用于最终混合的泄漏文件：
      1) 文件名以 0.1mm铜管泄漏_150kPa 开头
      2) 距离处于 [1.0, 5.0] m
    """
    if not path.stem.startswith(LEAK_NAME_PREFIX):
        return False

    distance = parse_distance_m(path.stem)
    if distance is None:
        return False

    return MIN_DISTANCE_M <= distance <= MAX_DISTANCE_M


def expected_full_segments(info: sf.SoundFile) -> int:
    frames_per_segment = int(round(info.samplerate * SEGMENT_SECONDS))
    if frames_per_segment <= 0:
        raise ValueError("SEGMENT_SECONDS 设置不合法。")
    return info.frames // frames_per_segment


def segment_output_dir(src_path: Path, src_root: Path, segment_root: Path) -> Path:
    """
    保留源文件的相对目录层级，避免 noise_data 下不同目录出现同名文件时冲突。

    例如：
    /noise_data/136mic/tonglu/xxx/HM....wav
    ->
    /snr_augmented/segments/noise/136mic/tonglu/xxx/HM..../
    """
    rel = src_path.relative_to(src_root)
    return segment_root / rel.parent / src_path.stem


def split_wav_into_1s(
    src_path: Path,
    src_root: Path,
    segment_root: Path,
    overwrite: bool = False,
) -> list[Path]:
    """
    将一个 136 通道 WAV 按完整 1 秒切片。
    返回所有切片路径。
    """
    info = sf.info(str(src_path))

    if info.channels != EXPECTED_CHANNELS:
        raise ValueError(
            f"通道数错误：{src_path}\n"
            f"检测到 {info.channels} 通道，期望 {EXPECTED_CHANNELS} 通道。"
        )

    frames_per_segment = int(round(info.samplerate * SEGMENT_SECONDS))
    total_full_segments = info.frames // frames_per_segment

    out_dir = segment_output_dir(src_path, src_root, segment_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob("sec_*.wav"))

    if not overwrite and len(existing) == total_full_segments:
        return existing

    # 如果数量不匹配，说明上次可能只生成了一部分，清理后重做。
    for old_file in existing:
        old_file.unlink()

    segment_paths: list[Path] = []

    with sf.SoundFile(str(src_path), mode="r") as f:
        for sec_idx in range(total_full_segments):
            data = f.read(
                frames=frames_per_segment,
                dtype="float32",
                always_2d=True,
            )

            if data.shape != (frames_per_segment, EXPECTED_CHANNELS):
                raise RuntimeError(
                    f"切片尺寸异常：{src_path}\n"
                    f"第 {sec_idx} 秒得到 {data.shape}，"
                    f"期望 {(frames_per_segment, EXPECTED_CHANNELS)}。"
                )

            out_path = out_dir / f"sec_{sec_idx:06d}.wav"

            sf.write(
                str(out_path),
                data,
                info.samplerate,
                subtype=OUTPUT_SUBTYPE,
                format="WAV",
            )
            segment_paths.append(out_path)

        # 当前默认丢弃不足 1 秒的尾段。
        if not DROP_LAST_PARTIAL_SECOND:
            remain = f.read(dtype="float32", always_2d=True)
            if len(remain) > 0:
                padded = np.zeros(
                    (frames_per_segment, EXPECTED_CHANNELS),
                    dtype=np.float32,
                )
                padded[: len(remain)] = remain
                out_path = out_dir / f"sec_{total_full_segments:06d}_padded.wav"
                sf.write(
                    str(out_path),
                    padded,
                    info.samplerate,
                    subtype=OUTPUT_SUBTYPE,
                    format="WAV",
                )
                segment_paths.append(out_path)

    return segment_paths


def signal_power(x: np.ndarray) -> float:
    """
    136 通道整体均方功率：
        mean(x^2)
    即同时对时间维和通道维求平均。
    """
    x64 = np.asarray(x, dtype=np.float64)
    return float(np.mean(x64 * x64))


def mix_signal_and_noise_at_snr(
    signal: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> tuple[np.ndarray, float, float]:
    """
    按“136 通道整体功率”控制 SNR。

    只给整段 136 通道噪声乘同一个 scale，
    不对每个通道分别缩放，从而保留原始噪声的通道间空间关系。

    Returns
    -------
    mixed:
        合成后的 136 通道数据
    noise_scale:
        噪声缩放系数
    achieved_snr_db:
        根据缩放后的信号/噪声重新计算的实际 SNR
    """
    if signal.shape != noise.shape:
        raise ValueError(
            f"signal/noise 尺寸不一致：signal={signal.shape}, noise={noise.shape}"
        )

    ps = signal_power(signal)
    pn = signal_power(noise)

    eps = 1e-20
    if ps <= eps:
        raise ValueError("泄漏信号功率接近 0，无法按 SNR 加噪。")
    if pn <= eps:
        raise ValueError("噪声功率接近 0，无法按 SNR 加噪。")

    target_noise_power = ps / (10.0 ** (snr_db / 10.0))
    noise_scale = math.sqrt(target_noise_power / pn)

    scaled_noise = noise.astype(np.float64) * noise_scale
    mixed = signal.astype(np.float64) + scaled_noise

    achieved_snr_db = 10.0 * math.log10(
        ps / signal_power(scaled_noise)
    )

    return mixed.astype(np.float32), noise_scale, achieved_snr_db


def snr_label(snr_db: float) -> str:
    if float(snr_db).is_integer():
        value = int(snr_db)
        if value < 0:
            return f"m{abs(value)}"
        if value > 0:
            return f"p{value}"
        return "0"

    text = f"{abs(snr_db):g}".replace(".", "p")
    if snr_db < 0:
        return f"m{text}"
    if snr_db > 0:
        return f"p{text}"
    return "0"


# ============================================================
# 3. 切分全部泄漏 / 噪声 WAV
# ============================================================

def build_segments():
    leak_files = find_wav_files(LEAK_ROOT)
    noise_files = find_wav_files(NOISE_ROOT)

    print(f"[INFO] 找到泄漏 WAV：{len(leak_files)} 个")
    print(f"[INFO] 找到噪声 WAV：{len(noise_files)} 个")

    leak_segments_by_source: dict[Path, list[Path]] = {}
    noise_segments: list[Path] = []

    print("\n========== 切分泄漏信号 ==========")
    for idx, wav_path in enumerate(leak_files, start=1):
        segs = split_wav_into_1s(
            wav_path,
            LEAK_ROOT,
            LEAK_SEG_ROOT,
            overwrite=OVERWRITE_SEGMENTS,
        )
        leak_segments_by_source[wav_path] = segs
        print(
            f"[LEAK {idx}/{len(leak_files)}] "
            f"{wav_path.name} -> {len(segs)} 段"
        )

    print("\n========== 切分噪声信号 ==========")
    for idx, wav_path in enumerate(noise_files, start=1):
        segs = split_wav_into_1s(
            wav_path,
            NOISE_ROOT,
            NOISE_SEG_ROOT,
            overwrite=OVERWRITE_SEGMENTS,
        )
        noise_segments.extend(segs)
        print(
            f"[NOISE {idx}/{len(noise_files)}] "
            f"{wav_path.name} -> {len(segs)} 段"
        )

    if not noise_segments:
        raise RuntimeError("没有得到任何完整的 1 秒噪声片段。")

    return leak_segments_by_source, noise_segments


# ============================================================
# 4. 构造噪声池：按采样率分组
# ============================================================

def group_noise_segments_by_samplerate(
    noise_segments: list[Path],
) -> dict[int, list[Path]]:
    pools: dict[int, list[Path]] = defaultdict(list)

    for path in noise_segments:
        info = sf.info(str(path))

        if info.channels != EXPECTED_CHANNELS:
            raise ValueError(
                f"噪声切片通道数异常：{path} -> {info.channels}"
            )

        pools[info.samplerate].append(path)

    rng = np.random.default_rng(RANDOM_SEED)

    for sr, paths in pools.items():
        # 同一采样率内部随机打乱，但随机种子固定，可复现。
        order = rng.permutation(len(paths))
        pools[sr] = [paths[i] for i in order]

    return dict(pools)


# ============================================================
# 5. 对目标泄漏条件进行 SNR 混合
# ============================================================

def synthesize_target_dataset(
    leak_segments_by_source: dict[Path, list[Path]],
    noise_segments: list[Path],
):
    noise_pools = group_noise_segments_by_samplerate(noise_segments)
    noise_cursor: dict[int, int] = defaultdict(int)

    target_sources = sorted(
        path
        for path in leak_segments_by_source
        if is_target_leak_file(path)
    )

    if not target_sources:
        raise RuntimeError(
            "没有找到符合条件的泄漏文件。\n"
            f"条件：文件名前缀={LEAK_NAME_PREFIX!r}，"
            f"距离={MIN_DISTANCE_M}~{MAX_DISTANCE_M} m"
        )

    print("\n========== 目标泄漏文件 ==========")
    for path in target_sources:
        print(f"[TARGET] {path.name}")

    MIXED_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_ROOT / "manifest.csv"

    fieldnames = [
        "mixed_wav",
        "leak_source_wav",
        "leak_segment_wav",
        "distance_m",
        "noise_segment_wav",
        "snr_db",
        "achieved_snr_db",
        "noise_scale",
        "sample_rate",
        "channels",
    ]

    rows: list[dict[str, object]] = []
    total_mixed = 0

    for leak_source in target_sources:
        distance_m = parse_distance_m(leak_source.stem)
        leak_segments = leak_segments_by_source[leak_source]

        for leak_seg in leak_segments:
            leak_info = sf.info(str(leak_seg))
            sr = leak_info.samplerate

            if sr not in noise_pools or not noise_pools[sr]:
                available = sorted(noise_pools.keys())
                raise RuntimeError(
                    f"找不到与泄漏采样率 {sr} Hz 匹配的噪声片段。\n"
                    f"泄漏片段：{leak_seg}\n"
                    f"当前噪声采样率：{available}"
                )

            pool = noise_pools[sr]

            # 一个泄漏片段只选一个噪声片段；
            # 同一泄漏片段的不同 SNR 使用同一噪声，仅改变 scale。
            cursor = noise_cursor[sr]
            noise_seg = pool[cursor % len(pool)]
            noise_cursor[sr] += 1

            signal, signal_sr = sf.read(
                str(leak_seg),
                dtype="float32",
                always_2d=True,
            )
            noise, noise_sr = sf.read(
                str(noise_seg),
                dtype="float32",
                always_2d=True,
            )

            if signal_sr != noise_sr:
                raise RuntimeError(
                    f"内部错误：采样率不一致，signal={signal_sr}, noise={noise_sr}"
                )

            if signal.shape[1] != EXPECTED_CHANNELS:
                raise RuntimeError(
                    f"泄漏切片不是 {EXPECTED_CHANNELS} 通道：{leak_seg}"
                )

            if noise.shape[1] != EXPECTED_CHANNELS:
                raise RuntimeError(
                    f"噪声切片不是 {EXPECTED_CHANNELS} 通道：{noise_seg}"
                )

            if signal.shape != noise.shape:
                raise RuntimeError(
                    f"1 秒片段长度不一致：\n"
                    f"signal={leak_seg}, shape={signal.shape}\n"
                    f"noise={noise_seg}, shape={noise.shape}"
                )

            # 输出结构：
            # mixed/<原始泄漏文件名>/sec_000000/snr_m5dB.wav
            sec_dir = (
                MIXED_ROOT
                / leak_source.stem
                / leak_seg.stem
            )
            sec_dir.mkdir(parents=True, exist_ok=True)

            for snr_db in SNR_LIST:
                mixed_path = sec_dir / f"snr_{snr_label(snr_db)}dB.wav"

                mixed, scale, actual_snr = mix_signal_and_noise_at_snr(
                    signal=signal,
                    noise=noise,
                    snr_db=snr_db,
                )

                if OVERWRITE_MIXED or not mixed_path.exists():
                    sf.write(
                        str(mixed_path),
                        mixed,
                        signal_sr,
                        subtype=OUTPUT_SUBTYPE,
                        format="WAV",
                    )

                rows.append(
                    {
                        "mixed_wav": str(mixed_path),
                        "leak_source_wav": str(leak_source),
                        "leak_segment_wav": str(leak_seg),
                        "distance_m": distance_m,
                        "noise_segment_wav": str(noise_seg),
                        "snr_db": snr_db,
                        "achieved_snr_db": f"{actual_snr:.8f}",
                        "noise_scale": f"{scale:.12g}",
                        "sample_rate": signal_sr,
                        "channels": EXPECTED_CHANNELS,
                    }
                )

                total_mixed += 1
                print(
                    f"[MIX] {leak_source.name} | {leak_seg.stem} | "
                    f"SNR={snr_db:g} dB | noise={noise_seg.name} | "
                    f"actual={actual_snr:.6f} dB"
                )

    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n========== 完成 ==========")
    print(f"合成 WAV 数量：{total_mixed}")
    print(f"输出根目录：{OUTPUT_ROOT}")
    print(f"清单文件：{manifest_path}")


# ============================================================
# 6. 主程序
# ============================================================

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    leak_segments_by_source, noise_segments = build_segments()

    synthesize_target_dataset(
        leak_segments_by_source=leak_segments_by_source,
        noise_segments=noise_segments,
    )


if __name__ == "__main__":
    main()
