"""Bounded local WAV validation and media previews (no playback)."""
from __future__ import annotations

import hashlib
import io
import math
import struct
from pathlib import Path

import numpy as np
import soundfile as sf

MAX_BYTES = 4 * 1024**3
MAX_SECONDS = 24 * 60 * 60
BLOCK_FRAMES = 65536
SUBTYPES = {"PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"}


def _container(path):
    """Validate RIFF chunk extents independently of libsndfile's tolerant reader."""
    size = path.stat().st_size
    if size > MAX_BYTES:
        raise ValueError("音频超过4GiB限制")
    with path.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12 or header[0:4] not in {b"RIFF", b"RF64"} or header[8:12] != b"WAVE":
            raise ValueError("必须是完整的 WAV/WAVEX/RF64 音频容器")
        rf64 = header[:4] == b"RF64"
        declared = struct.unpack("<I", header[4:8])[0]
        if not rf64 and declared + 8 != size:
            raise ValueError("WAV容器长度不完整或存在多余数据")
        if rf64 and declared != 0xFFFFFFFF:
            raise ValueError("RF64容器头无效")
        data_bytes = None
        rf64_data = None
        position = 12
        saw_format = False
        while position < size:
            stream.seek(position)
            chunk = stream.read(8)
            if len(chunk) != 8:
                raise ValueError("WAV块头被截断")
            kind, length = chunk[:4], struct.unpack("<I", chunk[4:])[0]
            if kind == b"ds64":
                if not rf64 or length < 28:
                    raise ValueError("RF64长度描述无效")
                ds64 = stream.read(28)
                if len(ds64) != 28:
                    raise ValueError("RF64长度描述被截断")
                riff_size, rf64_data, _, table_length = struct.unpack("<QQQI", ds64)
                if riff_size + 8 != size or length < 28 + table_length * 12:
                    raise ValueError("RF64容器长度不完整")
            if length == 0xFFFFFFFF:
                if kind != b"data" or rf64_data is None:
                    raise ValueError("不支持的RF64扩展块长度")
                length = rf64_data
            end = position + 8 + length
            if end > size or end + (length % 2) > size:
                raise ValueError("WAV数据块被截断")
            if kind == b"fmt ":
                saw_format = True
            if kind == b"data":
                if data_bytes is not None:
                    raise ValueError("不支持多个音频数据块")
                data_bytes = length
            position = end + length % 2
        if not saw_format or data_bytes is None or (rf64 and rf64_data is None):
            raise ValueError("缺少WAV格式或音频数据块")
    return size, data_bytes


def audio_info_validate(path, stop_requested=None):
    path = Path(path)
    size, data_bytes = _container(path)
    digest = hashlib.sha256()
    def cancelled():
        if stop_requested and stop_requested():
            raise InterruptedError("音频校验已取消")
    try:
        with sf.SoundFile(path) as audio:
            rate, channels, frames = audio.samplerate, audio.channels, audio.frames
            if audio.format not in {"WAV", "WAVEX", "RF64"} or audio.subtype not in SUBTYPES:
                raise ValueError("只接受WAV PCM16/24/32或浮点音频")
            if channels not in {1, 2} or not 8000 <= rate <= 96000:
                raise ValueError("音频须为单声道/双声道、8–96kHz")
            if frames <= 0 or frames / rate > MAX_SECONDS:
                raise ValueError("音频不能为空或超过24小时")
            sample_bytes = {"PCM_16": 2, "PCM_24": 3, "PCM_32": 4, "FLOAT": 4, "DOUBLE": 8}[audio.subtype]
            if data_bytes != frames * channels * sample_bytes:
                raise ValueError("音频实际数据长度与帧数不一致")
            count = 0
            while True:
                cancelled()
                block = audio.read(BLOCK_FRAMES, dtype="float64", always_2d=True)
                if not len(block):
                    break
                if not np.isfinite(block).all() or (np.abs(block) > 1).any():
                    raise ValueError("音频含非有限数值或超出[-1,1]范围")
                count += len(block)
            if count != frames:
                raise ValueError("音频实际读取帧数不一致或数据被截断")
            result = dict(duration_seconds=frames / rate, sample_rate=rate, channels=channels, frames=frames,
                          format=audio.format, subtype=audio.subtype, bytes=size)
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                cancelled()
                digest.update(block)
        return {**result, "sha256": digest.hexdigest()}
    except (sf.LibsndfileError, RuntimeError) as exc:
        raise ValueError("音频无法完整解码，请重新导出WAV") from exc


def _number(value, name):
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name}必须为有限秒数") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name}必须为有限秒数")
    return number


def _range(audio, start, end):
    start, end = _number(start, "开始时间"), _number(end, "结束时间")
    duration = audio.frames / audio.samplerate
    if start < 0 or end <= start or end > duration + 1e-9:
        raise ValueError("时间区间必须位于音频范围内且结束晚于开始")
    first = min(audio.frames, round(start * audio.samplerate))
    last = min(audio.frames, round(end * audio.samplerate))
    if last <= first:
        raise ValueError("所选时间区间不足一个采样点")
    return first, last


def read_mono_window(path, start_seconds, duration_seconds):
    start = _number(start_seconds, "开始时间")
    duration = _number(duration_seconds, "时长")
    if duration <= 0:
        raise ValueError("时长必须大于零")
    with sf.SoundFile(path) as audio:
        first, last = _range(audio, start, start + duration)
        audio.seek(first)
        samples = audio.read(last - first, dtype="float32", always_2d=True)
        if len(samples) != last - first:
            raise ValueError("音频窗口数据不完整")
        return samples.mean(axis=1, dtype=np.float32), audio.samplerate


def waveform(path, points=1600, start=0, end=None):
    if isinstance(points, bool) or not isinstance(points, int) or not 1 <= points <= 4000:
        raise ValueError("波形点数须在1–4000之间")
    with sf.SoundFile(path) as audio:
        duration = audio.frames / audio.samplerate
        first, last = _range(audio, start, duration if end is None else end)
        edges = np.linspace(first, last, min(points, last - first) + 1, dtype=np.int64)
        output = []
        audio.seek(first)
        for left, right in zip(edges[:-1], edges[1:]):
            minimum, maximum = math.inf, -math.inf
            remaining = int(right - left)
            while remaining:
                block = audio.read(min(remaining, BLOCK_FRAMES), dtype="float32", always_2d=True)
                if not len(block):
                    raise ValueError("波形数据被截断")
                mono = block.mean(axis=1, dtype=np.float32)
                minimum = min(minimum, float(mono.min()))
                maximum = max(maximum, float(mono.max()))
                remaining -= len(block)
            output.append(dict(time_seconds=float((left + right) / (2 * audio.samplerate)), min=minimum, max=maximum))
        return dict(duration_seconds=duration, start_seconds=first / audio.samplerate,
                    end_seconds=last / audio.samplerate, points=output)


def spectrogram_png(path, start=0, duration=10):
    duration = _number(duration, "时长")
    if not 0 < duration <= 60:
        raise ValueError("声谱图时长须大于0且不超过60秒")
    samples, rate = read_mono_window(path, start, duration)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from scipy.signal import resample_poly, spectrogram
    # Downsample only the preview to bound rendering work at high input rates.
    stride = max(1, math.ceil(rate / 16000))
    if stride > 1:
        samples = resample_poly(samples, 1, stride)
    view_rate = rate / stride
    nperseg = min(512, len(samples))
    if nperseg < 2:
        raise ValueError("所选区间过短，无法生成声谱图")
    frequencies, times, power = spectrogram(samples, fs=view_rate, nperseg=nperseg, noverlap=nperseg // 2)
    db = 10 * np.log10(np.maximum(power, 1e-16))
    db -= db.max()
    figure = Figure(figsize=(9, 3), dpi=100)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    image = axis.pcolormesh(times + float(start), frequencies, db, shading="auto", vmin=-80, vmax=0, cmap="magma")
    axis.set(xlabel="Time (s)", ylabel="Frequency (Hz)", title="Relative spectrum (not calibrated SPL)")
    figure.colorbar(image, ax=axis, label="Relative dB")
    figure.tight_layout()
    output = io.BytesIO()
    figure.savefig(output, format="png")
    figure.clear()
    return output.getvalue()


def deterministic_demo(path):
    """Generate synthetic audio; no real speaker, device or PSG observations."""
    rate = 16000
    events = [dict(start_seconds=a, end_seconds=b, label="synthetic_breath", source="demo", score=None, model_id=None)
              for a, b in [(15, 23), (40, 49), (65, 76)]]
    rng = np.random.default_rng(20261026)
    with sf.SoundFile(path, "w", samplerate=rate, channels=1, subtype="PCM_16", format="WAV") as audio:
        for second in range(90):
            t = second + np.arange(rate) / rate
            envelope = sum(((t >= e["start_seconds"]) & (t < e["end_seconds"])).astype(float) for e in events)
            samples = .002 * rng.normal(size=rate) + .035 * envelope * (0.5 + 0.5 * np.sin(2 * np.pi * .3 * t)) * np.sin(2 * np.pi * 150 * t)
            audio.write(samples)
    wearable = dict(source="user_import", time_unit="relative_seconds",
                    heart_rate=[dict(time_seconds=t, value=60 + 2 * math.sin(t / 15)) for t in range(0, 90, 5)],
                    stages=[dict(time_seconds=0, duration_seconds=30, stage="W"), dict(time_seconds=30, duration_seconds=60, stage="N2")],
                    acceleration=[], summary={"synthetic": True}, warnings=["合成演示时序，不是真人、设备或PSG观察"])
    return dict(description="确定性合成90秒音频和示例时序，非真人录音或PSG参考", events=events, wearable=wearable)
