"""语音 —— 能用就用，不能用就退，绝不假装。

## 为什么不做成"必须有"

设备上装 sherpa-onnx 要下载模型（几十到几百 MB）+ onnxruntime（ARM64 轮子）。
装上了当然更好（离线、不依赖浏览器）；但**没装的人也必须能用**：

| 能力 | 有 sherpa | 没有 sherpa |
|---|---|---|
| 听 | 服务端识别（wav 16k） | 浏览器 `SpeechRecognition`（Chrome/Edge 自带） |
| 说 | 服务端合成 wav | 浏览器 `SpeechSynthesis` |

网页就在局域网里，手机/电脑的浏览器本来就带这两样。所以第一版**默认走浏览器**，
sherpa 是可选升级 —— 这样你拿到板子当天就能说话，不用先折腾模型。

## 一个不能含糊的地方

浏览器录音给的是 `webm/opus`，sherpa 要 16k 单声道 PCM。中间必须转码，
而转码要 ffmpeg。所以服务端识别这条路在 `ffmpeg` 缺失时会**明确报错**，
而不是把 webm 硬喂给识别器然后返回一段乱码 —— 那种"能跑但全是错字"的结果
比报错难查得多。
"""

from __future__ import annotations

import io
import logging
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: 一段录音最多这么大。网页端半双工、一次说一句话，10MB 已经是几分钟，
#: 超过这个数基本是传错了文件；直接拒掉比让 ffmpeg 在内存里啃一遍好。
MAX_AUDIO_BYTES = 10 * 1024 * 1024
#: 太短的录音识别出来必然是空的/乱的，直接说"没听清"比返回一串错字好。
MIN_AUDIO_S = 0.2


class VoiceUnavailable(RuntimeError):
    """这台机器上做不了这件事。`hint` 说明缺什么、怎么补。"""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint


def _sherpa() -> Any | None:
    try:
        import sherpa_onnx  # type: ignore
        return sherpa_onnx
    except Exception:  # noqa: BLE001
        return None


def _model_dir(cfg: Any, which: str) -> Path | None:
    raw = str(getattr(cfg, f"{which}_model_dir", "") or "").strip()
    return Path(raw) if raw else None


def capabilities(cfg: Any) -> dict:
    """页面据此决定用哪一套。**不猜**：每一项都写清"为什么不行"。"""
    a = getattr(cfg, "assistant", None)
    sherpa = _sherpa() is not None
    asr_dir = _model_dir(a, "asr") if a is not None else None
    tts_dir = _model_dir(a, "tts") if a is not None else None

    asr_ok = bool(sherpa and asr_dir and asr_dir.is_dir())
    tts_ok = bool(sherpa and tts_dir and tts_dir.is_dir())
    return {
        "server_asr": asr_ok,
        "server_tts": tts_ok,
        "browser_asr": True,     # 由页面再探测一次（SpeechRecognition 不一定有）
        "browser_tts": True,
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "reason": {
            "asr": ("可用" if asr_ok else
                    ("没装 sherpa-onnx" if not sherpa else
                     "assistant.asr_model_dir 没配或不存在")),
            "tts": ("可用" if tts_ok else
                    ("没装 sherpa-onnx" if not sherpa else
                     "assistant.tts_model_dir 没配或不存在")),
        },
    }


# --------------------------------------------------------------------- 听


def _to_wav_16k(data: bytes, hint_name: str = "") -> bytes:
    """把任意音频转成 16k 单声道 wav。已经是 wav 就直接用。"""
    if data[:4] == b"RIFF":
        return data
    ff = shutil.which("ffmpeg")
    if not ff:
        raise VoiceUnavailable(
            "这段录音不是 wav，转码需要 ffmpeg，但这台机器上没装",
            hint="装 ffmpeg（apt install ffmpeg），或者让浏览器直接识别（网页端默认就是这条路）")
    try:
        out = subprocess.run(
            [ff, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
             "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1"],
            input=data, capture_output=True, timeout=60)
    except Exception as e:  # noqa: BLE001
        raise VoiceUnavailable(f"转码失败：{e}") from e
    if out.returncode != 0 or not out.stdout:
        raise VoiceUnavailable("ffmpeg 转码失败：" +
                               (out.stderr or b"").decode("utf-8", "ignore")[:200])
    return out.stdout


def transcribe(cfg: Any, audio: bytes) -> str:
    """服务端识别。不可用时抛 `VoiceUnavailable`（带怎么做）。"""
    a = getattr(cfg, "assistant", None)
    if a is None:
        raise VoiceUnavailable("没有配置 assistant 段")
    if not audio:
        raise VoiceUnavailable("这段录音是空的")
    if len(audio) > MAX_AUDIO_BYTES:
        raise VoiceUnavailable(
            f"录音太大（{len(audio) / 1024 / 1024:.1f}MB，上限 "
            f"{MAX_AUDIO_BYTES / 1024 / 1024:.0f}MB）—— 一句话就够，别一直录")
    sherpa = _sherpa()
    d = _model_dir(a, "asr")
    if sherpa is None:
        raise VoiceUnavailable(
            "服务端没装 sherpa-onnx",
            hint="pip install sherpa-onnx；或者用网页自带的语音识别（默认）")
    if d is None or not d.is_dir():
        raise VoiceUnavailable("assistant.asr_model_dir 没配",
                               hint="把 SenseVoice 模型解压到某个目录，然后填到配置里")

    model = d / "model.onnx"
    tokens = d / "tokens.txt"
    if not (model.exists() and tokens.exists()):
        raise VoiceUnavailable(
            f"{d} 里没找到 model.onnx / tokens.txt",
            hint="这个目录里要有这两个文件（sherpa-onnx 的 SenseVoice 中文模型）")

    wav = _to_wav_16k(audio)
    samples, rate = _read_wav(wav)
    if rate > 0 and len(samples) / rate < MIN_AUDIO_S:
        raise VoiceUnavailable(
            f"录音只有 {len(samples) / max(rate, 1):.2f} 秒，太短了",
            hint="点一下「说话」，说完一整句再点一下")
    try:
        rec = sherpa.OfflineRecognizer.from_sense_voice(
            model=str(model), tokens=str(tokens), num_threads=2, use_itn=True)
        st = rec.create_stream()
        st.accept_waveform(rate, samples)
        rec.decode_stream(st)
        text = (st.result.text or "").strip()
    except Exception as e:  # noqa: BLE001
        raise VoiceUnavailable(f"识别失败：{e}") from e
    return text


def _read_wav(data: bytes) -> tuple[list[float], int]:
    """wav → ([-1,1] 浮点采样, 采样率)。只认 16bit PCM —— 我们自己的转码产物就是它。"""
    import struct

    with wave.open(io.BytesIO(data), "rb") as w:
        rate = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        raise VoiceUnavailable(f"只支持 16bit PCM，这段是 {sw * 8}bit")
    ints = struct.unpack("<" + "h" * (len(raw) // 2), raw)
    if ch > 1:
        ints = ints[::ch]
    return [s / 32768.0 for s in ints], rate


# --------------------------------------------------------------------- 说


def synthesize(cfg: Any, text: str) -> bytes:
    """服务端合成 wav。不可用时抛 `VoiceUnavailable`。"""
    text = str(text or "").strip()
    if not text:
        raise VoiceUnavailable("没有要念的内容")
    a = getattr(cfg, "assistant", None)
    sherpa = _sherpa()
    d = _model_dir(a, "tts") if a is not None else None
    if sherpa is None:
        raise VoiceUnavailable("服务端没装 sherpa-onnx",
                               hint="不装也行：网页会用浏览器自带的朗读")
    if d is None or not d.is_dir():
        raise VoiceUnavailable("assistant.tts_model_dir 没配",
                               hint="把 VITS 中文模型解压到某个目录再填进来")

    model = d / "model.onnx"
    tokens = d / "tokens.txt"
    if not (model.exists() and tokens.exists()):
        raise VoiceUnavailable(f"{d} 里没找到 model.onnx / tokens.txt")
    try:
        tts = sherpa.OfflineTts(sherpa.OfflineTtsConfig(
            model=sherpa.OfflineTtsModelConfig(
                vits=sherpa.OfflineTtsVitsModelConfig(
                    model=str(model), tokens=str(tokens),
                    lexicon=str(d / "lexicon.txt") if (d / "lexicon.txt").exists() else ""),
                num_threads=2),
            max_num_sentences=1))
        audio = tts.generate(str(text)[:200],
                             sid=int(getattr(a, "tts_speaker", 0)))
    except Exception as e:  # noqa: BLE001
        raise VoiceUnavailable(f"合成失败：{e}") from e
    return _wav_bytes(audio.samples, audio.sample_rate)


def _wav_bytes(samples: Any, rate: int) -> bytes:
    import array
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(rate))
        arr = array.array("h", (max(-32767, min(32767, int(s * 32767)))
                                for s in samples))
        w.writeframes(arr.tobytes())
    return buf.getvalue()
