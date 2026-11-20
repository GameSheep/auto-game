"""语音：能听能说就用，不能就老老实实说"缺什么"。

这组用例不去下载真模型（几百 MB，也不该在单测里跑），而是用假 sherpa
把**调用形状**钉住：模型目录怎么找、wav 怎么读、立体声怎么降、空/短/超大
录音怎么拒、装不上时给出的提示够不够让人自己修好。

真模型只影响"识别得准不准"，那是另一类问题；这里保证的是"链路对不对"。
"""

from __future__ import annotations

import io
import wave
from types import SimpleNamespace

import pytest

from autogame.assistant import voice
from autogame.config import AppConfig

TOKEN = "t"


# ---------------- 假的 sherpa：只实现我们真正用到的形状 ----------------

class _Stream:
    def __init__(self, text="先停下"):
        self.result = SimpleNamespace(text=text)
        self.audio = None

    def accept_waveform(self, rate, samples):
        self.audio = (rate, list(samples))


class _Recognizer:
    def __init__(self):
        self.stream = _Stream()

    def create_stream(self):
        return self.stream

    def decode_stream(self, st):
        pass


class _TtsConfig:
    def __init__(self, **kw):
        self.kw = kw


class FakeSherpa:
    class OfflineRecognizer:
        @staticmethod
        def from_sense_voice(**kw):
            return _Recognizer()

    OfflineTtsConfig = _TtsConfig
    OfflineTtsModelConfig = _TtsConfig
    OfflineTtsVitsModelConfig = _TtsConfig

    class OfflineTts:
        def __init__(self, cfg):
            self.cfg = cfg

        def generate(self, text, sid=0):
            return SimpleNamespace(samples=[0.0] * 1600, sample_rate=16000)


def _model_dir(tmp_path, name="asr"):
    d = tmp_path / name
    d.mkdir()
    (d / "model.onnx").write_bytes(b"fake")
    (d / "tokens.txt").write_text("a 1\n", encoding="utf-8")
    return d


def _cfg(tmp_path, **kw):
    cfg = AppConfig()
    for k, v in kw.items():
        setattr(cfg.assistant, k, str(v) if k.endswith("_dir") else v)
    return cfg


def _wav(seconds=0.5, rate=16000, channels=1):
    n = int(seconds * rate)
    return voice._wav_bytes([0.1] * n * channels, rate) if channels == 1 else _stereo_wav(n, rate)


def _stereo_wav(frames, rate):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x10\x00\x10" * frames)
    return buf.getvalue()


# ═══════════ 1. 能力探测：说清"为什么不行" ═══════════

def test_capabilities_explain_missing_sherpa(tmp_path, monkeypatch):
    monkeypatch.setattr(voice, "_sherpa", lambda: None)
    caps = voice.capabilities(_cfg(tmp_path))
    assert caps["server_asr"] is False and caps["server_tts"] is False
    assert "sherpa" in caps["reason"]["asr"]
    assert caps["browser_asr"] and caps["browser_tts"], "浏览器那套永远可用（页面上再探测）"


def test_capabilities_ok_when_models_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(voice, "_sherpa", lambda: FakeSherpa)
    cfg = _cfg(tmp_path, asr_model_dir=_model_dir(tmp_path, "asr"),
               tts_model_dir=_model_dir(tmp_path, "tts"))
    caps = voice.capabilities(cfg)
    assert caps["server_asr"] and caps["server_tts"]


def test_capabilities_point_at_the_config_field(tmp_path, monkeypatch):
    monkeypatch.setattr(voice, "_sherpa", lambda: FakeSherpa)
    caps = voice.capabilities(_cfg(tmp_path))
    assert "asr_model_dir" in caps["reason"]["asr"]


# ═══════════ 2. 听 ═══════════

def test_transcribe_returns_model_text(tmp_path, monkeypatch):
    monkeypatch.setattr(voice, "_sherpa", lambda: FakeSherpa)
    cfg = _cfg(tmp_path, asr_model_dir=_model_dir(tmp_path))
    assert voice.transcribe(cfg, _wav()) == "先停下"


def test_transcribe_passes_16k_mono_to_the_model(tmp_path, monkeypatch):
    monkeypatch.setattr(voice, "_sherpa", lambda: FakeSherpa)
    cfg = _cfg(tmp_path, asr_model_dir=_model_dir(tmp_path))
    voice.transcribe(cfg, _wav(seconds=0.5))
    # 直接验证读 wav 的口径：立体声要降成单声道、采样率原样
    samples, rate = voice._read_wav(_stereo_wav(8000, 16000))
    assert rate == 16000 and len(samples) == 8000


def test_transcribe_rejects_empty_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(voice, "_sherpa", lambda: FakeSherpa)
    with pytest.raises(voice.VoiceUnavailable, match="空的"):
        voice.transcribe(_cfg(tmp_path), b"")


def test_transcribe_rejects_oversized_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(voice, "_sherpa", lambda: FakeSherpa)
    big = b"x" * (voice.MAX_AUDIO_BYTES + 1)
    with pytest.raises(voice.VoiceUnavailable, match="太大"):
        voice.transcribe(_cfg(tmp_path), big)


def test_transcribe_rejects_too_short_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(voice, "_sherpa", lambda: FakeSherpa)
    cfg = _cfg(tmp_path, asr_model_dir=_model_dir(tmp_path))
    with pytest.raises(voice.VoiceUnavailable, match="太短"):
        voice.transcribe(cfg, _wav(seconds=0.05))


def test_missing_ffmpeg_gives_an_actionable_message(tmp_path, monkeypatch):
    monkeypatch.setattr(voice, "_sherpa", lambda: FakeSherpa)
    monkeypatch.setattr(voice.shutil, "which", lambda *_a, **_k: None)
    cfg = _cfg(tmp_path, asr_model_dir=_model_dir(tmp_path))
    with pytest.raises(voice.VoiceUnavailable) as ei:
        voice.transcribe(cfg, b"\x1aE\xdf\xa3webm-not-wav")
    assert "ffmpeg" in str(ei.value) and ei.value.hint


def test_wav_passthrough_is_not_transcoded(tmp_path):
    """已经是 wav 就别再走 ffmpeg —— 那一步在板子上又慢又可能没装。"""
    raw = _wav()
    assert voice._to_wav_16k(raw) == raw


def test_read_wav_rejects_non_16bit():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(16000)
        w.writeframes(b"\x80" * 100)
    with pytest.raises(voice.VoiceUnavailable, match="16bit"):
        voice._read_wav(buf.getvalue())


def test_transcribe_without_model_dir_says_which_field(tmp_path, monkeypatch):
    monkeypatch.setattr(voice, "_sherpa", lambda: FakeSherpa)
    with pytest.raises(voice.VoiceUnavailable) as ei:
        voice.transcribe(_cfg(tmp_path), _wav())
    assert "asr_model_dir" in str(ei.value)


# ═══════════ 3. 说 ═══════════

def test_synthesize_returns_playable_wav(tmp_path, monkeypatch):
    monkeypatch.setattr(voice, "_sherpa", lambda: FakeSherpa)
    cfg = _cfg(tmp_path, tts_model_dir=_model_dir(tmp_path, "tts"))
    data = voice.synthesize(cfg, "已经停下了")
    assert data[:4] == b"RIFF"
    samples, rate = voice._read_wav(data)
    assert rate == 16000 and len(samples) == 1600


def test_synthesize_rejects_empty_text(tmp_path, monkeypatch):
    monkeypatch.setattr(voice, "_sherpa", lambda: FakeSherpa)
    with pytest.raises(voice.VoiceUnavailable, match="没有要念"):
        voice.synthesize(_cfg(tmp_path), "   ")


def test_synthesize_without_model_dir_says_which_field(tmp_path, monkeypatch):
    monkeypatch.setattr(voice, "_sherpa", lambda: FakeSherpa)
    with pytest.raises(voice.VoiceUnavailable) as ei:
        voice.synthesize(_cfg(tmp_path), "喂")
    assert "tts_model_dir" in str(ei.value)


def test_wav_roundtrip_keeps_length_and_rate():
    data = voice._wav_bytes([0.0, 0.5, -0.5], 8000)
    samples, rate = voice._read_wav(data)
    assert rate == 8000 and len(samples) == 3


# ═══════════ 4. HTTP：别名与失败口径 ═══════════

def _client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from autogame import assistant as asst_mod
    from autogame.executor import DryRunExecutor
    from autogame.server.app import create_app
    from autogame.server.hub import Hub
    from autogame.vision.pipeline import VisionPipeline

    cfg = AppConfig()
    cfg.server.token = TOKEN
    cfg.executor.min_action_interval_ms = 0
    cfg.assistant.enabled = False
    hub = Hub(cfg, DryRunExecutor(cfg.executor), VisionPipeline(cfg.vision))
    asst_mod.build(cfg, None, hub)
    return TestClient(create_app(cfg, hub, "<h1>x</h1>")), cfg


def test_http_voice_alias_exists_and_reports_unsupported(tmp_path, monkeypatch):
    monkeypatch.setattr(voice, "_sherpa", lambda: None)
    c, cfg = _client(tmp_path, monkeypatch)
    r = c.post("/api/voice/transcribe", headers={"X-Token": TOKEN},
               content=b"webm-bytes")
    d = r.json()
    assert d["ok"] is False and d["unsupported"] is True
    assert "sherpa" in d["error"] and d.get("hint")


def test_http_voice_runs_the_same_conversation(tmp_path, monkeypatch):
    monkeypatch.setattr(voice, "_sherpa", lambda: FakeSherpa)
    c, cfg = _client(tmp_path, monkeypatch)
    cfg.assistant.asr_model_dir = str(_model_dir(tmp_path))
    d = c.post("/api/assistant/voice", headers={"X-Token": TOKEN},
               content=_wav()).json()
    assert d["ok"] and d["heard"] == "先停下"
    assert d["mode"]["mode"] == "paused", "说'先停下'必须真的停"


def test_http_voice_without_audio_is_explicit(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    d = c.post("/api/assistant/voice", headers={"X-Token": TOKEN}, content=b"").json()
    assert d["ok"] is False and "没收到音频" in d["error"]


def test_http_tts_reports_unsupported_instead_of_noise(tmp_path, monkeypatch):
    monkeypatch.setattr(voice, "_sherpa", lambda: None)
    c, _ = _client(tmp_path, monkeypatch)
    d = c.post("/api/assistant/tts", headers={"X-Token": TOKEN},
               json={"text": "喂"}).json()
    assert d["ok"] is False and d["unsupported"] is True


def test_http_tts_returns_a_wav(tmp_path, monkeypatch):
    monkeypatch.setattr(voice, "_sherpa", lambda: FakeSherpa)
    c, cfg = _client(tmp_path, monkeypatch)
    cfg.assistant.tts_model_dir = str(_model_dir(tmp_path, "tts"))
    r = c.post("/api/assistant/tts", headers={"X-Token": TOKEN}, json={"text": "喂"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    assert r.content[:4] == b"RIFF"
