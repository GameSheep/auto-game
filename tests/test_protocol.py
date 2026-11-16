"""协议与安全模块测试。"""

import pytest

from autogame.protocol import pack_jpeg_frame, unpack_message
from autogame.security import ActionRateLimiter, check_token


def test_jpeg_frame_roundtrip():
    jpeg = b"\xff\xd8fakejpegdata\xff\xd9"
    raw = pack_jpeg_frame(jpeg, 1280, 720, 123456789)
    kind, payload = unpack_message(raw)
    assert kind == "jpeg"
    assert payload["jpeg"] == jpeg
    assert payload["w"] == 1280
    assert payload["h"] == 720
    assert payload["ts_ms"] == 123456789 & 0xFFFFFFFF


def test_json_message_roundtrip():
    kind, payload = unpack_message('{"type":"hello","agent":"pc"}')
    assert kind == "json"
    assert payload["type"] == "hello"


def test_bad_messages_rejected():
    with pytest.raises(ValueError):
        unpack_message(b"")
    with pytest.raises(ValueError):
        unpack_message(b"\x02\x00\x00")
    with pytest.raises(Exception):
        unpack_message("not json")


def test_token_constant_time_check():
    assert check_token("secret", "secret")
    assert not check_token("wrong", "secret")
    assert not check_token(None, "secret")
    assert not check_token("", "secret")


def test_rate_limiter_min_interval():
    rl = ActionRateLimiter(min_interval_ms=50, max_per_sec=100)
    assert rl.allow() is True
    assert rl.allow() is False  # 间隔不足
    assert rl.dropped == 1


def test_rate_limiter_window_cap():
    rl = ActionRateLimiter(min_interval_ms=0, max_per_sec=3)
    assert rl.allow()
    assert rl.allow()
    assert rl.allow()
    assert rl.allow() is False  # 超出窗口上限
