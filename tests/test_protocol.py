"""协议与安全模块测试。"""

import time

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


def test_rate_limiter_acquire_waits_instead_of_dropping():
    """连续合法命令必须被"等"出来，而不是被 sleep 精度问题丢掉。

    回归背景：Windows 默认定时器 15.6ms，请求 26ms 的 sleep 可能 16ms 就返回，
    旧的 sleep(delay_needed()) 方案实测会丢 26% 的连续命令 ——
    现场表现是"偶尔少按一下 / 点了没反应"，极难排查。
    """
    rl = ActionRateLimiter(min_interval_ms=20, max_per_sec=100)
    assert rl.acquire() is True
    t0 = time.monotonic()
    for _ in range(5):
        assert rl.acquire(0.5) is True, "连续命令必须全部放行"
    assert time.monotonic() - t0 >= 0.09, "必须真的等到最小间隔（5×20ms）"
    assert rl.dropped == 0


def test_rate_limiter_acquire_drops_immediately_on_flood():
    """速率窗口满（指令风暴）→ 立即丢弃，不做无谓等待。"""
    rl = ActionRateLimiter(min_interval_ms=0, max_per_sec=2)
    assert rl.acquire() and rl.acquire()
    t0 = time.monotonic()
    assert rl.acquire(0.5) is False
    assert time.monotonic() - t0 < 0.1, "指令风暴应立即丢弃，不该等满 max_wait"
    assert rl.dropped == 1
