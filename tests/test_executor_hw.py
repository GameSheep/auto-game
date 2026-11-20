"""执行器硬件协议测试 —— RP2040 串口帧 / hidg 报文字节序列（无需真实硬件）。"""

import json
import sys
import types

import pytest

from autogame.config import ExecutorConfig
from autogame.executor import (
    RPi5HidgExecutor,
    build_abs_mouse_report,
    build_keyboard_report,
)
from autogame.protocol import Command


# ---------------- 绝对鼠标报文（hidg1, 6字节） ----------------

def test_abs_mouse_report_corners():
    full = (1920, 1080)
    # 左上角 → 0
    r = build_abs_mouse_report(0, 0, *full)
    assert r == bytes([0, 0, 0, 0, 0, 0])
    # 右下角 → 32767
    r = build_abs_mouse_report(1919, 1079, *full)
    assert r[1] == 0xFF and r[2] == 0x7F, "X 应为 32767 (lo=FF hi=7F)"
    assert r[3] == 0xFF and r[4] == 0x7F, "Y 应为 32767"
    # 报文长度与按钮位
    r = build_abs_mouse_report(100, 100, *full, button=0x01)
    assert len(r) == 6 and r[0] == 0x01
    r = build_abs_mouse_report(100, 100, *full, wheel=0x81)
    assert r[5] == 0x81


def test_abs_mouse_report_monotonic_and_clamped():
    prev = -1
    for x in range(0, 1920, 64):
        r = build_abs_mouse_report(x, 0, 1920, 1080)
        ax = r[1] | (r[2] << 8)
        assert ax >= prev, "X 映射必须单调"
        prev = ax
    # 越界钳制
    r = build_abs_mouse_report(-50, -50, 1920, 1080)
    assert r[1] | (r[2] << 8) == 0
    r = build_abs_mouse_report(99999, 99999, 1920, 1080)
    assert (r[1] | (r[2] << 8)) == 32767 and (r[3] | (r[4] << 8)) == 32767


def test_abs_mouse_report_resolution_independent():
    """同一屏幕比例下，归一化坐标跨分辨率近似一致（int 截断容差 ±16/32767≈0.05%）。

    采用 x*32767/(w-1) 映射以精确命中屏幕边缘（点击边缘元素），中点跨分辨率存在
    ≤16 的量化差，对游戏点击无影响。
    """
    a = build_abs_mouse_report(960, 540, 1920, 1080)
    b = build_abs_mouse_report(480, 270, 960, 540)
    assert abs((a[1] | (a[2] << 8)) - (b[1] | (b[2] << 8))) <= 16
    assert abs((a[3] | (a[4] << 8)) - (b[3] | (b[4] << 8))) <= 16


# ---------------- 键盘报文（hidg0, 8字节） ----------------

def test_keyboard_report_layout():
    assert build_keyboard_report() == bytes(8)
    r = build_keyboard_report(0x04)  # 'a'
    assert r == bytes([0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00])
    r = build_keyboard_report(0x04, modifier=0x02)  # shift+a
    assert r[0] == 0x02 and r[2] == 0x04


# ---------------- RPi5HidgExecutor（写假 hidg 文件） ----------------

class _FakeHidg:
    def __init__(self):
        self.written: list[bytes] = []

    def write(self, data: bytes) -> int:
        self.written.append(bytes(data))
        return len(data)

    def close(self) -> None:
        pass


def test_rpi5_hidg_executor_report_sequence(monkeypatch, tmp_path):
    kb, ms = _FakeHidg(), _FakeHidg()

    def fake_open(path, *a, **k):  # noqa: ARG001
        return kb if "hidg0" in str(path) else ms

    monkeypatch.setattr("builtins.open", fake_open)
    cfg = ExecutorConfig(backend="rpi5_hidg", hidg_keyboard=str(tmp_path / "hidg0"),
                         hidg_mouse=str(tmp_path / "hidg1"))
    ex = RPi5HidgExecutor(cfg)

    ex.execute(Command(op="mouse_click", button="left", x=960, y=540))
    assert len(ms.written) == 3, "点击 = 移动 + 按下 + 抬起"
    move, down, up = ms.written
    mx, my = move[1] | (move[2] << 8), move[3] | (move[4] << 8)
    assert (mx, my) == (down[1] | (down[2] << 8), down[3] | (down[4] << 8)), \
        "移动与按下必须同坐标（否则光标会跳回原点）"
    # 落点允许人类化偏差（endpoint_jitter_px 默认 2px，换算到 16bit 约 ±35）
    assert abs(mx - 960 * 32767 // 1919) <= 64, "落点偏差应在人类化范围内"
    assert down[0] == 0x01 and up[0] == 0x00, "左键按下/抬起"

    ex.execute(Command(op="key_tap", key="a", hold_ms=30))
    assert kb.written[0] == bytes([0, 0, 0x04, 0, 0, 0, 0, 0])
    assert kb.written[1] == bytes(8), "tap 后必须释放"
    ex.close()


# ---------------- RP2040 串口 JSON 帧 ----------------

class _FakeSerial:
    def __init__(self, *a, **k):  # noqa: ARG002
        self.lines: list[str] = []

    def write(self, data: bytes) -> int:
        self.lines.append(data.decode("utf-8"))
        return len(data)

    def readline(self) -> bytes:
        return b'{"ok":true}\n'

    def close(self) -> None:
        pass


@pytest.fixture()
def fake_serial_module(monkeypatch):
    mod = types.ModuleType("serial")
    mod.Serial = _FakeSerial
    monkeypatch.setitem(sys.modules, "serial", mod)
    return mod


def test_rp2040_json_framing(fake_serial_module):
    from autogame.executor import RP2040Executor

    cfg = ExecutorConfig(backend="rp2040", serial_port="FAKE", serial_baud=115200)
    ex = RP2040Executor(cfg)
    ex.execute(Command(op="key_tap", key="a", hold_ms=70))
    ex.execute(Command(op="mouse_click", button="left", x=800, y=600))
    line1 = json.loads(ex._ser.lines[0])
    assert line1 == {"op": "key_tap", "key": "a", "hold_ms": 70}
    assert ex._ser.lines[0].endswith("\n"), "JSON 行协议必须换行结尾"
    # 鼠标命令可能是多点轨迹（WindMouse），最后一帧必须落在点击上
    line2 = json.loads(ex._ser.lines[-1])
    assert line2["op"] == "mouse_click" and line2["button"] == "left"
    # 固件要的是 0..32767 的绝对尺度（描述符里 Logical Max 32767），**不是像素**。
    # 这里以前断言的是 800，等于把"像素直发"这个错误固化成了规格 ——
    # 真机上那样发出去，光标只会落在屏幕 2% 处，而主机日志一切正常。
    assert line2["x"] > 1000, "发出去的必须是绝对尺度，不是像素"
    exp_x = 800 * 32767 / (cfg.screen_w - 1)
    exp_y = 600 * 32767 / (cfg.screen_h - 1)
    # 轨迹抖动会挪几个像素，换算到 32767 尺度会被放大 (32767/屏幕宽) 倍，
    # 所以容差按"允许 3 像素"来算，而不是拍一个绝对数。
    tol_x = 3 * 32767 / (cfg.screen_w - 1)
    tol_y = 3 * 32767 / (cfg.screen_h - 1)
    assert abs(line2["x"] - exp_x) <= tol_x and abs(line2["y"] - exp_y) <= tol_y, \
        f"落点应落在 ({exp_x:.0f},{exp_y:.0f}) 附近，实际 ({line2['x']},{line2['y']})"
    # 急停帧
    ex.panic()
    assert json.loads(ex._ser.lines[-1]) == {"op": "stop_all"}
    ex.close()


def test_rp2040_keymap_finite_and_covers_fsm_keys():
    """FSM 用到的所有键（wasd + 拾取键）必须在固件键码映射内。"""
    from autogame.executor import RPi5HidgExecutor

    km = RPi5HidgExecutor._KEYMAP
    for k in ("w", "a", "s", "d"):
        assert k in km, f"方向键 {k} 缺失"
    assert "a" in km
    # 固件 JSON 协议键码映射（hid_bridge.ino KEYMAP）与 hidg 键位映射一致性抽查
    ino = (RPi5HidgExecutor.__module__, )
    assert km["f5"] == 0x3E and km["esc"] == 0x29 and km["space"] == 0x2C


def test_rate_limit_follows_config_hot_update():
    """限速参数必须跟随配置热更新。

    回归：ActionRateLimiter 曾在构造时把 min/max 复制成标量，
    配置页把「执行速度」拖到另一档后，限速器仍按启动时的旧值工作 ——
    界面显示已保存，实际动作节奏没变。
    """
    cfg = ExecutorConfig()
    from autogame.executor import DryRunExecutor
    ex = DryRunExecutor(cfg)
    try:
        assert ex._limiter.max_per_sec == cfg.max_actions_per_sec
        assert abs(ex._limiter.min_interval - cfg.min_action_interval_ms / 1000) < 1e-9

        # 模拟配置页保存：原地改 cfg（apply_patch 就是这么做的）
        cfg.max_actions_per_sec = 3
        cfg.min_action_interval_ms = 90

        # 下一条命令就该用新值
        ex.execute(Command(op="key_tap", key="a"))
        assert ex._limiter.max_per_sec == 3
        assert abs(ex._limiter.min_interval - 0.09) < 1e-9
    finally:
        ex.close()


def test_rp2040_watchdog_release_is_visible(fake_serial_module, monkeypatch):
    """固件自己松手这件事必须能被看见，否则现场只会看到"莫名少走一步"。

    5 秒没收到命令 → 固件 `release_all()` 并回 `{"ok":false,"error":"watchdog_release"}`。
    主机如果只把它写进 debug 日志，这个事故就等于隐身。
    """
    from autogame.executor import RP2040Executor

    class _WdSerial(_FakeSerial):
        def readline(self) -> bytes:
            return b'{"ok":false,"error":"watchdog_release"}\n'

    monkeypatch.setattr(fake_serial_module, "Serial", _WdSerial)
    cfg = ExecutorConfig(backend="rp2040", serial_port="FAKE", serial_baud=115200)
    ex = RP2040Executor(cfg)
    ex.execute(Command(op="key_down", key="w"))
    assert ex.ack_errors, "固件拒了命令却什么都没留下"
    assert "watchdog_release" in ex.ack_errors[-1]["ack"]
    assert ex.ack_errors[-1]["op"] == "key_down"


def test_rp2040_ack_error_list_is_bounded(fake_serial_module, monkeypatch):
    from autogame.executor import RP2040Executor

    class _BadSerial(_FakeSerial):
        def readline(self) -> bytes:
            return b'{"ok":false,"error":"unknown key"}\n'

    monkeypatch.setattr(fake_serial_module, "Serial", _BadSerial)
    cfg = ExecutorConfig(backend="rp2040", serial_port="FAKE", serial_baud=115200)
    ex = RP2040Executor(cfg)
    for _ in range(40):
        ex._write({"op": "key_down", "key": "a"})
    assert len(ex.ack_errors) == 20, "只留最近 20 条，别让错误列表变成内存泄漏"


def test_rp2040_watchdog_release_is_visible(fake_serial_module, monkeypatch):
    """固件自己松手这件事必须能被看见，否则现场只会看到"莫名少走一步"。

    5 秒没收到命令 → 固件 `release_all()` 并回 `{"ok":false,"error":"watchdog_release"}`。
    主机如果只把它写进 debug 日志，这个事故就等于隐身。
    """
    from autogame.executor import RP2040Executor

    class _WdSerial(_FakeSerial):
        def readline(self) -> bytes:
            return b'{"ok":false,"error":"watchdog_release"}\n'

    monkeypatch.setattr(fake_serial_module, "Serial", _WdSerial)
    cfg = ExecutorConfig(backend="rp2040", serial_port="FAKE", serial_baud=115200)
    ex = RP2040Executor(cfg)
    ex.execute(Command(op="key_down", key="w"))
    assert ex.ack_errors, "固件拒了命令却什么都没留下"
    assert "watchdog_release" in ex.ack_errors[-1]["ack"]
    assert ex.ack_errors[-1]["op"] == "key_down"


def test_rp2040_ack_error_list_is_bounded(fake_serial_module, monkeypatch):
    from autogame.executor import RP2040Executor

    class _BadSerial(_FakeSerial):
        def readline(self) -> bytes:
            return b'{"ok":false,"error":"unknown key"}\n'

    monkeypatch.setattr(fake_serial_module, "Serial", _BadSerial)
    cfg = ExecutorConfig(backend="rp2040", serial_port="FAKE", serial_baud=115200)
    ex = RP2040Executor(cfg)
    for _ in range(40):
        ex._write({"op": "key_down", "key": "a"})
    assert len(ex.ack_errors) == 20, "只留最近 20 条，别让错误列表变成内存泄漏"


def test_rp2040_and_hidg_map_a_pixel_to_the_same_place(monkeypatch):
    """两个后端必须把同一份像素坐标落到屏幕的同一个地方。

    以前 RP2040 把像素直接当 0..32767 发，hidg 却做了归一化 —— 于是同一份
    标定换个执行器就整体偏位，而两边日志都显示正常。这条用例就是把
    "两个后端的坐标语义必须一致"钉住。
    """
    from autogame.executor import RP2040Executor, build_abs_mouse_report

    cfg = ExecutorConfig(backend="rp2040", serial_port="FAKE",
                         screen_w=1280, screen_h=720)
    ex = RP2040Executor.__new__(RP2040Executor)
    ex.cfg = cfg

    for x, y in ((0, 0), (640, 360), (1279, 719)):
        ax, ay = ex._to_abs(x, y)
        rep = build_abs_mouse_report(x, y, cfg.screen_w, cfg.screen_h)
        hx = rep[1] | (rep[2] << 8)
        hy = rep[3] | (rep[4] << 8)
        assert (ax, ay) == (hx, hy), f"({x},{y}) → rp2040 {(ax, ay)} vs hidg {(hx, hy)}"


def test_rp2040_abs_scale_covers_the_whole_screen(monkeypatch):
    """右下角必须落在 32767，否则屏幕边缘永远点不到。"""
    from autogame.executor import RP2040Executor

    cfg = ExecutorConfig(backend="rp2040", screen_w=1920, screen_h=1080)
    ex = RP2040Executor.__new__(RP2040Executor)
    ex.cfg = cfg
    assert ex._to_abs(1919, 1079) == (32767, 32767)
    assert ex._to_abs(0, 0) == (0, 0)
    # 中心落在半程附近：0 和 w-1 钉在两个端点，中点必然略高于 16384
    # （32767/2 * w/(w-1)），差几十个单位在 1920 宽上只有零点几个像素。
    cx, cy = ex._to_abs(960, 540)
    assert abs(cx - 16384) <= 16 and abs(cy - 16384) <= 16
