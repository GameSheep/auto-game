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
    assert (move[1] | (move[2] << 8)) == (down[1] | (down[2] << 8)) == \
        (960 * 32767 // 1919), "移动与按下应同坐标"
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
    assert len(ex._ser.lines) == 2
    line1 = json.loads(ex._ser.lines[0])
    assert line1 == {"op": "key_tap", "key": "a", "hold_ms": 70}
    assert ex._ser.lines[0].endswith("\n"), "JSON 行协议必须换行结尾"
    line2 = json.loads(ex._ser.lines[1])
    assert line2 == {"op": "mouse_click", "button": "left", "x": 800, "y": 600}
    # 急停帧
    ex.panic()
    assert json.loads(ex._ser.lines[2]) == {"op": "stop_all"}
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
