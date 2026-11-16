"""执行器抽象与各后端。

统一命令入口 Command → 各后端转换为物理输出：
  dryrun     : 仅记录日志（测试/联调默认，绝不触碰真实输入）
  rp2040     : USB 串口 JSON 协议 → RP2040 组合HID 固件（键盘+绝对鼠标）
  rpi5_hidg  : Linux USB gadget (/dev/hidg0 键盘 /dev/hidg1 绝对鼠标)
  pca9685    : PCA9685+舵机 机械按压（对话定稿机械方案，仅按键无鼠标移动）

所有后端共享限速器与急停语义。
"""

from __future__ import annotations

import abc
import json
import logging
import random
import threading
import time

from .base import BaseExecutor, _jitter, build_abs_mouse_report, build_keyboard_report
from .pca9685 import Pca9685Executor  # noqa: F401 (build_executor 按需构造)
from ..config import ExecutorConfig
from ..protocol import Command
from ..security import ActionRateLimiter

log = logging.getLogger(__name__)


class DryRunExecutor(BaseExecutor):
    """干跑执行器：只记录，不产生任何真实输入。测试与联调默认。"""

    backend_name = "dryrun"

    def _emit(self, cmd: Command, bypass: bool = False) -> None:
        log.info("[dryrun] %s", cmd.model_dump_json(exclude_none=True))


class RP2040Executor(BaseExecutor):
    """RP2040 USB 串口执行器（组合HID 固件见 firmware/rp2040/hid_bridge.ino）。

    协议：JSON 行（\n 结尾），固件回显 {"ok":true,...}。
    """

    backend_name = "rp2040"

    def __init__(self, cfg: ExecutorConfig) -> None:
        super().__init__(cfg)
        import serial  # pyserial

        self._ser = serial.Serial(cfg.serial_port, cfg.serial_baud, timeout=0.5)
        log.info("RP2040 串口已连接: %s @%d", cfg.serial_port, cfg.serial_baud)

    def _emit(self, cmd: Command, bypass: bool = False) -> None:
        payload = cmd.model_dump(exclude_none=True)
        # 帧最小化：各命令只保留固件用得到的字段
        keep = {"key_tap": ("op", "key", "hold_ms"),
                "key_down": ("op", "key"),
                "key_up": ("op", "key"),
                "mouse_move": ("op", "x", "y"),
                "mouse_click": ("op", "button", "x", "y"),
                "stop_all": ("op",)}[cmd.op]
        payload = {k: v for k, v in payload.items() if k in keep}
        self._ser.write((json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8"))
        if not bypass:
            # 尽力读取回显（非阻塞预算内）
            try:
                line = self._ser.readline()
                if line:
                    log.debug("rp2040 ack: %s", line.decode(errors="replace").strip())
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        try:
            self._ser.close()
        except Exception:  # noqa: BLE001
            pass


class RPi5HidgExecutor(BaseExecutor):
    """树莓派5 USB gadget 执行器（写入 /dev/hidg 键盘码 + 绝对鼠标报文）。

    前置：scripts/setup_rpi5_hidg.sh 已配置好 gadget（键盘 hidg0 + 鼠标 hidg1）。
    """

    backend_name = "rpi5_hidg"

    # 产生的 USB HID 用法码映射（常用键子集，够自动化使用）
    _KEYMAP = {
        "a": 0x04, "b": 0x05, "c": 0x06, "d": 0x07, "e": 0x08, "f": 0x09,
        "g": 0x0A, "h": 0x0B, "i": 0x0C, "j": 0x0D, "k": 0x0E, "l": 0x0F,
        "m": 0x10, "n": 0x11, "o": 0x12, "p": 0x13, "q": 0x14, "r": 0x15,
        "s": 0x16, "t": 0x17, "u": 0x18, "v": 0x19, "w": 0x1A, "x": 0x1B,
        "y": 0x1C, "z": 0x1D,
        "1": 0x1E, "2": 0x1F, "3": 0x20, "4": 0x21, "5": 0x22,
        "6": 0x23, "7": 0x24, "8": 0x25, "9": 0x26, "0": 0x27,
        "enter": 0x28, "esc": 0x29, "backspace": 0x2A, "tab": 0x2B,
        "space": 0x2C, "capslock": 0x39,
        "f1": 0x3A, "f2": 0x3B, "f3": 0x3C, "f4": 0x3D, "f5": 0x3E, "f6": 0x3F,
        "f7": 0x40, "f8": 0x41, "f9": 0x42, "f10": 0x43, "f11": 0x44, "f12": 0x45,
        "up": 0x52, "down": 0x51, "left": 0x50, "right": 0x4F,
    }

    def __init__(self, cfg: ExecutorConfig) -> None:
        super().__init__(cfg)
        self._kb = open(cfg.hidg_keyboard, "wb", buffering=0)  # noqa: SIM115
        self._ms = open(cfg.hidg_mouse, "wb", buffering=0)  # noqa: SIM115

    def _emit(self, cmd: Command, bypass: bool = False) -> None:
        if cmd.op == "key_tap":
            code = self._KEYMAP.get((cmd.key or "").lower())
            if code is None:
                raise ValueError(f"未知按键: {cmd.key}")
            hold = _jitter(cmd.hold_ms, 0.15) / 1000.0
            self._kb.write(build_keyboard_report(code))
            time.sleep(hold)
            self._kb.write(build_keyboard_report())
        elif cmd.op in ("key_down", "key_up"):
            code = self._KEYMAP.get((cmd.key or "").lower())
            if code is None:
                raise ValueError(f"未知按键: {cmd.key}")
            if cmd.op == "key_down":
                self._kb.write(build_keyboard_report(code))
            else:
                self._kb.write(build_keyboard_report())
        elif cmd.op == "mouse_move":
            self._mouse_abs(cmd.x or 0, cmd.y or 0)
        elif cmd.op == "mouse_click":
            # 绝对鼠标：按下/抬起报文必须保持同一坐标（否则光标会跳回原点）
            cx = cmd.x if cmd.x is not None else 0
            cy = cmd.y if cmd.y is not None else 0
            self._ms.write(build_abs_mouse_report(cx, cy, self.cfg.screen_w, self.cfg.screen_h))
            time.sleep(0.03)
            btn = 0x01 if cmd.button == "left" else 0x02
            self._ms.write(build_abs_mouse_report(cx, cy, self.cfg.screen_w,
                                                  self.cfg.screen_h, button=btn))
            time.sleep(0.05)
            self._ms.write(build_abs_mouse_report(cx, cy, self.cfg.screen_w, self.cfg.screen_h))
        elif cmd.op == "stop_all":
            self._kb.write(build_keyboard_report())
            self._ms.write(build_abs_mouse_report(0, 0, self.cfg.screen_w, self.cfg.screen_h))

    def _mouse_abs(self, x: int, y: int) -> None:
        self._ms.write(build_abs_mouse_report(x, y, self.cfg.screen_w, self.cfg.screen_h))

    def close(self) -> None:
        for f in (self._kb, self._ms):
            try:
                f.close()
            except Exception:  # noqa: BLE001
                pass


def build_executor(cfg: ExecutorConfig) -> BaseExecutor:
    if cfg.backend == "rp2040":
        return RP2040Executor(cfg)
    if cfg.backend == "rpi5_hidg":
        return RPi5HidgExecutor(cfg)
    if cfg.backend == "pca9685":
        return Pca9685Executor(cfg)
    return DryRunExecutor(cfg)
