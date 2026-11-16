"""执行器基类（独立模块，避免循环导入）。"""

from __future__ import annotations

import abc
import logging
import random
import threading
import time

from ..config import ExecutorConfig
from ..protocol import Command
from ..security import ActionRateLimiter

log = logging.getLogger(__name__)


def _jitter(value_ms: int, ratio: float) -> int:
    """按比例随机抖动（工程可靠性：避免机械等间隔输出）。"""
    if ratio <= 0:
        return value_ms
    delta = max(1, int(value_ms * ratio))
    return max(1, value_ms + random.randint(-delta, delta))


class BaseExecutor(abc.ABC):
    """执行器基类：限速 + 急停 + 统计。"""

    def __init__(self, cfg: ExecutorConfig) -> None:
        self.cfg = cfg
        self._limiter = ActionRateLimiter(cfg.min_action_interval_ms, cfg.max_actions_per_sec)
        self._panic = threading.Event()
        self.sent = 0
        self.dropped = 0
        self.journal: list[dict] = []  # 最近动作日志（仪表盘/测试断言用）
        self._lock = threading.Lock()

    @property
    def panicked(self) -> bool:
        return self._panic.is_set()

    def panic(self) -> None:
        """急停：立刻置位并输出 stop_all。"""
        self._panic.set()
        try:
            self._emit(Command(op="stop_all"), bypass=True)
        except Exception:  # noqa: BLE001
            log.exception("急停输出失败")
        log.warning("执行器已急停")

    def reset(self) -> None:
        self._panic.clear()
        self._limiter.reset()
        log.info("执行器复位")

    def execute(self, cmd: Command) -> bool:
        if self._panic.is_set() and cmd.op != "stop_all":
            self.dropped += 1
            return False
        if cmd.op != "stop_all":
            # 最小间隔：自适应等待（上限50ms），保证连续合法命令不被丢弃
            delay = self._limiter.delay_needed()
            if delay > 0:
                time.sleep(min(delay, 0.05))
            if not self._limiter.allow():
                self.dropped += 1
                return False
        try:
            self._emit(cmd)
        except Exception:  # noqa: BLE001
            log.exception("执行命令失败 op=%s", cmd.op)
            return False
        self.sent += 1
        with self._lock:
            self.journal.append(
                {"ts": time.time(), "op": cmd.op, "key": cmd.key, "x": cmd.x, "y": cmd.y,
                 "button": cmd.button, "hold_ms": cmd.hold_ms}
            )
            if len(self.journal) > 500:
                del self.journal[:-300]
        return True

    @abc.abstractmethod
    def _emit(self, cmd: Command, bypass: bool = False) -> None:
        """后端实际输出（绕过统计的急停路径由 panic 调用）。"""

    def close(self) -> None:
        pass




# ---------------- HID 报文构建（纯函数，字节级可测） ----------------

# /dev/hidg1 绝对鼠标报文：6 字节 [btn, x_lo, x_hi, y_lo, y_hi, wheel]
# 描述符见 scripts/setup_rpi5_hidg.sh（8按钮 + 16bit X/Y(1..32767) + 滚轮）
def build_abs_mouse_report(x: int, y: int, screen_w: int, screen_h: int,
                           button: int = 0x00, wheel: int = 0x00) -> bytes:
    ax = max(0, min(32767, int(x * 32767 / max(1, screen_w - 1))))
    ay = max(0, min(32767, int(y * 32767 / max(1, screen_h - 1))))
    return bytes([button & 0xFF, ax & 0xFF, ax >> 8, ay & 0xFF, ay >> 8, wheel & 0xFF])


# /dev/hidg0 标准键盘报文：8 字节 [mod, reserved, k1..k6]
def build_keyboard_report(key_code: int = 0x00, modifier: int = 0x00) -> bytes:
    report = bytearray(8)
    report[0] = modifier & 0xFF
    if key_code:
        report[2] = key_code & 0xFF
    return bytes(report)
