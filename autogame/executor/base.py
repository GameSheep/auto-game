"""执行器基类（独立模块，避免循环导入）。"""

from __future__ import annotations

import abc
import logging
import random
import threading
import time
from typing import Callable, Sequence

from ..config import ExecutorConfig
from ..humanize import PathPoint, build_mouse_path
from ..protocol import Command
from ..security import ActionRateLimiter

log = logging.getLogger(__name__)


def _jitter(value_ms: int, ratio: float) -> int:
    """按比例随机抖动（工程可靠性：避免机械等间隔输出）。"""
    if ratio <= 0:
        return value_ms
    delta = max(1, int(value_ms * ratio))
    return max(1, value_ms + random.randint(-delta, delta))


def _rand_in(pair: Sequence[int] | None, default: tuple[int, int]) -> int:
    """从 [lo, hi] 区间取随机整数（区间缺失/非法时回落到 default）。"""
    lo, hi = default
    if pair:
        vals = list(pair)[:2]
        if len(vals) == 2:
            lo, hi = vals
    lo, hi = int(min(lo, hi)), int(max(lo, hi))
    return random.randint(max(0, lo), max(0, hi))


class BaseExecutor(abc.ABC):
    """执行器基类：限速 + 急停 + 统计 + 人类化鼠标轨迹。"""

    #: 后端是否支持绝对坐标鼠标移动（机械舵机按压后端为 False，走位需配 keys 模式）
    supports_mouse_move: bool = True

    def __init__(self, cfg: ExecutorConfig) -> None:
        self.cfg = cfg
        self._limiter = ActionRateLimiter(cfg.min_action_interval_ms, cfg.max_actions_per_sec)
        self._rate_sig = (cfg.min_action_interval_ms, cfg.max_actions_per_sec)
        self._panic = threading.Event()
        self.sent = 0
        self.dropped = 0
        self.journal: list[dict] = []  # 最近动作日志（仪表盘/测试断言用）
        self._lock = threading.Lock()
        # 光标位置：作为下一次轨迹的起点（None=未知，首帧退化为单点）
        self.cursor: tuple[int, int] | None = None
        self.last_path: list[PathPoint] = []
        self.path_points_total = 0
        # 本次点击手势参数（execute 里定，后端只管执行）
        self.click_hover_ms = 0
        self.click_press_ms = 0
        #: 动作闸门（观察/暂停模式）。返回 False 时动作一律不下发，但照常计数。
        #: None = 总是放行（没装助手的时候就是这个状态）。
        self.action_gate: Callable[[], bool] | None = None
        #: 被闸门拦下的动作数。页面上要能看出"它想动但没动"，否则观察模式
        #: 看起来和"什么都不做"一模一样。
        self.suppressed = 0

    #: 后端名（子类覆盖）。放基类是为了探活的默认失败信息能说清是哪个后端。
    backend_name: str = "base"

    #: 这个后端会不会真的动手（dryrun = 不会）。自检要靠它判断"这台机器上
    #: 到底有没有人在按键盘"，否则联调通过 = 真机上没验证，是最常见的翻车方式。
    actuates: bool = True

    def probe(self) -> dict:
        """探活：**这台机器上动作能不能真的发出去**。

        返回 `{ok, backend, actuates, detail, hint}`。`ok=False` 的意思是
        "现在上真机，动作会石沉大海"，`hint` 给出具体该查什么 —— 比把
        `SerialException` 原样抛给人有用得多。

        默认实现刻意返回失败而不是 True：新加的后端如果忘了实现探活，
        自检应该拦下来（或至少说"这个后端没有探活"），不能默默放行。
        """
        return {"ok": False, "backend": self.backend_name, "actuates": self.actuates,
                "detail": f"{self.backend_name} 还没实现探活",
                "hint": "这个后端没有探活接口，无法确认动作能不能发出去"}

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
        if cmd.op != "stop_all" and self.action_gate is not None \
                and not self.action_gate():
            # 观察/暂停模式：想动，但不动。丢掉的计数要留着 —— 复盘时要能说
            # "我当时想做 A 但被暂停了"，而不是看起来整段什么都没发生。
            self.suppressed += 1
            with self._lock:
                self.journal.append(
                    {"ts": time.time(), "op": cmd.op, "key": cmd.key,
                     "x": cmd.x, "y": cmd.y, "button": cmd.button,
                     "hold_ms": cmd.hold_ms, "path": 0, "press_ms": 0,
                     "suppressed": True})
                if len(self.journal) > 500:
                    del self.journal[:-300]
            return False
        if cmd.op != "stop_all":
            # 限速参数每帧跟随 config（配置页热更新后立即生效，不用重启）
            rate = (self.cfg.min_action_interval_ms, self.cfg.max_actions_per_sec)
            if rate != self._rate_sig:
                self._rate_sig = rate
                self._limiter.sync(*rate)
            # 最小间隔：由限速器自己等到可发送（不依赖 sleep 精度，见 acquire 注释）
            if not self._limiter.acquire(0.25):
                self.dropped += 1
                return False
        # 鼠标类命令先规划人类化轨迹（WindMouse），后端按点输出
        self.last_path = self._plan_mouse_path(cmd)
        # 点击手势参数（悬停/按压）在这里定，后端只管执行 —— 日志才能如实记录
        if cmd.op == "mouse_click":
            self.click_hover_ms = _rand_in(self.cfg.mouse_path.hover_ms, (30, 110))
            self.click_press_ms = _rand_in(self.cfg.mouse_path.press_ms, (55, 135))
        else:
            self.click_hover_ms = 0
            self.click_press_ms = 0
        try:
            self._emit(cmd)
        except Exception:  # noqa: BLE001
            log.exception("执行命令失败 op=%s", cmd.op)
            return False
        self.sent += 1
        if self.last_path:
            self.cursor = (self.last_path[-1].x, self.last_path[-1].y)
        with self._lock:
            self.journal.append(
                {"ts": time.time(), "op": cmd.op, "key": cmd.key, "x": cmd.x, "y": cmd.y,
                 "button": cmd.button, "hold_ms": cmd.hold_ms, "path": len(self.last_path),
                 "press_ms": self.click_press_ms}
            )
            if len(self.journal) > 500:
                del self.journal[:-300]
        return True

    # ------------- 人类化鼠标轨迹 -------------

    def _plan_mouse_path(self, cmd: Command) -> list[PathPoint]:
        """为 mouse_move / mouse_click 规划轨迹；其他命令返回空列表。"""
        if cmd.op not in ("mouse_move", "mouse_click"):
            return []
        if cmd.x is None or cmd.y is None:
            return []
        tx, ty = int(cmd.x), int(cmd.y)
        start = self.cursor if self.cursor is not None else (tx, ty)
        path = build_mouse_path(start, (tx, ty), self.cfg.mouse_path)
        self.path_points_total += len(path)
        return path

    def _walk_path(self, path: list[PathPoint], move) -> None:
        """按轨迹逐点输出（move(x, y) 由后端实现），点间按 dt 等待。

        注意：轨迹点在 _emit 内部输出，绕过 ActionRateLimiter ——
        它们属于**同一条**逻辑命令，不应被限速器拆散或丢弃。
        """
        for p in path:
            if p.dt > 0:
                time.sleep(p.dt)
            move(p.x, p.y)

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
