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
import contextlib
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
    actuates = False

    def _emit(self, cmd: Command, bypass: bool = False) -> None:
        n = len(self.last_path)
        suffix = f" path={n}pts" if n > 1 else ""
        log.info("[dryrun] %s%s", cmd.model_dump_json(exclude_none=True), suffix)

    def probe(self) -> dict:
        # 干跑"永远通"，所以这里刻意写清"通了也不能证明真机能动"：
        # 自检的结论要是"可以上真机"，就必须是某个真的会动手的后端回了活。
        return {"ok": True, "backend": self.backend_name, "actuates": False,
                "detail": "dryrun：只记账不下发，这台机器不会真按键盘",
                "hint": "真要上真机，把 executor.backend 改成 rp2040 / rpi5_hidg / pca9685"}


class RP2040Executor(BaseExecutor):
    """RP2040 USB 串口执行器（组合HID 固件见 firmware/rp2040/hid_bridge.ino）。

    协议：JSON 行（\n 结尾），固件回显 {"ok":true,...}。
    """

    backend_name = "rp2040"

    def __init__(self, cfg: ExecutorConfig) -> None:
        super().__init__(cfg)
        import serial  # pyserial

        self._ser = serial.Serial(cfg.serial_port, cfg.serial_baud, timeout=0.5)
        #: 固件回过 {"ok":false} 的次数。**必须能被看见**：
        #: 其中 `watchdog_release` 的意思是"主机卡了 5 秒，按键是固件自己松的" ——
        #: 这种事故如果只进 debug 日志，表现就是"偶尔莫名其妙少走一步"。
        self.ack_errors: list[dict] = []
        log.info("RP2040 串口已连接: %s @%d", cfg.serial_port, cfg.serial_baud)

    def _drain(self) -> None:
        """非阻塞清空串口回显缓冲。

        轨迹模式下一条命令会下发几十行，固件每行都回 ack；
        主机不读，RP2040 的 CDC TX FIFO 会写满并**卡住固件**。
        """
        try:
            n = getattr(self._ser, "in_waiting", 0)
            if n:
                self._ser.read(n)
        except Exception:  # noqa: BLE001
            pass

    def _to_abs(self, x: int, y: int) -> tuple[int, int]:
        """像素 → 固件要的 0..32767 绝对尺度。

        **这里曾经少了一步，后果是鼠标完全不能用。** 固件的鼠标描述符声明的是
        绝对坐标、Logical Max 32767，它对收到的数字不做任何换算。主机如果直接
        把像素坐标发过去（比如 1280×720 画面里的 x=640），光标只会走到屏幕
        2% 的位置 —— 而主机侧一切"成功"，日志里看不出任何异常。

        换算公式和 hidg 后端的 `build_abs_mouse_report` 一致：同一份像素坐标
        在两个后端上必须落到屏幕的同一个地方，否则换后端就等于换了一套标定。
        """
        w = max(1, int(getattr(self.cfg, "screen_w", 1920)))
        h = max(1, int(getattr(self.cfg, "screen_h", 1080)))
        # 分母用 w-1 与 build_abs_mouse_report 保持一致（那个函数也这么算）：
        # 两边公式必须逐位相同，否则同一份像素坐标在两个后端会差几个像素。
        ax = max(0, min(32767, int(x * 32767 / max(1, w - 1))))
        ay = max(0, min(32767, int(y * 32767 / max(1, h - 1))))
        return ax, ay

    def _write(self, payload: dict, ack: bool = True) -> None:
        # 唯一的出口：坐标换算放这里，轨迹中间点、落点、无轨迹的直点全都能覆盖到。
        # 放在各个 _emit 分支里做的话，总有一条路径会被漏掉（这里就漏过一次）。
        if payload.get("op") in ("mouse_move", "mouse_click") and payload.get("x") is not None:
            ax, ay = self._to_abs(int(payload["x"]), int(payload.get("y") or 0))
            payload = {**payload, "x": ax, "y": ay}

        self._ser.write((json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8"))
        if not ack:
            self._drain()
            return
        try:
            line = self._ser.readline()
            if line:
                txt = line.decode(errors="replace").strip()
                log.debug("rp2040 ack: %s", txt)
                if '"ok":false' in txt.replace(" ", ""):
                    self.ack_errors = (self.ack_errors +
                                       [{"op": payload.get("op", ""), "ack": txt}])[-20:]
                    log.warning("RP2040 拒了这条命令（%s）：%s", payload.get("op", ""), txt)
        except Exception:  # noqa: BLE001
            pass

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

        if cmd.op in ("mouse_move", "mouse_click") and self.last_path:
            # 沿人类化轨迹逐点下发；中间点不等回显（串口 115200 下等回显会拖成幻灯片）
            for p in self.last_path[:-1]:
                if p.dt > 0:
                    time.sleep(p.dt)
                self._write({"op": "mouse_move", "x": p.x, "y": p.y}, ack=False)
            last = self.last_path[-1]
            if last.dt > 0:
                time.sleep(last.dt)
            if cmd.op == "mouse_move":
                self._write({"op": "mouse_move", "x": last.x, "y": last.y},
                            ack=not bypass)
            else:
                # 点击 = 悬停(主机侧) → 按下 → 保持(固件 delay) → 抬起
                if self.click_hover_ms > 0:
                    time.sleep(self.click_hover_ms / 1000.0)
                self._write({"op": "mouse_click", "button": cmd.button,
                             "x": last.x, "y": last.y, "hold_ms": self.click_press_ms},
                            ack=not bypass)
            return

        if cmd.op == "mouse_click":
            payload["hold_ms"] = self.click_press_ms

        self._write(payload, ack=not bypass)

    def close(self) -> None:
        try:
            self._ser.close()
        except Exception:  # noqa: BLE001
            pass

    def _ask(self, payload: dict, timeout_s: float = 1.5) -> str:
        """发一行、等一行，把回显原样返回（空串 = 对面没吭声）。

        和 `_write` 分开是因为探活要**看见**回显内容：`_write` 只关心
        `ok:false`，而探活要区分"回了 unknown op"和"一个字都没回"。
        """
        try:
            self._ser.timeout = timeout_s
        except Exception:  # noqa: BLE001
            pass
        try:
            # 先清干净：上一次被 kill 时可能留了半行，读到它会把"没回"误判成"回了"
            n = getattr(self._ser, "in_waiting", 0)
            if n:
                self._ser.read(n)
            self._ser.write((json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8"))
            # flush 失败只说明"缓冲没刷完"，不代表串口不通 —— 别让它冒充连接故障
            with contextlib.suppress(Exception):
                self._ser.flush()
            line = self._ser.readline()
            return line.decode(errors="replace").strip() if line else ""
        except Exception as e:  # noqa: BLE001
            return f"__error__{type(e).__name__}: {e}"

    def probe(self, timeout_s: float = 1.5) -> dict:
        """发 ping，看对面到底是谁。

        三种结果对应三种完全不同的修法，所以不能合成一个"不通"：

        | 回显 | 结论 | 该做什么 |
        |---|---|---|
        | `{"ok":true,"fw":"1.1.0"}` | 就是这套固件 | 可以跑 |
        | `{"ok":false,"error":"unknown op"}` | 是这套固件，但版本旧 | 重刷固件 |
        | 空 / 别的 | 对面不是这套固件，或卡死 | 查串口号、查线、查板子 |

        `ping` 不碰键盘鼠标 —— 探活本身不能产生输入。
        """
        base = {"backend": self.backend_name, "actuates": True}
        port = getattr(self.cfg, "serial_port", "?")
        txt = self._ask({"op": "ping"}, timeout_s=timeout_s)
        if txt.startswith("__error__"):
            return {**base, "ok": False,
                    "detail": f"{port} 打不通：{txt[len('__error__'):]}",
                    "hint": f"串口被别的程序占着？{port} 没插好？驱动没装？"}
        if not txt:
            return {**base, "ok": False,
                    "detail": f"{port} 没回话（{timeout_s:.1f} 秒内一个字都没有）",
                    "hint": "对面可能不是这块板子，或者固件卡死了；先拔插一次 USB，"
                            "再确认串口号（Windows 看设备管理器，Linux 看 /dev/ttyACM*）"}
        if '"ok":false' in txt.replace(" ", ""):
            if "unknown op" in txt:
                return {**base, "ok": False,
                        "detail": f"{port} 对面的固件太旧（不认识 ping）：{txt[:80]}",
                        "hint": "板子上是旧固件：重刷 firmware/rp2040/hid_bridge/hid_bridge.ino"}
            return {**base, "ok": False,
                    "detail": f"{port} 对面拒了探活：{txt[:80]}",
                    "hint": "这条回显不是这套协议的，确认串口没有被别的设备占用"}
        try:
            got = json.loads(txt)
        except Exception:  # noqa: BLE001
            return {**base, "ok": False,
                    "detail": f"{port} 回的不是 JSON：{txt[:80]}",
                    "hint": "对面是别的串口设备（波特率/串口号配错了）"}
        fw = str(got.get("fw") or "")
        up = got.get("up_ms")
        return {**base, "ok": bool(got.get("ok")), "fw": fw,
                "detail": f"{port} 在线，固件 {fw or '未知'}"
                          + (f"，已运行 {int(up) / 1000:.0f} 秒" if isinstance(up, (int, float)) else ""),
                "hint": "" if got.get("ok") else "固件回了 ping 但 ok 为假，看上面 detail",
                "watchdog_ms": got.get("watchdog_ms")}


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
            if self.last_path:
                self._walk_path(self.last_path, self._mouse_abs)
            else:
                self._mouse_abs(cmd.x or 0, cmd.y or 0)
        elif cmd.op == "mouse_click":
            # 绝对鼠标：按下/抬起报文必须保持同一坐标（否则光标会跳回原点）
            if self.last_path:
                # 先沿人类化轨迹滑过去（最后一点即落点），再按下
                self._walk_path(self.last_path, self._mouse_abs)
                cx, cy = self.last_path[-1].x, self.last_path[-1].y
            else:
                cx = cmd.x if cmd.x is not None else 0
                cy = cmd.y if cmd.y is not None else 0
                self._mouse_abs(cx, cy)
            # 点击手势：悬停（先看清）→ 按下 → 保持（真人不会 45ms 就松）→ 抬起
            if self.click_hover_ms > 0:
                time.sleep(self.click_hover_ms / 1000.0)
            btn = 0x01 if cmd.button == "left" else 0x02
            self._ms.write(build_abs_mouse_report(cx, cy, self.cfg.screen_w,
                                                  self.cfg.screen_h, button=btn))
            time.sleep(max(0.01, self.click_press_ms / 1000.0))
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

    def probe(self) -> dict:
        """hidg 没有"回话"这一层，能做的检查就是节点还在、还能写。

        节点消失 = gadget 掉了（拔线、重启、`setup_rpi5_hidg.sh` 没跑完）。
        注意：`open()` 成功只证明当时能开，设备拔了之后句柄还"有效" ——
        所以这里看的是路径本身，而不是句柄。
        """
        import os

        base = {"backend": self.backend_name, "actuates": True}
        paths = [str(getattr(self.cfg, "hidg_keyboard", "")),
                 str(getattr(self.cfg, "hidg_mouse", ""))]
        bad = [p for p in paths if not p or not os.path.exists(p)]
        if bad:
            return {**base, "ok": False,
                    "detail": f"hidg 节点不在：{', '.join(x or '(空)' for x in bad)}",
                    "hint": "跑 scripts/setup_rpi5_hidg.sh 配好 gadget；"
                            "已经配过就确认数据线插在支持 OTG 的口上"}
        ro = [p for p in paths if not os.access(p, os.W_OK)]
        if ro:
            return {**base, "ok": False,
                    "detail": f"hidg 节点不可写：{', '.join(ro)}",
                    "hint": "权限问题：`sudo chmod 666 /dev/hidg*`，或把用户加进相应组"}
        return {**base, "ok": True, "detail": f"{paths[0]} / {paths[1]} 都好，能写"}


def build_executor(cfg: ExecutorConfig) -> BaseExecutor:
    if cfg.backend == "rp2040":
        return RP2040Executor(cfg)
    if cfg.backend == "rpi5_hidg":
        return RPi5HidgExecutor(cfg)
    if cfg.backend == "pca9685":
        return Pca9685Executor(cfg)
    return DryRunExecutor(cfg)
