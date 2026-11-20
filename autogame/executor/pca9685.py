"""PCA9685 舵机机械执行器 —— 对话定稿的"机械结构按压键鼠"方案。

原理：PCA9685（I2C 16路PWM, 50Hz）驱动舵机，舵机末端"手指"物理压下键盘按键。
     对话数据：舵机响应 100-300ms，PCA9685 12位分辨率定位精度 0.05°。

通道映射（config.executor.servo_map）：
  每个键/鼠标按钮 → {channel, down_angle, up_angle}
  例: {"a": {"ch": 0, "down": 110, "up": 60}, "w": {...},
       "mouse_left": {"ch": 8, "down": 120, "up": 70}}

局限（对话明确讨论过）：机械方案延迟数百毫秒、无法实现绝对坐标鼠标移动 ——
鼠标移动命令在该后端下不支持（仅按键按压）；需要鼠标走位时用 mouse_click
模式改为"方向键走位"，或混合方案（机械按A + RP2040 鼠标）。

I2C 通过 smbus2 直写 PCA9685 寄存器（无重依赖）；测试用 mock 总线。
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..protocol import Command
from .base import BaseExecutor

log = logging.getLogger(__name__)

# PCA9685 寄存器
_PCA9685_ADDR = 0x40
_MODE1 = 0x00
_SLEEP = 0x10
_WAKE = 0x00
_AI = 0x20  # auto-increment
_PRESCALE = 0xFE
_LED0_ON_L = 0x06
_OSC_CLK = 25_000_000.0  # 内部振荡器


class _Pca9685Bus:
    """smbus2 直写 PCA9685（50Hz PWM，脉冲=angle/180*（max-min)+min）。"""

    def __init__(self, bus_id: int, addr: int, freq_hz: float = 50.0) -> None:
        from smbus2 import SMBus  # 延迟导入

        self._bus = SMBus(bus_id)
        self._addr = addr
        prescale = int(round(_OSC_CLK / (4096 * freq_hz)) - 1)
        old = self._read(_MODE1)
        self._write(_MODE1, (old & ~_WAKE) | _SLEEP)
        self._write(_PRESCALE, prescale)
        self._write(_MODE1, old)
        time.sleep(0.005)
        self._write(_MODE1, old | _AI | 0x80)  # wake + auto-increment + 外部时钟

    def _read(self, reg: int) -> int:
        return int(self._bus.read_byte_data(self._addr, reg))

    def _write(self, reg: int, val: int) -> None:
        self._bus.write_byte_data(self._addr, reg, val)

    def set_pulse(self, channel: int, pulse_us: float) -> None:
        """设置某通道脉冲宽度（微秒）。"""
        tick = pulse_us * 4096 / (1000000.0 / 50.0)  # 50Hz 一周期 20ms
        on = 0
        off = max(0, min(4095, int(tick)))
        base = _LED0_ON_L + 4 * channel
        self._write(base, on & 0xFF)
        self._write(base + 1, on >> 8)
        self._write(base + 2, off & 0xFF)
        self._write(base + 3, off >> 8)

    def close(self) -> None:
        try:
            self._bus.close()
        except Exception:  # noqa: BLE001
            pass


class Pca9685Executor(BaseExecutor):
    """舵机机械按压执行器：key_tap/key_down/key_up/mouse_click(按压) 支持。

    mouse_move 不支持（机械结构无绝对定位）—— 走位请配 decision.approach_mode=keys。
    """

    backend_name = "pca9685"
    supports_mouse_move = False  # 机械结构无绝对定位 → 走神/轨迹层自动跳过鼠标噪音

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self._map: dict[str, dict] = dict(cfg.servo_map)
        self._pressed: set[str] = set()
        if cfg.pca9685_mock:
            self._bus = _MockBus()
            log.info("PCA9685 MOCK 总线（测试）")
        else:
            self._bus = _Pca9685Bus(cfg.pca9685_bus, cfg.pca9685_addr)
            log.info("PCA9685 已连接 bus=%d addr=0x%02x", cfg.pca9685_bus, cfg.pca9685_addr)

    def _set_servo(self, name: str, down: bool) -> None:
        m = self._map.get(name)
        if m is None:
            raise ValueError(f"舵机映射缺少 {name}")
        angle = m["down"] if down else m["up"]
        # 0-180° → 500-2500µs 脉冲
        pulse = 500 + (2500 - 500) * angle / 180.0
        self._bus.set_pulse(int(m["ch"]), pulse)

    def _emit(self, cmd: Command, bypass: bool = False) -> None:
        if cmd.op == "key_tap":
            name = (cmd.key or "").lower()
            self._set_servo(name, down=True)
            time.sleep(max(0.08, cmd.hold_ms / 1000.0))  # 机械响应 100ms+
            self._set_servo(name, down=False)
        elif cmd.op == "key_down":
            name = (cmd.key or "").lower()
            self._set_servo(name, down=True)
            self._pressed.add(name)
        elif cmd.op == "key_up":
            name = (cmd.key or "").lower()
            self._set_servo(name, down=False)
            self._pressed.discard(name)
        elif cmd.op == "mouse_click":
            name = f"mouse_{cmd.button}"
            self._set_servo(name, down=True)
            time.sleep(0.1)
            self._set_servo(name, down=False)
        elif cmd.op == "stop_all":
            for name in list(self._pressed):
                self._set_servo(name, down=False)
            self._pressed.clear()
        elif cmd.op == "mouse_move":
            raise ValueError("PCA9685 机械后端不支持鼠标移动（用 approach_mode=keys）")

    def close(self) -> None:
        self._bus.close()

    def probe(self) -> dict:
        """读一次 MODE1 寄存器 —— 能读出来就说明 I2C 上有颗芯片在应答。

        这条检查能挡掉最常见的一种"配好了但没接"：`pca9685_bus` 写错（树莓派上
        I2C 总线号是 1，很多人填 0），或者 SDA/SCL 没接、芯片没供电。这些情况下
        写寄存器不会报错（smbus 是"发出去就算成功"），舵机就是一秒不动，
        而程序里一切正常 —— 只有读回来才能证明对面真的有东西。
        """
        base = {"backend": self.backend_name, "actuates": True}
        read = getattr(self._bus, "_read", None)
        if read is None:
            # mock 总线：连的是字典不是芯片，别让人以为"探活通过 = 舵机会动"
            return {**base, "ok": True, "actuates": False,
                    "detail": "mock 总线（pca9685_mock=true）：不会真的驱动舵机",
                    "hint": "上真机前把 executor.pca9685_mock 设成 false"}
        try:
            mode1 = int(read(_MODE1))
        except Exception as e:  # noqa: BLE001
            bus = getattr(self.cfg, "pca9685_bus", "?")
            return {**base, "ok": False,
                    "detail": f"I2C bus={bus} addr=0x{getattr(self.cfg, 'pca9685_addr', 0):02x} "
                              f"读不到：{type(e).__name__}: {e}",
                    "hint": "确认 `i2cdetect -y 1` 能在 0x40 看到芯片；"
                            "没看到就是接线/供电问题，看到但读不了多半是总线号填错"}
        return {**base, "ok": True,
                "detail": f"I2C 上有应答（MODE1=0x{mode1:02x}），"
                          f"{len(self._map)} 个通道已映射"}


class _MockBus:
    """测试用 mock PCA9685 总线。"""

    def __init__(self) -> None:
        self.pulses: dict[int, float] = []

    def set_pulse(self, channel: int, pulse_us: float) -> None:
        self.pulses.append((channel, pulse_us))

    def close(self) -> None:
        pass
