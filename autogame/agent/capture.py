"""采集后端 —— 游戏画面获取。

优先级（对话定稿）：
  1. dxcam  (Desktop Duplication API, GPU 直出, 极低延迟) —— pip install dxcam
  2. mss    (通用屏幕捕获兜底, 纯 CPU)

安全原则：只读屏幕画面，绝不注入游戏进程、绝不读内存。
"""

from __future__ import annotations

import abc
import logging

import numpy as np

log = logging.getLogger(__name__)


class BaseCapture(abc.ABC):
    """采集接口：grab() 返回 BGR np.ndarray (H,W,3) uint8。"""

    name = "base"

    @abc.abstractmethod
    def grab(self) -> np.ndarray | None: ...

    def close(self) -> None:
        pass


class MsseCapture(BaseCapture):
    """mss 屏幕捕获（跨平台兜底）。"""

    name = "mss"

    def __init__(self, region: tuple[int, int, int, int] | None, monitor: int = 1) -> None:
        import mss

        self._mss = mss
        self._sct = mss.mss()
        if region:
            left, top, w, h = region
            self._bbox = {"left": left, "top": top, "width": w, "height": h}
        else:
            mon = self._sct.monitors[min(monitor, len(self._sct.monitors) - 1)]
            self._bbox = {"left": mon["left"], "top": mon["top"],
                          "width": mon["width"], "height": mon["height"]}
        log.info("采集后端 mss 区域: %s", self._bbox)

    def grab(self) -> np.ndarray | None:
        img = self._sct.grab(self._bbox)
        # mss 返回 BGRA → 转 BGR
        frame = np.asarray(img)[:, :, :3]
        return np.ascontiguousarray(frame)


class DxcamCapture(BaseCapture):
    """dxcam 捕获（Windows, DXGI 桌面复制, GPU 加速）。"""

    name = "dxcam"

    def __init__(self, region: tuple[int, int, int, int] | None, monitor: int = 1) -> None:
        import dxcam

        device_idx = max(0, monitor - 1)
        self._cam = dxcam.create(device_idx=device_idx, output_idx=device_idx,
                                 output_color="BGR")
        if self._cam is None:
            raise RuntimeError("dxcam 初始化失败")
        self._region = tuple(region) if region else None
        self._last: np.ndarray | None = None
        log.info("采集后端 dxcam 区域: %s", self._region or "全屏")

    def grab(self) -> np.ndarray | None:
        frame = self._cam.grab(region=self._region)
        if frame is None:  # 无新帧（dxcam 特性）
            return self._last
        self._last = frame
        return frame

    def close(self) -> None:
        try:
            self._cam.release()
        except Exception:  # noqa: BLE001
            pass


def build_capture(backend: str, region: list[int] | None, monitor: int = 1):
    """按配置构建采集器，auto 模式优先 dxcam。"""
    reg = tuple(region) if region else None
    if backend in ("dxcam", "auto"):
        try:
            return DxcamCapture(reg, monitor)
        except Exception as e:  # noqa: BLE001
            if backend == "dxcam":
                raise
            log.info("dxcam 不可用(%s)，回退 mss", e)
    return MsseCapture(reg, monitor)
