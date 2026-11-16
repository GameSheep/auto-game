"""采集代理客户端 —— 连接服务端，循环上传 JPEG 帧。"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import cv2
import numpy as np
import websockets

from ..config import AppConfig
from ..protocol import pack_jpeg_frame, unpack_message
from .capture import build_capture

log = logging.getLogger(__name__)


class AgentClient:
    """采集代理：capture → JPEG → WebSocket。

    capture 可注入自定义帧源（如模拟器 SimCapture），便于联调测试。
    """

    def __init__(self, cfg: AppConfig, server_url: str, agent_name: str = "pc-agent",
                 capture=None) -> None:  # noqa: ANN001
        self.cfg = cfg
        self.url = server_url
        self.agent_name = agent_name
        self._capture = capture
        self._running = asyncio.Event()
        self._stop = asyncio.Event()
        self._last_hash: np.ndarray | None = None  # 异动触发：上帧感知哈希
        self._last_sent_t: float = 0.0

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        token = self.cfg.server.token or ""
        ws_url = (
            self.url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
            .rstrip("/")
        )
        url = f"{ws_url}/ws/agent?token={token}"
        backoff = self.cfg.capture.reconnect_s
        while not self._stop.is_set():
            try:
                async with websockets.connect(url, max_size=2**22, ping_interval=10) as ws:
                    log.info("已连接服务端 %s", self.url)
                    backoff = self.cfg.capture.reconnect_s
                    await self._session(ws)
            except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
                log.warning("连接断开(%s)，%.1fs 后重连", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, 30)
            except Exception:  # noqa: BLE001
                log.exception("代理异常")
                await asyncio.sleep(backoff)

    async def _session(self, ws) -> None:  # noqa: ANN001
        cap = self._capture or build_capture(
            self.cfg.capture.backend,
            self.cfg.capture.region,
            self.cfg.capture.monitor,
        )
        self._reset_motion_state()  # 重连恢复：重连后首帧必发
        try:
            # hello
            hello = json.dumps(
                {"type": "hello", "agent": self.agent_name, "version": "0.1.0"}
            )
            await ws.send(hello)
            # 接收服务端消息（配置更新等）的后台任务
            rx_task = asyncio.create_task(self._rx_loop(ws))
            try:
                interval = 1.0 / max(0.5, self.cfg.capture.fps)
                last = 0.0
                while not self._stop.is_set():
                    now = time.monotonic()
                    if now - last < interval:
                        await asyncio.sleep(interval - (now - last))
                    last = time.monotonic()
                    frame = cap.grab()
                    if frame is None:
                        continue
                    await self._send_frame(ws, frame)
            finally:
                rx_task.cancel()
        finally:
            if self._capture is None:
                cap.close()

    async def _rx_loop(self, ws) -> None:  # noqa: ANN001
        """处理服务端下发消息（非致命，断连时静默退出）。"""
        try:
            async for reply in ws:
                kind, payload = unpack_message(reply)
                if kind == "json" and payload.get("type") == "set_capture":
                    self._apply_capture_cfg(payload)
        except (websockets.ConnectionClosed, asyncio.CancelledError):
            pass

    def _apply_capture_cfg(self, payload: dict) -> None:
        fps = payload.get("fps")
        q = payload.get("jpeg_quality")
        if fps:
            self.cfg.capture.fps = float(fps)
        if q:
            self.cfg.capture.jpeg_quality = int(q)
        log.info("采集参数更新 fps=%s q=%s", fps, q)

    def _reset_motion_state(self) -> None:
        """重连恢复：清空异动哈希（静止场景重连后首帧必发）。"""
        self._last_hash = None

    def should_upload(self, frame: np.ndarray) -> bool:
        """异动触发判定（对话定稿：发现异动才传图）。

        感知哈希比对：静止场景跳帧；每 motion_force_s 强制保活一帧。
        未启用异动触发时恒为 True。
        """
        if not self.cfg.capture.motion_trigger:
            return True
        small = cv2.resize(frame, (16, 16))
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        h = g > g.mean()
        now_t = time.monotonic()
        if self._last_hash is not None:
            dist = int((h != self._last_hash).sum())
            if dist <= self.cfg.capture.motion_hash_thresh \
                    and now_t - self._last_sent_t < self.cfg.capture.motion_force_s:
                return False  # 无异动，跳帧
        self._last_hash = h
        self._last_sent_t = now_t
        return True

    async def _send_frame(self, ws, frame: np.ndarray) -> None:  # noqa: ANN001
        if not self.should_upload(frame):
            return
        h_, w_ = frame.shape[:2]
        ok, buf = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.capture.jpeg_quality)],
        )
        if not ok:
            return
        ts_ms = int(time.time() * 1000) & 0xFFFFFFFF
        await ws.send(pack_jpeg_frame(buf.tobytes(), w_, h_, ts_ms))
