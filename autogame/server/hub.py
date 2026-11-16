"""服务端中枢 —— 代理会话、最新帧缓冲、场景状态、决策循环。"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from collections import deque
from typing import Any
from pathlib import Path

import cv2
import numpy as np

from ..config import AppConfig
from ..decision.fsm import DecisionEngine, State
from ..executor import BaseExecutor
from ..protocol import SceneState
from ..vision.pipeline import VisionPipeline

log = logging.getLogger(__name__)


class Hub:
    """单代理会话中枢（一个服务端服务一个游戏 PC）。"""

    def __init__(self, cfg: AppConfig, executor: BaseExecutor, pipeline: VisionPipeline) -> None:
        self.cfg = cfg
        self.executor = executor
        self.pipeline = pipeline
        self.fsm = DecisionEngine(cfg.decision, executor)

        self._latest_jpeg: bytes | None = None
        self._latest_scene: SceneState | None = None
        self._latest_ts: float = 0.0
        self._frame_seq: int = 0
        self._processed_seq: int = 0
        self._history: deque[dict] = deque(maxlen=cfg.server.history_size)
        self._lock = threading.Lock()
        self._decide_event = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        # 事件持久化（JSONL 审计日志）：FSM 事件回调直写，无索引、不因内存裁剪丢事件
        self._event_log_path = cfg.server.event_log
        self._event_log_warned = False
        if self._event_log_path:
            self.fsm.on_event = self._append_event_log
        # 当前代理连接（新连接踢旧连接，防断线风暴下双路帧交错）
        self.agent_ws = None
        # UI 任务（交易行/装备/天赋/技能 —— 配方引擎，激活时接管决策循环）
        self.ui_task: Any | None = None
        self.ui_task_ends: deque = deque(maxlen=20)  # 已结束任务的完成记录

    def _emit_event_log(self, event: dict) -> None:
        event = {"ts": time.time(), **event}
        if self._event_log_path:
            self._append_event_log(event)

    def _append_event_log(self, event: dict) -> None:
        try:
            path = Path(self._event_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and \
                    path.stat().st_size > self.cfg.server.event_log_max_mb * 1024 * 1024:
                os.replace(path, path.with_suffix(path.suffix + ".1"))
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        except Exception:  # noqa: BLE001
            if not self._event_log_warned:
                log.warning("事件日志写入失败（已降级为仅内存）", exc_info=True)
                self._event_log_warned = True

    def _drain_event_log(self) -> None:
        """兼容保留：事件已通过 on_event 回调直写，无需再排空。"""
        return

    # ------------- 代理帧入口（WS handler 线程调用） -------------

    def on_frame(self, jpeg: bytes, ts_ms: int) -> dict:
        """收到代理帧：解码 → 视觉 → 缓存 → 唤醒决策循环。返回摘要。"""
        buf = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("jpeg 解码失败")

        scene = self.pipeline.process(frame, ts_ms)
        with self._lock:
            self._latest_jpeg = jpeg
            self._latest_scene = scene
            self._latest_ts = time.time()
            self._frame_seq += 1
            seq = self._frame_seq
        self._history.append(
            {"seq": seq, "ts": scene.ts_ms, "pillars": len(scene.pillars),
             "aframe": scene.aframe is not None, "state": self.fsm.state.value}
        )
        # 唤醒决策
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._decide_event.set)
        return {
            "seq": seq,
            "pillars": len(scene.pillars),
            "aframe": scene.aframe is not None,
            "state": self.fsm.state.value,
        }

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def decision_loop(self) -> None:
        """决策循环：每个新帧 tick 一次 FSM（默认 1Hz 由代理 fps 决定）。

        FSM 在线程池中执行：串口/HID 输出可能阻塞数百毫秒，不能卡事件循环。
        """
        while True:
            await self._decide_event.wait()
            self._decide_event.clear()
            with self._lock:
                scene = self._latest_scene
            if scene is None:
                continue
            loop = asyncio.get_running_loop()
            try:
                if self.ui_task is not None:
                    # UI 任务激活：配方引擎接管（FSM 暂停），解码最新帧推进
                    task = self.ui_task
                    with self._lock:
                        jpeg = self._latest_jpeg
                    if jpeg is not None:
                        buf = np.frombuffer(jpeg, dtype=np.uint8)
                        fimg = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                        if fimg is not None:
                            await loop.run_in_executor(None, task.tick, fimg, scene.ts_ms)
                    if task.done or task.failed:
                        end = {"event": "ui_task_end", "name": task.name,
                               "failed": task.failed, "step": task.step_i,
                               "total_steps": len(task.steps)}
                        self.ui_task_ends.append(end)
                        self._emit_event_log(end)
                        self.ui_task = None
                else:
                    await loop.run_in_executor(None, self.fsm.on_scene, scene)
            except Exception:  # noqa: BLE001
                log.exception("决策 tick 异常")
            finally:
                self._drain_event_log()

    # ------------- 仪表盘数据 -------------

    def snapshot(self) -> dict:
        with self._lock:
            scene = self._latest_scene
            jpeg = self._latest_jpeg
            seq = self._frame_seq
        d = {
            "state": self.fsm.state.value,
            "frame_seq": seq,
            "latest_ts": self._latest_ts,
            "executor": {
                "backend": getattr(self.executor, "backend_name", self.cfg.executor.backend),
                "sent": self.executor.sent,
                "dropped": self.executor.dropped,
                "panicked": self.executor.panicked,
            },
            "stats": dict(self.fsm.stats),
            "scene": scene.model_dump() if scene else None,
            "events": self.fsm.events[-40:],
            "ui_task_ends": list(self.ui_task_ends),
            "history_len": len(self._history),
        }
        return d

    def annotated_jpeg(self, quality: int | None = None) -> bytes | None:
        """带检测标注的 JPEG（仪表盘用）。"""
        with self._lock:
            scene = self._latest_scene
            jpeg = self._latest_jpeg
        if jpeg is None:
            return None
        buf = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        if scene is not None:
            scale_x = frame.shape[1] / max(1, scene.w)
            scale_y = frame.shape[0] / max(1, scene.h)
            for p in scene.pillars:
                x, y = int(p.cx * scale_x), int(p.cy * scale_y)
                w, h = int(max(12, p.w * scale_x)), int(max(12, p.h * scale_y))
                color = (0, 215, 255) if p.color == "gold" else (255, 105, 180)
                cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, 2)
                cv2.putText(frame, f"{p.color}:{round(p.distance)}px",
                            (x - w // 2, max(12, y - h // 2 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            if scene.aframe:
                a = scene.aframe
                ax, ay = int(a.x * scale_x), int(a.y * scale_y)
                aw, ah = int(max(10, a.w * scale_x)), int(max(10, a.h * scale_y))
                cv2.rectangle(frame, (ax - aw // 2, ay - ah // 2),
                              (ax + aw // 2, ay + ah // 2), (0, 255, 0), 2)
                cv2.putText(frame, "A-FRAME", (ax - aw // 2, max(12, ay - ah // 2 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if scene.rechallenge:
                rc = scene.rechallenge
                rx, ry = int(rc.x * scale_x), int(rc.y * scale_y)
                rw, rh = int(rc.w * scale_x), int(rc.h * scale_y)
                cv2.rectangle(frame, (rx - rw // 2, ry - rh // 2),
                              (rx + rw // 2, ry + rh // 2), (0, 128, 255), 2)
                cv2.putText(frame, "RECHALLENGE", (rx - rw // 2, max(12, ry - rh // 2 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1)
            ch = scene.character
            cv2.drawMarker(frame, (int(ch[0] * scale_x), int(ch[1] * scale_y)),
                           (255, 255, 255), cv2.MARKER_CROSS, 24, 1)
        st = self.fsm.state.value
        cv2.putText(frame, f"FSM:{st}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255) if st == "APPROACHING" else (200, 200, 200), 2)
        q = quality or self.cfg.server.dash_jpeg_quality
        ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
        return enc.tobytes() if ok else None

    # ------------- 控制命令（REST/WS 调用） -------------

    def control(self, action: str) -> dict:
        if action == "start":
            self.fsm.start()
        elif action == "stop":
            self.fsm.stop()
        elif action == "panic":
            self.fsm.panic()
        elif action == "reset":
            self.fsm.reset_panic()
        else:
            raise ValueError(f"未知 action: {action}")
        return {"ok": True, "state": self.fsm.state.value}
