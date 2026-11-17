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
from .notify import Notifier

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
        # 视觉模型顾问需要原始帧（卡壳时截图问多模态模型）
        self.fsm.frame_source = self._latest_jpeg_bytes
        # 帧流看门狗：代理掉线/游戏卡死时让"卡住"这件事在仪表盘可见
        self._stall_emitted = False
        self._stall_gap = 0.0
        # 事件持久化（JSONL 审计日志）：FSM 事件回调直写，无索引、不因内存裁剪丢事件
        self._event_log_path = cfg.server.event_log
        self._event_log_warned = False
        if self._event_log_path:
            self.fsm.on_event = self._append_event_log
        # 当前代理连接（新连接踢旧连接，防断线风暴下双路帧交错）
        self.agent_ws = None
        # 视觉看门狗 + 告警（"看不到游戏就停" / "刷够了提醒我"）
        self.notifier = Notifier(cfg.notify)
        self._started_at: float = 0.0          # 本轮运行开始时刻
        self._vision_lost_since: float = 0.0   # 连续看不到游戏的起点
        self._vision_lost_reason = ""
        self._auto_stopped = False
        self._manual_needed = 0                # 需要人工判断的次数
        self._last_pick_ts: float = 0.0        # 最近一次成功拾取
        self._last_pick_count: int = 0
        self._no_pick_since: float = 0.0
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

    def _latest_jpeg_bytes(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    # ------------- 帧流看门狗 -------------

    def check_frame_stall(self) -> None:
        """运行中长时间没有新帧 → 记事件（代理掉线/游戏卡死）。

        这里**不产生任何输入**：没有画面就没有决策依据，宁可不动也不盲按。
        作用是让"卡住"这件事在仪表盘/审计日志里立刻可见，而不是静默假运行。
        """
        auto = self.cfg.decision.autonomy
        if not auto.enabled or self._latest_ts == 0:
            return
        if self.fsm.state in (State.PAUSED, State.PANIC):
            self._stall_emitted = False
            return
        gap = time.time() - self._latest_ts
        if gap >= auto.frame_stall_s and not self._stall_emitted:
            self._stall_emitted = True
            self._stall_gap = gap
            ev = {"ts": time.time(), "state": self.fsm.state.value,
                  "event": "frame_stall", "gap_s": round(gap, 1)}
            self.fsm.events.append(ev)
            if len(self.fsm.events) > 400:
                del self.fsm.events[:-200]
            self._emit_event_log(ev)
            log.warning("帧流中断 %.1fs（代理掉线/游戏卡死？已暂停一切输入）", gap)
        elif gap < auto.frame_stall_s and self._stall_emitted:
            self._stall_emitted = False
            ev = {"ts": time.time(), "state": self.fsm.state.value,
                  "event": "frame_resumed", "gap_s": round(self._stall_gap, 1)}
            self.fsm.events.append(ev)
            self._emit_event_log(ev)
            log.info("帧流恢复（中断 %.1fs）", self._stall_gap)

    async def watchdog_loop(self) -> None:
        while True:
            await asyncio.sleep(2.0)
            try:
                self.check_frame_stall()
                self.check_vision_lost()
                self.check_notify_rules()
            except Exception:  # noqa: BLE001
                log.exception("看门狗异常（忽略）")

    # ------------- 视觉看门狗：看不到游戏画面就自动停止 -------------

    def check_vision_lost(self) -> None:
        """看不到游戏画面 → 自动停止。

        两类"看不到"分开计时，因为它们的正确处置不一样：
          - 完全没有新帧：相机被关/代理掉线/游戏退出。没有任何决策依据，
            多等一秒都是白跑，`no_frame_stop_s`（默认 30s）到点就停。
          - 有帧但判定不在游戏里：可能是菜单/弹窗，autonomy 会先按 ESC 自救
            （gate_window_s 默认 25s）。所以这里必须等更久（默认 45s），
            否则会在自救进行到一半时把程序掐掉。
        """
        wd = self.cfg.watchdog
        if not wd.enabled or self._auto_stopped:
            return
        if self.fsm.state in (State.PAUSED, State.PANIC):
            self._vision_lost_since = 0.0
            return
        now = time.time()

        reason = ""
        if self._latest_ts <= 0:
            return                                  # 还没收到过帧，不算"丢失"
        gap = now - self._latest_ts
        with self._lock:
            scene = self._latest_scene
        in_game = bool(scene.in_game) if scene is not None else False

        if gap >= wd.no_frame_stop_s:
            reason = f"连续 {gap:.0f}s 没有新帧（相机被关/代理掉线/游戏退出）"
        elif not in_game and gap < 5.0:
            # 有帧但判定不在游戏里：从第一次判定开始计时
            if self._vision_lost_since <= 0:
                self._vision_lost_since = now
            lost_for = now - self._vision_lost_since
            if lost_for >= wd.no_game_stop_s:
                reason = f"连续 {lost_for:.0f}s 看不到游戏画面（相机没对准/窗口被挡）"
        else:
            if self._vision_lost_since > 0 and in_game:
                log.info("画面恢复，取消自动停止计时")
            self._vision_lost_since = 0.0
            return

        if not reason:
            return
        self._vision_lost_reason = reason
        ev = {"ts": now, "state": self.fsm.state.value,
              "event": "vision_lost", "reason": reason,
              "auto_stop": bool(wd.auto_stop)}
        self.fsm.events.append(ev)
        if len(self.fsm.events) > 400:
            del self.fsm.events[:-200]
        self._emit_event_log(ev)
        log.error("视觉看门狗: %s", reason)

        if wd.auto_stop:
            self._auto_stopped = True
            self._manual_needed += 1
            try:
                self.fsm.stop()
            except Exception:  # noqa: BLE001
                log.exception("自动停止失败")
            self.notifier.alert(
                "已自动停止：看不到游戏画面",
                f"{reason}\n状态已切到 PAUSED，请检查相机位置/游戏窗口后手动重新开始。",
                level="error",
            )
        else:
            self.notifier.alert("看不到游戏画面（仅告警）", reason, level="warn")

    # ------------- 告警规则 -------------

    def metrics(self) -> dict[str, float]:
        """告警规则用的指标快照。"""
        now = time.time()
        stats = self.fsm.stats
        runtime = (now - self._started_at) if self._started_at > 0 else 0.0
        # "有目标但捡不到" → 疑似背包满（游戏玩法内程序判断不了的东西）
        with self._lock:
            scene = self._latest_scene
        has_targets = bool(scene is not None and (scene.pillars or scene.item_list))
        if stats["picks"] != self._last_pick_count:
            self._last_pick_count = stats["picks"]
            self._last_pick_ts = now
            self._no_pick_since = 0.0
        elif has_targets and self.fsm.state not in (State.PAUSED, State.PANIC):
            if self._no_pick_since <= 0:
                self._no_pick_since = now
        else:
            self._no_pick_since = 0.0
        no_pick_for = (now - self._no_pick_since) if self._no_pick_since > 0 else 0.0
        return {
            "rechallenges": float(stats.get("rechallenges", 0)),
            "picks": float(stats.get("picks", 0)),
            "runtime_s": float(runtime),
            "abandons": float(stats.get("abandons", 0)),
            "recoveries": float(stats.get("recoveries", 0)),
            "manual_needed": float(self._manual_needed),
            "no_pick_with_targets_s": float(no_pick_for),
            "blind_taps": float(stats.get("blind_tap_violations", 0)),
            "dropped_commands": float(self.executor.dropped),
        }

    def check_notify_rules(self) -> None:
        if not self.cfg.notify.enabled:
            return
        self.notifier.check(self.metrics())

    def note_manual_needed(self, reason: str) -> None:
        """标记"这件事程序判断不了，需要人看一眼"。"""
        self._manual_needed += 1
        self.notifier.alert("需要人工判断", reason, level="warn")

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
            "frame_gap_s": (round(time.time() - self._latest_ts, 1)
                            if self._latest_ts else None),
            "frame_stalled": self._stall_emitted,
            "executor": {
                "backend": getattr(self.executor, "backend_name", self.cfg.executor.backend),
                "sent": self.executor.sent,
                "dropped": self.executor.dropped,
                "panicked": self.executor.panicked,
            },
            "stats": dict(self.fsm.stats),
            "autonomy": self.fsm.autonomy_snapshot(),
            "scene": scene.model_dump() if scene else None,
            "events": self.fsm.events[-40:],
            "ui_task_ends": list(self.ui_task_ends),
            "history_len": len(self._history),
            "watchdog": {
                "enabled": self.cfg.watchdog.enabled,
                "auto_stopped": self._auto_stopped,
                "reason": self._vision_lost_reason,
                "vision_lost_for_s": (round(time.time() - self._vision_lost_since, 1)
                                      if self._vision_lost_since > 0 else 0.0),
                "no_frame_stop_s": self.cfg.watchdog.no_frame_stop_s,
                "no_game_stop_s": self.cfg.watchdog.no_game_stop_s,
            },
            "notify": self.notifier.snapshot(),
            "metrics": self.metrics(),
            "runtime_s": (round(time.time() - self._started_at, 0)
                          if self._started_at > 0 else 0),
        }
        return d

    # ------------- 相机标定反馈 -------------

    def calibration_report(self) -> dict:
        """最新一帧的标定诊断（页面"相机与标定"用）。

        关键点：**即使检不到也要给出"差多少"**。只告诉用户"检不到"等于没说，
        他会去瞎调相机方向；告诉他"A框模板 0.41 / 阈值 0.72"他就知道该重裁模板了。
        """
        from . import calib

        with self._lock:
            jpeg = self._latest_jpeg
            scene = self._latest_scene
            ts = self._latest_ts
        if jpeg is None:
            return {"ok": False, "reason": "尚无画面（代理未连接或相机未采集）"}
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return {"ok": False, "reason": "帧解码失败"}
        diag = getattr(self.pipeline.detector, "diag", {}) or {}
        rep = calib.analyze(frame, scene, diag, self.cfg.vision,
                            frame_age_s=(time.time() - ts) if ts else None)
        rep["ok"] = True
        rep["verdict"] = calib.verdict_label(rep["scores"]["overall"])
        return rep

    def sample_hsv(self) -> dict:
        """从当前画面自动采样 HSV 范围。

        颜色名 = HSV 字典的键，而 `vision.whitelist_colors` 用的是这些名字。
        所以按**现有键的顺序**填回，不能瞎起名（起成 auto0/auto1 会让
        whitelist_colors: [gold] 直接失效 —— 那是静默失效，最难查）。
        """
        from ..vision.auto_calib import auto_hsv_ranges

        with self._lock:
            jpeg = self._latest_jpeg
        if jpeg is None:
            return {"ok": False, "reason": "尚无画面"}
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return {"ok": False, "reason": "帧解码失败"}
        names = list(self.cfg.vision.hsv.keys()) or ["gold", "pink"]
        ranges = auto_hsv_ranges(frame, max_ranges=max(1, len(names) + 1))
        if not ranges:
            return {"ok": False, "reason": "画面里没有足够的高亮饱和像素，采样失败"
                                           "（先让游戏里出现光柱再点）"}
        new: dict = {}
        for i, r in enumerate(ranges):
            new[names[i] if i < len(names) else f"auto{i}"] = r
        self.cfg.vision.hsv = new
        return {"ok": True, "hsv": new,
                "note": "按现有颜色名顺序填回（第 1 个颜色名 = 像素最多的色簇）。"
                        "若金柱不是画面里最多的颜色，名字可能对不上，请核对再保存。"}

    # ------------- 试跑 / 直发 HID（页面上手动验证用） -------------

    def _hid_preview(self, cmd) -> dict:
        """一条命令最终会变成什么：RP2040 JSON 行 + 原始 HID 报文（十六进制）。"""
        from ..executor import RPi5HidgExecutor
        from ..executor.base import build_abs_mouse_report, build_keyboard_report

        sw, sh = self.cfg.executor.screen_w, self.cfg.executor.screen_h
        item: dict = {"op": cmd.op, "key": cmd.key, "x": cmd.x, "y": cmd.y,
                      "button": cmd.button, "hold_ms": cmd.hold_ms}
        if cmd.op in ("mouse_move", "mouse_click"):
            btn = 0x01 if (cmd.op == "mouse_click" and cmd.button == "left") else (
                0x02 if cmd.op == "mouse_click" else 0x00)
            rep = build_abs_mouse_report(int(cmd.x or 0), int(cmd.y or 0), sw, sh, button=btn)
            item["hid_hex"] = " ".join(f"{b:02X}" for b in rep)
            item["hid_desc"] = f"6B 绝对鼠标 btn=0x{btn:02X} x={cmd.x} y={cmd.y}"
            payload = {"op": cmd.op, "x": int(cmd.x or 0), "y": int(cmd.y or 0)}
            if cmd.op == "mouse_click":
                payload["button"] = cmd.button
                payload["hold_ms"] = self.executor.click_press_ms or 90
            item["serial_json"] = json.dumps(payload, ensure_ascii=False)
        else:
            code = RPi5HidgExecutor._KEYMAP.get((cmd.key or "").lower(), 0)
            rep = build_keyboard_report(code)
            item["hid_hex"] = " ".join(f"{b:02X}" for b in rep)
            item["hid_desc"] = f"8B 键盘 key={cmd.key} code=0x{code:02X}"
            item["serial_json"] = json.dumps(
                {"op": cmd.op, "key": cmd.key, "hold_ms": cmd.hold_ms},
                ensure_ascii=False)
        return item

    def test_plan(self) -> dict:
        """用一份**影子 FSM** 在当前帧上跑一拍，回答"接下来几步会做什么"。

        为什么用影子而不是线上那个：线上 FSM 有自己的状态（可能正在走位/冷却），
        直接借来跑会打乱它。影子实例同配置、干跑执行器，结论是"如果现在从扫描
        开始，它打算这么动" —— 对"我是不是配对了"这个问题，这已经够用。
        """
        with self._lock:
            scene = self._latest_scene
        if scene is None:
            return {"ok": False, "reason": "尚无画面（代理未连接或相机未采集）"}
        from ..config import ExecutorConfig
        from ..decision.fsm import DecisionEngine
        from ..executor import DryRunExecutor
        from ..protocol import Command

        dry = DryRunExecutor(ExecutorConfig(min_action_interval_ms=0,
                                            max_actions_per_sec=1000))
        shadow = DecisionEngine(self.cfg.decision, dry)
        shadow.start()
        shadow.on_scene(scene)
        cmds = [Command(**{k: j[k] for k in Command.model_fields if k in j})
                for j in dry.journal]
        return {
            "ok": True,
            "state": shadow.state.value,
            "intent": shadow.last_intent,
            "scene": {
                "in_game": scene.in_game,
                "pillars": len(scene.pillars),
                "aframe": scene.aframe is not None,
                "rechallenge": scene.rechallenge is not None,
                "item_list": len(scene.item_list),
                "ground_labels": len(scene.ground_labels),
                "monsters": len(scene.monsters),
            },
            "commands": [self._hid_preview(c) for c in cmds],
        }

    def test_send(self, commands: list[dict]) -> dict:
        """把命令真发给执行器（含 RP2040），返回每条的结果。

        这是**手动测试工具**，不走决策路径：唯一的额外保护是拾取键盲按会被
        标记 `blind_risk` 并写事件 —— 不拦，因为你可能就是在验证这个键通不通。
        """
        from ..protocol import Command

        if not commands:
            return {"ok": False, "reason": "没有要发送的命令"}
        pickup = (self.cfg.decision.pickup_key or "a").lower()
        results = []
        for raw in commands[:12]:
            try:
                cmd = Command(**{k: v for k, v in raw.items()
                                 if k in Command.model_fields})
            except Exception as e:  # noqa: BLE001
                results.append({"ok": False, "error": f"命令格式错误: {e}"})
                continue
            preview = self._hid_preview(cmd)
            blind_risk = (cmd.op in ("key_tap", "key_down")
                          and (cmd.key or "").lower() == pickup)
            if blind_risk:
                ev = {"ts": time.time(), "state": self.fsm.state.value,
                      "event": "manual_pickup_test", "key": cmd.key}
                self.fsm.events.append(ev)
                if len(self.fsm.events) > 400:
                    del self.fsm.events[:-200]
                self._emit_event_log(ev)
            ok = self.executor.execute(cmd)
            preview["ok"] = bool(ok)
            preview["blind_risk"] = bool(blind_risk)
            results.append(preview)
        return {"ok": True, "sent": sum(1 for r in results if r.get("ok")),
                "results": results}

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
            self._started_at = time.time()
            self._auto_stopped = False
            self._vision_lost_since = 0.0
            self._vision_lost_reason = ""
            self._last_pick_ts = 0.0
            self._no_pick_since = 0.0
        elif action == "stop":
            self.fsm.stop()
        elif action == "panic":
            self.fsm.panic()
        elif action == "reset":
            self.fsm.reset_panic()
        else:
            raise ValueError(f"未知 action: {action}")
        return {"ok": True, "state": self.fsm.state.value}
