"""视觉模型顾问（VLM）—— 卡壳时"看一眼画面"，给出人类式判断。

为什么需要它：
  规则视觉（HSV/模板）只能回答"有没有光柱/A框"，回答不了"我现在是不是卡在
  某个弹窗里 / 门在这边 / 该往哪走"。搬砖机器人跑几小时后必然遇到规则覆盖不到
  的局面，这时多模态模型看一眼截图，比继续按规则瞎试有效得多。

安全边界（这条线不能松）：
  - **只给建议，不直接按键**：所有动作先过白名单（walk_to/key/esc/wait/...），
    坐标做边界钳制，按键白名单**不含拾取键** —— 拾取永远只能由 A框检测裁决。
  - **预算**：每分钟上限 + 最小间隔 + 连续失败熔断，避免把 GPU/API 打爆。
  - **不阻塞**：后台线程请求，决策循环只取"已到货"的建议；没有就继续用启发式。

协议：
  ollama:  POST {base}/api/chat   {model, messages:[{role:user, content, images:[b64]}], stream:false}
  openai:  POST {base}/v1/chat/completions  {model, messages:[{role:user, content:[{type:text},{type:image_url}]}]}
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from collections import deque
from typing import Any, Callable

import cv2
import numpy as np

from ..config import VLMConfig

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "你是一个 ARPG 游戏自动化机器人的视觉顾问。用户会给你一帧游戏截图和当前状态摘要。\n"
    "你的任务：判断角色当前所处界面与处境，并给出**一个**能推进游戏的下一步动作。\n"
    "只输出一行 JSON，不要解释，不要 markdown：\n"
    '{"scene":"battle|menu|loading|dialog|dead|unknown",'
    '"stuck_reason":"一句话说明为什么卡住",'
    '"action":"walk_to|key|esc|wait|explore|reset_stage",'
    '"target":[x,y],"key":"w|a|s|d|space|esc|enter|tab",'
    '"confidence":0.0,"reason":"一句话"}\n'
    "规则：\n"
    "- 界面里有弹窗/菜单挡路 → action=esc\n"
    "- 战斗界面里目标（掉落物/怪物/门/传送点）可见 → action=walk_to，target 给目标在**截图像素**里的坐标\n"
    "- 看不到明确目标但需要继续推进 → action=explore\n"
    "- 画面是结算/失败界面且有『再次挑战』类按钮 → action=reset_stage\n"
    "- 完全无法判断 → action=wait，confidence 给低分\n"
    "禁止建议任何拾取类动作（拾取由检测器裁决）。"
)

_SAFE_KEYS_DEFAULT = {"w", "a", "s", "d", "space", "esc", "enter", "tab"}


class VisionAdvisor:
    """多模态顾问客户端（后台线程 + 预算 + 熔断 + 建议白名单）。"""

    def __init__(self, cfg: VLMConfig,
                 frame_source: Callable[[], bytes | None] | None = None) -> None:
        self.cfg = cfg
        self.frame_source = frame_source
        self._lock = threading.RLock()  # 可重入：snapshot() 会在持锁时调用 available()
        self._busy = False
        self._call_ts: deque[float] = deque(maxlen=max(8, cfg.max_calls_per_min * 4))
        self._last_call = 0.0
        self._fail_streak = 0
        self._open_until = 0.0  # 熔断打开到该时刻（monotonic）
        self._pending: dict[str, Any] | None = None
        self.last_advice: dict[str, Any] | None = None
        self.stats = {"calls": 0, "ok": 0, "fail": 0, "rejected": 0, "tripped": 0}

    # ------------- 预算 / 熔断 -------------

    def available(self, now: float | None = None) -> bool:
        if not self.cfg.enabled:
            return False
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._busy:
                return False
            if now < self._open_until:
                return False
            if now - self._last_call < self.cfg.min_interval_s:
                return False
            recent = [t for t in self._call_ts if now - t <= 60.0]
            return len(recent) < self.cfg.max_calls_per_min

    def _note_call(self, now: float) -> None:
        self._last_call = now
        self._call_ts.append(now)

    def _note_result(self, ok: bool) -> None:
        with self._lock:
            if ok:
                self._fail_streak = 0
                self.stats["ok"] += 1
            else:
                self._fail_streak += 1
                self.stats["fail"] += 1
                if self._fail_streak >= self.cfg.failure_threshold:
                    self._open_until = time.monotonic() + self.cfg.cooldown_after_failure_s
                    self._fail_streak = 0
                    self.stats["tripped"] += 1
                    log.warning("VLM 连续失败，熔断 %.0fs", self.cfg.cooldown_after_failure_s)

    # ------------- 请求 -------------

    def request(self, context: dict[str, Any], now: float | None = None) -> bool:
        """异步请求一次建议。返回是否真的发起了请求（预算不足时 False）。"""
        now = time.monotonic() if now is None else now
        if not self.available(now):
            return False
        jpeg = self.frame_source() if self.frame_source else None
        if not jpeg:
            return False
        with self._lock:
            self._busy = True
            self.stats["calls"] += 1
            self._note_call(now)

        def _run() -> None:
            ok = False
            try:
                img_b64 = self._encode(jpeg)
                raw = self._post(img_b64, context)
                advice = self.sanitize(self._parse(raw), context)
                self.last_advice = advice
                with self._lock:
                    self._pending = advice
                ok = advice.get("valid", False)
                log.info("VLM 建议: %s", advice)
            except Exception as e:  # noqa: BLE001
                log.warning("VLM 调用失败(降级启发式): %s", e)
            finally:
                self._note_result(ok)
                with self._lock:
                    self._busy = False

        threading.Thread(target=_run, daemon=True, name="vlm-advisor").start()
        return True

    def take_advice(self) -> dict[str, Any] | None:
        """取走一次性建议（消费后清空，避免重复执行）。"""
        with self._lock:
            a, self._pending = self._pending, None
            return a

    def clear(self) -> None:
        with self._lock:
            self._pending = None

    # ------------- 图像 -------------

    def _encode(self, jpeg: bytes) -> str:
        """JPEG → 等比缩小 → 重编码 → base64（省带宽/显存）。"""
        buf = np.frombuffer(jpeg, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("帧解码失败")
        h, w = img.shape[:2]
        if w > self.cfg.max_width:
            nh = max(1, int(h * self.cfg.max_width / w))
            img = cv2.resize(img, (self.cfg.max_width, nh), interpolation=cv2.INTER_AREA)
        ok, enc = cv2.imencode(".jpg", img,
                               [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.jpeg_quality)])
        if not ok:
            raise ValueError("JPEG 编码失败")
        return base64.b64encode(enc.tobytes()).decode("ascii")

    # ------------- 传输 -------------

    def _post(self, img_b64: str, context: dict[str, Any]) -> str:
        import httpx

        base = self.cfg.base_url.rstrip("/")
        text = (
            "当前状态摘要：\n"
            + json.dumps(context, ensure_ascii=False)
            + "\n请按系统提示只输出一行 JSON。"
        )
        if self.cfg.provider == "ollama":
            url = f"{base}/api/chat"
            payload = {
                "model": self.cfg.model,
                "messages": [{"role": "user", "content": text, "images": [img_b64]}],
                "stream": False,
                "options": {"temperature": 0.1},
            }
            headers = {}
        else:
            url = f"{base}/v1/chat/completions"
            payload = {
                "model": self.cfg.model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    ],
                }],
                "temperature": 0.1,
            }
            key = self.cfg.api_key or _env(self.cfg.api_key_env)
            headers = {"Authorization": f"Bearer {key}"} if key else {}

        resp = httpx.post(url, json=payload, headers=headers, timeout=self.cfg.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        if self.cfg.provider == "ollama":
            return data["message"]["content"]
        return data["choices"][0]["message"]["content"]

    # ------------- 解析 / 清洗 -------------

    def _parse(self, text: str) -> dict[str, Any]:
        text = (text or "").strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"VLM 回复无 JSON: {text[:120]}")
        obj = json.loads(text[start:end + 1])
        if not isinstance(obj, dict) or "action" not in obj:
            raise ValueError(f"VLM 回复缺 action: {text[:120]}")
        return obj

    def sanitize(self, obj: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
        """白名单 + 坐标钳制 + 置信度门槛。返回带 valid/why 的建议字典。"""
        ctx = context or {}
        w = int(ctx.get("screen_w") or 1280)
        h = int(ctx.get("screen_h") or 720)
        action = str(obj.get("action", "")).strip().lower()
        out: dict[str, Any] = {
            "action": action,
            "reason": str(obj.get("reason", ""))[:200],
            "scene": str(obj.get("scene", "unknown"))[:20],
            "stuck_reason": str(obj.get("stuck_reason", ""))[:200],
            "valid": False,
            "why": "",
        }
        try:
            conf = float(obj.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        out["confidence"] = round(max(0.0, min(1.0, conf)), 3)

        if action not in self.cfg.allowed_actions:
            out["why"] = f"动作不在白名单: {action}"
            return out
        if out["confidence"] < self.cfg.min_confidence:
            out["why"] = f"置信度过低: {out['confidence']}"
            return out

        if action == "walk_to":
            tgt = obj.get("target")
            if not (isinstance(tgt, (list, tuple)) and len(tgt) == 2):
                out["why"] = "walk_to 缺 target"
                return out
            try:
                x, y = int(tgt[0]), int(tgt[1])
            except (TypeError, ValueError):
                out["why"] = "target 非整数"
                return out
            # 边界钳制：模型给的坐标可能越界（幻觉），钳到画面内再执行
            out["target"] = [max(0, min(w - 1, x)), max(0, min(h - 1, y))]
            out["valid"] = True
        elif action == "key":
            key = str(obj.get("key", "")).strip().lower()
            allowed = set(self.cfg.allowed_keys) | _SAFE_KEYS_DEFAULT
            if key not in allowed:
                out["why"] = f"按键不在白名单: {key}"
                return out
            out["key"] = key
            out["valid"] = True
        else:
            out["valid"] = True
        return out

    # ------------- 观测 -------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.cfg.enabled,
                "provider": self.cfg.provider,
                "model": self.cfg.model,
                "available": self.available(),
                "tripped": time.monotonic() < self._open_until,
                "stats": dict(self.stats),
                "last_advice": self.last_advice,
            }


def _env(name: str) -> str:
    import os

    return os.environ.get(name, "")
