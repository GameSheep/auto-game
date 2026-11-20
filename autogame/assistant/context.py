"""最近的现场 —— 让"刚才"有东西可指。

## 问题

用户说"刚才为什么判断错了"，这句话交给模型是完全无信息量的：
模型没有"刚才"，只有你给它的文字。直接把它加上当前画面，它也只能看图猜。

所以需要一份**最近 60 秒的结构化记录**，随问题一起送过去：

    识别到了什么（界面、光柱数、A 框、置信度、门禁原因）
    做了什么判断（状态迁移、规则命中、模型建议）
    真的发出去了什么动作（按键、点击，以及**被拦下的**）
    当前用的是哪一代配方、哪条引擎版本

有了这些，"为什么"才有依据；没有这些，模型给的永远是听起来合理的编造。

## 两个刻意的取舍

1. **只存结构化的一行，不存整帧图**。60 秒 1Hz 就是 60 帧，每帧 720p 的 base64
   塞进 prompt 是几 MB —— 既传不上去，也没人看得懂。图单独按需取（`frame_ref`）。
2. **内存环形缓冲，不落盘**。它服务的是"刚才"，服务重启之后"刚才"就不存在了。
   要长期的看 `logs/events.jsonl`，那是另一件事（审计）。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any


def _fmt_value(v: Any) -> str:
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


class RecentLog:
    """最近若干秒的事件环形缓冲。线程安全（FSM 线程写、HTTP 线程读）。"""

    def __init__(self, window_s: float = 60.0, capacity: int = 600) -> None:
        self.window_s = float(window_s)
        self._items: deque[dict] = deque(maxlen=int(capacity))
        self._lock = threading.Lock()
        self._frame: dict | None = None      # 最近一帧的位置（不存图本身）

    # ---------------- 写 ----------------

    def add(self, kind: str, text: str, **data: Any) -> dict:
        item = {"ts": time.time(), "kind": str(kind), "text": str(text), "data": data}
        with self._lock:
            self._items.append(item)
        return item

    def note_frame(self, path: str | None, ts: float | None = None,
                   size: tuple[int, int] | None = None) -> None:
        """记一下最新关键帧在哪。**只存引用**：谁要图谁自己按路径去读。"""
        with self._lock:
            self._frame = {"path": path, "ts": time.time() if ts is None else ts,
                           "size": list(size) if size else None}

    # ---------------- 读 ----------------

    def items(self, seconds: float | None = None, *, now: float | None = None
              ) -> list[dict]:
        now = time.time() if now is None else now
        span = self.window_s if seconds is None else float(seconds)
        cutoff = now - span
        with self._lock:
            return [it for it in self._items if it["ts"] >= cutoff]

    def frame_ref(self) -> dict | None:
        with self._lock:
            return dict(self._frame) if self._frame else None

    def brief(self, seconds: float | None = None, *, limit: int = 40,
              now: float | None = None) -> str:
        """给模型看的紧凑事实块。**行数有上限** —— 超了只留最新的。

        格式刻意写成"相对时间 + 类别 + 一句话"，因为模型对
        `[t-12.4s] vision 界面=战斗 光柱=2 A框=无 (门禁=skillbar)` 这种
        半结构化的行的理解，比一串 JSON 好得多，也短得多。
        """
        now = time.time() if now is None else now
        rows = self.items(seconds, now=now)[-limit:]
        if not rows:
            return "（最近没有任何记录 —— 可能还没开始跑，或者帧流断了）"
        lines = []
        for it in rows:
            # 相对时间一律**不长于现在**：刚记下的那一条差值是 0.0001 秒，四舍五入
            # 会写成 `t+0.0s`，模型读到"未来的事件"会当成另一件事。夹到 0 并对
            # 最近的一条直接写"现在"。
            rel = min(0.0, it["ts"] - now)
            when = "现在" if rel > -0.05 else f"t{rel:+.1f}s"
            body = it["text"]
            if it["data"]:
                kv = " ".join(f"{k}={_fmt_value(v)}" for k, v in it["data"].items()
                              if v not in (None, "", [], {}))
                if kv:
                    body = f"{body} ({kv})"
            lines.append(f"[{when}] {it['kind']} {body}")
        return "\n".join(lines)

    def snapshot(self, seconds: float | None = None, *, limit: int = 60) -> dict:
        rows = self.items(seconds)[-limit:]
        return {"window_s": self.window_s, "count": len(self.items(seconds)),
                "frame": self.frame_ref(),
                "items": [{"ts": it["ts"], "ago_s": round(time.time() - it["ts"], 1),
                           "kind": it["kind"], "text": it["text"],
                           "data": it["data"]} for it in rows]}

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._frame = None
