"""安全模块 —— token 鉴权 + 执行器限速。

设计：
  - WS 连接必须在握手时携带 token（query 参数 ?token=... 或首条 hello 消息）。
  - 使用常数时间比较防时序攻击。
  - 执行器限速：任何后端（含干跑）都经过同一限速器，防止异常指令风暴。
"""

from __future__ import annotations

import hmac
import time
from collections import deque


def check_token(provided: str | None, expected: str) -> bool:
    """常数时间 token 比较。"""
    if not provided:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


class ActionRateLimiter:
    """执行器动作限速器。

    - min_action_interval_ms：两次动作之间的最小间隔（自适应等待而非丢弃，
      保证 mouse_click 等连续合法命令完整下发）
    - max_actions_per_sec：滑动窗口速率上限（超限丢弃，防指令风暴）
    """

    def __init__(self, min_interval_ms: int = 25, max_per_sec: int = 20) -> None:
        self.min_interval = min_interval_ms / 1000.0
        self.max_per_sec = max_per_sec
        self._window: deque[float] = deque()
        self._last_ts: float = 0.0
        self.dropped = 0
        self.allowed = 0

    def delay_needed(self, now: float | None = None) -> float:
        """距下次可发送还需等待的秒数（0=立即可发）。"""
        now = time.monotonic() if now is None else now
        if self._last_ts <= 0:
            return 0.0
        return max(0.0, self.min_interval - (now - self._last_ts))

    def allow(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if now - self._last_ts < self.min_interval:
            self.dropped += 1
            return False
        # 清理滑动窗口
        horizon = now - 1.0
        while self._window and self._window[0] < horizon:
            self._window.popleft()
        if len(self._window) >= self.max_per_sec:
            self.dropped += 1
            return False
        self._last_ts = now
        self._window.append(now)
        self.allowed += 1
        return True

    def reset(self) -> None:
        self._window.clear()
        self._last_ts = 0.0
