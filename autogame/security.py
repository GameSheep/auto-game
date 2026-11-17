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

    def sync(self, min_interval_ms: int, max_per_sec: int) -> None:
        """跟随配置热更新。

        构造时就复制数值的话，配置页把「执行速度」拖到另一档后，
        限速器还按启动时的旧值工作 —— 界面上看着生效了，实际没变。
        """
        self.min_interval = min_interval_ms / 1000.0
        self.max_per_sec = max_per_sec

    def delay_needed(self, now: float | None = None) -> float:
        """距下次可发送还需等待的秒数（0=立即可发）。"""
        now = time.monotonic() if now is None else now
        if self._last_ts <= 0:
            return 0.0
        return max(0.0, self.min_interval - (now - self._last_ts))

    def _try(self, now: float | None = None) -> bool:
        """尝试占用一个发送名额（不计数失败）。"""
        now = time.monotonic() if now is None else now
        if self._last_ts > 0 and now - self._last_ts < self.min_interval:
            return False
        # 清理滑动窗口
        horizon = now - 1.0
        while self._window and self._window[0] < horizon:
            self._window.popleft()
        if len(self._window) >= self.max_per_sec:
            return False
        self._last_ts = now
        self._window.append(now)
        self.allowed += 1
        return True

    def allow(self, now: float | None = None) -> bool:
        ok = self._try(now)
        if not ok:
            self.dropped += 1
        return ok

    def acquire(self, max_wait: float = 0.25) -> bool:
        """等到可以发送为止（最多 max_wait 秒），返回是否放行。

        为什么不用 sleep(delay_needed())：**Windows 的 sleep 精度不可靠**
        （默认定时器 15.6ms），请求 26ms 可能 16ms 就返回，于是
        "睡够了"其实没睡够，合法的连续命令被限速器丢掉。
        实测（200 次连续 key_tap+mouse_click）：朴素的 sleep 方案丢 26%，
        现场表现就是"偶尔少按一下 / 点了没反应"，极难排查。

        这里按"实际能否放行"轮询，而不是相信 sleep 的精度。
        """
        deadline = time.monotonic() + max_wait
        while True:
            if self._try():
                return True
            if self.delay_needed() <= 0:
                # 不是间隔问题（速率窗口满 = 指令风暴）→ 立即丢弃，不做无谓等待
                self.dropped += 1
                return False
            if time.monotonic() >= deadline:
                self.dropped += 1
                return False
            time.sleep(0.002)

    def reset(self) -> None:
        self._window.clear()
        self._last_ts = 0.0
