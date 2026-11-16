"""Ollama 决策顾问 —— 局域网 LLM 异常局面咨询（对话定稿：pc 上跑 ollama 决策模型）。

定位：**顾问而非决策者**。
  - 触发：连续放弃 ≥ N 次 / 连续空扫等异常局面
  - 输入：紧凑场景摘要（状态/统计/最近事件）
  - 输出：建议动作（JSON），只允许执行白名单安全动作（默认 wait/continue），其余仅记录
  - 安全：默认关闭；仅访问内网 Ollama；后台线程执行，绝不阻塞决策循环；失败静默降级
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable

from ..config import LLMConfig

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "你是一个游戏自动化决策顾问。根据场景摘要给出下一步建议，"
    "只输出一行 JSON：{\"action\": \"wait|continue|panic\", \"reason\": \"简短原因\"}。"
    "wait=暂停观察, continue=继续当前策略, panic=怀疑异常建议人工介入。"
)


class LLMAdvisor:
    """Ollama 顾问客户端（后台线程 + 冷却 + 白名单动作）。

    通过 `note_scene(summary, on_advice)` 注入场景；内部判断触发条件并异步咨询。
    """

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._consecutive_abandons = 0
        self._last_advice_ts = 0.0
        self._busy = False
        self.last_advice: dict[str, Any] | None = None
        self.queries = 0  # 发起的咨询次数（测试/监控）

    def note_abandon(self) -> None:
        self._consecutive_abandons += 1

    def note_success(self) -> None:
        self._consecutive_abandons = 0

    def should_consult(self) -> bool:
        if not self.cfg.enabled or self._busy:
            return False
        if self._consecutive_abandons < self.cfg.trigger_abandons:
            return False
        if time.monotonic() - self._last_advice_ts < self.cfg.cooldown_s:
            return False
        return True

    def build_prompt(self, summary: dict[str, Any]) -> str:
        return (
            f"{_SYSTEM_PROMPT}\n\n场景摘要:\n{json.dumps(summary, ensure_ascii=False)}"
        )

    def consult(self, summary: dict[str, Any],
                on_advice: Callable[[dict[str, Any]], None] | None = None) -> None:
        """异步咨询 Ollama（后台线程，失败静默）。"""
        if not self.cfg.enabled or self._busy:
            return
        self._busy = True
        self._last_advice_ts = time.monotonic()
        self.queries += 1

        def _run() -> None:
            try:
                advice = self._query_ollama(summary)
                self.last_advice = advice
                action = str(advice.get("action", "continue"))
                allowed = action in self.cfg.allowed_actions
                payload = {"advice": advice, "allowed": allowed}
                log.info("LLM 建议: %s (允许执行=%s)", advice, allowed)
                if on_advice:
                    on_advice(payload)
            except Exception as e:  # noqa: BLE001
                log.warning("LLM 咨询失败(静默降级): %s", e)
                if on_advice:
                    on_advice({"error": str(e), "allowed": False})
            finally:
                self._busy = False

        threading.Thread(target=_run, daemon=True, name="llm-advisor").start()

    def _query_ollama(self, summary: dict[str, Any]) -> dict[str, Any]:
        """调用 Ollama /api/chat（内网，非流式），解析建议 JSON。"""
        import httpx

        resp = httpx.post(
            f"{self.cfg.base_url.rstrip('/')}/api/chat",
            json={
                "model": self.cfg.model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(summary, ensure_ascii=False)},
                ],
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=self.cfg.timeout_s,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        return self._parse_advice(content)

    def _parse_advice(self, text: str) -> dict[str, Any]:
        """从回复中提取第一段 JSON（容忍模型包裹文字/代码块）。"""
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"LLM 回复无 JSON: {text[:120]}")
        obj = json.loads(text[start : end + 1])
        if not isinstance(obj, dict) or "action" not in obj:
            raise ValueError(f"LLM 回复缺 action: {text[:120]}")
        return obj

    def reset(self) -> None:
        self._consecutive_abandons = 0
