"""多轮对话状态 —— "确认" 指的是哪一件事。

## 为什么必须有这一层

聊天里那个流程是四句话：

    你：刚才这个操作不对，以后换一种方式处理。
    AI：你是希望调整触发条件，还是修改执行动作？     ← 提问
    你：修改执行动作，先后退，再执行原来的操作。
    AI：明白了。我已经生成修改方案。需要现在应用吗？   ← 提问
    你：确认。                                        ← 指的**是上一句**那件事

如果每一句都当成全新指令发给模型，"确认" 这两个字什么也确认不了。
所以提问必须**记住它在等什么答案**，下一句先拿回来对。

## 三条规矩

1. **问过的话必须有 id**。用户回答时用它关联；页面/语音都要回显同一个 id。
2. **会过期**。十分钟前问"要不要应用"，人早走开了，回来一句"确认"就把
   一个补丁打上 —— 这类事故不能靠"他应该记得"来避免。过期就重新问。
3. **可选项有编号**。语音里说"第二个"比复述一遍长句子容易，识别率也高。
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

#: 提问的有效期。超过就当作没问过 —— 宁可多问一次，也不要执行一个过期的确认。
PENDING_TTL_S = 600.0

#: 保留多少轮给模型看。太长没用（模型不会用），太短又记不住上文。
MAX_TURNS = 40


@dataclass
class Pending:
    """在等一个回答。"""

    id: str
    question: str
    kind: str = "question"        # question（需要选择/补充） / confirm（需要确认）
    options: list[str] = field(default_factory=list)
    payload: dict = field(default_factory=dict)   # 待应用的补丁等
    created: float = field(default_factory=time.time)

    def expired(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return now - self.created > PENDING_TTL_S

    def to_dict(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        return {"id": self.id, "question": self.question, "kind": self.kind,
                "options": list(self.options), "created": self.created,
                "age_s": round(now - self.created, 1),
                "payload": self.payload}


@dataclass
class Turn:
    role: str                      # user / assistant / system
    text: str
    ts: float = field(default_factory=time.time)
    source: str = "text"           # text / voice / local / model / offline
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ago_s"] = round(time.time() - self.ts, 1)
        return d


class Dialog:
    """一个会话的全部状态：说过的话、在等什么、最近改了什么。"""

    def __init__(self, max_turns: int = MAX_TURNS) -> None:
        self._turns: deque[Turn] = deque(maxlen=max_turns)
        self._pending: Pending | None = None
        self._lock = threading.RLock()

    # ---------------- 说话 ----------------

    def say(self, role: str, text: str, *, source: str = "text",
            **meta: Any) -> Turn:
        t = Turn(role=role, text=str(text), source=source, meta=meta)
        with self._lock:
            self._turns.append(t)
            if role == "user":
                # 用户一开口，"等回答"就结束了 —— 除非这一句正是回答（由 resolve 先处理）。
                pass
        return t

    def ask(self, question: str, *, options: Iterable[str] = (), kind: str = "question",
            payload: dict | None = None) -> Pending:
        p = Pending(id=uuid.uuid4().hex[:8], question=str(question), kind=kind,
                    options=[str(o) for o in options], payload=dict(payload or {}))
        with self._lock:
            self._pending = p
            self._turns.append(Turn(role="assistant", text=str(question),
                                    source="ask", meta={"pending_id": p.id,
                                                        "options": p.options}))
        return p

    # ---------------- 解析回答 ----------------

    def resolve(self, text: str, *, now: float | None = None) -> dict:
        """看看这句话是不是在回答上一条提问。

        返回 `{matched: bool, kind: "confirm"|"cancel"|"choose"|"text",
        option: int|None, pending: Pending|None, expired: bool}`。

        这里**只判断"是不是在回答"**，不执行任何东西 —— 执行在 brain 里，
        因为"确认应用补丁"和"确认删除一组"虽然都是 confirm，代价完全不同。
        """
        from . import localintent as li

        now = time.time() if now is None else now
        with self._lock:
            p = self._pending
            if p is None:
                return {"matched": False, "kind": "none", "option": None,
                        "pending": None, "expired": False}
            if p.expired(now):
                self._pending = None
                return {"matched": False, "kind": "expired", "option": None,
                        "pending": p.to_dict(now), "expired": True}

        s = li.normalize(text)
        if not s:
            return {"matched": False, "kind": "none", "option": None,
                    "pending": p.to_dict(now), "expired": False}

        # 1) "确认/取消" 先于选项文本匹配。
        #
        #    这一条不是"顺序无所谓"的细节，是真事故：确认框的选项通常**就是**
        #    `["确认", "取消"]`，如果先比选项文本，一句"取消"会命中第 2 个选项 →
        #    被当成"选了取消" → `_do_choice` 拿着 payload 里的补丁去应用。
        #    人越想取消，改动越会被打上。所以语义明确的整句命令必须优先。
        it = li.parse(text)
        if it.kind == li.CONFIRM:
            return {"matched": True, "kind": "confirm", "option": None,
                    "pending": p.to_dict(now), "expired": False}
        if it.kind == li.CANCEL:
            return {"matched": True, "kind": "cancel", "option": None,
                    "pending": p.to_dict(now), "expired": False}
        if it.kind == li.CHOOSE:
            idx = int(it.arg or 0)
            if 0 <= idx < len(p.options):
                return {"matched": True, "kind": "choose", "option": idx,
                        "pending": p.to_dict(now), "expired": False}
            return {"matched": False, "kind": "bad_choice", "option": idx,
                    "pending": p.to_dict(now), "expired": False}

        # 2) 整句等于某个选项（"先后退再打"）—— 复述选项比"第二个"更直接
        for i, opt in enumerate(p.options):
            if li.normalize(opt) == s:
                return {"matched": True, "kind": "choose", "option": i,
                        "pending": p.to_dict(now), "expired": False}

        # 3) 不是"确认/取消/编号"，但提问是开放式的（要一句描述）→ 当成回答内容
        if p.kind == "answer":
            return {"matched": True, "kind": "text", "option": None,
                    "pending": p.to_dict(now), "expired": False}
        return {"matched": False, "kind": "none", "option": None,
                "pending": p.to_dict(now), "expired": False}

    def take_pending(self) -> Pending | None:
        """取走并清空（消费一次）。"""
        with self._lock:
            p, self._pending = self._pending, None
            return p

    def clear_pending(self) -> None:
        with self._lock:
            self._pending = None

    # ---------------- 给模型看的 ----------------

    def history_text(self, limit: int = 10) -> str:
        with self._lock:
            rows = list(self._turns)[-limit:]
        if not rows:
            return "（这是本次会话的第一句）"
        who = {"user": "用户", "assistant": "你", "system": "系统"}
        return "\n".join(f"{who.get(t.role, t.role)}：{t.text}" for t in rows)

    def snapshot(self, limit: int = 20) -> dict:
        with self._lock:
            pending = self._pending.to_dict() if self._pending else None
            turns = [t.to_dict() for t in list(self._turns)[-limit:]]
        return {"pending": pending, "turns": turns}
