"""机器人的四种模式 —— 谁有权让动作停下来。

## 为什么先做这个，而不是先做"听懂话"

在聊天里最要紧的一句是：**"先停下" 不能依赖任何模型。**
如果它要经过录音上传 → 语音识别 → 模型理解 → 生成 JSON → 解析 → 再停，
那这几百毫秒到几秒里机器人还在按键盘。这不是"响应慢一点"，是**停不下来**。

所以顺序是反过来的：先有一个不含任何模型的开关，再往上加对话能力。
模式就是那个开关，而且它是所有动作的**唯一闸门** —— FSM 的自由刷、指令组的
逐步点击、页面上点「试跑」，全部经过执行器，执行器只问这一句"现在允许动吗"。

## 四种模式

| 模式 | 决策 | 动作 | 用来干什么 |
|---|---|---|---|
| `auto` | 跑 | 发 | 按脚本自己打 |
| `coop` | 跑 | 发 | 边打边听你说话（默认） |
| `observe` | 跑 | **不发** | "现在只观察" —— 决策照常产出，供事后复盘 |
| `paused` | 停 | 不发 | "先停下" —— 连决策都不做，但能聊天 |

`observe` 和 `paused` 的区别是刻意的：观察模式要**留下证据**（模型/规则当时想做什么），
暂停模式不产生任何东西。两者都不下发动作，这是唯一不可让步的一条。

## 为什么不用 FSM 的 State 来表达

FSM 的 `PAUSED/SCANNING/RECOVERING` 说的是"这一帧它在干什么"，一个视觉 tick 就能
改好几次。模式说的是"人让它处在什么状态"，只有人能改。混在一起的话，
恢复动作（RECOVERING）会被误当成"已暂停"，而真正暂停时又会因为 FSM 还在跑而
继续按键 —— 这正是聊天里点名的那个冲突："你说暂停了，但视觉脚本还在继续按键"。
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)


class Mode(str, enum.Enum):
    AUTO = "auto"
    COOP = "coop"
    OBSERVE = "observe"
    PAUSED = "paused"

    @property
    def allows_action(self) -> bool:
        """能不能真的按键/点鼠标。"""
        return self in (Mode.AUTO, Mode.COOP)

    @property
    def allows_decision(self) -> bool:
        """还要不要继续跑决策。观察模式要继续（不然没有可复盘的东西）。"""
        return self is not Mode.PAUSED

    @property
    def label(self) -> str:
        return {
            Mode.AUTO: "自动模式",
            Mode.COOP: "协作模式",
            Mode.OBSERVE: "观察模式",
            Mode.PAUSED: "暂停模式",
        }[self]


#: 会过期回到 `resume_to` 的暂停（"暂停十分钟"）。到期时间放在管理器里。
@dataclass
class ModeState:
    mode: Mode = Mode.COOP
    reason: str = "启动默认"
    since: float = field(default_factory=time.time)
    until: float | None = None       # 有值 = 到点自动恢复
    resume_to: Mode = Mode.COOP      # 到期回到哪个模式

    def to_dict(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        return {
            "mode": self.mode.value,
            "label": self.mode.label,
            "reason": self.reason,
            "since": self.since,
            "age_s": round(now - self.since, 1),
            "until": self.until,
            "left_s": round(self.until - now, 1) if self.until else None,
            "allows_action": self.mode.allows_action,
            "allows_decision": self.mode.allows_decision,
        }


class ModeManager:
    """模式 + 一只闸门。线程安全：页面、语音、FSM 线程都会碰它。"""

    def __init__(self, mode: Mode = Mode.COOP,
                 on_change: Callable[[Mode, Mode], None] | None = None) -> None:
        self._st = ModeState(mode=mode)
        self._lock = threading.RLock()
        #: `fn(旧模式, 新模式)`。**通知在锁外发**：回调里会去停引擎/松按键，
        #: 那些动作又会读模式，持锁通知等于给自己下套。
        self.on_change = on_change

    # ---------------- 读 ----------------

    @property
    def mode(self) -> Mode:
        """当前模式。**读的时候也要检查过期** —— "暂停十分钟"到点了是时间自己
        到的，没有哪一行代码会来通知；如果只有 `allows_action` 才顺手清，那么
        "现在什么模式"和"到底能不能动"就会给出两个不一样的答案。"""
        with self._lock:
            tr = self._expire_locked()
            cur = self._st.mode
        self._notify(tr)
        return cur

    def allows_action(self) -> bool:
        """执行器每个动作前问这一句。**这里是热的路径**，不要在这里加 I/O。"""
        with self._lock:
            tr = self._expire_locked()
            ok = self._st.mode.allows_action
        self._notify(tr)
        return ok

    def allows_decision(self) -> bool:
        with self._lock:
            tr = self._expire_locked()
            ok = self._st.mode.allows_decision
        self._notify(tr)
        return ok

    def snapshot(self, now: float | None = None) -> dict:
        with self._lock:
            tr = self._expire_locked(now)
            snap = self._st.to_dict(now)
        self._notify(tr)
        return snap

    # ---------------- 写 ----------------

    def set(self, mode: Mode, *, reason: str = "", until: float | None = None,
            resume_to: Mode | None = None) -> dict:
        with self._lock:
            tr = self._expire_locked()   # 先结算旧账，否则"暂停十分钟→自动"会记错来源
            old = self._st.mode
            self._st = ModeState(mode=mode, reason=reason or old.label,
                                 until=until,
                                 resume_to=resume_to or (old if old is not Mode.PAUSED
                                                         else Mode.COOP))
            if old is not mode:
                log.info("模式 %s → %s（%s）", old.value, mode.value, self._st.reason)
            snap = self._st.to_dict()
        self._notify(tr)
        self._notify((old, mode) if old is not mode else None)
        return snap

    def pause(self, seconds: float | None = None, *, reason: str = "语音/页面暂停") -> dict:
        """暂停。带秒数就到点自动回到暂停前的模式（"暂停十分钟"）。"""
        with self._lock:
            tr = self._expire_locked()
            cur = self._st.mode
        self._notify(tr)
        until = time.time() + seconds if seconds else None
        return self.set(Mode.PAUSED, reason=reason, until=until,
                        resume_to=cur if cur.allows_action else Mode.COOP)

    def resume(self, *, reason: str = "恢复") -> dict:
        """从暂停/观察回到能动手的模式。

        回到哪个模式不是"一律 auto"：如果是"暂停十分钟"到点了，应该回到暂停前那个
        模式；如果本来就是手动观察，恢复要回到协作而不是自作主张降到自动。
        """
        with self._lock:
            tr = self._expire_locked()
            target = self._st.resume_to if self._st.until else (
                Mode.COOP if self._st.mode is Mode.OBSERVE else Mode.AUTO)
            if target in (Mode.OBSERVE, Mode.PAUSED):
                target = Mode.COOP
        self._notify(tr)
        return self.set(target, reason=reason)

    def _expire_locked(self, now: float | None = None) -> tuple[Mode, Mode] | None:
        """到点自动恢复。调用方必须已经持锁，且**在锁外**把返回值交给 `_notify`。"""
        now = time.time() if now is None else now
        if self._st.until and now >= self._st.until and self._st.mode is Mode.PAUSED:
            target = self._st.resume_to
            if target in (Mode.PAUSED, Mode.OBSERVE):
                target = Mode.COOP
            log.info("暂停到点，自动恢复到 %s", target.value)
            self._st = ModeState(mode=target, reason="暂停到点自动恢复")
            return (Mode.PAUSED, target)
        return None

    def _notify(self, trans: tuple[Mode, Mode] | None) -> None:
        fn = self.on_change
        if fn is None or trans is None:
            return
        try:
            fn(trans[0], trans[1])
        except Exception:  # noqa: BLE001
            log.exception("模式切换回调失败（%s → %s）", trans[0].value, trans[1].value)


def install_gate(executor: Any, modes: ModeManager) -> Callable[[], bool]:
    """把闸门挂到执行器上。返回那个判定函数（测试要用）。

    挂在 `BaseExecutor.execute` 里而不是包一层代理，是因为"包一层"会漏：
    代理只拦自己看到的入口，而 `executor.journal` / `sent` / `cursor` 这些
    被别处直接读的属性会分成两份状态。往基类里加四行，所有后端一起生效。
    """
    def _gate() -> bool:
        return modes.allows_action()

    executor.action_gate = _gate
    if not hasattr(executor, "suppressed"):
        executor.suppressed = 0
    return _gate
