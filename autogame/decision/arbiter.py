"""三级裁决器 —— 认得就自己做，拿不准问小模型，小模型也不会才问大模型。

用户描述的正确流程（这就是本模块的规格）：

    ① 先分析画面。截图/文字说明里有过的局面 → 直接处理，不问模型。
    ② 稍微不明白 → 问 laya，但**把系统已支持的行为清单一起递过去**，让它选。
    ③ 清单里都不合适 → 选 others，这时代码才去请求大模型，问"该怎么做"，
       并且把当时的背景、已经试过什么、为什么不行都写进去。
    ④ 等答案的时候人物不许站着不动 —— 该打怪打怪，别被怪打死。
    ⑤ 拿到指示后执行，并把这套「局面 → 动作 → 结果」存进经验库，
       和人工整理的玩法说明放在同一个目录，作为以后的依据。

和 supervisor 的分工（两者都要，别混）：
    supervisor 回答"我这一串动作有没有用"（对账、刹车），事件驱动、低频。
    arbiter    回答"现在这种情况该做什么"（选择、规划），只在快通道答不上来时才介入。

四条硬约束：
  1. **永远不阻塞主循环**。请求在后台线程，tick() 立即返回。等待期返回保命动作
     或 None（None 时 FSM 按自己的规则继续跑，本来就是"该干啥干啥"）。
  2. **每一级只能从能力清单里选**。L1 是单选，L2 是计划，但每一步都必须能翻译回
     ActionRegistry 里的 id。翻译不了 → 整条计划丢弃（不执行残缺计划）。
  3. **不许指路**。参数里出现坐标一律拒（validate_params 里也挡了一道）。
  4. **经验先验证再生效**。模型给的新做法只是 candidate，要真的执行过并且对账
     显示有效，才升 confirmed；只有 confirmed 允许在 ① 阶段直出。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from .actions import ActionRegistry
from .knowledge import KnowledgeBase, fingerprint

log = logging.getLogger(__name__)

_LAY_SYSTEM = (
    "你是 ARPG 搬砖机器人身上的『快速选择器』。主控是确定性视觉状态机，"
    "它现在遇到一个自己处理不了的画面，需要你从**已支持的动作清单**里挑一个。\n"
    "输出一行 JSON，不要解释，不要 markdown：\n"
    '{"action":"<清单里的 id 或 others>","params":{},'
    '"confidence":0.0,"reason":"一句话"}\n'
    "规则：\n"
    "1. 只能选清单里出现过的 id。清单里没有能解决当前局面的动作，就填 others。\n"
    "2. 你**不能**指定去哪、点哪里、走多少像素。清单里的动作自己会从画面找目标。\n"
    "3. 不确定就 others，不要硬凑。others 会转给更强的模型，代价只是慢几秒。\n"
    "4. 人类提供的游戏说明优先级最高，与你的经验冲突时以它为准。"
)

_PLANNER_SYSTEM = (
    "你是 ARPG 搬砖机器人的『慢速规划器』。快速选择器已经确认：现有动作清单里"
    "没有能直接解决当前局面的选项，所以来问你。\n"
    "输出一行 JSON，不要解释，不要 markdown：\n"
    '{"plan":[{"action":"<动作id>","params":{}}],"reason":"为什么这么排",'
    '"confidence":0.0,"missing":"如果清单缺了关键能力，用一句话说明缺什么"}\n'
    "规则：\n"
    "1. plan 的每一步都必须是清单里存在的 id；一步都不许编。\n"
    "2. 你**不能**给坐标、不能写\"向右下方走 120px\"这类东西。"
    "要去哪由画面里的像素证据决定，你只决定\"做什么、按什么顺序\"。\n"
    "3. 最多 5 步。能一步解决就别排三步。\n"
    "4. 清单里真的没有可用动作时，plan 留空、missing 写清楚缺什么 —— "
    "这比编一个做不了的动作有价值得多。\n"
    "5. 已经试过并失败的做法（见 history）不要再排一遍。"
)


class Stage(str, Enum):
    KNOWLEDGE = "l0_knowledge"   # 经验直出
    LAYA = "l1_laya"             # 小模型从清单里选
    PLANNER = "l2_planner"       # 大模型自由规划
    KEEPALIVE = "keepalive"      # 等答案期间的保命动作


@dataclass
class Call:
    """一次要让 FSM 执行的动作。"""

    action_id: str
    params: dict[str, Any] = field(default_factory=dict)
    stage: str = Stage.LAYA.value
    exp_id: str = ""          # 经验条目 id（用于执行后回写）
    reason: str = ""
    expires_at: float = 0.0   # 过期就别执行了：局面已经变了

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action_id, "params": dict(self.params),
                "stage": self.stage, "exp_id": self.exp_id, "reason": self.reason}


@dataclass
class _Pending:
    """执行后待对账的一条。"""

    exp_id: str
    action_id: str
    base: dict[str, Any]
    due: float
    stage: str


class Arbiter:
    """三级裁决。tick() 永不阻塞。"""

    def __init__(self, cfg: Any, registry: ActionRegistry, kb: KnowledgeBase,
                 clock: Callable[[], float] = time.monotonic,
                 root: Path | None = None,
                 caller: Callable[..., str] | None = None) -> None:
        self.cfg = cfg
        self.registry = registry
        self.kb = kb
        self._clock = clock
        self._lock = threading.RLock()
        self._root = Path(root) if root else Path.cwd()
        self._caller = caller  # 测试注入；默认走 llm_client.chat

        self._busy = False
        self._queue: list[Call] = []
        self._pending: list[_Pending] = []
        self._last_ask = 0.0
        self._last_stage = ""
        self._last_call: Call | None = None
        self._shadow_call: Call | None = None
        self._unmatched_since: float | None = None
        self._idle_since: float | None = None
        self._keepalive_at = 0.0
        self._history: list[dict[str, Any]] = []   # 最近做过的（给模型看）
        self._fail_streak = 0
        self._open_until = 0.0
        self.stats = {
            "asks": 0, "l1_hits": 0, "l1_others": 0, "l2_plans": 0,
            "l2_rejected": 0, "executed": 0, "verified_hit": 0,
            "verified_miss": 0, "keepalive": 0, "shadow_would_run": 0,
            "expired": 0, "fail": 0, "tripped": 0,
        }

    # ---------------- 主入口（每帧，非阻塞） ----------------

    def tick(self, probe: dict[str, Any], now: float, *,
             idle_s: float = 0.0, intent: str = "") -> Call | None:
        """返回要执行的动作，或 None（不干预，FSM 继续按自己的规则跑）。

        注意参数 idle_s 是"FSM 多久没发过命令"——保命动作靠它判断该不该插手。
        """
        if self.cfg.mode == "off":
            return None
        self._settle(probe, now)
        self._track_idle(idle_s, now)

        with self._lock:
            if self._busy:
                # 等模型期间：不许站着不动
                return self._may_return(self._keep_alive(probe, now))
            if self._queue:
                call = self._take_queued(probe, now)
                if call is not None:
                    return self._may_return(call)

        trigger = self._need_help(probe, idle_s, now)
        if not trigger:
            return None

        fp = fingerprint(probe)
        exp = self.kb.find_usable(fp)
        if exp is not None:
            params = dict((exp.action or {}).get("params") or {})
            return self._may_return(self._emit(
                exp.action_id, params, Stage.KNOWLEDGE, exp.id,
                f"已验证经验（hit {exp.hits}/miss {exp.misses}）", now))

        self._ask(fp, probe, now, trigger, idle_s, intent)
        return self._may_return(self._keep_alive(probe, now))

    def execute_hook(self, call: Call, probe: dict[str, Any], now: float) -> None:
        """FSM 真的执行后回调：登记对账、写历史。没执行就别调。"""
        with self._lock:
            self.stats["executed"] += 1
            self._history.append({
                "action": call.action_id, "stage": call.stage,
                "intent": probe.get("intent", ""), "at": round(now, 1),
            })
            self._history = self._history[-12:]
            if call.exp_id:
                self._pending.append(_Pending(
                    exp_id=call.exp_id, action_id=call.action_id,
                    base=_baseline(probe),
                    due=now + float(self.cfg.verify_horizon_s), stage=call.stage))

    # ---------------- 触发判断 ----------------

    def _need_help(self, probe: dict[str, Any], idle_s: float, now: float) -> str:
        """什么时候才允许打扰模型。返回触发原因，空串表示不需要。"""
        if self.cfg.mode == "off" or now < self._open_until:
            return ""
        if now - self._last_ask < self.cfg.cooldown_s:
            return ""
        gate = str(probe.get("gate_reason") or "")
        if gate == "unmatched":
            if self._unmatched_since is None:
                self._unmatched_since = now
            if now - self._unmatched_since >= self.cfg.unmatched_trigger_s:
                return f"画面未标定持续 {now - self._unmatched_since:.0f}s"
        else:
            self._unmatched_since = None

        if idle_s >= self.cfg.idle_trigger_s:
            return f"无事可做 {idle_s:.0f}s"
        miss = int(probe.get("miss_streak") or 0)
        if miss >= self.cfg.miss_trigger:
            return f"动作连续落空 {miss} 次"
        return ""

    def _track_idle(self, idle_s: float, now: float) -> None:
        if idle_s <= 0.5:
            self._idle_since = None
        elif self._idle_since is None:
            self._idle_since = now

    # ---------------- 保命 ----------------

    def _keep_alive(self, probe: dict[str, Any], now: float) -> Call | None:
        """待机时别站着挨打：有怪就砍，没有就不插手（FSM 自己会探索）。

        只在"真的闲了很久"时出手 —— 否则会和 FSM 的常规攻击逻辑重复，
        变成每帧多发一条命令。
        """
        if not probe.get("in_game"):
            return None
        if now - self._keepalive_at < self.cfg.keepalive_interval_s:
            return None
        monsters = int(probe.get("monsters") or 0)
        idle_for = 0.0 if self._idle_since is None else now - self._idle_since
        if monsters <= 0 or idle_for < self.cfg.keepalive_after_s:
            return None
        ok, _ = self.registry.validate("attack_nearest", {}, probe)
        if not ok:
            return None
        self._keepalive_at = now
        self.stats["keepalive"] += 1
        log.info("保命：闲 %.1fs 且有 %d 只怪 → 先打一下再等模型", idle_for, monsters)
        return Call("attack_nearest", {}, Stage.KEEPALIVE.value, "",
                    "等待模型回复期间不站着不动", now + 2.0)

    # ---------------- 后台请求 ----------------

    def _ask(self, fp: str, probe: dict[str, Any], now: float, trigger: str,
             idle_s: float, intent: str) -> None:
        with self._lock:
            self._busy = True
            self._last_ask = now
            self.stats["asks"] += 1
        ctx = self._context(fp, probe, now, trigger, idle_s, intent)

        def _run() -> None:
            ok = False
            try:
                call = self._ask_laya(ctx, probe, now)
                if call is not None:
                    with self._lock:
                        self._queue.append(call)
                        self.stats["l1_hits"] += 1
                    ok = True
                else:
                    with self._lock:
                        self.stats["l1_others"] += 1
                    calls = self._ask_planner(ctx, probe, now)
                    with self._lock:
                        self._queue.extend(calls)
                        if calls:
                            self.stats["l2_plans"] += 1
                    ok = bool(calls)
            except Exception as e:  # noqa: BLE001
                log.warning("裁决请求失败（规则继续跑）: %s", e)
                with self._lock:
                    self.stats["fail"] += 1
                    self._fail_streak += 1
                    if self._fail_streak >= self.cfg.failure_threshold:
                        self._open_until = self._clock() + self.cfg.cooldown_after_failure_s
                        self._fail_streak = 0
                        self.stats["tripped"] += 1
                        log.warning("裁决层连续失败，熔断 %.0fs", self.cfg.cooldown_after_failure_s)
            finally:
                with self._lock:
                    self._busy = False
                _ = ok

        threading.Thread(target=_run, daemon=True, name="arbiter").start()

    def _ask_laya(self, ctx: dict[str, Any], probe: dict[str, Any], now: float) -> Call | None:
        """L1：从清单里选一个。选不出来（others/非法/低置信）返回 None 交给 L2。"""
        payload = {
            "现在的情况": ctx["situation"],
            "你可以做的动作（只能从这里选）": ctx["actions"],
            "人类提供的说明与历史经验": ctx["knowledge"] or "（暂无）",
            "最近做过什么": ctx["history"] or "（刚开始）",
        }
        raw = self._call(self.cfg.laya, _LAY_SYSTEM,
                         json.dumps(payload, ensure_ascii=False))
        obj = _parse(raw)
        action_id = str(obj.get("action", "")).strip()
        if action_id in ("", "others", "other", "none"):
            log.info("laya 选择 others：%s", str(obj.get("reason", ""))[:80])
            return None
        params = obj.get("params") if isinstance(obj.get("params"), dict) else {}
        ok, why = self.registry.validate(action_id, params, probe)
        if not ok:
            log.info("laya 选项被拒（%s）→ 升级大模型", why)
            return None
        conf = _conf(obj)
        if conf < self.cfg.min_confidence:
            log.info("laya 置信度 %.2f 不足 → 升级大模型", conf)
            return None
        return self._emit(action_id, params, Stage.LAYA, "",
                          str(obj.get("reason", ""))[:160], now)

    def _ask_planner(self, ctx: dict[str, Any], probe: dict[str, Any],
                     now: float) -> list[Call]:
        """L2：自由规划。整条计划必须全部可执行，否则一条都不执行。"""
        payload = {
            "现在的情况": ctx["situation"],
            "你能用的动作（plan 里每一步都必须是这里的 id）": ctx["actions"],
            "人类提供的说明与历史经验": ctx["knowledge"] or "（暂无）",
            "已经试过但没用的": ctx["history"] or "（无）",
            "为什么找上你": ctx["why"],
        }
        raw = self._call(self.cfg.planner, _PLANNER_SYSTEM,
                         json.dumps(payload, ensure_ascii=False))
        obj = _parse(raw)
        steps = obj.get("plan")
        if not isinstance(steps, list) or not steps:
            missing = str(obj.get("missing", "")).strip()
            log.info("规划器没有给出计划%s",
                     f"（缺能力：{missing}）" if missing else "")
            return []
        if len(steps) > 5:
            log.info("规划 %d 步超过上限 5，丢弃", len(steps))
            self.stats["l2_rejected"] += 1
            return []
        if _conf(obj) < self.cfg.min_confidence:
            log.info("规划置信度不足，丢弃")
            self.stats["l2_rejected"] += 1
            return []

        calls: list[Call] = []
        reason = str(obj.get("reason", ""))[:200]
        for step in steps:
            if not isinstance(step, dict):
                self.stats["l2_rejected"] += 1
                return []
            aid = str(step.get("action", "")).strip()
            params = step.get("params") if isinstance(step.get("params"), dict) else {}
            ok, why = self.registry.validate(aid, params, probe)
            if not ok:
                # 不执行残缺计划：前一步动了、后一步没动，局面会比现在更糟
                log.info("规划含非法步骤（%s: %s），整条丢弃", aid, why)
                self.stats["l2_rejected"] += 1
                return []
            calls.append(Call(aid, params, Stage.PLANNER.value, "",
                              reason, now + self.cfg.plan_ttl_s))

        exp = self.kb.remember(
            situation=str(ctx["situation"].get("summary", ""))[:200],
            action={"id": calls[0].action_id, "params": calls[0].params},
            source="planner", fp=ctx["fp"], note=reason)
        for c in calls:
            c.exp_id = exp.id
        return calls

    # ---------------- 队列 / 对账 ----------------

    def _take_queued(self, probe: dict[str, Any], now: float) -> Call | None:
        while self._queue:
            call = self._queue.pop(0)
            if now > call.expires_at > 0:
                self.stats["expired"] += 1
                log.info("计划已过期（局面变了），丢弃 %s", call.action_id)
                continue
            ok, why = self.registry.validate(call.action_id, call.params, probe)
            if not ok:
                self.stats["expired"] += 1
                log.info("执行前复核失败（%s），丢弃 %s", why, call.action_id)
                continue
            return call
        return None

    def _settle(self, probe: dict[str, Any], now: float) -> None:
        if not self._pending:
            return
        keep: list[_Pending] = []
        for p in self._pending:
            if now < p.due:
                keep.append(p)
                continue
            result, detail = _judge(p, probe)
            if result == "unknown":
                continue
            if p.exp_id:
                self.kb.verify(p.exp_id, result == "hit", detail)
            with self._lock:
                self.stats["verified_hit" if result == "hit" else "verified_miss"] += 1
            log.info("裁决动作对账 %s → %s（%s）", p.action_id, result, detail)
        self._pending = keep

    # ---------------- 输出闸门 ----------------

    def _emit(self, action_id: str, params: dict[str, Any], stage: Stage,
              exp_id: str, reason: str, now: float) -> Call:
        return Call(action_id, dict(params or {}), stage.value, exp_id, reason,
                    now + float(self.cfg.plan_ttl_s))

    def _may_return(self, call: Call | None) -> Call | None:
        """shadow 下只记账不执行 —— 这是"先看它想干什么"的唯一实现方式。"""
        if call is None:
            return None
        with self._lock:
            self._shadow_call = call
            if self.cfg.mode == "shadow":
                self.stats["shadow_would_run"] += 1
                self._audit(call, applied=False)
                return None
            self._last_call = call
        self._audit(call, applied=True)
        return call

    # ---------------- 上下文 ----------------

    def _context(self, fp: str, probe: dict[str, Any], now: float,
                 trigger: str, idle_s: float, intent: str) -> dict[str, Any]:
        situation = {
            "summary": _describe(probe, intent),
            "in_game": bool(probe.get("in_game")),
            "gate_reason": str(probe.get("gate_reason") or ""),
            "掉落光柱": int(probe.get("pillars") or 0),
            "怪物": int(probe.get("monsters") or 0),
            "脚下有拾取框": bool(probe.get("aframe")),
            "闲置秒数": round(idle_s, 1),
            "当前意图": intent or "（没有明确意图）",
            "画面尺寸": f"{probe.get('w', 0)}x{probe.get('h', 0)}",
        }
        return {
            "fp": fp,
            "trigger": trigger,
            "situation": situation,
            "actions": self.registry.prompt_block(probe),
            "knowledge": self.kb.prompt_block(fp),
            "history": self._history_block(),
            "why": trigger,
        }

    def _history_block(self) -> str:
        with self._lock:
            rows = list(self._history)[-8:]
        if not rows:
            return ""
        return "\n".join(f"- {r['at']}s 做了 {r['action']}（{r['stage']}）" for r in rows)

    # ---------------- 模型调用 ----------------

    def _call(self, model_cfg: Any, system: str, user: str) -> str:
        if self._caller is not None:
            return self._caller(model_cfg, system, user)
        from .llm_client import chat

        return chat(model_cfg, system, user)

    # ---------------- 观测 ----------------

    def _audit(self, call: Call, applied: bool) -> None:
        try:
            p = Path(self.cfg.audit_path)
            if not p.is_absolute():
                p = self._root / p
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": round(time.time(), 3), "mode": self.cfg.mode,
                    "applied": applied, **call.to_dict(),
                }, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001
            log.debug("裁决审计写入失败", exc_info=True)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            last = self._last_call or self._shadow_call
            return {
                "mode": self.cfg.mode,
                "busy": self._busy,
                "queued": len(self._queue),
                "tripped": self._clock() < self._open_until,
                "last": last.to_dict() if last else None,
                "applied": bool(self._last_call),
                "unmatched_for": (
                    round(self._clock() - self._unmatched_since, 1)
                    if self._unmatched_since else 0.0),
                "knowledge": self.kb.snapshot(),
                "stats": dict(self.stats),
            }

    def reset(self) -> None:
        with self._lock:
            self._queue.clear()
            self._pending.clear()
            self._history.clear()
            self._unmatched_since = None
            self._idle_since = None


# ---------------- 纯函数 ----------------

def _parse(raw: str) -> dict[str, Any]:
    from .llm_client import extract_json

    return extract_json(raw)


def _conf(obj: dict[str, Any]) -> float:
    try:
        v = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))


def _baseline(probe: dict[str, Any]) -> dict[str, Any]:
    return {
        "picks": int(probe.get("picks") or 0),
        "pillars": int(probe.get("pillars") or 0),
        "monsters": int(probe.get("monsters") or 0),
        "aframe": bool(probe.get("aframe")),
        "in_game": bool(probe.get("in_game")),
    }


def _judge(p: _Pending, probe: dict[str, Any]) -> tuple[str, str]:
    """拿实测比基准。判不出来就 unknown —— 不硬判，否则经验库会被噪声污染。"""
    b = p.base
    picks = int(probe.get("picks") or 0)
    mon = int(probe.get("monsters") or 0)
    if p.action_id == "pickup":
        if picks > int(b["picks"]):
            return "hit", f"拾取 {b['picks']}→{picks}"
        if b["aframe"] and not probe.get("aframe"):
            return "hit", "拾取框已消失"
        return "miss", f"拾取数未增加（{picks}），拾取框仍在"
    if p.action_id == "attack_nearest":
        if mon < int(b["monsters"]):
            return "hit", f"怪物 {b['monsters']}→{mon}"
        return "miss", f"怪物数未减少（{mon}）"
    if p.action_id == "esc":
        if probe.get("in_game") and not b["in_game"]:
            return "hit", "已回到战斗界面"
        return "miss", f"仍在非战斗界面（{probe.get('gate_reason', '')}）"
    if p.action_id == "approach_nearest":
        if int(probe.get("pillars") or 0) < int(b["pillars"]) and picks > int(b["picks"]):
            return "hit", "目标被捡走"
        return "unknown", ""
    if p.action_id in ("reset_stage", "rechallenge"):
        if probe.get("in_game"):
            return "hit", "已回到战斗"
        return "unknown", ""
    # explore / navigate / wait：位移无法从这几个字段判断，不猜
    return "unknown", ""


def _describe(probe: dict[str, Any], intent: str) -> str:
    """把 probe 说成人话。模型看数字表格不如看一句话，但这句话必须只由数字生成。"""
    parts: list[str] = []
    if probe.get("in_game"):
        parts.append("在战斗界面")
    else:
        parts.append(f"不在战斗界面（门禁：{probe.get('gate_reason') or '未知'}）")
    p = int(probe.get("pillars") or 0)
    m = int(probe.get("monsters") or 0)
    parts.append(f"掉落光柱 {p} 个" if p else "视野内没有掉落光柱")
    parts.append(f"附近怪物 {m} 只" if m else "附近没有怪物")
    if probe.get("aframe"):
        parts.append("脚下有拾取提示")
    if intent:
        parts.append(f"当前意图：{intent}")
    return "，".join(parts)
