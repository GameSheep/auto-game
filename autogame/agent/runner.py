"""游戏者（GamePlayer）：把"目标 → 技能 → 动作 → 效果"串成一帧一步的循环。

这不是又一个决策器，而是**上面三层的组合**：

    goals     现在想干什么（可教）
    skills    这件事分几步（可教）
    verbs     每一步具体做什么（与游戏无关，证据写死）
    effects   做完有没有用（可累计的证据，用来生成待验证假设）

## 每帧只做四件事

    1. 结账：上一次动作之后世界变了没有
    2. 选目标：没有目标、或原目标做不下去 → 挑下一个“现在真的做得到”的
    3. 推进一步技能：拿到一次动作意图（或这帧什么都不做）
    4. 发出去：命令交给 sink（执行器/测试收集器）

## 它不会假装

三件事在这里被显式记数，且都会出现在 `stats` 里：

  - `blocked_capability` 这个游戏里没教的能力被技能用到 → 立即换技能，不重试
  - `blocked_evidence` 想按但没有证据（没 A 框、没目标）→ 这一帧不按
  - `noop_completion` 技能"成功"了但一个动作都没发 → 说明判据写错，封掉它

这三条是"假运行"的三个来源：显示在跑、实际什么都没做。能看见，就不会假装。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..input.scheme import InputScheme
from ..protocol import Command, SceneState
from .cues import CueBook
from .effects import (EffectLedger, SceneSignature, classify, situation_of)
from .goals import Goal, GoalSet
from .knowledge import AgentKnowledge, goal_context, usable_skills
from .skills import CONDITIONS, Skill, SkillRun
from .verbs import VERBS, Outcome, VerbContext


@dataclass
class Pending:
    """一次已经发出去、还没结账的动作。"""

    verb: str
    situation: str
    before: SceneSignature
    at: float
    motion: bool = False
    goal: str = ""
    skill: str = ""
    step: int = 0


@dataclass
class GamePlayer:
    knowledge: AgentKnowledge
    scheme: InputScheme
    reach_px: int = 120
    dead_after: int = 3
    max_frame_age_ms: float = 400.0
    sink: Callable[[Command], bool] | None = None
    now_fn: Callable[[], float] = time.monotonic
    #: 判据（`cues.yaml`）。不给也能跑 —— 只是技能里写 `when: cue` 的步骤不成立。
    cues: CueBook | None = None

    ctx: VerbContext = field(init=False)
    goalset: GoalSet = field(init=False)
    ledger: EffectLedger = field(init=False)

    goal: Goal | None = field(default=None, init=False)
    run: SkillRun | None = field(default=None, init=False)
    pending: Pending | None = field(default=None, init=False)
    trace: list[dict] = field(default_factory=list, init=False)
    stats: dict[str, int] = field(default_factory=dict, init=False)

    _banned: set[str] = field(default_factory=set, init=False)
    _gave_up: dict[str, str] = field(default_factory=dict, init=False)
    _goal_started: float = field(default=0.0, init=False)
    _skill_tries: int = field(default=0, init=False)
    _noop: dict[str, int] = field(default_factory=dict, init=False)
    #: 每条判据被读过多少次、其中成立过多少次。用来发现"标错了"——
    #: 一条判据读了几百帧一次都没成立，要么这局真的没这个条件，要么
    #: 区域/颜色标到了别的地方。两种都该让人看见，不该静默不按。
    _cue_seen: dict[str, int] = field(default_factory=dict, init=False)
    _cue_hit: dict[str, int] = field(default_factory=dict, init=False)
    _last_ts: int = field(default=-1, init=False)
    _last_idle: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.ctx = goal_context(self.scheme, self.reach_px, self.max_frame_age_ms)
        self.goalset = GoalSet(goals=self.knowledge.goals)
        self.ledger = EffectLedger(dead_after=self.dead_after)
        for k in ("ticks", "actions", "commands", "blocked_capability",
                  "blocked_evidence", "pickup_refused_no_aframe",
                  "skill_ok", "skill_failed", "skill_blocked",
                  "noop_completion", "goals_taken", "idle_no_goal",
                  "idle_no_scene", "cue_frames", "cue_no_frame", "cue_errors"):
            self.stats.setdefault(k, 0)

    # ------------------------------------------------------------ 对外

    def usable_skill_ids(self) -> set[str]:
        return usable_skills(self.knowledge, self.scheme)

    def on_scene(self, scene: SceneState, now: float | None = None,
                 frame: Any = None) -> list[dict]:
        """喂一帧。返回这一帧产生的记录（给日志/页面/测试用）。

        `frame` 是可选的原始画面（BGR ndarray）。给了才判读 `cues.yaml`：
        判据是画面证据，`SceneState` 里没有它的字段（每款游戏都不一样）。
        不给就是"这套知识里没有判据"或"离线回放"—— 那时 `when: cue` 恒不成立，
        不是恒成立。这个方向很重要：缺证据只能让动作变少，不能让动作变多。
        """
        t = self.now_fn() if now is None else float(now)
        self.ctx.now = t
        self.stats["ticks"] += 1
        self._read_cues(frame)
        events: list[dict] = []

        # 同一帧来两次不结账：否则一次动作会被判两次"没效果"
        if scene.ts_ms != self._last_ts:
            self._settle(scene, t, events)
            self._last_ts = scene.ts_ms

        self._ensure_goal(scene, t, events)
        if self.goal is None or self.run is None:
            return events

        out = self.run.tick(scene, t)
        if self.run.finished:
            self._close_run(events)
            return events
        if out is None:
            return events
        self._dispatch(out, scene, t, events)
        return events

    # ------------------------------------------------------------ 判据

    def _read_cues(self, frame: Any) -> None:
        """把这一帧的画面读数写进上下文。

        没有画面时**清空**而不是"沿用上一帧"：上一帧的蓝量不能决定现在按不按。
        判据是"此刻的证据"，这一点和 A 框的帧时效是同一个道理。
        """
        if self.cues is None or not self.cues:
            return
        base = self.knowledge.source_dir or None
        values = self.cues.evaluate(frame, base_dir=base)
        self.ctx.cues = values
        self.ctx.cue_on = self.cues.on(values) if frame is not None else set()
        self.stats["cue_frames"] = self.stats.get("cue_frames", 0) + 1
        if frame is None:
            self.stats["cue_no_frame"] = self.stats.get("cue_no_frame", 0) + 1
        else:
            for name, val in values.items():
                self._cue_seen[name] = self._cue_seen.get(name, 0) + 1
                if name in self.ctx.cue_on:
                    self._cue_hit[name] = self._cue_hit.get(name, 0) + 1
        if self.cues.last_errors:
            self.stats["cue_errors"] = self.stats.get("cue_errors", 0) + 1

    # ------------------------------------------------------------ 结账

    def _settle(self, scene: SceneState, now: float, events: list[dict]) -> None:
        p = self.pending
        if p is None:
            return
        self.pending = None
        effect = classify(p.before, SceneSignature.of(scene),
                          motion_sensitive=p.motion)
        self.ledger.record(p.verb, p.situation, effect)
        self.stats["effects_" + effect] = self.stats.get("effects_" + effect, 0) + 1
        rec = {"kind": "effect", "ts": round(now, 3), "verb": p.verb,
               "situation": p.situation, "effect": effect,
               "goal": p.goal, "skill": p.skill, "step": p.step,
               "latency_s": round(now - p.at, 3)}
        events.append(rec)
        self._push(rec)

    # ------------------------------------------------------------ 选目标

    def _ensure_goal(self, scene: SceneState, now: float, events: list[dict]) -> None:
        # 目标超时 → 放弃（防止在一个做不到的目标上空转整晚）
        if self.goal is not None and self.goal.max_seconds > 0 \
                and now - self._goal_started > self.goal.max_seconds:
            self._gave_up[self.goal.id] = f"超过 {self.goal.max_seconds:.0f}s 没完成"
            rec = {"kind": "goal_giveup", "ts": round(now, 3),
                   "goal": self.goal.id, "reason": self._gave_up[self.goal.id]}
            events.append(rec)
            self._push(rec)
            self.goal, self.run = None, None

        if self.goal is not None:
            # 目标还在、但手上没有技能在跑（上一条跑完了）→ 重新起一条。
            # 这里就是"刷图循环"：一条技能跑完 → 立刻再来一条，而不是停在那儿。
            if self.run is None and self.goal.id not in self._gave_up:
                self._start_skill(self.goal, scene, now, events)
            return

        usable = self.usable_skill_ids() - self._banned
        # 放弃过的目标不再出现在候选里：否则每个 tick 都会重新选它、重新放弃
        live_goals = GoalSet(goals=[g for g in self.knowledge.goals
                                    if g.id not in self._gave_up])
        g, skipped = live_goals.select(scene, self.ctx, usable)
        live = [s for s in skipped if s["id"] not in self._gave_up]
        if g is None:
            self.stats["idle_no_goal"] += 1
            reason = "; ".join(f"{s['id']}: {s['reason']}" for s in live) or "没有适用目标"
            if reason != self._last_idle:      # 同一个原因只说一次，别刷屏
                self._last_idle = reason
                rec = {"kind": "idle", "ts": round(now, 3), "reason": reason}
                events.append(rec)
                self._push(rec)
            return

        self.goal = g
        self._goal_started = now
        self._skill_tries = 0
        self._last_idle = ""
        self.stats["goals_taken"] += 1
        rec = {"kind": "goal", "ts": round(now, 3), "goal": g.id, "name": g.name or g.id,
               "skills": list(g.skills)}
        events.append(rec)
        self._push(rec)
        self._start_skill(g, scene, now, events)

    def _start_skill(self, g: Goal, scene: SceneState, now: float,
                     events: list[dict]) -> bool:
        """挑一条技能起来。返回是否真的起来了。

        两种情况必须分开：
          - 候选技能**一项都用不了**（缺能力/都被封）→ 放弃目标，换下一个
          - 候选技能**现在不适用**（`when` 不满足，比如没目标时不该捡）→ 等
            等不是失败：看到光柱的那一帧自然会起得来。把它当失败会导致
            "每帧放弃一次目标"的空转。
        """
        usable = self.usable_skill_ids() - self._banned
        cands = [sid for sid in g.skills
                 if sid in usable and self.knowledge.skill(sid) is not None]
        if not cands:
            self._gave_up[g.id] = "没有可用技能（缺能力或都失败过）"
            self.goal, self.run = None, None
            return False
        for sid in cands:
            sk = self.knowledge.skill(sid)
            if sk is None:
                continue
            if sk.when and not CONDITIONS[sk.when](scene, self.ctx, {}):
                continue
            self.run = SkillRun(sk, self.ctx, now)
            self._skill_tries = 0
            rec = {"kind": "skill", "ts": round(now, 3), "skill": sk.id, "goal": g.id}
            events.append(rec)
            self._push(rec)
            return True
        self.stats["idle_no_scene"] += 1
        return False

    # ------------------------------------------------------------ 结果处理

    def _close_run(self, events: list[dict]) -> None:
        run = self.run
        if run is None:
            return
        rec = {"kind": "skill_end", "skill": run.skill.id, "status": run.status,
               "reason": run.reason, "attempts": run.attempts}
        self.run = None
        if run.status == "ok":
            self.stats["skill_ok"] += 1
            if run.attempts == 0:
                # "成功"但一个动作都没发：判据写错了（比如每步都被 when 跳过）
                self.stats["noop_completion"] += 1
                n = self._noop.get(run.skill.id, 0) + 1
                self._noop[run.skill.id] = n
                rec["warning"] = "技能 0 动作完成 —— 判据或 when 写错了"
                if n >= 3:
                    self._banned.add(run.skill.id)
                    rec["warning"] += "；已封掉，不再空转"
        elif run.status == "blocked":
            self.stats["skill_blocked"] += 1
            self._banned.add(run.skill.id)
        else:
            self.stats["skill_failed"] += 1
            self._skill_tries += 1
            if run.skill.retries and self._skill_tries <= run.skill.retries:
                self.run = SkillRun(run.skill, self.ctx, self.now_fn())
                rec["retry"] = self._skill_tries
            else:
                self._banned.add(run.skill.id)

        events.append(rec)
        self._push(rec)
        self._maybe_change_goal(events)

    def _maybe_change_goal(self, events: list[dict]) -> None:
        """当前目标的所有候选技能都用不了 → 放弃这个目标，换下一个。"""
        g = self.goal
        if g is None:
            return
        usable = (self.usable_skill_ids() - self._banned)
        if not any(s in usable for s in g.skills):
            self._gave_up[g.id] = "候选技能都不可用（缺能力或反复失败）"
            rec = {"kind": "goal_giveup", "goal": g.id,
                   "reason": self._gave_up[g.id]}
            events.append(rec)
            self._push(rec)
            self.goal, self.run = None, None

    # ------------------------------------------------------------ 发动作

    def _dispatch(self, out: Outcome, scene: SceneState, now: float,
                  events: list[dict]) -> None:
        goal_id = self.goal.id if self.goal else ""
        skill_id = self.run.skill.id if self.run else ""
        step_i = self.run.i if self.run else 0

        if out.blocked:
            if out.blocked == "capability":
                self.stats["blocked_capability"] += 1
            else:
                self.stats["blocked_evidence"] += 1
                if out.evidence == "aframe":
                    # 这条数必须能被看见：它是"想捡但没证据"的次数，
                    # 也就是铁律在真正生效的证据（应该 >0，而不是 0）
                    self.stats["pickup_refused_no_aframe"] += 1
            rec = {"kind": "blocked", "ts": round(now, 3), "blocked": out.blocked,
                   "reason": out.reason, "goal": goal_id, "skill": skill_id,
                   "step": step_i,
                   "verb": self.run.skill.steps[step_i].verb if self.run else ""}
            events.append(rec)
            self._push(rec)
            return

        if not out.commands:
            return

        self.stats["actions"] += 1
        verb = self.run.skill.steps[step_i].verb if self.run else ""
        sent = 0
        for cmd in out.commands:
            if self.sink is None or self.sink(cmd):
                sent += 1
        self.stats["commands"] += sent
        rec = {"kind": "action", "ts": round(now, 3), "verb": verb,
               "goal": goal_id, "skill": skill_id, "step": step_i,
               "commands": [c.op for c in out.commands], "sent": sent,
               "note": out.note}
        events.append(rec)
        self._push(rec)

        # 记录"动作前的世界"，下一帧结账
        verb_obj = VERBS.get(verb)
        self.pending = Pending(verb=verb, situation=situation_of(scene),
                               before=SceneSignature.of(scene), at=now,
                               motion=bool(verb_obj and verb_obj.effect_kind == "motion"),
                               goal=goal_id, skill=skill_id, step=step_i)

    # ------------------------------------------------------------ 报告

    def _push(self, rec: dict) -> None:
        self.trace.append(rec)
        if len(self.trace) > 400:
            del self.trace[:-250]

    def hypotheses(self) -> list[dict]:
        """攒够证据的"疑似问题"。**只提案，不改任何配置。**"""
        out = list(self.ledger.dead())
        for sid, n in sorted(self._noop.items()):
            if n >= 2:
                out.append({"verb": "", "situation": "", "attempts": n,
                            "hypothesis": f"技能「{sid}」连续 {n} 次 0 动作就「完成」了",
                            "hint": "检查每步的 when 是否把所有步骤都跳过了"})
        # 判据读了几百帧一次没成立：可能是标错了，也可能这局确实没这个条件。
        # 这一条必须由人判 —— 程序自己分不清"蓝一直是满的"和"区域指到了别处"。
        for name, seen in sorted(self._cue_seen.items()):
            if seen >= 200 and not self._cue_hit.get(name):
                out.append({"verb": "", "situation": "", "attempts": seen,
                            "hypothesis": f"判据「{name}」读了 {seen} 帧，一次都没成立",
                            "hint": "用 tools/learn_cue.py 按截图重标 region/color；"
                                    "如果这局本来就一直不满足（比如全程满蓝），这条可以忽略"})
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "goal": self.goal.id if self.goal else "",
            "skill": self.run.skill.id if self.run else "",
            "banned": sorted(self._banned),
            "gave_up": dict(self._gave_up),
            "stats": dict(self.stats),
            "effects": self.ledger.to_dict(),
            "cues": dict(self.ctx.cues),
            "cue_on": sorted(self.ctx.cue_on),
            "cue_stats": {k: {"seen": v, "on": self._cue_hit.get(k, 0)}
                          for k, v in self._cue_seen.items()},
            "hypotheses": self.hypotheses(),
            "trace": self.trace[-20:],
        }
