"""知识：把"这款游戏怎么玩"装进文件，而不是装进代码。

一个游戏的知识 = 操作方案（`scheme.yaml`，怎么按）+ 技能（分几步做，做到什么算完）
+ 目标（先干什么）。这里只管后两样，操作方案由 `autogame.input` 负责。

## 为什么知识需要"检查"，而且检查要能说人话

可教的东西一定会被教错：动词名打错一个字、引用了不存在的技能、`when` 和动词的
证据要求自相矛盾。这些错误不会抛异常 —— 程序照常跑，只是一步都不动或者一直等，
日志里看不出问题。所以知识必须能"编译期"报错，而且要报出**改哪一个字段**。

分三级：

    error    这份知识本身写错了（未知动词/未知条件/引用不存在的技能）
    blocked  知识没错，但这个游戏现在做不到（拾取键还没教）—— 是缺口，不是错误
    warn     能跑，但大概率不达预期（面板键没有 when、技能没有成功判据）

`blocked` 单独一级很重要：把它算成 error，人就会去"改知识"来消错，
而真正该做的是教键位；把它算成 warn，又会被忽略。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

from ..input.scheme import InputScheme
from ..protocol import SceneState
from .cues import CueBook
from .goals import Goal
from .skills import CONDITION_CN, CONDITIONS, Skill
from .verbs import (EVIDENCE, NEEDS_SLOT, NEEDS_TARGET, TARGET_KINDS, VERBS,
                    VerbContext, _find_skill)

#: 两种状态算"能用"。`partial`（WASD 无视角）算能用，但页面上会标出来。
USABLE_STATES = ("ready", "partial")

_STATE_CN = {"ready": "已教", "partial": "部分可用", "unknown": "还没教",
             "unsupported": "引擎不支持", "blocked": "方案冲突"}


class Issue(BaseModel):
    level: str = "warn"          # error / blocked / warn
    code: str = ""
    where: str = ""              # skill 或 goal 的 id
    message: str = ""
    fix: str = ""

    def line(self) -> str:
        tag = {"error": "错误", "blocked": "缺口", "warn": "提示"}.get(self.level, self.level)
        where = f"[{self.where}] " if self.where else ""
        s = f"{tag}：{where}{self.message}"
        return s + (f" → {self.fix}" if self.fix else "")


def capability_states(scheme: InputScheme) -> dict[str, str]:
    return {r["capability"]: r["state"] for r in scheme.capability_table()}


def skill_capabilities(skill: Skill) -> list[str]:
    """这条技能会用到哪些能力（按步骤顺序去重）。"""
    caps: list[str] = []
    for st in skill.steps:
        v = VERBS.get(st.verb)
        if v and v.capability and v.capability not in caps:
            caps.append(v.capability)
    return caps


def usable_skills(knowledge: "AgentKnowledge", scheme: InputScheme) -> set[str]:
    states = capability_states(scheme)
    out: set[str] = set()
    for sk in knowledge.skills:
        caps = skill_capabilities(sk)
        if all(states.get(c, "unknown") in USABLE_STATES for c in caps):
            out.add(sk.id)
    return out


class AgentKnowledge(BaseModel):
    """一个游戏的全部可教内容。"""

    skills: list[Skill] = Field(default_factory=list)
    goals: list[Goal] = Field(default_factory=list)
    #: 判据（画面里"现在该不该按"的证据）。没有也不影响跑，只是技能里
    #: 写 `when: cue` 的步骤会一直不成立 —— 那是知识缺口，不是错误。
    cues: CueBook = Field(default_factory=CueBook)
    source_dir: str = ""

    # ------------------------------------------------------------ 加载

    @classmethod
    def load(cls, path: str | Path) -> "AgentKnowledge":
        """从目录（skills.yaml + goals.yaml + cues.yaml）或单个文件加载。"""
        p = Path(path)
        if not p.exists():
            return cls(source_dir=str(p))
        if p.is_dir():
            files = [f for f in (p / "skills.yaml", p / "goals.yaml") if f.exists()]
            cues = CueBook.load(p)
            if not files and not cues:
                return cls(source_dir=str(p))
            data: dict[str, Any] = {"skills": [], "goals": []}
            for f in files:
                raw = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
                if isinstance(raw, list):
                    key = "skills" if f.name.startswith("skill") else "goals"
                    data[key].extend(raw)
                elif isinstance(raw, dict):
                    data["skills"].extend(raw.get("skills") or [])
                    data["goals"].extend(raw.get("goals") or [])
            src = str(p)
        else:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            if not isinstance(raw, dict):
                raise ValueError(f"{p} 应该是一个字典（含 skills/goals 两个键）")
            data = {"skills": raw.get("skills") or [], "goals": raw.get("goals") or []}
            cues = CueBook(cues=raw.get("cues") or [], source=str(p.parent))
            src = str(p.parent)

        try:
            return cls(skills=data["skills"], goals=data["goals"], cues=cues, source_dir=src)
        except ValidationError as e:
            raise ValueError(f"{path} 解析失败：{e}") from e

    # ------------------------------------------------------------ 查询

    def skill(self, sid: str) -> Skill | None:
        return next((s for s in self.skills if s.id == sid), None)

    def goal(self, gid: str) -> Goal | None:
        return next((g for g in self.goals if g.id == gid), None)

    def __bool__(self) -> bool:
        return bool(self.skills or self.goals)

    # ------------------------------------------------------------ 检查

    def lint(self, scheme: InputScheme) -> list[Issue]:
        issues: list[Issue] = []
        states = capability_states(scheme)
        used_by_goal = {sid for g in self.goals for sid in g.skills}

        # 判据本身写错（区域越界、没阈值、模板找不到）—— 这一类最阴：
        # 程序照跑，判据恒为 0，表现是"技能一步都不做"，日志里一个异常都没有。
        base = Path(self.source_dir) if self.source_dir else None
        cue_ids = self.cues.ids()
        for prob in self.cues.problems(base):
            issues.append(Issue(
                level="error", code="bad_cue", where=prob["id"],
                message=f"判据写错了：{prob['message']}",
                fix="用 tools/learn_cue.py 按截图重新标一次（它会给建议阈值）"))
        used_cues = {st.cue for sk in self.skills for st in sk.steps if st.cue}
        for c in self.cues.cues:
            if c.id not in used_cues:
                issues.append(Issue(
                    level="warn", code="orphan_cue", where=c.id,
                    message="这条判据没有被任何技能引用 —— 标了但没人用",
                    fix=f"在步骤上写 when: cue + cue: {c.id}，或删掉这一条"))

        for sk in self.skills:
            if sk.when and sk.when not in CONDITIONS:
                issues.append(Issue(
                    level="error", code="unknown_condition", where=sk.id,
                    message=f"技能的适用条件 when={sk.when!r} 不是已知条件",
                    fix=f"可用：{', '.join(sorted(CONDITIONS))}"))
            if not sk.steps:
                issues.append(Issue(level="error", code="empty_skill", where=sk.id,
                                    message="技能没有任何步骤",
                                    fix="至少写一步（verb: wait 也算）"))
            # 最后一步有真实的后置条件时，"步骤走完"本身就是判据（如 until: target_gone），
            # 不用再写 success。只有一步"做完就完"的技能才需要额外的成功判据。
            if not sk.success and sk.steps and sk.steps[-1].until == "after_action":
                issues.append(Issue(
                    level="warn", code="no_success", where=sk.id,
                    message="最后一步是「做一次就完」，没有成功判据：做没做成看不出来",
                    fix="给最后一步写 until（如 target_gone），或加 `success:`"))
            if sk.until and sk.until not in CONDITIONS:
                issues.append(Issue(
                    level="error", code="unknown_condition", where=sk.id,
                    message=f"循环的提前结束条件 until={sk.until!r} 不是已知条件",
                    fix=f"可用：{', '.join(sorted(CONDITIONS))}"))
            # 技能级的 when 没有步骤参数（没有 cue 名、没有 slot），带参条件放这儿
            # 会永远不成立 —— 表现是"这条技能从来轮不到"，而它明明写得"对"。
            if sk.when in ("cue", "skill_ready"):
                issues.append(Issue(
                    level="error", code="param_condition_on_skill", where=sk.id,
                    message=f"技能级 when={sk.when!r} 需要参数（cue 名 / slot），"
                            "放在技能上永远不成立",
                    fix="把条件下沉到步骤的 when 上，那里能写 cue: / slot:"))
            if sk.repeat > 1 and not sk.until and not sk.success:
                issues.append(Issue(
                    level="warn", code="loop_without_exit", where=sk.id,
                    message=f"repeat: {sk.repeat} 但既没有 until 也没有 success："
                            f"它会把 {sk.repeat} 轮跑满，不管中途有没有打成",
                    fix="加 `until`（满足就提前结束，如 no_monsters）"))

            for i, st in enumerate(sk.steps, 1):
                where = f"{sk.id}#{i}"
                v = VERBS.get(st.verb)
                if v is None:
                    issues.append(Issue(
                        level="error", code="unknown_verb", where=where,
                        message=f"未知动作 {st.verb!r}",
                        fix=f"可用：{', '.join(sorted(VERBS))}"))
                    continue
                # 条件名
                for field_name, cond in (("until", st.until), ("when", st.when)):
                    if cond and cond != "after_action" and cond not in CONDITIONS:
                        issues.append(Issue(
                            level="error", code="unknown_condition", where=where,
                            message=f"{field_name}={cond!r} 不是已知条件",
                            fix=f"可用：{', '.join(sorted(CONDITIONS))}"))
                # 参数齐不齐
                if st.verb in NEEDS_SLOT and not st.slot:
                    issues.append(Issue(
                        level="error", code="missing_slot", where=where,
                        message="use_skill 必须写 slot（技能 id 或键名）",
                        fix="slot 填 scheme.yaml 里 skill 的 id"))
                if st.verb in NEEDS_TARGET:
                    if st.target and st.target not in TARGET_KINDS:
                        issues.append(Issue(
                            level="error", code="bad_target", where=where,
                            message=f"target={st.target!r} 不是已知的目标选择器",
                            fix=f"可用：{', '.join(TARGET_KINDS)}"))
                    if st.target == "point" and (st.x is None or st.y is None):
                        issues.append(Issue(
                            level="error", code="missing_point", where=where,
                            message="target: point 需要 x / y",
                            fix="补上 x、y，或改用 nearest_pillar 这类选择器"))
                # 自相矛盾：一边说"没目标"一边做需要目标的事
                if st.when == "no_targets" and v.needs and "target" in v.needs:
                    issues.append(Issue(
                        level="error", code="contradictory_when", where=where,
                        message=f"{st.verb} 需要目标，但 when 写的是「画面没有目标」",
                        fix="去掉 when，或换成不需要目标的动词（open_panel / wait）"))
                # 拾取/交互的完成条件写成"某样东西不在"，会还没做就判完成
                if st.verb in ("pickup", "interact") and st.until in ("not_aframe", "no_targets"):
                    issues.append(Issue(
                        level="error", code="bad_until", where=where,
                        message=(f"{st.verb} 的完成条件写成「{CONDITION_CN[st.until]}」："
                                 "人还没走到时这个条件也成立，这一步会被直接跳过"),
                        fix="改成 until: target_gone（那个目标消失了才算做完）"))
                # 判据（画面证据）：名字对不上等于这一步永远不成立
                uses_cue = "cue" in (st.when, st.until)
                if uses_cue and not st.cue:
                    issues.append(Issue(
                        level="error", code="missing_cue", where=where,
                        message="条件写的是「画面判据成立」，但没写 cue 是哪一条",
                        fix="加 `cue: <cues.yaml 里的 id>`"))
                if st.cue and not uses_cue:
                    issues.append(Issue(
                        level="warn", code="unused_cue", where=where,
                        message=f"写了 cue: {st.cue}，但 when/until 都不是 cue —— 这个字段不起作用",
                        fix="把 when 或 until 改成 cue"))
                if st.cue and st.cue not in cue_ids:
                    issues.append(Issue(
                        level="error", code="unknown_cue", where=where,
                        message=(f"引用了不存在的判据 {st.cue!r}"
                                 + ("（这个游戏还没有 cues.yaml）" if not cue_ids else "")),
                        fix="在 agent/cues.yaml 里加一条，或用 tools/learn_cue.py 从截图标出来"))
                # skill_ready 但没写 CD：条件恒成立，等于没写
                if "skill_ready" in (st.when, st.until):
                    slot = _find_skill(scheme, st.params())
                    if slot is None:
                        issues.append(Issue(
                            level="error", code="unknown_slot", where=where,
                            message="skill_ready 要配合 slot，但方案里找不到这个技能",
                            fix="slot 填 scheme.yaml 里 skill 的 id"))
                    elif float(getattr(slot, "cooldown_s", 0.0) or 0.0) <= 0:
                        issues.append(Issue(
                            level="warn", code="no_cooldown", where=where,
                            message=f"skill_ready 判的是 CD，但技能 {slot.id or slot.key} "
                                    "没写 cooldown_s：条件永远成立",
                            fix="知道 CD 就写进 scheme.yaml（不知道就别写，别猜）"))
                # 无目标按键：允许，但要求写出条件
                if v.risky and not st.when:
                    issues.append(Issue(
                        level="warn", code="risky_without_when", where=where,
                        message=f"{st.verb} 是无目标按键，容易在战斗中误触",
                        fix="加 when（如 no_monsters），让它在安全的时刻才按"))

            # 能力缺口
            for cap in skill_capabilities(sk):
                state = states.get(cap, "unknown")
                if state not in USABLE_STATES:
                    issues.append(Issue(
                        level="blocked", code=f"cap_{cap}", where=sk.id,
                        message=f"用到「{cap}」，但它{_STATE_CN.get(state, state)}",
                        fix=("用 tools/learn_input.py 把这一项教上" if state == "unknown"
                             else "见「操作方案」卡上的冲突/不支持说明")))
            if sk.id not in used_by_goal:
                issues.append(Issue(level="warn", code="orphan_skill", where=sk.id,
                                    message="这条技能没有被任何目标引用",
                                    fix="在 goals.yaml 的 skills 里加上，或删掉"))

        for g in self.goals:
            if g.when and g.when not in CONDITIONS:
                issues.append(Issue(level="error", code="unknown_condition", where=g.id,
                                    message=f"when={g.when!r} 不是已知条件",
                                    fix=f"可用：{', '.join(sorted(CONDITIONS))}"))
            if not g.skills:
                issues.append(Issue(level="error", code="empty_goal", where=g.id,
                                    message="目标没有候选技能",
                                    fix="列出至少一条 skills.yaml 里的技能 id"))
            for sid in g.skills:
                if self.skill(sid) is None:
                    issues.append(Issue(level="error", code="unknown_skill", where=g.id,
                                        message=f"引用了不存在的技能 {sid!r}",
                                        fix="检查 skills.yaml 里的 id 拼写"))
        return issues

    # ------------------------------------------------------------ 报告（页面/预检）

    def report(self, scheme: InputScheme) -> dict[str, Any]:
        states = capability_states(scheme)
        usable = usable_skills(self, scheme)
        base = Path(self.source_dir) if self.source_dir else None
        goals: list[dict] = []
        for g in sorted(self.goals, key=lambda x: -x.priority):
            cand = [s for s in g.skills if s in usable]
            blocked_by: list[str] = []
            for sid in g.skills:
                sk = self.skill(sid)
                if sk is None:
                    continue
                for cap in skill_capabilities(sk):
                    if states.get(cap, "unknown") not in USABLE_STATES and cap not in blocked_by:
                        blocked_by.append(cap)
            goals.append({
                "id": g.id, "name": g.name or g.id, "priority": g.priority,
                "when": g.when, "skills": list(g.skills),
                "usable": bool(cand),
                "blocked_by": [] if cand else blocked_by,
                "reason": "" if cand else (
                    f"缺：{'、'.join(blocked_by)}" if blocked_by else "没有可用技能"),
            })
        return {
            "source": self.source_dir,
            "skills": [{"id": s.id, "name": s.name or s.id, "steps": len(s.steps),
                        "repeat": s.repeat, "until": s.until,
                        "capabilities": skill_capabilities(s),
                        "usable": s.id in usable} for s in self.skills],
            "cues": self.cues.report(base),
            "goals": goals,
            "usable_goals": [g["id"] for g in goals if g["usable"]],
            "blocked_goals": [{"id": g["id"], "reason": g["reason"]}
                              for g in goals if not g["usable"]],
            "issues": [i.model_dump() for i in self.lint(scheme)],
        }

    def summary(self, scheme: InputScheme) -> str:
        r = self.report(scheme)
        bad = [i for i in r["issues"] if i["level"] == "error"]
        blocked = [i for i in r["issues"] if i["level"] == "blocked"]
        s = (f"目标 {len(r['goals'])} 个（可用 {len(r['usable_goals'])}），"
             f"技能 {len(r['skills'])} 条，{self.cues.summary()}")
        if bad:
            s += f"，{len(bad)} 处写错"
        if blocked:
            s += f"，{len(blocked)} 处缺口"
        return s


def resolve_knowledge_dir(cfg_agent: Any, profile_root: str | Path | None) -> Path:
    """知识目录：配置里写相对路径时，相对 profile 根目录。

    相对路径是有意的：同一个 profile 的知识跟着 profile 走，换机器/换人
    不用改绝对路径。没有 profile 时退回仓库的 `knowledge/agent`。
    """
    raw = str(getattr(cfg_agent, "knowledge_dir", "") or "agent")
    p = Path(raw)
    if p.is_absolute():
        return p
    if profile_root:
        return Path(profile_root) / p
    return Path("knowledge") / raw


def goal_context(scheme: InputScheme, reach_px: int = 120,
                 max_frame_age_ms: float = 400.0) -> VerbContext:
    """目标选择和技能执行共用的上下文。"""
    return VerbContext(scheme=scheme,
                       move_enabled=scheme.to_decision_patch().get("move_enabled", True),
                       reach_px=reach_px, max_frame_age_ms=max_frame_age_ms)


__all__ = [
    "AgentKnowledge", "Issue", "capability_states", "goal_context",
    "resolve_knowledge_dir", "skill_capabilities", "usable_skills",
]
