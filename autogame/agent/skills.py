"""技能（skill）：一串语义动作 + 什么时候算做成了。

技能是**可教的内容**，不是代码。它回答"这一件小事分几步、每步做到什么程度算完"，
比如「刷一批怪」= 走过去 → 打 → 捡 → 直到这片地没东西。

## 为什么技能是数据而不是函数

上一版引擎里"怎么刷"是 `fsm.py` 里一段 2000 行的 if/else，换游戏只能改代码。
而"这游戏该怎么刷"本来就是玩家脑子里的知识：暗黑类要捡光柱、MMO 要捡尸体、
有的游戏根本不用捡。把它写成 YAML，换游戏就是换一份文件。

## 执行模型：帧驱动、不做循环

`tick(scene, now)` 每次只推进一步，没到条件就返回 `None`（这一帧什么都不做）。
不在这里 `while` 循环，是因为决策是帧驱动的：循环会把帧读空，然后拿着 5 秒前的
画面继续按 —— 这正是历史事故里"按键时帧已经过期"的来源。

## 三种"失败"必须分开

    blocked  这项能力这个游戏里没有（没教拾取键）→ 立即中止，别重试
    timeout  这一步在超时前没等到条件（A 框一直没出现）→ 整条技能失败，换下一个
    failed   成功判据没满足 → 同上

`blocked` 不重试是刻意的：没教的键重试 20 秒，产出的只是一堆"看起来很努力"的
按键，日志还全绿 —— 比直接停下更难查。
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, Field, field_validator

from ..protocol import SceneState
from .verbs import VERBS, VerbContext, resolve_target

# ---------------------------------------------------------------- 条件


def _any_target(scene: SceneState, ctx: VerbContext, p: dict) -> bool:
    return bool(scene.pillars or scene.monsters or scene.item_list
                or scene.ground_labels or scene.aframe is not None)


def _near_target(scene: SceneState, ctx: VerbContext, p: dict) -> bool:
    # 写了目标选择器就只认当场解析出来的那个；只有没写选择器时才退回上一次的落点。
    # 混用会拿"上一轮的目标"判"这一轮到了没有"，表现为莫名其妙跳过走位。
    pt = resolve_target(scene, p, ctx)
    if pt is None and not p.get("target"):
        pt = ctx.last_target
    if pt is None:
        return False
    ch = scene.character or (scene.w // 2, scene.h // 2)
    return ((pt[0] - ch[0]) ** 2 + (pt[1] - ch[1]) ** 2) ** 0.5 <= ctx.reach_px


def _target_gone(scene: SceneState, ctx: VerbContext, p: dict) -> bool:
    """刚才那个目标还在不在。

    这是"按完了有没有用"的正确判据。两个坑都踩过：

    1. 用 `not_aframe`：A 框在**还没走到**时也不在，于是这步会被直接跳过 ——
       表现是"技能跑得飞快、什么都没捡"。
    2. 用"屏幕上那个坐标附近还有没有东西"：相机跟着角色走，目标在屏幕上
       一直在移动，会被误判成"已经消失"。

    所以判据只能是**数量**：动之前记下有几根光柱/几只怪，现在少了就是捡到了。
    数量跟相机无关，也跟分辨率无关。
    """
    from .verbs import _COUNT_INDEX, target_counts

    idx = _COUNT_INDEX.get(ctx.last_kind)
    if idx is None or not any(ctx.last_counts):
        return False
    now = target_counts(scene)
    if ctx.last_kind == "aframe":
        # 拾取：**以"地上的东西少了一个"为准**。A 框消失不能当判据 ——
        # 人走开、目标被别的特效遮住，A 框都会消失，但东西还在。
        # 只有在游戏根本没有"地上物体数量"这回事时（纯交互提示），
        # 才退回看提示本身没了。
        if ctx.last_counts[0] > 0:
            return now[0] < ctx.last_counts[0]
        return scene.aframe is None
    return now[idx] < ctx.last_counts[idx]


def _cue_on(scene: SceneState, ctx: VerbContext, p: dict) -> bool:
    """画面判据成立吗（`cues.yaml` 里那条）。

    没写 `cue:` 参数时一律不成立 —— 不猜"最近那条"。猜错的后果是
    "蓝不够也放技能"，而日志里只会显示成"按了但没效果"，查起来极慢。
    """
    name = str(p.get("cue") or "").strip()
    if not name:
        return False
    return name in ctx.cue_on


def _skill_ready(scene: SceneState, ctx: VerbContext, p: dict) -> bool:
    """这个技能现在能不能放（按 CD 算）。

    `cooldown_s` 写在 `scheme.yaml` 的技能槽上（键的属性，跟 hold_ms 一样）。
    写 0 / 不写 = **不知道 CD**，此时一律允许 —— 宁可按了没效果（会计进
    效果账本，攒够证据吐假设），也不要因为"猜一个 8 秒"把技能锁死。
    """
    from .verbs import _find_skill

    slot = _find_skill(ctx.scheme, p)
    if slot is None:
        return False
    cd = float(getattr(slot, "cooldown_s", 0.0) or 0.0)
    if cd <= 0:
        return True
    last = ctx.last_use.get(slot.key)
    if last is None:
        return True
    return (ctx.now - last) >= cd


#: 条件名 → 判定。封闭集合：写错了 lint 报错，不会静默当成"永远不满足"。
CONDITIONS: dict[str, Callable[[SceneState, VerbContext, dict], bool]] = {
    "always": lambda s, c, p: True,
    "never": lambda s, c, p: False,
    "in_game": lambda s, c, p: bool(s.in_game),
    "out_of_game": lambda s, c, p: not s.in_game,
    "loading": lambda s, c, p: bool(s.loading),
    "aframe": lambda s, c, p: s.aframe is not None,
    "not_aframe": lambda s, c, p: s.aframe is None,
    "target": lambda s, c, p: resolve_target(s, p, c) is not None,
    "no_targets": lambda s, c, p: not _any_target(s, c, p),
    "near_target": _near_target,
    "target_gone": _target_gone,
    "monsters": lambda s, c, p: bool(s.monsters),
    "no_monsters": lambda s, c, p: not s.monsters,
    "cue": _cue_on,
    "skill_ready": _skill_ready,
}

CONDITION_CN = {
    "always": "总是", "never": "从不", "in_game": "在战斗界面", "out_of_game": "不在战斗界面",
    "loading": "在加载", "aframe": "A 框在", "not_aframe": "A 框不在",
    "target": "有目标", "no_targets": "画面没有目标", "near_target": "走到目标附近",
    "target_gone": "刚才那个目标消失", "monsters": "有怪", "no_monsters": "没有怪",
    "cue": "画面判据成立", "skill_ready": "技能不在冷却",
}

#: `until` 的默认值：做一次就进下一步（不用等某个条件）
AFTER_ACTION = "after_action"

# ---------------------------------------------------------------- 步骤


class Step(BaseModel):
    """技能的一步。"""

    verb: str
    #: 目标选择器（move_to / attack / interact 用）：nearest_pillar / nearest_monster /
    #: item / label / aframe / point / none
    target: str = ""
    x: int | None = None
    y: int | None = None
    #: use_skill 用：技能 id 或键名
    slot: str = ""
    #: 这一步做到什么时候算完。条件名，或 `after_action`（做一次就完）
    until: str = AFTER_ACTION
    #: 这一步什么情况下跳过。条件名；空=不跳过
    when: str = ""
    #: `when`/`until` 写 `cue` 时，指向 `cues.yaml` 里的哪一条判据。
    #: 写在这里而不是拼进条件名（`cue_mana_ok`），因为条件集是**封闭**的：
    #: 拼名字会让"写错一个字"变成一个看起来正常、实际永不成立的条件。
    cue: str = ""
    max_seconds: float = Field(8.0, gt=0)
    #: 两次动作之间的最小间隔（秒）。帧驱动下每帧都发会把键按成"抖"
    every_s: float = Field(0.25, ge=0)
    note: str = ""

    @field_validator("verb", mode="before")
    @classmethod
    def _v(cls, v: Any) -> str:
        return str(v or "").strip()

    def params(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.target:
            out["target"] = self.target
        if self.cue:
            out["cue"] = self.cue
        if self.x is not None:
            out["x"] = self.x
        if self.y is not None:
            out["y"] = self.y
        if self.slot:
            out["slot"] = self.slot
        return out

    def describe(self) -> str:
        s = self.verb
        if self.target:
            s += f"→{self.target}"
        if self.slot:
            s += f"[{self.slot}]"
        if self.until != AFTER_ACTION:
            s += f" until {self.until}"
        if self.cue:
            s += f"({self.cue})"
        return s


class Skill(BaseModel):
    """一条技能。`goal` 只是分组标签，真正的目标在 goals.yaml 里引用 id。"""

    id: str
    name: str = ""
    goal: str = ""
    #: 这条技能**什么情况下适用**（条件名）。空=总是适用。
    #: 有它才能写"有目标就捡、没目标就去找"这类分工 —— 否则只能靠失败来试错，
    #: 表现是先白等一轮超时再换技能。判断放在起始，不是执行中：执行中改主意会
    #: 让"做到一半的步骤"没有接续。
    when: str = ""
    steps: list[Step] = Field(default_factory=list)
    #: 全部步骤走完后要满足的条件；空 = 走完即成功。
    #: 建议写：不写的话"步骤走完但什么都没发生"会被当成成功。
    success: str = ""
    #: 整套步骤重复几轮。技能循环就靠它：`repeat: 3` = 最多放三轮技能。
    #: 默认 1（不循环）。有上限是刻意的 —— 无上限的循环在"这个技能永远不生效"
    #: 的时候会按满整晚，而那正是最该停下来的情况。
    repeat: int = Field(1, ge=1, le=50)
    #: 循环的提前结束条件（条件名）。满足就走完当前这轮直接结束，
    #: 不用等 repeat 用完 —— 写 `until: no_monsters` 就是"怪清完就别再放了"。
    until: str = ""
    max_seconds: float = Field(120.0, gt=0)
    #: 失败后怎么办：abort=交回上层（换技能/换目标），retry=原地重来
    on_fail: str = "abort"
    retries: int = Field(0, ge=0, le=5)
    note: str = ""

    @field_validator("id", mode="before")
    @classmethod
    def _id(cls, v: Any) -> str:
        s = str(v or "").strip()
        if not s:
            raise ValueError("技能必须有 id（日志和报告都要靠它对照）")
        return s

    def describe(self) -> str:
        s = f"{self.id}（{len(self.steps)} 步"
        if self.repeat > 1:
            s += f" × {self.repeat}"
            if self.until:
                s += f"，{CONDITION_CN.get(self.until, self.until)}为止"
        return s + "）"


# ---------------------------------------------------------------- 运行


class SkillRun:
    """一条技能的一次执行。帧驱动：`tick()` 每帧推一步。"""

    def __init__(self, skill: Skill, ctx: VerbContext, now: float) -> None:
        self.skill = skill
        self.ctx = ctx
        self.started = now
        self.i = 0
        self.step_started = now
        self.last_action_at = -1e9
        self.acted_once = False
        self.rounds = 1                # 已开始的轮数（repeat 的上限）
        #: 本轮开始时已经尝试过多少次。判"这一轮到底动过没有"用它，
        #: 不能用 `acted_once` —— 那个字段在每次切步时都被 `_enter` 清掉，
        #: 走到轮边界时它已经归 False，于是循环永远只跑一轮。
        self._round_mark = 0
        self.status = "running"       # running / ok / failed / blocked
        self.reason = ""
        self.attempts = 0
        self.log: list[dict] = []

    @property
    def finished(self) -> bool:
        return self.status != "running"

    def _enter(self, now: float) -> None:
        self.step_started = now
        self.acted_once = False

    def _finish(self, status: str, reason: str = "") -> None:
        self.status = status
        self.reason = reason

    def tick(self, scene: SceneState, now: float) -> Any:
        """推进一步。返回 `Outcome | None`（None = 这一帧不动）。"""
        if self.finished:
            return None
        if now - self.started > self.skill.max_seconds:
            self._finish("failed", f"技能 {self.skill.id} 超过 {self.skill.max_seconds:.0f}s 未完成")
            return None

        steps = self.skill.steps
        # 用 while 而不是单次判断：跳过条件、已完成步骤要能连着过掉
        while True:
            if self.i >= len(steps):
                # 一轮走完。先看提前结束条件，再决定要不要再来一轮。
                # `until` 只在这里判一次（不在轮内每帧判）：否则"打完再捡掉落"
                # 这种收尾会被"怪没了"当场砍掉 —— 捡不到东西，日志还全绿。
                if self.rounds < self.skill.repeat and self.attempts > self._round_mark:
                    until = self.skill.until
                    if until and CONDITIONS[until](scene, self.ctx, {}):
                        self._finish("ok", f"{CONDITION_CN.get(until, until)}，"
                                           f"不用放第 {self.rounds + 1} 轮")
                        return None
                    self.rounds += 1
                    self.i = 0
                    self._round_mark = self.attempts
                    self._enter(now)
                    continue
                s = self.skill.success
                if s and not CONDITIONS[s](scene, self.ctx, {}):
                    self._finish("failed", f"步骤走完但成功判据「{CONDITION_CN.get(s, s)}」没满足")
                else:
                    self._finish("ok")
                return None

            step = steps[self.i]
            params = step.params()

            if step.when and not CONDITIONS[step.when](scene, self.ctx, params):
                self._next(now)
                continue

            if step.until != AFTER_ACTION and CONDITIONS[step.until](scene, self.ctx, params):
                self._next(now)
                continue

            if self.acted_once and step.until == AFTER_ACTION:
                self._next(now)
                continue

            if now - self.step_started > step.max_seconds:
                self._finish("failed",
                             f"第 {self.i + 1} 步（{step.describe()}）超过 "
                             f"{step.max_seconds:.0f}s 未满足「{CONDITION_CN.get(step.until, step.until)}」")
                return None

            if now - self.last_action_at < step.every_s:
                return None

            verb = VERBS.get(step.verb)
            if verb is None:  # lint 会先报；这里兜底，不让 None 炸在运行期
                self._finish("blocked", f"未知动作 {step.verb}")
                return None

            self.last_action_at = now
            self.attempts += 1
            out = verb.perform(scene, self.ctx, params)
            self.log.append({"step": self.i, "verb": step.verb, "ok": out.ok,
                             "blocked": out.blocked, "reason": out.reason})
            if out.ok:
                self.acted_once = True
            if out.blocked_capability:
                # 这项能力这个游戏里没有 —— 重试只会产出假动作
                self._finish("blocked", f"第 {self.i + 1} 步做不到：{out.reason}")
            return out

    def _next(self, now: float) -> None:
        self.i += 1
        self._enter(now)
