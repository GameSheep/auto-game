"""目标（goal）：这个游戏里"我在干什么"。

技能是"怎么干一件小事"，目标是"为什么干"。刷图、清任务、回城补给都是目标。
目标是**按优先级排的一份清单**，程序每次挑第一个"现在适用、而且真做得到"的。

## 为什么要"真做得到"这一票否决

目标清单里可以写「打 Boss」，但如果这个游戏里还没教交互键、也没有 Boss 房的
识别模板，那它现在就是做不到。做法不是"试着跑跑看"，而是在选目标时就把
没有可用技能的目标标成 blocked，并写清缺哪一项 —— 这样"现在能玩到什么程度"
是一句可回答的话，而不是跑一晚上之后看日志猜。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from ..protocol import SceneState
from .skills import CONDITIONS, CONDITION_CN
from .verbs import VerbContext


class Goal(BaseModel):
    """一个目标。"""

    id: str
    name: str = ""
    #: 什么情况下这个目标适用（条件名）。默认"在战斗界面"。
    when: str = "in_game"
    #: 候选技能，按优先顺序。前一个失败/做不到就试下一个。
    skills: list[str] = Field(default_factory=list)
    #: 数值越大越先考虑
    priority: int = 0
    #: 干多久还没完成就放弃（0=不设上限）。防止在一个做不到的目标上空转。
    max_seconds: float = Field(0.0, ge=0)
    note: str = ""

    @field_validator("id", mode="before")
    @classmethod
    def _id(cls, v: Any) -> str:
        s = str(v or "").strip()
        if not s:
            raise ValueError("目标必须有 id")
        return s

    def describe(self) -> str:
        return f"{self.id}（{len(self.skills)} 个候选技能，priority={self.priority}）"


class GoalSet(BaseModel):
    goals: list[Goal] = Field(default_factory=list)

    def select(self, scene: SceneState, ctx: VerbContext,
               usable: set[str]) -> tuple[Goal | None, list[dict]]:
        """挑一个目标。返回 (选中的目标, 被跳过的目标及原因)。

        排序：priority 降序，同优先级按文件里的顺序（稳定）。
        `usable` 是"现在真的做得到"的技能 id 集合 —— 由 knowledge 层算好传进来，
        因为判断它需要操作方案（哪些能力已教），不该散在目标里。
        """
        skipped: list[dict] = []
        ordered = sorted(enumerate(self.goals), key=lambda kv: (-kv[1].priority, kv[0]))
        for _, g in ordered:
            cond = CONDITIONS.get(g.when)
            if cond is None:
                skipped.append({"id": g.id, "reason": f"条件名写错了：{g.when}"})
                continue
            if not cond(scene, ctx, {}):
                skipped.append({"id": g.id,
                                "reason": f"现在不适用（{CONDITION_CN.get(g.when, g.when)}）"})
                continue
            cand = [s for s in g.skills if s in usable]
            if not cand:
                skipped.append({"id": g.id, "reason": "没有可用的技能（见知识检查）"})
                continue
            return g, skipped
        return None, skipped
