"""游戏者（agent）—— 换游戏靠补知识，不靠改代码。

分五层，越往上越"可教"，越往下越"不可动"：

    goals      目标：现在想干什么            ← YAML，随便改
    skills     技能：分几步、做到什么算完    ← YAML，随便改
    cues       判据：画面哪里、读什么、多少算成立  ← YAML，按截图标定
    verbs      动作：走过去/打/捡/交互/技能  ← 代码，证据写死
    effects    效果：做了之后世界变没变      ← 代码，只累计证据

底层（verbs/effects）与游戏无关：它只知道"有目标才能走""有 A 框才能捡"这些
**任何游戏都成立的常识**。上层（goals/skills）是知识：某款游戏该刷什么、先打谁。
中间那层（cues）是"什么时候按"的证据：蓝够不够、CD 到没到、脚下有没有东西。

所以加一个游戏不是"写一个适配器"，是：

    1. 教操作方案（tools/learn_input.py）→ scheme.yaml
    2. 标判据（tools/learn_cue.py，有截图就能标）→ agent/cues.yaml
    3. 写这个游戏的目标与技能（agent/goals.yaml、skills.yaml）
    4. 跑 tools/agent_play.py --check 看哪些目标现在做得到、哪些缺什么
    5. 跑起来，看 effects 账本：哪个动作在这个情境下一直没效果
    6. 按账本的假设去补知识（键位/判据/选择器），回到 4

第六步不自动做 —— 账本能说"这个键好像没用"，不能自己改键位。
"""

from .cues import Cue, CueBook  # noqa: F401
from .effects import (  # noqa: F401
    CHANGED,
    NO_EFFECT,
    SCENE_FLIP,
    EffectLedger,
    SceneSignature,
    classify,
    situation_of,
)
from .goals import Goal, GoalSet  # noqa: F401
from .knowledge import (  # noqa: F401
    AgentKnowledge,
    Issue,
    capability_states,
    goal_context,
    resolve_knowledge_dir,
    skill_capabilities,
    usable_skills,
)
from .runner import GamePlayer  # noqa: F401
from .skills import (  # noqa: F401
    AFTER_ACTION,
    CONDITION_CN,
    CONDITIONS,
    Skill,
    SkillRun,
    Step,
)
from .verbs import (  # noqa: F401
    EVIDENCE,
    EVIDENCE_CN,
    TARGET_KINDS,
    VERBS,
    Outcome,
    Verb,
    VerbContext,
    resolve_target,
)

__all__ = [
    "AFTER_ACTION", "CHANGED", "CONDITION_CN", "CONDITIONS", "EVIDENCE",
    "EVIDENCE_CN", "NO_EFFECT", "SCENE_FLIP", "TARGET_KINDS", "VERBS",
    "AgentKnowledge", "Cue", "CueBook", "EffectLedger", "GamePlayer", "Goal",
    "GoalSet", "Issue",
    "Outcome", "SceneSignature", "Skill", "SkillRun", "Step", "Verb",
    "VerbContext", "capability_states", "classify", "goal_context",
    "resolve_knowledge_dir", "resolve_target", "skill_capabilities",
    "situation_of", "usable_skills",
]
