"""操作方案（InputScheme）—— 把"这个游戏怎么被操作"变成一份可教的数据。

原来的做法是 `decision.approach_mode` 两档 + 几个写死的键名字段。那只对
"点地面移动的 ARPG"成立；换一款 WASD + 1-9 技能栏的游戏，字段名对不上，
程序会照旧拿 `a` 当拾取、拿 `d` 当交互、拿 `w` 当向前 —— 表现是"能动但
每个动作都错"，比不能动更难查。

这里的分层很关键：

  - **方案（本包）** 只描述"哪个物理输入 = 游戏里的哪个动作"。
  - **铁律（决策层）** 描述"什么时候才允许按"。这一层**不许**被方案影响：
    拾取必须过场景门/A框/目标存在于的三道闸，`fail_closed` 不许被方案关掉。

为什么要把两件事分开：方案可以靠"问用户 + 试按"学出来，而"什么时候按"
一旦也能被学，就等于允许程序自己决定"这次可以不确认就按"——那正是
`blind_tap_violations` 想抓的东西。所以方案里没有、也不该有 `when` 字段。
"""

from .scheme import (  # noqa: F401
    CAPABILITIES,
    CAPABILITY_CN,
    PRESETS,
    Binding,
    Conflict,
    InputScheme,
    InputSchemeError,
    MoveScheme,
    SkillSlot,
    from_legacy,
    is_mouse_key,
    normalize_key,
    preset,
)

__all__ = [
    "CAPABILITIES",
    "CAPABILITY_CN",
    "PRESETS",
    "Binding",
    "Conflict",
    "InputScheme",
    "InputSchemeError",
    "MoveScheme",
    "SkillSlot",
    "from_legacy",
    "is_mouse_key",
    "normalize_key",
    "preset",
]
