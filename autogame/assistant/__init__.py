"""对话助手 —— 让人能直接参与"事件 → 诊断 → 改进 → 验证 → 晋升"这个闭环。

模块分工（各自只做一件事，坏了好定位）：

    modes        四种模式 + 动作闸门（没有模型也能停）
    localintent  一句话 → 一个控制命令（纯函数，毫秒级）
    context      最近 60 秒的现场，让"刚才"有东西可指
    dialog       多轮状态：这个"确认"指的是哪件事
    brain        三道兜底：本地 → 模型 → 离线；模型只提案不执行
    patch        改数据不改代码：预览 / 校验 / 应用 / 回滚
    voice        能听能说就用，不能就退到浏览器，不假装

## 一条贯穿全部的设计

**越靠近动作的东西，越不能依赖网络。**

    停止/暂停/继续/观察   本地，0ms，无模型
    改一条规则             哈希校验 + 预览 + 人确认 + 可回滚
    解释"刚才为什么"        模型，秒级，允许失败（失败就说"看不到"）

把这条顺序反过来（先做模型、再补控制），必然出现"网断了停不下来"。
"""

from __future__ import annotations

from .brain import Assistant
from .context import RecentLog
from .dialog import Dialog, Pending
from .modes import Mode, ModeManager, install_gate
from .patch import PatchStore

__all__ = ["Assistant", "Dialog", "Pending", "Mode", "ModeManager",
           "RecentLog", "PatchStore", "install_gate", "build"]


def build(cfg, profile=None, hub=None, *, config_path: str = "") -> Assistant:
    """给应用装一套：模式管理器 + 现场记录 + 补丁账本 + 闸门。

    闸门**先**装、助手**后**建 —— 顺序反了的话，在装配的这几毫秒里
    执行器是可以自由发动作的（虽然几乎不可能撞上，但这类"几乎"没有意义）。
    """
    asst = Assistant(cfg, profile, hub, config_path=config_path)
    ex = getattr(hub, "executor", None) if hub is not None else None
    if ex is not None:
        install_gate(ex, asst.modes)
    if hub is not None:
        setattr(hub, "assistant", asst)
        setattr(hub, "modes", asst.modes)
        setattr(hub, "recent", asst.recent)
        if profile is not None:
            # 经验候选跟着 profile 走：它是"这个游戏怎么玩"的一部分。
            # 没绑 profile 时不建 —— 候选没地方放，也没有手册可写。
            from ..experience import ExperienceStore

            hub.experience = ExperienceStore(profile)
    return asst
