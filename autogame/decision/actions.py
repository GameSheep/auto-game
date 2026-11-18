"""行为注册表 —— 系统"真能执行"的动作清单，模型只能从里面选。

为什么要有这层：
  直接问大模型"现在该怎么办"，它会给你一段听起来很对的话，然后你没法执行。
  把它约束成"从这份清单里选一个"，情况会好很多，但**清单必须是真的**——
  如果清单里有一个程序执行不了的动作（比如"打开背包找药水"），模型选了它，
  你只会得到一次静默失败，而且看起来像是模型判断错了。

  所以这里的每一条都必须对应 FSM 里一个真实存在的方法调用。谁加行为，谁负责
  在 FSM 的 action_table() 里挂上实现，注册表只描述"长什么样、什么时候可用"。

三条硬规则（写在代码里，不是写在文档里）：
  1. **不允许坐标参数**。目标位置永远由像素证据裁决，模型只选"做什么"，不选"去哪"。
     带 x/y/px 的参数一律视为非法（见 validate_params）。
  2. **风险分级**。reset_stage 这类会丢进度的动作单独标 risky，需要更高置信度。
  3. **可用性由程序判，不由模型猜**。requires 里的条件不满足就不出现在清单里，
     省得模型对着一个当前根本做不了的动作编理由。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 参数白名单里出现这些名字一律拒绝：坐标就是"指路"，不归模型管
_FORBIDDEN_PARAM_HINTS = ("x", "y", "px", "coord", "position", "pos", "target_x", "target_y")

RISK_SAFE = "safe"        # 不会改变游戏状态或可立即撤销
RISK_NORMAL = "normal"    # 常规推进动作
RISK_RISKY = "risky"      # 可能丢进度/花资源，需要更高把握


@dataclass(frozen=True)
class ActionSpec:
    """一条系统能力。id 是模型唯一能引用的东西。"""

    id: str
    label: str
    desc: str
    params: dict[str, str] = field(default_factory=dict)  # 名 -> "int"/"bool"/"str"
    requires: tuple[str, ...] = ()          # probe 里需要为真的键
    risk: str = RISK_NORMAL
    cost_s: float = 1.0                     # 一次执行大致占用多久
    changes_state: bool = True               # 是否会改变游戏局面（决定是否需要对账）

    def available(self, probe: dict[str, Any]) -> bool:
        return all(_truthy(probe.get(k)) for k in self.requires)

    def to_prompt(self) -> str:
        args = ""
        if self.params:
            args = " 参数: " + ", ".join(f"{k}:{v}" for k, v in self.params.items())
        return f"- {self.id}（{self.label}）：{self.desc}{args} [风险:{self.risk}]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "label": self.label, "desc": self.desc,
            "params": dict(self.params), "risk": self.risk,
            "cost_s": self.cost_s,
        }


def _truthy(v: Any) -> bool:
    """requires 判断：0 和空列表算"不可用"，字符串 "0"/"false" 也算不可用。

    为什么不用裸 bool(v)：配置里从 YAML/环境变量灌进来的常常是字符串，
    bool("false") 是 True —— 这种坑在"可用性判断"上代价很高（会放出一个做不了的动作）。
    """
    if isinstance(v, str):
        return v.strip().lower() not in ("", "0", "false", "no", "none", "off")
    return bool(v)


# 系统当前真实支持的行为。新增行为时同步改 fsm.action_table()，否则模型选了会失败。
DEFAULT_ACTIONS: tuple[ActionSpec, ...] = (
    ActionSpec(
        "pickup", "拾取", "按一次拾取键，捡脚下/最近的东西。A 框不在附近时通常无效。",
        risk=RISK_NORMAL, cost_s=0.7, changes_state=True,
    ),
    ActionSpec(
        "approach_nearest", "走向最近目标", "朝最近的掉落光柱走一步（位置由像素证据决定，不是模型决定）。",
        requires=("pillars",), risk=RISK_NORMAL, cost_s=1.2, changes_state=True,
    ),
    ActionSpec(
        "attack_nearest", "攻击最近怪物", "点击最近的怪物，用于清怪、脱困、别站着挨打。",
        requires=("monsters",), risk=RISK_NORMAL, cost_s=1.5, changes_state=True,
    ),
    ActionSpec(
        "explore", "换区域探索", "按预设方向序列走几步，用于没有目标/视野内没东西时脱离当前位置。",
        risk=RISK_NORMAL, cost_s=2.0, changes_state=True,
    ),
    ActionSpec(
        "navigate", "按小地图导航", "按小地图上的目标标记方向走一步，用于找路而不是找掉落。",
        requires=("nav_enabled",), risk=RISK_NORMAL, cost_s=1.0, changes_state=True,
    ),
    ActionSpec(
        "rechallenge", "再次挑战", "点击结算界面的『再次挑战』，用于刷图循环。",
        requires=("rechallenge_visible",), risk=RISK_NORMAL, cost_s=1.5, changes_state=True,
    ),
    ActionSpec(
        "esc", "关闭弹窗/返回", "按一次 ESC，用于关掉挡住画面的弹窗、回到上一层界面。",
        risk=RISK_SAFE, cost_s=0.5, changes_state=True,
    ),
    ActionSpec(
        "wait", "原地观察", "这一帧什么都不做，等画面稳定。人物不动，仅在确定安全时用。",
        params={"frames": "int"}, risk=RISK_SAFE, cost_s=1.0, changes_state=False,
    ),
    ActionSpec(
        "reset_stage", "重开一局", "放弃当前这一局重新开始。会丢失本局进度，最后才考虑。",
        risk=RISK_RISKY, cost_s=4.0, changes_state=True,
    ),
)

_ALWAYS_AVAILABLE = ("wait", "esc", "reset_stage")


class ActionRegistry:
    """行为清单 + 校验。校验失败一律拒绝执行，不做"猜你想要"。"""

    def __init__(self, specs: tuple[ActionSpec, ...] | list[ActionSpec] = DEFAULT_ACTIONS) -> None:
        self._specs: dict[str, ActionSpec] = {s.id: s for s in specs}
        if not self._specs:
            raise ValueError("行为清单不能为空，否则模型永远只能选 others")

    # ---------------- 查询 ----------------

    def get(self, action_id: str) -> ActionSpec | None:
        return self._specs.get(str(action_id or "").strip())

    def all(self) -> list[ActionSpec]:
        return list(self._specs.values())

    def available(self, probe: dict[str, Any]) -> list[ActionSpec]:
        """当前局面下真能做的事。等待/ESC/重开永远活着（任何局面下都得能退出）。"""
        out = []
        for s in self._specs.values():
            if s.id in _ALWAYS_AVAILABLE or s.available(probe):
                out.append(s)
        return out

    def ids(self) -> list[str]:
        return list(self._specs.keys())

    # ---------------- 给模型看 ----------------

    def prompt_block(self, probe: dict[str, Any]) -> str:
        """渲染成清单文本。只列当前可用的，模型不会对着做不了的动作编理由。"""
        avail = self.available(probe)
        lines = [s.to_prompt() for s in avail]
        missing = [s.id for s in self._specs.values() if s not in avail]
        out = "\n".join(lines)
        if missing:
            out += f"\n（当前不可用，不要选：{', '.join(missing)}）"
        return out

    def catalog(self) -> dict[str, Any]:
        return {"actions": [s.to_dict() for s in self._specs.values()]}

    # ---------------- 校验 ----------------

    def validate(self, action_id: str, params: dict[str, Any] | None,
                 probe: dict[str, Any]) -> tuple[bool, str]:
        """返回 (是否可执行, 理由)。理由会进审计，方便事后看模型为什么被拒。"""
        spec = self.get(action_id)
        if spec is None:
            return False, f"动作 {action_id!r} 不在系统能力清单里"
        if not spec.available(probe) and spec.id not in _ALWAYS_AVAILABLE:
            return False, f"动作 {spec.id} 当前不满足前置条件 {spec.requires}"
        ok, why = validate_params(spec, params or {})
        if not ok:
            return False, why
        return True, ""


def validate_params(spec: ActionSpec, params: dict[str, Any]) -> tuple[bool, str]:
    """参数校验：只认 schema 里声明的，且**绝不接受坐标**。

    坐标是最容易被"看起来很合理"地编出来的东西（"向右下方 120px 处"），
    而它一旦被执行就变成模型在指路。所以在入口处硬挡。
    """
    if not isinstance(params, dict):
        return False, "params 必须是对象"
    for key in params:
        low = str(key).lower()
        if any(h in low for h in _FORBIDDEN_PARAM_HINTS):
            return False, f"参数 {key} 看起来是坐标/位置 —— 模型不许指路"
        if key not in spec.params:
            return False, f"动作 {spec.id} 不接受参数 {key}"
    for key, typ in spec.params.items():
        if key not in params:
            continue
        val = params[key]
        if typ == "int":
            if isinstance(val, bool) or not isinstance(val, int):
                return False, f"参数 {key} 必须是整数"
            if not (1 <= val <= 30):
                return False, f"参数 {key} 超出范围 1~30"
        elif typ == "bool":
            if not isinstance(val, bool):
                return False, f"参数 {key} 必须是布尔值"
        elif typ == "str":
            if not isinstance(val, str) or len(val) > 64:
                return False, f"参数 {key} 必须是不超过 64 字的字符串"
    return True, ""
