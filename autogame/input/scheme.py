"""操作方案：一款游戏怎么被操作，写成一份可教、可校验的数据。

## 为什么要这一层

`decision.approach_mode` 只有两档（点地面 / 方向键），配套写死了 `pickup_key`、
`interact_key`、`mash.keys`。这对**火炬之光这类点地面 ARPG** 刚好，因为：

    左键点地面 = 走；右键 = 主攻；QWER = 技能；A = 捡；D = 交互/选目标

换成**魔兽世界这类 WASD + 技能栏**的游戏，同一套字段名会全部指向错的东西：

    W 不是"向上"，是"朝角色面向走"；1-9 才是技能；交互是右键点 NPC；
    拾取要么没有（自动拾取），要么右键点尸体

字段能塞进去，语义塞不进去。所以把"游戏怎么被操作"提成独立的一等对象，
由用户**教一次**（或从预设里挑一份），而不是在代码里一档一档加。

## 硬边界：方案不管"什么时候按"

方案里有 `pickup`，但没有 `pickup_when`。什么时候允许按，仍由决策层的
三道闸（场景门 / A框 / 目标存在）+ `fail_closed` 决定。

这不是保守，是两类知识的性质不同：

  - "A 是捡东西"——可以问出来、可以试出来，错了顶多捡不到；
  - "看到 A 框就按"——错了就是盲按，`blind_tap_violations` 抓的就是这个。

把后者也放进可学习的方案里，等于允许"学"会把铁律学松。所以本文件里
**任何字段都不影响闸门**，只影响"发哪个物理输入"。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------- 物理键名

#: 鼠标类物理键。写成 `mouse_right` 而不是 `rmb`，是因为它要和键盘键一起
#: 放进同一个 `set` 里做冲突检测 —— 两种命名会漏掉"右键既是主攻又是交互"。
MOUSE_KEYS = ("mouse_left", "mouse_right", "mouse_middle", "wheel_up", "wheel_down")

#: 常见写法的归一化。用户填 `RMB`、`右键`、`MouseRight` 都该认。
_KEY_ALIASES = {
    "lmb": "mouse_left", "rmb": "mouse_right", "mmb": "mouse_middle",
    "左键": "mouse_left", "右键": "mouse_right", "中键": "mouse_middle",
    "鼠标左键": "mouse_left", "鼠标右键": "mouse_right", "鼠标中键": "mouse_middle",
    "mouse1": "mouse_left", "mouse2": "mouse_right", "mouse3": "mouse_middle",
    "esc": "escape", "回车": "enter", "空格": "space",
    "左shift": "shift", "左ctrl": "ctrl", "左alt": "alt",
}


def is_mouse_key(key: str) -> bool:
    """这个物理键是不是鼠标动作（鼠标不走 `key_tap` 命令）。"""
    return normalize_key(key) in MOUSE_KEYS


def normalize_key(key: str) -> str:
    """把各种写法统一成 `a` / `f1` / `mouse_right` / `escape`。

    为什么必须归一：冲突检测靠字符串相等。`RMB` 与 `mouse_right` 不相等，
    于是"右键既是主攻又是交互"这种致死冲突会被漏掉 —— 结果是一个右键
    既放技能又想开箱子，日志里只看到"技能放了但没交互"。
    """
    k = str(key or "").strip()
    if not k:
        return ""
    low = k.lower().replace(" ", "").replace("-", "_")
    if low in _KEY_ALIASES:
        return _KEY_ALIASES[low]
    if low in MOUSE_KEYS:
        return low
    return low


# ---------------------------------------------------------------- 单个绑定


Action = Literal["tap", "hold", "mash"]


class Binding(BaseModel):
    """一个能力对应的一次物理输入。

    `action` 的语义（这三档是真实差异，不是风格问题）：

      - `tap`  按一下松开。拾取、交互、开关面板都属这类。
      - `hold` 按住不放，由持续时间决定。有的游戏主攻是"按住连发"，
               按一下只出一刀 —— 用 `tap` 去驱动会表现为"打得很慢但没报错"。
      - `mash` 连按（走 `executor.mash` 的连打模型，含簇长/预算）。
               技能类多半是这种，但**拾取键永远不许 mash**，
               由 `BaseExecutor.set_burst_forbid` 在运行时兜底。
    """

    key: str = Field("", description="物理键名：a / 1 / f1 / mouse_right")
    action: Action = "tap"
    hold_ms: int = Field(60, ge=10, le=3000, description="按住时长（tap/hold 生效）")
    #: 这个动作是否需要先有目标/先面向——留给"能不能做"的判断，
    #: 不用来放宽任何闸门。例如 WoW 的近战技能要求面向目标。
    requires: Literal["none", "target", "facing"] = "none"

    @field_validator("key", mode="before")
    @classmethod
    def _norm(cls, v: Any) -> str:
        return normalize_key(v) if v is not None else ""

    @property
    def is_mouse(self) -> bool:
        return self.key in MOUSE_KEYS

    @property
    def label(self) -> str:
        """给人看/给日志看的名字。"""
        cn = {"mouse_left": "鼠标左键", "mouse_right": "鼠标右键",
              "mouse_middle": "鼠标中键", "wheel_up": "滚轮上", "wheel_down": "滚轮下",
              "escape": "ESC", "space": "空格"}
        return cn.get(self.key, self.key.upper())

    def same_physical(self, other: "Binding | None") -> bool:
        return bool(other) and bool(self.key) and self.key == other.key


class SkillSlot(BaseModel):
    """技能栏一格。

    `id` 是人给的名字（"火球"、"冲刺"），`key` 是物理键。名字重要：
    没有名字的技能在日志里只能显示成"按了 3"，出问题时无法对照录像。
    """

    id: str = ""
    key: str = ""
    action: Action = "tap"
    hold_ms: int = Field(60, ge=10, le=3000)
    role: Literal["primary", "aoe", "buff", "defense", "movement", "unknown"] = "unknown"
    cost: str = Field("", description="资源类型（mana/rage/无），留给人填，程序暂不判")
    #: 冷却秒数。0 = **还不知道**，此时 `skill_ready` 一律放行。
    #: 这是键的属性（和 hold_ms 同类），不是"什么时候按" —— 方案里不许出现 when，
    #: 这条边界不能因为"CD 也算时机"就被推开。不知道就别填，程序宁可按了没效果
    #: （会计进效果账本、攒出假设），也不要按一个猜的 CD 把技能锁死。
    cooldown_s: float = Field(0.0, ge=0, le=600)
    note: str = ""

    @field_validator("key", mode="before")
    @classmethod
    def _norm(cls, v: Any) -> str:
        return normalize_key(v) if v is not None else ""

    @property
    def binding(self) -> Binding:
        return Binding(key=self.key, action=self.action, hold_ms=self.hold_ms)


# ---------------------------------------------------------------- 走位方案


class MoveScheme(BaseModel):
    """角色怎么从 A 点走到 B 点 —— 这是不同游戏差异最大的一环。

    三种形态：

      - `ground_click`：点地面，角色自己走过去（火炬、暗黑、MOBA、RTS）。
        引擎只需要"朝那个方向点一下"，不需要知道朝向。
      - `wasd`：按方向键持续移动（WoW、FPS、绝大多数 3D MMO）。
        **难点不在按键，在视角**：`camera=mouse_look` 时 W 是"朝面向走"，
        不先转视角就会撞墙；只有 `camera=fixed`（俯视 2.5D）才能把
        "屏幕右下"直接翻译成按键组合。
      - `stick`：虚拟摇杆（手游）。需要按住+拖动，当前后端只支持点击，
        所以标为 unsupported 而不是假装能用。
    """

    kind: Literal["ground_click", "wasd", "stick", "none"] = "ground_click"
    #: 点地面/摇杆用哪个物理键。有的 ARPG 把左键留给"选中"，走位要给右键 ——
    #: 写死 left 的后果是所有走位都点错键，表现为"人物偶尔动一下"，很像识别问题。
    button: str = "mouse_left"
    #: 方向键映射。只有 wasd 用得到。
    keys: dict[str, str] = Field(
        default_factory=lambda: {"forward": "w", "back": "s", "left": "a", "right": "d"})
    #: 视角关系：fixed=画面方向即世界方向；mouse_look=视角由鼠标控制。
    camera: Literal["fixed", "mouse_look"] = "fixed"

    @field_validator("kind", mode="before")
    @classmethod
    def _norm_kind(cls, v: Any) -> str:
        """手写时的常见叫法归一。`click` 是人话，`ground_click` 才是字段值。"""
        s = str(v or "").strip().lower().replace("-", "_")
        return {"click": "ground_click", "point": "ground_click",
                "groundclick": "ground_click", "mouse": "ground_click",
                "keyboard": "wasd", "keys": "wasd", "joystick": "stick"}.get(s, s)

    @field_validator("button", mode="before")
    @classmethod
    def _norm_btn(cls, v: Any) -> str:
        return normalize_key(v) if v is not None else "mouse_left"

    @field_validator("keys", mode="before")
    @classmethod
    def _norm_keys(cls, v: Any) -> dict[str, str]:
        if not isinstance(v, dict):
            return {"forward": "w", "back": "s", "left": "a", "right": "d"}
        return {str(k): normalize_key(x) for k, x in v.items()}

    @property
    def move_keys(self) -> set[str]:
        """**会被真的按下去的**移动键。

        `ground_click` 模式下 WASD 不用于走位（探索/导航退化成点地面），
        所以它们不参与冲突检测 —— 否则火炬的"拾取 A / 左移 A"会被
        误报成冲突，而它实际上从来没被当移动键按过。
        """
        if self.kind != "wasd":
            return set()
        return {k for k in (self.keys or {}).values() if k}


# ---------------------------------------------------------------- 冲突


class Conflict(BaseModel):
    """方案内部的一条矛盾。

    `error` = 这么配下去一定会出错，加载时就该拒（fail_closed）；
    `warn`  = 可能是游戏本身就这么设计的，允许，但要在页面上说清楚。
    """

    severity: Literal["error", "warn"] = "warn"
    codes: list[str] = Field(default_factory=list, description="涉及的能力名")
    keys: list[str] = Field(default_factory=list, description="涉及的物理键")
    message: str = ""
    fix: str = ""


#: 能力名单。顺序 = 页面/报告里的展示顺序，也 = 学习向导的提问顺序。
CAPABILITIES = ("move", "primary", "pickup", "interact", "cancel", "map", "skill")

CAPABILITY_CN = {
    "move": "走位",
    "primary": "主攻",
    "pickup": "拾取",
    "interact": "交互",
    "cancel": "返回/关闭",
    "map": "打开地图",
    "skill": "技能",
}


class InputScheme(BaseModel):
    """一份完整的操作方案。per-profile 一份，可以手写，也可以向导问出来。"""

    name: str = ""
    game: str = ""
    #: 这份方案是哪儿来的：preset=内置预设，learned=向导问出来的，
    #: manual=手写，legacy=从旧的 decision.* 字段反推（没经过学习）。
    source: Literal["preset", "learned", "manual", "legacy"] = "manual"

    move: MoveScheme = Field(default_factory=MoveScheme)
    primary: Binding | None = None
    skills: list[SkillSlot] = Field(default_factory=list)
    #: pickup/interact/cancel/map 这些"按一下就完事"的能力
    bindings: dict[str, Binding] = Field(default_factory=dict)

    # ------------------------------------------------------------ 查询

    def binding(self, cap: str) -> Binding | None:
        if cap == "move":
            if self.move.kind == "none":
                return None
            # wasd 没有"一个键"，用前进键当代表（真正按哪些键由 move.keys 决定）。
            # 这里只用于"学过没有"和页面展示，别拿它去发命令。
            key = (self.move.keys.get("forward", "w")
                   if self.move.kind == "wasd" else self.move.button)
            return Binding(key=key)
        if cap == "primary":
            return self.primary if self.primary and self.primary.key else None
        if cap == "skill":
            return self.skills[0].binding if self.skills else None
        b = self.bindings.get(cap)
        return b if b and b.key else None

    def learned(self, cap: str) -> bool:
        return self.binding(cap) is not None

    def all_skills(self) -> list[SkillSlot]:
        """主攻 + 技能栏，按发送顺序。"""
        out: list[SkillSlot] = []
        if self.primary and self.primary.key:
            out.append(SkillSlot(id="主攻", key=self.primary.key,
                                 action=self.primary.action,
                                 hold_ms=self.primary.hold_ms, role="primary"))
        out.extend(s for s in self.skills if s.key)
        return out

    # ------------------------------------------------------------ 冲突

    def conflicts(self) -> list[Conflict]:
        """找出这份方案里会互相踩的地方。

        真实事故的顺序都长得一样：先"能动"，再"偶发"，最后发现是某个键
        一码两用。所以这里宁可多报。
        """
        out: list[Conflict] = []
        move_keys = self.move.move_keys
        func = {c: self.binding(c) for c in ("pickup", "interact", "cancel", "map")}
        func = {c: b for c, b in func.items() if b}

        # ① 功能键撞移动键 —— 这是唯一致死的：WASD 走位时按到 A 就是盲捡。
        for cap, b in func.items():
            if b.key in move_keys:
                out.append(Conflict(
                    severity="error", codes=[cap, "move"], keys=[b.key],
                    message=(f"{CAPABILITY_CN[cap]}键 {b.label} 同时是移动键"
                             f"（走位方案 wasd 的 {sorted(k for k,v in self.move.keys.items() if v == b.key)}）"),
                    fix=("改用点地面走位（move.kind=ground_click），"
                         "或把这款游戏的移动键改成不和功能键重叠的布局")))

        # ② 同一物理键绑两个能力 —— 有的游戏确实这样（D 既交互又选目标），
        #    但**主攻和拾取同键**必然是配错了。
        seen: dict[str, list[str]] = {}
        for cap, b in func.items():
            seen.setdefault(b.key, []).append(cap)
        if self.primary and self.primary.key:
            seen.setdefault(self.primary.key, []).append("primary")
        for sk in self.skills:
            if sk.key:
                seen.setdefault(sk.key, []).append(f"skill:{sk.id or sk.key}")
        for key, caps in seen.items():
            if len(caps) < 2:
                continue
            sev = "error" if ("primary" in caps and "pickup" in caps) else "warn"
            out.append(Conflict(
                severity=sev, codes=caps, keys=[key],
                message=f"{key} 同时绑给了：{'、'.join(caps)}",
                fix=("主攻与拾取不能同键：按下去要么放技能要么捡空气，"
                     "两者都会静默失败" if sev == "error"
                     else "如果游戏本身就这么设计可以忽略，否则建议拆开")))

        # ③ 视角：WASD + 鼠标视角，引擎还不会转视角。
        #    不是"不支持这个游戏"，是"走位这一项还不会"——其余能力仍可用。
        if self.move.kind == "wasd" and self.move.camera == "mouse_look":
            out.append(Conflict(
                severity="warn", codes=["move"], keys=[],
                message=("走位 = WASD + 鼠标视角：引擎目前不会转视角，"
                         "按 W 会朝当前面向走而不是朝目标走"),
                fix=("先只用来跑「原地能做的动作」（交互/技能/开关面板）；"
                     "要走位再加视角控制，或改用可点击的地面")))

        # ④ 走位按钮必须是鼠标键（点地面），否则就是"按键盘当点击"。
        if self.move.kind in ("ground_click", "stick") and not is_mouse_key(self.move.button):
            out.append(Conflict(
                severity="error", codes=["move"], keys=[self.move.button],
                message=f"走位方式 {self.move.kind} 需要鼠标键，但配的是 {self.move.button}",
                fix="move.button 填 mouse_left / mouse_right"))
        return out

    def errors(self) -> list[Conflict]:
        return [c for c in self.conflicts() if c.severity == "error"]

    # ------------------------------------------------------------ 能力表

    def capability_table(self) -> list[dict[str, Any]]:
        """每个能力现在是什么状态、不能用会怎样、怎么补。

        这张表就是页面上"这个游戏能跑到什么程度"的数据源。它必须能
        如实说"不会"，因为"假运行"（界面显示在跑、动作全被拦）是这类
        系统最贵的故障。
        """
        conflicts = self.conflicts()
        blocked = {c for cf in conflicts if cf.severity == "error" for c in cf.codes}
        rows: list[dict[str, Any]] = []
        for cap in CAPABILITIES:
            b = self.binding(cap)
            if cap in blocked:
                state, why, fix = "blocked", "方案内部冲突，启动前必须解决", \
                    next(cf.fix for cf in conflicts if cf.severity == "error" and cap in cf.codes)
            elif b is None:
                state, why, fix = "unknown", "还没教过这个动作，程序不会去按", \
                    "用 tools/learn_input.py 教一次，或在 scheme.yaml 里填上"
            elif cap == "move" and self.move.kind == "wasd" and self.move.camera == "mouse_look":
                state, why, fix = "partial", "能按方向键，但不会转视角（走不准）", \
                    "加视角控制，或改用点地面移动"
            elif cap == "move" and self.move.kind == "stick":
                state, why, fix = "unsupported", "虚拟摇杆需要按住拖动，当前后端只会点按", \
                    "改用点地面，或给后端加拖动原语"
            else:
                state, why, fix = "ready", "", ""
            label = b.label if b else ""
            key = b.key if b else ""
            if cap == "move" and self.move.kind == "wasd":
                label = "WASD"
                key = "/".join(dict.fromkeys(self.move.keys.values()))
            rows.append({
                "capability": cap, "name": CAPABILITY_CN[cap], "state": state,
                "binding": label, "key": key,
                "consequence": why, "fix": fix,
            })
        return rows

    def gaps(self) -> list[str]:
        return [r["capability"] for r in self.capability_table() if r["state"] == "unknown"]

    # ------------------------------------------------------------ 派生到旧字段

    def to_decision_patch(self) -> dict[str, Any]:
        """把方案翻译成决策层现有的扁平字段。

        让 FSM 一行不改就能吃到方案。等 FSM 逐步改成直接读方案之后，
        这个函数仍然有用 —— 它是"兼容层"，不是临时补丁：
        页面/预检/日志都靠它给出"程序实际会按什么"。
        """
        patch: dict[str, Any] = {}
        pk = self.binding("pickup")
        ik = self.binding("interact")
        if pk:
            patch["pickup_key"] = pk.key
        if ik:
            patch["interact_key"] = ik.key
        if self.move.kind == "ground_click":
            patch["approach_mode"] = "mouse_click"
        elif self.move.kind in ("wasd", "stick"):
            patch["approach_mode"] = "keys" if self.move.kind == "wasd" else "mouse_click"
        # 不会走位（没教 / 摇杆后端不支持）→ 关掉走位总闸。
        # 为什么必须显式关：这些游戏里"点地面"不一定是走路（可能是选中目标/取消），
        # 让它继续点，表现是"有人在那儿动，但角色没动"——比不动更难判断。
        patch["move_enabled"] = self.move.kind in ("ground_click", "wasd")
        if self.move.kind in ("ground_click", "stick") and is_mouse_key(self.move.button):
            patch["move_button"] = self.move.button.split("_", 1)[1]   # mouse_left → left
        return patch

    def to_mash_keys(self) -> list[str]:
        """允许连按的键白名单 = 主攻 + 技能键（**不含**拾取/交互）。"""
        keys: list[str] = []
        for sk in self.all_skills():
            if sk.key and not is_mouse_key(sk.key) and sk.key not in keys:
                keys.append(sk.key)
        return keys

    # ------------------------------------------------------------ 与人/文件交换

    def describe(self) -> str:
        """一行摘要，进日志和页面标题。"""
        if self.move.kind == "ground_click":
            mv = f"点地面({self.move.button})"
        elif self.move.kind == "wasd":
            mv = "WASD" + ("+鼠标视角" if self.move.camera == "mouse_look" else "")
        elif self.move.kind == "stick":
            mv = "虚拟摇杆"
        else:
            mv = "不走位"
        parts = [f"走位={mv}"]
        if self.primary and self.primary.key:
            parts.append(f"主攻={self.primary.label}")
        if self.skills:
            parts.append("技能=" + "/".join(s.key.upper() for s in self.skills if s.key))
        for cap in ("pickup", "interact", "cancel", "map"):
            b = self.binding(cap)
            if b:
                parts.append(f"{CAPABILITY_CN[cap]}={b.label}")
        return f"{self.name or self.game}：" + "，".join(parts)

    def to_yaml_dict(self) -> dict[str, Any]:
        """给 `tools/learn_input.py` 落盘用的最小结构（不带空字段）。"""
        out: dict[str, Any] = {"name": self.name, "game": self.game,
                               "source": self.source}
        mv: dict[str, Any] = {"kind": self.move.kind}
        if self.move.kind in ("ground_click", "stick"):
            mv["button"] = self.move.button
        if self.move.kind == "wasd":
            mv["keys"] = dict(self.move.keys)
            mv["camera"] = self.move.camera
        out["move"] = mv
        if self.primary and self.primary.key:
            out["primary"] = {"key": self.primary.key, "action": self.primary.action,
                              "hold_ms": self.primary.hold_ms}
        if self.skills:
            out["skills"] = [
                {"id": s.id, "key": s.key, "action": s.action,
                 **({"role": s.role} if s.role != "unknown" else {}),
                 **({"note": s.note} if s.note else {})}
                for s in self.skills if s.key
            ]
        binds = {k: {"key": v.key, "action": v.action} for k, v in self.bindings.items()
                 if v and v.key}
        if binds:
            out["bindings"] = binds
        return out


# ---------------------------------------------------------------- 内置预设

#: 预设就是"别人已经教过的方案"。选一个当起点再改，比从空白开始快。
PRESETS: dict[str, dict[str, Any]] = {
    # 火炬之光：无限 PC 端。用户的真实键位（本轮对话确认）：
    # 左键走、右键主攻、QWER 辅助、A 捡、D 交互/选目标。
    # 注意 `move.kind=ground_click` 意味着 WASD 不参与走位，所以 A/D 与
    # 移动键不冲突 —— 这是这套键位能安全跑起来的前提。
    "tli_pc": {
        "name": "火炬之光：无限（PC）",
        "game": "火炬之光：无限",
        "source": "preset",
        "move": {"kind": "ground_click", "button": "mouse_left"},
        "primary": {"key": "mouse_right", "action": "hold", "hold_ms": 220},
        "skills": [
            {"id": "技能1", "key": "q", "action": "tap", "role": "aoe"},
            {"id": "技能2", "key": "w", "action": "tap", "role": "aoe"},
            {"id": "技能3", "key": "e", "action": "tap", "role": "defense"},
            {"id": "技能4", "key": "r", "action": "hold", "role": "buff"},
        ],
        "bindings": {
            "pickup": {"key": "a", "action": "tap"},
            "interact": {"key": "d", "action": "tap"},
            "cancel": {"key": "escape", "action": "tap"},
        },
    },
    # 魔兽世界。WASD + 鼠标视角是这类游戏的标配，也是目前引擎的短板：
    # 走位会"按得出来但不准"，所以 `capability_table` 会把它标 partial。
    # 技能栏 1-9 按用户说法"每人设置不同"——所以这里的 1-3 只是占位示例，
    # 真用之前必须用学习向导过一遍。
    "wow": {
        "name": "魔兽世界（默认布局）",
        "game": "魔兽世界",
        "source": "preset",
        "move": {"kind": "wasd", "camera": "mouse_look",
                 "keys": {"forward": "w", "back": "s", "left": "a", "right": "d"}},
        "primary": {"key": "1", "action": "mash"},
        "skills": [
            {"id": "技能2", "key": "2", "action": "tap", "role": "aoe"},
            {"id": "技能3", "key": "3", "action": "tap", "role": "buff"},
        ],
        "bindings": {
            "interact": {"key": "mouse_right", "action": "tap", "requires": "target"},
            "cancel": {"key": "escape", "action": "tap"},
            "map": {"key": "m", "action": "tap"},
        },
    },
}


def preset(name: str) -> InputScheme:
    """取一份内置预设。名字不认识就抛 —— 静默退回默认会让"选错了游戏"表现为乱按。"""
    key = str(name or "").strip().lower()
    if key not in PRESETS:
        raise KeyError(f"没有这个预设：{name}；可选 {sorted(PRESETS)}")
    return InputScheme(**PRESETS[key])


def from_legacy(decision: Any) -> InputScheme:
    """从旧的 `decision.*` 字段反推一份方案。

    用途：老 profile、老 config.runtime.yaml、以及测试里直接构造
    `DecisionConfig` 的地方，都还在用扁平字段。反推出来标 `source=legacy`，
    页面上会提示"这份方案是推断的，建议用向导过一遍"。
    """
    approach = str(getattr(decision, "approach_mode", "mouse_click") or "mouse_click")
    move = (MoveScheme(kind="ground_click", button="mouse_" + str(
        getattr(decision, "move_button", "left") or "left"))
        if approach == "mouse_click" else MoveScheme(kind="wasd", camera="fixed"))
    binds: dict[str, Binding] = {}
    pk = str(getattr(decision, "pickup_key", "") or "")
    ik = str(getattr(decision, "interact_key", "") or "")
    esc = str(getattr(getattr(decision, "autonomy", None), "escape_key", "") or "")
    if pk:
        binds["pickup"] = Binding(key=pk)
    if ik:
        binds["interact"] = Binding(key=ik)
    if esc:
        binds["cancel"] = Binding(key=esc)
    return InputScheme(name="（从旧配置推断）", source="legacy", move=move, bindings=binds)


InputScheme.model_rebuild()


class InputSchemeError(ValueError):
    """方案里有致死冲突 —— 拒绝加载，而不是带着错配跑起来。

    为什么是抛而不是"记个警告继续"：错配的表现是"能动但每个动作都错"。
    比如拾取键和左移同键时，程序会一边走一边捡空气，日志全绿、游戏里
    什么都捡不到；交互键和右移同键时，每往右走一次都开出关卡面板。
    这两种都要人反复看录像才能定位，而报错只要看一眼。
    """

    def __init__(self, conflicts: list[Conflict]) -> None:
        self.conflicts = conflicts
        lines = ["操作方案有 %d 处致死冲突，拒绝加载：" % len(conflicts)]
        for i, c in enumerate(conflicts, 1):
            keys = f"（{', '.join(c.keys)}）" if c.keys else ""
            lines.append(f"  {i}. {c.message}{keys}")
            if c.fix:
                lines.append(f"     → {c.fix}")
        super().__init__("\n".join(lines))
