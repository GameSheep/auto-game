"""语义动作（verb）—— 一个"游戏者"的动作词汇，和具体游戏无关。

## 为什么要有这一层

上一层的 `InputScheme` 只回答"哪个物理输入 = 哪个动作"，它仍然是**按**的粒度：
`key_tap a`。可一个"游戏者"不会想"我要按 A"，它想的是"我要把这个东西捡起来"。

所以这里把动作提成语义动词（move_to / attack / pickup / interact / open_panel /
use_skill / cancel / wait），每个动词做两件事：

  1. **判断此刻允不允许做**（证据齐不齐）。不齐就拒绝，并说清缺什么。
  2. 允许就产出 `Command` 列表，键位从方案里取。

动词**不执行**命令（执行器负责），也**不判断做没做成**（`effects.py` 负责）。
这三件事分开，才能在没有真实游戏时把"它想做什么"完整测出来。

## 硬边界：证据是动词自带的，不是调用方给的

`pickup` 的证据是 `("in_game", "target", "aframe")`，写死在动词里，技能文件
**无法关掉**。这一点是刻意的：技能是"教"出来的知识，知识可以错、可以被改，
而"没看到 A 框不许按拾取键"不能因为某个 YAML 写得随意就被绕过。

结果就是：技能文件里写 `verb: pickup` 而没有 A 框时，动作会被拒并记进
`blocked_evidence`，而不是"按一下试试"。少捡一件东西，好过捡一晚上空气。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..input.scheme import InputScheme, is_mouse_key
from ..protocol import Command, SceneState

# ---------------------------------------------------------------- 上下文


@dataclass
class VerbContext:
    """动词执行需要的外部事实（都是"这个游戏怎么被操作"，不含时机）。"""

    scheme: InputScheme
    #: 走位总闸。方案说不会走位（没教 / 摇杆后端不支持）时，move_to 必须停手。
    move_enabled: bool = True
    #: 判定"到了"的半径（像素）。跟拾取半径不是一回事：拾取半径由游戏决定，
    #: 这个只是"还要不要继续挪"。
    reach_px: int = 120
    #: 上一次解析到的目标点。给 `until: target_gone` 用 —— 目标消失才算捡到了。
    last_target: tuple[int, int] | None = None
    #: 探索方向轮换的下标。放上下文里是为了可复现（随机数在测试里没法断言）。
    roam_index: int = 0
    #: 按键用的帧最多能有多旧（毫秒）。0 = 不检查（离线回放）。
    #: 这是 `decision.press_max_frame_age_ms` 在游戏者层的同一个闸：
    #: 动词必须自己看帧龄，不能指望调用方记得检查 —— A 框是"过去某一帧在范围内"，
    #: 帧一旧人就可能已经走过去了（真实事故：用 0.8 秒前的 A 框按了 A）。
    max_frame_age_ms: float = 400.0
    #: 上一次动作针对的目标类型与"发出那一刻各类目标的数量"。
    #: `until: target_gone` 靠它判断"那个东西没了"——
    #: 不能拿屏幕坐标比：相机一跟，目标在屏幕上就跑了，会被误判成"消失"。
    last_kind: str = ""
    last_counts: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)
    #: 当前帧时间（秒，单调钟）。放这里是因为「技能还在不在 CD」需要它，
    #: 而动词是纯函数、拿不到外面的时钟 —— 让调用方每帧写一次，比让动词自己
    #: `time.monotonic()` 强：时间可注入，测试才能断言"3 秒后能放"。
    now: float = 0.0
    #: 判据读数（`cues.yaml` 里每条 cue 的 0..1 值）。
    #: 没有画面时是空 dict —— 条件 `cue` 于是永远不成立，而不是"默认允许"。
    cues: dict[str, float] = field(default_factory=dict)
    #: 这一帧成立的 cue 名集合（阈值已经判过）
    cue_on: set[str] = field(default_factory=set)
    #: 最后一次放这个技能的时刻（键 → 秒）。`skill_ready` 靠它判 CD。
    #: 键而不是 id：同一个键可能挂两个技能（切换），CD 是键的属性。
    last_use: dict[str, float] = field(default_factory=dict)

    def cue(self, name: str) -> float:
        return float(self.cues.get(str(name or "").strip(), 0.0))


# ---------------------------------------------------------------- 结果


@dataclass
class Outcome:
    """一次动词尝试的结果。

    `blocked` 是这次设计里最重要的字段：

      - `"capability"` 这个游戏里这一项**根本做不到**（没教 / 后端不支持）。
        上层必须**中止技能**，不能重试到超时 —— 重试只会产出"看起来很努力"的假动作。
      - `"evidence"` 此刻条件不满足（A 框还没出现、帧没到）。可以下一帧再看。
      - `""` 发出去了。
    """

    ok: bool = False
    commands: list[Command] = field(default_factory=list)
    reason: str = ""
    blocked: str = ""
    #: 被哪条证据挡下的（`aframe` / `target` ...）。上层要按证据种类分别计数，
    #: 因为"没 A 框所以没按"和"没目标所以没按"是完全不同的两件事。
    evidence: str = ""
    note: str = ""
    target: tuple[int, int] | None = None

    @property
    def blocked_capability(self) -> bool:
        return self.blocked == "capability"

    @property
    def blocked_evidence(self) -> bool:
        return self.blocked == "evidence"


# ---------------------------------------------------------------- 目标解析


#: 已知的目标选择器。YAML 里写错（比如 `nearest_pillar ` 多了空格、
#: 或写成 `pillar`）会解析成 None，表现是「这一步永远在等目标」——
#: 所以 lint 拿这个集合做白名单。
TARGET_KINDS = ("nearest_pillar", "nearest_monster", "item", "label",
                "aframe", "point", "none")


def resolve_target(scene: SceneState, params: dict[str, Any],
                   ctx: VerbContext | None = None) -> tuple[int, int] | None:
    """把"打到哪/走到哪/捡哪个"解析成一个屏幕点。

    只认几种选择器，不猜：`nearest_pillar` / `nearest_monster` / `item` /
    `label` / `point`。没有目标就返回 None —— 由调用方拒绝，而不是退化成
    "点屏幕中心碰碰运气"。
    """
    kind = str(params.get("target") or "nearest_pillar").strip().lower()
    ch = scene.character or (scene.w // 2, scene.h // 2)

    def dist(p: tuple[int, int]) -> float:
        return ((p[0] - ch[0]) ** 2 + (p[1] - ch[1]) ** 2) ** 0.5

    if kind == "point":
        if params.get("x") is None or params.get("y") is None:
            return None
        return int(params["x"]), int(params["y"])

    if kind == "nearest_pillar":
        # 光柱用底部中心（x,y）—— 那是地面点，点它才是"走过去"
        pts = [(p.x, p.y) for p in scene.pillars]
    elif kind == "nearest_monster":
        pts = [(m.x + m.w // 2, m.y + m.h // 2) for m in scene.monsters]
    elif kind == "item":
        pts = [(e.x, e.y) for e in scene.item_list]
    elif kind == "label":
        pts = [(e.x, e.y) for e in scene.ground_labels]
    elif kind == "aframe":
        a = scene.aframe
        return (a.x + a.w // 2, a.y + a.h // 2) if a else None
    elif kind == "none":
        return None
    else:
        return None
    if not pts:
        return None
    return min(pts, key=dist)


# ---------------------------------------------------------------- 证据


def _ev_in_game(scene: SceneState, ctx: VerbContext, p: dict) -> tuple[bool, str]:
    if scene.in_game:
        return True, ""
    return False, f"场景门未通过（{scene.gate_reason or '未识别界面'}）"


def _ev_not_loading(scene: SceneState, ctx: VerbContext, p: dict) -> tuple[bool, str]:
    return (not scene.loading), "正在加载/过场"


def _ev_target(scene: SceneState, ctx: VerbContext, p: dict) -> tuple[bool, str]:
    if resolve_target(scene, p, ctx):
        return True, ""
    return False, "画面里没有这个动作的目标"


def _ev_fresh(scene: SceneState, ctx: VerbContext, p: dict) -> tuple[bool, str]:
    limit = float(getattr(ctx, "max_frame_age_ms", 0.0) or 0.0)
    age = float(getattr(scene, "age_ms", 0.0) or 0.0)
    if limit <= 0 or age <= 0:
        return True, ""          # 不检查 / 离线回放（没有"现在"可言）
    if age <= limit:
        return True, ""
    return False, f"这帧是 {age:.0f}ms 前的（上限 {limit:.0f}ms）—— 等一帧新的"


def _ev_aframe(scene: SceneState, ctx: VerbContext, p: dict) -> tuple[bool, str]:
    if scene.aframe is not None:
        return True, ""
    return False, "没有 A 框（拾取提示）—— 没证据，不按"


#: 证据名 → 判定函数。名字是 YAML 里能写的封闭集合，写错了 lint 会报。
EVIDENCE: dict[str, Callable[[SceneState, VerbContext, dict], tuple[bool, str]]] = {
    "in_game": _ev_in_game,
    "fresh": _ev_fresh,
    "not_loading": _ev_not_loading,
    "target": _ev_target,
    "aframe": _ev_aframe,
}

EVIDENCE_CN = {
    "in_game": "战斗界面（场景门）",
    "fresh": "帧够新",
    "not_loading": "不在加载",
    "target": "有目标",
    "aframe": "A 框在",
}

# ---------------------------------------------------------------- 命令构造


def target_counts(scene: SceneState) -> tuple[int, int, int, int, int]:
    """(光柱, 怪, 列表条目, 地面标签, A框) 的数量。"""
    return (len(scene.pillars), len(scene.monsters), len(scene.item_list),
            len(scene.ground_labels), 1 if scene.aframe is not None else 0)


_COUNT_INDEX = {"nearest_pillar": 0, "nearest_monster": 1, "item": 2,
                "label": 3, "aframe": 4}


def _button_of(key: str) -> str:
    """`mouse_right` → `right`（Command 的 button 字段只有 left/right）。"""
    return "right" if key.endswith("right") else "left"


def press(key: str, action: str = "tap", hold_ms: int = 60,
          x: int | None = None, y: int | None = None) -> list[Command]:
    """按一次（键或鼠标）。

    `hold` 也走 `key_tap` + 长 `hold_ms`，**不用 key_down 留在按下状态**：
    发一个 key_down 然后等下一帧再 key_up，中间只要掉线/切窗/急停，键就卡住了，
    游戏里表现为角色一直往前跑。按住连发的语义由"每个 tick 重发一次 key_tap"实现，
    物理效果一样，但没有卡键的可能。
    """
    if not key:
        return []
    if is_mouse_key(key):
        return [Command(op="mouse_click", button=_button_of(key), x=x, y=y)]
    return [Command(op="key_tap", key=key, hold_ms=int(hold_ms))]


def _dir_keys(ctx: VerbContext, dx: float, dy: float) -> list[str]:
    """屏幕方向 → 要按的键（wasd）。拾取/交互键永远不许出现在这里。"""
    mk = ctx.scheme.move.keys or {}
    forbid = {b.key for b in ctx.scheme.bindings.values() if b and b.key}
    keymap = {k: v for k, v in mk.items()
              if v and v not in forbid and v != ctx.scheme.move.button}
    out: list[str] = []
    if dy < -1 and "forward" in keymap:
        out.append(keymap["forward"])
    if dy > 1 and "back" in keymap:
        out.append(keymap["back"])
    if dx < -1 and "left" in keymap:
        out.append(keymap["left"])
    if dx > 1 and "right" in keymap:
        out.append(keymap["right"])
    return out


def _find_skill(scheme: InputScheme, p: dict) -> Any:
    want = str(p.get("slot") or "").strip().lower()
    for slot in scheme.all_skills():
        if not want:
            return slot
        if want in (slot.id.lower(), slot.key.lower()):
            return slot
    return None


# ---------------------------------------------------------------- 各动词实现


def _run_move_to(scene: SceneState, ctx: VerbContext, p: dict) -> Outcome:
    if not ctx.move_enabled:
        return Outcome(blocked="capability",
                       reason="走位还没教 / 后端不支持（move_enabled=false）")
    pt = resolve_target(scene, p, ctx)
    if pt is None:
        return Outcome(blocked="evidence", reason="没有可走的目标")

    kind = ctx.scheme.move.kind
    if kind == "ground_click":
        return Outcome(ok=True, target=pt,
                       commands=[Command(op="mouse_click",
                                         button=_button_of(ctx.scheme.move.button),
                                         x=pt[0], y=pt[1])])
    if kind == "wasd":
        ch = scene.character
        dx, dy = pt[0] - ch[0], pt[1] - ch[1]
        keys = _dir_keys(ctx, dx, dy)
        if not keys:
            return Outcome(blocked="evidence", reason="已经站在目标上了（没有方向要按）",
                           target=pt)
        note = ""
        if ctx.scheme.move.camera == "mouse_look":
            # 不拦，但要如实说：W 是"朝面向走"，不转视角就会走偏。
            note = "视角未控制，方向可能偏"
        cmds: list[Command] = []
        for k in keys:
            cmds += press(k, "tap", 90)
        return Outcome(ok=True, commands=cmds, target=pt, note=note)
    return Outcome(blocked="capability",
                   reason=f"走位方式 {kind} 没有可用的动作原语")


def _run_attack(scene: SceneState, ctx: VerbContext, p: dict) -> Outcome:
    b = ctx.scheme.primary
    if b is None or not b.key:
        return Outcome(blocked="capability", reason="主攻还没教（方案里没有 primary）")
    want = p.get("target") not in (None, "", "none")
    pt = resolve_target(scene, p, ctx) if want else None
    if want and pt is None:
        # 鼠标主攻会"点目标"，没有目标就点不出坐标 —— 不能退化成点当前光标：
        # 那等于朝上一次鼠标停的地方挥一刀，日志里看不出任何异常。
        return Outcome(blocked="evidence", evidence="target", reason="没有可打的目标")
    if is_mouse_key(b.key):
        return Outcome(ok=True, target=pt,
                       commands=press(b.key, b.action, b.hold_ms,
                                      pt[0] if pt else None, pt[1] if pt else None))
    return Outcome(ok=True, target=pt, commands=press(b.key, b.action, b.hold_ms))


def _run_pickup(scene: SceneState, ctx: VerbContext, p: dict) -> Outcome:
    b = ctx.scheme.bindings.get("pickup")
    if b is None or not b.key:
        return Outcome(blocked="capability",
                       reason="拾取键还没教 —— 程序不会猜一个键去按")
    a = scene.aframe
    assert a is not None  # 证据闸已保证；这里取不到就是逻辑错误，宁可炸
    cx, cy = a.x + a.w // 2, a.y + a.h // 2
    return Outcome(ok=True, target=(cx, cy),
                   commands=press(b.key, "tap", b.hold_ms, cx, cy))


def _run_interact(scene: SceneState, ctx: VerbContext, p: dict) -> Outcome:
    b = ctx.scheme.bindings.get("interact")
    if b is None or not b.key:
        return Outcome(blocked="capability",
                       reason="交互键还没教（宝箱/传送门/NPC 都做不了）")
    pt = resolve_target(scene, p, ctx) if p.get("target") not in (None, "", "none") else None
    return Outcome(ok=True, target=pt,
                   commands=press(b.key, b.action, b.hold_ms,
                                  pt[0] if pt else None, pt[1] if pt else None))


def _run_open_panel(scene: SceneState, ctx: VerbContext, p: dict) -> Outcome:
    """打开面板（关卡选择/背包）。

    和 `interact` 分开是因为它**没有目标可依**：面板键是全局键。这是本层唯一
    允许"无目标按键"的动词，所以 `risky=True`，lint 会要求技能写出 `when`。
    """
    b = ctx.scheme.bindings.get("interact") or ctx.scheme.bindings.get("map")
    if b is None or not b.key:
        return Outcome(blocked="capability", reason="打开面板的键还没教")
    return Outcome(ok=True, note="无目标按键（面板键）",
                   commands=press(b.key, "tap", b.hold_ms))


def _run_use_skill(scene: SceneState, ctx: VerbContext, p: dict) -> Outcome:
    slot = _find_skill(ctx.scheme, p)
    if slot is None:
        return Outcome(blocked="capability",
                       reason=f"方案里没有这个技能：{p.get('slot') or '(未指定)'}")
    # 记账：`skill_ready` 条件靠它判 CD。记在**发出去之前**：
    # 命令已经交给执行器了，此刻就是"用掉了"，等回执再记会让 CD 晚一帧。
    ctx.last_use[slot.key] = ctx.now
    return Outcome(ok=True,
                   commands=press(slot.key, slot.action, slot.hold_ms))


def _run_roam(scene: SceneState, ctx: VerbContext, p: dict) -> Outcome:
    """没有目标时朝某个方向走一段（找路）。

    为什么不复用 `move_to`：`move_to` 必须有目标，这是它的安全属性 —— 没有目标
    就不该"往那儿点一下"。而探索恰恰是"没有目标也要走"，所以单独一个动词，
    把"我可以无目标移动"这件事**显式**写出来，而不是给 move_to 开个后门。
    """
    if not ctx.move_enabled:
        return Outcome(blocked="capability",
                       reason="走位还没教 / 后端不支持（move_enabled=false）")
    # 方向按顺序轮换而不是随机：随机在测试里不可复现，而"四个方向都走过"
    # 才是探索该有的样子 —— 老在一个角上打转看起来也像在动。
    dirs = ("right", "down", "left", "up")
    duty = str(p.get("direction") or "").strip().lower()
    if duty not in dirs:
        duty = dirs[ctx.roam_index % len(dirs)]
        ctx.roam_index += 1

    if ctx.scheme.move.kind == "ground_click":
        w, h = max(2, scene.w), max(2, scene.h)
        # 点屏幕边缘而不是屏幕中心：中心可能正好是角色自己/可拾取物，
        # 点它会变成"选中"而不是"走过去"。
        frac = {"right": (0.85, 0.5), "left": (0.15, 0.5),
                "up": (0.5, 0.2), "down": (0.5, 0.8)}[duty]
        return Outcome(ok=True, note=f"探索:{duty}",
                       commands=[Command(op="mouse_click",
                                         button=_button_of(ctx.scheme.move.button),
                                         x=int(w * frac[0]), y=int(h * frac[1]))])
    if ctx.scheme.move.kind == "wasd":
        keys = _dir_keys(ctx, *{"right": (1, 0), "left": (-1, 0),
                                "up": (0, -1), "down": (0, 1)}[duty])
        if not keys:
            return Outcome(blocked="capability", reason="方向键没配上")
        return Outcome(ok=True, note=f"探索:{duty}",
                       commands=[Command(op="key_tap", key=k, hold_ms=140) for k in keys])
    return Outcome(blocked="capability", reason=f"走位方式 {ctx.scheme.move.kind} 无法探索")


def _run_cancel(scene: SceneState, ctx: VerbContext, p: dict) -> Outcome:
    b = ctx.scheme.bindings.get("cancel")
    if b is None or not b.key:
        return Outcome(blocked="capability", reason="返回/关闭键还没教")
    return Outcome(ok=True, commands=press(b.key, "tap", b.hold_ms))


def _run_wait(scene: SceneState, ctx: VerbContext, p: dict) -> Outcome:
    return Outcome(ok=True, commands=[], reason="等")


# ---------------------------------------------------------------- 注册表


@dataclass(frozen=True)
class Verb:
    name: str
    capability: str
    needs: tuple[str, ...]
    desc: str
    run: Callable[[SceneState, VerbContext, dict], Outcome]
    #: 无目标按键（面板类）。lint 会要求技能写出 `when`，避免战斗中盲按。
    risky: bool = False
    #: 这个动词期望"世界有变化"（wait 不期望）。effects 只对期望变化的动词记账。
    expect_effect: bool = True
    #: 效果口径：motion=位移（走位），state=状态改变（捡到/打开/怪变少）。
    #: 为什么要分：相机跟随时角色恒在屏幕中心，走位的效果只能看"画面在动"；
    #: 而拾取若看"画面在动"，飘个特效就会被当成"捡到了"。
    effect_kind: str = "state"

    def perform(self, scene: SceneState, ctx: VerbContext, params: dict | None = None) -> Outcome:
        p = params or {}
        for ev in self.needs:
            fn = EVIDENCE.get(ev)
            if fn is None:  # 理论上 lint 会先拦；这里再兜一次，防止绕过 lint
                return Outcome(blocked="capability", reason=f"未知证据 {ev}")
            ok, why = fn(scene, ctx, p)
            if not ok:
                # 用证据函数自己写的句子，不再加"XX 不满足："前缀 ——
                # 拼起来会变成"有目标：画面里没有目标"这种自相矛盾的话
                return Outcome(blocked="evidence", evidence=ev,
                               reason=why or f"{EVIDENCE_CN.get(ev, ev)}不满足")
        out = self.run(scene, ctx, p)
        if out.target is not None:
            ctx.last_target = out.target
            ctx.last_counts = target_counts(scene)
            ctx.last_kind = "aframe" if self.name == "pickup" else str(p.get("target") or "")
        return out


VERBS: dict[str, Verb] = {
    "move_to": Verb(
        name="move_to", capability="move",
        needs=("in_game", "not_loading", "fresh", "target"),
        desc="走到目标（光柱/怪/掉落/指定点）",
        run=_run_move_to, effect_kind="motion"),
    "roam": Verb(
        name="roam", capability="move",
        # 无目标移动：只要求"在游戏里"，不要求有目标。这是探索的唯一入口，
        # 所以它必须单独存在 —— 给 move_to 开后门就等于取消了它的安全属性。
        needs=("in_game", "not_loading"),
        desc="没有目标时朝一个方向走一段（探索）",
        run=_run_roam, effect_kind="motion"),
    "attack": Verb(
        name="attack", capability="primary",
        needs=("in_game", "not_loading", "fresh"),
        desc="用主攻打（鼠标则点目标，键则按/连按）",
        run=_run_attack),
    "pickup": Verb(
        name="pickup", capability="pickup",
        # 三道闸在这里，别改成两道的：没有 A 框就按 = 盲捡
        needs=("in_game", "not_loading", "fresh", "target", "aframe"),
        desc="捡起 A 框指示的东西",
        run=_run_pickup),
    "interact": Verb(
        name="interact", capability="interact",
        # 交互也要目标：站在箱子/传送门旁边才按，不站在那儿按就是盲按
        needs=("in_game", "not_loading", "fresh", "target"),
        desc="对目标交互（开箱/进门/NPC）",
        run=_run_interact),
    "open_panel": Verb(
        name="open_panel", capability="interact",
        needs=("in_game", "not_loading"),
        desc="打开面板（关卡选择/背包）—— 无目标按键，需谨慎",
        run=_run_open_panel, risky=True),
    "use_skill": Verb(
        name="use_skill", capability="skill",
        # 技能的"配没配"由 run 判成能力缺失（该去改方案），不是"时机不对"。
        # 放进证据会让"没教这个技能"看起来像"等一下再来"，然后一直等。
        needs=("in_game", "not_loading", "fresh"),
        desc="放一个已配置的技能",
        run=_run_use_skill),
    "cancel": Verb(
        name="cancel", capability="cancel",
        # 刻意不要求 in_game：面板打开时恰恰是场景门不通过的时候
        needs=(),
        desc="返回/关闭面板",
        run=_run_cancel),
    "wait": Verb(
        name="wait", capability="",
        needs=(),
        desc="等一会儿（不按键）",
        run=_run_wait, expect_effect=False),
}

#: 需要 `slot` 参数的动词（lint 检查用）
NEEDS_SLOT = ("use_skill",)
#: 需要目标选择器的动词（lint 检查用；`target=point` 时还要 x/y）
NEEDS_TARGET = ("move_to",)
