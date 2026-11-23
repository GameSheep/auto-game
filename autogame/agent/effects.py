"""动作效果：按下去之后，世界变了吗？

## 为什么这是"会玩"和"照脚本按"的分界线

照脚本按的程序只关心"我发出去了"。游戏者关心"发出去之后有没有用"：

    按了拾取，A 框消失 → 有用
    按了拾取，A 框还在 → 这次没用（离太远 / 按错键 / 时机不对）
    按了 D，界面整个变了 → 有用（打开了面板）
    按了主攻，怪还在、血没掉 → 这次没用

没有这一层，程序永远不知道自己按得对不对；有了这一层，"这个键在这款游戏里
到底管不管用"就变成了可累计的证据，而不是靠人看录像。

## 只看语义变化，不看像素差

为什么不做像素差：特效、飘字、光照每秒都在变，像素差永远非零 —— 那样每条
动作都会被判成"有用"，学不到任何东西。这里比的是**场景的语义签名**
（有没有 A 框、怪的数量、界面是不是从战斗切走了、角色位置），
变化必定意味着"游戏对这次输入做出了反应"。

代价是漏掉"纯视觉反馈"（比如技能打出去了但什么都没改变）。这里如实承认：
`no_effect` 的意思是"从画面语义上看不出反应"，不是"绝对没有反应"，
所以它只用来生成**待验证的假设**，不直接改任何配置。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..protocol import SceneState

# ---------------------------------------------------------------- 语义签名

#: 角色位置量化粒度（像素）。真人手抖 2 个像素不该算"世界变了"。
CHAR_QUANT = 16


@dataclass(frozen=True)
class SceneSignature:
    """一帧画面的语义摘要。相等 = 从决策的角度看"什么都没变"。"""

    in_game: bool
    loading: bool
    n_pillars: int
    n_monsters: int
    n_items: int
    n_labels: int
    aframe: bool
    char: tuple[int, int]
    #: 画面变化量的分桶（0..10）。走位有没有效只能靠它 ——
    #: 相机跟随时角色永远在屏幕中心，位置签名不会变。
    motion: int

    @classmethod
    def of(cls, scene: SceneState) -> "SceneSignature":
        ch = scene.character or (0, 0)
        return cls(
            in_game=bool(scene.in_game),
            loading=bool(scene.loading),
            n_pillars=len(scene.pillars),
            n_monsters=len(scene.monsters),
            n_items=len(scene.item_list),
            n_labels=len(scene.ground_labels),
            aframe=scene.aframe is not None,
            char=(int(ch[0]) // CHAR_QUANT, int(ch[1]) // CHAR_QUANT),
            motion=int(max(0.0, min(100.0, float(scene.frame_delta or 0.0))) // 10),
        )

    def same(self, other: "SceneSignature") -> bool:
        return self == other

    def state_key(self) -> tuple:
        """只看"状态"的那部分（不含画面变化量）。

        拾取/攻击这类动作的效果是**状态改变**（东西少了、界面切了）。
        把 motion 算进来会让"特效在飘"冒充"动作生效"。
        """
        return (self.in_game, self.loading, self.n_pillars, self.n_monsters,
                self.n_items, self.n_labels, self.aframe)


#: 效果分类
CHANGED = "changed"          # 世界变了（多半是这次输入生效了）
NO_EFFECT = "no_effect"      # 语义上看不出任何变化
SCENE_FLIP = "scene_flip"    # 战斗界面进/出（开面板、切图、掉线）


def classify(before: SceneSignature, after: SceneSignature,
             *, motion_sensitive: bool = False) -> str:
    """`motion_sensitive`：走位类动作用 True（看画面有没有动），
    其余用 False（只看语义状态有没有变）。"""
    if before.in_game != after.in_game or before.loading != after.loading:
        return SCENE_FLIP
    if before.state_key() != after.state_key():
        return CHANGED
    if motion_sensitive and before.motion != after.motion:
        return CHANGED
    return NO_EFFECT


def situation_of(scene: SceneState) -> str:
    """这次动作是在什么情境下做的。

    账本按"动词 + 情境"分开记：`attack` 在战斗里没效果是真问题，
    在空地上没效果则完全正常（本来就没东西可打）。
    """
    if not scene.in_game:
        return "out_of_game"
    return "combat" if scene.monsters else "ground"


# ---------------------------------------------------------------- 账本


@dataclass
class Tally:
    attempts: int = 0
    changed: int = 0
    no_effect: int = 0
    scene_flip: int = 0

    def to_dict(self) -> dict:
        return {"attempts": self.attempts, "changed": self.changed,
                "no_effect": self.no_effect, "scene_flip": self.scene_flip}


#: 没效果时该往哪儿查。写具体一点，不然这条假设等于没说。
_DEAD_HINT = {
    "pickup": "先确认拾取键（tools/learn_input.py 的 --capture 能抓真实按键），"
              "再确认这一步的 until 写的是不是「A 框消失」——按了没消失说明人还在拾取半径外",
    "attack": "先确认主攻键是「按住」还是「连点」（action 写错会表现为打得很慢但没报错），"
              "再确认目标选择器是否真的选中了怪",
    "move_to": "走位没动：检查 move.kind 与实际游戏是否一致（点地面 / WASD），"
               "WASD + 鼠标视角的游戏还要先解决转视角",
    "interact": "交互没反应：确认交互键，以及人是否真的站到了目标旁边（目标选择器可能选错了那个）",
    "use_skill": "技能没反应：确认这一格在游戏里还有蓝/不在 CD，或这格根本不是技能",
    "open_panel": "面板没打开：确认面板键，以及当前是否处在允许打开面板的界面",
}


@dataclass
class EffectLedger:
    """"哪个动作在这个情境下一直没效果"的累计账本。

    它**只累计证据**，不改配置、不改键位。攒够了由 `dead()` 吐出一条
    待验证假设，人确认之后再走已有的补丁通道（可回滚）。
    这条边界是为了守住"学不许把铁律学松"：账本能说"这个键好像没用"，
    不能自己决定"那换个键试试"。
    """

    dead_after: int = 3
    tallies: dict[tuple[str, str], Tally] = field(default_factory=dict)

    def record(self, verb: str, situation: str, effect: str) -> None:
        key = (verb, situation)
        t = self.tallies.setdefault(key, Tally())
        t.attempts += 1
        if effect == CHANGED:
            t.changed += 1
        elif effect == SCENE_FLIP:
            t.scene_flip += 1
        else:
            t.no_effect += 1

    def dead(self) -> list[dict]:
        """试了 `dead_after` 次以上、一次都没让世界变化的组合。"""
        out: list[dict] = []
        for (verb, situation), t in sorted(self.tallies.items()):
            if t.attempts >= self.dead_after and t.changed == 0 and t.scene_flip == 0:
                out.append({
                    "verb": verb, "situation": situation,
                    "attempts": t.attempts, "no_effect": t.no_effect,
                    "hypothesis": f"「{verb}」在「{situation}」情境下按了 {t.attempts} 次，"
                                  f"画面一次都没变",
                    "hint": _DEAD_HINT.get(verb, "先核对键位，再核对这一步的判据"),
                })
        return out

    def to_dict(self) -> dict[str, dict]:
        return {f"{v}@{s}": t.to_dict() for (v, s), t in sorted(self.tallies.items())}
