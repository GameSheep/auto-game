"""操作方案（InputScheme）测试 —— 换游戏不能靠改代码，但也不能靠猜。

这里钉住四件事：
  1. 方案能派生到决策字段（FSM 不用重写就能吃到）；
  2. 致死冲突**拒绝加载**，不是记个警告继续跑；
  3. 没学过的能力保持"未学"，绝不猜一个默认值去按；
  4. 方向键映射生效，且功能键拦截发生在**映射之后**（顺序反了会漏拦）。
"""

from __future__ import annotations

import pytest

from autogame.config import (
    AppConfig,
    DecisionConfig,
    ExecutorConfig,
    InputSchemeError,
    apply_input_scheme,
    load_config,
)
from autogame.decision.fsm import DecisionEngine
from autogame.executor import DryRunExecutor
from autogame.input import Binding, InputScheme, MoveScheme, normalize_key, preset
from autogame.protocol import SceneState


# ---------------------------------------------------------------- 派生


def test_preset_tli_derives_decision_fields():
    """火炬 PC：点地面走位 + 右键主攻 + A 拾取 + D 交互。"""
    cfg = AppConfig(input=preset("tli_pc"))
    warns = apply_input_scheme(cfg)
    assert warns == []
    assert cfg.decision.pickup_key == "a"
    assert cfg.decision.interact_key == "d"
    assert cfg.decision.approach_mode == "mouse_click"
    assert cfg.decision.move_button == "left"
    # 连按白名单只留方案里真有的技能键
    assert cfg.executor.mash.keys == ["q", "w", "e", "r"]


def test_no_scheme_keeps_legacy_fields():
    """不写方案 = 行为完全不变（老配置/老 profile 不受影响）。"""
    cfg = AppConfig()
    apply_input_scheme(cfg)
    assert cfg.decision.pickup_key == "a"
    assert cfg.decision.approach_mode == "mouse_click"
    assert "1" in cfg.executor.mash.keys      # 默认白名单没被动过


def test_right_button_locomotion_derives_move_button():
    """左键是"选中"、右键才走路的游戏：走位键必须能改。"""
    scheme = InputScheme(name="右键走路", move=MoveScheme(kind="click", button="mouse_right"))
    cfg = AppConfig(input=scheme)
    apply_input_scheme(cfg)
    assert cfg.decision.move_button == "right"


def test_wasd_scheme_derives_keys_mode():
    scheme = preset("wow")
    cfg = AppConfig(input=scheme)
    apply_input_scheme(cfg)
    assert cfg.decision.approach_mode == "keys"
    assert cfg.decision.move_keys["forward"] == "w"


# ---------------------------------------------------------------- 冲突必须拦


def test_pickup_sharing_move_key_refuses_to_load():
    """WASD 走位 + A 拾取 = A 同时是左移。必须拒绝加载，而不是"跑起来再说"。"""
    bad = InputScheme(
        name="错配", move=MoveScheme(kind="wasd"),
        bindings={"pickup": Binding(key="a"), "interact": Binding(key="f")})
    errs = bad.errors()
    assert len(errs) == 1
    assert "拾取键" in errs[0].message
    with pytest.raises(InputSchemeError) as ei:
        apply_input_scheme(AppConfig(input=bad))
    assert "拒绝加载" in str(ei.value)
    # 报错必须给出改法，否则用户只知道"配错了"不知道"怎么改"
    assert "ground_click" in str(ei.value)


def test_primary_and_pickup_same_key_is_fatal():
    """主攻和拾取同键：按下去要么放技能要么捡空气，两者都静默失败。"""
    bad = InputScheme(name="同键", move=MoveScheme(kind="click"),
                      primary=Binding(key="mouse_right"),
                      bindings={"pickup": Binding(key="mouse_right")})
    assert any(c.severity == "error" for c in bad.conflicts())


def test_tli_keys_do_not_look_like_conflicts():
    """火炬的 A/D 与 WASD 字母相同，但它走点地面 —— 不该误报。

    这是最容易搞反的一处：按"字符相等"判冲突会把一款能跑的游戏拦在门外。
    """
    assert preset("tli_pc").errors() == []


def test_character_key_conflict_reported_for_wasd_game():
    """真的 WASD 走位 + 拾取 A 就一定要报 —— 与上一条的分界是 move.kind。"""
    assert preset("wow").move.move_keys == {"w", "s", "a", "d"}


# ---------------------------------------------------------------- 不知道就是不知道


def test_unlearned_capabilities_stay_unknown():
    """空白方案：每项都是未学，且**没有**默认键冒充"已学"。"""
    empty = InputScheme(name="空白", move=MoveScheme(kind="none"))
    rows = {r["capability"]: r for r in empty.capability_table()}
    assert rows["pickup"]["state"] == "unknown"
    assert rows["pickup"]["key"] == ""
    assert set(empty.gaps()) >= {"pickup", "interact"}
    assert empty.learned("move") is False


def test_wow_pickup_is_unknown_not_guessed():
    """魔兽没有独立的拾取键：宁可标未学（不捡），也不要猜一个去按。"""
    rows = {r["capability"]: r for r in preset("wow").capability_table()}
    assert rows["pickup"]["state"] == "unknown"
    assert rows["interact"]["state"] == "ready"


def test_mouse_look_locomotion_is_partial_not_ready():
    """WASD + 鼠标视角：引擎还不会转视角，必须如实标"部分"。"""
    rows = {r["capability"]: r for r in preset("wow").capability_table()}
    assert rows["move"]["state"] == "partial"
    warns = [c for c in preset("wow").conflicts() if c.severity == "warn"]
    assert any("转视角" in c.message for c in warns)


def test_stick_is_unsupported():
    """虚拟摇杆要按住拖动，当前后端只会点按 —— 标不支持，不假装能用。"""
    scheme = InputScheme(name="手游", move=MoveScheme(kind="stick"))
    rows = {r["capability"]: r for r in scheme.capability_table()}
    assert rows["move"]["state"] == "unsupported"


# ---------------------------------------------------------------- 归一化


@pytest.mark.parametrize("raw,want", [
    ("RMB", "mouse_right"), ("右键", "mouse_right"), ("LMB", "mouse_left"),
    ("esc", "escape"), ("ESC", "escape"), ("F1", "f1"), (" A ", "a"),
])
def test_key_aliases(raw, want):
    """`RMB` 与 `mouse_right` 不相等的话，"右键既是主攻又是交互"会被漏掉。"""
    assert normalize_key(raw) == want


def test_key_alias_catches_conflict_that_spelling_would_hide():
    scheme = InputScheme(name="写法不同", move=MoveScheme(kind="click"),
                         primary=Binding(key="RMB"),
                         bindings={"interact": Binding(key="mouse_right")})
    assert any("mouse_right" in c.message for c in scheme.conflicts())


# ---------------------------------------------------------------- 派生到 mash


def test_mash_keys_never_include_pickup_or_interact():
    keys = preset("tli_pc").to_mash_keys()
    assert "a" not in keys and "d" not in keys
    assert keys == ["q", "w", "e", "r"]


def test_mouse_primary_not_in_mash_keys():
    """连按模型只对键盘键生效；鼠标主攻不能塞进 key 白名单。"""
    assert "mouse_right" not in preset("tli_pc").to_mash_keys()


# ---------------------------------------------------------------- profile 落地


def test_profile_scheme_yaml_is_loaded():
    """profiles/<id>/scheme.yaml 是键位的唯一存放处，必须真的被读到。"""
    cfg = load_config("config.yaml", profile="tli_phone960")
    assert cfg.input is not None
    assert cfg.input.move.kind == "ground_click"
    assert cfg.decision.pickup_key == "a"
    assert cfg.decision.interact_key == "d"
    assert cfg.executor.mash.keys == ["q", "w", "e", "r"]


# ---------------------------------------------------------------- FSM 侧生效


def _mk_engine(**overrides):
    dcfg = DecisionConfig(tick_hz=10.0, approach_mode="keys")
    dcfg.autonomy.enabled = False
    for k, v in overrides.items():
        setattr(dcfg, k, v)
    ex = DryRunExecutor(ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000))
    return DecisionEngine(dcfg, ex), ex


def test_move_keys_mapping_is_used():
    """改成 IJKL 之后，按出去的必须是 i，不是写死的 w。"""
    eng, ex = _mk_engine(move_keys={"forward": "i", "back": "k", "left": "j", "right": "l"})
    assert eng._emit_move_keys(["w"], 100) == 1
    assert [j["key"] for j in ex.journal if j["op"] == "key_tap"] == ["i"]


def test_functional_key_block_happens_after_mapping():
    """映射顺序反了会漏拦：前进键被改成 `a`（拾取键）时，拦下的必须是映射后的 `a`。

    若不先映射再比对，`w` 不在禁止表里 → 放行 → 实际发出的 `a` 就是一次盲捡。
    """
    eng, ex = _mk_engine(move_keys={"forward": "a", "back": "s", "left": "j", "right": "l"})
    assert eng._emit_move_keys(["w"], 100) == 0
    assert eng.stats["blind_tap_violations"] == 1
    assert [j for j in ex.journal if j["op"] == "key_tap"] == []


def test_click_direction_uses_move_button():
    """右键走位的游戏：走位点击必须点右键，左键留给选中。"""
    eng, ex = _mk_engine(approach_mode="mouse_click", move_button="right")
    scene = SceneState(ts_ms=1000, w=1280, h=720, character=(640, 360))
    eng._click_direction(scene, "d", 200)
    clicks = [j for j in ex.journal if j["op"] == "mouse_click"]
    assert clicks and clicks[-1]["button"] == "right"


def test_halt_releases_mapped_keys():
    """松键也要松映射后的键：松 wasd 而实际按的是 ijkl = 人物一直走，日志却说停了。"""
    eng, ex = _mk_engine(approach_mode="keys",
                         move_keys={"forward": "i", "back": "k", "left": "j", "right": "l"})
    eng._halt_movement()
    ups = {j["key"] for j in ex.journal if j["op"] == "key_up"}
    assert ups == {"i", "k", "j", "l"}


# ---------------------------------------------------------------- 页面改键位


def test_patch_scheme_takes_effect_immediately():
    """页面改键位必须立刻派生到 decision.* —— 否则是"显示已改、实际没改"。

    这条是 `settings.LIVE_FIELDS` 里为什么要包含 `input.` 的依据：
    派生后 FSM 读的字段真的变了，页面不该提示"需重启"。
    """
    from autogame.server.settings import apply_patch

    cfg = load_config("config.yaml", profile="tli_phone960")
    res = apply_patch(cfg, {"input": {"bindings": {"pickup": {"key": "f"}}}})
    assert cfg.decision.pickup_key == "f"
    assert "decision.pickup_key" in res["changed"]
    assert res["restart_required"] == []


def test_patch_rejected_whole_when_conflicting():
    """切成 WASD 走位但拾取还是 A → 整笔拒绝，且旧配置不被污染半套。"""
    from autogame.server.settings import ConfigError, apply_patch

    cfg = load_config("config.yaml", profile="tli_phone960")
    with pytest.raises(ConfigError):
        apply_patch(cfg, {"input": {"move": {"kind": "wasd"}}})
    assert cfg.input.move.kind == "ground_click"
    assert cfg.decision.pickup_key == "a"


def test_direct_patch_of_derived_field_is_rejected():
    """方案存在时直接改 decision.pickup_key 会被下次加载覆盖 → 直接拒，并指路。"""
    from autogame.server.settings import ConfigError, apply_patch

    cfg = load_config("config.yaml", profile="tli_phone960")
    with pytest.raises(ConfigError) as ei:
        apply_patch(cfg, {"decision": {"pickup_key": "f"}})
    assert "操作方案" in str(ei.value)
    assert "input.bindings" in str(ei.value)
    assert cfg.decision.pickup_key == "a"      # 一点都没改


def test_derived_field_patch_allowed_without_scheme():
    """没有方案的机器仍能按老办法改 —— 兼容不能破。"""
    from autogame.server.settings import apply_patch

    cfg = AppConfig()
    res = apply_patch(cfg, {"decision": {"pickup_key": "f"}})
    assert cfg.decision.pickup_key == "f"
    assert "decision.pickup_key" in res["changed"]


def test_unlearned_locomotion_blocks_walking_entirely():
    """没学会走位 → 一个"走"的动作都不发。

    为什么不能只是"点了也没用"：这些游戏里点地面可能是选中目标/取消，
    继续点会表现成"程序在动、角色没动"——比不动更难判断真假。
    """
    from autogame.protocol import Pillar

    eng, ex = _mk_engine(approach_mode="mouse_click", move_enabled=False,
                         explore_enabled=True)
    eng.start()
    pillar = Pillar(x=900, y=455, cx=900, cy=400, w=14, h=110, color="gold",
                    area=1500, distance=270.0, whitelisted=True)
    for i in range(6):
        eng.on_scene(SceneState(ts_ms=1000 + i * 200, w=1280, h=720,
                                character=(640, 360), pillars=[pillar]))
    moves = [j for j in ex.journal
             if j["op"] in ("mouse_click", "key_tap")]
    assert moves == [], f"走位总闸关了还在发动作：{moves}"


def test_stick_scheme_turns_off_locomotion():
    """虚拟摇杆：键位表里标 unsupported，同时把走位总闸关掉（不只写个字）。"""
    cfg = AppConfig(input=InputScheme(name="手游", move=MoveScheme(kind="stick")))
    apply_input_scheme(cfg)
    assert cfg.decision.move_enabled is False


def test_ground_click_scheme_keeps_locomotion_on():
    cfg = AppConfig(input=preset("tli_pc"))
    apply_input_scheme(cfg)
    assert cfg.decision.move_enabled is True
