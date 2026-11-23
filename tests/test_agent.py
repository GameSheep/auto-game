"""游戏者层（agent）的测试。

这一层的价值全在"它不会假装"：没证据不按、做不到就停、按了没效果要能看见。
所以这里的用例大多不是在测"功能对不对"，而是在测**它有没有骗人**：
    - 没 A 框时，一个拾取键都不许发出去
    - 没教的键，不许猜一个按下去
    - 走位没学会时，一个走位动作都不许发
    - 技能空转（0 动作就"完成"）要被发现并封掉
"""

from __future__ import annotations

import pytest

from autogame.agent import (
    AFTER_ACTION,
    CHANGED,
    CONDITION_CN,
    CONDITIONS,
    NO_EFFECT,
    SCENE_FLIP,
    VERBS,
    AgentKnowledge,
    EffectLedger,
    GamePlayer,
    Goal,
    Issue,
    SceneSignature,
    Skill,
    SkillRun,
    Step,
    VerbContext,
    classify,
    resolve_knowledge_dir,
    situation_of,
)
from autogame.agent.knowledge import capability_states, skill_capabilities
from autogame.config import AppConfig
from autogame.input import InputScheme, MoveScheme, preset
from autogame.protocol import AFrame, Monster, Pillar, SceneState


# ---------------------------------------------------------------- 素材

def _scene(ts=1000, char=(640, 360), pillars=(), aframe=False, monsters=(),
           items=(), labels=(), in_game=True, delta=0.0) -> SceneState:
    return SceneState(
        ts_ms=ts, w=1280, h=720, character=char,
        pillars=[Pillar(x=x, y=y, cx=x, cy=y - 60, w=14, h=110, color="gold",
                        area=1500, distance=0.0) for x, y in pillars],
        monsters=[Monster(x=x, y=y, w=40, h=60) for x, y in monsters],
        aframe=AFrame(x=700, y=380, w=28, h=28, score=0.9) if aframe else None,
        in_game=in_game, frame_delta=delta)


def _ctx(scheme=None, **kw) -> VerbContext:
    return VerbContext(scheme=scheme or preset("tli_pc"), **kw)


def _run(verb: str, scene: SceneState, ctx: VerbContext | None = None, **params):
    return VERBS[verb].perform(scene, ctx or _ctx(), params)


# ---------------------------------------------------------------- 动词：走位

def test_move_to_clicks_ground_at_target():
    out = _run("move_to", _scene(pillars=[(900, 500)]), target="nearest_pillar")
    assert out.ok and not out.blocked
    cmd = out.commands[0]
    assert cmd.op == "mouse_click" and cmd.button == "left"
    assert (cmd.x, cmd.y) == (900, 500)


def test_move_to_without_target_is_refused_not_guessed():
    out = _run("move_to", _scene(), target="nearest_pillar")
    assert out.blocked_evidence
    assert out.commands == []
    assert "目标" in out.reason


def test_move_to_refused_when_locomotion_not_learned():
    """不会走位时一个走位动作都不发：别的游戏里点地面可能是"选中目标"。"""
    out = _run("move_to", _scene(pillars=[(900, 500)]), _ctx(move_enabled=False),
               target="nearest_pillar")
    assert out.blocked_capability
    assert out.commands == []


def test_wasd_move_presses_direction_keys_and_warns_about_camera():
    """WASD 走位：按方向键而不是点地面；视角没控制时要如实说"方向可能偏"。"""
    out = _run("move_to", _scene(char=(640, 360), pillars=[(1100, 360)]),
               _ctx(preset("wow")), target="nearest_pillar")
    assert out.ok
    keys = [c.key for c in out.commands]
    assert keys and set(keys) <= {"w", "a", "s", "d"}
    assert "视角" in out.note


def test_roam_cycles_four_directions_before_repeating():
    """探索要覆盖四个方向：老在一个角上打转，看起来也像在动。"""
    ctx, scene = _ctx(), _scene()
    xs = []
    for _ in range(4):
        out = _run("roam", scene, ctx)
        assert out.ok
        cmd = out.commands[0]
        assert cmd.op == "mouse_click"
        xs.append((cmd.x, cmd.y))
    assert len(set(xs)) == 4
    # 不许点屏幕正中：那可能是角色自己，点下去是"选中"不是"走过去"
    assert all(x != 640 or y != 360 for x, y in xs)


# ---------------------------------------------------------------- 动词：拾取（铁律）

def test_pickup_without_aframe_sends_nothing():
    """**这条是铁律的抓手**：没有 A 框，拾取键一个字节都不许发出去。"""
    out = _run("pickup", _scene(pillars=[(700, 400)]))
    assert not out.ok
    assert out.blocked_evidence
    assert out.evidence == "aframe"
    assert out.commands == []


def test_pickup_with_aframe_presses_learned_key():
    out = _run("pickup", _scene(pillars=[(700, 400)], aframe=True))
    assert out.ok
    assert [c.op for c in out.commands] == ["key_tap"]
    assert out.commands[0].key == "a"


def test_pickup_without_binding_never_guesses_a_key():
    """魔兽预设没教拾取：不许拿"a"或"右键"去试，宁可什么都不捡。"""
    out = _run("pickup", _scene(pillars=[(700, 400)], aframe=True), _ctx(preset("wow")))
    assert out.blocked_capability
    assert out.commands == []
    assert "猜" in out.reason or "没教" in out.reason


def test_pickup_uses_mouse_binding_when_that_is_how_game_works():
    scheme = InputScheme(name="鼠标捡", move={"kind": "ground_click"},
                         bindings={"pickup": {"key": "mouse_right"}})
    out = _run("pickup", _scene(pillars=[(700, 400)], aframe=True), _ctx(scheme))
    assert out.ok
    assert out.commands[0].op == "mouse_click" and out.commands[0].button == "right"


def test_stale_frame_blocks_the_press():
    """A 框是"过去某一帧在范围内"。帧一旧，人可能已经走过去了 —— 这一按就是盲按。"""
    sc = _scene(pillars=[(700, 400)], aframe=True)
    sc.age_ms = 900.0
    out = _run("pickup", sc)
    assert out.blocked_evidence and out.evidence == "fresh"
    assert out.commands == []

    sc.age_ms = 120.0
    assert _run("pickup", sc).ok


# ---------------------------------------------------------------- 动词：交互 / 技能 / 面板

def test_interact_requires_a_target():
    """站在箱子旁边才按 D；不站在那儿按就是盲按。"""
    assert _run("interact", _scene()).blocked_evidence
    out = _run("interact", _scene(pillars=[(700, 400)]), target="nearest_pillar")
    assert out.ok and out.commands[0].key == "d"


def test_open_panel_is_allowed_without_target_but_flagged_risky():
    """打开关卡面板没有目标可依 —— 允许，但动词自己标了 risky，lint 会要求写 when。"""
    out = _run("open_panel", _scene())
    assert out.ok and out.commands[0].key == "d"
    assert VERBS["open_panel"].risky is True


def test_use_skill_with_unknown_slot_is_capability_block():
    out = _run("use_skill", _scene(), slot="不存在的技能")
    assert out.blocked_capability
    out2 = _run("use_skill", _scene(), slot="技能1")
    assert out2.ok and out2.commands[0].key == "q"


def test_mouse_primary_clicks_the_monster():
    out = _run("attack", _scene(monsters=[(800, 400)]), target="nearest_monster")
    assert out.ok
    cmd = out.commands[0]
    assert cmd.op == "mouse_click" and cmd.button == "right"
    assert (cmd.x, cmd.y) == (820, 430)


def test_attack_with_selector_but_no_target_does_not_click_the_cursor():
    """没目标时不能退化成"朝当前光标点一下"——日志里看不出任何异常。"""
    out = _run("attack", _scene(), target="nearest_monster")
    assert out.blocked_evidence and out.commands == []


def test_cancel_is_allowed_when_out_of_game():
    """面板打开时恰恰是场景门不通过的时候，所以返回键不能要求 in_game。"""
    out = _run("cancel", _scene(in_game=False))
    assert out.ok and out.commands[0].key == "escape"


# ---------------------------------------------------------------- 效果

def test_motion_effect_only_counts_for_movement_verbs():
    a = SceneSignature.of(_scene(delta=0.0))
    b = SceneSignature.of(_scene(delta=60.0))
    assert classify(a, b) == NO_EFFECT            # 状态类动作：画面在动不算数
    assert classify(a, b, motion_sensitive=True) == CHANGED


def test_state_effect_sees_things_disappear():
    before = SceneSignature.of(_scene(pillars=[(700, 400), (800, 400)]))
    after = SceneSignature.of(_scene(pillars=[(700, 400)]))
    assert classify(before, after) == CHANGED
    assert classify(before, after, motion_sensitive=True) == CHANGED


def test_scene_flip_is_its_own_category():
    a = SceneSignature.of(_scene(in_game=True))
    b = SceneSignature.of(_scene(in_game=False))
    assert classify(a, b) == SCENE_FLIP


def test_dead_ledger_needs_zero_effects():
    led = EffectLedger(dead_after=3)
    for _ in range(3):
        led.record("pickup", "ground", NO_EFFECT)
    assert led.dead()[0]["verb"] == "pickup"

    led2 = EffectLedger(dead_after=3)
    led2.record("pickup", "ground", NO_EFFECT)
    led2.record("pickup", "ground", NO_EFFECT)
    led2.record("pickup", "ground", CHANGED)
    assert led2.dead() == []      # 有用过一次就不算"一直没用"


def test_situation_separates_combat_from_empty_ground():
    assert situation_of(_scene(monsters=[(800, 400)])) == "combat"
    assert situation_of(_scene()) == "ground"
    assert situation_of(_scene(in_game=False)) == "out_of_game"


# ---------------------------------------------------------------- 技能执行

def test_target_gone_uses_counts_not_screen_positions():
    """相机一跟，目标在屏幕上就跑了 —— 不能把"移动了"当成"消失了"。"""
    ctx = _ctx()
    paused = _scene(pillars=[(900, 500)])
    VERBS["pickup"].perform(_scene(pillars=[(900, 500)], aframe=True), ctx)
    # 光柱还在，只是位置变了（角色在走）：不算消失
    assert CONDITIONS["target_gone"](_scene(pillars=[(700, 520)]), ctx, {}) is False
    # 光柱少了：才算真消失
    assert CONDITIONS["target_gone"](_scene(), ctx, {}) is True
    _ = paused


def test_skill_run_advances_when_target_is_reached():
    sk = Skill(id="s", steps=[
        Step(verb="move_to", target="nearest_pillar", until="near_target", every_s=0),
        Step(verb="pickup", until=AFTER_ACTION, every_s=0)])
    ctx = _ctx()
    run = SkillRun(sk, ctx, 0.0)
    out = run.tick(_scene(char=(640, 360), pillars=[(900, 500)]), 0.0)
    assert out.ok and run.i == 0
    run.tick(_scene(char=(880, 480), pillars=[(900, 500)], aframe=True), 0.5)
    assert run.i == 1


def test_skill_timeout_fails_with_the_step_name():
    sk = Skill(id="s", steps=[
        Step(verb="pickup", until="target_gone", max_seconds=0.5, every_s=0)])
    run = SkillRun(sk, _ctx(), 0.0)
    scene = _scene(pillars=[(700, 400)], aframe=True)
    run.tick(scene, 0.0)
    run.tick(scene, 0.9)
    assert run.status == "failed"
    assert "第 1 步" in run.reason


# ---------------------------------------------------------------- 游戏者：闭环

def _knowledge(skills, goals):
    return AgentKnowledge(skills=skills, goals=goals)


def _pick_skill():
    return Skill(id="pick_one", when="target", steps=[
        Step(verb="move_to", target="nearest_pillar", until="near_target",
             every_s=0, max_seconds=5),
        Step(verb="pickup", until="target_gone", every_s=0, max_seconds=3)],
        success="target_gone", max_seconds=10)


def _far_then_near_scenes():
    return [
        _scene(1000, (640, 360), [(1100, 600)]),
        _scene(1200, (1050, 560), [(1100, 600)]),
        _scene(1400, (1050, 560), [(1100, 600)], aframe=True),
        _scene(1600, (1050, 560), []),
    ]


def test_player_runs_one_full_pick_cycle():
    kn = _knowledge([_pick_skill()], [Goal(id="farm", when="in_game", skills=["pick_one"])])
    sent = []
    p = GamePlayer(knowledge=kn, scheme=preset("tli_pc"),
                   sink=lambda c: sent.append(c) or True)
    for i, s in enumerate(_far_then_near_scenes()):
        p.on_scene(s, now=i * 0.3)
    assert [c.op for c in sent] == ["mouse_click", "key_tap"]
    assert sent[1].key == "a"
    assert p.stats["skill_ok"] >= 1
    assert p.stats["pickup_refused_no_aframe"] == 1   # 走近那帧确实忍住了


def test_player_never_presses_pickup_without_aframe():
    """把一帧"人在旁边但 A 框还没出现"喂进去：不许有拾取键漏出来。"""
    kn = _knowledge([_pick_skill()], [Goal(id="farm", skills=["pick_one"])])
    sent = []
    p = GamePlayer(knowledge=kn, scheme=preset("tli_pc"),
                   sink=lambda c: sent.append(c) or True)
    for i in range(6):
        p.on_scene(_scene(1000 + i * 200, (1050, 560), [(1100, 600)]), now=i * 0.3)
    assert [c.key for c in sent if c.op == "key_tap"] == []
    assert p.stats["pickup_refused_no_aframe"] >= 1


def test_player_gives_up_goal_when_capability_is_missing():
    """魔兽预设没有拾取键 → 这条技能永远做不了 → 目标被标成"缺能力"，不空转。"""
    sk = Skill(id="pick_one", when="target", steps=[
        Step(verb="pickup", until="target_gone", max_seconds=2)])
    kn = _knowledge([sk], [Goal(id="farm", when="in_game", skills=["pick_one"])])
    p = GamePlayer(knowledge=kn, scheme=preset("wow"), sink=lambda c: True)
    p.on_scene(_scene(pillars=[(700, 400)], aframe=True), now=0.0)
    idle = [e for e in p.trace if e["kind"] == "idle"]
    assert idle and "技能" in idle[-1]["reason"]      # 说清"为什么不动"
    assert p.stats["actions"] == 0                   # 一个动作都没发


def test_player_repeats_the_skill_after_success():
    kn = _knowledge([_pick_skill()], [Goal(id="farm", skills=["pick_one"])])
    p = GamePlayer(knowledge=kn, scheme=preset("tli_pc"), sink=lambda c: True)
    seq = _far_then_near_scenes() + [_scene(1800, (900, 500), [(1100, 550)])]
    for i, s in enumerate(seq):
        p.on_scene(s, now=i * 0.3)
    started = [e["skill"] for e in p.trace if e["kind"] == "skill"]
    assert len(started) >= 2       # 成功之后要接着来，而不是停下
    assert [e["kind"] for e in p.trace].count("skill_end") >= 1


def test_noop_skill_is_detected_and_banned():
    """每步都被 when 跳过 → "0 动作就完成" → 三次之后封掉，不再空转。"""
    sk = Skill(id="fake", steps=[Step(verb="move_to", target="nearest_pillar",
                                      when="never", until="never")])
    kn = _knowledge([sk], [Goal(id="farm", skills=["fake"])])
    p = GamePlayer(knowledge=kn, scheme=preset("tli_pc"), sink=lambda c: True)
    for i in range(5):
        p.on_scene(_scene(1000 + i * 200, pillars=[(900, 500)]), now=i * 0.3)
    assert p.stats["noop_completion"] >= 3
    assert "fake" in p.summary()["banned"]
    assert any("0 动作" in h["hypothesis"] for h in p.hypotheses())


def test_dead_action_becomes_a_hypothesis_without_changing_anything():
    """一直没效果 → 只出假设，配置/方案一个字都不改。"""
    sk = Skill(id="hit", steps=[Step(verb="attack", until=AFTER_ACTION, every_s=0)])
    kn = _knowledge([sk], [Goal(id="fight", skills=["hit"])])
    scheme = preset("tli_pc")
    before = scheme.model_dump()
    p = GamePlayer(knowledge=kn, scheme=scheme, dead_after=3, sink=lambda c: True)
    for i in range(6):
        p.on_scene(_scene(1000 + i * 200, monsters=[(800, 400)]), now=i * 0.3)
    dead = p.ledger.dead()
    assert dead and dead[0]["verb"] == "attack"
    assert scheme.model_dump() == before


# ---------------------------------------------------------------- 知识检查

def _sk(**kw):
    base = {"id": "s", "steps": [{"verb": "move_to", "target": "nearest_pillar"}]}
    base.update(kw)
    return Skill(**base)


def test_lint_rejects_pickup_until_not_aframe():
    sk = Skill(id="s", steps=[Step(verb="pickup", until="not_aframe")])
    issues = AgentKnowledge(skills=[sk]).lint(preset("tli_pc"))
    assert any(i.code == "bad_until" and i.level == "error" for i in issues)


def test_lint_rejects_unknown_verb_condition_and_skill_ref():
    bad = AgentKnowledge(
        skills=[Skill(id="s", steps=[Step(verb="飞"),
                                     Step(verb="wait", until="什么时候")])],
        goals=[Goal(id="g", skills=["不存在"])])
    codes = {i.code for i in bad.lint(preset("tli_pc"))}
    assert {"unknown_verb", "unknown_condition", "unknown_skill"} <= codes


def test_lint_rejects_contradictory_when():
    sk = Skill(id="s", steps=[Step(verb="interact", when="no_targets")])
    codes = {i.code for i in AgentKnowledge(skills=[sk]).lint(preset("tli_pc"))}
    assert "contradictory_when" in codes


def test_lint_marks_capability_gap_as_blocked_not_error():
    """缺键位是"缺口"不是"错误"：该做的是去教，不是去改知识。"""
    sk = Skill(id="s", steps=[Step(verb="pickup", until="target_gone")])
    issues = AgentKnowledge(skills=[sk],
                            goals=[Goal(id="g", skills=["s"])]).lint(preset("wow"))
    blocked = [i for i in issues if i.level == "blocked"]
    assert blocked and "pickup" in blocked[0].message
    assert not [i for i in issues if i.level == "error"]


def test_lint_warns_when_risky_verb_has_no_when():
    sk = Skill(id="s", steps=[Step(verb="open_panel")])
    codes = {i.code for i in AgentKnowledge(skills=[sk]).lint(preset("tli_pc"))}
    assert "risky_without_when" in codes


def test_report_separates_usable_and_blocked_goals():
    can = Skill(id="can", when="target", steps=[Step(verb="pickup", until="target_gone")])
    cant = Skill(id="cant", steps=[Step(verb="pickup", until="target_gone")])
    kn = AgentKnowledge(skills=[can, cant],
                        goals=[Goal(id="ok", skills=["can"]),
                               Goal(id="no", skills=["cant"])])
    rep = kn.report(preset("wow"))     # wow 没有拾取键
    assert rep["usable_goals"] == []
    reason = {g["id"]: g["reason"] for g in rep["goals"]}
    assert "pickup" in reason["no"]


def test_knowledge_loads_from_dir(tmp_path):
    (tmp_path / "skills.yaml").write_text(
        "skills:\n  - id: s\n    steps:\n      - verb: wait\n", encoding="utf-8")
    (tmp_path / "goals.yaml").write_text(
        "goals:\n  - id: g\n    skills: [s]\n", encoding="utf-8")
    kn = AgentKnowledge.load(tmp_path)
    assert [s.id for s in kn.skills] == ["s"]
    assert [g.id for g in kn.goals] == ["g"]


def test_knowledge_loads_from_single_file(tmp_path):
    f = tmp_path / "agent.yaml"
    f.write_text("skills:\n  - id: s\n    steps:\n      - verb: wait\n"
                 "goals:\n  - id: g\n    skills: [s]\n", encoding="utf-8")
    kn = AgentKnowledge.load(f)
    assert kn.skills and kn.goals


def test_empty_dir_is_an_empty_knowledge_not_an_error(tmp_path):
    kn = AgentKnowledge.load(tmp_path)
    assert not kn and kn.skills == []


# ---------------------------------------------------------------- 仓库里的真实知识

def test_tli_profile_knowledge_has_no_issues():
    from autogame.config import load_config
    from autogame.profile import load_profile

    cfg = load_config("config.yaml", profile="tli_phone960")
    kn = AgentKnowledge.load(resolve_knowledge_dir(cfg.agent,
                                                  load_profile("tli_phone960").root))
    assert kn.skills and kn.goals
    issues = kn.lint(cfg.input)
    assert [i.line() for i in issues] == []
    rep = kn.report(cfg.input)
    assert rep["blocked_goals"] == []
    assert "farm" in rep["usable_goals"]


def test_template_knowledge_is_valid_scaffold():
    """脚手架要能直接用 —— 复制过去就有东西能跑，不是一段会报错的注释。"""
    kn = AgentKnowledge.load("profiles/_template/agent")
    assert kn.skills and kn.goals
    issues = kn.lint(preset("tli_pc"))
    assert [i.line() for i in issues] == []


# ---------------------------------------------------------------- 配置 / 自检

def test_agent_is_off_by_default():
    """新能力默认关：没验证过的东西不许改变已经在跑的流程。"""
    cfg = AppConfig()
    assert cfg.agent.enabled is False
    assert cfg.agent.knowledge_dir == "agent"


def test_preflight_agent_check_skips_when_not_taught(tmp_path):
    from autogame.preflight import _check_agent

    cfg = AppConfig()
    cfg.agent.knowledge_dir = str(tmp_path / "空目录")
    c = _check_agent(cfg, "")
    assert c["level"] == "skip" and "还没有" in c["detail"]


def test_preflight_agent_check_fails_on_bad_knowledge_when_enabled(tmp_path):
    from autogame.preflight import _check_agent

    (tmp_path / "skills.yaml").write_text(
        "skills:\n  - id: s\n    steps:\n      - verb: 飞\n", encoding="utf-8")
    cfg = AppConfig()
    cfg.agent.knowledge_dir = str(tmp_path)
    cfg.agent.enabled = True
    c = _check_agent(cfg, "")
    assert c["level"] == "fail" and "写错" in c["detail"]

    cfg.agent.enabled = False
    assert _check_agent(cfg, "")["level"] == "warn"    # 没开的时候只是提示


def test_agent_report_api_shape():
    from autogame.config import load_config
    from autogame.profile import load_profile
    from autogame.server.settings import agent_report

    cfg = load_config("config.yaml", profile="tli_phone960")
    rep = agent_report(cfg, load_profile("tli_phone960").root)
    assert rep is not None
    assert rep["enabled"] is False
    assert any(g["id"] == "farm" for g in rep["goals"])

# ---------------------------------------------------------------- 接进服务端

def _hub(cfg):
    from autogame.executor import DryRunExecutor
    from autogame.server.hub import Hub
    from autogame.vision.pipeline import VisionPipeline

    cfg.executor.min_action_interval_ms = 0
    cfg.executor.max_actions_per_sec = 1000
    return Hub(cfg, DryRunExecutor(cfg.executor), VisionPipeline(cfg.vision))


def test_hub_keeps_fsm_when_agent_is_off():
    """默认关的时候，服务端连游戏者都不构建 —— 老流程一个字节都不变。"""
    hub = _hub(AppConfig())
    assert hub.agent_player() is None


def test_hub_refuses_to_switch_on_without_knowledge(tmp_path):
    """开了却没知识：不许静默换成"什么都不做"，要报出来并留在 FSM 上。"""
    cfg = AppConfig()
    cfg.agent.enabled = True
    cfg.agent.knowledge_dir = str(tmp_path / "空的知识")
    hub = _hub(cfg)
    assert hub.agent_player() is None
    assert "知识目录为空" in hub._agent_error
    assert hub._player_tried is True          # 只试一次，不每帧重试


def test_hub_starts_agent_with_real_knowledge():
    from autogame.config import load_config
    from autogame.profile import load_profile

    cfg = load_config("config.yaml", profile="tli_phone960")
    cfg.agent.enabled = True
    hub = _hub(cfg)
    hub.profile = load_profile("tli_phone960")
    player = hub.agent_player()
    assert player is not None
    assert player.scheme is cfg.input
    assert hub._agent_error == ""
    assert hub.agent_player() is player        # 只构建一次
