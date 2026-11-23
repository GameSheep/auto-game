"""判据层（cues）的测试。

这一层的价值只有一句话：**画面里能读出来的，才配决定"按不按"。**

所以用例集中在三个"它会骗人"的地方：

  1. 读不出来的时候必须是不成立，不是"默认成立"（缺证据只能让动作变少）
  2. 阈值方向写反/没写，必须编译期报错，不能静默恒真或恒假
  3. 分辨率换了读数不能变（归一化区域，和坐标归一化同一条规矩）
"""

from __future__ import annotations

import numpy as np
import pytest

from autogame.agent import (
    AFTER_ACTION,
    CONDITIONS,
    AgentKnowledge,
    Cue,
    CueBook,
    GamePlayer,
    Goal,
    Skill,
    SkillRun,
    Step,
    VerbContext,
)
from autogame.agent.cues import parse_color
from autogame.input import Binding, InputScheme, MoveScheme, SkillSlot
from autogame.protocol import Monster, SceneState
from autogame.sim.scene import SimWorld

# ---------------------------------------------------------------- 素材


def _scheme(cooldown_s: float = 0.0) -> InputScheme:
    return InputScheme(
        name="测试方案", game="test",
        move=MoveScheme(kind="ground_click", button="mouse_left"),
        primary=Binding(key="mouse_right", action="hold"),
        skills=[SkillSlot(id="fireball", key="q", action="tap", cooldown_s=cooldown_s)],
        bindings={"pickup": Binding(key="a"), "interact": Binding(key="d")},
    )


def _scene(ts: int = 1000, monsters=(), in_game: bool = True) -> SceneState:
    return SceneState(ts_ms=ts, w=1280, h=720, character=(640, 360),
                      monsters=[Monster(x=x, y=y, w=40, h=60) for x, y in monsters],
                      in_game=in_game)


def _frame(color=(0, 0, 0), w: int = 1280, h: int = 720) -> np.ndarray:
    return np.full((h, w, 3), color, np.uint8)


def _paint(frame: np.ndarray, region, bgr) -> None:
    """按归一化矩形涂满一块纯色。"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(region[0] * w), int(region[1] * h),
                     int(region[2] * w), int(region[3] * h))
    frame[y1:y2, x1:x2] = bgr


_MANA = Cue(id="mana_ok", kind="ratio", region=[0.1, 0.1, 0.3, 0.2],
            color="#2a6bd4", above=0.5)
#: 模拟器左下角那条蓝条 —— 端到端用例用它，验证"从真画面读出该不该按"
_MANA_BAR = Cue(id="mana_ok", kind="ratio", region=list(SimWorld.VITAL_MP),
                color="#2a6bd4", above=0.5)


# ---------------------------------------------------------------- 颜色


def test_parse_color_hex_named_and_garbage():
    assert parse_color("#2a6bd4") == (212, 107, 42)   # BGR
    assert parse_color("2a6bd4") == (212, 107, 42)
    assert parse_color("white") == (255, 255, 255)
    assert parse_color("zzz") is None
    assert parse_color("#12345") is None


# ---------------------------------------------------------------- 读数


def test_ratio_cue_reads_color_share():
    b = CueBook(cues=[_MANA])
    on = _frame()
    _paint(on, _MANA.region, (212, 107, 42))
    off = _frame()
    v_on = b.evaluate(on)
    v_off = b.evaluate(off)
    assert v_on["mana_ok"] == pytest.approx(1.0, abs=0.02)
    assert v_off["mana_ok"] == pytest.approx(0.0, abs=0.02)
    assert b.on(v_on) == {"mana_ok"} and b.on(v_off) == set()


def test_region_is_normalized_so_resolution_does_not_matter():
    """同一块内容在 1280x720 和 640x360 上的读数必须一致。

    这里用的是和坐标归一化同一条规矩：写归一化区域，不写像素。
    写成像素的话，换分辨率就不是"读数偏一点"，而是判据指到别的地方去。
    """
    b = CueBook(cues=[_MANA])
    big, small = _frame(w=1280, h=720), _frame(w=640, h=360)
    _paint(big, _MANA.region, (212, 107, 42))
    _paint(small, _MANA.region, (212, 107, 42))
    assert b.evaluate(big)["mana_ok"] == pytest.approx(b.evaluate(small)["mana_ok"], abs=0.03)


def test_bright_cue_counts_luminance():
    b = CueBook(cues=[Cue(id="dark", kind="bright", region=[0, 0, 1, 1], below=0.02)])
    assert b.on(b.evaluate(_frame((0, 0, 0)))) == {"dark"}
    assert b.on(b.evaluate(_frame((255, 255, 255)))) == set()


def test_template_cue_matches_and_missing_template_is_zero(tmp_path):
    """模板匹配：图在 → 分数高；图不在 → 0 并说得出原因（不抛、不当成立）。"""
    import cv2

    tpl = np.zeros((12, 12, 3), np.uint8)
    cv2.rectangle(tpl, (2, 2), (9, 9), (0, 0, 255), -1)
    cv2.imwrite(str(tmp_path / "tpl.png"), tpl)

    frame = _frame()
    h, w = frame.shape[:2]
    frame[int(0.30 * h):int(0.30 * h) + 12, int(0.30 * w):int(0.30 * w) + 12] = tpl
    book = CueBook(cues=[Cue(id="icon", kind="template", region=[0.28, 0.28, 0.42, 0.42],
                             template="tpl.png", min_score=0.7)])
    assert book.evaluate(frame, base_dir=tmp_path)["icon"] > 0.7
    assert book.on({"icon": 0.9}) == {"icon"}

    missing = book.evaluate(frame, base_dir=tmp_path / "nope")
    assert missing["icon"] == 0.0
    assert book.last_errors and "模板读不出来" in book.last_errors[0]


def test_evaluate_never_raises_and_never_defaults_to_on():
    """区域越界、模板缺失、没有画面 —— 一律 0，且不抛异常。

    方向很重要：缺证据只能让动作变少。反过来（读不出来当成立）就是
    "画面一片黑也照按"，那正是这一层要堵的东西。
    """
    bad = CueBook(cues=[
        Cue(id="no_template", kind="template", region=[0.1, 0.1, 0.5, 0.5],
             template="not_here.png"),
        Cue(id="bad_region", kind="ratio", region=[0.9, 0.9, 0.2, 0.2], color="white",
             above=0.1),
    ])
    vals = bad.evaluate(_frame())
    assert vals == {"no_template": 0.0, "bad_region": 0.0}
    assert bad.on(vals) == set()
    assert bad.last_errors                       # 说得出哪里读不出来

    empty = CueBook(cues=[_MANA])
    assert empty.evaluate(None) == {"mana_ok": 0.0}
    assert empty.on(empty.evaluate(None)) == set()


# ---------------------------------------------------------------- 阈值


def test_threshold_directions():
    above = Cue(id="a", kind="ratio", region=[0, 0, 1, 1], above=0.5)
    below = Cue(id="b", kind="ratio", region=[0, 0, 1, 1], below=0.5)
    both = Cue(id="c", kind="ratio", region=[0, 0, 1, 1], above=0.3, below=0.6)
    none = Cue(id="d", kind="ratio", region=[0, 0, 1, 1])
    assert above.on(0.6) and not above.on(0.4)
    assert below.on(0.4) and not below.on(0.6)
    assert both.on(0.45) and not both.on(0.2) and not both.on(0.8)
    # 没写阈值 = 永不成立（lint 会把它报成 error，这里兜住运行期）
    assert not none.on(1.0) and none.problem()


def test_problem_reports_bad_region_and_threshold():
    bad = Cue(id="x", kind="ratio", region=[1.2, 0.0, 0.5, 0.5], above=None, below=None)
    msgs = "\n".join(bad.problem())
    assert "0..1" in msgs            # 越界
    assert "x2/y2" in msgs           # 顺序反
    assert "above" in msgs           # 没阈值
    assert _MANA.problem() == []


# ---------------------------------------------------------------- 编译期


def _knowledge(cues=None, steps=None, **skill_kw) -> AgentKnowledge:
    steps = steps or [Step(verb="use_skill", slot="fireball", when="cue", cue="mana_ok")]
    return AgentKnowledge(
        skills=[Skill(id="burst", when="monsters", steps=steps, **skill_kw)],
        goals=[Goal(id="farm", skills=["burst"])],
        cues=CueBook(cues=cues if cues is not None else [_MANA_BAR]),
    )


def test_lint_flags_missing_unknown_and_unused_cue():
    kn = _knowledge(steps=[Step(verb="use_skill", slot="fireball", when="cue")])
    codes = {i.code for i in kn.lint(_scheme())}
    assert "missing_cue" in codes

    kn = _knowledge(steps=[Step(verb="use_skill", slot="fireball", when="cue",
                                cue="nope")])
    assert "unknown_cue" in {i.code for i in kn.lint(_scheme())}

    kn = _knowledge(steps=[Step(verb="use_skill", slot="fireball", cue="mana_ok")])
    assert "unused_cue" in {i.code for i in kn.lint(_scheme())}


def test_lint_flags_bad_cue_as_error_not_warning():
    kn = _knowledge(cues=[Cue(id="half", kind="ratio", region=[0.1, 0.1],
                              color="white", above=0.2)])
    bad = [i for i in kn.lint(_scheme()) if i.code == "bad_cue"]
    assert bad and bad[0].level == "error"
    assert "region" in bad[0].message


def test_lint_flags_skill_ready_without_cooldown_but_accepts_it_with_one():
    kn = _knowledge(steps=[Step(verb="use_skill", slot="fireball", when="skill_ready")])
    assert "no_cooldown" in {i.code for i in kn.lint(_scheme(0))}
    assert "no_cooldown" not in {i.code for i in kn.lint(_scheme(8))}


def test_lint_rejects_param_condition_on_skill_level():
    kn = _knowledge()
    kn.skills[0].when = "cue"        # 技能级没有 cue 名 → 永远不成立
    assert "param_condition_on_skill" in {i.code for i in kn.lint(_scheme())}


# ---------------------------------------------------------------- 条件


def test_skill_ready_uses_injected_clock():
    ctx = VerbContext(scheme=_scheme(cooldown_s=8))
    step = Step(verb="use_skill", slot="fireball")
    ready = CONDITIONS["skill_ready"]
    assert ready(_scene(), ctx, step.params())          # 从没放过 → 能放
    ctx.last_use["q"] = 100.0
    ctx.now = 103.0
    assert not ready(_scene(), ctx, step.params())      # 3 秒 < 8 秒 CD
    ctx.now = 109.0
    assert ready(_scene(), ctx, step.params())          # CD 到了


def test_use_skill_records_use_time_for_cooldown():
    from autogame.agent import VERBS

    ctx = VerbContext(scheme=_scheme(cooldown_s=8), now=42.0)
    out = VERBS["use_skill"].perform(_scene(), ctx, {"slot": "fireball"})
    assert out.ok and ctx.last_use["q"] == 42.0


# ---------------------------------------------------------------- 循环


def test_skill_repeat_runs_the_whole_sequence_again():
    sk = Skill(id="burst", repeat=3,
               steps=[Step(verb="use_skill", slot="fireball", every_s=0)])
    ctx = VerbContext(scheme=_scheme())
    run = SkillRun(sk, ctx, now=0.0)
    acts = 0
    t = 0.0
    while not run.finished and t < 10:
        if run.tick(_scene(), t) is not None:
            acts += 1
        t += 1.0
    assert acts == 3 and run.status == "ok" and run.rounds == 3


def test_skill_repeat_stops_early_when_until_is_met():
    """`until` 在轮边界判：这一轮走完发现怪没了 → 不再开下一轮。

    判在轮边界而不是每帧，是为了不砍掉"打完再捡掉落"这种收尾步骤。
    """
    sk = Skill(id="burst", repeat=5, until="no_monsters", steps=[
        Step(verb="use_skill", slot="fireball", every_s=0),
        Step(verb="wait", every_s=0),
    ])
    run = SkillRun(sk, VerbContext(scheme=_scheme()), now=0.0)
    run.tick(_scene(monsters=[(500, 500)]), 0.0)     # 放技能
    run.tick(_scene(monsters=[(500, 500)]), 1.0)     # 收尾那一步
    run.tick(_scene(monsters=[]), 2.0)               # 一轮走完：没怪了
    assert run.status == "ok" and run.rounds == 1
    assert "不用放第 2 轮" in run.reason


def test_repeat_does_not_spin_when_every_step_is_skipped():
    """一轮什么都没做就不再空转 —— 否则日志里"跑了 50 轮"，实际 0 动作。"""
    sk = Skill(id="burst", repeat=5, steps=[
        Step(verb="use_skill", slot="fireball", when="cue", cue="nope", every_s=0)])
    run = SkillRun(sk, VerbContext(scheme=_scheme()), now=0.0)
    run.tick(_scene(), 0.0)
    assert run.finished and run.rounds == 1


# ---------------------------------------------------------------- 端到端


class _Sink:
    def __init__(self) -> None:
        self.cmds: list = []

    def __call__(self, cmd) -> bool:
        self.cmds.append(cmd)
        return True


def _player(sink) -> GamePlayer:
    kn = _knowledge(steps=[Step(verb="use_skill", slot="fireball",
                                when="cue", cue="mana_ok")])
    return GamePlayer(knowledge=kn, scheme=_scheme(), sink=sink,
                      cues=kn.cues, max_frame_age_ms=100000)


def _sim_frame(mana: float) -> np.ndarray:
    w = SimWorld(width=1280, height=720, seed=7)
    w.mana = mana
    return w.render()


def test_player_presses_skill_when_cue_is_on():
    sink = _Sink()
    p = _player(sink)
    p.on_scene(_scene(monsters=[(500, 500)]), now=1.0, frame=_sim_frame(1.0))
    assert [c.key for c in sink.cmds] == ["q"]
    assert p.summary()["cue_on"] == ["mana_ok"]


def test_player_stays_put_when_cue_is_off():
    """蓝空了：一个有目标、有技能、什么都就绪的玩家，也一个键都不该按。"""
    sink = _Sink()
    p = _player(sink)
    p.on_scene(_scene(monsters=[(500, 500)]), now=1.0, frame=_sim_frame(0.0))
    assert sink.cmds == []
    assert p.summary()["cue_on"] == []


def test_player_without_frame_treats_cues_as_off():
    """离线回放没有画面：判据一律不成立，而不是沿用上一帧。

    否则"上一帧还有蓝"会一直放技能，直到技能和蓝都空了才发现。
    """
    sink = _Sink()
    p = _player(sink)
    p.on_scene(_scene(monsters=[(500, 500)]), now=1.0, frame=_sim_frame(1.0))
    assert sink.cmds                    # 有画面 → 放
    sink.cmds.clear()
    p.on_scene(_scene(ts=2000, monsters=[(500, 500)]), now=2.0)
    assert sink.cmds == []
    assert p.stats["cue_no_frame"] >= 1
