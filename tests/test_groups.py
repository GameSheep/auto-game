"""指令组的边界测试。

这一层要钉住的东西，几乎都不是"功能有没有"，而是**不会悄悄变坏**：

    写一段步骤就要能跑            —— 人写的自然语言，机器要能对上动作
    连着写的两个动作是一件事      —— 合并，不然第一步跑满 300 秒才进第二步
    引擎已经会的不要再要点模板    —— 「再次挑战」每帧都在扫，逼人裁图是倒退
    缺东西要说得出来              —— 而不是跑起来以后每一步都点不出去
    认不出比认错好                —— 歧义时返回候选，一个动作都不发
    新学一组不能把旧组挤掉        —— 一组一文件，谁也别覆盖谁
    世界层只增不删                —— 手册删一条，不该让别的组突然瞎掉
    判定被翻过来必须记账          —— 同一界面忽正忽反，门禁会乱杀人
    手册里删了组不等于要删文件    —— 人点头之前，它还在

这些用例全部走临时目录，不碰真实 profile —— 那里面有用户的截图和标答。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autogame import groups as gp
from autogame.profile import Profile

LESSON = """# 测试用手册

## 术语

| 我说的 | 标准说法 | 备注 |
|---|---|---|
| 搬砖 | 刷图 | 主要玩法 |
| 星图 | 异界关卡选择 | |

## 界面 · 异界星图

- 判定: 反例
- 模板名: nether_title
- 一眼认出: 左上角「‹ 异界」

## 界面 · 战斗

- 判定: 正例
- 模板名: skillwheel

## 界面 · 结算

- 判定: 反例
- 模板名: result_panel

## 物体 · 掉落光柱

- 颜色: 金色
- 高宽比: 3~8

## 指令组 · 刷k7

- 喊作: 刷k7、打k7、k7、刷图
- 起点: 异界星图
1. 在星图点「K7节点」，进图
2. 清怪，沿途捡掉落
3. 打完 Boss 出结算
4. 点「再次挑战」，回到第 2 步

## 指令组 · 刷一局

- 喊作: 刷一局、再来一局
- 起点: 战斗
1. 清怪，沿途捡掉落，直到结算
2. 点「再次挑战」，回到第 1 步

## 键位

- 拾取: A
- 返回: ESC
"""


@pytest.fixture()
def prof(tmp_path: Path) -> Profile:
    root = tmp_path / "prof"
    (root / "lessons").mkdir(parents=True)
    (root / "templates").mkdir(parents=True)
    (root / "lessons" / "t.md").write_text(LESSON, encoding="utf-8")
    return Profile(id="t", root=root)


def _write(prof: Profile, text: str) -> None:
    (Path(prof.root) / "lessons" / "t.md").write_text(text, encoding="utf-8")


# ---------------------------------------------------------------- 1. 解析

def test_numbered_steps_become_executable_actions(prof):
    """人写的四句话 → 三步可执行的东西。"""
    gp.sync_from_lessons(prof)
    g = {x.id: x for x in gp.load_groups(prof)}["刷k7"]
    kinds = [s.kind for s in g.steps]
    assert kinds == ["click", "delegate", "click"], kinds
    assert g.steps[0].target == "K7"
    assert g.steps[1].until == "结算"
    assert g.steps[2].builtin == "rechallenge"
    assert g.steps[2].jump == 2


def test_two_written_steps_become_one_delegate(prof):
    """「清怪」+「打完 Boss 出结算」是同一件事的两个说法，合并成一步。

    不合并的后果很具体：第一步要跑满 300 秒超时才进第二步，而这期间它一直在
    打已经打完的图。
    """
    gp.sync_from_lessons(prof)
    g = {x.id: x for x in gp.load_groups(prof)}["刷k7"]
    delegates = [s for s in g.steps if s.kind == "delegate"]
    assert len(delegates) == 1
    # 合并后仍要说得出两句原话，否则诊断时不知道它当时在干什么
    assert "清怪" in delegates[0].raw and "结算" in delegates[0].raw


def test_builtin_button_needs_no_template(prof):
    """「再次挑战」是引擎每帧扫出来的按钮，不该要求用户再裁一张图。"""
    gp.sync_from_lessons(prof)
    world = gp.load_world(prof)
    g = {x.id: x for x in gp.load_groups(prof)}["刷一局"]
    assert g.steps[-1].builtin == "rechallenge"
    # 这一组完全不依赖模板，所以必须能跑
    assert g.gaps(world, Path(prof.root) / "templates") == []


def test_missing_template_is_reported_not_hidden(prof):
    gp.sync_from_lessons(prof)
    world = gp.load_world(prof)
    g = {x.id: x for x in gp.load_groups(prof)}["刷k7"]
    gaps = g.gaps(world, Path(prof.root) / "templates")
    assert len(gaps) == 1 and "K7" in gaps[0]


def test_unknown_step_keeps_the_original_words(prof):
    """看不懂的一步要留着原话，还要在缺口里说出来 —— 静默跳过等于骗人。"""
    _write(prof, LESSON.replace("2. 清怪，沿途捡掉落", "2. 把它标记成好东西"))
    gp.sync_from_lessons(prof)
    g = {x.id: x for x in gp.load_groups(prof)}["刷k7"]
    bad = [s for s in g.steps if s.kind == "unknown"]
    assert bad and "标记" in bad[0].raw
    gaps = g.gaps(gp.load_world(prof), Path(prof.root) / "templates")
    assert any("看不懂" in x for x in gaps)


# ---------------------------------------------------------------- 2. 路由

@pytest.fixture()
def routed(prof):
    gp.sync_from_lessons(prof)
    gs = gp.load_groups(prof)
    w = gp.load_world(prof)
    return lambda t: gp.route(t, gs, w)


@pytest.mark.parametrize("text,name", [
    ("刷k7", "刷k7"),
    ("打k7", "刷k7"),
    ("k7", "刷k7"),
    ("帮我刷一下k7", "刷k7"),
    ("刷一局", "刷一局"),
    ("再来一局", "刷一局"),
    ("搬砖", "刷k7"),          # 术语换算：搬砖 → 刷图 → 别名命中
])
def test_route_hits_the_right_group(routed, text, name):
    r = routed(text)
    assert r.ok and r.group.name == name, (text, r.why, r.candidates)


@pytest.mark.parametrize("text", ["来一把", "刷", "今天天气不错", ""])
def test_route_refuses_to_guess(routed, text):
    """认不出就认不出。猜错的代价是几分钟错误操作，不猜只丢几秒。"""
    r = routed(text)
    assert not r.ok


def test_route_marks_ambiguity_instead_of_picking(prof):
    """两组都像的时候返回候选，不挑一个最像的。"""
    gp.sync_from_lessons(prof)
    a = gp.Group(id="a", name="刷图a", say=["刷"], steps=[gp.Step(kind="key", target="A")])
    b = gp.Group(id="b", name="刷图b", say=["刷"], steps=[gp.Step(kind="key", target="B")])
    r = gp.route("刷", [a, b], gp.load_world(prof))
    assert not r.ok
    assert len(r.candidates) == 2


def test_route_survives_a_typo(routed):
    assert routed("刷k77").group.name == "刷k7"     # 多打一个字
    assert routed("刷K7").group.name == "刷k7"      # 大小写
    assert routed("刷 k7").group.name == "刷k7"     # 空格


# ---------------------------------------------------------------- 3. 不遗忘

def test_new_group_does_not_touch_other_groups(prof):
    gp.sync_from_lessons(prof)
    before = (Path(prof.root) / "groups" / "刷一局.yaml").read_bytes()
    _write(prof, LESSON + "\n## 指令组 · 清包\n\n- 喊作: 清包\n1. 按 A\n")
    gp.sync_from_lessons(prof)
    assert (Path(prof.root) / "groups" / "刷一局.yaml").read_bytes() == before
    assert {g.id for g in gp.load_groups(prof)} == {"刷k7", "刷一局", "清包"}


def test_world_keeps_what_the_new_manual_forgot(prof):
    """手册被改小了，旧的认识不能跟着消失 —— 否则另一组突然就瞎了。"""
    gp.sync_from_lessons(prof)
    assert "结算" in gp.load_world(prof).scenes
    _write(prof, "# 只剩一屏\n\n## 界面 · 战斗\n- 判定: 正例\n- 模板名: skillwheel\n")
    gp.sync_from_lessons(prof)
    w = gp.load_world(prof)
    assert "结算" in w.scenes and "异界星图" in w.scenes
    assert w.keys.get("拾取") == "A"


def test_flipped_verdict_is_recorded_as_conflict(prof):
    """同一个界面一会儿算战斗一会儿算菜单，门禁会乱杀人，必须记账。"""
    gp.sync_from_lessons(prof)
    _write(prof, LESSON.replace(
        "## 界面 · 战斗\n\n- 判定: 正例", "## 界面 · 战斗\n\n- 判定: 反例"))
    info = gp.sync_from_lessons(prof)
    assert any(c["kind"] == "scene" and c["key"] == "战斗" for c in info["conflicts"])
    # 冲突记下来了，但采用新的（人最后写的那句算数）
    assert "反例" in gp.load_world(prof).scenes["战斗"]["verdict"]


def test_group_deleted_from_manual_is_kept_and_flagged(prof):
    """手册删掉一段，不等于玩家要忘掉这个任务。文件留着，等人点头。"""
    gp.sync_from_lessons(prof)
    _write(prof, LESSON.split("## 指令组 · 刷一局")[0])
    info = gp.sync_from_lessons(prof)
    assert info["orphans"] == ["刷一局"]
    assert (Path(prof.root) / "groups" / "刷一局.yaml").exists()
    assert any(g.id == "刷一局" for g in gp.load_groups(prof))
    # 确认后才真删
    assert gp.forget_group(prof, "刷一局")
    assert not any(g.id == "刷一局" for g in gp.load_groups(prof))


# ---------------------------------------------------------------- 4. 执行

class _Exec:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    def execute(self, cmd) -> None:  # noqa: ANN001
        self.sent.append(cmd.model_dump())


def _scene(**kw) -> SimpleNamespace:
    base = dict(ts_ms=0, in_game=True, gate_reason="skillbar", rechallenge=None,
                pillars=[], monsters=[], ground_labels=[], aframe=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_runner_clicks_then_moves_on(prof):
    gp.sync_from_lessons(prof)
    world = gp.load_world(prof)
    g = {x.id: x for x in gp.load_groups(prof)}["刷一局"]
    ex = _Exec()
    r = gp.GroupRunner(g, world, Path(prof.root) / "templates", executor=ex,
                       fsm=SimpleNamespace(on_scene=lambda s: None))
    hit = SimpleNamespace(x=100, y=50)
    # 第 1 步是等结算。给一个结算帧（反例）→ 立刻进第 2 步
    r.tick_frame(_scene(in_game=False, gate_reason="negative:result_panel"), np.zeros((10, 10, 3), np.uint8))
    assert r.step_i == 1
    # 第 2 步点「再次挑战」：没有按钮就不动
    r.tick_frame(_scene(in_game=False), np.zeros((10, 10, 3), np.uint8))
    assert not ex.sent
    # 按钮出现 → 点它 → 回到第 1 步（jump=1）
    r.tick_frame(_scene(in_game=False, rechallenge=hit), np.zeros((10, 10, 3), np.uint8))
    assert ex.sent and ex.sent[-1]["op"] == "mouse_click"
    assert (ex.sent[-1]["x"], ex.sent[-1]["y"]) == (100, 50)


def test_runner_delegates_to_fsm(prof):
    """「交给引擎」这一步必须真的把控制权给 FSM，否则组就变成空转。"""
    gp.sync_from_lessons(prof)
    world = gp.load_world(prof)
    g = {x.id: x for x in gp.load_groups(prof)}["刷一局"]
    seen: list[int] = []
    r = gp.GroupRunner(g, world, None, executor=_Exec(),
                       fsm=SimpleNamespace(on_scene=lambda s: seen.append(s.ts_ms)))
    r.tick_frame(_scene(ts_ms=7, in_game=True), np.zeros((10, 10, 3), np.uint8))
    assert seen == [7]
    assert r.step_i == 0          # 还在"等结算"，没瞎推进


def test_runner_stops_on_timeout_and_says_where(prof):
    gp.sync_from_lessons(prof)
    world = gp.load_world(prof)
    g = {x.id: x for x in gp.load_groups(prof)}["刷一局"]
    r = gp.GroupRunner(g, world, None, executor=_Exec(),
                       fsm=SimpleNamespace(on_scene=lambda s: None))
    r.tick_frame(_scene(ts_ms=0), np.zeros((10, 10, 3), np.uint8))
    r.tick_frame(_scene(ts_ms=400_000), np.zeros((10, 10, 3), np.uint8))
    assert r.done and r.failed
    assert "第 1 步" in r.failed        # 报告要能指到具体哪一步
    assert r.status()["total_steps"] == 2


def test_runner_refuses_group_with_missing_template(prof):
    """缺模板的组跑到点击那一步要立刻停，而不是每帧点空、最后说超时。"""
    gp.sync_from_lessons(prof)
    world = gp.load_world(prof)
    g = {x.id: x for x in gp.load_groups(prof)}["刷k7"]
    r = gp.GroupRunner(g, world, Path(prof.root) / "templates", executor=_Exec())
    r.tick_frame(_scene(), np.zeros((10, 10, 3), np.uint8))
    assert r.done and "没有模板" in r.failed


def test_a_menu_is_not_mistaken_for_the_result_screen(prof):
    """星图/背包都没有战斗 HUD，但**不是**结算页。

    最初把"反例界面"写成"没有 HUD"，结果在星图上喊「刷一局」会当成已经打完，
    直接跳到「点再次挑战」—— 一步不打就空转。这条用例是那个 bug 的护栏。
    """
    gp.sync_from_lessons(prof)
    world = gp.load_world(prof)
    g = {x.id: x for x in gp.load_groups(prof)}["刷一局"]
    fsm_calls: list[int] = []
    r = gp.GroupRunner(g, world, None, executor=_Exec(),
                       fsm=SimpleNamespace(on_scene=lambda s: fsm_calls.append(s.ts_ms)))
    menu = _scene(ts_ms=1, in_game=False, gate_reason="unmatched", rechallenge=None)
    for _ in range(3):
        r.tick_frame(menu, np.zeros((10, 10, 3), np.uint8))
    assert r.step_i == 0                 # 还在"打一局"，没跳到下一步
    assert len(fsm_calls) == 3            # 而且真的把控制权给引擎了


def test_result_screen_needs_the_rechallenge_button(prof):
    """结算页的判据是「没有 HUD + 有再次挑战」，缺一个都不算。"""
    gp.sync_from_lessons(prof)
    world = gp.load_world(prof)
    g = {x.id: x for x in gp.load_groups(prof)}["刷一局"]
    r = gp.GroupRunner(g, world, None, executor=_Exec(),
                       fsm=SimpleNamespace(on_scene=lambda s: None))
    r.tick_frame(_scene(ts_ms=0), np.zeros((10, 10, 3), np.uint8))
    assert r.step_i == 0
    r.tick_frame(_scene(ts_ms=10, in_game=False,
                        rechallenge=SimpleNamespace(x=1, y=2)),
                 np.zeros((10, 10, 3), np.uint8))
    assert r.step_i == 1                 # 这一步过了，进到"点再次挑战"


def test_delegate_fails_fast_when_engine_is_paused(prof):
    """"交给引擎"的时候引擎停着 = 空转。

    实测踩到的：FSM 处于 PAUSED，组照样每帧调 on_scene（直接 return），
    于是稳稳停在第 1 步，300 秒后才报"超时" —— 而真正的原因一个字都没提。
    停车要立刻说。
    """
    gp.sync_from_lessons(prof)
    world = gp.load_world(prof)
    g = {x.id: x for x in gp.load_groups(prof)}["刷一局"]
    called: list[int] = []
    r = gp.GroupRunner(g, world, None, executor=_Exec(),
                       fsm=SimpleNamespace(state="PAUSED",
                                           on_scene=lambda s: called.append(s.ts_ms)))
    r.tick_frame(_scene(ts_ms=0), np.zeros((10, 10, 3), np.uint8))
    assert r.done and "暂停" in r.failed
    assert not called                 # 没有假装在执行


def test_delegate_fails_fast_when_engine_is_paused(prof):
    """"交给引擎"的时候引擎停着 = 空转。

    实测踩到的：FSM 处于 PAUSED，组照样每帧调 on_scene（直接 return），
    于是稳稳停在第 1 步，300 秒后才报"超时" —— 而真正的原因一个字都没提。
    停车要立刻说。
    """
    gp.sync_from_lessons(prof)
    world = gp.load_world(prof)
    g = {x.id: x for x in gp.load_groups(prof)}["刷一局"]
    called: list[int] = []
    r = gp.GroupRunner(g, world, None, executor=_Exec(),
                       fsm=SimpleNamespace(state="PAUSED",
                                           on_scene=lambda s: called.append(s.ts_ms)))
    r.tick_frame(_scene(ts_ms=0), np.zeros((10, 10, 3), np.uint8))
    assert r.done and "暂停" in r.failed
    assert not called                 # 没有假装在执行
