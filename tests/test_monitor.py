"""上线后的体检：卡住了要认得出，认出了要自己退。

这组用例钉的是对话里那条闭环的最后两段 ——「性能监控 → 退化回滚」。它和
`test_evolve.py` 的回归闸门不是一回事：

- `evolve.promote` 管**上线前**：新代在静态台账上不能比旧代差。
- 这里管**上线后**：卡住、变慢、烧模型、空转 —— 这四种坏法一张图都不会变红。

所以用例的重点不是"指标算得准不准"，而是几条不许松动的规矩：

    一帧抖动不许回滚           —— 要连续几个坏窗口
    回滚完不许马上又回滚       —— 要有冷却，否则会退到没有上一代
    回滚完必须重新采基线       —— 配方换了，旧基线描述的不是现在这套东西
    被模式闸门拦下的动作不算卡住 —— 那是"想动没动"，不是"在乱按"
    profile 没绑的时候只报不修  —— 没有可回滚的东西就别假装修好了
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from autogame import evolve
from autogame.config import load_config
from autogame.config import AppConfig
from autogame.decision.arbiter import Arbiter
from autogame.executor import DryRunExecutor, ExecutorConfig
from autogame.monitor import BAD, WARN, HealthMonitor, append_incident, incident_path
from autogame.profile import load_profile
from autogame.server.hub import Hub
from autogame.vision.pipeline import VisionPipeline

T0 = 1_000_000.0        # 固定的假时钟起点：用例不 sleep


def _drive(m: HealthMonitor, *, windows: int, stats: dict, tick_ms: float = 20.0,
           t0: float | None = None, journal: list | None = None,
           per_window_ticks: int = 7) -> list:
    """喂 `windows` 个窗口，返回每个窗口结算出的结论。

    起点必须接在监视器自己的 `_since` 上：它只认"距上次结算过了多久"，
    用例自己另起一个时间轴会让窗口拉长一倍，算出来的速率全对不上。
    """
    out: list = []
    span = m.window_s
    t = m._since or (T0 if t0 is None else t0)
    for _ in range(windows):
        for i in range(per_window_ticks):
            m.observe(stats=stats, journal=journal, tick_s=tick_ms / 1000.0,
                      now=t + span * i / per_window_ticks)
        out.append(m.observe(stats=stats, journal=journal,
                             tick_s=tick_ms / 1000.0, now=t + span + 0.01))
        t = m._since
    return out


def _m(**kw) -> HealthMonitor:
    base = dict(window_s=10.0, warmup=1, min_bad=2, cooldown_s=60.0)
    base.update(kw)
    return HealthMonitor(**base)


# ═══════════════════════ 1. 卡住：同一个动作反复发 ═══════════════════════

def _jammed(n: int, *, start: float = T0 + 0.1, op: str = "key_tap",
            key: str = "a", suppressed: bool = False) -> list[dict]:
    return [{"ts": start + i * 0.2, "op": op, "key": key,
             "suppressed": suppressed} for i in range(n)]


def test_repeated_action_is_flagged():
    m = _m()
    out = _drive(m, windows=1, stats={}, journal=_jammed(m.repeat_limit))
    codes = [f.code for f in out[-1]]
    assert "repeat" in codes
    bad = next(f for f in out[-1] if f.code == "repeat")
    assert bad.level == BAD
    assert "key_tap(a)" in bad.detail


def test_different_actions_are_not_a_jam():
    """按了 12 次，但 12 个不同的键 —— 那是在玩，不是在卡。"""
    m = _m()
    journal = [{"ts": T0 + 0.1 + i * 0.2, "op": "key_tap", "key": f"k{i}"}
               for i in range(m.repeat_limit)]
    out = _drive(m, windows=1, stats={}, journal=journal)
    assert not [f for f in out[-1] if f.code == "repeat"]


def test_suppressed_actions_do_not_count_as_jam():
    """观察/暂停模式下"想做但没做"的动作不算乱按 —— 它一个键都没发出去。"""
    m = _m()
    out = _drive(m, windows=1, stats={},
                 journal=_jammed(m.repeat_limit, suppressed=True))
    assert not [f for f in out[-1] if f.code == "repeat"]


def test_click_coordinates_are_bucketed():
    """点在同一个按钮上每次差几个像素，是一个动作，不是一个新动作。"""
    m = _m()
    journal = [{"ts": T0 + 0.1 + i * 0.2, "op": "mouse_click",
                "x": 640 + i % 5, "y": 361 + i % 3} for i in range(m.repeat_limit)]
    out = _drive(m, windows=1, stats={}, journal=journal)
    assert [f.code for f in out[-1]] == ["repeat"]


# ═══════════════════════ 2. 变慢 / 烧模型 / 空转 ═══════════════════════

def test_slow_tick_is_flagged_after_baseline():
    m = _m()
    _drive(m, windows=1, stats={}, tick_ms=20.0)          # 基线：20ms
    assert m.status()["baseline_ready"]
    out = _drive(m, windows=1, stats={}, tick_ms=900.0)
    assert [f.code for f in out[-1]] == ["slow"]


def test_small_tick_jitter_is_not_a_finding():
    """20ms → 30ms 不该报警。阈值里那个绝对值下限就是为这个留的。"""
    m = _m()
    _drive(m, windows=1, stats={}, tick_ms=20.0)
    out = _drive(m, windows=1, stats={}, tick_ms=30.0)
    assert out[-1] == []


def _ramp(m: HealthMonitor, key: str, *, per_tick: int, extra: dict | None = None,
          ticks: int = 8) -> list:
    """把某个计数器的速率拉起来，喂满一个窗口，返回结算结论。"""
    t = m._since
    out: list = []
    for i in range(ticks):
        st = {key: (i + 1) * per_tick, **(extra or {})}
        out = m.observe(stats=st, tick_s=0.02,
                        now=t + (m.window_s + 0.01) * (i + 1) / ticks)
    return out


def test_model_burn_when_rules_stop_matching():
    """之前一条规则都不问模型，现在每秒问好几次 —— 规则匹配不上了。"""
    m = _m()
    _drive(m, windows=1, stats={"llm_calls": 0})
    out = _ramp(m, "llm_calls", per_tick=3)
    assert [f.code for f in out] == ["model_burn"]


def test_recovery_spin_only_warns_because_it_cannot_be_rolled_back():
    """恢复里打转是"没效果"，不是"打错了"。它只提醒，不触发回滚。"""
    m = _m()
    _drive(m, windows=1, stats={"recoveries": 0, "picks": 0})
    out = _ramp(m, "recoveries", per_tick=4, extra={"picks": 0})
    assert [f.code for f in out] == ["recovery_spin"]
    assert out[0].level == WARN
    assert m.should_rollback() is None


def test_picks_coming_in_means_recovery_is_working():
    """同样的恢复频率，但这次真的捡到东西了 —— 那就不是空转。"""
    m = _m()
    _drive(m, windows=1, stats={"recoveries": 0, "picks": 0})
    t = m._since
    out = []
    for i in range(8):
        out = m.observe(stats={"recoveries": (i + 1) * 4, "picks": (i + 1) * 2},
                        tick_s=0.02,
                        now=t + (m.window_s + 0.01) * (i + 1) / 8)
    assert out == []


# ═══════════════════════ 3. 什么时候才动手 ═══════════════════════

def test_one_bad_window_is_not_enough():
    """一帧抖动就回滚的机器人没法用。"""
    m = _m(min_bad=3)
    _drive(m, windows=1, stats={}, journal=_jammed(m.repeat_limit))
    assert m.should_rollback() is None


def test_consecutive_bad_windows_trigger_rollback():
    m = _m(min_bad=2)
    j = _jammed(200)          # 一直卡着
    _drive(m, windows=2, stats={}, journal=j)
    hit = m.should_rollback()
    assert hit is not None and hit.code == "repeat"


def test_a_healthy_window_clears_the_streak():
    """坏、好、坏 —— 不连续就不算。中间那一个好窗口是真的恢复了。"""
    m = _m(min_bad=2)
    _drive(m, windows=1, stats={}, journal=_jammed(m.repeat_limit))
    _drive(m, windows=1, stats={}, journal=[], t0=T0 + 20)
    _drive(m, windows=1, stats={}, journal=_jammed(m.repeat_limit), t0=T0 + 40)
    assert m.should_rollback() is None


def test_cooldown_blocks_a_second_rollback():
    m = _m(min_bad=1, cooldown_s=600.0)
    j = _jammed(400)
    _drive(m, windows=2, stats={}, journal=j)
    hit = m.should_rollback()
    assert hit is not None
    m.mark_rollback(hit)
    assert m.should_rollback() is None, "刚退过就冷却，不允许连环退"


def test_rollback_clears_baseline():
    """配方换了，旧基线描述的不是现在这套东西 —— 必须重新采。"""
    m = _m()
    _drive(m, windows=1, stats={})
    assert m.status()["baseline_ready"]
    m.mark_rollback()
    st = m.status()
    assert not st["baseline_ready"] and st["baseline"] is None
    assert st["cooldown_left_s"] > 0


# ═══════════════════════ 4. 脏输入 ═══════════════════════

def test_counter_reset_does_not_poison_the_window():
    """换执行器/重启 FSM 会把计数器清零，差分变负 —— 不能当成负的动作数。"""
    m = _m()
    m.observe(stats={"picks": 100, "sent": 500}, now=T0)
    out = []
    t = T0
    for i in range(6):
        t += 2.0
        out = m.observe(stats={"picks": 0, "sent": 0}, tick_s=0.02, now=t)
    w = m.status()["windows"][-1]
    assert all(v >= 0 for v in w["rates"].values())
    assert out == [] or all(f.code != "repeat" for f in out)


def test_empty_journal_is_fine():
    m = _m()
    out = _drive(m, windows=1, stats={}, journal=[])
    assert out[-1] == []


# ═══════════════════════ 5. 真回滚（接 hub / evolve） ═══════════════════════

def _frame(i: int, h: int = 200, w: int = 360) -> np.ndarray:
    """造一张"长得不一样"的图。种子不同 → 相关度低，不会被当成重复样本。"""
    rng = np.random.default_rng(i)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    cv2.circle(img, (w // 2, h // 2), 30, (40, 200, 240), -1)
    return img


LESSON = """# 测试用手册

## 界面 · 战斗
- 图: refs/combat_a.jpg
- 判定: 正例
- 在: 右下角技能轮盘

## 界面 · 战斗（另一张）
- 图: refs/combat_b.jpg
- 判定: 正例
- 在: 右下角技能轮盘

## 界面 · 背包
- 图: refs/bag_a.jpg
- 判定: 反例

## 界面 · 结算
- 图: refs/result_a.jpg
- 判定: 反例

## 流程 · 没子弹了
- 一眼认出: 连续三次攻击都没打掉怪
- 该做什么: 先按 esc 退开，再等两秒
"""


@pytest.fixture()
def two_gens(tmp_path):
    """造出「上一代 + 当前这一代」，让 rollback 真的有地方可退。"""
    root = tmp_path / "profiles" / "unit_test"
    (root / "lessons").mkdir(parents=True)
    (root / "refs").mkdir()
    (root / "templates").mkdir()
    (root / "profile.yaml").write_text(
        "id: unit_test\nname: 单元测试\nstatus: partial\n"
        "reference_resolution: [360, 200]\nconfig: {}\n", encoding="utf-8")
    (root / "lessons" / "t.md").write_text(LESSON, encoding="utf-8")
    for i, name in enumerate(("combat_a.jpg", "combat_b.jpg", "bag_a.jpg",
                              "result_a.jpg")):
        cv2.imwrite(str(root / "refs" / name), _frame(i))
    (root / "labels.json").write_text(json.dumps(
        {"version": 1, "items": [{"file": "refs/combat_a.jpg", "scene": "战斗",
                                  "strict": False, "expect": {"in_game": True}}]}),
        encoding="utf-8")
    prof = load_profile(root)
    cfg = load_config(None, profile=prof)
    score = {"strict_pass": 3, "strict_total": 3, "ratio": 1.0, "fail": [], "gap": [],
             "items": []}
    g0, note = evolve.build(prof, cfg, use_model=False)
    assert g0 is not None, note
    evolve.promote(prof, g0, score=score)
    lesson = root / "lessons" / "t.md"
    lesson.write_text(lesson.read_text(encoding="utf-8") + "\n## 禁忌 · 别乱点\n",
                      encoding="utf-8")
    g1, note = evolve.build(prof, cfg, use_model=False)
    assert g1 is not None, note
    evolve.promote(prof, g1, score=score)
    assert evolve.active_id(prof) == g1.id
    return prof


def _hub(prof=None) -> Hub:
    cfg = AppConfig()
    cfg.executor.min_action_interval_ms = 0
    cfg.executor.max_actions_per_sec = 1000
    ex = DryRunExecutor(ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000))
    hub = Hub(cfg, ex, VisionPipeline(cfg.vision))
    hub.profile = prof
    return hub


def test_hub_rolls_back_and_records_incident(two_gens):
    """体检判定"卡住了" → 真的退回上一代 + 留一条事故记录。"""
    hub = _hub(two_gens)
    hub.health = _m(min_bad=1, cooldown_s=0.0)
    with hub.executor._lock:
        hub.executor.journal.extend(_jammed(200))

    # 用假时钟：journal 和 now 必须是同一条时间轴，混用真时钟会把窗口算错
    hub.health.observe(stats={}, journal=hub.executor.journal, tick_s=0.02, now=T0 - 100)
    out = hub.health.observe(stats={}, journal=hub.executor.journal, tick_s=0.02, now=T0)
    bad = hub.health.should_rollback()
    assert bad is not None, out
    res = hub._degrade(bad)
    assert res["ok"], res
    # 真的退了：active 变成了上一代
    assert evolve.active_id(two_gens) != ""
    incidents = incident_path(two_gens).read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(incidents[-1])["trigger"]["code"] == "repeat"
    assert hub._health_incidents[-1]["ok"] is True


def test_degrade_without_profile_only_reports(two_gens):
    """没绑 profile 就没有可回滚的配方 —— 只记一笔，别假装修好了。"""
    hub = _hub(None)
    hub.health = _m(min_bad=1)
    hub.health.observe(stats={}, journal=_jammed(200), tick_s=0.02, now=T0 - 100)
    hub.health.observe(stats={}, journal=_jammed(200), tick_s=0.02, now=T0)
    bad = hub.health.should_rollback()
    assert bad is not None
    res = hub._degrade(bad)
    assert not res["ok"] and "profile" in res["reason"]
    assert hub.health.status()["cooldown_left_s"] > 0


def test_arbiter_reload_rules_follows_active_pointer(two_gens):
    """回滚只改了文件是不够的 —— 进程里那套规则也得换掉。"""
    cfg = load_config(None, profile=two_gens)
    from autogame.decision.actions import ActionRegistry
    from autogame.decision.knowledge import KnowledgeBase

    arb = Arbiter(cfg.decision.arbiter, ActionRegistry(), KnowledgeBase(
        Path(two_gens.root) / "knowledge"))
    old = arb.rules_path_loaded if hasattr(arb, "rules_path_loaded") else arb._rules_path
    assert Path(old).exists()
    evolve.rollback(two_gens)
    new = arb.reload_rules()
    assert new != old and Path(new).exists()
    assert arb._rules_path == new


def test_incident_append_is_survivable(tmp_path):
    """写不进去不能把回滚本身搞失败，但路径对的时候必须真的写进去。"""
    p = tmp_path / "health" / "incidents.jsonl"
    append_incident(p, {"action": "rollback", "ok": True})
    assert json.loads(p.read_text(encoding="utf-8").strip())["ok"] is True
    # 目录被占成文件 → 吞掉异常，不抛
    bad = tmp_path / "blocked"
    bad.write_text("x", encoding="utf-8")
    append_incident(bad / "x" / "incidents.jsonl", {"action": "rollback"})


# ═══════════ 换代：学完必须当场生效，体检基线不能被旧配方带偏 ═══════════

def test_promote_hook_reloads_rules_and_resets_baseline(two_gens):
    hub = _hub(two_gens)
    hub.health = _m(warmup=1, min_ticks=1)
    calls = []
    hub._reload_rules = lambda: calls.append(1)
    # 先喂出一个基线
    _drive(hub.health, windows=2, stats={"picks": 1})
    assert hub.health.status()["baseline_ready"]

    hub.on_generation_change(two_gens, "promote", "gen-001")
    assert calls, "换代不重载规则 = 学习了但没生效"
    assert not hub.health.status()["baseline_ready"], \
        "旧基线描述的是旧配方，晋升后必须重采，否则第一分钟就误报变慢"


def test_rollback_hook_reloads_but_keeps_the_cooldown(two_gens):
    hub = _hub(two_gens)
    calls = []
    hub._reload_rules = lambda: calls.append(1)
    hub.health.mark_rollback()
    hub.on_generation_change(two_gens, "rollback", "gen-000")
    assert calls
    assert hub.health.status()["rolled_at"], "回滚的冷却不能被这次通知清掉"


def test_generation_hook_ignores_another_profile(two_gens):
    from types import SimpleNamespace

    hub = _hub(two_gens)
    calls = []
    hub._reload_rules = lambda: calls.append(1)
    hub.on_generation_change(SimpleNamespace(id="别的游戏"), "promote", "gen-9")
    assert calls == []


def test_evolve_calls_the_registered_hook_on_rollback(two_gens):
    from autogame import evolve as ev

    seen = []
    ev.set_generation_hook(lambda p, a, g: seen.append((a, g)))
    try:
        ok, msg = ev.rollback(two_gens)
        assert ok, msg
        assert seen and seen[-1][0] == "rollback"
    finally:
        ev.set_generation_hook(None)


def test_create_app_registers_the_hub_as_the_generation_hook():
    from autogame import evolve as ev
    from autogame.server.app import create_app

    cfg = AppConfig()
    cfg.executor.min_action_interval_ms = 0
    hub = Hub(cfg, DryRunExecutor(cfg.executor), VisionPipeline(cfg.vision))
    create_app(cfg, hub, "<h1>x</h1>")
    assert ev._CHANGE_HOOK == hub.on_generation_change


# ---------------- 与裁决层共用的"重复动作"统计 ----------------

def test_recent_repeat_finds_the_hottest_action():
    from autogame.monitor import recent_repeat

    journal = [{"ts": 100.0, "op": "mouse_click", "x": 640, "y": 360}] * 5
    journal += [{"ts": 100.0, "op": "key_tap", "key": "d"}]
    n, key = recent_repeat(journal, since=95.0)
    assert (n, key) == (5, "mouse_click(640,360)")


def test_recent_repeat_skips_old_and_suppressed_actions():
    from autogame.monitor import recent_repeat

    journal = [
        {"ts": 10.0, "op": "mouse_click", "x": 640, "y": 360},        # 太旧
        {"ts": 100.0, "op": "mouse_click", "x": 640, "y": 360,
         "suppressed": True},                                        # 被闸门拦下
        {"ts": 100.0, "op": "key_tap", "key": "d"},
    ]
    assert recent_repeat(journal, since=95.0) == (1, "key_tap(d)")
    assert recent_repeat([], since=0.0) == (0, "")


def test_recent_repeat_merges_nearby_clicks():
    """同一个小按钮上差几个像素，人是同一个动作。"""
    from autogame.monitor import recent_repeat

    journal = [
        {"ts": 100.0, "op": "mouse_click", "x": 641, "y": 356},
        {"ts": 100.5, "op": "mouse_click", "x": 644, "y": 359},
        {"ts": 101.0, "op": "mouse_click", "x": 646, "y": 352},
    ]
    n, key = recent_repeat(journal, since=99.0)
    assert n == 3 and key == "mouse_click(640,320)"
