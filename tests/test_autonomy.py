"""自主恢复测试 —— 看门狗判定、恢复阶梯、铁律（绝不按拾取键）、循环兜底。

这些用例模拟"搬砖机器人跑了几小时之后会遇到的各种卡死"：
画面冻结、房间走空、弹窗挡路、时间戳跳变、代理掉线。
核心断言只有一句：**任何情况下都不能停在那里，也绝不能盲按拾取键。**
"""

from __future__ import annotations

import time
from types import SimpleNamespace

from autogame.config import (AppConfig, AutonomyConfig, DecisionConfig,
                             ExecutorConfig, VisionConfig)
from autogame.decision.fsm import DecisionEngine, State
from autogame.decision.recovery import ProgressMonitor, RecoveryLadder
from autogame.executor import DryRunExecutor
from autogame.protocol import Monster, Pillar, SceneState
from autogame.server.hub import Hub
from autogame.vision.pipeline import VisionPipeline


def _scene(ts, pillars=None, delta=50.0, in_game=True, loading=False,
           rechallenge=None, monsters=None, character=(640, 360)):
    return SceneState(
        ts_ms=ts, w=1280, h=720, character=character,
        pillars=pillars or [], rechallenge=rechallenge,
        monsters=monsters or [], in_game=in_game, loading=loading,
        frame_delta=delta,
    )


def _pillar(cx=900, cy=400):
    return Pillar(x=cx, y=cy + 55, cx=cx, cy=cy, w=14, h=110, color="gold",
                  area=1500, distance=((cx - 640) ** 2 + (cy - 360) ** 2) ** 0.5,
                  whitelisted=True, stable=True)


def _engine(**over):
    dcfg = DecisionConfig(tick_hz=10.0, approach_mode="mouse_click")
    a = dcfg.autonomy
    a.frozen_window_s = 5.0
    a.idle_window_s = 6.0
    a.stall_window_s = 12.0
    a.gate_window_s = 5.0
    a.level_interval_s = 2.0
    a.step_interval_s = 0.2
    a.cycle_backoff_s = 0.0
    for k, v in over.items():
        setattr(a, k, v)
    ex = DryRunExecutor(ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000))
    eng = DecisionEngine(dcfg, ex)
    eng.start()
    return eng, ex


def _events(eng, name):
    return [e for e in eng.events if e["event"] == name]


# ---------------- 看门狗判定 ----------------

def test_monitor_verdicts_progression():
    """ok → frozen → idle → stalled 的判定顺序。"""
    cfg = AutonomyConfig(frozen_window_s=3.0, idle_window_s=5.0, stall_window_s=15.0)
    mon = ProgressMonitor(cfg)
    mon.reset(0.0)

    s = mon.update(_scene(1000, pillars=[_pillar()], delta=50.0), 1.0, 0)
    assert s.verdict == "ok"

    # 画面不动（撞墙/假死）—— 前提是我们确实发出了动作（否则只是"没事干"）
    for t in (2.0, 3.0, 4.0):
        mon.note_action(t)
        s = mon.update(_scene(int(t * 1000), pillars=[_pillar()], delta=0.0), t, 0)
    assert s.verdict == "frozen" and s.frozen_for >= 3.0

    # 画面在动但没有目标（房间走空）
    for t in (5.0, 6.0, 7.0, 8.0, 9.0):
        s = mon.update(_scene(int(t * 1000), delta=50.0), t, 0)
    assert s.verdict == "idle" and s.idle_for >= 5.0

    # 目标一直在但长时间没有任何进展
    mon2 = ProgressMonitor(AutonomyConfig(frozen_window_s=99, idle_window_s=99,
                                          stall_window_s=5.0))
    mon2.reset(0.0)
    for t in (1.0, 3.0, 6.0):
        s = mon2.update(_scene(int(t * 1000), pillars=[_pillar()], delta=50.0), t, 0)
    assert s.verdict == "stalled" and s.stall_for >= 5.0


def test_monitor_pick_counts_as_progress():
    cfg = AutonomyConfig(stall_window_s=5.0)
    mon = ProgressMonitor(cfg)
    mon.reset(0.0)
    for t in (1.0, 3.0, 6.0):
        mon.update(_scene(int(t * 1000), delta=50.0), t, 0)
    s = mon.update(_scene(7000, delta=50.0), 7.0, 1)  # 拾取 +1
    assert s.stall_for == 0.0


# ---------------- 冻结 → 恢复阶梯 ----------------

def test_frozen_enters_recovery_and_escalates():
    eng, ex = _engine()
    for i in range(18):
        eng.on_scene(_scene(1000 + i * 1000, delta=0.0))

    assert eng.state == State.RECOVERING
    assert _events(eng, "recovery_start")
    steps = {e.get("step") for e in _events(eng, "recovery_step")}
    # 由轻到重：先重定位/探路，再 ESC/脱困
    assert {"retarget", "probe"} <= steps
    assert eng.recovery.level >= 2, f"恢复等级应随卡壳时长升级，实际 {eng.recovery.level}"
    assert eng.stats["recovery_steps"] > 0


def test_recovery_never_presses_pickup_key():
    """铁律：恢复动作里绝不能出现拾取键（否则就是盲捡）。"""
    eng, ex = _engine()
    for i in range(40):
        eng.on_scene(_scene(1000 + i * 1000, delta=0.0))

    keys = [j["key"] for j in ex.journal if j["op"] == "key_tap"]
    assert "a" not in keys, f"恢复过程中出现拾取键: {keys}"
    assert eng.stats["blind_tap_violations"] == 0
    assert "esc" in keys or "space" in keys, "应使用 ESC/脱困键自救"


def test_recovery_exits_when_progress_returns():
    eng, ex = _engine()
    for i in range(10):
        eng.on_scene(_scene(1000 + i * 1000, delta=0.0))
    assert eng.state == State.RECOVERING

    # 卡壳解除：画面恢复 + 出现可交互目标
    for i in range(10, 15):
        eng.on_scene(_scene(1000 + i * 1000, pillars=[_pillar()], delta=50.0))
    assert _events(eng, "recovery_done"), "有进展必须退出恢复，不能一直自嗨"
    assert eng.state in (State.SCANNING, State.APPROACHING, State.COOLDOWN)


def test_recovery_cycles_and_keeps_trying():
    """恢复失败 → 换一轮继续试（绝不放弃），并产生告警。"""
    eng, ex = _engine(level_interval_s=0.5, step_interval_s=0.1)
    for i in range(40):
        eng.on_scene(_scene(1000 + i * 1000, delta=0.0))

    assert eng.stats["recovery_cycles"] >= 1
    assert eng.stats["stuck_alarms"] >= 1
    assert eng.state == State.RECOVERING

    before = len(ex.journal)
    for i in range(40, 60):
        eng.on_scene(_scene(1000 + i * 1000, delta=0.0))
    assert len(ex.journal) > before, "多轮失败后仍必须继续尝试（不能停在那里）"


def test_pause_after_cycles_is_opt_in():
    """可选安全阀：配置了 pause_after_cycles 才会暂停（默认无限重试）。"""
    eng, ex = _engine(level_interval_s=0.3, step_interval_s=0.05,
                      cycle_backoff_s=0.0, pause_after_cycles=1)
    for i in range(40):
        eng.on_scene(_scene(1000 + i * 1000, delta=0.0))
    assert eng.state == State.PAUSED


def test_gated_escape_when_modal_blocks():
    """长时间不在战斗界面（弹窗/菜单）→ 自动 ESC 自救。"""
    eng, ex = _engine()
    for i in range(12):
        eng.on_scene(_scene(1000 + i * 1000, in_game=False, delta=50.0))
    assert eng.stats["gate_escapes"] >= 1
    assert any(j["key"] == "esc" for j in ex.journal if j["op"] == "key_tap")


def test_loading_hold_never_acts():
    eng, ex = _engine()
    for i in range(20):
        eng.on_scene(_scene(1000 + i * 1000, loading=True, delta=0.0))
    assert len(ex.journal) == 0, "加载/黑屏过场必须保持不动"
    assert eng.state != State.RECOVERING


def test_clock_rebase_on_agent_restart():
    """代理重启导致帧时间戳回退：必须重基准，不能把超时判据算成负数。"""
    eng, ex = _engine()
    eng.on_scene(_scene(10000, pillars=[_pillar()]))
    eng.on_scene(_scene(500, pillars=[_pillar()]))  # 时间倒流
    assert _events(eng, "clock_rebase"), "时间戳回退必须记录并重基准"
    assert eng.state != State.RECOVERING


# ---------------- 空闲自主性 ----------------

def test_idle_explores_by_clicking_ground():
    """没有目标时自己去找（点地面），而不是站着等刷新。"""
    eng, ex = _engine(frozen_window_s=999, idle_window_s=999, stall_window_s=999)
    for i in range(3):
        eng.on_scene(_scene(1000 + i * 1000, delta=50.0))
    clicks = [j for j in ex.journal if j["op"] == "mouse_click"]
    assert clicks, "空闲时必须主动探索"
    assert any(e.get("mode") == "click" for e in _events(eng, "explore_move"))
    assert not [j for j in ex.journal if j["op"] == "key_tap" and j["key"] == "a"]


def test_idle_attacks_monsters():
    eng, ex = _engine()
    eng.on_scene(_scene(1000, monsters=[Monster(x=800, y=400, w=40, h=40)], delta=50.0))
    assert _events(eng, "attack_monster"), "看到怪必须打（搬砖目标是通关）"


# ---------------- 恢复阶梯单元 ----------------

class _FakeVLM:
    cfg = SimpleNamespace(enabled=True)

    def __init__(self) -> None:
        self.asked = 0

    def request(self, ctx, now=None):
        self.asked += 1
        return True

    def take_advice(self):
        return {"action": "walk_to", "target": [300, 300], "valid": True,
                "reason": "左侧有门", "confidence": 0.8}

    def snapshot(self):
        return {}


def _ladder(level: str, **cfg_over):
    a = AutonomyConfig(step_interval_s=0.05)
    for k, v in cfg_over.items():
        setattr(a, k, v)
    vlm = _FakeVLM()
    lad = RecoveryLadder(a, vlm, None, forbidden_keys={"a"})
    lad.begin(0.0)
    lad.level = lad.STEPS.index(level)
    return lad, vlm


def test_ladder_vlm_request_then_execute():
    lad, vlm = _ladder("vlm")
    r1 = lad.step(0.0, _scene(1000))
    assert vlm.asked == 1 and "请求" in r1.note
    r2 = lad.step(1.0, _scene(1000))
    assert r2.commands and r2.commands[0].x == 300 and r2.commands[0].y == 300


def test_ladder_filters_forbidden_key():
    """即使恢复阶梯想按拾取键，也会被安全过滤掉。"""
    lad, _ = _ladder("spin")
    for i in range(8):
        r = lad.step(float(i), _scene(1000), {"approach_mode": "keys"})
        assert all((c.key or "") != "a" for c in r.commands), "拾取键必须被过滤"


def test_ladder_stage_reset_clicks_rechallenge():
    from autogame.protocol import ButtonHit

    lad, _ = _ladder("stage_reset")
    r = lad.step(0.0, _scene(1000, rechallenge=ButtonHit(x=640, y=570, w=240, h=52, score=0.9)))
    assert r.commands and r.commands[0].x == 640 and r.commands[0].y == 570


# ---------------- 帧流看门狗 ----------------

def test_frame_stall_watchdog():
    cfg = AppConfig()
    cfg.server.event_log = None
    cfg.decision.autonomy.frame_stall_s = 5.0
    hub = Hub(cfg, DryRunExecutor(ExecutorConfig(min_action_interval_ms=0,
                                                 max_actions_per_sec=1000)),
              VisionPipeline(VisionConfig()))
    hub.fsm.state = State.SCANNING
    hub._latest_ts = time.time() - 10
    hub.check_frame_stall()
    assert any(e["event"] == "frame_stall" for e in hub.fsm.events)

    hub._latest_ts = time.time()
    hub.check_frame_stall()
    assert any(e["event"] == "frame_resumed" for e in hub.fsm.events)


def test_frame_stall_ignored_when_paused():
    cfg = AppConfig()
    cfg.server.event_log = None
    cfg.decision.autonomy.frame_stall_s = 1.0
    hub = Hub(cfg, DryRunExecutor(ExecutorConfig(min_action_interval_ms=0,
                                                 max_actions_per_sec=1000)),
              VisionPipeline(VisionConfig()))
    hub._latest_ts = time.time() - 30
    hub.check_frame_stall()  # PAUSED → 不该报警
    assert not any(e["event"] == "frame_stall" for e in hub.fsm.events)


def test_escape_step_skipped_while_still_in_combat():
    """真实录像验证：卡壳时如果仍判定在战斗界面，ESC 只会开暂停菜单 → 绝不能按。

    （暗黑4 实战帧上曾连按 6 次 ESC：机器人把自己锁进菜单，比卡住更糟。）
    """
    eng, ex = _engine(level_interval_s=0.3, step_interval_s=0.05,
                      stall_window_s=1.0, frozen_window_s=999, idle_window_s=999)
    # 有目标但零进展 → stalled → 阶梯升级到 escape；in_game 始终 True
    for i in range(40):
        eng.on_scene(_scene(1000 + i * 500, pillars=[_pillar()], delta=50.0))
    esc = [j for j in ex.journal if j["op"] == "key_tap" and j["key"] == "esc"]
    assert esc == [], "在战斗界面按 ESC 会弹暂停菜单，恢复阶梯必须跳过这一级"
    assert len(ex.journal) > 0, "跳过 ESC 不等于停手：其它恢复动作仍要发"


def test_escape_step_works_when_gate_says_not_in_game():
    """门禁判定不在战斗界面（真弹窗/菜单）时，ESC 才是正确的自救动作。"""
    from autogame.decision.recovery import RecoveryLadder

    a = AutonomyConfig()
    lad = RecoveryLadder(a, None, None, forbidden_keys={"a"})
    lad.begin(0.0)
    lad.level = RecoveryLadder.STEPS.index("escape")
    scene = _scene(1000, in_game=False)
    res = lad.step(100.0, scene, {"in_game": False})
    assert any(c.op == "key_tap" and c.key == a.escape_key for c in res.commands), \
        "非战斗界面下 ESC 必须能发出来"
