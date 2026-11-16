"""卡壳自愈端到端测试 —— 真实模拟世界 + 视觉检测 + 执行器闭环。

场景：角色被"卡死"（走不动、没有任何掉落），机器人必须自己发现卡住、
换招自救；一旦卡壳解除（掉落出现），必须立刻回到拾取，且全程不盲按拾取键。
"""

from __future__ import annotations

from autogame.config import DecisionConfig, ExecutorConfig, VisionConfig
from autogame.decision.fsm import DecisionEngine, State
from autogame.executor import DryRunExecutor
from autogame.sim.scene import SimPillar, SimWorld
from autogame.vision.detector import Detector


def _rig():
    world = SimWorld(seed=3, pillar_spawn_interval_s=999, walk_speed_px_s=0.0)
    world.pillars.clear()
    det = Detector(VisionConfig())

    dcfg = DecisionConfig(tick_hz=2.0, approach_mode="mouse_click")
    dcfg.autonomy.frozen_window_s = 3.0
    dcfg.autonomy.idle_window_s = 999.0
    dcfg.autonomy.stall_window_s = 999.0
    dcfg.autonomy.level_interval_s = 1.0
    dcfg.autonomy.step_interval_s = 0.05
    dcfg.autonomy.cycle_backoff_s = 0.0
    ex = DryRunExecutor(ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000))
    eng = DecisionEngine(dcfg, ex)
    eng.start()

    state = {"ts": 1000, "keys": []}

    def tick() -> None:
        frame = world.render()
        det.process(frame, state["ts"])
        scene = det.process(frame, state["ts"] + 100)
        state["ts"] += 1000
        eng.on_scene(scene)
        for j in list(ex.journal):
            if j["op"] == "key_tap":
                state["keys"].append(j["key"])
            world.on_command(j["op"], key=j["key"], button=j["button"],
                             x=j["x"], y=j["y"], hold_ms=j["hold_ms"])
        ex.journal.clear()
        world.tick()

    return world, eng, ex, state, tick


def test_stuck_world_recovers_and_resumes_picking():
    world, eng, ex, st, tick = _rig()

    # 阶段1：卡死 12 秒（走不动 + 无掉落 + 画面完全静止）
    for _ in range(12):
        tick()
    assert eng.state == State.RECOVERING, f"卡死必须被看门狗抓到，事件={eng.events[-6:]}"
    assert eng.stats["recoveries"] >= 1
    assert world.blind_a_press == 0, "恢复过程中绝不能盲按拾取键"
    assert "a" not in st["keys"], f"恢复动作里出现了拾取键: {st['keys']}"
    # 必须真的发出过自救动作（点地面探路 / ESC / 脱困键）
    ops = [j["op"] for j in ex.journal]
    assert eng.stats["recovery_actions"] >= 1
    assert any(k in ("esc", "space") for k in st["keys"]) or eng.stats["recovery_actions"] > 0
    del ops

    # 阶段2：卡壳解除（能走了 + 掉落出现在脚边）
    world.walk_speed = 260.0
    world.pillars.append(SimPillar(0, 60, "gold"))
    for _ in range(12):
        tick()

    assert any(e["event"] == "recovery_done" for e in eng.events), "有目标后必须退出恢复"
    assert world.picked >= 1, f"恢复后必须能正常拾取，实际 picked={world.picked}"
    assert world.blind_a_press == 0, "全程不允许盲按拾取键"
    assert eng.stats["picks"] >= 1


def test_recovery_never_blocks_decision_loop():
    """恢复期间每帧都必须有决策（不能因为"在恢复"就停止 tick）。"""
    world, eng, ex, st, tick = _rig()
    for _ in range(8):
        tick()
    seq_before = len(eng.events)
    for _ in range(8):
        tick()
    assert len(eng.events) > seq_before, "恢复期间决策仍在推进（事件持续产生）"
    assert eng.stats["recovery_steps"] >= 1
