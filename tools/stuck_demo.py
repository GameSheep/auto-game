"""卡壳自愈演示 —— 用合成世界现场演示"卡住 → 自救 → 恢复拾取"。

为什么单独做一个演示脚本：
  自主恢复的价值只有在"真的卡住"时才看得到，而正常刷图演示里它永远不会触发。
  本脚本人为制造三种卡死（画面冻结 / 房间走空 / 弹窗挡路），把看门狗判定、
  恢复阶梯的每一级动作、以及恢复后重新拾取的过程全部打印出来。

用法：
  python tools/stuck_demo.py                 # 三种场景各演示一遍
  python tools/stuck_demo.py --scenario frozen --seconds 20
  python tools/stuck_demo.py --json          # 机器可读（CI/回归用）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autogame.config import DecisionConfig, ExecutorConfig, VisionConfig  # noqa: E402
from autogame.decision.fsm import DecisionEngine, State  # noqa: E402
from autogame.executor import DryRunExecutor  # noqa: E402
from autogame.sim.scene import SimPillar, SimWorld  # noqa: E402
from autogame.vision.detector import Detector  # noqa: E402


def _build(scenario: str):
    world = SimWorld(seed=3, pillar_spawn_interval_s=999,
                     walk_speed_px_s=0.0 if scenario != "gated" else 260.0)
    world.pillars.clear()

    dcfg = DecisionConfig(tick_hz=2.0, approach_mode="mouse_click")
    a = dcfg.autonomy
    a.frozen_window_s = 3.0
    a.idle_window_s = 4.0
    a.stall_window_s = 6.0
    a.gate_window_s = 3.0
    a.level_interval_s = 1.0
    a.step_interval_s = 0.05
    a.cycle_backoff_s = 0.0
    ex = DryRunExecutor(ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000))
    eng = DecisionEngine(dcfg, ex)
    eng.start()
    det = Detector(VisionConfig())
    return world, eng, ex, det


def run(scenario: str, seconds: int) -> dict:
    world, eng, ex, det = _build(scenario)
    ts = 1000
    keys: list[str] = []          # 全部按键（含合法拾取）
    keys_stuck: list[str] = []    # 卡壳期间按键（这里出现拾取键就是违规）
    events: list[dict] = []
    seen = 0
    phase = [1]

    def tick(force_in_game: bool | None = None) -> None:
        nonlocal ts, seen
        frame = world.render()
        det.process(frame, ts)
        scene = det.process(frame, ts + 100)
        if force_in_game is not None:
            scene.in_game = force_in_game
        ts += 1000
        eng.on_scene(scene)
        for j in list(ex.journal):
            if j["op"] == "key_tap":
                keys.append(j["key"])
                if phase[0] == 1:
                    keys_stuck.append(j["key"])
            world.on_command(j["op"], key=j["key"], button=j["button"],
                             x=j["x"], y=j["y"], hold_ms=j["hold_ms"])
        ex.journal.clear()
        world.tick()
        if len(eng.events) > seen:
            for e in eng.events[seen:]:
                if e["event"].startswith(("recovery", "gate_", "stuck", "clock", "explore")):
                    events.append(e)
            seen = len(eng.events)

    # 阶段1：制造卡壳
    for _ in range(seconds):
        tick(force_in_game=False if scenario == "gated" else None)

    stuck_state = eng.state.value
    stuck_cycle = eng.recovery.cycle

    # 阶段2：解除卡壳（能走 + 掉落出现在脚边）
    phase[0] = 2
    world.walk_speed = 260.0
    world.pillars.append(SimPillar(world.char_wx, world.char_wy + 60, "gold"))
    for _ in range(seconds):
        tick(force_in_game=True if scenario == "gated" else None)

    return {
        "scenario": scenario,
        "stuck_state": stuck_state,
        "recoveries": eng.stats["recoveries"],
        "recovery_cycles": eng.stats["recovery_cycles"],
        "recovery_actions": eng.stats["recovery_actions"],
        "gate_escapes": eng.stats["gate_escapes"],
        "picks": world.picked,
        "blind_a_press": world.blind_a_press,
        "pickup_key_in_moves": keys_stuck.count("a"),
        "legal_pickup_taps": keys.count("a") - keys_stuck.count("a"),
        "final_state": eng.state.value,
        "levels_seen": sorted({e.get("level") for e in events
                               if e["event"] == "recovery_step"}),
        "steps_seen": [e.get("step") for e in events if e["event"] == "recovery_step"],
        "events": [e["event"] for e in events],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="卡壳自愈演示（不发任何真实输入）")
    ap.add_argument("--scenario", choices=["frozen", "idle", "gated", "all"], default="all")
    ap.add_argument("--seconds", type=int, default=12, help="每个阶段时长（秒/帧）")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    scenarios = ["frozen", "idle", "gated"] if args.scenario == "all" else [args.scenario]
    out = [run(s, args.seconds) for s in scenarios]

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return

    for r in out:
        print("=" * 68)
        print(f" 场景: {r['scenario']}")
        print(f"  卡壳期间状态      : {r['stuck_state']}  "
              f"(进入恢复 {r['recoveries']} 次 / 循环 {r['recovery_cycles']} 轮)")
        print(f"  恢复动作数        : {r['recovery_actions']}  ESC 自救 {r['gate_escapes']} 次")
        print(f"  执行过的恢复等级  : {r['levels_seen']}")
        print(f"  执行过的恢复手段  : {' → '.join(r['steps_seen'][:10]) or '-'}")
        print(f"  解除后拾取        : {r['picks']} 个")
        print(f"  盲按拾取键        : {r['blind_a_press']} 次（必须为 0）")
        print(f"  卡壳期出现拾取键  : {r['pickup_key_in_moves']} 次（必须为 0）")
        print(f"  合法拾取按键      : {r['legal_pickup_taps']} 次（A框确认后）")
        print(f"  最终状态          : {r['final_state']}")
        ok = (r["picks"] >= 1 and r["blind_a_press"] == 0
              and r["pickup_key_in_moves"] == 0)
        print(f"  结论              : {'✅ 自愈成功' if ok else '❌ 未通过'}")
    print("=" * 68)


if __name__ == "__main__":
    main()
