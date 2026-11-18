"""监督层干跑演示 —— 不需要 Ollama，用假模型输出走完整链路。

为什么要有这个脚本：
  1. 装模型之前先看清这套机制到底怎么动手（不然它只是一段没人验证过的代码）；
  2. 改完代码快速回归"闸门还在不在"——比跑真实录像快得多。

它演示两件事（都对应真实踩过的坑）：
  A. 假走位：命令一直在发、画面没变、距离没缩短 → 对账连续落空 → 暂停/降速
  B. 突发：未识别画面持续 50s（可能是没见过的弹窗）→ 接管按 ESC

用法：
  py -3.12 tools/supervisor_demo.py              # shadow：看着像会拦，其实一条不拦
  py -3.12 tools/supervisor_demo.py --enforce    # enforce：真的拦，对比命令数
  py -3.12 tools/supervisor_demo.py --scenario a|b|both
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autogame.config import SupervisorConfig                    # noqa: E402
from autogame.decision.supervisor import Supervisor             # noqa: E402
from autogame.protocol import Command, SceneState               # noqa: E402


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


def _scene(in_game: bool = True, gate_reason: str = "skillbar") -> SceneState:
    return SceneState(ts_ms=0, w=960, h=432, character=(480, 216),
                      in_game=in_game, gate_reason=gate_reason)


def _sup(mode: str, reply: dict) -> tuple[Supervisor, FakeClock]:
    clock = FakeClock()
    cfg = SupervisorConfig(mode=mode, check_interval_s=1.0,
                           min_interval_s=1.0, max_calls_per_min=10)
    sup = Supervisor(cfg, clock=clock, root=ROOT)
    # 假模型：直接返回预设结论，不走网络
    sup._post = lambda ctx: json.dumps(reply, ensure_ascii=False)  # type: ignore[method-assign]
    return sup, clock


def scenario_a(mode: str) -> None:
    """假走位：点了地面、命令发出去了、距离一动不动。"""
    print("\n=== A. 假走位（命令在发，距离不变）===")
    reply = {
        "verdict": "hold",
        "confidence": 0.85,
        "diagnosis": "走位连续落空：命令在发但距离没有缩短，疑似寻路失败或点到了墙",
        "evidence": "miss_streak 3, walk 240→240",
    }
    sup, clock = _sup(mode, reply)
    sent = blocked = 0
    hit_tick = None

    for tick in range(24):
        scene = _scene()
        intent = "向右下移动接近 auto2 掉落光柱（距离 240px）"
        sup.note_frame(scene, intent, clock(), {"dist": 240.0, "picks": 0,
                                                "aframe": False, "in_game": True})
        # 每 3 帧发起一次走位，登记"距离应缩短"
        if tick % 3 == 0:
            sup.predict("walk", "走位后距离应缩短（当前 240px）",
                        base=240.0, now=clock(), horizon_s=3.0)
        for _ in range(2):
            if sup.gate(Command(op="mouse_click", x=520, y=330), clock()):
                sent += 1
                sup.note_command(clock())
            else:
                blocked += 1
        if sup._ledger and hit_tick is None and sup.stats["calls"]:
            hit_tick = tick
            d = sup._decision
            print(f"  t={tick:02d}s  模型裁决 → {d.verdict.value} "
                  f"(conf {d.confidence}) {d.diagnosis}")
            print(f"         证据: {d.evidence}")
        clock.advance(1)

    led = [(r.kind, r.result, r.detail) for r in list(sup._ledger)[-4:]]
    time.sleep(0.3)  # 等后台线程把裁决落盘（真实运行不需要，这里演示要看文件）
    print(f"  对账（最近 4 条）: {led}")
    print(f"  健康: miss_streak={sup.last_health.get('miss_streak')} "
          f"cmds={sup.last_health.get('cmds')}")
    print(f"  命令: 发出 {sent} 条, 被拦 {blocked} 条"
          f"（shadow 模式下 '想拦' {sup.stats['shadow_would_block']} 次）")
    if mode == "shadow":
        print("  → shadow：看得出来它想拦，但一条都没真拦（这正是影子期的意义）")
    else:
        print("  → enforce：真的拦下来了。判错的代价最多是慢几秒，不是做错事")


def scenario_b(mode: str) -> None:
    """突发：未识别画面持续 50s，可能是从没见过的弹窗。"""
    print("\n=== B. 突发（未识别画面持续 50s）===")
    reply = {
        "verdict": "takeover",
        "confidence": 0.9,
        "diagnosis": "连续 45 秒不在已标定的战斗界面，规则层仍在等，建议按一次 ESC",
        # 注意 evidence 里的数字必须能在输入里找到（"45" 命中 health.unmatched_for）
        "evidence": "unmatched_for 45, in_game false",
        "action": {"op": "esc", "reason": "未知界面挡住，先尝试关闭"},
    }
    sup, clock = _sup(mode, reply)
    got = None
    for tick in range(60):
        sup.note_frame(_scene(False, "unmatched"), "未识别的画面：不决策、不按键",
                       clock(), {"picks": 0, "aframe": False, "in_game": False})
        d = sup._decision
        if d.verdict.value == "takeover" and got is None and sup.stats["calls"]:
            got = True
            print(f"  t={tick:02d}s  模型裁决 → {d.verdict.value} (conf {d.confidence})")
            print(f"         证据: {d.evidence}")
            act = sup.take_takeover(clock())
            print(f"         接管动作: {act if act else '（shadow：结论成立但不执行）'}")
        clock.advance(1)

    print(f"  健康: unmatched_for={sup.last_health.get('unmatched_for')}s")
    time.sleep(0.3)
    if got:
        print("  → 接管只允许 esc/wait/reset_stage。它不能指定去哪、捡哪个")
    else:
        print("  → 没触发接管（检查触发阈值或 mode=off）")
    if mode == "shadow":
        print("  → shadow：take_takeover 恒为 None，不会真的按 ESC")


def main() -> int:
    ap = argparse.ArgumentParser(description="监督层干跑演示（不需要模型）")
    ap.add_argument("--enforce", action="store_true", help="用 enforce 模式（真的拦）")
    ap.add_argument("--scenario", choices=["a", "b", "both"], default="both")
    args = ap.parse_args()
    mode = "enforce" if args.enforce else "shadow"

    print(f"模式: {mode}   （监督层默认 shadow：照常判断、照常记账，但不执行）")
    if args.scenario in ("a", "both"):
        scenario_a(mode)
    if args.scenario in ("b", "both"):
        scenario_b(mode)
    print("\n说明：上面的『模型裁决』是脚本里写死的假回复，用来演示链路。")
    print("     真实运行时的每条裁决都会落盘到 logs/supervisor.jsonl，供人工复盘。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
