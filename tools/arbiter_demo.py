"""三级裁决干跑演示 —— 不装任何模型也能看清这条链路怎么走。

三个场景对应三种真实处境：

  A 认得        画面里有光柱、有拾取框 → 规则直接处理，**一次模型都不调**
  B 拿不准      laya 从「系统已支持的行为清单」里选一个 → 立刻能执行
  C 清单里没有  laya 选 others → 大模型规划 → 执行并验证 → 写进经验库
                → 同样局面再来时走 L0 直出，不再问模型

跑法：
    py -3.12 tools/arbiter_demo.py
    py -3.12 tools/arbiter_demo.py --scenario c
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autogame.config import ArbiterConfig, ArbiterModelConfig  # noqa: E402
from autogame.decision.actions import ActionRegistry  # noqa: E402
from autogame.decision.arbiter import Arbiter  # noqa: E402
from autogame.decision.knowledge import KnowledgeBase, fingerprint  # noqa: E402


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def probe(**kw) -> dict:
    p = {"in_game": True, "gate_reason": "skillbar", "pillars": 0, "monsters": 0,
         "aframe": False, "picks": 0, "dist": None, "w": 960, "h": 432,
         "nav_enabled": False, "rechallenge_visible": False, "intent": "",
         "miss_streak": 0}
    p.update(kw)
    return p


def build(tmp: Path, replies: dict[str, str], clock: Clock) -> tuple[Arbiter, KnowledgeBase]:
    cfg = ArbiterConfig(
        mode="enforce", cooldown_s=1.0, idle_trigger_s=2.0, unmatched_trigger_s=5.0,
        verify_horizon_s=1.0, keepalive_after_s=2.0, keepalive_interval_s=0.5,
        laya=ArbiterModelConfig(model="laya-3b"), planner=ArbiterModelConfig(model="planner-big"),
    )
    kb = KnowledgeBase(tmp / "knowledge")
    seen: list[str] = []

    def caller(model_cfg, system, user):
        seen.append(model_cfg.model)
        return replies.get(model_cfg.model, json.dumps(
            {"action": "others", "confidence": 0.1, "reason": "演示：没有预设回复"}))

    arb = Arbiter(cfg, ActionRegistry(), kb, clock=clock, root=tmp, caller=caller)
    arb.demo_calls = seen  # type: ignore[attr-defined]
    return arb, kb


def settle(arb: Arbiter) -> None:
    end = time.time() + 3
    while time.time() < end:
        if not arb._busy:
            return
        time.sleep(0.02)


def step(arb: Arbiter, clock: Clock, p: dict, idle: float = 5.0) -> None:
    """把"发起请求 → 取回结果 → 执行 → 对账"走一遍。"""
    arb.tick(p, clock(), idle_s=idle)
    settle(arb)
    call = arb.tick(p, clock(), idle_s=idle)
    if call:
        print(f"   → 执行 {call.action_id}（{call.stage}）{call.reason}")
        arb.execute_hook(call, p, clock())
    else:
        print("   → 这一帧没有动作")


def scene_a() -> None:
    print("\n[场景 A] 画面认得：有光柱、脚下有拾取框")
    with tempfile.TemporaryDirectory() as td:
        clock = Clock()
        arb, kb = build(Path(td), {}, clock)
        p = probe(pillars=2, monsters=1, aframe=True)
        for _ in range(3):
            call = arb.tick(p, clock(), idle_s=0.0)
            print(f"   tick → {call.action_id if call else 'None（规则自己会处理）'}")
            clock.advance(1.0)
        print(f"   模型调用次数：{len(arb.demo_calls)}  ← 认得就不问")
        print(f"   经验库：{kb.snapshot()['total']} 条")


def scene_b() -> None:
    print("\n[场景 B] 拿不准：画面没光柱也没怪，规则不知道干什么")
    with tempfile.TemporaryDirectory() as td:
        clock = Clock()
        # 假 laya：从清单里挑一个
        replies = {"laya-3b": json.dumps({
            "action": "explore", "params": {}, "confidence": 0.85,
            "reason": "视野内没目标，走动找下一批"})}
        arb, kb = build(Path(td), replies, clock)
        p = probe(pillars=0, monsters=0)
        step(arb, clock, p)
        print(f"   模型调用：{arb.demo_calls}  ← 只问了 laya，没惊动大模型")
        print(f"   经验：{kb.snapshot()['candidate']} 条待验证")


def scene_c() -> None:
    print("\n[场景 C] 清单里都没有：laya 选 others → 大模型规划 → 沉淀经验")
    with tempfile.TemporaryDirectory() as td:
        clock = Clock()
        replies = {
            "laya-3b": json.dumps({
                "action": "others", "confidence": 0.2,
                "reason": "被三只怪围住又没有拾取目标，清单里没有『先突围再回来捡』"}),
            "planner-big": json.dumps({
                "plan": [{"action": "attack_nearest", "params": {}},
                         {"action": "explore", "params": {}}],
                "reason": "先打掉最近的怪脱离包围，再换位置避免被围",
                "confidence": 0.8}),
        }
        arb, kb = build(Path(td), replies, clock)
        p = probe(pillars=1, monsters=3, aframe=False)
        fp = fingerprint(p)
        print(f"   局面指纹：{fp}")

        print("   —— 第一次遇到 ——")
        arb.tick(p, clock(), idle_s=5.0)
        settle(arb)
        call = arb.tick(p, clock(), idle_s=5.0)
        assert call is not None
        print(f"   → 执行 {call.action_id}（{call.stage}）")
        arb.execute_hook(call, p, clock())
        # 对账：怪从 3 只变 1 只
        clock.advance(2.0)
        arb.tick(probe(pillars=1, monsters=1), clock(), idle_s=1.0)
        snap = kb.snapshot()
        print(f"   模型调用：{arb.demo_calls}")
        print(f"   经验：{snap['total']} 条（候选 {snap['candidate']}，已验证 {snap['confirmed']}）")

        # 再验证一次 → confirmed
        for _ in range(1):
            clock.advance(2.0)
            arb.tick(p, clock(), idle_s=5.0)
            settle(arb)
            c2 = arb.tick(p, clock(), idle_s=5.0)
            if c2:
                arb.execute_hook(c2, p, clock())
                clock.advance(2.0)
                arb.tick(probe(pillars=1, monsters=1), clock(), idle_s=1.0)
        snap = kb.snapshot()
        print(f"   再验证一次后：候选 {snap['candidate']}，已验证 {snap['confirmed']}")

        print("   —— 同样局面再来 ——")
        before = len(arb.demo_calls)
        clock.advance(5.0)
        call = arb.tick(p, clock(), idle_s=5.0)
        print(f"   → {call.action_id}（{call.stage}）" if call else "   → 无动作")
        print(f"   这次模型调用：{len(arb.demo_calls) - before} 次  ← 经验直出，不再问")

        print("\n   经验库落盘内容：")
        for name in ("game.md", "experiences.jsonl"):
            p2 = Path(td) / "knowledge" / name
            if p2.exists():
                lines = p2.read_text(encoding="utf-8").strip().splitlines()
                print(f"     {name}: {len(lines)} 行")
                for line in lines[-1:]:
                    try:
                        d = json.loads(line)
                        print(f"       {d['status']}  {d['situation']} → {d['action']['id']}"
                              f"   hit {d['hits']}/miss {d['misses']}")
                    except json.JSONDecodeError:
                        print(f"       {line[:60]}")


def main() -> int:
    ap = argparse.ArgumentParser(description="三级裁决干跑演示")
    ap.add_argument("--scenario", choices=["a", "b", "c", "all"], default="all")
    args = ap.parse_args()
    print("=" * 68)
    print("三级裁决：认得自己做 → 拿不准问 laya → 都不行才问大模型")
    print("=" * 68)
    if args.scenario in ("a", "all"):
        scene_a()
    if args.scenario in ("b", "all"):
        scene_b()
    if args.scenario in ("c", "all"):
        scene_c()
    print("\n" + "-" * 68)
    print("真实运行时同样的事会写进 logs/arbiter.jsonl（含 shadow 模式下"
          "『本来想执行什么』）。")
    print("要先看影子日志再切 enforce：这个层会真的发命令。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
