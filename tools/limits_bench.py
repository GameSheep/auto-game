"""三档放行度的实测对比 —— 同一场景、同一时长，只有限制不同。

为什么需要这个工具：
  页面上的三档如果只是"配置里换几个数字"，那它和手改配置没有区别，用户没有理由信它。
  这里用模拟器把三档各跑一遍**真实链路**（渲染 → 检测 → FSM → 执行器 → 世界），
  同一颗随机种子、同样长的墙钟时间，只输出一件事：拾到了多少个。

用法：
  py -3.12 tools/limits_bench.py                  # 三档各 30 秒
  py -3.12 tools/limits_bench.py --seconds 60 --seed 7
  py -3.12 tools/limits_bench.py --tiers strict,unleashed

结论怎么读：
  picks/min 是产出。taps 是按键次数，taps/pick 越接近 1 说明没有在空按。
  sent 是真正发出去的动作数 —— 它和 picks 的比值就是"动作配额利用率"。
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autogame import limits as lm  # noqa: E402
from autogame.config import AppConfig, load_config  # noqa: E402
from autogame.decision.fsm import DecisionEngine  # noqa: E402
from autogame.executor import DryRunExecutor  # noqa: E402
from autogame.sim.scene import SimWorld  # noqa: E402
from autogame.vision.detector import Detector  # noqa: E402


def _cfg_for(tier: str, base: AppConfig) -> AppConfig:
    """在基准配置上叠该档 patch，并关掉"会让结果依赖机器负载"的东西。"""
    cfg = AppConfig.model_validate({**base.model_dump(), **lm.tier_patch(tier)})
    # 模型层在离线环境里不可用：留着它们只会让三档的差别被"连接失败"掩盖。
    cfg.decision.arbiter.mode = "off"
    cfg.decision.supervisor.mode = "off"
    cfg.decision.llm.enabled = False
    cfg.decision.vlm.enabled = False
    cfg.executor.backend = "dryrun"
    # 场景门必须关：模拟器画面里没有真实游戏的 HUD，19 张锚点模板一张都匹配不上，
    # fail_closed=true 会把每一帧都判成"不在游戏里"→ 整个闭环空转，三档都是 0 拾取。
    # 这不是"放宽限制换成绩"：三档同样关掉，对比较结果没有偏向。
    cfg.vision.scene_gate.enabled = False
    return cfg


def _bridge(executor: DryRunExecutor, world: SimWorld, stop: threading.Event) -> None:
    """把执行器 journal 里的命令回注到虚拟世界（闭环，和真机同一条路径）。"""
    idx = 0
    while not stop.is_set():
        journal = executor.journal
        while idx < len(journal):
            e = journal[idx]
            idx += 1
            if e.get("suppressed"):
                continue
            world.on_command(e["op"], key=e.get("key"), button=e.get("button"),
                             x=e.get("x"), y=e.get("y"), hold_ms=e.get("hold_ms") or 0)
        stop.wait(0.01)


def run_tier(tier: str, seconds: float, seed: int, base: AppConfig) -> dict:
    cfg = _cfg_for(tier, base)
    world = SimWorld(seed=seed, width=cfg.sim.width, height=cfg.sim.height,
                     walk_speed_px_s=cfg.sim.walk_speed_px_s,
                     pickup_radius_px=cfg.sim.pickup_radius_px,
                     pillar_spawn_interval_s=cfg.sim.pillar_spawn_interval_s,
                     picks_until_clear=cfg.sim.picks_until_clear)
    executor = DryRunExecutor(cfg.executor)
    engine = DecisionEngine(cfg.decision, executor)
    detector = Detector(cfg.vision)

    stop = threading.Event()
    th = threading.Thread(target=_bridge, args=(executor, world, stop), daemon=True)
    th.start()

    engine.start()
    period = 1.0 / max(0.5, float(cfg.decision.tick_hz))
    t0 = time.monotonic()
    next_at = t0
    frames = 0
    try:
        while time.monotonic() - t0 < seconds:
            world.tick()
            scene = detector.process(world.render())
            engine.on_scene(scene)
            frames += 1
            next_at += period
            delay = next_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_at = time.monotonic()   # 落后了就别追赶（追赶会打出连发）
    finally:
        engine.stop()
        stop.set()
        th.join(timeout=2.0)

    cost = time.monotonic() - t0
    st = engine.stats
    picks = int(world.picked)
    return {
        "tier": tier,
        "title": lm.TIER_TITLES[tier]["title"],
        "seconds": round(cost, 1),
        "frames": frames,
        "maps": int(getattr(world, "map_clears", 0)),
        "picks": picks,
        "per_min": round(picks / cost * 60, 1) if cost else 0.0,
        "taps": int(st.get("a_taps", 0)),
        "sent": int(executor.sent),
        "suppressed": int(getattr(executor, "suppressed", 0)),
        "dropped": int(getattr(executor, "dropped", 0)),
        "blind": int(st.get("blind_tap_violations", 0)),
        "breaks": int(st.get("human_breaks", 0)),
        "ignored": int(st.get("targets_ignored", 0)),
        "abandons": int(st.get("abandons", 0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--tiers", default=",".join(lm.TIERS))
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()

    base = load_config(args.config)
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    rows = []
    for t in tiers:
        if t not in lm.TIERS:
            print(f"跳过未知档位 {t}")
            continue
        print(f"跑 {t} …", flush=True)
        rows.append(run_tier(t, args.seconds, args.seed, base))
        r = rows[-1]
        print(f"  {r['picks']} 个 / {r['seconds']}s  →  {r['per_min']}/min"
              f"  动作 {r['sent']}  空按 {r['taps'] - r['picks']}", flush=True)

    if not rows:
        return 1

    print()
    head = f"{'档位':<14}{'拾取':>6}{'/分钟':>8}{'动作':>7}{'空按':>6}{'丢/拦':>8}{'盲按':>6}{'休息':>6}{'懒得捡':>7}"
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['title']:<14}{r['picks']:>6}{r['per_min']:>8.1f}{r['sent']:>7}"
              f"{r['taps'] - r['picks']:>6}{str(r['dropped']) + '/' + str(r['suppressed']):>8}"
              f"{r['blind']:>6}{r['breaks']:>6}{r['ignored']:>7}")

    base_row = rows[0]
    for r in rows[1:]:
        if base_row["per_min"]:
            ratio = r["per_min"] / base_row["per_min"]
            print(f"\n{r['title']} 相对 {base_row['title']}：产出 ×{ratio:.2f}，"
                  f"动作 ×{r['sent'] / max(1, base_row['sent']):.2f}")
    if any(r["blind"] for r in rows):
        print("\n⚠ 出现盲按违规 —— 那是 bug，不是放行度问题（任何档位都必须为 0）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
