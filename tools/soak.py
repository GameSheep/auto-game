"""过夜 soak 自检引擎 —— 整夜循环迭代验证。

每个周期依次执行（约 3 分钟）：
  1. 全量测试套件（subprocess pytest）
  2. 闭环压测：3 轮端到端闭环（拾取+刷图，捕捉 flaky）
  3. 100 张真实帧批量自检
  4. 用户刷图视频回放自检
  5. 进程内存/RSS 采样（泄漏观测）

全部结果追加到 assets/soak/soak_report.jsonl（逐周期）+ soak_summary.md（滚动摘要）。
任何异常不中断循环（过夜自检的核心：积累问题而非停止）。

用法：
  .venv/Scripts/python tools/soak.py --cycles 160          # 约跑 8 小时
  .venv/Scripts/python tools/soak.py --duration 8          # 按小时数
  .venv/Scripts/python tools/soak.py --cycles 2            # 冒烟
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import threading
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

SOAK_DIR = ROOT / "assets" / "soak"
PY = sys.executable


def run_pytest() -> dict:
    t0 = time.perf_counter()
    r = subprocess.run(
        [PY, "-m", "pytest", "tests/", "-q", "--tb=no"],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    dt = time.perf_counter() - t0
    passed = failed = 0
    import re
    m = re.search(r"(\d+) passed", r.stdout)
    if m:
        passed = int(m.group(1))
    m = re.search(r"(\d+) failed", r.stdout)
    if m:
        failed = int(m.group(1))
    ok = r.returncode == 0
    return {"step": "pytest", "ok": ok, "passed": passed, "failed": failed,
            "seconds": round(dt, 1),
            "tail": "\n".join(r.stdout.splitlines()[-3:]) if not ok else ""}


def run_closed_loop_stress(rounds: int = 3) -> dict:
    """端到端闭环压测：多轮捕捉 flaky（时序/竞态类问题过夜最容易现形）。"""
    results = []
    for i in range(rounds):
        t0 = time.perf_counter()
        try:
            import asyncio

            from test_e2e import ServerFixture, _run_bridge

            from autogame.agent.client import AgentClient
            from autogame.sim.scene import SimCapture, SimWorld

            srv = ServerFixture()
            srv.start()
            srv.cfg.capture.fps = 6.0
            srv.cfg.decision.tick_hz = 6.0
            world = SimWorld(seed=int(time.time()) % 100000,
                             picks_until_clear=2, walk_speed_px_s=420.0)
            agent_cfg = srv.cfg.model_copy(deep=True)
            client = AgentClient(agent_cfg, srv.url, agent_name=f"soak-{i}",
                                 capture=SimCapture(world))
            stop_event = __import__('threading').Event()
            bridge = threading.Thread(
                target=_run_bridge, args=(srv.executor, world, stop_event), daemon=True)
            bridge.start()

            async def main():
                import httpx

                async with httpx.AsyncClient(timeout=5) as hc:
                    await hc.post(f"{srv.url}/api/control/start",
                                  headers={"X-Token": "e2e-test-token"})
                    task = asyncio.create_task(asyncio.wait_for(client.run(), timeout=60))
                    deadline = time.monotonic() + 70
                    while time.monotonic() < deadline:
                        if world.picked >= 2 and world.map_clears >= 1:
                            break
                        await asyncio.sleep(0.5)
                    client.stop()
                    try:
                        await asyncio.wait_for(task, timeout=8)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass
                    await hc.post(f"{srv.url}/api/control/stop",
                                  headers={"X-Token": "e2e-test-token"})

            asyncio.run(main())
            stop_event.set()
            ok = world.picked >= 2 and world.blind_a_press == 0
            results.append({"round": i, "ok": ok, "picked": world.picked,
                            "blind": world.blind_a_press,
                            "clears": world.map_clears,
                            "seconds": round(time.perf_counter() - t0, 1)})
        except Exception as e:  # noqa: BLE001
            results.append({"round": i, "ok": False, "error": f"{type(e).__name__}: {e}",
                            "seconds": round(time.perf_counter() - t0, 1)})
    return {"step": "closed_loop_stress", "ok": all(r.get("ok") for r in results),
            "rounds": results}


def run_batch_validate() -> dict:
    t0 = time.perf_counter()
    r = subprocess.run(
        [PY, "tools/batch_validate.py"], cwd=ROOT,
        capture_output=True, text=True, timeout=300,
    )
    dt = time.perf_counter() - t0
    ok = "FAIL 0" in r.stdout
    return {"step": "batch_validate", "ok": ok, "seconds": round(dt, 1),
            "tail": r.stdout.strip().splitlines()[-6:] if r.stdout else ""}


def run_video_replay() -> dict:
    t0 = time.perf_counter()
    video = ROOT / "assets" / "videos" / "user_farm.mp4"
    if not video.exists():
        return {"step": "video_replay", "ok": True, "skipped": True}
    r = subprocess.run(
        [PY, "tools/video_replay.py", str(video), "--hz", "2",
         "--max-min", "2", "--csv", str(SOAK_DIR / "replay_latest.csv")],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    dt = time.perf_counter() - t0
    ok = "全部通过" in r.stdout
    return {"step": "video_replay", "ok": ok, "seconds": round(dt, 1),
            "tail": "\n".join(r.stdout.splitlines()[-8:])}


def rss_mb() -> float:
    try:
        import psutil  # 可选

        return round(psutil.Process().memory_info().rss / 1024 / 1024, 1)
    except Exception:  # noqa: BLE001
        try:
            import resource

            return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
        except Exception:  # noqa: BLE001
            return -1.0


def run_cycle(n: int) -> dict:
    cycle = {"cycle": n, "ts": datetime.now().isoformat(timespec="seconds"),
             "rss_mb": rss_mb(), "steps": []}
    steps = [
        ("pytest", run_pytest),
        ("stress", run_closed_loop_stress),
        ("batch", run_batch_validate),
        ("replay", run_video_replay),
    ]
    for name, fn in steps:
        try:
            cycle["steps"].append(fn())
        except subprocess.TimeoutExpired:
            cycle["steps"].append({"step": name, "ok": False, "error": "timeout"})
        except Exception:  # noqa: BLE001
            cycle["steps"].append({"step": name, "ok": False,
                                   "error": traceback.format_exc()[-500:]})
    cycle["ok"] = all(s.get("ok", False) for s in cycle["steps"])
    return cycle


def write_summary(cycles: list[dict]) -> None:
    total = len(cycles)
    ok_cycles = sum(1 for c in cycles if c["ok"])
    lines = [
        "# 过夜 soak 摘要",
        f"- 更新: {datetime.now().isoformat(timespec='seconds')}",
        f"- 周期: {total}（正常 {ok_cycles} / 异常 {total - ok_cycles}）",
        f"- 内存: 首 {cycles[0]['rss_mb'] if cycles else '-'}MB → "
        f"末 {cycles[-1]['rss_mb'] if cycles else '-'}MB",
        "",
        "| 周期 | 时间 | 结果 | pytest | 压测 | 批量 | 回放 | 内存MB |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in cycles:
        steps = {s["step"]: s for s in c["steps"]}
        row = [
            c["cycle"], c["ts"][11:19], "✓" if c["ok"] else "✗",
            f"{steps.get('pytest', {}).get('passed', '-')}/{steps.get('pytest', {}).get('failed', '-')}",
            "✓" if steps.get("closed_loop_stress", {}).get("ok") else "✗",
            "✓" if steps.get("batch_validate", {}).get("ok") else "✗",
            "✓" if steps.get("video_replay", {}).get("ok") else "✗",
            c["rss_mb"],
        ]
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    # 问题清单（所有非 ok 周期）
    problems = []
    for c in cycles:
        if c["ok"]:
            continue
        for s in c["steps"]:
            if not s.get("ok", False):
                problems.append(f"- 周期{c['cycle']} {s['step']}: {s.get('error', s.get('tail', ''))[:200]}")
    if problems:
        lines.append("\n## 问题积累（去重前 20 条）\n")
        seen = set()
        for p in problems:
            if p not in seen:
                seen.add(p)
                lines.append(p)
                if len(seen) >= 20:
                    break
    (SOAK_DIR / "soak_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="过夜 soak 自检引擎")
    parser.add_argument("--cycles", type=int, default=0)
    parser.add_argument("--duration", type=float, default=0, help="小时")
    args = parser.parse_args()
    if args.duration:
        deadline = time.time() + args.duration * 3600
    else:
        deadline = time.time() + args.cycles * 1e9  # cycles 模式无时间上限

    SOAK_DIR.mkdir(parents=True, exist_ok=True)
    report_path = SOAK_DIR / "soak_report.jsonl"
    cycles: list[dict] = []
    n = 0
    print(f"soak 启动: cycles={args.cycles or '∞'} duration={args.duration or '∞'}h")
    while True:
        n += 1
        t0 = time.perf_counter()
        cycle = run_cycle(n)
        cycle["wall_s"] = round(time.perf_counter() - t0, 1)
        cycles.append(cycle)
        with report_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(cycle, ensure_ascii=False) + "\n")
        write_summary(cycles)
        print(f"[周期 {n}] {'✓' if cycle['ok'] else '✗'} "
              f"wall={cycle['wall_s']}s rss={cycle['rss_mb']}MB "
              f"steps={[s['step'] + ('✓' if s.get('ok') else '✗') for s in cycle['steps']]}",
              flush=True)
        if args.cycles and n >= args.cycles:
            break
        if args.duration and time.time() >= deadline:
            break
        time.sleep(2)  # 周期间休眠，减轻磁盘压力


if __name__ == "__main__":
    main()
