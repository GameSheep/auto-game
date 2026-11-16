"""真实屏幕采集端到端验证 —— 真机（本机）链路，不使用任何模拟画面。

跑的是**生产代码路径**：
  MsseCapture 真实屏幕 → JPEG → WebSocket → 服务端 → Detector/VisionPipeline
  → FSM 决策 → DryRunExecutor（不产生真实输入）

为什么单独做这个工具：单元测试和 sim 闭环都在"合成画面"上跑，
真实屏幕引入的变量（多显示器、DPI 缩放、窗口失焦、真实 UI 密度）完全没覆盖。
本工具把这些变量放进链路，并产出可复查的证据文件。

用法：
  python tools/e2e_real_screen.py --seconds 12
  python tools/e2e_real_screen.py --seconds 20 --out assets/real_screen_e2e.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import uvicorn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autogame.agent.capture import build_capture  # noqa: E402
from autogame.agent.client import AgentClient  # noqa: E402
from autogame.config import load_config  # noqa: E402
from autogame.executor import build_executor  # noqa: E402
from autogame.server.app import create_app  # noqa: E402
from autogame.server.hub import Hub  # noqa: E402
from autogame.vision.pipeline import VisionPipeline  # noqa: E402


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main() -> None:
    ap = argparse.ArgumentParser(description="真实屏幕端到端验证")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--out", default=str(ROOT / "assets" / "real_screen_e2e.json"))
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.server.port = _free_port()
    cfg.server.host = "127.0.0.1"
    cfg.server.token = cfg.server.token or "real-screen-token"
    cfg.capture.fps = 2.0
    cfg.decision.tick_hz = 2.0
    # 真机验证用 dryrun：只验证"看到→判断→决策"链路，不产生任何键鼠输出
    cfg.executor.backend = "dryrun"

    executor = build_executor(cfg.executor)
    hub = Hub(cfg, executor, VisionPipeline(cfg.vision))
    app = create_app(cfg, hub, "<html>real-screen</html>")
    server = uvicorn.Server(uvicorn.Config(app, host=cfg.server.host, port=cfg.server.port,
                                          log_level="error", access_log=False,
                                          ws="websockets"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    if not server.started:
        raise SystemExit("服务端启动失败")
    url = f"http://{cfg.server.host}:{cfg.server.port}"
    print(f"服务端已启动 {url}（dryrun 执行器，零真实输入）")

    capture = build_capture(cfg.capture.backend, cfg.capture.region)
    print(f"采集后端: {type(capture).__name__}")

    # 单帧检测耗时（真实屏幕画面）
    t0 = time.perf_counter()
    frame = capture.grab()
    grab_ms = (time.perf_counter() - t0) * 1000
    if frame is None:
        raise SystemExit("真实屏幕抓帧失败（屏幕锁定？）")
    h, w = frame.shape[:2]
    print(f"真实帧: {w}x{h}  抓取 {grab_ms:.1f}ms")

    det_pipeline = VisionPipeline(cfg.vision)
    lat = []
    last_state = None
    for _ in range(10):
        t = time.perf_counter()
        last_state = det_pipeline.process(frame, int(time.time() * 1000))
        lat.append((time.perf_counter() - t) * 1000)
    lat.sort()
    print(f"检测耗时: 中位 {lat[len(lat)//2]:.1f}ms  p95 {lat[int(len(lat)*0.95)]:.1f}ms")

    client = AgentClient(cfg, url, agent_name="real-screen-agent", capture=capture)

    async def run_agent() -> None:
        try:
            await asyncio.wait_for(client.run(), timeout=args.seconds + 8)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

    loop = asyncio.new_event_loop()
    th = threading.Thread(target=lambda: loop.run_until_complete(run_agent()), daemon=True)
    th.start()

    # 启动决策（FSM 进入 SCANNING；场景门会拦截非游戏画面 → 零命令）
    import httpx

    time.sleep(1.0)
    with httpx.Client(timeout=5) as hc:
        hc.post(f"{url}/api/control/start", headers={"X-Token": cfg.server.token})

    deadline = time.monotonic() + args.seconds
    seqs = []
    while time.monotonic() < deadline:
        snap = hub.snapshot()
        seqs.append(snap.get("frame_seq", 0))
        time.sleep(0.5)
    client.stop()
    time.sleep(0.6)
    snap = hub.snapshot()
    server.should_exit = True
    capture.close()

    frames = snap.get("frame_seq", 0)
    scene = snap.get("scene") or {}
    stats = snap.get("stats") or {}
    execu = snap.get("executor") or {}
    cmds = execu.get("sent", 0)

    evidence = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "host": f"{sys.platform}",
        "capture_backend": type(capture).__name__,
        "frame_size": [w, h],
        "grab_ms": round(grab_ms, 1),
        "detect_ms_median": round(lat[len(lat)//2], 1),
        "detect_ms_p95": round(lat[int(len(lat) * 0.95)], 1),
        "frames_received": frames,
        "fps_effective": round(frames / max(args.seconds, 1e-6), 2),
        "scene": {
            "in_game": scene.get("in_game"),
            "loading": scene.get("loading"),
            "pillars": len(scene.get("pillars") or []),
            "aframe": bool(scene.get("aframe")),
            "rechallenge": bool(scene.get("rechallenge")),
            "frame_delta": scene.get("frame_delta"),
        },
        "stats": stats,
        "executor": {"backend": execu.get("backend"), "sent": cmds,
                     "dropped": execu.get("dropped")},
        "verdict": {
            "frames_flowed": frames >= 5,
            "detection_under_100ms": lat[len(lat)//2] < 100,
            "no_blind_tap": stats.get("blind_tap_violations", 0) == 0,
            "non_game_screen_blocked": (not scene.get("in_game")) and cmds == 0,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(evidence, ensure_ascii=False, indent=1),
                              encoding="utf-8")

    print("\n" + "=" * 62)
    print(f" 真实屏幕链路: {frames} 帧 / {args.seconds:.0f}s  "
          f"(有效 {evidence['fps_effective']} fps)")
    print(f" 检测: 中位 {evidence['detect_ms_median']}ms  p95 {evidence['detect_ms_p95']}ms")
    print(f" 场景门: in_game={scene.get('in_game')} → 命令 {cmds} 条 "
          f"(非游戏画面应为 0)")
    print(f" 铁律: blind_tap_violations={stats.get('blind_tap_violations')}")
    print(f" 证据: {args.out}")
    print("=" * 62)
    ok = all(evidence["verdict"].values())
    print(" 结论: " + ("PASS（真实屏幕链路可用）" if ok else "CHECK（见 verdict 明细）"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
