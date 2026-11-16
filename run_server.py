"""服务端入口 —— 局域网小主机（J4125 / RPi5）上运行。

用法：
  .venv/Scripts/python run_server.py                     # 使用 config.yaml 默认配置
  .venv/Scripts/python run_server.py --config my.yaml    # 指定配置
  .venv/Scripts/python run_server.py --sim               # 内置模拟器代理（快速本地联调）
  AUTOGAME_TOKEN=xxx .venv/Scripts/python run_server.py  # 指定 token
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import threading
import time
from pathlib import Path

import uvicorn

from autogame.config import load_config
from autogame.executor import build_executor
from autogame.server.app import create_app
from autogame.server.hub import Hub
from autogame.vision.pipeline import VisionPipeline

ROOT = Path(__file__).resolve().parent


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_dashboard() -> str:
    return (ROOT / "web" / "dashboard.html").read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="autogame 服务端")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--sim", action="store_true", help="内置模拟器代理，快速闭环联调")
    parser.add_argument("--nav", action="store_true", help="启用小地图导航（模拟器目标点走位）")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None, help="覆盖采集/决策帧率")
    args = parser.parse_args()

    _setup_logging()
    cfg = load_config(args.config)
    if args.host:
        cfg.server.host = args.host
    if args.port:
        cfg.server.port = args.port
    if args.fps:
        cfg.capture.fps = args.fps
    if args.nav:
        cfg.decision.nav_enabled = True

    executor = build_executor(cfg.executor)
    pipeline = VisionPipeline(cfg.vision)
    hub = Hub(cfg, executor, pipeline)

    app = create_app(cfg, hub, _load_dashboard())

    config = uvicorn.Config(
        app, host=cfg.server.host, port=cfg.server.port,
        log_level="warning", access_log=False, ws="websockets",
    )
    server = uvicorn.Server(config)

    url = f"http://{'127.0.0.1' if cfg.server.host == '0.0.0.0' else cfg.server.host}:{cfg.server.port}"
    print("=" * 62)
    print(f"  autogame 服务端  {url}")
    print(f"  仪表盘: {url}/  （Token 见 .token 文件或 AUTOGAME_TOKEN）")
    print(f"  代理连接: ws://<游戏PC> → {url}/ws/agent?token=<token>")
    print(f"  执行器: {cfg.executor.backend}  采集fps: {cfg.capture.fps}")
    print("=" * 62)

    async def _serve() -> None:
        # 决策循环由 FastAPI lifespan 启动（create_app 内）
        if args.sim:
            threading.Thread(target=_run_sim_agent, args=(cfg, executor), daemon=True).start()
        await server.serve()

    asyncio.run(_serve())


def _run_sim_agent(cfg, executor) -> None:  # noqa: ANN001
    """内置模拟器代理：SyntheticWorld → AgentClient + 执行器→世界桥接。"""
    import json

    import cv2
    import websockets

    from autogame.agent.client import AgentClient
    from autogame.sim.scene import SimCapture, SimWorld

    world = SimWorld(
        width=cfg.sim.width, height=cfg.sim.height,
        walk_speed_px_s=cfg.sim.walk_speed_px_s,
        pickup_radius_px=cfg.sim.pickup_radius_px,
        pillar_spawn_interval_s=cfg.sim.pillar_spawn_interval_s,
        picks_until_clear=cfg.sim.picks_until_clear,
    )
    capture = SimCapture(world)
    agent_cfg = cfg.model_copy(deep=True)
    client = AgentClient(agent_cfg, f"http://127.0.0.1:{cfg.server.port}",
                         agent_name="sim-agent", capture=capture)

    def bridge() -> None:
        """执行器干跑日志 → 虚拟世界输入（闭环）。"""
        idx = 0
        while True:
            journal = executor.journal
            while idx < len(journal):
                e = journal[idx]
                idx += 1
                world.on_command(e["op"], key=e.get("key"), button=e.get("button"),
                                 x=e.get("x"), y=e.get("y"), hold_ms=e.get("hold_ms") or 0)
            time.sleep(0.02)

    threading.Thread(target=bridge, daemon=True).start()
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
