"""独立模拟器代理 —— 在无游戏 PC 的机器上演示完整链路。

用法（先在另一终端启动 run_server.py）：
  .venv/Scripts/python run_sim_agent.py --server http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from autogame.config import load_config
from autogame.sim.scene import SimCapture, SimWorld

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="autogame 模拟器代理")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_config(args.config)
    if args.fps:
        cfg.capture.fps = args.fps

    world = SimWorld(
        width=cfg.sim.width, height=cfg.sim.height,
        walk_speed_px_s=cfg.sim.walk_speed_px_s,
        pickup_radius_px=cfg.sim.pickup_radius_px,
        pillar_spawn_interval_s=cfg.sim.pillar_spawn_interval_s,
        seed=args.seed,
    )
    capture = SimCapture(world)

    from autogame.agent.client import AgentClient

    client = AgentClient(cfg, args.server, agent_name="sim-agent", capture=capture)

    print(f"模拟器代理 → {args.server}  (fps={cfg.capture.fps})")
    print("注意：本进程不桥接执行器输出；闭环演示请用 run_server.py --sim")
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
