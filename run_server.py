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

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_dashboard() -> str:
    return (ROOT / "web" / "dashboard.html").read_text(encoding="utf-8")


def _lan_ips() -> list[str]:
    """本机对外的 IPv4 地址，用于打印「从别的电脑该用哪个地址」。

    部署在小主机（RK3566 等）上时，用户是在自己电脑上打开配置页的，
    提示 127.0.0.1 只会点错地方，所以启动时直接把局域网地址打出来。
    """
    import socket

    ips: list[str] = []

    # 1) UDP 连一下外网地址查出口网卡（不实际发包，不会产生流量）
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                ips.append(ip)
    except OSError:
        pass

    # 2) 补上其它网卡（多网口 / docker 网桥 / 无默认路由的情况）
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except OSError:
        pass

    return ips


def _ensure_port_free(host: str, port: int) -> None:
    """启动前探测端口；被占用时给出可执行的排查命令并退出。

    注意：只探测不绑定 —— 探测用 SO_REUSEADDR 关闭状态，绑定交给 uvicorn。
    """
    import socket

    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        if s.connect_ex((probe_host, port)) == 0:
            print("=" * 62, file=sys.stderr)
            print(f" 启动失败：端口 {port} 已被占用（{probe_host}）", file=sys.stderr)
            print(" 排查：", file=sys.stderr)
            if sys.platform == "win32":
                print(f'   netstat -ano | findstr :{port}      # 找 PID', file=sys.stderr)
                print('   taskkill /PID <PID> /F               # 结束旧进程', file=sys.stderr)
            else:
                print(f"   ss -lptn 'sport = :{port}'", file=sys.stderr)
                print("   systemctl stop autogame              # 若由 systemd 托管", file=sys.stderr)
            print(f" 或换端口：python run_server.py --port {port + 1}", file=sys.stderr)
            print("=" * 62, file=sys.stderr)
            raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="autogame 服务端")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--profile", default="",
                        help="游戏适配包 id（profiles/<id>/）：决定门禁模板、标答台账、"
                             "坐标基准。换游戏必须带这个，否则会用错 HUD 模板。")
    parser.add_argument("--sim", action="store_true", help="内置模拟器代理，快速闭环联调")
    parser.add_argument("--nav", action="store_true", help="启用小地图导航（模拟器目标点走位）")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None, help="覆盖采集/决策帧率")
    args = parser.parse_args()

    _setup_logging()
    prof = None
    if args.profile:
        from autogame.profile import load_profile
        prof = load_profile(args.profile)
        log.info("使用 profile: %s  基准 %dx%d  状态 %s  模板 %s",
                 prof.id, prof.reference_resolution[0], prof.reference_resolution[1],
                 prof.status, prof.templates_dir)
        if not prof.is_calibrated:
            log.warning("profile %s 状态为 %s：门禁可能拦下全部画面（fail_closed）。"
                        "补齐方式见 profiles/README.md", prof.id, prof.status)
    cfg = load_config(args.config, profile=prof)
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

    app = create_app(cfg, hub, _load_dashboard(), config_path=str(args.config))

    # 端口占用检查：先自己探测再启动，否则 uvicorn 在后台报错、前台仍打印
    # "服务端已启动"，用户会以为启动成功（真实踩过的坑：旧进程占着 8765）。
    _ensure_port_free(cfg.server.host, cfg.server.port)

    config = uvicorn.Config(
        app, host=cfg.server.host, port=cfg.server.port,
        log_level="warning", access_log=False, ws="websockets",
    )
    server = uvicorn.Server(config)

    port = cfg.server.port
    print("=" * 62)
    if cfg.server.host in ("0.0.0.0", "::", ""):
        ips = _lan_ips()
        local = f"http://127.0.0.1:{port}"
        print(f"  autogame 服务端  监听 {cfg.server.host}:{port}（局域网可访问）")
        print(f"  配置页: {local}/settings")
        if ips:
            print(f"          http://{ips[0]}:{port}/settings   ← 从别的电脑/手机用这个")
            for ip in ips[1:3]:
                print(f"          http://{ip}:{port}/settings")
        else:
            print(f"          http://<本机局域网IP>:{port}/settings   ← 从别的电脑/手机用这个")
        print(f"  仪表盘: {local}/")
        print(f"  代理连接: ws://<游戏PC> → {f'http://{ips[0]}' if ips else local}/ws/agent?token=<token>")
    else:
        url = f"http://{cfg.server.host}:{port}"
        print(f"  autogame 服务端  {url}")
        print(f"  配置页: {url}/settings")
        print(f"  仪表盘: {url}/")
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
