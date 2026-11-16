"""采集代理入口 —— 游戏 PC（Windows）上运行。只采集画面，不做任何输入。

用法：
  .venv/Scripts/python run_agent.py --server http://192.168.1.20:8765
  .venv/Scripts/python run_agent.py --server http://192.168.1.20:8765 --fps 2 --region 0,0,1920,1080
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from autogame.config import load_config
from autogame.agent.client import AgentClient

ROOT = Path(__file__).resolve().parent


def _fix_windows_console_encoding() -> None:
    """Windows 控制台中文乱码修复（PyInstaller 打包后尤其明显）。

    PyInstaller 默认以 UTF-8 写 stdout，而 cmd/PowerShell 控制台按活动代码页
    （简体中文为 cp936）解码 → 帮助信息与日志全是"锟斤拷"。
    这里按**控制台实际代码页**重设输出编码；代码页 65001 时用 utf-8。
    """
    if sys.platform != "win32":
        return
    code_page = "936"
    try:
        import ctypes

        code_page = str(ctypes.windll.kernel32.GetConsoleOutputCP() or 936)
    except Exception:  # noqa: BLE001
        pass
    enc = "utf-8" if code_page == "65001" else f"cp{code_page}"
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding=enc, errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass


def main() -> None:
    _fix_windows_console_encoding()
    parser = argparse.ArgumentParser(description="autogame 采集代理")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--server", required=True, help="服务端 URL，如 http://192.168.1.20:8765")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--quality", type=int, default=None)
    parser.add_argument("--region", default=None, help="采集区域 left,top,width,height")
    parser.add_argument("--name", default="game-pc")
    parser.add_argument("--backend", default=None, choices=["auto", "dxcam", "mss"])
    parser.add_argument("--token", default=None,
                        help="服务端 token（也可用 AUTOGAME_TOKEN 环境变量或 .token 文件）")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_config(args.config)
    if args.token:
        cfg.server.token = args.token
    if args.fps:
        cfg.capture.fps = args.fps
    if args.quality:
        cfg.capture.jpeg_quality = args.quality
    if args.backend:
        cfg.capture.backend = args.backend
    if args.region:
        cfg.capture.region = [int(v) for v in args.region.split(",")]

    print(f"采集代理: {args.name} → {args.server}")
    print(f"  fps={cfg.capture.fps} 质量={cfg.capture.jpeg_quality} "
          f"区域={cfg.capture.region or '全屏'} 后端={cfg.capture.backend}")
    print("  安全说明：本程序只读取屏幕画面，不模拟任何键鼠输入。")

    client = AgentClient(cfg, args.server, agent_name=args.name)
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("代理已停止")


if __name__ == "__main__":
    main()
