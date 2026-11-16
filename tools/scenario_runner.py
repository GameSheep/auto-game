"""场景运行器 —— 真实游戏截图 → 自适应检测 → FSM 决策 → 文本输出命令。

用法：
  # 单图
  .venv/Scripts/python tools/scenario_runner.py --image assets/samples/loot_01.png
  # 批量
  .venv/Scripts/python tools/scenario_runner.py --dir assets/samples --ticks 2
  # JSON 输出（供测试断言）
  .venv/Scripts/python tools/scenario_runner.py --dir assets/samples --json

流程（每张图）：
  1. 自适应标定：auto_hsv_ranges 从图像推断高亮饱和色簇（不同游戏/滤镜通吃）
  2. 视觉检测：光柱 / A框 / 再次挑战按钮 / 小地图标记
  3. 决策：FSM 以 1s/tick 喂同一帧 ticks 次（模拟连续观察），干跑收集命令
  4. 输出：文本命令（KEY_TAP a hold=70ms / MOUSE_CLICK left (842,561)）
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autogame.config import AppConfig, DecisionConfig, ExecutorConfig, VisionConfig  # noqa: E402
from autogame.decision.fsm import DecisionEngine  # noqa: E402
from autogame.executor import DryRunExecutor  # noqa: E402
from autogame.protocol import Command, SceneState, format_command  # noqa: E402
from autogame.vision.auto_calib import auto_hsv_ranges  # noqa: E402
from autogame.vision.detector import Detector  # noqa: E402


def detect_scene(frame, cfg, auto=True):  # noqa: ANN001
    """自适应检测一帧（auto=用图像推断 HSV）。"""
    if auto:
        ranges = auto_hsv_ranges(frame)
        if ranges:
            cfg.hsv = {f"auto{i}": r for i, r in enumerate(ranges)}
    det = Detector(cfg)
    return det.process(frame, int(time.time() * 1000)), (cfg.hsv if auto else None)


def run_image(path: Path, ticks: int, auto: bool = True) -> dict:
    """跑单张真实截图，返回检测与文本命令报告。"""
    frame = cv2.imread(str(path))
    if frame is None:
        raise SystemExit(f"无法读取图片: {path}")
    h, w = frame.shape[:2]

    cfg = VisionConfig()
    hsv_used = None
    if auto:
        ranges = auto_hsv_ranges(frame)
        if ranges:
            cfg.hsv = {f"auto{i}": r for i, r in enumerate(ranges)}
            hsv_used = cfg.hsv

    scene, _ = detect_scene(frame, cfg, auto=False)  # hsv 已注入
    det = Detector(cfg)

    # FSM 干跑：同帧多 tick（模拟连续观察 1s 间隔）
    executor = DryRunExecutor(ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000))
    engine = DecisionEngine(DecisionConfig(tick_hz=1.0), executor)
    engine.start()
    base_ts = 1000
    tick_cmds: list[list[str]] = []
    for i in range(ticks):
        scene = det.process(frame, base_ts + i * 1000)
        engine.on_scene(scene)
        cmds = [format_command(Command(**{k: v for k, v in j.items()
                                           if k in Command.model_fields})) for j in executor.journal]
        tick_cmds.append(cmds)
        executor.journal.clear()

    report = {
        "image": str(path),
        "size": [w, h],
        "hsv_ranges": hsv_used,
        "pillars": [p.model_dump() for p in scene.pillars],
        "aframe": scene.aframe.model_dump() if scene.aframe else None,
        "rechallenge": scene.rechallenge.model_dump() if scene.rechallenge else None,
        "minimap_marker": list(scene.minimap.marker) if scene.minimap and scene.minimap.marker else None,
        "commands": tick_cmds,
    }
    return report


def render_report(report: dict) -> str:
    """报告 → 文本输出。"""
    lines = []
    lines.append(f"== {report['image']} ({report['size'][0]}x{report['size'][1]}) ==")
    hsv = report.get("hsv_ranges")
    if hsv:
        parts = [f"h={r[0][0]}-{r[1][0]} S{r[0][1]}-{r[1][1]} V{r[0][2]}-{r[1][2]}"
                 for r in hsv.values()]
        lines.append("[标定] " + " | ".join(parts))
    else:
        lines.append("[标定] 未发现高亮色簇")
    ps = report["pillars"]
    if ps:
        descs = [f"{p['color']}@({p['cx']},{p['cy']}) 距中心{round(p['distance'])}px" for p in ps]
        lines.append(f"[视觉] 光柱 {len(ps)}: " + "; ".join(descs))
    else:
        lines.append("[视觉] 光柱: 无")
    af = report["aframe"]
    if af:
        lines.append(f"[视觉] A框: ({af['x']},{af['y']}) score={af['score']}")
    else:
        lines.append("[视觉] A框: 无")
    rc = report["rechallenge"]
    if rc:
        lines.append(f"[视觉] 再次挑战按钮: ({rc['x']},{rc['y']})")
    else:
        lines.append("[视觉] 再次挑战按钮: 无")
    for i, cmds in enumerate(report["commands"], 1):
        if cmds:
            lines.append(f"[决策] tick{i}: " + " → ".join(cmds))
        else:
            lines.append(f"[决策] tick{i}: (无动作)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="真实截图场景运行器：图片 → 检测 → 决策 → 文本命令",
        epilog="快速用法: python tools/scenario_runner.py 截图.png [第二张.png ...]",
    )
    parser.add_argument("images", nargs="*", help="直接跟图片路径（可多张）")
    parser.add_argument("--image", default=None)
    parser.add_argument("--dir", default=None)
    parser.add_argument("--ticks", type=int, default=2)
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    parser.add_argument("--no-auto", action="store_true", help="禁用自适应标定（用 config.yaml 默认 HSV）")
    args = parser.parse_args()

    paths: list[Path] = [Path(p) for p in args.images]
    if args.image:
        paths.append(Path(args.image))
    if args.dir:
        exts = ("*.png", "*.jpg", "*.jpeg")
        paths += [Path(p) for ext in exts for p in glob.glob(str(Path(args.dir) / ext))]
        paths = [p for p in paths if "contact_sheet" not in p.name and "sheet" not in p.name]
    if not paths:
        raise SystemExit("用法: python tools/scenario_runner.py <图片路径> （或 --dir 目录）")

    reports = []
    for p in sorted(paths):
        try:
            rep = run_image(p, args.ticks, auto=not args.no_auto)
            reports.append(rep)
            print(render_report(rep) if not args.json else json.dumps(rep, ensure_ascii=False))
            print()
        except Exception as e:  # noqa: BLE001
            print(f"== {p} ==\n[错误] {e}\n")

    if args.json:
        pass
    else:
        picked = sum(1 for r in reports for cl in r["commands"] for c in cl if c.startswith("KEY_TAP"))
        clicked = sum(1 for r in reports for cl in r["commands"] for c in cl if c.startswith("MOUSE_CLICK"))
        print(f"—— 汇总: {len(reports)} 张图, 命令输出 {clicked} 次鼠标 + {picked} 次按键 ——")


if __name__ == "__main__":
    main()
