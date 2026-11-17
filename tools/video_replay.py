"""视频全程回放 —— 把视频当实时画面流，逐帧检测+即时决策，输出操作时间线。

用法：
  # 基础：时间线 + 自检
  .venv/Scripts/python tools/video_replay.py 游戏录像.mp4

  # 生成"判断标注版视频"（每帧叠加检测结果与命令）
  .venv/Scripts/python tools/video_replay.py 游戏录像.mp4 --annotate out_judged.mp4

  # 决策频率（默认 2Hz，与生产 capture.fps=2 一致；最高 6Hz）
  .venv/Scripts/python tools/video_replay.py 游戏录像.mp4 --hz 2

流程：
  1. 首个战斗帧自适应标定 HSV（auto_hsv_ranges）
  2. 以 --hz 频率逐帧喂 检测器+FSM（保持时域跟踪/帧差/FSM 状态 —— 真实在线语义）
  3. 输出带时间戳的操作时间线；不变量实时自检（零盲按/门禁一致/坐标合法…）
  4. --annotate 时输出叠加判定的视频，可直接观看每一步判断
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autogame.config import load_config  # noqa: E402
from autogame.decision.fsm import DecisionEngine  # noqa: E402
from autogame.executor import DryRunExecutor  # noqa: E402
from autogame.executor.base import ExecutorConfig  # noqa: E402
from autogame.protocol import Command, format_command  # noqa: E402
from autogame.vision.auto_calib import auto_hsv_ranges  # noqa: E402
from autogame.vision.detector import Detector, ROOT_TEMPLATES  # noqa: E402

# 视频回放参考基准（模板/排除区归一化依据）
REF_W = 1280


def journal_to_cmd(j: dict) -> Command:
    return Command(**{k: v for k, v in j.items() if k in Command.model_fields})


def ts_str(ms: float) -> str:
    s = ms / 1000.0
    return f"{int(s // 60):02d}:{s % 60:05.2f}"


def draw_judgment(frame: np.ndarray, lr: dict, ts: str) -> np.ndarray:
    """在帧上叠加判定结果（光柱/列表/A框/状态/命令流）。"""
    out = frame.copy()
    for (cx, cy, _c) in lr.get("pillars", []):
        cv2.rectangle(out, (cx - 18, cy - 75), (cx + 18, cy + 15), (0, 215, 255), 2)
    for e in lr.get("item_list", []):
        cv2.rectangle(out, (e["x"] - e["w"] // 2, e["y"] - e["h"] // 2),
                      (e["x"] + e["w"] // 2, e["y"] + e["h"] // 2), (255, 200, 0), 2)
    if lr.get("aframe"):
        ax, ay = lr["aframe"]
        cv2.rectangle(out, (ax - 22, ay - 22), (ax + 22, ay + 22), (0, 255, 0), 2)
    if lr.get("loading"):
        cv2.putText(out, "LOADING...", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 2)
    st = "IN-GAME" if lr.get("in_game") else "GATED(UI)"
    col = (0, 255, 255) if lr.get("in_game") else (160, 160, 160)
    cv2.putText(out, f"[{ts}] {st} pillars={len(lr.get('pillars', []))}",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    for ci, c in enumerate(lr.get("commands", [])[-3:]):
        cv2.putText(out, f">> {c}", (10, 52 + ci * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="视频全程回放 —— 即时操作判断")
    parser.add_argument("video", help="视频路径")
    parser.add_argument("--hz", type=float, default=2.0, help="决策频率（默认2Hz）")
    parser.add_argument("--annotate", default=None, help="输出标注版视频路径")
    parser.add_argument("--json", default=None, help="时间线 JSON 输出路径")
    parser.add_argument("--csv", default=None, help="时间线 CSV 输出路径（复盘用）")
    parser.add_argument("--frames-dir", default=None,
                        help="导出每个决策 tick 的标注帧 + manifest.json（供审查网页浏览）")
    parser.add_argument("--aframe-template", default=None,
                        help="A框/拾取提示模板（真实游戏提示样式，配合光柱判定按A）")
    parser.add_argument("--max-min", type=float, default=10.0, help="最多回放分钟数")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"),
                        help="部署配置（默认用线上那套，避免验证的不是生产参数）")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step_frames = max(1, int(round(fps / args.hz)))
    print(f"视频: {args.video}  {fps:.0f}fps  {total}帧  决策{args.hz}Hz(每{step_frames}帧)")

    # 首帧自适应标定
    ok, first = cap.read()
    if not ok:
        raise SystemExit("视频无帧")
    ranges = auto_hsv_ranges(first)
    app_cfg = load_config(args.config)
    cfg = app_cfg.vision
    if ranges:
        cfg.hsv = {f"auto{i}": r for i, r in enumerate(ranges)}
    print(f"[标定] {len(ranges)} 个色簇: "
          + " | ".join(f"h={r[0][0]}-{r[1][0]}" for r in ranges))

    if args.aframe_template:
        cfg.aframe.template = args.aframe_template
        cfg.aframe.template_threshold = 0.8
    det = Detector(cfg, ROOT_TEMPLATES if ROOT_TEMPLATES.exists() else None)
    ex = DryRunExecutor(ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000))
    eng = DecisionEngine(app_cfg.decision.model_copy(update={"tick_hz": args.hz}), ex)
    eng.start()
    random.seed(2026)

    writer = None
    if args.annotate:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.annotate, fourcc, fps,
                                 (first.shape[1], first.shape[0]))

    timeline: list[dict] = []
    violations: list[str] = []
    frames_dir = Path(args.frames_dir) if args.frames_dir else None
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)
    frames_manifest: list[dict] = []
    frame_i = 0
    tick_i = 0
    last_result = {"in_game": True, "pillars": [], "aframe": None, "commands": []}
    t_start = time.perf_counter()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        video_ms = frame_i / fps * 1000.0
        if video_ms > args.max_min * 60000:
            break

        # 写标注版视频（用上一次判定结果叠加）
        if writer is not None:
            writer.write(draw_judgment(frame, last_result, ts_str(video_ms)))

        # 决策节拍
        if frame_i % step_frames == 0:
            tick_ms = int(video_ms)
            scene = det.process(frame, tick_ms)
            eng.on_scene(scene)
            cmds = [format_command(journal_to_cmd(j)) for j in ex.journal]
            ex.journal.clear()
            last_result = {
                "in_game": scene.in_game,
                "pillars": [(p.cx, p.cy, p.color) for p in scene.pillars],
                "aframe": (scene.aframe.x, scene.aframe.y) if scene.aframe else None,
                "item_list": [{"x": e.x, "y": e.y, "w": e.w, "h": e.h}
                              for e in scene.item_list],
                "loading": scene.loading,
                "commands": cmds,
            }
            timeline.append({"t": ts_str(video_ms), "ms": tick_ms,
                             **{k: v for k, v in last_result.items()},
                             "pillars_full": last_result["pillars"]})
            # 导出决策帧（审查网页素材）
            if frames_dir is not None:
                fname = f"tick_{tick_ms:09d}.png"
                cv2.imwrite(str(frames_dir / fname),
                            draw_judgment(frame, last_result, ts_str(video_ms)))
                frames_manifest.append({
                    "image": fname, "t": ts_str(video_ms),
                    "gate": "战斗" if last_result["in_game"] else "拦截",
                    "pillars": last_result["pillars"],
                    "aframe": last_result["aframe"],
                    "commands": last_result["commands"],
                })
            # 实时不变量
            if not scene.in_game and cmds:
                violations.append(f"[{ts_str(video_ms)}] R3 门禁关闭却有命令 {cmds}")
            for c in cmds:
                if c.startswith("KEY_TAP a") and scene.aframe is None:
                    violations.append(f"[{ts_str(video_ms)}] R4 盲按A（无A框）")
                if "(" in c and c.rstrip().endswith(")"):
                    try:
                        xy = c[c.rfind("(") + 1 : c.rfind(")")].split(",")
                        x, y = int(xy[0]), int(xy[1])
                        if not (0 <= x < scene.w and 0 <= y < scene.h):
                            violations.append(f"[{ts_str(video_ms)}] R5 坐标越界 {c}")
                    except Exception:
                        pass
            tick_i += 1

        frame_i += 1

    cap.release()
    if writer is not None:
        writer.release()
    elapsed = time.perf_counter() - t_start

    # ---------------- 汇总 ----------------
    n_cmd = sum(len(t["commands"]) for t in timeline)
    n_click = sum(1 for t in timeline for c in t["commands"] if c.startswith("MOUSE"))
    n_a = sum(1 for t in timeline for c in t["commands"] if c.startswith("KEY_TAP a"))
    n_ingame = sum(1 for t in timeline if t["in_game"])
    print("=" * 66)
    print(f" 回放完成: {ts_str(video_ms)} 视频 / {tick_i} 次决策 / 处理耗时 {elapsed:.1f}s")
    print(f" 场景: 战斗 {n_ingame} tick | 门禁拦截 {tick_i - n_ingame} tick")
    print(f" 命令: 共 {n_cmd} 条（走位点击 {n_click} / 按A {n_a}）")
    print(f" 自检: {'全部通过 ✓' if not violations else f'{len(violations)} 条违规!'}")
    for v in violations[:10]:
        print("  ", v)
    print("=" * 66)

    if args.json:
        Path(args.json).write_text(
            json.dumps({"video": args.video, "hz": args.hz,
                        "violations": violations, "timeline": timeline},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"时间线 JSON → {args.json}")

    if frames_dir is not None:
        mf = frames_dir / "manifest.json"
        mf.write_text(json.dumps({"video": args.video, "items": frames_manifest},
                                 ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"决策帧 {len(frames_manifest)} 张 + manifest → {frames_dir}")

    if args.csv:
        import csv as _csv
        with Path(args.csv).open("w", encoding="utf-8-sig", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["时间", "门禁", "加载", "光柱数", "光柱位置", "A框", "命令"])
            for t in timeline:
                w.writerow([
                    t["t"], "战斗" if t["in_game"] else "拦截",
                    "是" if t.get("loading") else "否",
                    len(t["pillars"]),
                    " ".join(f"({x},{y})" for x, y, _c in t["pillars"]),
                    f"({t['aframe'][0]},{t['aframe'][1]})" if t["aframe"] else "",
                    " -> ".join(t["commands"]) if t["commands"] else "",
                ])
        print(f"时间线 CSV → {args.csv}")

    # 打印有动作的时间段（操作步骤摘要）
    active = [t for t in timeline if t["commands"]]
    if active:
        print("\n—— 操作时间线（有命令的决策点）——")
        for t in active:
            print(f" [{t['t']}] {' → '.join(t['commands'])}")
    else:
        print("\n（全程无操作命令：未出现可拾取目标或均被门禁拦截）")


if __name__ == "__main__":
    main()
