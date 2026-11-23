"""真实录像批量回放 —— 视频 → 决策帧 → 每帧完整指令（用部署配置跑，不另设一套参数）。

与 `tools/video_replay.py` 的区别（那个是"看个大概"的交互工具）：
  * **用 `config.yaml` 的部署配置**跑检测与决策（含 A框模板/场景门/自主性），
    避免"验证的不是线上那套参数"；
  * 逐帧落 **原始截图**（不烧字，叠加层交给审查页画，坐标可核对）+
    **manifest.json**（检出/意图/完整命令对象/不变量判定），供审查页逐帧核对；
  * 只读视频、零输出到游戏（DryRunExecutor），命令仅记录不发送。

用法：
  python tools/video_batch.py assets/replay/src/xx.mp4
  python tools/video_batch.py a.mp4 b.mp4 --hz 2 --max-min 5 --out-root assets/frames_replay
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autogame.config import ExecutorConfig, load_config  # noqa: E402
from autogame.decision.fsm import DecisionEngine  # noqa: E402
from autogame.executor import DryRunExecutor  # noqa: E402
from autogame.protocol import Command, format_command  # noqa: E402
from autogame.vision.auto_calib import auto_hsv_ranges  # noqa: E402
from autogame.vision.detector import Detector, ROOT_TEMPLATES  # noqa: E402

FRAME_W = 960          # 落盘宽度（审查页够看，体积可控）
JPEG_Q = 78


def ts_str(ms: float) -> str:
    s = ms / 1000.0
    return f"{int(s // 60):02d}:{s % 60:05.2f}"


def _journal_to_cmd(j: dict) -> Command:
    """执行器 journal 条目（含 ts/path 等旁路字段）→ Command。"""
    return Command(**{k: v for k, v in j.items() if k in Command.model_fields})


def _cmd_dict(c: Command) -> dict:
    return {"op": c.op, "key": c.key, "button": c.button, "x": c.x, "y": c.y,
            "hold_ms": c.hold_ms, "text": format_command(c)}


def _pickup_evidence(scene, dec_cfg) -> bool:
    """这一帧按拾取键有没有像素证据（R4 铁律的判定口径）。

    "绝不盲按"的本意是"没有证据不许按"，不是"只有 A框能按"：
      * 暗黑4 类：A框（近距交互提示）出现 = 证据；
      * 火炬之光 PC 类：没有 A框，拾取提示就是地面标签 —— 标签落在跟前
        （`label_pickup_near_ratio × 画面短边`）同样是证据。
    两条都不满足才算盲按。
    """
    if scene.aframe is not None:
        return True
    if not getattr(dec_cfg, "label_pickup_when_near", False) or not scene.ground_labels:
        return False
    near = float(dec_cfg.label_pickup_near_ratio) * min(scene.w, scene.h)
    ch = scene.character
    return any(math.hypot(g.x - ch[0], g.y - ch[1]) <= near
               for g in scene.ground_labels)


def run_video(video: Path, out_root: Path, hz: float, max_min: float,
              cfg_path: Path, name: str | None = None,
              hsv_mode: str = "auto", jpeg: bool = True,
              gate_unmatched: str = "hold") -> dict:
    app_cfg = load_config(str(cfg_path))
    vcfg = app_cfg.vision
    # 审查语料用：门禁"没认出来"的帧按放行分析（纯离线，不产生真实输入）。
    # 生产默认 hold（未标定就不动手）；分析时 pass 才能看到感知层到底想干什么。
    if gate_unmatched == "pass":
        vcfg.scene_gate.fail_closed = False
    if hsv_mode == "auto":
        cap0 = cv2.VideoCapture(str(video))
        ok0, first = cap0.read()
        cap0.release()
        if ok0:
            ranges = auto_hsv_ranges(first)
            if ranges:
                vcfg.hsv = {f"auto{i}": r for i, r in enumerate(ranges)}

    name = name or video.stem
    out = out_root / name
    (out / "frames").mkdir(parents=True, exist_ok=True)
    # 不删旧帧：文件名按 tick 序号覆盖写，审查页只渲染 manifest 里列出的帧，
    # 上一次跑剩的多余文件不会被引用（也避免脚本批量删文件触发安全确认）。

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, int(round(fps / hz)))
    limit_ms = max_min * 60000.0

    det = Detector(vcfg, ROOT_TEMPLATES if ROOT_TEMPLATES.exists() else None)
    # 决策参数取部署配置（自主性/超时/铁律都按线上），只把 tick_hz 对齐回放节拍
    dec_cfg = app_cfg.decision.model_copy(update={"tick_hz": hz})
    ex = DryRunExecutor(ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=10000))
    eng = DecisionEngine(dec_cfg, ex)
    eng.start()
    random.seed(20261117)  # 拟人抖动可复现

    items: list[dict] = []
    violations: list[str] = []
    per_op: dict[str, int] = {}
    t0 = time.perf_counter()
    fi = 0
    n_tick = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        video_ms = fi / fps * 1000.0
        if video_ms > limit_ms:
            break
        if fi % step == 0:
            scene = det.process(frame, int(video_ms))
            eng.on_scene(scene)
            cmds = [_journal_to_cmd(j) for j in ex.journal]
            ex.journal.clear()
            for c in cmds:
                per_op[c.op] = per_op.get(c.op, 0) + 1

            # 不变量（与 batch_validate 同口径，逐帧记录）
            v: list[str] = []
            if not scene.in_game and cmds:
                v.append(f"R3 门禁关闭却有 {len(cmds)} 条命令")
            for c in cmds:
                if c.op == "key_tap" and (c.key or "").lower() == dec_cfg.pickup_key.lower() \
                        and not _pickup_evidence(scene, dec_cfg):
                    v.append("R4 盲按拾取键（该帧既无 A框、也无近距标签）")
                if c.x is not None and c.y is not None:
                    if not (0 <= c.x < scene.w and 0 <= c.y < scene.h):
                        v.append(f"R5 坐标越界 {format_command(c)}")
            # R6 单帧命令数（执行器正常节流时不该出现爆量）
            if len(cmds) > 6:
                v.append(f"R6 单帧 {len(cmds)} 条命令偏多")
            for x in v:
                violations.append(f"[{ts_str(video_ms)}] {x}")

            fname = f"t{n_tick:05d}.jpg"
            if jpeg:
                fh, fw = frame.shape[:2]
                if fw > FRAME_W:
                    frame_s = cv2.resize(frame, (FRAME_W, int(fh * FRAME_W / fw)),
                                         interpolation=cv2.INTER_AREA)
                else:
                    frame_s = frame
                cv2.imwrite(str(out / "frames" / fname), frame_s,
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])

            items.append({
                "image": fname,
                "t": ts_str(video_ms),
                "ms": int(video_ms),
                "frame_size": [scene.w, scene.h],
                "image_size": [int(frame.shape[1] * min(1, FRAME_W / frame.shape[1])),
                               int(frame.shape[0] * min(1, FRAME_W / frame.shape[1]))],
                "gate": "战斗" if scene.in_game else "拦截",
                "gate_reason": det.diag.get("gate_reason", ""),
                "in_game": scene.in_game,
                "loading": scene.loading,
                "frame_delta": scene.frame_delta,
                # 角色位置（原始帧坐标）：审查页要能核对"这次走位点离角色多远、
                # 方向对不对"。之前没记，导致复核时只能猜角色在屏幕中心。
                "character": [int(scene.character[0]), int(scene.character[1])],
                "state": eng.state.value,
                "intent": eng.last_intent,
                "pillars": [{"x": p.x, "y": p.y, "cx": p.cx, "cy": p.cy, "w": p.w, "h": p.h,
                             "color": p.color, "distance": round(p.distance, 1),
                             "stable": p.stable, "seen": p.seen_frames}
                            for p in scene.pillars],
                "aframe": None if scene.aframe is None else {
                    "x": scene.aframe.x, "y": scene.aframe.y,
                    "w": scene.aframe.w, "h": scene.aframe.h},
                "rechallenge": None if scene.rechallenge is None else {
                    "x": scene.rechallenge.x, "y": scene.rechallenge.y},
                "monsters": [{"x": m.x, "y": m.y, "w": m.w, "h": m.h} for m in scene.monsters],
                "ground_labels": [{"x": e.x, "y": e.y, "w": e.w, "h": e.h}
                                  for e in scene.ground_labels],
                "item_list": [{"x": e.x, "y": e.y, "w": e.w, "h": e.h}
                              for e in scene.item_list],
                "commands": [_cmd_dict(c) for c in cmds],
                "violations": v,
            })
            n_tick += 1
        fi += 1
    cap.release()
    elapsed = time.perf_counter() - t0

    summary = {
        "name": name,
        "video": str(video),
        "src": {"fps": round(fps, 2), "frames": total_frames,
                "size": [src_w, src_h],
                "duration_s": round(total_frames / fps, 1) if fps else 0},
        "hz": hz,
        "gate_unmatched": gate_unmatched,
        "replay_s": round(elapsed, 1),
        "ticks": n_tick,
        "gate_pass": sum(1 for it in items if it["in_game"]),
        "loading_ticks": sum(1 for it in items if it["loading"]),
        "ticks_with_commands": sum(1 for it in items if it["commands"]),
        "commands": sum(len(it["commands"]) for it in items),
        "per_op": per_op,
        "a_press": sum(1 for it in items for c in it["commands"]
                       if c["op"] == "key_tap" and (c["key"] or "") == dec_cfg.pickup_key),
        "ticks_with_pillars": sum(1 for it in items if it["pillars"]),
        "ticks_with_aframe": sum(1 for it in items if it["aframe"]),
        "ticks_with_monsters": sum(1 for it in items if it["monsters"]),
        "hsv": {k: v for k, v in vcfg.hsv.items()},
        "violations": violations,
    }
    (out / "manifest.json").write_text(
        json.dumps({"summary": summary, "items": items}, ensure_ascii=False), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"[{name}] {n_tick} 帧 / {summary['commands']} 条命令 / 门禁拦 "
          f"{n_tick - summary['gate_pass']} / 违规 {len(violations)} / {elapsed:.0f}s "
          f"/ 未标定帧={'放行分析' if gate_unmatched == 'pass' else '拦下'} → {out}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="真实录像批量回放（部署配置 + 逐帧完整指令）")
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--hz", type=float, default=2.0, help="决策/抽帧频率（对齐 capture.fps=2）")
    ap.add_argument("--max-min", type=float, default=5.0, help="每段最多回放分钟数")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--out-root", default=str(ROOT / "assets" / "frames_replay"))
    ap.add_argument("--hsv", choices=["auto", "config"], default="auto",
                    help="auto=首帧自适应标定（推荐）；config=用 config.yaml 的色域")
    ap.add_argument("--no-jpeg", action="store_true", help="只出 manifest 不落图（调试用）")
    ap.add_argument("--gate-unmatched", choices=["hold", "pass"], default="hold",
                    help="门禁没认出来时怎么处置：hold=拦下（生产默认）/ pass=当战斗跑（审查分析用）")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for v in args.videos:
        run_video(Path(v), out_root, args.hz, args.max_min, Path(args.config),
                  hsv_mode=args.hsv, jpeg=not args.no_jpeg,
                  gate_unmatched=args.gate_unmatched)


if __name__ == "__main__":
    main()
