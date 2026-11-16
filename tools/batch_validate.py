"""100 张真实图片批量自检 —— 逐张跑"标定→检测→FSM"并自动判定输出正确性。

判定规则（R1-R10，每张图全部执行）：
  R1  不崩溃：全流程可完成
  R2  确定性：同图跑两遍命令完全一致
  R3  门禁一致：in_game=False 的 tick 不得有任何命令
  R4  铁律：KEY_TAP a 的 tick 必须同帧检出 A框（绝不盲按）
  R5  坐标合法：所有命令坐标在画面范围内
  R6  速率合理：单 tick 命令数 ≤ 6
  R7  UI 场景零输出：无战斗 HUD（场景门拦截）的图 → 零命令
  R8  时域确认：第 1 tick 不得直接确认光柱（需跨帧）
  R9  性能：单图全流程 ≤ 300ms
  R10 A框空间合理性：A框必须邻近某光柱（物品名提示在物品上方）

输出：assets/batch_report.json + 控制台摘要；异常帧标注图存 assets/batch_flagged/
"""

from __future__ import annotations

import glob
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import random

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autogame.config import DecisionConfig, ExecutorConfig, VisionConfig, load_config  # noqa: E402
from autogame.decision.fsm import DecisionEngine  # noqa: E402
from autogame.executor import DryRunExecutor  # noqa: E402
from autogame.protocol import Command, format_command  # noqa: E402
from autogame.vision.auto_calib import auto_hsv_ranges  # noqa: E402
from autogame.vision.detector import Detector, ROOT_TEMPLATES  # noqa: E402

TICKS = 3
STEP_MS = 1000
MAX_CMD_PER_TICK = 6
MAX_MS = 300


def journal_to_cmd(j: dict) -> Command:
    keys = {k: v for k, v in j.items() if k in Command.model_fields}
    return Command(**keys)


def _semantic(cmds: list[str]) -> list[str]:
    """归一化命令用于确定性比较：忽略 hold_ms 拟人抖动（设计行为）。"""
    import re
    return [re.sub(r"hold=\d+ms", "hold=*", c) for c in cmds]


def run_image(path: Path) -> dict:
    frame = cv2.imread(str(path))
    if frame is None:
        return {"image": path.name, "verdict": "FAIL", "failures": ["R1 无法读取"]}

    h0, w0 = frame.shape[:2]
    # 用**部署配置**跑（含 A框模板/场景门/阈值），只把 HSV 换成逐图自动标定：
    # 这样自检覆盖的是现场真实生效的配置，而不是一份脱离部署的默认值。
    cfg = load_config(str(ROOT / "config.yaml")).vision
    ranges = auto_hsv_ranges(frame)
    if ranges:
        cfg.hsv = {f"auto{i}": r for i, r in enumerate(ranges)}

    det = Detector(cfg, ROOT_TEMPLATES if ROOT_TEMPLATES.exists() else None)
    ex = DryRunExecutor(ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000))
    eng = DecisionEngine(DecisionConfig(tick_hz=1.0), ex)
    eng.start()

    t0 = time.perf_counter()
    random.seed(12345)  # 固定拟人抖动 → 验证器聚焦真实非确定性
    ticks = []
    for i in range(TICKS):
        scene = det.process(frame, 1000 + i * STEP_MS)
        eng.on_scene(scene)
        cmds = [format_command(journal_to_cmd(j)) for j in ex.journal]
        intent = eng.last_intent
        ex.journal.clear()
        ticks.append({
            "tick": i + 1,
            "in_game": scene.in_game,
            "pillars": [(p.cx, p.cy, p.color) for p in scene.pillars],
            "aframe": (scene.aframe.x, scene.aframe.y) if scene.aframe else None,
            "rechallenge": (scene.rechallenge.x, scene.rechallenge.y) if scene.rechallenge else None,
            "intent": intent,
            "commands": cmds,
        })
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # 再跑一遍验证确定性（R2）：独立检测器+FSM 实例重放
    det2 = Detector(cfg, ROOT_TEMPLATES if ROOT_TEMPLATES.exists() else None)
    ex2 = DryRunExecutor(ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000))
    eng2 = DecisionEngine(DecisionConfig(tick_hz=1.0), ex2)
    eng2.start()
    run2_seq = []
    random.seed(12345)
    for i in range(TICKS):
        scene = det2.process(frame, 1000 + i * STEP_MS)
        eng2.on_scene(scene)
        run2_seq.append(_semantic([format_command(journal_to_cmd(j)) for j in ex2.journal]))
        ex2.journal.clear()
    deterministic = run2_seq == [_semantic(t["commands"]) for t in ticks]

    # ---------------- 判定 ----------------
    failures: list[str] = []
    warnings: list[str] = []

    # R2 确定性
    if not deterministic:
        failures.append("R2 两次运行命令不一致（非确定性）")

    for t in ticks:
        # R3 门禁一致
        if not t["in_game"] and t["commands"]:
            failures.append(f"R3 tick{t['tick']} in_game=False 但有命令 {t['commands']}")
        # R4 铁律：KEY_TAP a ⇒ 同 tick 有 A框
        for c in t["commands"]:
            if c.startswith("KEY_TAP a") and t["aframe"] is None:
                failures.append(f"R4 tick{t['tick']} 按 A 但该帧无 A框（盲按!）")
        # R5 坐标合法
        for c in t["commands"]:
            if "(" in c and c.rstrip().endswith(")"):
                try:
                    xy = c[c.rfind("(") + 1 : c.rfind(")")].split(",")
                    x, y = int(xy[0]), int(xy[1])
                    if not (0 <= x < w0 and 0 <= y < h0):
                        failures.append(f"R5 tick{t['tick']} 坐标越界 {c}")
                except Exception:
                    failures.append(f"R5 tick{t['tick']} 坐标解析失败 {c}")
        # R6 速率
        if len(t["commands"]) > MAX_CMD_PER_TICK:
            warnings.append(f"R6 tick{t['tick']} 命令数 {len(t['commands'])} 偏多")

    # R7 UI 场景零输出：所有 tick 都 in_game=False → 必须零命令
    if all(not t["in_game"] for t in ticks):
        total = sum(len(t["commands"]) for t in ticks)
        if total:
            failures.append(f"R7 UI 场景输出了 {total} 条命令")
    # R8 时域确认：tick1 不得已有确认光柱
    if ticks[0]["pillars"]:
        failures.append(f"R8 tick1 直接确认了 {len(ticks[0]['pillars'])} 个光柱（未跨帧）")
    # R9 性能
    if elapsed_ms > MAX_MS:
        warnings.append(f"R9 处理耗时 {elapsed_ms:.0f}ms 超预算")
    # R10 A框邻近光柱
    last = ticks[-1]
    if last["aframe"] is not None and last["pillars"]:
        ax, ay = last["aframe"]
        near = any(((p[0] - ax) ** 2 + (p[1] - ay) ** 2) ** 0.5 <= 300 for p in last["pillars"])
        if not near:
            warnings.append("R10 A框未邻近任何光柱")

    verdict = "FAIL" if failures else ("WARN" if warnings else "PASS")
    return {
        "image": path.name,
        "size": [w0, h0],
        "verdict": verdict,
        "failures": failures,
        "warnings": warnings,
        "elapsed_ms": round(elapsed_ms, 1),
        "in_game": [t["in_game"] for t in ticks],
        "pillars": [t["pillars"] for t in ticks],
        "aframe": last["aframe"],
        "rechallenge": last["rechallenge"],
        "commands": [t["commands"] for t in ticks],
        "intents": [t.get("intent", "") for t in ticks],
    }


def annotate(report: dict, src: Path, out_dir: Path) -> None:
    """异常帧标注图：画上检测与命令。"""
    frame = cv2.imread(str(src))
    if frame is None:
        return
    for ti, tick in enumerate(report["pillars"]):
        pass
    last = report["pillars"][-1]
    for (cx, cy, color) in last:
        cv2.rectangle(frame, (cx - 15, cy - 60), (cx + 15, cy + 10), (0, 215, 255), 2)
    if report["aframe"]:
        ax, ay = report["aframe"]
        cv2.rectangle(frame, (ax - 20, ay - 20), (ax + 20, ay + 20), (0, 255, 0), 2)
    st = "INGAME" if report["in_game"][-1] else "GATED"
    cv2.putText(frame, f"{report['verdict']} {st} {Path(src).name}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if report['verdict']=='FAIL' else (0,255,255), 2)
    for ti, cmds in enumerate(report["commands"]):
        cv2.putText(frame, f"t{ti+1}: {'; '.join(cmds) if cmds else '-'}",
                    (10, 50 + ti * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    out_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir / f"{Path(src).stem}_{report['verdict']}.png"), frame)


def main() -> None:
    frames = sorted(glob.glob("assets/batch/img_*.png"))
    if not frames:
        raise SystemExit("assets/batch 下无帧，先抽帧")
    flagged_dir = ROOT / "assets" / "batch_flagged"
    for f in flagged_dir.glob("*.png"):
        f.unlink()

    reports = []
    n_fail = n_warn = n_pass = 0
    t0 = time.perf_counter()
    for p in frames:
        rep = run_image(Path(p))
        reports.append(rep)
        n_fail += rep["verdict"] == "FAIL"
        n_warn += rep["verdict"] == "WARN"
        n_pass += rep["verdict"] == "PASS"
        if rep["verdict"] != "PASS":
            annotate(rep, Path(p), flagged_dir)

    total = len(reports)
    elapsed = time.perf_counter() - t0
    summary = {
        "total": total,
        "pass": n_pass,
        "warn": n_warn,
        "fail": n_fail,
        "wall_s": round(elapsed, 1),
        "avg_ms": round(sum(r["elapsed_ms"] for r in reports) / total, 1),
        "in_game_frames": sum(1 for r in reports if r["in_game"][-1]),
        "ui_frames": sum(1 for r in reports if not r["in_game"][-1]),
        "frames_with_pillars": sum(1 for r in reports if any(r["pillars"])),
        "frames_with_aframe": sum(1 for r in reports if r["aframe"]),
        "frames_with_commands": sum(1 for r in reports if any(any(c for c in tick) for tick in r["commands"])),
        "total_commands": sum(len(c) for r in reports for c in r["commands"]),
        "a_press_commands": sum(
            1 for r in reports for tick in r["commands"] for c in tick
            if c.startswith("KEY_TAP a")
        ),
        "failures_detail": {r["image"]: r["failures"] for r in reports if r["verdict"] == "FAIL"},
        "warnings_detail": {r["image"]: r["warnings"] for r in reports if r["verdict"] == "WARN"},
    }
    (ROOT / "assets" / "batch_report.json").write_text(
        json.dumps({"summary": summary, "reports": reports}, ensure_ascii=False, indent=1),
        encoding="utf-8")

    # Markdown 清单（图片位置 + 检出 + 操作步骤，供人直接阅读）
    md = ["# 100 张真实图片测试清单（自动刷新）",
          "",
          f"- 更新: {datetime.now().isoformat(timespec='seconds')}  "
          f"判定: PASS {n_pass} / WARN {n_warn} / FAIL {n_fail}",
          "",
          "| # | 图片位置 | 尺寸 | 场景门 | 检出 | 操作步骤 | 判定 |",
          "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(reports, 1):
        last = r["pillars"][-1] if r["pillars"] else []
        det = (f"光柱x{len(last)} " + " ".join(f"({cx},{cy})" for cx, cy, _c in last)
               if last else "")
        if r["aframe"]:
            det += f" A框({r['aframe'][0]},{r['aframe'][1]})"
        cmds = " → ".join(c for tick in r["commands"] for c in tick) or "无动作"
        gate = "战斗" if r["in_game"][-1] else "UI拦截"
        md.append(f"| {i} | assets/batch/{r['image']} | {r['size'][0]}x{r['size'][1]} "
                  f"| {gate} | {det or '无'} | {cmds} | {r['verdict']} |")
    (ROOT / "assets" / "batch_list.md").write_text("\n".join(md), encoding="utf-8")

    # CSV 清单（图片位置 + 检出 + 操作步骤）
    import csv as _csv
    with (ROOT / "assets" / "batch_list.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["序号", "图片位置", "尺寸", "场景门", "检出", "操作步骤", "意图", "判定", "问题"])
        for i, r in enumerate(reports, 1):
            last = r["pillars"][-1] if r["pillars"] else []
            det = (f"光柱x{len(last)} " + " ".join(f"({cx},{cy})" for cx, cy, _c in last)
                   if last else "")
            if r["aframe"]:
                det += f" A框({r['aframe'][0]},{r['aframe'][1]})"
            cmds = " -> ".join(c for tick in r["commands"] for c in tick) or "无动作"
            intents = " / ".join(r.get("intents", []))
            gate = "战斗" if r["in_game"][-1] else "UI拦截"
            w.writerow([i, "assets/batch/" + r["image"],
                        f"{r['size'][0]}x{r['size'][1]}", gate, det, cmds,
                        intents, r["verdict"], "; ".join(r["failures"] + r["warnings"])])

    print("=" * 66)
    print(f" 批量自检: {total} 张真实图片   耗时 {elapsed:.1f}s  平均 {summary['avg_ms']}ms/图")
    print(f" 判定: PASS {n_pass} / WARN {n_warn} / FAIL {n_fail}")
    print(f" 场景分布: 战斗HUD {summary['in_game_frames']} | UI拦截 {summary['ui_frames']}")
    print(f" 检出: 含确认光柱 {summary['frames_with_pillars']} | 含A框 {summary['frames_with_aframe']}")
    print(f" 命令: {summary['frames_with_commands']} 张图共 {summary['total_commands']} 条 "
          f"(按A {summary['a_press_commands']} 次)")
    print("=" * 66)
    for img, fails in summary["failures_detail"].items():
        print(f" [FAIL] {img}: {fails}")
    for img, warns in summary["warnings_detail"].items():
        print(f" [WARN] {img}: {warns}")


if __name__ == "__main__":
    main()
