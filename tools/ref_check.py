"""拿人工标定过的参考图台账（assets/ref/labels.json）去核对检测器 + FSM。

这是"用真实画面验证程序"的人工标答版：截图是人给的、答案是人对的，
程序跑出什么就打印什么，逐字段对比。判据口径与生产完全一致
（config.yaml 的 vision 段 + 2Hz 决策节拍 + DryRun 执行器）。

用法：
    python tools/ref_check.py                 # 全部条目，差异表
    python tools/ref_check.py --only tli_     # 只跑文件名含 tli_ 的
    python tools/ref_check.py -v              # 附带每帧检出详情

退出码：任一 strict 条目不符 → 1（CI/回归用）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autogame.config import DecisionConfig, ExecutorConfig, load_config  # noqa: E402
from autogame.decision.fsm import DecisionEngine  # noqa: E402
from autogame.executor import DryRunExecutor  # noqa: E402
from autogame.vision.detector import Detector, ROOT_TEMPLATES  # noqa: E402

LABELS = ROOT / "assets" / "ref" / "labels.json"
TICKS = 2  # 两个决策节拍：够光柱时域确认（beam_persistence_frames=2）走完一轮


def measure(frame, det: Detector, engine: DecisionEngine, tick_hz: float) -> dict:
    """跑 TICKS 个决策节拍，返回实测字段（与 labels.json 的 expect 同口径）。"""
    ts = 1000
    st = None
    for _ in range(TICKS):
        st = det.process(frame, ts)
        engine.on_scene(st)
        ts += int(1000 / tick_hz)
    return {
        "in_game": bool(st.in_game),
        "pillars": len(st.pillars),
        "aframe": st.aframe is not None,
        "ground_labels": len(st.ground_labels),
        "monsters": len(st.monsters),
        "commands": len(engine.exec.journal),
        "_pillar_colors": sorted({p.color for p in st.pillars}),
        "_gate_reason": det.diag.get("gate_reason", ""),
        "_gate_best": det.diag.get("gate_best", 0.0),
        "_gate_neg": det.diag.get("gate_neg_best", 0.0),
    }


def run(labels_path: Path, only: str = "", verbose: bool = False) -> int:
    """核对整本台账，返回 strict 不符条数（0 = 全对）。"""
    if not labels_path.exists():
        raise SystemExit(f"没有台账: {labels_path}")
    data = json.loads(labels_path.read_text(encoding="utf-8"))
    items = [it for it in data.get("items", [])
             if only in it.get("file", "")]

    cfg = load_config(str(ROOT / "config.yaml")).vision
    tick_hz = 2.0
    strict_fail = 0
    strict_total = 0

    for it in items:
        path = ROOT / it["file"]
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"[缺图] {it['file']}")
            strict_fail += 1 if it.get("strict") else 0
            strict_total += 1 if it.get("strict") else 0
            continue

        det = Detector(cfg, ROOT_TEMPLATES if ROOT_TEMPLATES.exists() else None)
        ex = DryRunExecutor(ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000))
        eng = DecisionEngine(DecisionConfig(tick_hz=tick_hz), ex)
        eng.start()
        got = measure(frame, det, eng, tick_hz)

        diffs = []
        for key, want in (it.get("expect") or {}).items():
            if want is None:
                continue
            have = got.get(key)
            if have != want:
                diffs.append(f"{key}: 标答={want} 实测={have}")

        strict = bool(it.get("strict"))
        strict_total += 1 if strict else 0
        if diffs and strict:
            strict_fail += 1
        mark = "OK  " if not diffs else ("FAIL" if strict else "GAP ")
        print(f"[{mark}] {Path(it['file']).name:26s} {it.get('scene', '')}")
        print(f"       门禁={got['_gate_reason']} (模板 {got['_gate_best']:.2f}"
              f" / 反例 {got['_gate_neg']:.2f})  光柱={got['pillars']}"
              f"{got['_pillar_colors'] if got['pillars'] else ''}  A框={got['aframe']}"
              f"  标签={got['ground_labels']}  怪={got['monsters']}  指令={got['commands']}")
        for d in diffs:
            print(f"       ✗ {d}")
        if verbose:
            print(f"       备注: {it.get('note', '')}")

    gaps = len(items) - strict_total
    print(f"\n合计 {len(items)} 条：strict {strict_total} 条，通过 "
          f"{strict_total - strict_fail}，不符 {strict_fail}；已知缺口 {gaps} 条。")
    if gaps:
        print("提示：GAP 条目是『还没标定好』，不是通过。补齐方式见条目里的备注。")
    return strict_fail


def main() -> int:
    ap = argparse.ArgumentParser(description="参考图台账核对")
    ap.add_argument("--labels", default=str(LABELS))
    ap.add_argument("--only", default="", help="只跑文件名包含该子串的条目")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    return 1 if run(Path(args.labels), args.only, args.verbose) else 0


if __name__ == "__main__":
    raise SystemExit(main())
