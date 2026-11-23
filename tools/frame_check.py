"""整帧核对：直接从视频均匀抽若干帧，画出检出的怪物（绿框），拼成两张图。

整帧比小裁图可靠得多——能同时看到"框住的东西"和"框旁边真正的敌人"。
和 `monster_sheet.py` 的分工：这个工具**不用跑完整回放**，改完检测参数后
几秒就能在指定帧上看效果；`monster_sheet.py` 看的是整段录像的统计。

用法：
    python tools/frame_check.py --ticks 47,57,559 --out assets/frames_check
    python tools/frame_check.py --n 12          # 均匀抽 12 帧

注意取帧对齐：tick k 对应原始帧 `step*k`（step=fps/2），k=0 直接读，
之后每个 tick 先 grab `step-1` 帧再 read —— `read()` 自己会消耗一帧，
写成"每个 tick 都 grab step-1 再 read"会让整段偏 `step-1` 帧。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autogame.config import load_config  # noqa: E402
from autogame.vision.detector import Detector, ROOT_TEMPLATES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="assets/replay/src/real_20261120_1933.mp4")
    ap.add_argument("--ticks", default="")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="assets/frames_check")
    args = ap.parse_args()

    cfg = load_config(str(ROOT / "config.yaml"))
    det = Detector(cfg.vision, ROOT_TEMPLATES)
    cap = cv2.VideoCapture(str(ROOT / args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = int(round(fps / 2))
    n_ticks = total // step
    want = ({int(t) for t in args.ticks.split(",")} if args.ticks
            else {int(round(i * (n_ticks - 1) / (args.n - 1))) for i in range(args.n)})

    outdir = ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    cells = []
    for k in range(max(want) + 1):
        # tick k 对应原始帧 step*k：k=0 直接读，之后每个 tick 先跳过 step-1 帧
        if k:
            for _ in range(step - 1):
                cap.grab()
        ok, frame = cap.read()
        if not ok:
            break
        st = det.process(frame, k * 500)
        if k not in want:
            continue
        h0, w0 = frame.shape[:2]
        sc = min(1.0, 1280.0 / w0)
        work = cv2.resize(frame, (int(w0 * sc), int(h0 * sc)),
                          interpolation=cv2.INTER_AREA)
        for m in st.monsters:
            x, y = int(m.x * sc), int(m.y * sc)
            bw, bh = int(m.w * sc), int(m.h * sc)
            cv2.rectangle(work, (x - bw // 2, y - bh // 2),
                          (x + bw // 2, y + bh // 2), (0, 255, 0), 2)
        cv2.putText(work, f"tick{k} monsters={len(st.monsters)} "
                          f"bars={det.diag.get('hpbar_bars')}/"
                          f"{det.diag.get('hpbar_bars_raw')}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        cells.append(cv2.resize(work, (640, 345), interpolation=cv2.INTER_AREA))
        print(f"tick{k}: monsters={len(st.monsters)} "
              f"bars={det.diag.get('hpbar_bars')}/{det.diag.get('hpbar_bars_raw')}")
    cap.release()

    if not cells:
        print("no cells")
        return 1
    half = (len(cells) + 1) // 2
    for name, part in (("a", cells[:half]), ("b", cells[half:])):
        while len(part) % 2:
            part.append(np.zeros_like(part[0]))
        rows = [np.hstack(part[i:i + 2]) for i in range(0, len(part), 2)]
        img = np.vstack(rows)
        cv2.imwrite(str(outdir / f"check_{name}.png"), img)
        print(name, img.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
