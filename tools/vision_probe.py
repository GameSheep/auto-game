"""视觉探针 —— 在真实视频帧上跑检测器，打印每帧检出并落标注图（排障用）。

用法：
  python tools/vision_probe.py 视频.mp4 --stamps 10,60,120 [--config config.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autogame.config import load_config  # noqa: E402
from autogame.vision.auto_calib import auto_hsv_ranges  # noqa: E402
from autogame.vision.detector import Detector, ROOT_TEMPLATES  # noqa: E402


def draw(frame: np.ndarray, sc) -> np.ndarray:
    out = frame.copy()
    for p in sc.pillars:
        cv2.rectangle(out, (p.x, p.y), (p.x + p.w, p.y + p.h), (0, 215, 255), 2)
    for e in sc.item_list:
        cv2.rectangle(out, (e.x - e.w // 2, e.y - e.h // 2),
                      (e.x + e.w // 2, e.y + e.h // 2), (255, 200, 0), 2)
    for e in sc.ground_labels:
        cv2.rectangle(out, (e.x - e.w // 2, e.y - e.h // 2),
                      (e.x + e.w // 2, e.y + e.h // 2), (0, 200, 255), 1)
    for m in sc.monsters:
        cv2.circle(out, (m.x, m.y), 14, (255, 0, 255), 2)
    if sc.aframe:
        a = sc.aframe
        cv2.rectangle(out, (a.x - a.w // 2, a.y - a.h // 2),
                      (a.x + a.w // 2, a.y + a.h // 2), (0, 255, 0), 2)
    if sc.rechallenge:
        r = sc.rechallenge
        cv2.rectangle(out, (r.x - 60, r.y - 18), (r.x + 60, r.y + 18), (64, 200, 255), 2)
    txt = (f"gate={'Y' if sc.in_game else 'N'} load={'Y' if sc.loading else 'N'} "
           f"pillars={len(sc.pillars)} aframe={'Y' if sc.aframe else 'N'} "
           f"mon={len(sc.monsters)} list={len(sc.item_list)} lbl={len(sc.ground_labels)} "
           f"rech={'Y' if sc.rechallenge else 'N'} d={sc.frame_delta:.2f}")
    cv2.putText(out, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(out, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--stamps", default="10,60,120")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-auto-hsv", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("打不开视频")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cfg = load_config(args.config).vision

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, first = cap.read()
    ranges = [] if args.no_auto_hsv else auto_hsv_ranges(first)
    if ranges:
        cfg.hsv = {f"auto{i}": r for i, r in enumerate(ranges)}
        print("自动标定:", [(r[0][0], r[1][0]) for r in ranges])
    det = Detector(cfg, ROOT_TEMPLATES if ROOT_TEMPLATES.exists() else None)

    out_dir = Path(args.out) if args.out else ROOT / "assets" / "replay" / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in [float(x) for x in args.stamps.split(",")]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(s * fps))
        ok, f = cap.read()
        if not ok:
            continue
        sc = det.process(f, int(s * 1000))
        print(f"[{s:6.1f}s] gate={sc.in_game} load={sc.loading} "
              f"pillars={[(p.cx, p.cy, p.color) for p in sc.pillars]} "
              f"aframe={None if not sc.aframe else (sc.aframe.x, sc.aframe.y)} "
              f"mon={len(sc.monsters)} lbl={len(sc.ground_labels)} "
              f"list={len(sc.item_list)} rech={bool(sc.rechallenge)} d={sc.frame_delta:.2f} "
              f"diag={{af:{det.diag['aframe_best']:.2f} gate:{det.diag['gate_best']:.2f}}}")
        cv2.imwrite(str(out_dir / f"probe_{int(s):05d}s.jpg"), draw(f, sc), 
                    [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main()
