"""从视频抓若干缩略帧用于人眼确认（不落库）。"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

src = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("assets/replay/peek")
stamps = [float(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["5", "60", "180"])]
out.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(src))
if not cap.isOpened():
    raise SystemExit("打不开 " + str(src))
fps = cap.get(cv2.CAP_PROP_FPS) or 23
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"{src.name}: {cap.get(3):.0f}x{cap.get(4):.0f} {fps:.2f}fps {n}帧 = {n / fps:.1f}s")
for s in stamps:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(s * fps))
    ok, f = cap.read()
    if ok:
        p = out / f"peek_{int(s):05d}s.png"
        cv2.imwrite(str(p), f)
        print("→", p)
cap.release()
