"""临时核对：把回放 manifest 里"放行帧上的怪物"逐格裁出来，人工看是不是真怪。

用途：血条闸门放宽后（gap 2.0×h、条宽 3.2×w）必须重新验证精度 ——
召回上去了但全是地形的话，等于把误报换了个方向。拼图里每格标注
帧号 / 坐标 / 尺寸，肉眼 3 秒能判断真假。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="assets/frames_replay/real_20261120_1933")
    ap.add_argument("--out", default="assets/monster_sheet.png")
    ap.add_argument("--only-passed", action="store_true", default=True)
    args = ap.parse_args()

    run = ROOT / args.run
    man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    cells = []
    for idx, it in enumerate(man["items"]):
        if not it.get("monsters"):
            continue
        if args.only_passed and not it.get("in_game"):
            continue
        fp = run / "frames" / Path(it["image"]).name
        img = cv2.imread(str(fp))
        if img is None:
            continue
        k = img.shape[1] / it["frame_size"][0]
        for m in it["monsters"]:
            cx, cy = int(m["x"] * k), int(m["y"] * k)
            w, h = int(m["w"] * k), int(m["h"] * k)
            pad = int(max(w, h) * 1.3)
            x0, y0 = max(0, cx - w // 2 - pad), max(0, cy - h // 2 - pad)
            x1, y1 = min(img.shape[1], cx + w // 2 + pad), min(img.shape[0], cy + h // 2 + pad)
            crop = img[y0:y1, x0:x1].copy()
            # 画框标出候选
            cv2.rectangle(crop, (cx - w // 2 - x0, cy - h // 2 - y0),
                          (cx + w // 2 - x0, cy + h // 2 - y0), (0, 255, 0), 1)
            cells.append((idx, it["t"], m["x"], m["y"], m["w"], m["h"], crop))

    if not cells:
        print("放行帧上没有怪物检出")
        return 0

    cw, ch = 210, 200
    cols = 6
    rows = (len(cells) + cols - 1) // cols
    sheet = np.full((rows * ch, cols * cw, 3), 18, np.uint8)
    for i, (idx, t, x, y, w, h, crop) in enumerate(cells):
        hh, ww = crop.shape[:2]
        k = min((cw - 8) / ww, (ch - 30) / hh)
        rs = cv2.resize(crop, (max(1, int(ww * k)), max(1, int(hh * k))),
                        interpolation=cv2.INTER_NEAREST)
        r0, c0 = (i // cols) * ch, (i % cols) * cw
        sheet[r0 + 26:r0 + 26 + rs.shape[0], c0 + 4:c0 + 4 + rs.shape[1]] = rs
        cv2.putText(sheet, f"#{idx} {t}", (c0 + 4, r0 + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(sheet, f"({x},{y}) {w}x{h}", (c0 + 4, r0 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 220, 180), 1, cv2.LINE_AA)
    out = ROOT / args.out
    cv2.imwrite(str(out), sheet)
    print(f"放行帧怪物 {len(cells)} 个 → {out}")
    for c in cells:
        print(f"  #{c[0]:03d} {c[1]} ({c[2]},{c[3]}) {c[4]}x{c[5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
