"""屏幕/画面采集工具 —— 为标定与模板制作提供素材。

用法：
  .venv/Scripts/python tools/grab_frame.py --out shot.png             # 抓当前屏幕
  .venv/Scripts/python tools/grab_frame.py --out shot.png --count 5   # 连抓5张
  .venv/Scripts/python tools/grab_frame.py --out shot.png --region 0,0,1920,1080

配套流程：
  1. 游戏里出现目标光柱 → grab_frame 抓图
  2. calibrate_hsv.py --image shot.png --rect 光柱区域 → 得到 HSV 范围
  3. 裁出 A框 小图存 templates/aframe.png（任意图像工具），启用模板匹配
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np


def grab(region: tuple[int, int, int, int] | None = None) -> np.ndarray:
    import mss

    with mss.mss() as sct:
        if region:
            left, top, w, h = region
            bbox = {"left": left, "top": top, "width": w, "height": h}
        else:
            mon = sct.monitors[1]
            bbox = {"left": mon["left"], "top": mon["top"],
                    "width": mon["width"], "height": mon["height"]}
        return np.ascontiguousarray(np.asarray(sct.grab(bbox))[:, :, :3])


def main() -> None:
    parser = argparse.ArgumentParser(description="屏幕截图工具")
    parser.add_argument("--out", required=True, help="输出 PNG 路径（--count>1 时自动加序号）")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--region", default=None, help="left,top,width,height")
    args = parser.parse_args()

    region = tuple(int(v) for v in args.region.split(",")) if args.region else None
    stem = args.out.rsplit(".", 1)[0]
    for i in range(args.count):
        img = grab(region)
        path = f"{stem}_{i + 1}.png" if args.count > 1 else args.out
        cv2.imwrite(path, img)
        print(f"[{i + 1}/{args.count}] {path}  {img.shape[1]}x{img.shape[0]}")
        if i + 1 < args.count:
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
