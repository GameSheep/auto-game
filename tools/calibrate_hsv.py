"""HSV 标定工具 —— 从游戏截图中标定光柱颜色范围。

模式一（交互 GUI，标定机上使用）：
  .venv/Scripts/python tools/calibrate_hsv.py --image shot.png
  拖动 6 个滑块调 H/S/V 范围，右侧实时预览掩码；按 s 打印 YAML 片段，按 q 退出。

模式二（非交互采样，可脚本化）：
  .venv/Scripts/python tools/calibrate_hsv.py --image shot.png --rect 600,300,80,160
  对指定矩形采样，输出建议 HSV 范围（百分位 + 余量）与 YAML 片段。
  不带 --rect 时对全图找高饱和亮色簇并给出建议。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_image(path: str | None) -> np.ndarray:
    if path:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"无法读取图片: {path}")
        return img
    # 无图片 → 抓当前屏幕（mss）
    import mss

    with mss.mss() as sct:
        mon = sct.monitors[1]
        img = np.asarray(sct.grab(mon))[:, :, :3]
        return np.ascontiguousarray(img)


def sample_range(hsv: np.ndarray, rect: tuple[int, int, int, int] | None,
                 percentile: float = 8.0, margin: float = 0.15) -> dict:
    """对区域采样，输出建议 HSV 范围（config.yaml 格式）。

    返回 ranges: [[[h_lo,s_lo,v_lo],[h_hi,s_hi,v_hi]], ...]
    H 环形量用向量均值求主色调；跨 0/179 环界时拆成两个区间。
    """
    if rect:
        x, y, w, h = rect
        roi = hsv[y : y + h, x : x + w]
        # 自适应预过滤：ROI 里通常混有暗背景，按 ROI 内最大 S/V 归一，
        # 只保留目标亮色像素（V≥0.6·maxV 且 S≥0.5·maxS），避免背景污染统计
        v_max, s_max = int(roi[:, :, 2].max()), int(roi[:, :, 1].max())
        keep = (roi[:, :, 2] >= max(60, v_max * 0.6)) & (roi[:, :, 1] >= max(50, s_max * 0.5))
        filtered = roi[keep]
        if len(filtered) >= 30:
            roi = filtered.reshape(-1, 3)
        else:
            roi = roi.reshape(-1, 3)
    else:
        # 全图模式：只统计高饱和高亮像素（光柱特征），避免背景污染
        mask = (hsv[:, :, 1] > 120) & (hsv[:, :, 2] > 180)
        roi = hsv[mask] if mask.any() else hsv.reshape(-1, 3)
        roi = roi.reshape(-1, 3)
    if roi.size == 0:
        raise SystemExit("采样区域为空")

    # H 是环形量：向量均值法求主色调，环形展开后取分位数
    h_deg = roi[:, 0].astype(np.float64) * 2.0  # 0-360
    ang = np.deg2rad(h_deg)
    center_deg = float(np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360)
    rel = ((h_deg - center_deg + 180) % 360) - 180
    lo_h_rel, hi_h_rel = np.percentile(rel, [percentile, 100 - percentile])
    lo_h_deg = (center_deg + lo_h_rel) % 360
    hi_h_deg = (center_deg + hi_h_rel) % 360

    s_lo, s_hi = np.percentile(roi[:, 1], [percentile, 100 - percentile])
    v_lo, v_hi = np.percentile(roi[:, 2], [percentile, 100 - percentile])
    s_lo_i = int(max(0, s_lo * (1 - margin)))
    s_hi_i = int(min(255, s_hi * (1 + margin)))
    v_lo_i = int(max(0, v_lo * (1 - margin)))
    v_hi_i = int(min(255, v_hi * (1 + margin)))

    def _rng(h_lo_deg: float, h_hi_deg: float) -> list[list[int]]:
        # OpenCV H: 0-179（度/2），带 margin 外扩
        h_lo = int(max(0, (h_lo_deg / 2) * (1 - margin)))
        h_hi = int(min(179, (h_hi_deg / 2) * (1 + margin)))
        return [[h_lo, s_lo_i, v_lo_i], [h_hi, s_hi_i, v_hi_i]]

    if hi_h_deg < lo_h_deg:  # 跨 0/360 环界 → 两段
        ranges = [_rng(lo_h_deg, 360.0), _rng(0.0, hi_h_deg)]
    else:
        ranges = [_rng(lo_h_deg, hi_h_deg)]
    return {
        "ranges": ranges,
        "h_center_deg": round(center_deg, 1),
        "samples": int(len(roi)),
    }


def yaml_snippet(name: str, ranges: list[list[list[int]]]) -> str:
    if len(ranges) == 1:
        (lo, hi) = ranges[0]
        return (f"    {name}: [[{lo[0]}, {lo[1]}, {lo[2]}], "
                f"[{hi[0]}, {hi[1]}, {hi[2]}]]")
    parts = []
    for i, (lo, hi) in enumerate(ranges):
        parts.append(f"    {name}_part{i + 1}: [[{lo[0]}, {lo[1]}, {lo[2]}], "
                     f"[{hi[0]}, {hi[1]}, {hi[2]}]]")
    note = f"    # {name} 跨 H 环界，拆成两个区间（检测器按多 key 分别检测）"
    return note + "\n" + "\n".join(parts)


def gui(image: np.ndarray) -> None:
    """交互滑块标定（需要桌面环境）。"""
    win = "calibrate_hsv (s=输出,q=退出)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("H_lo", win, 18, 179, lambda v: None)
    cv2.createTrackbar("H_hi", win, 35, 179, lambda v: None)
    cv2.createTrackbar("S_lo", win, 120, 255, lambda v: None)
    cv2.createTrackbar("S_hi", win, 255, 255, lambda v: None)
    cv2.createTrackbar("V_lo", win, 180, 255, lambda v: None)
    cv2.createTrackbar("V_hi", win, 255, 255, lambda v: None)

    view = image
    max_w = 1280
    if view.shape[1] > max_w:
        s = max_w / view.shape[1]
        view = cv2.resize(view, (max_w, int(view.shape[0] * s)))
    hsv = cv2.cvtColor(view, cv2.COLOR_BGR2HSV)

    while True:
        h_lo = cv2.getTrackbarPos("H_lo", win)
        h_hi = cv2.getTrackbarPos("H_hi", win)
        s_lo = cv2.getTrackbarPos("S_lo", win)
        s_hi = cv2.getTrackbarPos("S_hi", win)
        v_lo = cv2.getTrackbarPos("V_lo", win)
        v_hi = cv2.getTrackbarPos("V_hi", win)
        lo = np.array([h_lo, s_lo, v_lo], np.uint8)
        hi = np.array([max(h_lo, h_hi), max(s_lo, s_hi), max(v_lo, v_hi)], np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        both = np.hstack([view, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        cv2.imshow(win, both)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("s"):
            print(f"\n  config.yaml → vision.hsv:\n"
                  f"    custom: [[{h_lo}, {s_lo}, {v_lo}], [{h_hi}, {s_hi}, {v_hi}]]\n")
        elif k == ord("q") or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="HSV 标定工具")
    parser.add_argument("--image", default=None, help="截图路径；缺省抓当前屏幕")
    parser.add_argument("--rect", default=None, help="采样矩形 x,y,w,h（非交互模式）")
    parser.add_argument("--name", default="custom", help="输出配置键名")
    args = parser.parse_args()

    image = load_image(args.image)
    if args.rect:
        rect = tuple(int(v) for v in args.rect.split(","))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        result = sample_range(hsv, rect)
        print(f"采样点数: {result['samples']}  主色调: {result['h_center_deg']}°")
        print(f"建议范围: {result['ranges']}")
        print(f"\nconfig.yaml → vision.hsv:\n{yaml_snippet(args.name, result['ranges'])}")
    elif not args.image and sys.stdout.isatty():
        gui(image)
    else:
        # 无 GUI 环境的兜底：全图高亮簇采样
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        result = sample_range(hsv, None)
        print(f"全图高亮簇采样点数: {result['samples']}  主色调: {result['h_center_deg']}°")
        print(f"建议范围: {result['ranges']}")


if __name__ == "__main__":
    main()
