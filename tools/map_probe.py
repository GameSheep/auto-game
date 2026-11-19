"""小地图定位标定/体检 —— 拿真实截图验一遍，别等跑起来才发现 zoom 不对。

用法（标定 zoom 顺序）：

    # 1. 先看参数对不对（会自动扫 zoom，报告最佳值）
    python tools/map_probe.py --map "地图/整图.png" --frame 截图.png --roi 0,55,245,245 --sweep

    # 2. 定下来之后正式跑一次，看定位点落在哪（输出标注图）
    python tools/map_probe.py --map "地图/整图.png" --frame 截图.png \
        --roi 0,55,245,245 --zoom 0.12 --out debug_map_probe.png

    # 3. 批量验一批截图（人物走过地图各处），看命中率
    python tools/map_probe.py --map 整图.png --dir assets/frames --roi 0,55,245,245 --zoom 0.12

为什么要扫 zoom 而不是让人手算：zoom = 小地图像素/底图像素，它等于"小地图显示的世界
范围"的倒数，而这个数字游戏不会告诉你。扫一遍（0.02~0.6）每次 matchTemplate 都是
几十毫秒，几十个候选一秒内跑完，比量两张图靠谱。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autogame.vision.imgio import imread_any, imwrite_any  # noqa: E402
from autogame.vision.minimap import MapLocator  # noqa: E402


def parse_roi(s: str) -> list[int]:
    parts = [int(v) for v in s.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--roi 需要 x,y,w,h")
    return parts


def crop(frame: np.ndarray, roi: list[int]) -> np.ndarray:
    x, y, w, h = roi
    return frame[y: y + h, x: x + w]


def sweep_zoom(world: np.ndarray, patch: np.ndarray, inset: int) -> tuple[list[tuple[float, float]], int]:
    """扫一遍 zoom，返回 ([(zoom, score)] 按分数降序, 实际跑过的档数)。

    注意 zoom 太大时缩略底图比裁片还小（相当于"小地图显示的范围比整张图还大"），
    这种档位没法匹配，直接跳过 —— 它不算"没命中"，别误导排查方向。
    """
    out: list[tuple[float, float]] = []
    ran = 0
    ph, pw = patch.shape[:2]
    mh, mw = world.shape[:2]
    for z in [round(0.02 + i * 0.01, 3) for i in range(59)]:
        if pw >= max(1, int(mw * z)) or ph >= max(1, int(mh * z)):
            continue
        try:
            loc = MapLocator(world, zoom=z, threshold=0.2, edge_threshold=0.0, inset=inset)
            hit = loc.locate(patch)
        except Exception:  # noqa: BLE001 —— 极端 zoom 下 OpenCV 会拒收尺寸，跳过
            continue
        ran += 1
        if hit is not None:
            out.append((z, hit.score))
    out.sort(key=lambda t: -t[1])
    return out, ran


def annotate(world: np.ndarray, pos: tuple[int, int], patch: np.ndarray, text: str) -> np.ndarray:
    img = world.copy()
    x, y = pos
    cv2.drawMarker(img, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 60, 3)
    cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
    cv2.putText(img, text, (max(10, x - 120), max(30, y - 40)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    # 右下角贴一张小地图裁片，方便肉眼对照"我找的是这块"
    ph, pw = patch.shape[:2]
    scale = min(180 / max(1, pw), 180 / max(1, ph), 1.0)
    thumb = cv2.resize(patch, (int(pw * scale), int(ph * scale)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    img[h - thumb.shape[0] - 10: h - 10, w - thumb.shape[1] - 10: w - 10] = thumb
    cv2.rectangle(img, (w - thumb.shape[1] - 10, h - thumb.shape[0] - 10),
                  (w - 10, h - 10), (0, 255, 0), 1)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description="小地图定位标定/体检")
    ap.add_argument("--map", required=True, help="整张地图底图")
    ap.add_argument("--frame", help="一帧游戏截图")
    ap.add_argument("--dir", help="一批截图目录（与 --frame 二选一）")
    ap.add_argument("--roi", type=parse_roi, default=[0, 55, 245, 245],
                    help="小地图 ROI x,y,w,h（默认按 TLI 1280x720 布局）")
    ap.add_argument("--zoom", type=float, default=None, help="小地图像素/底图像素")
    ap.add_argument("--inset", type=int, default=6, help="四边裁掉多少像素（去边框）")
    ap.add_argument("--sweep", action="store_true", help="先扫一遍 zoom 并报告最佳值")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--edge-threshold", type=float, default=0.25)
    ap.add_argument("--out", default="debug_map_probe.png")
    args = ap.parse_args()

    world = imread_any(args.map)
    if world is None:
        print(f"底图读不出来：{args.map}", file=sys.stderr)
        return 2

    if args.frame:
        frames = [Path(args.frame)]
    elif args.dir:
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        frames = sorted(p for p in Path(args.dir).iterdir() if p.suffix.lower() in exts)
    else:
        print("要给 --frame 或 --dir", file=sys.stderr)
        return 2
    if not frames:
        print("没有可用的截图", file=sys.stderr)
        return 2

    first = imread_any(frames[0])
    if first is None:
        print(f"截图读不出来：{frames[0]}", file=sys.stderr)
        return 2
    patch = crop(first, args.roi)
    print(f"底图 {world.shape[1]}x{world.shape[0]}  裁片 {patch.shape[1]}x{patch.shape[0]}")

    zoom = args.zoom
    if args.sweep or zoom is None:
        rows, ran = sweep_zoom(world, patch, args.inset)
        if ran == 0:
            print("没有任何 zoom 能跑：裁片比缩略后的整张底图还大，说明 zoom 的合理范围在更高处，"
                  "或者 --roi 圈错了（圈到的不含地形）", file=sys.stderr)
            return 1
        if not rows:
            print(f"扫了 {ran} 个 zoom，都没有过 0.2 分：地形不像同一张图（换图没换底图？）",
                  file=sys.stderr)
            return 1
        print("\nzoom 扫描（前 8）：")
        for z, s in rows[:8]:
            print(f"  zoom={z:<6} score={s:.3f}")
        if zoom is None:
            zoom = rows[0][0]
            print(f"\n取最佳 zoom={zoom}（正式配置里再手调 1~2 档，看哪档在不同截图上更稳）")
        if zoom not in [z for z, _s in rows]:
            print(f"注意：指定的 zoom={zoom} 在扫描里没命中")

    loc = MapLocator(world, zoom=zoom, threshold=args.threshold,
                     edge_threshold=args.edge_threshold, inset=args.inset)
    hits = 0
    for fp in frames:
        frame = imread_any(fp)
        if frame is None:
            continue
        p = crop(frame, args.roi)
        hit = loc.locate(p)
        if hit is None:
            print(f"  ✗ {fp.name}  {loc.last_diag}")
            continue
        hits += 1
        print(f"  ✓ {fp.name}  ({hit.x},{hit.y}) u={hit.x / loc.map_w:.3f} "
              f"v={hit.y / loc.map_h:.3f} score={hit.score:.3f} edge={hit.edge_score:.3f}")
        if len(frames) == 1 or hits == 1:
            img = annotate(world, (hit.x, hit.y), p, f"{hit.score:.2f}")
            if imwrite_any(args.out, img):
                print(f"  标注图已写：{args.out}")

    print(f"\n命中 {hits}/{len(frames)}")
    if hits == 0:
        print("一条都没命中，按这个顺序排查：\n"
              "  1. --roi 是不是圈到了小地图（先 --out 存一张 crop 出来看）\n"
              "  2. zoom 对不对（--sweep 选最佳那档，再看它比次优高多少；只高一点说明地形太像）\n"
              "  3. 底图和小地图是不是同一张地图（换了图忘了换底图）\n"
              "  4. 小地图有没有被 UI 挡住/半透明遮罩（裁片要能看见地形纹理）")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
