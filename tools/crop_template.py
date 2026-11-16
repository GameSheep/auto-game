"""UI 模板裁剪工具 —— 从自己的游戏截图上裁出配方所需的按钮/面板锚点。

为什么需要它：UI 场景自动化（交易行/装备/天赋/技能）靠模板匹配定位按钮，
模板必须来自**你自己的分辨率/画质/语言**的截图。没有模板时服务端会直接
拒绝启动任务（409 + 缺失清单），而不是静默空转。

用法：
  # 先看某个配方需要哪些模板
  python tools/crop_template.py --list trade_buy

  # 从截图裁一个按钮（坐标是截图像素，左上角 + 宽高）
  python tools/crop_template.py --image shot.png --rect 950,110,120,40 --out templates/ui/sort.png

  # 带放大预览（人眼确认裁对了）
  python tools/crop_template.py --image shot.png --rect 950,110,120,40 \
      --out templates/ui/sort.png --preview assets/tpl_preview.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autogame.decision.recipe import PRESET_RECIPES, required_templates  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="从截图裁剪 UI 模板")
    ap.add_argument("--list", metavar="RECIPE", help="列出配方所需模板（all=全部配方）")
    ap.add_argument("--image", help="游戏截图路径")
    ap.add_argument("--rect", help="裁剪框 x,y,w,h（截图像素）")
    ap.add_argument("--out", default=None, help="输出路径（默认 templates/ui/<name>）")
    ap.add_argument("--preview", default=None, help="额外输出 4 倍放大预览图")
    args = ap.parse_args()

    if args.list:
        names = PRESET_RECIPES if args.list == "all" else {args.list: PRESET_RECIPES.get(args.list)}
        if names.get(args.list) is None and args.list != "all":
            raise SystemExit(f"未知配方: {args.list}（可用: {', '.join(PRESET_RECIPES)}）")
        for name, steps in names.items():
            if steps is None:
                continue
            print(f"\n{name}:")
            for tpl in required_templates(steps):
                print(f"  - templates/ui/{tpl}")
        print("\n裁剪示例：python tools/crop_template.py --image shot.png "
              "--rect 950,110,120,40 --out templates/ui/sort.png")
        return

    if not args.image or not args.rect:
        raise SystemExit("需要 --image 与 --rect（或用 --list 查看所需模板）")

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"无法读取图片: {args.image}")
    try:
        x, y, w, h = (int(v) for v in args.rect.split(","))
    except ValueError as e:
        raise SystemExit("--rect 格式应为 x,y,w,h") from e

    x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    x, y = max(0, x), max(0, y)
    if x1 - x < 4 or y1 - y < 4:
        raise SystemExit(f"裁剪区域太小/越界: ({x},{y})-({x1},{y1})，图尺寸 {img.shape[1]}x{img.shape[0]}")
    tpl = img[y:y1, x:x1]

    out = Path(args.out) if args.out else ROOT / "templates" / "ui" / "template.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), tpl)
    print(f"已写入 {out}  ({tpl.shape[1]}x{tpl.shape[0]})")

    if args.preview:
        pv = cv2.resize(tpl, (tpl.shape[1] * 4, tpl.shape[0] * 4),
                        interpolation=cv2.INTER_NEAREST)
        Path(args.preview).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.preview, pv)
        print(f"预览 {args.preview}")

    print("\n提示：模板要与运行时**同一分辨率**裁剪；换分辨率/画质后需重裁。")


if __name__ == "__main__":
    main()
