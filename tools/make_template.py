"""从一张截图里裁出场景门模板（正向技能栏 / 反例菜单），并按生产参数验证。

为什么需要这个工具：场景门只有两种可靠的判据 ——
  1. **正向**：技能栏/血球这类"只有战斗才出现"的 HUD 部件（模板名 skillbar*.png）
  2. **反例**：关卡选择/背包/结算这类整屏 UI 的静态 UI 元素（模板名 neg_*.png）
两者都是**每款游戏、每个分辨率单独标一次**的资产，不存在跨游戏通用模板
（真实录像验证：暗黑4 的技能栏模板在火炬之光手机端只有 0.25 分，
  反过来任何"通用形状判据"都会在技能特效上误报）。

用法：
    # 裁反例（整屏菜单左上角的返回箭头+标题）
    python tools/make_template.py --src menus/map.jpg --rect 8,2,200,52 \\
        --out templates/neg_mygame_map.png --check other_menus/*.jpg

    # 裁正向（战斗界面底部的技能栏/技能轮盘）
    python tools/make_template.py --src shots/combat.jpg --rect 690,190,960,410 \\
        --out templates/skillbar_mygame.png --check shots/combat2.jpg

裁完必须看它给出的分数：正向模板要在**所有**战斗截图上 ≥0.6，
反例模板要在**所有**菜单截图上 ≥0.8、在战斗截图上明显更低。做不到就换个区域。
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
from autogame.vision.detector import ROOT_TEMPLATES, _tpl_usable  # noqa: E402

BASE_WIDTH = 1280  # 模板统一按 1280 宽基准存储，运行时按画面宽度比例缩放


def crop_normalized(src: np.ndarray, rect: list[int]) -> np.ndarray:
    x0, y0, x1, y1 = rect
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(src.shape[1], x1), min(src.shape[0], y1)
    if x1 - x0 < 6 or y1 - y0 < 6:
        raise SystemExit(f"裁剪区域太小: {rect}")
    crop = src[y0:y1, x0:x1]
    factor = BASE_WIDTH / src.shape[1]
    return cv2.resize(crop, (max(6, int(round(crop.shape[1] * factor))),
                             max(6, int(round(crop.shape[0] * factor)))),
                      interpolation=cv2.INTER_AREA)


def score(img: np.ndarray, tpl: np.ndarray, whole_frame: bool) -> float:
    """按生产参数打分：门禁 ROI 内匹配（正向）或整帧匹配（反例）。"""
    h, w = img.shape[:2]
    roi = img if whole_frame else img[int(h * 0.65):, :]
    roi = cv2.resize(roi, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    norm = (w / BASE_WIDTH) * 0.5
    tw = max(2, int(tpl.shape[1] * norm))
    th = max(2, int(tpl.shape[0] * norm))
    if th >= roi.shape[0] or tw >= roi.shape[1]:
        return -1.0
    t = cv2.resize(tpl, (tw, th), interpolation=cv2.INTER_AREA)
    res = cv2.matchTemplate(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY),
                            cv2.cvtColor(t, cv2.COLOR_BGR2GRAY),
                            cv2.TM_CCOEFF_NORMED)
    return round(float(cv2.minMaxLoc(res)[1]), 3)


def main() -> int:
    ap = argparse.ArgumentParser(description="裁场景门模板并验证")
    ap.add_argument("--src", required=True, help="源截图")
    ap.add_argument("--rect", required=True, help="裁剪框 x0,y0,x1,y1（源图像素）")
    ap.add_argument("--out", required=True,
                    help="输出路径：skillbar*.png=正向，neg_*.png=反例")
    ap.add_argument("--check", nargs="*", default=[],
                    help="验证图（可多张/通配符）：应命中的场景")
    ap.add_argument("--against", nargs="*", default=[],
                    help="反例图（可多张/通配符）：不应命中的场景")
    args = ap.parse_args()

    src = cv2.imread(args.src)
    if src is None:
        raise SystemExit(f"读不到源图: {args.src}")
    rect = [int(v) for v in args.rect.split(",")]
    tpl = crop_normalized(src, rect)
    out = Path(args.out)
    name = out.name
    is_negative = name.startswith("neg_")
    if not (is_negative or name.startswith("skillbar")):
        raise SystemExit("输出名必须以 neg_ 或 skillbar 开头，否则检测器不会加载")

    if not _tpl_usable(tpl):
        raise SystemExit("裁出来的模板方差过低（几乎纯色），NCC 匹配无意义，换个区域")

    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), tpl)
    print(f"已保存 {out}  尺寸 {tpl.shape[1]}x{tpl.shape[0]}"
          f"  ({'反例' if is_negative else '正向'}模板, 1280 宽基准)")

    cfg = load_config(str(ROOT / "config.yaml")).vision
    neg_thr = cfg.scene_gate.negative_threshold
    pos_thr = cfg.scene_gate.threshold
    self_score = score(src, tpl, is_negative)
    print(f"  源图自身: {self_score:.3f}")

    def report(paths: list[str], want_hit: bool) -> None:
        files: list[Path] = []
        for pattern in paths:
            p = Path(pattern)
            files.extend(sorted(p.parent.glob(p.name)) if any(c in pattern for c in "*?") else [p])
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                print(f"  ! 读不到 {f}")
                continue
            s = score(img, tpl, is_negative)
            thr = neg_thr if is_negative else pos_thr
            ok = (s >= thr) if want_hit else (s < thr)
            flag = "OK  " if ok else "FAIL"
            label = "应命中" if want_hit else "不应命中"
            print(f"  [{flag}] {label} {f.name:38s} {s:.3f}  阈值 {thr:.2f}")

    if args.check:
        print("验证（应命中）：")
        report(args.check, True)
    if args.against:
        print("验证（不应命中）：")
        report(args.against, False)

    print(f"\n门禁行为：命中则判{'非战斗（拦下，零命令）' if is_negative else '战斗（放行，可操作）'}。")
    print("若分数不够，换一块更静态、更有辨识度的 UI（标题栏/固定边框比图标稳）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
