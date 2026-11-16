"""从真实帧自动裁剪检测模板（A框 / 再次挑战按钮）。

背景：`templates/aframe.png` 这类标定资产必须从**当前游戏版本、当前分辨率**的实机截图上裁剪，
手工裁容易裁歪、也容易忘记更新。本工具用检测器自身的形状启发式先定位目标，
再按定位框（带少量 padding）从**检测工作分辨率**裁出模板 —— 这样模板与
`vision.max_detect_width` 的缩放比例天然一致，1.0 尺度即可命中。

用法：
  # 单张：自动找 A框 并裁成 templates/aframe.png
  python tools/auto_template.py --image assets/samples/real_combat_hud.png --kind aframe

  # 目录：遍历所有图，取置信度最高的一帧
  python tools/auto_template.py --dir assets/batch --kind aframe

  # 再次挑战按钮
  python tools/auto_template.py --dir assets/batch --kind rechallenge

  # 只预览不落盘（打印候选框）
  python tools/auto_template.py --dir assets/batch --kind aframe --dry-run
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autogame.config import load_config  # noqa: E402
from autogame.vision.detector import Detector  # noqa: E402

SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp"}


def _iter_images(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(p for p in target.iterdir() if p.suffix.lower() in SUPPORTED)


def _work_frame(frame, max_width: int):  # noqa: ANN001
    """缩放到检测工作分辨率（与 Detector 内部一致）。"""
    h, w = frame.shape[:2]
    if max_width and w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
    return frame, scale


def _crop_box_aframe(det: Detector, work, scale: float, pad: int):
    """用启发式定位 A框，返回工作分辨率下的裁剪框。"""
    state = det.process(work, 0)
    af = state.aframe
    if af is None:
        return None
    # AFrame 坐标是原分辨率中心点 → 换算回工作分辨率
    cx, cy = af.x * scale, af.y * scale
    w, h = af.w * scale, af.h * scale
    x0, y0 = int(cx - w / 2) - pad, int(cy - h / 2) - pad
    x1, y1 = int(cx + w / 2) + pad, int(cy + h / 2) + pad
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(work.shape[1], x1), min(work.shape[0], y1)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    return (x0, y0, x1, y1), float(af.score), af


def _crop_box_button(det: Detector, work, scale: float, pad: int):
    """定位『再次挑战』按钮，返回工作分辨率下的裁剪框。"""
    state = det.process(work, 0)
    btn = state.rechallenge
    if btn is None:
        return None
    cx, cy = btn.x * scale, btn.y * scale
    w, h = btn.w * scale, btn.h * scale
    x0, y0 = int(cx - w / 2) - pad, int(cy - h / 2) - pad
    x1, y1 = int(cx + w / 2) + pad, int(cy + h / 2) + pad
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(work.shape[1], x1), min(work.shape[0], y1)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    return (x0, y0, x1, y1), float(btn.score), btn


def _detector_without_template(cfg, kind: str) -> Detector:
    """构造一个禁用模板的检测器（纯启发式定位）。"""
    c2 = copy.deepcopy(cfg.vision)
    if kind == "aframe":
        c2.aframe.template = None
    else:
        c2.rechallenge.template = None
    return Detector(c2)


def main() -> None:
    ap = argparse.ArgumentParser(description="从真实帧自动裁剪检测模板")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="单张图片路径")
    src.add_argument("--dir", help="图片目录（取置信度最高的一帧）")
    ap.add_argument("--kind", choices=["aframe", "rechallenge"], default="aframe")
    ap.add_argument("--rect", default=None,
                    help="手动指定裁剪框 x,y,w,h（原图坐标）；给了就不再自动定位")
    ap.add_argument("--out", default=None, help="输出路径（默认 templates/<kind>.png）")
    ap.add_argument("--pad", type=int, default=2, help="裁剪外扩像素（默认 2）")
    ap.add_argument("--min-score", type=float, default=0.0, help="低于该置信度不采用")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--dry-run", action="store_true", help="只打印候选框，不写文件")
    args = ap.parse_args()

    cfg = load_config(args.config)
    det = _detector_without_template(cfg, args.kind)
    finder = _crop_box_aframe if args.kind == "aframe" else _crop_box_button

    target = Path(args.image or args.dir)
    if not target.exists():
        raise SystemExit(f"路径不存在: {target}")
    images = _iter_images(target)
    if not images:
        raise SystemExit(f"{target} 下没有图片")

    # 手动框：直接按原图坐标裁（自动定位失败时的兜底，最可靠）
    if args.rect:
        try:
            rx, ry, rw, rh = (int(v) for v in args.rect.split(","))
        except ValueError as e:
            raise SystemExit("--rect 格式应为 x,y,w,h（原图坐标）") from e
        frame = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
        if frame is None:
            raise SystemExit(f"无法读取图片: {images[0]}")
        work, scale = _work_frame(frame, cfg.vision.max_detect_width)
        x0, y0 = int(rx * scale), int(ry * scale)
        x1, y1 = int((rx + rw) * scale), int((ry + rh) * scale)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(work.shape[1], x1), min(work.shape[0], y1)
        tpl = work[y0:y1, x0:x1]
        if tpl.size == 0:
            raise SystemExit("--rect 裁剪结果为空，检查坐标是否越界")
        out = Path(args.out) if args.out else ROOT / "templates" / f"{args.kind}.png"
        print(f"手动框裁剪: {tpl.shape[1]}x{tpl.shape[0]} → {out}")
        if not args.dry_run:
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), tpl)
            print("已写入")
        return


    best = None  # (score, path, box, work)
    for p in images:
        frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        work, scale = _work_frame(frame, cfg.vision.max_detect_width)
        found = finder(det, work, scale, args.pad)
        if found is None:
            continue
        box, score, _obj = found
        if score < args.min_score:
            continue
        print(f"候选 {p.name}: box={box} score={score:.3f}")
        if best is None or score > best[0]:
            best = (score, p, box, work)

    if best is None:
        raise SystemExit(
            f"没有在 {len(images)} 张图中找到 {args.kind} 候选；"
            "请换一帧更清晰的目标截图（A框需空心亮框、按钮需完整可见）"
        )

    score, path, (x0, y0, x1, y1), work = best
    tpl = work[y0:y1, x0:x1]
    out = Path(args.out) if args.out else ROOT / "templates" / f"{args.kind}.png"
    print(f"\n最佳帧: {path.name}  置信度={score:.3f}  模板尺寸={tpl.shape[1]}x{tpl.shape[0]}")

    if args.dry_run:
        print("[dry-run] 未写入文件")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), tpl)
    print(f"已写入: {out}")

    # 自检：用新模板重新检测同一帧，确认 1.0 尺度能命中
    verify_cfg = copy.deepcopy(cfg.vision)
    if args.kind == "aframe":
        verify_cfg.aframe.template = str(out)
        verify_cfg.aframe.template_threshold = 0.78
    else:
        verify_cfg.rechallenge.template = str(out)
    det2 = Detector(verify_cfg, templates_dir=out.parent)
    state = det2.process(work, 0)
    hit = state.aframe if args.kind == "aframe" else state.rechallenge
    if hit is None:
        print("⚠ 自检：新模板在同一帧上未命中（阈值过高或裁剪偏移），请人工确认")
    else:
        print(f"✓ 自检：新模板命中 ({hit.x},{hit.y}) w={hit.w} h={hit.h} score={hit.score}")
        print("\n在 config.yaml 中启用：")
        if args.kind == "aframe":
            print("  vision:\n    aframe:\n      template: templates/aframe.png")
        else:
            print("  vision:\n    rechallenge:\n      template: templates/rechallenge.png")


if __name__ == "__main__":
    main()
