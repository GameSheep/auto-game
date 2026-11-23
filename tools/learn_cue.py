"""按截图标定判据：把"画面里哪个位置、什么颜色、多少算成立"写成 cues.yaml。

判据是这一层唯一没法靠想的东西 —— 蓝条在第几行、光柱是什么色、宝箱提示长什么样，
只能看截图。所以这个工具只做三件事：

    1. 看：给一张截图 + 一个区域，告诉你那里主要是什么颜色、读出来是多少
    2. 定：给"成立"和"不成立"两张截图，自动算出中间那条阈值
    3. 写：追加/更新 profiles/<id>/agent/cues.yaml

用法（先看图，再定阈值，最后写）：

    # 1) 不知道颜色：把区域里的主要颜色列出来
    py tools/learn_cue.py --profile tli_phone960 --image shot.png --region 0.03,0.88,0.19,0.906

    # 2) 知道颜色了：算读数（蓝满 / 蓝空各一张）
    py tools/learn_cue.py --profile tli_phone960 --image full.png \\
        --region 0.03,0.88,0.19,0.906 --color "#2a6bd4"
    py tools/learn_cue.py --profile tli_phone960 --image empty.png \\
        --region 0.03,0.88,0.19,0.906 --color "#2a6bd4"

    # 3) 两张一起给，自动取中间值并写进去
    py tools/learn_cue.py --profile tli_phone960 --id mana_ok --name "蓝量够" \\
        --region 0.03,0.88,0.19,0.906 --color "#2a6bd4" \\
        --on-img full.png --off-img empty.png --write

    # 4) 模板类（图标/按钮/提示）：直接从截图里裁一块存成模板
    py tools/learn_cue.py --profile tli_phone960 --id chest_hint --kind template \\
        --region 0.35,0.55,0.65,0.80 --crop 520,430,620,470 --write

为什么阈值不让人自己填：手填的数字没人验证得动。给两张"成立/不成立"的截图，
中间值就是唯一能自证的数字 —— 而且以后误判了，回头看也知道当时按哪两张定的。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autogame.agent.cues import Cue, CueBook, parse_color  # noqa: E402


def _load_image(path: str):
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"读不出这张图：{path}")
    return img


def _region(value: str) -> list[float]:
    parts = [p.strip() for p in str(value).replace("，", ",").split(",") if p.strip()]
    if len(parts) != 4:
        raise SystemExit("--region 要 4 个数：x1,y1,x2,y2（归一化 0..1，从截图左上角算）")
    return [float(p) for p in parts]


def _cues_path(profile: str, explicit: str) -> Path:
    if explicit:
        return Path(explicit)
    if not profile:
        raise SystemExit("给 --profile 或 --cues")
    return Path("profiles") / profile / "agent" / "cues.yaml"


def _dominant(frame: np.ndarray, region: list[float], top: int = 4) -> list[tuple[str, float]]:
    """区域里的主要颜色（按 32 级量化后统计）。给人选 `--color` 用。"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(region[0] * w), int(region[1] * h),
                     int(region[2] * w), int(region[3] * h))
    roi = frame[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]
    if roi.size == 0:
        return []
    q = (roi.reshape(-1, 3) // 32) * 32 + 16
    colors, counts = np.unique(q, axis=0, return_counts=True)
    order = np.argsort(-counts)[:top]
    total = roi.reshape(-1, 3).shape[0]
    out = []
    for i in order:
        b, g, r = (int(c) for c in colors[i])
        out.append((f"#{r:02x}{g:02x}{b:02x}", float(counts[i]) / total))
    return out


def _read(cue: Cue, frame: np.ndarray, base: Path) -> float:
    return CueBook(cues=[cue]).evaluate(frame, base_dir=base)[cue.id]


def _auto_threshold(on_vals: list[float], off_vals: list[float],
                    direction: str) -> tuple[float | None, str]:
    """由"成立/不成立"两组读数算出阈值，返回 (阈值, 说明)。"""
    if on_vals and off_vals:
        lo_off, hi_on = max(off_vals), min(on_vals)
        if direction == "above":
            if hi_on <= lo_off:
                return None, (f"两组读数重叠（成立最低 {hi_on:.3f}，不成立最高 {lo_off:.3f}）"
                              "—— 区域或颜色选得不对，换一块再试")
            return round((lo_off + hi_on) / 2, 3), f"取中间值（不成立最高 {lo_off:.3f} / 成立最低 {hi_on:.3f}）"
        lo_on, hi_off = min(on_vals), max(off_vals)
        if lo_on >= hi_off:
            return None, (f"两组读数重叠（成立最高 {lo_on:.3f}，不成立最低 {hi_off:.3f}）"
                          "—— 区域或颜色选得不对")
        return round((lo_on + hi_off) / 2, 3), f"取中间值（成立最高 {lo_on:.3f} / 不成立最低 {hi_off:.3f}）"
    # 只有一组：按倍率推（说明里讲清楚这是估算，以后最好补另一组）
    vals = on_vals or off_vals
    if not vals:
        return None, "没有样本"
    v = vals[0]
    if direction == "above":
        t = v * 0.6 if on_vals else max(0.02, v * 1.5)
    else:
        t = v * 1.6 if on_vals else max(0.02, v * 0.5)
    return round(min(1.0, max(0.0, t)), 3), f"只有一组样本，按读数 {v:.3f} 估的 —— 建议再补一张反向截图"


def _write(path: Path, cue: Cue) -> None:
    data = {}
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        data = loaded if isinstance(loaded, dict) else {"cues": loaded}
    cues = [c for c in (data.get("cues") or []) if c.get("id") != cue.id]
    cues.append(cue.model_dump(exclude_defaults=True, exclude_none=True))
    data["cues"] = cues
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                    encoding="utf-8")
    print(f"\n已写入 {path}（{len(cues)} 条判据）")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="按截图标定判据（cues.yaml）")
    ap.add_argument("--profile", default="", help="哪个游戏")
    ap.add_argument("--cues", default="", help="cues.yaml 路径（默认 profiles/<id>/agent/cues.yaml）")
    ap.add_argument("--id", default="", help="判据名（技能里用 when: cue + cue: <这个名字>）")
    ap.add_argument("--name", default="", help="人读的名字")
    ap.add_argument("--kind", default="ratio", choices=["ratio", "bright", "template"])
    ap.add_argument("--region", default="", help="归一化区域 x1,y1,x2,y2")
    ap.add_argument("--color", default="", help="#rrggbb 或 white/red/...")
    ap.add_argument("--tol", type=int, default=40, help="颜色容差（默认 40）")
    ap.add_argument("--image", action="append", default=[], help="只看读数（可重复）")
    ap.add_argument("--on-img", action="append", default=[], help="代表「成立」的截图（可重复）")
    ap.add_argument("--off-img", action="append", default=[], help="代表「不成立」的截图（可重复）")
    ap.add_argument("--above", type=float, default=None, help="手动指定下限（覆盖自动）")
    ap.add_argument("--below", type=float, default=None, help="手动指定上限")
    ap.add_argument("--crop", default="", help="kind=template 时：像素区域 x1,y1,x2,y2，从截图裁模板")
    ap.add_argument("--template", default="", help="kind=template 时：模板文件（相对 cues.yaml）")
    ap.add_argument("--list", action="store_true", help="列出现有判据")
    ap.add_argument("--write", action="store_true", help="写进 cues.yaml")
    args = ap.parse_args(argv)

    path = _cues_path(args.profile, args.cues)
    base = path.parent
    book = CueBook.load(path) if path.exists() else CueBook()

    if args.list:
        if not book:
            print(f"还没有判据：{path}")
            return 0
        print(f"{path}（{len(book)} 条）")
        for c in book.cues:
            print(f"  {c.describe()}")
            if c.note:
                print(f"      {c.note}")
        for p in book.problems(base):
            print(f"  [错误] {p['id']}: {p['message']}")
        return 0

    if not args.region:
        raise SystemExit("给 --region（归一化 x1,y1,x2,y2）：先把截图放进来看一眼")

    region = _region(args.region)

    # 只看读数：列出主色 + 可选颜色的占比
    if args.image and not args.id:
        for img_path in args.image:
            frame = _load_image(img_path)
            print(f"\n{img_path}  区域 {region}")
            print("  主要颜色：")
            for hex_code, share in _dominant(frame, region):
                print(f"    {hex_code}  {share * 100:5.1f}%")
            if args.color:
                cue = Cue(id="_probe", kind=args.kind, region=region,
                          color=args.color, tol=args.tol, above=0.0)
                print(f"  {args.color} 占比：{_read(cue, frame, base):.3f}")
        print("\n把选中的颜色用 --color 再跑一次，或用 --id 直接标一条判据。")
        return 0

    if not args.id:
        raise SystemExit("给 --id（判据名），或只用 --image 看读数")

    # 模板类：从截图裁一块存成模板文件
    template_rel = args.template
    if args.kind == "template" and args.crop:
        import cv2

        if not args.on_img:
            raise SystemExit("裁模板需要一张截图：--on-img <图> --crop x1,y1,x2,y2")
        frame = _load_image(args.on_img[0])
        cx1, cy1, cx2, cy2 = (int(v) for v in _region(args.crop))
        tpl = frame[cy1:cy2, cx1:cx2]
        if tpl.size == 0:
            raise SystemExit("--crop 是像素坐标，且要在图里面")
        template_rel = template_rel or f"tpl_{args.id}.png"
        out = base / template_rel
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), tpl)
        print(f"模板已存：{out}（{tpl.shape[1]}x{tpl.shape[0]}）")

    cue = Cue(id=args.id, name=args.name, kind=args.kind, region=region,
              color=args.color or "#ffffff", tol=args.tol,
              template=template_rel, above=args.above, below=args.below)

    # 两组样本 → 自动阈值
    on_vals = [_read(cue, _load_image(p), base) for p in args.on_img]
    off_vals = [_read(cue, _load_image(p), base) for p in args.off_img]
    if on_vals:
        print(f"成立样本读数：{', '.join(f'{v:.3f}' for v in on_vals)}")
    if off_vals:
        print(f"不成立样本读数：{', '.join(f'{v:.3f}' for v in off_vals)}")

    if args.above is None and args.below is None:
        direction = "below" if args.kind == "bright" and on_vals and np.mean(on_vals) < 0.5 else "above"
        if args.kind == "template":
            t, why = (max(0.55, min(0.9, min(on_vals) * 0.9)) if on_vals else 0.72), "模板分数留一成余量"
        else:
            t, why = _auto_threshold(on_vals, off_vals, direction)
        if t is None:
            print(f"\n算不出阈值：{why}")
            return 2
        print(f"建议阈值：{direction} {t}（{why}）")
        cue = cue.model_copy(update={direction: t})

    print(f"\n判据：{cue.describe()}")
    problems = cue.problem(base)
    for p in problems:
        print(f"  [错误] {p}")

    if problems:
        return 2
    if args.write:
        _write(path, cue)
        print("下一步：在 skills.yaml 的步骤上写 when: cue + cue: " + cue.id)
    else:
        print("\n（没写文件。确认没问题后加 --write）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
