"""把一个新游戏的 inbox 截图变成"可核对的数字"，而不是一段猜测。

这个工具存在的理由：AI 看图只能提出候选，不能宣布结论。历史上"技能栏角落有个 Q"
"红色圆斑是血球""星图节点是光柱"三个判断全错，共同点是都有一个听起来合理的故事，
但没有任何分数支撑。所以这里只做三件事：

  1. 逐张跑生产判据，把门禁分数/反例分数/检出数打成一张表 —— 让"差多少"可见
  2. 把画面里所有"像光柱"的色块连坐标、尺寸、高宽比一起列出来 —— 让"这是不是光柱"
     变成你一眼能裁决的事，而不是我下结论
  3. 输出待办清单 + 现成的 make_template 命令 —— 缺什么补什么，不用你碰代码

用法：
    py -3.12 tools/profile_init.py --profile tli_phone960
    py -3.12 tools/profile_init.py --profile tli_phone960 --emit-labels   # 生成台账骨架

退出码：0 正常（缺标定不算错，会列清单）；2 profile 不存在 / inbox 为空。
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autogame.config import load_config  # noqa: E402
from autogame.profile import Profile, load_profile, list_profiles  # noqa: E402
from autogame.vision.detector import Detector  # noqa: E402

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
TICKS = 2  # 与 ref_check 一致：光柱时域确认需要连续 2 帧


def _iter_images(*dirs: Path) -> list[Path]:
    out: list[Path] = []
    for d in dirs:
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                out.append(p)
    return out


def probe_frame(img: np.ndarray, det: Detector, cfg) -> dict:
    """跑生产判据，返回一行结果。"""
    h, w = img.shape[:2]
    ts = 1000
    st = None
    for _ in range(TICKS):
        st = det.process(img, ts)
        ts += 500
    return {
        "w": w, "h": h,
        "in_game": bool(st.in_game),
        "gate_reason": det.diag.get("gate_reason", ""),
        "gate_best": float(det.diag.get("gate_best", 0.0)),
        "gate_neg": float(diag_neg(det)),
        "pillars": len(st.pillars),
        "labels": len(st.ground_labels),
        "monsters": len(st.monsters),
        "aframe": st.aframe is not None,
    }


def diag_neg(det: Detector) -> float:
    """反例最高分：命中与未命中两种情况下 diag 都会写，取不到按 0。"""
    try:
        return float(det.diag.get("gate_neg_best", 0.0))
    except (TypeError, ValueError):
        return 0.0


def pillar_candidates(img: np.ndarray, cfg, limit: int = 8) -> list[dict]:
    """列出"可能被当成光柱"的色块（不要求门禁通过）。

    输出包含高宽比 —— 光柱是竖直细条（实测 3~8），星图节点/团光 1.4~2.0，
    技能激光可能很长但宽度不稳定。有了这个数字，"这算不算光柱"你一眼就能裁决。
    """
    hsv_ranges = cfg.hsv or {}
    out: list[dict] = []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for color, (lo, hi) in hsv_ranges.items():
        mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w < 2 or h < 12:
                continue
            area = int(cv2.contourArea(c))
            if area < 20:
                continue
            out.append({
                "color": color, "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                "aspect": round(h / float(w), 1), "area": area,
            })
    out.sort(key=lambda d: -d["area"])
    return out[:limit]


def _dw(s: str) -> int:
    """终端显示宽度（中日韩全角算 2）——否则中文文件名会让表格整体错位。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in s)


def _pad(s: str, width: int) -> str:
    return s + " " * max(0, width - _dw(s))


def _clip(s: str, width: int) -> str:
    """按显示宽度截断（不是字符数），保证多字节中文不会把表格挤歪。"""
    if _dw(s) <= width:
        return s
    out, w = "", 0
    for ch in s:
        cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if w + cw > width - 1:
            return out + "…"
        out += ch
        w += cw
    return out


def _fmt_table(rows: list[dict], name_w: int = 30) -> str:
    head = (f"{_pad('文件', name_w)} {'尺寸':>9}  {'门禁':<18} {'正例':>5} {'反例':>5} "
            f"{'光柱':>4} {'标签':>4} {'怪':>3}")
    lines = [head, "-" * (name_w + 54)]
    for r in rows:
        size = f"{r['w']}x{r['h']}"
        lines.append(
            f"{_pad(r['name'], name_w)} {size:>9}  {_pad(_clip(r['gate_reason'], 18), 18)} "
            f"{r['gate_best']:>5.2f} {r['gate_neg']:>5.2f} "
            f"{r['pillars']:>4} {r['labels']:>4} {r['monsters']:>3}"
        )
    return "\n".join(lines)


def build_plan(prof: Profile, rows: list[dict], cands: dict[str, list[dict]],
               cfg) -> list[str]:
    """待办清单：只列"有依据的"缺项，不猜哪张是战斗。"""
    g = cfg.scene_gate
    out: list[str] = []
    tdir = prof.templates_dir
    pos = sorted(tdir.glob("gate_pos_*.png")) if tdir.exists() else []
    neg = sorted(tdir.glob("gate_neg_*.png")) if tdir.exists() else []
    legacy_pos = sorted(tdir.glob("skillbar*.png")) if tdir.exists() else []
    legacy_neg = sorted(tdir.glob("neg_*.png")) if tdir.exists() else []

    # 反例排在最前：门禁是"反例优先"，先做正例会让菜单类界面直接放行
    if not neg and not legacy_neg:
        out.append(
            "[P0] 没有反例模板 → 关卡选择/背包/结算这类界面只能靠「没有 HUD」兜底拦下，\n"
            "     一旦这些界面也带 HUD（暗黑4 开面板时血球技能栏都在），就会点错。\n"
            "     **先做反例再做正例**：门禁是反例优先，反过来做会让菜单直接放行。\n"
            "     每类菜单至少一张：\n"
            "       py -3.12 tools/make_template.py --src <菜单截图> \\\n"
            f"           --rect x0,y0,x1,y1 --out {tdir}/gate_neg_<名字>.png \\\n"
            "           --check <全部菜单截图> --against <全部战斗截图>"
        )
    else:
        weak = [r for r in rows if 0.70 <= r["gate_neg"] < g.negative_threshold]
        if weak:
            names = ", ".join(f"{r['name']}({r['gate_neg']:.2f})" for r in weak[:3])
            out.append(
                f"[P2] 反例分数接近阈值但没到（{names}）。\n"
                "     说明画面里「有点像某张菜单」，余量不足。建议重裁更独特的 UI 部件。"
            )
        if len(neg) + len(legacy_neg) < 2:
            out.append(
                "[P1] 只有 1 类反例。实际容易点错的不止一种界面，通常至少要覆盖\n"
                "     关卡选择 / 背包(角色面板) / 结算 三类——每类截 1~2 张、各做一张反例。"
            )

    if not pos and not legacy_pos:
        out.append(
            "[P0] 没有任何正向锚点模板 → 门禁 fail_closed 会拦下所有画面，生产 0 指令。\n"
            "     锚点要选「位置固定、只在战斗出现、不被技能特效覆盖」的 HUD 部件\n"
            "     （技能轮盘外圈、血球边框、小地图边框）。从投标高分的图里裁一块：\n"
            "       py -3.12 tools/make_template.py --src <战斗截图> \\\n"
            f"           --rect x0,y0,x1,y1 --out {tdir}/gate_pos_<名字>.png \\\n"
            "           --check <全部战斗截图> --against <全部菜单截图>\n"
            "     手机端/右侧技能栏务必带 --roi（默认 ROI 是底部 35%，会裁掉一半）。"
        )
    else:
        low = [r for r in rows if r["gate_best"] < g.threshold]
        if low and len(low) < len(rows):
            names = ", ".join(r["name"] for r in low[:3])
            out.append(
                f"[P1] 有正向模板，但 {names} 的分数 < {g.threshold:.2f}。\n"
                "     要么这几张是另一种界面（应该做反例），要么锚点余量不足需要重裁。"
            )

    # 光柱候选：这是最需要人裁决的一项
    for name, cs in cands.items():
        if not cs:
            continue
        tall = [c for c in cs if c["aspect"] >= 2.5]
        squat = [c for c in cs if c["aspect"] < 2.5]
        seg = [f"[?] {name} 有 {len(cs)} 个疑似光柱色块"]
        for c in cs[:4]:
            tag = "细高(像光柱)" if c["aspect"] >= 2.5 else "矮宽(不像光柱)"
            seg.append(f"       ({c['x']:>4},{c['y']:>4}) {c['w']:>3}x{c['h']:<3} "
                       f"高宽比 {c['aspect']:<4} {c['color']:<5} {tag}")
        if tall and not squat:
            seg.append("     → 全是细高条。如果这张图确实有掉落光柱，当前阈值可以直接用。")
        elif squat and not tall:
            seg.append("     → 全是矮宽块。这张图大概没有光柱（团光/节点/特效）。")
        else:
            seg.append("     → 混着。请指出哪几个是光柱，我据此调 pillar_min_aspect。")
        out.append("\n".join(seg))

    if not out:
        out.append("[OK] 正例/反例/台账都齐了，跑 tools/ref_check.py --profile "
                   f"{prof.id} 做回归。")
    return out


def emit_labels(prof: Profile, rows: list[dict]) -> dict:
    """生成台账骨架：expect 全 null（不猜），等你填答案。"""
    items = []
    for r in rows:
        items.append({
            "file": f"{r.get('src', 'inbox')}/{r['name']}",
            "game": prof.game or prof.name,
            "scene": "TODO 填这是什么界面（战斗/关卡选择/背包/结算…）",
            "strict": False,
            "expect": {
                "in_game": None,
                "pillars": None,
                "aframe": None,
                "ground_labels": None,
                "monsters": None,
                "commands": None,
            },
            "note": (f"TODO 一句话写清楚该判成什么。探针实测：门禁 {r['gate_reason']} "
                     f"正例{r['gate_best']:.2f} 反例{r['gate_neg']:.2f} "
                     f"光柱{r['pillars']} 标签{r['labels']} 怪{r['monsters']}。"
                     f"（这行是实测，不是答案）"),
        })
    return {"version": 1, "updated": "auto", "items": items}


def main() -> int:
    ap = argparse.ArgumentParser(description="新游戏截图探针 + 标定待办清单")
    ap.add_argument("--profile", required=True, help="profile id")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"), help="基础配置文件")
    ap.add_argument("--emit-labels", action="store_true",
                    help="把骨架写到 <profile>/labels.json（已存在则加 .new 后缀，不覆盖）")
    ap.add_argument("--quiet-cand", action="store_true", help="不列光柱候选色块")
    args = ap.parse_args()

    try:
        prof = load_profile(args.profile)
    except FileNotFoundError as e:
        print(f"[错误] {e}")
        print(f"可用 profile: {', '.join(list_profiles()) or '（无）'}")
        return 2

    cfg = load_config(args.config, profile=prof)
    images = _iter_images(prof.inbox_dir, prof.refs_dir)
    if not images:
        print(f"[错误] {prof.inbox_dir} 和 {prof.refs_dir} 里没有图片。")
        print("       把截图丢进 inbox/，每类界面至少 2 张。")
        return 2

    det = Detector(cfg.vision, prof.templates_dir)
    rows: list[dict] = []
    cands: dict[str, list[dict]] = {}
    for p in images:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[跳过] 读不了: {p}")
            continue
        r = probe_frame(img, det, cfg)
        r["name"] = p.name
        r["src"] = "inbox" if p.parent.name == "inbox" else "refs"
        rows.append(r)
        if not args.quiet_cand:
            cands[p.name] = pillar_candidates(img, cfg.vision)

    pos = len(list(prof.templates_dir.glob("gate_pos_*.png"))) if prof.templates_dir.exists() else 0
    neg = len(list(prof.templates_dir.glob("gate_neg_*.png"))) if prof.templates_dir.exists() else 0
    print(f"\nprofile: {prof.id}  ({prof.status})  基准 {prof.reference_resolution[0]}"
          f"x{prof.reference_resolution[1]}  模板目录 {prof.templates_dir}")
    print(f"模板: 正例 {pos} / 反例 {neg}      图片: {len(rows)} 张"
          f"（inbox {len(_iter_images(prof.inbox_dir))} + refs {len(_iter_images(prof.refs_dir))}）\n")
    print(_fmt_table(rows))

    if not args.quiet_cand:
        print("\n---- 光柱候选色块（需要你裁决，我不下结论）----")
        any_c = False
        for name, cs in cands.items():
            if cs:
                any_c = True
                print(f"{name}:")
                for c in cs:
                    tag = "细高" if c["aspect"] >= 2.5 else "矮宽"
                    print(f"   ({c['x']:>4},{c['y']:>4}) {c['w']:>3}x{c['h']:<3} "
                          f"高宽比 {c['aspect']:<4} {c['color']:<5} {tag}")
        if not any_c:
            print("   （没有任何色块命中当前 HSV 阈值——可能是这张图真没光柱，"
                  "也可能是这个游戏的光柱不是金/粉色，需要你说明颜色）")

    print("\n---- 待办 ----")
    for line in build_plan(prof, rows, cands, cfg.vision):
        print(line)

    if args.emit_labels:
        data = emit_labels(prof, rows)
        target = prof.labels_path
        if target.exists():
            target = target.with_name(target.stem + ".new" + target.suffix)
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n台账骨架已写入: {target}")
        print("  填 expect 里的答案（只填你能确定的，其余保持 null）后再跑 ref_check。")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
