"""文档 → 模板，一条命令。

用法：
    py -3.12 tools/teach_crop.py --profile tli_phone960
    py -3.12 tools/teach_crop.py --profile tli_phone960 --only 星图
    py -3.12 tools/teach_crop.py --profile tli_phone960 --dry-run     # 只看不写
    py -3.12 tools/teach_crop.py --profile tli_phone960 --write-back  # 顺手填回手删

它干什么：
    读 lessons/*.md 里每个「界面」「物体」段 → 按你写的方位/形状在截图里找框 →
    跨图验证（战斗图该命中、菜单图不该命中）→ 裁成 gate_pos_*/gate_neg_* →
    出一张 preview.html 让人过目。

它不干什么：
    不调大模型。理由写在 autogame/teach_crop.py 开头：人裁锚点靠四件可验证的
    事（位置/形状/唯一性/跨图对账），四件全是像素能算的。加模型只是把工作量
    从"量坐标"搬到"审模型"。

你只需要写：
    - 图: ../refs/xxx.jpg
    - 判定: 正例 / 反例
    - 在: 右下角技能轮盘        ← 这一行是新的，写方位就行，不用坐标
    （想给模板命名就加 `- 模板名: skillwheel`，不写就自动编号）
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autogame.config import load_config  # noqa: E402
from autogame.profile import load_profile  # noqa: E402
from autogame.teach import load_lessons  # noqa: E402
from autogame.teach_crop import ROLE_LABEL, _parse_set, roi_advice, run  # noqa: E402


def _dw(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in s)


def _pad(s: str, width: int) -> str:
    return s + " " * max(0, width - _dw(s))


def _clip(s: str, width: int) -> str:
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


def main() -> int:
    ap = argparse.ArgumentParser(description="把教学手册里的描述编译成门禁模板")
    ap.add_argument("--profile", required=True, help="profiles/<id>")
    ap.add_argument("--only", default="", help="只做标题包含该串的条目")
    ap.add_argument("--dry-run", action="store_true", help="只算不写（不落模板、不出页面）")
    ap.add_argument("--write-back", action="store_true",
                    help="把量到的框写回手册里空着/待补充的位置（人写过的框不动）")
    ap.add_argument("--config", default="", help="额外叠加的 config（默认 config.yaml）")
    ap.add_argument("--set", action="append", default=[], metavar="标题=x,y,w,h",
                    help="人直接指定某条的框（可多次）。会写进 crops.json 并标 locked，"
                         "以后重跑不再自动搜。")
    args = ap.parse_args()

    prof = load_profile(args.profile)
    cfg = load_config(str(ROOT / "config.yaml") if not args.config else args.config,
                      profile=prof)
    lessons = load_lessons(prof.root / "lessons")
    if not lessons:
        print(f"没有教学手册：{prof.root / 'lessons'}/*.md")
        return 1

    report = run(prof, cfg, lessons, only=args.only,
                 write=not args.dry_run, do_write_back=args.write_back,
                 forced=_parse_set(args.set))

    print(f"\n模板编译 · {prof.id}　（{report.elapsed:.1f}s"
          f"{'　DRAW-RUN 未写文件' if args.dry_run else ''}）\n")

    if not report.results:
        print("手册里没有可处理的「界面」或「物体」段。")
        return 0

    w = max(min(30, max(_dw(r.intent.title) for r in report.results)), 10)
    print(f"{_pad('界面', w)}  {_pad('角色', 10)} {_pad('结果', 10)} "
          f"{_pad('模板', 24)} 依据")
    print("-" * (w + 66))
    for r in report.results:
        role = _pad(ROLE_LABEL.get(r.intent.role, r.intent.role), 10)
        if r.best is None:
            print(f"{_pad(_clip(r.intent.title, w), w)}  {role} "
                  f"{_pad('✗ 没做出来', 10)} {_pad('—', 24)} {_clip(r.reason, 52)}")
            continue
        if not r.verified:
            st = "⚠️ 样本不足"      # 只有一侧样本时，分数再漂亮也只是它自己在自夸
        elif r.best.ev.get("ok"):
            st = "✅ 可用"
        else:
            st = "❌ 不达标"
        tpl = report.templates.get(r.intent.id, "")
        print(f"{_pad(_clip(r.intent.title, w), w)}  {role} "
              f"{_pad(st, 10)} {_pad(_clip(tpl, 22), 24)} {_clip(r.reason, 52)}")

    if report.roi_suggest:
        cur = report.roi_current
        print(f"建议匹配 ROI：{report.roi_suggest}"
              f"　（现在 profile.yaml 写的是 {cur or '（没写）'}）")
        print(f"　→ {roi_advice(report.roi_suggest, cur)}")

    if report.warnings:
        print()
        for wmsg in report.warnings:
            print(f"! {wmsg}")

    out_dir = prof.root / "teach" / "crops"
    if not args.dry_run:
        print(f"\n预览（建议先看这个）：{out_dir / 'preview.html'}")
        print(f"清单（可手改 rect + locked）：{out_dir / 'crops.json'}")
        if report.written:
            print(f"落盘 {len(report.written)} 项：{', '.join(report.written)}")
        print(f"\n下一步：py -3.12 tools/teach.py --profile {prof.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
