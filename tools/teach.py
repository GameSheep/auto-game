"""吃掉一个游戏的图文教程，吐成程序能执行的东西。

用法：
    py -3.12 tools/teach.py --profile tli_phone960
    py -3.12 tools/teach.py --profile tli_phone960 --dry-run    # 只看不写
    py -3.12 tools/teach.py --profile tli_phone960 --check      # 只核对（CI 用）

默认动作：解析 lessons/*.md → 核对每条 claim 与实测 → 写产物 → 打印报告。
--check 不写任何文件，有「你写的和程序做的不一致」就 exit 1。

三类输出分得很清楚：
  一致（ok）  → 你的标注已经能当回归用例，写进 labels.json 且 strict=true
  缺料（gap） → 缺图/缺框/配置没这个键，进 teach/pending.json 等你补
  不一致（fail）→ 程序现在的行为和你说的相反，这是最有价值的输出（抓到了 bug）
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autogame.config import load_config  # noqa: E402
from autogame.profile import load_profile  # noqa: E402
from autogame.teach import (  # noqa: E402
    KIND_LABEL,
    compile_lessons,
    load_lessons,
    write_game_md,
    write_glossary,
    write_labels,
    write_pending,
    write_report,
)

MARK = {"ok": "✅ 一致", "fail": "❌ 不一致", "gap": "⚠️ 缺料", "skip": "📝 已成文"}


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


def _print_checks(res) -> None:
    if not res.checks:
        print("（没有可核对的条目）")
        return
    w = max(min(28, max(_dw(c["title"]) for c in res.checks)), 8)
    head = f"{_pad('条目', w)}  {'类型':<4} {'结果':<9} 说明"
    print(head)
    print("-" * (w + 40))
    for c in res.checks:
        kind = KIND_LABEL.get(c["kind"], c["kind"])
        print(f"{_pad(_clip(c['title'], w), w)}  {_pad(kind, 4)} "
              f"{_pad(MARK.get(c['status'], c['status']), 9)} "
              f"{_clip(c['detail'], 60)}")
        if c["evidence"]:
            print(f"{'':<{w}}  {'':<4} {'':<9} ⟶ {c['evidence']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="教学编译：图文教程 → 模板/台账/说明")
    ap.add_argument("--profile", required=True, help="profiles/<id>")
    ap.add_argument("--lessons", default="", help="手册目录（默认 profiles/<id>/lessons）")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写任何文件")
    ap.add_argument("--check", action="store_true",
                    help="只核对不写；有 fail 就 exit 1（CI 用）")
    args = ap.parse_args()

    prof = load_profile(args.profile)
    lesson_dir = Path(args.lessons) if args.lessons else prof.root / "lessons"
    if not lesson_dir.exists():
        print(f"没有教学手册目录：{lesson_dir}")
        print(f"新建：mkdir -p {lesson_dir}/img，把教程写成 {lesson_dir}/<游戏>.md")
        print("格式参考 profiles/_template/lessons/example.md")
        return 2

    lessons = load_lessons(lesson_dir)
    if not lessons:
        print(f"{lesson_dir} 里没有 .md 手册（README/index 除外）")
        return 2

    cfg = load_config(ROOT / "config.yaml", profile=prof)

    labels_path = prof.labels_path
    labels_data = {}
    if labels_path.exists():
        try:
            labels_data = json.loads(labels_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            labels_data = {}

    res = compile_lessons(lessons, prof, cfg, labels_data)

    print(f"教学编译 · {prof.id}（{prof.name}）")
    print(f"读到 {len(lessons)} 份手册："
          + "，".join(f"{k} {v} 条" for k, v in res.counts.items()))
    for ls in lessons:
        if ls.title:
            print(f"  · {ls.path.name} — {ls.title}")
    print()

    _print_checks(res)
    print()

    if res.pending:
        print(f"要你动手的（{len(res.pending)} 条）：")
        for i, it in enumerate(res.pending, 1):
            print(f" {i}. [{it['action']}] {it['title']} — {_clip(it['detail'], 70)}")
            if it["evidence"]:
                print(f"      实测：{it['evidence']}")
        print()

    for w in res.warnings:
        print(f"[格式警告] {w}")
    if res.warnings:
        print()

    fails = len(res.failures)
    if args.check:
        ok = sum(1 for c in res.checks if c["status"] == "ok")
        gap = sum(1 for c in res.checks if c["status"] == "gap")
        skip = sum(1 for c in res.checks if c["status"] == "skip")
        line = f"核对 {ok + gap + fails} 条：一致 {ok}，不一致 {fails}，缺料 {gap}"
        if skip:
            line += f"（另有 {skip} 条是文字类，已转成说明，不用像素验）"
        print(line)
        return 1 if fails else 0

    if args.dry_run:
        print("（--dry-run：以下文件未写入）")
        print(f"  {prof.root / 'knowledge' / 'game.md'}（受管块）")
        print(f"  {prof.root / 'knowledge' / 'glossary.json'}（{len(res.glossary)} 条术语）")
        print(f"  {labels_path}（+{len(res.labels_new)} 条）")
        print(f"  {prof.root / 'teach' / 'report.md'}")
        return 0

    write_glossary(res, prof.root / "knowledge")
    write_game_md(res, prof.root / "knowledge")
    n_labels, _ = write_labels(res, labels_path)
    write_report(res, prof.root / "teach", prof.id)
    write_pending(res, prof.root / "teach")

    print("已写入：")
    print(f"  knowledge/game.md        受管块已刷新（块外你写的没动）")
    print(f"  knowledge/glossary.json  {len(res.glossary)} 条术语")
    print(f"  labels.json              +{n_labels} 条"
          + (f"（已存在跳过 {len(res.labels_seen)} 条）" if res.labels_seen else ""))
    print(f"  teach/report.md          本轮报告")
    print(f"  teach/pending.json       {len(res.pending)} 条待办")
    print()
    print(f"核对：`py -3.12 tools/teach.py --profile {prof.id} --check`")
    print(f"回归：`py -3.12 tools/ref_check.py --profile {prof.id}`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
