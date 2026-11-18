"""二次进化的命令行入口。

装到设备上之后，日常只会用到两条：

    py -3.12 tools/evolve.py --profile <id>          # 我加了图和描述，让它学一遍
    py -3.12 tools/evolve.py --profile <id> --rollback   # 学坏了，退回去

其余参数是排障用的：看材料够不够、看这代分多少、看它自己认为缺什么。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autogame import evolve  # noqa: E402
from autogame.config import load_config  # noqa: E402
from autogame.profile import load_profile  # noqa: E402

MARK = {"ok": "OK ", "bad": "缺  ", "warn": "注意"}


def _hr(title: str = "") -> None:
    if title:
        print(f"\n── {title} " + "─" * max(0, 64 - len(title)))
    else:
        print()


def print_audit(audit: evolve.Audit) -> None:
    _hr("材料体检")
    print(f"材料指纹 {audit.digest}　声明 {audit.n_lessons} 条　"
          f"strict 标答 {audit.n_labels} 条")
    for n in audit.needs:
        tag = MARK["ok"] if n.ok else (MARK["bad"] if n.blocking else MARK["warn"])
        warn = "" if n.blocking else "（不拦）"
        print(f"  [{tag}] {n.what:26s} 有 {n.have} / 需 {n.need}{warn}")
        if not n.ok:
            print(f"         为什么：{n.why}")
            print(f"         怎么补：{n.fix}")

    if audit.unused_images:
        print(f"\n  放了但手册没引用的图 {len(audit.unused_images)} 张"
              "（程序不会看这些）：")
        for name in audit.unused_images[:8]:
            print(f"    - {name}")
        if len(audit.unused_images) > 8:
            print(f"    …还有 {len(audit.unused_images) - 8} 张")
        print("    用法：在手册里写 `- 图: refs/<名字>` 并给判定，它才会进教学范围。")

    if audit.duplicates:
        print(f"\n  看起来重复的图 {len(audit.duplicates)} 组"
              "（相关 ≥0.96，多半是同一帧截了两次）：")
        for d in audit.duplicates[:5]:
            print(f"    - {d}")

    _hr()
    print(audit.summary())


def print_gens(prof) -> None:
    gens = evolve.list_generations(prof)
    if not gens:
        print("还没有任何一代。跑一次不带参数的 evolve 就有了。")
        return
    cur = evolve.active_id(prof)
    _hr("代数")
    for g in gens:
        mark = "← 当前" if g.id == cur else ""
        sc = g.score or {}
        frac = (f"{sc.get('strict_pass', '?')}/{sc.get('strict_total', '?')}"
                if sc else "-")
        print(f"  {g.id:9s} {g.status:9s} 台账 {frac:7s} "
              f"规则 {g.rules_source:6s} {mark}")
        if g.note:
            print(f"            {g.note[:100]}")


def main() -> int:
    ap = argparse.ArgumentParser(description="二次进化：吃截图与玩法，长出一代配方")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--config", default="", help="额外叠加的 config")
    ap.add_argument("--audit", action="store_true", help="只做材料体检")
    ap.add_argument("--build", action="store_true", help="只出一代，不跑回归")
    ap.add_argument("--eval", action="store_true", help="只跑回归（不切换 active）")
    ap.add_argument("--gen", default="", help="配合 --eval 指定某代（默认当前 active）")
    ap.add_argument("--promote", default="", help="把某代上线（会先跑回归）")
    ap.add_argument("--rollback", action="store_true", help="退回上一代")
    ap.add_argument("--wanted", action="store_true", help="自我纠正：列出它认为缺什么")
    ap.add_argument("--check-rules", action="store_true", help="只校验 rules.yaml")
    ap.add_argument("--no-model", action="store_true", help="不调模型，只编手册")
    ap.add_argument("--force", action="store_true",
                    help="材料不够或指纹没变也强行出一代（排障用，别在设备上这么干）")
    args = ap.parse_args()

    prof = load_profile(args.profile)
    cfg = load_config(args.config or "config.yaml", profile=prof)

    if args.check_rules:
        # 默认查"当前生效那代"，不是 evolve/ 根目录 —— 根目录下没有 rules.yaml，
        # 默认指错地方会让人以为"我的规则文件丢了"。
        if args.gen:
            target = evolve.gen_dir(prof, args.gen) / "rules.yaml"
        else:
            g = evolve.active_generation(prof)
            if g is None:
                print("还没有 active 代。先跑一次 --build。")
                return 2
            target = g.rules_path
        if not target.exists():
            print(f"没有 {target}")
            return 2
        import yaml
        doc = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        rules, errors = evolve.validate_rules(doc)
        if errors:
            print(f"拒绝：{len(errors)} 处问题")
            for e in errors:
                print(f"  ✗ {e}")
            return 1
        print(f"通过：{len(rules)} 条规则")
        for r in rules:
            print(f"  {r['id']:10s} → {r['then']['action']:16s} "
                  f"{r['params'] if 'params' in r else r['then']['params']}")
        return 0

    if args.rollback:
        ok, msg = evolve.rollback(prof)
        print(msg)
        if ok:
            print_gens(prof)
        return 0 if ok else 1

    if args.wanted:
        doc = evolve.collect_wanted(prof)
        _hr("它认为缺什么")
        if not doc["asks"]:
            print("  暂时没有信号 —— 说明没跑够，或者日志里没有 hold/panic/L2。")
        for a in doc["asks"]:
            print(f"  · {a}")
        if doc["samples"]:
            _hr("原始信号")
            for s in doc["samples"][:8]:
                print(f"  [{s['kind']}] {s['scene'] or '-'}　{s['why']}")
        print(f"\n已写入 {(evolve.evolve_root(prof) / 'wanted.json').as_posix()}")
        return 0

    if args.audit:
        print_audit(evolve.audit_materials(prof))
        print_gens(prof)
        return 0

    if args.eval:
        gen = None
        if args.gen:
            gen = next((g for g in evolve.list_generations(prof) if g.id == args.gen), None)
            if gen is None:
                print(f"没有 {args.gen}")
                return 2
        else:
            gen = evolve.active_generation(prof)
        s = evolve.score_generation(prof, gen)
        _hr(f"回归 {gen.id if gen else '（无 active，用 profile 自带模板）'}")
        print(f"strict 通过 {s['strict_pass']}/{s['strict_total']}"
              f"　比例 {s['ratio']:.2f}")
        for f in s.get("fail") or []:
            print(f"  ✗ {Path(str(f['file'])).name}  {f['scene']}")
            for d in f.get("diffs") or []:
                print(f"      {d}")
        for g in s.get("gap") or []:
            print(f"  ~ {Path(str(g['file'])).name}（非 strict，只提示）")
        return 0 if not s.get("fail") else 1

    if args.build:
        gen, note = evolve.build(prof, cfg, force=args.force,
                                 use_model=not args.no_model)
        print(note)
        if gen:
            print(f"出了 {gen.id}，还没上线。跑 --promote {gen.id} 才生效。")
            print_gens(prof)
        return 0 if gen else 1

    if args.promote:
        gen = next((g for g in evolve.list_generations(prof) if g.id == args.promote), None)
        if gen is None:
            print(f"没有 {args.promote}")
            return 2
        s = evolve.score_generation(prof, gen)
        ok, why = evolve.promote(prof, gen, score=s)
        print(f"{'已上线' if ok else '拒绝上线'}：{why}")
        print_gens(prof)
        return 0 if ok else 1

    # 默认：全流程
    res = evolve.evolve(prof, cfg, force=args.force, use_model=not args.no_model)
    print_audit(evolve.audit_materials(prof))

    _hr("结果")
    if res.get("stage") in ("audit", "build") and not res.get("gen"):
        print(f"没出代：{res['message']}")
        print_gens(prof)
        return 1

    print(f"{res['gen']}　台账 {res['score']['strict_pass']}/"
          f"{res['score']['strict_total']}")
    for f in res.get("fail") or []:
        print(f"  ✗ {Path(str(f.get('file')))}  {f.get('scene')}")
        for d in f.get("diffs") or []:
            print(f"      {d}")
    print(("已上线" if res.get("promoted") else "未上线") + f"：{res['message']}")
    print_gens(prof)
    if not res.get("promoted"):
        print("\n下一步：按上面『怎么补』补齐材料，或者 --rollback 退回上一代。")
    return 0 if res.get("promoted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
