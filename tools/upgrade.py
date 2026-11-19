"""设备侧升级入口 —— 一条命令换引擎，换不过去自动留在原地。

    python tools/upgrade.py                      # 我现在是什么版本、装过什么
    python tools/upgrade.py verify               # 引擎文件有没有被人手改过
    python tools/upgrade.py install pkg.tar.gz   # 装（先自检，过了才切指针）
    python tools/upgrade.py install pkg.tar.gz --dry-run   # 只校验，不动手
    python tools/upgrade.py rollback             # 退回上一版
    python tools/upgrade.py snapshots            # 升级前的快照
    python tools/upgrade.py snapshots --restore <名字>

## 为什么装完还提示"重启"

Python 进程的代码在启动时就加载了，换目录不等于换正在跑的东西。
服务由 systemd 托管（`Restart=always`），所以正确做法是：

    sudo systemctl restart autogame

页面上的「引擎版本」区会显示"重启后生效"，就是这个原因 ——
不骗人说已经换了。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autogame import upgrade as up  # noqa: E402


def _print_steps(steps: list[dict]) -> None:
    for s in steps:
        mark = "OK  " if s.get("ok") else "失败"
        detail = s.get("detail") or ""
        print(f"  [{mark}] {s['step']}  {detail}")


def _status(iroot: Path | None = None, droot: Path | None = None) -> int:
    st = up.status(iroot, droot, deep=True)
    print(f"引擎版本   {st['engine_version'] or '（没装 release，当前是目录直跑）'}")
    print(f"上一版     {st['previous'] or '—'}")
    print(f"安装根     {st['install_root']}")
    print(f"数据根     {st['data_root']}   （schema {st['data_schema']}）")
    v = st.get("verify") or {}
    if v:
        print("完整性     " + ("全部文件与清单一致"
                               if v.get("ok") else
                               f"有 {len(v.get('modified', []))} 个文件被改过、"
                               f"{len(v.get('missing', []))} 个缺失（升级会覆盖这些改动）"))
    if st["releases"]:
        print("\n已装的版本")
        for r in st["releases"]:
            tag = " ← 在用" if r["version"] == st["engine_version"] else ""
            when = r.get("built_at") or "?"
            rev = r.get("git_rev") or "?"
            print(f"  {r['version']:<10} 构建 {when}  git {rev}{tag}"
                  + ("  [目录不完整]" if r.get("broken") else ""))
    if st["history"]:
        print("\n最近升级")
        for h in st["history"][-5:]:
            import time as _t
            ts = _t.strftime("%m-%d %H:%M", _t.localtime(h.get("ts", 0)))
            print(f"  {ts}  {h.get('action')}  {h.get('from') or '—'} → {h.get('to')}  "
                  + ("成功" if h.get("ok") else "失败"))
    if st["snapshots"]:
        print("\n快照  " + "、".join(s["name"] for s in st["snapshots"][:4]))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="autogame 引擎升级（换目录，不就地改）")
    ap.add_argument("cmd", nargs="?", default="status",
                    choices=["status", "verify", "install", "rollback", "snapshots"])
    ap.add_argument("arg", nargs="?", default="", help="install 的包路径 / 快照名")
    ap.add_argument("--install-root", default="")
    ap.add_argument("--data-root", default="")
    ap.add_argument("--force", action="store_true", help="允许降级/覆盖已装版本")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--with-deps", action="store_true", help="解包后先在新 venv 里装依赖")
    ap.add_argument("--no-deps", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--profile", default="")
    ap.add_argument("--keep", type=int, default=up._KEEP_DEFAULT)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    iroot = Path(a.install_root) if a.install_root else None
    droot = Path(a.data_root) if a.data_root else None

    if a.cmd == "status":
        if a.json:
            print(json.dumps(up.status(iroot, droot, deep=True), ensure_ascii=False, indent=2))
            return 0
        return _status(iroot, droot)

    if a.cmd == "verify":
        st = up.status(iroot, droot, deep=True)
        v = st.get("verify") or {}
        if a.json:
            print(json.dumps(v, ensure_ascii=False, indent=2))
        else:
            print(("一致" if v.get("ok") else "不一致") + f"  清单 {v.get('count', 0)} 个文件")
            for name in (v.get("modified") or [])[:10]:
                print("  被改过: " + name)
            for name in (v.get("missing") or [])[:10]:
                print("  缺失:   " + name)
            for name in (v.get("extra") or [])[:10]:
                print("  多出来: " + name)
        return 0 if v.get("ok") else 1

    if a.cmd == "install":
        if not a.arg:
            print("要装哪个包？  python tools/upgrade.py install dist/autogame-0.2.1.tar.gz")
            return 2
        print(f"装 {a.arg}" + ("（演练）" if a.dry_run else ""))
        res = up.install(a.arg, install_root=iroot, data_root=droot, force=a.force,
                         dry_run=a.dry_run, with_deps=a.with_deps,
                         profile=a.profile, keep=a.keep)
        _print_steps(res.get("steps") or [])
        doc = res.get("doctor") or {}
        for c in doc.get("checks", []):
            if c.get("level") != "ok":
                print(f"  · {c['level']}: {c['name']} — {c['detail']}")
        if a.json:
            print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
        if not res.get("ok"):
            print("\n没装成：" + str(res.get("error")))
            print("现场没动：current 指针还是原来那个版本。")
            return 1
        print(f"\n已切到 {res['version']}。")
        if res.get("restart_required"):
            print("重启服务后生效：sudo systemctl restart autogame")
        return 0

    if a.cmd == "rollback":
        res = up.rollback(install_root=iroot, data_root=droot, to=a.arg, force=a.force)
        if a.json:
            print(json.dumps(res, ensure_ascii=False, indent=2))
        if not res.get("ok"):
            print("退不了：" + str(res.get("error")))
            if res.get("hint"):
                print(res["hint"])
            return 1
        print(f"已退回 {res['version']}。重启服务后生效：sudo systemctl restart autogame")
        return 0

    # snapshots
    if a.arg:
        res = up.restore_snapshot(droot or up.paths.data_root(), a.arg)
        print(f"已还原 {res['restored']} 个文件；还原前的状态快照在 {res['pre_snapshot']}")
        return 0
    snaps = up.list_snapshots(droot or up.paths.data_root())
    if not snaps:
        print("还没有快照（每次 install/rollback 前会自动拍一张）")
    for s in snaps:
        print(f"  {s['name']}  {s['files']} 个文件")
    return 0


if __name__ == "__main__":
    sys.exit(main())
