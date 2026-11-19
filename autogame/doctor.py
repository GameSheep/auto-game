"""上线前自检 —— 新引擎换上去之前，先在**真实数据**上证明它能跑。

## 谁在什么时机跑它

`upgrade install` 把新包解到 `releases/.staging-<ver>/` 后，用新引擎自己的
doctor 跑一遍，跑过才切 `current`。所以自检代码必须**随新引擎一起发**
（旧引擎不认识新引擎的毛病），这也是它放在包内、而不是升级工具里的原因。

## 拦什么、不拦什么

只拦"这台机器上跑不起来"：

    数据根可写          只读根 = 一启动就 EROFS
    配置能解析          pydantic 不认的字段会让服务起不来
    profile 能加载      profile.yaml 坏掉 = 门禁用不了
    模板能解码          cv2.imread 返回 None = 门禁永远 fail_closed
    依赖在（cv2/numpy） 缺库直接崩
    磁盘还有空间        升级过程中写满 = 半截数据

**不拦"数据有毛病"**（规则写错、标答缺字段）：那些是内容问题，
新引擎恰恰可能是来修它的；拦下来只会让人卡在旧版。它们以 warn 出现，
会一路带到页面上让人看见。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

_MIN_FREE_MB = 200

Check = dict[str, Any]


def _ok(name: str, detail: str = "") -> Check:
    return {"name": name, "level": "ok", "detail": detail}


def _warn(name: str, detail: str) -> Check:
    return {"name": name, "level": "warn", "detail": detail}


def _fail(name: str, detail: str) -> Check:
    return {"name": name, "level": "fail", "detail": detail}


def _check_paths() -> Check:
    from . import paths
    from .version import DATA_SCHEMA, SCHEMA_FILE

    root = paths.data_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".doctor-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as e:
        return _fail("数据根可写", f"{root}: {e}")
    marker = root / SCHEMA_FILE
    detail = f"{root}（schema {marker.read_text().strip() if marker.exists() else '未标记 → ' + str(DATA_SCHEMA)}）"
    return _ok("数据根可写", detail)


def _check_deps() -> Check:
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        import yaml  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return _fail("依赖齐全", f"{type(e).__name__}: {e}")
    return _ok("依赖齐全", f"cv2 {cv2.__version__}")


def _config_path():
    from . import paths

    for p in (paths.data("config.yaml"), paths.code("config.yaml")):
        if p.exists():
            return p
    return None


def _check_config(profile: str) -> tuple[Check, Any]:
    """按运行时同样的方式加载配置 —— 必须带 profile。

    不带 profile 加载出来的那份，`rules_path`/`knowledge_dir` 都是空的，
    于是"配方兼容"永远报"还没进化过"，自检就变成了走过场。
    """
    from .config import load_config

    path = _config_path()
    try:
        cfg = load_config(str(path) if path else None, profile=profile or None)
    except Exception as e:  # noqa: BLE001
        return _fail("配置能解析", f"{path or '(默认值)'}: {type(e).__name__}: {e}"), None
    detail = str(path) if path else "全部使用默认值"
    return _ok("配置能解析", detail + (f"（profile {profile}）" if profile else "")), cfg


def _resolve_profile_id(wanted: str) -> tuple[str, str]:
    """定下要检查哪个 profile：命令行 → 数据根指针 → 唯一的那一个。"""
    from .profile import active_profile_id, list_profiles

    if wanted:
        return wanted, ""
    name = active_profile_id()
    if name:
        return name, ""
    avail = list_profiles()
    if len(avail) == 1:
        return avail[0], ""
    if not avail:
        return "", "数据根里还没有 profile（新机器正常，标定后才有）"
    return "", f"有多个 profile，未指定要检查哪个：{', '.join(avail)}"


def _check_profile(cfg, wanted: str) -> tuple[Check, Check, Check, list[str]]:
    """返回 (profile, 模板, 标答, warns)。"""
    import cv2
    import yaml

    if not wanted:
        note = "没有指定 profile，也没有 profile 指针（新机器正常，标定后才有）"
        return (_warn("profile 能加载", note),
                _warn("模板可读", note),
                _warn("标答可读", note), [])
    try:
        from .profile import load_profile

        prof = load_profile(wanted)
    except Exception as e:  # noqa: BLE001
        # 指定了却加载不出来 = 硬伤：门禁用不了，必须拦
        msg = f"{type(e).__name__}: {e}"
        return _fail("profile 能加载", msg), _warn("模板可读", msg), _warn("标答可读", msg), []

    # ---- 模板：能读 + 能解码（解码失败才是硬伤：门禁会永远 fail_closed）
    tdir = Path(prof.templates_dir)
    gates = sorted(list(tdir.glob("gate_*.png")) + list(tdir.glob("*.png"))) if tdir.exists() else []
    if not gates:
        tpl = _warn("模板可读", f"{tdir} 里没有 png（未标定的 profile 正常）")
    else:
        bad = [p.name for p in gates if cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) is None]
        tpl = (_fail("模板可读", f"{len(bad)}/{len(gates)} 张解码失败：{', '.join(bad[:3])}")
               if bad else _ok("模板可读", f"{len(gates)} 张，{tdir.name}"))
    idx = tdir / "_index.json" if tdir.exists() else None
    warns: list[str] = []
    if idx and idx.exists():
        try:
            json.loads(idx.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            warns.append(f"templates/_index.json 解析失败（不影响启动）：{e}")

    # ---- 标答：坏了只是没法回归，不拦升级
    labels = Path(prof.root) / "labels.json"
    if not labels.exists():
        lab = _warn("标答可读", "没有 labels.json（还没标过）")
    else:
        try:
            json.loads(labels.read_text(encoding="utf-8"))
            lab = _ok("标答可读", str(labels))
        except Exception as e:  # noqa: BLE001
            lab = _warn("标答可读", f"解析失败：{e}")
    _ = yaml  # 保持导入，labels 也可能是 yaml 扩展时用
    return _ok("profile 能加载", f"{prof.id} [{prof.status}]"), tpl, lab, warns


def _check_recipes(cfg) -> Check:
    """当前这一代的规则能不能被新引擎读懂（配方版本是否兼容）。"""
    from .evolve import validate_rules, active_generation
    from .version import RECIPE_SCHEMA

    arb = getattr(getattr(cfg, "decision", None), "arbiter", None)
    path = str(getattr(arb, "rules_path", "") or "")
    if not path:
        return _warn("配方兼容", "没有绑定规则文件（还没进化过）")
    p = Path(path)
    if not p.exists():
        return _warn("配方兼容", f"规则文件不存在：{p}")
    try:
        import yaml as _yaml

        doc = _yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:  # noqa: BLE001
        return _warn("配方兼容", f"rules.yaml 解析失败（运行时会当空规则）：{e}")
    ver = doc.get("recipe_schema")
    if isinstance(ver, int) and ver > RECIPE_SCHEMA:
        # 这份配方比引擎还新 —— 通常是回退到了旧引擎，必须让人知道
        return _fail("配方兼容", f"配方 schema {ver} 比引擎支持的 {RECIPE_SCHEMA} 新，"
                                f"旧引擎读不了这份配方")
    rules, errors = validate_rules(doc)
    if errors:
        return _warn("配方兼容", f"{len(errors)} 条不合法，运行时会整份不用：{errors[0]}")
    _ = active_generation  # 便于将来按代数检查
    return _ok("配方兼容", f"{len(rules)} 条规则可执行")


def _check_disk() -> Check:
    from . import paths

    try:
        free_mb = shutil.disk_usage(paths.data_root()).free // (1024 * 1024)
    except OSError as e:
        return _warn("磁盘空间", f"读不到：{e}")
    return (_ok("磁盘空间", f"剩余 {free_mb} MB") if free_mb >= _MIN_FREE_MB
            else _fail("磁盘空间", f"只剩 {free_mb} MB，低于 {_MIN_FREE_MB} MB"))


def run(profile: str = "") -> dict:
    """跑全套自检。返回 {ok, checks, ...}，ok 只由 fail 决定。"""
    from . import paths
    from .version import info

    pid, pid_note = _resolve_profile_id(profile)
    checks: list[Check] = [_check_paths(), _check_deps()]
    cfg_check, cfg = _check_config(pid)
    checks.append(cfg_check)
    if not pid and pid_note:
        checks.append(_warn("profile 指针", pid_note))
    if cfg is not None:
        p_ok, tpl, lab, warns = _check_profile(cfg, pid)
        checks.extend([p_ok, tpl, lab])
        checks.extend(_warn("数据提示", w) for w in warns)
        checks.append(_check_recipes(cfg))
    checks.append(_check_disk())

    fails = [c for c in checks if c["level"] == "fail"]
    return {
        "ok": not fails,
        "checks": checks,
        "warns": [c["name"] for c in checks if c["level"] == "warn"],
        "paths": paths.describe(),
        **info(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="autogame 引擎自检（升级闸门）")
    ap.add_argument("--json", action="store_true", help="输出 JSON（升级工具解析用）")
    ap.add_argument("--profile", default="", help="要检查哪个 profile")
    args = ap.parse_args(argv)

    res = run(args.profile)
    if args.json:
        print(json.dumps(res, ensure_ascii=False))
    else:
        icon = {"ok": "OK  ", "warn": "注意", "fail": "失败"}
        for c in res["checks"]:
            print(f"[{icon.get(c['level'], c['level'])}] {c['name']}  {c['detail']}")
        print(f"\n引擎 {res['engine_version']}  数据 schema {res['data_schema']}  "
              f"代码根 {res['paths']['code']}  数据根 {res['paths']['data']}")
        print("结论：" + ("可以上线" if res["ok"] else "不能上线（见上面的失败项）"))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
