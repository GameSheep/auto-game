"""打升级包 —— 仓库 → 一份可校验、可回滚的 tar.gz。

## 包里有什么、没有什么

    进包：引擎（autogame/）、页面（web/）、工具（tools/）、部署（deploy/）、
          样例 profile（_template、generic_*）、默认 config.yaml
    不进：profiles/<具体游戏>/、logs/、assets/、.token、.venv

理由：**代码随版本走，数据留在设备上**。如果把 `profiles/tli_phone960/` 打进包，
一次升级就会把用户在设备上标了三个月的模板覆盖掉 —— 这正是"引擎不能改、但能升级"
最怕的事故。所以包里只有"引擎跑起来需要的最小集"，用户数据一个字都不带。

## 清单为什么带 git rev 和 dirty

同一句"升级到 0.2.0"，一个月后没人说得清当时装的是哪份代码。
带上 rev 就能回到那棵树；dirty 为真时 `build` 会提示 ——
从没提交的工作区打出来的包，别人复现不出来，只适合自己试。
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MANIFEST = "RELEASE.json"

#: 整个目录都进包（相对仓库根）
_INCLUDE_DIRS = ("autogame", "web", "tools", "deploy", "docs")
#: 只挑这几个子目录：样例/通用 profile 是"引擎的一部分"，具体游戏的不是
_INCLUDE_PROFILE_DIRS = ("_template", "generic_1280x720")
_INCLUDE_FILES = (
    "run_server.py", "run_agent.py", "run_sim_agent.py",
    "requirements.txt", "requirements.lock", "config.yaml",
    "README.md", "pytest.ini",
)
_SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", "logs", "assets", "dist",
              ".pytest_cache", ".idea", ".vscode", "node_modules", "snapshots"}
_SKIP_SUFFIX = {".pyc", ".pyo", ".log", ".mp4", ".avi", ".mkv", ".zip", ".tar", ".gz"}
_SKIP_NAMES = {".token", "config.runtime.yaml", "benchmark_result.json"}


def _git(root: Path, *args: str) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=str(root), capture_output=True,
                           text=True, encoding="utf-8", timeout=10)
        return (r.stdout or "").strip() if r.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


def collect(root: Path) -> list[Path]:
    """汇总要进包的文件（相对 root 的 Path）。判定顺序：目录白名单 → 排除表。"""
    root = Path(root)
    out: list[Path] = []
    for d in _INCLUDE_DIRS:
        base = root / d
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if any(part in _SKIP_DIRS for part in p.relative_to(root).parts):
                continue
            if p.suffix.lower() in _SKIP_SUFFIX or p.name in _SKIP_NAMES:
                continue
            out.append(p.relative_to(root))

    for name in _INCLUDE_PROFILE_DIRS:
        base = root / "profiles" / name
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() not in _SKIP_SUFFIX \
                    and "__pycache__" not in p.parts:
                out.append(p.relative_to(root))

    for name in _INCLUDE_FILES:
        p = root / name
        if p.is_file() and name not in _SKIP_NAMES:
            out.append(Path(name))
    return sorted(set(out))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build(root: str | Path, out_dir: str | Path | None = None,
          version: str = "", *, require_clean: bool = False) -> dict:
    """打一个包。返回 {ok, path, version, files, bytes, dirty, warnings}。"""
    root = Path(root)
    from autogame.version import (DATA_SCHEMA, ENGINE_VERSION, MIN_DATA_SCHEMA,
                                  RECIPE_SCHEMA)

    ver = version or ENGINE_VERSION
    files = collect(root)
    if not files:
        return {"ok": False, "error": f"{root} 下没找到要打包的文件"}

    rev = _git(root, "rev-parse", "--short", "HEAD")
    dirty = bool(_git(root, "status", "--porcelain"))
    warnings = []
    if not rev:
        warnings.append("不在 git 仓库里：包无法追溯到某次提交")
    if dirty:
        warnings.append("工作区有未提交改动：这个包装到设备上和仓库对不上")

    out = Path(out_dir or (root / "dist"))
    out.mkdir(parents=True, exist_ok=True)
    pkg = out / f"autogame-{ver}.tar.gz"

    entries = {p.as_posix(): _sha256(root / p) for p in files}
    total = sum((root / p).stat().st_size for p in files)
    manifest = {
        "engine_version": ver,
        "data_schema": DATA_SCHEMA,
        "min_data_schema": MIN_DATA_SCHEMA,
        "recipe_schema": RECIPE_SCHEMA,
        "entry": "run_server.py",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_rev": rev,
        "dirty": dirty,
        "count": len(entries),
        "bytes": total,
        "files": entries,
    }

    # 先写临时包再改名：中途失败不会留一个"看起来能装"的半包
    tmp = pkg.with_suffix(".tar.gz.part")
    with tarfile.open(tmp, "w:gz") as tf:
        info = tarfile.TarInfo(MANIFEST)
        blob = json.dumps(manifest, ensure_ascii=False, indent=1).encode("utf-8")
        info.size = len(blob)
        info.mtime = int(time.time())
        tf.addfile(info, io.BytesIO(blob))
        for p in files:
            tf.add(root / p, arcname=p.as_posix(), recursive=False)
    tmp.replace(pkg)

    digest = _sha256(pkg)
    (out / f"{pkg.name}.sha256").write_text(f"{digest}  {pkg.name}\n", encoding="utf-8")
    if require_clean and (dirty or not rev):
        pkg.unlink(missing_ok=True)
        return {"ok": False, "error": "要求干净的工作区（--require-clean）", "warnings": warnings}
    return {"ok": True, "path": str(pkg), "version": ver, "files": len(entries),
            "bytes": total, "sha256": digest, "dirty": dirty,
            "git_rev": rev, "warnings": warnings}


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="打 autogame 升级包")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out", default="")
    ap.add_argument("--version", default="")
    ap.add_argument("--require-clean", action="store_true",
                    help="工作区不干净就失败（发正式版用）")
    a = ap.parse_args(argv)

    res = build(a.root, a.out or None, a.version, require_clean=a.require_clean)
    if not res.get("ok"):
        print("失败：" + str(res.get("error")))
        return 1
    print(f"包: {res['path']}")
    print(f"版本: {res['version']}  文件: {res['files']}  大小: {res['bytes'] >> 10} KB")
    print(f"sha256: {res['sha256'][:16]}…  git: {res['git_rev'] or '(无)'}"
          + ("  [工作区有改动]" if res["dirty"] else ""))
    for w in res.get("warnings") or []:
        print("提示：" + w)
    print(f"\n装到设备：python tools/upgrade.py install {res['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
