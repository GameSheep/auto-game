"""引擎升级 —— 换整个目录，不就地改一个字节；数据原地不动；换不过去就换回来。

## 和"二次进化"的分工（这是两件不同的事）

    evolve   数据层，设备自己长：吃截图/手册 → 出配方（rules/templates/knowledge）
    upgrade  代码层，只有人能推：整包换引擎目录

混在一起就是把"能自学"和"能改自己"划了等号 —— 前者可退，后者会变砖。
所以这里的每一行都默认：**引擎目录只读**，升级=解包到 `releases/<ver>/` + 改指针。

## 为什么是"包 + 指针"而不是"就地覆盖"

    1. 就地覆盖中途断电 = 一半新一半旧，且没有任何办法知道坏在哪。
       解包到 staging、校验完、再原子 rename，失败时现场是完整的旧版。
    2. 想退回去，必须有旧版还在。目录留着就有，不用备份工具。
    3. 升级包自带每个文件的 sha256（RELEASE.json），
       所以"这台机器上的引擎有没有被人手改过"是**可回答**的问题（`verify`）。

## 包内容（tar.gz）

    RELEASE.json        版本三元组 + 每个文件的 sha256 + 构建时间/git rev
    run_server.py, autogame/, web/, tools/, deploy/, profiles/_template ...
    （不含 profiles/<游戏>/、logs/、assets/ —— 那些是用户数据，不随代码走）

## 一次升级的完整顺序

    校验包（sha256/路径穿越/多余文件）
      → 数据布局兼容（schema 区间）
      → 磁盘余量
      → 解包到 staging
      → **用新引擎自己的 doctor 在真实数据上自检**
      → 快照数据元信息
      → staging 转正 + 指针原子切换（previous 记旧版）
      → 写 upgrade.jsonl
    任何一步失败：删 staging，`current` 一个字节都没动。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

from . import paths
from .atomic import atomic_write_json, atomic_write_text
from .version import DATA_SCHEMA, MIN_DATA_SCHEMA, SCHEMA_FILE

MANIFEST = "RELEASE.json"
CURRENT = "current"
PREVIOUS = "previous"
LOG = "upgrade.jsonl"
SNAPSHOTS = "snapshots"

_CHUNK = 1 << 20
_KEEP_DEFAULT = 3
#: 快照只收"人写的/程序配的"小文本，不收图 —— 图是追加型的，回滚不该把它们删掉
_SNAP_SUFFIX = {".json", ".yaml", ".yml", ".md", ".txt", ".ini", ".cfg", ".schema"}
_SNAP_MAX = 4 << 20

Step = dict[str, Any]


# --------------------------------------------------------------------------- 路径

def releases_dir(install_root: Path) -> Path:
    return Path(install_root) / "releases"


def release_dir(install_root: Path, version: str) -> Path:
    return releases_dir(install_root) / str(version)


def current_version(install_root: Path | None = None) -> str | None:
    return _read_pointer(_ptr(install_root or paths.install_root(), CURRENT))


def previous_version(install_root: Path | None = None) -> str | None:
    return _read_pointer(_ptr(install_root or paths.install_root(), PREVIOUS))


def _ptr(install_root: Path, name: str) -> Path:
    return Path(install_root) / name


def _read_pointer(p: Path) -> str | None:
    try:
        v = p.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return v or None


def _write_pointer(p: Path, value: str) -> None:
    """原子写 + fsync。

    这是整套东西里最要紧的几十字节：`current` 空了，启动器就不知道跑哪一版；
    `previous` 空了，退不回去。rename 只保证"名字"换得干净，不保证内容落盘 ——
    eMMC 上掉电完全可能得到一个空文件，而它看起来只是"没装过"。
    """
    atomic_write_text(p, str(value).strip() + "\n")


# --------------------------------------------------------------------------- 哈希/清单

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest(src: Path) -> dict:
    """从目录或 tar.gz 里读 RELEASE.json。"""
    src = Path(src)
    if src.is_dir():
        p = src / MANIFEST
        if not p.exists():
            raise FileNotFoundError(f"没有 {MANIFEST}: {src}")
        return json.loads(p.read_text(encoding="utf-8"))
    with tarfile.open(src, "r:*") as tf:
        try:
            m = tf.getmember(MANIFEST)
        except KeyError as e:
            raise FileNotFoundError(f"升级包里没有 {MANIFEST}") from e
        f = tf.extractfile(m)
        if f is None:
            raise ValueError(f"{MANIFEST} 不是普通文件")
        return json.loads(f.read().decode("utf-8"))


def _safe_members(tf: tarfile.TarFile) -> list[tarfile.TarInfo]:
    """拒绝绝对路径、`..` 逃逸和链接 —— 升级包是唯一以 root 身份解包的东西。"""
    out: list[tarfile.TarInfo] = []
    for m in tf.getmembers():
        name = m.name.replace("\\", "/")
        if name.startswith("/") or name.startswith("../") or "/../" in name:
            raise ValueError(f"升级包里有越界路径: {m.name}")
        if m.issym() or m.islnk():
            raise ValueError(f"升级包里有链接，不接受: {m.name}")
        if not (m.isfile() or m.isdir()):
            raise ValueError(f"升级包里有特殊文件: {m.name}")
        out.append(m)
    return out


def verify_package(pkg: Path, *, manifest: dict | None = None) -> dict:
    """逐文件比对 sha256，并检查"包里有没有清单外的东西"。

    为什么连多余文件也要查：清单是"这个包应该是什么样子"的唯一声明。
    多出来的文件要么是构建时混进了用户数据（隐私），要么是有人往里塞了东西。
    """
    pkg = Path(pkg)
    man = manifest or read_manifest(pkg)
    want: dict[str, str] = dict(man.get("files") or {})
    seen: set[str] = set()
    missing: list[str] = []
    bad: list[str] = []
    extra: list[str] = []

    with tarfile.open(pkg, "r:*") as tf:
        for m in _safe_members(tf):
            if not m.isfile():
                continue
            name = m.name.replace("\\", "/")
            if name == MANIFEST:
                continue
            seen.add(name)
            expect = want.get(name)
            if expect is None:
                extra.append(name)
                continue
            f = tf.extractfile(m)
            h = hashlib.sha256()
            for block in iter(lambda: f.read(_CHUNK), b""):  # type: ignore[union-attr]
                h.update(block)
            if h.hexdigest() != expect:
                bad.append(name)

    missing = [n for n in want if n not in seen]
    return {"ok": not (missing or bad or extra), "missing": missing,
            "modified": bad, "extra": extra, "count": len(want)}


def verify_release(release: Path) -> dict:
    """已安装的 release 有没有被就地改过（"引擎绝对不能改"的执法手段）。"""
    release = Path(release)
    try:
        man = read_manifest(release)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "missing": [], "modified": [], "extra": [str(e)], "count": 0}
    want: dict[str, str] = dict(man.get("files") or {})
    missing, bad, extra = [], [], []
    for name, digest in want.items():
        p = release / name
        if not p.exists():
            missing.append(name)
        elif sha256_file(p) != digest:
            bad.append(name)
    for p in release.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(release).as_posix()
        if rel == MANIFEST or rel in want:
            continue
        if "__pycache__" in rel or rel.endswith(".pyc"):
            continue
        extra.append(rel)
    return {"ok": not (missing or bad or extra), "missing": missing,
            "modified": bad, "extra": extra[:50], "count": len(want)}


# --------------------------------------------------------------------------- 兼容性

def read_schema(data_root: Path) -> int:
    p = Path(data_root) / SCHEMA_FILE
    try:
        return int(p.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return MIN_DATA_SCHEMA  # 没有标记 = 最老的布局


def write_schema(data_root: Path, value: int = DATA_SCHEMA) -> None:
    atomic_write_text(Path(data_root) / SCHEMA_FILE, f"{int(value)}\n")


def check_compat(manifest: dict, data_root: Path) -> tuple[bool, str]:
    """数据版本要在目标引擎能读的区间里。

    两边都会出事：
        数据太旧（< 目标的 min_data_schema）  → 必须先升到中间版本，不能跳
        目标太旧（< 现在的数据 schema）        → 那是降级，会把新字段读丢
    """
    have = read_schema(data_root)
    lo = int(manifest.get("min_data_schema", MIN_DATA_SCHEMA))
    hi = int(manifest.get("data_schema", DATA_SCHEMA))
    if have < lo:
        return False, (f"数据是 schema {have}，这个引擎只认 {lo} 及以上。"
                       f"先装一个支持 schema {have} 的版本把数据升上来（不能跳版本）。")
    if have > hi:
        return False, (f"数据已经是 schema {have}，这个引擎只到 {hi}：退回去会把新字段读丢。"
                       f"确实要退，加 --force（会先快照）。")
    return True, f"数据 schema {have} ∈ [{lo}, {hi}]"


# --------------------------------------------------------------------------- 快照

def snapshot(data_root: Path, tag: str) -> dict:
    """把数据里的"配置类文本"拷一份，用于回滚到旧引擎时还原。

    不收图：回滚不该删用户的截图，而截图也不会被升级改动。
    不收 logs/*.jsonl：那是历史，回滚它等于篡改台账。
    """
    data_root = Path(data_root)
    dst = data_root / SNAPSHOTS / f"{time.strftime('%Y%m%d-%H%M%S')}-{tag}"
    dst.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for p in data_root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(data_root)
        if rel.parts[0] == SNAPSHOTS or (len(rel.parts) > 1 and rel.parts[0] == "logs"):
            continue
        if p.suffix.lower() not in _SNAP_SUFFIX:
            continue
        try:
            if p.stat().st_size > _SNAP_MAX:
                continue
        except OSError:
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        files[rel.as_posix()] = sha256_file(p)
    atomic_write_json(dst / MANIFEST,
                      {"tag": tag, "at": time.time(), "files": files}, indent=2)
    _prune_snapshots(data_root, keep=5)
    return {"path": str(dst), "files": len(files)}


def _prune_snapshots(data_root: Path, keep: int = 5) -> None:
    root = Path(data_root) / SNAPSHOTS
    if not root.exists():
        return
    items = sorted([p for p in root.iterdir() if p.is_dir()])
    for p in items[:-keep] if keep > 0 else []:
        shutil.rmtree(p, ignore_errors=True)


def list_snapshots(data_root: Path) -> list[dict]:
    root = Path(data_root) / SNAPSHOTS
    if not root.exists():
        return []
    out = []
    for p in sorted([p for p in root.iterdir() if p.is_dir()], reverse=True):
        try:
            man = json.loads((p / MANIFEST).read_text(encoding="utf-8"))
            n = len(man.get("files") or {})
        except Exception:  # noqa: BLE001
            n = 0
        out.append({"name": p.name, "files": n})
    return out


def restore_snapshot(data_root: Path, snap: str | Path) -> dict:
    """把快照里的文件写回数据根（先给当前状态再拍一张，免得越还原越乱）。"""
    data_root = Path(data_root)
    snap = Path(snap)
    if not snap.is_absolute():
        snap = data_root / SNAPSHOTS / str(snap)
    man = json.loads((snap / MANIFEST).read_text(encoding="utf-8"))
    pre = snapshot(data_root, "pre-restore")
    n = 0
    for rel in (man.get("files") or {}):
        src = snap / rel
        if not src.exists():
            continue
        dst = data_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n += 1
    return {"ok": True, "restored": n, "pre_snapshot": pre["path"]}


# --------------------------------------------------------------------------- 自检

def _python_for(install_root: Path) -> str:
    """优先用安装根下的 venv：设备上系统 python 没有 cv2，自检会假失败。"""
    for rel in ("venv/bin/python", ".venv/bin/python", ".venv/Scripts/python.exe"):
        p = Path(install_root) / rel
        if p.exists():
            return str(p)
    return sys.executable


def run_doctor(release: Path, data_root: Path, install_root: Path,
               *, timeout: float = 120.0, profile: str = "") -> dict:
    """用**新引擎自己的** doctor 在真实数据上自检。

    必须起子进程：同一个进程里 import 的还是旧代码，测不到新引擎。
    """
    env = os.environ.copy()
    env["AUTOGAME_DATA"] = str(data_root)
    env["AUTOGAME_HOME"] = str(install_root)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(release)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [_python_for(install_root), "-m", "autogame.doctor", "--json"]
    if profile:
        cmd += ["--profile", profile]
    try:
        r = subprocess.run(cmd, cwd=str(release), env=env, capture_output=True,
                           text=True, encoding="utf-8", timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "checks": [{"name": "自检", "level": "fail",
                                         "detail": f"超过 {timeout:.0f}s 没结束"}],
                "warns": [], "raw": ""}
    raw = (r.stdout or "").strip()
    line = next((ln for ln in reversed(raw.splitlines()) if ln.startswith("{")), "")
    if not line:
        return {"ok": False, "checks": [{"name": "自检", "level": "fail",
                                         "detail": (r.stderr or raw or "没有任何输出")[-400:]}],
                "warns": [], "raw": raw}
    try:
        res = json.loads(line)
    except json.JSONDecodeError as e:
        return {"ok": False, "checks": [{"name": "自检", "level": "fail",
                                         "detail": f"输出不是 JSON: {e}"}], "warns": [], "raw": raw}
    res["raw"] = raw
    if r.returncode != 0 and res.get("ok"):
        res["ok"] = False
        res.setdefault("checks", []).append(
            {"name": "自检退出码", "level": "fail", "detail": f"returncode={r.returncode}"})
    return res


# --------------------------------------------------------------------------- 安装

def _inside(child: Path, parent: Path) -> bool:
    try:
        Path(child).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def _data_conflict(install_root: Path, data_root: Path) -> str | None:
    rel = releases_dir(install_root)
    if _inside(data_root, rel):
        return (f"数据根 {data_root} 在 {rel} 里面：升级换目录会把数据一起换掉。"
                f"把 AUTOGAME_DATA 指到 releases 之外（设备上是 /var/lib/autogame）")
    return None


def _prune_releases(install_root: Path, keep: int, *, protect: set[str]) -> list[str]:
    rel = releases_dir(install_root)
    if not rel.exists():
        return []
    items = sorted([p for p in rel.iterdir() if p.is_dir() and not p.name.startswith(".")],
                   key=lambda p: p.stat().st_mtime)
    removed: list[str] = []
    for p in items[:max(0, len(items) - keep)]:
        if p.name in protect:
            continue
        shutil.rmtree(p, ignore_errors=True)
        removed.append(p.name)
    return removed


def _log(install_root: Path, entry: dict) -> None:
    p = Path(install_root) / LOG
    entry = {"ts": time.time(), **entry}
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def history(install_root: Path | None = None, limit: int = 20) -> list[dict]:
    p = Path(install_root or paths.install_root()) / LOG
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def install(pkg: str | Path, *, install_root: Path | None = None,
            data_root: Path | None = None, force: bool = False,
            dry_run: bool = False, with_deps: bool = False,
            profile: str = "", keep: int = _KEEP_DEFAULT) -> dict:
    """装一个升级包。返回 {ok, steps, version, previous, doctor, restart_required}。

    失败时**不改任何指针**，报错里带具体是哪一步、为什么。
    """
    pkg = Path(pkg)
    iroot = Path(install_root or paths.install_root())
    droot = Path(data_root or paths.data_root())
    steps: list[Step] = []

    def step(name: str, ok: bool, detail: str = "") -> bool:
        steps.append({"step": name, "ok": ok, "detail": detail})
        return ok

    if not pkg.exists():
        step("读包", False, f"文件不存在: {pkg}")
        return {"ok": False, "steps": steps, "error": f"文件不存在: {pkg}"}

    # 1) 清单 + 完整性
    try:
        man = read_manifest(pkg)
    except Exception as e:  # noqa: BLE001
        step("读包", False, str(e))
        return {"ok": False, "steps": steps, "error": str(e)}
    ver = str(man.get("engine_version") or "").strip()
    if not ver:
        step("读包", False, "RELEASE.json 里没有 engine_version")
        return {"ok": False, "steps": steps, "error": "升级包缺少版本号"}
    n_files = man.get("count", len(man.get("files") or {}))
    step("读包", True, f"{pkg.name} → {ver}（{n_files} 个文件）")

    veri = verify_package(pkg, manifest=man)
    if not veri["ok"]:
        detail = (f"缺 {len(veri['missing'])} 个、改 {len(veri['modified'])} 个、"
                  f"多 {len(veri['extra'])} 个")
        first = (veri["missing"] + veri["modified"] + veri["extra"])[:3]
        step("校验包", False, f"{detail}：{', '.join(first)}")
        return {"ok": False, "steps": steps, "error": "包内容和清单不一致（可能没下完或被改过）"}
    step("校验包", True, f"{veri['count']} 个文件 sha256 全对")

    # 2) 位置 + 兼容 + 空间
    conflict = _data_conflict(iroot, droot)
    if conflict:
        step("检查布局", False, conflict)
        return {"ok": False, "steps": steps, "error": conflict}
    step("检查布局", True, f"代码 {iroot}  数据 {droot}")

    ok_c, why_c = check_compat(man, droot)
    if not ok_c and not force:
        step("数据兼容", False, why_c)
        return {"ok": False, "steps": steps, "error": why_c}
    step("数据兼容", True, why_c + ("（--force 放行）" if not ok_c else ""))

    need = int(man.get("bytes") or 0) * 2
    try:
        free = shutil.disk_usage(iroot if iroot.exists() else iroot.parent).free
    except OSError:
        free = need
    if need and free < need:
        msg = f"需要约 {need >> 20} MB（包的两倍），只剩 {free >> 20} MB"
        step("磁盘空间", False, msg)
        return {"ok": False, "steps": steps, "error": msg}
    step("磁盘空间", True, f"需要 {need >> 20} MB，可用 {free >> 20} MB")

    if dry_run:
        step("演练", True, "只校验，没解包也没改指针")
        return {"ok": True, "dry_run": True, "version": ver,
                "current": current_version(iroot), "steps": steps, "manifest": _slim(man)}

    # 3) 解包到 staging（同分区，后面 rename 才是原子的）
    rel_root = releases_dir(iroot)
    rel_root.mkdir(parents=True, exist_ok=True)
    final = rel_root / ver
    staging = rel_root / f".staging-{ver}-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    try:
        with tarfile.open(pkg, "r:*") as tf:
            members = _safe_members(tf)
            if hasattr(tarfile, "data_filter"):
                tf.extractall(staging, members=members, filter="data")
            else:  # pragma: no cover - 3.12 之前
                tf.extractall(staging, members=members)
    except Exception as e:  # noqa: BLE001
        shutil.rmtree(staging, ignore_errors=True)
        step("解包", False, f"{type(e).__name__}: {e}")
        return {"ok": False, "steps": steps, "error": f"解包失败: {e}"}
    step("解包", True, f"{staging}")

    # 4) 依赖（可选）—— 新引擎可能要新库
    if with_deps:
        req = staging / "requirements.txt"
        if req.exists():
            try:
                r = subprocess.run([_python_for(iroot), "-m", "pip", "install", "-q",
                                    "-r", str(req)], capture_output=True, text=True,
                                   timeout=600, encoding="utf-8")
                okd, detail = r.returncode == 0, (r.stdout or r.stderr or "")[-300:]
            except Exception as e:  # noqa: BLE001
                okd, detail = False, f"{type(e).__name__}: {e}"
            step("装依赖", okd, detail or "已是最新")
            if not okd:
                shutil.rmtree(staging, ignore_errors=True)
                return {"ok": False, "steps": steps,
                        "error": "依赖没装上（venv 可能没网）。装完再试，或用 --no-deps"}

    # 5) 快照（在自检之前：doctor 会创建目录、写 .schema）
    snap = snapshot(droot, f"pre-{ver}")
    step("快照数据", True, f"{snap['files']} 个配置类文件 → {Path(snap['path']).name}")

    # 6) 新引擎自检（硬闸门）
    doc = run_doctor(staging, droot, iroot, profile=profile)
    fails = [c for c in doc.get("checks", []) if c.get("level") == "fail"]
    if not doc.get("ok"):
        shutil.rmtree(staging, ignore_errors=True)
        msg = "；".join(f"{c['name']}：{c['detail']}" for c in fails[:3]) or "自检未通过"
        step("新引擎自检", False, msg)
        return {"ok": False, "steps": steps, "doctor": _slim_doc(doc),
                "error": f"新引擎在这台机器上跑不起来，已放弃（指针没动）。{msg}"}
    step("新引擎自检", True, "；".join(c["name"] for c in doc.get("checks", [])
                                     if c.get("level") == "warn") or "全部通过")

    # 7) 转正 + 指针
    old = current_version(iroot)
    if final.exists():
        if not force:
            shutil.rmtree(staging, ignore_errors=True)
            msg = f"{ver} 已经装过了。要重装覆盖加 --force"
            step("转正", False, msg)
            return {"ok": False, "steps": steps, "error": msg}
        shutil.rmtree(final, ignore_errors=True)
    os.replace(staging, final)
    step("转正", True, str(final))

    if old and old != ver:
        _write_pointer(_ptr(iroot, PREVIOUS), old)
    elif not previous_version(iroot):
        _write_pointer(_ptr(iroot, PREVIOUS), ver)
    _write_pointer(_ptr(iroot, CURRENT), ver)
    step("切指针", True, f"{old or '（无）'} → {ver}")

    write_schema(droot, int(man.get("data_schema", DATA_SCHEMA)))
    removed = _prune_releases(iroot, keep, protect={ver, previous_version(iroot) or "", old or ""})
    if removed:
        step("清理旧版", True, "删掉 " + "、".join(removed))

    _log(iroot, {"action": "install", "ok": True, "from": old, "to": ver,
                 "package": str(pkg), "files": veri["count"]})
    return {"ok": True, "version": ver, "previous": old, "steps": steps,
            "doctor": _slim_doc(doc), "restart_required": old != ver,
            "removed": removed}


def rollback(*, install_root: Path | None = None, data_root: Path | None = None,
             to: str = "", force: bool = False) -> dict:
    """退回到上一个（或指定）版本。只切指针，不删文件 —— 退错了还能再退回来。"""
    iroot = Path(install_root or paths.install_root())
    droot = Path(data_root or paths.data_root())
    cur = current_version(iroot)
    target = to or previous_version(iroot)
    steps: list[Step] = []

    if not target:
        return {"ok": False, "error": "没有上一版可退（previous 指针为空）", "steps": steps}
    if target == cur:
        return {"ok": False, "error": f"上一版就是当前版（{cur}），没得退", "steps": steps}
    tdir = release_dir(iroot, target)
    if not (tdir / "run_server.py").exists():
        return {"ok": False, "error": f"{target} 的目录不完整（{tdir}），不能退", "steps": steps}

    try:
        man = read_manifest(tdir)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{target} 的清单读不出来: {e}", "steps": steps}

    ok_c, why_c = check_compat(man, droot)
    if not ok_c and not force:
        return {"ok": False, "error": why_c,
                "hint": "数据已被新引擎升过版，退回旧引擎会读不全。"
                        "确认要退：--force；想连数据一起回：upgrade restore <快照>"}

    snapshot(droot, f"pre-rollback-{target}")
    _write_pointer(_ptr(iroot, PREVIOUS), cur)
    _write_pointer(_ptr(iroot, CURRENT), target)
    _log(iroot, {"action": "rollback", "ok": True, "from": cur, "to": target})
    steps.append({"step": "切指针", "ok": True, "detail": f"{cur} → {target}"})
    return {"ok": True, "version": target, "previous": cur, "steps": steps,
            "restart_required": True}


# --------------------------------------------------------------------------- 状态

def _slim(man: dict) -> dict:
    return {k: v for k, v in man.items() if k != "files"}


def _slim_doc(doc: dict) -> dict:
    return {"ok": doc.get("ok"), "checks": doc.get("checks", []), "warns": doc.get("warns", [])}


def status(install_root: Path | None = None, data_root: Path | None = None,
           *, deep: bool = False) -> dict:
    iroot = Path(install_root or paths.install_root())
    droot = Path(data_root or paths.data_root())
    cur = current_version(iroot)
    releases = []
    rel = releases_dir(iroot)
    if rel.exists():
        for p in sorted([p for p in rel.iterdir() if p.is_dir() and not p.name.startswith(".")]):
            item = {"version": p.name, "path": str(p),
                    "built_at": "", "git_rev": ""}
            try:
                man = read_manifest(p)
                item.update(built_at=str(man.get("built_at") or ""),
                            git_rev=str(man.get("git_rev") or ""))
            except Exception:  # noqa: BLE001
                item["broken"] = True
            releases.append(item)

    out = {
        "ok": True,
        "engine_version": cur,
        "previous": previous_version(iroot),
        "install_root": str(iroot),
        "data_root": str(droot),
        "paths": paths.describe(),
        "data_schema": read_schema(droot),
        "releases": releases,
        "history": history(iroot, limit=8),
        "snapshots": list_snapshots(droot),
        "restart_required": False,
    }
    if deep and cur:
        out["verify"] = verify_release(release_dir(iroot, cur))
    return out


def main(argv: list[str] | None = None) -> int:
    """`python -m autogame.upgrade status|verify|install|rollback`（设备上没有 git 时的兜底入口）。"""
    import argparse

    ap = argparse.ArgumentParser(description="autogame 引擎升级")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("status")
    pv = sub.add_parser("verify")
    pv.add_argument("--version", default="")

    pi = sub.add_parser("install")
    pi.add_argument("package")
    pi.add_argument("--force", action="store_true")
    pi.add_argument("--dry-run", action="store_true")
    pi.add_argument("--with-deps", action="store_true")
    pi.add_argument("--profile", default="")
    pi.add_argument("--keep", type=int, default=_KEEP_DEFAULT)

    pr = sub.add_parser("rollback")
    pr.add_argument("--to", default="")
    pr.add_argument("--force", action="store_true")

    ps = sub.add_parser("snapshots")
    ps.add_argument("--restore", default="")
    ps.add_argument("--list", action="store_true")

    a = ap.parse_args(argv)
    if a.cmd == "install":
        res = install(a.package, force=a.force, dry_run=a.dry_run,
                      with_deps=a.with_deps, profile=a.profile, keep=a.keep)
    elif a.cmd == "rollback":
        res = rollback(to=a.to, force=a.force)
    elif a.cmd == "verify":
        iroot = paths.install_root()
        v = a.version or current_version(iroot) or ""
        res = verify_release(release_dir(iroot, v))
    elif a.cmd == "snapshots":
        if a.restore:
            res = restore_snapshot(paths.data_root(), a.restore)
        else:
            res = {"ok": True, "snapshots": list_snapshots(paths.data_root())}
    else:
        res = status(deep=True)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
