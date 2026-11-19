"""引擎升级 —— 假包跑状态机，真仓库跑一次端到端。

这一组测试守的是三句承诺：
    1. 换不过去就不换（自检失败 → 指针一个字节没动）
    2. 换过去能退回（previous 指针 + 快照）
    3. 数据不随代码走（包里有不有用户数据、数据根在 releases 里要拒绝）
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from autogame import upgrade as up
from autogame import version


# --------------------------------------------------------------------------- 造包

DOCTOR_OK = """
import json, sys
print(json.dumps({"ok": True, "checks": [{"name": "自检", "level": "ok", "detail": ""}]}))
"""

DOCTOR_FAIL = """
import json, sys
print(json.dumps({"ok": False, "checks": [{"name": "依赖齐全", "level": "fail",
                                           "detail": "cv2 不在"}]}))
sys.exit(1)
"""


def _files(doctor_src: str = DOCTOR_OK, extra: dict[str, str] | None = None) -> dict[str, str]:
    out = {
        "run_server.py": "# stub entry\n",
        "autogame/__init__.py": "",
        "autogame/doctor.py": doctor_src,
        "web/dashboard.html": "<html></html>",
    }
    out.update(extra or {})
    return out


def make_pkg(path: Path, ver: str, *, doctor_src: str = DOCTOR_OK,
             data_schema: int = 2, min_data_schema: int = 1,
             extra: dict[str, str] | None = None) -> Path:
    """按 release.py 的格式造一个最小包（含逐文件 sha256 清单）。"""
    files = _files(doctor_src, extra)
    entries = {name: hashlib.sha256(body.encode()).hexdigest() for name, body in files.items()}
    manifest = {
        "engine_version": ver,
        "data_schema": data_schema,
        "min_data_schema": min_data_schema,
        "recipe_schema": 1,
        "entry": "run_server.py",
        "built_at": "2026-11-19T00:00:00",
        "git_rev": "deadbee",
        "dirty": False,
        "count": len(entries),
        "bytes": sum(len(b) for b in files.values()),
        "files": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tf:
        blob = json.dumps(manifest).encode()
        info = tarfile.TarInfo(up.MANIFEST)
        info.size = len(blob)
        tf.addfile(info, io.BytesIO(blob))
        for name, body in files.items():
            data = body.encode()
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return path


@pytest.fixture()
def roots(tmp_path: Path):
    """一个干净的安装根 + 数据根（数据根在 releases 之外）。"""
    iroot = tmp_path / "opt" / "autogame"
    droot = tmp_path / "var" / "autogame"
    (iroot / "releases").mkdir(parents=True)
    (droot / "profiles").mkdir(parents=True)
    return iroot, droot


# --------------------------------------------------------------------------- 包校验

def test_collect_keeps_code_and_drops_user_data(tmp_path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "rel_tool", Path(__file__).resolve().parent.parent / "tools" / "release.py")
    rel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rel)  # type: ignore[union-attr]

    root = tmp_path
    (root / "autogame").mkdir()
    (root / "autogame" / "a.py").write_text("x=1", encoding="utf-8")
    (root / "profiles" / "_template").mkdir(parents=True)
    (root / "profiles" / "_template" / "p.yaml").write_text("a: 1", encoding="utf-8")
    (root / "profiles" / "tli_phone960").mkdir(parents=True)
    (root / "profiles" / "tli_phone960" / "labels.json").write_text("{}", encoding="utf-8")
    (root / "logs").mkdir()
    (root / "logs" / "x.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "assets" / "refs").mkdir(parents=True)
    (root / "assets" / "refs" / "a.png").write_bytes(b"png")
    (root / "run_server.py").write_text("print(1)", encoding="utf-8")

    names = {p.as_posix() for p in rel.collect(root)}
    assert "autogame/a.py" in names
    assert "run_server.py" in names
    assert "profiles/_template/p.yaml" in names
    # 用户的游戏数据绝不能进包：升级会把设备上标了几个月的模板覆盖掉
    assert not any(n.startswith("profiles/tli_phone960") for n in names)
    assert not any(n.startswith("logs/") for n in names)
    assert not any(n.startswith("assets/") for n in names)


def test_package_verifies_and_reports_tampering(tmp_path: Path):
    pkg = make_pkg(tmp_path / "a.tar.gz", "0.2.1")
    assert up.verify_package(pkg)["ok"]

    # 用清单外的内容重打一个包（模拟"下到一半/被人改过"）
    man = up.read_manifest(pkg)
    bad = tmp_path / "b.tar.gz"
    with tarfile.open(pkg, "r:*") as src, tarfile.open(bad, "w:gz") as dst:
        for m in src.getmembers():
            data = src.extractfile(m).read()  # type: ignore[union-attr]
            if m.name == "run_server.py":
                data = b"# evil\n"
            m.size = len(data)
            dst.addfile(m, io.BytesIO(data))
    res = up.verify_package(bad, manifest=man)
    assert not res["ok"]
    assert res["modified"] == ["run_server.py"]


def test_tar_slip_rejected(tmp_path: Path):
    pkg = tmp_path / "evil.tar.gz"
    with tarfile.open(pkg, "w:gz") as tf:
        ti = tarfile.TarInfo("../../etc/passwd")
        body = b"x"
        ti.size = 1
        tf.addfile(ti, io.BytesIO(body))
    with pytest.raises(ValueError):
        # 给一份空清单：越界路径必须在"看每个成员"这一步被拒，而不是等到解包
        up.verify_package(pkg, manifest={"files": {}})


# --------------------------------------------------------------------------- 安装

def test_install_switches_pointer_and_keeps_previous(roots):
    iroot, droot = roots
    pkg = make_pkg(iroot.parent / "v1.tar.gz", "0.2.0")
    res = up.install(pkg, install_root=iroot, data_root=droot)
    assert res["ok"], res
    assert up.current_version(iroot) == "0.2.0"
    assert (iroot / "releases" / "0.2.0" / "run_server.py").exists()
    assert (iroot / "releases" / "0.2.0" / up.MANIFEST).exists()
    # 首次安装：上一版就是自己（回退时会明确告诉人"没得退"）
    assert up.previous_version(iroot) == "0.2.0"


def test_install_doctor_failure_leaves_pointer_untouched(roots):
    iroot, droot = roots
    up.install(make_pkg(iroot.parent / "v1.tar.gz", "0.2.0"),
               install_root=iroot, data_root=droot)
    res = up.install(make_pkg(iroot.parent / "v2.tar.gz", "0.2.1", doctor_src=DOCTOR_FAIL),
                     install_root=iroot, data_root=droot)
    assert not res["ok"]
    assert "跑不起来" in res["error"] or "没装成" in res["error"] or res["error"]
    assert up.current_version(iroot) == "0.2.0"          # 指针没动
    assert not (iroot / "releases" / "0.2.1").exists()   # staging 已清掉
    assert not list((iroot / "releases").glob(".staging-*"))


def test_install_refuses_downgrade_when_data_is_newer(roots):
    iroot, droot = roots
    up.write_schema(droot, 5)
    res = up.install(make_pkg(iroot.parent / "v2.tar.gz", "0.2.0", data_schema=2),
                     install_root=iroot, data_root=droot)
    assert not res["ok"]
    assert "读丢" in res["error"]
    assert up.current_version(iroot) is None


def test_install_refuses_data_inside_releases(roots):
    iroot, droot = roots
    bad = iroot / "releases" / "data"
    bad.mkdir(parents=True, exist_ok=True)
    res = up.install(make_pkg(iroot.parent / "v1.tar.gz", "0.2.0"),
                     install_root=iroot, data_root=bad)
    assert not res["ok"]
    assert "releases" in res["error"]


def test_same_version_needs_force(roots):
    iroot, droot = roots
    pkg = make_pkg(iroot.parent / "v1.tar.gz", "0.2.0")
    assert up.install(pkg, install_root=iroot, data_root=droot)["ok"]
    again = up.install(pkg, install_root=iroot, data_root=droot)
    assert not again["ok"]
    assert "已经装过" in again["error"]
    forced = up.install(pkg, install_root=iroot, data_root=droot, force=True)
    assert forced["ok"]


def test_dry_run_touches_nothing(roots):
    iroot, droot = roots
    res = up.install(make_pkg(iroot.parent / "v1.tar.gz", "0.2.0"),
                     install_root=iroot, data_root=droot, dry_run=True)
    assert res["ok"] and res.get("dry_run")
    assert up.current_version(iroot) is None
    assert not (iroot / "releases" / "0.2.0").exists()


# --------------------------------------------------------------------------- 回退

def test_rollback_returns_to_previous(roots):
    iroot, droot = roots
    up.install(make_pkg(iroot.parent / "v1.tar.gz", "0.2.0"),
               install_root=iroot, data_root=droot)
    up.install(make_pkg(iroot.parent / "v2.tar.gz", "0.2.1"),
               install_root=iroot, data_root=droot)
    assert up.current_version(iroot) == "0.2.1"

    res = up.rollback(install_root=iroot, data_root=droot)
    assert res["ok"], res
    assert up.current_version(iroot) == "0.2.0"
    assert up.previous_version(iroot) == "0.2.1"

    # 退错了还能退回去：指针互换，不删文件
    res2 = up.rollback(install_root=iroot, data_root=droot)
    assert res2["ok"]
    assert up.current_version(iroot) == "0.2.1"


def test_rollback_to_broken_dir_refused(roots):
    iroot, droot = roots
    up.install(make_pkg(iroot.parent / "v1.tar.gz", "0.2.0"),
               install_root=iroot, data_root=droot)
    up._write_pointer(iroot / up.PREVIOUS, "0.1.0")   # 指向不存在的版本
    res = up.rollback(install_root=iroot, data_root=droot)
    assert not res["ok"]
    assert "不完整" in res["error"]
    assert up.current_version(iroot) == "0.2.0"


# --------------------------------------------------------------------------- 完整性/状态

def test_verify_release_detects_local_edit(roots):
    iroot, droot = roots
    up.install(make_pkg(iroot.parent / "v1.tar.gz", "0.2.0"),
               install_root=iroot, data_root=droot)
    rel = iroot / "releases" / "0.2.0"
    assert up.verify_release(rel)["ok"]

    (rel / "run_server.py").write_text("# 有人手改了\n", encoding="utf-8")
    v = up.verify_release(rel)
    assert not v["ok"]
    assert "run_server.py" in v["modified"]


def test_status_shape(roots):
    iroot, droot = roots
    up.install(make_pkg(iroot.parent / "v1.tar.gz", "0.2.0"),
               install_root=iroot, data_root=droot)
    st = up.status(iroot, droot, deep=True)
    assert st["engine_version"] == "0.2.0"
    assert st["data_root"] == str(droot)
    assert [r["version"] for r in st["releases"]] == ["0.2.0"]
    assert st["history"][-1]["action"] == "install"
    assert st["verify"]["ok"]


def test_prune_keeps_current_and_previous(roots):
    iroot, droot = roots
    for v in ("0.1.0", "0.1.1", "0.1.2"):
        up.install(make_pkg(iroot.parent / f"{v}.tar.gz", v),
                   install_root=iroot, data_root=droot, keep=2)
    left = sorted(p.name for p in (iroot / "releases").iterdir() if p.is_dir())
    assert "0.1.2" in left                        # 当前
    assert "0.1.1" in left                        # 上一版（回退要用）
    assert "0.1.0" not in left


# --------------------------------------------------------------------------- 数据快照

def test_snapshot_takes_metadata_not_images(roots):
    _iroot, droot = roots
    prof = droot / "profiles" / "g1"
    prof.mkdir()
    (prof / "profile.yaml").write_text("id: g1\n", encoding="utf-8")
    (prof / "refs").mkdir()
    (prof / "refs" / "a.png").write_bytes(b"\x89PNG" + b"0" * 100)
    (droot / "logs").mkdir()
    (droot / "logs" / "supervisor.jsonl").write_text("{}\n", encoding="utf-8")

    snap = up.snapshot(droot, "t")
    assert snap["files"] >= 1
    names = {p.name for p in Path(snap["path"]).rglob("*") if p.is_file()}
    assert "profile.yaml" in names
    assert "a.png" not in names            # 图不回滚，也不会被回滚删掉
    assert "supervisor.jsonl" not in names  # 台账是历史，不能篡改

    (prof / "profile.yaml").write_text("id: g1\nbroken: true\n", encoding="utf-8")
    res = up.restore_snapshot(droot, snap["path"])
    assert res["ok"] and res["restored"] >= 1
    assert (prof / "profile.yaml").read_text(encoding="utf-8") == "id: g1\n"


# --------------------------------------------------------------------------- 真仓库端到端

def test_real_repo_package_installs_and_self_checks(roots, tmp_path: Path):
    """用真引擎打包 → 真解包 → 真 doctor。这条过了才敢说"能升级"。"""
    from tools.release import build

    repo = Path(__file__).resolve().parent.parent
    res = build(repo, tmp_path / "dist", version="9.9.9")
    assert res["ok"], res
    pkg = Path(res["path"])

    iroot, droot = roots
    out = up.install(pkg, install_root=iroot, data_root=droot)
    assert out["ok"], (out.get("error"), out.get("doctor"))
    assert up.current_version(iroot) == "9.9.9"

    rel = iroot / "releases" / "9.9.9"
    assert (rel / "autogame" / "version.py").exists()
    # doctor 真的在数据根上跑过：报告里有版本号，不是空壳
    doc = out["doctor"]
    assert any(c["name"] == "依赖齐全" for c in doc["checks"])

    # 包里的版本号来自 autogame/version.py，而不是手填
    man = up.read_manifest(rel)
    assert man["engine_version"] == "9.9.9"
    assert man["data_schema"] == version.DATA_SCHEMA
    assert man["recipe_schema"] == version.RECIPE_SCHEMA


# --------------------------------------------------------------------------- 代码根/数据根分离

def test_data_root_follows_env(monkeypatch, tmp_path: Path):
    from autogame import paths

    monkeypatch.setenv("AUTOGAME_DATA", str(tmp_path / "data"))
    assert paths.data_root() == (tmp_path / "data").resolve()
    assert paths.data("profiles") == (tmp_path / "data").resolve() / "profiles"
    assert not paths.shared()
    # 相对路径按数据根解析，不看 cwd（systemd 下 cwd 是 "/"）
    assert paths.resolve("logs/x.jsonl") == (tmp_path / "data").resolve() / "logs" / "x.jsonl"


def test_token_persists_in_data_root(monkeypatch, tmp_path: Path):
    """token 必须落在数据根：引擎目录只读，而且换代码不该让所有代理掉线。"""
    from autogame.config import load_config

    data = tmp_path / "data"
    data.mkdir()
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("server:/n  token: ''\n", encoding="utf-8")
    monkeypatch.setenv("AUTOGAME_DATA", str(data))
    monkeypatch.delenv("AUTOGAME_TOKEN", raising=False)

    cfg = load_config(cfg_file)
    tok = data / ".token"
    assert tok.exists()
    assert tok.read_text(encoding="utf-8").strip() == cfg.server.token
    # 第二次读到的是同一个 token（不是每次启动都换新的）
    assert load_config(cfg_file).server.token == cfg.server.token


def test_default_config_prefers_data_root(monkeypatch, tmp_path: Path):
    import importlib

    monkeypatch.setenv("AUTOGAME_DATA", str(tmp_path))
    (tmp_path / "config.yaml").write_text("server: {}\n", encoding="utf-8")
    rs = importlib.import_module("run_server")
    assert Path(rs._default_config()) == tmp_path / "config.yaml"
