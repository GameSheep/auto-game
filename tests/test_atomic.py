"""落盘的边界：断电、半份文件、两个进程抢着写。

这里钉的不是"写成功了没有"，而是**出了问题之后会怎样**。三件事必须成立：

1. 写到一半失败 → 旧文件一个字节都不动（不是"旧文件被截断成半份"）
2. 读到半份/空文件 → 当作坏了并留证据，不是静默当成"全新的空状态"
3. 两个进程同时改 → 后写的不能把先写的吃掉

第 2 条是最容易漏的：空文件解析成 `{}` 不报错，于是"学过的状态被清空了"
看起来像一次正常的首次运行，日志里什么都没有。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from autogame import atomic, learner
from autogame.groups import Group, World, groups_root, load_world, update_world

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------- 原子写


def test_write_replaces_content_and_leaves_no_temp_files(tmp_path):
    p = tmp_path / "state.json"
    atomic.atomic_write_json(p, {"a": 1})
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 1}
    # 临时文件必须清干净：留下的 .state.json.xxx.tmp 会让用户以为有备份
    assert [f.name for f in tmp_path.iterdir()] == ["state.json"]


def test_failure_mid_write_keeps_old_file_intact(tmp_path, monkeypatch):
    """替换失败（磁盘满、句柄被占、断电前一刻）→ 旧内容必须还在。"""
    p = tmp_path / "state.json"
    atomic.atomic_write_json(p, {"gen": 1})

    def boom(*_a, **_k):
        raise OSError("no space left on device")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic.atomic_write_json(p, {"gen": 2})

    assert json.loads(p.read_text(encoding="utf-8")) == {"gen": 1}
    assert [f.name for f in tmp_path.iterdir()] == ["state.json"]


def test_write_fsyncs_the_file(tmp_path, monkeypatch):
    """没有 fsync 的 rename 只保证名字对，不保证内容已经落盘。"""
    calls: list[int] = []
    real = os.fsync

    def counting(fd):
        calls.append(fd)
        return real(fd)

    monkeypatch.setattr(os, "fsync", counting)
    atomic.atomic_write_text(tmp_path / "a.txt", "hello")
    assert calls, "写文件没有 fsync —— 断电后可能得到一个长度 0 的文件"


def test_written_bytes_are_readable_after_fsync(tmp_path):
    p = tmp_path / "汉字.json"
    atomic.atomic_write_json(p, {"词": "刷图"})
    assert json.loads(p.read_text(encoding="utf-8")) == {"词": "刷图"}


# --------------------------------------------------------------------- 坏了怎么办


def test_truncated_json_is_backed_up_not_silently_dropped(tmp_path):
    """半份 JSON（断电最典型的样子）→ 返回默认值，并把原件挪到一边。"""
    p = tmp_path / "learner.json"
    p.write_text('{"enabled": true, "last_ts": 17', encoding="utf-8")   # 断在这里
    assert atomic.load_json_tolerant(p, None) is None
    assert not p.exists()
    backups = list(tmp_path.glob("learner.json.corrupt-*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8").startswith('{"enabled"')


def test_empty_file_counts_as_corrupt(tmp_path):
    """0 字节是断电最常见的残留。不能解析成 {} 就当"全新状态"接着跑。"""
    p = tmp_path / "learner.json"
    p.write_bytes(b"")
    assert atomic.load_json_tolerant(p, None) is None
    assert list(tmp_path.glob("learner.json.corrupt-*"))


def test_missing_file_is_not_an_incident(tmp_path):
    """文件不存在 ≠ 文件坏了：没装过的东西不该留下 .corrupt-* 垃圾。"""
    assert atomic.load_json_tolerant(tmp_path / "nope.json", {"x": 1}) == {"x": 1}
    assert not list(tmp_path.glob("*.corrupt-*"))


def test_corrupt_backup_name_does_not_collide(tmp_path):
    p = tmp_path / "s.json"
    for _ in range(2):
        p.write_text("{", encoding="utf-8")
        assert atomic.backup_corrupt(p) is not None
    assert len(list(tmp_path.glob("s.json.corrupt-*"))) == 2


# --------------------------------------------------------------------- 锁


def test_lock_serializes_threads(tmp_path):
    counter = {"n": 0}
    p = tmp_path / "c.json"

    def bump():
        with atomic.file_lock(p):
            cur = counter["n"]
            time.sleep(0.001)
            counter["n"] = cur + 1

    ts = [threading.Thread(target=bump) for _ in range(12)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert counter["n"] == 12


def test_lock_timeout_when_already_held(tmp_path):
    p = tmp_path / "c.json"
    held = threading.Event()

    def hold():
        with atomic.file_lock(p):
            held.set()
            time.sleep(0.6)

    t = threading.Thread(target=hold)
    t.start()
    held.wait(2)
    with pytest.raises(TimeoutError):
        with atomic.file_lock(p, timeout=0.15):
            pass
    t.join()


def test_nested_lock_on_same_path_times_out(tmp_path):
    """刻意钉住这个坑：进程内 RLock 是可重入的，但系统锁会自己等自己。

    所以"读-改-写一整段"只能用 `locked_update_json`，不能在锁里再调会加锁的函数。
    """
    p = tmp_path / "c.json"
    with atomic.file_lock(p):
        with pytest.raises(TimeoutError):
            with atomic.file_lock(p, timeout=0.15):
                pass


@pytest.mark.skipif(os.name == "nt" and not sys.executable, reason="需要子进程")
def test_lock_excludes_another_process(tmp_path):
    """跨进程才是这道锁存在的理由 —— 进程内用 threading.Lock 就够了。"""
    p = tmp_path / "c.json"
    code = (
        "import sys, time\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from autogame.atomic import file_lock\n"
        f"ctx = file_lock({str(p)!r}, timeout=5)\n"
        "ctx.__enter__()\n"
        "print('HELD', flush=True)\n"
        "time.sleep(1.2)\n"
        "ctx.__exit__(None, None, None)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8")
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "HELD", proc.stderr.read() if proc.stderr else ""
        with pytest.raises(TimeoutError):
            with atomic.file_lock(p, timeout=0.3):
                pass
    finally:
        proc.communicate(timeout=10)


def test_locked_update_json_serializes_writers(tmp_path):
    p = tmp_path / "n.json"

    def bump(_cur):
        return {"n": (_cur or {}).get("n", 0) + 1}

    ts = [threading.Thread(target=atomic.locked_update_json, args=(p, bump))
          for _ in range(10)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert json.loads(p.read_text(encoding="utf-8"))["n"] == 10


# --------------------------------------------------------------------- 接到真模块上


@pytest.fixture()
def prof(tmp_path):
    root = tmp_path / "profiles" / "u"
    root.mkdir(parents=True)
    (root / "templates").mkdir()
    return SimpleNamespace(root=root, id="u", templates_dir=root / "templates",
                           labels_path=root / "labels.json")


def test_learner_state_survives_corrupt_file(prof):
    (prof.root / "evolve").mkdir(parents=True, exist_ok=True)
    p = learner.state_path(prof)
    p.write_text("", encoding="utf-8")            # 断电离场的样子
    st = learner.load_state(prof)
    assert st.enabled is None and st.tries == {}
    assert list((prof.root / "evolve").glob("learner.json.corrupt-*"))


def test_learn_does_not_clobber_concurrent_toggle(prof, monkeypatch):
    """学的几十秒里用户点了「自动学习」开关 —— 学习回执不能把开关盖回去。

    这就是原来 `load → 改几十个字段 → save` 的漏洞：整个 State 是开头读的那份，
    `enabled` 也被写回旧值，表现是"点了一下又跳回去了"。
    """
    learner.save_state(prof, learner.State(enabled=True, last_digest="old"))

    def fake(profile, cfg, *, force=False, use_model=True):
        learner.set_enabled(profile, None, False)     # 学习期间用户关掉
        return {"ok": True, "stage": "promote", "gen": "gen-001", "promoted": True,
                "message": "台账 2/2", "score": {}, "fail": []}

    monkeypatch.setattr(learner.ev, "evolve", fake)
    learner.learn(prof, None)

    st = learner.load_state(prof)
    assert st.enabled is False
    assert st.last_result == "promoted"
    assert st.last_digest and st.last_digest != "old"


def test_mutate_state_only_touches_what_the_callback_touches(prof):
    learner.save_state(prof, learner.State(enabled=None, last_message="上一代回归没过",
                                           tries={"d1": {"n": 1}}))
    learner.mutate_state(prof, lambda s: setattr(s, "enabled", True))
    st = learner.load_state(prof)
    assert st.enabled is True
    assert st.last_message == "上一代回归没过"
    assert st.tries == {"d1": {"n": 1}}


def test_corrupt_world_layer_is_backed_up(prof):
    groups_root(prof).mkdir(parents=True, exist_ok=True)
    (groups_root(prof) / "_world.json").write_text('{"objects": ', encoding="utf-8")
    w = load_world(prof)
    assert w.objects == {}
    assert list(groups_root(prof).glob("_world.json.corrupt-*"))


def test_update_world_merges_instead_of_overwriting(prof):
    """两次"学一句"连着来：第二句不能把第一句挤掉（世界层只增不删的前提）。"""
    def add(term: str, std: str):
        def fn(w: World) -> World:
            w.terms[term] = std
            return w
        update_world(prof, fn)

    add("搬砖", "刷图")
    add("刷本", "刷图")
    assert load_world(prof).terms == {"搬砖": "刷图", "刷本": "刷图"}


def test_group_roundtrip_is_atomic(prof):
    from autogame.groups import save_group

    g = Group(id="刷k7", name="刷k7", say=["刷k7", "开始刷k7"])
    save_group(prof, g)
    leftovers = [p.name for p in groups_root(prof).iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_corrupt_ledger_is_not_silently_wiped(tmp_path):
    """台账坏了：新条目照写，但旧条目必须还在某个文件里 —— 那是人标的答案。

    改之前是 `except: data = 空`，然后整份覆盖写回：一次教学编译就把全部
    人工核对过的条目抹掉，日志里只有一行 warn，文件再也回不来。
    """
    from autogame.teach import TeachResult, write_labels

    labels = tmp_path / "labels.json"
    labels.write_text('{"version": 1, "items": [{"file": "refs/a.png", ', encoding="utf-8")
    res = TeachResult()
    res.labels_new = [{"file": "refs/b.png", "scene": "战斗", "strict": True}]

    n, p = write_labels(res, labels)
    assert n == 1
    kept = json.loads(p.read_text(encoding="utf-8"))
    assert [it["file"] for it in kept["items"]] == ["refs/b.png"]
    backups = list(tmp_path.glob("labels.json.corrupt-*"))
    assert backups and "refs/a.png" in backups[0].read_text(encoding="utf-8")
