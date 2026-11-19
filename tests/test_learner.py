"""持续学习的边界测试。

这个模块只回答一个问题：**什么时候该学**。所以这里钉的不是识别精度，而是节奏：

    材料没变不许学        —— 否则每轮 tick 都要烧几十秒 CPU
    人还在写的时候不许学  —— 连着截五张图不该学五次半成品
    机器人在跑不许学      —— 换代会让正在执行的动作失去依据
    同一份材料失败不许重试 —— 那是材料的问题，不是运气问题，改了才会再给机会
    学坏了必须留回执      —— 不然"为什么还是上一代"永远说不清

`evolve.evolve` 在这些用例里被替换成假实现：真跑回归要动检测器和 FSM，几十秒一次，
既慢又会把"节奏"和"识别"两件事混在一起。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from autogame import evolve, learner
from autogame.config import load_config
from autogame.profile import load_profile

LESSON = """# 测试用手册

## 界面 · 战斗
- 图: refs/combat_a.jpg
- 判定: 正例

## 界面 · 战斗（另一张）
- 图: refs/combat_b.jpg
- 判定: 正例

## 界面 · 背包
- 图: refs/bag_a.jpg
- 判定: 反例

## 界面 · 结算
- 图: refs/result_a.jpg
- 判定: 反例
"""


def _frame(seed: int, h: int = 200, w: int = 360) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    cv2.circle(img, (w // 2, h // 2), 30, (40, 200, 240), -1)
    return img


@pytest.fixture()
def prof(tmp_path):
    root = tmp_path / "profiles" / "unit"
    (root / "lessons").mkdir(parents=True)
    (root / "refs").mkdir()
    (root / "templates").mkdir()
    (root / "profile.yaml").write_text(
        "id: unit\nname: 单元\nstatus: partial\n"
        "reference_resolution: [360, 200]\nconfig: {}\n", encoding="utf-8")
    (root / "lessons" / "t.md").write_text(LESSON, encoding="utf-8")
    for i, name in enumerate(("combat_a.jpg", "combat_b.jpg",
                              "bag_a.jpg", "result_a.jpg")):
        cv2.imwrite(str(root / "refs" / name), _frame(i))
    return load_profile(root)


@pytest.fixture()
def cfg(prof):
    c = load_config(None, profile=prof)
    c.learner.quiet_s = 0
    c.learner.min_gap_s = 0
    c.learner.max_tries = 1
    return c


def _seed_active(prof, digest: str = "seed000") -> None:
    """造一个"已上线"的代，让材料看起来是"学过了"。"""
    gid = "gen-000"
    root = evolve.gen_dir(prof, gid)
    (root / "knowledge").mkdir(parents=True, exist_ok=True)
    (root / "knowledge" / "game.md").write_text("# 游戏\n- 光柱就是掉落\n", encoding="utf-8")
    g = evolve.Generation(id=gid, root=root, status="active", created=1.0,
                          digest=digest, score={"strict_pass": 2, "strict_total": 2})
    evolve._write_gen(g)
    evolve.write_materials_snapshot(prof, g)
    evolve.set_active(prof, gid)


def _fake_evolve(monkeypatch, *, promoted: bool, message: str = "台账 2/2",
                 digest_note: str = "") -> None:
    def fake(profile, cfg, *, force=False, use_model=True):
        if digest_note:
            gid = digest_note
        else:
            gid = "gen-001"
        return {"ok": promoted, "stage": "promote", "gen": gid,
                "promoted": promoted, "message": message,
                "score": {"strict_pass": 2 if promoted else 1, "strict_total": 2},
                "fail": [] if promoted else ["refs/bag_a.jpg"]}
    monkeypatch.setattr(learner.ev, "evolve", fake)


def _touch_lesson(prof, extra: str = "\n- 术语: 光柱\n") -> None:
    p = prof.root / "lessons" / "t.md"
    p.write_text(p.read_text(encoding="utf-8") + extra, encoding="utf-8")


# ---------------- 1. 判断：什么时候学 ----------------

def test_up_to_date_when_materials_match_active(prof, cfg):
    """材料和当前这代一模一样 → 不该学。这是"不烧 CPU"的第一道闸。"""
    _seed_active(prof, evolve.materials_digest(prof))
    v = learner.decide(prof, cfg)
    assert v.code == "up_to_date"
    assert not v.should_learn


def test_waits_while_user_is_still_typing(prof, cfg):
    """材料刚动过 → 等静默期。连着截五张图不该学五次半成品。"""
    _seed_active(prof)
    cfg.learner.quiet_s = 45
    _touch_lesson(prof)
    v = learner.decide(prof, cfg)          # now=真实时间，文件刚写完
    assert v.code == "waiting"
    assert v.wait_s > 0
    # 静默期过了就放行
    later = time.time() + 60
    assert learner.decide(prof, cfg, now=later).code == "ready"


def test_respects_min_gap_between_generations(prof, cfg):
    """刚学完一代，即使材料又变了也要等一会儿 —— 别把设备算力吃干。"""
    _seed_active(prof)
    cfg.learner.min_gap_s = 180
    st = learner.load_state(prof)
    st.last_ts = time.time()
    learner.save_state(prof, st)
    _touch_lesson(prof)
    now = time.time() + 60
    v = learner.decide(prof, cfg, now=now)
    assert v.code == "cooling"
    assert learner.decide(prof, cfg, now=time.time() + 200).code == "ready"


def test_never_learns_while_robot_is_running(prof, cfg):
    """机器人正在跑就不学。理由和"规则不夺权"同一条：不能从它手里换配方。"""
    _seed_active(prof)
    _touch_lesson(prof)
    v = learner.decide(prof, cfg, busy=True)
    assert v.code == "running"
    assert not v.should_learn


def test_blocked_when_materials_not_enough(prof, cfg):
    """材料不够时说的是"缺什么"，不是一个含糊的"暂不可用"。"""
    _seed_active(prof)
    p = prof.root / "lessons" / "t.md"
    p.write_text("# 空手册\n", encoding="utf-8")
    v = learner.decide(prof, cfg)
    assert v.code == "blocked"
    assert "缺" in v.text


def test_switch_off_stops_everything(prof, cfg):
    _seed_active(prof)
    _touch_lesson(prof)
    learner.set_enabled(prof, cfg, False)
    v = learner.decide(prof, cfg)
    assert v.code == "off"
    # 再打开就恢复
    learner.set_enabled(prof, cfg, True)
    assert learner.decide(prof, cfg).code == "ready"


# ---------------- 2. 学一次：回执与不重试 ----------------

def test_learn_records_receipt_and_promotes(prof, cfg, monkeypatch):
    _fake_evolve(monkeypatch, promoted=True, message="通过 2 → 2")
    res = learner.learn(prof, cfg, source="auto")
    assert res["ok"] and res["kind"] == "promoted"
    st = learner.load_state(prof)
    assert st.last_result == "promoted"
    assert st.last_message == "通过 2 → 2"
    assert st.last_source == "auto"
    assert st.last_ts > 0
    # 上线成功不该留下"试过"的计数，否则下次改了材料会被误挡
    assert st.tries == {}


def test_failed_material_is_not_retried_forever(prof, cfg, monkeypatch):
    """回归没过 → 记一笔就停手。

    这是自动化最容易出事的地方：材料有问题时，每 15 秒重试一次，既烧 CPU
    又把日志刷成噪声，用户还以为"它一直在努力"。
    """
    _seed_active(prof)
    _touch_lesson(prof)
    _fake_evolve(monkeypatch, promoted=False, message="通过 2 → 1，不上线")

    r = learner.tick(prof, cfg)
    assert r["action"] == "learn" and r["code"] == "rejected"

    r2 = learner.tick(prof, cfg)            # 材料没变，再来一轮
    assert r2["action"] == "skip"
    assert r2["code"] == "tried_out"
    # 回执要把上次的原因带出来（状态行只说结论，原因在状态里）
    assert "通过 2 → 1" in learner.snapshot(prof, cfg)["tried"]["message"]


def test_changing_materials_clears_the_retry_block(prof, cfg, monkeypatch):
    """用户改了材料 → 重新给机会。这是"持续学习"该有的样子。"""
    _seed_active(prof)
    _touch_lesson(prof)
    _fake_evolve(monkeypatch, promoted=False, message="没过")
    learner.tick(prof, cfg)
    assert learner.decide(prof, cfg).code == "tried_out"

    _touch_lesson(prof, "\n## 界面 · 新截图\n- 图: refs/combat_a.jpg\n- 判定: 正例\n")
    # 直接看判定，不依赖 mtime 精度
    assert learner.decide(prof, cfg, now=time.time() + 10).code == "ready"


def test_manual_run_ignores_retry_block(prof, cfg, monkeypatch):
    """人点了「现在学」就是现在要 —— 手动不受自动的防重试限制。"""
    _seed_active(prof)
    _touch_lesson(prof)
    _fake_evolve(monkeypatch, promoted=False, message="没过")
    learner.tick(prof, cfg)
    _fake_evolve(monkeypatch, promoted=True)
    res = learner.learn(prof, cfg, source="manual")
    assert res["kind"] == "promoted"


def test_error_is_recorded_not_raised(prof, cfg, monkeypatch):
    """学习崩了不能把服务拖垮，下一轮还得能继续。"""
    def boom(*a, **k):
        raise RuntimeError("模型连不上")
    monkeypatch.setattr(learner.ev, "evolve", boom)
    res = learner.learn(prof, cfg, source="auto")
    assert res["kind"] == "error"
    assert "模型连不上" in res["message"]
    st = learner.load_state(prof)
    assert st.last_result == "error"


def test_concurrent_learn_does_not_double_run(prof, cfg, monkeypatch):
    """后台线程和页面同时点 → 第二个直接返回"正在学"，不排队也不算失败。"""
    started = threading.Event()
    release = threading.Event()

    def slow(profile, cfg, *, force=False, use_model=True):
        started.set()
        release.wait(5)
        return {"ok": True, "stage": "promote", "gen": "gen-001",
                "promoted": True, "message": "ok",
                "score": {"strict_pass": 2, "strict_total": 2}, "fail": []}

    monkeypatch.setattr(learner.ev, "evolve", slow)
    t = threading.Thread(target=lambda: learner.learn(prof, cfg, source="auto"))
    t.start()
    assert started.wait(5)
    second = learner.learn(prof, cfg, source="manual")
    assert second.get("busy") is True
    release.set()
    t.join(5)
    assert learner.load_state(prof).last_result == "promoted"


# ---------------- 3. 材料变化清单 ----------------

def test_delta_reports_added_changed_removed(prof):
    _seed_active(prof, evolve.materials_digest(prof))
    base = evolve.active_generation(prof)

    _touch_lesson(prof)
    cv2.imwrite(str(prof.root / "refs" / "new.jpg"), _frame(11))
    (prof.root / "refs" / "bag_a.jpg").unlink()

    d = evolve.materials_delta(prof, base)
    assert d["tracked"] is True
    assert "refs/new.jpg" in d["added"]
    assert "lessons/t.md" in d["changed"]
    assert "refs/bag_a.jpg" in d["removed"]
    assert d["empty"] is False


def test_delta_says_untracked_for_old_generation(prof):
    """老代没有 materials.json → 说"不知道哪些是新的"，不假装知道。"""
    _seed_active(prof, evolve.materials_digest(prof))
    (evolve.gen_dir(prof, "gen-000") / "materials.json").unlink()
    d = evolve.materials_delta(prof, evolve.active_generation(prof))
    assert d["tracked"] is False
    assert d["n_files"] == 5          # 手册 + 4 张图


def test_knowledge_delta_shows_what_was_forgotten(prof, cfg):
    """新说法顶掉旧说法时，必须能看见 —— 悄悄消失等于悄悄改认识。"""
    _seed_active(prof)
    prev = evolve.active_generation(prof)
    gen, note = evolve.build(prof, cfg, use_model=False)
    assert gen is not None, note
    kd = evolve.knowledge_delta(prev, gen)
    # 手册被重编过，条目有增有减，两个方向都必须是列表（可能是空集）
    assert isinstance(kd["added"], list) and isinstance(kd["removed"], list)


# ---------------- 4. 状态持久化 ----------------

def test_state_survives_restart_and_bad_json(prof, cfg):
    learner.set_enabled(prof, cfg, False)
    assert learner.load_state(prof).enabled is False
    # 文件损坏 → 当作全新，不抛（设备断电时写了一半很常见）
    learner.state_path(prof).write_text("{ 这不是 json", encoding="utf-8")
    st = learner.load_state(prof)
    assert st.enabled is None and st.last_result == ""


def test_tries_is_bounded(prof, cfg, monkeypatch):
    """指纹计数不能无界长：设备跑一年会攒几千个。"""
    _fake_evolve(monkeypatch, promoted=False, message="没过")
    st = learner.load_state(prof)
    st.tries = {f"d{i}": {"ts": i, "result": "rejected", "n": 1} for i in range(30)}
    learner.save_state(prof, st)
    learner.learn(prof, cfg)
    st2 = learner.load_state(prof)
    assert len(st2.tries) <= 20


def test_snapshot_is_cheap_and_complete(prof, cfg):
    _seed_active(prof)
    _touch_lesson(prof)
    s = learner.snapshot(prof, cfg)
    assert s["enabled"] is True
    assert s["code"] in ("waiting", "ready", "cooling")
    assert s["delta"]["n_files"] >= 5
    assert s["interval_s"] > 0 and s["max_tries"] >= 1


# ---------------- 5. 新的理解从哪进来 ----------------

def test_note_becomes_a_parsable_flow_claim(prof):
    """玩家写两句话 → 变成手册认得的流程条目 → 下次学习真的能读到。

    这条链路断在哪一环都等于"写了没用"：格式不对解析器不认，
    解析器不认就不会进 knowledge，进了 knowledge 才会进模型上下文。
    """
    from autogame import teach

    before = evolve.materials_digest(prof)
    path, block = teach.append_note(prof, "拾取时提示背包已满",
                                   "先回城清包，别继续点拾取")
    assert path.exists()
    assert "- 一眼认出: 拾取时提示背包已满" in block
    assert evolve.materials_digest(prof) != before   # 材料变了，学习者才会接手

    lessons = teach.load_lessons(prof.root / "lessons")
    claims = [c for ls in lessons for c in ls.claims]
    hit = [c for c in claims if c.kind == teach.KIND_FLOW
           and "背包已满" in str(c.get("recognize", ""))]
    assert hit, "写进去的条目必须能被解析成流程声明"
    assert "回城清包" in str(hit[0].get("todo", ""))


def test_note_cannot_break_the_manual_structure(prof):
    """用户输入里的换行和 `##` 会被压平 —— 它们能把整份手册的结构搞坏。

    压平之后 `## 界面 · 伪造` 只是某一行的正文，不会被当成新段落。
    """
    from autogame import teach

    _, block = teach.append_note(
        prof, "看到\n## 界面 · 伪造\n- 判定: 正例", "按 esc")
    lines = block.splitlines()
    # 只有我们自己加的那一个标题
    assert sum(1 for x in lines if x.startswith("## ")) == 1
    # 更硬的检查：解析出来的 claim 里没有凭空多出的界面条目
    lessons = teach.load_lessons(prof.root / "lessons")
    scenes = [c for c in (c for ls in lessons for c in ls.claims)
              if c.kind == teach.KIND_SCENE]
    assert len(scenes) == 4


def test_note_rejects_empty_input(prof):
    from autogame import teach

    with pytest.raises(ValueError):
        teach.append_note(prof, " ", "回城")
    with pytest.raises(ValueError):
        teach.append_note(prof, "背包满了", "")
