"""经验闭环：候选 → 复现 → 晋升 / 驳回。

这一组的重点不是"能不能写进手册"，而是**不该写的时候坚决不写**：

- 只出现一次 → 不算规律（一次卡住可能只是网络抖了）
- 两次挤在几秒内 → 还是同一次发作
- 人驳回过的 → 不许隔一个窗口又冒出来
- 晋升要过补丁校验，且写坏了能回滚
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from autogame import experience as xp
from autogame.assistant.patch import PatchStore
from autogame.config import AppConfig
from autogame.experience import ExperienceError, ExperienceStore
from autogame.executor import DryRunExecutor
from autogame.monitor import BAD, WARN, Finding
from autogame.server.hub import Hub
from autogame.vision.pipeline import VisionPipeline

TOKEN = "t"


def _prof(tmp_path):
    root = tmp_path / "profiles" / "u"
    (root / "lessons").mkdir(parents=True)
    (root / "templates").mkdir()
    (root / "lessons" / "t.md").write_text(
        "# 手册\n\n## 流程 · 打一圈\n- 一眼认出: 有怪\n- 该做什么: 点技能\n",
        encoding="utf-8")
    (root / "profile.yaml").write_text(
        "id: u\nname: 单元\nstatus: partial\nreference_resolution: [360, 200]\nconfig: {}\n",
        encoding="utf-8")
    from autogame.profile import load_profile
    return load_profile(root)


def _finding(code="repeat", **metric):
    base = {"repeat": 14, "key": "mouse_click(640,360)"}
    base.update(metric)
    return Finding(code, BAD, "卡住了", "20 秒里同一个动作发了 14 次", base,
                   ts=time.time())


def _store(prof, **kw):
    return ExperienceStore(prof, min_seen=kw.pop("min_seen", 2),
                           min_span_s=kw.pop("min_span_s", 0.0), **kw)


# ═══════════ 1. 观察：只累计证据，不写手册 ═══════════

def test_observe_creates_a_candidate_with_actionable_text(tmp_path):
    prof = _prof(tmp_path)
    st = _store(prof)
    row = st.observe(_finding(), now=1000.0)
    assert row and row["code"] == "repeat" and row["status"] == xp.CANDIDATE
    assert "mouse_click(640,360)" in row["when"]
    assert len(row["do"]) > 10, "不能只写一句空话"
    assert row["seen"] == 1
    # 候选阶段一个字节都不能写进手册
    assert (prof.root / "lessons" / "t.md").read_text(encoding="utf-8").count("卡住") == 0


def test_same_window_is_not_counted_twice(tmp_path):
    prof = _prof(tmp_path)
    st = _store(prof)
    st.observe(_finding(), now=1000.0)
    st.observe(_finding(), now=1001.0)          # 5 秒内
    assert st.items()[0]["seen"] == 1, "同一个窗口里连报两次还是同一次发作"


def test_later_occurrence_counts_and_updates_numbers(tmp_path):
    prof = _prof(tmp_path)
    st = _store(prof)
    st.observe(_finding(repeat=14), now=1000.0)
    row = st.observe(_finding(repeat=21), now=2000.0)
    assert row["seen"] == 2
    assert "21" in row["when"], "模板里应该显示最近一次的实测值，不是第一次那个数"
    assert len(row["evidence"]) == 2


def test_unknown_codes_are_ignored(tmp_path):
    prof = _prof(tmp_path)
    st = _store(prof)
    assert st.observe(Finding("something_new", BAD, "?", "?"), now=1000.0) is None
    assert st.items() == []


def test_rejected_candidate_does_not_come_back(tmp_path):
    prof = _prof(tmp_path)
    st = _store(prof, min_seen=1)
    row = st.observe(_finding(), now=1000.0)
    st.reject(row["id"], "这次是网络抖了")
    st.observe(_finding(), now=5000.0)
    assert [r["status"] for r in st.items()] == [xp.REJECTED]


def test_evidence_is_capped(tmp_path):
    prof = _prof(tmp_path)
    st = _store(prof)
    for i in range(20):
        st.observe(_finding(), now=1000.0 + i * 100)
    row = st.items()[0]
    assert row["seen"] == 20
    assert len(row["evidence"]) <= xp.MAX_EVIDENCE


def test_store_survives_a_corrupt_file(tmp_path):
    prof = _prof(tmp_path)
    p = xp.store_path(prof)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ 半截", encoding="utf-8")
    st = _store(prof)
    assert st.observe(_finding(), now=1000.0) is not None
    assert list(p.parent.glob("*.corrupt-*")), "坏文件要留档，不能静默丢掉"


def test_two_stores_do_not_lose_each_others_rows(tmp_path):
    """两个进程（页面 + 后台）同时记候选：读-改-写必须在锁里。"""
    prof = _prof(tmp_path)
    a, b = _store(prof), _store(prof)
    a.observe(Finding("repeat", BAD, "卡", "一", {"repeat": 12, "key": "k"}), now=1000.0)
    b.observe(Finding("model_burn", BAD, "烧", "二", {"now": 1.2, "base": 0}), now=1000.0)
    assert {r["code"] for r in a.items()} == {"repeat", "model_burn"}


# ═══════════ 2. 晋升：够了才让写，写了能退 ═══════════

def test_promote_requires_enough_occurrences(tmp_path):
    prof = _prof(tmp_path)
    st = _store(prof)
    row = st.observe(_finding(), now=1000.0)
    with pytest.raises(ExperienceError, match="只出现过 1 次"):
        st.promote(row["id"], PatchStore(prof))
    assert (prof.root / "lessons" / "t.md").read_text(encoding="utf-8").count("卡住") == 0


def test_promote_requires_a_real_gap_between_occurrences(tmp_path):
    """20 秒窗口连报 3 次还是同一次发作 —— 不能当规律。"""
    prof = _prof(tmp_path)
    st = _store(prof, min_seen=2, min_span_s=600)
    row = st.observe(_finding(), now=1000.0)
    st.observe(_finding(), now=1020.0)
    with pytest.raises(ExperienceError, match="同一次发作"):
        st.promote(row["id"], PatchStore(prof))


def test_promote_writes_the_lesson_and_marks_it(tmp_path):
    prof = _prof(tmp_path)
    st = _store(prof, min_seen=2)
    row = st.observe(_finding(), now=1000.0)
    st.observe(_finding(), now=2000.0)
    res = st.promote(row["id"], PatchStore(prof))
    body = (prof.root / "lessons" / "t.md").read_text(encoding="utf-8")
    assert "先松开所有键" in body
    assert res["candidate"]["status"] == xp.PROMOTED
    assert res["candidate"]["patch_id"] == res["patch"]["id"]
    assert res["patch"]["kind"] == "lesson"


def test_promoted_candidate_can_be_rolled_back(tmp_path):
    prof = _prof(tmp_path)
    st = _store(prof, min_seen=1)
    row = st.observe(_finding(), now=1000.0)
    res = st.promote(row["id"], PatchStore(prof))
    before = "# 手册\n\n## 流程 · 打一圈\n- 一眼认出: 有怪\n- 该做什么: 点技能\n"
    assert (prof.root / "lessons" / "t.md").read_text(encoding="utf-8") != before
    PatchStore(prof).rollback(res["patch"]["id"])
    assert (prof.root / "lessons" / "t.md").read_text(encoding="utf-8") == before


def test_promote_twice_is_refused(tmp_path):
    prof = _prof(tmp_path)
    st = _store(prof, min_seen=1)
    row = st.observe(_finding(), now=1000.0)
    st.promote(row["id"], PatchStore(prof))
    with pytest.raises(ExperienceError):
        st.promote(row["id"], PatchStore(prof))


def test_cannot_reject_something_already_in_the_handbook(tmp_path):
    prof = _prof(tmp_path)
    st = _store(prof, min_seen=1)
    row = st.observe(_finding(), now=1000.0)
    st.promote(row["id"], PatchStore(prof))
    with pytest.raises(ExperienceError, match="已经写进手册"):
        st.reject(row["id"])


def test_missing_candidate_is_reported(tmp_path):
    st = _store(_prof(tmp_path))
    with pytest.raises(ExperienceError, match="没有这条候选"):
        st.promote("x-nope", None)
    with pytest.raises(ExperienceError, match="没有这条候选"):
        st.reject("x-nope")


def test_public_view_hides_internals_but_shows_why_blocked(tmp_path):
    prof = _prof(tmp_path)
    st = _store(prof, min_seen=3)
    st.observe(_finding(), now=1000.0)
    row = st.as_public()[0]
    assert row["status"] == xp.CANDIDATE and row["ready"] is False
    assert "至少 3 次" in row["blocked_by"]
    assert "evidence" in row


# ═══════════ 3. 接进体检与页面 ═══════════

def _hub(prof=None):
    from autogame import assistant as asst_mod

    cfg = AppConfig()
    cfg.server.token = TOKEN
    cfg.server.event_log = ""
    cfg.executor.min_action_interval_ms = 0
    hub = Hub(cfg, DryRunExecutor(cfg.executor), VisionPipeline(cfg.vision))
    asst_mod.build(cfg, prof, hub)
    return cfg, hub


def test_health_findings_become_candidates(tmp_path):
    prof = _prof(tmp_path)
    cfg, hub = _hub(prof)
    hub.experience = _store(prof, min_seen=1)
    hub.health.observe = lambda **kw: [_finding()]
    hub.health_tick(tick_s=0.02)
    rows = hub.experience.items()
    assert rows and rows[0]["code"] == "repeat"


def test_warn_level_findings_also_become_candidates(tmp_path):
    """恢复空转只是提醒（不回滚），但它照样是"值得记下来"的经验。"""
    prof = _prof(tmp_path)
    cfg, hub = _hub(prof)
    hub.experience = _store(prof, min_seen=1)
    hub.health.observe = lambda **kw: [
        Finding("recovery_spin", WARN, "空转", "恢复 6 次没捡到东西",
                {"now": 0.6, "base": 0.1, "picks": 0})]
    hub.health_tick(tick_s=0.02)
    assert hub.experience.items()[0]["code"] == "recovery_spin"


def test_http_experience_endpoints(tmp_path):
    from fastapi.testclient import TestClient

    from autogame.server.app import create_app

    prof = _prof(tmp_path)
    cfg, hub = _hub(prof)
    app = create_app(cfg, hub, "<h1>x</h1>", profile=prof)
    # create_app 会按默认参数重建一份候选账本；用例里要的是"已经够格"的状态
    hub.experience = _store(prof, min_seen=1)
    hub.experience.observe(_finding(), now=time.time())
    c = TestClient(app)

    d = c.get("/api/experience", headers={"X-Token": TOKEN}).json()
    assert d["ok"] and len(d["items"]) == 1
    cid = d["items"][0]["id"]

    r = c.post("/api/experience/promote", headers={"X-Token": TOKEN},
               json={"id": cid}).json()
    assert r["ok"], r
    assert "先松开所有键" in (prof.root / "lessons" / "t.md").read_text(encoding="utf-8")

    d2 = c.get("/api/experience", headers={"X-Token": TOKEN}).json()
    assert d2["items"][0]["status"] == "promoted"


def test_http_reject(tmp_path):
    from fastapi.testclient import TestClient

    from autogame.server.app import create_app

    prof = _prof(tmp_path)
    cfg, hub = _hub(prof)
    c = TestClient(create_app(cfg, hub, "<h1>x</h1>", profile=prof))
    hub.experience = _store(prof, min_seen=1)
    hub.experience.observe(_finding(), now=time.time())
    cid = c.get("/api/experience", headers={"X-Token": TOKEN}).json()["items"][0]["id"]
    r = c.post("/api/experience/reject", headers={"X-Token": TOKEN},
               json={"id": cid, "why": "这次是网络抖了"}).json()
    assert r["ok"] and r["item"]["status"] == "rejected"


def test_http_experience_without_profile_is_explicit(tmp_path):
    from fastapi.testclient import TestClient

    from autogame.server.app import create_app

    cfg, hub = _hub(None)
    c = TestClient(create_app(cfg, hub, "<h1>x</h1>"))
    d = c.get("/api/experience", headers={"X-Token": TOKEN}).json()
    assert d["ok"] is False and "profile" in d["error"]


def test_every_template_renders_with_the_metrics_we_actually_pass(tmp_path):
    """模板占位符必须都是 finding 里真有的字段。

    写错一个字段不会报错 —— `.format` 会给个 0，于是手册里出现
    "进入恢复后 0 秒内没捡到东西"这种看着像话、其实没有信息量的句子。
    """
    prof = _prof(tmp_path)
    st = _store(prof, min_seen=1)
    cases = {
        "repeat": {"repeat": 14, "key": "mouse_click(640,360)"},
        "recovery_spin": {"now": 0.6, "base": 0.1, "picks": 0},
        "slow": {"now_ms": 820.0, "base_ms": 40.0},
        "model_burn": {"now": 1.2, "base": 0.0},
    }
    for code, metric in cases.items():
        st.observe(Finding(code, BAD, "?", "详情", metric), now=1000.0)
    rows = {r["code"]: r for r in st.items()}
    assert set(rows) == set(cases)
    for code, row in rows.items():
        when = row["when"]
        assert "{" not in when and "}" not in when, f"{code} 的模板没填完：{when}"
        assert "None" not in when
        assert any(ch.isdigit() for ch in when), f"{code} 的条件里没有任何数值：{when}"



# ═══════════ 4. 经验长成规则：会命中才叫学会 ═══════════

def _rule_gen(prof):
    """给 profile 造一代 active 配方（规则文件里先有一条旧规则）。"""
    import yaml as _yaml

    gd = prof.root / "evolve" / "gen-000"
    gd.mkdir(parents=True, exist_ok=True)
    (gd / "gen.yaml").write_text(
        _yaml.safe_dump({"id": "gen-000", "status": "active"}), encoding="utf-8")
    rules = gd / "rules.yaml"
    rules.write_text(_yaml.safe_dump(
        {"version": 1, "rules": [
            {"id": "r_old", "when": {"fact": "pillars", "op": ">=", "value": 1},
             "then": {"action": "pickup"}}]},
        allow_unicode=True, sort_keys=False), encoding="utf-8")
    (prof.root / "evolve" / "active").write_text("gen-000\n", encoding="utf-8")
    return rules


def _cond(rule, fact):
    for c in (rule.get("when") or {}).get("all") or []:
        if c.get("fact") == fact:
            return c
    return None


def test_candidate_carries_a_rule_that_can_actually_match(tmp_path):
    """候选除了"该怎么做"，还要带一条**能过校验**的规则。

    这条用例守的是"不硬造"：凡是写进候选的规则，都必须能过 validate_rules，
    而且条件里只能出现真的会产出的事实字段 —— 否则它会被写进 rules.yaml、
    在页面上显示成"已经学会"，然后一次都不命中。
    """
    from autogame import evolve as ev

    prof = _prof(tmp_path)
    st = _store(prof, min_seen=1)
    st.observe(_finding(), now=1000.0)
    rule = st.items()[0]["rule"]
    assert rule["trigger"] == "stall", "卡住类规则必须不受空闲门槛限制，否则轮不到它"
    assert rule["then"]["action"] == "explore"
    ok, errors = ev.validate_rules({"rules": [rule]})
    assert not errors and ok, errors
    used = {c["fact"] for c in rule["when"]["all"]}
    assert used <= set(ev.FACTS), f"用了不存在的事实：{used - set(ev.FACTS)}"
    assert _cond(rule, "repeat_key")["value"] == "mouse_click(640,360)"


def test_rule_threshold_follows_the_latest_measurement(tmp_path):
    """门槛跟着最近一次实测走：按最轻的那次定标准等于没定。"""
    prof = _prof(tmp_path)
    st = _store(prof, min_seen=1)
    st.observe(_finding(repeat=12), now=1000.0)
    low = _cond(st.items()[0]["rule"], "repeat")["value"]
    st.observe(_finding(repeat=20), now=2000.0)
    high = _cond(st.items()[0]["rule"], "repeat")["value"]
    assert high > low
    assert low < 12, "门槛要比体检告警线更早，等它按满 12 次再救已经白按了"


def test_slow_and_burn_are_lesson_only_with_a_reason(tmp_path):
    """写不出规则的两种要把理由写出来，不能假装漏做了。"""
    prof = _prof(tmp_path)
    st = _store(prof, min_seen=1)
    st.observe(Finding("slow", BAD, "慢", "每 tick 变慢",
                       {"now_ms": 820.0, "base_ms": 40.0}), now=1000.0)
    st.observe(Finding("model_burn", BAD, "烧", "每秒问模型",
                       {"now": 1.2, "base": 0.0}), now=1001.0)
    for row in st.items():
        assert row["rule"] == {}, row["code"]
        assert len(row["rule_blocked"]) > 8, row["code"]


def test_repeat_of_the_escape_action_gets_no_rule(tmp_path):
    """"卡住的动作就是走开"—— 再让它走开一遍帮不上忙，这条不生成规则。"""
    prof = _prof(tmp_path)
    st = _store(prof, min_seen=1)
    st.observe(_finding(key="explore()"), now=1000.0)
    row = st.items()[0]
    assert row["rule"] == {}
    assert "再让它做一遍同样的动作" in row["rule_blocked"]


def test_promote_writes_both_the_lesson_and_the_rule(tmp_path):
    import yaml

    prof = _prof(tmp_path)
    rules = _rule_gen(prof)
    st = _store(prof, min_seen=2)
    row = st.observe(_finding(), now=1000.0)
    st.observe(_finding(), now=2000.0)
    res = st.promote(row["id"], PatchStore(prof))

    assert "先松开所有键" in (prof.root / "lessons" / "t.md").read_text(encoding="utf-8")
    assert res["rule_patch"] and res["rule_patch"]["kind"] == "rule"
    assert not res["rule_error"]
    doc = yaml.safe_load(rules.read_text(encoding="utf-8"))
    ids = [r["id"] for r in doc["rules"]]
    assert "xp-repeat" in ids and "r_old" in ids, "原有规则不能被顶掉"
    got = next(r for r in doc["rules"] if r["id"] == "xp-repeat")
    assert got["trigger"] == "stall" and got["then"]["action"] == "explore"
    assert res["candidate"]["rule_patch_id"] == res["rule_patch"]["id"]


def test_promote_without_a_generation_still_writes_the_lesson(tmp_path):
    """还没有任何一代配方时，规则没地方放 —— 手册照写，原因带出来。"""
    prof = _prof(tmp_path)
    st = _store(prof, min_seen=1)
    row = st.observe(_finding(), now=1000.0)
    res = st.promote(row["id"], PatchStore(prof))
    assert res["rule_patch"] is None
    assert "还没有任何一代配方" in res["rule_error"]
    assert "先松开所有键" in (prof.root / "lessons" / "t.md").read_text(encoding="utf-8")
    assert res["candidate"]["status"] == xp.PROMOTED


def test_learned_rule_can_be_rolled_back(tmp_path):
    prof = _prof(tmp_path)
    rules = _rule_gen(prof)
    before = rules.read_text(encoding="utf-8")
    st = _store(prof, min_seen=1)
    row = st.observe(_finding(), now=1000.0)
    res = st.promote(row["id"], PatchStore(prof))
    assert "xp-repeat" in rules.read_text(encoding="utf-8")
    PatchStore(prof).rollback(res["rule_patch"]["id"])
    assert rules.read_text(encoding="utf-8") == before


def test_learned_rules_are_listed_for_the_page(tmp_path):
    prof = _prof(tmp_path)
    _rule_gen(prof)
    st = _store(prof, min_seen=1)
    row = st.observe(_finding(), now=1000.0)
    assert st.rules_learned() == []
    st.promote(row["id"], PatchStore(prof))
    learned = st.rules_learned()
    assert len(learned) == 1 and learned[0]["rule"]["id"] == "xp-repeat"
    assert st.as_public()[0]["rule_patch_id"]


def test_http_promote_reloads_the_rule_into_the_running_process(tmp_path):
    """页面点"记进手册" → 手册写了、规则写了、**正在跑的那份也换了**。

    少了最后一步，页面显示"已经学会"，可程序照着旧规则跑到下次重启 ——
    和"只画了个按钮"没有区别。
    """
    from fastapi.testclient import TestClient

    from autogame.server.app import create_app

    prof = _prof(tmp_path)
    rules = _rule_gen(prof)
    cfg, hub = _hub(prof)
    cfg.decision.arbiter.rules_path = str(rules)
    app = create_app(cfg, hub, "<h1>x</h1>", profile=prof)
    hub.experience = _store(prof, min_seen=1)
    hub.experience.observe(_finding(), now=time.time())
    c = TestClient(app)

    cid = c.get("/api/experience", headers={"X-Token": TOKEN}).json()["items"][0]["id"]
    r = c.post("/api/experience/promote", headers={"X-Token": TOKEN},
               json={"id": cid}).json()
    assert r["ok"], r
    assert "规则" in r["message"]

    live_ids = [x["id"] for x in hub.fsm.arbiter.rules.rules]
    assert "xp-repeat" in live_ids, f"规则写了但没重载，跑的还是 {live_ids}"
    assert "当场重载" in r["live"], r["live"]


def test_promoting_the_same_kind_twice_updates_the_rule_instead_of_piling_up(tmp_path):
    """同一个毛病学两次 → 改门槛，不是往 rules.yaml 里再塞一条。"""
    import yaml

    prof = _prof(tmp_path)
    rules = _rule_gen(prof)
    st = _store(prof, min_seen=1)
    first = st.observe(_finding(repeat=12), now=1000.0)
    st.promote(first["id"], PatchStore(prof))

    second = st.observe(_finding(repeat=30), now=5000.0)
    assert second["id"] != first["id"], "已晋升的那条不再是候选，新证据应另起一条"
    st.promote(second["id"], PatchStore(prof))

    doc = yaml.safe_load(rules.read_text(encoding="utf-8"))
    ids = [r["id"] for r in doc["rules"]]
    assert ids.count("xp-repeat") == 1, ids
    got = next(r for r in doc["rules"] if r["id"] == "xp-repeat")
    lvl = next(c["value"] for c in got["when"]["all"] if c["fact"] == "repeat")
    assert lvl >= 20, lvl


# ═══════════ 5. 治"负担"的结论：改配置，不改动作 ═══════════

def test_slow_becomes_a_config_change_not_a_rule(tmp_path):
    """变慢这件事动作清单治不了 —— 9 个动作全是游戏内行为，没有"减负"这一项。

    硬写规则只会在慢的时候让角色随便走两步，不但不治，还添乱。
    所以它必须变成配置补丁，并且**明确说明为什么不写规则**。
    """
    prof = _prof(tmp_path)
    st = _store(prof, min_seen=1)
    row = st.observe(Finding("slow", BAD, "变慢了", "每 tick 耗时从 40ms 涨到 820ms",
                             {"now_ms": 820.0, "base_ms": 40.0}, ts=time.time()),
                     now=1000.0)
    assert row["rule"] == {}
    assert row["config"] == {"decision": {"tick_hz": 0.5}}
    assert row["kind"] == "lesson+config"
    assert "减轻负担" in row["rule_blocked"], row["rule_blocked"]
    assert row["config_why"]


def test_model_burn_raises_the_model_cooldown(tmp_path):
    prof = _prof(tmp_path)
    st = _store(prof, min_seen=1)
    row = st.observe(Finding("model_burn", BAD, "烧模型", "每秒问模型 1.2 次",
                             {"now": 1.2, "base": 0.0}, ts=time.time()), now=1000.0)
    assert row["rule"] == {}
    assert row["config"]["decision"]["arbiter"]["cooldown_s"] == 60.0
    assert "冷却" in row["config_why"]


def test_slow_with_no_actual_slowdown_does_not_change_config(tmp_path):
    """数据对不上就别动配置 —— 体检的基线是滚动的，抖一下很正常。"""
    prof = _prof(tmp_path)
    st = _store(prof, min_seen=1)
    row = st.observe(Finding("slow", BAD, "变慢了", "其实没慢",
                             {"now_ms": 40.0, "base_ms": 40.0}, ts=time.time()),
                     now=1000.0)
    assert row["config"] == {}
    assert "并不比基线" in row["config_blocked"]


def test_promote_applies_the_config_patch_for_real(tmp_path):
    """晋升配置补丁：overlay 有内容、内存里的 cfg 真的变了、能回滚。"""
    import yaml

    from autogame.config import AppConfig, runtime_overlay_path
    from autogame.server.settings import apply_patch

    prof = _prof(tmp_path)
    cfg = AppConfig()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("", encoding="utf-8")
    st = _store(prof, min_seen=1)
    row = st.observe(Finding("slow", BAD, "变慢了", "慢",
                             {"now_ms": 820.0, "base_ms": 40.0}, ts=time.time()),
                     now=1000.0)
    patches = PatchStore(prof, cfg, config_path=str(cfg_path))
    res = st.promote(row["id"], patches)

    assert res["rule_patch"] is None
    assert res["config_patch"] and res["config_patch"]["kind"] == "config"
    assert not res["config_error"]

    ov = runtime_overlay_path(cfg_path)
    assert ov.exists() and "tick_hz" in ov.read_text(encoding="utf-8")
    doc = yaml.safe_load(ov.read_text(encoding="utf-8"))
    assert doc["decision"]["tick_hz"] == 0.5

    # 写到文件还不够 —— 内存里那份也得变，否则程序跑的还是旧参数
    assert cfg.decision.tick_hz == 1.0
    apply_patch(cfg, res["candidate"]["config"])
    assert cfg.decision.tick_hz == 0.5

    patches.rollback(res["config_patch"]["id"])
    assert not (ov.read_text(encoding="utf-8") or "").strip()


def test_config_patch_failure_does_not_lose_the_lesson(tmp_path):
    """配置补丁坏了，手册照写 —— 手册那部分已经是有价值的知识。"""
    from autogame.config import AppConfig

    prof = _prof(tmp_path)
    cfg = AppConfig()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("", encoding="utf-8")
    st = _store(prof, min_seen=1)
    row = st.observe(Finding("model_burn", BAD, "烧", "烧",
                             {"now": 1.2, "base": 0.0}, ts=time.time()), now=1000.0)
    patches = PatchStore(prof, cfg, config_path=str(cfg_path))

    def _boom(*a, **k):  # noqa: ARG001
        raise RuntimeError("盘满了")

    original = patches.preview

    def preview(kind, payload, **kw):
        if kind == "config":
            _boom()
        return original(kind, payload, **kw)

    patches.preview = preview  # type: ignore[method-assign]
    res = st.promote(row["id"], patches)
    assert res["candidate"]["status"] == xp.PROMOTED
    assert "盘满了" in res["config_error"]
    assert "停下来把当时的界面" in (prof.root / "lessons" / "t.md").read_text(encoding="utf-8")


def test_public_row_exposes_the_config_advice(tmp_path):
    prof = _prof(tmp_path)
    st = _store(prof, min_seen=1)
    st.observe(Finding("slow", BAD, "慢", "慢",
                       {"now_ms": 820.0, "base_ms": 40.0}, ts=time.time()), now=1000.0)
    pub = st.as_public()[0]
    assert pub["config"] == {"decision": {"tick_hz": 0.5}}
    assert pub["config_why"] and pub["kind"] == "lesson+config"


def test_http_promote_says_what_it_changed(tmp_path):
    from fastapi.testclient import TestClient

    from autogame.config import runtime_overlay_path
    from autogame.server.app import create_app

    prof = _prof(tmp_path)
    cfg, hub = _hub(prof)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("", encoding="utf-8")
    # 传 config_path：否则 overlay 会落到数据根，测试就会往仓库里写文件
    app = create_app(cfg, hub, "<h1>x</h1>", config_path=str(cfg_path), profile=prof)
    hub.experience = _store(prof, min_seen=1)
    hub.experience.observe(Finding("slow", BAD, "慢", "慢",
                                   {"now_ms": 820.0, "base_ms": 40.0},
                                   ts=time.time()), now=time.time())
    c = TestClient(app)
    cid = c.get("/api/experience", headers={"X-Token": TOKEN}).json()["items"][0]["id"]
    r = c.post("/api/experience/promote", headers={"X-Token": TOKEN},
               json={"id": cid}).json()
    assert r["ok"], r
    assert r["config_patch"]
    assert "改了配置" in r["message"]
    assert r["live"], "配置写完要当场套进内存，否则等于没改"
    assert cfg.decision.tick_hz == 0.5
    _ = runtime_overlay_path  # 路径由 PatchStore 决定，这里只验内存
