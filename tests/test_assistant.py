"""对话助手：能不能停下来，比能不能聊天重要 100 倍。

这组用例钉的是四件事，按重要性排：

1. **本地命令真的会停**（不等模型、把任务/FSM/按键一起停）
2. **认不准就不认**（"别停" 不是 "停"，长句多半是在讨论）
3. **模型只提案**（不在白名单的动作一律无效；改规则必须先预览再确认）
4. **改坏了能退回**（每次写盘都有原件，撤销真的恢复原样）

模型本身在这些用例里是假的：真调一次要几秒到几十秒，而且会把"控制链路对不对"
和"模型今天心情如何"混在一起。真实链路另外用 `--live` 手工验。
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi.testclient import TestClient

from autogame import assistant as asst_mod
from autogame.assistant import brain as brain_mod
from autogame.assistant import localintent as li
from autogame.assistant.context import RecentLog
from autogame.assistant.dialog import Dialog
from autogame.assistant.modes import Mode, ModeManager, install_gate
from autogame.assistant.patch import KIND_CONFIG, KIND_LESSON, PatchError, PatchStore
from autogame.config import AppConfig
from autogame.decision.fsm import State
from autogame.executor import DryRunExecutor
from autogame.executor import ExecutorConfig
from autogame.protocol import Command
from autogame.server.app import create_app
from autogame.server.hub import Hub
from autogame.vision.pipeline import VisionPipeline

TOKEN = "t"


# ═══════════════════════ 1. 本地意图：认不准就不认 ═══════════════════════

@pytest.mark.parametrize("text,kind", [
    ("停", li.STOP), ("先停下", li.STOP), ("停下！", li.STOP),
    ("停止操作", li.STOP), ("住手", li.STOP),
    ("继续", li.RESUME), ("接着跑", li.RESUME),
    ("只看不操作", li.OBSERVE), ("别操作", li.OBSERVE), ("现在只观察", li.OBSERVE),
    ("自动模式", li.AUTO),
    ("确认", li.CONFIRM), ("取消", li.CANCEL), ("不要了", li.CANCEL),
    ("撤销", li.UNDO), ("回退上一版", li.UNDO), ("把刚才的修改撤销", li.UNDO),
    ("现在什么情况", li.STATUS),
])
def test_local_commands_are_recognized(text, kind):
    assert li.parse(text).kind == kind


@pytest.mark.parametrize("text", [
    "别停",                       # 否定：正好反了
    "不要停止",                   # 同上
    "怎么停下来着",                # 在问问题
    "这个怪为什么停在那",           # 在讨论
    "停止刷图吗",                  # 疑问
    "刚才为什么判断错了？下次遇到一样的画面别这么处理。",
    "好累",                       # 单字"好"不能当确认
])
def test_ambiguous_sentences_are_not_commands(text):
    """宁可不认 —— 把"别停"当成"停"是这套系统里最危险的一次误判。"""
    it = li.parse(text)
    assert it.kind == li.UNKNOWN, it


def test_pause_duration_is_parsed():
    assert li.parse("暂停十分钟").arg == 600
    assert li.parse("暂停半小时").arg == 1800
    assert li.parse("暂停 30 秒").arg == 30
    assert li.parse("暂停").arg is None


def test_ordinal_choice():
    assert li.parse("2").arg == 1
    assert li.parse("第二个").arg == 1
    assert li.parse("选1").arg == 0


def test_normalize_handles_fullwidth_and_space():
    assert li.parse("　停　下！").kind == li.STOP


# ═══════════════════════ 2. 模式与闸门 ═══════════════════════

def test_mode_defaults_to_coop_and_allows_action():
    m = ModeManager()
    assert m.mode is Mode.COOP
    assert m.allows_action() and m.allows_decision()


def test_observe_decides_but_never_acts():
    m = ModeManager()
    m.set(Mode.OBSERVE, reason="测试")
    assert not m.allows_action()
    assert m.allows_decision(), "观察模式要继续决策，否则没有可复盘的东西"


def test_paused_blocks_even_decision():
    m = ModeManager()
    m.pause(reason="测试")
    assert not m.allows_action() and not m.allows_decision()


def test_pause_expires_back_to_previous_mode():
    m = ModeManager()
    m.set(Mode.AUTO, reason="测试")
    m.pause(0.05, reason="暂停一小会")
    assert m.mode is Mode.PAUSED
    time.sleep(0.08)
    assert m.mode is Mode.AUTO, "暂停到点应该回到暂停前那个模式，而不是一律自动"


def test_gate_blocks_actions_but_lets_stop_all_through():
    cfg = ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000)
    ex = DryRunExecutor(cfg)
    m = ModeManager()
    install_gate(ex, m)

    m.set(Mode.OBSERVE, reason="测试")
    assert ex.execute(Command(op="key_tap", key="a")) is False
    assert ex.suppressed == 1
    # 被拦下也要留痕：复盘时要能说"想做但没做"
    assert ex.journal[-1]["suppressed"] is True

    # stop_all 是"把已经按下的松开"，必须永远放行
    assert ex.execute(Command(op="stop_all")) is True

    m.set(Mode.AUTO, reason="测试")
    assert ex.execute(Command(op="key_tap", key="a")) is True


# ═══════════════════════ 3. 现场记录 ═══════════════════════

def test_recent_log_brief_is_compact_and_relative():
    r = RecentLog(window_s=60)
    r.add("vision", "画面=游戏内", 光柱=2)
    r.add("state", "APPROACHING→COOLDOWN")
    text = r.brief(now=time.time() + 12.4)          # 假装过了 12.4 秒
    assert "t-12.4s" in text and "画面=游戏内" in text and "光柱=2" in text
    assert text.count("\n") == 1, "一条记录一行，别把模型淹了"

    # 刚记下的那一条是"现在"，不能写成 t+0.0s —— 那等于告诉模型"这是未来的事"
    fresh = r.brief()
    assert "现在" in fresh and "t+" not in fresh


def test_recent_log_drops_old_items():
    r = RecentLog(window_s=0.01)
    r.add("vision", "旧的")
    time.sleep(0.03)
    assert r.items() == []
    assert "没有任何记录" in r.brief()


# ═══════════════════════ 4. 多轮：确认指的是哪件事 ═══════════════════════

def test_confirm_matches_pending_question():
    d = Dialog()
    d.ask("要现在应用吗？", options=["确认", "取消"], kind="confirm")
    assert d.resolve("确认")["kind"] == "confirm"
    assert d.resolve("算了")["kind"] == "cancel"


def test_choice_by_number_or_by_text():
    d = Dialog()
    d.ask("要改哪一个？", options=["触发条件", "执行动作"])
    assert d.resolve("第二个")["option"] == 1
    assert d.resolve("执行动作")["option"] == 1


def test_stale_question_expires_instead_of_applying():
    """十分钟前问"要不要应用"，回来一句"确认"不能把补丁打上。"""
    d = Dialog()
    p = d.ask("要现在应用吗？", options=["确认", "取消"], kind="confirm")
    p.created -= 3600
    r = d.resolve("确认")
    assert r["kind"] == "expired"
    assert d.resolve("确认")["matched"] is False, "过期一次之后就不该再认"


# ═══════════════════════ 5. 改规则：预览 / 校验 / 回滚 ═══════════════════════

@pytest.fixture()
def prof(tmp_path):
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


def test_lesson_patch_preview_apply_rollback(prof, tmp_path):
    store = PatchStore(prof)
    pv = store.preview(KIND_LESSON, {"when": "没血了", "do": "先往后撤"})
    assert pv["ok"], pv["checks"]
    assert "+- 一眼认出: 没血了" in pv["diff"]
    assert "没血了" not in (prof.root / "lessons" / "t.md").read_text(encoding="utf-8"), \
        "预览不能改文件"

    rec = store.apply(KIND_LESSON, {"when": "没血了", "do": "先往后撤"},
                      stamp=pv["stamp"], note="测试")
    body = (prof.root / "lessons" / "t.md").read_text(encoding="utf-8")
    assert "没血了" in body

    store.rollback(rec["id"])
    assert (prof.root / "lessons" / "t.md").read_text(encoding="utf-8") == \
        "# 手册\n\n## 流程 · 打一圈\n- 一眼认出: 有怪\n- 该做什么: 点技能\n"


def test_patch_refuses_when_target_changed_after_preview(prof):
    """看了三分钟才点确认 —— 这期间文件被别人改过，就必须重看，不能硬写。"""
    store = PatchStore(prof)
    pv = store.preview(KIND_LESSON, {"when": "没血了", "do": "先往后撤"})
    target = Path(pv["target"])
    target.write_text(target.read_text(encoding="utf-8") + "\n## 别人加的\n",
                      encoding="utf-8")
    with pytest.raises(PatchError):
        store.apply(KIND_LESSON, {"when": "没血了", "do": "先往后撤"},
                    stamp=pv["stamp"])


def test_bad_config_patch_fails_validation(tmp_path):
    cfg = AppConfig()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("decision:\n  tick_hz: 1.0\n", encoding="utf-8")
    store = PatchStore(None, cfg, config_path=str(cfg_path))
    pv = store.preview(KIND_CONFIG, {"decision": {"tick_hz": -5}}, text="太快点")
    assert not pv["ok"]
    assert any(not c["ok"] for c in pv["checks"])


# ═══════════════════════ 6. 主流程（假模型） ═══════════════════════

def _rig(prof=None):
    cfg = AppConfig()
    cfg.server.token = TOKEN
    cfg.executor.min_action_interval_ms = 0
    cfg.executor.max_actions_per_sec = 1000
    cfg.assistant.enabled = False        # 用例里不连真模型
    ex = DryRunExecutor(cfg.executor)
    hub = Hub(cfg, ex, VisionPipeline(cfg.vision))
    a = asst_mod.build(cfg, prof, hub)
    return cfg, hub, ex, a


def test_stop_command_really_stops():
    cfg, hub, ex, a = _rig()
    hub.fsm.start()
    assert hub.fsm.state is not State.PAUSED

    res = a.handle("先停下")
    assert res["mode"]["mode"] == "paused"
    assert not a.modes.allows_action()
    assert ex.journal[-1]["op"] == "stop_all", "停下时必须把按住的键松开"
    assert hub.fsm.state is State.PAUSED


def test_resume_wakes_the_engine():
    cfg, hub, ex, a = _rig()
    a.handle("停")
    assert hub.fsm.state is State.PAUSED
    a.handle("继续")
    assert hub.fsm.state is not State.PAUSED
    assert a.modes.allows_action()


def test_observe_mode_still_blocks_actions():
    cfg, hub, ex, a = _rig()
    a.handle("只看不操作")
    assert a.modes.mode is Mode.OBSERVE
    assert ex.execute(Command(op="key_tap", key="a")) is False


def test_offline_answer_uses_local_facts_not_guesses():
    cfg, hub, ex, a = _rig()
    a.recent.add("vision", "画面=游戏内", 光柱=1)
    res = a.handle("刚才为什么没捡那个")
    assert res["source"] == "offline"
    assert "光柱=1" in res["reply"] or "画面=游戏内" in res["reply"]
    assert "连不上模型" in res["reply"]


def test_model_action_outside_whitelist_is_ignored():
    """模型说"删库"也只能被当成 none —— 白名单之外一律不执行。"""
    cfg, hub, ex, a = _rig()
    res = a._on_model_obj("随便", {"reply": "我删了", "action": "drop_tables"}, source="model")
    assert res["ok"] and a.modes.mode is Mode.COOP
    assert not (res.get("action") or {}).get("kind") == "drop_tables"


def test_patch_flow_preview_then_confirm_then_undo(prof):
    cfg, hub, ex, a = _rig(prof)
    lesson = prof.root / "lessons" / "t.md"
    before = lesson.read_text(encoding="utf-8")

    # 模型给出一个改手册的方案
    res = a._on_model_obj("以后没血就先撤", {
        "reply": "我建议加一条：没血了先往后撤。",
        "action": "patch",
        "patch": {"kind": "lesson", "payload": {"when": "没血了", "do": "先往后撤"}},
    }, source="model")
    assert res["pending"], "改规则必须先问，不能直接写"
    assert lesson.read_text(encoding="utf-8") == before, "确认之前一个字节都不能动"

    a.handle("确认")
    assert "没血了" in lesson.read_text(encoding="utf-8")

    a.handle("撤销")
    assert lesson.read_text(encoding="utf-8") == before


def test_patch_cancel_leaves_file_untouched(prof):
    cfg, hub, ex, a = _rig(prof)
    lesson = prof.root / "lessons" / "t.md"
    before = lesson.read_text(encoding="utf-8")
    a._on_model_obj("改", {"reply": "加一条", "action": "patch",
                           "patch": {"kind": "lesson",
                                     "payload": {"when": "没血了", "do": "先撤"}}},
                    source="model")
    a.handle("取消")
    assert lesson.read_text(encoding="utf-8") == before


# ═══════════════════════ 7. HTTP 入口 ═══════════════════════

def _client(prof=None):
    cfg = AppConfig()
    cfg.server.token = TOKEN
    cfg.executor.min_action_interval_ms = 0
    cfg.executor.max_actions_per_sec = 1000
    cfg.assistant.enabled = False
    ex = DryRunExecutor(cfg.executor)
    hub = Hub(cfg, ex, VisionPipeline(cfg.vision))
    app = create_app(cfg, hub, "<h1>dash</h1>", profile=prof)
    return TestClient(app), hub


def test_http_ask_runs_local_command_immediately(prof):
    c, hub = _client(prof)
    hub.fsm.start()
    r = c.post("/api/assistant/ask", headers={"X-Token": TOKEN}, json={"text": "停"})
    assert r.status_code == 200
    d = r.json()
    assert d["mode"]["mode"] == "paused"
    assert hub.fsm.state is State.PAUSED


def test_http_mode_endpoint_matches_voice_path(prof):
    c, hub = _client(prof)
    r = c.post("/api/assistant/mode", headers={"X-Token": TOKEN},
               json={"mode": "observe"})
    assert r.json()["mode"]["mode"] == "observe"
    assert not hub.assistant.modes.allows_action()


def test_http_state_reports_voice_capabilities(prof):
    c, hub = _client(prof)
    d = c.get("/api/assistant", headers={"X-Token": TOKEN}).json()
    assert d["ok"] and d["mode"]["mode"] in ("coop", "auto", "observe", "paused")
    assert "server_asr" in d["voice"] and "reason" in d["voice"]


def test_http_patch_preview_and_apply(prof):
    c, hub = _client(prof)
    body = {"kind": "lesson", "payload": {"when": "背包满了", "do": "先回城"},
            "note": "手动改"}
    pv = c.post("/api/assistant/patch/preview", headers={"X-Token": TOKEN}, json=body).json()
    assert pv["ok"], pv
    rec = c.post("/api/assistant/patch/apply", headers={"X-Token": TOKEN},
                 json={**body, "stamp": pv["stamp"]}).json()
    assert rec["ok"]
    assert "背包满了" in (prof.root / "lessons" / "t.md").read_text(encoding="utf-8")
    back = c.post("/api/assistant/patch/rollback", headers={"X-Token": TOKEN},
                  json={"id": rec["patch"]["id"]}).json()
    assert back["ok"]
    assert "背包满了" not in (prof.root / "lessons" / "t.md").read_text(encoding="utf-8")


# ═══════════════════════ 8. 改完必须真的生效 ═══════════════════════

def _rule_gen(prof, rule_id="r_old"):
    """给 profile 造一代配方，规则文件里先有一条旧规则。"""
    gd = prof.root / "evolve" / "gen-000"
    gd.mkdir(parents=True, exist_ok=True)
    (gd / "rules.yaml").write_text(
        yaml.safe_dump({"version": 1, "rules": [
            {"id": rule_id, "when": {"fact": "in_game", "op": "==", "value": True},
             "then": {"action": "wait", "params": {"frames": 1}}}]},
            allow_unicode=True, sort_keys=False), encoding="utf-8")
    return gd / "rules.yaml"


def test_rule_patch_overwrites_an_existing_rule_for_real(prof, monkeypatch):
    """同 id 的规则补丁是"覆盖"，不是"新增"。

    这里踩过一个坑：`_build_rule` 里 `rules = [ ... ]` 只是重绑了局部变量，
    `doc["rules"]` 还指着原列表，于是覆盖静默不生效 —— 补丁账本记成已应用、
    校验还说"原有规则都在"，文件一个字没变。学习类补丁全靠这条路径改门槛，
    一旦失灵，表现就是"学了但没变"。
    """
    from types import SimpleNamespace

    import autogame.evolve as ev

    rules = _rule_gen(prof)
    monkeypatch.setattr(ev, "active_generation",
                        lambda p: SimpleNamespace(id="gen-000", rules_path=rules))
    store = PatchStore(prof)
    payload = {"rule": {"id": "r_old",
                        "when": {"fact": "monsters", "op": ">=", "value": 3},
                        "then": {"action": "attack_nearest"}}}
    pv = store.preview("rule", payload)
    assert pv["ok"], pv["checks"]
    assert any(c["name"] == "是覆盖已有规则" for c in pv["checks"])
    store.apply("rule", payload, stamp=pv["stamp"])

    body = rules.read_text(encoding="utf-8")
    assert "monsters" in body, body
    assert body.count("- id: r_old") == 1, body


def test_rule_patch_overwrites_an_existing_rule_for_real(prof, monkeypatch):
    """同 id 的规则补丁是"覆盖"，不是"新增"。

    这里踩过一个坑：`_build_rule` 里 `rules = [ ... ]` 只是重绑了局部变量，
    `doc["rules"]` 还指着原列表，于是覆盖静默不生效 —— 补丁账本记成已应用、
    校验还说"原有规则都在"，文件一个字没变。学习类补丁全靠这条路径改门槛，
    一旦失灵，表现就是"学了但没变"。
    """
    from types import SimpleNamespace

    import autogame.evolve as ev

    rules = _rule_gen(prof)
    monkeypatch.setattr(ev, "active_generation",
                        lambda p: SimpleNamespace(id="gen-000", rules_path=rules))
    store = PatchStore(prof)
    payload = {"rule": {"id": "r_old",
                        "when": {"fact": "monsters", "op": ">=", "value": 3},
                        "then": {"action": "attack_nearest"}}}
    pv = store.preview("rule", payload)
    assert pv["ok"], pv["checks"]
    assert any(c["name"] == "是覆盖已有规则" for c in pv["checks"])
    store.apply("rule", payload, stamp=pv["stamp"])

    body = rules.read_text(encoding="utf-8")
    assert "monsters" in body, body
    assert body.count("- id: r_old") == 1, body


def test_rule_patch_reloads_the_running_rule_set(prof, monkeypatch):
    """规则是数据，进程不会自己知道文件变了 —— 写完必须当场重载。

    不重载的话用户听到"改好了"，程序跑的还是旧那套，下次重启才突然变 ——
    这种"改了没反应，过几天又灵了"是排查成本最高的一类问题。
    """
    from types import SimpleNamespace

    import autogame.evolve as ev

    rules = _rule_gen(prof)
    monkeypatch.setattr(ev, "active_generation",
                        lambda p: SimpleNamespace(id="gen-000", rules_path=rules))

    cfg = AppConfig()
    cfg.executor.min_action_interval_ms = 0
    ex = DryRunExecutor(cfg.executor)
    hub = Hub(cfg, ex, VisionPipeline(cfg.vision))
    calls = []
    hub._reload_rules = lambda: calls.append(1)
    a = asst_mod.build(cfg, prof, hub)

    rule = {"id": "r_extra", "when": {"fact": "monsters", "op": ">=", "value": 2},
            "then": {"action": "esc", "params": {}}}
    res = a._on_model_obj("以后两只怪就先撤", {
        "reply": "加一条", "action": "patch",
        "patch": {"kind": "rule", "payload": {"rule": rule}}}, source="model")
    assert res["pending"], res
    done = a.handle("确认")
    assert "r_extra" in rules.read_text(encoding="utf-8")
    assert calls, "应用规则补丁后必须重载，否则跑的还是旧规则"


def test_rule_rollback_reloads_too(prof, monkeypatch):
    from types import SimpleNamespace

    import autogame.evolve as ev

    rules = _rule_gen(prof)
    monkeypatch.setattr(ev, "active_generation",
                        lambda p: SimpleNamespace(id="gen-000", rules_path=rules))
    cfg = AppConfig()
    cfg.executor.min_action_interval_ms = 0
    hub = Hub(cfg, DryRunExecutor(cfg.executor), VisionPipeline(cfg.vision))
    calls = []
    hub._reload_rules = lambda: calls.append(1)
    a = asst_mod.build(cfg, prof, hub)
    before = rules.read_text(encoding="utf-8")

    a._on_model_obj("加一条", {"reply": "加", "action": "patch",
                               "patch": {"kind": "rule",
                                         "payload": {"rule": {
                                             "id": "r_x",
                                             "when": {"fact": "pillars", "op": ">=",
                                                      "value": 1},
                                             "then": {"action": "pickup"}}}}},
                    source="model")
    a.handle("确认")
    assert calls
    a.handle("撤销")
    assert rules.read_text(encoding="utf-8") == before
    assert len(calls) >= 2, "回滚也要重载"


def test_config_patch_applies_to_live_config(tmp_path):
    """页面改配置走 apply_patch 原地生效，对话改配置必须走同一条路。"""
    from autogame.assistant.brain import Assistant

    cfg = AppConfig()
    cfg.decision.tick_hz = 2.0
    cfg_path = tmp_path / "config.yaml"
    a = Assistant(cfg, None, None, config_path=str(cfg_path))
    pv = a.patches.preview(KIND_CONFIG, {"decision": {"tick_hz": 5.0}})
    assert pv["ok"], pv["checks"]
    a.patches.apply(KIND_CONFIG, {"decision": {"tick_hz": 5.0}}, stamp=pv["stamp"])
    msg = a.make_live("config", {"decision": {"tick_hz": 5.0}})
    assert cfg.decision.tick_hz == 5.0, "内存里的配置没变，等于没生效"
    assert "当场生效" in msg


def test_config_patch_reports_fields_that_need_a_restart(tmp_path):
    from autogame.assistant.brain import Assistant

    cfg = AppConfig()
    cfg_path = tmp_path / "config.yaml"
    a = Assistant(cfg, None, None, config_path=str(cfg_path))
    pv = a.patches.preview(KIND_CONFIG, {"server": {"port": 8899}})
    assert pv["ok"], pv["checks"]
    a.patches.apply(KIND_CONFIG, {"server": {"port": 8899}}, stamp=pv["stamp"])
    msg = a.make_live("config", {"server": {"port": 8899}})
    assert "重启" in msg, "不能假装端口这类字段已经生效"


def test_http_legacy_aliases_stay_working(prof):
    """对话里给的接口清单用的是 /api/assistant/interpret 与 /api/scripts/...。

    别名不是为了兼容老代码 —— 是为了让对着那份清单写的外部客户端能直接用，
    而不用为了一个路径名改一遍。
    """
    c, hub = _client(prof)
    d = c.post("/api/assistant/interpret", headers={"X-Token": TOKEN},
               json={"text": "现在什么情况"}).json()
    assert d["ok"]

    body = {"kind": "lesson", "payload": {"when": "背包满了", "do": "先回城"}}
    pv = c.post("/api/scripts/patch/preview", headers={"X-Token": TOKEN},
                json=body).json()
    assert pv["ok"]
    rec = c.post("/api/scripts/patch/apply", headers={"X-Token": TOKEN},
                 json={**body, "stamp": pv["stamp"]}).json()
    assert rec["ok"]
    back = c.post("/api/scripts/rollback", headers={"X-Token": TOKEN},
                  json={"id": rec["patch"]["id"]}).json()
    assert back["ok"]


# ═══════════════════════ 9. 模式的副作用（观察≠暂停） ═══════════════════════

def test_observe_keeps_deciding_but_stops_acting():
    """观察要留下"当时想做什么"，所以引擎得继续跑；只是动作一个都不发。"""
    cfg, hub, ex, a = _rig()
    hub.fsm.start()
    a.handle("只看不操作")
    assert a.modes.mode is Mode.OBSERVE
    assert hub.fsm.state is not State.PAUSED, "观察模式不该把引擎停掉"
    assert ex.execute(Command(op="key_tap", key="a")) is False
    assert ex.journal[-1]["op"] == "stop_all" or ex.suppressed >= 1


def test_pause_still_stops_the_engine():
    cfg, hub, ex, a = _rig()
    hub.fsm.start()
    a.handle("先停下")
    assert hub.fsm.state is State.PAUSED


def test_pause_expiry_wakes_the_engine_again():
    """"暂停十分钟，到点自己接着跑" —— 到点没人点继续，也必须自己点亮。"""
    cfg, hub, ex, a = _rig()
    hub.fsm.start()
    a.modes.pause(0.05, reason="测试")
    assert hub.fsm.state is State.PAUSED
    time.sleep(0.1)
    assert a.modes.mode is Mode.COOP          # 读一次触发到期
    assert hub.fsm.state is not State.PAUSED


def test_mode_change_callback_fires_on_expiry_without_being_asked():
    from autogame.assistant.modes import ModeManager

    seen = []
    m = ModeManager()
    m.on_change = lambda old, new: seen.append((old, new))
    m.pause(0.05)
    time.sleep(0.1)
    assert m.allows_action() is True           # 热路径会结算过期
    assert seen and seen[-1][1] is Mode.COOP


def test_hub_skips_decisions_when_paused_but_not_when_observing():
    cfg, hub, ex, a = _rig()
    assert hub._may_decide() is True
    a.handle("先停下")
    assert hub._may_decide() is False, "暂停还继续决策只是白烧 CPU"
    a.handle("只看不操作")
    assert hub._may_decide() is True, "观察要继续决策"


def test_hub_without_assistant_still_decides():
    cfg = AppConfig()
    cfg.executor.min_action_interval_ms = 0
    hub = Hub(cfg, DryRunExecutor(cfg.executor), VisionPipeline(cfg.vision))
    assert hub._may_decide() is True


# ═══════════════════════ 10. 模型那一路（假模型） ═══════════════════════

def _rig_model(monkeypatch, payload, prof=None, *, raw: str | None = None):
    import json

    import autogame.decision.llm_client as llm

    cfg, hub, ex, a = _rig(prof)
    cfg.assistant.enabled = True
    cfg.assistant.base_url = "http://127.0.0.1:11434"
    cfg.assistant.model = "fake"

    calls: list[dict] = []

    def chat(_cfg, system, user, **kw):
        calls.append({"system": system, "user": user, "kw": kw})
        return raw if raw is not None else json.dumps(payload, ensure_ascii=False)

    monkeypatch.setattr(llm, "chat", chat)
    return cfg, hub, ex, a, calls


def test_model_sees_recent_facts_and_answers(monkeypatch):
    cfg, hub, ex, a, calls = _rig_model(monkeypatch, {
        "reply": "刚才有 2 根光柱，但你一直没走过去。",
        "speak": "有两根光柱",
        "action": "none"})
    a.recent.add("vision", "画面=战斗", 光柱=2)
    res = a.handle("刚才为什么没捡")
    assert res["source"] == "model"
    assert "2 根光柱" in res["reply"]
    assert "光柱=2" in calls[0]["user"], "模型必须看到现场事实，否则只能编"
    assert calls[0]["kw"].get("json_mode") is True


def test_model_question_becomes_a_pending_choice_and_choice_runs(monkeypatch):
    cfg, hub, ex, a, _ = _rig_model(monkeypatch, {
        "reply": "要切观察吗？", "action": "observe",
        "question": "现在切到观察模式？", "options": ["切", "算了"]})
    res = a.handle("能不能先别动")
    assert res["pending"] and res["pending"]["question"] == "现在切到观察模式？"
    assert res["pending"]["options"] == ["切", "算了"]

    a.handle("1")
    assert a.modes.mode is Mode.OBSERVE


def test_model_action_pause_is_executed_but_unknown_is_not(monkeypatch):
    cfg, hub, ex, a, _ = _rig_model(monkeypatch, {
        "reply": "先停一下", "action": "pause", "action_arg": {"seconds": 5}})
    a.handle("现在该怎么处理")
    assert a.modes.mode is Mode.PAUSED

    cfg2, hub2, ex2, b, _ = _rig_model(monkeypatch, {
        "reply": "我删库了", "action": "rm_rf_/", "action_arg": {}})
    b.handle("随便干点什么")
    assert b.modes.mode is Mode.COOP, "白名单之外的动作一律不执行"


def test_model_bad_json_falls_back_to_offline(monkeypatch):
    cfg, hub, ex, a, _ = _rig_model(monkeypatch, {}, raw="我觉得应该先撤")
    a.recent.add("vision", "画面=战斗", 光柱=1)
    res = a.handle("刚才怎么回事")
    assert res["source"] == "offline"
    assert "连不上模型" in res["reply"]


def test_model_cannot_patch_when_patching_is_disabled(monkeypatch):
    cfg, hub, ex, a, _ = _rig_model(monkeypatch, {
        "reply": "我改一下", "action": "patch",
        "patch": {"kind": "lesson", "payload": {"when": "没血", "do": "撤"}}})
    cfg.assistant.allow_patch = False
    res = a.handle("以后没血就先撤")
    assert "权限" in res["reply"]
    assert not res["pending"]


def test_allow_patch_off_does_not_touch_the_file(monkeypatch, prof):
    lesson = prof.root / "lessons" / "t.md"
    before = lesson.read_text(encoding="utf-8")
    cfg, hub, ex, a, _ = _rig_model(monkeypatch, {
        "reply": "我改一下", "action": "patch",
        "patch": {"kind": "lesson", "payload": {"when": "没血", "do": "撤"}}}, prof)
    cfg.assistant.allow_patch = False
    a.handle("以后没血就先撤")
    assert lesson.read_text(encoding="utf-8") == before


def test_empty_input_does_not_call_the_model(monkeypatch):
    cfg, hub, ex, a, calls = _rig_model(monkeypatch, {"reply": "在", "action": "none"})
    res = a.handle("   ")
    assert res["ok"] and not calls
    assert "还没说话" in res["reply"]


def test_offline_with_no_records_says_so():
    cfg, hub, ex, a = _rig()
    a.recent.clear()
    res = a._offline("刚才发生了什么", source="local")
    assert "什么都没记到" in res["reply"]


def test_expired_pending_is_not_treated_as_an_answer(monkeypatch):
    import autogame.assistant.dialog as dlg

    cfg, hub, ex, a = _rig()
    monkeypatch.setattr(dlg, "PENDING_TTL_S", 0.01)
    a.dialog.ask("要不要改？", options=["确认", "取消"], kind="confirm")
    time.sleep(0.03)
    res = a.handle("确认")
    assert "已经过去太久" in res["reply"]


def test_unknown_group_is_not_run(prof):
    cfg, hub, ex, a = _rig(prof)
    res = a._run_group("不存在的组", source="model")
    assert "没认出" in res["reply"]
    assert hub.ui_task is None


def test_page_buttons_and_voice_share_the_same_switch():
    """仪表盘上的「启动/暂停/急停」必须和语音走同一个开关。

    只动 FSM 不动模式 → 页面显示在跑、动作全被拦下（假运行）；
    只动模式不动 FSM → 模式说停了、引擎还在想。
    """
    cfg, hub, ex, a = _rig()
    hub.control("stop")
    assert a.modes.mode is Mode.PAUSED and hub.fsm.state is State.PAUSED

    hub.control("start")
    assert a.modes.allows_action(), "点了启动却还停在暂停模式"
    assert hub.fsm.state is not State.PAUSED

    hub.control("panic")
    assert not a.modes.allows_action(), "急停之后闸门必须关上"


def test_page_buttons_and_voice_share_the_same_switch():
    """仪表盘上的「启动/暂停/急停」必须和语音走同一个开关。

    只动 FSM 不动模式 → 页面显示在跑、动作全被拦下（假运行）；
    只动模式不动 FSM → 模式说停了、引擎还在想。
    """
    cfg, hub, ex, a = _rig()
    hub.control("stop")
    assert a.modes.mode is Mode.PAUSED and hub.fsm.state is State.PAUSED

    hub.control("start")
    assert a.modes.allows_action(), "点了启动却还停在暂停模式"
    assert hub.fsm.state is not State.PAUSED

    hub.control("panic")
    assert not a.modes.allows_action(), "急停之后闸门必须关上"


def test_vision_lost_auto_stop_also_closes_the_gate():
    """看不到画面时自动停车，不能只停 FSM —— 闸门不关，指令组还能继续点。"""
    cfg, hub, ex, a = _rig()
    cfg.watchdog.no_frame_stop_s = 1.0
    hub.control("start")
    hub._latest_ts = time.time() - 10.0
    hub.check_vision_lost()
    assert hub.fsm.state is State.PAUSED
    assert not a.modes.allows_action(), "自动停车没关闸门，等于只停了一半"


def test_stop_clears_a_running_group():
    cfg, hub, ex, a = _rig()
    hub.ui_task = object()
    hub.control("stop")
    assert hub.ui_task is None, "点了暂停，正在跑的指令组还在后台点"


def test_vision_lost_auto_stop_also_closes_the_gate():
    """看不到画面时自动停车，不能只停 FSM —— 闸门不关，指令组还能继续点。"""
    cfg, hub, ex, a = _rig()
    cfg.watchdog.no_frame_stop_s = 1.0
    hub.control("start")
    hub._latest_ts = time.time() - 10.0
    hub.check_vision_lost()
    assert hub.fsm.state is State.PAUSED
    assert not a.modes.allows_action(), "自动停车没关闸门，等于只停了一半"


def test_stop_clears_a_running_group():
    cfg, hub, ex, a = _rig()
    hub.ui_task = object()
    hub.control("stop")
    assert hub.ui_task is None, "点了暂停，正在跑的指令组还在后台点"
