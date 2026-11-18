"""三级裁决的边界测试 —— 钉的是"它不许干什么"，不是"它能跑通"。

这一层的危险在于它真的会发命令。所以每条用例都在回答同一个问题：
**模型说了一句听起来很有道理的话，程序会不会照做？**

覆盖：
  1. 清单外的动作、带坐标的动作、前置不满足的动作 —— 一律不执行
  2. L1 选不出来（others/非法/低置信）→ 升级 L2，不是硬执行
  3. L2 计划里有一步非法 → 整条丢弃（不执行残缺计划）
  4. tick() 永不阻塞（模型慢不会拖住 1Hz 主循环）
  5. shadow 下只记账不执行
  6. 经验未验证不直出；已验证的才能跳过模型
  7. 对账把"有效/无效"回写到经验库，达阈值才升级状态
  8. 等模型期间有怪就砍（不站着挨打）
"""

from __future__ import annotations

import json
import time

import pytest

from autogame.config import ArbiterConfig, ArbiterModelConfig
from autogame.decision.actions import ActionRegistry, validate_params
from autogame.decision.arbiter import Arbiter, Stage
from autogame.decision.knowledge import (
    STATUS_CANDIDATE,
    STATUS_CONFIRMED,
    STATUS_REJECTED,
    KnowledgeBase,
    fingerprint,
)


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _cfg(mode: str = "enforce", **kw) -> ArbiterConfig:
    base = dict(
        mode=mode,
        laya=ArbiterModelConfig(model="fake-laya", timeout_s=2.0),
        planner=ArbiterModelConfig(model="fake-planner", timeout_s=2.0),
        cooldown_s=1.0, idle_trigger_s=2.0, unmatched_trigger_s=5.0,
        min_confidence=0.5, verify_horizon_s=1.0, plan_ttl_s=10.0,
        keepalive_after_s=2.0, keepalive_interval_s=0.5,
    )
    base.update(kw)
    return ArbiterConfig(**base)


def _probe(**kw) -> dict:
    p = {
        "in_game": True, "gate_reason": "skillbar", "pillars": 0, "monsters": 0,
        "aframe": False, "picks": 0, "dist": None, "w": 960, "h": 432,
        "nav_enabled": False, "rechallenge_visible": False, "intent": "",
        "miss_streak": 0,
    }
    p.update(kw)
    return p


def _arb(tmp_path, cfg=None, reply=None, clock=None):
    """构造带假模型的 arbiter。reply 可以是字符串或 callable(cfg, system, user)。"""
    cfg = cfg or _cfg()
    clock = clock or Clock()
    kb = KnowledgeBase(tmp_path / "knowledge")
    reg = ActionRegistry()
    calls: list[str] = []

    def caller(model_cfg, system, user):
        calls.append(model_cfg.model)
        if callable(reply):
            return reply(model_cfg, system, user)
        return reply if reply is not None else json.dumps(
            {"action": "others", "reason": "不知道", "confidence": 0.1})

    arb = Arbiter(cfg, reg, kb, clock=clock, root=tmp_path, caller=caller)
    return arb, kb, calls, clock


def _wait_idle(arb, seconds: float = 3.0) -> None:
    """等后台线程跑完（真实线程，不是 mock —— 这条链路的价值就在异步）。"""
    end = time.time() + seconds
    while time.time() < end:
        with arb._lock:
            if not arb._busy:
                return
        time.sleep(0.02)
    raise AssertionError("后台裁决线程没有在预期时间内结束")


# ---------------- 1. 清单与参数 ----------------

def test_unknown_action_is_rejected():
    reg = ActionRegistry()
    ok, why = reg.validate("open_inventory", {}, _probe())
    assert not ok
    assert "不在系统能力清单" in why


def test_coordinate_param_is_rejected():
    """模型最爱的表达就是"往右下走 120px"。这里必须硬挡。"""
    reg = ActionRegistry()
    for params in ({"x": 120}, {"target_y": 200}, {"px": 30}, {"pos": "右下"}):
        ok, why = reg.validate("explore", params, _probe())
        assert not ok, params
        assert "坐标" in why or "不接受参数" in why


def test_undeclared_param_is_rejected():
    reg = ActionRegistry()
    ok, why = reg.validate("pickup", {"speed": 3}, _probe())
    assert not ok
    assert "不接受参数" in why


def test_action_requiring_target_is_unavailable_when_absent():
    """没有光柱时 approach_nearest 不该出现在清单里，也不该被执行。"""
    reg = ActionRegistry()
    p = _probe(pillars=0)
    assert "approach_nearest" not in [a.id for a in reg.available(p)]
    ok, why = reg.validate("approach_nearest", {}, p)
    assert not ok
    assert "前置条件" in why


def test_always_available_actions_survive_empty_scene():
    """再糟的局面也得能退出：wait/esc/reset 永远活着。"""
    reg = ActionRegistry()
    ids = [a.id for a in reg.available({})]
    assert {"wait", "esc", "reset_stage"} <= set(ids)


def test_prompt_block_hides_unavailable_actions():
    reg = ActionRegistry()
    text = reg.prompt_block(_probe(pillars=0, monsters=0))
    assert "attack_nearest" not in text.split("当前不可用")[0]
    assert "当前不可用" in text


def test_param_schema_enforced():
    from autogame.decision.actions import ActionSpec

    spec = ActionSpec("wait", "等待", "等", params={"frames": "int"})
    assert validate_params(spec, {"frames": 3})[0]
    assert not validate_params(spec, {"frames": "三"})[0]
    assert not validate_params(spec, {"frames": 999})[0]


# ---------------- 2. 三级升级 ----------------

def test_l1_choice_is_executed_without_calling_planner(tmp_path):
    reply = json.dumps({"action": "explore", "params": {}, "confidence": 0.9,
                        "reason": "没目标就换个地方"})
    arb, _kb, calls, clock = _arb(tmp_path, reply=reply)
    arb.tick(_probe(), clock(), idle_s=5.0)
    _wait_idle(arb)
    call = arb.tick(_probe(), clock(), idle_s=5.0)
    assert call is not None
    assert call.action_id == "explore"
    assert call.stage == Stage.LAYA.value
    assert calls == ["fake-laya"], "L1 成功时不该再问规划器"


def test_l1_others_escalates_to_planner(tmp_path):
    def reply(model_cfg, system, user):
        if model_cfg.model == "fake-laya":
            return json.dumps({"action": "others", "reason": "清单里没有能解决这个的",
                               "confidence": 0.2})
        return json.dumps({"plan": [{"action": "explore", "params": {}}],
                           "reason": "先离开原地", "confidence": 0.8})

    arb, _kb, calls, clock = _arb(tmp_path, reply=reply)
    arb.tick(_probe(), clock(), idle_s=5.0)
    _wait_idle(arb)
    call = arb.tick(_probe(), clock(), idle_s=5.0)
    assert call is not None and call.action_id == "explore"
    assert call.stage == Stage.PLANNER.value
    assert calls == ["fake-laya", "fake-planner"]


def test_l1_illegal_choice_escalates_instead_of_executing(tmp_path):
    """L1 选了一个不存在的动作 —— 不能"凑合执行"，要升级让大模型重排。"""
    def reply(model_cfg, system, user):
        if model_cfg.model == "fake-laya":
            return json.dumps({"action": "dodge_roll", "confidence": 0.95})
        return json.dumps({"plan": [{"action": "wait", "params": {"frames": 2}}],
                           "confidence": 0.8})

    arb, _kb, calls, clock = _arb(tmp_path, reply=reply)
    arb.tick(_probe(), clock(), idle_s=5.0)
    _wait_idle(arb)
    call = arb.tick(_probe(), clock(), idle_s=5.0)
    assert call is not None
    assert call.action_id == "wait"
    assert calls == ["fake-laya", "fake-planner"]


def test_l1_low_confidence_escalates(tmp_path):
    def reply(model_cfg, system, user):
        if model_cfg.model == "fake-laya":
            return json.dumps({"action": "explore", "confidence": 0.2})
        return json.dumps({"plan": [{"action": "wait", "params": {}}], "confidence": 0.9})

    arb, _kb, calls, clock = _arb(tmp_path, reply=reply)
    arb.tick(_probe(), clock(), idle_s=5.0)
    _wait_idle(arb)
    call = arb.tick(_probe(), clock(), idle_s=5.0)
    assert call.action_id == "wait"
    assert call.stage == Stage.PLANNER.value


def test_planner_plan_with_coordinates_is_dropped_entirely(tmp_path):
    def reply(model_cfg, system, user):
        if model_cfg.model == "fake-laya":
            return json.dumps({"action": "others", "confidence": 0.1})
        return json.dumps({"plan": [
            {"action": "explore", "params": {}},
            {"action": "attack_nearest", "params": {"x": 400, "y": 300}},
        ], "confidence": 0.9})

    arb, kb, _calls, clock = _arb(tmp_path, reply=reply)
    arb.tick(_probe(monsters=2), clock(), idle_s=5.0)
    _wait_idle(arb)
    call = arb.tick(_probe(monsters=2), clock(), idle_s=5.0)
    assert call is None, "计划里有一带坐标的步骤，整条都不许执行"
    assert arb.stats["l2_rejected"] == 1
    assert kb.snapshot()["total"] == 0, "被拒的计划不许写进经验库"


def test_planner_empty_plan_is_not_an_error(tmp_path):
    """模型说"清单里真没有能用的，缺 X 能力" —— 这是合法输出，要接受。"""
    def reply(model_cfg, system, user):
        if model_cfg.model == "fake-laya":
            return json.dumps({"action": "others", "confidence": 0.1})
        return json.dumps({"plan": [], "missing": "缺少『打开背包使用药水』的能力",
                           "confidence": 0.9})

    arb, kb, _calls, clock = _arb(tmp_path, reply=reply)
    arb.tick(_probe(), clock(), idle_s=5.0)
    _wait_idle(arb)
    assert arb.tick(_probe(), clock(), idle_s=5.0) is None
    assert kb.snapshot()["total"] == 0


# ---------------- 3. 不阻塞 ----------------

def test_tick_does_not_block_while_model_is_thinking(tmp_path):
    def slow(model_cfg, system, user):
        time.sleep(0.4)
        return json.dumps({"action": "others", "confidence": 0.1})

    arb, _kb, _calls, clock = _arb(tmp_path, reply=slow)
    t0 = time.perf_counter()
    arb.tick(_probe(), clock(), idle_s=5.0)
    first = time.perf_counter() - t0
    assert first < 0.15, f"第一次 tick 花了 {first*1000:.0f}ms，说明请求是同步的"

    t0 = time.perf_counter()
    arb.tick(_probe(), clock(), idle_s=5.0)
    second = time.perf_counter() - t0
    assert second < 0.05, f"等待期间 tick 又阻塞了 {second*1000:.0f}ms"
    _wait_idle(arb)


def test_keepalive_attacks_instead_of_standing_still(tmp_path):
    """等模型的时候有怪 —— 先砍一刀，别站着挨打。"""
    def slow(model_cfg, system, user):
        time.sleep(0.4)
        return json.dumps({"action": "others", "confidence": 0.1})

    arb, _kb, _calls, clock = _arb(tmp_path, reply=slow)
    arb.tick(_probe(monsters=3), clock(), idle_s=1.0)   # 建立 idle 起点
    clock.advance(3.0)
    call = arb.tick(_probe(monsters=3), clock(), idle_s=3.0)
    assert call is not None
    assert call.action_id == "attack_nearest"
    assert call.stage == Stage.KEEPALIVE.value
    _wait_idle(arb)


def test_no_keepalive_when_no_monsters(tmp_path):
    def slow(model_cfg, system, user):
        time.sleep(0.4)
        return json.dumps({"action": "others", "confidence": 0.1})

    arb, _kb, _calls, clock = _arb(tmp_path, reply=slow)
    arb.tick(_probe(monsters=0), clock(), idle_s=1.0)
    clock.advance(3.0)
    assert arb.tick(_probe(monsters=0), clock(), idle_s=3.0) is None
    _wait_idle(arb)


# ---------------- 4. 影子模式 ----------------

def test_shadow_records_but_never_returns_a_call(tmp_path):
    reply = json.dumps({"action": "explore", "confidence": 0.95})
    arb, _kb, _calls, clock = _arb(tmp_path, cfg=_cfg(mode="shadow"), reply=reply)
    arb.tick(_probe(), clock(), idle_s=5.0)
    _wait_idle(arb)
    assert arb.tick(_probe(), clock(), idle_s=5.0) is None
    assert arb.stats["shadow_would_run"] >= 1
    snap = arb.snapshot()
    assert snap["last"] is not None and snap["last"]["action"] == "explore"
    assert snap["applied"] is False


# ---------------- 5. 经验库 ----------------

def test_candidate_is_not_used_for_direct_execution(tmp_path):
    """模型说过的做法在验证之前，不能变成"以后都这么干"。"""
    arb, kb, calls, clock = _arb(tmp_path, reply=json.dumps(
        {"action": "explore", "confidence": 0.9}))
    p = _probe()
    fp = fingerprint(p)
    kb.remember("没目标", {"id": "explore", "params": {}}, "planner", fp)
    assert kb.find_usable(fp) is None

    arb.tick(p, clock(), idle_s=5.0)
    _wait_idle(arb)
    assert calls, "没有可用经验时应该去问模型，而不是直接执行 candidate"


def test_confirmed_experience_skips_the_model(tmp_path):
    arb, kb, calls, clock = _arb(tmp_path, reply=json.dumps(
        {"action": "explore", "confidence": 0.9}))
    p = _probe()
    fp = fingerprint(p)
    exp = kb.remember("没目标", {"id": "explore", "params": {}}, "planner", fp)
    kb.verify(exp.id, True)
    kb.verify(exp.id, True)
    assert kb.find_usable(fp) is not None

    call = arb.tick(p, clock(), idle_s=5.0)
    assert call is not None
    assert call.action_id == "explore"
    assert call.stage == Stage.KNOWLEDGE.value
    assert call.exp_id == exp.id
    assert calls == [], "已验证的经验不该再去问模型"


def test_verification_promotes_and_rejects(tmp_path):
    kb = KnowledgeBase(tmp_path / "knowledge")
    fp = fingerprint(_probe())
    good = kb.remember("a", {"id": "explore"}, "planner", fp)
    bad = kb.remember("b", {"id": "wait"}, "planner", fp)

    assert kb.verify(good.id, True) == STATUS_CANDIDATE
    assert kb.verify(good.id, True) == STATUS_CONFIRMED
    assert kb.verify(bad.id, False) == STATUS_CANDIDATE
    assert kb.verify(bad.id, False) == STATUS_REJECTED
    assert kb.find_usable(fp).id == good.id


def test_rejected_experience_is_kept_as_counterexample(tmp_path):
    """失败的经验不删 —— 它是"别再这么干"的证据，模型看历史时要能看到。"""
    kb = KnowledgeBase(tmp_path / "knowledge")
    fp = fingerprint(_probe())
    e = kb.remember("卡住", {"id": "wait"}, "planner", fp)
    kb.verify(e.id, False)
    kb.verify(e.id, False)
    assert kb.find_usable(fp) is None
    block = kb.prompt_block(fp)
    assert "已知无效" in block


def test_experience_survives_restart(tmp_path):
    kb = KnowledgeBase(tmp_path / "knowledge")
    fp = fingerprint(_probe())
    e = kb.remember("x", {"id": "explore"}, "planner", fp, note="先离开原地")
    kb.verify(e.id, True)
    kb.verify(e.id, True)

    kb2 = KnowledgeBase(tmp_path / "knowledge")
    again = kb2.find_usable(fp)
    assert again is not None and again.id == e.id
    assert again.note == "先离开原地"


def test_fingerprint_is_stable_and_discriminating(tmp_path):
    a = _probe(pillars=2, monsters=1)
    b = _probe(pillars=3, monsters=1)      # 同一桶 → 同指纹
    c = _probe(pillars=0, monsters=5)      # 明显不同局面
    assert fingerprint(a) == fingerprint(b)
    assert fingerprint(a) != fingerprint(c)
    assert "pil=2-3" in fingerprint(a)
    assert "mon=3+" in fingerprint(c)


def test_executed_action_is_settled_against_reality(tmp_path):
    """模型给的动作执行后要用实测对账，不能凭"看起来做了"就算成功。"""
    exp = None
    arb, kb, _calls, clock = _arb(tmp_path, reply=json.dumps(
        {"plan": [{"action": "attack_nearest", "params": {}}], "confidence": 0.9}))
    # 先让 L1 失败升级到 L2
    p = _probe(monsters=3)
    arb.tick(p, clock(), idle_s=5.0)
    _wait_idle(arb)
    call = arb.tick(p, clock(), idle_s=5.0)
    assert call is not None and call.action_id == "attack_nearest"
    exp = kb._items[call.exp_id]

    arb.execute_hook(call, p, clock())
    clock.advance(2.0)
    arb.tick(_probe(monsters=0), clock(), idle_s=0.0)   # 怪没了 → hit
    assert kb._items[exp.id].hits == 1


# ---------------- 6. 熔断 ----------------

def test_repeated_failures_open_the_circuit(tmp_path):
    def boom(model_cfg, system, user):
        raise RuntimeError("模型服务挂了")

    arb, _kb, _calls, clock = _arb(tmp_path, cfg=_cfg(failure_threshold=1), reply=boom)
    arb.tick(_probe(), clock(), idle_s=5.0)
    _wait_idle(arb)
    assert arb.stats["fail"] == 1
    assert arb.stats["tripped"] == 1
    # 熔断期内不再打扰模型
    clock.advance(1.0)
    arb.tick(_probe(), clock(), idle_s=9.0)
    time.sleep(0.1)
    assert arb.stats["asks"] == 1


def test_off_mode_is_zero_cost(tmp_path):
    arb, _kb, calls, clock = _arb(tmp_path, cfg=_cfg(mode="off"))
    for i in range(20):
        assert arb.tick(_probe(), clock(), idle_s=30.0) is None
        clock.advance(1.0)
    assert calls == [] and arb.stats["asks"] == 0
