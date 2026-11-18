"""LLM 监督层测试 —— 钉住的不是"能跑"，是**权力边界**。

这个层最危险的地方不是它判断错，而是"判断错了还能动手"。所以这里每条用例
都对应一类真实事故：

  - shadow 模式必须一条命令都不拦（否则影子期就没有意义，等于直接上线）
  - stop_all 永远放行（安全动作被拦 = 想停停不下来）
  - 低置信度/无证据的 takeover 必须降级（模型编一句"界面可疑"就按 ESC）
  - panic 必须两票（单次幻觉直接停车，人工不在场就干等）
  - hold 有上限（一次误判永久卡死）
  - 预测对账要看**实测值**（拿旧距离对账永远 hit，等于没对账）
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autogame.config import SupervisorConfig
from autogame.decision.supervisor import (
    Decision,
    LedgerRow,
    Supervisor,
    Verdict,
    _evidence_ok,
)
from autogame.protocol import Command, SceneState


class Clock:
    """可推进的假时钟（结论有效期/熔断都靠它，不能等真实时间）。"""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


def _scene(in_game: bool = True, gate_reason: str = "skillbar") -> SceneState:
    return SceneState(ts_ms=0, w=960, h=432, character=(480, 216),
                      in_game=in_game, gate_reason=gate_reason)


def _sup(**kw) -> tuple[Supervisor, Clock]:
    clock = Clock()
    cfg = SupervisorConfig(**kw)
    return Supervisor(cfg, clock=clock, root=Path("/tmp/autogame-sup-test")), clock


def _force(sup: Supervisor, verdict: Verdict, until: float,
           conf: float = 0.9, action: dict | None = None) -> None:
    """直接把结论塞进去（跳过网络），用于测闸门行为。"""
    sup._decision = Decision(verdict=verdict, confidence=conf, until=until,
                             action=action, at=0.0)


# ---------------- 闸门：谁能拦，拦什么 ----------------

def test_shadow_mode_never_blocks_but_counts():
    """影子期一条命令都不许拦——否则"先观察"就是空话。"""
    sup, clock = _sup(mode="shadow", check_interval_s=1)
    _force(sup, Verdict.HOLD, until=clock() + 10)
    cmd = Command(op="mouse_click", x=10, y=10)
    assert sup.gate(cmd, clock()) is True
    assert sup.stats["shadow_would_block"] == 1
    assert sup.stats["blocked_cmds"] == 0


def test_enforce_hold_blocks_action_but_lets_panic_through():
    """hold 拦主动动作，但 stop_all 必须永远放行（想停必须停得下来）。"""
    sup, clock = _sup(mode="enforce", check_interval_s=1)
    _force(sup, Verdict.HOLD, until=clock() + 10)
    assert sup.gate(Command(op="key_tap", key="a"), clock()) is False
    assert sup.gate(Command(op="stop_all"), clock()) is True
    assert sup.stats["blocked_cmds"] == 1


def test_hold_expires_and_stops_blocking():
    """hold 必须自己过期，不能等人工复位（人不在场就永久卡死）。"""
    sup, clock = _sup(mode="enforce")
    _force(sup, Verdict.HOLD, until=clock() + 6)
    assert sup.gate(Command(op="key_tap", key="w"), clock()) is False
    clock.advance(7)
    assert sup.gate(Command(op="key_tap", key="w"), clock()) is True


def test_throttle_slows_without_blocking():
    """throttle 是最轻的干预：只降速，不否决。判错也只是慢一点。"""
    sup, clock = _sup(mode="enforce", throttle_pace=2.5)
    _force(sup, Verdict.THROTTLE, until=clock() + 10)
    assert sup.gate(Command(op="mouse_click", x=1, y=1), clock()) is True
    assert sup.pace_factor(clock()) == pytest.approx(2.5)


def test_off_mode_is_inert():
    sup, clock = _sup(mode="off")
    assert sup.note_frame(_scene(), "x", clock()) == {}
    assert sup.pace_factor(clock()) == 1.0
    assert sup.take_takeover(clock()) is None


# ---------------- 结论收敛：模型不能想干嘛就干嘛 ----------------

def _ctx() -> dict:
    return {"trigger": "t", "health": {"miss_streak": 4, "cmds": 12},
            "ledger": [{"kind": "pick", "result": "miss", "detail": "A框仍在"}]}


def test_low_confidence_downgrades_to_allow():
    """置信度不够就放行。监督层的第一原则是"宁可少拦，不要乱拦"。"""
    sup, clock = _sup()
    d = Decision(verdict=Verdict.TAKEOVER, confidence=0.3,
                 action={"op": "esc"}, until=clock() + 10)
    out = sup._restrict(d, _ctx(), clock())
    assert out.verdict is Verdict.ALLOW
    assert out.action is None
    assert sup.stats["downgraded"] == 1


def test_takeover_without_numeric_evidence_downgrades_to_hold():
    """证据必须复述输入里的数字——挡"编一句理由就按 ESC"。"""
    sup, clock = _sup()
    d = Decision(verdict=Verdict.TAKEOVER, confidence=0.95,
                 evidence="界面上有可疑内容，建议关闭",
                 action={"op": "esc"}, until=clock() + 10)
    out = sup._restrict(d, _ctx(), clock())
    assert out.verdict is Verdict.HOLD
    assert out.action is None


def test_takeover_with_evidence_passes():
    sup, clock = _sup()
    d = Decision(verdict=Verdict.TAKEOVER, confidence=0.95,
                 evidence="miss_streak 4 且 cmds 12 全落空",
                 action={"op": "esc", "reason": "疑似弹窗"}, until=clock() + 10)
    out = sup._restrict(d, _ctx(), clock())
    assert out.verdict is Verdict.TAKEOVER
    assert out.action == {"op": "esc", "reason": "疑似弹窗"}


def test_takeover_rejects_non_whitelisted_or_coordinate_action():
    """接管动作只能是 esc/wait/reset_stage，且不许带坐标（不许指路）。"""
    sup, clock = _sup()
    d = Decision(verdict=Verdict.TAKEOVER, confidence=0.95,
                 evidence="miss 4",
                 action={"op": "walk_to", "target": [100, 200]}, until=clock() + 10)
    out = sup._restrict(d, _ctx(), clock())
    assert out.verdict is Verdict.HOLD
    assert out.action is None


def test_panic_needs_two_votes():
    """单次幻觉不停车：第一票降 hold，第二票才 panic。"""
    sup, clock = _sup(mode="enforce", panic_confirm=2)
    d1 = Decision(verdict=Verdict.PANIC, confidence=0.95, evidence="miss 4",
                  until=clock() + 10)
    assert sup._restrict(d1, _ctx(), clock()).verdict is Verdict.HOLD
    d2 = Decision(verdict=Verdict.PANIC, confidence=0.95, evidence="miss 4",
                  until=clock() + 10)
    assert sup._restrict(d2, _ctx(), clock()).verdict is Verdict.PANIC


def test_hold_is_capped():
    """hold 时长必须被 hold_max_s 截断，哪怕模型说"等我十分钟"。"""
    sup, clock = _sup(hold_max_s=6.0)
    d = Decision(verdict=Verdict.HOLD, confidence=0.9, until=clock() + 600)
    out = sup._restrict(d, _ctx(), clock())
    assert out.until - clock() <= 6.0


def test_evidence_ok_requires_a_number_from_context():
    ctx = {"health": {"miss_streak": 4}}
    assert _evidence_ok("miss_streak 4 明显", ctx) is True
    assert _evidence_ok("界面不对劲", ctx) is False
    assert _evidence_ok("疑似 999 次失败", ctx) is False  # 数字编的


# ---------------- 对账：结果到底正不正向 ----------------

def test_walk_prediction_settles_on_measured_distance():
    """走位对账必须用**现算距离**：200→100 是 hit，200→205 是 miss。"""
    sup, clock = _sup()
    now = clock()
    sup.predict("walk", "距离应缩短", base=200.0, now=now, horizon_s=3.0)
    clock.advance(4)
    sup.note_frame(_scene(), "正在接近光柱", clock(), {"dist": 100.0, "picks": 0})
    assert [r.result for r in sup._ledger] == ["hit"]

    sup.predict("walk", "距离应缩短", base=200.0, now=clock(), horizon_s=3.0)
    clock.advance(4)
    sup.note_frame(_scene(), "正在接近光柱", clock(), {"dist": 205.0, "picks": 0})
    assert [r.result for r in sup._ledger] == ["hit", "miss"]


def test_pick_prediction_hits_when_aframe_disappears():
    """按完拾取键 A框消失 = 捡到了，这是最强的正向信号。"""
    sup, clock = _sup()
    sup.predict("pick", "A框应消失", base=0.0, now=clock(), horizon_s=1.5)
    clock.advance(2)
    sup.note_frame(_scene(), "按拾取键", clock(), {"aframe": False, "picks": 0})
    assert sup._ledger[-1].result == "hit"


def test_miss_streak_triggers_review():
    """连续落空才叫模型——这是"频繁判断"的节流阀，不能每帧都问。"""
    sup, clock = _sup(miss_streak_trigger=3)
    for _ in range(3):
        sup._ledger.append(LedgerRow(clock(), "walk", "x", "miss", "200→210"))
    trig = sup._pick_trigger(sup._health(_scene(), {"picks": 0}, clock()), clock())
    assert "落空" in trig


def test_unmatched_screen_has_a_trigger():
    """未识别画面持续太久要问模型（但也只是问，规则层仍然不动）。"""
    sup, clock = _sup(unmatched_trigger_s=10)
    for _ in range(4):
        clock.advance(5)
        sup.note_frame(_scene(False, "unmatched"), "未识别", clock(), {"picks": 0})
    h = sup._health(_scene(False, "unmatched"), {"picks": 0}, clock())
    assert h["unmatched_for"] >= 15
    assert "未识别" in sup._pick_trigger(h, clock())


# ---------------- 预算 ----------------

def test_budget_and_failure_breaker():
    """连续失败要熔断，否则一个挂掉的模型地址会被每 20 秒敲一次。"""
    sup, clock = _sup(min_interval_s=1, max_calls_per_min=2,
                      failure_threshold=2, cooldown_after_failure_s=60)
    for _ in range(2):
        sup._fail_streak += 1
        if sup._fail_streak >= sup.cfg.failure_threshold:
            sup._open_until = clock() + sup.cfg.cooldown_after_failure_s
            sup._fail_streak = 0
            sup.stats["tripped"] += 1
    assert sup.available(clock()) is False
    clock.advance(61)
    assert sup.available(clock()) is True
