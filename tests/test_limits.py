"""放行度 —— 三档必须"点了就能用"，而且绝不能顺手把铁律也放开。

这一组测试盯的是三类事故：
  1. 三档里任何一个值超出 pydantic 约束 → 用户点一下得到"配置校验失败"，
     而错误信息里只有一串字段路径，没人知道是表格哪一行写错了。
  2. 档位 patch 漏了某条限制 → 页面显示"已切到搬砖"，实际有两条还是保守值。
  3. 安全铁律被混进 patch → 这是最严重的，放宽它不会变快，只会变错。
"""

from __future__ import annotations

from autogame import limits as lm
from autogame.config import AppConfig


def _apply(cfg: AppConfig, patch: dict) -> AppConfig:
    return AppConfig.model_validate({**cfg.model_dump(), **patch})


# ---------------- 三档本身 ----------------

def test_every_tier_validates_against_real_config():
    """三档都必须过 pydantic —— 判据不是"看起来合理"，是"能装上"。"""
    base = AppConfig()
    for tier in lm.TIERS:
        cfg = _apply(base, lm.tier_patch(tier))
        assert cfg is not None
        # 抽查几条跨越了不同子模型的字段，确认 patch 真的落到了对应对象上
        assert cfg.decision.tick_hz == lm.tier_patch(tier)["decision"]["tick_hz"]


def test_tier_patch_covers_every_limit():
    """没有漏项：三条限制表里的每一条，三档 patch 里都要有。"""
    for tier in lm.TIERS:
        patch = lm.tier_patch(tier)
        for lim in lm.LIMITS:
            cur = patch
            for part in lim.id.split("."):
                assert part in cur, f"{tier} 缺 {lim.id}"
                cur = cur[part]
            assert cur == lim.value_for(tier)


def test_tiers_are_actually_different():
    """三档不能有一档是空 patch —— 那等于按钮没接上。"""
    patches = [lm.tier_patch(t) for t in lm.TIERS]
    assert patches[0] != patches[1]
    assert patches[1] != patches[2]
    assert patches[0] != patches[2]


def test_unleashed_is_looser_on_the_three_that_matter_most():
    """最关键的三条必须真的更松：决策更快、轨迹更短、休息更短。

    这三条是"跑一夜只干了半晚"的主要来源。它们如果有任何一条在搬砖档里
    反而更紧，说明值写反了（这类错误不会报错，只会表现成"切了没感觉"）。
    """
    base = AppConfig()
    strict = _apply(base, lm.tier_patch("strict"))
    loose = _apply(base, lm.tier_patch("unleashed"))
    assert loose.decision.tick_hz > strict.decision.tick_hz
    assert loose.executor.mouse_path.max_duration_s < strict.executor.mouse_path.max_duration_s
    assert loose.decision.humanize.fatigue.break_seconds[1] < \
        strict.decision.humanize.fatigue.break_seconds[1]
    assert loose.decision.humanize.mood.skip_prob < strict.decision.humanize.mood.skip_prob


def test_capture_fps_keeps_up_with_tick_hz():
    """采集帧率必须 ≥ 决策频率的两倍，否则"决策更快"只是拿旧帧反复决策。"""
    base = AppConfig()
    for tier in lm.TIERS:
        cfg = _apply(base, lm.tier_patch(tier))
        assert cfg.capture.fps >= cfg.decision.tick_hz * 2, f"{tier} 帧率跟不上决策"


def test_safety_never_enters_any_patch():
    """铁律不进档位。

    这里逐条检查"能在配置里表达的那些"：场景门、A框确认帧数。
    其余几条（拾取三道闸、坐标换算）是代码，改不了，所以也不会出现在配置里。
    """
    for tier in lm.TIERS:
        patch = lm.tier_patch(tier)
        assert "vision" not in patch, f"{tier} 动了视觉阈值 —— 预设不该碰检测参数"
        aframe = patch.get("decision", {}).get("aframe_confirm_frames")
        assert aframe is None, f"{tier} 动了 A框确认帧数（防按空），这是铁律"


def test_safety_list_is_not_empty_and_has_locations():
    """安全清单是给用户看的，每条都要有"它在哪"，否则无从核对。"""
    assert len(lm.SAFETY) >= 5
    for s in lm.SAFETY:
        assert s.label and s.what and s.where


# ---------------- 当前档位判定 ----------------

def test_current_tier_matches_default_config():
    """出厂配置就是保守档 —— 否则"默认"这件事本身没有意义。"""
    assert lm.current_tier(AppConfig()) == "strict"
    assert lm.current_tier(_apply(AppConfig(), lm.tier_patch("normal"))) == "normal"
    assert lm.current_tier(_apply(AppConfig(), lm.tier_patch("unleashed"))) == "unleashed"


def test_current_tier_survives_manual_tweaks():
    """手改两三个滑块之后仍要能判出档位，不能变成"未知"。"""
    cfg = _apply(AppConfig(), lm.tier_patch("normal"))
    cfg.decision.tick_hz = 2.2          # 手动微调
    cfg.capture.fps = 5.5
    assert lm.current_tier(cfg) in ("normal", "unleashed")


# ---------------- 对账 ----------------

def test_ledger_counts_only_known_counters():
    """没见过的键不渲染 —— 页面上的每一行都要有解释，不能冒出无名字的数字。"""
    res = lm.ledger({"picks": 10, "a_taps": 12, "dropped": 3})
    keys = {r["key"] for r in res["rows"]}
    assert "dropped" in keys
    assert "picks" not in keys        # picks 是产出，不是"被拦"
    assert res["lost"] == 3


def test_ledger_calls_out_wasted_quota():
    """一半动作没变成产出时必须明说 —— 这种状态最容易被当成"跑得慢"。"""
    res = lm.ledger({"dropped": 50, "targets_ignored": 50})
    assert res["wasted_ratio"] >= 0.25
    assert "配额" in res["summary"] or "没用" in res["summary"]


def test_ledger_quiet_when_nothing_blocked():
    res = lm.ledger({"suppressed": 0, "dropped": 0})
    assert res["lost"] == 0
    assert res["rows"], "即使全是 0 也要给出行，否则页面会空一块"


def test_ledger_uses_executor_counters():
    """执行器上的计数器是运行时唯一真相，不能被 stats 里的缺省值盖掉。"""

    class _Ex:
        sent = 40
        suppressed = 6
        dropped = 2

    res = lm.ledger({}, _Ex())
    assert res["sent"] == 40
    assert res["lost"] == 8


# ---------------- report ----------------

def test_report_shape_for_page():
    r = lm.report(AppConfig(), {"dropped": 1}, None)
    assert set(r) >= {"tier", "tier_title", "tiers", "kinds", "limits", "safety", "ledger"}
    assert len(r["tiers"]) == len(lm.TIERS)
    row = r["limits"][0]
    assert set(row) >= {"id", "label", "kind", "kind_title", "why", "now",
                        "strict", "normal", "unleashed", "relaxed", "tightened"}


def test_every_limit_has_a_reason():
    """每条限制都要写清"它在防什么"。写不出来的限制就是没想清楚该不该留。"""
    for lim in lm.LIMITS:
        assert lim.why and len(lim.why) >= 8, f"{lim.id} 缺少说明"
        assert lim.kind in lm.KINDS, f"{lim.id} 的类别 {lim.kind} 不在 KINDS 里"


def test_limit_paths_all_exist_on_config():
    """点分路径必须真的存在 —— 写错一个字母，页面会静默显示默认值。"""
    cfg = AppConfig()
    for lim in lm.LIMITS:
        obj = cfg
        for part in lim.id.split("."):
            assert hasattr(obj, part), f"{lim.id} 在 {part} 处断掉"
            obj = getattr(obj, part)


# ---------------- HTTP：页面点按钮走的就是这两条 ----------------

def _client(cfg):
    from fastapi.testclient import TestClient

    from autogame.server.app import create_app
    from autogame.server.hub import Hub
    from autogame.vision.pipeline import VisionPipeline
    from autogame.executor import DryRunExecutor

    cfg.server.event_log = ""
    cfg.executor.min_action_interval_ms = 0
    hub = Hub(cfg, DryRunExecutor(cfg.executor), VisionPipeline(cfg.vision))
    return TestClient(create_app(cfg, hub, "<h1>x</h1>")), hub


def test_http_limits_state():
    from autogame.config import AppConfig as C

    cfg = C()
    cfg.server.token = "t"
    c, _ = _client(cfg)
    d = c.get("/api/limits", headers={"X-Token": "t"}).json()
    assert d["tier"] == "strict"
    assert len(d["limits"]) == len(lm.LIMITS)
    assert d["safety"], "安全清单不能为空"
    assert "ledger" in d


def test_http_limits_apply_changes_the_live_config():
    """切档必须落到**正在跑的那个 cfg 对象**上，不是造一份新的。"""
    from autogame.config import AppConfig as C

    cfg = C()
    cfg.server.token = "t"
    c, hub = _client(cfg)
    r = c.post("/api/limits/unleashed", headers={"X-Token": "t"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["state"]["tier"] == "unleashed"
    # 同一个对象（hub.cfg 就是 cfg）
    assert cfg.decision.tick_hz == 4.0
    assert hub.cfg.decision.tick_hz == 4.0
    assert body["restart_required"] == [] or isinstance(body["restart_required"], list)


def test_http_limits_unknown_tier_is_404():
    from autogame.config import AppConfig as C

    cfg = C()
    cfg.server.token = "t"
    c, _ = _client(cfg)
    assert c.post("/api/limits/ludicrous", headers={"X-Token": "t"}).status_code == 404


def test_http_limits_needs_token_when_enabled():
    from autogame.config import AppConfig as C

    cfg = C()
    cfg.server.token = "t"
    cfg.server.require_token = True
    c, _ = _client(cfg)
    assert c.get("/api/limits").status_code == 401


# ---------------- 自治开关：只展示，不进档位 ----------------

def test_modes_report_what_it_dares_to_decide():
    """三档改的是快慢；这一组改的是"谁做决定"，必须分开呈现。"""
    r = lm.report(AppConfig())
    keys = {m["key"] for m in r["modes"]}
    assert keys == {"arbiter", "supervisor", "humanize", "autonomy"}
    for m in r["modes"]:
        assert m["note"], f"{m['key']} 没有解释 —— 用户只会看到 shadow 不知道意味着什么"


def test_modes_never_enter_tier_patches():
    """enforce 不在任何档位里：先跑影子期是纪律，不是性能刻度。"""
    for tier in lm.TIERS:
        patch = lm.tier_patch(tier)
        arb = patch.get("decision", {}).get("arbiter", {})
        sup = patch.get("decision", {}).get("supervisor", {})
        assert "mode" not in arb, f"{tier} 动了裁决模式"
        assert "mode" not in sup, f"{tier} 动了监督层模式"
        assert "humanize" not in patch.get("decision", {}) or \
            "enabled" not in patch["decision"]["humanize"], f"{tier} 把拟人化整个关掉"
