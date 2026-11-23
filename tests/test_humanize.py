"""拟人化行为层测试 —— 轨迹 / 疲劳 / 反应时间 / 走神 / 社交噪音。

本文件**显式打开**拟人化（conftest 默认关掉以保证其余测试确定性），
并用固定 seed 的 Random 让随机行为可复现。
"""

import math
import random
import statistics

from autogame.config import (
    DecisionConfig,
    DriftConfig,
    ExecutorConfig,
    FatigueConfig,
    HumanizeConfig,
    MoodConfig,
    MousePathConfig,
    SocialConfig,
)
from autogame.decision.fsm import DecisionEngine
from autogame.executor import DryRunExecutor
from autogame.humanize import (
    FatigueModel,
    Humanizer,
    build_mouse_path,
    wind_mouse,
)
from autogame.protocol import AFrame, ButtonHit, Pillar, SceneState


# ═══════════════════════════ 鼠标轨迹 ═══════════════════════════


def test_wind_mouse_path_is_curved_not_straight():
    """轨迹必须偏离直线（风力扰动），否则和"两点一线"没区别。"""
    pts = wind_mouse((100, 100), (900, 500), rng=random.Random(1))
    assert len(pts) > 10, "长距离移动必须有足够采样点"

    # 起点→终点直线，计算中间点到直线的最大偏离
    x0, y0 = 100, 100
    x1, y1 = 900, 500
    dx, dy = x1 - x0, y1 - y0
    norm = math.hypot(dx, dy)
    max_dev = 0.0
    for px, py in pts:
        # 点到直线距离
        dev = abs(dy * (px - x0) - dx * (py - y0)) / norm
        max_dev = max(max_dev, dev)
    assert max_dev > 3.0, f"轨迹必须带弧度，实际最大偏离 {max_dev:.2f}px"


def test_wind_mouse_ends_at_target():
    end = (900, 500)
    pts = wind_mouse((0, 0), end, rng=random.Random(2))
    assert pts[-1] == (float(end[0]), float(end[1])), "最后一点必须精确落在目标"


def test_wind_mouse_progress_is_monotonic_toward_target():
    """整体必须是在靠近目标（允许局部抖动，不允许整体跑偏）。"""
    pts = wind_mouse((0, 0), (600, 0), rng=random.Random(3))
    dists = [math.hypot(600 - x, 0 - y) for x, y in pts]
    assert dists[0] > dists[-1]
    assert dists[-1] < 1.0
    # 前 80% 的点里，不能出现"离目标比起点还远"的离谱漂移
    assert max(dists) <= dists[0] + 60


def test_build_mouse_path_disabled_is_single_point():
    cfg = MousePathConfig(enabled=False)
    path = build_mouse_path((10, 10), (800, 600), cfg, rng=random.Random(4))
    assert len(path) == 1
    assert (path[0].x, path[0].y) == (800, 600)
    assert path[0].dt == 0.0


def test_build_mouse_path_enabled_has_points_and_bounded_duration():
    cfg = MousePathConfig()
    path = build_mouse_path((0, 0), (900, 600), cfg, rng=random.Random(5))
    assert 2 < len(path) <= cfg.max_points + 3
    total = sum(p.dt for p in path)
    # 时长受 min/max 钳制，且中途停顿最多 max_micro_pauses 次
    assert cfg.min_duration_s <= total <= cfg.max_duration_s + \
        cfg.max_micro_pauses * cfg.micro_pause_ms[1] / 1000.0
    # 落点允许人类化偏差
    assert abs(path[-1].x - 900) <= cfg.endpoint_jitter_px
    assert abs(path[-1].y - 600) <= cfg.endpoint_jitter_px


def test_build_mouse_path_short_move_skips_curve():
    """几像素的挪动不该硬套曲线（那只会拖慢节奏）。"""
    cfg = MousePathConfig()
    path = build_mouse_path((100, 100), (102, 101), cfg, rng=random.Random(6))
    assert len(path) == 1


def test_build_mouse_path_deterministic_with_seed():
    cfg = MousePathConfig()
    a = build_mouse_path((0, 0), (700, 400), cfg, rng=random.Random(99))
    b = build_mouse_path((0, 0), (700, 400), cfg, rng=random.Random(99))
    assert a == b, "同一 seed 必须得到同一轨迹（可复现/可回归）"


# ═══════════════════════════ 疲劳度 ═══════════════════════════


def test_fatigue_accumulates_and_triggers_break():
    fm = FatigueModel(FatigueConfig(per_action=0.05, break_threshold=0.5),
                      rng=random.Random(7))
    assert not fm.needs_break()
    for i in range(12):
        fm.note_action(float(i))
    assert fm.level >= 0.5 and fm.needs_break()
    assert fm.latency_scale() > 1.0, "疲劳必须拉长延时"


def test_fatigue_rest_recovers():
    fm = FatigueModel(FatigueConfig(per_action=0.05, break_recovery=0.5,
                                    break_seconds=[10.0, 60.0]), rng=random.Random(8))
    for i in range(12):
        fm.note_action(float(i))
    before = fm.level
    fm.note_rest(60.0)
    assert fm.level < before
    assert fm.breaks == 1


def test_fatigue_changes_latency_mean_and_variance():
    """核心断言：延时不是固定区间随机 —— 累/不累的**均值与方差**都不同。

    固定区间随机（uniform(lo,hi)）无论跑多久，均值和方差都锁在同一个盒子里；
    两层状态模型下，疲劳度一变，整个分布跟着平移并展宽。
    """
    fresh = _human()
    tired = _human()
    for i in range(400):
        tired.note_action(float(i))
    s_fresh = [fresh.reaction_ms() for _ in range(300)]
    s_tired = [tired.reaction_ms() for _ in range(300)]
    assert statistics.mean(s_tired) > statistics.mean(s_fresh) * 1.15, "疲劳必须抬高均值"
    assert statistics.pstdev(s_tired) > statistics.pstdev(s_fresh), "疲劳必须放大方差"


def test_fatigue_idle_decays():
    cfg = FatigueConfig(per_action=0.05, idle_after_s=2.0, rest_decay_per_s=0.1)
    fm = FatigueModel(cfg, rng=random.Random(12))
    for i in range(6):
        fm.note_action(float(i))
    lvl = fm.level
    for t in range(0, 20):
        fm.tick(float(t))
    assert fm.level < lvl, "长时间没有动作应视为休息，疲劳自然衰减"


# ═══════════════════════════ 反应时间 ═══════════════════════════


def _human(**over) -> Humanizer:
    cfg = HumanizeConfig(enabled=True)
    for k, v in over.items():
        setattr(cfg, k, v)
    return Humanizer(cfg, pickup_key="a", seed=42)


def test_reaction_ms_in_range_and_right_skewed():
    h = _human()
    samples = [h.reaction_ms() for _ in range(400)]
    lo, hi = h.cfg.reaction_ms
    assert min(samples) >= lo * 0.6 - 1
    assert max(samples) <= hi * 2.5 + 1
    # 右偏：均值 > 中位数（对数正态的典型特征）
    assert statistics.mean(samples) > statistics.median(samples)


def test_reaction_ms_disabled_returns_zero():
    h = _human(enabled=False)
    assert h.reaction_ms() == 0, "关闭时必须返回 0，让调用方回落到确定性确认窗"


def test_latency_and_jitter_scale_with_fatigue():
    h = _human()
    base_lat = h.latency_scale(0.0)
    base_jit = h.jitter_ratio(0.0, 0.15)
    for i in range(400):
        h.note_action(float(i))
    assert h.latency_scale(400.0) > base_lat
    assert h.jitter_ratio(400.0, 0.15) > base_jit


# ═══════════════════════════ 疲劳休息窗口 ═══════════════════════════


def test_paused_break_cycle():
    cfg = HumanizeConfig(enabled=True)
    cfg.fatigue = FatigueConfig(per_action=0.05, break_threshold=0.5,
                                break_seconds=[5.0, 5.0], break_recovery=0.9)
    h = Humanizer(cfg, seed=1)
    for i in range(12):
        h.note_action(float(i))
    assert h.paused(20.0) is True, "超过阈值必须停手"
    assert h.paused(21.0) is True, "休息窗口内持续停手"
    assert h.paused(26.0) is False, "休息结束后必须恢复"
    assert h.break_count == 1
    assert h.fatigue.level < 0.5


# ═══════════════════════════ 走神 ═══════════════════════════


def test_drift_respects_startup_grace():
    cfg = HumanizeConfig(enabled=True)
    cfg.drift = DriftConfig(startup_grace_s=60.0, mean_interval_s=5.0)
    h = Humanizer(cfg, seed=2)
    for t in range(0, 60):
        assert h.maybe_drift(float(t), 1280, 720) is None, "宽限期内不得走神"


def test_drift_emits_only_mouse_move_by_default():
    cfg = HumanizeConfig(enabled=True)
    cfg.drift = DriftConfig(startup_grace_s=0.0, mean_interval_s=5.0,
                            cooldown_s=1.0, noise_keys=[])
    h = Humanizer(cfg, seed=3)
    h.maybe_drift(0.0, 1280, 720)  # 初始化调度
    cmds = None
    for t in range(1, 600):
        cmds = h.maybe_drift(float(t), 1280, 720)
        if cmds:
            break
    assert cmds, "足够长时间后必须出现走神"
    assert all(c.op == "mouse_move" for c in cmds), "默认走神只允许鼠标移动"
    assert all(0 <= c.x < 1280 and 0 <= c.y < 720 for c in cmds), "漂移点必须在画面内"


def test_drift_never_uses_pickup_key_as_noise_key():
    cfg = HumanizeConfig(enabled=True)
    cfg.drift = DriftConfig(startup_grace_s=0.0, mean_interval_s=5.0,
                            cooldown_s=1.0, noise_keys=["a", "i"])
    h = Humanizer(cfg, seed=4)
    h.maybe_drift(0.0, 1280, 720)
    found = None
    for t in range(1, 600):
        found = h.maybe_drift(float(t), 1280, 720)
        if found:
            break
    assert found
    keys = [c.key for c in found if c.op == "key_tap"]
    assert "a" not in keys, "拾取键绝不能作为走神噪音键（铁律）"
    assert "i" in keys


# ═══════════════════════════ 社交噪音 ═══════════════════════════


def test_social_disabled_by_default():
    assert SocialConfig().enabled is False
    h = _human()
    assert h.maybe_social(0.0) is None


def test_validate_phrase_blocks_hotkey_chars():
    h = _human()
    ok, why = h.validate_phrase("hello")
    assert ok, why
    for bad in ("hi a", "wave", "see you", "sword"):
        ok, why = h.validate_phrase(bad)
        assert not ok, f"{bad!r} 含移动/拾取键字符，必须拒绝"
        assert "禁用字符" in why
    ok, _ = h.validate_phrase("有人在吗")
    assert not ok, "中文无法用 HID 键盘输入，必须拒绝"


def test_social_enabled_sends_typing_sequence():
    cfg = HumanizeConfig(enabled=True)
    cfg.social = SocialConfig(enabled=True, phrases=["gg"], min_interval_s=30.0,
                              chat_open_key="enter", chat_send_key="enter")
    h = Humanizer(cfg, seed=5)
    h.maybe_social(0.0)  # 初始化调度
    cmds = None
    for t in range(1, 200):
        cmds = h.maybe_social(float(t))
        if cmds:
            break
    assert cmds
    assert cmds[0].key == "enter" and cmds[-1].key == "enter"
    typed = [c.key for c in cmds[1:-1]]
    assert typed == ["g", "g"]
    assert h.social_count == 1


def test_social_refuses_unsafe_phrase():
    cfg = HumanizeConfig(enabled=True)
    cfg.social = SocialConfig(enabled=True, phrases=["wave"], min_interval_s=30.0)
    h = Humanizer(cfg, seed=6)
    h.maybe_social(0.0)
    sent = None
    for t in range(1, 200):
        sent = h.maybe_social(float(t))
        if h.social_refused:
            break
    assert sent is None, "含热键字符的台词必须被拒绝，不得'尽力发送'"
    assert h.social_refused >= 1
    assert h.social_count == 0


# ═══════════════════════════ 执行器集成 ═══════════════════════════


def test_executor_plans_path_and_tracks_cursor():
    ex = DryRunExecutor(ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000))
    from autogame.protocol import Command

    ex.execute(Command(op="mouse_move", x=100, y=100))
    assert ex.cursor is not None

    ex.execute(Command(op="mouse_click", x=900, y=600))
    assert len(ex.last_path) > 1, "长距离点击必须规划多点轨迹"
    assert abs(ex.cursor[0] - 900) <= 2

    # 第二次点击从上一个落点出发
    prev = ex.cursor
    ex.execute(Command(op="mouse_click", x=200, y=200))
    assert ex.last_path[0].x == prev[0] and ex.last_path[0].y == prev[1]
    assert ex.journal[-1]["path"] == len(ex.last_path)


def test_executor_path_disabled_single_point():
    ecfg = ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000)
    ecfg.mouse_path.enabled = False
    ex = DryRunExecutor(ecfg)
    from autogame.protocol import Command

    ex.execute(Command(op="mouse_click", x=900, y=600))
    assert len(ex.last_path) == 1


def test_executor_key_commands_have_no_path():
    ex = DryRunExecutor(ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000))
    from autogame.protocol import Command

    ex.execute(Command(op="key_tap", key="w", hold_ms=50))
    assert ex.last_path == []


# ═══════════════════════════ FSM 集成 ═══════════════════════════


def _scene(ts, pillars=None, aframe=None, rechallenge=None, delta=50.0):
    return SceneState(
        ts_ms=ts, w=1280, h=720, character=(640, 360),
        pillars=pillars or [], aframe=aframe, rechallenge=rechallenge,
        frame_delta=delta,
    )


def _pillar(cx=700, cy=400):
    return Pillar(x=cx, y=cy + 55, cx=cx, cy=cy, w=14, h=110, color="gold",
                  area=1500, distance=((cx - 640) ** 2 + (cy - 360) ** 2) ** 0.5,
                  whitelisted=True)


def _engine(humanize: HumanizeConfig, seed=7):
    cfg = DecisionConfig(tick_hz=10.0, approach_mode="mouse_click")
    cfg.autonomy.enabled = False
    cfg.humanize = humanize
    # 亚帧打散默认 400ms：真实环境下这是防帧锁相的关键，但在测试里每点一次
    # 就多睡最多 400ms，50 多个用例会拖到分钟级。这里关掉，单独用
    # test_desync_* 验证接线（那两条用例把值调小到可测的量级）。
    cfg.action_desync_ms = 0
    ex = DryRunExecutor(ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000))
    eng = DecisionEngine(cfg, ex)
    # 固定 seed：FSM 自己 new 的 Humanizer 无 seed，这里替换成可复现的
    eng.human = Humanizer(humanize, pickup_key=cfg.pickup_key, seed=seed)
    eng.start()
    return eng, ex


def test_fsm_reaction_delay_before_pickup_tap():
    """看到 A框 → 反应时间到达前不得按拾取键（这是"人"而不是"反射")。"""
    hz = HumanizeConfig(enabled=True)
    hz.reaction_ms = [400, 450]
    hz.reaction_sigma = 0.01
    eng, ex = _engine(hz)
    af = AFrame(x=700, y=250, w=32, h=32, score=0.8)

    eng.on_scene(_scene(1000, [_pillar()], aframe=af))   # 0ms：刚看到
    eng.on_scene(_scene(1200, [_pillar()], aframe=af))   # 200ms：还没到反应时间
    taps = [j for j in ex.journal if j["op"] == "key_tap" and j["key"] == "a"]
    assert not taps, "反应时间未到不得按拾取键"

    for i in range(3, 8):
        eng.on_scene(_scene(1000 + i * 200, [_pillar()], aframe=af))
    taps = [j for j in ex.journal if j["op"] == "key_tap" and j["key"] == "a"]
    assert taps, "反应时间到达后必须按拾取键"


def test_fsm_fatigue_break_suppresses_all_commands():
    hz = HumanizeConfig(enabled=True)
    hz.fatigue = FatigueConfig(per_action=0.05, break_threshold=0.25,
                               break_seconds=[30.0, 30.0], break_recovery=0.9)
    eng, ex = _engine(hz)
    # 连续走位消耗动作 → 触发休息
    hit = -1
    for i in range(20):
        eng.on_scene(_scene(1000 + i * 1000, [_pillar()]))
        if any(e["event"] == "human_break" for e in eng.events):
            hit = i
            break
    assert hit >= 0, "疲劳到阈值必须停手"

    before = len(ex.journal)
    for k in range(1, 6):
        eng.on_scene(_scene(1000 + (hit + k) * 1000, [_pillar()]))
    assert len(ex.journal) == before, "休息窗口内不得发出任何命令"
    assert eng.stats["human_breaks"] == 1


def test_fsm_drift_emits_noise_and_skips_a_beat():
    hz = HumanizeConfig(enabled=True)
    hz.drift = DriftConfig(startup_grace_s=0.0, mean_interval_s=5.0, cooldown_s=1.0)
    eng, ex = _engine(hz, seed=11)

    hit = False
    for i in range(600):
        ex.journal.clear()
        eng.on_scene(_scene(1000 + i * 500))
        if eng.events and eng.events[-1]["event"] == "drift":
            hit = True
            break
    assert hit, "长时间空闲必须出现走神"
    assert eng.stats["drifts"] >= 1
    # 走神那一拍只允许鼠标移动
    assert all(j["op"] == "mouse_move" for j in ex.journal)


def test_fsm_hesitation_delays_rechallenge_click():
    hz = HumanizeConfig(enabled=True)
    hz.drift = DriftConfig(hesitate_prob=1.0, hesitate_s=[1.0, 1.0])
    eng, ex = _engine(hz)
    btn = ButtonHit(x=640, y=570, w=240, h=52, score=0.9)

    eng.on_scene(_scene(1000, rechallenge=btn))   # 首帧只记录
    eng.on_scene(_scene(1200, rechallenge=btn))   # 确认 → 开始犹豫
    clicks = [j for j in ex.journal if j["op"] == "mouse_click"]
    assert not clicks, "犹豫窗口内不得点击"

    for i in range(3, 6):   # 1200ms + 1s 犹豫 → 2200ms 之后才点
        eng.on_scene(_scene(1000 + i * 500, rechallenge=btn))
    clicks = [j for j in ex.journal if j["op"] == "mouse_click"]
    assert clicks, "犹豫结束后必须点击『再次挑战』"
    assert eng.stats["rechallenges"] == 1


def test_fsm_fatigue_causes_ground_misclicks_only():
    """疲劳失误只发生在**地面点击**上：走错一步自己纠正，绝不涉及按键。

    这里把 error_prob 直接钉成 1.0（而不是靠概率碰运气），
    验证的是"失误接线正确且不越界"，而不是随机数本身。
    """
    hz = HumanizeConfig(enabled=True)
    hz.fatigue = FatigueConfig(per_action=0.05, break_threshold=1.0,
                               error_boost=0.3, latency_boost=0.0)
    eng, ex = _engine(hz, seed=13)
    for i in range(15):
        eng.human.note_action(float(i))
    assert 0.0 < eng.human.error_prob(20.0) <= 0.5, "疲劳必须抬高失误概率"

    eng.human.error_prob = lambda now, base=0.0: 1.0  # 强制每次都失误
    for i in range(3):
        eng.on_scene(_scene(1000 + i * 1000, [_pillar()]))

    assert eng.stats["misclicks"] >= 1, "失误接线必须生效"
    assert any(e["event"] == "human_misclick" for e in eng.events)
    # 失误全部是 mouse_click，且没有任何拾取键按键
    assert all(j["op"] == "mouse_click" for j in ex.journal)
    assert not [j for j in ex.journal if j["op"] == "key_tap"]
    assert eng.stats["blind_tap_violations"] == 0


# ═══════════════════════ 走位等待（动作链：点一下 ≠ 走到位） ═══════════════════════
#
# 真实链路是：点地面 → 人物在游戏里走过去（1-3s）→ 进到一定距离 → A框出现 → 按拾取键。
# 这些测试锁死"中间那几步"：走位期间不许重复点击、停下要给 A框宽限、
# 有效接近要能继续点、无进展要能进撞墙逻辑。


def _pillar_d(cx=900, cy=400, dist=520.0):
    """同屏位置不变、但距离可变（角色走过去时相机跟随，光柱屏幕坐标不变）。"""
    return Pillar(x=cx, y=cy + 55, cx=cx, cy=cy, w=14, h=110, color="gold",
                  area=1500, distance=dist, whitelisted=True)


def _clicks(ex):
    return [j for j in ex.journal if j["op"] == "mouse_click"]


def test_walk_wait_suppresses_reclick_while_moving():
    """人物还在走（帧差高）→ 不许重复点同一落点。"""
    eng, ex = _engine(HumanizeConfig(enabled=True))
    eng.on_scene(_scene(1000, [_pillar_d()], delta=40.0))
    assert len(_clicks(ex)) == 1, "首次接近应该点一次地面"

    eng.on_scene(_scene(2000, [_pillar_d(dist=460)], delta=40.0))  # 1s：还在走
    eng.on_scene(_scene(3000, [_pillar_d(dist=380)], delta=40.0))  # 2s：还在走
    assert len(_clicks(ex)) == 1, "走位等待期间不得重复点击"


def test_walk_wait_reclicks_after_budget():
    """走位预算耗尽（人还没到）→ 允许再点一次。"""
    eng, ex = _engine(HumanizeConfig(enabled=True))
    eng.on_scene(_scene(1000, [_pillar_d()], delta=40.0))
    for i in range(1, 6):
        eng.on_scene(_scene(1000 + i * 1000, [_pillar_d(dist=520 - i * 20)], delta=40.0))
    assert len(_clicks(ex)) >= 2, "预算耗尽后必须重新发起走位"
    assert eng.stats["walk_starts"] >= 2
    assert eng.stats["walk_ends"] >= 1


def test_walk_settle_with_progress_reclicks_immediately():
    """人物提前停下且确实靠近了 → 立刻再点（不用等满预算）。"""
    eng, ex = _engine(HumanizeConfig(enabled=True))
    eng.on_scene(_scene(1000, [_pillar_d(dist=520)], delta=40.0))
    assert len(_clicks(ex)) == 1
    # 画面停了（帧差低）+ 距离明显缩短 = 走了一段但没到
    for i in range(1, 4):
        eng.on_scene(_scene(1000 + i * 1000, [_pillar_d(dist=300)], delta=0.5))
    assert eng.stats["walk_settles"] >= 1, "停下应被识别"
    assert eng.stats["walk_progress"] >= 1, "距离缩短必须算有效接近"
    assert len(_clicks(ex)) >= 2, "有效接近后应继续发起走位"
    assert eng.stats["abandons"] == 0, "有进展不得放弃"


def test_walk_settle_without_progress_goes_to_stuck_logic():
    """点了没走近（撞墙）→ 不算进展，进入撞墙判定而不是无限重试。"""
    eng, ex = _engine(HumanizeConfig(enabled=True))
    eng.on_scene(_scene(1000, [_pillar_d(dist=520)], delta=40.0))
    for i in range(1, 8):
        eng.on_scene(_scene(1000 + i * 1000, [_pillar_d(dist=520)], delta=0.5))
    assert eng.stats["walk_no_progress"] >= 1
    assert eng._alt_direction_tried or eng.stats["abandons"] >= 1, \
        "无进展必须走撞墙/放弃路径，不能原地重试"


def test_aframe_during_walk_ends_wait_and_taps():
    """走位途中 A框出现（= 走到拾取距离内）→ 结束等待 → 反应后按键。"""
    eng, ex = _engine(HumanizeConfig(enabled=True))
    eng.on_scene(_scene(1000, [_pillar_d(dist=520)], delta=40.0))
    assert eng._walking is True

    af = AFrame(x=900, y=250, w=32, h=32, score=0.8)
    eng.on_scene(_scene(2000, [_pillar_d(dist=120)], aframe=af, delta=40.0))
    assert eng._walking is False, "A框出现即证明到位，走位等待必须结束"
    assert any(e["event"] == "walk_end" and e.get("reason") == "aframe"
               for e in eng.events)

    # 2000 那一帧是"看见 A框"→先停步（停步本身不算确认帧），所以要多等一帧
    eng.on_scene(_scene(3000, [_pillar_d(dist=120)], aframe=af, delta=40.0))
    eng.on_scene(_scene(4000, [_pillar_d(dist=120)], aframe=af, delta=40.0))
    taps = [j for j in ex.journal if j["op"] == "key_tap" and j["key"] == "a"]
    assert taps, "到位后必须按拾取键"


def test_explore_does_not_spam_clicks():
    """探索（无目标）也走同一条链：点一次 → 等人走 → 再点，不是每秒一次。"""
    eng, ex = _engine(HumanizeConfig(enabled=True))
    eng.cfg.walk.budget_s = 4.0
    eng.cfg.explore_enabled = True   # 打开探索（_engine 里默认关 autonomy）

    eng.on_scene(_scene(1000, [], delta=40.0))     # 无目标 → 探索点一次
    first = len(_clicks(ex))
    assert first == 1

    eng.on_scene(_scene(2000, [], delta=40.0))
    eng.on_scene(_scene(3000, [], delta=40.0))
    assert len(_clicks(ex)) == first, "探索走位期间不得重复点击"

    eng.on_scene(_scene(5000, [], delta=40.0))     # 超过预算
    assert len(_clicks(ex)) > first, "预算耗尽后继续探索"


def test_walk_disabled_falls_back_to_old_cadence():
    """关闭走位等待 → 退回旧的"每 tick 重复点"（给排障留退路）。"""
    hz = HumanizeConfig(enabled=True)
    eng, ex = _engine(hz)
    eng.cfg.walk.enabled = False
    for i in range(3):
        eng.on_scene(_scene(1000 + i * 1000, [_pillar_d(dist=520 - i * 10)], delta=40.0))
    assert len(_clicks(ex)) >= 2, "关闭后应恢复旧的重复点击节拍"


def test_fsm_humanize_snapshot_exposed():
    hz = HumanizeConfig(enabled=True)
    eng, _ = _engine(hz)
    snap = eng.autonomy_snapshot()["humanize"]
    assert snap["enabled"] is True
    assert "fatigue" in snap and "drifts" in snap


def test_executor_click_gesture_press_duration_varies():
    """一次点击不是一个瞬间事件：按压时长必须随机（不是固定 45ms）。"""
    from autogame.protocol import Command

    ex = DryRunExecutor(ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000))
    for _ in range(20):
        ex.execute(Command(op="mouse_click", x=600, y=400))
    presses = [j["press_ms"] for j in ex.journal]
    assert len(presses) == 20
    assert all(55 <= p <= 135 for p in presses), f"按压时长越界: {presses}"
    assert len(set(presses)) > 3, "按压时长必须随机变化，不能是固定值"
    assert ex.dropped == 0, "连续点击不得被限速器丢弃"


# ═══════════════════════ 情绪/偏好：选哪个目标 ═══════════════════════
#
# 最硬的破绽不是"怎么走"，而是"选哪个"：原实现是 (stables or candidates)[0]，
# 永远第一个。下面是这条确定性规则的替代方案。


def _pillar_c(cx, cy, color="gold", dist=300.0):
    return Pillar(x=cx, y=cy + 55, cx=cx, cy=cy, w=14, h=110, color=color,
                  area=1500, distance=dist, whitelisted=True, stable=True)


def _mood_engine(**mood_over):
    hz = HumanizeConfig(enabled=True)
    for k, v in mood_over.items():
        setattr(hz.mood, k, v)
    eng, ex = _engine(hz, seed=21)
    return eng, ex


def test_softmax_choice_is_not_argmax():
    """softmax 不是 argmax：次优项也会被选中（这正是"人味"）。"""
    from autogame.humanize import softmax_choice

    rng = random.Random(5)
    picks = [softmax_choice([1.0, 0.85], 0.35, rng) for _ in range(400)]
    assert picks.count(0) > 0 and picks.count(1) > 0, "两项都必须被选到"
    assert picks.count(0) > picks.count(1), "最优项仍应更常被选（不是纯随机）"


def test_mood_utility_impatient_prefers_near():
    """没耐心 + 不贪 → 只想要近的。"""
    from autogame.humanize import FatigueModel, MoodModel

    cfg = MoodConfig(patience=0.0, greed=0.0, novelty=0.0, skip_prob=0.0)
    mood = MoodModel(cfg, FatigueModel(FatigueConfig(), random.Random(1)), random.Random(1))
    near = mood.utility(80.0, "pink", 1.0)
    far = mood.utility(900.0, "gold", 1.0)
    assert near > far, "没耐心时不贪心也不愿意走远"


def test_mood_utility_greedy_prefers_valuable():
    """贪心 → 愿意为稀有度走远。"""
    from autogame.humanize import FatigueModel, MoodModel

    cfg = MoodConfig(patience=0.1, greed=1.0, novelty=0.0, skip_prob=0.0)
    mood = MoodModel(cfg, FatigueModel(FatigueConfig(), random.Random(2)), random.Random(2))
    far_gold = mood.utility(900.0, "gold", 1.0)
    near_pink = mood.utility(100.0, "pink", 1.0)
    assert far_gold > near_pink, "贪心时应当愿意绕远捡金"


def test_mood_fatigue_reduces_patience_and_greed():
    from autogame.humanize import FatigueModel, MoodModel

    fat = FatigueModel(FatigueConfig(per_action=0.05), random.Random(3))
    mood = MoodModel(MoodConfig(patience=0.8, greed=0.8), fat, random.Random(3))
    before = mood.effective()
    for i in range(20):
        fat.note_action(float(i))
    after = mood.effective()
    assert after["patience"] < before["patience"]
    assert after["greed"] < before["greed"]


def test_mood_drifts_over_time():
    """偏好必须漂移，不能是固定权重（否则又变成一条确定规则）。"""
    from autogame.humanize import FatigueModel, MoodModel

    mood = MoodModel(MoodConfig(drift_sigma=0.15), FatigueModel(FatigueConfig()),
                     random.Random(7))
    seen = set()
    for t in range(0, 120):
        mood.tick(float(t))
        seen.add(round(mood.patience, 2))
    assert len(seen) > 3, "耐心值必须在漂移"


def test_pick_target_chooses_valuable_when_greedy():
    """贪心 + 低温度 → 选远处的金而不是脚边的粉。"""
    eng, _ = _mood_engine(greed=1.0, patience=0.1, novelty=0.0,
                          temperature=0.01, skip_prob=0.0)
    scene = _scene(1000, [_pillar_c(700, 360, "pink", dist=80),
                          _pillar_c(1100, 360, "gold", dist=900)], delta=50.0)
    t = eng._pick_target(scene, 1.0)
    assert t is not None and t.color == "gold"


def test_pick_target_chooses_near_when_impatient():
    eng, _ = _mood_engine(greed=0.0, patience=0.0, novelty=0.0,
                          temperature=0.01, skip_prob=0.0)
    scene = _scene(1000, [_pillar_c(1100, 360, "gold", dist=900),
                          _pillar_c(700, 360, "pink", dist=80)], delta=50.0)
    t = eng._pick_target(scene, 1.0)
    assert t is not None and t.distance == 80


def test_pick_target_falls_back_to_first_when_humanize_off():
    """拟人化关闭 → 退回确定性选择（旧行为，可复现）。"""
    eng, _ = _engine(HumanizeConfig(enabled=False), seed=5)
    assert eng.human.mood_active is False
    scene = _scene(1000, [_pillar_c(1100, 360, "gold", dist=900),
                          _pillar_c(700, 360, "pink", dist=80)], delta=50.0)
    for _ in range(10):
        t = eng._pick_target(scene, 1.0)
        assert t is not None and t.cx == 1100, "关闭时必须恒选第一个候选"


def test_pick_target_ignores_and_returns_none_when_all_skipped():
    """全部被"懒得捡"过滤 → 返回 None（继续走，不弯腰）。"""
    eng, _ = _mood_engine(skip_prob=0.5)
    eng.human.mood.should_skip = lambda d: True     # 强制无视
    scene = _scene(1000, [_pillar_c(700, 360), _pillar_c(900, 400)], delta=50.0)
    assert eng._pick_target(scene, 1.0) is None
    assert eng.stats["targets_ignored"] == 1
    assert any(e["event"] == "target_ignored" for e in eng.events)
    # 无视后一段时间内不再考虑它们
    assert eng._pick_target(scene, 2.0) is None


def test_ignore_expires():
    eng, _ = _mood_engine(skip_prob=0.5)
    eng.human.mood.should_skip = lambda d: True
    scene = _scene(1000, [_pillar_c(700, 360), _pillar_c(900, 400)], delta=50.0)
    assert eng._pick_target(scene, 1.0) is None
    eng.human.mood.should_skip = lambda d: False
    assert eng._pick_target(scene, 200.0) is not None, "无视必须会过期"


def test_novelty_penalizes_repeat_same_color_and_place():
    eng, _ = _mood_engine()
    p = _pillar_c(700, 360, "gold")
    assert eng._novelty(p, 1.0) == 1.0
    eng._note_recent_pick(p, 1.0)
    assert eng._novelty(p, 2.0) < 1.0, "刚捡过同色同位置必须打折"
    other = _pillar_c(200, 600, "pink")
    assert eng._novelty(other, 2.0) > eng._novelty(p, 2.0)


def test_mood_snapshot_exposed():
    eng, _ = _mood_engine()
    snap = eng.autonomy_snapshot()["humanize"]["mood"]
    assert "patience" in snap and "effective" in snap


# ═══════════════ 防"按空"：绝不盲按拾取键 ═══════════════
#
# 帧是"过去"的信息（采集→编码→传输→检测，几百毫秒）。按 A 的那一刻，人物
# 可能早已离开拾取半径 —— 模拟器里就记为 blind_a_press（铁律违规）。
# 两个硬判据：① A框连续确认够帧数；② 距离没有变大（在远离就不按）。


def _aframe_engine(confirm_frames=2):
    hz = HumanizeConfig(enabled=True)
    hz.reaction_ms = [200, 210]
    hz.reaction_sigma = 0.01
    eng, ex = _engine(hz)
    eng.cfg.aframe_confirm_frames = confirm_frames
    return eng, ex


def _taps(ex):
    return [j for j in ex.journal if j["op"] == "key_tap" and j["key"] == "a"]


def test_tap_requires_aframe_confirmation_frames():
    """必须连续确认够帧数才按（单帧 A框可能是旧信息/误检闪烁）。"""
    eng, ex = _aframe_engine(confirm_frames=3)
    af = AFrame(x=900, y=250, w=32, h=32, score=0.8)
    # 第 1 帧在 SCANNING（选目标），从第 2 帧起才计入确认帧数
    for i in range(3):
        eng.on_scene(_scene(1000 + i * 1000, [_pillar_d(dist=120)], aframe=af, delta=0.5))
    assert not _taps(ex), "确认帧数不够时不得按 A"

    # 第 2 帧是停步帧（看见 A框先停步），确认帧从它之后重新数 → 需要再一帧
    eng.on_scene(_scene(4000, [_pillar_d(dist=120)], aframe=af, delta=0.5))
    eng.on_scene(_scene(5000, [_pillar_d(dist=120)], aframe=af, delta=0.5))
    assert _taps(ex), "确认够帧数后必须按（不能永远等下去）"


def test_no_tap_when_character_moving_away():
    """距离在变大（路过/走过头）→ 绝不按，否则必然按空。"""
    eng, ex = _aframe_engine(confirm_frames=2)
    af = AFrame(x=900, y=250, w=32, h=32, score=0.8)
    # 首帧距离 120，之后一路变大（人物在离开）
    dists = [120, 120, 160, 220, 300, 380]
    for i, d in enumerate(dists):
        eng.on_scene(_scene(1000 + i * 1000, [_pillar_d(dist=d)], aframe=af, delta=40.0))
    assert not _taps(ex), "人物在远离时按 A 必然是盲按"


def test_tap_when_approaching_and_confirmed():
    """在靠近 + 确认够帧数 → 正常按。"""
    eng, ex = _aframe_engine(confirm_frames=2)
    af = AFrame(x=900, y=250, w=32, h=32, score=0.8)
    # 112 那一帧是停步帧，确认帧从它之后重新数，所以补一帧 110
    for i, d in enumerate([120, 118, 112, 110]):
        eng.on_scene(_scene(1000 + i * 1000, [_pillar_d(dist=d)], aframe=af, delta=40.0))
    assert _taps(ex), "在靠近且确认够帧数时必须按"
