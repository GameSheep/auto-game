"""FSM 决策状态机单元测试 —— 核心铁律验证。

FSM 全部计时基于帧时间戳（scene.ts_ms），测试用显式时间序列保证确定性。
"""

from autogame.config import DecisionConfig, ExecutorConfig
from autogame.decision.fsm import DecisionEngine, State
from autogame.executor import DryRunExecutor
from autogame.protocol import AFrame, Pillar, SceneState


def _scene(pillars=None, aframe=None, delta=50.0, ts_ms=1000):
    return SceneState(
        ts_ms=ts_ms,
        w=1280,
        h=720,
        character=(640, 360),
        pillars=pillars or [],
        aframe=aframe,
        frame_delta=delta,
    )


def _pillar(cx=900, cy=400, distance=None, whitelisted=True):
    return Pillar(
        x=cx, y=cy + 55, cx=cx, cy=cy, w=14, h=110,
        color="gold", area=1500, distance=distance or ((cx - 640) ** 2 + (cy - 360) ** 2) ** 0.5,
        whitelisted=whitelisted,
    )


def _mk_engine(**cfg_overrides):
    # 单元测试隔离：自主探索/恢复会引入额外命令，这里默认关（自主性见 test_autonomy.py）
    dcfg = DecisionConfig(tick_hz=10.0, approach_mode="keys")
    dcfg.autonomy.enabled = False
    for k, v in cfg_overrides.items():
        setattr(dcfg, k, v)
    ecfg = ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000)
    ex = DryRunExecutor(ecfg)
    return DecisionEngine(dcfg, ex), ex


def test_no_blind_a_press_without_aframe():
    """铁律：A框未出现时绝不按 A。"""
    eng, ex = _mk_engine()
    eng.start()
    for i in range(5):
        eng.on_scene(_scene([_pillar(cx=700, cy=400)], ts_ms=1000 + i * 100))
    a_cmds = [j for j in ex.journal if j["op"] == "key_tap" and j["key"] == "a"]
    assert len(a_cmds) == 0, "无 A框时绝不允许按 A"
    assert eng.stats["blind_tap_violations"] == 0


def test_a_press_only_when_aframe_visible():
    """A框出现 → 按拾取键；A框消失 → 判定拾取成功。"""
    eng, ex = _mk_engine()
    eng.start()
    af = AFrame(x=700, y=250, w=32, h=32, score=0.8)
    # 0.2s 步进：第2帧(0.2s≥0.15s确认窗)触发拾取
    for i in range(4):
        eng.on_scene(_scene([_pillar(cx=700, cy=400)], aframe=af, ts_ms=1000 + i * 200))
    a_cmds = [j for j in ex.journal if j["op"] == "key_tap" and j["key"] == "a"]
    assert len(a_cmds) >= 1, "A框出现必须按拾取键"
    assert eng.stats["a_taps"] == len(a_cmds)
    # 下一帧 A框消失 → 拾取成功
    eng.on_scene(_scene([], ts_ms=1000 + 4 * 200))
    assert eng.stats["picks"] == 1
    assert eng.state in (State.COOLDOWN, State.SCANNING)


def test_whitelisted_only():
    """非白名单目标不触发任何动作。"""
    eng, ex = _mk_engine()
    eng.start()
    eng.on_scene(_scene([_pillar(whitelisted=False)], ts_ms=1000))
    assert len(ex.journal) == 0
    assert eng.state == State.SCANNING


def test_timeout_abandon_and_blacklist():
    """走位超时 → 放弃 + 区域拉黑。"""
    eng, ex = _mk_engine(approach_timeout_s=0.0)
    eng.start()
    eng.on_scene(_scene([_pillar(cx=900, cy=400)], ts_ms=1000))
    assert eng.state == State.APPROACHING
    eng.on_scene(_scene([_pillar(cx=900, cy=400)], ts_ms=2500))  # 超时
    assert eng.stats["abandons"] == 1
    assert eng.state == State.COOLDOWN
    # 拉黑后同位置目标不再被选中（ts=2800 必然仍在冷却 [2.9,3.4] 之前）
    eng.on_scene(_scene([_pillar(cx=900, cy=400)], ts_ms=2800))
    assert eng.state == State.COOLDOWN  # 还在冷却
    # 跳过冷却（帧时间 30s 后）→ 同位置仍被拉黑
    eng.on_scene(_scene([_pillar(cx=900, cy=400)], ts_ms=1000 + 30000))
    assert eng.state == State.SCANNING
    assert eng.stats["approach_starts"] == 1, "拉黑期内同位置不得再次触发走位"


def test_stuck_wall_detection():
    """连续帧差极低 → 撞墙 → 侧向脱困 → 仍卡则放弃。

    注意：现在"点一下"之后有走位等待（decision.walk）——人物没走过去之前
    不会重复点击。所以这里用 1s 步进（真实决策节拍），等走位预算耗尽
    后才进入撞墙判定。
    """
    eng, ex = _mk_engine(stuck_frames=2, stuck_delta_thresh=6.0)
    eng.start()
    ts = 1000
    eng.on_scene(_scene([_pillar(cx=900, cy=400)], delta=100, ts_ms=ts)); ts += 1000
    assert eng.state == State.APPROACHING
    for _ in range(8):
        if eng._alt_direction_tried:
            break
        eng.on_scene(_scene([_pillar(cx=900, cy=400)], delta=1.0, ts_ms=ts)); ts += 1000
    # 触发侧向脱困
    assert eng._alt_direction_tried is True
    for _ in range(6):
        if eng.state == State.COOLDOWN:
            break
        eng.on_scene(_scene([_pillar(cx=900, cy=400)], delta=1.0, ts_ms=ts)); ts += 1000
    assert eng.stats["abandons"] == 1, "侧向脱困无效必须放弃"
    assert eng.state == State.COOLDOWN


def test_panic_stops_everything():
    eng, ex = _mk_engine()
    eng.start()
    eng.on_scene(_scene([_pillar(cx=700, cy=400)],
                        aframe=AFrame(x=700, y=250, w=32, h=32, score=0.8), ts_ms=1000))
    eng.panic()
    assert eng.state == State.PANIC
    assert ex.panicked
    n = len(ex.journal)
    eng.on_scene(_scene([_pillar(cx=700, cy=400)], ts_ms=2000))
    assert len(ex.journal) == n, "PANIC 后不得再产生输出"
    eng.reset_panic()
    assert eng.state == State.SCANNING


def test_cooldown_then_rescan():
    """拾取成功后进入冷却，冷却结束恢复扫描。"""
    eng, ex = _mk_engine()
    eng.start()
    af = AFrame(x=700, y=250, w=32, h=32, score=0.8)
    # 第1帧起步走位；第2帧看见A框→先停步；第3帧停稳确认；第4帧按键；第5帧A框消失→拾取成功
    eng.on_scene(_scene([_pillar(cx=700, cy=400)], aframe=af, ts_ms=1000))
    eng.on_scene(_scene([_pillar(cx=700, cy=400)], aframe=af, ts_ms=1300))
    eng.on_scene(_scene([_pillar(cx=700, cy=400)], aframe=af, ts_ms=1600))
    eng.on_scene(_scene([_pillar(cx=700, cy=400)], aframe=af, ts_ms=1900))
    eng.on_scene(_scene([], ts_ms=2200))  # A框消失 → 成功
    assert eng.stats["picks"] == 1
    assert eng.state == State.COOLDOWN
    eng.on_scene(_scene([], ts_ms=2000))
    assert eng.state == State.COOLDOWN  # 冷却未结束
    eng.on_scene(_scene([], ts_ms=1000 + 5000))  # 5s 后冷却必已结束
    assert eng.state == State.SCANNING


def test_aframe_while_walking_stops_first_then_presses():
    """铁律补强：走位中看见 A框，必须先停步、只认停步后的新帧，才允许按 A。

    为什么：A框是过去某一帧的证据，而点地面的航点不会因为"看见 A框"被取消。
    采集→检测几百毫秒里人物还在走，A键落下时可能已出拾取半径 —— 真实后果就是
    "明明看见提示却按空"（模拟器 blind_a_press，2026-11-23 全量负载下复现）。
    """
    eng, ex = _mk_engine(approach_mode="mouse_click")
    eng.start()
    af = AFrame(x=700, y=250, w=32, h=32, score=0.8)
    pil = _pillar(cx=700, cy=400)

    eng.on_scene(_scene([pil], ts_ms=1000))        # 起步走位（点地面）
    assert eng._walking, "第一帧必须真的发起走位，否则这条用例没测到东西"

    eng.on_scene(_scene([pil], aframe=af, ts_ms=1300))   # 走位途中看见 A框
    taps = [j for j in ex.journal if j["op"] == "key_tap" and j["key"] == "a"]
    assert not taps, "走位中看见 A框不能直接按 A（人还在走，按下可能就空）"
    stops = [j for j in ex.journal
             if j["op"] == "mouse_click" and (j["x"], j["y"]) == (640, 360)]
    assert stops, "必须先点自己脚下停步"
    assert eng.stats["pick_stops"] == 1

    # 停步那一帧及更早的帧都不算"已停稳"：同一时间戳再来一次也不按
    eng.on_scene(_scene([pil], aframe=af, ts_ms=1300))
    assert not [j for j in ex.journal if j["op"] == "key_tap" and j["key"] == "a"]

    # 停步之后的新帧：连续确认够 2 帧才按
    eng.on_scene(_scene([pil], aframe=af, ts_ms=1600))
    assert not [j for j in ex.journal if j["op"] == "key_tap" and j["key"] == "a"], \
        "只来了一帧新证据，还不够"
    eng.on_scene(_scene([pil], aframe=af, ts_ms=1900))
    assert [j for j in ex.journal if j["op"] == "key_tap" and j["key"] == "a"], \
        "停稳后的连续两帧都看见 A框 → 该按"


def test_stop_before_pick_can_be_turned_off():
    """关卡/副本里点自己脚下会触发别的动作时，得能关掉这条停步。"""
    eng, ex = _mk_engine(approach_mode="mouse_click", stop_before_pick=False)
    eng.start()
    af = AFrame(x=700, y=250, w=32, h=32, score=0.8)
    pil = _pillar(cx=700, cy=400)
    for i in range(4):
        eng.on_scene(_scene([pil], aframe=af, ts_ms=1000 + i * 300))
    assert not [j for j in ex.journal
                if j["op"] == "mouse_click" and (j["x"], j["y"]) == (640, 360)], \
        "关掉后不该再发停步点击"
    assert eng.stats.get("pick_stops", 0) == 0


def test_single_frame_aframe_flicker_not_counted_as_pick():
    """铁律：A框闪现一帧就消失且从未按过A → 绝不能记为拾取成功。"""
    eng, ex = _mk_engine()
    eng.start()
    eng.on_scene(_scene([_pillar(cx=700, cy=400)], ts_ms=1000))
    eng.on_scene(_scene([_pillar(cx=700, cy=400)],
                        aframe=AFrame(x=700, y=250, w=32, h=32, score=0.8), ts_ms=1300))
    eng.on_scene(_scene([_pillar(cx=700, cy=400)], ts_ms=1600))  # 闪烁消失（未按过A）
    assert eng.stats["picks"] == 0, "未按A的A框闪现不能算拾取"
    assert eng.state == State.APPROACHING  # 继续走位而非错误进入冷却


def test_tap_recheck_blocks_rapid_second_press():
    """复核冷却：按 A 后 500ms 内不得再次按键（防连续按 A 捡垃圾）。"""
    eng, ex = _mk_engine()
    eng.start()
    af = AFrame(x=700, y=250, w=32, h=32, score=0.8)
    # tick_hz=10 → 100ms/tick；tap 发生在 0.3s，复核冷却到 0.8s
    ts = 1000
    for i in range(5):
        eng.on_scene(_scene([_pillar(cx=700, cy=400)], aframe=af, ts_ms=ts))
        ts += 100
    a_cmds = [j for j in ex.journal if j["op"] == "key_tap" and j["key"] == "a"]
    assert len(a_cmds) == 1, f"复核冷却期内只允许 1 次按键，实际 {len(a_cmds)}"
    # A框消失（拾取成功）
    eng.on_scene(_scene([], ts_ms=ts + 100))
    assert eng.stats["picks"] == 1


# ═══════════ 规则要吃的事实：真的在 probe 里 ═══════════

def test_probe_reports_repeat_and_recoveries():
    """`repeat` / `repeat_key` / `recoveries` 必须真的产出。

    这三个字段以前 probe 里根本没有，于是引用它们的规则会求值成 unknown、
    静默不生效 —— 而页面/手册上看起来"已经学会了"。
    """
    import time

    eng, ex = _mk_engine()
    eng.start()
    sc = _scene([_pillar()], ts_ms=1000)
    eng.on_scene(sc)

    ex.journal.clear()          # 先把 FSM 自己刚发的命令清掉，只看注入的这几条
    now = time.time()
    for _ in range(9):
        ex.journal.append({"ts": now - 1, "op": "mouse_click", "x": 640, "y": 360,
                           "button": "left"})
    probe = eng._arb_probe(sc)
    assert probe["repeat"] == 9
    assert probe["repeat_key"] == "mouse_click(640,360)"
    assert probe["recoveries"] == 0


def test_probe_ignores_stale_and_suppressed_actions():
    """10 秒前的那次点击不算"还在重复"；被模式闸门拦下的也不算。"""
    import time

    eng, ex = _mk_engine()
    eng.start()
    sc = _scene([_pillar()], ts_ms=1000)
    eng.on_scene(sc)

    ex.journal.clear()
    now = time.time()
    ex.journal.extend([
        {"ts": now - 30, "op": "mouse_click", "x": 640, "y": 360},
        {"ts": now - 1, "op": "key_tap", "key": "a", "suppressed": True},
        {"ts": now - 1, "op": "key_tap", "key": "d"},
    ])
    probe = eng._arb_probe(sc)
    assert probe["repeat"] == 1 and probe["repeat_key"] == "key_tap(d)"


def test_probe_counts_recoveries_inside_the_last_minute():
    """累计恢复次数不能当条件用（跑一小时之后它必然大于任何阈值）。"""
    from autogame.decision.recovery import StallSignals

    eng, _ = _mk_engine()
    eng.start()
    sc = _scene([_pillar()], ts_ms=1000)
    eng.on_scene(sc)

    eng._recovery_ts.extend([100.0, 110.0, 120.0])
    eng._recovery_ts.append(eng._now - 200.0)      # 三分钟前那次不算
    probe = eng._arb_probe(sc)
    assert probe["recoveries"] == 3
    # 真进一次恢复也要记上（走的是 _enter_recovery 这条唯一入口）
    eng._enter_recovery(sc, eng._now, StallSignals(verdict="stalled", stall_for=6.0))
    assert eng._arb_probe(sc)["recoveries"] == 4


def test_window_facts_are_absent_when_the_supervisor_is_off():
    """没在测的时候不能填 0：`picks_delta == 0` 会因此命中，规则就治错病了。"""
    from autogame.config import SupervisorConfig

    eng, _ = _mk_engine(supervisor=SupervisorConfig(mode="off"))
    eng.start()
    sc = _scene([_pillar()], ts_ms=1000)
    eng.on_scene(sc)
    probe = eng._arb_probe(sc)
    for k in ("picks_delta", "no_progress_for", "intent_churn", "cmds"):
        assert k not in probe, k


def test_rule_written_from_experience_matches_a_real_probe():
    """经验生成的规则 + 真 probe → 必须命中。

    这条是整条链的接缝：规则里的条件字段、probe 里的字段名、求值口径，
    三处任何一处对不上，写出来的规则都是"能过校验、永不命中"。
    """
    import time

    from autogame import evolve as ev
    from autogame import experience as xp

    eng, ex = _mk_engine()
    eng.start()
    sc = _scene([_pillar()], ts_ms=1000)
    eng.on_scene(sc)

    ex.journal.clear()
    now = time.time()
    for _ in range(14):
        ex.journal.append({"ts": now - 1, "op": "mouse_click", "x": 640, "y": 360,
                           "button": "left"})
    probe = eng._arb_probe(sc)

    rule, blocked = xp.rule_for("repeat", {"repeat": 14, "key": "mouse_click(640,360)"})
    assert rule and not blocked
    rules, errors = ev.validate_rules({"rules": [rule]})
    assert not errors, errors
    hit = ev.evaluate(rules, ev.facts_from_probe(probe))
    assert hit["hit"] is True, f"规则没命中；事实={ev.facts_from_probe(probe)}"
    assert hit["action"] == "explore"


# ═══════════ 帧时效：过期画面不许动手 ═══════════

def test_stale_frame_is_not_decided():
    """帧年龄超限 → 这一帧什么都不做（连目标都丢掉，别拿旧目标继续走）。"""
    eng, ex = _mk_engine(max_frame_age_ms=1000)
    eng.start()
    sc = _scene([_pillar(cx=700, cy=400)], ts_ms=1000)
    sc.age_ms = 5000
    eng.on_scene(sc)
    assert not ex.journal, "过期帧不得产生任何命令"
    assert eng.stats["stale_frames"] == 1
    assert eng._target is None


def test_fresh_frame_is_decided_normally():
    """对照：同样的帧，年龄正常就必须照常决策（闸门不能把正常帧也拦了）。"""
    eng, ex = _mk_engine(max_frame_age_ms=1000)
    eng.start()
    sc = _scene([_pillar(cx=700, cy=400)], ts_ms=1000)
    sc.age_ms = 120
    eng.on_scene(sc)
    assert ex.journal, "新鲜帧应该正常走位"
    assert eng.stats.get("stale_frames", 0) == 0


def test_same_frame_is_not_decided_twice():
    """同一帧重复投递不能重复决策 —— 否则"连续确认帧"会被同一帧刷满。"""
    eng, ex = _mk_engine()
    eng.start()
    sc = _scene([_pillar(cx=700, cy=400)], ts_ms=1000)
    eng.on_scene(sc)
    n = len(ex.journal)
    eng.on_scene(sc)          # 同一帧再来一次
    eng.on_scene(sc)
    assert len(ex.journal) == n, "同一帧重复决策产生了多余命令"
    assert eng.stats["duplicate_frames"] == 2


def test_press_requires_fresh_frame():
    """按拾取键比走位更挑时效：画面旧了就等下一帧，绝不拿过期 A 框按键。"""
    eng, ex = _mk_engine(press_max_frame_age_ms=400)
    eng.start()
    af = AFrame(x=700, y=250, w=32, h=32, score=0.8)
    pil = _pillar(cx=700, cy=400)
    sc = _scene([pil], aframe=af, ts_ms=1000)
    sc.age_ms = 900      # 0.9 秒前的画面
    for i in range(4):
        sc.ts_ms = 1000 + i * 300
        sc.age_ms = 900
        eng.on_scene(sc)
    a_cmds = [j for j in ex.journal if j["op"] == "key_tap" and j["key"] == "a"]
    assert not a_cmds, "过期画面里看见的 A 框不能按"
    assert eng.stats["stale_press_skips"] >= 1


def test_protocol_frame_age_unwraps_32bit_ts():
    """代理发的 ts 是 32 位回卷的墙钟：直接相减会得到"56 年"，必须先补高位。"""
    import time as _t

    from autogame.protocol import frame_age_ms

    now = _t.time() * 1000
    packed = int(now) & 0xFFFFFFFF
    age = frame_age_ms(packed, now)
    assert 0 <= age < 50, f"刚拍的帧算出来 {age}ms，说明高位没补对"
    # 40 秒前的帧：回卷后依然能算对
    old = int(now - 40_000) & 0xFFFFFFFF
    assert 39_000 < frame_age_ms(old, now) < 41_000


# ---------------- 交互键 D：和右移同键，绝不许当移动键 ----------------

def test_interact_key_is_never_emitted_as_a_movement_key():
    """`interact_key=d` 与 WASD 右移同键 —— 当移动键发出去就会一路开面板/走传送门。

    这比"把拾取键当左移"更严重：按 A 顶多捡空气，按 D 会打开关卡面板、
    选传送门、进 Boss 房 —— 人物已经不在原来那张图了，而日志里只是一次"向右走"。
    """
    eng, ex = _mk_engine()
    eng.start()
    sent = eng._emit_move_keys(["d"], hold_ms=80)
    assert sent == 0, "交互键不许作为移动键发出"
    assert not [j for j in ex.journal if j.get("op") == "key_tap" and j.get("key") == "d"]
    assert eng.stats["blind_tap_violations"] == 1, "被拦下必须记账（监控靠它判健康）"


def test_interact_key_is_not_mashable():
    """连按 D = 连开面板/连选传送门，比连按拾取键危险，必须从连按白名单里排掉。"""
    from autogame.config import ExecutorConfig

    eng, ex = _mk_engine()
    assert eng.cfg.interact_key in ex.burst_forbid
    assert eng.cfg.pickup_key in ex.burst_forbid
    names = [k.lower() for k in ExecutorConfig().mash.keys]
    assert eng.cfg.interact_key.lower() not in names, \
        f"交互键出现在连按白名单里：{names}"


def test_recovery_never_presses_functional_keys():
    """恢复阶梯的禁用键必须含拾取键与交互键。

    脱困时按 D 可能开出关卡面板 —— 那时人还卡在原地，画面已经换了地图，
    事后完全无法从日志复原发生了什么。
    """
    eng, _ex = _mk_engine()
    forbidden = {k.lower() for k in eng.recovery.forbidden}
    assert eng.cfg.pickup_key.lower() in forbidden
    assert eng.cfg.interact_key.lower() in forbidden


def test_keys_mode_moves_right_by_clicking_instead_of_pressing_d():
    """键盘模式下"往右走"必须退化成点地面，而不是按 D（D 会开面板/进传送门）。

    这条测的是"退化之后还能走"：只拦不补会让探索少掉 1/4 方向，
    表现为"人物老是往左/上走，右边的东西永远不捡"——比报错难查得多。
    """
    eng, ex = _mk_engine()          # approach_mode="keys"
    eng.start()
    # cy=300 → 目标的 y 与人物基本齐平，方向只有"向右"这一个键（否则会出对角 w/s）
    scene = _scene([_pillar(cx=980, cy=300)], ts_ms=1000)
    eng.on_scene(scene)
    # 直接走一次"往右"的走位分支。不靠帧循环：静止帧会先触发撞墙侧移，
    # 那时发出的方向键是 w/s，测不到"向右"这条。
    eng._move_toward(scene, scene.pillars[0], 2.0, True)
    taps_d = [j for j in ex.journal if j.get("op") == "key_tap" and j.get("key") == "d"]
    assert not taps_d, "向右走不许按 D"
    assert eng.stats["blind_tap_violations"] == 0, "这是主动退化，不是被拦"
    clicks = [j for j in ex.journal if j.get("op") == "mouse_click"]
    assert clicks, "退化后必须点地面（否则右边永远走不到）"


def test_pressable_filters_all_functional_keys():
    eng, _ex = _mk_engine()
    assert eng._pressable(["d"]) == []
    assert eng._pressable(["a"]) == []
    assert eng._pressable(["d", "w"]) == ["w"]
    assert eng._pressable(["w", "a", "s", "d"]) == ["w", "s"]
