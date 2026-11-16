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
    dcfg = DecisionConfig(tick_hz=10.0, approach_mode="keys")
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
    """连续帧差极低 → 撞墙 → 侧向脱困 → 仍卡则放弃。"""
    eng, ex = _mk_engine(stuck_frames=2, stuck_delta_thresh=6.0)
    eng.start()
    ts = 1000
    eng.on_scene(_scene([_pillar(cx=900, cy=400)], delta=100, ts_ms=ts)); ts += 100
    assert eng.state == State.APPROACHING
    eng.on_scene(_scene([_pillar(cx=900, cy=400)], delta=1.0, ts_ms=ts)); ts += 100
    eng.on_scene(_scene([_pillar(cx=900, cy=400)], delta=1.0, ts_ms=ts)); ts += 100
    # 触发侧向脱困
    assert eng._alt_direction_tried is True
    eng.on_scene(_scene([_pillar(cx=900, cy=400)], delta=1.0, ts_ms=ts)); ts += 100
    eng.on_scene(_scene([_pillar(cx=900, cy=400)], delta=1.0, ts_ms=ts)); ts += 100
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
    # 第1帧启动走位（不处理A框）；第2帧确认A框；第3帧按键；第4帧A框消失→拾取成功
    eng.on_scene(_scene([_pillar(cx=700, cy=400)], aframe=af, ts_ms=1000))
    eng.on_scene(_scene([_pillar(cx=700, cy=400)], aframe=af, ts_ms=1300))
    eng.on_scene(_scene([_pillar(cx=700, cy=400)], aframe=af, ts_ms=1600))
    eng.on_scene(_scene([], ts_ms=1900))  # A框消失 → 成功
    assert eng.stats["picks"] == 1
    assert eng.state == State.COOLDOWN
    eng.on_scene(_scene([], ts_ms=2000))
    assert eng.state == State.COOLDOWN  # 冷却未结束
    eng.on_scene(_scene([], ts_ms=1000 + 5000))  # 5s 后冷却必已结束
    assert eng.state == State.SCANNING


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
