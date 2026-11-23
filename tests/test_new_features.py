"""对话需求补齐项测试 —— locate API / 列表点击 / PCA9685 / 加载保持。"""

import json

import cv2
import numpy as np
import pytest

from autogame.config import DecisionConfig, ExecutorConfig, VisionConfig
from autogame.decision.fsm import DecisionEngine, State
from autogame.executor import DryRunExecutor
from autogame.executor.pca9685 import _MockBus
from autogame.executor import Pca9685Executor
from autogame.protocol import Command, ListEntry, SceneState


# ---------------- PCA9685 机械执行器 ----------------

def _mk_pca():
    cfg = ExecutorConfig(backend="pca9685", pca9685_mock=True,
                         min_action_interval_ms=0, max_actions_per_sec=1000)
    return Pca9685Executor(cfg)


def test_pca9685_key_tap_pulse_sequence():
    ex = _mk_pca()
    ex.execute(Command(op="key_tap", key="a", hold_ms=80))
    assert len(ex._bus.pulses) == 2, "按下+抬起两条脉冲"
    ch, down = ex._bus.pulses[0]
    assert ch == 0  # servo_map a→ch0
    angle = (down - 500) / 2000 * 180
    assert 105 <= angle <= 115, f"按下角度应≈110°，实际 {angle:.0f}°"


def test_pca9685_stop_all_releases():
    ex = _mk_pca()
    ex.execute(Command(op="key_down", key="w"))
    assert "w" in ex._pressed
    ex.execute(Command(op="stop_all"))
    assert len(ex._pressed) == 0


def test_pca9685_mouse_move_rejected():
    ex = _mk_pca()
    ok = ex.execute(Command(op="mouse_move", x=100, y=200))
    assert ok is False, "机械后端不支持鼠标移动，必须安全拒绝"
    assert ex.dropped >= 0


def test_pca9685_unknown_key_rejected():
    ex = _mk_pca()
    ok = ex.execute(Command(op="key_tap", key="f13", hold_ms=50))
    assert ok is False, "servo_map 缺少的键必须拒绝"


# ---------------- 右侧 UI 列表掉落 ----------------

def _scene_with_list(ts=1000):
    return SceneState(
        ts_ms=ts, w=1280, h=720, character=(640, 360),
        pillars=[], item_list=[ListEntry(x=1050, y=200, w=200, h=30)],
        frame_delta=50.0,
    )


def test_list_entry_click_flow():
    eng_cfg = DecisionConfig(tick_hz=10.0)
    eng_cfg.autonomy.enabled = False  # 只验证列表点击：自主探索会额外点一次地面
    ecfg = ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000)
    ex = DryRunExecutor(ecfg)
    eng = DecisionEngine(eng_cfg, ex)
    eng.start()
    eng.on_scene(_scene_with_list(1000))   # 首见
    eng.on_scene(_scene_with_list(1400))   # ≥300ms → 点击
    clicks = [j for j in ex.journal if j["op"] == "mouse_click"]
    assert len(clicks) == 1, "列表条目稳定可见后点击一次"
    assert clicks[0]["x"] == 1050 and clicks[0]["y"] == 200
    assert eng.state == State.COOLDOWN


def test_list_entry_flicker_no_click():
    eng_cfg = DecisionConfig(tick_hz=10.0)
    eng_cfg.autonomy.enabled = False  # 只验证防抖：自主探索会产生无关走位命令
    ex = DryRunExecutor(ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000))
    eng = DecisionEngine(eng_cfg, ex)
    eng.start()
    eng.on_scene(_scene_with_list(1000))
    eng.on_scene(_scene_with_list(1200, ))  # 200ms < 300ms 防抖
    eng.on_scene(SceneState(ts_ms=1400, w=1280, h=720, character=(640, 360),
                            item_list=[], frame_delta=50.0))
    assert len(ex.journal) == 0, "防抖期内条目消失不得点击"


def test_item_list_detection_on_synthetic():
    """合成右侧列表：亮色条行被检出且坐标正确。"""
    frame = np.full((720, 1280, 3), (28, 24, 22), dtype=np.uint8)
    for y in (150, 250, 350):
        frame[y:y + 26, 950:1230] = (120, 170, 255)  # 亮色条行
    cfg = VisionConfig()
    cfg.item_list.enabled = True
    from autogame.vision.detector import Detector, ROOT_TEMPLATES
    det = Detector(cfg, ROOT_TEMPLATES)
    st = det.process(frame, 1000)
    assert len(st.item_list) >= 2, f"应检出≥2条列表条目，实际 {st.item_list}"
    for e in st.item_list:
        assert 900 <= e.x <= 1280, "条目应在右侧 ROI 内"


# ---------------- 加载/黑屏过场保持 ----------------

def test_loading_screen_holds_fsm():
    eng_cfg = DecisionConfig(tick_hz=10.0)
    eng_cfg.autonomy.enabled = False  # 只验证防抖：自主探索会产生无关走位命令
    ex = DryRunExecutor(ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000))
    eng = DecisionEngine(eng_cfg, ex)
    eng.start()
    # 黑屏帧
    loading = SceneState(ts_ms=1000, w=1280, h=720, character=(640, 360),
                         loading=True, frame_delta=0.0)
    eng.on_scene(loading)
    assert any(e["event"] == "loading_hold" for e in eng.events)
    # 加载中不允许动作
    eng.on_scene(SceneState(ts_ms=2000, w=1280, h=720, character=(640, 360),
                            loading=True, pillars=[], frame_delta=0.0))
    assert len(ex.journal) == 0, "加载画面不得产生命令"


def test_loading_flag_from_dark_frame():
    from autogame.vision.detector import Detector, ROOT_TEMPLATES
    cfg = VisionConfig()
    det = Detector(cfg, ROOT_TEMPLATES)
    black = np.zeros((720, 1280, 3), dtype=np.uint8)
    st = det.process(black, 1000)
    assert st.loading is True, "全黑帧必须判定为加载/过场"
    bright = np.full((720, 1280, 3), (28, 24, 22), dtype=np.uint8)
    st2 = det.process(bright, 1001)
    assert st2.loading is False


def test_pca9685_right_click_uses_the_right_servo_channel():
    """技能槽第 5 格 = 鼠标右键 → 舵机必须打 `mouse_right`（ch9），不是左键 ch8。

    三套执行器里，右键这条路以前一次都没被发过（左键是唯一调用方）。
    舵机后端按 `mouse_{button}` 查通道，通道名写错就是 KeyError —— 真机上表现为
    "一放技能程序就崩"，而在测试里完全看不出来。
    """
    ex = _mk_pca()
    ex.execute(Command(op="mouse_click", button="right", x=100, y=100))
    chans = [ch for ch, _ in ex._bus.pulses]
    assert chans == [9, 9], f"右键应打 ch9（按下+抬起），实际 {chans}"
