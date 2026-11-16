"""刷图循环测试 —— 『再次挑战』按钮检测 + 完整刷图闭环。

验证对话定稿的主循环：拾取 → 地图结算 → 检测『再次挑战』→ 点击 → 下一张图。
"""

import time

import cv2
import numpy as np

from autogame.config import VisionConfig
from autogame.sim.scene import SimPillar, SimWorld
from autogame.vision.detector import Detector


def process2(det, frame, ts=1000):
    """喂同一帧两次（时域确认）。"""
    det.process(frame, ts)
    return det.process(frame, ts + 1000)


def _cleared_world():
    world = SimWorld(seed=5, picks_until_clear=2)
    world.pillars = [SimPillar(60, 20, "gold"), SimPillar(-70, 40, "pink")]
    world.char_wx, world.char_wy = 60, 30
    world.picked = 2  # 直接置为结算态
    world.map_cleared = True
    world.pillars.clear()
    return world


def test_rechallenge_button_detected():
    world = _cleared_world()
    frame = world.render()
    det = Detector(VisionConfig())
    state = process2(det, frame)
    assert state.rechallenge is not None, "结算画面必须检出『再次挑战』按钮"
    btn = state.rechallenge
    # 位置应接近模拟器按钮矩形中心
    bx, by, bw, bh = world.rechallenge_button_rect()
    assert abs(btn.x - (bx + bw / 2)) <= 30
    assert abs(btn.y - (by + bh / 2)) <= 30
    # 点击坐标在按钮矩形内
    assert bx <= btn.x <= bx + bw and by <= btn.y <= by + bh


def test_rechallenge_button_not_detected_when_playing():
    world = SimWorld(seed=5)
    world.pillars = [SimPillar(60, 20, "gold")]
    frame = world.render()
    det = Detector(VisionConfig())
    state = process2(det, frame)
    assert state.rechallenge is None, "游玩画面不得误报再次挑战按钮"


def test_rechallenge_click_restarts_map():
    """点击按钮矩形内 → 重开；点击按钮外 → 不重开。"""
    world = _cleared_world()
    bx, by, bw, bh = world.rechallenge_button_rect()
    world.on_command("mouse_click", x=bx + bw // 2, y=by + bh // 2)
    assert world.map_clears == 1 and not world.map_cleared
    # 立即刷怪
    world._advance(time.monotonic())
    assert len(world.pillars) > 0, "重开后必须立即刷光柱"

    # 点外面不重开
    world2 = _cleared_world()
    world2.on_command("mouse_click", x=100, y=100)
    assert world2.map_clears == 0 and world2.map_cleared


def _no_auto():
    """关闭自主探索：本测试只验证『再次挑战』防抖，不接受额外命令。"""
    from autogame.config import DecisionConfig
    cfg = DecisionConfig(tick_hz=10.0)
    cfg.autonomy.enabled = False
    return cfg


def test_rechallenge_fsm_clicks_button():
    """FSM：无光柱 + 按钮可见（连续2帧）→ 点击按钮中心。"""
    from autogame.config import DecisionConfig, ExecutorConfig
    from autogame.decision.fsm import DecisionEngine, State
    from autogame.executor import DryRunExecutor
    from autogame.protocol import ButtonHit, SceneState

    ecfg = ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000)
    ex = DryRunExecutor(ecfg)
    eng = DecisionEngine(_no_auto(), ex)
    eng.start()

    def scene(ts, btn=True):
        return SceneState(
            ts_ms=ts, w=1280, h=720, character=(640, 360),
            pillars=[],
            rechallenge=ButtonHit(x=640, y=570, w=240, h=52, score=0.9) if btn else None,
            frame_delta=0.0,
        )

    eng.on_scene(scene(1000))  # 首帧只见不点（防抖）
    eng.on_scene(scene(1200))  # 第二帧 → 点击
    clicks = [j for j in ex.journal if j["op"] == "mouse_click"]
    assert len(clicks) == 1, "按钮连续可见必须点击一次"
    assert clicks[0]["x"] == 640 and clicks[0]["y"] == 570
    assert eng.stats["rechallenges"] == 1
    assert eng.state == State.COOLDOWN

    # 按钮闪现一帧不得点击
    ex2 = DryRunExecutor(ecfg)
    eng2 = DecisionEngine(_no_auto(), ex2)
    eng2.start()
    eng2.on_scene(scene(1000))
    eng2.on_scene(scene(1200, btn=False))
    assert len(ex2.journal) == 0


def test_farming_loop_e2e_world():
    """世界级刷图闭环：拾满 → 结算 → 点按钮 → 下一图 → 继续拾取。"""
    world = SimWorld(seed=11, picks_until_clear=3)
    # 手动放拾取半径内的光柱，依次拾取
    world.pillars = [SimPillar(50, 10, "gold"), SimPillar(60, 20, "gold"),
                     SimPillar(70, 30, "gold"), SimPillar(80, 40, "gold"),
                     SimPillar(90, 50, "gold"), SimPillar(100, 60, "gold")]
    for i in range(3):
        assert not world.map_cleared
        n_before = world.picked
        world.on_command("key_tap", key="a")
        assert world.picked == n_before + 1
    assert world.map_cleared and world.picked == 3
    # 结算后拾取半径内无光柱（已清空）→ 盲按会违规，FSM 不应触发
    # 点击按钮重开
    bx, by, bw, bh = world.rechallenge_button_rect()
    world.on_command("mouse_click", x=bx + bw // 2, y=by + bh // 2)
    assert world.map_clears == 1
    world._advance(time.monotonic())
    assert len(world.pillars) > 0, "重开后自动刷怪（远距 300-700px）"
    # 放一个拾取半径内的光柱再拾取（自动刷的怪在远处，A框未出现不能按 A）
    world.pillars.append(SimPillar(world.char_wx + 50, world.char_wy + 10, "gold"))
    world.on_command("key_tap", key="a")
    assert world.picked == 4, "下一张图继续拾取"
    assert world.blind_a_press == 0
