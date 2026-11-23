"""小地图导航测试 —— 标记检测 + 方向键映射 + 导航闭环。"""

import time

import cv2

from autogame.config import DecisionConfig, ExecutorConfig, VisionConfig
from autogame.decision.fsm import DecisionEngine, State
from autogame.executor import DryRunExecutor
from autogame.sim.scene import SimWorld
from autogame.vision.detector import Detector


def process2(det, frame, ts=1000):
    """喂同一帧两次（时域确认）。"""
    det.process(frame, ts)
    return det.process(frame, ts + 1000)


def _nav_cfg(**overrides):
    dcfg = DecisionConfig(tick_hz=10.0, approach_mode="keys", nav_enabled=True)
    dcfg.autonomy.enabled = False  # 本文件只验证导航数学，自主探索会干扰断言
    for k, v in overrides.items():
        setattr(dcfg, k, v)
    ecfg = ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000)
    return DecisionEngine(dcfg, DryRunExecutor(ecfg))


def test_keys_for_angle_mapping():
    eng = _nav_cfg()
    assert eng.keys_for_angle(0) == ["d"]        # 右
    assert eng.keys_for_angle(45) == ["s", "d"]  # 右下
    assert eng.keys_for_angle(90) == ["s"]       # 下
    assert eng.keys_for_angle(-90) == ["w"]      # 上
    assert eng.keys_for_angle(180) == ["a"]      # 左
    assert eng.keys_for_angle(-180) == ["a"]     # 左
    assert eng.keys_for_angle(-45) == ["w", "d"] # 右上
    assert eng.keys_for_angle(135) == ["s", "a"] # 左下


def test_minimap_marker_detected():
    world = SimWorld(seed=77)
    frame = world.render()
    cfg = VisionConfig()
    cfg.minimap.enabled = True
    det = Detector(cfg)
    state = process2(det, frame)
    assert state.minimap is not None
    mm = state.minimap
    assert mm.marker is not None, "小地图内应有青色目标标记"
    assert mm.angle_deg is not None and mm.distance_ratio is not None
    assert 0 < mm.distance_ratio < 1.0, "标记应在小地图范围内且不在中心"


def test_nav_closed_loop_reaches_objective():
    """导航闭环：FSM 朝小地图标记走位 → 角色到达目标点。

    禁用刷怪（光柱优先于导航是正确设计；本测试只验证导航链路）。
    """
    world = SimWorld(seed=77, pillar_spawn_interval_s=999)
    world.pillars.clear()
    objective = world.objective_world
    eng = _nav_cfg()

    # 直接用世界驱动（不经过服务端，验证导航数学与 FSM 逻辑）
    frame = world.render()
    cfg = VisionConfig()
    cfg.minimap.enabled = True
    det = Detector(cfg)
    ts = 1000
    eng.start()
    for _ in range(220):
        state = det.process(world.render(), ts)
        det.process(world.render(), ts + 50)  # 时域确认第二帧
        state = det.process(world.render(), ts + 100)
        eng.on_scene(state)
        # 注入导航按键到世界
        for j in eng.exec.journal:
            # 回放**所有**移动命令，不只 key_tap：`d` 是交互键、不许当移动键发，
            # 于是"往右走"会退化成点地面。只回放按键的话，退化后的点击被丢掉，
            # 表现为"导航永远到不了"（其实已经走过去了，只是测试台没接）。
            if j["op"] == "key_tap" and j["key"] in ("w", "a", "s", "d"):
                world.on_command("key_tap", key=j["key"], hold_ms=j["hold_ms"])
            elif j["op"] == "mouse_click" and j.get("x") is not None:
                world.on_command("mouse_click", x=j["x"], y=j["y"], button="left")
        eng.exec.journal.clear()
        world.tick()
        ts += 200
        if world.objectives_reached >= 1:
            break
    assert world.objectives_reached >= 1, (
        f"导航必须到达目标点，位置=({world.char_wx:.0f},{world.char_wy:.0f}) "
        f"目标={objective} events={eng.events[-5:]}"
    )


def test_nav_disabled_when_no_marker():
    """标记不在小地图视野内 → 不发导航按键。"""
    world = SimWorld(seed=77)
    world.objective_world = (world.char_wx + 5000, world.char_wy)  # 超出小地图
    cfg = VisionConfig()
    cfg.minimap.enabled = True
    det = Detector(cfg)
    eng = _nav_cfg()
    eng.start()
    eng.on_scene(process2(det, world.render()))
    assert len(eng.exec.journal) == 0, "无标记不得发导航按键"


def test_nav_arrived_marker_no_move():
    """标记已在到达圈内 → 不再移动。"""
    world = SimWorld(seed=77)
    # 把目标点放到角色当前位置附近 → 标记在小地图中心 → arrived
    world.objective_world = (world.char_wx + 30, world.char_wy)
    cfg = VisionConfig()
    cfg.minimap.enabled = True
    det = Detector(cfg)
    eng = _nav_cfg()
    eng.start()
    eng.on_scene(process2(det, world.render()))
    moves = [j for j in eng.exec.journal if j["op"] == "key_tap"]
    assert len(moves) == 0, "已到达不得继续移动"


def test_nav_idle_only():
    """有拾取目标时导航必须让位（拾取优先）。"""
    from autogame.protocol import Pillar

    world = SimWorld(seed=77)
    cfg = VisionConfig()
    cfg.minimap.enabled = True
    det = Detector(cfg)
    eng = _nav_cfg()
    eng.start()
    scene = process2(det, world.render())
    scene.pillars.append(
        Pillar(x=700, y=455, cx=700, cy=400, w=14, h=110, color="gold",
               area=1500, distance=100, whitelisted=True)
    )
    eng.on_scene(scene)
    assert eng.state == State.APPROACHING, "有光柱必须走拾取流程"
    assert not any(e["event"] == "nav_move" for e in eng.events), \
        "拾取优先，不得同时导航"


def test_nav_never_presses_wasd_in_click_mode():
    """点地寻路的游戏里，导航必须**只点地面**，一个方向键都不许发。

    目标游戏是点地移动，WASD 不是移动键：`w`/`s` 轻则无反应，重则撞上别的热键；
    `a`/`d` 更是拾取键/交互键（按 D 会开面板、走传送门）。所以导航在
    `approach_mode=mouse_click` 时一律点击，不经过键盘路径。
    """
    world = SimWorld(seed=77, pillar_spawn_interval_s=999)
    world.pillars.clear()
    eng = _nav_cfg(approach_mode="mouse_click")
    cfg = VisionConfig()
    cfg.minimap.enabled = True
    det = Detector(cfg)
    ts = 1000
    eng.start()
    for _ in range(40):
        state = det.process(world.render(), ts)
        det.process(world.render(), ts + 50)
        state = det.process(world.render(), ts + 100)
        eng.on_scene(state)
        ts += 200
    keys = [j for j in eng.exec.journal
            if j["op"] == "key_tap" and j.get("key") in ("w", "a", "s", "d")]
    assert not keys, f"点地模式下导航不许按方向键，实际发了 {keys}"
    clicks = [j for j in eng.exec.journal if j["op"] == "mouse_click"]
    assert clicks, "导航必须点地面（否则不会移动）"
