"""视觉检测器测试 —— 用合成帧验证光柱/A框检测精度。

注意：光柱默认需要时域确认（连续 2 帧），统一用 process2 辅助函数喂两帧。
"""

import numpy as np

from autogame.config import VisionConfig
from autogame.sim.scene import SimWorld
from autogame.vision.detector import Detector


def process2(det: Detector, frame, ts=1000):  # noqa: ANN001
    """喂同一帧两次（时域确认）。"""
    det.process(frame, ts)
    return det.process(frame, ts + 1000)


def _empty_frame(w=1280, h=720):
    frame = np.full((h, w, 3), (28, 24, 22), dtype=np.uint8)
    return frame


def _draw_gold_pillar(frame, sx, sy, w_pillar=14, h_pillar=110):
    # 金柱 + 白热芯（真实光柱特征）
    color = (36, 190, 255)  # BGR → HSV hue≈21, S≈230, V≈255
    frame[sy - h_pillar : sy, sx - w_pillar // 2 : sx + w_pillar // 2] = color
    frame[sy - h_pillar + 6 : sy - 4, sx - 2 : sx + 2] = (255, 255, 255)


def test_empty_scene_no_detection():
    cfg = VisionConfig()
    det = Detector(cfg)
    state = det.process(_empty_frame(), 1000)
    assert len(state.pillars) == 0
    assert state.aframe is None
    assert state.character == (640, 360)


def test_gold_pillar_detected_with_position():
    cfg = VisionConfig()
    det = Detector(cfg)
    frame = _empty_frame()
    sx, sy = 900, 500  # 光柱底部地面点
    _draw_gold_pillar(frame, sx, sy)
    state = process2(det, frame)
    assert len(state.pillars) == 1
    p = state.pillars[0]
    assert p.color == "gold"
    # 底部中心应接近绘制点（容差：合并光斑后的 bbox 误差）
    assert abs(p.x - sx) <= 16
    assert abs(p.y - sy) <= 20
    assert p.distance > 0


def test_pillar_out_of_range_not_false_positive():
    cfg = VisionConfig()
    det = Detector(cfg)
    frame = _empty_frame()
    # 方形金块（宽高比1:1）不应被判为光柱
    frame[300:380, 300:380] = (36, 190, 255)
    state = process2(det, frame)
    assert len(state.pillars) == 0


def test_single_frame_beam_not_confirmed():
    """时域确认：单帧出现的光柱（特效闪烁）不得触发。"""
    cfg = VisionConfig()
    det = Detector(cfg)
    frame = _empty_frame()
    _draw_gold_pillar(frame, 900, 500)
    state = det.process(frame, 1000)  # 只喂一帧
    assert len(state.pillars) == 0, "单帧光柱（未确认）不得输出"


def test_sim_frame_detection():
    """模拟器整帧：光柱 + 近距A框都能检出。"""
    world = SimWorld(seed=42)
    # 手动放置一个近距光柱（世界坐标 (60,20) → 屏幕约 (700,380)，在拾取半径内）
    world.pillars.clear()
    from autogame.sim.scene import SimPillar

    world.pillars.append(SimPillar(60, 20, "gold"))
    frame = world.render()
    cfg = VisionConfig()
    det = Detector(cfg)
    state = process2(det, frame)
    assert len(state.pillars) == 1
    assert state.aframe is not None, "拾取半径内必须检出 A框"
    # A框应在光柱附近（<220px）
    p, a = state.pillars[0], state.aframe
    dist = ((p.cx - a.x) ** 2 + (p.cy - a.y) ** 2) ** 0.5
    assert dist < 220


def test_frame_delta_changes_when_world_moves():
    world = SimWorld(seed=7)
    cfg = VisionConfig()
    det = Detector(cfg)
    f1 = det.process(world.render(), 1)
    world.char_wx += 100  # 移动后画面明显变化
    f2 = det.process(world.render(), 2)
    assert f2.frame_delta > 1.0, "世界滚动必须产生帧差（防撞墙依赖）"


def test_aframe_requires_brightness():
    """暗背景上没有任何 A框 → 不误报。"""
    cfg = VisionConfig()
    det = Detector(cfg)
    state = det.process(_empty_frame(), 1000)
    assert state.aframe is None


def test_in_game_gate_blocks_detections():
    """场景门：非战斗界面（in_game=False）时不产出可行动目标。"""
    world = SimWorld(seed=42)
    world.pillars.clear()
    from autogame.sim.scene import SimPillar

    world.pillars.append(SimPillar(60, 20, "gold"))
    frame = world.render()
    cfg = VisionConfig()
    det = Detector(cfg)
    st1 = det.process(frame, 1000)
    st2 = det.process(frame, 2000)
    # 模拟器渲染了技能栏 → 场景门应放行
    assert st2.in_game is True, "模拟器帧含技能栏 → in_game 必须 True"
    assert len(st2.pillars) == 1
    # 手动构造 UI 场景（无技能栏）→ 门禁拦截
    frame_ui = _empty_frame()
    _draw_gold_pillar(frame_ui, 900, 500)
    det2 = Detector(VisionConfig(), templates_dir=None)
    det2.process(frame_ui, 1000)
    st_ui = det2.process(frame_ui, 2000)
    assert st_ui.in_game is False, "无技能栏的 UI 画面 → in_game 必须 False"


# ---------------- 怪物检测：真实录像暴露的误报（技能图标/漂浮文字） ----------------

def _draw_blob(frame, cx, cy, r=16, color=(200, 60, 220)):
    """画一个"像怪物"的饱和色块（面积/尺寸/填充率都在怪物判据内）。"""
    import cv2 as _cv2

    _cv2.circle(frame, (cx, cy), r, color, -1)


def test_static_blob_is_not_a_monster():
    """静态图标（技能轮盘/背包键）不是怪物：只出现、不动、不变 → 永不确认。"""
    cfg = VisionConfig()
    det = Detector(cfg)
    frame = _empty_frame()
    _draw_blob(frame, 900, 400)
    for i in range(4):
        st = det.process(frame, 1000 + i * 500)
        assert st.monsters == [], f"第{i+1}帧静态色块被当成怪物（真实录像里的误报源）"


def test_moving_blob_becomes_monster():
    """会动的才是怪：位移超过容差 → 第二帧起确认。"""
    cfg = VisionConfig()
    det = Detector(cfg)
    seen = 0
    for i in range(4):
        frame = _empty_frame()
        _draw_blob(frame, 860 + i * 12, 400)  # 每帧右移 12px
        st = det.process(frame, 1000 + i * 500)
        seen += len(st.monsters)
    assert seen >= 1, "移动的色块必须被确认为怪物"


def test_blob_inside_hud_exclusion_is_masked():
    """HUD 排除区内的色块（移动端技能按钮就压在战斗区里）不得成为怪物。"""
    cfg = VisionConfig(hud_exclusions=[[820, 330, 200, 160]])
    det = Detector(cfg)
    for i in range(3):
        frame = _empty_frame()
        _draw_blob(frame, 900 + i * 12, 400)  # 在排除区内移动
        st = det.process(frame, 1000 + i * 500)
        assert st.monsters == [], "HUD 排除区内的候选必须被抹掉"


def test_blob_overlapping_ground_label_is_not_monster():
    """与掉落标签重叠的候选是文字，不是怪。"""
    from autogame.protocol import ListEntry

    cfg = VisionConfig()
    det = Detector(cfg)
    for i in range(3):
        frame = _empty_frame()
        _draw_blob(frame, 860 + i * 12, 400)
        st = det.process(frame, 1000 + i * 500)
        # 直接把标签按候选位置喂进互斥逻辑，断言不产出怪物
        got = det._detect_monsters(frame, frame.shape[1], frame.shape[0], 1.0,
                                   avoid=[ListEntry(x=860 + i * 12, y=400, w=90, h=24)])
        assert got == [], "与掉落标签重叠的色块不能当怪物点"
        _ = st
