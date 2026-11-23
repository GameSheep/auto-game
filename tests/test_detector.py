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


def test_pillar_detection_can_be_disabled():
    """掉落不走光柱的游戏（火炬之光 PC）关掉光柱检测：零误报且省 CPU。

    真实录像实测：334 秒里光柱帧 1 个、全是散点误报，按 A 次数 0 ——
    拾取提示实际是地面文字标签（decision.label_pickup_when_near）。
    """
    cfg = VisionConfig(pillar_enabled=False)
    det = Detector(cfg)
    frame = _empty_frame()
    _draw_gold_pillar(frame, 900, 500)
    state = process2(det, frame)
    assert state.pillars == [], "光柱检测关闭后不得产出光柱"


def test_aframe_detection_can_be_disabled():
    """没有暗底 A 提示框的游戏关掉 A框检测：避免伤害数字/技能图标误报。"""
    world = SimWorld(seed=42)
    world.pillars.clear()
    from autogame.sim.scene import SimPillar

    world.pillars.append(SimPillar(60, 20, "gold"))
    frame = world.render()

    on = Detector(VisionConfig())
    assert process2(on, frame).aframe is not None, "默认开：模拟器帧里应检出 A框"

    cfg = VisionConfig()
    cfg.aframe.enabled = False
    off = Detector(cfg)
    assert process2(off, frame).aframe is None, "关闭后不得产出 A框"


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

def _draw_blob(frame, cx, cy, r=16, color=(200, 60, 220), hpbar=True):
    """画一个"像怪物"的饱和色块（面积/尺寸/填充率都在怪物判据内）。

    `hpbar=True` 时在头顶画一条红色细横条 —— 真实游戏里敌人都有血条，
    而 `monster_hpbar`（默认开）要求候选框正上方有横条才认。
    """
    import cv2 as _cv2

    _cv2.circle(frame, (cx, cy), r, color, -1)
    if hpbar:
        bar_w = max(24, int(r * 2.4))
        _cv2.rectangle(frame, (cx - bar_w // 2, cy - r - 20),
                       (cx + bar_w // 2, cy - r - 16), (0, 0, 220), -1)


def test_player_nameplate_bar_is_not_monster_evidence():
    """玩家自己头顶的铭牌血条不能当"怪物证据"。

    真实录像里玩家铭牌（绿 HP + 蓝 MP）固定在角色头顶、位置和填充比例
    5 分钟不变 —— 它是全画面最稳定的条，把角色旁边的岩石一块块"验证"成了
    怪物（8 个检出里 4 个是它干的）。角色附近的条一律不算证据。
    """
    import cv2 as _cv2

    cfg = VisionConfig()
    det = Detector(cfg)
    frame = _empty_frame()
    r = 16
    cx, cy = 700, 330           # 离角色中心 (640,360) 不远
    _cv2.circle(frame, (cx, cy), r, (200, 60, 220), -1)
    bar_w = 60
    bar_y = cy - r - 30         # 角色头顶的铭牌位置
    _cv2.rectangle(frame, (660 - bar_w // 2, bar_y),
                   (660 + bar_w // 2, bar_y + 5), (0, 0, 220), -1)
    seen = 0
    for i in range(4):
        seen += len(det.process(frame, 1000 + i * 500).monsters)
    assert seen == 0, "玩家铭牌条被当成怪物证据（真实录像里的误报源）"


def test_bar_away_from_character_still_confirms_monster():
    """远离角色的条照常认怪（别把闸门做得连真怪都挡了）。"""
    import cv2 as _cv2

    cfg = VisionConfig()
    det = Detector(cfg)
    frame = _empty_frame()
    r = 16
    cx, cy = 1040, 330
    _cv2.circle(frame, (cx, cy), r, (200, 60, 220), -1)
    bar_w = 60
    bar_y = cy - r - 30
    _cv2.rectangle(frame, (cx - bar_w // 2, bar_y),
                   (cx + bar_w // 2, bar_y + 5), (0, 0, 220), -1)
    seen = 0
    for i in range(4):
        seen += len(det.process(frame, 1000 + i * 500).monsters)
    assert seen >= 1, "远离角色的血条怪被误挡"


def test_character_itself_is_not_a_monster():
    """玩家精灵自己不是怪（同样高饱和、尺寸也在怪物判据里）。"""
    cfg = VisionConfig()
    det = Detector(cfg)
    frame = _empty_frame()
    _draw_blob(frame, 640, 360, r=18)
    seen = 0
    for i in range(4):
        seen += len(det.process(frame, 1000 + i * 500).monsters)
    assert seen == 0, "角色自身色块被当成怪物"


def test_static_blob_without_hpbar_is_not_a_monster():
    """静止 + 没有血条 = 不是怪物（技能图标/背包键/地形装饰）。

    旧版这里守的是"静止就永不确认"，因为当时唯一的判据是"动没动"。
    加了血条闸门之后，判据变成"头顶有没有条"：**站着不动放技能的怪**框
    根本不位移，只认运动会把它们全漏掉（真实录像里 6 个可见怪只认出 2 个）。
    所以不变量改为"没有血条的证据才不算怪" —— 这条比"静止"更接近物理事实。
    """
    cfg = VisionConfig()
    det = Detector(cfg)
    frame = _empty_frame()
    _draw_blob(frame, 900, 400, hpbar=False)
    for i in range(4):
        st = det.process(frame, 1000 + i * 500)
        assert st.monsters == [], f"第{i+1}帧无血条的静态色块被当成怪物"


def test_static_blob_with_hpbar_is_a_monster():
    """静止 + 头顶有血条 = 怪物（站着放技能的怪不能漏）。

    真实录像（火炬之光 2560×1380）实测：血条在怪身上方 1.0~1.5 倍身高处、
    比怪身宽约 2 倍，且只有一小段是彩色（剩余血量），其余是灰色空槽。
    这三条决定了 HPBAR_GAP / HPBAR_W_RATIO / 横向闭运算的取值。
    """
    cfg = VisionConfig()
    det = Detector(cfg)
    frame = _empty_frame()
    _draw_blob(frame, 900, 400)  # hpbar=True
    seen = 0
    for i in range(4):
        st = det.process(frame, 1000 + i * 500)
        seen += len(st.monsters)
    assert seen >= 1, "静止但头顶有血条的色块必须被确认为怪物"


def test_hpbar_far_above_and_wider_than_blob_still_matches():
    """血条远在 1.2 倍身高上方、且比怪身宽 2 倍时仍要认（TLI 实测形态）。

    旧判据（gap ≤0.85h、条宽 ≤1.6 倍怪身宽）在这款游戏上一条都不满足，
    真怪被整批拒掉 —— 这是"6 个可见怪只认出 2 个"的直接原因。
    """
    import cv2 as _cv2

    cfg = VisionConfig()
    det = Detector(cfg)
    frame = _empty_frame()
    r = 16
    cx, cy = 900, 400
    _cv2.circle(frame, (cx, cy), r, (200, 60, 220), -1)
    # 条宽 = 2 倍怪身宽，条中心距框顶 1.2 倍身高
    bar_w = int(r * 2 * 2.0)
    bar_y = cy - r - int(r * 2 * 1.2)
    _cv2.rectangle(frame, (cx - bar_w // 2, bar_y),
                   (cx + bar_w // 2, bar_y + 5), (0, 0, 220), -1)
    seen = 0
    for i in range(4):
        seen += len(det.process(frame, 1000 + i * 500).monsters)
    assert seen >= 1, "远距宽血条被拒（TLI 真怪召回回归）"


def test_bar_without_dark_frame_is_not_monster_evidence():
    """红色横条周围没有黑框 → 不是 UI 血条，不能当怪物证据。

    真实录像（2560 宽）实测：真血条上下各有 1~2px 纯黑描边（外圈/填充
    ≈ 0.24），而地形红色亮边、技能特效的柔和色带外圈≈填充（0.65~0.97）。
    这一条把"看着像条的红斑"整类挡掉 —— 它是唯一不依赖"动没动"、
    也不依赖几何比例的物理判据（运动判据在 2Hz 采样下实测无效）。
    """
    import cv2 as _cv2

    cfg = VisionConfig()
    det = Detector(cfg)
    frame = _empty_frame()
    cx, cy, r = 900, 400, 16
    _cv2.circle(frame, (cx, cy), r, (200, 60, 220), -1)
    bar_w = int(r * 2 * 2.0)
    bar_y = cy - r - int(r * 2 * 1.2)
    # 条周围铺一层亮的橙色（H≈21，不在红色区间，但 V 很高）：
    # 外圈亮度 ≈ 填充亮度 → 黑框校验必须判"没有黑框"
    _cv2.rectangle(frame, (cx - bar_w // 2 - 4, bar_y - 4),
                   (cx + bar_w // 2 + 4, bar_y + 9), (0, 140, 255), -1)
    _cv2.rectangle(frame, (cx - bar_w // 2, bar_y),
                   (cx + bar_w // 2, bar_y + 5), (0, 0, 220), -1)
    seen = 0
    for i in range(4):
        seen += len(det.process(frame, 1000 + i * 500).monsters)
    assert seen == 0, "没有 UI 黑框的红条被当成了血条（地形色带误报回归）"


def test_bar_much_narrower_than_body_is_not_a_pair():
    """条宽只有身体 1/4 → 物理上不可能是"头顶血条"，不许配对。

    真实录像里最常见的误报是"血条是真的，但下面的身体配到了地形块"：
    岩石/树/珊瑚的色块正好落在条下方窗口里。头顶血条是按怪物宽度画的，
    宽度比落在 0.15~0.35 的一定是配错了。
    """
    import cv2 as _cv2

    cfg = VisionConfig()
    det = Detector(cfg)
    frame = _empty_frame()
    cx, cy, r = 900, 400, 40          # 身体 80px 宽
    _cv2.circle(frame, (cx, cy), r, (200, 60, 220), -1)
    bar_w = 20                        # 条只有 20px 宽 → 0.25 倍
    bar_y = cy - r - 24
    _cv2.rectangle(frame, (cx - bar_w // 2, bar_y),
                   (cx + bar_w // 2, bar_y + 5), (0, 0, 220), -1)
    seen = 0
    for i in range(4):
        seen += len(det.process(frame, 1000 + i * 500).monsters)
    assert seen == 0, "过窄的条被配给了过宽的身体（地形块误报回归）"


def test_bar_overlapping_body_upper_part_still_matches():
    """条压在身体上部（高个怪）仍要认 —— 不能要求"身体顶边严格在条下方"。

    TLI 的冰晶簇类怪：血条画在身体上部，实测条底 y=82 而身体包围盒
    y=68~137。旧判据要求 `top >= bottom + 1`，这类怪全被拒掉。
    """
    import cv2 as _cv2

    cfg = VisionConfig()
    det = Detector(cfg)
    frame = _empty_frame()
    cx, cy, r = 900, 400, 30
    _cv2.circle(frame, (cx, cy), r, (200, 60, 220), -1)
    bar_w = int(r * 2 * 0.9)
    bar_y = cy - r + 6                 # 落在身体上部
    # 游戏画的条自带 1px 黑描边（压在怪物身上也是黑的），先铺黑再铺红
    _cv2.rectangle(frame, (cx - bar_w // 2 - 1, bar_y - 1),
                   (cx + bar_w // 2 + 1, bar_y + 5), (0, 0, 0), -1)
    _cv2.rectangle(frame, (cx - bar_w // 2, bar_y),
                   (cx + bar_w // 2, bar_y + 4), (0, 0, 220), -1)
    seen = 0
    for i in range(4):
        seen += len(det.process(frame, 1000 + i * 500).monsters)
    assert seen >= 1, "条压在身体上部的怪被拒（TLI 冰晶簇召回回归）"


def test_moving_blob_becomes_monster():
    """会动 + 有血条才是怪：位移超过容差 → 第二帧起确认。"""
    cfg = VisionConfig()
    det = Detector(cfg)
    seen = 0
    for i in range(4):
        frame = _empty_frame()
        _draw_blob(frame, 860 + i * 12, 400)  # 每帧右移 12px
        st = det.process(frame, 1000 + i * 500)
        seen += len(st.monsters)
    assert seen >= 1, "移动且带血条的色块必须被确认为怪物"


def test_moving_blob_without_hpbar_is_not_monster():
    """没有血条的移动色块不是怪物 —— 这是真实录像里 564 个误报的根因。

    火炬之光 PC 2560×1380 录屏：红岩/冰面/火焰/技能特效全部通过"高饱和
    色块 + 2px 位移"两道旧判据，564 个检出里 0 个是真怪。加上血条闸门后
    这些必须全部消失，而真正的怪（有头顶横条）仍然要能被认出来。
    """
    det = Detector(VisionConfig())
    seen = 0
    for i in range(4):
        frame = _empty_frame()
        _draw_blob(frame, 860 + i * 12, 400, hpbar=False)
        seen += len(det.process(frame, 1000 + i * 500).monsters)
    assert seen == 0, f"无血条的移动色块被当成怪物 {seen} 次（误报回归）"


def test_hpbar_gate_can_be_disabled():
    """闸门可关：老 profile / 没有血条的游戏显式关闭后回到旧行为。"""
    det = Detector(VisionConfig(monster_hpbar=False))
    seen = 0
    for i in range(4):
        frame = _empty_frame()
        _draw_blob(frame, 860 + i * 12, 400, hpbar=False)
        seen += len(det.process(frame, 1000 + i * 500).monsters)
    assert seen >= 1, "关掉血条闸门后，移动色块应恢复为怪物（兼容旧行为）"



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


# ---------------- 角色定位与坐标基准（真实 2560×1380 录像暴露） ----------------

def test_character_coords_use_original_frame_space():
    """character 必须是**原始帧坐标**，不是缩放后的 work 坐标。

    真实事故：`char_x, char_y = ww // 2, wh // 2` 用的是缩放后尺寸，而
    `SceneState.w/h` 是原始尺寸。2560 宽录像被缩到 1280 检测，于是
    character=(640,345) 落在 2560 宽画面的左 1/4，所有走位点整体偏到左边
    （回放里 232 条命令有 154 条落在 (640,125)/(860,345) 这四个点上）。
    """
    cfg = VisionConfig()
    cfg.max_detect_width = 1280
    cfg.scene_gate.enabled = False
    det = Detector(cfg)
    frame = _empty_frame(2560, 1380)
    st = det.process(frame, 1000)
    assert st.w == 2560 and st.h == 1380
    cx, cy = st.character
    assert 1100 <= cx <= 1460, f"character.x={cx} 不是原始帧中心附近（坐标基准混用了）"
    assert 590 <= cy <= 790, f"character.y={cy} 不是原始帧中心附近"


def test_character_tracks_static_block_against_moving_background():
    """镜头跟随角色：背景整体在动、角色几乎不动 → 角色位置应偏到静止块那边。

    写死屏幕中心会让所有走位点固定在"中心 ± 距离"的几个坐标上；这里验证
    帧差修正真的会把角色位置拉到"最静的那一块"。
    """
    cfg = VisionConfig()
    cfg.scene_gate.enabled = False
    det = Detector(cfg)
    rng = np.random.default_rng(3)
    hero_x, hero_y = 560, 400          # 明显偏离中心（640, 360）
    hero = rng.integers(0, 255, (150, 120, 3), dtype=np.uint8)

    got = None
    for i in range(8):
        # 背景每帧全换（模拟镜头平移），角色块固定不动
        frame = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
        frame[hero_y - 75:hero_y + 75, hero_x - 60:hero_x + 60] = hero
        got = det.process(frame, 1000 + i * 120).character

    cx, cy = got
    assert abs(cx - hero_x) < 120 and abs(cy - hero_y) < 110, (
        f"角色位置估计 {got} 没跟上静止块 ({hero_x},{hero_y})（还锁在屏幕中心？）")


def test_character_track_can_be_disabled():
    """老行为可退回：关掉帧差修正就永远是屏幕中心。"""
    cfg = VisionConfig(character_track=False)
    cfg.scene_gate.enabled = False
    det = Detector(cfg)
    rng = np.random.default_rng(5)
    for i in range(4):
        frame = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
        st = det.process(frame, 1000 + i * 120)
    assert st.character == (640, 360), "关掉 character_track 后必须回到屏幕中心"


# ---------------------------------------------------------------- 真光柱（不是模拟器那种）

def _draw_ingame_pillar(frame, sx, sy, w=4, h=120, core=1,
                        hue=25, sat=95, val=150):
    """按**真实游戏**的样子画光柱：暗背景上半透明的淡金 + 白热芯。

    参数取真实帧实测值（S≈70~120 / V≈120~170），而不是模拟器的 (36,190,255)
    —— 后者 S=218/V=255，是当初阈值被调错的原因。
    """
    import cv2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    y0 = sy - h
    hsv[y0:sy, sx - w // 2 : sx - w // 2 + w] = (hue, sat, val)
    if core:
        hsv[y0:sy, sx - core // 2 : sx - core // 2 + core] = (hue, 30, 235)
    frame[:] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_ingame_thin_pillar_is_detected():
    """3~6px 宽的真光柱必须能检出。

    这是曾经"真光柱一根都看不见"的根因：只按饱和度取金边 → 白芯（S 很低）
    把 4px 的光柱劈成左右各 1px 的两条碎边 → 被 min_width / 高宽比逐个筛掉。
    结果只有 14px 粗的模拟器光柱能过。
    """
    for w, core in ((3, 1), (4, 1), (6, 2), (8, 3)):
        cfg = VisionConfig()
        det = Detector(cfg)
        frame = _empty_frame()
        _draw_ingame_pillar(frame, 640, 500, w=w, h=120, core=core)
        state = process2(det, frame)
        assert len(state.pillars) == 1, f"{w}px 宽的真光柱漏检了"
        assert state.pillars[0].color == "gold"


def test_pure_white_streak_is_not_a_pillar():
    """纯白竖条不是光柱 —— 真光柱是"淡金边 + 白热芯"，必须有彩色成分。

    （技能特效、白色文字、UI 分隔线都长这样，只靠白芯判据挡不住。）
    """
    cfg = VisionConfig()
    det = Detector(cfg)
    frame = _empty_frame()
    _draw_ingame_pillar(frame, 640, 500, w=6, h=120, core=6, hue=0, sat=10, val=235)
    state = process2(det, frame)
    assert state.pillars == []


def test_short_gold_fragment_is_not_a_pillar():
    """19px 高的黄色碎片不是光柱（真实帧上实测会误报两次）。"""
    cfg = VisionConfig()
    det = Detector(cfg)
    frame = _empty_frame()
    _draw_ingame_pillar(frame, 640, 500, w=5, h=19, core=1)
    state = process2(det, frame)
    assert state.pillars == []
