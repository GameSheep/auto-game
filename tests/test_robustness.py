"""视觉鲁棒性测试 —— 打破"检测器和模拟器循环论证"。

覆盖真实场景的退化条件：
  - JPEG 压缩（代理端 q40-q90）
  - 高斯噪声 / 亮度偏移（环境光变化）
  - 多分辨率（720p/1080p/1440p/2160p，降采样路径）
  - 干扰色块（非光柱颜色不应误报）
  - 多尺度模板匹配（UI 缩放 0.8x-1.25x）
  - 多光柱场景（去重与近距优先）
"""

import cv2
import numpy as np
import pytest

from autogame.config import VisionConfig
from autogame.sim.scene import SimPillar, SimWorld
from autogame.vision.detector import Detector


def process2(det, frame, ts=1000):
    """喂同一帧两次（时域确认）。"""
    det.process(frame, ts)
    return det.process(frame, ts + 1000)


def _make_frame(seed=99, pillars=None):
    world = SimWorld(seed=seed)
    if pillars is not None:
        world.pillars = pillars
    return world.render()


def _gold_pillar_at(wx, wy):
    return SimPillar(wx, wy, "gold")


def _find_gold(state, expect_min=1):
    golds = [p for p in state.pillars if p.color == "gold"]
    assert len(golds) >= expect_min, f"应检出≥{expect_min}个金色光柱，实际 {state.pillars}"
    return golds[0]


# ---------------- JPEG 压缩鲁棒性 ----------------

@pytest.mark.parametrize("quality", [40, 60, 80, 90])
def test_detection_survives_jpeg_compression(quality):
    frame = _make_frame(pillars=[_gold_pillar_at(200, 50)])
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    det = Detector(VisionConfig())
    state = process2(det, dec)
    p = _find_gold(state)
    # 压缩下位置仍需在光柱附近（±30px）
    sx = int(200 - 0 + 640)
    assert abs(p.cx - sx) <= 30, f"压缩 q{quality} 后位置漂移过大: {p.cx} vs {sx}"


# ---------------- 噪声与光照鲁棒性 ----------------

def test_detection_survives_gaussian_noise():
    frame = _make_frame(pillars=[_gold_pillar_at(150, 30)])
    rng = np.random.default_rng(7)
    noisy = frame.astype(np.int16) + rng.normal(0, 10, frame.shape).astype(np.int16)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    det = Detector(VisionConfig())
    state = process2(det, noisy)
    _find_gold(state)


def test_detection_survives_moderate_brightness_shift():
    frame = _make_frame(pillars=[_gold_pillar_at(150, 30)])
    bright = cv2.convertScaleAbs(frame, alpha=1.0, beta=30)
    det = Detector(VisionConfig())
    state = process2(det, bright)
    _find_gold(state)


def test_no_false_positive_on_random_colorful_textures():
    """随机彩色纹理（非金/粉色系）不得误报为光柱。"""
    rng = np.random.default_rng(42)
    frame = np.full((720, 1280, 3), (28, 24, 22), dtype=np.uint8)
    # 撒 30 个随机彩色竖条（避开金/粉色 HSV 区间）
    for _ in range(30):
        x, y = rng.integers(0, 1100), rng.integers(100, 600)
        w, h = int(rng.integers(6, 20)), int(rng.integers(40, 130))
        # 蓝/绿/青色系（hue 60-130）
        hue = int(rng.integers(60, 130))
        bgr = cv2.cvtColor(np.uint8([[[hue, 230, 230]]]), cv2.COLOR_HSV2BGR)[0, 0]
        frame[y : y + h, x : x + w] = bgr
    det = Detector(VisionConfig())
    state = process2(det, frame)
    assert len(state.pillars) == 0, f"彩色干扰纹理误报: {[p.color for p in state.pillars]}"
    assert state.aframe is None


# ---------------- 多分辨率 ----------------

@pytest.mark.parametrize("size", [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)])
def test_multi_resolution_scaling(size):
    w, h = size
    # 光柱世界坐标按 1280x720 基准设计（屏幕 (840,460)），再整帧缩放
    world = SimWorld(seed=99)
    world.pillars = [_gold_pillar_at(200, 100)]
    base_frame = world.render()
    det = Detector(VisionConfig())
    base = process2(det, base_frame, 1)
    bp = _find_gold(base)  # 基准检测值（720p 无缩放路径）

    if size != (1280, 720):
        frame = cv2.resize(base_frame, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        frame = base_frame
    state = process2(det, frame)
    p = _find_gold(state)
    # 期望位置 = 基准检测坐标 × 整帧缩放系数（detector 输出还原到原分辨率坐标系）
    fx, fy = w / 1280, h / 720
    tol = max(25, fx * 25)
    assert abs(p.cx - bp.cx * fx) <= tol, f"@{w}x{h}: cx={p.cx} vs {bp.cx * fx}"
    assert abs(p.cy - bp.cy * fy) <= tol, f"@{w}x{h}: cy={p.cy} vs {bp.cy * fy}"


# ---------------- 多光柱 ----------------

def test_multi_pillar_nearest_first_and_dedup():
    world = SimWorld(seed=99)
    world.pillars = [
        _gold_pillar_at(400, 200),   # 距离 ~533
        _gold_pillar_at(100, 30),    # 距离 ~605
        _gold_pillar_at(300, -100),  # 距离 ~400 近
        SimPillar(405, 195, "gold"), # 重叠 → 去重
    ]
    frame = world.render()
    det = Detector(VisionConfig())
    state = process2(det, frame)
    golds = [p for p in state.pillars if p.color == "gold"]
    assert len(golds) >= 2
    # 排序按距离近优先
    assert golds[0].distance <= golds[-1].distance
    # 位置互不重叠（去重生效）
    for i in range(len(golds)):
        for j in range(i + 1, len(golds)):
            assert abs(golds[i].cx - golds[j].cx) > 30 or abs(golds[i].cy - golds[j].cy) > 40


# ---------------- 多尺度模板匹配 ----------------

def test_multiscale_template_matching(tmp_path):
    """模板图缩放 ±25% 后仍能命中（模拟游戏 UI 缩放）。"""
    # 从 sim 帧裁出真实 A框作为模板
    world = SimWorld(seed=99)
    world.pillars = [_gold_pillar_at(60, 20)]  # 距离 63 < 130 → A框可见
    frame = world.render()
    det0 = Detector(VisionConfig())
    st0 = det0.process(frame, 1000)
    assert st0.aframe is not None
    ax, ay, aw, ah = st0.aframe.x, st0.aframe.y, st0.aframe.w, st0.aframe.h
    tpl = frame[ay - ah // 2 : ay + ah // 2, ax - aw // 2 : ax + aw // 2]
    tpl_path = tmp_path / "aframe.png"
    cv2.imwrite(str(tpl_path), tpl)

    cfg = VisionConfig()
    cfg.aframe.template = str(tpl_path)
    cfg.aframe.template_scales = [0.8, 0.9, 1.0, 1.1, 1.25]
    det = Detector(cfg, templates_dir=tmp_path)
    assert det._template is not None

    # 同一帧：模板匹配应命中
    st = det.process(frame, 2000)
    assert st.aframe is not None, "模板匹配未命中原始 A框"

    # UI 缩放 0.85x 的场景帧：多尺度仍应命中
    for s in (0.85, 1.15):
        scaled = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        st2 = det.process(scaled, 3000)
        assert st2.aframe is not None, f"多尺度模板在 {s}x 缩放下未命中"


def test_template_miss_below_threshold():
    """没有 A框的画面，模板匹配不得误报（高阈值下不相关模板不命中）。"""
    frame = _make_frame(pillars=[])  # 无光柱无 A框
    # 构造一个有纹理的不相关模板（纯色模板对 TM_CCOEFF_NORMED 是零方差退化输入）
    tpl = np.zeros((30, 30, 3), dtype=np.uint8)
    tpl[:, :, 0] = 200  # 蓝色底
    cv2.putText(tpl, "X", (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        cv2.imwrite(f.name, tpl)
        cfg = VisionConfig()
        cfg.aframe.template = f.name
        cfg.aframe.template_threshold = 0.85  # 高阈值
        det = Detector(cfg)
        state = det.process(frame, 1000)
        assert state.aframe is None


# ---------------- 退化画面 ----------------

def test_black_frame_and_solid_frame_no_crash():
    det = Detector(VisionConfig())
    black = np.zeros((720, 1280, 3), dtype=np.uint8)
    st1 = det.process(black, 1000)
    assert st1.pillars == [] and st1.aframe is None
    solid = np.full((720, 1280, 3), 128, dtype=np.uint8)
    st2 = det.process(solid, 1001)
    assert st2.aframe is None  # 灰屏 S 低 → 不该有 A框
    # 奇数尺寸帧
    odd = np.full((721, 1279, 3), (28, 24, 22), dtype=np.uint8)
    st3 = det.process(odd, 1002)
    assert st3.character == (639, 360)


def test_frame_delta_metric_sanity():
    """帧差指标：静止 0，世界滚动 > 1%（防撞墙信号有效性）。"""
    world = SimWorld(seed=99)
    det = Detector(VisionConfig())
    s1 = det.process(world.render(), 1)
    assert s1.frame_delta == 0.0
    s2 = det.process(world.render(), 2)  # 完全静止
    assert s2.frame_delta < 0.3, "静止时帧差必须≈0"
    world.char_wx += 60
    s3 = det.process(world.render(), 3)
    assert s3.frame_delta > 1.0, "滚动 60px 帧差必须显著"
