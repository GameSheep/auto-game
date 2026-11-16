"""A框检测精度回归 —— 真实帧上的误报必须为 0。

背景（真实数据驱动）：
  早期 A框检测只看"亮 + 镂空 + 方形"，在 100 张真实帧上检出 11 个候选，
  人工核验后发现 **11/11 全是误报**（伤害数字 7.8亿 / 83.2亿、技能图标、
  紫色 Boss 特效、血条高亮）—— 因为它们同样是"亮色 + 有孔洞"。

修复：加入结构校验（框内必须是**暗底 + 亮字形**），并启用从实机截图裁出的
真实 A框模板。修复后同样 100 张帧只命中 1 个，且该命中经人工核验是真正的
"暗底白字 A + 物品名牌"拾取提示。

本文件锁死这两条：真阳性必须检出、已知误报帧必须全部拒绝。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from autogame.config import AFrameConfig, VisionConfig
from autogame.vision.detector import Detector

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "templates" / "aframe.png"
# 真实帧素材：tests/fixtures 入库（9 帧，12MB），assets/batch 是本地全量 100 帧
FIXTURES = ROOT / "tests" / "fixtures" / "frames"
BATCH = FIXTURES if FIXTURES.exists() else (ROOT / "assets" / "batch")


def _cfg() -> VisionConfig:
    return VisionConfig(
        aframe=AFrameConfig(
            template=str(TEMPLATE),
            template_threshold=0.72,
            glyph_white_only=True,  # 与 config.yaml 部署配置一致
        ),
    )


def _frame(name: str) -> np.ndarray:
    img = cv2.imread(str(BATCH / name), cv2.IMREAD_COLOR)
    assert img is not None, f"缺少测试帧 {name}"
    return img


def _process_twice(det: Detector, frame: np.ndarray):  # noqa: ANN001
    det.process(frame, 0)
    return det.process(frame, 200)


def test_template_asset_exists_and_is_small():
    """模板必须存在且是"暗底白字"小方框（不是整张 UI 截图）。"""
    assert TEMPLATE.exists(), "templates/aframe.png 缺失（标定资产未入库）"
    tpl = cv2.imread(str(TEMPLATE), cv2.IMREAD_COLOR)
    h, w = tpl.shape[:2]
    assert 8 <= w <= 40 and 8 <= h <= 40, f"模板尺寸异常 {w}x{h}（应为小方框）"
    assert abs(w - h) <= 6, "A框模板应接近正方形"


def test_real_aframe_is_detected():
    """真实帧里的 A 提示（暗底白字 A + 物品名牌）必须检出。"""
    det = Detector(_cfg())
    state = _process_twice(det, _frame("img_098_user_farm_f2940.png"))
    assert state.aframe is not None, "真实 A 提示未检出（模板/阈值失效）"
    af = state.aframe
    # 人工核验位置：(853,455)
    assert abs(af.x - 853) <= 10 and abs(af.y - 455) <= 10, f"A框位置偏移 {af.x},{af.y}"
    assert af.score >= 0.72


def test_damage_numbers_are_not_aframe():
    """伤害数字（7.8亿 / 83.2亿 / 38）与技能图标绝不能被当成 A框。"""
    det = Detector(_cfg())
    # 早期误报帧（人工核验：全是伤害数字/图标/Boss特效）
    for name in [
        "img_008_legend_c_f420.png",
        "img_060_user_farm_f1800.png",
        "img_014_user_farm_f540.png",
        "img_080_user_farm_f2400.png",
        "img_084_user_farm_f2520.png",
    ]:
        state = _process_twice(det, _frame(name))
        assert state.aframe is None, f"{name} 仍误报 A框 @({state.aframe.x},{state.aframe.y})"


def test_red_icon_is_not_aframe():
    """暗底 + **红色**字形（Buff/状态图标）不是 A 提示：真实 A 是白字。

    这三帧曾在两尺度配置下被误报在同一位置 (308,172)，人工核验是红色图标。
    """
    det = Detector(_cfg())
    for name in [
        "img_031_user_farm_f960.png",
        "img_037_user_farm_f1110.png",
        "img_040_user_farm_f1200.png",
    ]:
        state = _process_twice(det, _frame(name))
        assert state.aframe is None, f"{name} 红色图标仍被误报为 A框"


def test_bright_core_is_rejected():
    """亮色填充的"空心框"（内部不是暗底）必须被结构校验拒绝。

    构造：亮色边框 + 中亮填充（V≈150，高于暗底上限 110，低于亮字形下限 200）。
    形状校验（镂空率）会放行，只有结构校验能拦住 —— 这就是伤害数字的真实形态。
    """
    frame = np.full((360, 640, 3), 20, dtype=np.uint8)
    # 外框亮色
    cv2.rectangle(frame, (300, 160), (340, 200), (0, 220, 255), -1)
    # 内部中亮填充（不是暗底）
    cv2.rectangle(frame, (306, 166), (334, 194), (60, 150, 150), -1)
    # 留一个暗洞，让"镂空率"过关
    cv2.rectangle(frame, (315, 175), (325, 185), (10, 10, 10), -1)
    det = Detector(_cfg())
    state = _process_twice(det, frame)
    assert state.aframe is None, "亮底框被误判为 A框（结构校验失效）"


def test_without_template_still_works():
    """模板缺失时降级为形状启发式，不崩（现场未标定也能跑）。"""
    cfg = VisionConfig(aframe=AFrameConfig(template=None))
    det = Detector(cfg)
    state = _process_twice(det, _frame("img_098_user_farm_f2940.png"))
    # 不要求检出（真实 A框是暗底白字，形状启发式本就抓不到），但不能抛异常
    assert state.w > 0
