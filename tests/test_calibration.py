"""标定工具测试 —— HSV 采样建议逻辑（非交互模式）。"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from calibrate_hsv import sample_range, yaml_snippet  # noqa: E402


def _frame_with_gold_pillar():
    frame = np.full((720, 1280, 3), (28, 24, 22), dtype=np.uint8)
    frame[300:410, 600:614] = (36, 190, 255)  # 金色光柱
    frame[306:406, 605:609] = (255, 255, 255)  # 白热芯（真实光柱特征，窄条）
    return frame


def test_sample_range_on_gold_pillar_rect():
    frame = _frame_with_gold_pillar()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    r = sample_range(hsv, (590, 290, 40, 130))
    assert r["samples"] > 0
    assert len(r["ranges"]) == 1, "金柱场景应得单一区间"
    (lo_h, lo_s, lo_v), (hi_h, hi_s, hi_v) = r["ranges"][0]
    # 建议 range 必须覆盖光柱主体真实 HSV (h≈21, s≈219, v≈255)
    assert lo_h <= 21 <= hi_h, f"H 范围未覆盖: {lo_h}-{hi_h}"
    assert lo_s <= 219 <= hi_s, f"S 范围未覆盖: {lo_s}-{hi_s}"
    assert lo_v <= 255 <= hi_v or hi_v == 255, f"V 范围未覆盖: {lo_v}-{hi_v}"


def test_sample_range_suggested_range_detects_pillar():
    """采样建议的范围直接喂给检测器应能检出该光柱（标定闭环有效性）。"""
    frame = _frame_with_gold_pillar()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    r = sample_range(hsv, (590, 290, 40, 130))
    assert len(r["ranges"]) == 1, "金柱场景应得单一区间"
    from autogame.config import VisionConfig
    from autogame.vision.detector import Detector

    cfg = VisionConfig()
    cfg.hsv = {"calibrated": r["ranges"][0]}
    det = Detector(cfg)
    det.process(frame, 1000)
    state = det.process(frame, 2000)  # 时域确认需两帧
    golds = [p for p in state.pillars if p.color == "calibrated"]
    assert len(golds) == 1, "标定建议的范围必须能检出原光柱"
    assert abs(golds[0].cx - 607) <= 15


def test_yaml_snippet_single_range():
    s = yaml_snippet("gold", [[[18, 120, 180], [35, 255, 255]]])
    assert "gold" in s and "[[18, 120, 180]" in s


def test_yaml_snippet_wrapped_range():
    s = yaml_snippet("pink", [[[170, 90, 160], [179, 255, 255]],
                              [[0, 90, 160], [5, 255, 255]]])
    assert "pink_part1" in s and "pink_part2" in s
