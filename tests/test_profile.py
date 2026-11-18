"""profile（游戏适配包）加载链的回归测试。

钉死三件容易反复出错的事：
  1. profile 的 reference_resolution 必须进到 vision.reference_resolution
     （否则 HUD 排除区整体错位，HUD 图标会被当成掉落物）
  2. **不能**顺手把 template_base_width 也设成 profile 的基准宽
     （模板经 make_template 统一归一化存 1280 宽；改成 960 会让反例从 1.00 掉到 0.41，
      表现是"菜单拦不住"，但看起来像"模板失效"，极难排查）
  3. 没有正向模板 + fail_closed=True 时必须拦下，不能放行
     （历史行为是"门禁停用恒 True"，等于未标定的游戏直接盲操作）
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autogame.config import VisionConfig, load_config
from autogame.profile import (
    PROFILES_DIR,
    list_profiles,
    load_profile,
)
from autogame.vision.detector import Detector

ROOT = Path(__file__).resolve().parent.parent


def test_builtin_profiles_exist():
    ids = list_profiles()
    assert "generic_1280x720" in ids
    assert "tli_phone960" in ids
    assert not any(i.startswith("_") for i in ids), "下划线开头的是脚手架，不该被列出"


def test_load_profile_reads_meta():
    p = load_profile("tli_phone960")
    assert p.id == "tli_phone960"
    assert p.reference_resolution == (960, 432)
    assert p.status in ("uncalibrated", "partial", "calibrated")
    assert p.templates_dir == PROFILES_DIR / "tli_phone960" / "templates"


def test_load_profile_accepts_path():
    p = load_profile(PROFILES_DIR / "tli_phone960")
    assert p.id == "tli_phone960"


def test_missing_profile_raises():
    with pytest.raises(FileNotFoundError):
        load_profile("__does_not_exist__")


def test_reference_resolution_is_injected():
    prof = load_profile("tli_phone960")
    cfg = load_config(ROOT / "config.yaml", profile=prof)
    assert cfg.vision.reference_resolution == [960, 432]


def test_template_base_width_stays_1280():
    """模板是 1280 归一化的，base_width 跟着画面尺寸走会让匹配整体失败。"""
    prof = load_profile("tli_phone960")
    cfg = load_config(ROOT / "config.yaml", profile=prof)
    assert cfg.vision.scene_gate.template_base_width == 1280


def test_profile_config_overrides_base():
    """profile 的 config 段要能盖过 config.yaml（游戏特定设置优先）。"""
    prof = load_profile("tli_phone960")
    cfg = load_config(ROOT / "config.yaml", profile=prof)
    assert cfg.vision.max_detect_width == 960
    assert cfg.vision.scene_gate.roi == [600, 235, 360, 197]


def test_hud_exclusion_scaled_from_reference(tmp_path):
    """基准 960x432 的排除区，在 1920x864 帧上要等比放大 2 倍。"""
    cfg = VisionConfig(
        reference_resolution=[960, 432],
        hud_exclusions=[[0, 0, 100, 50]],
    )
    det = Detector(cfg, tmp_path)
    import numpy as np

    frame = np.full((864, 1920, 3), 200, dtype="uint8")
    out = det._apply_hud_exclusions(frame, 1920, 864, 1.0)
    # 100x50 → 200x100，左上角被清零
    assert out[10, 10].max() == 0
    assert out[99, 199].max() == 0
    assert out[110, 210].min() == 200, "超出排除区的不该被动"


def test_no_positive_template_fails_closed(tmp_path):
    """一张正向模板都没有时必须拦下（未标定游戏不许盲操作）。"""
    cfg = VisionConfig()
    det = Detector(cfg, tmp_path)  # 空目录：既无 gate_pos 也无 skillbar
    import numpy as np

    frame = np.full((720, 1280, 3), 60, dtype="uint8")
    st = det.process(frame, 1000)
    assert st.in_game is False
    assert det.diag.get("gate_reason") == "no_template"


def test_no_positive_template_can_be_opened(tmp_path):
    """显式关掉 fail_closed 才允许放行（离线分析场景用）。"""
    cfg = VisionConfig()
    cfg.scene_gate.fail_closed = False
    det = Detector(cfg, tmp_path)
    import numpy as np

    frame = np.full((720, 1280, 3), 60, dtype="uint8")
    st = det.process(frame, 1000)
    assert st.in_game is True


def test_template_index_base_width(tmp_path):
    """_index.json 里声明的 base_width 要能覆盖默认值。"""
    import json

    cfg = VisionConfig()
    (tmp_path / "_index.json").write_text(
        json.dumps({"gate_neg_x.png": {"base_width": 960}}), encoding="utf-8")
    det = Detector(cfg, tmp_path)
    assert det._base_width("gate_neg_x.png") == 960
    assert det._base_width("not_in_index.png") == 1280
