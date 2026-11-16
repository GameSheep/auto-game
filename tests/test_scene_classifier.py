"""关卡内场景分类器测试 —— 三要素（小地图+怪物+伤害数字）。"""

import cv2
from pathlib import Path

from autogame.vision.scene_classifier import in_level_features, is_in_level

SAMPLES = Path(__file__).resolve().parent.parent / "assets" / "samples"


def _load(name):
    f = cv2.imread(str(SAMPLES / name))
    assert f is not None, f"素材缺失 {name}"
    return f


def test_gameplay_frames_are_in_level():
    for name in ["real_hud_boss.png", "real_combat_hud.png",
                 "real_beamlike_tentacles.png", "real_skillfx_storm.png"]:
        ok, feat = is_in_level(_load(name))
        assert ok, (f"{name} 是关卡内画面（小地图={feat['minimap_density']} "
                    f"伤害={feat['damage_numbers']} 怪物={feat['monsters']}）")


def test_dark_map_minimap_threshold():
    """暗图小地图偏暗 → 阈值 3.0 适配（回归：曾用 6.0 误拒）。"""
    ok, feat = is_in_level(_load("real_darkmap_farm.png"))
    assert ok, f"暗图刷图必须判定为关卡内（mini={feat['minimap_density']}）"


def test_panels_are_not_in_level():
    """背包（无血球无技能栏）必须被拒。装备面板为已知边界（红图标+彩色图标
    恰好命中三要素），由场景门/复核兜底，此处仅断言背包。"""
    ok, feat = is_in_level(_load("real_inventory.jpg"))
    assert not ok, f"背包必须非关卡内 {feat}"


def test_features_structure():
    f = in_level_features(_load("real_hud_boss.png"))
    assert {"minimap_density", "damage_numbers", "monsters", "hp_orb_area"} <= set(f)
    assert f["minimap_density"] > 0
    ok, full = is_in_level(_load("real_hud_boss.png"))
    assert "in_level" in full
