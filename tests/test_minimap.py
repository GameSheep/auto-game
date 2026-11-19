"""小地图定位：滚动小地图 + 离线底图 → 人物在整图的位置。

测试用的是合成地图（随机地形 + 方块地标 + 模糊），不是真实截图 —— 真实截图该进
`tests/fixtures`，而这里要验的是"配准算得对不对、该不该采信"，和画面内容无关。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from autogame.config import MinimapConfig
from autogame.vision.minimap import MapLocator, MinimapLocator, _ncc

MAP_W = 1600
MAP_H = 1200
ZOOM = 0.1          # 小地图像素 / 底图像素
PATCH = 60          # 小地图裁片边长（1:1 视角下 = 底图 600px）


def make_map(seed: int = 7) -> np.ndarray:
    """合成一张"像地图"的底图：模糊噪声打底 + 若干方块地标（提供边缘）。"""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, (MAP_H, MAP_W), dtype=np.uint8)
    base = cv2.GaussianBlur(noise, (0, 0), 1.0)   # 轻模糊：留纹理，别抹成纯色
    for _ in range(40):
        x = int(rng.integers(0, MAP_W - 60))
        y = int(rng.integers(0, MAP_H - 60))
        w = int(rng.integers(20, 70))
        h = int(rng.integers(20, 70))
        cv2.rectangle(base, (x, y), (x + w, y + h), int(rng.integers(0, 255)), -1)
    return cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)


def crop_minimap(world: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """在底图上取一块（这就是游戏中滚动小地图显示的内容），按 zoom 缩小。"""
    half = int(round(PATCH / ZOOM / 2))
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    win = world[y0: y0 + half * 2, x0: x0 + half * 2]
    return cv2.resize(win, (PATCH, PATCH), interpolation=cv2.INTER_AREA)


def textured_center(world: np.ndarray, cx: int, cy: int, span: int = 400) -> tuple[int, int]:
    """就近找一个"有地形"的中心。

    合成地图里的方块是纯色填充，落在里面时裁片方差接近 0 —— 真实世界也有这种地方
    （纯色水面、全黑阴影），那种裁片本来就不该给出坐标（见 _MIN_PATCH_STD）。
    所以测试要挑有纹理的位置，而不是赌随机数刚好不在方块上。
    """
    half = int(round(PATCH / ZOOM / 2))
    # 贴边点会让 crop_minimap 夹到 0，实际中心就不是给的那个点了 —— 先夹到内部
    cx = min(max(cx, half + 1), MAP_W - half - 1)
    cy = min(max(cy, half + 1), MAP_H - half - 1)
    for d in range(0, span, 40):
        for dx, dy in ((d, 0), (-d, 0), (0, d), (0, -d), (d, d), (-d, -d)):
            x, y = cx + dx, cy + dy
            if not (half < x < MAP_W - half and half < y < MAP_H - half):
                continue
            win = world[y - half: y + half, x - half: x + half]
            if float(cv2.cvtColor(win, cv2.COLOR_BGR2GRAY).std()) > 15:
                return x, y
    return cx, cy


def test_locate_finds_true_position():
    world = make_map()
    loc = MapLocator(world, zoom=ZOOM, threshold=0.5, edge_threshold=0.2, inset=0)
    cx, cy = textured_center(world, 800, 600)
    patch = crop_minimap(world, cx, cy)
    hit = loc.locate(patch)
    assert hit is not None, f"定位失败：{loc.last_diag}"
    # 1 个缩略像素 = 1/zoom 个底图像素，容差给 2 个缩略像素
    assert abs(hit.x - cx) <= 2 / ZOOM
    assert abs(hit.y - cy) <= 2 / ZOOM
    assert hit.score > 0.5


def test_locate_respects_inset():
    """inset 裁掉边框：裁片带一圈纯色边框时应仍能定位（边框是真实小地图的常态）。"""
    world = make_map()
    loc = MapLocator(world, zoom=ZOOM, threshold=0.5, edge_threshold=0.2, inset=5)
    cx, cy = textured_center(world, 500, 400)
    patch = crop_minimap(world, cx, cy)
    patch = cv2.copyMakeBorder(patch, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    hit = loc.locate(patch)
    assert hit is not None, f"定位失败：{loc.last_diag}"
    assert abs(hit.x - cx) <= 2 / ZOOM
    assert abs(hit.y - cy) <= 2 / ZOOM


def test_hint_window_marks_hinted_and_still_correct():
    world = make_map()
    loc = MapLocator(world, zoom=ZOOM, threshold=0.5, edge_threshold=0.2,
                     search_radius_px=200, inset=0)
    cx, cy = textured_center(world, 900, 700)
    patch = crop_minimap(world, cx, cy)
    hit = loc.locate(patch, hint=(cx + 10, cy + 5))
    assert hit is not None
    assert hit.hinted is True
    assert abs(hit.x - cx) <= 2 / ZOOM


def test_hint_far_away_falls_back_to_global():
    """上一帧位置错了（比如刚切图），窗口里没有答案，必须退全图而不是给窗口里的最优。"""
    world = make_map()
    loc = MapLocator(world, zoom=ZOOM, threshold=0.5, edge_threshold=0.2,
                     search_radius_px=120, inset=0)
    tx, ty = textured_center(world, 200, 900)
    patch = crop_minimap(world, tx, ty)
    hit = loc.locate(patch, hint=(1400, 200))
    assert hit is not None, loc.last_diag
    assert hit.hinted is False
    assert abs(hit.x - tx) <= 2 / ZOOM
    assert abs(hit.y - ty) <= 2 / ZOOM


def test_flat_patch_is_rejected():
    """纯色裁片（黑屏/被面板盖住）不该给坐标：NCC 在零方差下处处 1.0，那不是定位。"""
    loc = MapLocator(make_map(), zoom=ZOOM, threshold=0.5, inset=0)
    flat = np.full((PATCH, PATCH, 3), 40, np.uint8)
    assert loc.locate(flat) is None
    assert "没有地形信息" in loc.last_diag["why"]


def test_unrelated_patch_is_rejected():
    """拿另一张图里的裁片来问，必须返回 None —— 宁可不给坐标，也不给一个像的。"""
    world = make_map(seed=7)
    other = make_map(seed=99)
    loc = MapLocator(world, zoom=ZOOM, threshold=0.55, edge_threshold=0.2, inset=0)
    hit = loc.locate(crop_minimap(other, 400, 400))
    assert hit is None, f"不该给出坐标，却给了 {hit}"


def test_patch_bigger_than_map_is_rejected():
    loc = MapLocator(make_map(), zoom=ZOOM, threshold=0.5, inset=0)
    huge = np.zeros((MAP_H + 10, MAP_W + 10, 3), np.uint8)
    assert loc.locate(huge) is None


def test_zoom_must_be_positive():
    with pytest.raises(ValueError):
        MapLocator(make_map(), zoom=0.0)


def test_ncc_basics():
    a = np.arange(9, dtype=np.float32).reshape(3, 3)
    assert _ncc(a, a) == pytest.approx(1.0, abs=1e-5)
    assert _ncc(a, a[::-1]) < 0.5
    assert _ncc(a, np.zeros((3, 3), np.float32)) == 0.0     # 无方差 → 不算相关
    assert _ncc(a, np.zeros((4, 4), np.float32)) == 0.0     # 尺寸不符


# ---------------------------------------------------------------- MinimapLocator

def _write_map(tmp_path, world):
    p = tmp_path / "map.png"
    cv2.imwrite(str(p), world)
    return p


def test_locator_explains_why_unavailable(tmp_path):
    world = make_map()
    path = _write_map(tmp_path, world)

    off = MinimapLocator(MinimapConfig(enabled=False))
    assert off.ready is False and "未启用" in off.why

    no_map = MinimapLocator(MinimapConfig(enabled=True))
    assert no_map.ready is False and "底图" in no_map.why

    no_zoom = MinimapLocator(MinimapConfig(enabled=True, map_image=str(path)))
    assert no_zoom.ready is False and "zoom" in no_zoom.why

    missing = MinimapLocator(MinimapConfig(enabled=True, map_image=str(tmp_path / "nope.png"), zoom=0.1))
    assert missing.ready is False and "读不出来" in missing.why


def test_locator_updates_position_with_roi(tmp_path):
    world = make_map()
    path = _write_map(tmp_path, world)
    cfg = MinimapConfig(enabled=True, map_image=str(path), zoom=ZOOM,
                        roi=[0, 0, PATCH, PATCH], locate_threshold=0.5, inset=0)
    loc = MinimapLocator(cfg, base_dir=tmp_path)
    assert loc.ready is True, loc.why

    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[0:PATCH, 0:PATCH] = crop_minimap(world, *textured_center(world, 800, 600))
    pos = loc.update(frame)
    assert pos is not None, loc.locator.last_diag
    assert 0.0 <= pos.u <= 1.0 and 0.0 <= pos.v <= 1.0
    assert 0 <= pos.x <= MAP_W


def test_locator_drops_hint_after_repeated_misses(tmp_path):
    """连续认不出就丢掉 hint，否则会一直搜一个早就不在的窗口。"""
    world = make_map()
    path = _write_map(tmp_path, world)
    cfg = MinimapConfig(enabled=True, map_image=str(path), zoom=ZOOM,
                        roi=[0, 0, PATCH, PATCH], locate_threshold=0.5, inset=0)
    loc = MinimapLocator(cfg, base_dir=tmp_path)
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[0:PATCH, 0:PATCH] = crop_minimap(world, *textured_center(world, 800, 600))
    assert loc.update(frame) is not None
    assert loc._last is not None

    blank = np.zeros((720, 1280, 3), np.uint8)
    for _ in range(3):
        assert loc.update(blank) is None
    assert loc._last is None


def test_locator_rejects_single_frame_jump(tmp_path):
    """一帧跳几百像素不可能，先不采信；连续两帧都这么说才认（真传送/切图）。"""
    world = make_map()
    path = _write_map(tmp_path, world)
    cfg = MinimapConfig(enabled=True, map_image=str(path), zoom=ZOOM,
                        roi=[0, 0, PATCH, PATCH], locate_threshold=0.5, inset=0,
                        max_jump_px=200, jump_confirm=2)
    loc = MinimapLocator(cfg, base_dir=tmp_path)

    frame = np.zeros((720, 1280, 3), np.uint8)
    a = textured_center(world, 200, 200)
    b = textured_center(world, 1300, 900)
    frame[0:PATCH, 0:PATCH] = crop_minimap(world, *a)
    assert loc.update(frame) is not None

    frame2 = frame.copy()
    frame2[0:PATCH, 0:PATCH] = crop_minimap(world, *b)
    assert loc.update(frame2) is None          # 第一帧跳变：不采信
    assert loc.update(frame2) is not None      # 第二帧一致：认了

    loc.reset()
    assert loc._last is None
