"""物品名词表（字条模板）匹配。

合成字条用 cv2.putText 渲染——真实字条是从游戏截图上裁的，但这里要验的是
"词表能不能认出、认错的会不会被拦住、低质量字条会不会被拒收"，字体长相不重要。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from autogame.config import NameTagConfig
from autogame.vision.imgio import imwrite_any
from autogame.vision.nametags import NameTagBank


def strip(text: str, w: int = 150, h: int = 24) -> np.ndarray:
    """一张"名字条"：深底 + 白字（游戏里掉落物名字就是这种形态）。"""
    img = np.full((h, w, 3), 24, np.uint8)
    cv2.putText(img, text, (4, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1,
                cv2.LINE_AA)
    return img


def scene(names: list[tuple[int, int, str]], w: int = 420, h: int = 320) -> np.ndarray:
    """黑画面 + 若干名字条（给一个位置）。"""
    img = np.full((h, w, 3), 12, np.uint8)
    for x, y, text in names:
        s = strip(text)
        img[y: y + s.shape[0], x: x + s.shape[1]] = s
    return img


@pytest.fixture()
def bank(tmp_path):
    imwrite_any(tmp_path / "强化石.png", strip("GOLD_STONE"))
    imwrite_any(tmp_path / "药水.png", strip("POTION"))
    b = NameTagBank(tmp_path, threshold=0.72)
    assert b.load() == 2
    return b


def test_identify_returns_name(bank):
    frame = scene([(120, 80, "GOLD_STONE")])
    hit = bank.identify(frame, (100, 60, 200, 60))
    assert hit is not None
    assert hit[0] == "强化石"
    assert hit[1] > 0.9


def test_identify_picks_the_right_one_among_several(bank):
    """两条不同名字同时在画面里，ROI 框住哪条就该返回哪条。"""
    frame = scene([(100, 60, "POTION"), (100, 160, "GOLD_STONE")])
    top = bank.identify(frame, (90, 50, 220, 50))
    bottom = bank.identify(frame, (90, 150, 220, 50))
    assert top is not None and top[0] == "药水"
    assert bottom is not None and bottom[0] == "强化石"


def test_unknown_text_is_not_guessed(bank):
    """画面里是一句词表里没有的话 —— 必须返回 None，不能给"最像的"。"""
    frame = scene([(120, 80, "SOMETHING_ELSE")])
    assert bank.identify(frame, (100, 60, 240, 60)) is None


def test_blank_region_is_rejected(bank):
    frame = np.full((200, 400, 3), 12, np.uint8)
    assert bank.identify(frame, (50, 50, 260, 60)) is None


def test_threshold_is_enforced(tmp_path):
    imwrite_any(tmp_path / "强化石.png", strip("GOLD_STONE"))
    strict = NameTagBank(tmp_path, threshold=0.98)
    assert strict.load() == 1
    # 同一张字条被压缩过（模拟 JPEG 噪声）分数会掉，0.98 的线应该拦得住"像但不一样"
    frame = scene([(120, 80, "GOLD_STONE")])
    noisy = cv2.GaussianBlur(frame, (3, 3), 0.8)
    hit = strict.identify(noisy, (100, 60, 200, 60))
    assert hit is None or hit[1] >= 0.98


def test_low_variance_template_is_skipped(tmp_path):
    """纯色/一块背景的"字条"会让 NCC 处处高分（q_letter.png 的老毛病），装载时就拒收。"""
    imwrite_any(tmp_path / "假的.png", np.full((24, 120, 3), 30, np.uint8))
    imwrite_any(tmp_path / "真.png", strip("POTION"))
    b = NameTagBank(tmp_path, threshold=0.72)
    assert b.load() == 1
    assert b.names == ["真"]
    assert any("方差" in why for _n, why in b.skipped)


def test_tiny_template_is_skipped(tmp_path):
    imwrite_any(tmp_path / "小.png", np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8))
    b = NameTagBank(tmp_path, threshold=0.72)
    assert b.load() == 0
    assert any("太小" in why for _n, why in b.skipped)


def test_multiple_samples_share_one_name(tmp_path):
    """同一物品多张样本：强化石__1.png / 强化石__2.png 都归到「强化石」。"""
    imwrite_any(tmp_path / "强化石__1.png", strip("GOLD_STONE"))
    imwrite_any(tmp_path / "强化石__2.png", strip("GOLD STONE", w=170))
    b = NameTagBank(tmp_path, threshold=0.6)
    assert b.load() == 1                   # load 返回的是**词条数**，不是文件数
    assert b.names == ["强化石"]
    assert len(b._tpl["强化石"]) == 2      # 两条样本都进了候选


def test_scales_variant_still_matches(tmp_path):
    """UI 缩放变了：字条按 1.0 裁的，画面里是 0.85 倍，靠 scales 补一档。"""
    imwrite_any(tmp_path / "强化石.png", strip("GOLD_STONE"))
    b = NameTagBank(tmp_path, threshold=0.6, scales=(1.0, 0.85))
    assert b.load() == 1
    s = strip("GOLD_STONE")
    small = cv2.resize(s, None, fx=0.85, fy=0.85, interpolation=cv2.INTER_AREA)
    frame = np.full((200, 400, 3), 12, np.uint8)
    frame[60: 60 + small.shape[0], 100: 100 + small.shape[1]] = small
    hit = b.identify(frame, (80, 40, 220, 70))
    assert hit is not None and hit[0] == "强化石"


def test_identify_near_anchors_on_pillar(bank):
    """名字在柱子上方：带子以柱心为锚，柱动了名字跟着动。"""
    cfg = NameTagConfig(enabled=True, threshold=0.72, band_up=46, band_down=18, band_w=240)
    pillar_cx, pillar_cy, pillar_w, pillar_h = 200, 260, 8, 80
    name_top = pillar_cy - pillar_h // 2 - 40          # 柱顶再往上 40px
    frame = scene([(140, name_top, "POTION")], h=400)
    cv2.line(frame, (pillar_cx, pillar_cy - pillar_h // 2),
             (pillar_cx, pillar_cy + pillar_h // 2), (0, 200, 255), pillar_w)

    hit = bank.identify_near(frame, pillar_cx, pillar_cy, pillar_w, pillar_h, cfg, ts_ms=1000)
    assert hit is not None, "按柱心锚定取带子应该能框到名字"
    assert hit[0] == "药水"


def test_identify_near_misses_when_name_far_above(bank):
    """名字离柱顶超过 band_up —— 不该硬凑一个结果。"""
    cfg = NameTagConfig(enabled=True, threshold=0.72, band_up=20, band_down=10, band_w=240)
    pillar_cx, pillar_cy, pillar_w, pillar_h = 200, 260, 8, 80
    frame = scene([(140, 120, "POTION")], h=400)   # 远在带子之上
    assert bank.identify_near(frame, pillar_cx, pillar_cy, pillar_w, pillar_h, cfg,
                              ts_ms=1000) is None


def test_identify_near_caches_by_position(bank):
    """同一位置两秒内不重复匹配；缓存也记"什么都没认出"。"""
    cfg = NameTagConfig(enabled=True, threshold=0.72)
    frame = scene([(140, 200, "POTION")], h=400)
    a = bank.identify_near(frame, 200, 260, 8, 80, cfg, ts_ms=1000)
    b = bank.identify_near(frame, 202, 261, 8, 80, cfg, ts_ms=1200)   # ±40px 内命中缓存
    assert a is not None and b is not None
    assert a[0] == b[0]
    assert len(bank._cache) == 1
