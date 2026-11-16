"""关卡内场景分类器 —— 按"必然有 小地图 + 怪物 + 伤害数字"判定。

对话定稿的关卡内判据（用户明确给出）：
  ① 地图：左上角小地图区域有结构化内容（边框+高边缘密度）
  ② 怪物：战斗区域内存在中型饱和色块（怪模 20-160px）
  ③ 伤害数字：战斗区域内多个小型高亮文字块（白/黄伤害数字 4-40px）

三要素齐备 → in_level=True。纯启发式（无模型），逐帧毫秒级。
"""

from __future__ import annotations

import cv2
import numpy as np


def _components(mask: np.ndarray, min_area: int, max_area: int,
                min_w: int, max_w: int, min_h: int, max_h: int) -> list[tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        a = cv2.contourArea(c)
        if not (min_area <= a <= max_area):
            continue
        if not (min_w <= w <= max_w and min_h <= h <= max_h):
            continue
        out.append((x, y, w, h))
    return out


def in_level_features(frame: np.ndarray) -> dict:
    """提取三要素特征（不判定，只量化）。"""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ① 小地图：左上角区域边缘密度 + 有界内容
    mx0, mx1 = int(w * 0.01), int(w * 0.28)
    my0, my1 = int(h * 0.04), int(h * 0.36)
    mini = gray[my0:my1, mx0:mx1]
    edges = cv2.Canny(mini, 60, 150)
    minimap_density = float((edges > 0).mean() * 100)

    # 战斗区域（排除常见 HUD：顶部 8%、底部 18%、左侧 22%、右侧 8%）
    px0, px1 = int(w * 0.24), int(w * 0.92)
    py0, py1 = int(h * 0.10), int(h * 0.80)
    play = hsv[py0:py1, px0:px1]

    # ② 伤害数字：小型高亮文字块（白/黄，V≥200，低-中饱和）
    bright = cv2.inRange(play, np.array([0, 0, 200], np.uint8),
                         np.array([179, 130, 255], np.uint8))
    dmg_components = _components(bright, 12, 900, 4, 70, 5, 40)

    # ③ 怪物：中型饱和色块（20-160px 见方）
    sat_mask = cv2.inRange(play, np.array([0, 110, 70], np.uint8),
                           np.array([179, 255, 255], np.uint8))
    monster_components = _components(sat_mask, 350, 14000, 22, 170, 18, 170)

    # ④ 红血球 HUD 锚点：所有 build 左下固定有红色六边形血球
    #   （背包/全屏面板没有 —— 排除彩色 UI 误报）
    x0, x1 = int(w * 0.28), int(w * 0.45)
    y0, y1 = int(h * 0.80), int(h * 1.0)
    orb_roi = hsv[y0:y1, x0:x1]
    m1 = cv2.inRange(orb_roi, np.array([0, 90, 90], np.uint8), np.array([10, 255, 255], np.uint8))
    m2 = cv2.inRange(orb_roi, np.array([170, 90, 90], np.uint8), np.array([179, 255, 255], np.uint8))
    orb_area = 0
    for c in cv2.findContours(m1 | m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        orb_area = max(orb_area, cv2.contourArea(c))

    return {
        "minimap_density": round(minimap_density, 1),
        "damage_numbers": len(dmg_components),
        "monsters": len(monster_components),
        "hp_orb_area": int(orb_area),
    }


def is_in_level(frame: np.ndarray, *,
                minimap_density_min: float = 3.0,
                damage_numbers_min: int = 2,
                monsters_min: int = 1,
                hp_orb_min: int = 1500) -> tuple[bool, dict]:
    """判定是否关卡内画面（四要素：小地图+怪物+伤害数字+红血球HUD）。"""
    f = in_level_features(frame)
    ok = (f["minimap_density"] >= minimap_density_min
          and f["damage_numbers"] >= damage_numbers_min
          and f["monsters"] >= monsters_min
          and f["hp_orb_area"] >= hp_orb_min)
    f["in_level"] = ok
    return ok, f
