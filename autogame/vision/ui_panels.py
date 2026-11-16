"""UI 面板场景分析 —— 交易行 / 装备 / 天赋 / 技能面板的通用元素检测。

设计（业界 UI bot 标准做法）：面板差异巨大，但**结构元素是通用的**：
  - 网格槽位（交易物品格 / 装备格 / 天赋节点 / 技能槽）→ 等间距矩形网格检测
  - 高亮态（选中 / 可买 / 可点 / 可学）→ 槽位亮度/颜色相对偏差
  - 按钮（购买 / 装备 / 确认 / 卸下）→ 模板匹配（复用 ButtonHit）
  - 文本行（价格 / 物品名 / 词缀）→ 横向亮色行检测（复用 item_list 逻辑）

配色无关：一律按 亮+饱和 判定（稀有度/选中态颜色随物品变化）。
配合 decision/recipe.py 的配方引擎，四类场景作为预置配方运行。
"""

from __future__ import annotations

import cv2
import numpy as np
from pydantic import BaseModel, Field

from ..protocol import ButtonHit


class UiSlot(BaseModel):
    """UI 网格槽位（物品格/天赋节点/技能槽）。"""

    x: int
    y: int           # 中心
    w: int
    h: int
    index: int = 0   # 网格序号（行优先）
    bright: float = 0.0   # 亮度均值（相对高亮判定）
    highlighted: bool = False  # 是否为高亮/选中/可交互态


class UiPanel(BaseModel):
    """一次 UI 面板分析结果。"""

    slots: list[UiSlot] = Field(default_factory=list)
    highlighted: UiSlot | None = None
    buttons: list[ButtonHit] = Field(default_factory=list)
    rows: list[dict] = Field(default_factory=list)  # 文本行 {x,y,w,h}


class GridConfig(BaseModel):
    """网格槽位检测配置（针对具体面板微调）。"""

    roi: list[int]                       # [x,y,w,h] 原分辨率坐标
    cell_min: int = Field(24, ge=6)      # 槽位最小边
    cell_max: int = Field(140, ge=10)    # 槽位最大边
    min_slots: int = Field(4, ge=1)
    max_slots: int = Field(60, ge=1)
    brightness_s_max: int = Field(70)    # 亮度均值的饱和上限（灰度基准）


def detect_grid_slots(frame: np.ndarray, gcfg: GridConfig) -> list[UiSlot]:
    """通用网格槽位检测：ROI 内找等尺寸矩形槽位（按面积聚类 + 行列对齐）。

    算法：亮/饱和掩码 → 轮廓 → 尺寸过滤 → 按尺寸中位数剔除异常 → 行优先排序。
    对交易行物品格、装备格、天赋节点、技能槽通用。
    """
    x0, y0, rw, rh = gcfg.roi
    wh, ww = frame.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    rw, rh = min(rw, ww - x0), min(rh, wh - y0)
    if rw < gcfg.cell_min or rh < gcfg.cell_min:
        return []
    roi = frame[y0 : y0 + rh, x0 : x0 + rw]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # 亮或饱和的内容掩码（槽位内容总是比面板底板亮/艳）
    mask = cv2.inRange(hsv, np.array([0, 40, 70], np.uint8), np.array([179, 255, 255], np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if not (gcfg.cell_min <= w <= gcfg.cell_max and gcfg.cell_min <= h <= gcfg.cell_max):
            continue
        boxes.append((x, y, w, h))
    if len(boxes) < gcfg.min_slots:
        return []

    # 尺寸中位数剔除异常（更大的高亮框/装饰不算槽位）
    ws = sorted(b[2] for b in boxes)
    hs = sorted(b[3] for b in boxes)
    mw, mh = ws[len(ws) // 2], hs[len(hs) // 2]
    slots_raw = [b for b in boxes
                 if 0.6 * mw <= b[2] <= 1.5 * mw and 0.6 * mh <= b[3] <= 1.5 * mh]

    # 行优先排序：按 y 聚类成行，行内按 x 排序
    slots_raw.sort(key=lambda b: (b[1] // max(1, mh), b[0]))
    hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    out: list[UiSlot] = []
    for i, (x, y, w, h) in enumerate(slots_raw[: gcfg.max_slots]):
        cx, cy = x0 + x + w // 2, y0 + y + h // 2
        gx, gy = x0 + x, y0 + y
        cell = hsv_full[gy : gy + h, gx : gx + w]
        bright = float(cell[:, :, 2].mean())
        out.append(UiSlot(x=int(cx), y=int(cy), w=int(w), h=int(h), index=i,
                          bright=round(bright, 1)))
    if out:
        # 高亮槽：亮度显著高于中位数（选中/可点/悬停态更亮）
        meds = float(np.median([s.bright for s in out]))
        for s in out:
            s.highlighted = s.bright >= meds * 1.35 + 10
        # 至少一个高亮：取最亮者
        if not any(s.highlighted for s in out):
            brightest = max(out, key=lambda s: s.bright)
            brightest.highlighted = brightest.bright >= meds * 1.15
    return out


def detect_text_rows(frame: np.ndarray, roi: list[int],
                     min_h: int = 12, max_h: int = 44,
                     width_ratio: float = 0.3, max_rows: int = 10) -> list[dict]:
    """通用文本/价格行检测（横向亮色行），返回 [{x,y,w,h}] 原分辨率坐标。"""
    x0, y0, rw, rh = roi
    wh, ww = frame.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    rw, rh = min(rw, ww - x0), min(rh, wh - y0)
    if rw < 40 or rh < min_h:
        return []
    roi_img = frame[y0 : y0 + rh, x0 : x0 + rw]
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 100, 160], np.uint8),
                       np.array([179, 255, 255], np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rows = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if not (min_h <= h <= max_h) or w < rw * width_ratio:
            continue
        rows.append({"x": x0 + x + w // 2, "y": y0 + y + h // 2,
                     "w": w, "h": h})
    rows.sort(key=lambda r: r["y"])
    return rows[:max_rows]


def match_button(frame: np.ndarray, template: np.ndarray,
                 roi: list[int] | None, threshold: float = 0.72) -> ButtonHit | None:
    """按钮模板匹配（购买/装备/确认/卸下等，复用通用 ButtonHit）。

    roi=None 时全帧搜索（按钮位置未知），半分辨率匹配提速（坐标还原到原分辨率）。
    """
    wh, ww = frame.shape[:2]
    half = False
    if roi:
        rx, ry, rw, rh = roi
        rx, ry = max(0, rx), max(0, ry)
        rw, rh = min(rw, ww - rx), min(rh, wh - ry)
        r = frame[ry : ry + rh, rx : rx + rw]
    else:
        r = frame
        half = True
    if r.shape[0] < 10 or r.shape[1] < 10:
        return None
    tpl = template
    if half:
        r = cv2.resize(r, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        tpl = cv2.resize(tpl, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    th, tw = tpl.shape[:2]
    if th >= r.shape[0] or tw >= r.shape[1]:
        return None
    res = cv2.matchTemplate(r, tpl, cv2.TM_CCOEFF_NORMED)
    _, max_v, _, max_loc = cv2.minMaxLoc(res)
    if max_v < threshold:
        return None
    k = 2.0 if half else 1.0
    if roi:
        cx, cy = rx + int((max_loc[0] + tw / 2) * k), ry + int((max_loc[1] + th / 2) * k)
    else:
        cx, cy = int((max_loc[0] + tw / 2) * k), int((max_loc[1] + th / 2) * k)
    return ButtonHit(x=cx, y=cy, w=int(tw * k), h=int(th * k),
                     score=round(float(max_v), 3))
