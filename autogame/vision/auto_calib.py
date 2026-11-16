"""真实图像自适应标定 —— 无人工干预从实机截图推断光柱 HSV 范围。

实机画面调色板与 config.yaml 默认值不同（不同游戏/场景/滤镜），
本模块从图像本身推断"高亮饱和色簇"（掉落光柱特征），产出可直接喂给
Detector.hsv 的范围列表。行业做法：圆形统计的色相聚类 + 分位数边界。
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import VisionConfig


def auto_hsv_ranges(
    frame_bgr: np.ndarray,
    max_ranges: int = 3,
    v_min: int = 170,
    s_min: int = 110,
    percentile: float = 8.0,
    margin: float = 0.12,
) -> list[list[list[int]]]:
    """从图像推断高亮饱和色簇的 HSV 范围（config.yaml hsv 段格式）。

    - 只统计 V>=v_min 且 S>=s_min 的像素（光柱/高亮特效特征）
    - 色相是环形量：按向量均值聚簇（1° 步长直方图 + 平滑峰值）
    - 每个簇用百分位 + margin 外扩生成 [lo,hi] 区间（S/V 全局共享分位数）
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 2] >= v_min) & (hsv[:, :, 1] >= s_min)
    pts = hsv[mask]
    if len(pts) < 200:
        return []

    h_deg = pts[:, 0].astype(np.float64) * 2.0  # 0-360
    # 圆形直方图（36 bins，权重按饱和度*亮度，突出真正的"特效色"）
    weights = (pts[:, 1].astype(np.float64) / 255.0) * (pts[:, 2].astype(np.float64) / 255.0)
    hist, edges = np.histogram(h_deg, bins=36, range=(0, 360), weights=weights)
    # 环形平滑（宽 3 bins）
    kernel = np.array([0.25, 0.5, 0.25])
    hist = np.convolve(np.concatenate([hist[-1:], hist, hist[:1]]), kernel, mode="valid")
    total = hist.sum()
    if total <= 0:
        return []

    # 取峰值簇（抑制已选峰 ±25° 后重复取，最多 max_ranges 个）
    order = np.argsort(hist)[::-1]
    chosen_centers: list[float] = []
    for idx in order:
        center = (edges[idx] + edges[idx + 1]) / 2.0
        if hist[idx] < total * 0.05:  # 忽略 <5% 的小簇
            break
        if any(abs((center - c + 180) % 360 - 180) < 25 for c in chosen_centers):
            continue
        chosen_centers.append(center)
        if len(chosen_centers) >= max_ranges:
            break

    # S/V 分位数（全局）
    s_lo, s_hi = np.percentile(pts[:, 1], [percentile, 100 - percentile])
    v_lo, v_hi = np.percentile(pts[:, 2], [percentile, 100 - percentile])
    s_lo_i = int(max(0, s_lo * (1 - margin)))
    s_hi_i = int(min(255, s_hi * (1 + margin)))
    v_lo_i = int(max(0, v_lo * (1 - margin)))
    v_hi_i = 255

    ranges: list[list[list[int]]] = []
    for center in chosen_centers:
        rel = ((h_deg - center + 180) % 360) - 180
        cluster = np.abs(rel) <= 30  # 峰值 ±30° 视为同簇
        if not cluster.any():
            continue
        c_h = h_deg[cluster]
        rel_c = ((c_h - center + 180) % 360) - 180
        lo_rel, hi_rel = np.percentile(rel_c, [percentile, 100 - percentile])
        lo_h_deg = (center + lo_rel) % 360
        hi_h_deg = (center + hi_rel) % 360
        h_lo = int(max(0, (lo_h_deg / 2) * (1 - margin)))
        h_hi = int(min(179, (hi_h_deg / 2) * (1 + margin)))
        if h_hi < h_lo:  # 环界分裂：扩为覆盖区间
            h_lo, h_hi = h_lo, 179
        ranges.append([[h_lo, s_lo_i, v_lo_i], [h_hi, s_hi_i, v_hi_i]])
    return ranges


def auto_vision_config(frame_bgr: np.ndarray, base: VisionConfig | None = None,
                       keep_defaults: bool = True) -> VisionConfig:
    """以 base 配置为底，将 hsv 替换为图像自适应推断的范围。"""
    cfg = (base or VisionConfig()).model_copy(deep=True)
    ranges = auto_hsv_ranges(frame_bgr)
    if ranges:
        cfg.hsv = {f"auto{i}": r for i, r in enumerate(ranges)}
    elif not keep_defaults:
        cfg.hsv = {}
    return cfg
