"""视觉管道 —— Detector + OCR 组合，产出最终 SceneState。

OCR 白名单过滤：
  - 光柱：OCR 光柱上方物品名区域，命中白名单才保留（whitelisted/label）
  - 右侧列表条目：OCR 条目区域，命中白名单才保留（对话定稿高级方案）
  - 结果缓存：按位置（±40px）缓存 OCR 文本 2s —— 同一物品连续帧不重复 OCR，
    Tesseract 单条 100ms+，2Hz 多目标下无缓存会超决策预算
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ..config import VisionConfig
from ..protocol import SceneState
from .detector import Detector
from .ocr import BaseOCR, NullOCR, fuzzy_contains

log = logging.getLogger(__name__)

_OCR_CACHE_TTL_MS = 2000
_OCR_CACHE_TOL_PX = 40


class VisionPipeline:
    """检测 → OCR 白名单过滤（带位置缓存）。"""

    def __init__(self, cfg: VisionConfig, ocr: BaseOCR | None = None) -> None:
        self.cfg = cfg
        self.detector = Detector(cfg)
        self.ocr = ocr or NullOCR()
        # OCR 缓存: [(x, y, ts_ms, text)]
        self._cache: list[tuple[int, int, int, str]] = []
        # 白名单未配置时明确告知（否则"只捡指定物品"的铁律其实是关着的，
        # 所有光柱都会被当成目标 —— 这是真实部署最容易误解的一点）
        if not cfg.whitelist and not cfg.whitelist_colors:
            log.warning(
                "白名单未配置：所有颜色的光柱都会被视为目标。"
                "只捡指定稀有度请设 vision.whitelist_colors（如 [gold]），"
                "只捡指定物品名请设 vision.whitelist + vision.ocr"
            )

    def _apply_color_whitelist(self, state: SceneState) -> None:
        """稀有度白名单：按光柱颜色簇过滤（不需要 OCR）。"""
        if not self.cfg.whitelist_colors:
            return
        allowed = set(self.cfg.whitelist_colors)
        kept = []
        for p in state.pillars:
            if p.color in allowed:
                p.whitelisted = True
                if not p.label:
                    p.label = p.color
                kept.append(p)
        dropped = len(state.pillars) - len(kept)
        if dropped:
            log.debug("稀有度白名单 %s 过滤掉 %d 个光柱", sorted(allowed), dropped)
        state.pillars = kept

    def _ocr_cached(self, frame_bgr: np.ndarray, x: int, y: int, w: int, h: int,
                    ts_ms: int) -> str:
        """带位置缓存的 OCR：命中未过期缓存则跳过识别。"""
        if not isinstance(self.ocr, NullOCR):
            for cx, cy, ts, text in self._cache:
                if ts_ms - ts <= _OCR_CACHE_TTL_MS \
                        and abs(cx - x) <= _OCR_CACHE_TOL_PX \
                        and abs(cy - y) <= _OCR_CACHE_TOL_PX:
                    return text
        text = self.ocr.read_text(frame_bgr, x, y, w, h)
        if isinstance(self.ocr, NullOCR):
            return text
        self._cache.append((x, y, ts_ms, text))
        if len(self._cache) > 64:
            del self._cache[:-32]
        return text

    def process(self, frame_bgr: np.ndarray, ts_ms: int | None = None) -> SceneState:
        ts_ms = int(time.time() * 1000) if ts_ms is None else ts_ms
        state = self.detector.process(frame_bgr, ts_ms)
        # 缓存清理
        if self._cache:
            self._cache = [c for c in self._cache if ts_ms - c[2] <= _OCR_CACHE_TTL_MS]

        # 稀有度白名单（颜色，无 OCR 也能用）——先过滤，再做 OCR 名称白名单
        self._apply_color_whitelist(state)

        # 白名单过滤（OCR 启用且有白名单时）
        if self.cfg.whitelist and not isinstance(self.ocr, NullOCR):
            kept = []
            for p in state.pillars:
                # OCR 目标区域：光柱附近向上扩展（物品名通常在光柱上方/旁边）
                w = max(160, p.w * 6)
                h = max(28, 40)
                x0 = p.cx - w // 2
                y0 = max(0, p.cy - p.h // 2 - h)
                text = self._ocr_cached(frame_bgr, x0, y0, w, h, ts_ms)
                hit = fuzzy_contains(text, self.cfg.whitelist, self.cfg.whitelist_fuzzy)
                if hit:
                    p.whitelisted = True
                    p.label = hit
                    kept.append(p)
                else:
                    log.debug("光柱未命中白名单 text=%r", text)
            state.pillars = kept

            # 列表条目白名单过滤（条目区域 OCR，命中才保留）
            kept_entries = []
            for e in state.item_list:
                text = self._ocr_cached(frame_bgr, e.x - e.w // 2, e.y - e.h // 2,
                                        e.w, e.h, ts_ms)
                if fuzzy_contains(text, self.cfg.whitelist, self.cfg.whitelist_fuzzy):
                    kept_entries.append(e)
            state.item_list = kept_entries
            state.ocr_used = True
        return state
