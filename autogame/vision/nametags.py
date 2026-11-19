"""掉落物名词表 —— 不识字，认"长得像"。

对话定稿：物品名是固定词表，那就把每个名字裁一张字条做模板匹配，不走 OCR。

为什么不是 OCR（这不是偷懒，是尺度决定的）：
  手机 1280x720 上，光柱旁边的名字通常只有 10~14px 高，笔画 1~2px。tesseract 的
  中文模型是按 30px 上下训练的，到了这个尺度上误识率很高（"强化石"→"强化右"），
  而认错的代价是**捡错东西**。既然词表固定，"背下来"比"读出来再比对"少一层误差，
  也快一个数量级（几毫秒 vs 100ms+）。

代价说清楚：
  - 新物品上线必须补一张字条，否则**认不出来也不会猜**（返回 None，宁可不捡）。
  - 字体渲染缩放变了（UI 缩放设置、换了分辨率）就要重裁，或者往 `scales` 里加一档。

沿用 A 框那次教训：低方差模板在 TM_CCOEFF_NORMED 下会"哪里都像"（历史上
`q_letter.png` 一块图标角落就对每帧都给 0.88）。所以装载时先算模板灰度标准差，
过低的直接拒收 —— 一张没有笔画的字条不是字条。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from ..config import NameTagConfig
from .imgio import imread_any

log = logging.getLogger(__name__)

_MIN_STD = 8.0        # 灰度标准差下限（与 detector._tpl_usable 同一口径）
_MIN_TPL = 8          # 模板最小边长
_MULTI_SUFFIX = "__"  # 同一物品多张样本：强化石__1.png / 强化石__2.png
_CACHE_TTL_MS = 2000  # 同一位置的光柱不重复匹配（名字不会在两秒内变）
_CACHE_TOL_PX = 40


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


class NameTagBank:
    """字条模板库：文件名（去掉 __N 后缀）= 物品名。"""

    def __init__(self, template_dir: str | Path, threshold: float = 0.72,
                 scales: tuple[float, ...] | list[float] = (1.0,)) -> None:
        self.dir = Path(template_dir)
        self.threshold = float(threshold)
        self.scales = tuple(float(s) for s in scales) or (1.0,)
        # name -> [(scale, gray_template), ...]
        self._tpl: dict[str, list[tuple[float, np.ndarray]]] = {}
        self.skipped: list[tuple[str, str]] = []   # (文件名, 原因)
        self._cache: list[tuple[int, int, float, str]] = []

    # ------------------------------------------------------------------ 装载
    @property
    def names(self) -> list[str]:
        return sorted(self._tpl)

    def load(self) -> int:
        self._tpl.clear()
        self.skipped.clear()
        if not self.dir.is_dir():
            log.warning("字条模板目录不存在：%s", self.dir)
            return 0
        for path in sorted(self.dir.iterdir()):
            if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                continue
            name = path.stem.split(_MULTI_SUFFIX)[0]
            img = imread_any(path)
            if img is None:
                self.skipped.append((path.name, "读不出来"))
                continue
            gray = _gray(img)
            if min(gray.shape[:2]) < _MIN_TPL:
                self.skipped.append((path.name, f"太小 {gray.shape[1]}x{gray.shape[0]}"))
                continue
            if float(gray.std()) < _MIN_STD:
                # 没有笔画的图（纯色条/一块背景）会让门禁常开
                self.skipped.append((path.name, f"方差 {gray.std():.1f} < {_MIN_STD}，不是字条"))
                continue
            variants = []
            for s in self.scales:
                if abs(s - 1.0) < 1e-6:
                    variants.append((1.0, gray))
                else:
                    w = max(_MIN_TPL, int(round(gray.shape[1] * s)))
                    h = max(_MIN_TPL, int(round(gray.shape[0] * s)))
                    variants.append((s, cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)))
            self._tpl.setdefault(name, []).extend(variants)
        if self.skipped:
            for fn, why in self.skipped:
                log.warning("字条 %s 未装载：%s", fn, why)
        log.info("词表装载 %d 条（%s）", len(self._tpl), self.dir)
        return len(self._tpl)

    # ------------------------------------------------------------------ 匹配
    def identify(self, frame_bgr: np.ndarray, roi: tuple[int, int, int, int]) -> tuple[str, float] | None:
        """在 roi=[x,y,w,h] 里找最像的一个词。没有过线的返回 None。"""
        if not self._tpl:
            return None
        x, y, w, h = (int(v) for v in roi)
        fh, fw = frame_bgr.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(fw, x + w), min(fh, y + h)
        if x1 - x0 < _MIN_TPL or y1 - y0 < _MIN_TPL:
            return None
        patch = _gray(frame_bgr[y0:y1, x0:x1])

        best: tuple[str, float] | None = None
        for name, variants in self._tpl.items():
            name_best = 0.0
            for _scale, tpl in variants:
                if tpl.shape[0] > patch.shape[0] or tpl.shape[1] > patch.shape[1]:
                    continue  # 字条比搜索区还大，说明 ROI 给错了，跳过而不是乱缩
                res = cv2.matchTemplate(patch, tpl, cv2.TM_CCOEFF_NORMED)
                _, mv, _, _ = cv2.minMaxLoc(res)
                name_best = max(name_best, float(mv))
            if name_best < self.threshold:
                continue
            if best is None or name_best > best[1]:
                best = (name, round(name_best, 3))
        return best

    def identify_near(self, frame_bgr: np.ndarray, cx: int, cy: int,
                      pillar_w: int, pillar_h: int, cfg: NameTagConfig,
                      ts_ms: int | None = None) -> tuple[str, float] | None:
        """以光柱为基准取一条横带，在其中找名字。

        名字通常在柱**上方**（band_up），少数游戏压在柱身上（band_down）。
        带子以柱的横轴中心为准 —— 名字和柱子是同一个物体的两个部分，锚在柱上比
        锚在屏幕坐标上稳（人物走动时柱子会动，名字跟着动）。

        `cx/cy` 是光柱外接框中心（`Pillar.cx/cy`），不是左上角。
        """
        ts_ms = int(time.time() * 1000) if ts_ms is None else ts_ms
        x = int(cx) - cfg.band_w // 2
        y = int(cy - pillar_h // 2 - cfg.band_up)
        w = cfg.band_w
        h = cfg.band_up + cfg.band_down + pillar_h

        if self._tpl:
            for px, py, ts, text in self._cache:
                if ts_ms - ts <= _CACHE_TTL_MS and abs(px - cx) <= _CACHE_TOL_PX \
                        and abs(py - y) <= _CACHE_TOL_PX:
                    # 缓存的是"这一带刚才认出了什么"，含"什么都没认出"（空串）
                    return None if not text else (text, 1.0)
        hit = self.identify(frame_bgr, (x, y, w, h))
        self._cache.append((cx, y, ts_ms, hit[0] if hit else ""))
        if len(self._cache) > 128:
            del self._cache[:-64]
        return hit
