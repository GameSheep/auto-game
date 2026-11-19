"""小地图定位 —— 从左上角那一小块，反推人物在整张地图的哪个位置。

对话确认过一件事：小地图是**随人物滚动**的局部视图，屏幕中心另外那张半透明的
是整张地图。所以箭头永远在小地图中心，它只告诉你"屏幕朝哪"，不告诉你"我在哪"。
要回答"我在哪"，只能把小地图这块地形拿去和一张离线底图对齐 —— 这是图像配准
（registration），不是目标检测。

三个让它跑得动的前提（都靠标定一次，之后每帧便宜）：

1. **把底图缩小，而不是把小地图放大。**
   小地图是底图的 1/N（`zoom`），所以先把底图 `zoom` 倍一次缩好缓存；小地图裁片
   过来是 1:1。反过来放大裁片会丢高频、分数虚高，还慢。
2. **有上一帧就只搜附近。**
   人物一帧走不过几百像素，全图搜索是浪费。±`search_radius_px` 内搜，失败才退全图。
3. **灰度 + 边缘两张图，load 时算好。**
   每帧只做一次 matchTemplate，加一次局部 NCC 复核，量级是毫秒。

一条刻意的保守：**不信最高峰**。平滑地形上 TM_CCOEFF_NORMED 会出现大片高分区
（"哪都像"），最高峰经常偏十几到几十像素，而它看上去分数很漂亮。所以取前 K 个峰
逐个用 Sobel 边缘图复核 —— 地形纹理对上了，边缘方向才会对上。这跟 A 框那次
"11/11 误报"是同一个教训：分数是必要条件，不是充分条件。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..config import MinimapConfig
from ..protocol import MapPos
from .imgio import imread_any

log = logging.getLogger(__name__)

_MIN_PATCH = 16  # 比这还小的裁片没有地形信息，匹配出来纯属噪声
_MIN_PATCH_STD = 8.0  # 灰度方差下限：纯色裁片（黑屏/被面板盖住）会让 NCC 处处是 1.0


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _edges(gray8: np.ndarray) -> np.ndarray:
    """Sobel 梯度幅值，模糊一下再归一化到 0..1。

    模糊是必要的：不模糊时边缘是细线，两张图错 1px 就完全不相交，NCC 掉到 0。
    离体后变成"哪里有一条边"，容忍几个像素的配准误差。
    """
    gx = cv2.Sobel(gray8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray8, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.GaussianBlur(mag, (3, 3), 0)
    peak = float(mag.max())
    return (mag / peak).astype(np.float32) if peak > 1e-6 else np.zeros_like(mag, np.float32)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """零均值归一化相关；尺寸不一致或方差为零时返回 0（不猜）。"""
    if a.shape != b.shape:
        return 0.0
    af = a.astype(np.float32).ravel()
    bf = b.astype(np.float32).ravel()
    af = af - af.mean()
    bf = bf - bf.mean()
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    return float(np.dot(af, bf) / denom) if denom > 1e-6 else 0.0


@dataclass
class Locate:
    """一次定位结果。x/y 是底图像素（裁片中心），score/edge 是两条通过线。"""

    x: int
    y: int
    score: float
    edge_score: float
    hinted: bool = False


class MapLocator:
    """纯几何定位器（不依赖配置，可单独测）。

    构造时吃一张底图；之后每帧只吃小地图裁片。`hint` 是上一帧结果（底图坐标），
    给了就只搜附近。
    """

    def __init__(
        self,
        map_image: np.ndarray,
        zoom: float,
        threshold: float = 0.55,
        edge_threshold: float = 0.25,
        search_radius_px: int = 220,
        top_k: int = 3,
        inset: int = 6,
    ) -> None:
        if zoom <= 0:
            raise ValueError("zoom 必须 > 0（小地图像素/底图像素）")
        gray = _gray(map_image)
        self.zoom = float(zoom)
        self.threshold = float(threshold)
        self.edge_threshold = float(edge_threshold)
        self.search_radius_px = int(search_radius_px)
        self.top_k = max(1, int(top_k))
        self.inset = max(0, int(inset))
        self.map_w = int(gray.shape[1])
        self.map_h = int(gray.shape[0])
        # 底图按 zoom 缩一次，之后与小地图裁片 1:1
        sw = max(1, int(round(self.map_w * self.zoom)))
        sh = max(1, int(round(self.map_h * self.zoom)))
        self.map_small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
        self.edge_small = _edges(self.map_small)
        self.last_diag: dict = {}

    # ------------------------------------------------------------------ 内部
    def _peaks(self, res: np.ndarray, patch_w: int, patch_h: int,
               floor: float) -> list[tuple[float, int, int]]:
        """取前 K 个峰，逐个抑制邻域（同一处只留一个，否则前 3 名挤在一起）。"""
        out: list[tuple[float, int, int]] = []
        work = res.copy()
        rw = max(2, patch_w // 2)
        rh = max(2, patch_h // 2)
        for _ in range(self.top_k):
            _, mv, _, ml = cv2.minMaxLoc(work)
            if mv < floor:
                break
            x, y = int(ml[0]), int(ml[1])
            out.append((float(mv), x, y))
            work[max(0, y - rh): y + rh, max(0, x - rw): x + rw] = -1.0
        return out

    def _windows(self, pw: int, ph: int, hint: tuple[int, int] | None):
        """搜索窗口（map_small 坐标）。hinted 的窗口排在前面，命中就不用搜全图。

        hint 是裁片**中心**，而 matchTemplate 的结果坐标是裁片**左上角**，所以窗口
        要按 `tl = hint*zoom - patch/2` 再各向外扩 r，并且宽度至少留一个裁片 ——
        否则算出 2r 宽（r<pw 时）就直接小于裁片、整个窗口被跳过，hint 静默失效。
        """
        sh, sw = self.map_small.shape[:2]
        wins: list[tuple[int, int, int, int, bool]] = []
        if hint is not None and self.search_radius_px > 0:
            r = int(round(self.search_radius_px * self.zoom))
            tlx = int(round(hint[0] * self.zoom)) - pw // 2
            tly = int(round(hint[1] * self.zoom)) - ph // 2
            x0, y0 = max(0, tlx - r), max(0, tly - r)
            x1, y1 = min(sw, tlx + r + pw), min(sh, tly + r + ph)
            if x1 - x0 >= pw and y1 - y0 >= ph:
                wins.append((x0, y0, x1, y1, True))
        wins.append((0, 0, sw, sh, False))
        return wins

    # ------------------------------------------------------------------ 对外
    def locate(self, minimap_bgr: np.ndarray, hint: tuple[int, int] | None = None) -> Locate | None:
        """小地图裁片 → 人物在底图上的位置。认不出返回 None（不返回"最像的那个"）。"""
        gray = _gray(minimap_bgr)
        ins = self.inset
        if ins:
            h, w = gray.shape[:2]
            if h <= ins * 2 or w <= ins * 2:
                return None
            gray = gray[ins: h - ins, ins: w - ins]
        ph, pw = gray.shape[:2]
        sh, sw = self.map_small.shape[:2]
        if ph < _MIN_PATCH or pw < _MIN_PATCH or ph >= sh or pw >= sw:
            self.last_diag = {"why": f"裁片 {pw}x{ph} 与底图 {sw}x{sh} 尺寸不合"}
            return None
        if float(gray.std()) < _MIN_PATCH_STD:
            # 纯色裁片（黑屏、被面板盖住、小地图还没加载）在 TM_CCOEFF_NORMED 下
            # 会给出"处处 1.0"。这不是"定位不准"，是"没有东西可定位"。
            self.last_diag = {"why": f"裁片没有地形信息（std={gray.std():.1f}）"}
            return None

        patch_edge = _edges(gray)
        best_raw = 0.0
        for wx0, wy0, wx1, wy1, hinted in self._windows(pw, ph, hint):
            sub = self.map_small[wy0:wy1, wx0:wx1]
            if sub.shape[0] < ph or sub.shape[1] < pw:
                continue
            res = cv2.matchTemplate(sub, gray, cv2.TM_CCOEFF_NORMED)
            for score, px, py in self._peaks(res, pw, ph, floor=max(0.2, self.threshold - 0.25)):
                best_raw = max(best_raw, score)
                gx, gy = wx0 + px, wy0 + py
                if score < self.threshold:
                    continue
                region = self.edge_small[gy: gy + ph, gx: gx + pw]
                es = _ncc(patch_edge, region)
                if es < self.edge_threshold:
                    log.debug("小地图候选 (%.3f) 边缘复核不过 %.3f，跳过", score, es)
                    continue
                cx = (gx + pw / 2) / self.zoom
                cy = (gy + ph / 2) / self.zoom
                self.last_diag = {"score": round(score, 3), "edge": round(es, 3),
                                  "hinted": hinted, "best_raw": round(best_raw, 3)}
                return Locate(
                    x=int(round(min(max(cx, 0), self.map_w - 1))),
                    y=int(round(min(max(cy, 0), self.map_h - 1))),
                    score=round(float(score), 3),
                    edge_score=round(float(es), 3),
                    hinted=hinted,
                )
        self.last_diag = {"why": "没有候选同时过两条线", "best_raw": round(best_raw, 3),
                          "threshold": self.threshold, "edge_threshold": self.edge_threshold}
        return None


class MinimapLocator:
    """运行时入口：配置 + 帧 → MapPos。负责裁 ROI、维护上一帧位置、拒绝单帧跳变。

    状态全在这里，MapLocator 保持无状态 —— 这样"配准算法对不对"和"该不该采信这一帧"
    是两件事，可以分开测。
    """

    def __init__(self, cfg: MinimapConfig, base_dir: str | Path | None = None) -> None:
        self.cfg = cfg
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.locator: MapLocator | None = None
        self.why = ""          # 为什么不可用，直接给页面/日志看
        self._last: tuple[int, int] | None = None
        self._misses = 0
        self._jumps = 0
        self._build()

    # ------------------------------------------------------------------ 装载
    def _resolve(self, p: str) -> Path:
        path = Path(p)
        if path.is_absolute():
            return path
        for root in (self.base_dir, self.base_dir.parent, self.base_dir.parent.parent):
            cand = root / path
            if cand.exists():
                return cand
        return self.base_dir / path

    def _build(self) -> None:
        cfg = self.cfg
        if not cfg.enabled:
            self.why = "未启用（vision.minimap.enabled=false）"
            return
        if not cfg.map_image:
            self.why = "没配底图（vision.minimap.map_image）"
            return
        if cfg.zoom <= 0:
            self.why = "zoom 未标定（vision.minimap.zoom=0）"
            return
        path = self._resolve(cfg.map_image)
        img = imread_any(path)
        if img is None:
            self.why = f"底图读不出来：{path}"
            log.warning("小地图定位不可用：%s", self.why)
            return
        try:
            self.locator = MapLocator(
                img, zoom=cfg.zoom, threshold=cfg.locate_threshold,
                edge_threshold=cfg.edge_threshold,
                search_radius_px=cfg.search_radius_px, inset=cfg.inset,
            )
        except ValueError as exc:  # zoom 非法
            self.why = str(exc)
            log.warning("小地图定位不可用：%s", exc)
            return
        self.why = ""
        log.info("小地图定位就绪：底图 %dx%d，zoom=%.4f，缩略 %dx%d",
                 self.locator.map_w, self.locator.map_h, cfg.zoom,
                 self.locator.map_small.shape[1], self.locator.map_small.shape[0])

    @property
    def ready(self) -> bool:
        return self.locator is not None

    def reset(self) -> None:
        """切图/换角色时调用：上一帧位置对下一张图没意义，留着只会让它搜错窗口。"""
        self._last = None
        self._misses = 0
        self._jumps = 0

    # ------------------------------------------------------------------ 每帧
    def update(self, frame_bgr: np.ndarray) -> MapPos | None:
        if self.locator is None:
            return None
        roi = self.cfg.roi or [0, 0, 320, 320]
        x, y, w, h = (int(v) for v in roi[:4])
        fh, fw = frame_bgr.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(fw, x + w), min(fh, y + h)
        if x1 - x0 < _MIN_PATCH or y1 - y0 < _MIN_PATCH:
            return None
        crop = frame_bgr[y0:y1, x0:x1]
        loc = self.locator.locate(crop, hint=self._last)

        if loc is None:
            # 单帧认不出不算故障：半透明遮罩/小地图被 UI 挡住都会这样。
            # 连续 3 帧认不出就丢掉 hint —— 否则会一直搜一个早就不在的窗口。
            self._misses += 1
            if self._misses >= 3:
                self._last = None
            return None
        self._misses = 0

        if self._last is not None:
            jump = ((loc.x - self._last[0]) ** 2 + (loc.y - self._last[1]) ** 2) ** 0.5
            if jump > self.cfg.max_jump_px:
                # 一帧走出 max_jump_px 不可能，多半是漂到一块相似地形上。
                # 但真的传送/切图就是这样 —— 所以连续 jump_confirm 帧都这么说就认。
                self._jumps += 1
                self._last = (loc.x, loc.y)
                if self._jumps < self.cfg.jump_confirm:
                    log.debug("小地图跳变 %.0fpx，先不采信（%d/%d）",
                              jump, self._jumps, self.cfg.jump_confirm)
                    return None
            else:
                self._jumps = 0
        self._last = (loc.x, loc.y)
        mw = max(1, self.locator.map_w)
        mh = max(1, self.locator.map_h)
        return MapPos(
            x=loc.x, y=loc.y,
            u=round(loc.x / mw, 4), v=round(loc.y / mh, 4),
            score=loc.score, edge_score=loc.edge_score, hinted=loc.hinted,
        )
