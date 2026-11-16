"""视觉检测器 —— 光柱检测 + A框检测。

对应对话定稿的视觉锚点方案：
  - 光柱（金/粉高亮特效）= 掉落物位置 → HSV 阈值分割 + 轮廓筛选（细高条状）。
  - A框 = "距离足够近可以拾取"的 UI 提示 → 模板匹配 / 颜色+形状启发式。
  - 角色固定在屏幕中心（ARPG 通用）。

性能要点：
  - 检测前按 max_detect_width 等比缩放（1Hz 下 CPU 占用极低）。
  - 全程 numpy/OpenCV 向量化，无逐像素 Python 循环。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from ..config import VisionConfig
from ..protocol import AFrame, ButtonHit, ListEntry, MinimapInfo, Monster, Pillar, SceneState

log = logging.getLogger(__name__)

# 仓库默认模板目录（场景门 skillbar.png 随仓库分发）
ROOT_TEMPLATES = Path(__file__).resolve().parent.parent.parent / "templates"


class Detector:
    """检测器（含时域光柱跟踪：跨帧持续出现的候选才确认为光柱）。"""

    def __init__(self, cfg: VisionConfig, templates_dir: Path | None = None) -> None:
        self.cfg = cfg
        self._template: np.ndarray | None = None
        self._template_name: str | None = None
        self._rechallenge_template: np.ndarray | None = None
        self._skillbar_templates: list[np.ndarray] = []
        self._letter_template: np.ndarray | None = None
        self._gate_cache_until: float = 0.0
        self._gate_cache_value: bool = True
        self._last_gray: np.ndarray | None = None
        # 光柱时域跟踪: [(cx, cy, 剩余寿命, 连续可见帧数)]
        self._beam_track: list[tuple[int, int, int, int]] = []
        self._templates_dir = templates_dir
        self._load_template(templates_dir)
        self._load_rechallenge_template(templates_dir)
        self._load_skillbar_template(templates_dir)
        # 键位字母锚点模板（随仓库分发）
        for p in [ROOT_TEMPLATES / "q_letter.png",
                  Path(__file__).resolve().parent.parent.parent / "templates" / "q_letter.png"]:
            if p.exists():
                self._letter_template = cv2.imread(str(p), cv2.IMREAD_COLOR)
                break

    # ---------------- 模板加载 ----------------

    def _load_template(self, templates_dir: Path | None) -> None:
        name = self.cfg.aframe.template
        if not name:
            return
        self._template = self._read_template_img(name, templates_dir)
        if self._template is not None:
            log.info("A框模板已加载: %s", self._template_name)
        else:
            log.warning("A框模板加载失败，回退颜色启发式: %s", name)

    def _load_rechallenge_template(self, templates_dir: Path | None) -> None:
        name = self.cfg.rechallenge.template
        if not name:
            return
        self._rechallenge_template = self._read_template_img(name, templates_dir)
        if self._rechallenge_template is not None:
            log.info("再次挑战按钮模板已加载: %s", self._template_name)

    def _read_template_img(self, name: str, templates_dir: Path | None
                           ) -> np.ndarray | None:
        candidates = [Path(name)]
        if templates_dir:
            candidates.append(templates_dir / name)
        for p in candidates:
            try:
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is not None:
                    self._template_name = str(p)
                    return img
            except Exception:  # noqa: BLE001
                continue
        return None

    def _load_skillbar_template(self, templates_dir: Path | None) -> None:
        if not self.cfg.scene_gate.enabled:
            return
        # 多模板锚点：skillbar*.png 全部加载（不同职业/画质 HUD 外观不同，任意命中即通过）
        import glob as _glob

        paths: list[Path] = []
        base = templates_dir or (ROOT_TEMPLATES if ROOT_TEMPLATES.exists() else None)
        if base:
            for p in sorted(_glob.glob(str(Path(base) / "skillbar*.png"))):
                paths.append(Path(p))
        if not paths:
            log.warning("场景门模板缺失(skillbar*.png) → 门禁停用，in_game 恒为 True")
            return
        self._skillbar_templates = []
        for p in paths:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None:
                self._skillbar_templates.append(img)
        log.info("场景门模板已加载 %d 张", len(self._skillbar_templates))

    # ---------------- 主入口 ----------------

    def process(self, frame_bgr: np.ndarray, ts_ms: int | None = None) -> SceneState:
        """输入 BGR 帧，输出 SceneState。"""
        h0, w0 = frame_bgr.shape[:2]
        ts_ms = int(time.time() * 1000) if ts_ms is None else ts_ms

        # 1) 等比缩放（提升性能 & 归一化坐标空间）
        scale = 1.0
        work = frame_bgr
        max_w = self.cfg.max_detect_width
        if max_w and w0 > max_w:
            scale = max_w / w0
            work = cv2.resize(
                frame_bgr, (max_w, max(1, int(h0 * scale))), interpolation=cv2.INTER_AREA
            )
        wh, ww = work.shape[:2]

        # 2) 场景门：先于 HUD 排除（门要在原始画面上找战斗 HUD）
        in_game = self._check_scene_gate(work, w0, h0, scale)

        # 3) 小地图目标标记（导航输入）—— 在 HUD 排除之前（小地图本身就是要读的 HUD）
        minimap = self._detect_minimap(work, scale) if self.cfg.minimap.enabled else None
        # 3b) 右侧 UI 掉落列表条目（对话定稿的高级方案）
        item_list = (self._detect_item_list(work, scale)
                     if self.cfg.item_list.enabled else [])
        # 3b-2) 地面掉落标签（TLI 类：黄条=掉落物）—— HUD 排除后检测（标签在战斗区域）
        ground_labels = (self._detect_ground_labels(work, scale)
                         if self.cfg.ground_labels.enabled else [])
        # 3b-3) 怪物检测（战斗区域中型饱和色块 —— 主动清怪目标）
        monsters = self._detect_monsters(work, ww, wh, scale)

        # 3c) 加载/黑屏过场判定（暗像素占比，于排除前的原始画面）
        gray_raw = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        loading = bool((gray_raw < self.cfg.loading_dark_v).mean()
                       >= self.cfg.loading_dark_ratio)

        # 4) HUD 排除区置黑（小地图/任务/聊天/技能栏 —— 地面检测不需要这些区域）
        work = self._apply_hud_exclusions(work, w0, h0, scale)

        # 角色中心：屏幕中心（缩放后坐标系）
        char_x, char_y = ww // 2, wh // 2

        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)

        # 5) 帧差（防撞墙）：变化像素占比（0-100），在 1/4 分辨率上计算（提速）
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        gray_q = cv2.resize(gray, (max(1, gray.shape[1] // 4), max(1, gray.shape[0] // 4)),
                            interpolation=cv2.INTER_AREA)
        frame_delta = 0.0
        if self._last_gray is not None and self._last_gray.shape == gray_q.shape:
            diff = cv2.absdiff(gray_q, self._last_gray)
            frame_delta = float((diff > 10).mean() * 100.0)
        self._last_gray = gray_q

        # 6) A框检测（先于光柱，用于过滤 A框边缘误报）
        aframe = self._detect_aframe(work, hsv, scale)

        # 7) 『再次挑战』按钮检测（ROI 内模板/启发式）
        rechallenge = self._detect_rechallenge(work, scale) if self.cfg.rechallenge.enabled \
            else None

        # 8) 光柱检测（白芯校验 + HUD 排除 + 时域确认）
        pillars = self._detect_pillars(hsv, work, ww, wh, char_x, char_y, scale)
        if aframe is not None:
            # A框边缘产生的竖向细条不是光柱：丢弃与 A框重叠的候选
            ax0, ax1 = aframe.x - aframe.w // 2 - 2, aframe.x + aframe.w // 2 + 2
            ay0, ay1 = aframe.y - aframe.h // 2 - 2, aframe.y + aframe.h // 2 + 2
            pillars = [
                p for p in pillars
                if not (ax0 <= p.cx <= ax1 and ay0 <= p.cy <= ay1)
            ]

        return SceneState(
            ts_ms=ts_ms,
            w=w0,
            h=h0,
            character=(char_x, char_y),
            pillars=pillars,
            aframe=aframe,
            rechallenge=rechallenge,
            minimap=minimap,
            item_list=item_list,
            ground_labels=ground_labels,
            monsters=monsters,
            in_game=in_game,
            loading=loading,
            frame_delta=round(frame_delta, 3),
            ocr_used=False,
        )

    # ---------------- 右侧 UI 掉落列表 ----------------

    def _detect_item_list(self, work: np.ndarray, scale: float) -> list[ListEntry]:
        """右侧掉落列表条目检测（对话定稿高级方案）。

        ROI 内找横向亮色条行（物品名/框），行中心=点击坐标。
        """
        il = self.cfg.item_list
        wh, ww = work.shape[:2]
        if il.roi:
            rx, ry, rw, rh = (int(v * scale) for v in il.roi)
        else:
            rx, rw = int(ww * 0.72), int(ww * 0.26)
            ry, rh = int(wh * 0.10), int(wh * 0.60)
        rx, ry = max(0, rx), max(0, ry)
        rw, rh = min(rw, ww - rx), min(rh, wh - ry)
        if rw <= 30 or rh <= 30:
            return []
        roi = work[ry : ry + rh, rx : rx + rw]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv_roi,
            np.array([0, il.bright_s_min, il.bright_v_min], np.uint8),
            np.array([179, 255, 255], np.uint8),
        )
        # 横向膨胀连接成行
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rows: list[tuple[int, int, int, int]] = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if h < il.row_min_height or h > il.row_max_height:
                continue
            if w < rw * il.row_min_width_ratio:
                continue
            rows.append((x, y, w, h))
        rows.sort(key=lambda r: r[1])
        out: list[ListEntry] = []
        for x, y, w, h in rows[: il.max_rows]:
            out.append(
                ListEntry(
                    x=int((rx + x + w / 2) / scale),
                    y=int((ry + y + h / 2) / scale),
                    w=int(w / scale),
                    h=int(h / scale),
                )
            )
        return out

    # ---------------- HUD 排除 / 场景门 ----------------

    def _apply_hud_exclusions(self, work: np.ndarray, w0: int, h0: int,
                              scale: float) -> np.ndarray:
        # 不能原地修改调用方的帧（同一帧会被多次处理）
        work = work.copy()
        # 排除区坐标按 1280x720 基准定义，任意分辨率归一化
        fx, fy = w0 / 1280.0, h0 / 720.0
        for ex in self.cfg.hud_exclusions:
            x, y, w, h = ex
            x0, y0 = int(x * fx * scale), int(y * fy * scale)
            x1 = min(work.shape[1], int((x + w) * fx * scale))
            y1 = min(work.shape[0], int((y + h) * fy * scale))
            if x1 > x0 and y1 > y0:
                work[y0:y1, x0:x1] = 0
        return work

    def _check_scene_gate(self, work: np.ndarray, w0: int, h0: int, scale: float) -> bool:
        """技能栏多模板任一命中 → in_game=True；无模板 → True（门禁停用）。

        性能：ROI 与模板统一降半分辨率匹配（结构锚点精度足够，速度 ×4）。
        """
        templates = self._skillbar_templates
        if not templates:
            return True
        g = self.cfg.scene_gate
        wh, ww = work.shape[:2]
        if g.roi:
            rx, ry, rw, rh = (int(v * scale) for v in g.roi)
        else:
            ry, rh = int(wh * 0.65), wh - int(wh * 0.65)
            rx, rw = 0, ww
        rx, ry = max(0, rx), max(0, ry)
        rw, rh = min(rw, ww - rx), min(rh, wh - ry)
        if rw <= 10 or rh <= 10:
            return False
        roi = cv2.resize(work[ry : ry + rh, rx : rx + rw], None, fx=0.5, fy=0.5,
                         interpolation=cv2.INTER_AREA)
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # 结构锚点用灰度（1/3 计算）
        # 模板按 1280 宽基准制作，任意分辨率视频按宽度比例缩放后再半分辨率匹配
        norm = (w0 / 1280.0) * scale * 0.5
        for tpl in templates:
            th, tw = tpl.shape[:2]
            tw_s, th_s = max(2, int(tw * norm)), max(2, int(th * norm))
            if th_s >= roi.shape[0] or tw_s >= roi.shape[1]:
                continue
            tpl_s = cv2.resize(tpl, (tw_s, th_s), interpolation=cv2.INTER_AREA)
            tpl_gray = cv2.cvtColor(tpl_s, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(roi_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
            _, max_v, _, _ = cv2.minMaxLoc(res)
            if max_v >= g.threshold:
                return True
        # 键位字母锚点（Q/W/E/R 同游戏字体恒定，跨 build/跨画质）
        # 模板可能因 build 技能图标不同而失效 → 红血球结构锚点兜底
        # （红色六边形血球所有 build 固定在左下；模板在即视为命中）
        if self._check_key_letter(work, w0, h0, scale):
            return True
        return self._check_red_orb(work, w0, h0, scale)

    def _check_key_letter(self, work: np.ndarray, w0: int, h0: int, scale: float) -> bool:
        """键位字母 Q 锚点：底部条带灰度匹配（阈值 0.7，实测战斗≥0.788/背包0.638）。"""
        qtpl = self._letter_template
        if qtpl is None:
            return False
        wh, ww = work.shape[:2]
        ry, rh = int(wh * 0.78), wh - int(wh * 0.78)
        strip = work[ry : ry + rh, :]
        s = cv2.resize(strip, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        t = cv2.resize(qtpl, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        g1 = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(g1, g2, cv2.TM_CCOEFF_NORMED)
        return cv2.minMaxLoc(res)[1] >= 0.7

    def _check_red_orb(self, work: np.ndarray, w0: int, h0: int, scale: float) -> bool:
        """红色生命球结构锚点：左下区域的大块红色色斑（build 无关）。"""
        wh, ww = work.shape[:2]
        x0, x1 = int(ww * 0.28), int(ww * 0.45)
        y0, y1 = int(wh * 0.80), int(wh * 1.0)
        if x1 <= x0 or y1 <= y0:
            return False
        roi = work[y0:y1, x0:x1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, 90, 90], np.uint8), np.array([10, 255, 255], np.uint8))
        m2 = cv2.inRange(hsv, np.array([170, 90, 90], np.uint8), np.array([179, 255, 255], np.uint8))
        mask = m1 | m2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) >= 1500:
                return True
        return False

    # ---------------- 光柱 ----------------

    def _detect_pillars(
        self,
        hsv: np.ndarray,
        work: np.ndarray,
        ww: int,
        wh: int,
        char_x: int,
        char_y: int,
        scale: float,
    ) -> list[Pillar]:
        out: list[Pillar] = []
        acfg = self.cfg.aframe
        for color_name, (lo, hi) in self.cfg.hsv.items():
            lo_a = np.array(lo, dtype=np.uint8)
            hi_a = np.array(hi, dtype=np.uint8)
            mask = cv2.inRange(hsv, lo_a, hi_a)
            # 形态学去噪：开运算去掉孤立噪点，闭运算连接光柱断条
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < self.cfg.pillar_min_area:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                if w <= 0:
                    continue
                # 宽度过滤：光柱是窄条；排除 A框边缘细条与大面积色块
                if w < self.cfg.pillar_min_width or w > self.cfg.pillar_max_width:
                    continue
                aspect = h / w
                if aspect < self.cfg.pillar_min_aspect or aspect > self.cfg.pillar_max_aspect:
                    continue
                # 白芯校验：真实掉落光柱有白热芯（低饱和高亮），Boss/技能特效是纯饱和色
                if self.cfg.require_white_core:
                    roi = work[y : y + h, x : x + w]
                    hsv_roi = hsv[y : y + h, x : x + w]
                    core = ((hsv_roi[:, :, 1] <= self.cfg.white_core_s_max)
                            & (hsv_roi[:, :, 2] >= self.cfg.white_core_v_min))
                    if float(core.mean()) < self.cfg.white_core_min_ratio:
                        continue
                cx, cy = x + w // 2, y + h // 2
                bx = int((x + w / 2) / scale)
                by = int((y + h) / scale)  # 底部中心（缩放回原坐标）
                fcx = int(cx / scale)
                fcy = int(cy / scale)
                out.append(
                    Pillar(
                        x=bx,
                        y=by,
                        cx=fcx,
                        cy=fcy,
                        w=int(w / scale),
                        h=int(h / scale),
                        color=color_name,
                        area=int(area),
                        distance=float(np.hypot(fcx - char_x, fcy - char_y)),
                        whitelisted=True,
                        label="",
                    )
                )
        # 去重：同位置不同颜色保留面积大者
        out.sort(key=lambda p: p.area, reverse=True)
        dedup: list[Pillar] = []
        for p in out:
            if all(abs(p.cx - q.cx) > 30 or abs(p.cy - q.cy) > 40 for q in dedup):
                dedup.append(p)
        # 按距离排序（近优先）
        dedup.sort(key=lambda p: p.distance)
        # 时域确认：跨帧持续出现的候选才算真光柱（单帧特效闪烁被滤掉）
        return self._confirm_persistent_beams(dedup[:12], scale)

    def _confirm_persistent_beams(self, candidates: list[Pillar], scale: float = 1.0) -> list[Pillar]:
        """光柱时域跟踪：候选与上一帧轨迹匹配 → 寿命递减到 0 即确认。

        确认后的轨迹保留（继续存在则一直确认）；消失的轨迹移除。
        """
        need = max(1, self.cfg.beam_persistence_frames)
        stable_need = self.cfg.beam_stable_frames
        tol = self.cfg.beam_track_tol_px * scale  # 时间感知容差（config）
        confirmed: list[Pillar] = []
        new_track: list[tuple[int, int, int, int]] = []
        for p in candidates:
            matched = False
            for i, (tx, ty, life, seen) in enumerate(self._beam_track):
                if abs(p.cx - tx) <= tol and abs(p.cy - ty) <= tol:
                    matched = True
                    new_life = life - 1
                    new_seen = seen + 1
                    if new_life <= 0:
                        p.seen_frames = new_seen
                        p.stable = stable_need > 0 and new_seen >= stable_need
                        confirmed.append(p)
                        new_track.append((p.cx, p.cy, 0, new_seen))
                    else:
                        new_track.append((p.cx, p.cy, new_life, new_seen))
                    self._beam_track.pop(i)
                    break
            if not matched:
                new_track.append((p.cx, p.cy, need - 1, 1))
        # 未匹配的旧轨迹自然消亡（不带入新帧）
        self._beam_track = new_track
        return confirmed

    # ---------------- 再次挑战按钮 ----------------

    def _detect_rechallenge(self, work: np.ndarray, scale: float) -> ButtonHit | None:
        rc = self.cfg.rechallenge
        wh, ww = work.shape[:2]
        # ROI（工作坐标系）：默认屏幕中部居中（避开底部技能栏 HUD）
        if rc.roi:
            rx, ry, rw, rh = (int(v * scale) for v in rc.roi)
        else:
            rx, rw = int(ww * 0.25), int(ww * 0.5)
            ry, rh = int(wh * 0.30), int(wh * 0.25)
        rx, ry = max(0, rx), max(0, ry)
        rw, rh = min(rw, ww - rx), min(rh, wh - ry)
        if rw <= 10 or rh <= 10:
            return None
        roi = work[ry : ry + rh, rx : rx + rw]

        # 模式1：多尺度模板匹配
        if self._rechallenge_template is not None:
            best_v, best_loc, best_shape = -1.0, None, self._rechallenge_template.shape[:2]
            for s in rc.template_scales:
                tpl = self._rechallenge_template if abs(s - 1.0) < 1e-6 else cv2.resize(
                    self._rechallenge_template, None, fx=s, fy=s, interpolation=cv2.INTER_AREA
                )
                th, tw = tpl.shape[:2]
                if th >= roi.shape[0] or tw >= roi.shape[1]:
                    continue
                res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
                _, max_v, _, max_loc = cv2.minMaxLoc(res)
                if max_v > best_v:
                    best_v, best_loc, best_shape = max_v, max_loc, (th, tw)
            if best_loc is not None and best_v >= rc.template_threshold:
                th, tw = best_shape
                return ButtonHit(
                    x=int((rx + best_loc[0] + tw / 2) / scale),
                    y=int((ry + best_loc[1] + th / 2) / scale),
                    w=int(tw / scale),
                    h=int(th / scale),
                    score=round(float(best_v), 3),
                )
            return None

        # 模式2：颜色+横条启发式（按钮高亮色宽横条，实心）
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lo = np.array(rc.color_lo, dtype=np.uint8)
        hi = np.array(rc.color_hi, dtype=np.uint8)
        mask = cv2.inRange(hsv_roi, lo, hi)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: ButtonHit | None = None
        for c in contours:
            area = cv2.contourArea(c)
            if area < rc.min_area or area > rc.max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if h <= 0:
                continue
            aspect = w / h
            if aspect < rc.min_aspect or aspect > rc.max_aspect:
                continue
            cand = ButtonHit(
                x=int((rx + x + w / 2) / scale),
                y=int((ry + y + h / 2) / scale),
                w=int(w / scale),
                h=int(h / scale),
                score=round(float(area) / (w * h), 3),
            )
            if best is None or cand.w * cand.h > best.w * best.h:
                best = cand
        return best

    # ---------------- 地面掉落标签 ----------------

    def _detect_ground_labels(self, work: np.ndarray, scale: float) -> list:
        from ..protocol import ListEntry

        gl = self.cfg.ground_labels
        wh, ww = work.shape[:2]
        # 战斗区域（HUD 排除已置黑，这里再取中带）
        rx, rw = int(ww * 0.24), int(ww * 0.68)
        ry, rh = int(wh * 0.10), int(wh * 0.68)
        roi = work[ry : ry + rh, rx : rx + rw]
        if roi.shape[0] < 20 or roi.shape[1] < 40:
            return []
        hsv_r = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_r,
                           np.array([gl.hue_lo, gl.s_min, gl.v_min], np.uint8),
                           np.array([gl.hue_hi, 255, 255], np.uint8))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if not (gl.row_min_h <= h <= gl.row_max_h):
                continue
            if w < gl.row_min_w:
                continue
            # 点击点 = 标签中心下方（地面）
            out.append(ListEntry(
                x=int((rx + x + w / 2) / scale),
                y=int((ry + y + h / 2 + gl.click_below_px) / scale),
                w=int(w / scale), h=int(h / scale)))
        out.sort(key=lambda e: (e.y, e.x))
        return out[: gl.max_labels]

    # ---------------- 怪物检测 ----------------

    def _detect_monsters(self, work: np.ndarray,
                         ww: int, wh: int, scale: float) -> list[Monster]:
        """战斗区域怪物：中型饱和色块（排除 HUD 排除区/光柱/标签重合区）。

        怪物=移动的敌人精灵：面积 400-12000px、宽 20-170、不规则填充。
        配色无关（怪物颜色多样），以饱和+亮度范围取掩码。
        """
        px0, px1 = int(ww * 0.24), int(ww * 0.92)
        py0, py1 = int(wh * 0.10), int(wh * 0.80)
        roi = cv2.cvtColor(work[py0:py1, px0:px1], cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(roi, np.array([0, 100, 80], np.uint8),
                           np.array([179, 255, 255], np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: list[Monster] = []
        for c in contours:
            a = cv2.contourArea(c)
            if not (400 <= a <= 12000):
                continue
            x, y, w, h = cv2.boundingRect(c)
            if not (20 <= w <= 170 and 18 <= h <= 170):
                continue
            fill = a / (w * h) if w * h else 0
            if fill < 0.25 or fill > 0.95:  # 怪物精灵：中密度不规则
                continue
            out.append(Monster(
                x=int((px0 + x + w / 2) / scale),
                y=int((py0 + y + h / 2) / scale),
                w=int(w / scale), h=int(h / scale)))
            if len(out) >= 10:
                break
        return out

    # ---------------- 小地图目标标记 ----------------

    def _detect_minimap(self, work: np.ndarray, scale: float) -> MinimapInfo | None:
        mm = self.cfg.minimap
        wh, ww = work.shape[:2]
        if mm.roi:
            rx, ry, rw, rh = (int(v * scale) for v in mm.roi)
        else:
            rw = rh = int(min(ww, wh) * 0.25)  # 默认左上角方形区域
            rx, ry = int(ww * 0.03), int(wh * 0.04)
        rx, ry = max(0, rx), max(0, ry)
        rw, rh = min(rw, ww - rx), min(rh, wh - ry)
        if rw <= 20 or rh <= 20:
            return None
        roi = work[ry : ry + rh, rx : rx + rw]
        cx, cy = rx + rw // 2, ry + rh // 2
        radius = min(rw, rh) // 2

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lo = np.array(mm.marker_hsv_lo, dtype=np.uint8)
        hi = np.array(mm.marker_hsv_hi, dtype=np.uint8)
        mask = cv2.inRange(hsv_roi, lo, hi)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < mm.marker_min_area or area > mm.marker_max_area:
                continue
            if area > best_area:
                mx, my, mw, mh = cv2.boundingRect(c)
                best = (int((rx + mx + mw / 2) / scale), int((ry + my + mh / 2) / scale))
                best_area = area
        if best is None:
            return MinimapInfo(center=(int(cx / scale), int(cy / scale)),
                               radius=int(radius / scale), marker=None)

        dx = best[0] - cx / scale
        dy = best[1] - cy / scale
        dist = (dx * dx + dy * dy) ** 0.5
        ratio = dist / max(1.0, radius / scale)
        return MinimapInfo(
            center=(int(cx / scale), int(cy / scale)),
            radius=int(radius / scale),
            marker=best,
            angle_deg=round(float(np.degrees(np.arctan2(dy, dx))), 1),
            distance_ratio=round(float(ratio), 3),
            arrived=ratio <= mm.arrived_ratio,
        )

    # ---------------- A框 ----------------

    def _aframe_patch_ok(self, hsv: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
        """结构校验：A框内部必须是**暗底 + 亮字形**。

        真实帧上的伤害数字、技能图标、血条高亮同样是"亮+镂空"，但内部是亮色填充。
        只靠形状会全部误报（实测 100 张真实帧 11/11 误报），必须看内部亮度分布。
        """
        acfg = self.cfg.aframe
        if not acfg.verify_dark_core:
            return True
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(hsv.shape[1], x + w), min(hsv.shape[0], y + h)
        if x1 - x0 < 6 or y1 - y0 < 6:
            return False
        inset = max(1, min(x1 - x0, y1 - y0) // 6)
        inner = hsv[y0 + inset : y1 - inset, x0 + inset : x1 - inset]
        if inner.size == 0:
            return False
        v = inner[..., 2]
        mean_v = float(v.mean())
        bright = v >= acfg.bright_v_min
        if acfg.glyph_white_only:
            # 真实 A 提示是"白字暗底"：白色 = 亮 + 低饱和
            bright = bright & (inner[..., 1] <= acfg.glyph_s_max)
        ratio = float(bright.mean())
        return (mean_v <= acfg.dark_core_v_max
                and acfg.glyph_min_ratio <= ratio <= acfg.glyph_max_ratio)

    def _aframe_template_hit(self, work: np.ndarray, hsv: np.ndarray,
                             scale: float) -> AFrame | None:
        """A框模板匹配 —— 粗到精两级搜索（性能关键）。

        性能实测：17x17 模板直接在 1280x720 全彩图上匹配 = 75ms/尺度，5 个尺度 375ms，
        远超 1Hz 决策预算（端到端 37ms）。因此：
          ① 粗搜：灰度 + 1/2 分辨率 + 3 个尺度（7.2ms/尺度），取前 K 个峰；
          ② 精搜：回到全分辨率，在每个峰附近 ±6px 局部窗口重匹配（<1ms），
             再做"暗底+亮字形"结构校验。
        分辨率选择有实测依据：1/4 分辨率时 4x4 模板噪声太大，真目标（GT 853,455）
        连前 6 峰都进不去；1/2 分辨率时真目标稳定是最高峰（0.864）。
        """
        acfg = self.cfg.aframe
        assert self._template is not None
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        tpl_g = cv2.cvtColor(self._template, cv2.COLOR_BGR2GRAY)
        coarse = 0.5
        small = cv2.resize(gray, None, fx=coarse, fy=coarse, interpolation=cv2.INTER_AREA)

        peaks: list[tuple[float, int, int, float]] = []
        for s in acfg.template_scales:
            tw = max(3, int(round(tpl_g.shape[1] * s * coarse)))
            th = max(3, int(round(tpl_g.shape[0] * s * coarse)))
            if tw >= small.shape[1] or th >= small.shape[0]:
                continue
            tq = cv2.resize(tpl_g, (tw, th), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(small, tq, cv2.TM_CCOEFF_NORMED)
            for _ in range(2):  # 每尺度取前 2 个峰
                _, mv, _, ml = cv2.minMaxLoc(res)
                if mv < 0.30:  # 粗搜门槛放低，由精搜+结构校验把关
                    break
                peaks.append((float(mv), int(ml[0]), int(ml[1]), s))
                y0 = max(0, ml[1] - th // 2)
                x0 = max(0, ml[0] - tw // 2)
                res[y0 : y0 + th, x0 : x0 + tw] = -1.0  # 邻域抑制，避免同一处重复

        peaks.sort(reverse=True)
        for _mv, px, py, s in peaks[:6]:
            tw = max(3, int(round(tpl_g.shape[1] * s)))
            th = max(3, int(round(tpl_g.shape[0] * s)))
            fx, fy = px / coarse, py / coarse
            pad = 6
            x0 = int(max(0, fx - pad))
            y0 = int(max(0, fy - pad))
            x1 = int(min(gray.shape[1], fx + tw + pad))
            y1 = int(min(gray.shape[0], fy + th + pad))
            roi = gray[y0:y1, x0:x1]
            if roi.shape[0] < th or roi.shape[1] < tw:
                continue
            tp = cv2.resize(tpl_g, (tw, th), interpolation=cv2.INTER_AREA)
            r = cv2.matchTemplate(roi, tp, cv2.TM_CCOEFF_NORMED)
            _, rv, _, rl = cv2.minMaxLoc(r)
            if rv < acfg.template_threshold:
                continue
            ax, ay = x0 + rl[0], y0 + rl[1]
            if not self._aframe_patch_ok(hsv, ax, ay, tw, th):
                log.debug("A框模板命中 %.3f 但结构校验未通过（非暗底字形）", rv)
                continue
            return AFrame(
                x=int((ax + tw / 2) / scale),
                y=int((ay + th / 2) / scale),
                w=int(tw / scale),
                h=int(th / scale),
                score=round(float(rv), 3),
            )
        return None

    def _detect_aframe(self, work: np.ndarray, hsv: np.ndarray, scale: float) -> AFrame | None:
        acfg = self.cfg.aframe

        # 模式1：模板匹配（粗到精，见 _aframe_template_hit）
        if self._template is not None:
            hit = self._aframe_template_hit(work, hsv, scale)
            if hit is not None:
                return hit

        # 模式2：形状启发式（A框：亮色框线 + 内部镂空，区别于实心光柱）
        # 色相无关：框色随稀有度变化（金/紫/粉…），按 亮+饱和 判定 + 镂空率排除实心物
        if acfg.hue_agnostic:
            mask = cv2.inRange(
                hsv,
                np.array([0, acfg.bright_s_min, acfg.bright_v_min], dtype=np.uint8),
                np.array([179, 255, 255], dtype=np.uint8),
            )
        else:
            lo = np.array(acfg.color_lo, dtype=np.uint8)
            hi = np.array(acfg.color_hi, dtype=np.uint8)
            mask = cv2.inRange(hsv, lo, hi)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: AFrame | None = None
        for c in contours:
            area = cv2.contourArea(c)
            if area < acfg.min_area or area > acfg.max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if h <= 0:
                continue
            # 尺寸限制（真实 A框是小方框）
            if w < acfg.min_size or h < acfg.min_size or w > acfg.max_size or h > acfg.max_size:
                continue
            aspect = w / h
            if aspect < acfg.min_aspect or aspect > acfg.max_aspect:
                continue
            # 镂空率过滤：外轮廓填充面积中未被高亮像素覆盖的比例。
            # 空心框线 ≈0.5+，实心色块（光柱/光斑）≈0 → 直接排除。
            roi = mask[y : y + h, x : x + w]
            filled = np.zeros(roi.shape, dtype=np.uint8)
            cv2.drawContours(filled, [c - (x, y)], -1, 255, thickness=-1)
            filled_px = int((filled > 0).sum())
            if filled_px <= 0:
                continue
            hole_ratio = 1.0 - float((roi > 0).sum()) / filled_px
            if hole_ratio < acfg.hole_min_ratio:
                continue
            # 结构校验：内部必须是暗底 + 亮字形（否则就是伤害数字/技能图标/血条高亮）
            if not self._aframe_patch_ok(hsv, x, y, w, h):
                continue
            cand = AFrame(
                x=int((x + w / 2) / scale),
                y=int((y + h / 2) / scale),
                w=int(w / scale),
                h=int(h / scale),
                score=round(hole_ratio, 3),
            )
            if best is None or cand.w * cand.h > best.w * best.h:
                best = cand
        return best
