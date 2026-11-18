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

import glob as _glob
import json
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


def _box_overlap(cx1: float, cy1: float, w1: float, h1: float,
                 cx2: float, cy2: float, w2: float, h2: float) -> bool:
    """两个"中心点+宽高"的框是否重叠（用于怪物 vs 文字标签互斥）。"""
    return (abs(cx1 - cx2) * 2 < (w1 + w2)) and (abs(cy1 - cy2) * 2 < (h1 + h2))


def _tpl_usable(tpl: np.ndarray) -> bool:
    """模板是否可用于 NCC 匹配：需要真实结构（方差）且不是一条线。

    TM_CCOEFF_NORMED 在低方差模板上会退化成"哪里都像"，历史上 q_letter.png
    （一块图标角落）就因此对每个画面都给出 0.88，等于门禁常开。
    """
    if min(tpl.shape[:2]) < 6:
        return False
    gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY) if tpl.ndim == 3 else tpl
    return float(gray.std()) >= 8.0



class Detector:
    """检测器（含时域光柱跟踪：跨帧持续出现的候选才确认为光柱）。"""

    # 怪物时域确认参数（真实录像验证后定：静态 HUD 图标/悬停物品名必须被滤掉）
    MONSTER_NEAR_PX = 45        # 跨帧同一目标的中心位移容差
    MONSTER_MOVE_PX = 2         # 判定"动了"的最小位移（精灵动画/走位都超过）
    MONSTER_SIZE_TOL = 0.35     # 尺寸变化超过该比例也算"活了"（挥击/受击缩放）
    MONSTER_CONFIRM_HITS = 1    # 首次确认所需命中次数（0 → 需两帧都见过）

    def __init__(self, cfg: VisionConfig, templates_dir: Path | None = None) -> None:
        self.cfg = cfg
        self._template: np.ndarray | None = None
        self._template_name: str | None = None
        self._rechallenge_template: np.ndarray | None = None
        # 正向锚点模板：[(文件名, 图像, 裁制基准帧宽)]
        self._skillbar_templates: list[tuple[str, np.ndarray, int]] = []
        # 反例模板（整屏 UI 的静态裁块）：[(名字, 图像, 裁制基准帧宽)]
        self._neg_templates: list[tuple[str, np.ndarray, int]] = []
        self._tpl_index: dict[str, dict] = {}
        self._gate_cache_until: float = 0.0
        self._gate_cache_value: bool = True
        # 标定诊断：最近一次检测的关键分数（页面"相机标定"用来给反馈，不参与决策）
        self.diag: dict = {}
        self._diag_aframe_best: float = 0.0
        self._diag_gate_best: float = 0.0
        self._diag_gate_neg_best: float = 0.0
        self._diag_gate_neg_name: str = ""
        self._diag_gate_pos_name: str = ""
        self._diag_gate_reason: str = ""
        self._last_gray: np.ndarray | None = None
        # 光柱时域跟踪: [(cx, cy, 剩余寿命, 连续可见帧数)]
        self._beam_track: list[tuple[int, int, int, int]] = []
        # 怪物时域确认: [(x, y, w, h, 命中次数)] —— 静态 HUD 图标/漂浮文字永不累加
        self._monster_track: list[tuple[int, int, int, int, int]] = []
        # 模板目录优先级：显式传参 > 配置里的 vision.templates_dir（profile 注入）
        # > 仓库根 templates/。换游戏必须换目录，否则会拿另一款游戏的 HUD 当锚点。
        if templates_dir is not None:
            self._templates_dir: Path | None = Path(templates_dir)
        elif getattr(cfg, "templates_dir", None):
            self._templates_dir = Path(cfg.templates_dir)
        else:
            self._templates_dir = None
        tdir = self._templates_dir
        self._tpl_index = self._load_template_index(tdir)
        self._load_template(tdir)
        self._load_rechallenge_template(tdir)
        self._load_skillbar_template(tdir)
        self._load_negative_templates(tdir)

    # ---------------- 模板加载 ----------------

    def _load_template_index(self, templates_dir: Path | None) -> dict[str, dict]:
        """读模板目录下的 _index.json：记录每张模板"是按多宽的帧裁的"。

        默认路径（tools/make_template.py）会把裁块统一缩放存成 1280 宽，此时
        base_width 恒为 1280，这里有没有记录都一样。之所以还要支持逐张覆盖：
        手工裁的原生尺寸模板（比如直接从 960 录屏截一块没缩放）必须声明真实基准，
        否则会被按 1280 反推缩放，分数直接掉到噪声水平。

        注意区分两件事——**跨分辨率不通用**和**跨游戏不通用**是两回事：
        模板按 1280 归一化后，同一游戏换分辨率基本还能用（0.9→0.7 量级）；
        换游戏则经常 0.25 分，那不是缩放错了，是两块 HUD 本来就长得不一样。
        """
        base = templates_dir or (ROOT_TEMPLATES if ROOT_TEMPLATES.exists() else None)
        if not base:
            return {}
        p = Path(base) / "_index.json"
        if not p.exists():
            return {}
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            log.warning("模板索引解析失败（已忽略）：%s", p, exc_info=True)
            return {}
        out: dict[str, dict] = {}
        for k, v in (data or {}).items():
            if isinstance(v, dict):
                out[Path(k).name] = v
            elif isinstance(v, (int, float)):
                out[Path(k).name] = {"base_width": int(v)}
        return out

    def _base_width(self, name: str) -> int:
        """该模板的裁制基准帧宽：单张覆盖 > 配置默认。"""
        entry = self._tpl_index.get(Path(name).name) or {}
        try:
            w = int(entry.get("base_width") or 0)
        except (TypeError, ValueError):
            w = 0
        return w if w >= 320 else int(self.cfg.scene_gate.template_base_width)

    def _collect_templates(self, templates_dir: Path | None,
                           pattern: str, legacy: str = "") -> list[Path]:
        """按通配收集模板；pattern 无命中时回退 legacy 通配（兼容老命名）。"""
        base = templates_dir or (ROOT_TEMPLATES if ROOT_TEMPLATES.exists() else None)
        if not base:
            return []
        found = sorted(Path(p) for p in _glob.glob(str(Path(base) / pattern)))
        if not found and legacy:
            found = sorted(Path(p) for p in _glob.glob(str(Path(base) / legacy)))
        return found
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
        # 先判存在再读：cv2.imread 对不存在的路径会往 stderr 打 WARN，搅乱日志
        candidates: list[Path] = []
        p0 = Path(name)
        if p0.is_file():
            candidates.append(p0)
        if templates_dir:
            p1 = Path(templates_dir) / name
            if p1.is_file() and (not p0.is_file() or p1.resolve() != p0.resolve()):
                candidates.append(p1)
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
        g = self.cfg.scene_gate
        # 多锚点：不同游戏/职业/画质 HUD 外观不同，任意一张命中即放行。
        # gate_pos_*.png 是标准名；skillbar*.png 是历史命名，继续认。
        paths = self._collect_templates(templates_dir, g.positive_glob, "skillbar*.png")
        entries: list[tuple[str, Path | None]] = [(p.name, p) for p in paths]
        # 老配置里显式指定的单张模板也要算进去（不存在就静默跳过：
        # 默认值 skillbar.png 指的是仓库根 templates/，profile 目录里没有它很正常）
        if g.template and Path(g.template).name not in {n for n, _ in entries}:
            exists = Path(g.template).is_file() or (
                templates_dir is not None and (Path(templates_dir) / g.template).is_file())
            if exists:
                entries.append((Path(g.template).name, None))
        if not entries:
            log.warning("场景门正向模板缺失（%s）→ 未标定，按 scene_gate.fail_closed=%s 处理"
                        "（True=全部拦下，生产 0 指令；这是安全默认）",
                        g.positive_glob, g.fail_closed)
            return
        self._skillbar_templates = []
        for name, p in entries:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR) if p is not None else None
            if img is None:
                img = self._read_template_img(name, templates_dir)
            if img is None:
                log.warning("正向模板缺失: %s", name)
                continue
            if not _tpl_usable(img):
                log.warning("正向模板方差过低已跳过（NCC 无意义）: %s", name)
                continue
            self._skillbar_templates.append((name, img, self._base_width(name)))
        log.info("场景门正向模板已加载 %d 张: %s", len(self._skillbar_templates),
                 ", ".join(f"{n}(base {b})" for n, _, b in self._skillbar_templates))

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
        # 先抹掉 HUD 排除区：移动端技能轮盘/背包键就压在右侧战斗区里，不抹掉会被
        # 当成怪物点掉（真实录像验证发现的误报源）；再交给时域确认滤静态图标。
        work_ground = self._apply_hud_exclusions(work, w0, h0, scale)
        monsters = self._detect_monsters(work_ground, ww, wh, scale,
                                         avoid=[*ground_labels, *item_list])

        # 3c) 加载/黑屏过场判定（暗像素占比，于排除前的原始画面）
        gray_raw = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        loading = bool((gray_raw < self.cfg.loading_dark_v).mean()
                       >= self.cfg.loading_dark_ratio)

        # 4) HUD 排除区置黑（小地图/任务/聊天/技能栏 —— 地面检测不需要这些区域）
        # 复用上面为怪物检测算好的副本，避免同一帧再复制一次
        work = work_ground

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

        # 怪物 vs 光柱互斥：光柱是"地上的光"，从它身上点一下等于点空地
        if monsters and pillars:
            monsters = [
                m for m in monsters
                if not any(_box_overlap(m.x, m.y, m.w, m.h, p.cx, p.cy, p.w, p.h)
                           for p in pillars)
            ]

        # 标定诊断（页面"相机标定"用）：即使没检出也要给出"差多少"，否则用户
        # 只能看到"检不到"，不知道该调亮度、重裁模板还是挪相机。
        self.diag = {
            "aframe_best": self._diag_aframe_best,
            "aframe_threshold": float(self.cfg.aframe.template_threshold),
            "gate_best": self._diag_gate_best,
            "gate_threshold": float(self.cfg.scene_gate.threshold),
            "gate_reason": self._diag_gate_reason,
            "gate_neg_best": self._diag_gate_neg_best,
            "gate_neg_name": self._diag_gate_neg_name,
            "gate_pos_name": self._diag_gate_pos_name,
            "gate_neg_threshold": float(self.cfg.scene_gate.negative_threshold),
            "scale": round(float(scale), 3),
            "detect_w": int(w0),
            "detect_h": int(h0),
        }
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
            gate_reason=self._diag_gate_reason,
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

    def _ref_scale(self, w0: int, h0: int) -> tuple[float, float]:
        """基准分辨率 → 实际帧的缩放系数（坐标统一按配置的基准书写）。

        换游戏/换录屏分辨率时只改 vision.reference_resolution 一处，不用把
        hud_exclusions、各 ROI 全部重算一遍——重算过一次，漏掉一个框就会把
        HUD 里的图标当掉落物。
        """
        ref = self.cfg.reference_resolution or [1280, 720]
        rw = float(ref[0]) if len(ref) > 0 and ref[0] else 1280.0
        rh = float(ref[1]) if len(ref) > 1 and ref[1] else 720.0
        return w0 / rw, h0 / rh

    def _apply_hud_exclusions(self, work: np.ndarray, w0: int, h0: int,
                              scale: float) -> np.ndarray:
        # 不能原地修改调用方的帧（同一帧会被多次处理）
        work = work.copy()
        # 排除区坐标按 reference_resolution 基准定义，按实际帧等比换算
        fx, fy = self._ref_scale(w0, h0)
        for ex in self.cfg.hud_exclusions:
            x, y, w, h = ex
            x0, y0 = int(x * fx * scale), int(y * fy * scale)
            x1 = min(work.shape[1], int((x + w) * fx * scale))
            y1 = min(work.shape[0], int((y + h) * fy * scale))
            if x1 > x0 and y1 > y0:
                work[y0:y1, x0:x1] = 0
        return work

    def _check_scene_gate(self, work: np.ndarray, w0: int, h0: int, scale: float) -> bool:
        """反例优先 → 技能栏模板 → 兜底拦下。

        判定顺序就是优先级：反例能拿出"这就是我标过的某张菜单"的证据，任何正向
        模板都压不过它；两条都拿不出证据时，未标定的画面按 fail_closed 拦下。

        性能：ROI 与模板统一降半分辨率匹配（结构锚点精度足够，速度 ×4）。
        """
        g = self.cfg.scene_gate
        if not g.enabled:
            self._diag_gate_reason = "disabled"
            return True
        # 1) 反例优先：整屏 UI（关卡选择/背包/结算）靠"有没有 HUD"判不出来，
        #    靠"这块 UI 长得像不像我标过的某张菜单"才判得出来。
        neg = self._check_negative_ui(work, w0)
        if neg is not None:
            self._diag_gate_reason = f"negative:{neg}"
            return False

        templates = self._skillbar_templates
        if not templates:
            # 一张正向模板都没有 = 这个游戏没标定过。此时"放行"等于盲操作：
            # 菜单里任何一次点击都可能卖装备/洗点。所以和"没匹配上"一样处理，
            # 由 fail_closed 决定拦不拦。想临时放行就把 fail_closed 关掉。
            self._diag_gate_reason = "no_template"
            return not g.fail_closed
        wh, ww = work.shape[:2]
        rf = self._ref_scale(w0, h0)
        if g.roi:
            rf_x, rf_y = rf
            rx, ry, rw, rh = (int(g.roi[0] * rf_x * scale), int(g.roi[1] * rf_y * scale),
                              int(g.roi[2] * rf_x * scale), int(g.roi[3] * rf_y * scale))
        else:
            ry, rh = int(wh * 0.65), wh - int(wh * 0.65)
            rx, rw = 0, ww
        rx, ry = max(0, rx), max(0, ry)
        rw, rh = min(rw, ww - rx), min(rh, wh - ry)
        if rw <= 10 or rh <= 10:
            self._diag_gate_reason = "roi_too_small"
            return False
        roi = cv2.resize(work[ry : ry + rh, rx : rx + rw], None, fx=0.5, fy=0.5,
                         interpolation=cv2.INTER_AREA)
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # 结构锚点用灰度（1/3 计算）
        # 模板按各自裁制时的基准帧宽缩放：实际帧宽 / 模板基准宽，再乘 work 缩放与降采样
        for tpl_name, tpl, base_w in templates:
            norm = (w0 / float(base_w)) * scale * 0.5
            th, tw = tpl.shape[:2]
            tw_s, th_s = max(2, int(round(tw * norm))), max(2, int(round(th * norm)))
            if th_s >= roi.shape[0] or tw_s >= roi.shape[1]:
                continue
            tpl_s = cv2.resize(tpl, (tw_s, th_s), interpolation=cv2.INTER_AREA)
            tpl_gray = cv2.cvtColor(tpl_s, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(roi_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
            _, max_v, _, _ = cv2.minMaxLoc(res)
            self._diag_gate_best = round(float(max_v), 3)
            if max_v >= g.threshold:
                self._diag_gate_reason = "skillbar"
                self._diag_gate_pos_name = tpl_name
                return True
        # 一个锚点都没命中：未标定的游戏/未知界面 —— 默认拦下（宁可不动手）
        self._diag_gate_reason = "unmatched"
        if not g.fail_closed:
            return True
        return False

    def _load_negative_templates(self, templates_dir: Path | None) -> None:
        """加载反例模板（整屏 UI 的静态裁块）。

        空配置 → 自动加载 gate_neg_*.png，再回退历史命名 neg_*.png。低方差模板直接
        跳过：TM_CCOEFF_NORMED 在近乎纯色/无结构的模板上会给出无意义的高分
        （历史教训：q_letter.png 实际是技能图标的一个角落，对全部画面都给
        0.87~0.88，把场景门变成了常开）。
        """
        g = self.cfg.scene_gate
        if not g.enabled:
            return
        base = templates_dir or (ROOT_TEMPLATES if ROOT_TEMPLATES.exists() else None)
        names: list[str] = list(g.negative_templates)
        if not names and base:
            paths = self._collect_templates(templates_dir, "gate_neg_*.png", "neg_*.png")
            names = [str(p) for p in paths]
        for name in names:
            img = self._read_template_img(name, templates_dir)
            if img is None:
                log.warning("反例模板缺失: %s", name)
                continue
            if not _tpl_usable(img):
                log.warning("反例模板方差过低已跳过（NCC 无意义）: %s", name)
                continue
            self._neg_templates.append(
                (Path(name).name, img, self._base_width(Path(name).name)))
        if self._neg_templates:
            log.info("反例模板已加载 %d 张: %s", len(self._neg_templates),
                     ", ".join(f"{n}(base {b})" for n, _, b in self._neg_templates))

    def _check_negative_ui(self, work: np.ndarray, w0: int) -> str | None:
        """整屏反例匹配：任一命中即返回模板名（命中=这是菜单/面板，不是战斗）。"""
        if not self._neg_templates:
            return None
        g = self.cfg.scene_gate
        full = cv2.resize(work, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        if min(full.shape[:2]) < 12:
            return None
        gray = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY)
        best, best_name = -1.0, ""
        for name, tpl, base_w in self._neg_templates:
            norm = (w0 / float(base_w)) * 0.5
            th, tw = tpl.shape[:2]
            tw_s, th_s = max(2, int(round(tw * norm))), max(2, int(round(th * norm)))
            if th_s >= gray.shape[0] or tw_s >= gray.shape[1]:
                continue
            t = cv2.resize(tpl, (tw_s, th_s), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(gray, cv2.cvtColor(t, cv2.COLOR_BGR2GRAY),
                                    cv2.TM_CCOEFF_NORMED)
            v = float(cv2.minMaxLoc(res)[1])
            if v > best:
                best, best_name = v, name
            if v >= g.negative_threshold:
                self._diag_gate_neg_best = round(v, 3)
                self._diag_gate_neg_name = name
                return name
        self._diag_gate_neg_best = round(best, 3)
        self._diag_gate_neg_name = best_name
        return None

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
                         ww: int, wh: int, scale: float,
                         avoid: list | None = None) -> list[Monster]:
        """战斗区域怪物：中型饱和色块 + **时域确认**（静态 HUD/文字一律不算怪物）。

        真实录像（1080p 暗黑4 / 火炬之光手游）上纯形状启发式的命中几乎全是误报：
        命中的是技能图标、漂浮的物品名、爆炸特效。所以这里加三道过滤：
          1. HUD 排除区在调用前已抹黑（技能轮盘/背包键/聊天框不再是候选）；
          2. 与已检出的掉落标签/列表条目重叠的候选丢弃（文字不是怪）；
          3. 时域确认：上一帧同位置也在，且**位移或尺寸有变化** —— 怪物会走、
             会打、会受击缩放；静态图标和悬停不动的物品名不会累加命中。
        """
        px0, px1 = int(ww * 0.24), int(ww * 0.92)
        py0, py1 = int(wh * 0.10), int(wh * 0.80)
        roi = cv2.cvtColor(work[py0:py1, px0:px1], cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(roi, np.array([0, 100, 80], np.uint8),
                           np.array([179, 255, 255], np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw: list[tuple[int, int, int, int]] = []
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
            cx, cy = int((px0 + x + w / 2) / scale), int((py0 + y + h / 2) / scale)
            bw, bh = int(w / scale), int(h / scale)
            if avoid and any(_box_overlap(cx, cy, bw, bh,
                                          int(o.x), int(o.y), int(o.w), int(o.h))
                             for o in avoid):
                continue  # 与掉落标签/列表条目重叠 → 是文字不是怪
            raw.append((cx, cy, bw, bh))

        # 时域确认：与前一次检测的追踪表按中心距离配对
        track: list[tuple[int, int, int, int, int]] = []
        out: list[Monster] = []
        for (cx, cy, w, h) in raw:
            hits = 0
            best: tuple[float, int, int, int] | None = None
            for (tx, ty, tw, th, thits) in self._monster_track:
                d = ((tx - cx) ** 2 + (ty - cy) ** 2) ** 0.5
                if d <= self.MONSTER_NEAR_PX and (best is None or d < best[0]):
                    best = (d, tw, th, thits)
            if best is not None:
                d, tw, th, thits = best
                size_change = abs(w * h - tw * th) / max(1, tw * th)
                alive = d >= self.MONSTER_MOVE_PX or size_change >= self.MONSTER_SIZE_TOL
                hits = thits + 1 if alive else thits
            track.append((cx, cy, w, h, hits))
            if hits >= self.MONSTER_CONFIRM_HITS and len(out) < 10:
                out.append(Monster(x=cx, y=cy, w=w, h=h))
        self._monster_track = track
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
        best_rv = 0.0
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
            best_rv = max(best_rv, float(rv))
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
        self._diag_aframe_best = round(best_rv, 3)
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
