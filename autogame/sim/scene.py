"""合成场景模拟器 —— 无游戏环境下的闭环联调。

模拟 ARPG 关键机制（与对话定稿一致）：
  - 角色固定屏幕中心，世界随角色移动滚动
  - 光柱（金/粉）标记掉落物，可远距离看到
  - 靠近光柱（< pickup_radius）时物品上方出现 "A框" 提示
  - 按拾取键 A：仅当 A框 可见时拾取最近目标；否则视为违规盲按（记录事件）

SimCapture 实现 grab() 接口，可直接注入 AgentClient 闭环运行。
"""

from __future__ import annotations

import logging
import math
import random
import time

import cv2
import numpy as np

log = logging.getLogger(__name__)

# BGR 颜色
_BG = (28, 24, 22)
_GRID = (46, 40, 38)
_GOLD_BGR = (36, 190, 255)      # HSV hue≈26 高饱和高亮
_PINK_BGR = (147, 86, 255)      # HSV hue≈160
_CHAR_BGR = (235, 235, 235)
_HP_BGR = (48, 48, 208)         # 红（BGR）
_MP_BGR = (212, 107, 42)        # #2a6bd4 蓝，`cues.yaml` 里按这个颜色判"蓝够不够"
_VITAL_SLOT = (26, 24, 22)      # 空槽（深灰，低饱和，不会被颜色判据认成有蓝）

#: 模拟器认为的"技能键"：q/e/r（不含 wasd 与拾取键 a）。
#: 不写 qwer 全套是因为 w 在 wasd 方案里是前进 —— 模拟器不能同时把 w 当两种东西。
_SIM_SKILL_KEYS = {"q", "e", "r"}


class SimPillar:
    def __init__(self, wx: float, wy: float, color: str) -> None:
        self.wx = wx
        self.wy = wy
        self.color = color  # gold / pink
        self.id = f"{color}-{id(self) % 10000}"

    def screen_pos(self, char_wx: float, char_wy: float, w: int, h: int) -> tuple[int, int]:
        sx = int(self.wx - char_wx + w / 2)
        sy = int(self.wy - char_wy + h / 2)
        return sx, sy


class SimWorld:
    """虚拟世界：角色世界坐标 + 光柱 + 拾取规则 + 刷图循环（再次挑战）。"""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        walk_speed_px_s: float = 260.0,
        pickup_radius_px: float = 130.0,
        pillar_spawn_interval_s: float = 2.2,
        picks_until_clear: int = 0,  # >0 时启用刷图循环：拾满 N 个后出"再次挑战"
        seed: int | None = None,
    ) -> None:
        self.w = width
        self.h = height
        self.walk_speed = walk_speed_px_s
        self.pickup_radius = pickup_radius_px
        self.spawn_interval = pillar_spawn_interval_s
        self.picks_until_clear = picks_until_clear
        self.rng = random.Random(seed)
        # 角色从世界原点出发
        self.char_wx = 0.0
        self.char_wy = 0.0
        self.waypoint: tuple[float, float] | None = None
        self.pillars: list[SimPillar] = []
        self._last_spawn = time.monotonic()
        # 刷图循环状态
        self.map_cleared = False
        self.map_clears = 0
        self._clear_at: float | None = None  # 结算开始时刻
        # 小地图导航目标（世界坐标），每张图随机一个
        self.objective_world = self._new_objective()
        self.objectives_reached = 0
        # 统计
        self.picked = 0
        self.blind_a_press = 0  # 违规盲按次数（FSM 正确实现应为 0）
        self.picked_ids: list[str] = []
        # 按键状态（键盘走位模式）
        self._keys_down: set[str] = set()
        self._last_update = time.monotonic()
        # 资源条（左下 HUD）。没有它，"蓝够不够"这条判据在陪练里永远不成立，
        # 只能上真机才知道写没写对 —— 模拟器得先长得像真机，才配叫陪练。
        self.hp = 1.0
        self.mana = 1.0
        self.mana_spend = 0.3          # 放一次技能扣多少
        self.mana_regen_per_s = 0.12   # 每秒回多少
        self.skill_presses = 0

    # ------------- 输入事件（执行器干跑日志驱动） -------------

    def on_command(self, op: str, key: str | None = None, button: str | None = None,
                   x: int | None = None, y: int | None = None, hold_ms: int = 0) -> None:
        """把执行器输出注入虚拟世界（闭环）。"""
        now = time.monotonic()
        self._advance(now)
        if op == "mouse_click":
            if x is None or y is None:
                return
            # 刷图循环：结算界面点击『再次挑战』按钮 → 重开
            if self.map_cleared and self._in_rechallenge_button(x, y):
                self._start_next_map(now)
                return
            # 屏幕坐标 → 世界坐标（角色恒在屏幕中心）
            wx = x - self.w / 2 + self.char_wx
            wy = y - self.h / 2 + self.char_wy
            self.waypoint = (wx, wy)
        elif op == "key_tap":
            if key == "a":
                self._try_pickup()
            elif (key or "").lower() in _SIM_SKILL_KEYS:
                # 技能键：扣蓝（模拟器的技能栏就那几格）。走位键 w/a/s/d 不算 ——
                # a 是拾取、w 在 wasd 方案里是前进，扣蓝会把"走位"和"放技能"混一起。
                self.skill_presses += 1
                self.mana = max(0.0, self.mana - self.mana_spend)
            else:
                # 方向键 tap：按 hold 时长移动
                dist = self.walk_speed * max(0.05, hold_ms / 1000.0)
                dx, dy = {
                    "w": (0, -1), "s": (0, 1), "a": (-1, 0), "d": (1, 0),
                }.get((key or "").lower(), (0, 0))
                self.char_wx += dx * dist
                self.char_wy += dy * dist
        elif op == "key_down":
            self._keys_down.add((key or "").lower())
        elif op == "key_up":
            self._keys_down.discard((key or "").lower())

    # ------------- 刷图循环 -------------

    def rechallenge_button_rect(self) -> tuple[int, int, int, int]:
        """按钮屏幕矩形 (x, y, w, h)：屏幕中部居中（避开底部技能栏 HUD）。"""
        bw, bh = 240, 52
        bx = self.w // 2 - bw // 2
        by = int(self.h * 0.42)
        return bx, by, bw, bh

    def _in_rechallenge_button(self, x: int, y: int) -> bool:
        bx, by, bw, bh = self.rechallenge_button_rect()
        return bx <= x <= bx + bw and by <= y <= by + bh

    def _start_next_map(self, now: float) -> None:
        self.map_clears += 1
        self.map_cleared = False
        self._clear_at = None
        # 立即刷怪。多留 1ms：`now - interval` 再被 `now' - _last_spawn` 减回来，
        # 浮点会得到 2.19999999995（差 4.6e-11），`>= interval` 就判成"还没到点"，
        # 表现为"点了『再次挑战』却没刷出光柱"。时间戳判据不要卡在边界上。
        self._last_spawn = now - self.spawn_interval - 1e-3
        self.objective_world = self._new_objective()
        self._objective_hit = False
        log.info("模拟器: 『再次挑战』点击成功 → 第 %d 张图", self.map_clears + 1)

    def _new_objective(self) -> tuple[float, float]:
        angle = self.rng.uniform(0, 2 * math.pi)
        dist = self.rng.uniform(400, 800)
        return (math.cos(angle) * dist, math.sin(angle) * dist)

    def check_objective(self) -> None:
        """走到目标点附近 → 记一次到达（导航闭环验收用）。

        阈值与小地图导航到达圈一致：arrived_ratio(0.12) × 700世界px ≈ 84 → 取 100。
        """
        if math.hypot(self.objective_world[0] - self.char_wx,
                      self.objective_world[1] - self.char_wy) <= 100:
            if not getattr(self, "_objective_hit", False):
                self._objective_hit = True
                self.objectives_reached += 1
                log.info("模拟器: 到达小地图目标点 (累计 %d)", self.objectives_reached)

    def _try_pickup(self) -> None:
        """拾取规则：必须 A框可见（目标在拾取半径内）才捡最近的一个。"""
        near = self._near_pillars()
        if not near:
            # 游戏会捡范围内"最好的垃圾" —— 模拟器记录违规盲按
            self.blind_a_press += 1
            # 打上"当时最近的目标有多远、这一拍世界推进了多久"：盲按要么是
            # 旧帧（证据早过期），要么是世界在两拍之间跳了一大步（步长被放大）。
            # 没有这两项就只能看到"盲按了"，查不出是哪一种。
            nearest = min((math.hypot(p.wx - self.char_wx, p.wy - self.char_wy)
                           for p in self.pillars), default=float("inf"))
            since = time.monotonic() - getattr(self, "_last_abox_at", 0.0)
            log.warning("模拟器: 盲按 A（无近距目标）—— FSM 违规! 最近目标 %.0fpx"
                        "（半径 %.0f）本拍推进 %.2fs，上次画出 A 框在 %.2fs 前"
                        "（距离 %.0fpx）", nearest, self.pickup_radius,
                        getattr(self, "_last_dt", 0.0), since,
                        getattr(self, "_last_abox_d", -1))
            return
        target = near[0]
        self.pillars.remove(target)
        self.picked += 1
        self.picked_ids.append(target.id)
        log.info("模拟器: 拾取 %s (累计 %d)", target.id, self.picked)
        # 拾满 → 地图结算（刷图循环）
        if self.picks_until_clear and self.picked >= self.picks_until_clear * (self.map_clears + 1) \
                and not self.map_cleared:
            self.map_cleared = True
            self._clear_at = time.monotonic()
            self.pillars.clear()
            self.waypoint = None
            log.info("模拟器: 地图结算 → 出现『再次挑战』按钮")

    def _near_pillars(self) -> list[SimPillar]:
        near = []
        for p in self.pillars:
            d = math.hypot(p.wx - self.char_wx, p.wy - self.char_wy)
            if d <= self.pickup_radius:
                near.append((d, p))
        near.sort(key=lambda t: t[0])
        return [p for _, p in near]

    def aframe_visible(self) -> bool:
        return bool(self._near_pillars())

    # ------------- 世界推进 -------------

    def _advance(self, now: float) -> None:
        # 时间戳可能相等（同一 tick 内二次推进 / 时钟回拨）：dt 夹到 [0, 0.5]，
        # **不能直接 return** —— 刷怪/结算都是"时间戳判据"，被吞掉就会表现为
        # "点了再次挑战却没刷出东西"（e2e 刷图循环曾因此失败）。
        dt = max(0.0, min(0.5, now - self._last_update))
        self._last_dt = dt
        self._last_update = max(now, self._last_update)
        # 蓝量自然回复（夹在 [0,1]：不回复会让"放完一轮就再也放不出"永远成立，
        # 那样陪练里看到的是"技能被 CD 锁死"，而不是真实的"蓝不够，等了两秒"）
        self.mana = min(1.0, self.mana + dt * self.mana_regen_per_s)
        # 键盘持续按住移动
        dx = dy = 0
        if "w" in self._keys_down:
            dy -= 1
        if "s" in self._keys_down:
            dy += 1
        if "a" in self._keys_down:
            dx -= 1
        if "d" in self._keys_down:
            dx += 1
        if dx or dy:
            norm = math.hypot(dx, dy)
            self.char_wx += dx / norm * self.walk_speed * dt
            self.char_wy += dy / norm * self.walk_speed * dt
            self.waypoint = None
        # 鼠标寻路
        if self.waypoint is not None:
            wx, wy = self.waypoint
            d = math.hypot(wx - self.char_wx, wy - self.char_wy)
            step = self.walk_speed * dt
            if d <= step or d < 2.0:
                self.char_wx, self.char_wy = wx, wy
                self.waypoint = None
            else:
                self.char_wx += (wx - self.char_wx) / d * step
                self.char_wy += (wy - self.char_wy) / d * step
        # 自动刷光柱（在角色周围 300~700px 环带随机生成）；结算期间不刷
        if not self.map_cleared and now - self._last_spawn >= self.spawn_interval:
            self._last_spawn = now
            placed = False
            for _ in range(8):  # 重roll避开顶部 Boss 血条排除带（y<90 且 x∈[380,900]）
                angle = self.rng.uniform(0, 2 * math.pi)
                dist = self.rng.uniform(300, 700)
                wx = self.char_wx + math.cos(angle) * dist
                wy = self.char_wy + math.sin(angle) * dist
                sx = int(wx - self.char_wx + self.w / 2)
                sy = int(wy - self.char_wy + self.h / 2)
                if 380 <= sx <= 900 and 0 <= sy <= 90:
                    continue
                color = self.rng.choice(["gold", "gold", "pink"])
                self.pillars.append(SimPillar(wx, wy, color))
                placed = True
                break
            if not placed:
                # 兜底：连续 8 次都落在 Boss 血条带里也要刷 —— "重开却刷不出东西"
                # 会让模拟器自己卡死（e2e 刷图循环断言会失败）。角色正下方永远安全。
                self.pillars.append(SimPillar(self.char_wx, self.char_wy + 360, "gold"))
            # 上限控制
            if len(self.pillars) > 8:
                self.pillars.pop(0)

    def tick(self, now: float | None = None) -> None:
        """每帧调用：推进世界。

        `now` 可注入。默认用真实时钟（实机演示、tools/stuck_demo.py 都走这条）；
        测试里给一个**固定步长的合成时钟**，否则"每帧走多远"取决于机器负载：
        慢机器上 dt 顶到 0.5s 的上限，一帧走 130px，正好等于拾取半径，
        角色会一步跨过去 —— 测试于是只在跑全量、CPU 被其他用例占满时才红。
        """
        self._advance(time.monotonic() if now is None else now)
        self.check_objective()

    # ------------- 渲染 -------------

    def render(self) -> np.ndarray:
        frame = np.full((self.h, self.w, 3), _BG, dtype=np.uint8)
        self._draw_world_background(frame)

        # 光柱（含物品名文字 & 近距 A框）
        for p in self.pillars:
            sx, sy = p.screen_pos(self.char_wx, self.char_wy, self.w, self.h)
            if not (-40 <= sx < self.w + 40 and -120 <= sy < self.h + 40):
                continue
            color_bgr = _GOLD_BGR if p.color == "gold" else _PINK_BGR
            # 光柱主体：细高渐变条（高70 宽14），中心白热芯（真实光柱特征）
            top = sy - 110
            cv2.rectangle(frame, (sx - 7, top), (sx + 7, sy), color_bgr, -1)
            cv2.rectangle(frame, (sx - 3, top + 6), (sx + 3, sy - 4), (255, 255, 255), -1)
            glow = tuple(int(c * 0.55) for c in color_bgr)
            cv2.rectangle(frame, (sx - 10, top - 4), (sx + 10, sy + 4), glow, 1)
            # 底部光斑（地面）
            cv2.ellipse(frame, (sx, sy + 2), (22, 9), 0, 0, 360, glow, -1)
            # 物品名（白字，OCR 目标）
            cv2.putText(frame, f"Item {p.id}", (sx - 46, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (250, 250, 250), 1)
            # A框：距离足够近时出现（对话定稿的核心锚点）
            d = math.hypot(p.wx - self.char_wx, p.wy - self.char_wy)
            if d <= self.pickup_radius:
                # 诊断：记下最后一次"真的画了 A 框"的时刻与距离。盲按时用它算
                # "按键落下的那一刻，A 框证据已经过期多久"。
                self._last_abox_at = time.monotonic()
                self._last_abox_d = d
                ax, ay = sx, sy - 150
                cv2.rectangle(frame, (ax - 16, ay - 16), (ax + 16, ay + 16), color_bgr, 2)
                cv2.rectangle(frame, (ax - 14, ay - 14), (ax + 14, ay + 14), (0, 0, 0), -1)
                cv2.rectangle(frame, (ax - 16, ay - 16), (ax + 16, ay + 16), color_bgr, 2)
                # 字形用**白色**：真实游戏的 A 提示就是暗底白字。
                # 若这里画成彩色，`aframe.glyph_white_only: true`（真实部署配置）
                # 会把模拟器自己的 A 框判成"彩色图标"拒掉 —— 模拟器必须长得像真机。
                cv2.putText(frame, "A", (ax - 6, ay + 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 『再次挑战』按钮（地图结算）：黄橙色宽横条，底部居中
        if self.map_cleared:
            bx, by, bw, bh = self.rechallenge_button_rect()
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (60, 200, 255), -1)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (40, 30, 20), 2)
            cv2.putText(frame, "Play Again", (bx + 58, by + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)

        # 小地图（左上角）：暗底 + 白色角色点(中心) + 青色目标标记(随世界滚动)
        self._draw_minimap(frame)
        # 技能栏（底部中央）：战斗 HUD 锚点（场景门依赖）
        self._draw_skillbar(frame)
        # 血/蓝条（左下）：判据的落点（"蓝够不够"要能被读出来）
        self._draw_vitals(frame)

        # 角色（屏幕中心白十字）
        cx, cy = self.w // 2, self.h // 2
        cv2.drawMarker(frame, (cx, cy), _CHAR_BGR, cv2.MARKER_CROSS, 18, 2)
        return frame

    def _draw_skillbar(self, frame: np.ndarray) -> None:
        """底部技能栏：半透明暗条 + 六边形技能槽（模拟真实战斗 HUD）。"""
        bx = int(self.w * 0.30)
        by = int(self.h * 0.83)
        bw, bh = int(self.w * 0.40), int(self.h * 0.15)
        overlay = frame.copy()
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (45, 38, 32), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (70, 62, 55), 2)
        # 6 个技能槽 + 底部键位
        n = 6
        slot = bh // 2
        gap = (bw - n * slot) // (n + 1)
        colors = [(60, 60, 200), (200, 120, 40), (40, 160, 200),
                  (150, 60, 180), (60, 140, 60), (200, 60, 90)]
        for i in range(n):
            sx = bx + gap + i * (slot + gap)
            sy = by + 10
            cv2.circle(frame, (sx + slot // 2, sy + slot // 2), slot // 2,
                       colors[i], 2)
            # 第 5 格标 `RMB` 而不是 `A`：真实游戏这一格是**鼠标右键**，
            # 键盘 A 只用于拾取。以前这里画成 A，看模拟器截图会以为"按 a 放技能"，
            # 进而怀疑 pickup_key=a 定错了（实际没错）——模拟器不该教错这件事。
            key = ["Q", "W", "E", "R", "RMB", "S"][i]
            # 按实测文本宽度居中：以前写死 -5px 是按单字符算的，`RMB` 会偏出格子
            scale = 0.4
            (tw, _th), _base = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            cv2.putText(frame, key, (sx + slot // 2 - tw // 2, by + bh - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (230, 230, 230), 1)

    #: 资源条布局（归一化 [x1,y1,x2,y2]）。绘制和判据标定共用同一组数：
    #: 两处各写一份的话，改一处就"画在那儿、按这儿判"，永远调不准。
    VITAL_HP = (0.03, 0.848, 0.19, 0.874)
    VITAL_MP = (0.03, 0.880, 0.19, 0.906)

    def _draw_vitals(self, frame: np.ndarray) -> None:
        """左下角的血/蓝条（红 HP + 蓝 MP）。"""
        x1, y1, x2, y2 = self.VITAL_HP
        bw = int(self.w * (x2 - x1))
        bh = max(6, int(self.h * (y2 - y1)))
        px1 = int(self.w * x1)
        for i, (val, bgr, tag) in enumerate(((self.hp, _HP_BGR, "HP"),
                                             (self.mana, _MP_BGR, "MP"))):
            py1 = int(self.h * (y1 + i * 0.032))
            cv2.rectangle(frame, (px1, py1), (px1 + bw, py1 + bh), _VITAL_SLOT, -1)
            fill = int(bw * max(0.0, min(1.0, float(val))))
            if fill > 0:
                cv2.rectangle(frame, (px1, py1), (px1 + fill, py1 + bh), bgr, -1)
            cv2.rectangle(frame, (px1, py1), (px1 + bw, py1 + bh), (72, 68, 62), 1)
            # 标签写在条右侧，不压在条上：压在条上会占掉一块纯色，
            # "蓝量占比"的读数就永远到不了 0（空蓝时也剩几个像素）。
            cv2.putText(frame, tag, (px1 + bw + 5, py1 + bh - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (205, 205, 205), 1)

    def vitals_region(self, which: str = "mp") -> list[float]:
        """资源条的归一化矩形 —— 直接抄进 `cues.yaml` 的 `region`。"""
        box = self.VITAL_HP if which == "hp" else self.VITAL_MP
        return [round(v, 4) for v in box]

    def _draw_minimap(self, frame: np.ndarray) -> None:
        size = int(min(self.w, self.h) * 0.25)
        mx, my = int(self.w * 0.03), int(self.h * 0.04)
        half = size // 2
        ccx, ccy = mx + half, my + half
        # 深色底 + 边框（S 低 V 低，不会被任何检测器误认）
        cv2.rectangle(frame, (mx, my), (mx + size, my + size), (18, 16, 15), -1)
        cv2.rectangle(frame, (mx, my), (mx + size, my + size), (60, 55, 50), 2)
        # 目标标记：青色点（HSV hue≈85 高饱和亮），比例尺 = 半径/700世界px
        scale = half / 700.0
        owx, owy = self.objective_world
        mcx = int(ccx + (owx - self.char_wx) * scale)
        mcy = int(ccy + (owy - self.char_wy) * scale)
        if mx < mcx < mx + size and my < mcy < my + size:
            cv2.circle(frame, (mcx, mcy), 6, (255, 255, 0), -1)  # BGR 青色
        # 角色点（小地图中心白色）
        cv2.circle(frame, (ccx, ccy), 4, (250, 250, 250), -1)

    def _draw_world_background(self, frame: np.ndarray) -> None:
        """世界坐标背景：网格 + 纹理块随角色移动滚动（真实感 & 防撞墙帧差信号）。"""
        cell = 200
        gx = int(-self.char_wx) % 80
        gy = int(-self.char_wy) % 80
        for x in range(gx, self.w, 80):
            cv2.line(frame, (x, 0), (x, self.h), _GRID, 1)
        for y in range(gy, self.h, 80):
            cv2.line(frame, (0, y), (self.w, y), _GRID, 1)

        # 视野覆盖的世界网格单元，每个单元按种子固定放几块纹理（滚动）
        wx0 = int(self.char_wx - self.w / 2) // cell - 1
        wx1 = int(self.char_wx + self.w / 2) // cell + 1
        wy0 = int(self.char_wy - self.h / 2) // cell - 1
        wy1 = int(self.char_wy + self.h / 2) // cell + 1
        for cy in range(wy0, wy1 + 1):
            for cx in range(wx0, wx1 + 1):
                rng = random.Random(f"{cx},{cy}")
                for _ in range(rng.randint(2, 3)):
                    bw = rng.randint(60, 170)
                    bh = rng.randint(45, 120)
                    bx = cx * cell + rng.randint(0, cell - bw)
                    by = cy * cell + rng.randint(0, cell - bh)
                    sx = int(bx - self.char_wx + self.w / 2)
                    sy = int(by - self.char_wy + self.h / 2)
                    # 视口裁剪
                    if sx >= self.w or sy >= self.h or sx + bw <= 0 or sy + bh <= 0:
                        continue
                    shade = rng.randint(33, 46)
                    cv2.rectangle(frame, (sx, sy), (sx + bw, sy + bh),
                                  (shade, shade - 4, shade - 6), -1)
                    # 块内斑点纹理（移动时产生明显帧差，模拟地形细节）
                    for _ in range(bw * bh // 120):
                        px = sx + rng.randint(2, max(3, bw - 3))
                        py = sy + rng.randint(2, max(3, bh - 3))
                        if 0 <= px < self.w and 0 <= py < self.h:
                            c = shade - rng.randint(6, 12)
                            frame[py, px] = (c, c, c)


class SimCapture:
    """实现 capture 接口的模拟帧源（注入 AgentClient）。"""

    name = "sim"

    def __init__(self, world: SimWorld) -> None:
        self.world = world

    def grab(self) -> np.ndarray | None:
        self.world.tick()
        return self.world.render()

    def close(self) -> None:
        pass
