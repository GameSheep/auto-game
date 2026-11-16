"""决策状态机 —— 拾取闭环 + 自主恢复。

状态流：
  PAUSED → SCANNING → APPROACHING → (A框出现) → 拾取确认 → COOLDOWN → SCANNING
                        │
                        ├─ 走位超时 ──→ 放弃+拉黑 → COOLDOWN → SCANNING
                        └─ 撞墙(帧差低) → 试一次侧向 → 仍卡 → 放弃+拉黑
  任意状态 ──看门狗判定 frozen/stalled──→ RECOVERING（分级恢复阶梯）→ SCANNING

铁律（对话原文要点，代码层面强制）：
  1. 只有当检测到 A框（近距拾取提示）时才按拾取键 —— 绝不盲按。
     自主恢复里的所有动作同样禁止出现拾取键（RecoveryLadder 二次过滤）。
  2. 单目标走位超时自动放弃并拉黑该区域，绝不原地卡死。
  3. 每目标最多按 N 次拾取键，A框不消失即放弃（防误捡循环）。
  4. panic 立即停止一切输出。
  5. 绝不"停在那里"：卡住 → 分级恢复；恢复失败 → 换招循环重试（可降速告警）。
"""

from __future__ import annotations

import enum
import logging
import random
import time
from typing import Callable

from ..config import DecisionConfig
from ..decision.llm_advisor import LLMAdvisor
from ..decision.recovery import ProgressMonitor, RecoveryLadder, StallSignals
from ..decision.vlm import VisionAdvisor
from ..executor import BaseExecutor
from ..protocol import Command, Pillar, SceneState

log = logging.getLogger(__name__)

# 目标匹配容差 & 拉黑半径（像素）
TARGET_MATCH_TOL_PX = 70
BLACKLIST_RADIUS_PX = 90


class State(str, enum.Enum):
    PAUSED = "PAUSED"
    SCANNING = "SCANNING"
    APPROACHING = "APPROACHING"
    COOLDOWN = "COOLDOWN"
    RECOVERING = "RECOVERING"  # 自主恢复中（卡壳看门狗触发）
    PANIC = "PANIC"


class DecisionEngine:
    """事件驱动 FSM：每个视觉帧 tick 一次（默认 1Hz，对话定稿）。"""

    def __init__(self, cfg: DecisionConfig, executor: BaseExecutor,
                 advisor: LLMAdvisor | None = None,
                 on_event: Callable[[dict], None] | None = None,
                 vlm: VisionAdvisor | None = None) -> None:
        self.cfg = cfg
        self.exec = executor
        self.advisor = advisor or LLMAdvisor(cfg.llm)
        self.on_event = on_event  # 事件回调（持久化等），同步调用，需快速返回
        self.state: State = State.PAUSED
        self.events: list[dict] = []  # 事件日志（仪表盘+测试）
        self.stats = {
            "picks": 0,
            "a_taps": 0,
            "abandons": 0,
            "approach_starts": 0,
            "rechallenges": 0,
            "blind_tap_violations": 0,  # 恒为 0 才正确（防御性断言）
            "recoveries": 0,            # 进入自主恢复次数
            "recovery_steps": 0,        # 恢复阶梯执行步数
            "recovery_cycles": 0,       # 完整恢复循环轮数
            "recovery_actions": 0,      # 恢复动作发出的命令数
            "gate_escapes": 0,          # 弹窗/菜单 ESC 次数
            "stuck_alarms": 0,          # 多轮仍失败告警次数
        }
        # 自主性：看门狗 + 恢复阶梯 + 视觉顾问
        self.monitor = ProgressMonitor(cfg.autonomy)
        self.vlm = vlm or VisionAdvisor(cfg.vlm)
        self.frame_source: Callable[[], bytes | None] | None = None
        self.vlm.frame_source = self._current_frame
        self.recovery = RecoveryLadder(
            cfg.autonomy, self.vlm, self.advisor,
            forbidden_keys={cfg.pickup_key},  # 铁律：恢复动作永不按拾取键
        )
        self._recovery_reason = ""
        self._recovery_picks = 0
        self._recovery_targets0 = 0
        self._recovery_ok_ticks = 0
        self._recovery_pause_until = 0.0
        self._recovery_started = 0.0
        self._last_recovery_step = ""
        self._need_monitor_reset = True
        self._now = 0.0
        self._ts_offset_ms = 0
        self._last_ts_ms = 0
        self._gate_escape_ts = 0.0
        self._auto_explore_until = 0.0
        self._last_rechallenge: tuple[int, int] | None = None
        self._rechallenge_first_seen: float | None = None
        self._list_first_seen_ms: int | None = None
        self._label_first_seen_ms: int | None = None
        self._last_nav_ms: int = 0
        self._gate_hold_emitted: bool = False
        self.last_intent: str = ""
        self._explore_i: int = 0
        self._last_explore_ms: int = 0
        self._loading_hold_emitted: bool = False
        self._target: Pillar | None = None
        self._target_started: float = 0.0
        self._last_click_ts: float = 0.0
        self._aframe_first_seen_ms: int | None = None
        self._taps_this_target: int = 0
        self._stuck_ticks: int = 0
        self._alt_direction_tried: bool = False
        self._cooldown_until: float = 0.0
        self._last_tap_ts: float = 0.0
        self._tap_block_until_ms: int = 0
        self._blacklist: list[tuple[int, int, float]] = []  # (x, y, until_frame_ts)

    # ------------- 对外控制 -------------

    def start(self) -> None:
        if self.state == State.PANIC:
            log.warning("PANIC 状态需先 reset_panic")
            return
        self.exec.reset()
        self.state = State.SCANNING
        self._need_monitor_reset = True
        self.recovery.reset()
        self._recovery_reason = ""
        self._recovery_pause_until = 0.0
        self._emit_event("started")
        log.info("FSM 启动 → SCANNING")

    def stop(self) -> None:
        self.state = State.PAUSED
        self._drop_target("stopped")
        self._act(Command(op="stop_all"))
        self._emit_event("stopped")
        log.info("FSM 停止")

    def panic(self) -> None:
        self.state = State.PANIC
        self._drop_target("panic")
        self.exec.panic()
        self._emit_event("panic")
        log.warning("FSM 急停")

    def reset_panic(self) -> None:
        self.exec.reset()
        if self.state == State.PANIC:
            self.state = State.SCANNING
        self._emit_event("panic_reset")

    # ------------- 主 tick -------------

    def on_scene(self, scene: SceneState) -> None:
        """处理一帧场景状态（决策频率 = 帧率，建议 1Hz）。

        所有内部计时使用帧时间戳（scene.ts_ms），保证确定性、可测试。
        场景门：非战斗界面（in_game=False，如背包/菜单）禁止一切输出。
        """
        if self.state in (State.PAUSED, State.PANIC):
            return
        now = self._frame_time(scene.ts_ms)
        self._now = now
        if self._need_monitor_reset:
            self.monitor.reset(now)
            self._need_monitor_reset = False
        signals = self.monitor.update(scene, now, self.stats["picks"])

        # 加载/黑屏过场：保持不动（等加载完成），不产生任何命令
        if scene.loading:
            if not self._loading_hold_emitted:
                self._emit_event("loading_hold")
                self._loading_hold_emitted = True
            self._drop_target("loading")
            return
        self._loading_hold_emitted = False

        if not scene.in_game:
            self._tick_gated(scene, now, signals)
            return
        self._gate_hold_emitted = False

        # 拉黑清理
        if self._blacklist:
            self._blacklist = [b for b in self._blacklist if b[2] > now]

        # ---- 卡壳看门狗：卡住就进恢复，恢复中优先恢复 ----
        if self.cfg.autonomy.enabled:
            if self.state == State.RECOVERING:
                if self._recovery_can_exit(scene, signals):
                    self._exit_recovery(now)
                else:
                    self._tick_recovery(scene, now, signals)
                    return
            elif signals.verdict in ("frozen", "stalled") \
                    and self.state != State.APPROACHING:
                # 走位中不归看门狗管：单目标撞墙/超时由 _tick_approach 自己处理
                # （否则"站在光柱旁边画面静止"会被误判成卡死，来回抖动）
                self._enter_recovery(scene, now, signals)
                return

        if self.state == State.COOLDOWN:
            if now >= self._cooldown_until:
                self.state = State.SCANNING
                self._emit_event("cooldown_done")
            return

        if self.state == State.SCANNING:
            self._tick_scan(scene, now)
        elif self.state == State.APPROACHING:
            self._tick_approach(scene, now)

    # ------------- 自主恢复 -------------

    def _frame_time(self, ts_ms: int) -> float:
        """帧时间戳 → 秒（自动处理代理重启/时钟跳变导致的时间回退）。"""
        ts_ms = int(ts_ms)
        if ts_ms < self._last_ts_ms - 1000:
            # 时间倒流：把偏移重基准，保持内部时间单调，否则所有超时判据失效
            self._ts_offset_ms = ts_ms - self._last_ts_ms + self._ts_offset_ms
            self._emit_event("clock_rebase", from_ms=self._last_ts_ms, to_ms=ts_ms)
        self._last_ts_ms = ts_ms
        return (ts_ms - self._ts_offset_ms) / 1000.0

    def _current_frame(self) -> bytes | None:
        try:
            return self.frame_source() if self.frame_source else None
        except Exception:  # noqa: BLE001
            log.debug("取帧失败（VLM 跳过）", exc_info=True)
            return None

    def _tick_gated(self, scene: SceneState, now: float, signals: StallSignals) -> None:
        """不在战斗界面：短时间等待；长时间（弹窗/菜单/结算）按 ESC 自救。"""
        if not self._gate_hold_emitted:
            self._emit_event("scene_gate_hold")
            self._gate_hold_emitted = True
        self._drop_target("scene_gate")
        if self.state == State.RECOVERING:
            # 恢复中被弹窗挡住：继续按阶梯走（ESC 就在阶梯里）
            self._tick_recovery(scene, now, signals)
            return
        auto = self.cfg.autonomy
        if not auto.enabled or signals.verdict != "gated":
            return
        if self.stats["gate_escapes"] >= 6:
            return
        if now - self._gate_escape_ts < 4.0:
            return
        self._gate_escape_ts = now
        self.stats["gate_escapes"] += 1
        self._act(Command(op="key_tap", key=auto.escape_key, hold_ms=60))
        self._intent("长时间不在战斗界面，按 ESC 关闭弹窗/菜单自救")
        self._emit_event("gate_escape", attempt=self.stats["gate_escapes"],
                         gate_for=round(signals.gate_for, 1))

    def _enter_recovery(self, scene: SceneState, now: float, signals: StallSignals) -> None:
        self.state = State.RECOVERING
        self._recovery_reason = signals.verdict
        self._recovery_picks = self.stats["picks"]
        self._recovery_targets0 = signals.targets
        self._recovery_ok_ticks = 0
        self._recovery_pause_until = 0.0
        self._recovery_started = now
        self.recovery.begin(now)
        self._drop_target("recovery")
        self.stats["recoveries"] += 1
        self._emit_event("recovery_start", reason=signals.verdict, signals=signals.as_dict())
        self._intent(f"自主恢复启动（{signals.verdict}）")
        self._tick_recovery(scene, now, signals)

    def _recovery_can_exit(self, scene: SceneState, signals: StallSignals) -> bool:
        """恢复成功判据（任一即可）：

          1. 拿到新拾取（最硬的证据）；
          2. 出现 A框 —— 说明"现在就能交互"，立刻回到正常拾取流程；
          3. **新增**了可交互目标（卡壳前没有的东西）。

        注意不能只看"有目标"：贴着墙角卡在光柱旁边时目标一直存在，
        若按"有目标=恢复成功"就会在恢复与正常之间来回抖动。
        """
        if self.stats["picks"] > self._recovery_picks:
            return True
        if scene.aframe is not None:
            return True
        if signals.targets > self._recovery_targets0:
            self._recovery_ok_ticks += 1
            return self._recovery_ok_ticks >= 2
        self._recovery_ok_ticks = 0
        return False

    def _exit_recovery(self, now: float) -> None:
        self.state = State.SCANNING
        # 重新计时：刚恢复成功应给一个完整观察窗口，否则会立刻被"停滞"判据再抓回去
        self.monitor.note_progress(now, "recovery_done")
        self._recovery_ok_ticks = 0
        self._last_recovery_step = ""
        self._emit_event("recovery_done", reason=self._recovery_reason,
                         level=self.recovery.level, cycle=self.recovery.cycle,
                         steps=self.recovery.steps_done)
        self._intent("自主恢复成功，回到扫描")
        self.recovery.reset()

    def _tick_recovery(self, scene: SceneState, now: float, signals: StallSignals) -> None:
        if now < self._recovery_pause_until:
            return  # 循环间的降速等待（避免"卡住→疯狂乱按"）
        ctx = {
            "character": scene.character,
            "approach_mode": self.cfg.approach_mode,
            "state": self.state.value,
            "signals": signals.as_dict(),
            "stats": dict(self.stats),
            "recent_events": [e.get("event") for e in self.events[-8:]],
            "on_llm_advice": self._on_llm_advice,
        }
        res = self.recovery.step(now, scene, ctx)

        if res.step == "retarget" and res.commands == []:
            # 内部重定位：清空拉黑与探索序列，下一轮从零开始找目标
            self._blacklist.clear()
            self._explore_i = 0
            self._last_explore_ms = 0

        for cmd in res.commands:
            if self._act(cmd):
                self.stats["recovery_actions"] += 1
        if res.commands:
            self.stats["recovery_steps"] += 1

        # 事件：有动作 / 升级 / 换了手段 都记录（纯内部重定位也要可见）
        if res.commands or res.new_level or res.step != self._last_recovery_step:
            self._emit_event("recovery_step", level=res.level, cycle=res.cycle,
                             step=res.step, note=res.note, commands=len(res.commands))
            self._last_recovery_step = res.step
        if res.note:
            self._intent(f"自主恢复 L{res.level}/{res.step}: {res.note}")

        if res.cycle_done:
            self.stats["recovery_cycles"] += 1
            self._emit_event("recovery_cycle", cycle=res.cycle, reason=self._recovery_reason)
            # 每轮失败后降速：轮数越多等越久（上限 60s），但不停止尝试
            backoff = min(60.0, self.cfg.autonomy.cycle_backoff_s * (1 + res.cycle))
            self._recovery_pause_until = now + backoff
            if res.cycle >= 2:
                self.stats["stuck_alarms"] += 1
                self._emit_event("stuck_alarm", cycle=res.cycle, reason=self._recovery_reason)
                limit = self.cfg.autonomy.pause_after_cycles
                if limit and res.cycle >= limit:
                    log.error("连续 %d 轮自主恢复失败，按配置暂停并等待人工介入", res.cycle)
                    self.stop()

    # ------------- SCANNING -------------

    def _tick_scan(self, scene: SceneState, now: float) -> None:
        target = self._pick_target(scene, now)
        if target is None:
            if not self._maybe_rechallenge(scene, now):
                if not self._maybe_click_list_entry(scene, now):
                    if not self._maybe_approach_ground_label(scene, now):
                        if not self._maybe_attack_monster(scene, now):
                            if not self._maybe_navigate(scene, now):
                                self._maybe_explore(scene, now)
            return
        self._target = target
        self._target_started = now
        self._last_click_ts = 0.0
        self._aframe_first_seen_ms = None
        self._taps_this_target = 0
        self._stuck_ticks = 0
        self._alt_direction_tried = False
        self.stats["approach_starts"] += 1
        self.state = State.APPROACHING
        self._emit_event(
            "approach_start",
            target={"x": target.x, "y": target.y, "color": target.color,
                    "label": target.label, "distance": round(target.distance)},
        )
        # 立即发出第一步走位
        self._move_toward(scene, target, now, first=True)

    def _pick_target(self, scene: SceneState, now: float) -> Pillar | None:
        # 优先取 stable 光柱（掉落物存续长，特效短命）；无 stable 回退全部候选
        candidates = [p for p in scene.pillars
                      if p.whitelisted and not self._is_blacklisted(p, now)]
        if not candidates:
            return None
        stables = [p for p in candidates if p.stable]
        return (stables or candidates)[0]

    def _is_blacklisted(self, p: Pillar, now: float) -> bool:
        for bx, by, until in self._blacklist:
            if until > now and abs(p.cx - bx) <= BLACKLIST_RADIUS_PX \
                    and abs(p.cy - by) <= BLACKLIST_RADIUS_PX:
                return True
        return False

    # ------------- 刷图循环：再次挑战 -------------

    def _maybe_rechallenge(self, scene: SceneState, now: float) -> None:
        """无拾取目标时：检测到『再次挑战』按钮 → 点击 → 重开下一图。

        防抖：按钮需连续可见 ≥200ms（防 UI 过场闪烁误点）。毫秒整数比较避免浮点误差。
        """
        if not self.cfg.rechallenge_enabled:
            return
        btn = scene.rechallenge
        if btn is None:
            self._rechallenge_first_seen = None
            return
        self._last_rechallenge = (btn.x, btn.y)
        if self._rechallenge_first_seen is None:
            self._rechallenge_first_seen = scene.ts_ms
            return
        if scene.ts_ms - self._rechallenge_first_seen < 200:
            return
        self._rechallenge_first_seen = None
        self._intent("点击『再次挑战』进入下一层（通关循环）")
        self.stats["rechallenges"] += 1
        self._act(Command(op="mouse_click", button="left", x=btn.x, y=btn.y))
        self._emit_event("rechallenge_click", x=btn.x, y=btn.y)
        self._cooldown_until = now + max(self.cfg.rechallenge_cooldown_s,
                                         random.uniform(2.5, 3.5))
        self.state = State.COOLDOWN
        return True  # 已处理本 tick

    # ------------- 右侧 UI 掉落列表（对话定稿高级方案） -------------

    def _maybe_click_list_entry(self, scene: SceneState, now: float) -> bool:
        """无地面光柱时：检测到右侧掉落列表条目 → 点击条目（角色走过去拾取）。

        条目需跨帧稳定（复用 _rechallenge 防抖语义）；无 OCR 白名单时全部条目可点。
        """
        entries = scene.item_list
        if not entries:
            self._list_first_seen_ms = None
            return False
        if self._list_first_seen_ms is None:
            self._list_first_seen_ms = scene.ts_ms
            return False
        if scene.ts_ms - self._list_first_seen_ms < 300:  # 连续 ≥300ms 可见才点击
            return False
        self._list_first_seen_ms = None
        e = entries[0]
        self._intent("点击右侧掉落列表条目（走过去拾取）")
        self.stats["list_clicks"] = self.stats.get("list_clicks", 0) + 1
        self._act(Command(op="mouse_click", button="left", x=e.x, y=e.y))
        self._emit_event("list_entry_click", x=e.x, y=e.y, entries=len(entries))
        self._cooldown_until = now + random.uniform(1.0, 1.6)
        self.state = State.COOLDOWN
        return True

    # ------------- 主动清怪 + 探索（通关目标循环） -------------

    def _maybe_attack_monster(self, scene: SceneState, now: float) -> bool:
        """主动清怪：最近的怪物 → 点击（ARPG：点击=移动+攻击）。

        自主模式（autonomy.auto_attack）下即使 attack_monsters 未显式开启也执行 ——
        搬砖机器人的目标是通关刷图，站着不打怪等于卡住。
        """
        auto_attack = self.cfg.attack_monsters or (
            self.cfg.autonomy.enabled and self.cfg.autonomy.auto_attack)
        if not auto_attack or not scene.monsters:
            return False
        ch = scene.character
        m = min(scene.monsters,
                key=lambda mo: (mo.x - ch[0]) ** 2 + (mo.y - ch[1]) ** 2)
        dx, dy = m.x - ch[0], m.y - ch[1]
        self._act(Command(op="mouse_click", button="left", x=m.x, y=m.y))
        self._intent(f"攻击怪物（点击 {m.x},{m.y}，方向{_dir_text(dx, dy)}）")
        self._emit_event("attack_monster", x=m.x, y=m.y)
        self._cooldown_until = now + random.uniform(0.6, 1.0)
        self.state = State.COOLDOWN
        return True

    def _maybe_explore(self, scene: SceneState, now: float) -> bool:
        """探索：无目标时按方向序列移动（通关=走完整个关卡）。

        自主模式（autonomy.auto_explore）默认开启：没有掉落/怪物时自己去找，
        而不是站在原地等刷新 —— 这是"更高自主性"最直接的体现。
        """
        auto = self.cfg.autonomy
        if not (self.cfg.explore_enabled or (auto.enabled and auto.auto_explore)
                or now < self._auto_explore_until):
            return False
        nav_interval_ms = int(1000 / max(0.5, self.cfg.tick_hz))
        now_ms = int(now * 1000)
        if self._last_explore_ms and now_ms - self._last_explore_ms < nav_interval_ms:
            return False
        self._last_explore_ms = now_ms
        dirs = self.cfg.explore_dirs or auto.explore_dirs or ["w", "d", "s", "a"]
        key = dirs[self._explore_i % len(dirs)]
        self._explore_i += 1
        step_ms = self.cfg.explore_step_ms if self.cfg.explore_enabled else auto.explore_step_ms
        hold = _jitter_ms(step_ms, self.cfg.jitter_ratio)
        if self.cfg.approach_mode == "mouse_click" or key.lower() == self.cfg.pickup_key.lower():
            # 点地面探索：既能走，又不会把拾取键当移动键（铁律）
            self._click_direction(scene, key, 220)
            self._intent(f"向{_DIR_CN.get(key, key)}方向点地面探索（寻找怪物/掉落）")
            self._emit_event("explore_move", key=key, mode="click")
        else:
            self._emit_move_keys([key], hold)
            self._intent(f"向{_DIR_CN.get(key, key)}方向探索移动（寻找怪物/掉落）")
            self._emit_event("explore_move", key=key, mode="key")
        return True

    # ------------- 地面掉落标签走位（TLI 类） -------------

    def _maybe_approach_ground_label(self, scene: SceneState, now: float) -> bool:
        """无光柱/列表时：地面掉落标签（黄条）→ 点击标签下方地面走过去。

        标签需连续 ≥400ms 可见（防伤害数字/聊天闪烁误点）。
        """
        labels = scene.ground_labels
        if not labels:
            self._label_first_seen_ms = None
            return False
        if self._label_first_seen_ms is None:
            self._label_first_seen_ms = scene.ts_ms
            return False
        if scene.ts_ms - self._label_first_seen_ms < 400:
            return False
        self._label_first_seen_ms = None
        e = labels[0]
        self._intent("走向地面掉落物标签（点击标签下方地面）")
        self.stats["label_walks"] = self.stats.get("label_walks", 0) + 1
        self._act(Command(op="mouse_click", button="left", x=e.x, y=e.y))
        self._emit_event("label_walk", x=e.x, y=e.y, labels=len(labels))
        self._cooldown_until = now + random.uniform(0.8, 1.3)
        self.state = State.COOLDOWN
        return True

    # ------------- 小地图导航 -------------

    # 8 方向键映射（角度: 0=右, 90=下, 屏幕坐标系 y 向下）
    _NAV_SECTORS = [
        (-22.5, 22.5, ["d"]),
        (22.5, 67.5, ["s", "d"]),
        (67.5, 112.5, ["s"]),
        (112.5, 157.5, ["s", "a"]),
        (-67.5, -22.5, ["w", "d"]),
        (-112.5, -67.5, ["w"]),
        (-157.5, -112.5, ["w", "a"]),
    ]

    def keys_for_angle(self, angle_deg: float) -> list[str]:
        """小地图标记方向 → 方向键（纯函数，可测）。"""
        if angle_deg > 157.5 or angle_deg < -157.5:
            return ["a"]
        for lo, hi, keys in self._NAV_SECTORS:
            if lo <= angle_deg <= hi:
                return keys
        return ["d"]

    def _maybe_navigate(self, scene: SceneState, now: float) -> None:
        """空闲且无拾取目标时：朝小地图目标标记方向走位（对话定稿的 laya 判路）。

        标记在到达圈内 → 不动（等游戏逻辑推进）；未启用 → 跳过。
        节流：每个决策间隔最多发一次导航步（按帧时间，状态机无关）。
        """
        if not self.cfg.nav_enabled or scene.minimap is None:
            return
        mm = scene.minimap
        if mm.marker is None or mm.arrived:
            return
        # 防抖等待期 → 保持站位
        if self._label_first_seen_ms is not None or                 self._rechallenge_first_seen is not None or                 self._list_first_seen_ms is not None:
            return
        # 节流：导航步间隔 = 1 个决策周期（1Hz≈1s / 6Hz≈167ms）
        nav_interval_ms = int(1000 / max(0.5, self.cfg.tick_hz))
        now_ms = int(now * 1000)
        if self._last_nav_ms and now_ms - self._last_nav_ms < nav_interval_ms:
            return
        keys = self.keys_for_angle(mm.angle_deg or 0.0)
        if not keys:
            return
        self._last_nav_ms = now_ms
        self.stats["nav_moves"] = self.stats.get("nav_moves", 0) + 1
        hold = _jitter_ms(self.cfg.nav_step_ms, self.cfg.jitter_ratio)
        sent = self._emit_move_keys(keys, hold)
        if sent == 0:
            # 方向键全部被铁律拦下（该方向键=拾取键）→ 改用点地面，绝不按拾取键
            self._click_direction(scene, "".join(keys), 200)
        self._emit_event("nav_move", angle=mm.angle_deg, keys="".join(keys),
                         distance_ratio=mm.distance_ratio)

    # ------------- 走位安全闸门 -------------

    def _act(self, cmd: Command) -> bool:
        """执行一条命令并登记"我确实动过"（看门狗据此区分"卡住"与"没事干"）。"""
        ok = self.exec.execute(cmd)
        if ok and cmd.op != "stop_all":
            self.monitor.note_action(self._now)
        return ok

    def _emit_move_keys(self, keys: list[str], hold_ms: int) -> int:
        """发出移动按键，**拦截拾取键**（铁律：拾取键只能由 A框裁决触发）。

        现实里很多游戏拾取键与左移同键（本项目默认 pickup_key=a），
        自主探索/导航一旦直接用方向键就会变成"盲捡"。这里做最后一道闸门，
        被拦下的次数记入 blind_tap_violations（监控可见，恒应为 0 才健康）。
        """
        sent = 0
        for k in keys:
            if (k or "").lower() == self.cfg.pickup_key.lower():
                self.stats["blind_tap_violations"] += 1
                log.error("拦截：禁止把拾取键 %s 当移动键使用", k)
                continue
            if self._act(Command(op="key_tap", key=k, hold_ms=hold_ms)):
                sent += 1
        return sent

    def _click_direction(self, scene: SceneState, keys: str, dist: int = 200) -> None:
        """方向 → 点击角色该方向的地面（不依赖键盘，天然避开拾取键冲突）。"""
        from ..decision.recovery import direction_vec

        dx, dy = direction_vec(keys)
        if dx == 0 and dy == 0:
            return
        x = max(0, min(scene.w - 1, int(scene.character[0]) + dx * dist))
        y = max(0, min(scene.h - 1, int(scene.character[1]) + dy * dist))
        self._act(Command(op="mouse_click", button="left", x=x, y=y))

    # ------------- APPROACHING -------------

    def _tick_approach(self, scene: SceneState, now: float) -> None:
        target = self._target
        if target is None:
            self.state = State.SCANNING
            return

        # 1) A框裁决 —— 最高优先级：出现且邻近目标才按拾取键（唯一合法按 A 时机）
        aframe = scene.aframe
        aframe_for_target = (
            aframe is not None
            and self._aframe_near_target(aframe, target)
        )
        if aframe_for_target:
            if self._aframe_first_seen_ms is None:
                self._aframe_first_seen_ms = scene.ts_ms
            self._stuck_ticks = 0
            # 停止移动再拾取（走位中直接松开方向键更稳）
            self._halt_movement()
            # 拾取复核冷却：上次按键后等游戏处理拾取动画（防连续按 A 捡垃圾）
            if scene.ts_ms >= self._tap_block_until_ms \
                    and scene.ts_ms - self._aframe_first_seen_ms >= 150:  # 连续≥150ms可见才动作
                self._tap_pickup(now)
            # 等待拾取确认
            self._check_pick_result(scene, now, aframe_present=True)
            return

        # A框消失：
        #   已按过拾取键 → 拾取成功（物品被捡走提示消失）
        #   未按过       → 单帧误检闪烁 → 清除状态继续走位（绝不能凭空判拾取成功）
        if self._aframe_first_seen_ms is not None:
            if self._taps_this_target > 0:
                self._emit_event("aframe_gone")
                self._finish_target("picked", success=True, now=now)
                return
            self._emit_event("aframe_flicker")
            self._aframe_first_seen_ms = None

        # A框存在但不属于当前目标（属于别的物品）：忽略，不按 A
        if aframe is not None:
            self._stuck_ticks = 0

        # 2) 目标跟踪：光柱还在吗？
        tracked = self._reacquire_target(scene, target, now)
        if tracked is None:
            # 光柱消失且无 A框：可能被别人捡走/拾取成功/瞬时遮挡
            self._emit_event("target_lost")
            self._finish_target("target_lost", success=True, now=now)
            return
        self._target = tracked

        # 3) 走位超时 → 放弃 + 拉黑
        if now - self._target_started > self.cfg.approach_timeout_s:
            self._abandon("timeout", now)
            return

        # 4) 撞墙检测：连续 N tick 画面几乎没变
        if scene.frame_delta < self.cfg.stuck_delta_thresh:
            self._stuck_ticks += 1
        else:
            self._stuck_ticks = 0
        if self._stuck_ticks >= self.cfg.stuck_frames:
            if not self._alt_direction_tried:
                self._alt_direction_tried = True
                self._stuck_ticks = 0
                self._emit_event("stuck_alt_direction")
                self._jiggle_alternate(scene, tracked)
                return
            self._abandon("stuck_wall", now)
            return

        # 5) 继续走位（按节流间隔重复点击/按键）
        if now - self._last_click_ts >= self._move_interval_s():
            self._move_toward(scene, tracked, now, first=False)

    # ------------- 走位动作 -------------

    def _move_interval_s(self) -> float:
        base = 1.0 / max(0.5, self.cfg.tick_hz)
        return base * random.uniform(0.9, 1.1)

    def _move_toward(self, scene: SceneState, target: Pillar, now: float, first: bool) -> None:
        self._last_click_ts = now
        dx, dy = target.cx - scene.character[0], target.y - scene.character[1]
        self._intent(f"向{_dir_text(dx, dy)}移动接近{target.color}掉落光柱（距离{round(target.distance)}px）")
        if self.cfg.approach_mode == "mouse_click":
            # 点击光柱下方地面（对话定稿：光柱悬空，点地面人物才会走过去）。
            # 只发单条 mouse_click(x,y)：RP2040/RPi5 后端内部先移动后点击。
            gx = int(target.x + random.randint(-8, 8))
            gy = int(target.y + self.cfg.ground_click_offset_px + random.randint(-6, 6))
            self._act(Command(op="mouse_click", button="left", x=gx, y=gy))
            if first:
                self._emit_event("move_click", x=gx, y=gy)
        else:
            keys = self._direction_keys(scene, target)
            if not keys:
                return
            hold = _jitter_ms(self.cfg.key_step_ms, self.cfg.jitter_ratio)
            if self._emit_move_keys(keys, hold) == 0:
                self._click_direction(scene, "".join(keys), 200)
            if first:
                self._emit_event("move_keys", keys="".join(keys))

    def _direction_keys(self, scene: SceneState, target: Pillar) -> list[str]:
        """按屏幕象限给出方向键（纯键盘走位兜底方案）。"""
        dx = target.cx - scene.character[0]
        dy = target.y - scene.character[1]
        dead = 40
        keys: list[str] = []
        if dx > dead:
            keys.append("d")
        elif dx < -dead:
            keys.append("a")
        if dy > dead:
            keys.append("s")
        elif dy < -dead:
            keys.append("w")
        if not self.cfg.diagonal_keys and len(keys) == 2:
            keys = [keys[0]]  # 主方向优先
        return keys

    def _jiggle_alternate(self, scene: SceneState, target: Pillar) -> None:
        """撞墙后的侧向脱困：垂直于目标方向轻移一步。"""
        if self.cfg.approach_mode == "mouse_click":
            # 垂直方向点击一个偏移点
            dx = target.cx - scene.character[0]
            dy = target.y - scene.character[1]
            # 旋转90°
            px, py = -dy, dx
            norm = max(1.0, (px * px + py * py) ** 0.5)
            gx = int(scene.character[0] + px / norm * 160)
            gy = int(scene.character[1] + py / norm * 160)
            self._act(Command(op="mouse_click", button="left", x=gx, y=gy))
        else:
            dx = target.cx - scene.character[0]
            dy = target.y - scene.character[1]
            k = "w" if dy < 0 else "s"
            hold = _jitter_ms(self.cfg.key_step_ms, self.cfg.jitter_ratio)
            if self._emit_move_keys([k], hold) == 0:
                self._click_direction(scene, k, 160)

    def _halt_movement(self) -> None:
        if self.cfg.approach_mode == "keys":
            for k in ("w", "a", "s", "d"):
                self._act(Command(op="key_up", key=k))

    # ------------- 拾取 -------------

    def _tap_pickup(self, now: float) -> None:
        if self._taps_this_target >= 2:
            self._abandon("max_taps", now)
            return
        # 防御性不变量：此处必然已检测到 A框（调用路径保证）
        self._intent(f"按拾取键 [{self.cfg.pickup_key.upper()}] 拾取物品（A框确认）")
        self.stats["a_taps"] += 1
        self._taps_this_target += 1
        hold = _jitter_ms(70, self.cfg.jitter_ratio)
        self._act(Command(op="key_tap", key=self.cfg.pickup_key, hold_ms=hold))
        self._last_tap_ts = now
        # 复核冷却：给游戏拾取动画+视觉确认留时间，期间不得再按
        self._tap_block_until_ms = now * 1000.0 + self.cfg.tap_recheck_ms
        self._emit_event("pickup_tap", tap=self._taps_this_target)

    def _check_pick_result(self, scene: SceneState, now: float, aframe_present: bool) -> None:
        """A框持续超过 2.5s 且已按 2 次仍未消失 → 放弃（防误捡垃圾循环）。"""
        if self._aframe_first_seen_ms is None or self._taps_this_target < 2:
            return
        if scene.ts_ms - self._aframe_first_seen_ms > 2500:
            self._abandon("aframe_stuck", now)

    # ------------- 目标管理 -------------

    def _aframe_near_target(self, aframe, target: Pillar) -> bool:  # noqa: ANN001
        """A框必须出现在当前目标光柱附近（对话定稿：A框标在就近物品名字上）。"""
        d = ((aframe.x - target.cx) ** 2 + (aframe.y - target.cy) ** 2) ** 0.5
        return d <= self.cfg.aframe_near_target_px

    def _reacquire_target(self, scene: SceneState, target: Pillar, now: float) -> Pillar | None:
        """在最新场景里找回当前目标（位置容差内最近者）。"""
        best: Pillar | None = None
        best_d = float("inf")
        for p in scene.pillars:
            if not p.whitelisted or self._is_blacklisted(p, now):
                continue
            d = ((p.cx - target.cx) ** 2 + (p.cy - target.cy) ** 2) ** 0.5
            if d <= TARGET_MATCH_TOL_PX and d < best_d:
                best, best_d = p, d
        return best

    def _abandon(self, reason: str, now: float) -> None:
        target = self._target
        if target is not None:
            until = now + self.cfg.abandon_blacklist_s
            self._blacklist.append((target.cx, target.cy, until))
        self.stats["abandons"] += 1
        self._finish_target(reason, success=False, now=now)

    def _finish_target(self, reason: str, success: bool, now: float) -> None:
        # 只有走完 "A框出现 → 按A → A框消失" 完整循环的才算真实拾取
        if reason == "picked":
            self.stats["picks"] += 1
            self.advisor.note_success()
            self.monitor.note_progress(now, "pick")
        elif reason in ("timeout", "stuck_wall", "max_taps", "aframe_stuck"):
            self.advisor.note_abandon()
            self._maybe_consult_llm(reason)
        self._emit_event("target_done", reason=reason, success=success)
        self._drop_target(reason)
        self._cooldown_until = now + random.uniform(0.4, 0.9)
        self.state = State.COOLDOWN

    # ------------- LLM 顾问（可选，异常局面咨询） -------------

    def _maybe_consult_llm(self, reason: str) -> None:
        advisor = self.advisor
        if not advisor.should_consult():
            return

        def _on_advice(payload: dict) -> None:
            self._on_llm_advice(payload, trigger=reason)

        summary = {
            "state": self.state.value,
            "trigger": reason,
            "stats": dict(self.stats),
            "recent_events": [e.get("event") for e in self.events[-8:]],
            "blacklist_size": len(self._blacklist),
        }
        advisor.consult(summary, on_advice=_on_advice)

    def _on_llm_advice(self, payload: dict, trigger: str = "recovery") -> None:
        """LLM 建议落地（后台线程回调，只做简单赋值 + 追加事件）。

        可执行动作（白名单在 cfg.llm.allowed_actions）：
          wait        → 拉长冷却（观察）
          explore     → 强制探索 15s（换区域找目标）
          reset_stage → 有『再次挑战』就点（重开一局）
          panic       → 急停（模型怀疑异常，交人工）
        其余仅记录，不影响决策。
        """
        self._emit_event("llm_advice", trigger=trigger, **payload)
        advice = payload.get("advice") or {}
        action = advice.get("action")
        if not payload.get("allowed"):
            return
        now = time.monotonic()
        if action == "wait":
            self._cooldown_until = max(self._cooldown_until, now + 3.0)
        elif action == "explore":
            self._auto_explore_until = now + 15.0
            self._emit_event("llm_policy", policy="explore", seconds=15)
        elif action == "reset_stage":
            btn = self._last_rechallenge
            if btn is not None:
                self._act(Command(op="mouse_click", button="left",
                                          x=btn[0], y=btn[1]))
                self._emit_event("llm_policy", policy="reset_stage")
        elif action == "panic":
            self.panic()

    def _drop_target(self, reason: str) -> None:
        self._target = None
        self._aframe_first_seen_ms = None
        self._taps_this_target = 0
        self._tap_block_until_ms = 0

    # ------------- 自主性观测（仪表盘/接口） -------------

    def autonomy_snapshot(self) -> dict:
        """自主恢复与模型调用状态（仪表盘展示 + 排障）。"""
        return {
            "enabled": self.cfg.autonomy.enabled,
            "signals": self.monitor.last.as_dict(),
            "recovery": {
                "active": self.state == State.RECOVERING,
                "reason": self._recovery_reason,
                "level": self.recovery.level,
                "step": self.recovery.last_step,
                "cycle": self.recovery.cycle,
                "steps_done": self.recovery.steps_done,
            },
            "vlm": self.vlm.snapshot(),
            "llm": {
                "enabled": self.cfg.llm.enabled,
                "model": self.cfg.llm.model,
                "queries": self.advisor.queries,
                "last_advice": self.advisor.last_advice,
            },
            "stats": {k: self.stats.get(k, 0) for k in (
                "recoveries", "recovery_steps", "recovery_cycles",
                "recovery_actions", "gate_escapes", "stuck_alarms")},
        }

    # ------------- 事件 -------------

    def _intent(self, text: str) -> None:
        """记录当前决策意图（人类可读，审查页展示）。"""
        self.last_intent = text

    def _emit_event(self, name: str, **detail) -> None:
        e = {"ts": time.time(), "state": self.state.value, "event": name, **detail}
        self.events.append(e)
        if len(self.events) > 400:
            del self.events[:-200]
        if self.on_event is not None:
            try:
                self.on_event(e)
            except Exception:  # noqa: BLE001
                log.exception("事件回调异常（忽略）")
        log.info("FSM[%s] %s %s", self.state.value, name, detail if detail else "")


_DIR_CN = {"w": "上", "s": "下", "a": "左", "d": "右",
          "wd": "右上", "wa": "左上", "sd": "右下", "sa": "左下"}


def _dir_text(dx: float, dy: float) -> str:
    """位移向量 → 中文方向（屏幕坐标系 y 向下）。"""
    import math as _m
    ang = _m.degrees(_m.atan2(dy, dx))
    sectors = [(22.5, "右"), (67.5, "右下"), (112.5, "下"), (157.5, "左下"),
               (-22.5, "右"), (-67.5, "右上"), (-112.5, "上"), (-157.5, "左上")]
    for lo, hi, name in [(22.5, 67.5, "右下"), (67.5, 112.5, "下"),
                         (112.5, 157.5, "左下"), (-22.5, 22.5, "右"),
                         (-67.5, -22.5, "右上"), (-112.5, -67.5, "上"),
                         (-157.5, -112.5, "左上")]:
        if lo <= ang < hi:
            return name
    return "左"


def _jitter_ms(value_ms: int, ratio: float) -> int:
    if ratio <= 0:
        return value_ms
    delta = max(1, int(value_ms * ratio))
    return max(10, value_ms + random.randint(-delta, delta))
