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
import math
import random
import time
from collections import deque
from pathlib import Path
from typing import Callable

from ..config import DecisionConfig
from .. import paths
from ..decision.actions import ActionRegistry
from ..decision.arbiter import Arbiter
from ..decision.knowledge import KnowledgeBase
from ..decision.llm_advisor import LLMAdvisor
from ..decision.recovery import ProgressMonitor, RecoveryLadder, StallSignals
from ..decision.supervisor import Supervisor, Verdict
from ..decision.vlm import VisionAdvisor
from ..executor import BaseExecutor
from ..humanize import Humanizer, softmax_choice
from ..monitor import recent_repeat
from ..protocol import Command, Pillar, SceneState

log = logging.getLogger(__name__)

#: 台账/日志是数据，写到数据根（设备上是 /var/lib/autogame），升级换代码时不能带走。
_DATA = paths.data_root()


def _knowledge_dir(cfg) -> Path:
    """经验库目录：profile 指定优先，否则落在默认 profile 下。

    不放到仓库根，是为了让"一个游戏一个目录"这条约定不出例外 ——
    否则用户会看到根的 knowledge/ 和 profiles/<id>/knowledge/ 两份，不知道该填哪个。
    相对路径按**数据根**解析（不是 cwd）：systemd 下 cwd 是 `/`。
    """
    raw = str(getattr(cfg, "knowledge_dir", "") or "").strip()
    if raw:
        return paths.resolve(raw)
    return paths.data("profiles", "generic_1280x720", "knowledge")

# 目标匹配容差 & 拉黑半径（像素）
TARGET_MATCH_TOL_PX = 70
BLACKLIST_RADIUS_PX = 90
# A框确认窗下限（毫秒）：低于该值不动作，防单帧误检闪烁
AFRAME_CONFIRM_MS = 150

# 规则要用的事实窗口。都取得比体检窗口（20s）短：规则的用处是**当场**把动作
# 拐回来，等 20 秒窗口结算完，那 20 秒已经按完了。
#: "同一个动作重复了几次"看最近 5 秒
_REPEAT_WINDOW_S = 5.0
#: "最近进了几次恢复"看最近 60 秒
_RECOVERY_WINDOW_S = 60.0


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
                 vlm: VisionAdvisor | None = None,
                 supervisor: Supervisor | None = None,
                 arbiter: Arbiter | None = None) -> None:
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
            "drifts": 0,                # 走神（非目标导向噪音操作）次数
            "misclicks": 0,             # 疲劳导致的走位失误次数
            "human_breaks": 0,          # 疲劳休息次数
            "social_msgs": 0,           # 社交噪音发送次数
            "walk_starts": 0,           # 发起走位（点地面/按方向键）次数
            "walk_ends": 0,             # 走位等待结束次数
            "walk_progress": 0,         # 有效接近（距离真的缩短了）
            "walk_no_progress": 0,      # 点了没走近（撞墙/寻路失败）
            "walk_settles": 0,          # 人物提前停下（等待 A框的宽限触发）
            "targets_ignored": 0,       # 选择性无视（懒得捡）次数
            "off_pick": 0,              # 选了非首个候选的次数（偏好证据）
            "supervisor_blocks": 0,     # 被监督层拦下的命令数（shadow 下恒 0）
            "supervisor_takeovers": 0,  # 监督层接管次数（esc/reset_stage）
            "supervisor_panics": 0,     # 监督层停车次数
            "arbiter_actions": 0,       # 三级裁决实际执行的动作数（shadow 下恒 0）
        }
        # 自主性：看门狗 + 恢复阶梯 + 视觉顾问
        self.monitor = ProgressMonitor(cfg.autonomy)
        # 拟人化：疲劳/反应时间/走神/社交（只调时序与噪音，不改动作语义）
        self.human = Humanizer(cfg.humanize, pickup_key=cfg.pickup_key)
        self.vlm = vlm or VisionAdvisor(cfg.vlm)
        self.frame_source: Callable[[], bytes | None] | None = None
        self.vlm.frame_source = self._current_frame
        # LLM 监督层：对账"动作有没有用"，有权踩刹车（默认 shadow，不真的拦）
        self.supervisor = supervisor or Supervisor(cfg.supervisor, root=_DATA)
        # 三级裁决：经验直出 → laya 从能力清单里选 → 都不行才问大模型。
        # 行为清单必须由 action_table() 提供实现，清单里没有实现的动作执行不了，
        # 所以 registry 只描述、FSM 才拥有"能不能做"的最终解释权。
        self.actions = ActionRegistry()
        self.kb = KnowledgeBase(_knowledge_dir(cfg.arbiter))
        self.arbiter = arbiter or Arbiter(cfg.arbiter, self.actions, self.kb, root=_DATA)
        self._scene: SceneState | None = None
        self._last_action_at = 0.0
        self._arbiter_paused_until = 0.0
        self.recovery = RecoveryLadder(
            cfg.autonomy, self.vlm, self.advisor,
            forbidden_keys={cfg.pickup_key},  # 铁律：恢复动作永不按拾取键
        )
        self._recovery_reason = ""
        self._recovery_picks = 0
        self._recovery_targets0 = 0
        self._recovery_ok_ticks = 0
        # 进入恢复的时刻（只留最近一两分钟）。规则要用的 `recoveries` 事实是
        # "最近 60 秒进了几次恢复"，而 stats 里的 recoveries 是**累计值**——
        # 累计值当条件用毫无意义（跑一小时之后它必然大于任何阈值）。
        self._recovery_ts: deque[float] = deque(maxlen=64)
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
        self._gate_uncalibrated_emitted: bool = False
        self.last_intent: str = ""
        self._explore_i: int = 0
        self._last_explore_ms: int = 0
        self._loading_hold_emitted: bool = False
        self._target: Pillar | None = None
        self._target_started: float = 0.0
        self._last_click_ts: float = 0.0
        self._aframe_first_seen_ms: int | None = None
        self._reaction_ms: int = AFRAME_CONFIRM_MS  # 本目标的反应时间（看到A框→按下）
        self._aframe_seen_frames: int = 0           # 本目标 A框连续出现帧数（防按空）
        self._aframe_dist0: float = 0.0             # A框首次出现时的距离（判断"在靠近还是远离"）
        self._taps_this_target: int = 0
        self._stuck_ticks: int = 0
        self._alt_direction_tried: bool = False
        self._cooldown_until: float = 0.0
        self._last_tap_ts: float = 0.0
        self._tap_block_until_ms: int = 0
        self._hesitate_until: float | None = None   # 进关卡前"愣一下"的截止时刻
        self._break_emitted: bool = False           # 疲劳休息事件只发一次
        # 走位等待：点完地面后人物在游戏里自己走过去，这段期间不该再点
        self._walking: bool = False
        self._walk_started: float = 0.0
        self._walk_until: float = 0.0
        self._walk_dist0: float = 0.0
        self._walk_still: int = 0
        self._walk_settled_at: float | None = None
        self._ignored: list[tuple[int, int, float]] = []   # 选择性无视（拟人化）
        self._recent: list[tuple[int, int, str, float]] = []  # 最近拾取（新鲜感记忆）
        self._blacklist: list[tuple[int, int, float]] = []  # (x, y, until_frame_ts)

    # ------------- 对外控制 -------------

    def start(self) -> None:
        if self.state == State.PANIC:
            log.warning("PANIC 状态需先 reset_panic")
            return
        self.exec.reset()
        self.state = State.SCANNING
        self._need_monitor_reset = True
        self._hesitate_until = None
        self._break_emitted = False
        self._walking = False
        self._ignored.clear()
        self._recent.clear()
        self.recovery.reset()
        self._recovery_ts.clear()
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
        # 人工复位时一并清掉监督层结论：否则那条 panic 会挂到 ttl 到期，
        # 复位后每帧又被拦，看起来像"复位没用"。
        self.supervisor.reset()
        self.arbiter.reset()
        self._recovery_ts.clear()
        self._last_action_at = 0.0
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
        self._scene = scene
        if self._need_monitor_reset:
            self.monitor.reset(now)
            self._need_monitor_reset = False
        signals = self.monitor.update(scene, now, self.stats["picks"])

        # ---- LLM 监督层：每帧本地对账（零成本），异常才异步叫模型 ----
        # 它只能踩刹车（hold/takeover/panic），不能指定去哪——目标仍由像素证据裁决。
        if self.supervisor.cfg.mode != "off":
            self.supervisor.note_frame(scene, self.last_intent, now,
                                       self._sup_probe(scene))
            if self._sup_maybe_act(scene, now):
                return

        # ---- 拟人化：状态推进 + 疲劳休息（累了就停手，真人不连轴转） ----
        self.human.tick(now)
        if self.human.paused(now):
            if not self._break_emitted:
                self._break_emitted = True
                self.stats["human_breaks"] += 1
                self._emit_event("human_break",
                                 seconds=round(self.human.break_seconds_left(now), 1),
                                 fatigue=round(self.human.fatigue.level, 3))
                self._intent("疲劳到阈值，停手歇一会儿（休息窗口内不发任何命令）")
            self._drop_target("human_break")
            return
        if self._break_emitted:
            self._break_emitted = False
            # 休息后重新计时，否则刚睁眼就被"停滞"判据抓回去
            self.monitor.reset(now)
            self._emit_event("human_resume", fatigue=round(self.human.fatigue.level, 3))
            self._intent("休息结束，恢复作业")

        # 加载/黑屏过场：保持不动（等加载完成），不产生任何命令
        if scene.loading:
            if not self._loading_hold_emitted:
                self._emit_event("loading_hold")
                self._loading_hold_emitted = True
            self._drop_target("loading")
            return
        self._loading_hold_emitted = False

        # ---- 三级裁决：只在快通道答不上来时介入（闲置/未标定/连续落空） ----
        # 认得就自己做（上面那些规则），认不得才问。它返回的是一条高层动作，
        # 由 action_table() 翻译成本机的真实操作，绝不接受模型给的坐标。
        if self._tick_arbiter(scene, now):
            return

        if not scene.in_game:
            self._tick_gated(scene, now, signals)
            return
        self._gate_hold_emitted = False
        self._gate_uncalibrated_emitted = False

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

    # ------------- LLM 监督层 -------------

    def _sup_probe(self, scene: SceneState) -> dict:
        """当前帧的实测值（给监督层对账用）。

        距离必须**现算**：self._target 是上一帧的对象，它的 .distance 只在检测时
        算过一次。拿旧距离去做"有没有走近"的对账，永远得到 hit —— 那等于没对账。
        """
        tgt = self._target
        dist = None
        if tgt is not None:
            dist = math.hypot(tgt.cx - scene.character[0],
                              tgt.y - scene.character[1])
        return {
            "dist": dist,
            "aframe": scene.aframe is not None,
            "picks": self.stats["picks"],
            "in_game": scene.in_game,
        }

    def _sup_maybe_act(self, scene: SceneState, now: float) -> bool:
        """监督层生效：停车 / 接管一次。返回 True = 本帧已处理，不要再决策。

        shadow 模式下 should_panic / take_takeover 都返回 False —— 只记账不动手。
        """
        sup = self.supervisor
        if sup.should_panic(now):
            snap = sup.snapshot()
            self.stats["supervisor_panics"] += 1
            self._emit_event("supervisor_panic",
                             diagnosis=snap.get("diagnosis", ""),
                             evidence=snap.get("evidence", ""))
            self._intent(f"监督层判定异常，停车交人工：{snap.get('diagnosis', '')}")
            self.panic()
            return True

        act = sup.take_takeover(now)
        if not act:
            return False
        op = str(act.get("op", ""))
        reason = str(act.get("reason", ""))
        self.stats["supervisor_takeovers"] += 1
        self._emit_event("supervisor_takeover", op=op, reason=reason)
        if op == "esc":
            self._intent(f"监督层接管：按 ESC 关闭可疑界面（{reason or '界面可疑'}）")
            self._act(Command(op="key_tap", key=self.cfg.autonomy.escape_key, hold_ms=60))
        elif op == "reset_stage":
            btn = self._last_rechallenge
            if btn is not None:
                self._intent("监督层接管：点『再次挑战』重开当前关卡")
                self._act(Command(op="mouse_click", button="left", x=btn[0], y=btn[1]))
            else:
                self._intent("监督层要重开，但画面没有『再次挑战』，改为观察")
        else:
            self._intent(f"监督层接管：暂停观察（{reason or '等一帧'}）")
        return True

    # ------------- 三级裁决（arbiter） -------------

    def _arb_probe(self, scene: SceneState) -> dict:
        """给裁决层的实测快照。字段名与 knowledge.fingerprint 对齐。

        多出来的这几个字段不是为了给模型看，是给**指纹**用的：
        同一个界面在不同帧的像素完全不同，唯一稳定的匹配依据就是这些定性特征。
        """
        p = self._sup_probe(scene)
        now = self._now
        # 注意时钟：`self._now` 是**帧时间**（scene.ts_ms/1000，从关卡开始算），
        # 而执行器 journal 里的 ts 是 time.time()（墙钟）。拿帧时间去筛 journal
        # 一条都筛不出来，`repeat` 会恒为 0 —— 规则于是"写了也不命中"。
        # 这里按 journal 自己的口径取"最近 5 秒"。
        repeat, repeat_key = recent_repeat(
            getattr(self.exec, "journal", None) or [],
            time.time() - _REPEAT_WINDOW_S)
        p.update({
            "gate_reason": str(getattr(scene, "gate_reason", "") or ""),
            # 这几个是二次进化的规则要用的。必须在这里显式给：规则引用的事实
            # 若在 probe 里不存在，会求值成 unknown 然后静默不生效，人只会看到
            # "我写的规则没反应"，而原因藏在这一层。
            "in_game": bool(scene.in_game),
            "aframe": scene.aframe is not None,
            "ground_labels": len(scene.ground_labels or []),
            "pillars": len(scene.pillars or []),
            "monsters": len(scene.monsters or []),
            "w": scene.w, "h": scene.h,
            "nav_enabled": bool(self.cfg.nav_enabled),
            "rechallenge_visible": self._last_rechallenge is not None,
            "intent": self.last_intent,
            "miss_streak": int(self.supervisor.last_health.get("miss_streak", 0) or 0),
            # "同一个动作在重复"和"一直在恢复"是体检能判出来、也是人最想写规则
            # 治住的两件事。窗口取最近 5s / 60s：规则要能**当场**命中，不能用
            # 体检那个 20 秒结算窗口（等它结算完，那 20 秒已经白按了）。
            "repeat": int(repeat),
            "repeat_key": str(repeat_key),
            "recoveries": sum(1 for t in self._recovery_ts
                              if now - t <= _RECOVERY_WINDOW_S),
        })
        # 监督层的窗口指标（捡到几个、在动但没结果多久、意图切了几次）。
        # **只在真算出过结论时才给**：监督层关掉时这里是空的，缺字段让规则
        # 求值成 unknown（日志里会说哪条没求值），而不是填 0 ——
        # 填 0 会让 `picks_delta == 0` 这类条件在"根本没在测"的情况下命中。
        health = self.supervisor.last_health or {}
        if health:
            p.update({
                "picks_delta": int(health.get("picks_delta") or 0),
                "no_progress_for": int(health.get("no_progress_for") or 0),
                "intent_churn": int(health.get("intent_churn") or 0),
                "cmds": int(health.get("cmds") or 0),
            })
        return p

    def action_table(self) -> dict[str, Callable[[], bool]]:
        """行为清单的实现端 —— 注册表描述"叫什么"，这里决定"真能不能做"。

        为什么把映射留在 FSM：动作需要访问内部状态（当前目标、走位等待窗口、
        上次看到的按钮位置）。如果让注册表自己实现，它会不得不把 FSM 的一大半
        状态复制一遍，然后两份状态开始漂移。
        """
        return {
            "pickup": self._arb_pickup,
            "approach_nearest": self._arb_approach,
            "attack_nearest": self._arb_attack,
            "explore": self._arb_explore,
            "navigate": self._arb_navigate,
            "rechallenge": self._arb_rechallenge,
            "esc": self._arb_esc,
            "wait": self._arb_wait,
            "reset_stage": self._arb_rechallenge,   # 同一入口：点『再次挑战』
        }

    def _arb_pickup(self) -> bool:
        self._tap_pickup(self._now)
        return True

    def _arb_approach(self) -> bool:
        scene, now = self._scene, self._now
        if scene is None:
            return False
        tgt = self._target or self._pick_target(scene, now)
        if tgt is None:
            return False
        self._target = tgt
        self._move_toward(scene, tgt, now, first=True)
        return True

    def _arb_attack(self) -> bool:
        if self._scene is None:
            return False
        return self._maybe_attack_monster(self._scene, self._now)

    def _arb_explore(self) -> bool:
        if self._scene is None:
            return False
        return self._maybe_explore(self._scene, self._now)

    def _arb_navigate(self) -> bool:
        if self._scene is None:
            return False
        self._maybe_navigate(self._scene, self._now)
        return True

    def _arb_rechallenge(self) -> bool:
        btn = self._last_rechallenge
        if btn is None:
            return False
        self._act(Command(op="mouse_click", button="left", x=btn[0], y=btn[1]))
        return True

    def _arb_esc(self) -> bool:
        # 未标定的画面不许按 ESC：它可能是没标定的战斗画面，一按就进暂停菜单。
        # 这条规则以前在 _tick_gated 里，现在裁决层也能发 ESC，所以必须在这里再挡一次。
        if self._scene is not None and str(getattr(self._scene, "gate_reason", "")) == "unmatched":
            return False
        self._act(Command(op="key_tap", key=self.cfg.autonomy.escape_key, hold_ms=60))
        return True

    def _arb_wait(self) -> bool:
        return True

    def _tick_arbiter(self, scene: SceneState, now: float) -> bool:
        """问裁决层：现在该做什么。返回 True = 本帧已由它处理。

        关键：arbiter.tick() 不会阻塞。模型在后台跑，这一帧该干什么还干什么。
        """
        if self.arbiter.cfg.mode == "off":
            return False
        if now < self._arbiter_paused_until:
            return False
        idle_s = (now - self._last_action_at) if self._last_action_at else 0.0
        call = self.arbiter.tick(self._arb_probe(scene), now,
                                 idle_s=max(0.0, idle_s), intent=self.last_intent)
        if call is None:
            return False
        fn = self.action_table().get(call.action_id)
        if fn is None:
            # 清单和实现对不上 = 代码 bug。拒绝执行并留痕，不要静默失败。
            log.error("裁决动作没有实现: %s（清单与 action_table 不同步）", call.action_id)
            self._emit_event("arbiter_missing_impl", action=call.action_id)
            self._arbiter_paused_until = now + 30.0
            return False
        try:
            ok = bool(fn())
        except Exception:  # noqa: BLE001
            log.exception("裁决动作执行异常: %s", call.action_id)
            ok = False
        if not ok:
            # 执行不了（比如没有目标/没有按钮）：不记对账，否则会污染经验库
            self._emit_event("arbiter_noop", action=call.action_id)
            return False
        self.arbiter.execute_hook(call, self._arb_probe(scene), now)
        self.stats["arbiter_actions"] += 1
        self._emit_event("arbiter_action", action=call.action_id,
                         stage=call.stage, reason=call.reason[:120])
        self._intent(f"三级裁决[{call.stage}] {call.action_id}：{call.reason or '模型选择'}")
        return True

    def _tick_gated(self, scene: SceneState, now: float, signals: StallSignals) -> None:
        """不在战斗界面：短时间等待；长时间（弹窗/菜单/结算）按 ESC 自救。

        但**门禁说"没认出来"（unmatched）时一律不动**，连 ESC 都不按：
        未标定的画面可能是菜单，也可能是没标定的战斗画面——在战斗画面按 ESC
        等于自己把游戏按进暂停菜单（真实录像上连按 6 次，比卡住更糟）。
        """
        if not self._gate_hold_emitted:
            self._emit_event("scene_gate_hold")
            self._gate_hold_emitted = True
        self._drop_target("scene_gate")
        # 清掉上一帧的战斗意图：否则审查页/仪表盘会把"接近 auto2 掉落光柱"这种
        # 上一帧的残留意图挂在关卡选择页上（真实事故，看起来像程序在乱认画面）。
        self._intent("非战斗界面（关卡选择/背包/菜单）：本帧不决策、不点击")
        if scene.gate_reason == "unmatched":
            if not self._gate_uncalibrated_emitted:
                self._gate_uncalibrated_emitted = True
                self._emit_event("scene_gate_uncalibrated")
            self._intent("未识别的画面（HUD 未标定）：不决策、不按键，先标定技能栏")
            return
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
        # 登记预期：按完 ESC 该回到战斗界面。没回去说明按错了界面（或这根本不是弹窗），
        # 连续几次就说明"按 ESC 自救"这条路在这个画面上是错的。
        self.supervisor.predict("escape", "ESC 后应回到战斗界面", base=0.0,
                                now=now, horizon_s=4.0)

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
        self._recovery_ts.append(now)
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
        # 走神 / 社交噪音：偶尔做点和当前任务无关的事（默认只动鼠标，不点击不按键）
        if self._maybe_drift(scene, now):
            return
        if self._maybe_social(now):
            return
        target = self._pick_target(scene, now)
        if target is None:
            if not self._maybe_rechallenge(scene, now):
                # 上一次点地面还在走 → 别重发同一指令（点一下 ≠ 走到位）
                if self._walk_wait_active(scene, now):
                    return
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
        self._aframe_seen_frames = 0
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

    def _maybe_drift(self, scene: SceneState, now: float) -> bool:
        """走神：偶尔做一件和当前任务无关的事（默认只有鼠标移动）。

        真人跑图会鼠标划过无关角落、在背包里翻两下又关掉、进关卡前停一下。
        这类操作不会被任务逻辑自然生成 —— 恰恰是真人行为里最"像人"的部分。
        安全默认：不点击、不按键，移动光标在绝大多数游戏里没有副作用。
        """
        if not getattr(self.exec, "supports_mouse_move", True):
            return False  # 机械舵机后端没有绝对定位，鼠标噪音无从谈起
        cmds = self.human.maybe_drift(now, scene.w, scene.h)
        if not cmds:
            return False
        for c in cmds:
            self._act(c)
        self.stats["drifts"] += 1
        self._intent("走神：鼠标飘向无关角落（非目标导向的噪音操作）")
        self._emit_event("drift", commands=len(cmds))
        return True

    def _maybe_social(self, now: float) -> bool:
        """社交噪音：偶尔在公共频道说句废话（默认关闭，见 SocialConfig 风险说明）。"""
        cmds = self.human.maybe_social(now)
        if not cmds:
            return False
        for c in cmds:
            self._act(c)
        self.stats["social_msgs"] += 1
        self._intent("社交噪音：在公共频道发送一句无意义台词")
        self._emit_event("social_msg", commands=len(cmds))
        return True

    def _pick_target(self, scene: SceneState, now: float) -> Pillar | None:
        """挑一个目标 —— 不是"永远第一个"。

        原实现是 `(stables or candidates)[0]`：确定性规则，样本一多必然暴露。
        真人挑目标有自己的偏好，而且偏好会漂移：
          - 累的时候只捡近的（耐心↓）；
          - 精神好时愿意绕远捡贵的（贪心↑）；
          - 刚捡过同色/同位置会腻（新鲜感）；
          - 偶尔就是懒得捡（直接无视，继续走）。

        拟人化关闭时退回旧行为（确定性选择），保证测试与排障可复现。
        只有一个候选时也直接选它 —— "选择性无视"是关于**在多个目标里挑**，
        不是"唯一的目标也放弃"（那属于 _abandon/拉黑的语义）。
        """
        # 优先取 stable 光柱（掉落物存续长，特效短命）；无 stable 回退全部候选
        candidates = [p for p in scene.pillars
                      if p.whitelisted and not self._is_blacklisted(p, now)
                      and not self._is_ignored(p, now)]
        if not candidates:
            return None
        stables = [p for p in candidates if p.stable]
        pool = stables or candidates
        if not self.human.mood_active or len(pool) == 1:
            return pool[0]

        mood = self.human.mood
        # 懒得捡：逐个候选按"距离 + 耐心"决定要不要无视（人真的会漏捡）。
        # 注意是**逐个**过滤，不是"懒得捡一个就全都不捡"。
        kept = []
        for p in pool:
            if mood.should_skip(p.distance):
                self._ignore(p, now)
            else:
                kept.append(p)
        if not kept:
            self.stats["targets_ignored"] += 1
            self._emit_event("target_ignored", count=len(pool))
            self._intent("懒得捡：这批掉落全部无视，继续走")
            return None
        if len(kept) == 1:
            return kept[0]

        utils = [mood.utility(p.distance, p.color, self._novelty(p, now)) for p in kept]
        idx = softmax_choice(utils, self.cfg.humanize.mood.temperature, self.human.rng)
        chosen = kept[idx]
        best = max(range(len(kept)), key=lambda i: utils[i])
        if idx != best:
            # 统计"挑了次优目标"——这是 softmax 与 argmax 的差别，也是最像人的地方
            self.stats["off_pick"] += 1
            self._emit_event("target_choice", index=idx, of=len(kept),
                             color=chosen.color, distance=round(chosen.distance))
        return chosen

    def _is_blacklisted(self, p: Pillar, now: float) -> bool:
        for bx, by, until in self._blacklist:
            if until > now and abs(p.cx - bx) <= BLACKLIST_RADIUS_PX \
                    and abs(p.cy - by) <= BLACKLIST_RADIUS_PX:
                return True
        return False

    # ------------- 选择性无视 + 新鲜感（拟人化目标偏好） -------------

    def _ignore(self, p: Pillar, now: float) -> None:
        """无视某个目标一段时间（人真的会漏捡，不是每个掉落都值得弯腰）。"""
        lo, hi = (list(self.cfg.humanize.mood.skip_ignore_s) + [20.0, 60.0])[:2]
        ttl = random.uniform(min(lo, hi), max(lo, hi))
        self._ignored.append((p.cx, p.cy, now + ttl))
        if len(self._ignored) > 32:
            del self._ignored[:-24]

    def _is_ignored(self, p: Pillar, now: float) -> bool:
        if not self._ignored:
            return False
        self._ignored = [i for i in self._ignored if i[2] > now]
        for ix, iy, _ in self._ignored:
            if abs(p.cx - ix) <= BLACKLIST_RADIUS_PX and abs(p.cy - iy) <= BLACKLIST_RADIUS_PX:
                return True
        return False

    def _note_recent_pick(self, p: Pillar, now: float) -> None:
        self._recent.append((p.cx, p.cy, p.color, now))
        cap = max(1, self.cfg.humanize.mood.recent_memory)
        if len(self._recent) > cap:
            del self._recent[:-cap]

    def _novelty(self, p: Pillar, now: float) -> float:
        """新鲜感：刚捡过同色/同位置的会腻（1=很新鲜，0=腻了）。"""
        if not self._recent:
            return 1.0
        hl = max(5.0, self.cfg.humanize.mood.recent_halflife_s)
        pen = 0.0
        for x, y, color, ts in self._recent:
            age = max(0.0, now - ts)
            decay = 0.5 ** (age / hl)
            if color == p.color:
                pen += 0.15 * decay
            if abs(p.cx - x) < 150 and abs(p.cy - y) < 150:
                pen += 0.35 * decay
        return max(0.0, 1.0 - pen)

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
            self._hesitate_until = None
            return
        self._last_rechallenge = (btn.x, btn.y)
        if self._rechallenge_first_seen is None:
            self._rechallenge_first_seen = scene.ts_ms
            return
        if scene.ts_ms - self._rechallenge_first_seen < 200:
            return
        # 确认要进：先在门口"愣一下"（真人不会确认完 200ms 就点下去）
        if self._hesitate_until is None:
            self._hesitate_until = now + self.human.hesitation_s()
        if now < self._hesitate_until:
            self._intent("准备进下一层，先在门口停顿一下")
            return
        self._hesitate_until = None
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
            self._begin_walk(now, None)   # 点完等人物走过去，别每秒重发
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
        """执行一条命令并登记"我确实动过"（看门狗据此区分"卡住"与"没事干"）。

        所有命令的唯一出口 —— 监督层的 hold 闸门就架在这里。它只能否决，
        不会改写命令：被拦下就是这帧不动，而不是"改成去别处"。
        """
        if not self.supervisor.gate(cmd, self._now):
            self.stats["supervisor_blocks"] += 1
            log.debug("监督层拦下命令: %s", cmd.op)
            return False
        ok = self.exec.execute(cmd)
        if ok and cmd.op != "stop_all":
            self.monitor.note_action(self._now)
            self.human.note_action(self._now)
            self.supervisor.note_command(self._now)
            # 裁决层的闲置判据：真的发出命令才算"动了"，被拦下的不算
            self._last_action_at = self._now
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

        # 当前帧的距离：A框判据必须用**这一帧**的数据。
        # 用 self._target（上一帧的对象）会让"在靠近还是远离"永远滞后一帧，
        # 于是"人物已经在走远"这件事永远看不见 —— 那就是盲按的根源。
        _fresh = self._reacquire_target(scene, target, now)
        if _fresh is not None:
            target = self._target = _fresh   # 用当前帧的对象（距离/位置都是最新的）
        dist_now = float(target.distance)

        # 1) A框裁决 —— 最高优先级：出现且邻近目标才按拾取键（唯一合法按 A 时机）
        aframe = scene.aframe
        aframe_for_target = (
            aframe is not None
            and self._aframe_near_target(aframe, target)
        )
        if aframe_for_target:
            if self._aframe_first_seen_ms is None:
                self._aframe_first_seen_ms = scene.ts_ms
                self._aframe_dist0 = dist_now
                # 反应时间：看到 A框 → 按下拾取键（对数正态右偏，疲劳会拉长）
                self._reaction_ms = max(AFRAME_CONFIRM_MS, self.human.reaction_ms())
            self._aframe_seen_frames += 1
            self._stuck_ticks = 0
            # 停止移动再拾取（走位中直接松开方向键更稳）
            self._halt_movement()
            # 走到位了：结束走位等待（A框出现本身就是"到了"的证据）
            self._end_walk(now, reason="aframe", gained=self._walk_dist0 - dist_now)
            # 防"按空"（铁律：绝不盲按拾取键）。帧是"过去"的信息，
            # 采集→编码→传输→检测往往几百毫秒，这一帧的 A框可能早已失效。
            # 两个硬判据，任一不满足就再等一帧：
            #   ① 连续确认够 N 帧（不看帧差 —— 人物减速时帧差可能低于阈值，但仍在外移）；
            #   ② 距离没有变大（在远离 = 路过/走过头，按下去必然空）。
            approaching = dist_now <= self._aframe_dist0 + 8.0
            confirmed = self._aframe_seen_frames >= max(1, self.cfg.aframe_confirm_frames)
            if not confirmed:
                self._intent("A框已出现，等连续确认（防旧帧按空）")
                return
            if not approaching:
                self._intent("A框出现但人物在远离（路过/走过头）—— 不按，避免按空")
                return
            # 拾取复核冷却：上次按键后等游戏处理拾取动画（防连续按 A 捡垃圾）
            if scene.ts_ms >= self._tap_block_until_ms \
                    and scene.ts_ms - self._aframe_first_seen_ms >= self._reaction_ms:
                self._tap_pickup(now)
                self._check_pick_result(scene, now, aframe_present=True)
                return
            # 按不了（没确认 / 在远离 / 反应时间没到）。**不能死等**：
            # 光柱可能已丢失检测、A框属于旁边另一件物品、或者根本没进拾取半径。
            # 长期按不下去 → 再走一步（真人也会"咦，怎么捡不到，挪一下"）。
            if self._aframe_seen_frames >= max(2, self.cfg.aframe_retry_frames) \
                    and self._taps_this_target == 0:
                self._aframe_first_seen_ms = None
                self._aframe_seen_frames = 0
                self._aframe_dist0 = 0.0
                self._emit_event("aframe_reapproach", distance=round(dist_now))
                self._intent("A框在但按不下去 —— 再走一步靠近")
                self._move_toward(scene, target, now, first=False)
                return
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
            self._end_walk(now, reason="timeout")
            self._abandon("timeout", now)
            return

        # 4) 走位等待：点完地面只是"发起移动"，人物走过去是游戏侧的耗时步骤
        #    （1-3s，且 A框要到一定距离内才出现）。这段期间**不重复点击**，
        #    只观察：画面在动=在走；画面停了=A框宽限；超预算=没走到，重新点。
        if self._walking and self._walk_step(scene, tracked, now):
            return

        # 5) 撞墙检测：连续 N tick 画面几乎没变
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

        # 6) 继续走位（按节流间隔重复点击/按键）
        if now - self._last_click_ts >= self._move_interval_s(now):
            self._move_toward(scene, tracked, now, first=False)

    # ------------- 走位等待（点一下 ≠ 走到位） -------------

    def _begin_walk(self, now: float, target: Pillar | None) -> None:
        """发起一次走位后进入"等待人物走过去"状态。

        预算与 approach_timeout 联动：至少给 2 轮机会，否则走位超时会把
        "还在路上"误判成"走不到"（预算 3s + 超时 8s 时原本只能试 2 次）。
        """
        walk = self.cfg.walk
        if not walk.enabled:
            return
        remaining = self.cfg.approach_timeout_s - (now - self._target_started)
        budget = min(walk.budget_s, max(1.0, remaining / 2.0))
        if self.cfg.approach_mode == "keys":
            budget = max(budget, self.cfg.key_step_ms / 1000.0 + walk.key_extra_s)
        self._walking = True
        self._walk_started = now
        self._walk_until = now + budget
        self._walk_dist0 = float(target.distance) if target is not None else 0.0
        self._walk_still = 0
        self._walk_settled_at = None
        self.stats["walk_starts"] += 1
        # 登记预期：这次走位到期时距离该缩短。监督层用它判断"走位是不是假动作"
        # （点了地面、命令发出去了、人物其实没动——这是规则层最难自证的一类故障）。
        if target is not None:
            self.supervisor.predict(
                "walk", f"走位后距离应缩短（当前 {target.distance:.0f}px）",
                base=float(target.distance), now=now, horizon_s=budget + 1.0)

    def _end_walk(self, now: float, reason: str, gained: float | None = None) -> None:
        if not self._walking:
            return
        self._walking = False
        self.stats["walk_ends"] += 1
        self._emit_event("walk_end", reason=reason,
                         waited=round(now - self._walk_started, 2),
                         gained=None if gained is None else round(gained, 1))

    def _walk_step(self, scene: SceneState, target: Pillar, now: float) -> bool:
        """走位等待一步。返回 True = 本 tick 什么都不做（人物还在走）。"""
        walk = self.cfg.walk
        moving = scene.frame_delta >= walk.motion_delta_thresh
        if now < self._walk_until:
            if moving:
                self._walk_still = 0
                self._walk_settled_at = None
                return True                     # 人还在走，等
            self._walk_still += 1
            if self._walk_still < walk.settle_frames:
                return True                     # 可能只是动画间隙，再确认一帧
            if self._walk_settled_at is None:
                self._walk_settled_at = now
                self.stats["walk_settles"] += 1
            if now - self._walk_settled_at < walk.settle_grace_s:
                # 人物停了，但 A框提示有延迟 —— 先给游戏一点时间
                return True
        gained = self._walk_dist0 - float(target.distance)
        self._end_walk(now, reason="settled" if now < self._walk_until else "budget",
                       gained=gained)
        if gained >= walk.min_progress_px:
            # 确实靠近了：不算撞墙，立刻允许再点一次继续接近
            self._stuck_ticks = 0
            self._last_click_ts = 0.0
            self.stats["walk_progress"] += 1
        else:
            self.stats["walk_no_progress"] += 1
        return False

    def _walk_wait_active(self, scene: SceneState, now: float) -> bool:
        """SCANNING 里的走位等待（没有目标可量距离，只看时间与画面动静）。

        探索/点地面走路时同样不该每秒重发同一指令。
        """
        if not self._walking:
            return False
        if now >= self._walk_until:
            self._end_walk(now, reason="budget")
            return False
        if scene.frame_delta >= self.cfg.walk.motion_delta_thresh:
            return True
        self._end_walk(now, reason="settled")   # 人停了，不用等满预算
        return False

    # ------------- 走位动作 -------------

    def _move_interval_s(self, now: float) -> float:
        """两次走位动作的间隔：基准节拍 × 抖动 × 疲劳倍数 × 监督层降速。

        监督层判 throttle 时这里会变慢——降速是它能做的三种干预里最轻的一种，
        好处是即使判错也只是慢一点，不会让机器人做错事。
        """
        base = 1.0 / max(0.5, self.cfg.tick_hz)
        return (base * random.uniform(0.9, 1.1) * self.human.latency_scale(now)
                * self.supervisor.pace_factor(now))

    def _move_toward(self, scene: SceneState, target: Pillar, now: float, first: bool) -> None:
        self._last_click_ts = now
        dx, dy = target.cx - scene.character[0], target.y - scene.character[1]
        self._intent(f"向{_dir_text(dx, dy)}移动接近{target.color}掉落光柱（距离{round(target.distance)}px）")
        if self.cfg.approach_mode == "mouse_click":
            # 点击光柱下方地面（对话定稿：光柱悬空，点地面人物才会走过去）。
            # 只发单条 mouse_click(x,y)：RP2040/RPi5 后端内部先移动后点击。
            gx = int(target.x + random.randint(-8, 8))
            gy = int(target.y + self.cfg.ground_click_offset_px + random.randint(-6, 6))
            # 疲劳失误：偶尔点到偏离目标的地面（走错一步，下一拍自己纠正）。
            # 只在**地面点击**上做手脚 —— 绝不涉及按键，铁律不受影响。
            if self.human.error_prob(now) > random.random():
                off = random.choice((-1, 1)) * random.randint(40, 110)
                gx = max(0, min(scene.w - 1, gx + off))
                self.stats["misclicks"] = self.stats.get("misclicks", 0) + 1
                self._emit_event("human_misclick", x=gx, y=gy)
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
        # 发起移动 → 进入"等人物走过去"（点一下不等于走到位）
        self._begin_walk(now, target)

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
        # 按压时长：抖动比例随疲劳放大（越累手越不稳）
        hold = _jitter_ms(70, self.human.jitter_ratio(now, self.cfg.jitter_ratio))
        self._act(Command(op="key_tap", key=self.cfg.pickup_key, hold_ms=hold))
        self._last_tap_ts = now
        # 登记预期：按完这一下，A框该消失（或拾取数该涨）。连续落空= 在捡空气，
        # 这正是"结果是否正向"最直接的证据。
        self.supervisor.predict(
            "pick", "按拾取键后 A框应消失", base=float(self.stats["picks"]),
            now=now, horizon_s=max(1.0, self.cfg.tap_recheck_ms / 1000.0 + 1.0))
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
            if self._target is not None:
                # 记入"新鲜感"记忆：刚捡过的颜色/位置下次会被打折
                self._note_recent_pick(self._target, now)
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
        self._aframe_seen_frames = 0
        self._aframe_dist0 = 0.0
        self._taps_this_target = 0
        self._tap_block_until_ms = 0
        self._walking = False   # 目标没了，走位等待一并作废

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
            "supervisor": self.supervisor.snapshot(),
            "arbiter": self.arbiter.snapshot(),
            "humanize": self.human.snapshot(),
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
