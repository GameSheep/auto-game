"""自主恢复 —— 进度看门狗 + 分级恢复阶梯。

搬砖机器人真正的难点不是"能捡一次"，而是"连续跑 8 小时不卡住"。
本模块解决两件事：

1. **ProgressMonitor（看门狗）**：判断"现在是不是卡了、卡在哪一层"。
   - frozen  ：画面完全不动（撞墙/窗口失焦/游戏假死）
   - idle    ：画面在动但没有任何可交互目标（房间走完了/该换区了）
   - stalled ：有目标但长时间没有任何进展（走不到、捡不到、来回绕圈）
   - gated   ：长时间不在战斗界面（弹窗/菜单/结算界面没退出去）
   - loading ：加载/黑屏过场（正常等待，不算卡）

2. **RecoveryLadder（恢复阶梯）**：由轻到重换招，**永不放弃**。
   重定位 → 四向探路 → ESC 关弹窗 → 脱困键 → 重新聚焦 → 八向脱困
   → 重开关卡 → 视觉模型看一眼 → 文本模型换策略 → 循环（带降速）

安全铁律（和主决策一致，代码强制）：
  - 恢复动作里**永远不出现拾取键** —— 拾取只能由 A框检测裁决；
  - 所有按键/坐标都过白名单与边界钳制；
  - 每级动作有最小间隔，避免"卡住→疯狂乱按"这种更糟的失败模式。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..config import AutonomyConfig
from ..protocol import Command, SceneState

log = logging.getLogger(__name__)

# 八方向（屏幕坐标系：y 向下）
_SPIN_KEYS: list[list[str]] = [
    ["w"], ["w", "d"], ["d"], ["s", "d"], ["s"], ["s", "a"], ["a"], ["w", "a"],
]

_DIR_VEC: dict[str, tuple[int, int]] = {
    "w": (0, -1), "s": (0, 1), "a": (-1, 0), "d": (1, 0),
    "wd": (1, -1), "dw": (1, -1), "wa": (-1, -1), "aw": (-1, -1),
    "sd": (1, 1), "ds": (1, 1), "sa": (-1, 1), "as": (-1, 1),
}


def direction_vec(key: str) -> tuple[int, int]:
    """方向键 → 屏幕位移向量（y 向下）。"""
    return _DIR_VEC.get((key or "").lower(), (0, 0))


@dataclass
class StallSignals:
    """一帧的卡壳信号（全部单位=秒，基于帧时间戳）。"""

    verdict: str = "ok"          # ok / frozen / idle / stalled / gated / loading
    frozen_for: float = 0.0
    idle_for: float = 0.0
    stall_for: float = 0.0
    gate_for: float = 0.0
    targets: int = 0
    nearest: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "frozen_for": round(self.frozen_for, 1),
            "idle_for": round(self.idle_for, 1),
            "stall_for": round(self.stall_for, 1),
            "gate_for": round(self.gate_for, 1),
            "targets": self.targets,
            "nearest": None if self.nearest is None else round(self.nearest),
        }


class ProgressMonitor:
    """进度看门狗（纯逻辑，帧时间戳驱动，可确定性测试）。"""

    def __init__(self, cfg: AutonomyConfig) -> None:
        self.cfg = cfg
        self._last_motion = 0.0
        self._last_target = 0.0
        self._last_progress = 0.0
        self._last_action = 0.0
        self._gated_since: float | None = None
        self._nearest: float | None = None
        self._picks = 0
        self._interactions = 0
        self.last = StallSignals()

    def reset(self, now: float) -> None:
        self._last_motion = now
        self._last_target = now
        self._last_progress = now
        self._last_action = 0.0
        self._gated_since = None
        self._nearest = None
        self.last = StallSignals()

    def note_action(self, now: float) -> None:
        """登记"我确实发出了动作"（用于区分"卡住"与"没事干"）。"""
        self._last_action = now

    def note_progress(self, now: float, kind: str = "") -> None:
        """外部确认的"有效进展"（拾取/点击列表/再次挑战/攻击命中…）。"""
        self._last_progress = now
        self._interactions += 1
        if kind:
            log.debug("progress(%s) @%.1fs", kind, now)

    def update(self, scene: SceneState, now: float, picks: int) -> StallSignals:
        cfg = self.cfg
        targets = (
            len(scene.pillars) + len(scene.item_list) + len(scene.ground_labels)
            + len(scene.monsters) + (1 if scene.rechallenge else 0)
        )
        if targets > 0:
            self._last_target = now

        # 拾取数增加 = 最硬的进展证据
        if picks > self._picks:
            self._picks = picks
            self._last_progress = now

        # 最近目标距离明显缩短 = 走位有效（避免"在走但走不到"被判成进展）
        nearest = min((p.distance for p in scene.pillars), default=None)
        if nearest is not None and self._nearest is not None and nearest < self._nearest - 15:
            self._last_progress = now
        self._nearest = nearest

        if scene.loading:
            self._gated_since = None
            self.last = StallSignals(verdict="loading", targets=targets, nearest=nearest)
            return self.last

        if not scene.in_game:
            if self._gated_since is None:
                self._gated_since = now
            gate_for = now - self._gated_since
            verdict = "gated" if gate_for >= cfg.gate_window_s else "gate_wait"
            self.last = StallSignals(verdict=verdict, gate_for=gate_for,
                                     targets=targets, nearest=nearest)
            return self.last
        self._gated_since = None

        if scene.frame_delta >= cfg.motion_delta_thresh:
            self._last_motion = now

        frozen_for = now - self._last_motion
        idle_for = now - self._last_target
        stall_for = now - self._last_progress

        # frozen 的前提是"我们确实动过手却没反应"：
        # 只是站着没事干（空闲）不该算卡死，否则空闲探索会被误判成卡壳。
        acted_since_motion = self._last_action >= self._last_motion
        if stall_for >= cfg.stall_window_s:
            verdict = "stalled"
        elif frozen_for >= cfg.frozen_window_s and acted_since_motion:
            verdict = "frozen"
        elif idle_for >= cfg.idle_window_s:
            verdict = "idle"
        else:
            verdict = "ok"

        self.last = StallSignals(verdict=verdict, frozen_for=frozen_for,
                                 idle_for=idle_for, stall_for=stall_for,
                                 targets=targets, nearest=nearest)
        return self.last


@dataclass
class RecoveryResult:
    step: str = ""
    level: int = 0
    cycle: int = 0
    commands: list[Command] = field(default_factory=list)
    note: str = ""
    cycle_done: bool = False
    new_level: bool = False


class RecoveryLadder:
    """分级恢复阶梯（由轻到重，循环重试，永不放弃）。"""

    STEPS = ["retarget", "probe", "escape", "dodge", "refocus",
             "spin", "stage_reset", "vlm", "llm"]

    def __init__(self, cfg: AutonomyConfig, vlm=None, advisor=None,  # noqa: ANN001
                 forbidden_keys: set[str] | None = None) -> None:
        self.cfg = cfg
        self.vlm = vlm
        self.advisor = advisor
        self.forbidden = {k.lower() for k in (forbidden_keys or set())}
        self.level = 0
        self.cycle = 0
        self.steps_done = 0
        self._level_started: float | None = None
        self._last_action = 0.0
        self._probe_i = 0
        self._spin_i = 0
        self._vlm_requested = False
        self._llm_requested = False
        self._escape_taps = 0
        self.last_step = ""

    # ------------- 生命周期 -------------

    def reset(self) -> None:
        self.level = 0
        self.cycle = 0
        self.steps_done = 0
        self._level_started = None
        self._last_action = 0.0
        self._probe_i = 0
        self._spin_i = 0
        self._vlm_requested = False
        self._llm_requested = False
        self._escape_taps = 0
        self.last_step = ""

    def begin(self, now: float) -> None:
        self.level = 0
        self._level_started = now
        self._last_action = 0.0
        self._vlm_requested = False
        self._llm_requested = False
        self._probe_i = 0
        self._spin_i = 0
        self._escape_taps = 0

    def step_name(self) -> str:
        return self.STEPS[min(self.level, len(self.STEPS) - 1)]

    # ------------- 主入口 -------------

    def step(self, now: float, scene: SceneState, ctx: dict[str, Any] | None = None) -> RecoveryResult:
        ctx = ctx or {}
        if self._level_started is None:
            self.begin(now)

        # 升级：每 level_interval_s 换更重的手段；走完一轮 → 新循环
        cycle_done = False
        new_level = False
        if now - (self._level_started or now) >= self.cfg.level_interval_s:
            self._level_started = now
            self.level += 1
            new_level = True
            if self.level >= len(self.STEPS):
                self.level = 0
                self.cycle += 1
                cycle_done = True
                self._vlm_requested = False
                self._llm_requested = False
                self._escape_taps = 0

        name = self.step_name()
        # 同一步骤内节流：避免一秒内连发同一条恢复动作
        if not new_level and self._last_action and now - self._last_action < self.cfg.step_interval_s:
            return RecoveryResult(step=name, level=self.level, cycle=self.cycle,
                                  cycle_done=cycle_done, new_level=new_level)

        result = getattr(self, f"_step_{name}")(now, scene, ctx)
        result.level = self.level
        result.cycle = self.cycle
        result.cycle_done = cycle_done
        result.new_level = new_level
        if result.commands or result.note:
            self._last_action = now
        self.last_step = name
        self.steps_done += 1
        result.commands = self._safe(result.commands)
        return result

    # ------------- 各步骤 -------------

    def _step_retarget(self, now: float, scene: SceneState, ctx: dict) -> RecoveryResult:
        """重定位：清拉黑、丢目标、重置探索序列（纯内部动作，不发命令）。"""
        return RecoveryResult(step="retarget", note="清空拉黑与当前目标，重新扫描")

    def _step_probe(self, now: float, scene: SceneState, ctx: dict) -> RecoveryResult:
        """四向探路：朝角色四周点一下，找出"哪个方向能走"。"""
        ch = ctx.get("character") or scene.character
        offs = self.cfg.probe_offsets or [[0, -140], [140, 0], [0, 140], [-140, 0]]
        dx, dy = offs[self._probe_i % len(offs)]
        self._probe_i += 1
        x = _clamp(int(ch[0]) + int(dx), 0, scene.w - 1)
        y = _clamp(int(ch[1]) + int(dy), 0, scene.h - 1)
        return RecoveryResult(
            step="probe", commands=[Command(op="mouse_click", button="left", x=x, y=y)],
            note=f"探路点击 ({x},{y})",
        )

    def _step_escape(self, now: float, scene: SceneState, ctx: dict) -> RecoveryResult:
        """ESC：关掉可能挡住操作的弹窗/菜单（最多连按 3 次，防在菜单里反复开关）。"""
        if self._escape_taps >= 3:
            return RecoveryResult(step="escape", note="ESC 已按 3 次，跳过")
        self._escape_taps += 1
        return RecoveryResult(
            step="escape",
            commands=[Command(op="key_tap", key=self.cfg.escape_key, hold_ms=60)],
            note=f"按 {self.cfg.escape_key} 关闭弹窗/菜单（第 {self._escape_taps} 次）",
        )

    def _step_dodge(self, now: float, scene: SceneState, ctx: dict) -> RecoveryResult:
        """脱困键（空格=闪避/位移）：卡在墙角时常常一下就能出来。"""
        key = (self.cfg.dodge_key or "").strip().lower()
        if not key:
            return RecoveryResult(step="dodge", note="未配置脱困键，跳过")
        cmds = [Command(op="key_tap", key=key, hold_ms=90)
                for _ in range(max(1, self.cfg.dodge_taps))]
        return RecoveryResult(step="dodge", commands=cmds,
                              note=f"按脱困键 {key} ×{len(cmds)}")

    def _step_refocus(self, now: float, scene: SceneState, ctx: dict) -> RecoveryResult:
        """重新聚焦：点一下角色（不移动），激活游戏窗口/取消奇怪的选择态。"""
        ch = ctx.get("character") or scene.character
        x = _clamp(int(ch[0]), 0, scene.w - 1)
        y = _clamp(int(ch[1]), 0, scene.h - 1)
        return RecoveryResult(
            step="refocus",
            commands=[Command(op="mouse_move", x=x, y=y),
                      Command(op="mouse_click", button="left", x=x, y=y)],
            note="点击角色重新聚焦窗口",
        )

    def _step_spin(self, now: float, scene: SceneState, ctx: dict) -> RecoveryResult:
        """八向脱困：沿八个方向依次短走，总能找到一条能走的路。

        走位方式跟随主决策：mouse_click 模式用"点地面"（**绝不能按拾取键** ——
        很多游戏拾取键与左移同键，按键盘走位会变成盲捡）。
        """
        keys = _SPIN_KEYS[self._spin_i % len(_SPIN_KEYS)]
        self._spin_i += 1
        if ctx.get("approach_mode") == "mouse_click":
            ch = ctx.get("character") or scene.character
            dx, dy = direction_vec("".join(keys))
            x = _clamp(int(ch[0]) + dx * 180, 0, scene.w - 1)
            y = _clamp(int(ch[1]) + dy * 180, 0, scene.h - 1)
            return RecoveryResult(step="spin",
                                  commands=[Command(op="mouse_click", button="left", x=x, y=y)],
                                  note=f"八向脱困点击 ({x},{y})")
        hold = _clamp(int(self.cfg.explore_step_ms * 0.6), 120, 900)
        cmds = [Command(op="key_tap", key=k, hold_ms=hold) for k in keys]
        return RecoveryResult(step="spin", commands=cmds,
                              note=f"八向脱困：{''.join(keys)}")

    def _step_stage_reset(self, now: float, scene: SceneState, ctx: dict) -> RecoveryResult:
        """重开关卡：检测到『再次挑战』就点；没有就交给下一级（VLM 看画面）。"""
        btn = scene.rechallenge
        if btn is None:
            return RecoveryResult(step="stage_reset", note="未见『再次挑战』按钮，跳过")
        x = _clamp(int(btn.x), 0, scene.w - 1)
        y = _clamp(int(btn.y), 0, scene.h - 1)
        return RecoveryResult(
            step="stage_reset",
            commands=[Command(op="mouse_click", button="left", x=x, y=y)],
            note=f"点击『再次挑战』重开关卡 ({x},{y})",
        )

    def _step_vlm(self, now: float, scene: SceneState, ctx: dict) -> RecoveryResult:
        """视觉模型：截图问"我在哪、该往哪走"，按返回动作执行。"""
        if self.vlm is None or not getattr(self.vlm.cfg, "enabled", False):
            return RecoveryResult(step="vlm", note="VLM 未启用，跳过")
        if not self._vlm_requested:
            self._vlm_requested = True
            req_ctx = {
                "state": ctx.get("state", ""),
                "signals": ctx.get("signals", {}),
                "stats": ctx.get("stats", {}),
                "screen_w": scene.w,
                "screen_h": scene.h,
                "recent_events": ctx.get("recent_events", []),
            }
            if self.vlm.request(req_ctx, now=time.monotonic()):
                return RecoveryResult(step="vlm", note="已请求视觉模型分析画面")
            return RecoveryResult(step="vlm", note="视觉模型预算不足/不可用，跳过")

        advice = self.vlm.take_advice()
        if not advice:
            return RecoveryResult(step="vlm", note="等待视觉模型返回")
        if not advice.get("valid"):
            return RecoveryResult(step="vlm",
                                  note=f"视觉建议未采纳：{advice.get('why') or advice.get('action')}")
        cmds = self._vlm_commands(advice, scene, ctx)
        return RecoveryResult(step="vlm", commands=cmds,
                              note=f"视觉模型建议 {advice.get('action')}：{advice.get('reason', '')}")

    def _vlm_commands(self, advice: dict, scene: SceneState,
                      ctx: dict | None = None) -> list[Command]:
        action = advice.get("action")
        if action == "walk_to":
            x, y = advice.get("target", [scene.w // 2, scene.h // 2])
            return [Command(op="mouse_click", button="left", x=int(x), y=int(y))]
        if action == "key":
            return [Command(op="key_tap", key=str(advice.get("key", "space")), hold_ms=120)]
        if action == "esc":
            return [Command(op="key_tap", key=self.cfg.escape_key, hold_ms=60)]
        if action == "explore":
            keys = _SPIN_KEYS[self._spin_i % len(_SPIN_KEYS)]
            self._spin_i += 1
            if (ctx or {}).get("approach_mode") == "mouse_click":
                ch = (ctx or {}).get("character") or scene.character
                dx, dy = direction_vec("".join(keys))
                return [Command(op="mouse_click", button="left",
                                x=_clamp(int(ch[0]) + dx * 180, 0, scene.w - 1),
                                y=_clamp(int(ch[1]) + dy * 180, 0, scene.h - 1))]
            return [Command(op="key_tap", key=k, hold_ms=400) for k in keys]
        if action == "reset_stage":
            btn = scene.rechallenge
            if btn is not None:
                return [Command(op="mouse_click", button="left",
                                x=_clamp(int(btn.x), 0, scene.w - 1),
                                y=_clamp(int(btn.y), 0, scene.h - 1))]
        return []  # wait / 无法执行

    def _step_llm(self, now: float, scene: SceneState, ctx: dict) -> RecoveryResult:
        """文本模型：换策略（探索/清怪/等待/重开），不直接给坐标。"""
        if self.advisor is None:
            return RecoveryResult(step="llm", note="LLM 未配置，跳过")
        if self._llm_requested:
            return RecoveryResult(step="llm", note="已咨询 LLM")
        try:
            ready = self.advisor.should_consult(ignore_abandons=True)
        except TypeError:  # 兼容旧签名
            ready = self.advisor.should_consult()
        if not ready:
            return RecoveryResult(step="llm", note="LLM 触发条件未满足，跳过")
        self._llm_requested = True
        summary = {
            "state": ctx.get("state", ""),
            "trigger": "recovery_stalled",
            "signals": ctx.get("signals", {}),
            "stats": ctx.get("stats", {}),
            "recent_events": ctx.get("recent_events", []),
            "hint": "机器人已尝试探路/ESC/脱困键/八向移动仍无进展，请给策略建议",
        }
        self.advisor.consult(summary, on_advice=ctx.get("on_llm_advice"))
        return RecoveryResult(step="llm", note="已咨询 LLM 策略建议")

    # ------------- 安全过滤 -------------

    def _safe(self, cmds: list[Command]) -> list[Command]:
        """安全过滤：禁止拾取键、禁止越界坐标、禁止空 key。"""
        out: list[Command] = []
        for c in cmds:
            if c.op in ("key_tap", "key_down", "key_up"):
                k = (c.key or "").lower()
                if not k or k in self.forbidden:
                    log.warning("恢复动作被安全过滤：按键 %s 禁止", k)
                    continue
            if c.op in ("mouse_click", "mouse_move"):
                if c.x is None or c.y is None:
                    continue
            out.append(c)
        return out


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))
