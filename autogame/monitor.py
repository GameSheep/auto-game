"""上线之后谁在盯着 —— 性能监控与自动退化回滚。

对话里那条闭环的最后两段是「性能监控 → 退化回滚」。`evolve.promote` 只管
**上线前**：新一代在静态台账上不能比旧代差。可是上线之后坏掉的方式完全不同，
而且**没有一种会让台账变红** —— 台账是固定的那几十张图，图没变、判决也没变：

| 坏法 | 现场长什么样 | 台账为什么看不出来 |
|---|---|---|
| 卡住 | 同一个动作一秒发十几次：撞墙、打不动的怪、反复点一个按钮 | 图还是那张图，判断还是对的 |
| 变慢 | 每 tick 耗时翻倍，决策跟不上帧率 | 回归是干跑，没有真实负载 |
| 烧模型 | 某条规则永远匹配不上，每帧都在问模型 | 回归根本不调模型 |
| 空转 | 恢复次数暴涨，可一个东西也没捡到 | 恢复「成功执行」了，只是没用 |

所以这层必须存在，而且必须**自己**能动手。等人在群里看见「它卡了半小时」
再回滚，代价是半小时的无效操作。

## 四条约束

1. **热路径只做加法。** `observe()` 每 tick 调一次，读计数器做差分、攒着；
   一个窗口（默认 20 秒）才结算一次，排序、判断都在结算里。
2. **基线不许被退化拉低。** 基线只在「这一窗口判定健康」时才缓慢更新。
   否则连续退化会把基线也拖下去，越坏越觉得正常 —— 这是监控版的「越学越坏」。
3. **连续几个坏窗口才动手，动手后冷却。** 一帧抖动就回滚的机器人没法用；
   回滚完立刻又回滚，会把基线也退没了，最后退到没有上一代。
4. **回滚不是目的，止损才是。** 回滚之后清空基线重新采 —— 配方换了，
   旧基线描述的已经不是现在这套东西了。
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: 只是提个醒，不动手。
WARN = "warn"
#: 要动手的（连续够次数就回滚）。
BAD = "bad"

#: 要跟的计数器。都来自 `fsm.stats` / `executor.sent`，缺了当 0 算。
TRACKED = ("picks", "recoveries", "recovery_actions", "llm_calls", "sent")

_DEFAULTS: dict[str, Any] = {
    "window_s": 20.0,      # 结算窗口长度
    "warmup": 3,           # 用前几个健康窗口定基线（3×20s = 1 分钟）
    "min_bad": 3,          # 连续几个坏窗口才回滚（又是 1 分钟）
    "cooldown_s": 600.0,   # 回滚之后多久不再回滚
    "min_ticks": 3,        # 一个窗口至少这么多 tick 才拿来定基线（防空转把基线拉平）
    "repeat_limit": 12,    # 同一动作在一个窗口里重复多少次算卡住
    "slow_ratio": 2.0,     # 每 tick 耗时涨到基线的几倍算变慢
    "slow_floor_s": 0.4,   # 且绝对值要到这个数（10ms → 20ms 不该报警）
    "burn_ratio": 3.0,     # 模型调用频率涨到基线的几倍算烧
    "burn_floor": 0.1,     # 且每秒至少这么多次
    "spin_ratio": 3.0,     # 恢复频率涨到基线的几倍算空转
    "spin_floor": 0.02,    # 且每秒至少这么多次
    "keep_windows": 60,    # 留多少个窗口的明细（页面/复盘用）
}


@dataclass
class Window:
    """一个结算窗口的指标快照。"""

    ts: float
    seconds: float
    ticks: int = 0
    tick_s: float = 0.0                 # 平均每 tick 耗时（秒）
    repeat: int = 0                     # 窗口内最热的那个动作重复了几次
    repeat_key: str = ""
    rates: dict[str, float] = field(default_factory=dict)   # 每秒计数

    def to_dict(self) -> dict:
        return {"ts": round(self.ts, 1), "seconds": round(self.seconds, 2),
                "ticks": self.ticks, "tick_ms": round(self.tick_s * 1000, 1),
                "repeat": self.repeat, "repeat_key": self.repeat_key,
                "rates": {k: round(v, 3) for k, v in self.rates.items()}}


@dataclass
class Finding:
    """一条体检结论。`code` 是对外的稳定标识，`detail` 是给人看的。"""

    code: str
    level: str
    label: str
    detail: str
    metric: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"code": self.code, "level": self.level, "label": self.label,
                "detail": self.detail, "metric": self.metric,
                "ts": round(self.ts, 1)}


def _action_key(it: dict) -> str:
    """把一条动作折成"同一个动作"。坐标按 40px 网格归并 —— 点在同一个按钮上
    每次差几个像素，人是同一个动作，代码里是三个不同的 (x, y)。"""
    op = str(it.get("op") or "?")
    if op in ("mouse_click", "mouse_move"):
        x = int(it.get("x") or 0) // 40 * 40
        y = int(it.get("y") or 0) // 40 * 40
        return f"{op}({x},{y})"
    if op.startswith("key"):
        return f"{op}({it.get('key') or ''})"
    return op


def recent_repeat(journal: list, since: float, *, limit: int = 500) -> tuple[int, str]:
    """`since` 之后最热的那个动作重复了几次。返回 (次数, 动作键)。

    抽成模块级函数是为了**只留一份实现**：体检用它在窗口结算时判"卡住"，
    裁决层用它在 probe 里给规则一个 `repeat` 事实。两处各写一份，迟早会出现
    "体检说它卡住了，规则里的 repeat 还是 3"这种自相矛盾的现场。

    只看最近 `limit` 条：journal 是环形的，全量扫一遍在跑久了之后是白费。
    """
    counts: dict[str, int] = {}
    for it in list(journal or [])[-limit:]:
        if float(it.get("ts") or 0) < since:
            continue
        if it.get("suppressed"):      # 被模式闸门拦下的不算"它在乱按"
            continue
        key = _action_key(it)
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return 0, ""
    key, n = max(counts.items(), key=lambda kv: kv[1])
    return n, key


def _median(values: list[float]) -> float:
    rows = sorted(float(v) for v in values)
    if not rows:
        return 0.0
    mid = len(rows) // 2
    return rows[mid] if len(rows) % 2 else (rows[mid - 1] + rows[mid]) / 2


class HealthMonitor:
    """运行时体检。**不 import evolve、不写盘**，回滚动作由调用方给。"""

    def __init__(self, **kw: Any) -> None:
        o = {**_DEFAULTS, **kw}
        self.window_s = float(o["window_s"])
        self.warmup = int(o["warmup"])
        self.min_bad = int(o["min_bad"])
        self.cooldown_s = float(o["cooldown_s"])
        self.min_ticks = int(o["min_ticks"])
        self.repeat_limit = int(o["repeat_limit"])
        self.slow_ratio = float(o["slow_ratio"])
        self.slow_floor_s = float(o["slow_floor_s"])
        self.burn_ratio = float(o["burn_ratio"])
        self.burn_floor = float(o["burn_floor"])
        self.spin_ratio = float(o["spin_ratio"])
        self.spin_floor = float(o["spin_floor"])

        self._since = 0.0
        self._acc: dict[str, int] = {}
        self._tick_cost: list[float] = []
        self._last_stats: dict[str, int] = {}
        self._windows: deque[Window] = deque(maxlen=int(o["keep_windows"]))
        self._warm: list[Window] = []
        self._base: dict[str, float] | None = None
        self._bad_streak = 0
        self._last_bad: Finding | None = None
        self._last_findings: list[Finding] = []
        self._rolled_at = 0.0
        self._rollback_reason = ""

    # ---------------- 采 ----------------

    def observe(self, *, stats: dict | None = None, journal: list | None = None,
                tick_s: float | None = None, now: float | None = None
                ) -> list[Finding]:
        """每 tick 调一次。返回**这一 tick 结算出的结论**（绝大多数时候是空表）。"""
        now = time.time() if now is None else now
        cur = {k: int((stats or {}).get(k) or 0) for k in TRACKED}

        if not self._since:                     # 第一次：只记起点
            self._since = now
            self._last_stats = cur
            self._acc = {k: 0 for k in TRACKED}
            return []

        self._acc["ticks"] = self._acc.get("ticks", 0) + 1
        if tick_s is not None:
            self._tick_cost.append(max(0.0, float(tick_s)))
        for k in TRACKED:
            d = cur[k] - self._last_stats.get(k, cur[k])
            # 计数器被重置（换执行器/重启 FSM）时差值为负：当 0，不要让负数污染窗口
            self._acc[k] = self._acc.get(k, 0) + (d if d > 0 else 0)
        self._last_stats = cur

        span = now - self._since
        if span < self.window_s:
            return []

        w = Window(ts=now, seconds=span, ticks=int(self._acc.get("ticks") or 0),
                   tick_s=(sum(self._tick_cost) / len(self._tick_cost))
                   if self._tick_cost else 0.0)
        w.rates = {k: v / span for k, v in self._acc.items() if k != "ticks"}
        if journal:
            w.repeat, w.repeat_key = self._top_repeat(journal, now - span)

        self._since = now
        self._acc = {k: 0 for k in TRACKED}
        self._tick_cost.clear()

        self._windows.append(w)
        findings = self._judge(w)
        self._last_findings = findings
        if any(f.level == BAD for f in findings):
            self._bad_streak += 1
            self._last_bad = next(f for f in findings if f.level == BAD)
        else:
            self._bad_streak = 0
            self._learn(w)
        for f in findings:
            log.warning("体检发现 %s（%s）：%s", f.label, f.code, f.detail)
        return findings

    def _top_repeat(self, journal: list, since: float) -> tuple[int, str]:
        return recent_repeat(journal, since)

    # ---------------- 判 ----------------

    def _judge(self, w: Window) -> list[Finding]:
        out: list[Finding] = []
        if w.repeat >= self.repeat_limit:
            out.append(Finding(
                "repeat", BAD, "卡住了",
                f"{w.seconds:.0f} 秒里「{w.repeat_key}」发了 {w.repeat} 次，"
                "位置和参数都没变 —— 多半是撞墙、打不动的怪，或者反复点同一个按钮",
                {"repeat": w.repeat, "key": w.repeat_key}))

        b = self._base
        if not b:
            return out

        if w.tick_s and w.tick_s >= max(self.slow_floor_s, b.get("tick_s", 0) * self.slow_ratio):
            out.append(Finding(
                "slow", BAD, "变慢了",
                f"每 tick 耗时从 {b.get('tick_s', 0) * 1000:.0f}ms 涨到 "
                f"{w.tick_s * 1000:.0f}ms，同样的画面要花更多时间才算完",
                {"now_ms": round(w.tick_s * 1000, 1),
                 "base_ms": round(b.get("tick_s", 0) * 1000, 1)}))

        llm = w.rates.get("llm_calls", 0.0)
        if llm >= max(self.burn_floor, b.get("llm_calls", 0.0) * self.burn_ratio):
            out.append(Finding(
                "model_burn", BAD, "在烧模型",
                f"每秒问模型 {llm:.2f} 次（之前 {b.get('llm_calls', 0.0):.2f} 次）—— "
                "多半是某条规则永远匹配不上，每帧都在兜底问模型",
                {"now": round(llm, 3), "base": round(b.get("llm_calls", 0.0), 3)}))

        rec = w.rates.get("recoveries", 0.0)
        picks = w.rates.get("picks", 0.0)
        if rec >= max(self.spin_floor, b.get("recoveries", 0.0) * self.spin_ratio) and picks <= 0:
            out.append(Finding(
                "recovery_spin", WARN, "在恢复里打转",
                f"每秒进恢复 {rec:.2f} 次，可这 {w.seconds:.0f} 秒一个都没捡到 —— "
                "恢复动作在「成功执行」，但没有解决任何问题",
                {"now": round(rec, 3), "base": round(b.get("recoveries", 0.0), 3),
                 "picks": round(picks, 3)}))
        return out

    def _learn(self, w: Window) -> None:
        """健康窗口才喂基线。空转窗口（ticks 太少）也不喂 —— 否则暂停十分钟就把
        基线拉平了，之后正常的模型调用会看起来像"暴涨"。"""
        if w.ticks < self.min_ticks:
            return
        sample = {"tick_s": w.tick_s, **{k: w.rates.get(k, 0.0) for k in TRACKED}}
        if self._base is None:
            self._warm.append(w)
            if len(self._warm) < self.warmup:
                return
            self._base = {k: _median([getattr(x, "tick_s", 0.0) if k == "tick_s"
                                      else x.rates.get(k, 0.0) for x in self._warm])
                          for k in sample}
            self._warm.clear()
            log.info("体检基线已建立：%s",
                     {k: round(v, 3) for k, v in self._base.items()})
            return
        # 缓慢跟随（0.2）：配置/关卡变了要能适应，但不能一夜之间漂过去
        for k, v in sample.items():
            self._base[k] = 0.8 * float(self._base.get(k, v)) + 0.2 * float(v)

    # ---------------- 动手 ----------------

    def should_rollback(self, now: float | None = None) -> Finding | None:
        """该不该回滚。**连续** `min_bad` 个坏窗口才点头，且回滚后有冷却。"""
        now = time.time() if now is None else now
        if self._last_bad is None or self._bad_streak < self.min_bad:
            return None
        if self._rolled_at and now - self._rolled_at < self.cooldown_s:
            return None
        return self._last_bad

    def mark_rollback(self, finding: Finding | None = None, *,
                      now: float | None = None) -> None:
        """记一次回滚：进入冷却，并且**清空基线**重新采。

        配方已经换了，旧基线描述的不是现在这套东西；拿旧基线继续比，
        回滚之后还会立刻"发现变慢"，然后把基线也退没了。
        """
        now = time.time() if now is None else now
        self._rolled_at = now
        self._bad_streak = 0
        self._last_bad = None
        self._base = None
        self._warm.clear()
        self._rollback_reason = finding.code if finding else ""
        log.warning("体检触发回滚（%s），冷却 %d 秒", self._rollback_reason,
                    int(self.cooldown_s))

    # ---------------- 对外 ----------------

    def status(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        left = (max(0.0, self.cooldown_s - (now - self._rolled_at))
                if self._rolled_at else 0.0)
        return {
            "ok": self._bad_streak < self.min_bad,
            "bad_streak": self._bad_streak,
            "min_bad": self.min_bad,
            "window_s": self.window_s,
            "baseline": ({k: round(v, 4) for k, v in self._base.items()}
                         if self._base else None),
            "baseline_ready": self._base is not None,
            "windows": [w.to_dict() for w in list(self._windows)[-3:]],
            "findings": [f.to_dict() for f in self._last_findings],
            "rolled_at": round(self._rolled_at, 1) or None,
            "rollback_reason": self._rollback_reason,
            "cooldown_left_s": round(left, 1),
        }

    def reset(self) -> None:
        """重新开始采（换了配方、换了关卡之后调）。"""
        self._since = 0.0
        self._acc = {}
        self._tick_cost.clear()
        self._windows.clear()
        self._warm.clear()
        self._base = None
        self._bad_streak = 0
        self._last_bad = None
        self._last_findings = []


def incident_path(profile: Any) -> Path:
    """退化事件落到哪：`profiles/<id>/health/incidents.jsonl`。

    为什么不塞进 `logs/`：它回答的是"这一版配方上线后出过什么事"，
    回退配方时要能一起看见 —— 它属于 profile 的历史，不属于机器的日志。
    """
    return Path(profile.root) / "health" / "incidents.jsonl"


def append_incident(path: Path, row: dict) -> None:
    """追加一条事件。**不能因为写日志失败而让回滚回滚不掉**，所以异常吞掉但记。"""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("a", encoding="utf-8") as f:
            f.write(json.dumps({**row, "ts": round(time.time(), 1)},
                               ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        log.debug("退化事件写入失败", exc_info=True)
