"""LLM 监督层 —— 判断快通道的"结果是否正向"，并处理突发局面。

为什么需要它（以及为什么不能让它当驾驶员）：
  OpenCV+FSM 是快通道：1Hz、确定性、每帧必跑，但只会回答"有没有光柱/A框"，
  回答不了"我这一串动作到底有没有让局面变好"。规则覆盖不到的地方（弹窗、
  死亡、假走位、目标来回抖）它会一直按，而且按得理直气壮。

  但把 1Hz 的控制权直接交给大模型是错的，三个原因：
    1. 延迟：本地 7B 一次 1~3s，云端 2~15s，每帧都问会把 1Hz 变成 0.3Hz；
    2. 成本：一小时 3600 帧；
    3. **方差**：视觉阈值确定、可回归；LLM 每帧给的答案会飘。拿一个飘的东西
       逐帧控制，比开环更差。

  所以这里的定位是：**断路器 + 审计员，不是驾驶员。**

三条硬约束（缺一条这个层就不该开）：
  1. 只做减法：allow / throttle / hold / takeover / panic，**永远不给目标坐标**。
     去哪、捡哪个，只能由像素证据裁决。松掉这条就是"模型幻觉驱动机器人"，
     而且完全无法回归测试。
  2. 判断"正向"用数值对账，不用截图：每次关键动作登记一条 prediction
     （走位→距离应缩短、拾取→A框应消失、ESC→应回到战斗），到期用实测值结算
     hit/miss。喂给模型的是一串可核对的数字，不是"你觉得这样对吗"。
  3. 先影子后生效：mode=shadow 时照常判断、照常记账、照常出 verdict，但不执行。
     跑几小时看它想拦什么，确认全对再切 enforce。没有影子模式的监督层不该上线。

预算与熔断（搬砖机器人会跑几小时，模型调用不能失控）：
  - 本地体检每 check_interval_s 一次（纯本地、零成本），只有体检异常才叫 LLM；
  - LLM 每分钟上限 + 最小间隔 + 连续失败熔断；
  - 后台线程请求，决策循环只取"已到货且未过期"的结论；没有就继续用规则。

风险动作要有据可查：takeover/panic 的 evidence 必须复述输入里的数字，否则降级
为 allow —— 这是挡"模型编一个理由就停车"的硬机制，不是礼貌建议。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from ..config import SupervisorConfig

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "你是 ARPG 搬砖机器人的监督员。主控是确定性的视觉状态机，它只会按规则动，"
    "看不出自己的动作有没有效果。你的任务：看它最近一段的数值对账，判断"
    "「这一串动作是在推进，还是在空转/走错界面」，必要时喊停。\n"
    "只输出一行 JSON，不要解释，不要 markdown：\n"
    '{"verdict":"allow|throttle|hold|takeover|panic",'
    '"confidence":0.0,"diagnosis":"一句话",'
    '"evidence":"引用输入里的具体数字",'
    '"action":{"op":"esc|wait|reset_stage","reason":"一句话"}}\n'
    "判定口径：\n"
    "- allow：有进展，或只是暂时没目标。绝大多数情况选这个。\n"
    "- throttle：在动但效果很差（walk_no_progress 高、miss 连击），降速观察。\n"
    "- hold：怀疑当前这一帧不该出手（界面可疑、预测连续落空），停几秒。\n"
    "- takeover：明确知道该做什么安全动作（弹窗挡住→esc；结算界面→reset_stage；"
    "需要观察→wait）。必须给 action。\n"
    "- panic：明确异常必须停车交人工（卡死、反复进同一个界面、疑似被反外挂拦截）。\n"
    "硬规则：\n"
    "1. 你**不能**指定去哪、捡哪个——那不归你管，输出里也不要写坐标。\n"
    "2. evidence 必须复述输入里的数字（例如 \"miss 4/6, walk_no_progress 7\"），"
    "空话会被丢弃。\n"
    "3. 没把握就 allow，confidence 给低分。宁可少拦，不要乱拦。\n"
    "4. 证据只支持「界面可疑/空转」时用 hold，别直接 panic。"
)


class Verdict(str, Enum):
    """监督结论。除 ALLOW 外都只是"更保守"，没有一条能让机器人做新动作。"""

    ALLOW = "allow"
    THROTTLE = "throttle"
    HOLD = "hold"
    TAKEOVER = "takeover"
    PANIC = "panic"


# 允许接管时执行的安全动作（不含任何拾取/移动语义）
_TAKEOVER_OPS = {"esc", "wait", "reset_stage"}


@dataclass
class Prediction:
    """一条"预期"：到期用实测值结算。这是"结果是否正向"的最小单位。"""

    kind: str          # walk / pick / escape
    expect: str        # 人类可读预期
    base: float        # 登记时的基准值
    due: float         # 结算时刻（monotonic）
    ts: float
    settled: bool = False


@dataclass
class LedgerRow:
    """预测-实测的对账结果（给 LLM 看的证据，也是人工复盘的材料）。"""

    ts: float
    kind: str
    expect: str
    result: str        # hit / miss / unknown
    detail: str


@dataclass
class Decision:
    """一次监督结论（后台线程写入，主线程读取）。"""

    verdict: Verdict = Verdict.ALLOW
    confidence: float = 0.0
    diagnosis: str = ""
    evidence: str = ""
    action: dict | None = None
    trigger: str = ""
    until: float = 0.0     # 有效期截止（monotonic）；过期自动回落 allow
    applied: bool = False  # 是否真的生效（shadow 模式下恒 False）
    at: float = 0.0


class Supervisor:
    """LLM 监督层：本地对账（每帧）+ 模型裁决（事件驱动）+ 命令闸门。"""

    def __init__(self, cfg: SupervisorConfig,
                 clock: Callable[[], float] = time.monotonic,
                 root: Path | None = None) -> None:
        self.cfg = cfg
        self._clock = clock
        self._lock = threading.RLock()
        self._root = root or Path.cwd()
        # 对账账本（最近 N 条，给 LLM 当证据）
        self._ledger: deque[LedgerRow] = deque(maxlen=60)
        self._pending: list[Prediction] = []
        # 健康窗口
        self._last_check = 0.0
        self._progress_marks: deque[tuple[float, int]] = deque(maxlen=600)
        self._beat: deque[tuple[float, int]] = deque(maxlen=600)  # (ts, 命令数)
        self._intent_changes: deque[float] = deque(maxlen=200)
        self._last_intent = ""
        self._last_cmd_count = 0
        self._unmatched_since: float | None = None
        self._no_progress_since: float | None = None
        # 模型侧
        self._decision = Decision()
        self._busy = False
        self._call_ts: deque[float] = deque(maxlen=64)
        self._last_call = 0.0
        self._fail_streak = 0
        self._open_until = 0.0
        self._panic_votes: deque[bool] = deque(maxlen=8)
        self._pending_takeover: dict | None = None
        # 统计
        self.stats = {
            "checks": 0, "calls": 0, "ok": 0, "fail": 0,
            "tripped": 0, "blocked_cmds": 0, "throttled_frames": 0,
            "takeovers": 0, "panics": 0, "shadow_would_block": 0,
            "downgraded": 0,
        }
        self.last_health: dict[str, Any] = {}

    # ---------------- 对外：每帧 ----------------

    def note_frame(self, scene: Any, intent: str, now: float,
                   probe: dict[str, Any] | None = None) -> dict[str, Any]:
        """每帧调用（纯本地，零成本）。返回本次体检结果（大多数帧是上一次的）。

        做三件事：1) 结算到期的预测；2) 按节拍算健康指标；3) 异常就异步叫模型。
        """
        if self.cfg.mode == "off":
            return {}
        probe = probe or {}
        with self._lock:
            self._settle(scene, probe, now)
            if intent and intent != self._last_intent:
                self._intent_changes.append(now)
                self._last_intent = intent
            if now - self._last_check < self.cfg.check_interval_s:
                return self.last_health
            self._last_check = now
            self.stats["checks"] += 1
            health = self._health(scene, probe, now)
            self.last_health = health

        trigger = self._pick_trigger(health, now)
        if trigger:
            self.request_review(trigger, health)
        return health

    def predict(self, kind: str, expect: str, base: float, now: float,
                horizon_s: float = 4.0) -> None:
        """登记一条预期（FSM 在关键动作后调用）：走位/拾取/自救。"""
        if self.cfg.mode == "off":
            return
        with self._lock:
            self._pending.append(Prediction(
                kind=kind, expect=expect, base=float(base),
                due=now + horizon_s, ts=now))

    # ---------------- 对外：闸门 ----------------

    def gate(self, cmd: Any, now: float) -> bool:
        """命令出口：返回 False 表示本帧不许发这条命令。

        只拦"可恢复的主动动作"，`stop_all` 永远放行（安全动作不该被拦）。
        shadow 模式下不拦，只记账 —— 这是先观察后生效的关键。
        """
        if self.cfg.mode == "off":
            return True
        if getattr(cmd, "op", "") == "stop_all":
            return True
        with self._lock:
            d = self._decision
            active = d.verdict in (Verdict.HOLD, Verdict.PANIC) and now <= d.until
        if not active:
            return True
        if self.cfg.mode == "shadow":
            with self._lock:
                self.stats["shadow_would_block"] += 1
            return True
        with self._lock:
            self.stats["blocked_cmds"] += 1
        return False

    def pace_factor(self, now: float) -> float:
        """THROTTLE 时返回 >1 的倍率，FSM 用它拉长移动/点击间隔。"""
        if self.cfg.mode == "off":
            return 1.0
        with self._lock:
            d = self._decision
            if d.verdict is Verdict.THROTTLE and now <= d.until:
                if self.cfg.mode == "enforce":
                    self.stats["throttled_frames"] += 1
                return float(self.cfg.throttle_pace)
        return 1.0

    def take_takeover(self, now: float) -> dict | None:
        """取走一次接管动作（消费后清空；shadow 模式返回 None）。"""
        if self.cfg.mode != "enforce":
            return None
        with self._lock:
            d = self._decision
            if d.verdict is not Verdict.TAKEOVER or now > d.until:
                self._pending_takeover = None
                return None
            act, self._pending_takeover = self._pending_takeover, None
        if act:
            with self._lock:
                self.stats["takeovers"] += 1
        return act

    def should_panic(self, now: float) -> bool:
        """PANIC 是否成立（enforce 下才真正停车）。"""
        if self.cfg.mode != "enforce":
            return False
        with self._lock:
            d = self._decision
            return d.verdict is Verdict.PANIC and now <= d.until

    # ---------------- 对账 ----------------

    def _settle(self, scene: Any, probe: dict[str, Any], now: float) -> None:
        if not self._pending:
            return
        keep: list[Prediction] = []
        for p in self._pending:
            if now < p.due:
                keep.append(p)
                continue
            result, detail = self._judge(p, scene, probe)
            self._ledger.append(LedgerRow(now, p.kind, p.expect, result, detail))
        self._pending = keep

    def _judge(self, p: Prediction, scene: Any, probe: dict[str, Any]) -> tuple[str, str]:
        """到点结算：拿实测值比基准值。判不出来就 unknown，不硬判。"""
        try:
            if p.kind == "walk":
                cur = probe.get("dist")
                if cur is None:
                    return "unknown", "目标已丢失"
                cur = float(cur)
                # 距离缩短 ≥10% 才算真的走近（走位抖动会小幅波动）
                if cur <= p.base * 0.9:
                    return "hit", f"{p.base:.0f}→{cur:.0f}px"
                return "miss", f"{p.base:.0f}→{cur:.0f}px（未缩短）"
            if p.kind == "pick":
                picks = int(probe.get("picks") or 0)
                if picks > int(p.base):
                    return "hit", f"拾取 {p.base:.0f}→{picks}"
                if probe.get("aframe") is False:
                    return "hit", "A框已消失"
                return "miss", f"拾取数未增加（{picks}），A框仍在"
            if p.kind == "escape":
                if getattr(scene, "in_game", False):
                    return "hit", "已回到战斗界面"
                return "miss", f"仍不在战斗界面（{getattr(scene, 'gate_reason', '')}）"
        except Exception:  # noqa: BLE001
            log.debug("预测结算异常", exc_info=True)
        return "unknown", ""

    # ---------------- 健康体检 ----------------

    def _health(self, scene: Any, probe: dict[str, Any], now: float) -> dict[str, Any]:
        """零成本指标：进展率、预测失配、界面可疑度、抖动、盲动。"""
        picks = int(probe.get("picks") or 0)
        self._progress_marks.append((now, picks))
        self._beat.append((now, self._last_cmd_count))

        win = 60.0
        marks = [p for t, p in self._progress_marks if now - t <= win]
        progressed = (marks[-1] - marks[0]) if len(marks) >= 2 else 0
        recent = [r for r in self._ledger if now - r.ts <= win]
        miss = sum(1 for r in recent if r.result == "miss")
        hit = sum(1 for r in recent if r.result == "hit")

        # 连续未识别（unmatched）持续时长：未标定战场按 ESC 会把游戏按进暂停菜单
        if not getattr(scene, "in_game", True) and \
                getattr(scene, "gate_reason", "") == "unmatched":
            if self._unmatched_since is None:
                self._unmatched_since = now
        else:
            self._unmatched_since = None
        unmatched_for = 0.0 if self._unmatched_since is None else now - self._unmatched_since

        # "在动但没结果"：命令在发、拾取不涨 —— 最危险的假动作
        cmds_recent = self._cmd_count(now) - self._cmd_count(now - win)
        if cmds_recent > 0 and progressed == 0:
            if self._no_progress_since is None:
                self._no_progress_since = now
        else:
            self._no_progress_since = None
        no_progress_for = 0.0 if self._no_progress_since is None else now - self._no_progress_since

        churn = len([t for t in self._intent_changes if now - t <= win])
        return {
            "window_s": win,
            "picks_delta": progressed,
            "hit": hit,
            "miss": miss,
            "miss_streak": self._miss_streak(),
            "cmds": cmds_recent,
            "no_progress_for": round(no_progress_for, 1),
            "unmatched_for": round(unmatched_for, 1),
            "intent_churn": churn,
            "in_game": bool(getattr(scene, "in_game", True)),
            "gate_reason": str(getattr(scene, "gate_reason", "")),
            "pillars": len(getattr(scene, "pillars", []) or []),
            "aframe": bool(getattr(scene, "aframe", None)),
            "monsters": len(getattr(scene, "monsters", []) or []),
        }

    def _cmd_count(self, now: float) -> int:
        total = 0
        for t, n in self._beat:
            if t <= now:
                total = n
        return total

    def note_command(self, now: float) -> None:
        """命令计数（gate 放行后调用），用于"在动但没进展"判据。"""
        with self._lock:
            self._last_cmd_count += 1

    def _miss_streak(self) -> int:
        streak = 0
        for r in reversed(self._ledger):
            if r.result == "miss":
                streak += 1
            elif r.result == "hit":
                break
        return streak

    def _pick_trigger(self, h: dict[str, Any], now: float) -> str:
        """异常才叫模型 —— 这是"频繁判断"的节流阀。"""
        c = self.cfg
        if h.get("miss_streak", 0) >= c.miss_streak_trigger:
            return f"预测连续落空 {h['miss_streak']} 次"
        if h.get("no_progress_for", 0) >= c.no_progress_trigger_s:
            return f"在动但 {h['no_progress_for']:.0f}s 无进展"
        if h.get("unmatched_for", 0) >= c.unmatched_trigger_s:
            return f"未识别画面持续 {h['unmatched_for']:.0f}s"
        if h.get("intent_churn", 0) >= c.churn_trigger:
            return f"意图抖动 {h['intent_churn']} 次/分"
        return ""

    # ---------------- 模型调用 ----------------

    def available(self, now: float | None = None) -> bool:
        if self.cfg.mode == "off":
            return False
        now = self._clock() if now is None else now
        with self._lock:
            if self._busy or now < self._open_until:
                return False
            if now - self._last_call < self.cfg.min_interval_s:
                return False
            recent = [t for t in self._call_ts if now - t <= 60.0]
            return len(recent) < self.cfg.max_calls_per_min

    def request_review(self, trigger: str, health: dict[str, Any] | None = None,
                       now: float | None = None) -> bool:
        """异步请求一次裁决。预算不足/正在忙时直接返回 False（规则继续跑）。"""
        now = self._clock() if now is None else now
        if not self.available(now):
            return False
        ctx = self._context(trigger, health or self.last_health)
        with self._lock:
            self._busy = True
            self.stats["calls"] += 1
            self._last_call = now
            self._call_ts.append(now)

        def _run() -> None:
            ok = False
            try:
                raw = self._post(ctx)
                d = self._parse(raw, ctx, trigger, now)
                d = self._restrict(d, ctx, now)
                with self._lock:
                    self._decision = d
                    self._pending_takeover = d.action if d.verdict is Verdict.TAKEOVER else None
                    if d.verdict is Verdict.PANIC:
                        self.stats["panics"] += 1
                ok = True
                self.stats["ok"] += 1
                log.info("监督裁决: %s (conf %.2f) %s", d.verdict.value,
                         d.confidence, d.diagnosis)
                self._audit(ctx, d)
            except Exception as e:  # noqa: BLE001
                log.warning("监督层调用失败(降级为规则): %s", e)
                with self._lock:
                    self.stats["fail"] += 1
                    self._fail_streak += 1
                    if self._fail_streak >= self.cfg.failure_threshold:
                        self._open_until = self._clock() + self.cfg.cooldown_after_failure_s
                        self._fail_streak = 0
                        self.stats["tripped"] += 1
                        log.warning("监督层连续失败，熔断 %.0fs",
                                    self.cfg.cooldown_after_failure_s)
            finally:
                with self._lock:
                    self._busy = False
                _ = ok

        threading.Thread(target=_run, daemon=True, name="llm-supervisor").start()
        return True

    def _context(self, trigger: str, health: dict[str, Any]) -> dict[str, Any]:
        """给模型的输入：一串可核对的数字，不是"你觉得如何"。"""
        with self._lock:
            rows = [
                {"kind": r.kind, "expect": r.expect, "result": r.result, "detail": r.detail}
                for r in list(self._ledger)[-8:]
            ]
        return {
            "trigger": trigger,
            "health": health,
            "ledger": rows,
            "note": "ledger 是最近的『预期 vs 实测』对账；result=miss 表示动作没产生预期效果。",
        }

    def _post(self, ctx: dict[str, Any]) -> str:
        import httpx

        base = self.cfg.base_url.rstrip("/")
        text = json.dumps(ctx, ensure_ascii=False)
        if self.cfg.provider == "ollama":
            url = f"{base}/api/chat"
            payload = {
                "model": self.cfg.model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                "stream": False,
                "options": {"temperature": 0.0},
            }
            headers: dict[str, str] = {}
        else:
            url = f"{base}/v1/chat/completions"
            payload = {
                "model": self.cfg.model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                "temperature": 0.0,
            }
            key = self.cfg.api_key or _env(self.cfg.api_key_env)
            headers = {"Authorization": f"Bearer {key}"} if key else {}
        resp = httpx.post(url, json=payload, headers=headers, timeout=self.cfg.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        if self.cfg.provider == "ollama":
            return data["message"]["content"]
        return data["choices"][0]["message"]["content"]

    def _parse(self, text: str, ctx: dict[str, Any], trigger: str, now: float) -> Decision:
        text = (text or "").strip()
        s, e = text.find("{"), text.rfind("}")
        if s < 0 or e <= s:
            raise ValueError(f"监督层回复无 JSON: {text[:120]}")
        obj = json.loads(text[s:e + 1])
        if not isinstance(obj, dict) or "verdict" not in obj:
            raise ValueError(f"监督层回复缺 verdict: {text[:120]}")
        try:
            verdict = Verdict(str(obj["verdict"]).strip().lower())
        except ValueError:
            verdict = Verdict.ALLOW
        try:
            conf = float(obj.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        action = obj.get("action") if isinstance(obj.get("action"), dict) else None
        return Decision(
            verdict=verdict, confidence=round(conf, 3),
            diagnosis=str(obj.get("diagnosis", ""))[:160],
            evidence=str(obj.get("evidence", ""))[:200],
            action=action, trigger=trigger,
            until=now + self.cfg.verdict_ttl_s, at=now,
        )

    def _restrict(self, d: Decision, ctx: dict[str, Any], now: float) -> Decision:
        """把模型结论收敛到"只能更保守"：置信度门槛、动作白名单、证据校验。"""
        c = self.cfg
        # 1) 置信度不够 → 放行（宁可少拦）
        floor = c.takeover_min_confidence if d.verdict in (
            Verdict.TAKEOVER, Verdict.PANIC) else c.min_confidence
        if d.confidence < floor:
            self.stats["downgraded"] += 1
            d.verdict, d.action = Verdict.ALLOW, None
            d.diagnosis = f"置信度 {d.confidence} < {floor}，降级放行；原判：{d.diagnosis}"
            return d

        # 2) 高风险动作必须有据可查（evidence 要复述输入里的数字）
        if d.verdict in (Verdict.TAKEOVER, Verdict.PANIC) and not _evidence_ok(d.evidence, ctx):
            self.stats["downgraded"] += 1
            d.verdict, d.action = Verdict.HOLD, None
            d.diagnosis = f"证据不含输入数字，降级为 hold；原判：{d.diagnosis}"
            d.until = now + min(c.hold_max_s, c.verdict_ttl_s)
            return d

        # 3) 接管动作必须在白名单，且不能带坐标（不许指路）
        if d.verdict is Verdict.TAKEOVER:
            act = d.action or {}
            op = str(act.get("op", "")).strip().lower()
            if op not in _TAKEOVER_OPS:
                self.stats["downgraded"] += 1
                d.verdict, d.action = Verdict.HOLD, None
                d.diagnosis = f"接管动作不在白名单({op or '空'})，降级为 hold"
                d.until = now + min(c.hold_max_s, c.verdict_ttl_s)
                return d
            d.action = {"op": op, "reason": str(act.get("reason", ""))[:120]}
            d.until = now + min(c.verdict_ttl_s, 10.0)

        # 4) HOLD 有上限，过期自动放行（防止模型一次误判永久卡死）
        if d.verdict is Verdict.HOLD:
            d.until = now + min(d.until - now, c.hold_max_s)

        # 5) PANIC 要连续两次一致（单次幻觉不停车）
        if d.verdict is Verdict.PANIC:
            self._panic_votes.append(True)
            if sum(1 for v in self._panic_votes if v) < c.panic_confirm:
                d.verdict, d.diagnosis = Verdict.HOLD, f"恐慌票不足，先 hold；{d.diagnosis}"
                d.until = now + min(c.hold_max_s, c.verdict_ttl_s)
        else:
            self._panic_votes.append(False)
        return d

    def _audit(self, ctx: dict[str, Any], d: Decision) -> None:
        """每条裁决落盘：这是把 LLM 输出变成回归用例的原料。"""
        try:
            p = Path(self.cfg.audit_path)
            if not p.is_absolute():
                p = self._root / p
            p.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "ts": round(time.time(), 3),
                "mode": self.cfg.mode,
                "verdict": d.verdict.value,
                "confidence": d.confidence,
                "diagnosis": d.diagnosis,
                "evidence": d.evidence,
                "action": d.action,
                "trigger": d.trigger,
                "health": ctx.get("health"),
                "ledger": ctx.get("ledger"),
            }
            with self._lock:
                with p.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001
            log.debug("监督层审计落盘失败", exc_info=True)

    # ---------------- 观测 ----------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            d = self._decision
            return {
                "mode": self.cfg.mode,
                "provider": self.cfg.provider,
                "model": self.cfg.model,
                "available": self.available(),
                "tripped": self._clock() < self._open_until,
                "verdict": d.verdict.value,
                "confidence": d.confidence,
                "diagnosis": d.diagnosis,
                "evidence": d.evidence,
                "trigger": d.trigger,
                "applied": self.cfg.mode == "enforce",
                "health": self.last_health,
                "ledger_recent": [
                    {"kind": r.kind, "result": r.result, "detail": r.detail}
                    for r in list(self._ledger)[-5:]
                ],
                "stats": dict(self.stats),
            }

    def reset(self) -> None:
        with self._lock:
            self._decision = Decision()
            self._pending.clear()
            self._ledger.clear()
            self._panic_votes.clear()
            self._pending_takeover = None
            self._unmatched_since = None
            self._no_progress_since = None


def _evidence_ok(evidence: str, ctx: dict[str, Any]) -> bool:
    """evidence 里至少要有一个数字能在输入里找到。

    为什么这么较真：模型最容易生成的幻觉是"看起来很有道理的一句话"。
    要求它复述输入里的具体数字，成本几乎为零，却能挡掉大部分空话式判断。
    """
    import re

    nums = re.findall(r"\d+(?:\.\d+)?", evidence or "")
    if not nums:
        return False
    blob = json.dumps(ctx, ensure_ascii=False)
    return any(n in blob for n in nums)


def _env(name: str) -> str:
    import os

    return os.environ.get(name, "")
