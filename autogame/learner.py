"""持续学习 —— 把"学一遍"从一个按钮变成一件一直在做的事。

## 为什么要单独一个模块

`evolve()` 是**一次**进化：体检 → 出代 → 回归 → 上线。它回答"现在这批材料能学到什么"。
但设备装好之后，用户的行为不是"学一次"，而是：

    玩了两把，发现结算界面不认识 → 截两张图丢进来 → 补一句"这是结算"
    → 过几天发现背包要按 B 打开 → 再补一句
    → 又发现之前那句"看到光柱就走过去"说得不准确 → 改成"先看清是不是自己的光柱"

也就是说，材料是**一点点长出来**的，还经常**推翻自己之前说的话**。
这种节奏下，"点一下学一遍"有两个毛病：人得记得点；点的时候常常材料还没写完。

这个模块只干一件事：盯着材料，静默一会儿就学，学坏了不硬学第二遍。

## 三条纪律

1. **防抖**：材料的最后修改时间距今不足 `quiet_s` 就不动手。用户连续截五张图，
   不该让它学五次半成品。
2. **同一份材料只自动试 `max_tries` 次**：回归挂了，说明是材料/模板的问题，
   不是运气问题。反复重试只会把 RK3566 的算力吃干净，还会让日志变成噪声。
   材料一变（指纹变），计数归零，重新给机会。
3. **绝不绕过闸门**：这里调的仍然是 `evolve()`，回归不过就不上线。自动化的
   只是"什么时候学"，不是"学完了算不算数"。

## 状态放哪

`profiles/<id>/evolve/learner.json`。和 generations 放在一起，升级换引擎时
跟着数据走。服务重启后接着上次的认知继续 —— 不会因为重启就把一份坏材料再学一遍。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import evolve as ev

log = logging.getLogger(__name__)

#: 同一 profile 的"学"要串行：后台 tick 和页面「现在学」不能同时动同一批文件。
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(profile: Any) -> threading.Lock:
    key = str(getattr(profile, "id", "") or profile.root)
    with _LOCKS_GUARD:
        if key not in _LOCKS:
            _LOCKS[key] = threading.Lock()
        return _LOCKS[key]


@dataclass
class State:
    """学习者记得的事。全部可重建 —— 丢了最多多学一次，不会学坏。"""

    enabled: bool | None = None          # None = 跟随 config
    last_ts: float = 0.0
    last_digest: str = ""
    last_result: str = ""                # promoted / rejected / error / duplicate
    last_message: str = ""
    last_source: str = ""                # auto / manual
    tries: dict = field(default_factory=dict)   # digest → {ts, result, n}

    def to_dict(self) -> dict:
        return asdict(self)


def state_path(profile: Any) -> Path:
    return ev.evolve_root(profile) / "learner.json"


def load_state(profile: Any) -> State:
    p = state_path(profile)
    if not p.exists():
        return State()
    try:
        d = json.loads(p.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        log.warning("learner.json 读不出，按全新处理: %s", p, exc_info=True)
        return State()
    return State(enabled=d.get("enabled"), last_ts=float(d.get("last_ts") or 0),
                 last_digest=str(d.get("last_digest") or ""),
                 last_result=str(d.get("last_result") or ""),
                 last_message=str(d.get("last_message") or ""),
                 last_source=str(d.get("last_source") or ""),
                 tries=dict(d.get("tries") or {}))


def save_state(profile: Any, st: State) -> None:
    p = state_path(profile)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st.to_dict(), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(p)          # 原子换：断电不会留下半份状态


def _cfg_num(cfg: Any, name: str, default: float) -> float:
    sec = getattr(cfg, "learner", None)
    try:
        return float(getattr(sec, name, default))
    except (TypeError, ValueError):
        return default


def is_enabled(profile: Any, cfg: Any) -> bool:
    """页面上的开关存在 learner.json；没动过就跟随 config.yaml。"""
    st = load_state(profile)
    if st.enabled is not None:
        return bool(st.enabled)
    return bool(getattr(getattr(cfg, "learner", None), "enabled", True))


def set_enabled(profile: Any, cfg: Any, on: bool) -> State:
    st = load_state(profile)
    st.enabled = bool(on)
    save_state(profile, st)
    log.info("自动学习 %s", "打开" if on else "关闭")
    return st


# ---------------------------------------------------------------------------
# 现在该不该学
# ---------------------------------------------------------------------------

@dataclass
class Verdict:
    code: str      # off / up_to_date / running / blocked / tried_out / waiting / cooling / ready
    text: str      # 给页面看的原话
    wait_s: float = 0.0

    @property
    def should_learn(self) -> bool:
        return self.code == "ready"


def decide(profile: Any, cfg: Any, *, now: float | None = None,
           busy: bool = False) -> Verdict:
    """只判断，不动手。反复调用无副作用 —— 页面和后台循环共用这一份判断。

    `busy`：机器人正在跑。这时候不学，理由和"规则层不夺权"是同一条 ——
    换代会让正在执行的动作失去依据（模板换了、规则换了，打了一半的目标没了）。
    材料不会跑掉，等它停手。
    """
    now = time.time() if now is None else now
    if not is_enabled(profile, cfg):
        return Verdict("off", "自动学习已关（关掉之后只在页面点「现在学」才动）")

    audit = ev.audit_materials(profile)
    cur = ev.active_generation(profile)

    if cur is not None and cur.digest and cur.digest == audit.digest:
        return Verdict("up_to_date", "材料没有新东西，不用再学")

    if busy:
        return Verdict("running", "机器人正在跑，等它停手再学（不在它手里换配方）")

    if not audit.ready:
        return Verdict("blocked", audit.summary())

    st = load_state(profile)
    mx = max(1, int(_cfg_num(cfg, "max_tries", 1)))
    hit = st.tries.get(audit.digest) or {}
    if hit and int(hit.get("n") or 0) >= mx:
        # 具体原因不写在状态行里：它会很长，而页面下面还要单独列一次。
        return Verdict("tried_out",
                       "这批材料上次没通过，自动学习不再重复烧它。"
                       "改了材料会重新试，也可以点「现在学」硬跑一次。")

    delta = ev.materials_delta(profile, cur)
    quiet = _cfg_num(cfg, "quiet_s", 45.0)
    if delta.get("newest"):
        idle = now - float(delta["newest"])
        if idle < quiet:
            left = quiet - idle
            return Verdict("waiting",
                           f"材料刚动过，等你写顺手了再学（还有 {int(left)} 秒没有新文件）",
                           wait_s=left)

    gap = _cfg_num(cfg, "min_gap_s", 180.0)
    if st.last_ts and now - st.last_ts < gap:
        left = gap - (now - st.last_ts)
        return Verdict("cooling", f"刚学过，{int(left)} 秒后再看", wait_s=left)

    return Verdict("ready", "材料有变化，可以学了")


# ---------------------------------------------------------------------------
# 学一次
# ---------------------------------------------------------------------------

def learn(profile: Any, cfg: Any, *, source: str = "auto",
          use_model: bool = True, reason: str = "") -> dict:
    """跑一次完整进化（出代 + 回归 + 上线），并把回执写进状态。

    `source` 只影响记录和防重试口径：手动点的时候不受"同一份材料试过几次"限制 ——
    人是知道自己在干什么的（比如刚改完模板想再跑一次回归）。
    """
    lock = _lock_for(profile)
    if not lock.acquire(blocking=False):
        return {"ok": False, "busy": True,
                "message": "已经在学了，等这一轮跑完"}

    st = load_state(profile)
    digest = ""
    try:
        digest = ev.materials_digest(profile)
        res = ev.evolve(profile, cfg, force=False, use_model=use_model)
        stage = str(res.get("stage") or "")
        if stage == "build" and not res.get("gen"):
            # 指纹一致 / 材料不够：不是失败，别记成"试过并失败"
            out = {"ok": bool(res.get("ok")), "kind": "skipped",
                   "message": res.get("message") or "没出代"}
        else:
            ok = bool(res.get("promoted"))
            out = {"ok": ok, "kind": "promoted" if ok else "rejected",
                   "message": res.get("message") or "",
                   "gen": res.get("gen"), "score": res.get("score"),
                   "fail": res.get("fail") or []}
        result = out["kind"]
        msg = out["message"]
    except Exception as e:  # noqa: BLE001
        log.exception("自动学习失败")
        out = {"ok": False, "kind": "error", "message": f"{type(e).__name__}: {e}"}
        result, msg = "error", out["message"]
    finally:
        lock.release()

    if not digest:
        try:
            digest = ev.materials_digest(profile)
        except Exception:  # noqa: BLE001
            digest = ""

    now = time.time()
    st.last_ts = now
    st.last_digest = digest
    st.last_result = result
    st.last_message = msg
    st.last_source = source
    if result in ("rejected", "error"):
        # 只对"真的跑过并没过"计数。skipped 不计数：那是材料没变，不是失败。
        slot = st.tries.get(digest) or {"ts": now, "result": result, "n": 0}
        slot["ts"] = now
        slot["result"] = result
        slot["message"] = msg[:200]
        slot["n"] = int(slot.get("n") or 0) + 1
        st.tries[digest] = slot
    else:
        st.tries.pop(digest, None)
    # 只留最近 20 个指纹的计数，别让这个文件无界长
    if len(st.tries) > 20:
        keep = sorted(st.tries.items(), key=lambda kv: kv[1].get("ts") or 0)[-20:]
        st.tries = dict(keep)
    save_state(profile, st)

    out["source"] = source
    out["digest"] = digest
    if reason:
        out["reason"] = reason
    log.info("学习回执[%s] %s：%s", source, result, msg)
    return out


def tick(profile: Any, cfg: Any, *, now: float | None = None,
         use_model: bool = True, busy: bool = False) -> dict:
    """后台循环调这个。该学就学，不该学就返回一句为什么。"""
    v = decide(profile, cfg, now=now, busy=busy)
    if not v.should_learn:
        return {"action": "skip", "code": v.code, "message": v.text,
                "wait_s": round(v.wait_s, 1)}
    res = learn(profile, cfg, source="auto", use_model=use_model)
    return {"action": "learn", "code": res.get("kind"),
            "message": res.get("message") or "", "result": res}


# ---------------------------------------------------------------------------
# 页面读的快照（便宜、不跑学习）
# ---------------------------------------------------------------------------

def snapshot(profile: Any, cfg: Any, *, now: float | None = None,
             busy: bool = False) -> dict:
    now = time.time() if now is None else now
    st = load_state(profile)
    cur = ev.active_generation(profile)
    delta = ev.materials_delta(profile, cur)
    v = decide(profile, cfg, now=now, busy=busy)
    mx = max(1, int(_cfg_num(cfg, "max_tries", 1)))
    try:
        digest = ev.materials_digest(profile)
    except Exception:  # noqa: BLE001
        digest = ""
    hit = st.tries.get(digest) or {}
    return {
        "enabled": is_enabled(profile, cfg),
        "code": v.code,
        "text": v.text,
        "wait_s": round(v.wait_s, 1),
        "interval_s": _cfg_num(cfg, "interval_s", 15.0),
        "quiet_s": _cfg_num(cfg, "quiet_s", 45.0),
        "min_gap_s": _cfg_num(cfg, "min_gap_s", 180.0),
        "last_ts": st.last_ts,
        "last_ago_s": round(now - st.last_ts, 1) if st.last_ts else None,
        "last_result": st.last_result,
        "last_message": st.last_message,
        "last_source": st.last_source,
        "delta": delta,
        "max_tries": mx,
        "tried": hit,
    }
