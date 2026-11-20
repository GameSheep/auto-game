"""经验闭环 —— 从"又出事了"到"手册里多一条"，中间不许跳步。

## 为什么不能"出一次问题就写进手册"

对话里把这条链写得很清楚：

    事件记录 → 异常诊断 → 经验提取 → 候选策略 → 验证 → 晋升 → 性能监控 → 退化回滚

其中「候选 → 验证 → 晋升」最容易做错：抓到一条体检结论就写成经验，
下一轮进化把它当规律用，结果是**在噪声上训练**。一次卡住可能只是网络抖了；
一次 "recovery_spin" 也可能只是刚好那 20 秒没目标。把这种东西写进手册，
下一代的规则就会照着它走，越学越偏。

所以这里的规矩是：

1. **只记候选，不直接写手册**。候选带证据（每次的 detail / 指标），人能在
   页面上看见"它凭什么这么认为"。
2. **要复现才算数**。同一类问题至少出现 `min_seen` 次，且第一次到最近一次
   跨过 `min_span_s` 秒 —— 同一个 20 秒窗口里连报三次是"一次发作"，不是三次。
3. **晋升走补丁那条路**。写进手册用的是 `PatchStore`：有预览、有校验、有原件、
   能回滚。经验不是"直接改文件"的特权通道。
4. **驳回要留档**。驳回过的候选不会隔一个窗口又冒出来 —— 否则页面会变成
   一个永远清不掉的告警列表。

## 候选长什么样

    {"id": "x-...", "code": "repeat", "kind": "lesson",
     "when": "同一个动作连续 14 次位置和参数都没变（mouse_click(640,360)）",
     "do": "先松开所有键停 1 秒，再后退一步重新识别；连续两次还这样，就换路线",
     "rule": {"id": "xp-repeat", "trigger": "stall", "when": {...}, "then": {...}},
     "rule_blocked": "",
     "evidence": [{"ts": ..., "detail": "...", "metric": {...}}],
     "seen": 3, "first_seen": ..., "last_seen": ...,
     "status": "candidate|promoted|rejected", "patch_id": ""}

晋升时写**两样**：手册（给人看）+ 规则（给程序用）。规则不是每条结论都写得出来
—— 只有条件能落到 `evolve.FACTS` 里真会产出的字段上，才允许生成；写不出来的
把理由放进 `rule_blocked`，页面照原样显示。硬造一条"能过校验但永远不命中"的
规则，比不写更坏：页面上它会显示成"已经学会了"。
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from .atomic import atomic_write_json, load_json_tolerant, locked_update_json

log = logging.getLogger(__name__)

CANDIDATE = "candidate"
PROMOTED = "promoted"
REJECTED = "rejected"

#: 同一类问题的两次记录至少隔这么久才算"又一次发作"（体检窗口本身 20 秒）
DEDUPE_S = 5.0
#: 证据最多留几条（页面只展示最近几条，留太多只是让文件变大）
MAX_EVIDENCE = 8

_TEMPLATES: dict[str, tuple[str, str]] = {
    "repeat": (
        "同一个动作连续 {repeat} 次位置和参数都没变（{key}）",
        "先松开所有键停 1 秒，再后退一步重新识别；连续两次还这样，"
        "就换路线或放弃这个目标，别用同一个动作硬顶",
    ),
    "recovery_spin": (
        "每秒进恢复 {now} 次（之前 {base} 次），可一直没捡到东西（这窗口 {picks} 次拾取）",
        "第二次恢复不要重复同一个动作：回到上一个稳定点重新识别，"
        "或者直接放弃当前目标换一个",
    ),
    "slow": (
        "每 tick 耗时从 {base_ms}ms 涨到 {now_ms}ms（同一画面要花更多时间）",
        "先切观察模式确认是帧率/负载问题，不要在这个状态下继续按键；"
        "恢复后从上一个稳定点继续，别接着刚才那一步硬走",
    ),
    "model_burn": (
        "每秒问模型 {now} 次（之前 {base} 次）—— 规则大面积匹配不上",
        "停下来把当时的界面补进手册（一张图 + 一眼认出 + 该做什么），"
        "让规则先匹配上再继续；不要靠每帧问模型兜底",
    ),
}

_LABELS = {
    "repeat": "卡住时怎么脱身",
    "recovery_spin": "恢复没效果时怎么换招",
    "slow": "变慢时先做什么",
    "model_burn": "规则匹配不上时补什么",
}

#: 这两条结论**写不成会命中的规则**，理由要跟着候选一起给人看：
#: 一句"这条不生成规则"没有信息量，用户会以为是漏做了。
#:
#: 它们改的是系统参数（见 `config_for`）—— 动作清单里只有 9 个游戏内行为
#: （走开/打/捡/等/导航…），没有"降级""少问模型"这类动作。变慢时让角色
#: 随便走两步，不但治不了问题，还添乱。
_NO_RULE = {
    "slow": ("变慢是每 tick 耗时的变化，而动作清单里没有「减轻负担」这类动作 —— "
             "写成规则只会在慢的时候让角色乱走。改成调参数：把决策频率降下来"),
    "model_burn": ("烧模型的原因是「没有规则匹配得上」，再加一条规则治不了它。"
                   "改成调参数：把模型裁决的冷却从 20 秒拉到 60 秒"),
}

#: 治"负担"的两条结论 → 配置补丁。值取**保守的绝对值**而不是"减半"：
#: 减半要知道当前值，而观察的时候拿不到配置；而且人看到"降到 0.5Hz"
#: 比看到"降到 0.5Hz（原来是 0.7）"更容易判断要不要采纳。
_CONFIG_ADVICE: dict[str, dict] = {
    "slow": {
        "payload": {"decision": {"tick_hz": 0.5}},
        "why": "把决策频率降到 0.5Hz（2 秒一次）：同样的负载下每帧能分到双倍时间，"
               "而且慢的时候本来就不该按那么快",
    },
    "model_burn": {
        "payload": {"decision": {"arbiter": {"cooldown_s": 60.0}}},
        "why": "把模型裁决的冷却从 20 秒拉到 60 秒：它会继续问，但不会每秒都在问",
    },
}

#: 生成规则时用到的动作。都取自 `evolve.action_catalog()`，不写死新动作。
_ESCAPE_ACTION = "explore"


def _key_op(key: str) -> str:
    """`mouse_click(640,360)` → `mouse_click`。"""
    return str(key or "").split("(")[0]


def rule_for(code: str, metric: dict) -> tuple[dict | None, str]:
    """把一条体检结论翻成**真能命中**的规则。返回 (规则, 不写的理由)。

    两条硬规矩：

    1. **条件只能引用 `evolve.FACTS` 里、且 probe 每个 tick 都会产出的字段。**
       引用一个永远缺席的字段，规则能过校验、能写进 rules.yaml，然后一次都不
       命中 —— 这比不写更坏，因为页面上它会显示成"已学会"。
    2. **门槛要比体检的告警线更早。** 体检在重复 12 次时告警，规则取 `0.75×N`
       （12 次 → 9 次）介入：等它按满 12 次再救，那 12 次已经白按了。

    不写规则的那两种（`slow` / `model_burn`）把理由一起返回，页面照原样显示。
    """
    if code in _NO_RULE:
        return None, _NO_RULE[code]

    if code == "repeat":
        key = str(metric.get("key") or "")
        n = int(metric.get("repeat") or 0)
        if not key:
            return None, "不知道是哪个动作在重复，写不出条件"
        if _key_op(key) in (_ESCAPE_ACTION, "wait"):
            # 卡住的动作本身就是"走开/等一等"，再让它走开一遍没有意义
            return None, f"卡住的动作就是 {key}，再让它做一遍同样的动作帮不上忙"
        if n < 6:
            return None, f"只重复了 {n} 次，还说明不了卡住"
        level = max(4, int(round(n * 0.75)))
        return {
            "id": "xp-repeat", "trigger": "stall",
            "when": {"all": [
                {"fact": "in_game", "op": "==", "value": True},
                {"fact": "repeat", "op": ">=", "value": level},
                {"fact": "repeat_key", "op": "==", "value": key},
            ]},
            "then": {"action": _ESCAPE_ACTION},
            "why": (f"「{key}」重复到 {n} 次时卡死了：位置和参数都没变，"
                    "再按下去只是白按。先走开换个位置重新找目标"),
            "confidence": 0.5,
        }, ""

    if code == "recovery_spin":
        # 这里的 3 次 / 30 秒是**固定的**，不从 metric 推：metric 里给的是
        # "每秒进恢复几次"（20 秒窗口的速率），换算成"60 秒内几次"要靠乘窗口长度，
        # 那是猜；而 `recoveries` 这个事实本身就是"最近 60 秒进了几次"，
        # 直接说"一分钟里三次还没捡到东西"更清楚，也更好解释给人听。
        return {
            "id": "xp-recovery_spin", "trigger": "stall",
            "when": {"all": [
                {"fact": "in_game", "op": "==", "value": True},
                {"fact": "recoveries", "op": ">=", "value": 3},
                {"fact": "picks_delta", "op": "==", "value": 0},
                {"fact": "no_progress_for", "op": ">=", "value": 30},
            ]},
            "then": {"action": _ESCAPE_ACTION},
            "why": ("一分钟里进恢复 3 次还一个都没捡到 —— 在原地换招已经没用了，"
                    "先离开这片区域再重新找目标"),
            "confidence": 0.5,
        }, ""

    return None, f"不认识的结论 {code}"


def config_for(code: str, metric: dict) -> tuple[dict | None, str]:
    """把"治负担"的结论翻成一条配置补丁。返回 (补丁, 为什么不改)。

    这类结论和 `rule_for` 是**互补**关系，不是二选一：
    规则管"遇到某种局面该做什么动作"，配置管"整个系统跑多快、多久打扰一次模型"。

    补丁本身不落盘 —— 它跟着候选一起等人点「记进手册」，走了
    `PatchStore` 的 config 那条路（写到 runtime overlay，删掉就回滚，
    不碰手写的 config.yaml）。
    """
    adv = _CONFIG_ADVICE.get(code)
    if adv is None:
        return None, ""
    if code == "slow":
        now = float(metric.get("now_ms") or 0)
        base = float(metric.get("base_ms") or 0)
        if not now or not base:
            return None, "这次没测到耗时数值（now_ms/base_ms），不知道该不该降"
        # 只有在"确实变慢了"的时候才建议动配置；数据对不上就别乱改
        if now <= base:
            return None, f"耗时 {now:.0f}ms 并不比基线 {base:.0f}ms 慢，不用调"
    if code == "model_burn":
        now = float(metric.get("now") or 0)
        if now <= 0:
            return None, "没有测到模型调用频率，不知道该不该拉长冷却"
    return dict(adv["payload"]), ""


def config_reason(code: str) -> str:
    """配置补丁的人话解释（页面/手册里用）。"""
    return str((_CONFIG_ADVICE.get(code) or {}).get("why") or "")


class ExperienceError(ValueError):
    """这条候选现在不能晋升（次数不够 / 校验没过 / 状态不对）。"""


def store_path(profile: Any) -> Path:
    """候选存在 profile 里：它属于"这个游戏怎么玩"，不属于机器日志。"""
    return Path(profile.root) / "experience" / "candidates.json"


def _fmt(tpl: str, metric: dict) -> str:
    vals: dict[str, Any] = {}
    for k in _keys(tpl):
        v = metric.get(k, 0)
        # 820.0ms 这种显示不好看；整数浮点就当整数说
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        vals[k] = v
    try:
        return tpl.format(**vals)
    except Exception:  # noqa: BLE001
        return tpl


def _kind_of(rule: dict | None, cfg_patch: dict | None) -> str:
    """这条候选晋升时会写什么。页面靠它区分"教它做事"和"调参数"。"""
    parts = ["lesson"]
    if rule:
        parts.append("rule")
    if cfg_patch:
        parts.append("config")
    return "+".join(parts)


def _keys(tpl: str) -> set[str]:
    import string

    return {name for _, name, _, _ in string.Formatter().parse(tpl) if name}


class ExperienceStore:
    """一个 profile 的经验候选账本。读写都在跨进程锁里。"""

    def __init__(self, profile: Any, *, min_seen: int = 2, min_span_s: float = 60.0,
                 path: Path | None = None) -> None:
        self.profile = profile
        self.min_seen = max(1, int(min_seen))
        self.min_span_s = max(0.0, float(min_span_s))
        self.path = Path(path) if path else store_path(profile)

    # ---------------- 读 ----------------

    def _read(self) -> dict:
        doc = load_json_tolerant(self.path, {"version": 1, "items": []})
        if not isinstance(doc, dict) or not isinstance(doc.get("items"), list):
            doc = {"version": 1, "items": []}
        return doc

    def items(self, status: str | None = None, *, limit: int = 50) -> list[dict]:
        rows = [r for r in self._read()["items"] if isinstance(r, dict)]
        if status:
            rows = [r for r in rows if r.get("status") == status]
        rows.sort(key=lambda r: float(r.get("last_seen") or 0), reverse=True)
        return rows[:limit]

    def get(self, cid: str) -> dict | None:
        for r in self._read()["items"]:
            if isinstance(r, dict) and r.get("id") == cid:
                return r
        return None

    def ready(self, row: dict) -> tuple[bool, str]:
        """够不够格晋升。返回 (能不能, 不能的理由)。"""
        if row.get("status") != CANDIDATE:
            return False, f"状态是 {row.get('status')}，不是候选"
        seen = int(row.get("seen") or 0)
        if seen < self.min_seen:
            return False, (f"只出现过 {seen} 次，还不够当规律"
                           f"（至少 {self.min_seen} 次）")
        span = float(row.get("last_seen") or 0) - float(row.get("first_seen") or 0)
        if span < self.min_span_s:
            return False, (f"两次之间只隔了 {span:.0f} 秒，可能还是同一次发作"
                           f"（要隔 {self.min_span_s:.0f} 秒以上）")
        return True, ""

    # ---------------- 写：观察 ----------------

    def observe(self, finding: Any, *, now: float | None = None,
                extra: dict | None = None) -> dict | None:
        """把一条体检结论折进候选。返回更新后的候选（不构成候选时返回 None）。

        同一类问题在 `DEDUPE_S` 内重复报只算一次 —— 体检窗口是 20 秒，
        但抖动/重试可能让同一件事连着进来。
        """
        now = float(now if now is not None else time.time())
        code = str(getattr(finding, "code", "") or
                   (finding.get("code") if isinstance(finding, dict) else "") or "")
        if not code or code not in _TEMPLATES:
            return None
        metric = dict(getattr(finding, "metric", None)
                      or (finding.get("metric") if isinstance(finding, dict) else {})
                      or {})
        detail = str(getattr(finding, "detail", "")
                     or (finding.get("detail") if isinstance(finding, dict) else "") or "")
        if extra:
            metric = {**metric, **extra}

        when_tpl, do = _TEMPLATES[code]
        rule, blocked = rule_for(code, metric)
        cfg_patch, cfg_blocked = config_for(code, metric)
        updated: dict = {}

        def mutate(doc: dict) -> dict:
            items = doc.setdefault("items", [])
            row = next((r for r in items
                        if isinstance(r, dict) and r.get("code") == code
                        and r.get("status") == CANDIDATE), None)
            if row is None:
                # 驳回过的同类问题不再自动复活：人已经说过"不采纳"了。
                if any(isinstance(r, dict) and r.get("code") == code
                       and r.get("status") == REJECTED for r in items):
                    return doc
                row = {
                    "id": "x-" + uuid.uuid4().hex[:8], "code": code,
                    "label": _LABELS.get(code, code),
                    "kind": _kind_of(rule, cfg_patch),
                    "when": _fmt(when_tpl, metric), "do": do,
                    "rule": rule or {}, "rule_blocked": blocked,
                    "config": cfg_patch or {}, "config_blocked": cfg_blocked,
                    "config_why": config_reason(code),
                    "evidence": [], "seen": 0,
                    "first_seen": now, "last_seen": 0.0,
                    "status": CANDIDATE, "patch_id": "", "note": "",
                }
                items.append(row)
            if now - float(row.get("last_seen") or 0) < DEDUPE_S and row.get("seen"):
                updated.update(row)
                return doc
            row["seen"] = int(row.get("seen") or 0) + 1
            row["last_seen"] = now
            ev = row.setdefault("evidence", [])
            ev.append({"ts": round(now, 1), "detail": detail[:160],
                       "metric": {k: v for k, v in metric.items()
                                  if isinstance(v, (int, float, str, bool))}})
            del ev[:-MAX_EVIDENCE]
            # 模板里的数字用最近一次的实测值，别一直显示第一次那个数
            row["when"] = _fmt(when_tpl, metric)
            # 规则也跟着实测值走：卡住的次数每次不一样（12/14/20），
            # 门槛留在第一次那个数，等于按最轻的那次定标准
            row["rule"], row["rule_blocked"] = rule_for(code, metric)
            row["rule"] = row["rule"] or {}
            row["config"], row["config_blocked"] = config_for(code, metric)
            row["config"] = row["config"] or {}
            row["kind"] = _kind_of(row["rule"], row["config"])
            updated.clear()
            updated.update(row)
            return doc

        try:
            locked_update_json(self.path, mutate,
                               default={"version": 1, "items": []})
        except Exception:  # noqa: BLE001
            log.exception("记录经验候选失败（不影响运行）")
            return None
        return dict(updated) if updated else None

    # ---------------- 写：晋升 / 驳回 ----------------

    def promote(self, cid: str, patches: Any, *, actor: str = "experience") -> dict:
        """候选 → 手册（+ 能做到就再加一条规则）。走补丁那条路：预览、校验、留原件、可回滚。

        为什么两样都写：手册是**给人看**的（下次同类局面人一眼就知道该怎么做），
        规则是**给程序用**的（下一次同样的局面它自己就换了招，不用等人）。
        只写手册等于"记下来了但不会用"，只写规则等于"会用了但不知道为什么"。

        规则写不进去（校验没过 / 还没有任何一代配方）**不影响手册**：
        手册那部分已经是有价值的知识，不能因为规则没落地就一起扔掉。
        失败原因会跟着返回值出去，页面上照原样显示。
        """
        row = self.get(cid)
        if row is None:
            raise ExperienceError(f"没有这条候选: {cid}")
        ok, why = self.ready(row)
        if not ok:
            raise ExperienceError(why)
        if patches is None:
            raise ExperienceError("没有绑定可写的对象（缺 profile）")

        payload = {"when": str(row.get("when") or ""), "do": str(row.get("do") or "")}
        note = f"经验：{row.get('label') or row.get('code')}（出现 {row.get('seen')} 次）"
        pv = patches.preview("lesson", payload, note=note,
                             text=str(row.get("evidence", [{}])[-1].get("detail", "")))
        if not pv.get("ok"):
            bad = [c for c in pv.get("checks", []) if not c["ok"]]
            raise ExperienceError("写进手册前校验没过：" +
                                  "；".join(f"{c['name']}（{c['detail']}）" for c in bad))
        rec = patches.apply("lesson", payload, note=note,
                            text=str(row.get("when") or ""),
                            stamp=pv["stamp"], actor=actor)

        rule_rec, rule_error = self._apply_rule(row, patches, actor=actor)
        cfg_rec, cfg_error = self._apply_config(row, patches, actor=actor)

        def mutate(doc: dict) -> dict:
            for r in doc.get("items", []):
                if isinstance(r, dict) and r.get("id") == cid:
                    r["status"] = PROMOTED
                    r["patch_id"] = rec["id"]
                    r["rule_patch_id"] = (rule_rec or {}).get("id", "")
                    r["rule_error"] = rule_error
                    r["config_patch_id"] = (cfg_rec or {}).get("id", "")
                    r["config_error"] = cfg_error
                    r["promoted_at"] = time.time()
            return doc

        locked_update_json(self.path, mutate, default={"version": 1, "items": []})
        wrote = [x for x in (f"规则补丁 {(rule_rec or {}).get('id')}" if rule_rec else "",
                             f"配置补丁 {(cfg_rec or {}).get('id')}" if cfg_rec else "") if x]
        log.info("经验已晋升 %s（%s）→ 手册补丁 %s%s", cid, row.get("code"), rec["id"],
                 "、" + "、".join(wrote) if wrote else "（没有随附规则/配置）")
        return {"candidate": self.get(cid), "patch": rec,
                "rule_patch": rule_rec, "rule_error": rule_error,
                "config_patch": cfg_rec, "config_error": cfg_error}

    def _apply_rule(self, row: dict, patches: Any, *,
                    actor: str) -> tuple[dict | None, str]:
        """把候选里那条规则写进当前这一代。返回 (补丁记录, 没写成的原因)。

        规则 id 固定成 `xp-<结论>`：同类结论第二次晋升是**改门槛**，不是再堆一条。
        否则同一个毛病每学一次就往 rules.yaml 里加一条，越长越没法看，
        后面的人也不知道哪条才算数。
        """
        from .assistant.patch import PatchError

        rule = row.get("rule")
        if not isinstance(rule, dict) or not rule.get("when"):
            return None, str(row.get("rule_blocked") or "这条结论没有对应的规则形态")
        note = f"经验规则：{row.get('label') or row.get('code')}"
        try:
            rp = patches.preview("rule", {"rule": rule}, note=note,
                                 text=str(row.get("when") or ""))
            if not rp.get("ok"):
                bad = [c for c in rp.get("checks", []) if not c["ok"]]
                return None, ("规则没敢写进去：" +
                              "；".join(f"{c['name']}（{c['detail']}）" for c in bad))
            rec = patches.apply("rule", {"rule": rule}, note=note,
                                text=str(row.get("when") or ""),
                                stamp=rp["stamp"], actor=actor)
        except PatchError as e:
            # 最典型的一种：profile 还没有任何一代配方，规则没地方放。
            # 手册已经写了，这里只把原因带出去，不回滚、不假装成功。
            return None, f"规则没敢写进去：{e}"
        except Exception as e:  # noqa: BLE001
            log.exception("规则补丁应用失败（手册已写）")
            return None, f"规则没敢写进去：{type(e).__name__}: {e}"
        return rec, ""

    def _apply_config(self, row: dict, patches: Any, *,
                      actor: str) -> tuple[dict | None, str]:
        """把"治负担"的那条配置补丁写进 runtime overlay。返回 (补丁记录, 没写成的原因)。

        走 overlay 而不是 config.yaml：经验是**试出来的**，用户想撤的时候删掉
        overlay 或回滚这条补丁就行，手写的那份配置一个字都不动。

        写完不等重启 —— 调用方（`make_live`）会当场 `apply_patch` 到内存。
        这条链之前断过一次：补丁写进文件，程序照旧跑着旧参数，
        页面上却显示"已经学会"，直到下次重启才突然变。
        """
        from .assistant.patch import PatchError

        cfg_patch = row.get("config")
        if not isinstance(cfg_patch, dict) or not cfg_patch:
            return None, str(row.get("config_blocked") or "")
        note = f"经验配置：{row.get('label') or row.get('code')}"
        try:
            pv = patches.preview("config", cfg_patch, note=note,
                                 text=config_reason(str(row.get("code") or "")))
            if not pv.get("ok"):
                bad = [c for c in pv.get("checks", []) if not c["ok"]]
                return None, ("配置没敢改：" +
                              "；".join(f"{c['name']}（{c['detail']}）" for c in bad))
            rec = patches.apply("config", cfg_patch, note=note,
                                text=config_reason(str(row.get("code") or "")),
                                stamp=pv["stamp"], actor=actor)
        except PatchError as e:
            return None, f"配置没敢改：{e}"
        except Exception as e:  # noqa: BLE001
            log.exception("配置补丁应用失败（手册已写）")
            return None, f"配置没敢改：{type(e).__name__}: {e}"
        return rec, ""

    def reject(self, cid: str, why: str = "") -> dict:
        def mutate(doc: dict) -> dict:
            for r in doc.get("items", []):
                if isinstance(r, dict) and r.get("id") == cid:
                    if r.get("status") == PROMOTED:
                        raise ExperienceError("这条已经写进手册了，要退请回滚对应的补丁")
                    r["status"] = REJECTED
                    r["note"] = str(why or "")[:200]
                    r["rejected_at"] = time.time()
            return doc

        locked_update_json(self.path, mutate, default={"version": 1, "items": []})
        row = self.get(cid)
        if row is None:
            raise ExperienceError(f"没有这条候选: {cid}")
        return row

    def reset(self) -> None:
        """清空（测试/换了游戏之后用）。"""
        atomic_write_json(self.path, {"version": 1, "items": []}, indent=2)

    # ---------------- 给页面 / 助手 ----------------

    def summary(self) -> dict:
        rows = self.items(limit=200)
        pending = [r for r in rows if r.get("status") == CANDIDATE]
        return {
            "counts": {s: sum(1 for r in rows if r.get("status") == s)
                       for s in (CANDIDATE, PROMOTED, REJECTED)},
            "ready": [r["id"] for r in pending if self.ready(r)[0]],
            "pending": len(pending),
            "min_seen": self.min_seen,
            "min_span_s": self.min_span_s,
        }

    def as_public(self, limit: int = 20) -> list[dict]:
        out = []
        for r in self.items(limit=limit):
            ok, why = self.ready(r)
            out.append({
                "id": r.get("id"), "code": r.get("code"),
                "label": r.get("label") or r.get("code"),
                "kind": r.get("kind") or "lesson",
                "when": r.get("when"), "do": r.get("do"),
                "seen": int(r.get("seen") or 0),
                "status": r.get("status"), "patch_id": r.get("patch_id") or "",
                "rule": r.get("rule") or {},
                "rule_blocked": r.get("rule_blocked") or "",
                "rule_patch_id": r.get("rule_patch_id") or "",
                "rule_error": r.get("rule_error") or "",
                "config": r.get("config") or {},
                "config_blocked": r.get("config_blocked") or "",
                "config_why": r.get("config_why") or "",
                "config_patch_id": r.get("config_patch_id") or "",
                "config_error": r.get("config_error") or "",
                "first_seen": r.get("first_seen"), "last_seen": r.get("last_seen"),
                "ready": ok, "blocked_by": why,
                "evidence": (r.get("evidence") or [])[-3:],
            })
        return out

    def rules_learned(self) -> list[dict]:
        """已经写进手册的那些候选规则（页面上"它现在会自己做的事"）。"""
        return [{"id": r.get("id"), "code": r.get("code"),
                 "rule": r.get("rule") or {}, "patch_id": r.get("rule_patch_id") or ""}
                for r in self.items(status=PROMOTED, limit=50)
                if isinstance(r.get("rule"), dict) and r.get("rule")]
