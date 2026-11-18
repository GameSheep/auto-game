"""经验库 —— 人写的玩法说明 + 机器踩出来的经验，放在同一个目录里。

用户的诉求原话："这次行为要作为经验存储出来，和我最开始整理的游戏内容放一起，
作为以后的依据和行为数据库。"

所以目录就是那个"一起"：

    profiles/<游戏id>/knowledge/
      game.md              人写的：玩法、界面、术语、禁忌（只读，机器不许改）
      experiences.jsonl    机器沉淀的：局面 → 动作 → 结果（可增可改状态）

两条设计原则，都是为了不把幻觉固化成"经验"：

1. **局面用结构化指纹，不用像素**。同一类局面（"未识别界面 + 3 只怪 + 无光柱"）
   在不同帧里像素完全不同，用 hash 永远匹配不上；用定性特征桶才稳定。

2. **经验必须先验证才能直出**。模型给的新做法默认 candidate，只有真的执行过、
   并且对账显示有效，才升 confirmed；直出（L0）只认 confirmed。
   没有这条，模型编一句"这种情况应该先按 ESC"，第二天就会被当成真理每帧执行。

状态机：
    candidate --验证成功×2--> confirmed --再失败×2--> rejected
    candidate --验证失败×2--> rejected（保留，作为"别这么做"的反例）
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

STATUS_CANDIDATE = "candidate"
STATUS_CONFIRMED = "confirmed"
STATUS_REJECTED = "rejected"

# 升级/降级阈值。故意定得保守：宁可多问几次模型，也不要让一条错经验变真理。
CONFIRM_HITS = 2
REJECT_MISSES = 2
MAX_MISS_RATE = 0.34

# 落盘时额外写的字段（ts 是审计用的，不属于 Experience 本体）
_EXP_FIELDS = {"id", "fp", "situation", "action", "source", "status",
               "hits", "misses", "created", "updated", "note"}


def bucket(n: int, edges: tuple[int, ...], labels: tuple[str, ...]) -> str:
    for e, lab in zip(edges, labels):
        if n <= e:
            return lab
    return labels[-1]


def fingerprint(probe: dict[str, Any]) -> str:
    """局面指纹：把连续量装进桶里，得到稳定可匹配的短串。

    只放"决定该做什么"的维度。放太多会让匹配率趋近于零（每条经验只能用一次）；
    放太少会把不同局面混为一谈（比如"有怪"和"没怪"必须分开，否则"别站着挨打"
    这条经验会被套到安全局面上去）。
    """
    in_game = "1" if probe.get("in_game") else "0"
    gate = str(probe.get("gate_reason") or "none")
    # 门禁原因分两类：认出来了(skillbar/negative) 与 没认出来(unmatched/none)
    if gate.startswith("negative"):
        gate = "menu"
    elif gate in ("skillbar", ""):
        gate = "combat" if probe.get("in_game") else "unknown"
    p = int(probe.get("pillars") or 0)
    m = int(probe.get("monsters") or 0)
    a = "1" if probe.get("aframe") else "0"
    return (f"in={in_game}|gate={gate}|pil={bucket(p, (0, 1, 3), ('0', '1', '2-3', '4+'))}"
            f"|mon={bucket(m, (0, 1, 2), ('0', '1', '2', '3+'))}|a={a}")


@dataclass
class Experience:
    """一条经验：什么局面下、做了什么、结果如何。"""

    id: str
    fp: str
    situation: str
    action: dict[str, Any]          # {"id": "attack_nearest", "params": {...}}
    source: str = "laya"            # laya / planner / human
    status: str = STATUS_CANDIDATE
    hits: int = 0
    misses: int = 0
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    note: str = ""                  # 模型当时给的理由，便于人工复核

    @property
    def action_id(self) -> str:
        return str((self.action or {}).get("id", ""))

    @property
    def miss_rate(self) -> float:
        total = self.hits + self.misses
        return self.misses / total if total else 0.0

    @property
    def usable(self) -> bool:
        """能不能被直接执行（L0 直出）。只有 confirmed 可以。"""
        return self.status == STATUS_CONFIRMED and self.miss_rate <= MAX_MISS_RATE


class KnowledgeBase:
    """一个游戏一个库。不存在就建，空的也能跑（就是全都得问模型）。"""

    def __init__(self, knowledge_dir: Path) -> None:
        # 参数就是那个目录本身（profile 下的 knowledge/），不再往下拼一层，
        # 否则会变成 knowledge/knowledge —— 用户找 game.md 会找不到。
        self.knowledge_dir = Path(knowledge_dir)
        self.root = self.knowledge_dir.parent
        self.exp_path = self.knowledge_dir / "experiences.jsonl"
        self.game_md = self.knowledge_dir / "game.md"
        self._lock = threading.RLock()
        self._items: dict[str, Experience] = {}
        self._game_cache: tuple[float, str] | None = None
        self._loaded = False
        self.stats = {"recalls": 0, "hits": 0, "remembered": 0, "confirmed": 0,
                      "rejected": 0, "verified": 0}

    # ---------------- 加载 ----------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self.knowledge_dir.mkdir(parents=True, exist_ok=True)
            if self.exp_path.exists():
                for line in self.exp_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        exp = Experience(**{k: v for k, v in obj.items()
                                            if k in _EXP_FIELDS})
                    except Exception:  # noqa: BLE001
                        log.warning("经验库有一行解析失败（已跳过）: %s", line[:80])
                        continue
                    self._items[exp.id] = exp
            self._loaded = True

    # ---------------- 人写的部分 ----------------

    def read_game_notes(self, max_chars: int = 2000) -> str:
        """读 game.md。人写的东西优先于任何模型输出 —— 这是"百分百正确的参考"。"""
        if not self.game_md.exists():
            return ""
        try:
            mtime = self.game_md.stat().st_mtime
            if self._game_cache and self._game_cache[0] == mtime:
                text = self._game_cache[1]
            else:
                text = self.game_md.read_text(encoding="utf-8")
                self._game_cache = (mtime, text)
        except Exception:  # noqa: BLE001
            return ""
        text = text.strip()
        return text[:max_chars]

    # ---------------- 检索 ----------------

    def recall(self, fp: str, limit: int = 4) -> list[Experience]:
        """按指纹取经验：精确命中优先，其次同门禁类别（gate= 段相同）。

        允许模糊，是因为严格指纹匹配率太低 —— 桶边界上差一个"2-3 只怪 / 3+ 只怪"
        就把可用经验全滤掉了。但模糊只降级"参考"，不降级"可用"：
        usable 仍然只认精确 fp。
        """
        self._ensure_loaded()
        with self._lock:
            self.stats["recalls"] += 1
            exact = [e for e in self._items.values()
                     if e.fp == fp and e.status != STATUS_REJECTED]
            if exact:
                self.stats["hits"] += 1
                return sorted(exact, key=lambda e: (not e.usable, -e.hits))[:limit]
            gate = _gate_of(fp)
            loose = [e for e in self._items.values()
                     if _gate_of(e.fp) == gate and e.status == STATUS_CONFIRMED]
            return sorted(loose, key=lambda e: -e.hits)[:limit]

    def find_usable(self, fp: str) -> Experience | None:
        """L0 直出用：只有同指纹 + confirmed + 失败率达标才算。"""
        self._ensure_loaded()
        with self._lock:
            for e in self._items.values():
                if e.fp == fp and e.usable:
                    return e
        return None

    def _rejected(self, fp: str, limit: int = 2) -> list[Experience]:
        self._ensure_loaded()
        with self._lock:
            rows = [e for e in self._items.values()
                    if e.fp == fp and e.status == STATUS_REJECTED]
        return sorted(rows, key=lambda e: -e.misses)[:limit]

    # ---------------- 写入 ----------------

    def remember(self, situation: str, action: dict[str, Any], source: str,
                 fp: str, note: str = "") -> Experience:
        """记一条新经验（candidate）。同一指纹下同一动作去重，避免刷爆文件。"""
        self._ensure_loaded()
        with self._lock:
            for e in self._items.values():
                if e.fp == fp and e.action_id == str((action or {}).get("id")):
                    e.updated = time.time()
                    if note:
                        e.note = note[:300]
                    self._append(asdict(e))
                    return e
            exp = Experience(
                id=self._next_id(), fp=fp, situation=situation[:300],
                action=dict(action or {}), source=source, note=note[:300],
            )
            self._items[exp.id] = exp
            self.stats["remembered"] += 1
            self._append(asdict(exp))
            log.info("沉淀经验 %s [%s] %s ← %s", exp.id, source, exp.action_id, fp)
            return exp

    def verify(self, exp_id: str, ok: bool, detail: str = "") -> str:
        """对账回写：有效/无效。返回新状态。"""
        self._ensure_loaded()
        with self._lock:
            e = self._items.get(exp_id)
            if e is None:
                return ""
            if ok:
                e.hits += 1
            else:
                e.misses += 1
            e.updated = time.time()
            self.stats["verified"] += 1
            old = e.status
            if e.status != STATUS_REJECTED:
                if e.hits >= CONFIRM_HITS and e.miss_rate <= MAX_MISS_RATE:
                    e.status = STATUS_CONFIRMED
                elif e.misses >= REJECT_MISSES:
                    e.status = STATUS_REJECTED
            if e.status != old:
                self.stats["confirmed" if e.status == STATUS_CONFIRMED else "rejected"] += 1
                log.info("经验 %s 状态 %s → %s（hit %d / miss %d）%s",
                         e.id, old, e.status, e.hits, e.misses, detail[:80])
            self._append(asdict(e))
            return e.status

    def _next_id(self) -> str:
        best = 0
        for k in self._items:
            if k.startswith("exp_"):
                try:
                    best = max(best, int(k[4:]))
                except ValueError:
                    continue
        return f"exp_{best + 1:04d}"

    def _append(self, row: dict[str, Any]) -> None:
        """追加一行。整条记录都是"最后一次状态"，重放时后写覆盖先写。"""
        try:
            self.knowledge_dir.mkdir(parents=True, exist_ok=True)
            with self.exp_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({**row, "ts": round(time.time(), 3)},
                                   ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001
            log.debug("经验写入失败", exc_info=True)

    # ---------------- 给模型看 ----------------

    def prompt_block(self, fp: str, limit: int = 4) -> str:
        """把"人说的"和"机器试过的"拼成一段给模型的背景。人说的排前面。"""
        parts: list[str] = []
        notes = self.read_game_notes()
        if notes:
            parts.append("## 人类提供的游戏说明（最高优先级，与之冲突以它为准）\n" + notes)
        exp = self.recall(fp, limit)
        # 失败的经验单独补进来：它在 recall 里被排除（不许直出），但"这么做没用"
        # 对模型是有价值的约束 —— 否则它会建议一个已经证明无效的做法。
        rejected = self._rejected(fp, limit=2)
        if rejected:
            exp = exp + rejected
        if exp:
            lines = []
            for e in exp:
                mark = {STATUS_CONFIRMED: "已验证", STATUS_CANDIDATE: "待验证",
                        STATUS_REJECTED: "已知无效"}[e.status]
                lines.append(f"- [{mark}] 局面 {e.situation} → {e.action_id}"
                             f"（hit {e.hits}/miss {e.misses}）")
            parts.append("## 这台机器在类似局面上试过的做法\n" + "\n".join(lines))
        return "\n\n".join(parts)

    def snapshot(self) -> dict[str, Any]:
        self._ensure_loaded()
        with self._lock:
            items = list(self._items.values())
            return {
                "path": str(self.knowledge_dir),
                "has_game_notes": self.game_md.exists(),
                "total": len(items),
                "confirmed": sum(1 for e in items if e.status == STATUS_CONFIRMED),
                "candidate": sum(1 for e in items if e.status == STATUS_CANDIDATE),
                "rejected": sum(1 for e in items if e.status == STATUS_REJECTED),
                "recent": [
                    {"id": e.id, "situation": e.situation, "action": e.action_id,
                     "status": e.status, "hits": e.hits, "misses": e.misses}
                    for e in sorted(items, key=lambda x: -x.updated)[:5]
                ],
                "stats": dict(self.stats),
            }


def _gate_of(fp: str) -> str:
    for seg in fp.split("|"):
        if seg.startswith("gate="):
            return seg[5:]
    return ""
