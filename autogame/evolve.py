"""二次进化 —— 在设备上吃材料、长本事，但每一代都能退回去。

## 这个模块解决什么

基础机器人（引擎）不该为某个游戏改代码。游戏差异全部落在 `profiles/<id>/`：
模板、阈值、动作、术语、经验。用户装好设备后，往 profile 里不断放截图 + 写玩法，
程序据此"再生"出一份针对这个游戏的配方 —— 这就是二次进化。

## 为什么不是"让模型改 Python"

在 RK3566 上，引擎代码被模型改坏的后果是变砖：没有 git、没有编译器、没有 shell
能救回来。而且"模型改的这行是什么意思"永远说不清。所以：

    进化的是 **配方（rules + templates + knowledge）**，不是引擎。

配方有两种形态，对应两种确定性：

    rules.yaml     受限制的可执行规则（数据，不是代码）——错了立刻能看出是哪一条
    knowledge/     进 prompt 的文字（术语、流程、经验）——模型读了会参考，但不是硬编码

一条硬规则：**没有标答的材料只能进 knowledge，不能进 rules**。
因为 rules 命中就直接派动作、不问模型，错了没人兜；而 prompt 里的文字错了，
模型还有机会不采信。这个区分是"敢不敢让它自己动手"的分水岭。

## 为什么一代一个目录

    profiles/<id>/evolve/
        active              指针文件，内容是 "gen-001"（写临时文件再 rename，断电不留半代）
        history.jsonl       每代的材料指纹、分数、接受/拒绝原因（追加写）
        wanted.json         自我纠正：系统认为"我这儿不确定，需要哪种样本"
        gen-000/            出厂基线（= 手册第一遍编译的产物）
        gen-001/            第一次吃到新材料后的一代

原地改参数的话，学坏了没人能回到上一版 —— 而这正是"持续追加截图"最常出的事：
新样本带来的改动把旧样本搞坏了（灾难性遗忘）。所以每代必须独立可跑回归，
过了才切换指针。

## 回归闸门（这是"进化"和"乱改"的唯一分界）

新代必须满足：strict 通过的条数 >= 当前 active 代，且没有新增失败。
差一条就不上线 —— 保留在目录里标 `rejected`，并写清是哪条退化了。
宁可停在原地，也不要为了认一张新图而把手册里已经对的画面搞错。

## 材料体检（回答"图不对/不够/太模糊"）

进化前先算"这些材料够不够学"，不够就不许出代，只输出缺什么。三项最关键：

    1. 正例要 >= 2 张不同帧。一张图过阈值毫无信息量 —— 模板是从它身上裁的，
       自匹配 0.99 是必然的，证明不了"别的战斗帧也能认出来"。
    2. 反例要 >= 2 张。一张反例只能挡住一个具体画面（暗黑类游戏开背包时血球
       技能栏都还在，只靠"有没有 HUD"判不出来）。
    3. 放进去的图必须被手册引用过。用户常放 20 张图、手册只写了 3 张，
       剩下 17 张程序根本不会看 —— 这是"图给了却没用"的头号原因。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

log = logging.getLogger(__name__)

#: 规则里允许出现的事实字段。**只列检测器/裁决层真的会产出、且已进台账的**。
#: 不加"猜出来的"字段（比如 hp_frac）—— 画面里没有这个字段，引用它的规则
#: 永远求值为 unknown，表现为"这条规则从来不生效"，比报错更难查。
FACTS: dict[str, tuple[str, str]] = {
    "in_game": ("bool", "门禁判定：这是不是游戏内可操作画面"),
    "gate": ("str", "门禁结论：skillbar / negative:<名字> / unmatched / no_template"),
    "pillars": ("int", "掉落光柱数量"),
    "monsters": ("int", "可见怪物数量"),
    "aframe": ("bool", "当前是否有可拾取物 A 框"),
    "ground_labels": ("int", "地面物品文字标签数"),
    "intent": ("str", "当前意图（FSM 阶段名，如 explore / attack / loot）"),
    "miss_streak": ("int", "连续失败次数（监督层体检结果）"),
}

OPS: dict[str, str] = {
    "==": "等于", "!=": "不等于",
    ">=": "不小于", "<=": "不大于", ">": "大于", "<": "小于",
    "in": "属于（右侧给列表）", "not_in": "不属于",
}

#: 禁止出现在规则里的东西。不是安全考虑（规则是数据，执行不了任意代码），
#: 而是这些写法会让"为什么这条规则没生效"变成一道谜题。
_BAD_TOKENS = ("__", "import", "eval", "exec", "open(", "lambda", "os.", "sys.")


# ---------------------------------------------------------------------------
# 动作白名单：从注册表派生，不硬编码
# ---------------------------------------------------------------------------

def action_catalog() -> dict[str, set[str]]:
    """动作 id → 允许的参数名。派生自 ActionRegistry，加新动作不需要改这里。"""
    from .decision.actions import ActionRegistry

    reg = ActionRegistry()
    out: dict[str, set[str]] = {}
    for spec in reg.all():
        params = getattr(spec, "params", None) or {}
        if isinstance(params, dict):
            names = {str(k) for k in params}
        else:
            names = {str(x) for x in params}
        out[str(spec.id)] = names
    return out


# ---------------------------------------------------------------------------
# 代（generation）目录
# ---------------------------------------------------------------------------

def evolve_root(profile: Any) -> Path:
    return Path(profile.root) / "evolve"


def gen_dir(profile: Any, gen_id: str) -> Path:
    return evolve_root(profile) / gen_id


@dataclass
class Generation:
    id: str
    root: Path
    status: str = "baseline"        # baseline / active / rejected / retired
    parent: str | None = None
    created: float = 0.0
    digest: str = ""                # 材料指纹
    score: dict = field(default_factory=dict)
    rules_source: str = "none"      # model / manual / none
    note: str = ""

    @property
    def rules_path(self) -> Path:
        return self.root / "rules.yaml"

    @property
    def templates_dir(self) -> Path:
        """本代模板目录。空目录 = 沿用 profile 的模板（绝大多数代都是这样）。"""
        return self.root / "templates"

    def resolved_templates(self, profile: Any) -> Path | None:
        """真正要用的模板目录。本代有自己的就用本代的，否则用 profile 的。"""
        d = self.templates_dir
        if d.is_dir() and any(d.glob("*.png")):
            return d
        return Path(profile.templates_dir)

    def load_rules(self) -> list[dict]:
        if not self.rules_path.exists():
            return []
        doc = yaml.safe_load(self.rules_path.read_text(encoding="utf-8")) or {}
        return list(doc.get("rules") or [])

    def to_dict(self) -> dict:
        return {"id": self.id, "status": self.status, "parent": self.parent,
                "created": self.created, "digest": self.digest,
                "score": self.score, "rules_source": self.rules_source,
                "note": self.note}


def _read_gen(profile: Any, path: Path) -> Generation | None:
    meta = path / "gen.yaml"
    if not meta.exists():
        return None
    try:
        d = yaml.safe_load(meta.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        log.warning("gen.yaml 读不出: %s", meta, exc_info=True)
        return None
    return Generation(
        id=str(d.get("id") or path.name),
        root=path,
        status=str(d.get("status") or "candidate"),
        parent=d.get("parent"),
        created=float(d.get("created") or 0.0),
        digest=str(d.get("digest") or ""),
        score=dict(d.get("score") or {}),
        rules_source=str(d.get("rules_source") or "none"),
        note=str(d.get("note") or ""),
    )


def _write_gen(gen: Generation) -> None:
    gen.root.mkdir(parents=True, exist_ok=True)
    doc = gen.to_dict()
    (gen.root / "gen.yaml").write_text(
        yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")


def list_generations(profile: Any) -> list[Generation]:
    root = evolve_root(profile)
    if not root.is_dir():
        return []
    out = []
    for p in sorted(root.iterdir()):
        if not p.is_dir() or not p.name.startswith("gen-"):
            continue
        g = _read_gen(profile, p)
        if g:
            out.append(g)
    return out


def active_id(profile: Any) -> str | None:
    p = evolve_root(profile) / "active"
    if not p.exists():
        return None
    s = p.read_text(encoding="utf-8").strip()
    return s or None


def active_generation(profile: Any) -> Generation | None:
    gid = active_id(profile)
    if not gid:
        return None
    p = gen_dir(profile, gid)
    return _read_gen(profile, p) if p.is_dir() else None


def set_active(profile: Any, gen_id: str) -> None:
    """切换指针。写临时文件再 rename —— 断电时要么是旧的、要么是新的，不留半代。"""
    root = evolve_root(profile)
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / ".active.tmp"
    tmp.write_text(gen_id + "\n", encoding="utf-8")
    os.replace(tmp, root / "active")


def _history_append(profile: Any, row: dict) -> None:
    root = evolve_root(profile)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "history.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def history(profile: Any, limit: int = 50) -> list[dict]:
    p = evolve_root(profile) / "history.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-limit:]


# ---------------------------------------------------------------------------
# 材料指纹：材料没变就不该出新代
# ---------------------------------------------------------------------------

def materials_digest(profile: Any) -> str:
    """把"程序会看的全部材料"折成一个指纹。

    只算会看的东西：手册文本、refs 里的图（名字 + 尺寸）、templates 里的模板。
    inbox 不算 —— 那是待标定的，还没进教学范围；算进去会让"丢一张图进 inbox"
    就触发一整代验证，白烧设备算力。
    """
    h = hashlib.sha1()
    root = Path(profile.root)

    for p in sorted((root / "lessons").glob("*.md")) if (root / "lessons").is_dir() else []:
        h.update(p.name.encode())
        h.update(p.read_bytes())

    refs = root / "refs"
    if refs.is_dir():
        for p in sorted(refs.iterdir()):
            if not p.is_file():
                continue
            st = p.stat()
            h.update(f"{p.name}:{st.st_size}".encode())

    tdir = Path(profile.templates_dir)
    if tdir.is_dir():
        for p in sorted(tdir.glob("*.png")):
            st = p.stat()
            h.update(f"tpl:{p.name}:{st.st_size}".encode())

    labels = Path(profile.labels_path)
    if labels.exists():
        h.update(b"labels:")
        h.update(labels.read_bytes())

    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# 材料体检
# ---------------------------------------------------------------------------

@dataclass
class Need:
    what: str
    have: int
    need: int
    why: str
    fix: str
    blocking: bool = True

    @property
    def ok(self) -> bool:
        return self.have >= self.need

    def to_dict(self) -> dict:
        return {"what": self.what, "have": self.have, "need": self.need,
                "ok": self.ok, "blocking": self.blocking,
                "why": self.why, "fix": self.fix}


@dataclass
class Audit:
    needs: list[Need] = field(default_factory=list)
    unused_images: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    n_lessons: int = 0
    n_labels: int = 0
    digest: str = ""

    @property
    def blocking(self) -> list[Need]:
        return [n for n in self.needs if n.blocking and not n.ok]

    @property
    def ready(self) -> bool:
        return not self.blocking

    def to_dict(self) -> dict:
        return {
            "ready": self.ready,
            "digest": self.digest,
            "n_lessons": self.n_lessons,
            "n_labels": self.n_labels,
            "needs": [n.to_dict() for n in self.needs],
            "unused_images": self.unused_images,
            "duplicates": self.duplicates,
        }

    def summary(self) -> str:
        if self.ready:
            return "材料够用，可以出一代"
        items = "；".join(f"{n.what} 有{n.have}缺{n.need - n.have}" for n in self.blocking)
        return f"材料不够，缺：{items}"


def _thumb(path: Path, size: int = 48) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 4 or b.std() < 4:
        return 0.0
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def audit_materials(profile: Any) -> Audit:
    """算"这些材料够不够学"。不够就明说缺什么，不假装能学。

    这是用户反复说的"图不对、不够明确、太少"的直接回答。以前工具不管材料多少
    都硬出一版结果，看起来是交付了，其实是把"材料不足"藏进了不可靠的分数里。
    """
    from .teach import (KIND_OBJECT, KIND_SCENE, _verdict_positive,
                        load_lessons)

    root = Path(profile.root)
    out = Audit(digest=materials_digest(profile))

    lessons = load_lessons(root / "lessons") if (root / "lessons").is_dir() else []
    # Lesson 是"一份文件"，声明是 Claim。数样本要数 Claim —— 一份手册里可以写
    # 十几个界面条目，按文件数会得到"我有 1 份手册"，毫无意义。
    claims = [c for ls in lessons for c in ls.claims]
    out.n_lessons = len(claims)

    scenes = [c for c in claims if c.kind == KIND_SCENE]
    objects = [c for c in claims if c.kind == KIND_OBJECT]

    pos = [c for c in scenes if _verdict_positive(c.get("verdict", "")) is True]
    neg = [c for c in scenes if _verdict_positive(c.get("verdict", "")) is False]

    out.needs.append(Need(
        what="应命中的界面样本（正例）", have=len(pos), need=2,
        why="一张图过阈值毫无信息量：模板就是从它身上裁的，自匹配 0.99 是必然的，"
            "证明不了别的战斗帧也认得出。至少两张不同帧才算「能泛化」。",
        fix="再截 1~2 张同为战斗、但场景/光照/怪不同的图，写进手册并标『判定: 正例』"))

    out.needs.append(Need(
        what="应拦下的界面样本（反例）", have=len(neg), need=2,
        why="一张反例只挡得住一个具体画面。暗黑类游戏开背包/角色面板时血条和技能栏"
            "都还在，只靠「有没有 HUD」判不出来，判错的代价是点一下就换装/洗点。",
        fix="截背包、角色面板、结算页各 1 张，标『判定: 反例』"))

    with_rect = [c for c in objects if c.rect or c.get("pos", "")]
    out.needs.append(Need(
        what="物体样本（要能自动量出框）", have=len(with_rect), need=1,
        why="光柱/掉落物这类目标要有至少一张「确实有它」的图，才能反推颜色、高宽比、"
            "最小像素数。没有样本只能沿用 PC 端的阈值，手机端往往一个都检不出。",
        fix="截一张画面里确实有掉落光柱的图，写『物体 · 掉落光柱』+『在: 画面中央』",
        blocking=False))

    # 台账：没有标答就不能判"这代有没有学坏"
    n_strict = 0
    labels_path = Path(profile.labels_path)
    if labels_path.exists():
        try:
            data = json.loads(labels_path.read_text(encoding="utf-8"))
            n_strict = sum(1 for it in (data.get("items") or []) if it.get("strict"))
        except Exception:  # noqa: BLE001
            log.warning("台账解析失败: %s", labels_path, exc_info=True)
    # 标答有两个来源，都算数：台账里的 strict 条目；手册里带图的「界面」条目
    # （它的『判定: 正例/反例』就是 in_game 的标答，可以直接拿来跑回归）。
    # 只认台账的话，用户写完手册还得再手工抄一遍答案，抄这一步没有任何信息增益。
    n_scene_with_img = sum(1 for c in scenes if c.get("file", ""))
    out.n_labels = max(n_strict, n_scene_with_img)
    out.needs.append(Need(
        what="带标答的样本", have=out.n_labels, need=3,
        why="回归闸门靠它。没有标答就分不清「这代变好了」还是「只是换了个错法」，"
            "那不如不出新代。两个来源都算：台账的 strict 条目，或手册里带图的"
            "『界面』条目（判定: 正例/反例 就是 in_game 的答案）。",
        fix="在手册里给 3 个界面条目补上 `- 图:` 和 `- 判定:`；"
            "或者跑 tools/profile_init.py 生成台账。"))

    # 放进去了但手册没引用 —— "图给了却没用"的头号原因
    used: set[str] = set()
    for c in claims:
        f = c.get("file", "")
        if f:
            used.add(Path(f).name)
    refs = root / "refs"
    if refs.is_dir():
        for p in sorted(refs.iterdir()):
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                if p.name not in used:
                    out.unused_images.append(p.name)

    # 冗余样本：相关 >= 0.96 基本是同一帧的重复
    if refs.is_dir():
        thumbs: dict[str, np.ndarray] = {}
        for p in sorted(refs.iterdir()):
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                t = _thumb(p)
                if t is not None:
                    thumbs[p.name] = t
        names = list(thumbs)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if _corr(thumbs[names[i]], thumbs[names[j]]) >= 0.96:
                    out.duplicates.append(f"{names[i]} ≈ {names[j]}")

    return out


# ---------------------------------------------------------------------------
# 受限 DSL：校验 + 求值
# ---------------------------------------------------------------------------

@dataclass
class RuleError:
    index: int
    rule_id: str
    message: str

    def __str__(self) -> str:
        return f"第 {self.index + 1} 条（{self.rule_id or '无 id'}）：{self.message}"


def _check_cond(cond: Any, path: str, catalog: dict[str, set[str]],
                errors: list[str]) -> None:
    """递归校验一个条件。允许 cond / all / any 三种组合，其余一律拒。"""
    if not isinstance(cond, dict):
        errors.append(f"{path}: 条件必须是对象，收到 {type(cond).__name__}")
        return
    if "all" in cond or "any" in cond:
        key = "all" if "all" in cond else "any"
        subs = cond[key]
        if not isinstance(subs, list) or not subs:
            errors.append(f"{path}.{key}: 必须是至少一项的列表")
            return
        for i, s in enumerate(subs):
            _check_cond(s, f"{path}.{key}[{i}]", catalog, errors)
        extra = set(cond) - {key}
        if extra:
            errors.append(f"{path}: {key} 里不能再混 {sorted(extra)}")
        return

    fact = cond.get("fact")
    if fact not in FACTS:
        errors.append(f"{path}: 未知事实 {fact!r}；可用 {sorted(FACTS)}")
        return
    op = cond.get("op", "==")
    if op not in OPS:
        errors.append(f"{path}: 未知比较符 {op!r}；可用 {sorted(OPS)}")
        return
    if "value" not in cond:
        errors.append(f"{path}: 缺 value")
        return

    want_type = FACTS[fact][0]
    val = cond["value"]
    lists = op in ("in", "not_in")
    if lists:
        if not isinstance(val, list) or not val:
            errors.append(f"{path}: {op} 的 value 必须是非空列表")
            return
        vals = val
    else:
        vals = [val]
    for v in vals:
        if want_type == "bool" and not isinstance(v, bool):
            errors.append(f"{path}: 事实 {fact} 是布尔，value 却是 {v!r}")
        elif want_type == "int" and (not isinstance(v, int) or isinstance(v, bool)):
            errors.append(f"{path}: 事实 {fact} 是整数，value 却是 {v!r}")
        elif want_type == "str" and not isinstance(v, str):
            errors.append(f"{path}: 事实 {fact} 是字符串，value 却是 {v!r}")


def validate_rules(doc: Any) -> tuple[list[dict], list[RuleError]]:
    """校验 rules.yaml。返回 (能用的规则, 错误列表)。

    口径：**只要有错，整份文档不进规则集**。不做"能用的先用上"——半份规则跑起来，
    行为会变得无法从文档推断，排查时人会以为"我明明写了这条"。错误全部报出来，
    模型可以重写一次，人也可以直接改。
    """
    catalog = action_catalog()
    if not isinstance(doc, dict):
        return [], [RuleError(0, "", "rules.yaml 顶层必须是对象（含 rules: 列表）")]
    rules_in = doc.get("rules")
    if rules_in is None:
        return [], []
    if not isinstance(rules_in, list):
        return [], [RuleError(0, "", "rules 必须是列表")]

    errors: list[RuleError] = []
    out: list[dict] = []
    seen: set[str] = set()

    for i, r in enumerate(rules_in):
        rid = str((r or {}).get("id") or "") if isinstance(r, dict) else ""
        if not isinstance(r, dict):
            errors.append(RuleError(i, rid, "规则必须是对象"))
            continue
        problems: list[str] = []

        if not rid:
            problems.append("缺 id（要能一眼指认是哪条）")
        elif rid in seen:
            problems.append(f"id 重复：{rid}")
        elif rid.startswith("__"):
            problems.append("id 不能以双下划线开头")

        if "when" not in r:
            problems.append("缺 when")
        else:
            _check_cond(r["when"], "when", catalog, problems)

        then = r.get("then")
        if not isinstance(then, dict):
            problems.append("then 必须是对象 {action: ...}")
        else:
            aid = then.get("action")
            if aid not in catalog:
                problems.append(f"未知动作 {aid!r}；可用 {sorted(catalog)}")
            else:
                params = then.get("params") or {}
                if not isinstance(params, dict):
                    problems.append("then.params 必须是对象")
                else:
                    bad = set(params) - catalog[aid]
                    if bad:
                        problems.append(
                            f"动作 {aid} 不认参数 {sorted(bad)}；可用 {sorted(catalog[aid])}")
                    for k, v in params.items():
                        if isinstance(v, (dict, list)):
                            problems.append(
                                f"参数 {k} 只接受标量（数值/字符串/布尔），"
                                "嵌套结构在动作层没有对应的处理逻辑")

        for tok in _BAD_TOKENS:
            blob = json.dumps(r, ensure_ascii=False)
            if tok in blob:
                problems.append(f"出现不允许的写法 {tok!r}")

        if problems:
            for p in problems:
                errors.append(RuleError(i, rid, p))
            continue

        seen.add(rid)
        out.append({
            "id": rid,
            "when": r["when"],
            "then": {"action": then["action"],
                     "params": dict(then.get("params") or {})},
            "why": str(r.get("why") or ""),
            "confidence": float(r.get("confidence") or 0.5),
        })

    if errors:
        return [], errors
    return out, []


def _fact_value(facts: dict, name: str) -> Any:
    if name not in facts:
        return None
    return facts[name]


def _match(cond: dict, facts: dict) -> bool | None:
    """三值逻辑：True / False / None（未知）。

    未知不能当 False —— 那会让"缺一个字段"变成"这条规则静默失效"，人会以为
    规则写错了。上层拿到 None 就记 audit 日志，说清是哪个字段没产出。
    """
    if "all" in cond:
        vals = [_match(c, facts) for c in cond["all"]]
        if any(v is False for v in vals):
            return False
        return None if any(v is None for v in vals) else True
    if "any" in cond:
        vals = [_match(c, facts) for c in cond["any"]]
        if any(v is True for v in vals):
            return True
        return None if any(v is None for v in vals) else False

    v = _fact_value(facts, cond["fact"])
    if v is None:
        return None
    op = cond.get("op", "==")
    want = cond["value"]
    try:
        if op == "==":
            return v == want
        if op == "!=":
            return v != want
        if op == ">=":
            return v >= want
        if op == "<=":
            return v <= want
        if op == ">":
            return v > want
        if op == "<":
            return v < want
        if op == "in":
            return v in want
        if op == "not_in":
            return v not in want
    except TypeError:
        return None
    return None


def evaluate(rules: list[dict], facts: dict) -> dict:
    """按文件顺序取第一条命中。顺序即优先级 —— 人不用去猜 confidence 谁大。

    返回 {hit, rule_id, action, params, unknown}。unknown 列的是"因为字段缺失
    而求值不出结果"的规则 id，用来在日志里说清"这条为什么不生效"。
    """
    unknown: list[str] = []
    for r in rules:
        m = _match(r["when"], facts)
        if m is True:
            return {"hit": True, "rule_id": r["id"],
                    "action": r["then"]["action"], "params": dict(r["then"]["params"]),
                    "unknown": unknown}
        if m is None:
            unknown.append(r["id"])
    return {"hit": False, "rule_id": "", "action": "", "params": {},
            "unknown": unknown}


# ---------------------------------------------------------------------------
# 运行时接线：让 rules.yaml 真的被用到
# ---------------------------------------------------------------------------

def facts_from_probe(probe: dict) -> dict:
    """把裁决层的 probe 翻成规则能用的事实。

    只翻**真的存在**的字段，翻不出来就是 None。不在这里填默认值 ——
    一个假的默认值会让规则在错误的场景下命中，那比"不生效"危险得多：
    不生效只是没帮上忙，错误命中是主动做错事。
    """
    src = dict(probe or {})
    if "gate" not in src and "gate_reason" in src:
        src["gate"] = src["gate_reason"]
    return {k: src[k] for k in FACTS if k in src and src[k] is not None}


class RuleSet:
    """一代规则的可执行形态。加载时校验，不合法就整份不用。"""

    def __init__(self, rules: list[dict] | None = None, source: str = ""):
        self.rules = list(rules or [])
        self.source = source

    @classmethod
    def load(cls, path: Any) -> "RuleSet":
        """从 rules.yaml 加载。文件不存在 → 空规则集（不是错误）。

        校验不过时**大声报错并退回空集**：静默降级会让人以为"我写的规则在跑"，
        然后花半天排查为什么行为没变。
        """
        if not path:
            return cls()
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            log.error("rules.yaml 解析失败，本代规则不生效: %s", p, exc_info=True)
            return cls()
        rules, errors = validate_rules(doc)
        if errors:
            log.error("rules.yaml 有 %d 处问题，整份不生效（%s）：%s",
                      len(errors), p, "; ".join(str(e) for e in errors[:3]))
            return cls()
        if rules:
            log.info("规则集已加载：%d 条（%s）", len(rules), p)
        return cls(rules, str(p))

    def __bool__(self) -> bool:
        return bool(self.rules)

    def decide(self, probe: dict) -> dict | None:
        """命中的规则 → {action, params, rule_id, unknown}；没命中 → None。

        动作合法性在这里再查一次：规则是数据，但动作 id 可能过期
        （比如某次重构把 dodge 改名了）。宁可这一步拒掉，也不要让 FSM 拿到
        一个不存在的动作然后整帧空转。
        """
        if not self.rules:
            return None
        facts = facts_from_probe(probe)
        if not facts:
            return None
        hit = evaluate(self.rules, facts)
        if not hit["hit"]:
            if hit["unknown"]:
                log.debug("规则因字段缺失未求值: %s", hit["unknown"])
            return None
        if hit["action"] not in action_catalog():
            log.error("规则 %s 指向未知动作 %s，已忽略", hit["rule_id"], hit["action"])
            return None
        return hit


# ---------------------------------------------------------------------------
# 回归：跑台账
# ---------------------------------------------------------------------------

def _synthetic_labels(profile: Any) -> list[dict]:
    """从手册的『界面』条目合成标答。只取"有图 + 有明确判定"的，判不出的跳过。"""
    from .teach import KIND_SCENE, _verdict_positive, load_lessons

    root = Path(profile.root)
    if not (root / "lessons").is_dir():
        return []
    out = []
    for ls in load_lessons(root / "lessons"):
        for c in ls.claims:
            if c.kind != KIND_SCENE:
                continue
            img = c.get("file", "")
            verdict = _verdict_positive(c.get("verdict", ""))
            if not img or verdict is None:
                continue
            rel = img
            if not (root / rel).exists() and not Path(rel).is_absolute():
                # 手册里常写 ../refs/x.jpg（相对 lessons/），换算成相对 profile
                cand = (root / "lessons" / rel).resolve()
                if cand.exists():
                    rel = cand.as_posix()
            out.append({"file": rel, "scene": c.title, "strict": True,
                        "expect": {"in_game": bool(verdict)},
                        "note": "由手册界面条目合成（判定: " + c.get("verdict", "") + "）"})
    return out


def _measure(frame: np.ndarray, det: Any, engine: Any, tick_hz: float) -> dict:
    """跑两个决策节拍，取实测字段。口径与 tools/ref_check.py 完全一致。

    为什么两个节拍：光柱有 beam_persistence_frames=2 的时域确认，只跑一拍会因为
    "还没确认完"而数出 0 根，表现为"这张图明明有光柱却判没有"。
    """
    ts = 1000
    st = None
    for _ in range(2):
        st = det.process(frame, ts)
        engine.on_scene(st)
        ts += int(1000 / tick_hz)
    return {
        "in_game": bool(st.in_game),
        "pillars": len(st.pillars),
        "aframe": st.aframe is not None,
        "ground_labels": len(st.ground_labels),
        "monsters": len(st.monsters),
        "commands": len(engine.exec.journal),
    }


def score_generation(profile: Any, gen: Generation | None = None,
                     only: str = "") -> dict:
    """跑台账，出分数。gen 为空时用 profile 自己的模板（= 无进化状态）。

    返回 {strict_pass, strict_total, ratio, fail:[...], gap:[...], items:[...]}。
    没台账直接返回 ratio=0 并说明 —— 不能返回 1.0 让"没有验证"看起来像"全过"。
    """
    from .config import DecisionConfig, ExecutorConfig, load_config
    from .decision.fsm import DecisionEngine
    from .executor import DryRunExecutor
    from .vision.detector import Detector

    labels_path = Path(profile.labels_path)
    items: list[dict] = []
    if labels_path.exists():
        data = json.loads(labels_path.read_text(encoding="utf-8"))
        items = [it for it in (data.get("items") or []) if only in it.get("file", "")]
    if not items:
        # 台账还没建（新手最常处在这个状态）→ 用手册里带图的『界面』条目合成。
        # 这不是编答案：『判定: 正例』本来就是人写的"程序应该认出这是战斗页"。
        # 没有这一步，用户写完手册还得手工把同样的答案抄进 labels.json，
        # 而抄写不会带来任何新信息，只会带来"抄错了"这种新问题。
        items = _synthetic_labels(profile)
    if not items:
        return {"strict_pass": 0, "strict_total": 0, "ratio": 0.0,
                "fail": [], "gap": [], "items": [],
                "error": "没有台账，手册里也没有带图的界面条目 —— 没有可验证的东西"}

    # 不能靠 cwd 找 config.yaml：装在 RK3566 上当 systemd 服务跑时，cwd 是 "/"，
    # 依赖 cwd 会让回归悄悄跑成"全部默认阈值"的版本，分数看着正常其实不是我想要的那套。
    cfg_file = Path(profile.root).parent.parent / "config.yaml"
    if not cfg_file.exists():
        cfg_file = Path(__file__).resolve().parent.parent / "config.yaml"
    cfg = load_config(str(cfg_file) if cfg_file.exists() else None, profile=profile)
    tdir = gen.resolved_templates(profile) if gen else Path(profile.templates_dir)
    tick_hz = 2.0

    def _resolve(rel: str) -> Path:
        p = Path(profile.root).parent.parent / rel
        if p.exists():
            return p
        return labels_path.parent / rel

    n_pass = n_total = 0
    fails: list[dict] = []
    gaps: list[dict] = []
    detail: list[dict] = []

    for it in items:
        path = _resolve(it.get("file", ""))
        frame = cv2.imread(str(path))
        strict = bool(it.get("strict"))
        if strict:
            n_total += 1
        if frame is None:
            row = {"file": it.get("file"), "diffs": [f"图读不出来：{path}"],
                   "scene": it.get("scene", ""), "strict": strict}
            detail.append(row)
            if strict:
                fails.append(row)
            continue

        det = Detector(cfg.vision, tdir if tdir and Path(tdir).is_dir() else None)
        ex = DryRunExecutor(ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000))
        eng = DecisionEngine(DecisionConfig(tick_hz=tick_hz), ex)
        eng.start()
        got = _measure(frame, det, eng, tick_hz)

        diffs = []
        for key, want in (it.get("expect") or {}).items():
            if want is None:
                continue
            have = got.get(key)
            if have != want:
                diffs.append(f"{key}: 标答={want} 实测={have}")

        row = {"file": it.get("file"), "scene": it.get("scene", ""),
               "strict": strict, "diffs": diffs,
               "gate": det.diag.get("gate_reason", "")}
        detail.append(row)
        if not diffs:
            if strict:
                n_pass += 1
        elif strict:
            fails.append(row)
        else:
            gaps.append(row)

    ratio = (n_pass / n_total) if n_total else 0.0
    return {"strict_pass": n_pass, "strict_total": n_total,
            "ratio": round(ratio, 4), "fail": fails, "gap": gaps,
            "items": detail}


# ---------------------------------------------------------------------------
# 规则草稿：这一步才需要模型
# ---------------------------------------------------------------------------

_RULE_SYS = """你在把一份人类写的游戏玩法手册，翻译成机器可执行的规则表。

只输出 JSON。格式：
{"rules": [{"id": "r_xxx", "when": <条件>, "then": {"action": "<动作>", "params": {}}, "why": "<一句话依据>"}]}

条件只有三种写法，可以嵌套：
  {"fact": "<事实名>", "op": "<比较符>", "value": <值>}
  {"all": [<条件>, ...]}        # 全部满足
  {"any": [<条件>, ...]}        # 任一满足

事实只有这些（用别的名字会被拒）：
__FACTS__

动作只能是这些（用别的 id 会被拒）：
__ACTIONS__

硬要求：
- 手册里没有明确依据的规则，一条都不要写。宁可 rules 为空。
- 每条 why 必须引用手册原文的关键词，不要自己发挥。
- 不要写"提高胜率""更聪明"这类无法验证的规则。
- 最多 8 条。顺序即优先级。
"""


def _fact_block() -> str:
    return "\n".join(f"  {k}（{t}）：{d}" for k, (t, d) in FACTS.items())


def _action_block() -> str:
    cat = action_catalog()
    return "\n".join(f"  {aid}：参数 {sorted(ps) or '无'}" for aid, ps in sorted(cat.items()))


def draft_rules_with_model(profile: Any, cfg: Any, *, timeout_s: float = 60.0) -> tuple[
        list[dict], list[RuleError], str]:
    """让模型把手册翻成规则草稿，再走 DSL 校验。

    返回 (规则, 错误, 说明)。模型不通就返回空规则 + 说明 —— 这是正常路径，
    不是失败：知识进 prompt 那一半照样生效，只是没有硬规则可用。
    """
    from .decision import llm_client
    from .teach import (KIND_FLOW, KIND_LABEL, KIND_SCENE, KIND_TABOO,
                        load_lessons)

    lessons = load_lessons(Path(profile.root) / "lessons") \
        if (Path(profile.root) / "lessons").is_dir() else []
    if not lessons:
        return [], [], "没有手册，跳过规则草稿"

    want = (KIND_FLOW, KIND_TABOO, KIND_SCENE)
    body = []
    for ls in lessons:
        for c in ls.claims:
            if c.kind not in want:
                continue
            fields = "；".join(f"{k}={v}" for k, v in (c.fields or {}).items())
            body.append(f"## {KIND_LABEL.get(c.kind, c.kind)} · {c.title}\n{fields}\n"
                        + "\n".join(c.body[:8]))
    if not body:
        return [], [], "手册里没有『流程/禁忌/界面』条目，没有可翻译的规则素材"

    arb = getattr(getattr(cfg, "decision", None), "arbiter", None)
    model_cfg = getattr(arb, "laya", None) or getattr(arb, "planner", None)
    if model_cfg is None:
        return [], [], "没有配置模型（decision.arbiter.laya / planner 都为空）"

    # 用 replace 而不是 format：模板里全是 JSON 花括号，format 会把 {"rules"}
    # 当成占位符（踩过，报 KeyError: '"rules"'）。
    system = _RULE_SYS.replace("__FACTS__", _fact_block()).replace(
        "__ACTIONS__", _action_block())
    try:
        raw = llm_client.chat(model_cfg, system,
                              "手册如下：\n\n" + "\n\n".join(body),
                              temperature=0.0, json_mode=True)
        doc = llm_client.extract_json(raw)
    except Exception as e:  # noqa: BLE001
        log.warning("规则草稿生成失败: %s", e)
        return [], [], f"模型不可用或输出不是 JSON：{e}"

    rules, errors = validate_rules(doc)
    if errors:
        return [], errors, f"模型给了 {len(doc.get('rules') or [])} 条，全部被校验拒绝"
    return rules, [], f"模型给了 {len(rules)} 条规则"


# ---------------------------------------------------------------------------
# 出一代
# ---------------------------------------------------------------------------

def _next_id(existing: list[Generation]) -> str:
    nums = []
    for g in existing:
        try:
            nums.append(int(g.id.split("-")[-1]))
        except (ValueError, IndexError):
            continue
    return f"gen-{(max(nums) + 1) if nums else 0:03d}"


def build(profile: Any, cfg: Any, *, force: bool = False,
          use_model: bool = True) -> tuple[Generation | None, str]:
    """吃材料，出一代。返回 (新代, 说明)。材料不够时返回 (None, 原因)。

    流程：体检 → 指纹是否变过 → 编译手册（knowledge）→ 规则草稿（可选模型）
    → 落盘。
    注意这里**不切换 active** —— 上线要走 promote，先过回归。
    """
    from .teach import compile_lessons, load_lessons, write_game_md, write_glossary

    audit = audit_materials(profile)
    if not audit.ready and not force:
        return None, f"材料不够，不出新代。{audit.summary()}"

    gens = list_generations(profile)
    digest = audit.digest
    if not force:
        same = [g for g in gens if g.digest == digest]
        if same:
            return None, (f"材料和 {same[0].id} 完全一样（指纹 {digest}），"
                          "不会为了重跑而多出一代。改了手册或加了图再来。")

    gid = _next_id(gens)
    root = gen_dir(profile, gid)
    root.mkdir(parents=True, exist_ok=True)

    # 1) 编译手册 → knowledge（这一半不需要模型）
    lessons = load_lessons(Path(profile.root) / "lessons") \
        if (Path(profile.root) / "lessons").is_dir() else []
    kdir = root / "knowledge"
    kdir.mkdir(parents=True, exist_ok=True)
    teach_note = ""
    if lessons:
        res = compile_lessons(lessons, profile, cfg)
        write_game_md(res, kdir)
        write_glossary(res, kdir)
        n_claims = sum(len(ls.claims) for ls in lessons)
        teach_note = f"手册 {n_claims} 条声明 / {len(lessons)} 份文件"

    # 2) 规则草稿
    rules: list[dict] = []
    rule_note = ""
    rule_errors: list[RuleError] = []
    if use_model:
        rules, rule_errors, rule_note = draft_rules_with_model(profile, cfg)
    else:
        rule_note = "按参数跳过模型（--no-model）"

    top = {"version": 1,
           "howto": [
               "这是二次进化产出的一代配方。改了要重跑 tools/evolve.py --build 才生效。",
               "when 里的事实字段只能是 autogame/evolve.py 的 FACTS 里那几个；"
               "动作只能是动作注册表里有的。写错会被整份拒绝，不是静默跳过。",
               "顺序即优先级：第一条命中的规则生效。",
               "没有标答的材料不会进这里，只进 knowledge/ 的文字。"],
           "rules": rules}
    (root / "rules.yaml").write_text(
        yaml.safe_dump(top, allow_unicode=True, sort_keys=False), encoding="utf-8")

    prev_active = active_id(profile)
    gen = Generation(id=gid, root=root, status="candidate", parent=prev_active,
                     created=time.time(), digest=digest, rules_source=(
                         "model" if rules else "none"),
                     note="；".join(x for x in (teach_note, rule_note) if x))
    _write_gen(gen)

    if rule_errors:
        (root / "rule_errors.txt").write_text(
            "\n".join(str(e) for e in rule_errors), encoding="utf-8")

    return gen, gen.note


def promote(profile: Any, gen: Generation, *, score: dict | None = None) -> tuple[bool, str]:
    """过闸门才上线。返回 (是否上线, 原因)。

    闸门：strict 通过数不能比当前 active 少，且不能有新增失败。
    这条是"持续学习"和"越学越坏"的唯一分界 —— 新样本带来的改动经常会把旧样本搞坏。
    """
    cur = active_generation(profile)
    if score is None:
        score = score_generation(profile, gen)
    gen.score = {k: score[k] for k in ("strict_pass", "strict_total", "ratio")
                 if k in score}
    gen.score["fail"] = [f.get("file") for f in score.get("fail") or []]

    if score.get("strict_total", 0) == 0:
        gen.status = "rejected"
        gen.note = (gen.note + "；" if gen.note else "") + \
            "没有 strict 标答，无法验证，不上线"
        _write_gen(gen)
        _history_append(profile, {"ts": time.time(), "gen": gen.id, "action": "reject",
                                  "reason": gen.note})
        return False, gen.note

    if cur is None:
        # 第一代：只要台账能全过就算基线。过不了说明手册/模板本来就有问题，
        # 这时候上线等于把问题固化成"基线"，以后再也看不出退化了。
        ok = score["ratio"] >= 1.0
        reason = ("第一代基线，台账全过" if ok
                  else f"第一代就有 {len(score.get('fail') or [])} 条不符，"
                       "先修手册/模板，别急着定基线")
    else:
        prev = int(cur.score.get("strict_pass") or 0)
        now = int(score.get("strict_pass") or 0)
        new_fail = set(gen.score.get("fail") or []) - set(cur.score.get("fail") or [])
        ok = now >= prev
        if ok and new_fail:
            ok = False
            reason = f"通过数没跌，但新增失败：{sorted(new_fail)}"
        elif ok:
            reason = f"通过 {prev} → {now}" + ("（持平）" if now == prev else "")
        else:
            reason = f"通过数从 {prev} 跌到 {now}，不上线"

    if not ok:
        gen.status = "rejected"
        gen.note = (gen.note + "；" if gen.note else "") + reason
        _write_gen(gen)
        _history_append(profile, {"ts": time.time(), "gen": gen.id,
                                  "action": "reject", "reason": reason,
                                  "score": gen.score})
        return False, reason

    if cur is not None:
        cur.status = "retired"
        _write_gen(cur)
    gen.status = "active"
    _write_gen(gen)
    set_active(profile, gen.id)
    _history_append(profile, {"ts": time.time(), "gen": gen.id, "action": "promote",
                              "reason": reason, "score": gen.score})
    return True, reason


def rollback(profile: Any) -> tuple[bool, str]:
    """退回上一代。事故现场的第一动作，必须一条命令能用。"""
    cur = active_generation(profile)
    if cur is None:
        return False, "当前没有 active 代"
    parent = cur.parent
    if not parent:
        return False, f"{cur.id} 是基线，没有上一代可退"
    p = gen_dir(profile, parent)
    if not p.is_dir():
        return False, f"上一代 {parent} 的目录不见了"
    prev = _read_gen(profile, p)
    if prev is None:
        return False, f"上一代 {parent} 的 gen.yaml 读不出来"
    cur.status = "retired"
    _write_gen(cur)
    prev.status = "active"
    _write_gen(prev)
    set_active(profile, prev.id)
    _history_append(profile, {"ts": time.time(), "gen": prev.id,
                              "action": "rollback", "reason": f"从 {cur.id} 退回"})
    return True, f"已退回 {prev.id}"


def evolve(profile: Any, cfg: Any, *, force: bool = False,
           use_model: bool = True) -> dict:
    """一条命令走完：体检 → 出代 → 回归 → 上线或拒绝。"""
    audit = audit_materials(profile)
    if not audit.ready and not force:
        return {"ok": False, "stage": "audit", "audit": audit.to_dict(),
                "message": audit.summary()}

    gen, note = build(profile, cfg, force=force, use_model=use_model)
    if gen is None:
        return {"ok": False, "stage": "build", "audit": audit.to_dict(),
                "message": note}

    score = score_generation(profile, gen)
    ok, reason = promote(profile, gen, score=score)
    return {"ok": ok, "stage": "promote", "gen": gen.id, "audit": audit.to_dict(),
            "score": {k: score[k] for k in ("strict_pass", "strict_total", "ratio")},
            "fail": score.get("fail") or [], "message": reason,
            "promoted": ok}


# ---------------------------------------------------------------------------
# 自我纠正：从运行日志里挖"我在哪儿不确定"
# ---------------------------------------------------------------------------

def collect_wanted(profile: Any, *, limit: int = 200) -> dict:
    """扫审计日志，把"我不确定/我错了"聚成一份"要你补什么材料"。

    这是"自我纠正"唯一诚实的形态。模型自己改参数不叫纠正，叫漂移；
    真正能纠正的是：系统知道自己在哪类画面上没把握，然后向人要那一类样本。

    三个信号：
        supervisor 判 hold/panic   → 它不认识这个画面，可能要动却是菜单
        arbiter 走了 L2             → L0 经验 + L1 清单都没有合适的，规则有缺口
        evaluate 里 unknown 非空    → 规则引用了没产出的字段
    """
    from .teach import load_lessons

    root = Path(profile.root)
    logs = Path("logs")
    signals: dict[str, int] = {}
    samples: list[dict] = []

    sup = logs / "supervisor.jsonl"
    if sup.exists():
        for line in sup.read_text(encoding="utf-8").splitlines()[-limit:]:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = str(row.get("verdict") or row.get("decision") or "")
            if v in ("hold", "panic", "takeover"):
                signals[f"supervisor:{v}"] = signals.get(f"supervisor:{v}", 0) + 1
                if len(samples) < 20:
                    samples.append({"kind": f"supervisor:{v}",
                                    "why": str(row.get("reason") or "")[:160],
                                    "scene": str(row.get("scene") or "")})

    arb = logs / "arbiter.jsonl"
    if arb.exists():
        for line in arb.read_text(encoding="utf-8").splitlines()[-limit:]:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            tier = str(row.get("tier") or row.get("level") or "")
            if tier.upper() in ("L2", "PLANNER"):
                signals["arbiter:L2"] = signals.get("arbiter:L2", 0) + 1
                if len(samples) < 20:
                    samples.append({"kind": "arbiter:L2",
                                    "why": str(row.get("situation") or "")[:160],
                                    "scene": str(row.get("scene") or "")})

    asks: list[str] = []
    for k, n in sorted(signals.items(), key=lambda t: -t[1]):
        if k.startswith("supervisor:panic"):
            asks.append(f"有 {n} 次触发急停：这是「程序完全不认识的画面」。"
                        "截一张当时的样子丢进 inbox/，说清它是哪个界面。")
        elif k.startswith("supervisor:hold"):
            asks.append(f"有 {n} 次被拦住不许动：程序拿不准该不该操作。"
                        "要的是那个界面的反例图（判定: 反例）。")
        elif k.startswith("supervisor:takeover"):
            asks.append(f"有 {n} 次被迫接管：预测和实测对不上，多半是动作没生效。"
                        "补一张那个状态下的截图，并在手册里写清该按什么键。")
        elif k == "arbiter:L2":
            asks.append(f"有 {n} 次不得不问大模型：现有规则和经验覆盖不到。"
                        "把这些情况写成『流程』条目（什么条件下做什么），规则才有依据。")

    lessons = load_lessons(root / "lessons") if (root / "lessons").is_dir() else []
    if not lessons:
        asks.append("手册还是空的。先按 profiles/_template/lessons/example.md 的格式写几条，"
                    "程序才有可执行的东西。")

    doc = {"ts": time.time(), "profile": profile.id,
           "signals": signals, "samples": samples, "asks": asks}
    out = evolve_root(profile) / "wanted.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc
