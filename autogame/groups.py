"""指令组 —— 学完之后能被喊出来的整套流程。

为什么需要这一层
----------------
引擎里已经有三样东西，但它们都回答不了用户问的那句话（"说一声刷k7，它就该懂"）：

    FSM      是"手"：会走、会捡、会点、会打，但不知道自己在做哪个任务
    rules    是"条件反射"：看到光柱就捡，没有先后、没有目标
    recipes  是"面板脚本"：点格子点按钮，只管一屏 UI，管不了"进图打一圈再来一局"

中间缺的是**"一组有序的步骤 + 一个能喊的名字"**。指令组就是填这个缺口的。

它是数据，不是代码
------------------
组能表达的东西只有四种步骤：点某个模板、按某个键、等到某个界面、交给已有循环去跑。
表达不了条件分支、表达不了任意逻辑 —— 这是故意的。想加新玩法就加一个 yaml，
永远不需要改引擎（设备上没有编译器，改引擎等于变砖）。

三层不能混
----------
    world（世界层）  术语、界面、物体、键位、禁忌。**全游戏唯一一份**
    groups（组层）   刷k7、打材料、清包。一组一文件，各写各的
    每代配方          rules.yaml + templates/。自动生成，可整代回退

"进了另一组就把上一组忘了"，根因是把**认识世界**的东西也存在组里。所以这里
定死一条：组只存「步骤 + 引用」，认识世界的那部分永远只有一份，新组只能引用
它，不能自带一套。世界层只增不删；真要删或改，写进冲突记录让人点头，不静默覆盖。

为什么路由"宁可不猜"
--------------------
猜错的代价不是一次点击，是几分钟的错误操作，还可能把装备分解掉。所以分差太小
就返回候选让人选，而不是挑一个最像的。
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

log = logging.getLogger(__name__)

#: 步骤种类。就这四种，加新的要同时改执行器和校验，不能只在 yaml 里写。
STEP_KINDS = ("click", "key", "wait", "delegate")
KIND_LABEL = {
    "click": "点一下", "key": "按键", "wait": "等界面", "delegate": "交给引擎",
}

#: 起名上限，防止把一整句话当组名（页面会挤爆）
MAX_NAME = 24
MAX_STEPS = 24
#: 分差小于这个值就不猜：两个都像，让人选。
AMBIGUOUS_GAP = 0.12


# ===========================================================================
# 世界层
# ===========================================================================

@dataclass
class World:
    """这个游戏"是什么样"的唯一一份描述。组只能引用，不能覆盖。

    故意做成扁平的三张表 + 一个禁忌列表：组里写「点 K7 节点」时，要靠 scenes
    找到它对应哪个模板；写「清怪」时，要靠 keys 知道攻击键。少了任何一张，
    组就得自己带一份副本 —— 那正是"忘掉别的组"的来源。
    """

    terms: dict[str, str] = field(default_factory=dict)        # 别名 → 标准说法
    scenes: dict[str, dict] = field(default_factory=dict)      # 界面名 → {tpl, verdict, recognize}
    objects: dict[str, dict] = field(default_factory=dict)     # 物体名 → {tpl, color, ...}
    keys: dict[str, str] = field(default_factory=dict)         # 动作 → 键
    taboos: list[str] = field(default_factory=list)

    # ---------------- 查询 ----------------

    def canon(self, name: str) -> str:
        """别名 → 标准说法。认不出就原样返回（不留空，免得组里的字消失）。"""
        n = _norm(name)
        for alias, std in self.terms.items():
            if _norm(alias) == n:
                return std
        return name.strip()

    def scene(self, name: str) -> dict | None:
        """按名字找界面。先精确，再按标准说法，最后按"包含"（"战斗"匹配"战斗（异界深处）"）。"""
        n = _norm(self.canon(name))
        for k, v in self.scenes.items():
            if _norm(k) == n:
                return {**v, "name": k}
        for k, v in self.scenes.items():
            if n and (n in _norm(k) or _norm(k) in n):
                return {**v, "name": k}
        return None

    def template_for(self, name: str) -> str:
        """名字 → 模板文件名。界面/物体/术语都可能给出来，找不到返回空串。"""
        c = self.canon(name)
        for table in (self.scenes, self.objects):
            for k, v in table.items():
                if _norm(k) == _norm(c) or _norm(c) in _norm(k) or _norm(k) in _norm(c):
                    tpl = str(v.get("tpl") or "")
                    if tpl:
                        return tpl
        # 名字本身就是模板名（"rechallenge_btn"）的情况。只在带下划线/扩展名时认 ——
        # 否则「K7」这种关卡名会被当成模板文件名，然后"看起来没有缺口"，
        # 实际执行时匹配不到任何东西。
        if re.fullmatch(r"[A-Za-z0-9_\-]+", c or "") and ("_" in c or c.endswith(".png")):
            return c
        return ""

    def has_template(self, tpl: str, tpl_dir: Path | None) -> bool:
        if not tpl:
            return False
        if tpl_dir is None:
            return True          # 没给目录就不做文件级校验，别假装知道
        d = Path(tpl_dir)
        if not d.is_dir():
            return False
        return (d / tpl).exists() or (d / f"{tpl}.png").exists()

    def known_scenes(self) -> list[str]:
        return sorted(self.scenes)

    def to_dict(self) -> dict:
        return {"terms": self.terms, "scenes": self.scenes, "objects": self.objects,
                "keys": self.keys, "taboos": self.taboos}

    @classmethod
    def from_dict(cls, d: dict | None) -> "World":
        d = d or {}
        return cls(terms=dict(d.get("terms") or {}),
                   scenes=dict(d.get("scenes") or {}),
                   objects=dict(d.get("objects") or {}),
                   keys=dict(d.get("keys") or {}),
                   taboos=list(d.get("taboos") or []))


def build_world(claims: Iterable[Any]) -> World:
    """从手册声明编译世界层。纯函数，不碰磁盘 —— 冲突检测和写盘在外面。"""
    from .teach import KIND_KEY, KIND_OBJECT, KIND_SCENE, KIND_TABOO

    w = World()
    for c in claims:
        if c.kind == KIND_SCENE:
            w.scenes[c.title] = {
                "tpl": c.get("tpl_name", ""),
                "verdict": c.get("verdict", ""),
                "recognize": c.get("recognize", ""),
                "todo": c.get("todo", ""),
            }
        elif c.kind == KIND_OBJECT:
            w.objects[c.title] = {k: v for k, v in c.fields.items()
                                  if k in ("tpl_name", "color", "shape", "aspect",
                                           "rect", "count", "confusable")}
        elif c.kind == KIND_KEY:
            for action, key in c.fields.items():
                k = str(key).strip()
                if k and k != "待补充":
                    w.keys[str(action)] = k
        elif c.kind == KIND_TABOO:
            for line in c.body:
                s = line.strip().lstrip("-*+ ").strip()
                if s:
                    w.taboos.append(s)
    return w


def merge_world(old: World, new: World) -> tuple[World, list[dict]]:
    """合并世界层：只增不删，改要记账。

    为什么不是直接覆盖：用户补一段新描述时，很可能只写了自己关心的那几屏。
    如果按"这份手册就是全部"覆盖，没写到的界面会一起消失 —— 上一组就瞎了。
    所以旧条目默认保留，新条目补上，两者不一致的记成冲突。

    冲突分两种，危险程度差一个数量级：
      * 判定翻转（正例↔反例）：同一个界面一会儿算战斗一会儿算菜单，门禁会乱杀人
      * 其他字段变化：模板换了、说明改了，一般是有意的
    """
    out = World(terms=dict(old.terms), scenes={k: dict(v) for k, v in old.scenes.items()},
                objects={k: dict(v) for k, v in old.objects.items()},
                keys=dict(old.keys), taboos=list(old.taboos))
    conflicts: list[dict] = []

    for alias, std in new.terms.items():
        prev = next((v for k, v in out.terms.items() if _norm(k) == _norm(alias)), None)
        if prev is not None and _norm(prev) != _norm(std):
            conflicts.append({"kind": "term", "key": alias, "old": prev, "new": std,
                              "why": "同一个叫法指向了两个意思"})
        out.terms[alias] = std

    for name, meta in new.scenes.items():
        oldv = out.scenes.get(name)
        if oldv and _flip(oldv.get("verdict", ""), meta.get("verdict", "")):
            conflicts.append({"kind": "scene", "key": name,
                              "old": oldv.get("verdict"), "new": meta.get("verdict"),
                              "why": "同一个界面的判定被翻过来了，门禁会忽正忽反"})
        out.scenes[name] = {**oldv, **meta} if oldv else dict(meta)

    for name, meta in new.objects.items():
        out.objects[name] = {**(out.objects.get(name) or {}), **meta}

    for action, key in new.keys.items():
        prev = out.keys.get(action)
        if prev and _norm(prev) != _norm(key):
            conflicts.append({"kind": "key", "key": action, "old": prev, "new": key,
                              "why": "同一个动作用了两个键"})
        out.keys[action] = key

    for line in new.taboos:
        if line not in out.taboos:
            out.taboos.append(line)
    return out, conflicts


def _flip(a: str, b: str) -> bool:
    from .teach import _verdict_positive

    pa, pb = _verdict_positive(a or ""), _verdict_positive(b or "")
    return pa is not None and pb is not None and pa != pb


# ===========================================================================
# 组
# ===========================================================================

@dataclass
class Step:
    """一步。`raw` 永远保存用户原话 —— 诊断时说"卡在第 2 步 清怪，沿途捡掉落"，
    比说"卡在 delegate 步"有用一百倍。"""

    kind: str
    name: str = ""
    target: str = ""            # 点什么（原文或模板名）/ 按什么键 / 交给哪个循环
    until: str = ""             # 等到哪个界面算这一步完
    timeout_s: float = 20.0
    jump: int = 0               # 回到第 N 步（1 基）。0 = 不跳
    builtin: str = ""           # 引擎自带能力（rechallenge），不需要模板
    raw: str = ""
    note: str = ""              # 解析时的疑问，展示给人看

    def to_dict(self) -> dict:
        return {"kind": self.kind, "name": self.name, "target": self.target,
                "until": self.until, "timeout_s": self.timeout_s,
                "jump": self.jump, "builtin": self.builtin,
                "raw": self.raw, "note": self.note}

    @classmethod
    def from_dict(cls, d: dict) -> "Step":
        return cls(kind=str(d.get("kind") or "unknown"), name=str(d.get("name") or ""),
                   target=str(d.get("target") or ""), until=str(d.get("until") or ""),
                   timeout_s=float(d.get("timeout_s") or 20.0),
                   jump=int(d.get("jump") or 0), builtin=str(d.get("builtin") or ""),
                   raw=str(d.get("raw") or ""), note=str(d.get("note") or ""))


@dataclass
class Group:
    """一个有名字的任务。"""

    id: str
    name: str
    say: list[str] = field(default_factory=list)       # 用户会怎么喊
    start: str = ""                                     # 从哪一屏开始（世界层里的界面名）
    steps: list[Step] = field(default_factory=list)
    on_fail: str = "交回自由模式并报告卡在第几步"
    source: str = ""                                    # 来自哪份手册
    origin: str = "lesson"                              # lesson / page / manual

    # ---------------- 就绪检查 ----------------

    def gaps(self, world: World, tpl_dir: Path | None) -> list[str]:
        """跑不了或会跑错的地方。空列表 = 可以喊。

        这里只查"确定会失败"的项，不查"可能不准"。不确定的东西塞进 gaps 会让列表
        永远非空，人就不看了 —— 那等于什么都没查。
        """
        out: list[str] = []
        if self.start and not world.scene(self.start):
            out.append(f"起点「{self.start}」不在界面表里，不知道从哪一屏算开始")
        if not self.steps:
            out.append("一步都没有")
        for i, s in enumerate(self.steps, 1):
            if s.kind not in STEP_KINDS:
                out.append(f"第 {i} 步看不懂类型（{s.raw or s.name}）")
                continue
            if s.kind == "click":
                if s.builtin:
                    continue          # 引擎自带能力，不需要模板
                tpl = world.template_for(s.target)
                if not tpl:
                    out.append(f"第 {i} 步要点的「{s.target}」没有对应模板")
                elif not world.has_template(tpl, tpl_dir):
                    out.append(f"第 {i} 步要点的「{s.target}」缺模板 {tpl}")
        return out

    @property
    def ready_hint(self) -> str:
        return f"{len(self.steps)} 步"

    # ---------------- 序列化 ----------------

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "say": list(self.say),
                "start": self.start, "steps": [s.to_dict() for s in self.steps],
                "on_fail": self.on_fail, "source": self.source, "origin": self.origin}

    @classmethod
    def from_dict(cls, d: dict) -> "Group":
        return cls(id=str(d.get("id") or ""), name=str(d.get("name") or ""),
                   say=[str(x) for x in (d.get("say") or [])],
                   start=str(d.get("start") or ""),
                   steps=[Step.from_dict(x) for x in (d.get("steps") or [])],
                   on_fail=str(d.get("on_fail") or "交回自由模式并报告卡在第几步"),
                   source=str(d.get("source") or ""),
                   origin=str(d.get("origin") or "lesson"))

    def say_line(self) -> str:
        return " / ".join([self.name] + [s for s in self.say if s != self.name])


# ===========================================================================
# 从手册解析
# ===========================================================================

#: 交给引擎跑的动词。这些动作 FSM 早就会做，组不该重写一遍走位和拾取。
_DELEGATE_WORDS = ("清怪", "打怪", "刷怪", "捡", "拾取", "打一圈", "刷一圈", "farm",
                   "走位", "自由", "打boss", "清图", "打完", "打掉", "过图")
_CLICK_WORDS = ("点", "点击", "按一下", "选中", "打开")
_KEY_RE = re.compile(r"^按\s*[「\"']?([A-Za-z0-9]{1,8})[」\"']?")
_QUOTED = re.compile(r"[「\"'《]([^」\"'》]{1,24})[」\"'》]")
_JUMP_RE = re.compile(r"回到第\s*(\d+)\s*步|重复第\s*(\d+)\s*步|再从头")
_NUM_STEP = re.compile(r"^\s*(?:\d+[.、)]|[-*+]\s|\d+\s|第\s*\d+\s*步[：:]?)\s*")

#: 引擎里已经有的能力，不该逼用户再截一张模板。
#: 「再次挑战」是扫描出来的按钮（SceneState.rechallenge），不是模板匹配 ——
#: 要求用户为它裁图，等于把"机器已经会的事"重新压给人做。
_BUILTIN_CLICK = {
    "再次挑战": "rechallenge", "再挑战": "rechallenge", "再来一局": "rechallenge",
    "重新挑战": "rechallenge", "rechallenge": "rechallenge",
}
_BUILTIN_LABEL = {"rechallenge": "「再次挑战」按钮（视觉检出）"}


def parse_groups(claims: Iterable[Any], world: World) -> list[Group]:
    """手册里的「指令组」条目 → Group。解析出的疑问写在 step.note 里，不静默丢。"""
    from .teach import KIND_GROUP

    out: list[Group] = []
    for c in claims:
        if c.kind != KIND_GROUP:
            continue
        g = _group_from_claim(c, world)
        if g is not None:
            out.append(g)
    return out


def _group_from_claim(c: Any, world: World) -> Group | None:
    title = str(c.title or "").strip()
    if not title:
        return None
    say = _split_say(c.get("say", ""))
    name = say[0] if say else title
    gid = _slug(name or title)
    if not gid:
        return None
    steps = _steps_from(c, world)
    return Group(id=gid, name=name, say=[s for s in say if s != name],
                 start=c.get("start", "").strip(), steps=steps,
                 on_fail=c.get("on_fail", "").strip() or
                 "交回自由模式并报告卡在第几步",
                 source=str(getattr(c, "source", "") or ""), origin="lesson")


def _split_say(raw: str) -> list[str]:
    if not raw:
        return []
    parts = re.split(r"[、,，/|·；;]|\s{2,}", raw)
    return [p.strip() for p in parts if p.strip()]


def _steps_from(c: Any, world: World) -> list[Step]:
    """字段式的 `- 进图: 点 nether_node_k7` 优先；没有就用正文里的编号步骤。"""
    steps: list[Step] = []
    # 字段式：除组元信息外的 key 都当一步，key 就是步骤名
    reserved = {"say", "start", "on_fail", "until", "loop", "target", "note", "tpl_name"}
    for k, v in c.fields.items():
        if k in reserved:
            continue
        s = _step_from_text(f"{k}:{v}", world, default_name=k)
        s.name = s.name or k
        steps.append(s)
    if not steps:
        for line in c.body:
            raw = line.strip()
            if not raw:
                continue
            s = _step_from_text(raw, world)
            if s.kind == "unknown" and not s.raw:
                continue
            steps.append(s)
    return _tidy_steps(steps, world)


def _tidy_steps(steps: list[Step], world: World) -> list[Step]:
    """把人写的连续动作整理成可执行的步序列。

    用户写的是给人看的：
        2. 清怪，沿途捡掉落
        3. 打完 Boss 出结算
    机器要的是：
        2. 交给引擎跑，直到结算
    两步连着写不是两个任务，是同一件事的两个说法。不合并的话，第一步会跑满
    300 秒超时才进第二步 —— 而"清怪"本来就应该到出结算为止。
    """
    out: list[Step] = []
    for s in steps:
        if s.kind == "delegate" and out and out[-1].kind == "delegate":
            prev = out[-1]
            # 合并：名字留第一个（"清怪"比"打完 Boss 出结算"更像任务描述），
            # 完成条件取第一个写明的（越靠后越接近这件事的终点）
            if not prev.until:
                prev.until = s.until or _scene_mention(s.raw, world)
            prev.raw = f"{prev.raw}；{s.raw}" if prev.raw and s.raw else prev.raw or s.raw
            if not prev.name:
                prev.name = s.name
            continue
        out.append(s)

    # delegate 后面紧跟一个纯等待步 → 完成条件提到 delegate 上，删掉等待步
    merged: list[Step] = []
    i = 0
    while i < len(out):
        s = out[i]
        if (s.kind == "delegate" and not s.until and i + 1 < len(out)
                and out[i + 1].kind == "wait"):
            nxt = out[i + 1]
            s.until = nxt.until or _scene_mention(nxt.raw, world)
            if s.until:
                s.raw = f"{s.raw}；{nxt.raw}" if s.raw else nxt.raw
                merged.append(s)
                i += 2
                continue
        merged.append(s)
        i += 1
    return merged


def _scene_mention(text: str, world: World) -> str:
    """这段话里提到了哪个界面。提到多个就取最长（"战斗"和"战斗（Boss）"取后者）。"""
    if not text:
        return ""
    hit = ""
    for name in world.scenes:
        if name and name in text and len(name) > len(hit):
            hit = name
    return hit


def _step_from_text(line: str, world: World, default_name: str = "") -> Step:
    """一句话 → 一步。判不出类型就 unknown，并把原话留着（人在页面上一眼能改）。"""
    raw = line.strip()
    text = _NUM_STEP.sub("", raw).strip()
    s = Step(kind="unknown", name=default_name, raw=raw, timeout_s=20.0)
    # 关键词匹配去空格：「打完 Boss」要能命中「打完」
    flat = re.sub(r"\s+", "", text).lower()

    jump = _JUMP_RE.search(text)
    if jump:
        s.jump = int(next(g for g in jump.groups() if g) or 0) if any(jump.groups()) else 1
    # 「回到第 2 步」这句话本身不是动作，只是循环条件，摘掉再判类型
    text = _JUMP_RE.sub("", text).strip()

    # 「直到/等到 X」：先把完成条件摘出来，剩下的才是动作
    until = ""
    m = re.search(r"(?:直到|等到|等它|一直(?:到|刷到))\s*([^，,。;；]{1,20})", text)
    if m:
        until = m.group(1).strip()
    # 段落里提到过的界面名也算完成条件（"打完 Boss 出结算" → 结算）
    s.until = until or _scene_mention(text, world)

    # 按键：按 A / 按 Esc
    mk = _KEY_RE.search(text)
    if mk:
        s.kind = "key"
        s.target = mk.group(1).upper()
        if not s.name:
            s.name = f"按 {s.target}"
        return s

    # 点某个东西：优先引号里的内容，其次是「点 XX 节点/按钮」
    if any(w in flat for w in _CLICK_WORDS):
        target = _click_target(text)
        if target:
            s.kind = "click"
            s.target = target
            builtin = (_BUILTIN_CLICK.get(_norm(target))
                       or _BUILTIN_CLICK.get(_norm(world.canon(target))))
            if builtin:
                # 引擎自己会找这个按钮，不该逼用户再裁一张模板
                s.builtin = builtin
                s.target = _BUILTIN_LABEL.get(builtin, builtin)
                s.note = ""
            elif world.template_for(target):
                s.note = ""
            else:
                s.note = f"认不出「{target}」是哪个模板，确认一下"
            if not s.name:
                s.name = f"点{target}"
            return s

    # 交给引擎：有"清怪/捡/刷/打完"这类动词，且没有明确点击对象
    if any(w in flat for w in _DELEGATE_WORDS):
        s.kind = "delegate"
        s.target = "farm"
        s.timeout_s = 300.0
        if not s.name:
            s.name = "交给引擎"
        return s

    # 纯等待
    if until or re.search(r"^(等|wait)", text):
        s.kind = "wait"
        s.timeout_s = 30.0
        if not s.name:
            s.name = f"等到{until}" if until else "等一会"
        return s

    if not s.name:
        s.name = text[:MAX_NAME]
    s.note = "没看懂这一步要做什么，请改成「点 X」「按 A」「清怪」中的一种"
    return s


def _click_target(text: str) -> str:
    """从"点「再次挑战」"里取出"再次挑战"；没引号就取「点」后面那一小段。"""
    quoted = _QUOTED.findall(text)
    if quoted:
        t = quoted[-1].strip()
    else:
        m = re.search(r"(?:点|点击|选中|打开)\s*([^，,。;；：:]{1,16})", text)
        if not m:
            return ""
        t = m.group(1).strip()
        # 去掉方位/目的尾巴："目标节点，进图" / "K7节点进图"
        t = re.split(r"[，,。;；]|进图|进入|打开|后", t)[0].strip()
    # "K7节点" 和 "K7 节点" 是同一个东西；只剥后缀，不剥内容
    t2 = re.sub(r"^(?:那个|这个)", "", t).strip()
    t2 = re.sub(r"(节点|按钮|图标)$", "", t2).strip()
    return (t2 or t)[:MAX_NAME]




# ===========================================================================
# 存盘：一组一文件
# ===========================================================================

def groups_root(profile: Any) -> Path:
    return Path(profile.root) / "groups"


def _slug(text: str) -> str:
    """组 id。中文保留 —— 文件名就是人认得的名字，比 k7_001 好用。"""
    s = unicodedata.normalize("NFKC", str(text or "")).strip()
    s = re.sub(r"[\\/:*?\"<>|\s]+", "_", s)
    return s.strip("._")[:32]


def load_groups(profile: Any) -> list[Group]:
    root = groups_root(profile)
    if not root.is_dir():
        return []
    out: list[Group] = []
    for p in sorted(root.glob("*.yaml")):
        if p.name.startswith("_"):
            continue
        try:
            d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            g = Group.from_dict(d)
            if g.name:
                out.append(g)
        except Exception:  # noqa: BLE001
            log.warning("指令组文件解析失败，已跳过: %s", p, exc_info=True)
    return out


def save_group(profile: Any, g: Group) -> Path:
    root = groups_root(profile)
    root.mkdir(parents=True, exist_ok=True)
    p = root / f"{g.id}.yaml"
    body = yaml.safe_dump(g.to_dict(), allow_unicode=True, sort_keys=False,
                          default_flow_style=False)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(p)
    return p


def load_world(profile: Any) -> World:
    p = groups_root(profile) / "_world.json"
    if not p.exists():
        return World()
    try:
        return World.from_dict(json.loads(p.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        log.warning("世界层文件坏了，按空的算: %s", p, exc_info=True)
        return World()


def save_world(profile: Any, w: World) -> Path:
    root = groups_root(profile)
    root.mkdir(parents=True, exist_ok=True)
    p = root / "_world.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(w.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)
    return p


def sync_from_lessons(profile: Any) -> dict:
    """手册 → 世界层 + 组。这是"学完就能喊"的入口。

    三条规矩，都是为了让"学到的东西不会互相挤掉"：
      1. 世界层合并而不是覆盖（见 merge_world）
      2. 组按 id 覆盖自己那一份，别的组文件一个都不动
      3. 手册里消失的组**不删**，记进 _orphans.json 等人决定 ——
         手册改了一句话就让一个能跑的任务凭空消失，是最让人不信任的行为
    """
    from .teach import _verdict_positive, load_lessons  # noqa: F401

    root = Path(profile.root)
    lessons_dir = root / "lessons"
    claims: list[Any] = []
    lessons: list[Any] = []
    if lessons_dir.is_dir():
        lessons = load_lessons(lessons_dir)
        for ls in lessons:
            claims.extend(ls.claims)

    fresh = build_world(claims)
    # 术语在 Lesson 上（表格解析的结果），不在 Claim 里 —— 不补这一句，
    # 世界层的 terms 永远是 0，而路由恰恰要靠它把"搬砖"换算成"刷图"。
    for ls in lessons:
        for alias, std, _note in getattr(ls, "terms", []) or []:
            if alias and std:
                fresh.terms[alias] = std
    world, conflicts = merge_world(load_world(profile), fresh)
    save_world(profile, world)

    parsed = parse_groups(claims, world)
    existing = {g.id: g for g in load_groups(profile)}
    written: list[str] = []
    for g in parsed:
        save_group(profile, g)
        written.append(g.id)

    orphans = sorted(set(existing) - set(written))
    if orphans:
        _write_json(groups_root(profile) / "_orphans.json",
                    {"orphans": orphans,
                     "why": "手册里已经没有这些组了。文件保留着 —— 删掉是我做的事，"
                            "该由你点头。在页面上确认后再删。"})
    else:
        (groups_root(profile) / "_orphans.json").unlink(missing_ok=True)

    _write_json(groups_root(profile) / "_conflicts.json",
                {"conflicts": conflicts} if conflicts else {})
    if conflicts:
        log.warning("世界层有 %d 处冲突，需要人裁决（groups/_conflicts.json）", len(conflicts))

    return {"world": {"scenes": len(world.scenes), "objects": len(world.objects),
                      "keys": len(world.keys), "terms": len(world.terms)},
            "groups": written, "orphans": orphans, "conflicts": conflicts}


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def forget_group(profile: Any, gid: str) -> bool:
    """显式删掉一组。只由人触发 —— 学习过程永远不调它。"""
    p = groups_root(profile) / f"{gid}.yaml"
    if not p.exists():
        return False
    p.unlink()
    return True


# ===========================================================================
# 命令路由
# ===========================================================================

@dataclass
class Route:
    group: Group | None = None
    score: float = 0.0
    why: str = ""
    candidates: list[tuple[Group, float]] = field(default_factory=list)
    text: str = ""

    @property
    def ok(self) -> bool:
        return self.group is not None

    @property
    def ambiguous(self) -> bool:
        return self.group is None and len(self.candidates) >= 2

    def to_dict(self) -> dict:
        return {"ok": self.ok, "group": self.group.name if self.group else "",
                "id": self.group.id if self.group else "",
                "score": round(self.score, 3), "why": self.why,
                "ambiguous": self.ambiguous, "text": self.text,
                "candidates": [{"id": g.id, "name": g.name, "score": round(s, 3)}
                               for g, s in self.candidates[:4]]}


def _norm(text: str) -> str:
    """全角转半角、去空格、去标点、小写。用户不会照着别名一字不差地打字。"""
    s = unicodedata.normalize("NFKC", str(text or "")).lower()
    s = re.sub(r"[\s\u3000]+", "", s)
    s = re.sub(r"[，,。.！!？?、~～\-—_·:：;；\"'“”‘’()（）\[\]【】]", "", s)
    return s


def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def route(text: str, groups: list[Group], world: World | None = None) -> Route:
    """一句话 → 哪一组。分数相同就不猜，把候选摆出来。

    四路匹配，按可信度递减：
      1. 完全相同（组名或别名）
      2. 包含（"帮我刷一下k7" 里有 "k7"）—— 别名越长越可信
      3. 术语换算后再比（用户说"搬砖"，标准说法是"刷图"）
      4. 打错一个字（"刷k77"）—— 只在长度 ≥3 时启用，短名字容错会乱指
    """
    t = _norm(text)
    r = Route(text=text)
    if not t or not groups:
        r.why = "没有可用的指令组"
        return r

    scored: list[tuple[Group, float, str]] = []
    for g in groups:
        names = [g.name, g.id] + list(g.say)
        best = 0.0
        why = ""
        for nm in names:
            n = _norm(nm)
            if not n:
                continue
            if n == t:
                best, why = 1.0, f"和「{nm}」一字不差"
                break
            if len(n) >= 2 and n in t:
                s = 0.55 + 0.45 * (len(n) / max(1, len(t)))
                if s > best:
                    best, why = s, f"你的话里含「{nm}」"
            if len(t) >= 3 and len(n) >= 3 and _lev(n, t) <= 1:
                if 0.72 > best:
                    best, why = 0.72, f"「{t}」和「{nm}」只差一个字"
            if world is not None:
                c = _norm(world.canon(t))
                if c and c != t and (c == n or (len(n) >= 2 and n in c)):
                    s = 0.9 if c == n else 0.6
                    if s > best:
                        best, why = s, f"按术语表换算成「{c}」后对上「{nm}」"
        if best > 0:
            scored.append((g, best, why))

    if not scored:
        r.why = "没认出是哪一组"
        return r
    scored.sort(key=lambda x: -x[1])
    r.candidates = [(g, s) for g, s, _ in scored]
    top_g, top_s, top_why = scored[0]
    if len(scored) > 1 and top_s - scored[1][1] < AMBIGUOUS_GAP:
        r.why = (f"「{top_g.name}」和「{scored[1][0].name}」都像，分不清。"
                 "说准确一点，或者直接选")
        return r
    r.group, r.score, r.why = top_g, top_s, top_why
    return r


# ===========================================================================
# 执行
# ===========================================================================

class GroupRunner:
    """跑一组。和 RecipeRunner 同一个形状（tick/done/failed/status），
    hub 那段代码不用改 —— 执行器只认这四个东西。

    和 RecipeRunner 的区别：RecipeRunner 只管一屏 UI 的点击序列；组可以跨越
    "菜单 → 战斗 → 结算"多个界面，中间的"打一圈"不自己实现，交给 FSM。
    """

    #: hub 看到这个标记就改调 tick_frame（组需要整帧场景，不只是图像）
    wants_scene = True

    def __init__(self, group: Group, world: World, templates_dir: Path | None,
                 executor: Any = None, fsm: Any = None) -> None:
        self.group = group
        self.world = world
        self.templates_dir = Path(templates_dir) if templates_dir else None
        self.executor = executor
        self.fsm = fsm
        self.step_i = 0
        self.done = False
        self.failed = ""
        # 用 None 表示"还没开始"，不能用 0 —— 第一帧的 ts_ms 就是 0，
        # 拿它当哨兵的话，起跑计时永远停在 0，超时永远判不出来。
        self.started_ms: int | None = None
        self.step_started_ms: int | None = None
        self.log: list[dict] = []
        self._tpl_cache: dict[str, Any] = {}
        self._clicked = False
        self._last_frame = None
        self.last_scene = ""

    # ---------------- 模板 ----------------

    def _template(self, name: str):  # noqa: ANN202
        import cv2

        tpl = self.world.template_for(name)
        if not tpl:
            return None, ""
        if tpl in self._tpl_cache:
            return self._tpl_cache[tpl], tpl
        img = None
        if self.templates_dir is not None:
            for cand in (self.templates_dir / tpl, self.templates_dir / f"{tpl}.png"):
                if cand.exists():
                    img = cv2.imread(str(cand), cv2.IMREAD_COLOR)
                    break
        self._tpl_cache[tpl] = img
        return img, tpl

    # ---------------- 推进 ----------------

    @property
    def steps(self) -> list[Step]:
        return self.group.steps

    @property
    def name(self) -> str:
        return self.group.name

    def tick(self, frame, ts_ms: int) -> list[str]:  # noqa: ANN001
        """没有场景时的退化入口（只有图）。点击能用，delegate 用不了。"""
        return self._tick(frame, ts_ms, scene=None)

    def tick_frame(self, scene: Any, frame) -> list[str]:  # noqa: ANN001
        return self._tick(frame, int(getattr(scene, "ts_ms", 0) or 0), scene=scene)

    def _tick(self, frame, ts_ms: int, scene: Any) -> list[str]:  # noqa: ANN001
        if self.done:
            return []
        if self.started_ms is None:
            self.started_ms = ts_ms
            self.log.append({"t": ts_ms, "event": "start", "group": self.group.id})
        if self.step_i >= len(self.steps):
            self._finish(ts_ms, "")
            return []

        s = self.steps[self.step_i]
        if self.step_started_ms is None:
            self.step_started_ms = ts_ms
            self.log.append({"t": ts_ms, "event": "step_start", "i": self.step_i,
                             "kind": s.kind, "raw": s.raw})
        elapsed = (ts_ms - self.step_started_ms) / 1000.0
        if elapsed > s.timeout_s:
            # 超时不是"这一步失败"，是"整组停下来"。继续跑下一步会越跑越错。
            self._finish(ts_ms, f"第 {self.step_i + 1} 步「{s.name or s.raw}」"
                                f"等了 {s.timeout_s:.0f}s 没过去")
            return []

        cmds: list[str] = []
        try:
            if s.kind == "click":
                cmds = self._do_click(s, frame, ts_ms, scene)
            elif s.kind == "key":
                cmds = self._do_key(s, ts_ms)
            elif s.kind == "wait":
                cmds = self._do_wait(s, scene, elapsed, ts_ms)
            elif s.kind == "delegate":
                cmds = self._do_delegate(s, scene, ts_ms)
            else:
                self._finish(ts_ms, f"第 {self.step_i + 1} 步看不懂：{s.raw or s.name}")
        except Exception as e:  # noqa: BLE001
            self.log.append({"t": ts_ms, "event": "error", "reason": str(e)})
            self._finish(ts_ms, f"第 {self.step_i + 1} 步出错：{e}")
        return cmds

    # ---------------- 各步骤 ----------------

    def _do_click(self, s: Step, frame, ts_ms: int, scene: Any = None) -> list[str]:  # noqa: ANN001
        from .vision.ui_panels import match_button

        if not self._clicked:
            x = y = -1
            if s.builtin == "rechallenge":
                # 引擎每帧都在扫这个按钮（SceneState.rechallenge），直接拿来用
                hit = getattr(scene, "rechallenge", None) if scene is not None else None
                if hit is None:
                    return []      # 这一帧还没出现，下一帧再看
                x, y = int(hit.x), int(hit.y)
            else:
                tpl_img, tpl_name = self._template(s.target)
                if tpl_img is None:
                    self._finish(ts_ms, f"第 {self.step_i + 1} 步要点的「{s.target}」没有模板"
                                        f"（{tpl_name or '认不出名字'}），跑不了")
                    return []
                hit = match_button(frame, tpl_img, None, threshold=0.72)
                if hit is None:
                    return []      # 这一帧没看到，下一帧再看，超时兜底
                x, y = int(hit.x), int(hit.y)
            self._exec({"op": "mouse_click", "button": "left", "x": x, "y": y})
            self._clicked = True
            self.log.append({"t": ts_ms, "event": "click", "target": s.target,
                             "builtin": s.builtin, "at": [x, y]})
            return [f"MOUSE_CLICK left ({x},{y})"]
        # 点完了：等它生效。写了「进图」这类完成条件就等界面，但最多等 5 秒 ——
        # 界面识别不准的时候（反例模板没裁好），不该让整组卡死在这里。
        elapsed = ts_ms - (self.step_started_ms or 0)
        if elapsed >= 600:
            if s.until and elapsed < 5000 and not self._scene_is(s.until, scene):
                return []
            self._advance(ts_ms)
        return []

    def _do_key(self, s: Step, ts_ms: int) -> list[str]:
        key = s.target or "a"
        self._exec({"op": "key_tap", "key": key, "hold_ms": 60})
        self.log.append({"t": ts_ms, "event": "key", "key": key})
        self._advance(ts_ms)
        return [f"KEY_TAP {key} hold=60ms"]

    def _do_wait(self, s: Step, scene: Any, elapsed: float, ts_ms: int) -> list[str]:
        if self._scene_is(s.until, scene) or (not s.until and elapsed >= 1.0):
            self._advance(ts_ms)
        return []

    def _do_delegate(self, s: Step, scene: Any, ts_ms: int) -> list[str]:
        """把这一步交给 FSM 自由跑，直到 until 指定的界面出现。

        为什么不自己实现"打一圈"：走位、脱困、拾取判定、地图黑名单这些东西
        FSM 里已经调了几十轮，重写一遍只会更差。组负责"什么时候开始、什么时候
        算完"，引擎负责"怎么打"—— 这也是"本质是一样的"在代码里的样子：
        所有组共用同一个执行体。
        """
        if scene is None:
            self.log.append({"t": ts_ms, "event": "delegate_no_scene",
                             "why": "没有场景输入，跳过"})
            self._advance(ts_ms)
            return []
        # 引擎停着的时候"交给引擎"等于空转：组会一直停在这一步直到超时，
        # 然后报一句"第 1 步等了 300s 没过去"，而真正的原因是 CPU 根本没在动。
        # 停车要立刻说，不要等到超时。
        state = str(getattr(self.fsm, "state", "") or "") if self.fsm is not None else ""
        if state.endswith("PANIC"):
            self._finish(ts_ms, "引擎处于急停状态，这一组无法继续（先解除急停）")
            return []
        if state.endswith("PAUSED"):
            self._finish(ts_ms, "引擎是暂停状态，交给引擎的步骤无法执行（先启动引擎）")
            return []
        if s.until and self._scene_is(s.until, scene):
            self._advance(ts_ms)
            return []
        if self.fsm is not None:
            self.fsm.on_scene(scene)
        return []

    def _scene_is(self, name: str, scene: Any) -> bool:
        """界面判定：只认硬证据。

        最初写成"反例界面 = 没有战斗 HUD"，实测立刻出错：星图、背包、加载过场
        全是没有 HUD 的，于是在星图上喊「刷一局」会当成已经结算过，直接跳到
        「点再次挑战」—— 一步都不打就空转。

        所以顺序是：
          1. 门禁说命中了这个界面的模板 → 就是它（最硬）
          2. 结算页的独有特征：「再次挑战」按钮只有结算页才有
          3. 都没有 → 不判。宁可等超时让人看见"卡在哪一步"，也不瞎猜。
        """
        if not name or scene is None:
            return False
        meta = self.world.scene(name)
        if meta is None:
            return False
        from .teach import _verdict_positive

        gate = str(getattr(scene, "gate_reason", "") or "")
        in_game = bool(getattr(scene, "in_game", False))
        tpl = str(meta.get("tpl") or "")
        want = _verdict_positive(str(meta.get("verdict") or ""))

        if tpl and tpl in gate:
            return True if want is False else in_game
        if want is True:
            return in_game
        if want is False:
            # 结算页：没有战斗 HUD，同时有「再次挑战」按钮。两个条件都满足才认。
            if not in_game and getattr(scene, "rechallenge", None) is not None:
                return True
            return False
        return False

    # ---------------- 收尾 ----------------

    def _advance(self, ts_ms: int) -> None:
        s = self.steps[self.step_i] if self.step_i < len(self.steps) else None
        self.log.append({"t": ts_ms, "event": "step_done", "i": self.step_i,
                         "kind": s.kind if s else "", "raw": s.raw if s else ""})
        self._clicked = False
        self.step_started_ms = None
        nxt = self.step_i + 1
        if s is not None and s.jump:
            # 「回到第 2 步」= 整组循环。刷图类任务几乎都是这种，所以支持它。
            nxt = max(0, s.jump - 1)
            self.log.append({"t": ts_ms, "event": "jump", "to": s.jump})
        self.step_i = nxt
        if self.step_i >= len(self.steps):
            self._finish(ts_ms, "")

    def _finish(self, ts_ms: int, reason: str) -> None:
        self.done = True
        self.failed = reason
        self.log.append({"t": ts_ms, "event": "fail" if reason else "done",
                         "reason": reason})

    def _exec(self, cmd: dict) -> None:
        if self.executor is None:
            return
        from .protocol import Command

        try:
            self.executor.execute(Command(**cmd))
        except Exception:  # noqa: BLE001
            log.warning("组执行器下发命令失败", exc_info=True)

    # ---------------- 状态 ----------------

    def status(self) -> dict:
        i = min(self.step_i, len(self.steps) - 1)
        cur = self.steps[i] if self.steps else None
        return {"name": self.group.name, "id": self.group.id,
                "step": self.step_i, "total_steps": len(self.steps),
                "current": (cur.raw or cur.name) if cur else "",
                "done": self.done, "failed": self.failed,
                "log": self.log[-12:]}
