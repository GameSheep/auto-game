"""教学编译器：把"人写的图文教程"编译成程序能执行的资产。

用户的原话就是需求：

    "每个游戏都要有这些内容，我希望有一个过程让程序吃掉并完全理解用户写的东西。"

"完全理解"这个词必须翻译成可验证的东西，否则它就是一句空话——模型读完点头说
"我懂了"，第二天照样把星图节点当光柱。所以这里定义的理解是：

    你写的每一条，要么变成程序能判的判据（模板/阈值/键位），
    要么变成程序会读的文字（术语/流程/禁忌，进模型上下文），
    要么变成一条明确的"我来问你要"的待办（缺图/缺框/阈值不符）。

三种归宿，没有第四种"大概懂了"。

为什么不让模型直接读教程就完事
------------------------------
模型读一遍能答"这个界面该按什么"，但答不出"这一帧现在该按什么"——后者需要
像素证据。教程的作用是告诉我们**去哪里找证据、找到的算不算**，不是替我们判断。
所以编译器只做机械翻译，判断留给分数。

输入（每个游戏一套，可以多份，按文件名排序依次吃）
--------------------------------------------------
    profiles/<id>/lessons/
        tli.md           图文教程（格式见 profiles/_template/lessons/example.md）
        img/*.png|jpg    教程里引用的图；也可以直接引用 ../refs/ 下的图

输出（全部可追溯，全部可重跑）
------------------------------
    knowledge/game.md        <!-- teach:begin --> 块被重写，块外你写的一个字不动
    knowledge/glossary.json  黑话/简称对照表（模型和报表都用它）
    labels.json              界面类条目自动进台账（只有核对通过的才 strict）
    teach/report.md          本轮吃了什么、卡在哪、下一步裁哪块
    teach/pending.json       需要你裁决的清单（缺图/缺框/键位冲突/阈值不符）

三条硬约束
----------
1. **不覆盖人写的东西**：game.md 只动 teach 标记块；labels.json 只追加不存在的条目；
   已有的 note 永不改写（那是人的话，比生成的更值钱）。
2. **核对不过就不许 strict**：你说"这是战斗页"、程序认不出来，条目仍然入库，
   但 strict=false + 备注写清原因。让缺口可见，不让 CI 变红也变瞎。
3. **量不到就说量不到**：没给框就不编高宽比，没给图就不编颜色。宁可 pending
   多一条，也不要一个看起来合理的阈值。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

# ---------------- 类型与别名 ----------------

KIND_SCENE = "scene"
KIND_OBJECT = "object"
KIND_FLOW = "flow"
KIND_KEY = "key"
KIND_TABOO = "taboo"
KIND_TERM = "term"
KIND_UNKNOWN = "unknown"

KIND_LABEL = {
    KIND_SCENE: "界面", KIND_OBJECT: "物体", KIND_FLOW: "流程",
    KIND_KEY: "键位", KIND_TABOO: "禁忌", KIND_TERM: "术语",
    KIND_UNKNOWN: "没看懂",
}

_SECTION_ALIASES = {
    "界面": KIND_SCENE, "场景": KIND_SCENE, "画面": KIND_SCENE, "screen": KIND_SCENE,
    "物体": KIND_OBJECT, "掉落物": KIND_OBJECT, "掉落的": KIND_OBJECT,
    "object": KIND_OBJECT, "item": KIND_OBJECT,
    "流程": KIND_FLOW, "玩法": KIND_FLOW, "循环": KIND_FLOW, "步骤": KIND_FLOW,
    "flow": KIND_FLOW,
    "键位": KIND_KEY, "按键": KIND_KEY, "操作": KIND_KEY, "key": KIND_KEY, "keys": KIND_KEY,
    "禁忌": KIND_TABOO, "别做": KIND_TABOO, "注意事项": KIND_TABOO, "taboo": KIND_TABOO,
    "术语": KIND_TERM, "黑话": KIND_TERM, "简称": KIND_TERM, "叫法": KIND_TERM,
    "term": KIND_TERM,
}

# 字段别名：用户怎么写都认（中英文、口语化）
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "file": ("图", "图片", "截图", "证据图", "file", "image", "src"),
    "verdict": ("判定", "是", "类型", "verdict", "positive"),
    "recognize": ("一眼认出", "认出", "特征", "判据", "怎么认", "looks"),
    "todo": ("该做什么", "处理", "做什么", "怎么处理", "do"),
    "rect": ("框", "框选", "区域", "范围", "rect", "roi"),
    "color": ("颜色", "color", "色"),
    "shape": ("形状", "shape"),
    "aspect": ("高宽比", "aspect"),
    "count": ("数量", "几束", "count"),
    "confusable": ("不是", "别当成", "易混淆", "别认成", "confusable"),
    "note": ("备注", "note", "说明"),
    "pos": ("位置", "在哪", "pos", "在", "方位", "方向", "画面位置"),
    "tpl_name": ("模板名", "文件名", "name", "slug", "tpl"),
}

# 键位动作 → (decision 下的点路径, 别名)。路径为空 = 配置里没这个概念，单独报。
# 注意 pickup_key 挂在 DecisionConfig，escape/dodge 挂在 DecisionConfig.autonomy ——
# 写错归属会让核对永远报"配置里没写"，看起来像手册写错了，其实是查错了地方。
_KEY_ACTIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "pickup": ("pickup_key", ("拾取", "捡", "捡东西", "拾起", "pickup")),
    "escape": ("autonomy.escape_key",
               ("返回", "关闭", "退出", "关掉", "关弹窗", "escape", "esc")),
    "dodge": ("autonomy.dodge_key", ("闪避", "翻滚", "位移", "脱困", "dodge")),
    "attack": ("", ("攻击", "普攻", "打", "attack")),
    "map": ("", ("地图", "打开地图", "map")),
}

_POSITIVE_WORDS = ("正例", "正面", "战斗", "是战斗", "in_game", "游戏中", "对")
_NEGATIVE_WORDS = ("反例", "负例", "菜单", "不是战斗", "非战斗", "negative", "界面", "不让点")

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

HEAD_BEGIN = "<!-- teach:begin 由 tools/teach.py 从 lessons/ 生成；块内会被重写 -->"
HEAD_END = "<!-- teach:end -->"


# ---------------- 数据结构 ----------------

@dataclass
class Claim:
    """一条声明：一段标题 + 若干字段。教程的最小单位。"""

    kind: str
    title: str
    line: int
    fields: dict[str, str] = field(default_factory=dict)
    body: list[str] = field(default_factory=list)   # 非 key:value 的正文/步骤
    source: str = ""                                 # 哪份文件

    @property
    def file(self) -> str:
        return self.fields.get("file", "").strip()

    @property
    def rect(self) -> list[int] | None:
        return parse_rect(self.fields.get("rect", ""))

    def get(self, key: str, default: str = "") -> str:
        return self.fields.get(key, default)


@dataclass
class Lesson:
    path: Path
    title: str = ""
    claims: list[Claim] = field(default_factory=list)
    terms: list[tuple[str, str, str]] = field(default_factory=list)  # (别名, 标准, 备注)
    warnings: list[str] = field(default_factory=list)


@dataclass
class TeachResult:
    lessons: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    glossary: dict[str, dict] = field(default_factory=dict)
    md_block: str = ""
    labels_new: list[dict] = field(default_factory=list)
    labels_seen: list[str] = field(default_factory=list)
    pending: list[dict] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def failures(self) -> list[dict]:
        return [c for c in self.checks if c["status"] == "fail"]

    @property
    def gaps(self) -> list[dict]:
        return [c for c in self.checks if c["status"] in ("gap", "skip")]


# ---------------- 解析 ----------------

def parse_rect(raw: str) -> list[int] | None:
    """`0,0,300,80` / `(0 0 300 80)` / `[0,0,300,80]` 都认。"""
    if not raw:
        return None
    nums = re.findall(r"-?\d+", raw)
    if len(nums) < 4:
        return None
    return [int(n) for n in nums[:4]]


def _norm_field_key(raw: str) -> str:
    k = raw.strip().lower()
    for key, aliases in _FIELD_ALIASES.items():
        if k in aliases:
            return key
    return ""


def _split_heading(text: str) -> tuple[str, str]:
    """`界面 · 异界星图` → ('scene', '异界星图')；认不出类型就返回 ('', 原文)。"""
    for sep in ("·", "｜", "|", "—", "--", "-", "：", ":", " "):
        if sep in text:
            head, rest = text.split(sep, 1)
            kind = _SECTION_ALIASES.get(head.strip().lower()) or \
                _SECTION_ALIASES.get(head.strip())
            if kind and rest.strip():
                return kind, rest.strip()
    for word, kind in _SECTION_ALIASES.items():
        if text.strip().startswith(word):
            rest = text.strip()[len(word):].lstrip(" ·｜|-—:：")
            # `## 键位` 这种只有类型没有标题的也认——不能因为少了后半截就整段丢掉
            return kind, rest or text.strip()
    return "", text.strip()


def _infer_kind(fields: dict[str, str], body: list[str]) -> str:
    """标题没写类型时，按字段反推。推不出来就认 unknown，不瞎猜。"""
    if "file" in fields and "verdict" in fields:
        return KIND_SCENE
    if "file" in fields and ({"color", "shape", "aspect", "rect"} & set(fields)):
        return KIND_OBJECT
    # 整段都是 `- 动作: 键` 且没有图 → 键位表
    pairs = [re.match(r"^\s*[-*+]\s*([^:：]{1,12})\s*[:：]\s*(.+)$", ln) for ln in body]
    pairs = [m for m in pairs if m]
    if pairs and len(pairs) == len(body) and not any(
            m.group(1).strip() in ("图", "图片", "截图") for m in pairs):
        return KIND_KEY
    if any(re.match(r"^\s*(\d+[.、)]|[-*])\s*\S", ln) for ln in body):
        return KIND_FLOW
    return KIND_UNKNOWN


def _parse_terms(lines: list[str], lesson: Lesson) -> None:
    """术语表：markdown 表格，`| 我说的 | 标准说法 | 备注 |`。"""
    for ln in lines:
        s = ln.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 2:
            continue
        a, b = cells[0], cells[1]
        if not a or not b:
            continue
        if set(a) <= set("-: ") or a in ("我说的", "别名", "黑话", "原话"):
            continue
        note = cells[2] if len(cells) > 2 else ""
        lesson.terms.append((a, b, note))


def parse_lessons(path: Path) -> Lesson:
    """解析一份教学手册。格式错的地方给 warning+行号，不静默丢。"""
    lesson = Lesson(path=path)
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        lesson.warnings.append(f"{path.name}: 读不出来（{exc}）")
        return lesson

    # 先剔掉 HTML 注释，但**保留行号**（用等量换行占位）：
    # 注释里几乎必然出现「## 界面」这类示例文本，而且常是多行；不剔掉会被当成
    # 真的段落解析出一堆幽灵 claim，或者混进流程的正文里。
    text = re.sub(r"<!--.*?-->",
                  lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)

    lines = text.splitlines()
    # 一级标题当手册名
    for ln in lines:
        if ln.startswith("# "):
            lesson.title = ln[2:].strip()
            break

    cur: Claim | None = None
    section_lines: list[str] = []

    def _flush() -> None:
        nonlocal cur, section_lines
        if cur is not None:
            if cur.kind == KIND_TERM:
                # 术语走表格，不走 claim —— 它不是"某张图的一条判据"
                _parse_terms(section_lines, lesson)
            elif cur.kind != KIND_UNKNOWN:
                lesson.claims.append(cur)
        cur = None
        section_lines = []

    for i, ln in enumerate(lines, 1):
        s = ln.rstrip()
        if s.startswith("## "):
            _flush()
            kind, title = _split_heading(s[3:].strip())
            cur = Claim(kind=kind, title=title or s[3:].strip(), line=i,
                        source=str(path.name))
            continue
        if cur is None:
            continue
        section_lines.append(s)
        body = s.strip()
        if not body or body.startswith("<!--") or body.startswith("#"):
            continue
        m = re.match(r"^\s*[-*+]\s*([^:：]{1,12})\s*[:：]\s*(.+)$", body)
        if m:
            key = _norm_field_key(m.group(1))
            if key:
                cur.fields[key] = m.group(2).strip()
                continue
            # 键位段里字段名就是动作名（拾取/返回/闪避…），不能按通用别名过滤，
            # 否则 `- 拾取: A` 会掉进正文，整段键位表就白写了。
            if cur.kind == KIND_KEY:
                cur.fields[m.group(1).strip()] = m.group(2).strip()
                continue
        cur.body.append(body)

    _flush()

    for c in lesson.claims:
        if not c.kind:
            c.kind = _infer_kind(c.fields, c.body)
        elif c.kind == KIND_UNKNOWN:
            c.kind = _infer_kind(c.fields, c.body)
    # 二次过滤：推断后仍是 unknown 的，只留 warning
    keep = [c for c in lesson.claims if c.kind != KIND_UNKNOWN]
    dropped = [c for c in lesson.claims if c.kind == KIND_UNKNOWN]
    for c in dropped:
        lesson.warnings.append(
            f"{path.name}:{c.line} 这一段没看懂类型（『{c.title}』），已跳过。"
            f" 标题里带上「界面/物体/流程/键位/禁忌/术语」之一即可")
    lesson.claims = keep

    # 类型已经定下来的，把 term 段里的表格清掉（它不是 claim 的字段）
    return lesson


def load_lessons(lesson_dir: Path) -> list[Lesson]:
    if not lesson_dir.exists():
        return []
    out = [parse_lessons(p) for p in sorted(lesson_dir.glob("*.md"))
           if p.name.lower() not in ("readme.md", "index.md")]
    return out


# ---------------- 测量（只量人给过框的东西） ----------------

def measure_region(img: np.ndarray, rect: list[int], cfg: Any) -> dict:
    """量一块矩形：尺寸、高宽比、主色、明亮像素占比。

    只回答"这块在像素上是什么样"，不回答"它该不该算光柱"——后者拿去和阈值比，
    由调用方判定。这样量出来的数字可以复用（换游戏、换阈值都不用重标）。
    """
    h_img, w_img = img.shape[:2]
    x, y, w, h = rect
    x = max(0, min(int(x), w_img - 1))
    y = max(0, min(int(y), h_img - 1))
    w = max(1, min(int(w), w_img - x))
    h = max(1, min(int(h), h_img - y))
    crop = img[y:y + h, x:x + w]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    best_color, best_frac = "", 0.0
    for color, rng in (getattr(cfg, "hsv", None) or {}).items():
        lo, hi = np.array(rng[0], np.uint8), np.array(rng[1], np.uint8)
        frac = float(cv2.inRange(hsv, lo, hi).mean()) / 255.0
        if frac > best_frac:
            best_color, best_frac = str(color), frac

    v = hsv[:, :, 2]
    bright = float((v >= 200).mean())
    return {
        "x": x, "y": y, "w": w, "h": h,
        "aspect": round(h / float(w), 2),
        "color": best_color,
        "color_fill": round(best_frac, 3),   # 这块里命中该颜色的像素占比
        "bright": round(bright, 3),          # 白芯/高光的强度
    }


def default_probe(cfg: Any, templates_dir: Path | None) -> Callable[[np.ndarray], dict]:
    """生产口径探针：2 个决策节拍（够光柱时域确认走一轮）。"""
    from .vision.detector import Detector

    def _run(img: np.ndarray) -> dict:
        det = Detector(cfg, templates_dir)
        st = None
        ts = 1000
        for _ in range(2):
            st = det.process(img, ts)
            ts += 500
        return {
            "in_game": bool(st.in_game),
            "gate_reason": det.diag.get("gate_reason", ""),
            "gate_best": float(det.diag.get("gate_best", 0.0) or 0.0),
            "gate_neg": float(det.diag.get("gate_neg_best", 0.0) or 0.0),
            "pillars": len(st.pillars),
            "pillar_colors": sorted({p.color for p in st.pillars}),
            "labels": len(st.ground_labels),
            "monsters": len(st.monsters),
        }

    return _run


# ---------------- 核对（claim vs 实测） ----------------

def _resolve_img(lesson: Lesson, rel: str, root: Path) -> Path | None:
    """图片路径：先按手册所在目录，再按 profile 根，最后按仓库根。"""
    if not rel:
        return None
    p = Path(rel)
    for base in (lesson.path.parent, root, Path.cwd()):
        cand = p if p.is_absolute() else (base / p)
        if cand.exists():
            return cand
    return None


def _verdict_positive(raw: str) -> bool | None:
    s = raw.strip().lower()
    if not s:
        return None
    for w in _NEGATIVE_WORDS:
        if w.lower() in s:
            return False
    for w in _POSITIVE_WORDS:
        if w.lower() in s:
            return True
    return None


def _norm_key(s: str) -> str:
    s = s.strip().lower()
    return {"空格": "space", "空": "space", "回": "enter", "回车": "enter",
            "esc": "esc", "escape": "esc"}.get(s, s)


def check_claims(lessons: list[Lesson], prof: Any, cfg: Any,
                 probe: Callable[[np.ndarray], dict] | None = None) -> list[dict]:
    """逐条核对。返回 [{title, kind, status, detail, evidence}]。

    cfg 是**完整 AppConfig**（不是 cfg.vision）：键位类要读 decision.pickup_key，
    视觉类读 vision.hsv —— 只传 vision 会让键位核对永远报"配置里没写"。

    status:
      ok    —— 程序现在的行为和你写的一致
      fail  —— 明确不一致（这是最有价值的输出：你的标注抓到了一个 bug）
      gap   —— 缺东西，量不了 / 认不了（缺图、缺框、没有锚点）
      skip  —— 这类本身不可量化（流程/禁忌/术语），已转成文字
    """
    vcfg = getattr(cfg, "vision", cfg)
    run = probe or default_probe(vcfg, getattr(prof, "templates_dir", None))
    rows: list[dict] = []

    for lesson in lessons:
        for c in lesson.claims:
            row = {"title": c.title, "kind": c.kind, "status": "skip",
                   "detail": "", "evidence": "", "source": c.source, "line": c.line}

            if c.kind in (KIND_FLOW, KIND_TABOO, KIND_TERM):
                row["detail"] = "文字类：已写进 game.md，供模型参考（无法用像素验）"
                rows.append(row)
                continue

            if c.kind == KIND_KEY:
                pairs = [(k, v) for k, v in c.fields.items()
                         if k not in _NON_ACTION_FIELDS]
                if not pairs:
                    row["status"] = "gap"
                    row["detail"] = "没读到『动作: 键』这一行（写成 `- 拾取: A` 这样的列表）"
                    rows.append(row)
                for act, key in pairs:
                    sub = dict(row, title=f"{act}", status="skip", evidence="")
                    rows.append(_check_key(act, key, cfg, sub))
                continue

            img_path = _resolve_img(lesson, c.file, prof.root)
            if img_path is None:
                row["status"] = "gap"
                row["detail"] = (f"缺图：{c.file or '（没写 图: 字段）'}。"
                                 f"截图丢进 lessons/img/ 再跑一次")
                rows.append(row)
                continue

            if c.kind == KIND_SCENE:
                rows.append(_check_scene(c, img_path, run, row))
            elif c.kind == KIND_OBJECT:
                rows.append(_check_object(c, img_path, vcfg, row))
            else:
                row["status"] = "gap"
                row["detail"] = "类型不认识，跳过"
                rows.append(row)
    return rows


def _read_img(path: Path) -> np.ndarray | None:
    try:
        return cv2.imread(str(path))
    except Exception:  # noqa: BLE001
        return None


def _check_scene(c: Claim, img_path: Path, run: Callable, row: dict) -> dict:
    want = _verdict_positive(c.get("verdict"))
    if want is None:
        row["status"] = "gap"
        row["detail"] = (f"『判定』没写清楚（现在是『{c.get('verdict')}』）。"
                         f"写「正例」（战斗页）或「反例」（菜单/面板/结算）")
        return row
    img = _read_img(img_path)
    if img is None:
        row["status"] = "gap"
        row["detail"] = f"图读不出来：{img_path.name}"
        return row
    got = run(img)
    same = bool(got["in_game"]) == want
    row["evidence"] = (f"门禁={got['gate_reason']} 模板={got['gate_best']:.2f}"
                       f" 反例={got['gate_neg']:.2f}")
    if same:
        row["status"] = "ok"
        row["detail"] = f"程序判{'战斗' if want else '非战斗'}，与你的判定一致"
    else:
        row["status"] = "fail"
        if want:
            row["detail"] = ("你说这是战斗页，程序没认出来 → 缺一块固定 HUD 锚点。"
                             "跑 tools/teach_crop.py 会自动从这张图量出 "
                             "gate_pos_*.png，不用手写坐标")
        else:
            row["detail"] = ("你说这不是战斗页，程序却放行了 → 会在这里乱按键，"
                             "需要一块反例模板 gate_neg_*.png"
                             "（tools/teach_crop.py 会挑一张只有这张菜单才有的静态块）")
    return row


def judge_object(m: dict, cfg: Any) -> list[str]:
    """拿一块量出来的尺寸/颜色去和生产阈值比，返回不通过的原因。

    单独抽出来是因为有两个调用方：`_check_object`（核对手册标的框）和
    `teach_crop`（从文档自动量出一束光柱后再判一次）。两边必须用同一套阈值
    和同一句话术，否则"手册说能检出、自动量说检不出"会互相打脸。
    """
    min_asp = float(getattr(cfg, "pillar_min_aspect", 2.5))
    max_asp = float(getattr(cfg, "pillar_max_aspect", 15.0))
    min_w = int(getattr(cfg, "pillar_min_width", 2))
    max_w = int(getattr(cfg, "pillar_max_width", 70))
    min_area = int(getattr(cfg, "pillar_min_area", 40))
    area = m["w"] * m["h"]

    problems: list[str] = []
    if m["aspect"] < min_asp:
        problems.append(f"高宽比 {m['aspect']} < 阈值 {min_asp}（会被当成团光/节点）")
    if m["aspect"] > max_asp:
        problems.append(f"高宽比 {m['aspect']} > 上限 {max_asp}（会被当成特效/长条）")
    if m["w"] < min_w or m["w"] > max_w:
        problems.append(f"宽 {m['w']}px 不在 [{min_w}, {max_w}] 内")
    if area < min_area:
        problems.append(f"面积 {area}px² < 阈值 {min_area}")
    if not m["color"] or m["color_fill"] < 0.08:
        problems.append("色相不落在任何已配置的颜色区间（hsv 里没这个色）")
    return problems


def _check_object(c: Claim, img_path: Path, cfg: Any, row: dict) -> dict:
    rect = c.rect
    img = _read_img(img_path)
    if img is None:
        row["status"] = "gap"
        row["detail"] = f"图读不出来：{img_path.name}"
        return row
    if rect is None:
        where = c.get("pos").strip()
        row["status"] = "gap"
        if where:
            row["detail"] = (f"有方位（『{where}』）但没量出框。"
                             f"跑 tools/teach_crop.py 会自动量，不用手写坐标")
        else:
            row["detail"] = ("缺框：这是整帧里的一块，得告诉我它在哪。"
                             "加一行 `- 在: 右下角`，或者直接给 `- 框: x,y,w,h`")
        return row

    m = measure_region(img, rect, cfg)
    row["evidence"] = (f"{m['w']}x{m['h']}px 高宽比={m['aspect']} "
                       f"主色={m['color'] or '未知'}({m['color_fill']:.0%}) "
                       f"亮部={m['bright']:.0%}")

    problems = judge_object(m, cfg)
    if problems:
        row["status"] = "fail"
        row["detail"] = "按现在的判据检不出来：" + "；".join(problems)
    else:
        row["status"] = "ok"
        row["detail"] = "按现在的判据能检出（尺寸/颜色/高宽比都在范围内）"
    return row


_NON_ACTION_FIELDS = ("file", "verdict", "recognize", "todo", "rect", "note",
                      "pos", "tpl_name")


def _decision_attr(cfg: Any, path: str) -> str:
    """按点路径取 decision 下的值：`pickup_key` / `autonomy.escape_key`。"""
    obj = getattr(cfg, "decision", None)
    for part in path.split("."):
        if obj is None:
            return ""
        obj = getattr(obj, part, None)
    return str(obj or "")


def _check_key(action: str, key: str, cfg: Any, row: dict) -> dict:
    """一条『- 拾取: A』。cfg 是完整 AppConfig（键位分散在 decision 与 autonomy）。"""
    attr = ""
    matched = ""
    for aid, (cfg_attr, aliases) in _KEY_ACTIONS.items():
        if action in aliases:
            attr, matched = cfg_attr, aid
            break
    if not matched:
        # 「走位: 点地面」这种不是键，是描述。别报成缺口，它已经进 game.md 了。
        row["status"] = "skip"
        row["detail"] = "不是键位（自由文本），已写进 game.md"
        return row
    if not attr or getattr(cfg, "decision", None) is None:
        row["status"] = "gap"
        row["evidence"] = f"动作={matched} 配置字段=（无）"
        row["detail"] = (f"『{action}』程序目前没有对应的键位配置项，"
                         f"需要在 profile 的 config 里加一个才能用")
        return row
    row["evidence"] = f"动作={matched} 配置字段=decision.{attr}"

    have = _norm_key(_decision_attr(cfg, attr))
    want = _norm_key(key)
    row["evidence"] = f"decision.{attr}: 配置={have or '（空）'} 手册={want}"
    if have == want:
        row["status"] = "ok"
        row["detail"] = f"键位一致（{want}）"
    elif not have:
        row["status"] = "gap"
        row["detail"] = f"配置里没写这个键位，按手册填 {want} 即可"
    else:
        row["status"] = "fail"
        row["detail"] = (f"冲突：手册说 {want}，程序配置是 {have}。"
                         f"哪个对？以游戏实际为准，改 profile.yaml 的 decision.{attr}")
    return row


# ---------------- 编译产物 ----------------

def _clean(s: str, n: int = 200) -> str:
    return re.sub(r"\s+", " ", s).strip()[:n]


def build_glossary(lessons: list[Lesson]) -> dict[str, dict]:
    terms: dict[str, dict] = {}
    for lesson in lessons:
        for alias, canonical, note in lesson.terms:
            terms[alias] = {"canonical": canonical, "note": note,
                            "source": lesson.path.name}
    return terms


def build_md_block(lessons: list[Lesson]) -> str:
    """生成 game.md 的受管块。写给人看，也给模型看——所以要有表格，不要 JSON。"""
    scenes = [c for ls in lessons for c in ls.claims if c.kind == KIND_SCENE]
    objects = [c for ls in lessons for c in ls.claims if c.kind == KIND_OBJECT]
    flows = [c for ls in lessons for c in ls.claims if c.kind == KIND_FLOW]
    keys = [c for ls in lessons for c in ls.claims if c.kind == KIND_KEY]
    taboos = [c for ls in lessons for c in ls.claims if c.kind == KIND_TABOO]

    out: list[str] = [HEAD_BEGIN, ""]

    if scenes:
        out += ["### 界面：怎么认、该怎么处理", "",
                "| 界面 | 判定 | 一眼认出 | 该做什么 |", "|---|---|---|---|"]
        for c in scenes:
            want = _verdict_positive(c.get("verdict"))
            mark = "正例（战斗中）" if want else ("反例（不在战斗）" if want is False else "未定")
            out.append(f"| {c.title} | {mark} | {_clean(c.get('recognize'), 60)} "
                       f"| {_clean(c.get('todo'), 60)} |")
        out.append("")

    if objects:
        out += ["### 物体：长什么样", ""]
        for c in objects:
            out.append(f"- **{c.title}**")
            if c.get("color"):
                out.append(f"  - 颜色：{c.get('color')}")
            if c.get("shape"):
                out.append(f"  - 形状：{c.get('shape')}")
            if c.get("aspect"):
                out.append(f"  - 高宽比：{c.get('aspect')}")
            if c.get("count"):
                out.append(f"  - 一张图里的数量：{c.get('count')}")
            if c.get("recognize"):
                out.append(f"  - 认法：{_clean(c.get('recognize'), 120)}")
            if c.get("confusable"):
                out.append(f"  - 别认成：{c.get('confusable')}")
        out.append("")

    if flows:
        out += ["### 流程", ""]
        for c in flows:
            out.append(f"**{c.title}**")
            for ln in c.body:
                out.append(ln if re.match(r"^\s*\d+[.、)]", ln) else f"- {ln}")
            out.append("")

    if keys:
        out += ["### 键位", ""]
        for c in keys:
            for k, v in c.fields.items():
                if k in _NON_ACTION_FIELDS:
                    continue
                out.append(f"- {k}：{v}")
        out.append("")

    if taboos:
        out += ["### 禁忌（最高优先级，冲突时以此为准）", ""]
        for c in taboos:
            items = c.body or [c.title]
            for it in items:
                out.append(f"- {it.lstrip('-* ').strip()}")
        out.append("")

    out.append(HEAD_END)
    return "\n".join(out)


def _markdown_terms(lessons: list[Lesson]) -> str:
    lines: list[str] = []
    for lesson in lessons:
        for alias, canonical, note in lesson.terms:
            lines.append(f"| {alias} | {canonical} | {note} |")
    return "\n".join(lines)


def build_labels(lessons: list[Lesson], checks: list[dict], prof: Any,
                 existing: dict) -> tuple[list[dict], list[str]]:
    """界面类 claim → 台账条目。已有同图条目一律跳过（人的话更值钱）。"""
    have = {str(it.get("file", "")).replace("\\", "/") for it in existing.get("items", [])}
    have_names = {Path(h).name for h in have}
    new: list[dict] = []
    skipped: list[str] = []
    root = prof.root

    for lesson in lessons:
        for c in lesson.claims:
            if c.kind != KIND_SCENE:
                continue
            want = _verdict_positive(c.get("verdict"))
            img = _resolve_img(lesson, c.file, root)
            if want is None or img is None:
                continue
            try:
                rel = img.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                rel = img.as_posix()
            if rel in have or img.name in have_names:
                skipped.append(rel)
                continue

            chk = next((x for x in checks
                        if x["source"] == c.source and x["line"] == c.line), None)
            ok = bool(chk and chk["status"] == "ok")
            note = f"教学手册：{_clean(c.get('recognize'), 80) or c.title}。来源 {c.source}:{c.line}"
            if not ok and chk:
                note += f"｜待标定：{chk['detail']}"
            new.append({
                "file": rel,
                "game": getattr(prof, "game", "") or prof.id,
                "scene": c.title,
                "strict": ok,
                "expect": {"in_game": want},
                "note": note,
            })
            have.add(rel)
            have_names.add(img.name)
    return new, skipped


def compile_lessons(lessons: list[Lesson], prof: Any, cfg: Any,
                    labels_data: dict | None = None,
                    probe: Callable | None = None) -> TeachResult:
    """把教程编译成产物。纯函数：不写文件，写入交给 write_artifacts。"""
    res = TeachResult(lessons=[str(ls.path) for ls in lessons])
    for ls in lessons:
        res.warnings.extend(ls.warnings)
        for c in ls.claims:
            res.counts[KIND_LABEL[c.kind]] = res.counts.get(KIND_LABEL[c.kind], 0) + 1
        if ls.terms:
            res.counts["术语"] = res.counts.get("术语", 0) + len(ls.terms)

    res.glossary = build_glossary(lessons)
    res.md_block = build_md_block(lessons)
    res.checks = check_claims(lessons, prof, cfg, probe)
    res.labels_new, res.labels_seen = build_labels(lessons, res.checks, prof,
                                                   labels_data or {})

    # 待裁决清单：从 checks 里抽 fail/gap，形成"你要干什么"的具体动作
    for chk in res.checks:
        if chk["status"] == "fail":
            res.pending.append({
                "kind": chk["kind"], "title": chk["title"],
                "action": "定夺", "detail": chk["detail"],
                "evidence": chk["evidence"], "source": f"{chk['source']}:{chk['line']}",
            })
        elif chk["status"] == "gap":
            res.pending.append({
                "kind": chk["kind"], "title": chk["title"],
                "action": "补料", "detail": chk["detail"],
                "evidence": chk["evidence"], "source": f"{chk['source']}:{chk['line']}",
            })
    return res


# ---------------- 落盘 ----------------

def write_glossary(res: TeachResult, knowledge_dir: Path) -> Path:
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    p = knowledge_dir / "glossary.json"
    p.write_text(json.dumps({
        "version": 1,
        "updated": time.strftime("%Y-%m-%d"),
        "note": "用户的黑话/简称对照表。由 tools/teach.py 从 lessons/ 生成。",
        "terms": res.glossary,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return p


def write_game_md(res: TeachResult, knowledge_dir: Path) -> Path:
    """只替换标记块。块外一个字不动——那是人写的，比生成的更值钱。"""
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    p = knowledge_dir / "game.md"
    old = p.read_text(encoding="utf-8") if p.exists() else ""
    block = res.md_block.strip()

    if HEAD_BEGIN in old and HEAD_END in old:
        head, rest = old.split(HEAD_BEGIN, 1)
        _, tail = rest.split(HEAD_END, 1)
        # 块尾的空行必须先 normalize 再拼：直接 "\n" + tail 会让每跑一次多一个
        # 空行，跑十次就是十个——文件看不出来，diff 里全是噪声。
        tail_body = tail.strip("\n")
        new = head.rstrip() + "\n\n" + block + "\n"
        if tail_body.strip():
            new += "\n" + tail_body.rstrip() + "\n"
    else:
        sep = "\n\n" if old.strip() else ""
        new = old.rstrip() + sep + block + "\n"
    p.write_text(new, encoding="utf-8")
    return p


def write_labels(res: TeachResult, labels_path: Path) -> tuple[int, Path]:
    data: dict = {"version": 1, "howto": [], "items": []}
    if labels_path.exists():
        try:
            data = json.loads(labels_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            data = {"version": 1, "howto": [], "items": []}
    items = data.setdefault("items", [])
    items.extend(res.labels_new)
    data["updated"] = time.strftime("%Y-%m-%d")
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return len(res.labels_new), labels_path


def write_report(res: TeachResult, out_dir: Path, profile_id: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "report.md"
    counts = "，".join(f"{k} {v} 条" for k, v in res.counts.items()) or "（空）"
    lines = [
        f"# 教学编译报告 · {profile_id}",
        "",
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M')}",
        f"读到：{len(res.lessons)} 份手册，{counts}",
        "",
        "## 我读到了什么，各自去了哪",
        "",
        "| 条目 | 类型 | 结果 | 依据 |",
        "|---|---|---|---|",
    ]
    mark = {"ok": "✅ 一致", "fail": "❌ 不一致", "gap": "⚠️ 缺料", "skip": "📝 已成文"}
    for c in res.checks:
        lines.append(f"| {c['title']} | {KIND_LABEL.get(c['kind'], c['kind'])} "
                     f"| {mark.get(c['status'], c['status'])} | "
                     f"{_clean(c['detail'], 90)} |")

    if res.pending:
        lines += ["", "## 要你动手的（按这个顺序补）", ""]
        for i, it in enumerate(res.pending, 1):
            lines.append(f"{i}. **[{it['action']}] {it['title']}** — {it['detail']}")
            if it["evidence"]:
                lines.append(f"   - 实测：{it['evidence']}")
            lines.append(f"   - 出处：{it['source']}")

    if res.labels_new or res.labels_seen:
        lines += ["", "## 台账变更", ""]
        if res.labels_new:
            for it in res.labels_new:
                lines.append(f"- 新增（strict={it['strict']}）：`{it['file']}` "
                             f"→ {it['scene']}")
        if res.labels_seen:
            lines.append(f"- 已存在跳过 {len(res.labels_seen)} 条："
                         + "、".join(Path(s).name for s in res.labels_seen))

    if res.warnings:
        lines += ["", "## 格式警告", ""] + [f"- {w}" for w in res.warnings]

    lines += ["", "---", "",
              "重新编译：`py -3.12 tools/teach.py --profile " + profile_id + "`",
              "核对一致：`py -3.12 tools/teach.py --profile " + profile_id + " --check`",
              "把「界面」段变成模板（不用手写坐标，出 preview.html 供核对）："
              "`py -3.12 tools/teach_crop.py --profile " + profile_id + "`"]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def write_pending(res: TeachResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "pending.json"
    p.write_text(json.dumps({"updated": time.strftime("%Y-%m-%d %H:%M"),
                             "items": res.pending}, ensure_ascii=False, indent=2) + "\n",
                 encoding="utf-8")
    return p
