"""中间层：把「人写的文档」翻成「能进模板的资产」。

用户原话：

    "进模板还是很难的，就要一个文档转成模板，需要一个中间 ai 给转换，
     符合人类使用行为。"

手工做一块门禁模板，人要干四件事：量坐标、裁图、缩放到基准宽、拿别的图验分数。
四件全是机器比人准的活，却全压在写文档的人身上——这就是"进模板难"的全部原因。
这个模块把它们接走：文档里写「右下角技能轮盘」，这里给出 gate_pos_skillwheel.png
和一张能过目的预览。

为什么叫"中间 ai"却不调模型
----------------------
人类裁锚点时不靠猜，靠四件可验证的事：

    1. 我知道它在哪（右下角）              → 文档里的方位词
    2. 我知道它长什么样（一圈图标/一条细高的柱） → 文档里的形状词
    3. 它得是固定的、别处没有的            → 唯一性（匹配第二峰）
    4. 别的战斗图上也有、菜单上没有        → 跨图命中 / 误报

四条全部是像素能算的。让模型进来只会多一个"看起来挺像"的黑盒，还得多加一道
人工复核——那是把工作量从"量坐标"搬到"审模型"，并没有变少。所以这里做的是
"把人的判断步骤写成算法"，不是"让模型看图猜"。

「符合人类使用行为」落在七个具体决定上
------------------------------------
1. **人说方位，不说坐标。** 人只会说「右下角技能轮盘」，不会写 640,255,260,135。
   所以 `- 在: 右下角` 就够了。没写方位时按"人扫 HUD 先看四周"的先验，只搜四条
   边带——UI 几乎都在边上，中央那一大片是地面和特效，搜了只会选错。
2. **人挑不变的那块，不挑最显眼的那块。** 星图中间 7 个发光节点很显眼，但每张
   星图长得都不一样 → 当锚点等于没标；左上角「‹ 异界」标题块每张都一样 → 能当
   锚点。这一条对应"唯一性"评分（匹配第二峰）。
3. **人会拿别的图对一下。** "这块另一张战斗图上也在吧？菜单上应该没有。"——跨图
   命中率 + 误报率，两项都有硬阈值，不是加权平均能糊过去的。
4. **人一次只想看一个结论。** 给一个 best，备选收进 crops.json，不甩一屏候选让
   人自己挑。
5. **人要能当场改。** 量错了不用重跑整个流程：改 `teach/crops.json` 里的 rect 加
   `"locked": true`，再跑一次就按你给的裁。
6. **人要能一眼过目。** `teach/crops/preview.html`：原图 + 红框 + 裁出的模板 + 分数
   并排放。没有这一步，前面五条都是黑箱。
7. **人不喜欢被要求填表。** 缺图就报缺图，缺方位就报缺方位，绝不让工具的懒变成
   人写文档的负担。

坐标口径（两套，别混）
--------------------
    手册 `- 框: x,y,w,h`   → 左上角 + 宽高，源图像素
    crops.json 的 rect     → 同上（保持一致，人手改不用换算）
    tools/make_template.py → `--rect x0,y0,x1,y1`（历史口径，两角点）
"""

from __future__ import annotations

import base64
import html
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .teach import (
    KIND_OBJECT,
    KIND_SCENE,
    Lesson,
    _resolve_img,
    _verdict_positive,
    judge_object,
    measure_region,
    parse_rect,
)
from .vision.gate_score import (
    DEFAULT_BASE_WIDTH,
    match_best,
    match_peaks,
    normalize_crop,
    usable,
)

# ---------------- 角色 ----------------

ROLE_POS = "gate_pos"
ROLE_NEG = "gate_neg"
ROLE_OBJ = "object"
ROLE_LABEL = {ROLE_POS: "正向锚点", ROLE_NEG: "反例锚点", ROLE_OBJ: "物体样本"}
ROLE_PREFIX = {ROLE_POS: "gate_pos_", ROLE_NEG: "gate_neg_", ROLE_OBJ: "obj_"}

# ---------------- 人怎么写方位，这里就怎么认 ----------------
#
# 归一化坐标 (x0, y0, x1, y1)，相对整帧。不追求精确到边——人的"右下角"本来
# 就是个大概，精确边界交给后面的框搜索。
_DIR_ALIASES: dict[str, tuple[str, ...]] = {
    "TL": ("左上角", "左上", "左上方", "左上侧", "画面左上", "屏幕左上"),
    "TR": ("右上角", "右上", "右上方", "右上侧", "画面右上", "屏幕右上"),
    "BL": ("左下角", "左下", "左下方", "左下侧", "画面左下", "屏幕左下"),
    "BR": ("右下角", "右下", "右下方", "右下侧", "画面右下", "屏幕右下"),
    "MID": ("中间", "中央", "正中", "画面中间", "画面中央", "屏幕中间", "中部"),
    "TOP": ("顶部", "上方", "上边", "顶端", "最上面", "顶边"),
    "BOTTOM": ("底部", "下方", "下边", "底边", "最下面"),
    "LEFT": ("左侧", "左边", "左面", "最左边", "左边缘"),
    "RIGHT": ("右侧", "右边", "右面", "最右边", "右边缘"),
    "FULL": ("整屏", "全屏", "整个画面", "整张图", "全局"),
}
_DIR_RECT: dict[str, tuple[float, float, float, float]] = {
    "TL": (0.00, 0.00, 0.46, 0.46),
    "TR": (0.54, 0.00, 1.00, 0.46),
    "BL": (0.00, 0.54, 0.46, 1.00),
    "BR": (0.54, 0.54, 1.00, 1.00),
    "MID": (0.22, 0.22, 0.78, 0.78),
    "TOP": (0.00, 0.00, 1.00, 0.42),
    "BOTTOM": (0.00, 0.58, 1.00, 1.00),
    "LEFT": (0.00, 0.00, 0.40, 1.00),
    "RIGHT": (0.60, 0.00, 1.00, 1.00),
    "FULL": (0.00, 0.00, 1.00, 1.00),
}
# 没写方位时的默认搜索范围：四条边带。
# 这不是偷懒，是在复刻"人扫 HUD 先看四周"——UI 贴边，地面/怪/特效在中间，
# 全帧搜索会把 Boss 光环、技能轨迹这类高对比度的东西排在真锚点前面。
_DEFAULT_REGIONS: tuple[str, ...] = ("BR", "BL", "TOP", "BOTTOM")

# 人描述形状用的词 → 候选高宽比（h/w）。只提供几个档位，不做精确拟合：
# "一圈图标"本来就不是 1.00 和 1.03 的区别。
_SHAPE_HINTS: tuple[tuple[tuple[str, ...], tuple[float, ...]], ...] = (
    (("轮盘", "圆盘", "圆圈", "转盘", "方向盘"), (0.75, 1.0, 1.3)),
    (("摇杆", "摇把"), (0.7, 1.0, 1.5)),
    (("圆", "圈", "环形", "按钮", "图标", "技能键"), (0.7, 1.0, 1.4)),
    (("光柱", "光条", "细高", "竖直", "竖条", "柱"), (2.0, 3.4, 5.0)),
    (("标题", "栏", "横幅", "顶栏", "菜单栏", "血条"), (0.22, 0.34, 0.5)),
    (("格子", "背包", "物品列表", "网格"), (0.6, 0.9, 1.3)),
)
_DEFAULT_ASPECTS: tuple[float, ...] = (0.42, 0.68, 1.0, 1.5)

# 说"轮盘/圆/摇杆"这类词时，走圆检测而不是撒热点格——这是整个工具最重要的一条。
# 在 960x432 的真实战斗帧上，纯按"右下角"撒热点，排名第一的是一块聊天文字：
# 它在两张战斗图上逐字相同（同一段录屏的连续帧），跨图命中 1.00、第二峰 0.23，
# 三条验证全过。可它是最烂的锚点——聊天内容下一分钟就变了。人不会犯这个错，
# 因为人看到"右下角技能轮盘"就知道要找的是一个圈，不会去看左下角的字。
_CIRCLE_WORDS = ("轮盘", "圆盘", "圆圈", "转盘", "圆", "圈", "环形",
                 "光环", "摇杆", "节点", "按钮", "图标", "技能键")
_BAR_WORDS = ("标题", "栏", "横幅", "顶栏", "菜单栏", "血条", "进度条")
_NEGATION = ("没有", "没", "无", "不是", "别", "非", "看不到")

# 候选尺寸上限（占整帧的比例）。这是人的直觉：锚点是"HUD 的一个零件"，不是
# "小半屏"。实测纯靠"越独特越好"排序会稳定选中大块：大块包含更多内容，第二峰
# 天然更低，看起来更独特，但它把会变的东西（怪、特效、飘字）也框进去了，换一帧
# 就失配。人标的技能盘是 260x135（占帧 27%x31%），框一块 375x281（39%x65%）的
# 没有人会这么干。
_MAX_W_FRAC = 0.36
_MAX_H_FRAC = 0.42
_MAX_AREA_FRAC = 0.13

# 宽度档按**整帧宽**算，不按搜索区域宽算：否则同一个"0.48 档"在右下角给 212px、
# 在顶部横带给 460px，用户完全没法预期。
_WIDTH_STEPS: tuple[float, ...] = (0.10, 0.16, 0.24, 0.34)


# ---------------- 数据结构 ----------------

@dataclass
class Hint:
    """一处"我指的是什么、在哪个大方向"。

    方位必须和它后面的名词绑在一起看。"右下角技能轮盘 + 左下角摇杆，左上小地图"
    这句话里有三处，混成一个"右下|左下|左上"的方位集合就丢了最关键的信息——
    哪个方向对应哪个东西。而"轮盘"两个字决定了要用圆检测，"小地图"没有形状词
    就退回撒热点。
    """

    region: tuple[float, float, float, float]
    region_key: str
    noun: str
    kind: str = ""     # "circle" / "bar" / ""（没形状词，走通用路径）
    # 高宽比也按各自的名词定，不能用整句：同一句里"标题块"（扁）和"圆形节点"
    # （方）差着一倍，共用一套档位的结果是两边都框不准——实测标题块一次都没
    # 进候选，因为 width*0.7 算出的是个方块。
    aspects: tuple[float, ...] = _DEFAULT_ASPECTS

    @property
    def label(self) -> str:
        return f"{self.region_key}:{self.noun}" if self.noun else self.region_key


@dataclass
class Intent:
    """一条"我要一块模板"的意图，来自手册里的一个 `## 界面/物体` 段。"""

    id: str
    title: str
    role: str
    kind: str
    source: str
    line: int
    image: Path | None = None
    image_rel: str = ""
    where: str = ""                                   # 人写的方位原文
    hints: list[Hint] = field(default_factory=list)
    aspects: tuple[float, ...] = _DEFAULT_ASPECTS
    given_rect: list[int] | None = None               # 人已经量好的框，最高优先级
    tpl_name: str = ""                                # 人指定的模板名
    gap: str = ""                                     # 做不了的原因

    @property
    def hit_positive(self) -> bool:
        """这块模板应该"命中"的是战斗页还是菜单页。"""
        return self.role == ROLE_POS


@dataclass
class PoolItem:
    path: Path
    rel: str
    img: np.ndarray | None
    cls: bool | None      # True=战斗（正例） False=菜单（反例） None=判定不明
    title: str


@dataclass
class Cand:
    rect: list[int]          # x,y,w,h 源图像素
    tpl: np.ndarray
    self_score: float = 0.0  # 在源图上的匹配分（框歪了就会掉）
    second: float = 1.0      # 第二峰：越高越"到处都是"，越低越独特
    src: str = "hotspot"     # given / circle / hotspot —— 这块是哪条路径找到的
    weight: float = 0.45     # 那条路径有多硬（见 propose_rects）
    textness: float = 0.0    # 越像文字越高。人的底线是"绝不拿文字当锚点"
    area_frac: float = 0.0   # 框面积占整帧的比例，给 _size_fit 用
    ev: dict = field(default_factory=dict)

    @property
    def unique(self) -> float:
        return round(1.0 - max(0.0, self.second), 3)


@dataclass
class CropResult:
    intent: Intent
    best: Cand | None = None
    alts: list[Cand] = field(default_factory=list)
    reason: str = ""
    verified: bool = False      # 是否真的做了跨图验证
    locked: bool = False        # 框是人指定的（--set 或 crops.json 里的 locked）
    object_note: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.best and self.best.ev.get("ok"))


@dataclass
class CropReport:
    profile: str
    results: list[CropResult] = field(default_factory=list)
    roi_suggest: list[int] | None = None
    roi_current: list[int] | None = None
    templates: dict[str, str] = field(default_factory=dict)   # intent.id → 模板文件名
    written: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def done(self) -> list[CropResult]:
        return [r for r in self.results if r.best is not None]

    @property
    def gaps(self) -> list[CropResult]:
        return [r for r in self.results if r.best is None]


# ---------------- 第一步：读文档，变成意图 ----------------

def _kind_of(noun: str) -> str:
    """名词 → 用什么算子去找。认不出就给空串，走通用撒热点。"""
    if any(w in noun for w in _CIRCLE_WORDS):
        return "circle"
    if any(w in noun for w in _BAR_WORDS):
        return "bar"
    return ""


def parse_hints(text: str) -> list[Hint]:
    """"右下角技能轮盘 + 左下角摇杆，左上小地图" → 三条 Hint。

    方位词按"最长优先"匹配，命中后占位，避免"右下"和"右下角"各算一次。
    每个方位词后面、到下一个方位词（或标点）之前的文字，就是它指向的东西。

    否定要单独判：「右下**没有**技能轮盘」说的是这里什么都没有，不是让你去找。
    这种句子在真实手册里一定会出现——写"一眼认出"时人习惯把不在的东西也列上。
    """
    if not text:
        return []
    taken: list[tuple[int, int]] = []
    hits: list[tuple[int, int, str]] = []      # (start, end, region_key)
    pairs = sorted(((w, k) for k, ws in _DIR_ALIASES.items() for w in ws),
                   key=lambda t: -len(t[0]))
    for word, key in pairs:
        start = 0
        while True:
            i = text.find(word, start)
            if i < 0:
                break
            span = (i, i + len(word))
            if all(span[1] <= a or span[0] >= b for a, b in taken):
                taken.append(span)
                hits.append((span[0], span[1], key))
            start = i + len(word)
    hits.sort()

    out: list[Hint] = []
    seen: set[tuple[str, str]] = set()
    for idx, (start, end, key) in enumerate(hits):
        nxt = hits[idx + 1][0] if idx + 1 < len(hits) else len(text)
        noun = text[end:nxt]
        # 到标点为止：逗号/句号/+ 之后一般是另一件事
        noun = re.split(r"[，,。；;、+]", noun, maxsplit=1)[0]
        noun = noun.strip(" 　:：.、（）()【】[]「」『』\"'“”")
        if any(noun.startswith(neg) or noun[:2] == neg for neg in _NEGATION):
            continue
        if (key, noun) in seen:
            continue
        seen.add((key, noun))
        out.append(Hint(region=_DIR_RECT[key], region_key=key, noun=noun,
                        kind=_kind_of(noun), aspects=aspects_from(noun)))
    return out


def aspects_from(text: str) -> tuple[float, ...]:
    """从描述里猜这块的大致高宽比。猜不出给默认档。"""
    if not text:
        return _DEFAULT_ASPECTS
    for words, aspects in _SHAPE_HINTS:
        if any(w in text for w in words):
            return aspects
    return _DEFAULT_ASPECTS


def _looks_like_placeholder(raw: str) -> bool:
    s = (raw or "").strip()
    return not s or s in ("待补充", "待定", "无", "-", "?", "？", "unknown", "TBD")


def plan_intents(prof: Any, lessons: list[Lesson]) -> list[Intent]:
    """手册里的每个「界面」「物体」段 → 一条意图。判定不明的跳过并记账。"""
    out: list[Intent] = []
    for lesson in lessons:
        for c in lesson.claims:
            if c.kind not in (KIND_SCENE, KIND_OBJECT):
                continue
            where = c.get("pos").strip()
            describe = " ".join([c.title, c.get("recognize", ""), c.get("shape", ""),
                                 c.get("note", ""), where])
            intent = Intent(
                id=f"{c.kind}|{c.title}",
                title=c.title,
                role=ROLE_OBJ if c.kind == KIND_OBJECT else ROLE_POS,
                kind=c.kind,
                source=c.source,
                line=c.line,
                where=where,
                aspects=aspects_from(describe),
                given_rect=c.rect,
                tpl_name=c.get("tpl_name").strip(),
            )
            if c.kind == KIND_SCENE:
                want = _verdict_positive(c.get("verdict"))
                if want is None:
                    intent.gap = (f"『判定』没写清楚（现在是『{c.get('verdict')}』），"
                                  f"不知道这块模板该判命中还是该判拦下")
                    out.append(intent)
                    continue
                intent.role = ROLE_POS if want else ROLE_NEG

            if _looks_like_placeholder(c.file):
                intent.gap = "缺图：这一段还没指定截图（`- 图:` 是空的或待补充）"
                out.append(intent)
                continue
            img = _resolve_img(lesson, c.file, prof.root)
            if img is None:
                intent.gap = f"缺图：{c.file} 找不到（按手册目录、profile 根、仓库根都试过了）"
                out.append(intent)
                continue
            intent.image = img
            try:
                intent.image_rel = img.resolve().relative_to(prof.root.resolve()).as_posix()
            except ValueError:
                intent.image_rel = img.as_posix()
            hints = parse_hints(describe)
            if where:
                # 人显式写了 `- 在:` 就以它为准，排在前面
                wh = parse_hints(where)
                if wh:
                    known = {h.label for h in wh}
                    hints = wh + [h for h in hints if h.label not in known]
            intent.hints = hints or [Hint(region=_DIR_RECT[k], region_key=k, noun="")
                                     for k in _DEFAULT_REGIONS]
            out.append(intent)
    return out


def load_pool(lessons: list[Lesson], prof: Any) -> dict[Path, PoolItem]:
    """验证集：手册里所有指定了图的界面，按人的判定分成 应命中 / 不应命中 两堆。

    这就是"人会拿别的图对一下"里的"别的图"。它完全来自人已经写的东西，不需要
    额外标注——写文档的时候就顺手写完了。
    """
    pool: dict[Path, PoolItem] = {}
    for lesson in lessons:
        for c in lesson.claims:
            if c.kind != KIND_SCENE or _looks_like_placeholder(c.file):
                continue
            img_path = _resolve_img(lesson, c.file, prof.root)
            if img_path is None:
                continue
            key = img_path.resolve()
            cls = _verdict_positive(c.get("verdict"))
            if key in pool:
                if pool[key].cls != cls and cls is not None:
                    pool[key].cls = None      # 同图两个相反判定 → 谁都不信
                continue
            try:
                rel = key.relative_to(prof.root.resolve()).as_posix()
            except ValueError:
                rel = key.as_posix()
            pool[key] = PoolItem(path=key, rel=rel, img=cv2.imread(str(key)),
                                 cls=cls, title=c.title)
    return pool


# ---------------- 第二步：找框 ----------------

def _hotspots(sub: np.ndarray, k: int) -> list[tuple[int, int]]:
    """"视线落点"：局部边缘能量最高处。

    人扫一眼画面，目光会落在有边框、有字、有图标的地方，不会落在草地和天空。
    Sobel 幅值 + 大核模糊 = 一块区域整体"花不花"，正好对上这个行为。核开大一
    点（约画面宽的 1/20）是为了让整块 UI 一起亮，而不是每个图标各亮一下。
    """
    if sub.size == 0:
        return []
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY) if sub.ndim == 3 else sub
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    energy = cv2.magnitude(gx, gy)
    kw = max(9, (sub.shape[1] // 20) | 1)
    kh = max(9, (sub.shape[0] // 20) | 1)
    energy = cv2.blur(energy, (kw, kh))

    dil = cv2.dilate(energy, np.ones((kh, kw), np.uint8))
    peaks = (energy >= dil - 1e-6) & (energy > float(energy.mean()))
    ys, xs = np.nonzero(peaks)
    if len(xs) == 0:
        return []
    order = np.argsort(-energy[ys, xs])
    out: list[tuple[int, int]] = []
    min_d = max(kw, kh) // 2
    for idx in order:
        x, y = int(xs[idx]), int(ys[idx])
        if all(abs(x - ox) > min_d or abs(y - oy) > min_d for ox, oy in out):
            out.append((x, y))
        if len(out) >= k:
            break
    return out


def _iou(a: list[int], b: list[int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / float(union) if union > 0 else 0.0


def _crowded(rect: list[int], others: list[list[int]], thr: float) -> bool:
    return any(_iou(rect, o) > thr for o in others)


def _circle_groups(sub: np.ndarray, max_groups: int = 6) -> list[tuple[list[int], int]]:
    """找"一圈图标"：先找所有圆，再用单链接聚类把挨着的圆并成一簇。

    这一步替代的是人眼最自然的一次动作——看到 4 个技能图标围成一圈，框住的是
    整圈，不是某一个图标。单靠"高宽比接近 1"的方框撞不出这种结构：它只会给出
    成百上千个差不多方的方块，其中 99% 是地面纹理和特效。

    为什么用单链接（并查集）而不是"每个圆当种子收邻居"：后者在图标疏密不均时
    会把同一圈切成两三段，各段分别成为候选，最后选中的往往是其中最小的那段
    （实测把 4 个技能图标切成了"右下那个按钮 + 时间戳水印"）。单链接只要链上
    就连成一体，一圈必然是一簇。

    返回 [(外接框, 圆的数量)]，数量用来判断"这是不是一圈"——1 个圆是单个按钮，
    ≥3 个圆才是"一圈图标"。
    """
    H, W = sub.shape[:2]
    if min(H, W) < 40:
        return []
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
    short = min(W, H)
    r_min = max(7, int(short * 0.035))
    r_max = max(r_min + 6, int(short * 0.22))

    found = None
    # 从严到松试三档：先只要最像圆的，找不到再放宽。宁缺勿滥——放宽后误检的圆
    # 会把一堆特效围成的"假轮盘"顶上来，比"没找到"更难排查。
    for p2 in (30, 24, 19):
        c = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
                             minDist=max(18, r_min * 2), param1=110, param2=p2,
                             minRadius=r_min, maxRadius=r_max)
        if c is not None and c.shape[1] > 0:
            found = c[0][:30]
            break
    if found is None:
        return []

    circles = [(float(x), float(y), float(r)) for x, y, r in found]
    radii = sorted(c[2] for c in circles)
    med_r = radii[len(radii) // 2]
    reach = 3.5 * med_r          # 技能图标间距约 2.5~3.5 倍半径，正好连成一圈

    parent = list(range(len(circles)))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            if dx * dx + dy * dy <= reach * reach:
                ri, rj = _find(i), _find(j)
                if ri != rj:
                    parent[ri] = rj

    clusters: dict[int, list[int]] = {}
    for i in range(len(circles)):
        clusters.setdefault(_find(i), []).append(i)

    boxes: list[tuple[list[int], int]] = []
    for members in clusters.values():
        cs = [circles[i] for i in members]
        x0 = min(c[0] - c[2] for c in cs)
        x1 = max(c[0] + c[2] for c in cs)
        y0 = min(c[1] - c[2] for c in cs)
        y1 = max(c[1] + c[2] for c in cs)
        w, h = x1 - x0, y1 - y0
        # 外扩 10%：圆检测给的是图标可视区域，人框的时候会把外圈描边一起框进来。
        # 外扩后可能超出搜索区域（右下角贴着画面边），必须再 clamp 一次——漏这一步
        # 会返回一个比区域还高的框，随后被"框高不超过帧高 42%"的规则整条丢掉，
        # 表现是"明明找到了一整圈图标，最后却只剩一个单按钮"。
        w2 = min(int(w * 1.2), W)
        h2 = min(int(h * 1.2), H)
        bx = int(max(0, min(W - w2, x0 - w * 0.1)))
        by = int(max(0, min(H - h2, y0 - h * 0.1)))
        boxes.append(([bx, by, w2, h2], len(cs)))

    out: list[tuple[list[int], int]] = []
    for b, n in sorted(boxes, key=lambda t: (-t[1], -t[0][2] * t[0][3])):
        if _crowded(b, [o for o, _ in out], 0.80):
            continue
        out.append((b, n))
        if len(out) >= max_groups:
            break
    return out


def _textness(tpl: np.ndarray) -> float:
    """"这块里文字像素占多大比例"。

    只用来**提醒**，不用来排序。理由是实测过：按"小连通域个数占比"算，聊天框
    0.93、技能盘 0.87、整块技能盘 0.93——根本分不开；按"文字像素面积占比"算，
    聊天框 0.090、左上小地图 0.072、Boss 光环 0.079——还是分不开。游戏画面本身
    就满是碎块，像素级判不出"这是字"。

    所以这里不做"我帮你排除了文字"这种承诺（那是假装）。真正的解法在
    `parse_hints`：人说"右下角技能轮盘"就只在圆里找，聊天框压根进不了候选池。
    而这个分数用来在预览里写一句"这块有 X% 是文字，文字会变，确认一下"。
    """
    if tpl is None or tpl.size == 0:
        return 0.0
    gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY) if tpl.ndim == 3 else tpl
    if min(gray.shape[:2]) < 8:
        return 0.0
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(th, 8)
    total = float(gray.shape[0] * gray.shape[1])
    lim_h = max(8, gray.shape[0] * 0.40)
    lim_w = max(8, gray.shape[1] * 0.25)
    text_px = 0
    for i in range(1, n):
        w, h, a = int(stats[i][2]), int(stats[i][3]), int(stats[i][4])
        if a < 6:
            continue
        if h <= lim_h and w <= lim_w and a <= 500:
            text_px += a
    return round(min(1.0, text_px / total), 3)


def propose_rects(intent: Intent, img: np.ndarray, *,
                  hotspots: int = 4, limit: int = 200
                  ) -> list[tuple[list[int], str, float]]:
    """按 hint 铺候选框，返回 [(rect, src, weight)]。

    weight 是"这条路径的证据有多硬"：人手写的框 1.0；圆检测聚成 3 个以上的簇
    （一圈图标）1.0，单个圆 0.33（可能只是个按钮）；纯撒热点 0.45。排序时它在
    跨图分数之后起作用——数据说话优先，这条只用来打破同分。

    两类路径，优先级不同：

        hint 说了"轮盘/圆/摇杆/图标"  → 只信圆检测的结果。找到了就不混通用候选，
                                      否则聊天框、任务文字又会挤回来
        没说形状（"小地图""标题块"）  → 撒热点 + 人的形状词定高宽比

    三条硬约束拦在 `_add` 里：宽 ≤ 36% 帧宽、高 ≤ 42% 帧高、面积 ≤ 13%。
    这不是为了省算力，是因为"锚点是小而固定的一个零件"——超过这个尺度的框必然
    把会变的内容（怪、特效、飘字）框进去，跨帧就废。
    """
    H, W = img.shape[:2]
    max_w = int(W * _MAX_W_FRAC)
    max_h = int(H * _MAX_H_FRAC)
    max_area = int(W * H * _MAX_AREA_FRAC)

    out: list[tuple[list[int], str, float]] = []
    placed: list[list[int]] = []

    def _add(rect: list[int], src: str, weight: float = 0.45) -> None:
        x, y, w, h = [int(v) for v in rect]
        if src == "circle":
            # 圆框是算出来的，不截断（截了就不是那个圆了），只做合理性检查
            if w < 20 or h < 20 or w > W * 0.55 or h > H * 0.55:
                return
        else:
            w, h = min(w, max_w), min(h, max_h)
            if w < 24 or h < 14 or w * h > max_area:
                return
        if w > W or h > H:
            return
        x = max(0, min(W - w, x))
        y = max(0, min(H - h, y))
        r = [x, y, w, h]
        if _crowded(r, placed, 0.82):
            return
        placed.append(r)
        out.append((r, src, float(weight)))

    for hint in intent.hints:
        rx0, ry0, rx1, ry1 = hint.region
        x0, y0 = int(rx0 * W), int(ry0 * H)
        x1, y1 = min(W, int(rx1 * W)), min(H, int(ry1 * H))
        if x1 - x0 < 24 or y1 - y0 < 24:
            continue
        sub = img[y0:y1, x0:x1]
        RW, RH = x1 - x0, y1 - y0

        if hint.kind == "circle":
            groups = _circle_groups(sub)
            if groups:
                for (gx, gy, gw, gh), n in groups:
                    # 3 个以上的圆才算"一圈"，单个圆只给 1/3 权重
                    _add([x0 + gx, y0 + gy, gw, gh], "circle", min(1.0, n / 3.0))
                continue
            # 一个圆都没找到 → 退回通用路径。人说"轮盘"可能实际是个方框图标，
            # 不该因为算子找空就整条放弃。

        hots = _hotspots(sub, hotspots) or [(RW // 2, RH // 2)]
        for (hx, hy) in hots:
            for wr in _WIDTH_STEPS:
                w = int(W * wr)
                if w >= RW:
                    w = int(RW * 0.92)
                if w < 24:
                    continue
                for asp in (hint.aspects or intent.aspects):
                    h = int(w * asp)
                    if h < 14 or h > RH * 1.25:
                        continue
                    _add([x0 + hx - w // 2, y0 + hy - h // 2, w, h], "hotspot")
        if len(out) >= limit:
            break
    return out
    return out


# ---------------- 第三步：验框（"拿别的图对一下"） ----------------

def evaluate(tpl: np.ndarray, intent: Intent, pool: dict[Path, PoolItem],
             cfg: Any) -> dict:
    """这块模板在验证集上的表现。

    `ok` 的两个条件是硬性的，不做加权折中：
        该命中的每一张都过阈值     ← 少一张就会在那一帧里"认不出自己在战斗"
        不该命中的一张都不过阈值   ← 多一张就会在菜单里乱按键

    宁可 `ok=False` 让人看一眼，也不要一个"平均分 0.72 挺好"但其实两头顶穿的
    折中结果。
    """
    g = cfg.vision.scene_gate
    pos_thr = float(g.threshold)
    neg_thr = float(g.negative_threshold)
    hit_thr = pos_thr if intent.hit_positive else neg_thr
    miss_thr = neg_thr if intent.hit_positive else pos_thr
    mdw = int(getattr(cfg.vision, "max_detect_width", 0) or 0)
    ref = list(getattr(cfg.vision, "reference_resolution", [1280, 720]) or [1280, 720])

    hit: list[float] = []
    miss: list[float] = []
    detail: list[str] = []
    for item in pool.values():
        if item.img is None or item.cls is None:
            continue
        s = match_best(item.img, tpl, whole_frame=True,
                       base_width=DEFAULT_BASE_WIDTH, ref_res=ref,
                       max_detect_width=mdw)
        detail.append(f"{Path(item.rel).name}={s:.2f}"
                      f"{'✓' if item.cls == intent.hit_positive else '✗'}")
        (hit if item.cls == intent.hit_positive else miss).append(s)

    ok_hit = bool(hit) and min(hit) >= hit_thr
    ok_miss = (not miss) or max(miss) < miss_thr
    margin = (min(hit) - max(miss)) if (hit and miss) else None
    return {
        "hit_min": round(min(hit), 3) if hit else None,
        "hit_mean": round(sum(hit) / len(hit), 3) if hit else None,
        "miss_max": round(max(miss), 3) if miss else None,
        "margin": round(margin, 3) if margin is not None else None,
        "hit_thr": hit_thr,
        "miss_thr": miss_thr,
        "n_hit": len(hit),
        "n_miss": len(miss),
        "ok": bool(ok_hit and ok_miss),
        "detail": " ".join(detail),
    }


def _size_fit(area_frac: float) -> float:
    """框的面积是否"像个 HUD 零件"。人的锚点大致落在整帧的 4%~10%。

    太小（<3.5%）框不住足够独特的结构，随便一片地面纹理都能匹配上；太大
    （>10%）必然把会变的东西带进来。两头都扣分，中间给满分——这是实测人标的
    两块锚点（技能盘 8.5%、星图标题块 5.8%）反推出来的区间。
    """
    if 0.030 <= area_frac <= 0.11:
        return 1.0
    if area_frac < 0.030:
        # 小框不是"更精致"，是"更容易在别处误匹配"：70x70 的技能按钮在整帧里
        # 能找到几十个长得差不多的圆。所以这头罚得比大框重。
        return round(0.35 * (area_frac / 0.030), 3)
    return round(max(0.0, 1.0 - (area_frac - 0.11) / 0.08), 3)


def _rank_key(c: Cand) -> tuple:
    """排序：硬条件 → 别是文字 → 分离度 → 余量 → 来源 → 唯一性。

    顺序是有讲究的，每一条都对应一次真实踩坑或一次人的直觉：

    1. `ok` 放第一位："能用的差模板"和"好用的不能用模板"之间没有可比性，
       综合打分只会让后者偶尔冒到前面，被人一眼看出不对，白费一次信任。
    2. 分离度、余量：跨图命中 vs 误命中，越宽松越好。这是唯一有真实证据的一项。
    3. 来源权重：人明确说了"轮盘"，圆检测聚出的一圈就比瞎撒热点更值得信。
    4. 尺寸是否像个 HUD 零件：太小容易在别处误匹配，太大必然框进会变的内容。
    5. 唯一性：最后才看"这块是不是到处都是"。

    文字没有放进排序——试过，分不开（见 `_textness` 的注释）。挡住聊天框的是
    `parse_hints` 的"形状词绑定"，不是这里。
    """
    ev = c.ev
    margin = ev.get("margin")
    return (
        1 if ev.get("ok") else 0,
        round(margin if margin is not None else -1.0, 3),
        round((ev.get("hit_min") or 0.0) - (ev.get("hit_thr") or 0.0), 3),
        round(c.weight, 3),
        round(_size_fit(c.area_frac), 3),
        c.unique,
        round(c.self_score, 3),
    )


def search(intent: Intent, pool: dict[Path, PoolItem], cfg: Any, *,
           tpl_dir: Path | None = None, keep_alts: int = 3,
           top: int = 32) -> CropResult:
    """给一条意图找出最好的框。"""
    if intent.gap:
        return CropResult(intent, reason=intent.gap)
    if intent.image is None:
        return CropResult(intent, reason="没有可用的截图")
    img = cv2.imread(str(intent.image))
    if img is None:
        return CropResult(intent, reason=f"图读不出来：{intent.image}")

    if intent.role == ROLE_OBJ:
        return _search_object(intent, img, cfg, tpl_dir)

    rects = propose_rects(intent, img)
    if intent.given_rect:
        # 人写的框永远排第一：先验它，但照样验分。不合理就明说，不偷偷换掉。
        rects = [(intent.given_rect, "given", 1.0)] + [
            (r, s, w) for (r, s, w) in rects if _iou(r, intent.given_rect) < 0.7]
    if not rects:
        return CropResult(intent, reason="方位区域太小，铺不出候选框（检查 `- 在:` 写的位置）")

    mdw = int(getattr(cfg.vision, "max_detect_width", 0) or 0)
    # 粗筛：只看"这块独不独特"。一次整帧匹配同时给出自身分和第二峰，很便宜。
    pre: list[Cand] = []
    for r, src, weight in rects:
        tpl = normalize_crop(img, r, DEFAULT_BASE_WIDTH)
        if tpl.size == 0 or not usable(tpl):
            continue
        best, second = match_peaks(img, tpl, base_width=DEFAULT_BASE_WIDTH,
                                  max_detect_width=mdw)
        if best < 0.55:
            continue                      # 框自己都匹配不上自己 → 框歪了
        pre.append(Cand(rect=r, tpl=tpl, self_score=best, second=second,
                        src=src, weight=weight, textness=_textness(tpl),
                        area_frac=r[2] * r[3] / float(img.shape[0] * img.shape[1])))

    if not pre:
        return CropResult(intent, reason="这块区域里没有结构足够的块（纯色/过曝？）")

    # 第二峰太高 = 这玩意到处都是，当锚点没意义。留一点余量给纹理丰富的真 HUD。
    pre = [c for c in pre if c.second <= 0.93]
    if not pre:
        return CropResult(intent, reason="候选块在画面里都有大量相似项（不独特），"
                                        "往更静态的 UI 上找（标题栏/固定边框比图标稳）")
    # 人写的框先占位：picked 用"互不重叠"筛，谁先进谁留下。不排在最前面的话，
    # 一个和它重叠 0.6 的自动框会先占位，人的框反被筛掉。
    pre.sort(key=lambda c: (0 if c.src == "given" else 1, c.second, -c.self_score))

    # 挑一批互不重叠的候选去精算：否则 top32 会全是同一块技能盘的微调
    picked: list[Cand] = []
    for c in pre:
        if _crowded(c.rect, [p.rect for p in picked], 0.55):
            continue
        picked.append(c)
        if len(picked) >= top:
            break

    for c in picked:
        c.ev = evaluate(c.tpl, intent, pool, cfg)
    picked.sort(key=_rank_key, reverse=True)

    # 人写的框只要验得过，就直接用它 —— 不参与排序。
    # 这和"放候选列表第一位"不是一回事：排序按跨图分数走，自动找的"整盘"框
    # 往往比人标的"下半盘"分数更高，于是人的框被顶掉。人的判断不该输给分数
    # 的第三位小数。
    if intent.given_rect:
        manual = next((c for c in picked if c.rect == intent.given_rect), None)
        if manual is not None and manual.ev.get("ok"):
            picked.remove(manual)
            picked.insert(0, manual)

    best = picked[0]
    n_hit = best.ev.get("n_hit", 0)
    n_miss = best.ev.get("n_miss", 0)
    verified = n_hit >= 2 and n_miss >= 1
    res = CropResult(intent, best=best, alts=picked[1:1 + keep_alts], verified=verified)
    res.reason = _explain(best, verified, n_hit, n_miss)
    return res


_SRC_TEXT = {"circle": "按你说的形状找圆", "hotspot": "按方位扫热点",
             "given": "用你给的框"}


def _explain(c: Cand, verified: bool, n_hit: int, n_miss: int) -> str:
    ev = c.ev
    how = _SRC_TEXT.get(c.src, c.src)
    if not verified:
        by = "" if c.src == "given" else f"按唯一性选的（第二峰 {c.second:.2f}）。"
        return (f"{how}；只有 {n_hit} 张应命中 / {n_miss} 张不应命中的样本，没法跨图验证。"
                f"{by}再补一两张别的界面，这块的可靠性才敢签字")
    if ev.get("ok"):
        return (f"{how}；在 {n_hit} 张应命中的图上最低 {ev['hit_min']:.2f}"
                f"（阈值 {ev['hit_thr']:.2f}），在 {n_miss} 张不该命中的图上最高 "
                f"{ev['miss_max']:.2f}（阈值 {ev['miss_thr']:.2f}），两头都留了余量")
    if (ev.get("hit_min") or 0) < ev["hit_thr"]:
        return (f"{how}；没有一块能在所有应命中的图上过阈值：最好的这块最低只有 "
                f"{ev['hit_min']:.2f} < {ev['hit_thr']:.2f}。"
                f"通常是那几张图的 HUD 长得不一样（换分辨率/换职业/换画质）")
    return (f"这块在不应命中的图上就给了 {ev['miss_max']:.2f} ≥ {ev['miss_thr']:.2f}，"
            f"会误拦/误放。换一块更贴近这块菜单独有的 UI（标题、固定边框）")


# ---------------- 物体：不裁模板，量样本 ----------------

def _scan_tall(img: np.ndarray, k: int = 4) -> list[list[int]]:
    """"最像光柱的竖长亮块"：亮 + 有颜色 + 竖长。

    这条分支只在生产检测器一束都检不出来时走。目的不是"我帮你找到光柱"，
    而是"把你漏掉的那束量给你看，好定阈值该松多少"。
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v, s = hsv[:, :, 2], hsv[:, :, 1]
    mask = (((v >= 140) & (s >= 55)).astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out: list[tuple[float, list[int]]] = []
    for i in range(1, n):
        x, y, w, h, area = (int(vv) for vv in stats[i])
        if w < 3 or h < 10 or area < 40:
            continue
        asp = h / float(w)
        if not (1.6 <= asp <= 14.0):
            continue
        bonus = 1.0 if 2.5 <= asp <= 8.0 else 0.55
        out.append((area * bonus, [x, y, w, h]))
    out.sort(key=lambda t: -t[0])
    return [r for _, r in out[:k]]


def _search_object(intent: Intent, img: np.ndarray, cfg: Any,
                   tpl_dir: Path | None = None) -> CropResult:
    """光柱这类"满地跑、位置不固定"的东西不能裁模板，只能量样本。

    人找光柱也不是靠记模板，是靠"又细又高、亮的、金色的"。所以这里直接跑生产
    检测器：检出的就是"现在的判据能认的"；一束都检不出就退回尺寸搜索，把差距
    量出来——这样"阈值太严"就是一句有数字的话，而不是一句抱怨。
    """
    from .vision.detector import Detector

    cands: list[Cand] = []
    detected = 0
    try:
        det = Detector(cfg.vision, tpl_dir if tpl_dir and Path(tpl_dir).exists() else None)
        ts = 1000
        st = None
        for _ in range(2):
            st = det.process(img, ts)
            ts += 500
        pillars = list(getattr(st, "pillars", []) or [])
        detected = len(pillars)
        for p in pillars:
            rect = [int(p.cx - p.w // 2), int(p.cy - p.h // 2), int(p.w), int(p.h)]
            crop = normalize_crop(img, rect, DEFAULT_BASE_WIDTH)
            cands.append(Cand(rect=rect, tpl=crop,
                              ev={"source": "生产检测器检出", "color": p.color,
                                  "stable": bool(getattr(p, "stable", False))}))
    except Exception as exc:  # noqa: BLE001
        detected = -1
        cands = []

    if not cands:
        for rect in _scan_tall(img):
            crop = normalize_crop(img, rect, DEFAULT_BASE_WIDTH)
            cands.append(Cand(rect=rect, tpl=crop, ev={"source": "尺寸搜索（检测器未检出）"}))
    if not cands:
        return CropResult(intent, reason="这张图里既没检出光柱，也找不到像光柱的竖长亮块")

    # 逐块量尺寸/颜色，和生产阈值比一次
    for c in cands:
        m = measure_region(img, c.rect, cfg.vision)
        problems = judge_object(m, cfg.vision)
        c.ev.update({"measure": m, "problems": problems,
                     "ok": not problems,
                     "hit_min": m["aspect"], "margin": None})
        c.self_score = 1.0
        c.second = 0.0

    ok = [c for c in cands if c.ev.get("ok")]
    cands.sort(key=lambda c: (1 if c.ev.get("ok") else 0,
                              float(c.ev["measure"]["color_fill"]),
                              -abs(float(c.ev["measure"]["aspect"]) - 4.8)),
               reverse=True)
    best = cands[0]
    res = CropResult(intent, best=best, alts=cands[1:4], verified=detected > 0)
    if detected > 0:
        res.object_note = (f"生产检测器在这张图上检出 {detected} 束；"
                           f"标定样本取的是最典型的这一束")
    elif detected == 0:
        res.object_note = ("生产检测器在这张图上一束都没检出 → 说明阈值比画面严，"
                           "下面列出最像光柱的几块，按它们的实际尺寸调 plateau 阈值")
    else:
        res.object_note = "检测器跑不起来（见日志），只给了尺寸搜索结果"
    res.reason = res.object_note + ("" if ok else
                                    "；注意：选中的这块按现判据也检出不了，见下面的差距")
    return res


# ---------------- 第四步：落盘 ----------------

def template_name(intent: Intent, index: int) -> str:
    """模板文件名：人指定 > 标题里的 ASCII > 按角色序号。

    文件名必须 ASCII —— cv2.imwrite 在 Windows 上写不了中文路径。所以不能直接
    用标题。想让它有意义就写 `- 模板名: skillwheel`。
    """
    if intent.tpl_name:
        stem = re.sub(r"[^0-9A-Za-z_\-]+", "_", intent.tpl_name).strip("_") or "tpl"
        return f"{ROLE_PREFIX[intent.role]}{stem}.png"
    ascii_words = re.findall(r"[0-9A-Za-z]{2,}", intent.title)
    stem = "_".join(w.lower() for w in ascii_words[:2])
    if stem:
        return f"{ROLE_PREFIX[intent.role]}{stem}.png"
    return f"{ROLE_PREFIX[intent.role]}{index}.png"


def _index_path(templates_dir: Path) -> Path:
    return templates_dir / "_index.json"


def read_index(templates_dir: Path) -> dict:
    p = _index_path(templates_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return {}


def _same_as_existing(templates_dir: Path, tpl: np.ndarray, role: str) -> str | None:
    """已有同角色模板和它几乎一样 → 复用旧名，不再堆一份新文件。

    人会反复改手册、反复重跑。每次多一张 99% 相同的 gate_pos_2.png，过两天就
    分不清哪张是当前那一版，而门禁"任一命中即放行"，重复的那几张只是白白多跑
    几次匹配。内容相同就复用文件名，_index.json 里的来源照旧更新到最新。
    """
    if not templates_dir.exists() or tpl is None or tpl.size == 0:
        return None
    rb = tpl.shape[1] / max(1, tpl.shape[0])
    for p in sorted(templates_dir.glob(ROLE_PREFIX[role] + "*.png")):
        old = cv2.imread(str(p))
        if old is None or min(old.shape[:2]) < 8:
            continue
        ra = old.shape[1] / max(1, old.shape[0])
        if max(ra, rb) / max(1e-6, min(ra, rb)) > 1.15:
            continue                      # 宽高比差这么多，必然不是同一块
        h = min(old.shape[0], tpl.shape[0])
        w = min(old.shape[1], tpl.shape[1])
        a = cv2.resize(cv2.cvtColor(old, cv2.COLOR_BGR2GRAY), (w, h),
                       interpolation=cv2.INTER_AREA)
        b = cv2.resize(cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY), (w, h),
                       interpolation=cv2.INTER_AREA)
        if float(a.std()) < 8.0 or float(b.std()) < 8.0:
            continue
        corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        if corr >= 0.97:
            return p.name
    return None


def save_template(prof: Any, intent: Intent, cand: Cand, name: str) -> Path:
    """落盘模板 + 登记来源。_index.json 记的"哪张图哪块裁的"是日后能重裁的唯一线索。"""
    tdir = prof.templates_dir
    tdir.mkdir(parents=True, exist_ok=True)
    out = tdir / name
    ok, buf = cv2.imencode(".png", cand.tpl)
    if not ok:
        raise RuntimeError(f"模板编码失败: {name}")
    buf.tofile(str(out))       # tofile 而不是 imwrite：后者对非 ASCII 路径会静默失败
    idx = read_index(tdir)
    # 指针指向不存在文件的条目要清掉：改过模板名或删过模板之后，这些孤儿记录
    # 会让 "这张模板是哪张图裁的" 变成一道谜题。
    for stale in [k for k in idx if not (tdir / k).exists()]:
        idx.pop(stale, None)
    idx[name] = {
        "base_width": DEFAULT_BASE_WIDTH,
        "source": Path(intent.image_rel).name if intent.image_rel else "",
        "rect": list(cand.rect),
        "note": f"{intent.title}｜由 tools/teach_crop.py 自动量得",
    }
    _index_path(tdir).write_text(json.dumps(idx, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    return out


def roi_advice(suggest: list[int], current: list[int] | None) -> str:
    """把"建议值 vs 现在值"翻成一句能直接照做的话。

    不这么说的话，报告里永远是"差别较大就更新"——多大算大？人只能猜，猜的结果
    通常是"算了不改"，然后真出问题时没人记得这里有过提示。
    """
    if not current:
        return ("profile.yaml 里现在没写 roi。留空会退回「底部 35% 全宽」的 PC 假设，"
                "手机端 UI 在右下角时对不上，务必加上")
    ix = max(0, min(suggest[0] + suggest[2], current[0] + current[2]) - max(suggest[0], current[0]))
    iy = max(0, min(suggest[1] + suggest[3], current[1] + current[3]) - max(suggest[1], current[1]))
    inter = ix * iy
    union = suggest[2] * suggest[3] + current[2] * current[3] - inter
    iou = inter / union if union > 0 else 0.0
    if iou >= 0.85:
        return "和算出来的一致，不用改"
    if iou >= 0.5:
        return ("大体重合，可以不改。只有当战斗帧里正向模板分数莫名偏低"
                "（说明锚点有一部分被 ROI 裁掉了）再换成建议值")
    return "差别很大，建议换成建议值——现在的 ROI 很可能把锚点裁掉了一部分"


def suggest_roi(results: list[CropResult], prof: Any) -> list[int] | None:
    """从正向模板的位置反推 scene_gate.roi。

    这一步是把用户踩过的坑自动化掉：正向模板的匹配 ROI 如果留默认（底部 35% 全宽，
    PC ARPG 的假设），手机端技能盘在右下角就会被裁掉一半，分数从 0.99 掉到 0.3，
    看起来像"模板不管用"。ROI 本来就该由"这块 UI 在哪"决定，而现在这个信息就在
    刚刚量出来的框里。
    """
    ref_w, ref_h = (list(prof.reference_resolution) + [1280, 720])[:2]
    boxes: list[list[int]] = []
    for r in results:
        if r.intent.role != ROLE_POS or r.best is None:
            continue
        try:
            img = cv2.imread(str(r.intent.image))
        except Exception:  # noqa: BLE001
            img = None
        if img is None:
            continue
        fx = ref_w / float(img.shape[1])
        fy = ref_h / float(img.shape[0])
        x, y, w, h = r.best.rect
        boxes.append([int(x * fx), int(y * fy), int(w * fx), int(h * fy)])
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    pad_x = max(6, int((x1 - x0) * 0.10))
    pad_y = max(6, int((y1 - y0) * 0.10))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(ref_w, x1 + pad_x)
    y1 = min(ref_h, y1 + pad_y)
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


def read_crops(out_dir: Path) -> dict:
    p = out_dir / "crops.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return {}


def write_crops(prof: Any, results: list[CropResult], out_dir: Path,
                templates: dict[str, str]) -> Path:
    """人可改的那份清单。`locked: true` 的条目下次直接采信，不重新搜。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    old = read_crops(out_dir)
    old_by_id = {it.get("id"): it for it in (old.get("items") or []) if it.get("id")}
    items: list[dict] = []
    for r in results:
        prev = old_by_id.get(r.intent.id) or {}
        item: dict = {
            "id": r.intent.id,
            "title": r.intent.title,
            "role": r.intent.role,
            "source": f"{r.intent.source}:{r.intent.line}",
            "image": r.intent.image_rel,
            "where": r.intent.where,
            "template": templates.get(r.intent.id, ""),
        }
        if r.best is not None:
            b = r.best
            item["rect"] = list(b.rect)
            item["score"] = {k: b.ev.get(k) for k in
                             ("ok", "hit_min", "miss_max", "margin", "hit_thr", "miss_thr",
                              "n_hit", "n_miss")}
            item["unique"] = b.unique
            item["why"] = r.reason
            item["alts"] = [{"rect": list(a.rect),
                             "hit_min": a.ev.get("hit_min"),
                             "miss_max": a.ev.get("miss_max")} for a in r.alts]
        else:
            item["gap"] = r.reason
        # 人指定过的条目：原样保留 rect 与标记，不再让算法改回去
        if r.locked or prev.get("locked"):
            item["locked"] = True
            if r.locked and r.best is not None:
                item["rect"] = list(r.best.rect)
            else:
                item["rect"] = prev.get("rect", item.get("rect"))
            item["source_note"] = "人工指定（locked=true，算法不再覆盖）"
        items.append(item)

    data = {
        "version": 1,
        "profile": prof.id,
        "updated": time.strftime("%Y-%m-%d %H:%M"),
        "howto": [
            "这是自动量出来的框，可以直接改。rect 用 [x,y,w,h]（源图像素，左上角原点），"
            "和手册里 `- 框:` 同一个口径。",
            "改完把该条目的 \"locked\" 设成 true，再跑一次 tools/teach_crop.py，"
            "就会按你给的框重裁，不再自动搜。",
            "（这个文件是生成物，不进版本库。要长期保留某个框，用 --write-back 写回手册的"
            " `- 框:`，或者直接在手册里写死。）",
        ],
        "roi_suggest": suggest_roi(results, prof),
        "items": items,
    }
    p = out_dir / "crops.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return p


# ---------------- 预览（人一眼过目） ----------------

def _b64_img(arr: np.ndarray, *, fmt: str = ".jpg", quality: int = 72,
             max_w: int = 760, upscale: int = 1) -> str:
    """编码成 data URI。缩到 760 宽是给眼睛看的分辨率，再大只是浪费（图会内嵌）。"""
    if arr is None or arr.size == 0:
        return ""
    h, w = arr.shape[:2]
    if w > max_w:
        f = max_w / float(w)
        arr = cv2.resize(arr, (max_w, max(1, int(round(h * f)))),
                         interpolation=cv2.INTER_AREA)
    elif upscale > 1 and w * upscale <= 640:
        arr = cv2.resize(arr, (w * upscale, h * upscale), interpolation=cv2.INTER_NEAREST)
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt == ".jpg" else \
             [cv2.IMWRITE_PNG_COMPRESSION, 6]
    ok, buf = cv2.imencode(fmt, arr, params)
    if not ok:
        return ""
    mime = "jpeg" if fmt == ".jpg" else "png"
    return f"data:image/{mime};base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _pct(v: float) -> str:
    return f"{max(0.0, min(100.0, v * 100)):.2f}%"


def _box_style(rect: list[int], shape: tuple[int, int]) -> str:
    H, W = shape
    x, y, w, h = rect
    return (f"left:{_pct(x / W)};top:{_pct(y / H)};"
            f"width:{_pct(w / W)};height:{_pct(h / H)}")


def write_preview(prof: Any, report: CropReport, out_dir: Path,
                  templates: dict[str, str]) -> Path:
    """一张页面，把"我选了什么、凭什么"摆在一起给人过目。

    用 HTML 而不是拼一张 PNG：中文标注不用折腾字体，框能随浏览器缩放，单张图
    不用内嵌整帧原图（只嵌缩略图）。这个文件可以单独发出去，不依赖仓库。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    for r in report.results:
        intent = r.intent
        role = ROLE_LABEL.get(intent.role, intent.role)
        if r.best is None:
            cards.append(f"""
    <section class="card gap">
      <div class="head"><h2>{html.escape(intent.title)}</h2>
        <span class="tag tag-gap">{role}</span></div>
      <p class="why">{html.escape(r.reason or '没做出来')}</p>
      <p class="meta">出处 {html.escape(intent.source)}:{intent.line}</p>
    </section>""")
            continue

        img = cv2.imread(str(intent.image)) if intent.image else None
        shot = _b64_img(img) if img is not None else ""
        shape = (img.shape[0], img.shape[1]) if img is not None else (1, 1)
        boxes = (f'<i class="box best" style="{_box_style(r.best.rect, shape)}">'
                 f'<b>选中</b></i>')
        for a in r.alts[:2]:
            boxes += f'<i class="box alt" style="{_box_style(a.rect, shape)}"></i>'
        tpl_uri = _b64_img(r.best.tpl, fmt=".png", upscale=2)
        ev = r.best.ev
        status = "ok" if ev.get("ok") else "warn"
        status_text = "可用" if ev.get("ok") else ("样本不足" if not r.verified else "不达标")
        if intent.role == ROLE_OBJ:
            m = ev.get("measure") or {}
            rows = [
                ("尺寸", f"{m.get('w')}x{m.get('h')}px"),
                ("高宽比", str(m.get("aspect"))),
                ("主色", f"{m.get('color') or '未知'} {m.get('color_fill', 0):.0%}"),
                ("亮部", f"{m.get('bright', 0):.0%}"),
            ]
            problems = ev.get("problems") or []
            extra = ("" if not problems else
                     '<p class="bad">按现判据检不出：' +
                     html.escape("；".join(problems)) + "</p>")
        else:
            rows = [
                ("在应命中的图上", f"最低 {ev.get('hit_min')}（阈值 {ev.get('hit_thr')}）"),
                ("在不应命中的图上", f"最高 {ev.get('miss_max')}（阈值 {ev.get('miss_thr')}）"),
                ("唯一性", f"{r.best.unique:.2f}（越高越独特）"),
                ("样本", f"{ev.get('n_hit')} 张应命中 / {ev.get('n_miss')} 张不应命中"),
            ]
            extra = ""
        table = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"
                        for k, v in rows)
        tpl_name = templates.get(intent.id, "")
        cards.append(f"""
    <section class="card">
      <div class="head"><h2>{html.escape(intent.title)}</h2>
        <span class="tag tag-{intent.role}">{role}</span>
        <span class="pill pill-{status}">{status_text}</span></div>
      <div class="body">
        <div class="shot">{f'<img src="{shot}" alt="">' if shot else ''}{boxes}</div>
        <aside class="panel">
          <table class="kv">{table}</table>
          {extra}
          <p class="why">{html.escape(r.reason)}</p>
          <div class="tpl"><span>裁出的模板</span>
            <img src="{tpl_uri}" alt=""></div>
          <p class="meta">{html.escape(tpl_name)}
            ｜源 {html.escape(Path(intent.image_rel).name)}
            ｜{html.escape(intent.source)}:{intent.line}</p>
        </aside>
      </div>
    </section>""")

    roi = report.roi_suggest
    roi_note = ""
    if roi:
        roi_note = (
            "<p>正向模板的匹配 ROI 建议 "
            f"<code>[{', '.join(str(v) for v in roi)}]</code>"
            f"（基准分辨率 {prof.reference_resolution[0]}x{prof.reference_resolution[1]}）"
            f"　—　<b>{html.escape(roi_advice(roi, report.roi_current))}</b></p>")

    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>模板候选 · {html.escape(report.profile)}</title>
<style>
  :root{{color-scheme:dark}}
  *{{box-sizing:border-box}}
  body{{margin:0;padding:28px 32px 60px;background:#0e1016;color:#e6e9ef;
       font:14px/1.65 "Microsoft YaHei",system-ui,-apple-system,sans-serif}}
  h1{{font-size:20px;margin:0 0 6px}}
  .lead{{color:#98a2b3;margin:0 0 22px;max-width:min(100%,980px)}}
  .lead b{{color:#e6e9ef;font-weight:600}}
  code{{background:#1b202b;padding:1px 6px;border-radius:4px;font-size:12.5px;
        color:#8ecdf7}}
  .card{{background:#151922;border:1px solid #232937;border-radius:12px;
         padding:16px 18px;margin:0 0 18px}}
  .card.gap{{border-color:#4a3a1c;background:#1a1710}}
  .head{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px}}
  .head h2{{font-size:15px;margin:0;font-weight:600}}
  .tag{{font-size:12px;padding:2px 8px;border-radius:99px;border:1px solid #2f3748;
        color:#a7b2c4}}
  .tag-gate_pos{{color:#7fd1a0;border-color:#255c40;background:#12241b}}
  .tag-gate_neg{{color:#f0b429;border-color:#5c4a1c;background:#221c10}}
  .tag-object{{color:#8ecdf7;border-color:#264a63;background:#101f29}}
  .tag-gap{{color:#f0846a;border-color:#5c2a1c;background:#221410}}
  .pill{{margin-left:auto;font-size:12px;padding:2px 10px;border-radius:99px}}
  .pill-ok{{background:#173d29;color:#7fd1a0}}
  .pill-warn{{background:#3d3117;color:#f0b429}}
  .body{{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:18px}}
  @media(max-width:900px){{.body{{grid-template-columns:1fr}}}}
  .shot{{position:relative;line-height:0;border-radius:8px;overflow:hidden;
         border:1px solid #232937}}
  .shot img{{width:100%;display:block}}
  .box{{position:absolute;border:2px solid #ff5f57;box-shadow:0 0 0 1px #0006}}
  .box b{{position:absolute;left:-2px;top:-2px;transform:translateY(-100%);
          background:#ff5f57;color:#fff;font-size:11px;padding:0 6px;
          border-radius:3px 3px 0 0;line-height:18px;font-weight:500}}
  .box.alt{{border:1px dashed #f0b429;opacity:.75}}
  .panel{{min-width:0}}
  table.kv{{width:100%;border-collapse:collapse;margin-bottom:10px}}
  table.kv th{{text-align:left;font-weight:400;color:#8b96a8;font-size:12.5px;
               padding:3px 0;width:46%}}
  table.kv td{{padding:3px 0;font-variant-numeric:tabular-nums}}
  .why{{color:#98a2b3;font-size:12.5px;margin:8px 0}}
  .bad{{color:#f0846a;font-size:12.5px;background:#221410;border-left:2px solid #f0846a;
        padding:6px 10px;border-radius:0 4px 4px 0}}
  .tpl{{margin-top:12px;border-top:1px dashed #2a3141;padding-top:10px}}
  .tpl span{{display:block;color:#6d7789;font-size:12px;margin-bottom:6px}}
  .tpl img{{max-width:100%;max-height:170px;image-rendering:pixelated;
            background:#0b0d12;border:1px solid #232937;border-radius:6px;padding:4px}}
  .meta{{color:#5f6a7d;font-size:11.5px;margin:10px 0 0}}
  .foot{{color:#6d7789;font-size:12.5px;border-top:1px solid #232937;padding-top:16px;
         margin-top:26px}}
</style></head>
<body>
  <h1>模板候选 · {html.escape(report.profile)}</h1>
  <p class="lead">
    每张卡片是手册里的一个界面段。<b>红框</b>是这次选中的锚点，黄虚线是备选。
    右边是裁出来的模板和它的分数。
    <br>看着不对就改 <code>teach/crops.json</code> 里的 <code>rect</code>，
    把那条的 <code>locked</code> 设成 <code>true</code>，再跑一次即可。
  </p>
  {roi_note}
  {''.join(cards) if cards else '<p class="lead">没有可处理的界面段。</p>'}
  <p class="foot">由 tools/teach_crop.py 生成 · {time.strftime('%Y-%m-%d %H:%M')} ·
    耗时 {report.elapsed:.1f}s</p>
</body></html>"""
    p = out_dir / "preview.html"
    p.write_text(page, encoding="utf-8")
    return p


# ---------------- 报告 ----------------

def write_report(prof: Any, report: CropReport, out_dir: Path,
                 templates: dict[str, str]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# 模板编译报告 · {prof.id}",
        "",
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M')}　耗时 {report.elapsed:.1f}s",
        "",
        "## 结论",
        "",
        "| 界面 | 角色 | 结果 | 模板 | 一句话依据 |",
        "|---|---|---|---|---|",
    ]
    mark = {"ok": "✅ 可用", "warn": "⚠️ 待确认", "gap": "❌ 没做出来"}
    for r in report.results:
        if r.best is None:
            lines.append(f"| {r.intent.title} | {ROLE_LABEL.get(r.intent.role, '')} "
                         f"| {mark['gap']} | — | {r.reason} |")
            continue
        st = "ok" if r.best.ev.get("ok") else "warn"
        lines.append(f"| {r.intent.title} | {ROLE_LABEL.get(r.intent.role, '')} "
                     f"| {mark[st]} | `{templates.get(r.intent.id, '')}` | {r.reason} |")

    if report.roi_suggest:
        cur = report.roi_current
        lines += ["", "## 建议的匹配 ROI", "",
                  f"`vision.scene_gate.roi: [{', '.join(str(v) for v in report.roi_suggest)}]`"
                  f"　（基准分辨率 {prof.reference_resolution[0]}x{prof.reference_resolution[1]}）",
                  ""]
        if cur:
            lines.append(f"现在写的是 `[{', '.join(str(v) for v in cur)}]`。"
                         f"**{roi_advice(report.roi_suggest, cur)}**")
        else:
            lines.append(f"**{roi_advice(report.roi_suggest, None)}**")

    nodata = [r for r in report.results if r.best is not None and not r.verified]
    if nodata:
        lines += ["", "## 样本不足，这些还没法签字", ""]
        for r in nodata:
            lines.append(f"- **{r.intent.title}**：{r.reason}")

    if report.gaps:
        lines += ["", "## 还缺什么（按这个顺序补）", ""]
        for r in report.gaps:
            lines.append(f"- **{r.intent.title}**（{r.intent.source}:{r.intent.line}）："
                         f"{r.reason}")

    if report.warnings:
        lines += ["", "## 提醒", ""] + [f"- {w}" for w in report.warnings]

    lines += ["", "---", "",
              f"看预览：`{out_dir / 'preview.html'}`",
              "改框：编辑 `teach/crops.json` 的 rect 并置 `\"locked\": true`，然后重跑",
              f"重跑：`py -3.12 tools/teach_crop.py --profile {prof.id}`",
              f"回填手册：`py -3.12 tools/teach_crop.py --profile {prof.id} --write-back`",
              f"之后再编译一次教学台账：`py -3.12 tools/teach.py --profile {prof.id}`"]
    p = out_dir / "report.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


# ---------------- 回填手册（可选，默认不动人的字） ----------------

def write_back(prof: Any, lessons: list[Lesson], report: CropReport) -> list[str]:
    """把量到的框写回手册里空着的位置。人已经写了框的，一个字不动。

    只替换「空 / 待补充 / 待定」，插在段尾（若原本没有 `- 框:` 行）。同一文件里
    多个段从后往前改，避免行号错位。
    """
    by_name = {ls.path.name: ls.path for ls in lessons}
    pending: dict[Path, list[tuple[int, list[int], str]]] = {}
    for r in report.results:
        if r.best is None:
            continue
        path = by_name.get(r.intent.source)
        if path is None:
            continue
        pending.setdefault(path, []).append(
            (r.intent.line, list(r.best.rect), r.intent.title))

    touched: list[str] = []
    for path, jobs in pending.items():
        lines = path.read_text(encoding="utf-8").splitlines()
        jobs.sort(key=lambda j: -j[0])
        changed = False
        for line_no, rect, _title in jobs:
            start = line_no - 1
            if start < 0 or start >= len(lines) or not lines[start].lstrip().startswith("## "):
                continue
            end = start + 1
            while end < len(lines) and not lines[end].lstrip().startswith("## "):
                end += 1
            hit = -1
            for i in range(start + 1, end):
                if re.match(r"^\s*[-*+]\s*(框|框选|区域|范围|rect|roi)\s*[:：]", lines[i]):
                    hit = i
                    break
            new_line = f"- 框: {rect[0]},{rect[1]},{rect[2]},{rect[3]}"
            if hit >= 0:
                value = lines[hit].split("：", 1)[-1] if "：" in lines[hit] else \
                        lines[hit].split(":", 1)[-1]
                if _looks_like_placeholder(value):
                    lines[hit] = new_line
                    changed = True
                # 人写了值 → 跳过
            else:
                insert_at = end
                while insert_at - 1 > start and not lines[insert_at - 1].strip():
                    insert_at -= 1
                lines.insert(insert_at, new_line)
                changed = True
        if changed:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            touched.append(path.name)
    return touched


# ---------------- 编排 ----------------

def _locked_rect(out_dir: Path, intent_id: str) -> list[int] | None:
    for it in read_crops(out_dir).get("items") or []:
        if it.get("id") == intent_id and it.get("locked") and it.get("rect"):
            try:
                return [int(v) for v in it["rect"]][:4]
            except Exception:  # noqa: BLE001
                return None
    return None


def _parse_set(values: list[str]) -> dict[str, list[int]]:
    """`--set "标题或id=0,0,300,80"` → {key: rect}。多个用多次 --set。"""
    out: dict[str, list[int]] = {}
    for raw in values or []:
        if "=" not in raw:
            raise SystemExit(f"--set 要写成 『标题或id=x,y,w,h』，收到: {raw}")
        key, val = raw.rsplit("=", 1)
        rect = parse_rect(val)
        if rect is None:
            raise SystemExit(f"--set 的框要 4 个数字（x,y,w,h），收到: {val}")
        out[key.strip()] = rect
    return out


def run(prof: Any, cfg: Any, lessons: list[Lesson], *,
        only: str = "", write: bool = True,
        do_write_back: bool = False,
        forced: dict[str, list[int]] | None = None) -> CropReport:
    """一条命令走完：文档 → 框 → 模板 → 分数 → 预览。

    `forced` 是人在命令行上直接指定的框（`--set "标题=x,y,w,h"`）。走这条路时
    照样跑验证、照样出预览，只是不再搜索——人的判断永远压过算法，但人也要看到
    自己的判断在数据上是什么样。
    """
    forced = forced or {}
    t0 = time.time()
    out_dir = prof.root / "teach" / "crops"
    report = CropReport(profile=prof.id)

    intents = plan_intents(prof, lessons)
    if only:
        key = only.lower()
        intents = [i for i in intents if key in i.title.lower() or key in i.id.lower()]
    if not intents:
        report.warnings.append("手册里没有可处理的「界面」或「物体」段")
        report.elapsed = time.time() - t0
        return report

    pool = load_pool(lessons, prof)
    n_pos = sum(1 for i in pool.values() if i.cls is True)
    n_neg = sum(1 for i in pool.values() if i.cls is False)
    if n_pos < 2:
        report.warnings.append(f"应命中的样本只有 {n_pos} 张，跨图验证会失效——"
                               f"再截一张同类界面（换个位置/换个房间）能显著提高可靠性")
    if n_neg < 1:
        report.warnings.append("一张反例都没有：所有模板都没法验证『会不会误命中』。"
                               "截一张关卡选择/背包/结算丢进 refs/ 并写进手册即可")

    templates: dict[str, str] = {}
    for idx, intent in enumerate(intents, 1):
        # 优先级：命令行 --set > crops.json 里人标过 locked > 手册里的 `- 框:`
        # > 自动搜。只有前两种会在 crops.json 里留 locked，手册的框不算——手册
        # 常年会改，锁死了反而让人以为改了没生效。
        hand = forced.get(intent.id) or forced.get(intent.title)
        if not hand:
            # 人懒得打全名："--set 星图=..." 也该认
            hand = next((v for k, v in forced.items()
                         if k and (k in intent.title or intent.title in k)), None)
        pinned = None
        if not hand:
            pinned = _locked_rect(out_dir, intent.id)
        if hand:
            intent.given_rect = hand
        elif pinned and not intent.given_rect:
            intent.given_rect = pinned
        res = search(intent, pool, cfg, tpl_dir=prof.templates_dir)
        res.locked = bool(hand) or bool(pinned and intent.given_rect == pinned)
        report.results.append(res)
        if res.best is None:
            continue
        name = template_name(intent, idx)
        same = _same_as_existing(prof.templates_dir, res.best.tpl, intent.role)
        if same:
            # 和已有模板内容一致 → 复用旧名，不新增文件
            name = same
        templates[intent.id] = name
        if write:
            save_template(prof, intent, res.best, name)
            report.written.append(name)

    report.roi_suggest = suggest_roi(report.results, prof)
    report.roi_current = list(getattr(cfg.vision.scene_gate, "roi", None) or []) or None
    report.templates = templates
    report.elapsed = time.time() - t0

    if write:
        write_report(prof, report, out_dir, templates)
        write_preview(prof, report, out_dir, templates)
        write_crops(prof, report.results, out_dir, templates)
    if do_write_back and write:
        names = write_back(prof, lessons, report)
        if names:
            report.written.extend(f"{n}（已回填 - 框:）" for n in names)

    return report
