"""本地意图：一句话 → 一个动作，**不碰模型**。

## 谁会走这条路

停止、暂停、继续、只看、自动、撤销、确认、取消、选第几个 —— 这些都是"控制"，
不是"理解"。它们必须在一句话说完的时候就已经生效，不能等远程模型。

## 最要紧的一条：宁可不认，也不要认错

快通道的代价是**没有上下文**。同一个"停"字：
- "停" / "停下" / "先别动"     → 是命令
- "怎么停下来着？" / "这个怪为什么停在那" → 是在问问题
- "别停"                        → 是**反**命令

所以这里的判断不是"包含关键词"，而是三条同时成立：

1. **句子短**（去掉标点后 ≤ 12 字）。长句多半在讨论，不是在命令。
2. **整句就是命令**（可带少量语气词前缀，如"先/快/马上/帮我"），
   而不是"命令词 + 别的意思"。用"去掉前缀后剩下的部分必须落在同义词表里"来判。
3. **没有否定词**（"别/不要/不用/先别"）。有否定一律 `unknown`，交给模型 ——
   把"别停"当成"停"是这套系统里最危险的一次误判。

认不出就返回 `unknown`，调用方会去问模型。**慢一点可以，动错了不行。**
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

#: 意图种类。加新种类时**必须**同时在 brain 的工具表里给出处理方式，
#: 否则模型/页面会收到一个没人处理的 action。
STOP = "stop"
PAUSE = "pause"
RESUME = "resume"
OBSERVE = "observe"
AUTO = "auto"
CONFIRM = "confirm"
CANCEL = "cancel"
CHOOSE = "choose"
UNDO = "undo"
STATUS = "status"
UNKNOWN = "unknown"

#: 可以带时长的（"暂停十分钟"）
_HOUR = r"(?:小时|钟头|个点)"
_MIN = r"(?:分钟|分)"
_SEC = r"(?:秒钟|秒)"

_NUM_CN = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "半": 0.5}

#: 时长：阿拉伯数字或中文数字 + 单位。只认到"小时"，再长没有意义（那时候人早回来了）。
_DUR_RE = re.compile(
    rf"(\d+(?:\.\d+)?|[一二两三四五六七八九十半]+)\s*个?\s*({_HOUR}|{_MIN}|{_SEC})")

#: 命令前允许出现的语气/礼貌前缀。去掉它们之后，剩下的必须**整体**是同义词。
_PREFIXES = ("请", "帮我", "麻烦", "快", "赶快", "马上", "立刻", "立即", "先",
             "现在", "给我", "你", "那就", "就", "赶紧")

#: 否定词。出现任何一个 → 不算命令。
_NEGATIONS = ("别", "不要", "不用", "不必", "甭", "勿", "not")

#: 同义词表。**长的在前**：先匹配"只看不操作"，否则会被"停"吃掉。
_SYNONYMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (STOP, ("急停", "停下", "停止", "停手", "住手", "停下来", "停止操作",
            "停止执行", "别动了", "停一下", "停", "stop", "halt")),
    (PAUSE, ("暂停", "暂停一下", "歇一会", "歇一会儿", "等一下", "pause")),
    (RESUME, ("继续", "恢复", "接着", "接着跑", "继续跑", "继续执行", "开始跑",
              "继续刷", "resume", "go on", "continue")),
    (OBSERVE, ("观察模式", "只看", "只观察", "只看不操作", "别操作", "不要操作",
               "看着就行", "只记录", "观察")),
    (AUTO, ("自动模式", "自动", "自动跑", "自己跑", "正常执行")),
    (CONFIRM, ("确认", "确定", "是的", "对", "好", "行", "可以", "应用", "就这么办",
               "同意", "没错", "ok", "yes", "y")),
    (CANCEL, ("取消", "算了", "不用了", "不要了", "否", "不行", "别改", "no", "n")),
    (UNDO, ("撤销", "撤回", "回退", "回退上一版", "退回上一版", "回到上一版",
            "恢复上一版", "undo", "revert")),
    (STATUS, ("什么情况", "现在什么情况", "状态", "在干什么", "在做什么",
              "现在怎么样", "status")),
)

#: "第 2 个" / "第二个" / "选 1"
_CHOOSE_RE = re.compile(r"^(?:选|要|第)?\s*(\d+|[一二三四五])\s*(?:个|条|项|种)?$")
_ORDINAL = re.compile(r"^第\s*(\d+|[一二三四五])\s*(?:个|条|项|种)$")

_PUNCT = "，。、！？；：“”‘’（）《》…—~!?,.;:\"'()<>[]{} \t\n\r"


@dataclass
class Intent:
    """一句本地命令的解析结果。"""

    kind: str
    text: str
    arg: Any = None          # 时长秒数 / 选项序号 / 说明
    why: str = ""            # 命中理由（页面和日志要能回答"它为什么这么做"）
    raw: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def is_command(self) -> bool:
        return self.kind != UNKNOWN

    def to_dict(self) -> dict:
        return {"kind": self.kind, "text": self.text, "arg": self.arg,
                "why": self.why, "raw": self.raw}


def normalize(text: str) -> str:
    """全角转半角、去标点、压空白。中文里"停 下"和"停下"是同一句话。"""
    s = unicodedata.normalize("NFKC", str(text or "")).strip().lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    return re.sub(r"\s+", "", s)


def _strip_prefix(s: str) -> str:
    changed = True
    while changed:
        changed = False
        for p in _PREFIXES:
            if s.startswith(p) and len(s) > len(p):
                s = s[len(p):]
                changed = True
                break
    return s


def _duration(text: str) -> float | None:
    """从整句里抠出时长（秒）。抠不出来返回 None。"""
    m = _DUR_RE.search(text)
    if not m:
        return None
    num, unit = m.group(1), m.group(2)
    if re.fullmatch(r"\d+(?:\.\d+)?", num):
        val = float(num)
    else:
        val = 0.0
        for ch in num:
            val += _NUM_CN.get(ch, 0)
        if num.startswith("十"):        # "十五" 之类
            val = _NUM_CN.get("十", 10) + sum(_NUM_CN.get(c, 0) for c in num[1:])
    if unit in ("小时", "钟头", "个点"):
        val *= 3600
    elif unit in ("分钟", "分"):
        val *= 60
    return val


def _clean_duration(text: str) -> str:
    """把时长短语从句子中去掉，剩下的才是"命令本体"（暂停十分钟 → 暂停）。"""
    return _DUR_RE.sub("", text)


def parse(text: str) -> Intent:
    """解析一句话。纯函数、无 I/O、毫秒级 —— 可以每句话都先过它。"""
    raw = str(text or "")
    s = normalize(raw)
    if not s:
        return Intent(UNKNOWN, "", raw=raw, why="空输入")

    # 0) 整句就是"取消/撤销/只看"：这几类词**本身带否定**（"不要了""别操作"），
    #    必须先于否定检查，否则会被自己的否定词拦掉 —— "别操作" 正是"只观察"。
    for kind in (CANCEL, UNDO, OBSERVE):
        for w in dict(_SYNONYMS)[kind]:
            if s == w:
                return Intent(kind, s, raw=raw, why=f"整句匹配「{w}」")

    # 1) 否定优先：有否定词就不猜（"别停" 不是 "停"）
    if any(n in s for n in _NEGATIONS):
        return Intent(UNKNOWN, s, raw=raw, why="含否定词，交给模型判断")

    # 2) 序号选择（"2" / "第二个" / "选 1"）—— 多轮提问时用
    m = _CHOOSE_RE.match(s) or _ORDINAL.match(s)
    if m:
        num = m.group(1)
        idx = int(num) if num.isdigit() else _NUM_CN.get(num, 0)
        if idx:
            return Intent(CHOOSE, s, arg=idx - 1, raw=raw,
                          why=f"选第 {idx} 个")

    body = _strip_prefix(_clean_duration(s))

    # 3) 整句必须**等于**某个同义词（长的先试），不做"包含"匹配
    for kind, words in _SYNONYMS:
        for w in words:
            if body == w:
                arg = None
                if kind == PAUSE:
                    arg = _duration(s)
                return Intent(kind, s, arg=arg, raw=raw,
                              why=f"整句匹配「{w}」")

    # 4) 只在**多字**同义词上允许"命令 + 极少量补充"。
    #    单字（好/行/对/停）不走这一步："好累"不是确认，"停下来了吗"不是停止。
    for kind, words in _SYNONYMS:
        for w in words:
            if len(w) >= 2 and body.startswith(w) and len(body) - len(w) <= 2:
                if kind is PAUSE and _duration(s):
                    return Intent(kind, s, arg=_duration(s), raw=raw,
                                  why=f"以「{w}」开头且无其它内容")

    if len(body) > 12:
        return Intent(UNKNOWN, s, raw=raw, why="句子太长，多半是在讨论而不是命令")

    # 5) 中文里"撤销/撤回"常放在句尾（"把刚才的修改撤销"）。只对撤销类放行，
    #    因为它是**不可逆方向**里唯一安全的：猜错最多白撤一次，模型还会再问。
    for w in ("撤销", "撤回", "回退"):
        if body.endswith(w) and len(body) - len(w) <= 8:
            return Intent(UNDO, s, raw=raw, why=f"句尾是「{w}」")

    return Intent(UNKNOWN, s, raw=raw, why="没有匹配到本地命令")


def is_short_command(text: str) -> bool:
    """给调用方的快捷判断：值不值得先走本地通道。"""
    return parse(text).is_command
