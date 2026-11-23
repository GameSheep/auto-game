"""判据：画面里那点"现在该不该按"的证据，从截图里标定出来，写成 YAML。

## 为什么单独一层

技能能写出来，不代表能跑。`use_skill` 缺的从来不是"按哪个键"（方案里有），而是
**什么时候按**：蓝够不够、技能在不在 CD、脚下有没有可交互的东西、面板是不是开着的。
这些东西在 `SceneState` 里没有字段，因为它们每款游戏都不一样 —— 硬塞进协议，
协议就变成了某一款游戏的形状。

所以另立一层：`cues.yaml` 里一条 cue = 一块屏幕区域 + 一种读法 + 一个阈值。

    cues:
      - id: mana_ok
        kind: ratio                 # 这块区域里"像这个颜色"的像素占比
        region: [0.30, 0.94, 0.42, 0.97]
        color: "#2a6bd4"
        above: 0.25                 # 占比高于 → 成立

读出来的是 0..1 的数，`above` / `below` 负责变成"成立/不成立"。技能里写
`when: cue` + `cue: mana_ok` 就能用，**不用改代码**。

## 三条设计约束

1. **区域一律归一化**（0..1），跟坐标归一化同一条规矩：换分辨率不用重标。
2. **读不出来就是 0，不抛异常**。模板缺文件、区域越界、帧比区域还小 —— 都记进
   `last_errors` 并返回 0。理由和 `probe()` 默认返回失败一样：新写的 cue 要能被
   拦下，而不是让程序"看起来在判读"。
3. **阈值不猜**。没写 `above` 也没写 `below` 的 cue 是写错了（lint 报 error），
   不是"永远成立"。默认值会让标错的东西静默生效 —— 那比不生效更难查。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, Field, PrivateAttr, field_validator

#: 颜色名 → BGR。写死几个常用的，省得为了 `white` 去查 hex。
NAMED_BGR: dict[str, tuple[int, int, int]] = {
    "white": (255, 255, 255), "black": (0, 0, 0),
    "red": (0, 0, 255), "green": (0, 200, 0), "blue": (255, 0, 0),
    "cyan": (255, 255, 0), "yellow": (0, 255, 255), "gold": (0, 200, 255),
    "magenta": (255, 0, 255), "orange": (0, 140, 255),
}


def parse_color(raw: str) -> tuple[int, int, int] | None:
    """`#2a6bd4` / `2a6bd4` / `white` → BGR。解析不了返回 None（lint 报错）。"""
    s = str(raw or "").strip().lower()
    if not s:
        return None
    if s in NAMED_BGR:
        return NAMED_BGR[s]
    s = s.lstrip("#")
    if len(s) != 6:
        return None
    try:
        r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except ValueError:
        return None
    return (b, g, r)


#: 模板缓存：key = (绝对路径, mtime) → 图。不做 lru_cache：
#: ndarray 可变，缓存住被调用方改过的图会出"改了文件不生效"的怪事。
_TEMPLATES: dict[tuple[str, float], np.ndarray] = {}
_TEMPLATE_MAX = 32


def _load_template(path: Path) -> np.ndarray | None:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    key = (str(path), mtime)
    hit = _TEMPLATES.get(key)
    if hit is not None:
        return hit
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    if len(_TEMPLATES) >= _TEMPLATE_MAX:
        _TEMPLATES.clear()
    _TEMPLATES[key] = img
    return img


class Cue(BaseModel):
    """一条判据。"""

    id: str
    name: str = ""
    #: ratio  = 区域里接近 `color` 的像素占比（血条/蓝条/光柱/按钮染色）
    #: bright = 区域平均亮度（全黑屏幕、加载遮罩、面板变暗）
    #: template = 模板匹配最高分（图标、按钮、提示文字位）
    kind: Literal["ratio", "bright", "template"] = "ratio"
    #: 归一化矩形 [x1, y1, x2, y2]，0..1
    region: list[float] = Field(default_factory=list)
    color: str = "#ffffff"
    tol: int = Field(40, ge=1, le=255)
    above: float | None = None
    below: float | None = None
    template: str = ""
    min_score: float = Field(0.72, ge=0.0, le=1.0)
    note: str = ""

    @field_validator("id", mode="before")
    @classmethod
    def _id(cls, v: Any) -> str:
        s = str(v or "").strip()
        if not s:
            raise ValueError("cue 必须有 id（技能里按名字引用它）")
        return s

    @field_validator("region", mode="before")
    @classmethod
    def _region(cls, v: Any) -> list[float]:
        if v is None:
            return []
        if isinstance(v, (list, tuple)):
            return [float(x) for x in v]
        return []

    def problem(self, base_dir: Path | None = None) -> list[str]:
        """这条 cue 写得对不对（lint 用）。返回人话问题列表。"""
        out: list[str] = []
        r = self.region
        if len(r) != 4:
            out.append("region 要 4 个数 [x1,y1,x2,y2]（归一化 0..1）")
        else:
            if any(not (0.0 <= v <= 1.0) for v in r):
                out.append("region 里有超出 0..1 的值 —— 这里用的是归一化坐标，不是像素")
            if r[2] <= r[0] or r[3] <= r[1]:
                out.append("region 的 x2/y2 必须大于 x1/y1")
        if self.above is None and self.below is None and self.kind != "template":
            out.append("above / below 至少写一个 —— 没有阈值的判据永远不成立")
        if self.above is not None and self.below is not None and self.above > self.below:
            out.append(f"above({self.above}) 大于 below({self.below})，区间是空的")
        if self.kind in ("ratio", "bright"):
            if parse_color(self.color) is None:
                out.append(f"color={self.color!r} 解析不了（写 #rrggbb 或 white/red/...）")
        if self.kind == "template":
            if not self.template:
                out.append("kind: template 必须给 template 路径（相对 profile 根目录）")
            elif base_dir is not None and not (Path(base_dir) / self.template).exists():
                out.append(f"模板图不存在：{Path(base_dir) / self.template}")
        return out

    def threshold(self) -> tuple[float | None, float | None]:
        """(下限, 上限)。模板匹配的下限就是 `min_score` —— 不再要求写第二遍 above。

        一个阈值只写一遍：写两遍的地方迟早会出现"min_score 改了、above 忘改"，
        那时判据按旧值成立，模板早就不匹配了。
        """
        lo, hi = self.above, self.below
        if self.kind == "template" and lo is None:
            lo = self.min_score
        return lo, hi

    def on(self, value: float) -> bool:
        """读数 → 成立与否。只看写了的那一边：写 above 就只管下限。"""
        lo, hi = self.threshold()
        if lo is None and hi is None:
            return False
        if lo is not None and value < lo - 1e-9:
            return False
        if hi is not None and value > hi + 1e-9:
            return False
        return True

    def describe(self) -> str:
        lo, hi = self.threshold()
        bits = []
        if lo is not None:
            bits.append(f"≥{lo:g}")
        if hi is not None:
            bits.append(f"≤{hi:g}")
        what = {"ratio": f"占比 {self.color}", "bright": "亮度",
                "template": f"匹配 {Path(self.template).name or '(未填)'}"}[self.kind]
        return f"{self.id}：{what} {' 且 '.join(bits)}"


class CueBook(BaseModel):
    """一个游戏的全部判据。"""

    cues: list[Cue] = Field(default_factory=list)
    source: str = ""

    _last_errors: list[str] = PrivateAttr(default_factory=list)

    # ------------------------------------------------------------ 加载

    @classmethod
    def load(cls, path: str | Path) -> "CueBook":
        p = Path(path)
        if p.is_dir():
            p = p / "cues.yaml"
        if not p.exists():
            return cls(source=str(p.parent))
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict):
            items = raw.get("cues") or []
        else:
            raise ValueError(f"{p} 应该是 cues: 列表")
        return cls(cues=items, source=str(p.parent))

    # ------------------------------------------------------------ 查询

    def get(self, cid: str) -> Cue | None:
        return next((c for c in self.cues if c.id == cid), None)

    def ids(self) -> set[str]:
        return {c.id for c in self.cues}

    def __bool__(self) -> bool:
        return bool(self.cues)

    def __len__(self) -> int:
        return len(self.cues)

    @property
    def last_errors(self) -> list[str]:
        return list(self._last_errors)

    # ------------------------------------------------------------ 判读

    def evaluate(self, frame: np.ndarray | None, base_dir: str | Path | None = None
                 ) -> dict[str, float]:
        """一帧 → {cue id: 读数 0..1}。

        没有帧、区域空、模板缺失 —— 都返回 0 并记进 `last_errors`。
        不抛异常的理由：这一层跑在决策循环里，一条写错的 cue 不该让整局停摆，
        但也不能静默变成"永远成立"。
        """
        self._last_errors = []
        out: dict[str, float] = {c.id: 0.0 for c in self.cues}
        if frame is None or not self.cues:
            if frame is None:
                self._last_errors.append("没有画面，判据全部按 0 处理")
            return out
        h, w = frame.shape[:2]
        for c in self.cues:
            try:
                out[c.id] = self._read(c, frame, h, w, base_dir)
            except Exception as e:  # noqa: BLE001  一条坏 cue 不能拖垮整个 tick
                out[c.id] = 0.0
                self._last_errors.append(f"{c.id}：判读失败（{e}）")
        return out

    def _roi(self, c: Cue, frame: np.ndarray, h: int, w: int):
        r = c.region
        if len(r) != 4:
            raise ValueError("region 不是 4 个数")
        x1 = max(0, min(w - 1, int(round(r[0] * w))))
        y1 = max(0, min(h - 1, int(round(r[1] * h))))
        x2 = max(x1 + 1, min(w, int(round(r[2] * w))))
        y2 = max(y1 + 1, min(h, int(round(r[3] * h))))
        return frame[y1:y2, x1:x2]

    def _read(self, c: Cue, frame: np.ndarray, h: int, w: int,
              base_dir: str | Path | None) -> float:
        roi = self._roi(c, frame, h, w)
        if c.kind == "bright":
            return float(roi.mean()) / 255.0
        if c.kind == "ratio":
            bgr = parse_color(c.color)
            if bgr is None:
                raise ValueError(f"颜色 {c.color!r} 解析不了")
            px = roi.reshape(-1, 3).astype(np.int16)
            diff = np.abs(px - np.array(bgr, dtype=np.int16))
            return float(np.all(diff <= c.tol, axis=1).mean())
        # template
        if not c.template:
            raise ValueError("没写 template")
        base = Path(base_dir) if base_dir else Path(".")
        tpl = _load_template(base / c.template)
        if tpl is None:
            raise ValueError(f"模板读不出来：{base / c.template}")
        th, tw = tpl.shape[:2]
        if roi.shape[0] < th or roi.shape[1] < tw:
            raise ValueError(f"区域({roi.shape[1]}x{roi.shape[0]})比模板({tw}x{th})小")
        import cv2

        res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
        return float(np.clip(res.max(), 0.0, 1.0))

    def on(self, values: dict[str, float]) -> set[str]:
        """读数 → 成立的 cue 名集合。"""
        return {c.id for c in self.cues if c.on(float(values.get(c.id, 0.0)))}

    # ------------------------------------------------------------ 报告

    def problems(self, base_dir: str | Path | None = None) -> list[dict[str, str]]:
        """写错的地方（lint 直接吃这个）。"""
        out: list[dict[str, str]] = []
        seen: set[str] = set()
        for c in self.cues:
            if c.id in seen:
                out.append({"id": c.id, "message": f"cue id 重复：{c.id}"})
            seen.add(c.id)
            for msg in c.problem(Path(base_dir) if base_dir else None):
                out.append({"id": c.id, "message": msg})
        return out

    def report(self, base_dir: str | Path | None = None) -> list[dict[str, Any]]:
        return [{
            "id": c.id, "name": c.name or c.id, "kind": c.kind,
            "region": list(c.region), "on_at_zero": c.on(0.0),
            "describe": c.describe(),
            "problem": c.problem(Path(base_dir) if base_dir else None),
        } for c in self.cues]

    def summary(self) -> str:
        if not self.cues:
            return "判据 0 条"
        kinds: dict[str, int] = {}
        for c in self.cues:
            kinds[c.kind] = kinds.get(c.kind, 0) + 1
        detail = "、".join(f"{k} {v}" for k, v in sorted(kinds.items()))
        return f"判据 {len(self.cues)} 条（{detail}）"


__all__ = ["Cue", "CueBook", "NAMED_BGR", "parse_color"]
