"""配方引擎 —— 交易行 / 处理装备 / 点天赋 / 换技能 的 UI 自动化执行。

设计：**配方 = 步骤列表**，每步 = 期望视觉验证 + 动作，逐帧验证推进：
  步骤类型：
    {"expect_panel": "面板锚点模板名"}            # 等待面板出现（模板匹配）
    {"click_slot": {"highlighted": true}}         # 点高亮槽位（可买/可学/选中）
    {"click_slot": {"index": 3}}                  # 点指定序号槽位
    {"click_row": {"index": 0}}                   # 点第 N 文本行（最便宜价=排序后首行）
    {"click_button": "buy.png"}                   # 点模板按钮（购买/装备/确认/卸下）
    {"key": "a", "hold_ms": 60}                   # 按键
    {"wait_ms": 500}                              # 等待
  每步带超时与重试；任何一步超时 → 任务失败并报告卡在哪一步。

预置配方（四场景，配置 ui_task.recipes 可改）：
  trade_buy   交易行购买：确认面板 → 点排序按钮 → 点首行(最低价) → 点购买 → 确认
  equip_best  处理装备：确认面板 → 点高亮槽位(最优) → 点装备按钮 → 确认
  talent_alloc 点天赋：确认面板 → 依次点高亮天赋节点 → 确认
  skill_swap  学技能/换技能：确认面板 → 点新技能槽 → 点目标技能槽 → 确认

模板放到 templates/ui/<面板锚点>/<按钮>.png（用户从自己的截图裁剪）。
"""

from __future__ import annotations

import contextlib

import logging
import time
from pathlib import Path
from typing import Any

import cv2

from ..protocol import Command
from ..vision.ui_panels import GridConfig, UiSlot, detect_grid_slots, detect_text_rows, match_button

log = logging.getLogger(__name__)

PRESET_RECIPES: dict[str, list[dict]] = {
    "trade_buy": [
        {"expect_panel": "trade"},
        {"click_button": "sort.png"},
        {"click_row": {"index": 0}},
        {"click_button": "buy.png"},
        {"click_button": "confirm.png"},
    ],
    "equip_best": [
        {"expect_panel": "equip"},
        {"click_slot": {"highlighted": True}},
        {"click_button": "equip.png"},
        {"click_button": "confirm.png"},
    ],
    "talent_alloc": [
        {"expect_panel": "talent"},
        {"click_slot": {"highlighted": True}},
        {"click_slot": {"highlighted": True, "next": True}},
        {"click_button": "confirm.png"},
    ],
    "skill_swap": [
        {"expect_panel": "skill"},
        {"click_slot": {"index": 0}},
        {"click_button": "learn.png"},
        {"click_slot": {"index": 1}},
        {"click_button": "confirm.png"},
    ],
}


def required_templates(steps: list[dict]) -> list[str]:
    """配方所需的模板文件名（面板锚点 + 按钮），按出现顺序去重。

    用途：任务启动前做预检 —— 缺模板时**快速失败**，而不是静默 skip 掉所有步骤
    假装"任务已启动"（历史 bug：实测 trade_buy 5 步全 skip，一条命令都不发）。
    """
    need: list[str] = []
    for step in steps:
        if "expect_panel" in step:
            need.append(f"panel_{step['expect_panel']}.png")
        if "click_button" in step:
            need.append(str(step["click_button"]))
    seen: set[str] = set()
    out: list[str] = []
    for n in need:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


class RecipeRunner:
    """配方执行器：逐帧推进步骤，每步带期望验证与超时。"""

    def __init__(self, name: str, steps: list[dict], gcfg: GridConfig,
                 templates_dir: Path | None,
                 step_timeout_ms: int = 6000, executor=None,  # noqa: ANN001
                 rows_roi: list[int] | None = None) -> None:
        self.name = name
        self.steps = steps
        self.gcfg = gcfg
        self.rows_roi = rows_roi or gcfg.roi
        self.templates_dir = templates_dir
        self.step_timeout_ms = step_timeout_ms
        self.executor = executor
        self.step_i = 0
        self.step_started_ms = 0
        self.done = False
        self.failed = ""
        self.log: list[dict] = []
        self._btn_cache: dict[str, np.ndarray | None] = {}

    # ------------- 模板 -------------

    def template_path(self, fname: str) -> Path | None:
        """模板文件的真实路径（找不到返回 None）。"""
        for p in [self.templates_dir / "ui" / fname if self.templates_dir else None,
                  self.templates_dir / fname if self.templates_dir else None,
                  Path(fname)]:
            if p and Path(p).exists():
                return Path(p)
        return None

    def missing_templates(self) -> list[str]:
        """预检：返回缺失的模板文件名列表（空=齐备）。"""
        return [n for n in required_templates(self.steps) if self.template_path(n) is None]

    def _button_template(self, fname: str) -> np.ndarray | None:
        if fname in self._btn_cache:
            return self._btn_cache[fname]
        p = self.template_path(fname)
        img = cv2.imread(str(p), cv2.IMREAD_COLOR) if p else None
        self._btn_cache[fname] = img
        return img

    def _panel_anchor(self, name: str) -> np.ndarray | None:
        return self._button_template(f"panel_{name}.png")

    # ------------- 执行 -------------

    def tick(self, frame, ts_ms: int) -> list[dict]:  # noqa: ANN001
        """每个决策 tick 推进一次；返回本 tick 发出的命令文本。"""
        if self.done:
            return []
        if self.step_i >= len(self.steps):
            self.done = True
            self.log.append({"t": ts_ms, "event": "done"})
            return []
        step = self.steps[self.step_i]
        if self.step_started_ms == 0:
            self.step_started_ms = ts_ms
            self.log.append({"t": ts_ms, "event": "step_start", "step": step})
        if ts_ms - self.step_started_ms > self.step_timeout_ms:
            self.failed = f"step {self.step_i} 超时: {step}"
            self.done = True
            self.log.append({"t": ts_ms, "event": "fail", "reason": self.failed})
            return []

        cmds: list[dict] = []
        advanced = False
        try:
            if "expect_panel" in step:
                anchor = self._panel_anchor(step["expect_panel"])
                if anchor is None:
                    # 无锚点模板 → 跳过验证（记录）
                    self.log.append({"t": ts_ms, "event": "skip_panel_check",
                                     "panel": step["expect_panel"]})
                    advanced = True
                else:
                    hit = match_button(frame, anchor, None, threshold=0.6)
                    advanced = hit is not None
            elif "click_slot" in step:
                flt = step["click_slot"]
                slots = detect_grid_slots(frame, self.gcfg)
                target = self._pick_slot(slots, flt)
                if target is not None:
                    cmds.append({"op": "mouse_click", "button": "left",
                                 "x": target.x, "y": target.y})
                    advanced = True
            elif "click_row" in step:
                rows = detect_text_rows(frame, self.rows_roi)
                idx = int(step["click_row"].get("index", 0))
                if len(rows) > idx:
                    r = rows[idx]
                    cmds.append({"op": "mouse_click", "button": "left",
                                 "x": r["x"], "y": r["y"]})
                    advanced = True
            elif "click_button" in step:
                tpl = self._button_template(step["click_button"])
                if tpl is None:
                    # 无按钮模板 → 无法验证，跳过该步（记录）
                    self.log.append({"t": ts_ms, "event": "skip_button",
                                     "button": step["click_button"]})
                    advanced = True
                else:
                    hit = match_button(frame, tpl, None, threshold=0.72)
                    if hit is not None:
                        cmds.append({"op": "mouse_click", "button": "left",
                                     "x": hit.x, "y": hit.y})
                        advanced = True
            elif "key" in step:
                cmds.append({"op": "key_tap", "key": step["key"],
                             "hold_ms": int(step.get("hold_ms", 60))})
                advanced = True
            elif "wait_ms" in step:
                advanced = ts_ms - self.step_started_ms >= int(step["wait_ms"])
        except Exception as e:  # noqa: BLE001
            self.log.append({"t": ts_ms, "event": "error", "reason": str(e)})

        if cmds and self.executor is not None:
            # 配方是精确的 UI 操作序列（点槽位/确认买入），多按两下就点错。
            # 连按是给"打怪"用的，不是给"填表单"用的。
            pause = getattr(self.executor, "burst_paused", None)
            ctx = pause() if callable(pause) else contextlib.nullcontext()
            with ctx:
                for c in cmds:
                    self.executor.execute(Command(**c))

        if advanced:
            self.step_i += 1
            self.step_started_ms = ts_ms
            self.log.append({"t": ts_ms, "event": "step_done", "step": step,
                             "commands": cmds})
            if self.step_i >= len(self.steps):
                self.done = True
                self.log.append({"t": ts_ms, "event": "done"})
        return [format_cmd(c) for c in cmds]

    def _pick_slot(self, slots: list[UiSlot], flt: dict) -> UiSlot | None:
        if not slots:
            return None
        if flt.get("highlighted"):
            highs = [s for s in slots if s.highlighted]
            if not highs:
                return None
            if flt.get("next"):
                # 依次点各高亮槽（天赋连点）：跳过已点过的位置
                clicked = {(s.get("x"), s.get("y")) for s in self.log
                           if s.get("event") == "step_done" and s.get("commands")
                           for s in s["commands"][:0]}  # 占位，实际用 _clicked_xy
            clicked = getattr(self, "_clicked_xy", set())
            for s in highs:
                if (s.x, s.y) not in clicked:
                    self._clicked_xy = getattr(self, "_clicked_xy", set()) | {(s.x, s.y)}
                    return s
            return highs[0]
        if "index" in flt and flt["index"] < len(slots):
            return slots[flt["index"]]
        return None

    # ------------- 状态 -------------

    def status(self) -> dict:
        return {"name": self.name, "done": self.done, "failed": self.failed,
                "step": self.step_i, "total_steps": len(self.steps),
                "log": self.log[-12:]}


def format_cmd(c: dict) -> str:
    op = c.get("op")
    if op == "mouse_click":
        return f"MOUSE_CLICK {c.get('button', 'left')} ({c.get('x')},{c.get('y')})"
    if op == "key_tap":
        return f"KEY_TAP {c.get('key')} hold={c.get('hold_ms')}ms"
    return json_dumps_short(c)


def json_dumps_short(c: dict) -> str:
    import json

    return json.dumps(c, ensure_ascii=False)
