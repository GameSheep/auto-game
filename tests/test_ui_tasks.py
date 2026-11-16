"""UI 场景任务测试 —— 交易行/装备/天赋/技能 配方引擎 + 网格/高亮/行检测。

全部用合成面板帧（OpenCV 绘制）验证检测器与配方引擎，无需真实截图。
"""

import cv2
from pathlib import Path
import numpy as np
import pytest

from autogame.config import DecisionConfig, ExecutorConfig, VisionConfig
from autogame.decision.recipe import PRESET_RECIPES, RecipeRunner
from autogame.decision.fsm import DecisionEngine
from autogame.executor import DryRunExecutor
from autogame.protocol import Command, ListEntry, SceneState
from autogame.vision.ui_panels import GridConfig, detect_grid_slots, detect_text_rows

GOLD = (36, 190, 255)   # BGR 亮金


def _mk_trade_panel(highlight_idx=5):
    """合成交易行面板：4x3 物品格网格 + 价格行 + 排序/购买/确认按钮。"""
    f = np.full((720, 1280, 3), (30, 26, 24), dtype=np.uint8)
    gx, gy, cw, ch = 200, 100, 60, 60
    idx = 0
    for r in range(3):
        for c in range(4):
            x, y = gx + c * 80, gy + r * 80
            bright = GOLD if idx == highlight_idx else (90, 120, 160)
            cv2.rectangle(f, (x, y), (x + cw, y + ch), bright, -1)
            cv2.rectangle(f, (x, y), (x + cw, y + ch), (30, 30, 30), 1)
            idx += 1
    # 价格行（3 行亮色横条）
    for i, y in enumerate((430, 470, 510)):
        cv2.rectangle(f, (200, y), (620, y + 24), (120, 170, 255), -1)
    # 按钮：排序(右上) / 购买(中下) / 确认(中下) —— 带文字与边框（真实按钮有内容，
    # 纯色块模板对 TM_CCOEFF_NORMED 是零方差退化输入）
    for (x0, y0, x1, y1, label) in [
        (950, 110, 1070, 150, "SORT"),
        (520, 560, 660, 600, "BUY"),
        (520, 620, 660, 660, "OK"),
    ]:
        cv2.rectangle(f, (x0, y0), (x1, y1), (200, 160, 90), -1)
        cv2.rectangle(f, (x0, y0), (x1, y1), (40, 40, 40), 2)
        cv2.putText(f, label, (x0 + 12, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (20, 20, 20), 2)
    return f


def _crop_templates(panel, tmp_path, with_panel_anchor: bool = True):
    """从面板裁出按钮模板（模拟用户从自己截图裁剪）。

    panel_trade.png 是面板锚点：任务启动前用它确认"面板真的打开了"。
    缺它时配方启动会被预检拒绝（避免空转假成功）。
    """
    for name, (x0, y0, x1, y1) in {
        "sort.png": (950, 110, 1070, 150),
        "buy.png": (520, 560, 660, 600),
        "confirm.png": (520, 620, 660, 660),
    }.items():
        cv2.imwrite(str(tmp_path / name), panel[y0:y1, x0:x1])
    if with_panel_anchor:
        # 面板锚点取标题栏（稳定不变的区域）
        cv2.imwrite(str(tmp_path / "panel_trade.png"), panel[10:50, 150:640])
    return tmp_path


GRID = GridConfig(roi=[180, 90, 400, 250], cell_min=30, cell_max=90)


# ---------------- 网格槽位检测 ----------------

def test_grid_slots_detected_and_highlighted():
    panel = _mk_trade_panel(highlight_idx=5)
    slots = detect_grid_slots(panel, GRID)
    assert len(slots) >= 10, f"4x3 网格应检出 ≥10 槽位，实际 {len(slots)}"
    # 高亮槽（第 5 号，金色）应被标记
    hl = [s for s in slots if s.highlighted]
    assert len(hl) >= 1, "必须标记高亮槽位"
    hl = hl[0]
    expect_x = 200 + (5 % 4) * 80 + 30
    expect_y = 100 + (5 // 4) * 80 + 30
    assert abs(hl.x - expect_x) <= 10 and abs(hl.y - expect_y) <= 10


def test_grid_slots_empty_roi():
    panel = np.full((720, 1280, 3), (30, 26, 24), dtype=np.uint8)
    assert detect_grid_slots(panel, GRID) == []


# ---------------- 文本行检测 ----------------

def test_text_rows_detected():
    panel = _mk_trade_panel()
    rows = detect_text_rows(panel, [180, 410, 500, 130])
    assert len(rows) >= 3, f"应检出 3 条价格行，实际 {len(rows)}"
    ys = sorted(r["y"] for r in rows)
    assert ys[1] - ys[0] >= 30, "行间距应分开"


# ---------------- 配方引擎：trade_buy 全流程 ----------------

def test_recipe_trade_buy_completes(tmp_path):
    panel = _mk_trade_panel()
    tdir = _crop_templates(panel, tmp_path)
    ex = DryRunExecutor(ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000))
    runner = RecipeRunner("trade_buy", PRESET_RECIPES["trade_buy"], GRID,
                          templates_dir=tdir, step_timeout_ms=6000, executor=ex,
                          rows_roi=[180, 410, 500, 130])
    ts = 1000
    for _ in range(8):
        runner.tick(panel, ts)
        ts += 1000
        if runner.done:
            break
    assert runner.done and not runner.failed, f"配方应完成: {runner.failed} log={runner.log[-4:]}"
    clicks = [j for j in ex.journal if j["op"] == "mouse_click"]
    assert len(clicks) == 4, f"应有 4 次点击（排序/行/购买/确认），实际 {len(clicks)}"
    # 第 2 次点击 = 价格首行（最低价）
    assert abs(clicks[1]["x"] - 410) <= 30 and abs(clicks[1]["y"] - 442) <= 20
    # 第 3 次点击 = 购买按钮中心
    assert abs(clicks[2]["x"] - 590) <= 10 and abs(clicks[2]["y"] - 580) <= 10


def test_recipe_times_out_when_button_missing(tmp_path):
    """按钮模板存在但画面上没有该按钮 → 单步超时 → 任务失败并报告步骤。"""
    panel = _mk_trade_panel()
    tdir = _crop_templates(panel, tmp_path)
    # 画面换成空面板（无按钮）
    blank = np.full((720, 1280, 3), (30, 26, 24), dtype=np.uint8)
    ex = DryRunExecutor(ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000))
    runner = RecipeRunner("trade_buy", PRESET_RECIPES["trade_buy"], GRID,
                          templates_dir=tdir, step_timeout_ms=1500, executor=ex)
    ts = 1000
    for _ in range(5):
        runner.tick(blank, ts)
        ts += 1000
        if runner.done:
            break
    assert runner.done and runner.failed, "无按钮必须失败并报告"
    assert "超时" in runner.failed


def test_all_presets_exist():
    assert set(PRESET_RECIPES.keys()) == {"trade_buy", "equip_best",
                                          "talent_alloc", "skill_swap"}


def test_ground_label_detection_on_real_frame():
    """真实 tartarus 帧：地面掉落标签（黄条）检出。"""
    from autogame.vision.detector import Detector, ROOT_TEMPLATES
    fp = Path("tests/fixtures/frames/tar_0150.png")
    if not fp.exists():
        fp = Path("assets/frames/tar_0150.png")
    frame = cv2.imread(str(fp))
    assert frame is not None
    det = Detector(VisionConfig(), ROOT_TEMPLATES if ROOT_TEMPLATES.exists() else None)
    st = det.process(frame, 1000)
    assert len(st.ground_labels) >= 3, f"标签密集帧应检出 ≥3 个掉落标签，实际 {len(st.ground_labels)}"
    for e in st.ground_labels:
        assert 0 < e.x < 1280 and 0 < e.y < 720


def test_ground_label_walk_fsm():
    """FSM：无光柱 → 地面标签连续 400ms 可见 → 点击标签下方走位。"""
    eng_cfg = DecisionConfig(tick_hz=10.0)
    ex = DryRunExecutor(ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000))
    eng = DecisionEngine(eng_cfg, ex)
    eng.start()

    def scene(ts, labels=True):
        return SceneState(
            ts_ms=ts, w=1280, h=720, character=(640, 360),
            pillars=[],
            ground_labels=[ListEntry(x=700, y=400, w=120, h=20)] if labels else [],
            frame_delta=50.0,
        )

    eng.on_scene(scene(1000))
    eng.on_scene(scene(1500))   # ≥400ms → 点击
    clicks = [j for j in ex.journal if j["op"] == "mouse_click"]
    assert len(clicks) == 1
    assert clicks[0]["x"] == 700 and clicks[0]["y"] == 400
    assert eng.stats["label_walks"] == 1

    # 防抖：闪烁消失不点击
    ex2 = DryRunExecutor(ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000))
    eng2 = DecisionEngine(eng_cfg, ex2)
    eng2.start()
    eng2.on_scene(scene(1000))
    eng2.on_scene(scene(1200, labels=False))
    assert len(ex2.journal) == 0


# ---------------- 启动前模板预检 ----------------

def test_missing_templates_reported(tmp_path):
    """缺面板锚点 → 预检必须报出来（不再静默跳过）。"""
    panel = _mk_trade_panel()
    _crop_templates(panel, tmp_path, with_panel_anchor=False)
    runner = RecipeRunner("trade_buy", PRESET_RECIPES["trade_buy"], GRID,
                          templates_dir=tmp_path)
    assert runner.missing_templates() == ["panel_trade.png"]


def test_missing_templates_all_reported_when_empty_dir(tmp_path):
    """空模板目录 → 4 个模板全部缺失，且顺序稳定（面板锚点在前）。"""
    runner = RecipeRunner("trade_buy", PRESET_RECIPES["trade_buy"], GRID,
                          templates_dir=tmp_path)
    assert runner.missing_templates() == [
        "panel_trade.png", "sort.png", "buy.png", "confirm.png"]


def test_no_missing_when_all_present(tmp_path):
    """模板齐备 → 预检通过，且能解析出真实路径。"""
    panel = _mk_trade_panel()
    _crop_templates(panel, tmp_path)
    runner = RecipeRunner("trade_buy", PRESET_RECIPES["trade_buy"], GRID,
                          templates_dir=tmp_path)
    assert runner.missing_templates() == []
    assert runner.template_path("sort.png") is not None
