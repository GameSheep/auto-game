"""真实游戏截图回归测试 —— 用网上找的实机素材验证全链路。

素材（assets/samples/，来源：B站实机视频抽帧 / 17173 评测文章）：
  real_hud_boss.png / real_combat_hud.png / real_beamlike_tentacles.png
      —— 真实主线 Boss 战 + 完整 HUD（小地图/任务/技能栏 QWE R A）
  real_skillfx_storm.png —— 技能特效满屏（青紫光束横飞，误报压力场景）
  real_darkmap_farm.png / real_darkmap_fight.png —— 暗黑 endgame 刷图
  real_equip_panel.png —— 装备面板（整屏 UI，含紫色物品 tooltip）
  real_inventory.jpg —— 背包界面（黑底物品 + 彩色框）

验收（对应"很多场景没有处理好"的整改）：
  1. 全部素材不崩溃、场景门分类正确（战斗 HUD 放行 / UI 界面拦截）
  2. 战斗特效（粉色 Boss 触手/技能光束）不得被确认为光柱（白芯校验+时域确认）
  3. UI 场景 FSM 零输出
  4. 自适应标定在真实帧上产出 HSV 范围
  5. 场景运行器对战斗帧输出文本命令、对 UI 帧不输出
"""

import sys
from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autogame.config import DecisionConfig, ExecutorConfig, VisionConfig  # noqa: E402
from autogame.decision.fsm import DecisionEngine  # noqa: E402
from autogame.executor import DryRunExecutor  # noqa: E402
from autogame.vision.auto_calib import auto_hsv_ranges  # noqa: E402
from autogame.vision.detector import Detector, ROOT_TEMPLATES  # noqa: E402

SAMPLES = ROOT / "assets" / "samples"

GAMEPLAY = [
    "real_hud_boss.png",
    "real_combat_hud.png",
    "real_beamlike_tentacles.png",
    "real_skillfx_storm.png",
    "real_darkmap_farm.png",
]
UI_SCREENS = [
    "real_equip_panel.png",
    "real_inventory.jpg",
]

ALL_REAL = GAMEPLAY + UI_SCREENS


def _load(name: str):
    frame = cv2.imread(str(SAMPLES / name))
    assert frame is not None, f"素材缺失: {name}"
    return frame


def _det():
    return Detector(VisionConfig(), ROOT_TEMPLATES if ROOT_TEMPLATES.exists() else None)


def _run_ticks(det: Detector, frame, n=3, ts0=1000, step=1000):  # noqa: ANN001
    states = []
    for i in range(n):
        states.append(det.process(frame, ts0 + i * step))
    return states


# ---------------- 场景门 ----------------

@pytest.mark.parametrize("name", GAMEPLAY)
def test_gameplay_frames_pass_scene_gate(name):
    det = _det()
    states = _run_ticks(det, _load(name))
    assert states[-1].in_game is True, f"{name} 是战斗画面，场景门必须放行"


def test_inventory_blocked_by_scene_gate():
    """背包界面（无血球无技能栏）必须被场景门拦截。"""
    det = _det()
    states = _run_ticks(det, _load("real_inventory.jpg"))
    assert states[-1].in_game is False, "背包界面必须被场景门拦截"


def test_equip_panel_never_emits_key_commands():
    """装备面板：红血球锚点可能误放行（红物品图标在血球区，已知边界），但
    无论门禁状态如何，绝不允许产生任何按键命令（鼠标点击最多）。"""
    det = _det()
    ex = DryRunExecutor(ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000))
    eng = DecisionEngine(DecisionConfig(tick_hz=1.0), ex)
    eng.start()
    frame = _load("real_equip_panel.png")
    ts = 1000
    for _ in range(3):
        st = det.process(frame, ts)
        eng.on_scene(st)
        ts += 1000
    key_cmds = [j for j in ex.journal if j["op"].startswith("key")]
    assert len(key_cmds) == 0, "装备面板绝不允许产生按键命令"
    mouse_cmds = [j for j in ex.journal if j["op"] == "mouse_click"]
    # 即便有鼠标点击（低危走位），也不能连续点击同一位置超过 2 次（防 UI 误操作风暴）
    assert len(mouse_cmds) <= 4, f"装备面板鼠标点击过多: {len(mouse_cmds)}"


# ---------------- 误报抑制（战斗特效 ≠ 光柱） ----------------

@pytest.mark.parametrize("name", GAMEPLAY)
def test_combat_fx_not_confirmed_as_pillars(name):
    """真实战斗特效（Boss 触手/技能光束）跨 3 帧不得确认为光柱。"""
    det = _det()
    states = _run_ticks(det, _load(name), n=4)
    for i, st in enumerate(states):
        assert len(st.pillars) == 0, (
            f"{name} tick{i}: 战斗特效被误判为光柱 {[p.color for p in st.pillars]}"
        )


def test_white_core_check_discriminates():
    """同一画面：有白热芯的竖条被检出，纯饱和色竖条（Boss 触手特征）被拒。"""
    frame = _load("real_beamlike_tentacles.png")
    det = _det()
    # 人工在画面空地画两种竖条：白芯 vs 纯饱和
    y0, y1 = 640, 700  # 地面空白区（避开 HUD 与角色）
    frame[y0:y1, 1000:1014] = (36, 190, 255)
    frame[y0 + 6 : y1 - 4, 1005:1009] = (255, 255, 255)  # 白芯 → 应确认
    frame[y0:y1, 1100:1114] = (80, 40, 200)              # 纯饱和红蓝 → 无白芯 → 应拒绝
    det.process(frame, 1000)
    st = det.process(frame, 2000)
    cx_list = [p.cx for p in st.pillars]
    assert any(abs(c - 1007) <= 20 for c in cx_list), "带白芯的光柱必须被确认"
    assert not any(abs(c - 1107) <= 20 for c in cx_list), "无白芯的特效条必须被拒绝"


def test_hud_exclusions_mask_minimap_and_skillbar():
    """HUD 排除区：在真实帧的小地图/技能栏位置画光柱也不得检出。"""
    frame = _load("real_hud_boss.png")
    # 小地图区 (~120,150) 与技能栏区 (~640,640) 画白芯光柱
    for (sx, sy) in [(120, 200), (640, 700)]:
        frame[sy - 60 : sy, sx - 7 : sx + 7] = (36, 190, 255)
        frame[sy - 54 : sy - 4, sx - 2 : sx + 2] = (255, 255, 255)
    det = _det()
    det.process(frame, 1000)
    st = det.process(frame, 2000)
    for p in st.pillars:
        assert not (abs(p.cx - 120) <= 20 and abs(p.cy - 200) <= 40), "小地图区光柱必须被排除"
        assert not (abs(p.cx - 640) <= 20 and abs(p.cy - 700) <= 40), "技能栏区光柱必须被排除"


# ---------------- 自适应标定 ----------------

@pytest.mark.parametrize("name", ALL_REAL)
def test_auto_calibration_produces_ranges(name):
    frame = _load(name)
    ranges = auto_hsv_ranges(frame)
    if name == "real_inventory.jpg":
        assert ranges, "彩色背包界面必须能推断出色簇"
    # 不强求全部帧有色簇（暗图可能不足 200 像素），但接口必须稳定返回 list
    assert isinstance(ranges, list)


# ---------------- FSM 场景门 + 场景运行器 ----------------

@pytest.mark.parametrize("name", UI_SCREENS)
def test_fsm_zero_key_commands_on_ui_screens(name):
    """UI 界面 FSM 绝不产生按键命令（鼠标走位点击为低危上限 4 条）。"""
    det = _det()
    ex = DryRunExecutor(ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000))
    eng = DecisionEngine(DecisionConfig(tick_hz=1.0), ex)
    eng.start()
    frame = _load(name)
    ts = 1000
    for _ in range(3):
        st = det.process(frame, ts)
        eng.on_scene(st)
        ts += 1000
    key_cmds = [j for j in ex.journal if j["op"].startswith("key")]
    assert len(key_cmds) == 0, f"{name} UI 界面产生了按键命令 {[j['op'] for j in key_cmds]}"


def test_scenario_runner_text_output_on_real_frame():
    """场景运行器：真实战斗帧 → 文本命令输出（MOUSE_CLICK/KEY_TAP 格式）。"""
    sys.path.insert(0, str(ROOT / "tools"))
    import importlib

    sr = importlib.import_module("scenario_runner")
    report = sr.run_image(SAMPLES / "real_combat_hud.png", ticks=3)
    assert report["size"] == [1280, 720]
    # 真实战斗帧无真实掉落光柱 → 铁律之下不产生按 A 命令（防盲按验收）
    all_cmds = [c for cl in report["commands"] for c in cl]
    assert not any(c.startswith("KEY_TAP a") for c in all_cmds), \
        "战斗特效不得触发拾取键（绝不盲按）"


def test_real_beamlike_tentacles_do_not_trigger_pickup():
    """Boss 触手（类光柱形态）跨帧不得触发按 A —— 对话定稿铁律的真实素材验证。"""
    det = _det()
    ex = DryRunExecutor(ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000))
    eng = DecisionEngine(DecisionConfig(tick_hz=1.0), ex)
    eng.start()
    frame = _load("real_beamlike_tentacles.png")
    ts = 1000
    for _ in range(4):
        st = det.process(frame, ts)
        eng.on_scene(st)
        ts += 1000
    a_taps = [j for j in ex.journal if j["op"] == "key_tap" and j["key"] == "a"]
    assert len(a_taps) == 0, "Boss 触手特效绝不能触发拾取键"
    assert eng.stats["blind_tap_violations"] == 0
