"""场景门：反例模板优先 + 未命中即拦下（fail-closed）。

背景（真实事故）：部署到 RK3566 后打开配置页，用户在**关卡选择界面**看到
「向右下移动接近 auto2 掉落光柱（距离237px）」。两个独立缺陷叠在一起：
  1. 场景门靠一张裁错的 q_letter.png 兜底 → 对所有画面都给 0.8+，等于常开；
  2. 光柱判据只要高宽比 ≥1.1 → 星图节点（1.4~2.0）全被当成光柱。
这两条现在都有回归测试（本文件 + tests/test_real_scenarios.py）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autogame.config import DecisionConfig, ExecutorConfig, VisionConfig  # noqa: E402
from autogame.decision.fsm import DecisionEngine  # noqa: E402
from autogame.executor import DryRunExecutor  # noqa: E402
from autogame.vision.detector import Detector, ROOT_TEMPLATES  # noqa: E402

SAMPLES = ROOT / "assets" / "samples"
REF = ROOT / "assets" / "ref"
COMBAT = SAMPLES / "real_hud_boss.png"      # 技能栏模板命中（0.93）
MAGIC_MAP = REF / "tli_nether_map_01.jpg"   # 关卡选择（无光柱）


def _det(**gate_over) -> Detector:
    cfg = VisionConfig()
    for k, v in gate_over.items():
        setattr(cfg.scene_gate, k, v)
    return Detector(cfg, ROOT_TEMPLATES if ROOT_TEMPLATES.exists() else None)


def _load(p: Path) -> np.ndarray:
    img = cv2.imread(str(p))
    assert img is not None, f"素材缺失: {p}"
    return img


def test_negative_template_wins_over_skillbar(tmp_path):
    """反例优先：即使技能栏命中，命中反例也一律判非战斗。"""
    frame = _load(COMBAT)
    assert _det().process(frame, 1000).in_game is True  # 基线：技能栏放行

    neg = tmp_path / "neg_combat.png"
    cv2.imwrite(str(neg), frame[300:360, 200:400])  # 从同一张图裁一块当"反例"
    det = _det(negative_templates=[str(neg)])
    st = det.process(frame, 1000)
    assert st.in_game is False, "反例命中必须压过正向模板"
    assert det.diag["gate_reason"].startswith("negative:")


def test_flat_negative_template_is_rejected(tmp_path):
    """方差过低的模板必须被拒载：NCC 在纯色块上会给无意义的高分。

    （q_letter.png 就是这么坑的：它是一块图标角落，却让门禁对 8 张真实截图
      中的 7 张都放行。）
    """
    flat = tmp_path / "neg_flat.png"
    cv2.imwrite(str(flat), np.full((40, 80, 3), 128, dtype=np.uint8))
    det = _det(negative_templates=[str(flat)])
    assert det._neg_templates == [], "低方差模板不该进入匹配池"
    assert det.process(_load(COMBAT), 1000).in_game is True, "不该被一张纯色模板拦住"


def test_unmatched_fails_closed():
    """没有任何锚点命中 → 拦下（默认），而不是"未知即放行"。"""
    menu = _load(MAGIC_MAP)
    det = _det(negative_templates=["__nonexistent__.png"])  # 只剩正向锚点
    st = det.process(menu, 1000)
    assert st.in_game is False
    assert det.diag["gate_reason"] == "unmatched"


def test_fail_closed_can_be_turned_off():
    """fail_closed=False → 未匹配就当战斗（老部署的兼容开关，不推荐）。

    注意这条跟反例模板是两回事：反例命中是**确定的**非战斗，任何时候都拦；
    fail_closed 只影响"什么都认不出来"时怎么处置。
    """
    uncalibrated = SAMPLES / "real_darkmap_fight.png"  # 无锚点覆盖的战斗帧
    det = _det(fail_closed=False)
    assert det.process(_load(uncalibrated), 1000).in_game is True
    det2 = _det(fail_closed=True)
    assert det2.process(_load(uncalibrated), 1000).in_game is False


def test_nether_map_has_no_beam_and_no_command():
    """用户实测的那一帧：关卡选择页必须 0 光柱 / 0 命令 / 提示语不提"靠近光柱"。"""
    det = _det()
    ex = DryRunExecutor(ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000))
    eng = DecisionEngine(DecisionConfig(tick_hz=2.0), ex)
    eng.start()
    ts = 1000
    for _ in range(2):
        eng.on_scene(det.process(_load(MAGIC_MAP), ts))
        ts += 500
    st = det.process(_load(MAGIC_MAP), ts)

    assert st.in_game is False
    assert det.diag["gate_reason"].startswith("negative:"), det.diag
    assert st.pillars == [], "星图节点不是掉落光柱（高宽比 1.4~2.0，低于 2.5 下限）"
    assert ex.journal == [], f"门禁拦下时不允许任何指令，实际 {ex.journal}"
    assert "光柱" not in eng.last_intent, \
        f"拦截时意图里不许残留上一帧的『接近光柱』：{eng.last_intent}"


def test_boss_combat_frame_still_reads_and_acts():
    """反向保护：改了门禁/光柱阈值后，正常战斗帧仍要放行且能产出决策。"""
    det = _det()
    ex = DryRunExecutor(ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000))
    eng = DecisionEngine(DecisionConfig(tick_hz=2.0), ex)
    eng.start()
    ts = 1000
    for _ in range(3):
        eng.on_scene(det.process(_load(COMBAT), ts))
        ts += 500
    assert det.diag["gate_reason"] == "skillbar"
    assert len(ex.journal) > 0, "战斗画面必须还能动（别把门禁改成永远拦）"


@pytest.mark.parametrize("h_min", [1.4, 1.85, 2.0])
def test_squat_glow_is_not_a_pillar(h_min):
    """高宽比低于 2.5 的团光/地图节点不算光柱（含白芯也不行）。"""
    cfg = VisionConfig()
    cfg.hsv = {"gold": ([20, 120, 120], [40, 255, 255])}
    det = Detector(cfg, None)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    w = 20
    h = int(w * h_min)
    x, y = 640, 400
    cv2.rectangle(frame, (x - w // 2, y), (x + w // 2, y + h), (60, 190, 255), -1)
    cv2.rectangle(frame, (x - w // 4, y + 4), (x + w // 4, y + h - 4), (255, 255, 255), -1)
    st = det.process(frame, 1000)
    assert st.pillars == [], f"高宽比 {h_min}:1 的团光被当成了光柱"


def test_tall_thin_beam_is_a_pillar():
    """反向保护：真正的细高光柱（14x110，高宽比 7.9）必须能被确认。"""
    cfg = VisionConfig()
    cfg.hsv = {"gold": ([20, 120, 120], [40, 255, 255])}
    cfg.beam_persistence_frames = 1
    det = Detector(cfg, None)
    got = 0
    for i in range(2):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.rectangle(frame, (633, 290), (647, 400), (60, 190, 255), -1)
        cv2.rectangle(frame, (637, 296), (643, 396), (255, 255, 255), -1)
        got += len(det.process(frame, 1000 + i * 500).pillars)
    assert got >= 1, "竖直细条光柱必须被检出"


def test_negative_template_survives_downscale(tmp_path):
    """源帧宽 > max_detect_width 时，反例模板仍必须命中。

    真实事故（2560×1380 的录屏）：`_check_negative_ui` 里模板缩放漏乘了
    `scale`，模板被放大 2 倍去和半分辨率的整屏比对，星图/结算/过场**一次都没
    匹配上**，门禁把这些界面全当战斗放行（71 条命令打在了选图界面上）。
    源帧宽恰好等于 max_detect_width 时 scale=1，缺陷完全隐形——所以必须
    用"同一画面放大 2 倍"来钉这个不变量。
    """
    rng = np.random.default_rng(0)
    patch = rng.integers(0, 255, (60, 120, 3), dtype=np.uint8)
    base = np.zeros((690, 1280, 3), dtype=np.uint8)
    base[100:160, 200:320] = patch
    big = cv2.resize(base, (2560, 1380), interpolation=cv2.INTER_NEAREST)

    tpl_dir = tmp_path / "templates"
    tpl_dir.mkdir()
    cv2.imwrite(str(tpl_dir / "gate_neg_synth.png"), patch)   # 1280 宽基准

    cfg = VisionConfig()
    cfg.max_detect_width = 1280          # 2560 的帧会被缩到 1280 → scale=0.5
    cfg.scene_gate.enabled = True
    cfg.scene_gate.fail_closed = True
    det = Detector(cfg, tpl_dir)
    assert det._neg_templates, "反例模板没被加载，测试前提不成立"

    st = det.process(big, 1000)
    assert not st.in_game, "放大 2 倍的同一张界面必须仍被反例拦下"
    assert det.diag["gate_reason"].startswith("negative:"), (
        f"反例没命中，实际原因 {det.diag['gate_reason']}（缩放没跟着 max_detect_width 走）")


def test_negative_template_survives_downscale(tmp_path):
    """源帧宽 > max_detect_width 时，反例模板仍必须命中。

    真实事故（2560×1380 的录屏）：`_check_negative_ui` 里模板缩放漏乘了
    `scale`，模板被放大 2 倍去和半分辨率的整屏比对，星图/结算/过场**一次都没
    匹配上**，门禁把这些界面全当战斗放行（71 条命令打在了选图界面上）。
    源帧宽恰好等于 max_detect_width 时 scale=1，缺陷完全隐形——所以必须
    用"同一画面放大 2 倍"来钉这个不变量。
    """
    rng = np.random.default_rng(0)
    patch = rng.integers(0, 255, (60, 120, 3), dtype=np.uint8)
    base = np.zeros((690, 1280, 3), dtype=np.uint8)
    base[100:160, 200:320] = patch
    big = cv2.resize(base, (2560, 1380), interpolation=cv2.INTER_NEAREST)

    tpl_dir = tmp_path / "templates"
    tpl_dir.mkdir()
    cv2.imwrite(str(tpl_dir / "gate_neg_synth.png"), patch)   # 1280 宽基准

    cfg = VisionConfig()
    cfg.max_detect_width = 1280          # 2560 的帧会被缩到 1280 → scale=0.5
    cfg.scene_gate.enabled = True
    cfg.scene_gate.fail_closed = True
    det = Detector(cfg, tpl_dir)
    assert det._neg_templates, "反例模板没被加载，测试前提不成立"

    st = det.process(big, 1000)
    assert not st.in_game, "放大 2 倍的同一张界面必须仍被反例拦下"
    assert det.diag["gate_reason"].startswith("negative:"), (
        f"反例没命中，实际原因 {det.diag['gate_reason']}（缩放没跟着 max_detect_width 走）")


def test_negative_beats_positive_when_hud_still_visible(tmp_path):
    """反例必须压过正向：角色面板把 HUD 留在画面上时，门禁仍要拦。

    真实事故（火炬之光 PC 录屏）：打开角色/装备面板时，底部技能栏、血条、
    经验条**全都还在**（面板是半透明覆盖层），于是三张正向锚点全部命中
    （0.83，阈值 0.60），门禁判"这是战斗"，12 帧里点了 5 下 —— 其中
    (736,950)/(755,952) 正好落在「鉴定所有」上。这类面板一旦被放行，
    点一下就是换装/洗点，比漏检更贵。

    不变量：**反例优先于正向**，且与正向得分高低无关。
    """
    rng = np.random.default_rng(7)
    hud = rng.integers(0, 255, (40, 120, 3), dtype=np.uint8)       # 模拟技能栏
    panel = rng.integers(0, 255, (70, 150, 3), dtype=np.uint8)     # 模拟面板按钮行
    frame = np.zeros((690, 1280, 3), dtype=np.uint8)
    frame[600:640, 300:420] = hud          # 底部 HUD 在
    frame[390:460, 960:1110] = panel       # 面板同时在

    tpl_dir = tmp_path / "templates"
    tpl_dir.mkdir()
    cv2.imwrite(str(tpl_dir / "gate_pos_synth_hud.png"), hud)
    cv2.imwrite(str(tpl_dir / "gate_neg_synth_panel.png"), panel)

    cfg = VisionConfig()
    cfg.scene_gate.enabled = True
    cfg.scene_gate.fail_closed = True
    cfg.scene_gate.threshold = 0.6
    det = Detector(cfg, tpl_dir)
    assert det._skillbar_templates and det._neg_templates, "两类模板都要加载到，测试前提不成立"

    st = det.process(frame, 1000)
    assert not st.in_game, "HUD 还在也不能放行——面板开着就不是战斗画面"
    assert det.diag["gate_reason"].startswith("negative:"), (
        f"反例没优先于正向，实际原因 {det.diag['gate_reason']}"
        f"（正向 {det.diag['gate_best']:.3f} / 反例 {det.diag['gate_neg_best']:.3f}）")
