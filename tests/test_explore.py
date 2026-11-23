"""探索落点与地面标签拾取（真实录像 232 条命令里 154 条是固定四点 → 改掉）。"""

import math

from autogame.config import DecisionConfig, ExecutorConfig
from autogame.decision.fsm import DecisionEngine
from autogame.executor import DryRunExecutor
from autogame.protocol import ListEntry, SceneState


def _engine(**kw):
    cfg = DecisionConfig(tick_hz=10.0, **kw)
    cfg.autonomy.enabled = False          # 只看走位本身，别被自主恢复插入命令
    ex = DryRunExecutor(ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000))
    eng = DecisionEngine(cfg, ex)
    eng.start()
    return eng, ex


def _scene(ts=1000, labels=None, w=2560, h=1380):
    return SceneState(
        ts_ms=ts, w=w, h=h, character=(w // 2, h // 2),
        pillars=[], ground_labels=labels or [], frame_delta=50.0,
    )


def test_explore_points_disperse():
    """探索落点必须散开：唯一坐标够多、扇区覆盖够广。

    旧实现是"4 个方向 × 固定 220px"，落点永远是 (640,125)/(860,345) 那 4 个。
    真实回放里 232 条命令有 154 条就是这 4 个坐标 —— 一眼看出是脚本。
    """
    eng, _ = _engine()
    sc = _scene()
    pts = [eng._pick_explore_point(sc) for _ in range(24)]
    uniq = set(pts)
    assert len(uniq) >= 18, f"24 次探索只有 {len(uniq)} 个不同落点（还是固定点位？）"

    # 覆盖的扇区数：按相对角色的角度分 8 档
    cx, cy = sc.character
    sectors = {int(((math.degrees(math.atan2(y - cy, x - cx)) + 360) % 360) // 45) for x, y in pts}
    assert len(sectors) >= 5, f"只走到 {len(sectors)} 个扇区，探索方向太单一"


def test_explore_points_stay_inside_safe_area():
    """落点不许落在底部 HUD 带 / 屏幕边缘（那里点了不是走路，是点界面）。"""
    eng, _ = _engine()
    sc = _scene()
    w, h = sc.w, sc.h
    for _ in range(30):
        x, y = eng._pick_explore_point(sc)
        assert 0.05 * w <= x <= 0.95 * w, f"落点 x={x} 贴到屏幕左右边缘"
        assert 0.07 * h <= y <= 0.87 * h, f"落点 y={y} 落到顶部/底部 HUD 带"


def test_explore_point_avoids_repeating_last_one():
    """连续两次探索不该落在几乎同一个点上（原地踏步 = 卡住的样子）。"""
    eng, _ = _engine()
    sc = _scene()
    base = min(sc.w, sc.h)
    last = eng._pick_explore_point(sc)
    for _ in range(10):
        cur = eng._pick_explore_point(sc)
        assert math.hypot(cur[0] - last[0], cur[1] - last[1]) >= base * 0.05, (
            f"两次落点几乎重合 {last} → {cur}")
        last = cur


# ---------------- 地面标签近距拾取（TLI：没有独立 A框） ----------------

def test_ground_label_pickup_when_near():
    """标签走到跟前 + 连续可见 → 按一次拾取键（这是 TLI 唯一的合法按 A 时机）。"""
    eng, ex = _engine(label_pickup_when_near=True)
    # 标签就在角色旁边（1280×720 短边 720，0.09 → 64.8px 内）
    label = [ListEntry(x=660, y=370, w=120, h=20)]
    for i in range(4):
        eng.on_scene(_scene(ts=1000 + i * 200, labels=label, w=1280, h=720))
    taps = [j for j in ex.journal if j["op"] == "key_tap" and j.get("key") == "a"]
    assert len(taps) == 1, f"近距标签连续 4 帧应只按 1 次拾取键，实际 {len(taps)} 次"


def test_ground_label_pickup_needs_confirm_frames():
    """只出现 1 帧的标签不按（旧帧/闪烁，按下去就是盲按）。"""
    eng, ex = _engine(label_pickup_when_near=True)
    label = [ListEntry(x=660, y=370, w=120, h=20)]
    eng.on_scene(_scene(ts=1000, labels=label, w=1280, h=720))
    taps = [j for j in ex.journal if j["op"] == "key_tap"]
    assert taps == [], "只出现一帧的标签不能按拾取键"


def test_ground_label_pickup_skips_far_labels():
    """标签在远处 → 只走过去，不按拾取键。"""
    eng, ex = _engine(label_pickup_when_near=True)
    label = [ListEntry(x=1200, y=650, w=120, h=20)]   # 离角色 (640,360) 很远
    for i in range(5):
        eng.on_scene(_scene(ts=1000 + i * 200, labels=label, w=1280, h=720))
    taps = [j for j in ex.journal if j["op"] == "key_tap"]
    assert taps == [], "远处的标签不能按拾取键（要先走过去）"


def test_ground_label_pickup_off_by_default():
    """默认关闭：没有拾取键的游戏不该被这条路径按到键盘。"""
    eng, ex = _engine()
    label = [ListEntry(x=660, y=370, w=120, h=20)]
    for i in range(5):
        eng.on_scene(_scene(ts=1000 + i * 200, labels=label, w=1280, h=720))
    taps = [j for j in ex.journal if j["op"] == "key_tap"]
    assert taps == [], "默认配置下地面标签拾取必须是关的"


# ---------------- R4 铁律口径：近距标签也算"有证据" ----------------

def test_pickup_evidence_accepts_near_ground_label():
    """R4 检查的口径：没有 A框但有"跟前的地面标签"不算盲按。

    TLI 没有 A框，拾取提示就是地面标签。回放第一次跑出 3 条 R4 违规，
    全是这条新路径按的 —— 如果直接放宽检查就是自欺，所以要把"证据"定义
    写进代码：A框出现，**或** 最近标签落在 label_pickup_near_ratio 之内。
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    from video_batch import _pickup_evidence

    cfg = DecisionConfig(label_pickup_when_near=True)
    cfg.autonomy.enabled = False
    sc = SceneState(ts_ms=1000, w=1280, h=720, character=(640, 360),
                    pillars=[], ground_labels=[ListEntry(x=680, y=380, w=120, h=20)],
                    frame_delta=50.0)
    assert _pickup_evidence(sc, cfg), "近距标签应该算证据"

    far = SceneState(ts_ms=1000, w=1280, h=720, character=(640, 360),
                     pillars=[], ground_labels=[ListEntry(x=1250, y=690, w=120, h=20)],
                     frame_delta=50.0)
    assert not _pickup_evidence(far, cfg), "远处的标签不是证据"

    off = DecisionConfig()          # 默认关 → 标签路径不成立
    off.autonomy.enabled = False
    assert not _pickup_evidence(sc, off), "配置没开时标签不能当证据"
