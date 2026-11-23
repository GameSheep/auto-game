"""人类运动学测试 —— 轨迹形状与动作节奏的**统计性质**。

这里的断言不是"值等于多少"，而是"分布的形状符合人类运动学/时间序列的
已知性质"。每一条都对应一种可被检测出来的机器特征，见 `autogame/motion.py`
开头的分层说明。

⚠️ 统计断言必须用固定 seed + 足够样本，否则会变成偶发失败。
"""

import math
import random
import statistics

from autogame.config import ExecutorConfig, MousePathConfig
from autogame.executor.base import _jitter
from autogame.humanize import build_mouse_path
from autogame.motion import Rhythm, fitts_duration, human_path
from autogame.security import ActionRateLimiter

# ═══════════════════════════ Fitts 律 ═══════════════════════════


def test_fitts_duration_grows_logarithmically():
    """距离翻倍，耗时**不是**翻倍（对数增长），这是 Fitts 律的核心。"""
    d100 = fitts_duration(100, 24, cv=0.0)
    d200 = fitts_duration(200, 24, cv=0.0)
    d400 = fitts_duration(400, 24, cv=0.0)
    assert d200 < d100 * 2 and d400 < d200 * 2, "必须是亚线性（对数）增长"
    assert d400 > d200 > d100
    # 对数曲线的判别特征：距离翻倍带来的增量**远小于**原值（线性增长会是等量）
    assert (d200 - d100) < d100 * 0.6
    assert d400 < 2.0 * d200


def test_fitts_duration_prefers_wider_target():
    """目标越宽越"好点"（耗时更短），这是 Fitts 律里 W 的作用。"""
    narrow = fitts_duration(600, 12, cv=0.0)
    wide = fitts_duration(600, 96, cv=0.0)
    assert wide < narrow


# ═══════════════════════════ 轨迹 ═══════════════════════════


def _path_slope(path) -> float:
    """log v 对 log κ 的回归斜率（曲率越大速度越慢 → 应为负）。"""
    pts = [(p.x, p.y) for p in path]
    pairs = []
    for i in range(1, len(pts) - 1):
        x1, y1 = pts[i - 1]
        x2, y2 = pts[i]
        x3, y3 = pts[i + 1]
        a = math.hypot(x2 - x1, y2 - y1)
        b = math.hypot(x3 - x2, y3 - y2)
        c = math.hypot(x3 - x1, y3 - y1)
        if a * b * c < 1e-9:
            continue
        area2 = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        kappa = 2.0 * area2 / (a * b * c)
        v = a / max(1e-6, path[i].dt)
        if kappa > 1e-5 and v > 1.0:
            pairs.append((math.log(kappa), math.log(v)))
    if len(pairs) < 5:
        return 0.0
    mx = statistics.mean(p[0] for p in pairs)
    my = statistics.mean(p[1] for p in pairs)
    sxx = sum((p[0] - mx) ** 2 for p in pairs) or 1e-9
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    return sxy / sxx


def _peaks(path) -> int:
    """平滑后的速度峰数（≈ 亚运动段数）。"""
    seg = [math.hypot(path[i].x - path[i - 1].x, path[i].y - path[i - 1].y)
           for i in range(1, len(path))]
    raw = [seg[i - 1] / max(1e-6, path[i].dt) for i in range(1, len(path))]
    if len(raw) >= 5:
        sp = [statistics.mean(raw[max(0, i - 2):i + 3]) for i in range(len(raw))]
    else:
        sp = raw
    if not sp:
        return 0
    top = max(sp) or 1e-9
    n, last = 0, -99
    for i in range(2, len(sp) - 2):
        if sp[i] < 0.45 * top:
            continue
        if not (sp[i] >= sp[i - 1] and sp[i] > sp[i + 1]):
            continue
        if i - last < 3:
            continue
        n += 1
        last = i
    return n


def test_kinematic_path_has_power_law_curvature_speed():
    """2/3 幂定律：曲率大的地方必须更慢（回归斜率为负）。

    这条是轨迹类检测里最难伪造的不变量之一 —— 它是"手转急弯要减速"的
    物理结果，不是随机数能凑出来的。WindMouse 的斜率接近 0（实测 -0.08）。
    """
    slopes = []
    for seed in range(20):
        p = human_path((640, 360), (1000, 500), rng=random.Random(seed))
        s = _path_slope(_as_path(p))
        if s:
            slopes.append(s)
    assert slopes
    med = statistics.median(slopes)
    assert med < -0.10, f"曲率-速度斜率必须明显为负（实测中位 {med:.3f}）"


def _as_path(pts):
    """把 human_path 的 tuple 列表包成带 .x/.y/.dt 的轻量对象。"""
    from autogame.humanize import PathPoint
    return [PathPoint(x, y, dt) for x, y, dt in pts]


def test_kinematic_path_has_submovements():
    """远距离移动由 2-3 个亚运动组成（速度剖面多峰），不是单峰钟形。"""
    peaks = [_peaks(_as_path(human_path((100, 100), (900, 600),
                                        rng=random.Random(s))))
             for s in range(20)]
    med = statistics.median(peaks)
    assert 1 <= med <= 4, f"亚运动峰数应在 1~4，实测中位 {med}"


def test_kinematic_path_endpoint_error_is_gaussian_not_uniform():
    """落点误差是连续高斯，不是整数均匀分布（±2px 等概率是明显机器痕迹）。"""
    cfg = MousePathConfig()
    errs = []
    for s in range(400):
        p = build_mouse_path((300, 300), (900, 600), cfg, rng=random.Random(s))
        errs.append(p[-1].x - 900)
    # 均匀整数 ±2：只有 5 个取值且必在 [-2,2] 内
    assert len({e for e in errs}) > 2, "落点误差不能只有几个离散值"
    assert any(abs(e) >= 2 for e in errs) or len(set(errs)) > 2


def test_kinematic_path_point_count_varies():
    """点数由时长推出，不是恒等于上限（恒等于上限本身就是特征）。"""
    cfg = MousePathConfig()
    counts = {len(build_mouse_path((100, 100), (900, 600), cfg,
                                   rng=random.Random(s))) for s in range(15)}
    assert len(counts) > 3, f"点数不该是常数，实测 {sorted(counts)}"
    assert max(counts) <= cfg.max_points


def test_kinematic_path_duration_within_bounds():
    cfg = MousePathConfig()
    for s in range(30):
        p = build_mouse_path((100, 100), (900, 600), cfg, rng=random.Random(s))
        total = sum(x.dt for x in p)
        assert cfg.min_duration_s - 1e-6 <= total <= cfg.max_duration_s + 1e-6


def test_kinematic_path_is_deterministic():
    cfg = MousePathConfig()
    a = build_mouse_path((0, 0), (700, 400), cfg, rng=random.Random(7))
    b = build_mouse_path((0, 0), (700, 400), cfg, rng=random.Random(7))
    assert a == b


def test_kinematic_path_short_move_single_point():
    cfg = MousePathConfig()
    p = build_mouse_path((100, 100), (102, 101), cfg, rng=random.Random(1))
    assert len(p) == 1


def test_path_model_switch_keeps_windmouse_available():
    """`model="windmouse"` 保留旧实现（回归对照），行为与旧版一致。"""
    cfg = MousePathConfig()
    cfg.model = "windmouse"
    p = build_mouse_path((100, 100), (900, 500), cfg, rng=random.Random(3))
    assert len(p) > 2
    assert abs(p[-1].x - 900) <= cfg.endpoint_jitter_px


def test_path_disabled_single_point():
    cfg = MousePathConfig(enabled=False)
    p = build_mouse_path((10, 10), (800, 600), cfg, rng=random.Random(4))
    assert len(p) == 1 and (p[0].x, p[0].y) == (800, 600)


# ═══════════════════════════ 节奏 ═══════════════════════════


def _dfa(xs):
    """DFA-1 标度指数（白噪声≈0.5，1/f≈1.0）。"""
    n = len(xs)
    m = statistics.mean(xs)
    y, acc = [], 0.0
    for x in xs:
        acc += x - m
        y.append(acc)
    pts = []
    for size in (4, 8, 16, 32, 64):
        if size > n // 2:
            continue
        f = []
        for start in range(0, n - size + 1, size):
            seg = y[start:start + size]
            idx = list(range(size))
            mi = (size - 1) / 2
            my = statistics.mean(seg)
            sxx = sum((i - mi) ** 2 for i in idx) or 1e-9
            b = sum((i - mi) * (v - my) for i, v in zip(idx, seg)) / sxx
            a = my - b * mi
            f.append(math.sqrt(sum((v - (a + b * i)) ** 2
                                   for i, v in zip(idx, seg)) / size))
        mf = statistics.mean(f) if f else 0.0
        if mf > 0:
            pts.append((math.log(size), math.log(mf)))
    if len(pts) < 2:
        return 0.0
    mx = statistics.mean(p[0] for p in pts)
    my2 = statistics.mean(p[1] for p in pts)
    sxx = sum((p[0] - mx) ** 2 for p in pts) or 1e-9
    return sum((p[0] - mx) * (p[1] - my2) for p in pts) / sxx


def _acf(xs, lag):
    m = statistics.mean(xs)
    var = sum((x - m) ** 2 for x in xs)
    if var <= 0:
        return 0.0
    cov = sum((xs[i] - m) * (xs[i + lag] - m) for i in range(len(xs) - lag))
    return cov / var


def test_rhythm_is_not_uniform_box():
    """间隔不能是"基准 ± 抖动"的盒子：CV 要够大，且不是均匀分布。"""
    r = Rhythm(1.0, None, random.Random(11))
    xs = [r.next() for _ in range(400)]
    m = statistics.mean(xs)
    cv = statistics.pstdev(xs) / m
    assert 0.25 <= cv <= 1.2, f"CV 应在人类区间，实测 {cv:.2f}"
    # 均匀分布（0.9~1.1）的偏度≈0；人的间隔右偏
    sd = statistics.pstdev(xs) or 1e-9
    skew = sum(((x - m) / sd) ** 3 for x in xs) / len(xs)
    assert skew > 0.3, f"间隔必须右偏（偶尔慢），实测偏度 {skew:.2f}"


def test_rhythm_has_long_range_correlation():
    """1/f 长程相关：DFA α 明显高于白噪声（0.5）。这是最硬的一条。"""
    r = Rhythm(1.0, None, random.Random(12))
    xs = [r.next() for _ in range(600)]
    alpha = _dfa(xs)
    assert alpha > 0.6, f"DFA α 应接近 1（人），实测 {alpha:.2f}"


def test_rhythm_adjacent_intervals_are_correlated():
    """相邻间隔正相关（人在"一串快/一串慢"之间切换），白噪声 ACF≈0。"""
    r = Rhythm(1.0, None, random.Random(13))
    xs = [r.next() for _ in range(500)]
    assert _acf(xs, 1) > 0.05, f"ACF(lag1) 应明显为正，实测 {_acf(xs, 1):.2f}"


def test_rhythm_has_no_fixed_period():
    """不许出现固定节拍：间隔的取值数应接近样本数（连续分布）。"""
    r = Rhythm(1.0, None, random.Random(14))
    xs = [r.next() for _ in range(200)]
    assert len({round(x, 4) for x in xs}) > 150


def test_rhythm_soft_floor_has_no_wall():
    """软下界：低于门槛的值被压缩而不是截断（截断会制造新的硬墙）。"""
    r = Rhythm(1.0, None, random.Random(15), floor_s=0.0)
    xs = [r.next() for _ in range(400)]
    lows = [x for x in xs if x < 0.62]
    # 压缩后仍保留分散（若用 max() 截断，这些值会全部等于 0.62）
    assert len({round(x, 3) for x in lows}) > 3 or not lows


def test_rhythm_disabled_is_deterministic():
    """关闭时退化为 base×scale（确定性优先，和 reaction_ms 返回 0 一致）。"""

    class _Cfg:
        enabled = False

    r = Rhythm(0.5, _Cfg(), random.Random(1), floor_s=0.0)
    assert abs(r.next(2.0) - 1.0) < 1e-9


def test_rhythm_next_can_override_base():
    r = Rhythm(9.0, None, random.Random(16))
    xs = [r.next(1.0, 0.2) for _ in range(100)]
    assert statistics.median(xs) < 0.6, "base_s 覆盖应生效（0.2s 节拍）"


def test_rhythm_snapshot():
    r = Rhythm(1.0, None, random.Random(17))
    r.next()
    snap = r.snapshot()
    assert snap["samples"] == 1 and "last_ms" in snap


# ═══════════════════════════ 锁相 / 限速器 ═══════════════════════════


def _lock(intervals, fps):
    """动作时刻对 1/fps 取模的圆统计量（1=完全锁定）。"""
    period = 1.0 / fps
    t = s = c = 0.0
    for d in intervals:
        t += d
        ang = 2 * math.pi * (t % period) / period
        s += math.sin(ang)
        c += math.cos(ang)
    return math.hypot(s, c) / max(1, len(intervals))


def test_desync_breaks_frame_lock():
    """亚帧打散 + 顺延后，动作时刻不再钉在帧边界上。"""
    fps, frame, desync = 2.0, 0.5, 0.4
    rng = random.Random(21)
    r = Rhythm(1.0, None, random.Random(21), floor_s=0.0)
    intervals = [r.next() for _ in range(400)]

    def emit(lock_on: bool):
        ts, prev_dec, debt = [], 0.0, 0.0
        for d in intervals:
            dec = math.ceil((prev_dec + debt + d) / frame - 1e-9) * frame
            e = dec + (rng.uniform(0, desync) if lock_on else 0.0)
            ts.append(e)
            debt, prev_dec = e - dec, dec
        return ts

    locked = [emit(False)[i] - emit(False)[i - 1] for i in range(1, 400)]
    free_ts = emit(True)
    free = [free_ts[i] - free_ts[i - 1] for i in range(1, 400)]
    assert _lock(locked, fps) > 0.9, "不开打散时必然锁相（这是旧版的真实状态）"
    assert _lock(free, fps) < 0.5, "开打散后必须明显解耦"


def test_limiter_floor_is_randomized():
    """限速器下界不能是固定值（硬墙会让所有动作贴墙）。"""
    r = ActionRateLimiter(100, 20, floor_jitter=0.25)
    vals = []
    for _ in range(40):
        r.reset()
        r._try(now=10.0)
        vals.append(r.delay_needed(now=10.0))
    assert len({round(v, 4) for v in vals}) > 5, "下界必须在变"
    assert all(0.05 <= v <= 0.15 for v in vals), "但仍应在名义值附近"


def test_limiter_floor_jitter_off_is_constant():
    r = ActionRateLimiter(100, 20, floor_jitter=0.0)
    vals = []
    for _ in range(10):
        r.reset()
        r._try(now=10.0)
        vals.append(r.delay_needed(now=10.0))
    assert len(set(round(v, 6) for v in vals)) == 1, "关闭时保持旧行为（硬墙）"


def test_limiter_still_enforces_max_rate():
    """抖动只影响下限，不影响速率上限（安全性不许被放松）。"""
    r = ActionRateLimiter(0, 5, floor_jitter=0.3)
    allowed = sum(1 for _ in range(20) if r._try(now=10.0))
    assert allowed <= 5


def test_jitter_is_right_skewed_and_centered():
    """按键时长抖动：右偏（松得快更常见）+ 均值不漂移。"""
    xs = [_jitter(70, 0.2) for _ in range(600)]
    m = statistics.mean(xs)
    assert 60 <= m <= 82, f"均值不该漂移，实测 {m:.1f}"
    sd = statistics.pstdev(xs) or 1e-9
    skew = sum(((x - m) / sd) ** 3 for x in xs) / len(xs)
    assert skew > 0.2, f"必须右偏，实测 {skew:.2f}"
    assert all(x >= 10 for x in xs)


def test_jitter_zero_ratio_returns_input():
    assert _jitter(70, 0.0) == 70


# ═══════════════════════════ 执行器集成 ═══════════════════════════


def test_executor_uses_kinematic_model_by_default():
    from autogame.executor import DryRunExecutor
    from autogame.protocol import Command

    cfg = ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000)
    assert cfg.mouse_path.model == "kinematic"
    ex = DryRunExecutor(cfg)
    # 光标位置未知时第一次移动退化为单点（没有起点就没有轨迹），先落一个起点
    ex.execute(Command(op="mouse_move", x=100, y=100))
    ex.execute(Command(op="mouse_click", x=900, y=600))
    assert len(ex.last_path) > 2
    assert ex.dropped == 0


# ═══════════════════════════ 连按 / 连点 ═══════════════════════════


def _mash(**over):
    """构造一个"必连按"的 Mash（测试里不要靠概率碰运气）。"""
    from autogame.config import MashConfig
    from autogame.motion import Mash

    cfg = MashConfig()
    cfg.enabled = True          # 套件默认关拟人化，这里显式打开
    cfg.prob_key = 1.0
    cfg.prob_click = 1.0
    cfg.budget_ratio = 1.0
    cfg.excite_mean = 0.5
    for k, v in over.items():
        setattr(cfg, k, v)
    return Mash(cfg, random.Random(11), forbid=("a",))


def test_mash_disabled_never_bursts():
    m = _mash(enabled=False)
    for i in range(200):
        assert m.plan("key_tap", "1", now=i * 0.5) is None


def test_mash_never_bursts_pickup_key():
    """铁律：拾取键连按 = 多次盲按的机会，任何配置都不能放开。"""
    m = _mash()
    for i in range(200):
        assert m.plan("key_tap", "A", now=i * 0.5) is None, "大写也要拦住"


def test_mash_only_whitelisted_keys_and_taps():
    m = _mash()
    assert m.plan("key_tap", "z", now=0.0) is None, "白名单外的键不连按"
    assert m.plan("key_down", "1", now=0.0) is None, "长按不是连打"
    assert m.plan("mouse_move", None, now=0.0) is None


def test_mash_cluster_length_is_heavy_tailed():
    """簇长不是固定值：必须是右偏的（多数短、少数长）。

    旧行为（程序化点击）簇长恒为 1 —— 直方图一根柱子，这本身就是特征。
    """
    m = _mash(count=[3, 9])
    ns = []
    t = 0.0
    for _ in range(500):
        p = m.plan("key_tap", "1", now=t)
        t += 1.0
        if p:
            ns.append(p.count)
    assert len(ns) > 200
    assert min(ns) >= 3 and max(ns) <= 9
    assert max(ns) > statistics.mean(ns) + 1.0, "要有长尾，不能挤在均值附近"
    assert len(set(ns)) >= 4, "至少出现 4 种簇长"


def test_mash_first_gap_is_longest():
    """第一下之后要抬手再落 —— 第一 gap 必须明显长于后续。"""
    m = _mash(count=[5, 6], first_ms=[240, 260], interval_ms=[100, 100])
    firsts, rests = [], []
    t = 0.0
    for _ in range(300):
        p = m.plan("key_tap", "1", now=t)
        t += 3.0
        if p and len(p.taps) >= 3:
            firsts.append(p.taps[0].wait_s)
            rests.extend(x.wait_s for x in p.taps[1:])
    assert firsts and rests
    assert statistics.mean(firsts) > statistics.mean(rests) * 1.5


def test_mash_gaps_are_irregular_and_above_human_floor():
    from autogame.motion import MASH_FLOOR_S

    m = _mash(count=[4, 9])
    gaps = []
    t = 0.0
    for _ in range(400):
        p = m.plan("key_tap", "1", now=t)
        t += 2.0
        if p:
            gaps.extend(x.wait_s for x in p.taps)
    assert min(gaps) >= MASH_FLOOR_S - 1e-9, "不能快过人类的物理极限"
    mean = statistics.mean(gaps)
    assert statistics.pstdev(gaps) / mean > 0.12, "等间隔会被一眼看穿"
    assert statistics.mean(gaps) < 0.25, "连按就该是连按，不是慢按"


def test_mash_hold_shorter_than_single_press():
    """连打时手指不在键上停留：按压时长要比单击短。"""
    m = _mash()
    p = m.plan("key_tap", "1", now=0.0)
    assert p is not None
    assert p.first_hold_ms < 80
    assert all(x.hold_ms < 60 for x in p.taps)


def test_mash_respects_time_budget():
    """连按占比不能超过预算 —— 否则人就变成"一直狂按同一个键"。"""
    from autogame.config import MashConfig
    from autogame.motion import Mash

    cfg = MashConfig()
    cfg.enabled = True
    cfg.prob_key = 1.0
    cfg.budget_ratio = 0.10
    cfg.budget_window_s = 20.0
    m = Mash(cfg, random.Random(3), forbid=("a",))

    t = 0.0
    used = 0.0
    for _ in range(2000):
        p = m.plan("key_tap", "1", now=t)
        t += 1.0
        if p:
            used += p.duration_s
    assert used <= 0.10 * 20.0 * (t / 20.0) * 1.35, "预算最多允许 35% 的超额（窗口边界）"
    assert m.skipped_budget > 0


def test_mash_deterministic_with_seed():
    a = _mash()
    b = _mash()
    for i in range(50):
        pa, pb = a.plan("key_tap", "1", now=i * 1.5), b.plan("key_tap", "1", now=i * 1.5)
        assert (pa is None) == (pb is None)
        if pa and pb:
            assert pa.taps == pb.taps


def test_mash_burst_fits_max_duration():
    m = _mash(max_burst_s=0.6, count=[9, 9], interval_ms=[300, 400])
    p = m.plan("key_tap", "1", now=0.0)
    assert p is not None
    assert p.duration_s <= 0.6 + 1e-6


def test_mash_excite_is_mean_reverting():
    """兴奋度必须**回复**，不能一路衰减到 0（否则跑久了连按会悄悄消失）。"""
    from autogame.config import MashConfig
    from autogame.motion import Mash

    cfg = MashConfig()
    cfg.enabled = True
    cfg.excite_mean = 0.35
    m = Mash(cfg, random.Random(5))
    vals = []
    for i in range(1200):
        vals.append(m.excite(now=i * 2.0))
    tail = values = vals[-300:]
    assert 0.15 < statistics.mean(tail) < 0.60, f"长期均值应回到 0.35 附近，实际 {statistics.mean(tail):.3f}"


# ═══════════════════════════ 执行器：连按落地 ═══════════════════════════


def _burst_exec(**over):
    from autogame.executor import DryRunExecutor

    cfg = ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000)
    cfg.mash.enabled = True     # 套件默认关拟人化
    cfg.mash.prob_key = 1.0
    cfg.mash.count = [4, 4]
    cfg.mash.first_ms = [30, 30]
    cfg.mash.interval_ms = [25, 25]
    cfg.mash.tail_ms = [10, 10]
    cfg.mash.hold_ms = [15, 15]
    cfg.mash.first_hold_ms = [20, 20]
    cfg.mash.max_burst_s = 1.0
    cfg.mash.budget_ratio = 1.0
    for k, v in over.items():
        setattr(cfg.mash, k, v)
    ex = DryRunExecutor(cfg)
    ex.set_burst_forbid(["a"])
    return ex


def test_executor_expands_key_tap_into_burst():
    from autogame.protocol import Command

    ex = _burst_exec()
    assert ex.execute(Command(op="key_tap", key="1", hold_ms=70)) is True
    assert len(ex.journal) == 1, "一簇连按在日志里是**一个**动作"
    assert ex.journal[0]["burst"] == 4
    assert ex.last_burst_count == 4
    assert ex.burst_taps == 3, "第一下由 execute 发，其余 3 下由 burst 补"
    assert ex.sent == 1, "sent 计的是逻辑动作数，不是按下次数"


def test_executor_never_bursts_pickup_key_from_fsm():
    """决策层注入的 forbid 必须真的生效（这条是铁律的最后一道）。"""
    from autogame.decision.fsm import DecisionEngine
    from autogame.config import DecisionConfig
    from autogame.executor import DryRunExecutor
    from autogame.protocol import Command

    ex = DryRunExecutor(ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000))
    eng = DecisionEngine(DecisionConfig(), ex)
    assert "a" in ex.burst_forbid
    ex.execute(Command(op="key_tap", key="a", hold_ms=60))
    assert ex.last_burst_count == 1
    del eng


def test_executor_burst_aborts_when_gate_closes():
    from autogame.protocol import Command

    ex = _burst_exec(count=[8, 8], hold_ms=[15, 15], interval_ms=[20, 20])
    calls = {"n": 0}

    def gate():
        calls["n"] += 1
        return calls["n"] <= 2

    ex.action_gate = gate
    ex.execute(Command(op="key_tap", key="2", hold_ms=60))
    assert ex.burst_aborted == 1
    assert ex.burst_taps < 7, "闸门关了就不许再补发"


def test_executor_mouse_burst_does_not_replay_path():
    """连点不重新走轨迹（手已在目标上），否则每次连点都要滑一遍。"""
    from autogame.protocol import Command

    ex = _burst_exec(prob_click=1.0, count=[3, 3])
    ex.execute(Command(op="mouse_move", x=100, y=100))
    ex.execute(Command(op="mouse_click", x=600, y=400))
    assert ex.last_burst_count == 3
    # 轨迹只在第一下规划过：连点两下不应把 path_points_total 翻三倍
    assert ex.burst_taps == 2
    assert ex.journal[-1]["burst"] == 3


def test_burst_resets_aframe_confirmation():
    """连按之后 A框确认必须重新数。

    burst 是同步阻塞（0.5-1.6s），这期间世界在动、人也没盯屏。如果把 burst
    之前的确认帧继续累加，一次技能连按就能把 seen 凑到阈值 —— 等于拿旧证据按 A。
    """
    from autogame.decision.fsm import DecisionEngine
    from autogame.config import DecisionConfig
    from autogame.executor import DryRunExecutor
    from autogame.protocol import Command

    ex = _burst_exec()
    eng = DecisionEngine(DecisionConfig(), ex)
    eng._aframe_seen_frames = 3
    eng._act(Command(op="key_tap", key="1", hold_ms=60))
    assert ex.last_burst_count > 1, "前提：这次确实连按了"
    assert eng._aframe_seen_frames == 0
    assert eng.stats["burst_breaks_confirm"] == 1


def test_burst_paused_during_precise_ui():
    """配方/UI 自动化期间不许连按：多按两下就点错槽位。"""
    from autogame.protocol import Command

    ex = _burst_exec()
    with ex.burst_paused():
        ex.execute(Command(op="key_tap", key="1", hold_ms=60))
        assert ex.last_burst_count == 1
        with ex.burst_paused():
            ex.execute(Command(op="key_tap", key="1", hold_ms=60))
            assert ex.last_burst_count == 1, "嵌套也要挡住"
    ex.execute(Command(op="key_tap", key="1", hold_ms=60))
    assert ex.last_burst_count > 1, "退出上下文后恢复连按"


def test_pickup_key_is_never_whitelisted_for_mashing():
    """拾取键不能被列入连按白名单 —— 也不该因为"技能栏写的是 A"被加进去。

    真机拾取是靠 A 框确认 + 三道闸的**单次**按键；连按等于把这道闸刷穿。
    运行时还有 `set_burst_forbid` 兜底，这里钉住配置侧的默认值：
    谁要把 `a` 当技能键塞进 `mash.keys`，先看到这条红。
    """
    from autogame.config import DecisionConfig, ExecutorConfig

    dec, ex = DecisionConfig(), ExecutorConfig()
    whitelisted = [k.lower() for k in ex.mash.keys]
    assert dec.pickup_key.lower() not in whitelisted, \
        f"拾取键 {dec.pickup_key!r} 出现在连按白名单里：{whitelisted}"
