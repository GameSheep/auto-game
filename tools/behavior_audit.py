"""行为审计 —— 用反作弊侧会算的统计量，检验"像不像人"。

═══════════════════════════════════════════════════════════════════════
这个工具回答两个问题，不给结论性形容词
═══════════════════════════════════════════════════════════════════════
1. **鼠标轨迹**：一次移动的形状是否满足人类运动学不变量？
   - 曲率-速度关系（2/3 幂定律）：log v 对 log κ 的回归斜率应 ≈ -1/3；
   - 速度剖面峰数：远距离移动应有 2-3 个峰（亚运动），不是单峰钟形；
   - 路径效率 / 最大横向偏离：真人 0.93-0.99、偏离随距离亚线性；
   - 采样间隔的取值数：真实鼠标上报间隔是连续量，不该只有几个离散值。

2. **操作节奏**：一串动作的间隔序列有没有"程序味"？
   - CV / 偏度：人右偏（长尾），均匀抖动是盒状；
   - ACF(lag1..k)：固定节拍会在基频处出现尖峰；
   - DFA α：长程相关指数，人 0.7-1.05，白噪声 ≈0.5；
   - 最小值 / 名义值：限速器硬墙会让所有间隔贴在同一个下界；
   - 锁相强度 R：动作时刻对 1/fps 取模的圆统计量，固定节拍会接近 1。

用法：
    python tools/behavior_audit.py                # 新旧对比表
    python tools/behavior_audit.py --json out.json
    python tools/behavior_audit.py --fps 6        # 按实际采集帧率算锁相

⚠️ 这不是"安全证明"。所有在这里变好看的指标，都只说明**单条特征更像人**；
一旦有人盯着摄像头看十分钟，程序驱动这件事本身是藏不住的。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autogame.config import MousePathConfig  # noqa: E402
from autogame.humanize import build_mouse_path  # noqa: E402
from autogame.motion import Rhythm  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
# 统计工具（都是标准做法，不引入依赖）
# ═══════════════════════════════════════════════════════════════════════


def _cv(xs: list[float]) -> float:
    m = statistics.mean(xs) if xs else 0.0
    return (statistics.pstdev(xs) / m) if m else 0.0


def _skew(xs: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    m = statistics.mean(xs)
    sd = statistics.pstdev(xs) or 1e-9
    return sum(((x - m) / sd) ** 3 for x in xs) / n


def _kurt(xs: list[float]) -> float:
    n = len(xs)
    if n < 4:
        return 0.0
    m = statistics.mean(xs)
    sd = statistics.pstdev(xs) or 1e-9
    return sum(((x - m) / sd) ** 4 for x in xs) / n - 3.0


def _acf(xs: list[float], lag: int) -> float:
    n = len(xs)
    if n <= lag + 2:
        return 0.0
    m = statistics.mean(xs)
    var = sum((x - m) ** 2 for x in xs)
    if var <= 0:
        return 0.0
    cov = sum((xs[i] - m) * (xs[i + lag] - m) for i in range(n - lag))
    return cov / var


def _dfa_alpha(xs: list[float], sizes: tuple[int, ...] = (4, 8, 16, 32, 64)) -> float:
    """DFA-1：去趋势波动分析的标度指数。

    白噪声（均匀抖动）α≈0.5；1/f 噪声（人类节奏）α≈1.0。
    这条是"间隔高度标准化"最硬的反证 —— 它看的是**长程结构**，
    不是单个间隔的大小，加噪声也伪造不出来（要么真有，要么没有）。
    """
    n = len(xs)
    if n < 64:
        return 0.0
    m = statistics.mean(xs)
    y: list[float] = []
    acc = 0.0
    for x in xs:
        acc += x - m
        y.append(acc)

    pts: list[tuple[float, float]] = []
    for size in sizes:
        if size > n // 2:
            continue
        fluct = []
        for start in range(0, n - size + 1, size):
            seg = y[start:start + size]
            # 线性去趋势
            idx = list(range(size))
            mi = (size - 1) / 2.0
            my = statistics.mean(seg)
            sxx = sum((i - mi) ** 2 for i in idx) or 1e-9
            sxy = sum((i - mi) * (v - my) for i, v in zip(idx, seg))
            b = sxy / sxx
            a = my - b * mi
            resid = [v - (a + b * i) for i, v in zip(idx, seg)]
            fluct.append(math.sqrt(sum(r * r for r in resid) / size))
        f = statistics.mean(fluct) if fluct else 0.0
        if f > 0:
            pts.append((math.log(size), math.log(f)))
    if len(pts) < 2:
        return 0.0
    mx = statistics.mean(p[0] for p in pts)
    my2 = statistics.mean(p[1] for p in pts)
    sxx = sum((p[0] - mx) ** 2 for p in pts) or 1e-9
    sxy = sum((p[0] - mx) * (p[1] - my2) for p in pts)
    return sxy / sxx


def _phase_lock(intervals: list[float], fps: float) -> float:
    """锁相强度 R ∈ [0,1]：动作时刻对 1/fps 取模后的圆统计量。

    决策是帧驱动的：如果节拍与帧周期有整数关系，动作会**永远落在同一相位**，
    R 接近 1。R 接近 0 才说明动作时刻与采集时钟无关。
    这是"操作间隔高度标准化"里最隐蔽的一条，因为单看间隔值看不出来。
    """
    if not intervals or fps <= 0:
        return 0.0
    period = 1.0 / fps
    t = 0.0
    s = c = 0.0
    for d in intervals:
        t += d
        ang = 2.0 * math.pi * (t % period) / period
        s += math.sin(ang)
        c += math.cos(ang)
    n = len(intervals)
    return math.hypot(s, c) / max(1, n)


# ═══════════════════════════════════════════════════════════════════════
# 轨迹指标
# ═══════════════════════════════════════════════════════════════════════


def path_metrics(path, target: tuple[float, float]) -> dict:
    """一条轨迹的运动学指标。"""
    pts = [(p.x, p.y) for p in path]
    if len(pts) < 3:
        return {}
    # 路程与效率
    seg = [math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
           for i in range(1, len(pts))]
    total = sum(seg) or 1e-9
    straight = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]) or 1e-9
    efficiency = min(1.0, straight / total)

    # 最大横向偏离（相对起点→终点直线）
    x0, y0 = pts[0]
    x1, y1 = pts[-1]
    dx, dy = x1 - x0, y1 - y0
    nrm = math.hypot(dx, dy) or 1e-9
    max_dev = max(abs(dy * (px - x0) - dx * (py - y0)) / nrm for px, py in pts)

    # 速度剖面：用"每点位移/该点 dt"，**先平滑再找峰**。
    # 不平滑的话，信号相关噪声会让每个点都变成一个小峰，"亚运动数"就失去意义。
    raw_speed = [seg[i - 1] / max(1e-6, path[i].dt) for i in range(1, len(path))]
    if len(raw_speed) >= 5:
        speed = [statistics.mean(raw_speed[max(0, i - 2):i + 3])
                 for i in range(len(raw_speed))]
    else:
        speed = raw_speed
    peaks = 0
    if speed:
        top = max(speed) or 1e-9
        last_peak = -99
        for i in range(2, len(speed) - 2):
            if speed[i] < 0.45 * top:
                continue
            if not (speed[i] >= speed[i - 1] and speed[i] > speed[i + 1]):
                continue
            # 峰间距至少 3 个采样点：真人不会在 1 个点上"再峰一次"
            if i - last_peak < 3:
                continue
            peaks += 1
            last_peak = i

    # 2/3 幂律：log v 对 log κ 的斜率（曲率大 → 慢）
    pairs: list[tuple[float, float]] = []
    for i in range(1, len(pts) - 1):
        x1_, y1_ = pts[i - 1]
        x2_, y2_ = pts[i]
        x3_, y3_ = pts[i + 1]
        a = math.hypot(x2_ - x1_, y2_ - y1_)
        b = math.hypot(x3_ - x2_, y3_ - y2_)
        c = math.hypot(x3_ - x1_, y3_ - y1_)
        if a * b * c < 1e-9:
            continue
        area2 = abs((x2_ - x1_) * (y3_ - y1_) - (x3_ - x1_) * (y2_ - y1_))
        kappa = 2.0 * area2 / (a * b * c)
        v = seg[i - 1] / max(1e-6, path[i].dt)
        if kappa > 1e-5 and v > 1.0:
            pairs.append((math.log(kappa), math.log(v)))
    slope = 0.0
    if len(pairs) >= 5:
        mx = statistics.mean(p[0] for p in pairs)
        my = statistics.mean(p[1] for p in pairs)
        sxx = sum((p[0] - mx) ** 2 for p in pairs) or 1e-9
        sxy = sum((p[0] - mx) * (p[1] - my) for p in pairs)
        slope = sxy / sxx

    return {
        "points": len(pts),
        "duration": sum(p.dt for p in path),
        "efficiency": efficiency,
        "max_dev": max_dev,
        "peaks": peaks,
        "power_slope": slope,
        "distinct_dt": len({round(p.dt, 4) for p in path if p.dt > 0}),
        "endpoint_err": math.hypot(pts[-1][0] - target[0], pts[-1][1] - target[1]),
    }


# ═══════════════════════════════════════════════════════════════════════
# 两套生成器（旧 vs 新）
# ═══════════════════════════════════════════════════════════════════════


def collect_paths(kind: str, n: int, rng: random.Random) -> list[dict]:
    cfg = MousePathConfig()
    cfg.model = "windmouse" if kind == "old" else "kinematic"
    # 旧版 max_points 是 40；这里对新版保留 64，避免"改默认值时顺便偷分"
    out: list[dict] = []
    for i in range(n):
        dist = rng.uniform(120, 900)
        ang = rng.uniform(0, 2 * math.pi)
        end = (640 + dist * math.cos(ang), 360 + dist * math.sin(ang))
        start = (640.0, 360.0)
        p = build_mouse_path((int(start[0]), int(start[1])),
                             (int(end[0]), int(end[1])), cfg, rng=rng)
        m = path_metrics(p, end)
        if m:
            out.append(m)
    return out


def _emit_times(intervals: list[float], fps: float, desync_ms: float,
                rng: random.Random) -> list[float]:
    """期望间隔 → 实际发出时刻（含帧量化 + 亚帧打散 + 顺延补偿）。

    三段都绕不开，少模拟任何一段对比都是自欺：
      ① **帧量化**：决策是帧驱动的，动作最早只能落在该帧到达的时刻；
      ② **亚帧打散**：动作前再等 U(0, desync)，把相位从帧边界挪开。
         没有 ②，fps=2 时所有间隔都是 500ms 的整数倍；
      ③ **顺延**：② 的延迟必须计入下一拍（`_desync_debt`）。否则前一个
         动作拖后、后一个抽到 0 延迟，两拍之间就只剩一帧 —— 真人不会在
         166ms 内连续两次点地面。
    """
    frame = 1.0 / max(0.1, fps)
    out: list[float] = []
    prev_dec = 0.0
    debt = 0.0
    for d in intervals:
        target = prev_dec + debt + d
        # 决策循环按帧唤醒：只能在这一帧边界上判断"时间到了没有"
        dec = math.ceil(target / frame - 1e-9) * frame
        emit = dec + rng.uniform(0.0, desync_ms / 1000.0)
        out.append(emit)
        debt = emit - dec
        prev_dec = dec
    return out


def _diffs(ts: list[float]) -> list[float]:
    return [ts[i] - ts[i - 1] for i in range(1, len(ts))]


def collect_old_intervals(n: int, base: float, rng: random.Random,
                          fps: float = 2.0, desync_ms: float = 0.0) -> list[float]:
    """旧行为：固定节拍 × uniform(0.9,1.1)，**帧驱动**（每帧重抽直到跨过阈值）。

    动作时刻被钉在帧边界上。fps=2 时这一条比"均匀抖动"致命得多：
    间隔只可能是 500ms 或 1000ms 两个值。
    """
    intervals = [base * rng.uniform(0.9, 1.1) for _ in range(n)]
    emits = _emit_times(intervals, fps, desync_ms, rng)
    return _diffs(emits)


def collect_new_intervals(n: int, base: float, rng: random.Random,
                          fps: float = 2.0, desync_ms: float = 400.0) -> list[float]:
    """新行为：Rhythm 生成的连续间隔，同样经过帧量化 + 亚帧打散 + 顺延。"""
    r = Rhythm(base, None, rng, floor_s=0.025)
    intervals = [r.next() for _ in range(n)]
    emits = _emit_times(intervals, fps, desync_ms, rng)
    return _diffs(emits)


# ═══════════════════════════════════════════════════════════════════════
# 连按结构（簇长 / 簇内间隔 / 簇的出现过程）
# ═══════════════════════════════════════════════════════════════════════


def collect_mash(kind: str, n_actions: int, seed: int,
                 base: float = 1.0) -> dict:
    """模拟 `n_actions` 次"按技能键"，统计连按的结构。

    旧行为 = **没有连按这个概念**：每个动作都是一次孤立按键，簇长恒为 1。
    这本身就是特征 —— 真实日志里按键的簇长分布是重尾的。
    """
    if kind == "old":
        return {"counts": [1] * n_actions, "isis": [], "used": 0.0,
                "span": n_actions * base, "clusters": 0, "starts": []}

    from autogame.config import MashConfig
    from autogame.motion import Mash

    m = Mash(MashConfig(), random.Random(seed), forbid=("a",))
    counts: list[int] = []
    isis: list[float] = []
    starts: list[float] = []
    used = 0.0
    t = 0.0
    for _ in range(n_actions):
        p = m.plan("key_tap", "1", now=t)
        # 连按会**阻塞**决策线程，所以下一次"按技能"不可能在之前开始
        cost = p.duration_s if p else 0.0
        if p:
            counts.append(p.count)
            isis.extend(x.wait_s for x in p.taps)
            starts.append(t)
            used += cost
        t += base + cost
    return {"counts": counts, "isis": isis, "used": used, "span": t,
            "clusters": len(counts), "starts": starts}


def mash_metrics(d: dict) -> dict:
    counts = d["counts"] or [1]
    isis = d["isis"]
    starts = d["starts"]
    gaps = [starts[i] - starts[i - 1] for i in range(1, len(starts))]
    return {
        "count_mean": statistics.mean(counts),
        "count_max": max(counts),
        "count_kinds": len(set(counts)),
        "isi_cv": _cv(isis) if len(isis) > 2 else 0.0,
        "used_ratio": d["used"] / max(1e-9, d["span"]),
        # 簇出现过程的过度离散：泊松过程方差/均值≈1，人做任务会成堆出现
        "cluster_disp": (statistics.pvariance(gaps) / max(1e-9, statistics.mean(gaps)))
                        if len(gaps) > 3 else float("nan"),
    }


# ═══════════════════════════════════════════════════════════════════════
# 报告
# ═══════════════════════════════════════════════════════════════════════

#: 人类参考带（来自运动控制文献的量级，不是精确阈值）
HUMAN = {
    "power_slope": (-0.45, -0.22),
    "peaks": (1.0, 3.2),
    "efficiency": (0.93, 0.995),
    "distinct_dt_ratio": (0.5, 1.0),
    "cv": (0.35, 0.95),
    "skew": (0.3, 2.5),
    "acf1": (0.10, 0.55),
    "dfa": (0.65, 1.10),
    "min_over_base": (0.55, 0.95),
    "lock_R": (0.0, 0.35),
}


def _fmt(v: float, nd: int = 2) -> str:
    return f"{v:.{nd}f}"


def _mark(value: float, band: tuple[float, float]) -> str:
    lo, hi = band
    return "✓" if lo <= value <= hi else "✗"


def main() -> int:
    ap = argparse.ArgumentParser(description="行为审计（轨迹 + 节奏）")
    ap.add_argument("--paths", type=int, default=200, help="轨迹样本数")
    ap.add_argument("--intervals", type=int, default=800, help="间隔样本数")
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--fps", type=float, default=6.0, help="采集帧率（算锁相用）")
    ap.add_argument("--base", type=float, default=1.0, help="名义节拍（秒）")
    ap.add_argument("--desync", type=float, default=400.0,
                    help="亚帧打散上限（毫秒），应与 action_desync_ms 一致")
    ap.add_argument("--actions", type=int, default=600,
                    help="连按样本：模拟按多少次技能键")
    ap.add_argument("--json", type=str, default="", help="把指标写到 json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    old_p = collect_paths("old", args.paths, rng)
    rng = random.Random(args.seed)
    new_p = collect_paths("new", args.paths, rng)

    rng = random.Random(args.seed)
    old_i = collect_old_intervals(args.intervals, args.base, rng,
                                  fps=args.fps, desync_ms=0.0)
    rng = random.Random(args.seed)
    new_i = collect_new_intervals(args.intervals, args.base, rng,
                                  fps=args.fps, desync_ms=args.desync)

    def agg(rows: list[dict], key: str) -> float:
        return statistics.median([r[key] for r in rows if key in r])

    def ratio(rows: list[dict]) -> float:
        vals = [r["distinct_dt"] / max(1, r["points"]) for r in rows if r.get("points")]
        return statistics.median(vals) if vals else 0.0

    rows = []
    rows.append(("曲率-速度斜率", _fmt(agg(old_p, "power_slope"), 3),
                 _fmt(agg(new_p, "power_slope"), 3), "-0.33", "2/3 幂定律"))
    rows.append(("速度峰数(中位)", _fmt(agg(old_p, "peaks"), 1),
                 _fmt(agg(new_p, "peaks"), 1), "1~3", "亚运动结构"))
    rows.append(("路径效率", _fmt(agg(old_p, "efficiency"), 3),
                 _fmt(agg(new_p, "efficiency"), 3), "0.93~0.99", "不该是直线也不该乱飘"))
    rows.append(("最大横向偏离px", _fmt(agg(old_p, "max_dev"), 1),
                 _fmt(agg(new_p, "max_dev"), 1), "4~40", "弯曲量级"))
    rows.append(("点数", _fmt(agg(old_p, "points"), 0),
                 _fmt(agg(new_p, "points"), 0), "随时长变", "不该恒等于上限"))
    rows.append(("dt 不同取值比", _fmt(ratio(old_p), 2), _fmt(ratio(new_p), 2),
                 "0.5~1.0", "间隔量化痕迹"))
    rows.append(("落点误差px", _fmt(agg(old_p, "endpoint_err"), 2),
                 _fmt(agg(new_p, "endpoint_err"), 2), "0~2.5", "高斯而非均匀"))

    def stat(xs: list[float]) -> dict:
        return {
            "cv": _cv(xs),
            "skew": _skew(xs),
            "kurt": _kurt(xs),
            "acf1": _acf(xs, 1),
            "acf2": _acf(xs, 2),
            "dfa": _dfa_alpha(xs),
            "min_over_base": min(xs) / args.base,
            "lock": _phase_lock(xs, args.fps),
            "distinct": len({round(x, 3) for x in xs}),
        }

    so, sn = stat(old_i), stat(new_i)
    rows += [
        ("间隔 CV", _fmt(so["cv"], 2), _fmt(sn["cv"], 2), "0.35~0.95", "离散程度"),
        ("间隔偏度", _fmt(so["skew"], 2), _fmt(sn["skew"], 2), "0.3~2.5", "右偏=偶尔慢"),
        ("间隔峰度", _fmt(so["kurt"], 2), _fmt(sn["kurt"], 2), ">1", "尖峰+长尾"),
        ("ACF(lag1)", _fmt(so["acf1"], 2), _fmt(sn["acf1"], 2), "0.10~0.55", "相邻间隔相关"),
        ("DFA α", _fmt(so["dfa"], 2), _fmt(sn["dfa"], 2), "0.65~1.10", "长程相关（最硬）"),
        ("最小/名义", _fmt(so["min_over_base"], 2), _fmt(sn["min_over_base"], 2),
         "0.55~0.95", "硬墙→斜坡"),
        ("锁相 R", _fmt(so["lock"], 2), _fmt(sn["lock"], 2), "0.00~0.35", "与帧率解耦"),
        ("间隔取值数", _fmt(so["distinct"], 0), _fmt(sn["distinct"], 0),
         f"≈{len(old_i)}", "不该是有限集合"),
    ]

    mo = mash_metrics(collect_mash("old", args.actions, args.seed))
    mn = mash_metrics(collect_mash("new", args.actions, args.seed))
    rows += [
        ("簇长均值", _fmt(mo["count_mean"], 1), _fmt(mn["count_mean"], 1),
         "3~9", "一簇按几下"),
        ("簇长种类", _fmt(mo["count_kinds"], 0), _fmt(mn["count_kinds"], 0),
         ">3", "旧版只有「1」这一种"),
        ("簇内间隔 CV", _fmt(mo["isi_cv"], 2), _fmt(mn["isi_cv"], 2),
         "0.15~0.50", "连打不是等间隔"),
        ("连按时间占比", _fmt(mo["used_ratio"], 2), _fmt(mn["used_ratio"], 2),
         "0.10~0.40", "狂按也该有度"),
        ("簇出现离散度", "—", _fmt(mn["cluster_disp"], 2),
         ">1.2", "成堆出现，不是泊松"),
    ]

    w0, w1, w2, w3 = 18, 14, 14, 26
    print()
    print(f"轨迹样本 {len(new_p)} 条 / 节奏样本 {args.intervals} 个 / 连按 {args.actions} 次 "
          f"（fps={args.fps:g}，名义节拍 {args.base:g}s，亚帧打散 {args.desync:g}ms，seed={args.seed}）")
    print("─" * (w0 + w1 + w2 + w3 + 8))
    print(f"{'指标':<{w0}}{'旧(windmouse/固定)':<{w1}}{'新(运动学/节奏)':<{w2}}"
          f"{'人类参考':<{w3}}说明")
    print("─" * (w0 + w1 + w2 + w3 + 8))
    for name, a, b, human, note in rows:
        print(f"{name:<{w0}}{a:<{w1}}{b:<{w2}}{human:<{w3}}{note}")
    print("─" * (w0 + w1 + w2 + w3 + 8))

    # 达标情况
    checks = [
        ("曲率-速度斜率", agg(new_p, "power_slope"), HUMAN["power_slope"]),
        ("速度峰数", agg(new_p, "peaks"), HUMAN["peaks"]),
        ("路径效率", agg(new_p, "efficiency"), HUMAN["efficiency"]),
        ("dt 取值比", ratio(new_p), HUMAN["distinct_dt_ratio"]),
        ("间隔 CV", sn["cv"], HUMAN["cv"]),
        ("间隔偏度", sn["skew"], HUMAN["skew"]),
        ("ACF(lag1)", sn["acf1"], HUMAN["acf1"]),
        ("DFA α", sn["dfa"], HUMAN["dfa"]),
        ("最小/名义", sn["min_over_base"], HUMAN["min_over_base"]),
        ("锁相 R", sn["lock"], HUMAN["lock_R"]),
        ("簇长均值", mn["count_mean"], (3.0, 9.0)),
        ("簇内间隔 CV", mn["isi_cv"], (0.15, 0.50)),
        ("连按时间占比", mn["used_ratio"], (0.10, 0.40)),
        ("簇出现离散度", mn["cluster_disp"], (1.2, 12.0)),
    ]
    ok = sum(1 for _, v, b in checks if b[0] <= v <= b[1])
    print(f"\n落在人类参考带内：{ok}/{len(checks)}")
    for name, v, b in checks:
        if not (b[0] <= v <= b[1]):
            print(f"  · {name} = {v:.3f} 不在 [{b[0]}, {b[1]}]")

    if args.json:
        payload = {
            "path_old": {k: agg(old_p, k) for k in
                         ("points", "efficiency", "max_dev", "peaks", "power_slope")},
            "path_new": {k: agg(new_p, k) for k in
                         ("points", "efficiency", "max_dev", "peaks", "power_slope")},
            "interval_old": so,
            "interval_new": sn,
            "mash_old": mo,
            "mash_new": mn,
            "human_band": {k: list(v) for k, v in HUMAN.items()},
        }
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {args.json}")

    print("\n⚠️ 这些指标只说明**单条特征更像人**。轨迹可以骗过统计模型，")
    print("   骗不过「盯屏十分钟再人工复核」。风险不为零，别当安全方案。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
