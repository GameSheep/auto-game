"""人类运动学 —— 轨迹形状与动作节奏的生成器（纯计算，不碰业务）。

═══════════════════════════════════════════════════════════════════════
先说要对抗的是什么（不然改完只是"换一种假"）
═══════════════════════════════════════════════════════════════════════
一、轨迹类检测（看**一次**移动的形状）
  1. 两点一线 + 匀速：最原始，现在没人这么写；
  2. **WindMouse 系**：公开算法，反作弊侧有现成指纹 —— 它的单步步长序列
     自相关很高、横向偏移与距离几乎线性、步长上限被 clip 所以出现"等幅平台"。
     用知名脚本库的默认参数，等于自带签名；
  3. 真正难伪造的是**运动学不变量**（这些是人的神经-肌肉特性，不是随机数）：
     - Fitts 律  MT = a + b·log2(D/W + 1)：距离-耗时是**对数**关系。
       用"距离 ÷ 固定速度"得到的是线性关系，回归一下就能把两者分开；
     - 2/3 幂定律  v ∝ κ^(-1/3)：曲率大的地方必须**更慢**（转大弯要减速）；
     - 亚运动（submovement）：一次远距离移动由 2-3 个 ballistic 段组成，
       速度剖面有 2-3 个峰，后段是"修正"而不是"重来"；段间速度接近 0；
     - 信号相关噪声  σ_v ∝ v：快的地方抖得厉害，慢的地方几乎不抖
       （Harris & Wolpert）。全局等幅抖动反而暴露；
     - 落点误差是**二维高斯**（各向同性），不是整数均匀分布。

二、时序类检测（看**一串**动作的间隔）
  1. 固定节拍（1000/tick_hz）→ 间隔序列的 ACF 在基频处有尖峰，肉眼可见；
  2. 均匀抖动（uniform ±ratio）→ 分布是"盒状"，样本一多方差就锁死；
  3. 硬下限（限速器 min_interval 钳位）→ 间隔直方图有一堵陡墙，所有动作
     都贴着墙走（"每次都刚好 25ms 以上一点"）；
  4. **帧率锁相**：决策是帧驱动的，若节拍与 1/fps 有整数关系，动作时刻对
     1/fps 取模后会聚成一簇 —— 这条最隐蔽，也最容易抓。

三、手势类检测（看**一个动作内部**的微观结构）
  1. 一次按键 = 一次 keydown+keyup，按下时长恒定 → 机械；
  2. 连按（打技能/CD/抢怪）在真实日志里是**成簇**的：3-9 下一次，
     簇内间隔先长后短、有卡壳、按下时长比单次更短；
  3. 程序化实现通常"点完就开始下一步"，**根本没有连按这个概念** ——
     于是整份日志里每个动作都是孤立的单次事件，这本身就是特征
     （真人日志里点击与按键的簇长分布是重尾的）。

本模块给出的东西
════════════════
  - `human_path()`：Fitts 时长 + 亚运动 + 2/3 幂律 + 信号相关噪声 + 非均匀采样；
  - `Rhythm`：双态（专注/松散）× 1/f 长程相关 × 对数正态 × 长尾打断，
    并把物理下限**本身**随机化（把"墙"变成"坡"）；
  - `Mash`：连按/连点爆发 —— 簇长重尾、簇内先慢后快、按下时长变短、
    成簇出现（兴奋度慢变量）、总时长有预算。

三样都是纯函数/小状态，可 seed 复现，可被 `tools/behavior_audit.py` 量化检验。
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

# ── minimum-jerk 速度剖面的峰值：v(u) = 30u²(1-u)²，峰值 1.875（u=0.5）──
MIN_JERK_PEAK = 1.875

#: 曲率减速的参考曲率（1/px）。κ 远大于它时速度按 κ^(-1/3) 衰减。
#: 0.004 ≈ 曲率半径 250px 处开始明显减速，和真人转急弯的体感量级一致。
KAPPA_REF = 0.004

#: 亚运动边界处的速度下限（相对峰值）。真人在两段之间手会**停一下**，
#: 但不会停到无穷久；没有这个下限，∫ds/v 会发散。
SEG_V_FLOOR = 0.06


def _rng_or_default(rng: Any) -> Any:
    return random if rng is None else rng


# ═══════════════════════════════════════════════════════════════════════
# 一、Fitts 律时长
# ═══════════════════════════════════════════════════════════════════════


def fitts_duration(
    dist_px: float,
    width_px: float = 24.0,
    *,
    a: float = 0.09,
    b: float = 0.10,
    cv: float = 0.15,
    rng: Any = None,
) -> float:
    """Fitts 律：MT = a + b·log2(D/W + 1)，再乘个体差异（对数正态）。

    为什么不是 D / speed：那样距离翻倍时长翻倍；真人翻倍距离只多花一点点
    （对数增长）。这条关系在任何一段操作日志里都能回归出来，是轨迹类检测
    最先看的东西之一。
    """
    r = _rng_or_default(rng)
    w = max(1.0, float(width_px))
    d = max(0.0, float(dist_px))
    mt = float(a) + float(b) * math.log2(d / w + 1.0)
    return max(0.01, mt * math.exp(r.gauss(0.0, max(0.0, float(cv)))))


# ═══════════════════════════════════════════════════════════════════════
# 二、几何：贝塞尔弓形 + 低频横向漂移
# ═══════════════════════════════════════════════════════════════════════


def _curvature(pts: Sequence[tuple[float, float]], i: int) -> float:
    """三点外接圆曲率 κ = 4A/(abc)（单位 1/px），退化返回 0。"""
    x1, y1 = pts[i - 1]
    x2, y2 = pts[i]
    x3, y3 = pts[i + 1]
    a = math.hypot(x2 - x1, y2 - y1)
    b = math.hypot(x3 - x2, y3 - y2)
    c = math.hypot(x3 - x1, y3 - y1)
    denom = a * b * c
    if denom < 1e-9:
        return 0.0
    area2 = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))  # = 2×面积
    return 2.0 * area2 / denom


def _catmull(p0, p1, p2, p3, t: float) -> tuple[float, float]:
    """Catmull-Rom 插值（过控制点，C1 连续）。"""
    t2, t3 = t * t, t * t * t
    return (
        0.5 * ((2.0 * p1[0]) + (-p0[0] + p2[0]) * t
               + (2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0]) * t2
               + (-p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0]) * t3),
        0.5 * ((2.0 * p1[1]) + (-p0[1] + p2[1]) * t
               + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * t2
               + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * t3),
    )


def _geometry(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    bow: float,
    wobble_px: float,
    n: int,
    bounds: Sequence[float],
    rng: Any,
) -> list[tuple[float, float]]:
    """密采样几何轨迹：分段折线（折点带横向偏移）+ Catmull-Rom 平滑。

    ⚠️ 这里刻意**不用单一贝塞尔**。贝塞尔的最大曲率在中段，而中段正好是
    速度最快的部分 —— 于是"曲率大的地方更快"，和 2/3 幂定律（曲率大必须
    减速）反号。审计工具一回归就能看出来。
    真人的结构是：**主运动近似直线（快），段间修正是一条小弧（慢）**。
    所以这里把弯曲集中在段边界，主段保持接近直线。
    """
    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    dx, dy = ex - sx, ey - sy
    dist = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / dist, dx / dist

    # 控制点：沿直线的分段点，各自带法向偏移。
    # 主段（第一段）偏移小 → 直；修正段偏移大 → 弯曲集中在低速区。
    ctrl: list[tuple[float, float]] = [(sx, sy)]
    n_seg = max(1, len(bounds) - 1)
    for i in range(1, len(bounds) - 1):
        p = float(bounds[i])
        # 越靠后的段越小（修正量递减），主段终点承担整体弧顶
        shrink = 1.0 if i == 1 else (0.7 ** (i - 1))
        base_amp = bow * (dist ** 0.7) * 0.5
        amp = (base_amp + rng.uniform(1.0, 5.0)) * shrink * rng.choice((-1.0, 1.0))
        ctrl.append((sx + dx * p + nx * amp, sy + dy * p + ny * amp))
        # 每段中点再放一个控制点：让主段有极轻的弧，修正段更像"小弧"
        if i < len(bounds) - 2:
            mid = 0.5 * (p + float(bounds[i + 1]))
            amp2 = amp * 0.55 + rng.uniform(-1.0, 1.0)
            ctrl.append((sx + dx * mid + nx * amp2, sy + dy * mid + ny * amp2))
    ctrl.append((ex, ey))
    if len(ctrl) < 3:
        ctrl = [(sx, sy), ((sx + ex) / 2 + nx * bow * dist * 0.5,
                            (sy + ey) / 2 + ny * bow * dist * 0.5), (ex, ey)]

    # 端点镜像补两个虚拟点，保证首尾切线自然
    p_first = (2 * ctrl[0][0] - ctrl[1][0], 2 * ctrl[0][1] - ctrl[1][1])
    p_last = (2 * ctrl[-1][0] - ctrl[-2][0], 2 * ctrl[-1][1] - ctrl[-2][1])
    ext = [p_first] + ctrl + [p_last]

    spans = len(ext) - 3
    steps = max(2, int(math.ceil(n / max(1, spans))))
    out: list[tuple[float, float]] = []
    for i in range(spans):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        last_step = steps + (1 if i == spans - 1 else 0)
        for j in range(last_step):
            out.append(_catmull(p0, p1, p2, p3, j / float(steps)))

    # AR(1) 低频横向漂移：连续平滑的手抖，不是白噪声
    rho = 0.9
    scale = math.sqrt(max(0.0, 1.0 - rho * rho))
    w = 0.0
    for i, (x, y) in enumerate(out):
        w = w * rho + rng.gauss(0.0, float(wobble_px) * scale)
        out[i] = (x + nx * w, y + ny * w)
    out[0] = (sx, sy)
    out[-1] = (ex, ey)
    return out


# ═══════════════════════════════════════════════════════════════════════
# 三、轨迹主函数
# ═══════════════════════════════════════════════════════════════════════


def _submove_bounds(k: int, rng: Any) -> list[float]:
    """亚运动的路程分段边界（0..1）。第一段是主运动，吃掉大部分路程。"""
    if k <= 1:
        return [0.0, 1.0]
    b1 = rng.uniform(0.74, 0.94)          # 主运动结束位置（真人 85% 左右）
    if k == 2:
        return [0.0, b1, 1.0]
    b2 = rng.uniform(b1 + 0.02, min(0.995, b1 + 0.18))
    return [0.0, b1, b2, 1.0]


def _temporal_profile(
    prog: Sequence[float],
    kappa: Sequence[float],
    bounds: Sequence[float],
) -> list[float]:
    """把"沿路程的比例"映射成"时间比例"。

    两步叠在一起，缺一个都会露：
      ① 亚运动：段内 minimum-jerk 速度剖面（钟形），段间速度掉到接近 0
         —— 这就是真人"冲过去 → 停一下 → 微调"的多峰结构；
      ② 2/3 幂律：速度再乘 (κ_ref/(κ+κ_ref))^(1/3)，曲率越大越慢。

    时间 = ∫ ds / v，最后归一化到 [0,1]，总时长由调用方（Fitts）决定。
    """
    n = len(prog)
    ts = [0.0] * n
    acc = 0.0
    seg_i = 0
    for i in range(1, n):
        p_mid = 0.5 * (prog[i - 1] + prog[i])
        while seg_i + 1 < len(bounds) - 1 and p_mid > bounds[seg_i + 1]:
            seg_i += 1
        lo, hi = bounds[seg_i], bounds[seg_i + 1]
        span = max(1e-6, hi - lo)
        u = min(1.0, max(0.0, (p_mid - lo) / span))
        v = MIN_JERK_PEAK * u * u * (1.0 - u) * (1.0 - u)
        v = max(SEG_V_FLOOR, v)
        k = max(0.0, kappa[i])
        v *= (KAPPA_REF / (k + KAPPA_REF)) ** (1.0 / 3.0)
        ds = abs(prog[i] - prog[i - 1])
        acc += ds / max(1e-9, v)
        ts[i] = acc
    if acc <= 0:
        return [i / max(1, n - 1) for i in range(n)]
    return [t / acc for t in ts]


def human_path(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    bow: float = 0.06,
    wobble_px: float = 1.6,
    sample_hz: float = 90.0,
    noise_px: float = 0.5,
    speed_noise: float = 0.10,
    target_width_px: float = 24.0,
    fitts_a: float = 0.09,
    fitts_b: float = 0.10,
    fitts_cv: float = 0.15,
    min_duration_s: float = 0.06,
    max_duration_s: float = 0.45,
    max_points: int = 64,
    endpoint_sigma_px: float = 1.2,
    endpoint_clip_px: float = 0.0,
    min_dist_px: float = 6.0,
    rng: Any = None,
) -> list[tuple[int, int, float]]:
    """生成一条人类化轨迹：`[(x, y, 到该点前等待秒), ...]`，末点落在 (±σ) 内。

    ⚠️ 落点带高斯偏差（真人不点数学中心），`endpoint_clip_px>0` 时按该值钳制
    上下限（真人偏得太远会重试，不会硬着头皮点下去）。
    dt 之和 = 整段时长（受 min/max 钳制）。
    """
    r = _rng_or_default(rng)
    sx, sy = float(start[0]), float(start[1])
    # 落点误差：二维各向同性高斯。整数均匀分布（±2px 等概率）是明显的机器痕迹。
    sig = max(0.0, float(endpoint_sigma_px))
    tx = float(end[0]) + (r.gauss(0.0, sig) if sig else 0.0)
    ty = float(end[1]) + (r.gauss(0.0, sig) if sig else 0.0)
    clip = max(0.0, float(endpoint_clip_px))
    if clip > 0:
        tx = min(float(end[0]) + clip, max(float(end[0]) - clip, tx))
        ty = min(float(end[1]) + clip, max(float(end[1]) - clip, ty))

    dist = math.hypot(tx - sx, ty - sy)
    if dist < max(0.0, float(min_dist_px)):
        return [(int(round(tx)), int(round(ty)), 0.0)]

    # ── 时长：Fitts 律，再钳到带宽预算内 ──
    dur = fitts_duration(dist, target_width_px, a=fitts_a, b=fitts_b,
                         cv=fitts_cv, rng=r)
    lo = max(0.01, float(min_duration_s))
    hi = max(lo, float(max_duration_s))
    dur = min(hi, max(lo, dur))

    # ── 亚运动段数：近距离 1 段，中距 1-2 段，远距 2-3 段（带随机）──
    # 必须在几何之前定：弯曲要集中在段边界（低速区），主段保持接近直线。
    if dist < 120.0:
        k = 1 if r.random() < 0.85 else 2
    elif dist < 600.0:
        k = 1 if r.random() < 0.35 else 2
    else:
        k = 2 if r.random() < 0.55 else 3
    bounds = _submove_bounds(k, r)

    # ── 几何：密采样 → 弧长参数化 ──
    n_dense = 420
    geom = _geometry((sx, sy), (tx, ty), bow=bow, wobble_px=wobble_px,
                     n=n_dense, bounds=bounds, rng=r)
    arc = [0.0]
    for i in range(1, len(geom)):
        arc.append(arc[-1] + math.hypot(geom[i][0] - geom[i - 1][0],
                                        geom[i][1] - geom[i - 1][1]))
    total = arc[-1]
    if total < 1e-6:
        return [(int(round(tx)), int(round(ty)), 0.0)]
    prog = [a / total for a in arc]

    kappa = [0.0] * len(geom)
    for i in range(1, len(geom) - 1):
        kappa[i] = _curvature(geom, i)

    time_prog = _temporal_profile(prog, kappa, bounds)

    # ── 采样时刻：非均匀（真实鼠标上报间隔本来就有抖动）──
    # 采样时刻（归一化 0..1）：间隔带一点抖动（真实 HID 上报本来就不均匀），
    # 但抖动幅度压在 12% 以内 —— 再大就会把速度剖面污染成噪声。
    mean_dt = 1.0 / max(10.0, float(sample_hz))
    times: list[float] = [0.0]
    t = 0.0
    guard = 0
    while t < 1.0 and guard < 4000:
        guard += 1
        step = mean_dt * math.exp(r.gauss(0.0, 0.12))
        t = min(1.0, t + max(0.0025, step))
        times.append(t)
    if len(times) < 3:
        times = [0.0, 0.5, 1.0]

    def _lookup(table: Sequence[float], x: float) -> float:
        """在单调表里定位 x 的下标（返回浮点下标，用于插值）。"""
        lo_i, hi_i = 0, len(table) - 1
        while lo_i < hi_i:
            mid = (lo_i + hi_i) // 2
            if table[mid] < x:
                lo_i = mid + 1
            else:
                hi_i = mid
        if lo_i == 0:
            return 0.0
        prev, cur = table[lo_i - 1], table[lo_i]
        span = cur - prev
        if span <= 1e-12:
            return float(lo_i)
        return (lo_i - 1) + (x - prev) / span

    raw: list[tuple[float, float, float, float]] = []   # (x, y, 步长, 归一化时刻)
    prev_xy = (sx, sy)
    for tk in times:
        idx = _lookup(time_prog, tk)
        i0 = int(math.floor(idx))
        frac = idx - i0
        i1 = min(len(geom) - 1, i0 + 1)
        # 位置：先按时间比例找到"路程比例"，再在几何上插值
        p = prog[i0] + (prog[i1] - prog[i0]) * frac
        j = _lookup(prog, p)
        j0 = int(math.floor(j))
        f2 = j - j0
        j1 = min(len(geom) - 1, j0 + 1)
        x = geom[j0][0] + (geom[j1][0] - geom[j0][0]) * f2
        y = geom[j0][1] + (geom[j1][1] - geom[j0][1]) * f2

        step_px = math.hypot(x - prev_xy[0], y - prev_xy[1])
        raw.append((x, y, step_px, tk))
        prev_xy = (x, y)

    # ── 信号相关噪声：σ ∝ 速度（快的地方抖、慢的地方稳），方向垂直于运动 ──
    total_len = sum(s[2] for s in raw) or 1.0
    out: list[tuple[float, float, float, float]] = []
    prev = (sx, sy)
    for x, y, step_px, tk in raw:
        dx, dy = x - prev[0], y - prev[1]
        nrm = math.hypot(dx, dy)
        if nrm > 1e-9:
            nx, ny = -dy / nrm, dx / nrm
        else:
            nx, ny = 0.0, 0.0
        v_norm = step_px / max(1e-9, total_len) * len(raw)   # 相对速度（≈1 为平均）
        sigma = float(noise_px) + float(speed_noise) * v_norm
        jitter = r.gauss(0.0, sigma)
        out.append((x + nx * jitter, y + ny * jitter, step_px, tk))
        prev = (x, y)

    # ── 抽稀：目标点数**也随机**（撞上限时不能每次都正好等于上限）──
    limit = max(2, int(round(max(2, int(max_points)) * r.uniform(0.72, 1.0))))
    if len(out) > limit:
        m = len(out)
        step = (m - 1) / (limit - 1)
        picked = [out[min(m - 1, int(round(i * step)))] for i in range(limit)]
    else:
        picked = out

    # ── 组装 PathPoint：(x, y, dt)，dt 之和 = dur，末点精确落在（带偏差的）目标 ──
    #
    # dt 取**采样时刻的差分**（随点一起抽稀保留）。这是关键：
    #   - 位置序列已经编码了速度剖面（快的地方位移大）；
    #   - dt 再独立抖一层 → 位置与时间脱钩，"速度"出现假峰，审计工具会把
    #     噪声当成亚运动（实测峰数从 3 涨到 11）；
    #   - dt 完全与位移成正比 → 速度被抹成直线，2/3 幂律和亚运动结构全没了。
    # 唯一正确的做法是让两者来自同一套时间轴。
    npts = len(picked)
    res: list[tuple[int, int, float]] = [(int(round(sx)), int(round(sy)), 0.0)]
    t_prev = picked[0][3]
    for i in range(1, npts):
        dt = max(1e-4, (picked[i][3] - t_prev) * dur)
        t_prev = picked[i][3]
        res.append((int(round(picked[i][0])), int(round(picked[i][1])), dt))
    # 抽稀会丢掉一些时间，按比例缩回去，保证 sum(dt) == dur
    ssum = sum(p[2] for p in res) or 1.0
    res = [(p[0], p[1], (p[2] * dur / ssum) if p[2] > 0 else 0.0) for p in res]
    res[-1] = (int(round(tx)), int(round(ty)), res[-1][2])
    return res


# ═══════════════════════════════════════════════════════════════════════
# 四、节奏：双态 + 1/f + 对数正态 + 长尾
# ═══════════════════════════════════════════════════════════════════════

_RHYTHM_DEFAULTS = {
    "enabled": True,
    "cv": 0.26,                 # 对数正态乘性离散度
    "pink_strength": 0.22,      # 1/f 分量强度（乘性，取 exp）
    "pink_octaves": 5,          # Voss-McCartney 层数（相关长度 ≈ 2^5 个样本）
    "drift_strength": 0.06,     # 基频慢漂移（打破与帧率锁相）
    "tail_prob": 0.02,          # 长尾打断概率
    "tail_range": (1.4, 2.2),   # 打断时长倍数
    "focus_factor": (0.90, 1.00),
    "loose_factor": (1.26, 1.68),
    "p_focus_to_loose": 0.07,
    "p_loose_to_focus": 0.30,
    "floor_jitter": 0.18,       # 物理下限本身的抖动比例
    "min_factor": 0.70,         # 相对名义节拍的软下界（真人的手快也有个谱）
}


class Rhythm:
    """动作间隔生成器 —— 让"节奏"不是一条直线。

    三层叠加，每层对应一种可被检测出的规律：
      ① **双态**（专注/松散）：人做重复任务是"一串快动作 + 一个停顿"，
         不是均匀节拍。用两态 Markov 链，各自不同的间隔尺度；
      ② **1/f 长程相关**：人类节奏的时间序列有长程相关性（DFA 指数
         0.7-1.0），而均匀抖动/白噪声接近 0.5。Voss-McCartney 算法用
         几个不同更新速率的随机行走叠加，近似出 pink noise；
      ③ **长尾**：偶尔被打断（看手机、喝口水），概率乘 2-5 倍。指数/对数
         正态这种"有界盒"永远生不出长尾，而长尾恰恰是真实日志最显著的特征。

    另外把**物理下限本身**随机化：限速器的 min_interval 是一条硬墙，
    所有动作都贴着墙走会形成直方图上的陡边。每次抽一个 ±floor_jitter 的
    下界，墙就变成坡。
    """

    def __init__(
        self,
        base_s: float,
        cfg: Any = None,
        rng: Any = None,
        floor_s: float = 0.0,
    ) -> None:
        self.base = max(1e-3, float(base_s))
        self.cfg = cfg
        self._rng = _rng_or_default(rng)
        self.floor_s = max(0.0, float(floor_s))
        k = max(1, int(self._get("pink_octaves")))
        self._rows = [self._rng.gauss(0.0, 1.0) for _ in range(k)]
        self._counter = 0
        self._loose = False
        self.samples = 0
        self.last = self.base

    def _get(self, name: str) -> Any:
        if self.cfg is None:
            return _RHYTHM_DEFAULTS[name]
        return getattr(self.cfg, name, _RHYTHM_DEFAULTS[name])

    def _pink(self) -> float:
        """Voss-McCartney 近似 1/f：越慢的层更新越稀疏，叠加后长程相关。"""
        k = len(self._rows)
        self._counter += 1
        for i in range(k):
            if self._counter % (1 << i) == 0:
                self._rows[i] = self._rng.gauss(0.0, 1.0)
        return sum(self._rows) / (math.sqrt(k) or 1.0)

    def _slow(self) -> float:
        """只取最慢的两层：给基频一个缓慢的漂移，专门用来打破帧率锁相。"""
        if not self._rows:
            return 0.0
        return (self._rows[-1] + self._rows[max(0, len(self._rows) - 2)]) / 2.0

    def next(self, scale: float = 1.0, base_s: Optional[float] = None) -> float:
        """下一个间隔（秒）。

        `base_s` 覆盖实例的基准节拍（同一个 Rhythm 可以服务不同任务：
        走位节拍 1/tick_hz、按键冷却 tap_recheck_ms）；`scale` 用于叠加
        疲劳/监督层降速这类"整体变慢"。
        """
        base = self.base if base_s is None else max(1e-3, float(base_s))
        if not self._get("enabled"):
            return max(self.floor_s, base * max(0.0, scale))
        r = self._rng

        # ① 双态切换
        if self._loose:
            if r.random() < float(self._get("p_loose_to_focus")):
                self._loose = False
        else:
            if r.random() < float(self._get("p_focus_to_loose")):
                self._loose = True
        lo_f = (
            self._get("loose_factor") if self._loose else self._get("focus_factor")
        )
        lo, hi = (list(lo_f) + [0.9, 1.0])[:2]
        state = r.uniform(min(lo, hi), max(lo, hi))

        # ② 1/f 乘性 + 慢漂移（去锁相）
        pink = math.exp(float(self._get("pink_strength")) * self._pink())
        drift = 1.0 + float(self._get("drift_strength")) * self._slow()

        # ③ 对数正态（人的间隔是右偏的，不是对称抖动）
        ln = math.exp(r.gauss(0.0, float(self._get("cv"))))

        v = base * state * pink * drift * ln * max(0.0, scale)

        # ④ 长尾：偶尔被打断
        if r.random() < float(self._get("tail_prob")):
            tr = list(self._get("tail_range")) or [2.0, 5.0]
            v *= r.uniform(min(tr), max(tr))

        # ⑤ 软下界：真人手快也有个谱。**不能用截断** —— 截断会制造一堵新的
        #    硬墙（所有低于门槛的值都堆在门槛上），那正是我们要消掉的痕迹。
        #    这里把过低的值压缩进 [0.7·low, low]，保留分散。
        low = base * max(0.0, float(self._get("min_factor")))
        if low > 0 and v < low:
            v = low * (0.82 + 0.18 * max(0.0, v / low))

        # ⑥ 物理下限（限速器）本身随机化（把墙变坡）
        fj = max(0.0, float(self._get("floor_jitter")))
        floor = self.floor_s * (1.0 + fj * (2.0 * r.random() - 1.0))
        v = max(max(0.0, floor), v)
        self.last = v
        self.samples += 1
        return v

    def next_ms(self, scale: float = 1.0) -> int:
        return int(round(self.next(scale) * 1000.0))

    def snapshot(self) -> dict:
        return {
            "base_ms": round(self.base * 1000.0, 1),
            "last_ms": round(self.last * 1000.0, 1),
            "loose": self._loose,
            "samples": self.samples,
        }


# ═══════════════════════════════════════════════════════════════════════
# 连按 / 连点（mashing）
# ═══════════════════════════════════════════════════════════════════════
#
# 为什么单列一层，而不是"让 Rhythm 偶尔生成一个 90ms 的间隔"：
#
#   真人连打技能不是"间隔变小了"，而是**换了一种行为模式**：
#     ① 簇长是重尾的（3-9 下，偶尔十几下），不是"每 N 秒一次";
#     ② 簇内第一下之后有个回收间隔（抬手再落），比后续都长；
#     ③ 之后越按越快（ramp），但偶尔"卡壳"（手指没跟上/键位偏了）；
#     ④ 连按时的按下时长比单次动作**短**（手指没在键上停留）；
#     ⑤ 收尾有停一下（看技能出来没有），不是按完就走；
#     ⑥ 连按成簇出现（连着好几次操作都在连打），不是独立同分布 ——
#        这对应一个慢变的"兴奋度"，用 AR(1) 做；
#     ⑦ 一分钟里连按占比有上限（真人不会 80% 时间在连打）。
#
# 缺了①-⑦里的任何一条，"高频点击"就退化成"间隔乘了 0.3 的同一个剧本"，
# 而那种序列在簇长直方图上只有一根柱子 —— 比不连按更好认。


#: 单指最快敲击下限（秒）。人类 tapping 纪录约 10-14 次/秒（ISI 70-100ms），
#: 游戏里持续按技能很少快过 9 次/秒。低于这个值的"连按"不是人类物理能做到的。
MASH_FLOOR_S = 0.070

_MASH_DEFAULTS = {
    "enabled": True,
    "keys": ["1", "2", "3", "4", "5", "6", "q", "e", "r", "f"],
    "prob_key": 0.42,           # 按技能键时进入连按的基准概率
    "prob_click": 0.16,         # 鼠标连点（双击/三击）的基准概率
    "click_interval_ms": (70, 140),
    "interval_ms": (95, 165),   # 簇内后续间隔基准
    "interval_sigma": 0.16,     # 间隔的对数正态 σ（右偏）
    "first_ms": (150, 340),     # 第一下 → 第二下（抬手再落，必然更慢）
    "ramp": (1.16, 0.90),       # 越按越快：首 gap 放大，末 gap 收缩
    "slip_prob": 0.07,          # 卡壳概率（某一下明显慢了）
    "slip_range": (1.3, 2.1),
    "count": (3, 9),            # 簇长区间（重尾，均值≈lo+2）
    "hold_ms": (28, 58),        # 连按时的按压时长（比单次短）
    "first_hold_ms": (35, 75),
    "tail_ms": (220, 700),      # 收尾停顿（看结果）
    "excite_mean": 0.35,        # 兴奋度的回复中心（长期平均连按倾向）
    "excite_tau_s": 25.0,       # 兴奋度时间常数（连按成簇的时间尺度）
    "excite_sigma": 0.30,       # 兴奋度的随机游走强度
    "excite_gain": 1.00,        # 兴奋度对概率的放大幅度
    "budget_ratio": 0.30,       # 窗口内连按总时长占比上限
    "budget_window_s": 45.0,
    "max_burst_s": 1.60,        # 单次连按总时长上限（含收尾）
}


@dataclass(frozen=True)
class Tap:
    """连按中的一次按键：先等 `wait_s`，再按住 `hold_ms`。"""

    wait_s: float
    hold_ms: int


@dataclass
class MashPlan:
    """一次连按的完整计划（第一下由调用方发出，这里只描述第 2..n 下）。"""

    taps: list[Tap] = field(default_factory=list)
    first_hold_ms: int = 60
    tail_s: float = 0.0

    @property
    def count(self) -> int:
        """这一簇总共按了几下（含第一下）。"""
        return len(self.taps) + 1

    @property
    def duration_s(self) -> float:
        return sum(t.wait_s for t in self.taps) + self.tail_s


class Mash:
    """把"一次按键"扩展成"一簇连按"。纯状态机 + rng，可 seed 复现。

    ⚠️ 铁律：`forbid` 里的键**永不**连按。拾取键（默认 `a`）连按等于绕开
    A 框确认闸门 —— 那是本项目唯一不允许被"拟人化"覆盖的东西。
    同理 `stop_all` / 逃生键也不连按（按得越多越危险）。

    用法：
        m = Mash(cfg.executor.mash, rng, forbid=(cfg.decision.pickup_key,))
        plan = m.plan(op=cmd.op, key=cmd.key, now=time.monotonic())
        if plan: ...   # 发第一下 → 按 plan.taps 依次发 → 等 tail
    """

    def __init__(self, cfg: Any = None, rng: Any = None,
                 forbid: Sequence[str] = ()) -> None:
        self.cfg = cfg
        self._rng = _rng_or_default(rng)
        self.forbid = {(k or "").lower() for k in forbid if k}
        self._excite = 0.30          # 兴奋度（0..1），先给个中间值
        self._last_t = 0.0
        self._bursts: list[tuple[float, float]] = []   # (结束时刻, 时长)
        self.bursts = 0
        self.skipped_budget = 0
        self.last_plan: Optional[MashPlan] = None

    # ---------------- 配置 ----------------

    def _get(self, name: str) -> Any:
        if self.cfg is None:
            return _MASH_DEFAULTS[name]
        return getattr(self.cfg, name, _MASH_DEFAULTS[name])

    # ---------------- 慢变兴奋度 ----------------

    def excite(self, now: Optional[float] = None) -> float:
        """连按倾向的慢变量。人不是"每次按键独立掷骰子"——

        抢怪/打 boss 的那两分钟里连续连打，跑图的路上一次都不连。
        独立同分布会让"连按簇"的出现过程变成泊松（方差/均值≈1），
        而真实日志是过度离散的（方差/均值 2-4）。
        """
        now = time.monotonic() if now is None else now
        if self._last_t <= 0:
            self._last_t = now
            return self._excite
        dt = max(0.0, min(300.0, now - self._last_t))
        self._last_t = now
        if dt <= 0:
            return self._excite
        tau = max(1.0, float(self._get("excite_tau_s")))
        decay = math.exp(-dt / tau)
        sig = max(0.0, float(self._get("excite_sigma"))) * math.sqrt(max(1e-9, 1.0 - decay * decay))
        # ⚠️ 必须是**均值回复**的 OU，不能是纯衰减 + 噪声：
        #    纯衰减的长期均值是 0，跑十几分钟之后兴奋度沉到 0，
        #    "连按"这个行为就悄悄消失了 —— 日志里看不出原因，只看到越跑越呆。
        mu = min(1.0, max(0.0, float(self._get("excite_mean"))))
        self._excite = min(1.0, max(0.0, mu + (self._excite - mu) * decay
                                  + self._rng.gauss(0.0, sig)))
        return self._excite

    # ---------------- 预算 ----------------

    def _budget_ok(self, need_s: float, now: float) -> bool:
        win = max(5.0, float(self._get("budget_window_s")))
        self._bursts = [(t, d) for t, d in self._bursts if t >= now - win]
        used = sum(d for _, d in self._bursts)
        limit = max(0.05, float(self._get("budget_ratio"))) * win
        return used + need_s <= limit

    # ---------------- 规划 ----------------

    def _allowed(self, op: str, key: Optional[str]) -> bool:
        if not bool(self._get("enabled")):
            return False
        if op == "mouse_click":
            return True
        if op != "key_tap":
            return False
        k = (key or "").lower()
        if not k or k in self.forbid:
            return False
        keys = {str(x).lower() for x in (self._get("keys") or [])}
        return (not keys) or (k in keys)

    def plan(self, op: str, key: Optional[str] = None,
             now: Optional[float] = None) -> Optional[MashPlan]:
        """规划一次连按；不该连按就返回 None（= 保持单次动作）。"""
        now = time.monotonic() if now is None else now
        if not self._allowed(op, key):
            return None

        ex = self.excite(now)
        p0 = float(self._get("prob_click" if op == "mouse_click" else "prob_key"))
        gain = max(0.0, float(self._get("excite_gain")))
        # 上限 0.92：永远"每次都连按"等于换了一种整齐（真人会漏）。
        # 但配置里显式写 1.0 时尊重它 —— 那是调试/审计要的可复现入口。
        p = 1.0 if p0 >= 1.0 else min(0.92, max(0.0, p0 * (0.60 + gain * ex)))
        if self._rng.random() >= p:
            return None

        r = self._rng
        lo, hi = (int(x) for x in self._get("count"))
        lo = max(1, lo)
        hi = max(lo, hi)
        # 簇长重尾：Gamma(2,1) 均值 2、右偏，尾部自然到 hi
        n = lo + int(round(r.gammavariate(2.0, 1.0)))
        n = min(hi, max(lo, n))
        if n < 2:
            return None

        if op == "mouse_click":
            i_lo, i_hi = (float(x) for x in self._get("click_interval_ms"))
            first_lo, first_hi = i_lo * 1.6, i_hi * 2.2
            sigma = 0.12
        else:
            i_lo, i_hi = (float(x) for x in self._get("interval_ms"))
            first_lo, first_hi = (float(x) for x in self._get("first_ms"))
            sigma = max(0.0, float(self._get("interval_sigma")))

        ramp0, ramp1 = (float(x) for x in self._get("ramp"))
        slip_p = max(0.0, float(self._get("slip_prob")))
        slip_lo, slip_hi = (float(x) for x in self._get("slip_range"))
        floor_s = MASH_FLOOR_S

        taps: list[Tap] = []
        total = 0.0
        max_total = max(0.2, float(self._get("max_burst_s")))
        h_lo, h_hi = (int(x) for x in self._get("hold_ms"))
        fh_lo, fh_hi = (int(x) for x in self._get("first_hold_ms"))

        for i in range(n - 1):
            if i == 0:
                gap = r.uniform(first_lo, first_hi) / 1000.0
            else:
                frac = (i - 1) / max(1, n - 3) if n > 3 else 0.5
                ramp = ramp0 + (ramp1 - ramp0) * min(1.0, max(0.0, frac))
                gap = r.uniform(i_lo, i_hi) / 1000.0 * ramp
                gap *= math.exp(r.gauss(0.0, sigma))
                if slip_p > 0 and r.random() < slip_p:
                    gap *= r.uniform(slip_lo, slip_hi)
            gap = max(floor_s, gap)
            if total + gap > max_total:
                break
            total += gap
            taps.append(Tap(gap, r.randint(h_lo, max(h_lo, h_hi))))

        if not taps:
            return None
        tail = r.uniform(*(float(x) for x in self._get("tail_ms"))) / 1000.0
        if total + tail > max_total:
            tail = max(0.0, max_total - total - 0.05)

        plan = MashPlan(taps=taps, first_hold_ms=r.randint(fh_lo, max(fh_lo, fh_hi)),
                        tail_s=max(0.0, tail))
        if not self._budget_ok(plan.duration_s, now):
            self.skipped_budget += 1
            return None
        self._bursts.append((now + plan.duration_s, plan.duration_s))
        self.bursts += 1
        self.last_plan = plan
        return plan

    def snapshot(self) -> dict:
        return {
            "bursts": self.bursts,
            "excite": round(self._excite, 3),
            "skipped_budget": self.skipped_budget,
            "last_count": self.last_plan.count if self.last_plan else 0,
        }
