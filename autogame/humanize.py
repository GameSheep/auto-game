"""拟人化行为层 —— 手抖（轨迹）、疲劳（节奏）、走神（噪音）、社交。

═══════════════════════════════════════════════════════════════════════
先认清这一层能做什么、不能做什么（不自欺欺人）
═══════════════════════════════════════════════════════════════════════
能做：把"每一条单独的特征"做得更像人 —— 轨迹有弧度、反应时间右偏、
      延时均值随时间漂移、偶尔做和目标无关的事。这让**纯统计模型**的
      判定门槛变高。
不能做：改变"程序驱动"这个本质。这套分布的形状是我们设计的，不是由一个
      会累、会烦、会分心的人自然长出来的。行为数据一旦积累到足够量级并
      触发**人工复核**，"摄像头看屏 → 程序判断 → 机械臂执行 → 无人值守"
      在人工眼里就是自动化系统，加多少层随机都一样。
结论：这是**降低短期统计风险**的手段，不是安全方案。别把封号风险当成 0。

分层职责（保持解耦，别把业务逻辑写进这里）：
  - 本模块（humanize）   ：只有"人的状态"和纯函数，不碰视觉、不碰业务；
  - 执行器（executor）   ：消费 MousePathConfig，决定"手怎么动"；
  - 状态机（decision）   ：消费 Humanizer，决定"人累不累、要不要走神"。
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .config import (
    DriftConfig,
    FatigueConfig,
    HumanizeConfig,
    MoodConfig,
    MousePathConfig,
    RhythmConfig,
    SocialConfig,
)
from .motion import Rhythm, human_path
from .protocol import Command

log = logging.getLogger(__name__)


def _rng_or_default(rng: Any, seed: int | None) -> Any:
    """随机源选择。

    未显式给 rng/seed 时**复用全局 random 模块** —— 这样 `random.seed(x)` 就能
    一次性固定整条链路（含本模块）的随机性。工具/校验器（batch_validate、
    video_replay）正是靠这个保证"同一输入两次运行命令一致"。
    """
    if rng is not None:
        return rng
    if seed is not None:
        return random.Random(seed)
    return random


# ═══════════════════════════════════════════════════════════════════════
# 一、鼠标轨迹：WindMouse（重力吸附 + 风力扰动 + 端点减速）
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PathPoint:
    """轨迹采样点：坐标 + 到达该点前应等待的秒数。"""

    x: int
    y: int
    dt: float = 0.0


def wind_mouse(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    gravity: float = 9.0,
    wind: float = 3.0,
    max_step: float = 10.0,
    damping: float = 0.86,
    target_area: float = 8.0,
    max_points: int = 200,
    rng: Any = None,
) -> list[tuple[float, float]]:
    """WindMouse 轨迹：目标像"重力源"吸引，同时被"风"随机推偏。

    相比两点一线：
      - 路径天然带弧度（风力横向分量），不是直线；
      - 速度动态变化（重力加速 → 靠近目标风力衰减 → 自然减速）；
      - 落点前会有细微抖动（target_area 内风力快速衰减）。
    纯函数 + 显式 rng，可复现、可单测。
    """
    rng = _rng_or_default(rng, None)
    sqrt3, sqrt5 = math.sqrt(3.0), math.sqrt(5.0)
    x, y = float(start[0]), float(start[1])
    tx, ty = float(end[0]), float(end[1])
    wx = wy = vx = vy = 0.0
    # 轨迹必须包含起点本身：否则第一个采样点已经偏移，鼠标会"凭空跳一下"
    pts: list[tuple[float, float]] = [(x, y)]

    for _ in range(max(1, max_points)):
        dist = math.hypot(tx - x, ty - y)
        if dist < 1.0:
            break
        w = min(wind, dist)
        if dist > target_area:
            # 风力做低通（/sqrt3）后再注入新扰动（/sqrt5）——经典 WindMouse 形状
            wx = wx / sqrt3 + (2.0 * w * rng.random() - w) / sqrt5
            wy = wy / sqrt3 + (2.0 * w * rng.random() - w) / sqrt5
        else:
            wx /= sqrt3
            wy /= sqrt3
        vx = (vx + wx + gravity * (tx - x) / dist) * damping
        vy = (vy + wy + gravity * (ty - y) / dist) * damping
        mag = math.hypot(vx, vy)
        if mag > max_step:
            # 随机化单步上限：避免"每步都顶到上限"这种可识别的等幅特征
            clip = max_step * (0.5 + 0.5 * rng.random())
            vx = vx / mag * clip
            vy = vy / mag * clip
        x += vx
        y += vy
        pts.append((max(0.0, x), max(0.0, y)))

    pts.append((max(0.0, tx), max(0.0, ty)))
    return pts


def _downsample(pts: Sequence[tuple[float, float]], limit: int) -> list[tuple[float, float]]:
    """均匀抽稀到 limit 个点（保留首尾）。"""
    n = len(pts)
    if n <= limit or limit < 2:
        return list(pts)
    step = (n - 1) / (limit - 1)
    idx = [min(n - 1, int(round(i * step))) for i in range(limit)]
    out = [pts[i] for i in idx]
    out[-1] = pts[-1]
    return out


def _assign_timing(
    pts: Sequence[tuple[float, float]], cfg: MousePathConfig, rng: Any
) -> list[PathPoint]:
    """给几何轨迹配时间：两端慢、中段快，总时长按名义速度钳制。"""
    n = len(pts)
    if n < 2:
        p = pts[0] if pts else (0.0, 0.0)
        return [PathPoint(int(round(p[0])), int(round(p[1])), 0.0)]

    seg = [math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]) for i in range(1, n)]
    total = sum(seg)
    if total < 1e-6:
        return [PathPoint(int(round(pts[-1][0])), int(round(pts[-1][1])), 0.0)]

    duration = min(cfg.max_duration_s, max(cfg.min_duration_s, total / max(1.0, cfg.speed_px_s)))

    # 每段"单位距离耗时"权重：端点慢（w 小 → 耗时多），中段快
    raw: list[float] = []
    cum = 0.0
    for d in seg:
        u = (cum + d / 2.0) / total
        cum += d
        w = 0.18 + 0.82 * math.sin(math.pi * min(1.0, max(0.0, u))) ** 0.6
        raw.append(d / w)
    s = sum(raw) or 1.0

    out = [PathPoint(int(round(pts[0][0])), int(round(pts[0][1])), 0.0)]
    for i, r in enumerate(raw):
        out.append(
            PathPoint(int(round(pts[i + 1][0])), int(round(pts[i + 1][1])), duration * r / s)
        )

    # 中途微小停顿：真人手不是匀速滑过去的，偶尔"卡"一下
    if len(out) > 3 and cfg.max_micro_pauses > 0:
        lo, hi = (cfg.micro_pause_ms + [20, 70])[:2]
        for _ in range(cfg.max_micro_pauses):
            if rng.random() >= cfg.micro_pause_prob:
                continue
            i = rng.randint(1, len(out) - 2)
            out[i] = PathPoint(out[i].x, out[i].y, out[i].dt + rng.uniform(lo, hi) / 1000.0)
    return out


def _kinematic_path(
    start: tuple[int, int],
    end: tuple[int, int],
    cfg: MousePathConfig,
    rng: Any,
) -> list[PathPoint]:
    """运动学轨迹（`motion.human_path`）→ PathPoint 列表。

    与 WindMouse 的区别在于"伪造的是什么"：WindMouse 伪造的是**抖动形状**，
    这里伪造的是**运动学不变量**（Fitts 律时长 / 2/3 幂律 / 亚运动多峰 /
    速度相关噪声）。后者是人的神经-肌肉特性，不是随机数能凑出来的。
    """
    if abs(start[0] - end[0]) < 2 and abs(start[1] - end[1]) < 2:
        return [PathPoint(int(end[0]), int(end[1]), 0.0)]
    j = max(0, int(cfg.endpoint_jitter_px))
    pts = human_path(
        start, end,
        bow=cfg.bow, wobble_px=cfg.wobble_px, sample_hz=cfg.sample_hz,
        noise_px=cfg.noise_px, speed_noise=cfg.speed_noise,
        target_width_px=cfg.target_width_px,
        fitts_a=cfg.fitts_a, fitts_b=cfg.fitts_b, fitts_cv=cfg.fitts_cv,
        min_duration_s=cfg.min_duration_s, max_duration_s=cfg.max_duration_s,
        max_points=cfg.max_points,
        endpoint_sigma_px=cfg.endpoint_sigma_px,
        # 落点偏差按 endpoint_jitter_px 兜底钳制：偏得太远真人会重试，
        # 不会"硬着头皮点下去"，这里保持同一个语义。
        endpoint_clip_px=float(j) if j else 0.0,
        rng=rng,
    )
    return [PathPoint(x, y, dt) for x, y, dt in pts]


def build_mouse_path(
    start: tuple[int, int],
    end: tuple[int, int],
    cfg: MousePathConfig,
    rng: Any = None,
) -> list[PathPoint]:
    """起点 → 终点的人类化轨迹（关闭时退化为单点，行为与旧版完全一致）。

    默认走 `motion.human_path()`（Fitts 律 + 2/3 幂律 + 亚运动 + 速度相关噪声），
    因为它伪造的是**运动学不变量**而不是"随机弯一下"；`cfg.model="windmouse"`
    保留旧实现，用于回归对照。
    """
    rng = _rng_or_default(rng, None)
    ex, ey = int(end[0]), int(end[1])
    if not cfg.enabled:
        return [PathPoint(ex, ey, 0.0)]

    if getattr(cfg, "model", "kinematic") == "kinematic":
        return _kinematic_path(start, (ex, ey), cfg, rng)

    # 落点偏差：真人不点"数学中心"，差 1-3px 很正常
    j = max(0, int(cfg.endpoint_jitter_px))
    fx = max(0, ex + (rng.randint(-j, j) if j else 0))
    fy = max(0, ey + (rng.randint(-j, j) if j else 0))

    if abs(start[0] - fx) < 2 and abs(start[1] - fy) < 2:
        return [PathPoint(fx, fy, 0.0)]
    # 近距移动不规划轨迹：几像素的挪动硬套一条曲线只会拖慢节奏（真人也不会）
    if math.hypot(start[0] - fx, start[1] - fy) < 6.0:
        return [PathPoint(fx, fy, 0.0)]

    pts = wind_mouse(
        start, (fx, fy),
        gravity=cfg.gravity, wind=cfg.wind, max_step=cfg.max_step_px,
        damping=cfg.damping, max_points=max(4, cfg.max_points * 4), rng=rng,
    )
    pts = _downsample(pts, max(2, cfg.max_points))

    # 冲过头再拉回：真人快速移动常见，尤其远距离
    if cfg.overshoot_max_px > 0 and rng.random() < cfg.overshoot_prob:
        p0, p1 = pts[-2], pts[-1]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        norm = math.hypot(dx, dy) or 1.0
        ov = rng.uniform(2.0, cfg.overshoot_max_px)
        ux, uy = dx / norm, dy / norm
        pts.append((max(0.0, fx + ux * ov), max(0.0, fy + uy * ov)))
        pts.append((max(0.0, fx + ux * ov * 0.35), max(0.0, fy + uy * ov * 0.35)))

    pts.append((float(fx), float(fy)))
    return _assign_timing(pts, cfg, rng)


# ═══════════════════════════════════════════════════════════════════════
# 二、疲劳度：两层状态（慢变量疲劳 + 快变量唤醒）
# ═══════════════════════════════════════════════════════════════════════

class FatigueModel:
    """疲劳度模型 —— 为什么"固定区间随机"不够。

    固定区间随机（uniform(0.8, 1.2)）的方差有上界：样本一多，方差就收敛到
    那个盒子里，长程统计一眼就看出是程序。真人的反应时间方差本身在变。

    这里用两层状态：
      - fatigue（慢变量）：连续操作累积、休息衰减 → **均值**随时间漂移；
      - arousal（快变量）：随机游走 + 向 0 回归 → **方差**本身在波动。
    于是延迟分布既有均值漂移又有方差波动，长程统计不会稳定在一个固定范围。
    """

    def __init__(self, cfg: FatigueConfig, rng: Any = None) -> None:
        self.cfg = cfg
        self._rng = _rng_or_default(rng, None)
        self.level = 0.0      # 0..1 疲劳度
        self.arousal = 0.0    # 瞬时唤醒度（可正可负：负=精神好，正=走神/烦躁）
        self.actions = 0
        self.breaks = 0
        self._last_action = 0.0
        self._last_tick: float | None = None

    def tick(self, now: float) -> None:
        """每帧调用：唤醒度随机游走 + 空闲时疲劳自然衰减。"""
        if self._last_tick is None:
            self._last_tick = now
            return
        dt = max(0.0, min(5.0, now - self._last_tick))
        self._last_tick = now
        if dt <= 0:
            return
        # 向 0 回归（decay^dt）叠加随机游走（σ√dt 是标准布朗桥缩放）
        self.arousal = self.arousal * (self.cfg.arousal_decay ** dt) + self._rng.gauss(
            0.0, self.cfg.arousal_sigma * math.sqrt(dt)
        )
        self.arousal = max(-1.0, min(1.0, self.arousal))
        # 连续 idle_after_s 秒没有动作 → 当作"歇着"，疲劳按秒衰减
        if self._last_action and now - self._last_action > self.cfg.idle_after_s:
            self.level = max(0.0, self.level - self.cfg.rest_decay_per_s * dt)

    def note_action(self, now: float, cost: float = 1.0) -> None:
        self.actions += 1
        self._last_action = now
        self.level = min(1.0, self.level + self.cfg.per_action * max(0.0, cost))

    def note_rest(self, seconds: float) -> None:
        self.breaks += 1
        span = max(1.0, (self.cfg.break_seconds or [30.0])[-1])
        self.level = max(0.0, self.level - self.cfg.break_recovery * min(1.0, seconds / span))
        self.arousal *= 0.3

    def needs_break(self) -> bool:
        return self.level >= self.cfg.break_threshold

    def latency_scale(self) -> float:
        """延时倍数：疲劳拉长 + 唤醒度扰动（正=更慢，负=进入状态更快）。"""
        s = 1.0 + self.level * self.cfg.latency_boost + max(0.0, self.arousal) * 0.5
        return max(0.6, min(3.0, s))

    def jitter_scale(self) -> float:
        """抖动比例倍数：越累手越不稳（方差随疲劳放大）。"""
        return max(0.2, (1.0 + self.level * self.cfg.jitter_boost) * (1.0 + abs(self.arousal)))

    def error_prob(self, base: float) -> float:
        return max(0.0, min(0.5, base + self.level * self.cfg.error_boost))


# ═══════════════════════════════════════════════════════════════════════
# 三、情绪/偏好：让"选哪个目标"也有不确定性
# ═══════════════════════════════════════════════════════════════════════


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def softmax_choice(values: Sequence[float], temperature: float, rng: Any) -> int:
    """softmax 加权随机选择，返回下标。

    为什么不用 argmax：argmax 是"永远选最优" —— 这本身就是机器特征。
    真人会偶尔挑个次优的（顺手、看着顺眼、懒得多走两步），softmax 天然带这个性质，
    而且温度可调（温度越高越随性）。
    """
    n = len(values)
    if n <= 1:
        return 0
    t = max(1e-3, float(temperature))
    mx = max(values)
    weights = [math.exp((v - mx) / t) for v in values]
    total = sum(weights)
    if total <= 0 or not math.isfinite(total):
        return int(rng.randrange(n))
    r = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return n - 1


class MoodModel:
    """情绪/偏好：耐心、贪心、新鲜感三个慢变量。

    解决的问题：原来的目标选择是 `stables[0]` —— **永远第一个**。
    这比任何时序抖动都致命，因为它是一个确定性规则，样本一多必然暴露。

    三个变量都做"随机游走 + 向基线回归 + 疲劳耦合"：
      - 随机游走 → 偏好会漂移，不是固定权重；
      - 向基线回归 → 不会漂到离谱（人也不会一直想捡远处的垃圾）；
      - 疲劳耦合 → 累了耐心和贪心都下降（只捡近的、懒得绕路）。
    """

    def __init__(self, cfg: MoodConfig, fatigue: "FatigueModel",
                 rng: Any = None) -> None:
        self.cfg = cfg
        self.fatigue = fatigue
        self._rng = _rng_or_default(rng, None)
        self.patience = clamp01(cfg.patience)
        self.greed = clamp01(cfg.greed)
        self.novelty = clamp01(cfg.novelty)
        self._last_tick: float | None = None

    def tick(self, now: float) -> None:
        if self._last_tick is None:
            self._last_tick = now
            return
        dt = max(0.0, min(5.0, now - self._last_tick))
        self._last_tick = now
        if dt <= 0:
            return
        k = min(1.0, self.cfg.reversion * dt)
        sigma = self.cfg.drift_sigma * math.sqrt(dt)
        self.patience = clamp01(
            self.patience + (self.cfg.patience - self.patience) * k
            + self._rng.gauss(0.0, sigma))
        self.greed = clamp01(
            self.greed + (self.cfg.greed - self.greed) * k
            + self._rng.gauss(0.0, sigma))
        self.novelty = clamp01(
            self.novelty + (self.cfg.novelty - self.novelty) * k
            + self._rng.gauss(0.0, sigma))

    def effective(self) -> dict[str, float]:
        """当前生效的偏好（叠加疲劳影响）。"""
        f = clamp01(self.fatigue.level)
        return {
            "patience": clamp01(self.patience * (1.0 - self.cfg.fatigue_patience_drop * f)),
            "greed": clamp01(self.greed * (1.0 - self.cfg.fatigue_greed_drop * f)),
            "novelty": clamp01(self.novelty),
        }

    def utility(self, distance: float, color: str, novelty: float) -> float:
        """目标效用：近的、贵的、新鲜的更值得捡。

        near 用指数衰减而不是线性：真人不会觉得"远 10px 就差一点"，
        而是"再远一点就不想去了"，指数形状更贴近。
        """
        e = self.effective()
        w_dist = 0.2 + e["patience"]        # 0.2..1.2
        w_value = e["greed"] * 1.2
        w_novelty = e["novelty"] * 0.8
        near = math.exp(-max(0.0, float(distance)) / max(1.0, self.cfg.distance_scale_px))
        value = self.cfg.color_value.get(color, 0.5)
        return w_dist * near + w_value * value + w_novelty * clamp01(novelty)

    def should_skip(self, best_distance: float) -> bool:
        """懒得捡：距离越远、耐心越低，越可能直接无视（人真的会漏捡）。"""
        if self.cfg.skip_prob <= 0:
            return False
        e = self.effective()
        scale = max(1.0, self.cfg.distance_scale_px)
        far = max(0.0, float(best_distance) / scale - 0.6)   # 超过 0.6 倍尺度才开始加码
        p = self.cfg.skip_prob * (1.0 + far) * (1.5 - e["patience"])
        return self._rng.random() < max(0.0, min(0.6, p))

    def snapshot(self) -> dict:
        e = self.effective()
        return {
            "patience": round(self.patience, 3),
            "greed": round(self.greed, 3),
            "novelty": round(self.novelty, 3),
            "effective": {k: round(v, 3) for k, v in e.items()},
        }


# ═══════════════════════════════════════════════════════════════════════
# 四、Humanizer：决策层门面（疲劳 + 反应时间 + 走神 + 社交 + 情绪）
# ═══════════════════════════════════════════════════════════════════════


class Humanizer:
    """把"人的状态"打包给状态机用。

    只做三件事，且都在安全边界内：
      1. 调整**时序**（延时/抖动/反应时间）—— 不改任何动作语义；
      2. 偶尔插入**无害噪音**（默认只有鼠标移动，不点击、不按键）；
      3. 累了就**停手歇一会儿**（休息窗口内不输出任何命令）。

    铁律不变：拾取键永远只能由 A框裁决触发，走神/社交都不得绕过。
    """

    def __init__(
        self,
        cfg: HumanizeConfig,
        pickup_key: str = "a",
        rng: Any = None,
        seed: int | None = None,
    ) -> None:
        self.cfg = cfg
        self._rng = _rng_or_default(rng, seed)
        self.pickup_key = (pickup_key or "a").lower()
        self.fatigue = FatigueModel(cfg.fatigue, self._rng)
        # 情绪/偏好：影响"选哪个目标"，与疲劳共用随机源
        self.mood = MoodModel(cfg.mood, self.fatigue, self._rng)
        # 节奏：双态 + 1/f + 长尾。base 由调用方按任务节拍设定（rhythm.base）。
        self.rhythm = Rhythm(1.0, cfg.rhythm, rng=self._rng)

        self._break_until = 0.0
        self._break_started = 0.0
        self._breaking = False
        self.break_count = 0

        self._drift_next: float | None = None
        self._drift_cool_until = 0.0
        self.drift_count = 0

        self._social_next: float | None = None
        self.social_count = 0
        self.social_refused = 0

        if cfg.enabled and cfg.social.enabled:
            log.warning(
                "社交噪音已启用：台词会通过键盘输入进游戏。"
                "请确认聊天框打开键已标定，且台词不含任何游戏热键字符。"
            )

    # ---- 状态推进 ----

    def tick(self, now: float) -> None:
        if self.cfg.enabled:
            self.fatigue.tick(now)
            self.mood.tick(now)

    def note_action(self, now: float, cost: float = 1.0) -> None:
        if self.cfg.enabled:
            self.fatigue.note_action(now, cost)

    @property
    def rng(self) -> Any:
        """共享随机源（供目标选择等外部逻辑使用，便于统一 seed）。"""
        return self._rng

    @property
    def mood_active(self) -> bool:
        """情绪模型是否生效（拟人化总开关 + mood 子开关）。"""
        return bool(self.cfg.enabled and self.cfg.mood.enabled)

    # ---- 时序调整 ----

    def latency_scale(self, now: float) -> float:
        return self.fatigue.latency_scale() if self.cfg.enabled else 1.0

    def jitter_ratio(self, now: float, base: float) -> float:
        if not self.cfg.enabled:
            return base
        return max(0.0, min(0.6, base * self.fatigue.jitter_scale()))

    def error_prob(self, now: float, base: float = 0.0) -> float:
        return self.fatigue.error_prob(base) if self.cfg.enabled else base

    def interval_s(self, base_s: float, scale: float = 1.0) -> float:
        """按人类节奏给出一个间隔（秒）。

        `base_s` 是任务决定的节拍（例如 1/tick_hz），`scale` 叠加疲劳/监督降速。
        关闭拟人化时退化为 `base_s * scale`（确定性优先，和 `reaction_ms`
        返回 0 的契约一致）。

        为什么不能沿用"base × uniform(0.9,1.1)"：那个分布的支撑集是固定盒子，
        样本一多方差收敛、ACF 在 lag>1 直接掉到 0；而真人的节奏序列有长程
        相关（1/f）和长尾打断。见 `motion.Rhythm` 的分层说明。
        """
        if not self.cfg.enabled or not self.cfg.rhythm.enabled:
            return max(0.0, float(base_s) * max(0.0, float(scale)))
        return self.rhythm.next(max(0.0, float(scale)), float(base_s))

    def interval_ms(self, base_ms: float, scale: float = 1.0) -> int:
        return int(round(self.interval_s(base_ms / 1000.0, scale) * 1000.0))

    def reaction_ms(self) -> int:
        """看到 A框 → 按下拾取键的反应时间（对数正态，右偏长尾）。

        为什么不用均匀分布：人的反应时间分布是右偏的（偶尔"手慢"到区间外），
        均匀分布两头截断，样本一多统计上就不像人。

        契约：拟人化关闭时返回 0，由调用方回落到自己的最小确认窗（确定性优先）。
        """
        if not self.cfg.enabled:
            return 0
        lo, hi = (list(self.cfg.reaction_ms) + [180, 620])[:2]
        lo, hi = float(min(lo, hi)), float(max(lo, hi))
        if hi <= 0:
            return 0
        mu = math.log(max(1.0, (lo + hi) / 2.0))
        v = self._rng.lognormvariate(mu, max(0.01, self.cfg.reaction_sigma))
        v = min(hi * 2.5, max(lo * 0.6, v))          # 允许长尾，但不许离谱
        return int(max(0, v * self.fatigue.latency_scale()))

    def hesitation_s(self) -> float:
        """进关卡/点按钮前的"愣一下"（概率触发）。"""
        d = self.cfg.drift
        if not self.cfg.enabled or not d.enabled:
            return 0.0
        if self._rng.random() >= d.hesitate_prob:
            return 0.0
        lo, hi = (list(d.hesitate_s) + [0.3, 1.6])[:2]
        return self._rng.uniform(min(lo, hi), max(lo, hi))

    # ---- 疲劳休息（停手窗口） ----

    def paused(self, now: float) -> bool:
        """到阈值就停手歇一会儿。休息期间调用方不得输出任何命令。"""
        if not self.cfg.enabled or not self.cfg.fatigue.enabled:
            return False
        if self._breaking:
            if now < self._break_until:
                return True
            self._breaking = False
            self.fatigue.note_rest(now - self._break_started)
            self.break_count += 1
            return False
        if self.fatigue.needs_break():
            lo, hi = (list(self.cfg.fatigue.break_seconds) + [20.0, 90.0])[:2]
            self._break_started = now
            self._break_until = now + self._rng.uniform(min(lo, hi), max(lo, hi))
            self._breaking = True
            return True
        return False

    def break_seconds_left(self, now: float) -> float:
        return max(0.0, self._break_until - now) if self._breaking else 0.0

    # ---- 走神：非目标导向的无害噪音 ----

    def maybe_drift(
        self, now: float, scene_w: int, scene_h: int
    ) -> Optional[list[Command]]:
        """偶尔做点和当前任务无关的事。

        真人跑图会：鼠标划过某个无关角落、在背包里翻两下又关掉、进关卡前停一下。
        这类操作**不会被任务逻辑自然生成**，恰恰是真人行为里最常见的"噪音"。

        安全默认：只发 mouse_move（不点击、不按键）——
        移动光标在绝大多数游戏里没有副作用，是最便宜的噪音。
        """
        if not self.cfg.enabled or not self.cfg.drift.enabled:
            return None
        d: DriftConfig = self.cfg.drift
        if self._drift_next is None:
            # 启动宽限期：真人不会一上线就发呆
            self._drift_next = (now + max(0.0, d.startup_grace_s)
                                + self._rng.expovariate(1.0 / max(1.0, d.mean_interval_s)))
            return None
        if now < self._drift_next or now < self._drift_cool_until:
            return None

        # 指数分布：真人走神是泊松过程，不是"每 45 秒整一次"
        self._drift_next = now + self._rng.expovariate(1.0 / max(1.0, d.mean_interval_s))
        self._drift_cool_until = now + d.cooldown_s
        self.drift_count += 1

        cmds: list[Command] = []
        moves = list(d.moves) or [1]
        n = self._rng.randint(min(moves), max(moves))
        m = max(0, int(d.margin_px))
        w = max(m + 1, int(scene_w))
        h = max(m + 1, int(scene_h))
        for _ in range(max(1, n)):
            # 偏向画面边缘（余光扫视），不落在角色正下方 —— 那是"有目的"的位置
            if self._rng.random() < 0.5:
                x = self._rng.choice([m, max(m, w - m)])
            else:
                x = self._rng.randint(m, max(m, w - m))
            y = self._rng.randint(m, max(m, h - m))
            cmds.append(Command(op="mouse_move", x=int(x), y=int(y)))

        for k in d.noise_keys:
            if (k or "").lower() == self.pickup_key:
                log.error("拒绝把拾取键 %s 当走神噪音键（铁律：拾取键只能由 A框触发）", k)
                continue
            cmds.append(Command(op="key_tap", key=k, hold_ms=int(d.noise_key_hold_ms)))
        return cmds

    # ---- 社交噪音（默认关闭） ----

    def validate_phrase(self, phrase: str) -> tuple[bool, str]:
        """台词安全校验 —— 这是社交模块唯一的硬闸门。

        为什么必须拦：打字前要先聚焦聊天框，焦点没对时每个字符都会变成游戏热键。
        台词里只要有一个拾取键/移动键字符，就等于"盲按拾取键"，直接违反铁律。
        所以宁可拒绝发送，也不"尽力发送"。
        """
        s: SocialConfig = self.cfg.social
        if not phrase:
            return False, "空台词"
        if len(phrase) > s.max_phrase_len:
            return False, f"超长（>{s.max_phrase_len}）"
        low = phrase.lower()
        forbidden = set((s.forbidden_chars or "").lower()) | {self.pickup_key}
        bad = sorted({c for c in low if c in forbidden})
        if bad:
            return False, f"含禁用字符 {''.join(bad)}（会被游戏当成热键）"
        allowed = set("abcdefghijklmnopqrstuvwxyz0123456789 .,!?")
        bad2 = sorted({c for c in low if c not in allowed})
        if bad2:
            return False, f"含执行器键盘映射不支持的字符 {bad2[0]!r}"
        return True, ""

    def maybe_social(self, now: float) -> Optional[list[Command]]:
        """偶尔在公共频道说句废话（默认关闭，见 SocialConfig 的风险说明）。"""
        if not self.cfg.enabled or not self.cfg.social.enabled:
            return None
        s: SocialConfig = self.cfg.social
        if not s.phrases:
            return None
        if self._social_next is None:
            self._social_next = now + s.min_interval_s * self._rng.uniform(0.6, 1.6)
            return None
        if now < self._social_next:
            return None
        self._social_next = now + s.min_interval_s * self._rng.uniform(0.6, 1.6)

        phrase = self._rng.choice(s.phrases)
        ok, why = self.validate_phrase(phrase)
        if not ok:
            self.social_refused += 1
            log.warning("拒绝发送社交台词 %r：%s", phrase, why)
            return None

        self.social_count += 1
        cmds = [Command(op="key_tap", key=s.chat_open_key, hold_ms=60)]
        for ch in phrase:
            key = "space" if ch == " " else ch.lower()
            cmds.append(Command(op="key_tap", key=key, hold_ms=int(s.char_hold_ms)))
        cmds.append(Command(op="key_tap", key=s.chat_send_key, hold_ms=60))
        return cmds

    # ---- 观测 ----

    def snapshot(self) -> dict:
        return {
            "enabled": self.cfg.enabled,
            "fatigue": round(self.fatigue.level, 3),
            "arousal": round(self.fatigue.arousal, 3),
            "latency_scale": round(self.fatigue.latency_scale(), 2),
            "actions": self.fatigue.actions,
            "breaks": self.break_count,
            "breaking": self._breaking,
            "drifts": self.drift_count,
            "rhythm": self.rhythm.snapshot(),
            "mood": self.mood.snapshot() if self.mood_active else {"enabled": False},
            "social": {
                "enabled": self.cfg.social.enabled,
                "sent": self.social_count,
                "refused": self.social_refused,
            },
        }
