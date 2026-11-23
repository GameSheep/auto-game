"""执行器基类（独立模块，避免循环导入）。"""

from __future__ import annotations

import abc
import contextlib
import logging
import math
import random
import threading
import time
from typing import Callable, Sequence

from ..config import ExecutorConfig
from ..humanize import PathPoint, build_mouse_path
from ..motion import Mash, MashPlan
from ..protocol import Command
from ..security import ActionRateLimiter

log = logging.getLogger(__name__)

#: 自旋收尾窗口（秒）。sleep 的精度不够，最后这一段用忙等兜住。
_SPIN_S = 0.0015


def _sleep_until(deadline: float) -> None:
    """精确睡到 `deadline`（perf_counter 基准）。

    实测结论（本机 Windows 11 + Python 3.12，40 次 × 5 档）：
      - 裸 `time.sleep(4/8/12/20/33ms)` 的均值误差 0.4~0.5ms，**没有**老资料里
        说的 15.6ms 量化 —— 现代 Windows 的定时器精度已经够了；
      - 但单次最大误差能到 1.4ms，且 Linux/小板子（RK3566）上的调度抖动更大。

    所以这里不是"修好了 15.6ms 量化"，而是：
      1. 用**绝对 deadline**而不是相对 sleep —— `move()` 本身的耗时（HID 写一次
         1-3ms）会被自动吸收，整段轨迹不会越走越慢（这才是真问题）；
      2. sleep 提前返回时用自旋兜住最后 1.5ms（Linux 上 sleep 提前返回更常见）；
      3. 自旋只占整段的 5~20%，不会把一个核吃满。
    """
    while True:
        left = deadline - time.perf_counter()
        if left <= 0:
            return
        if left > _SPIN_S * 2:
            time.sleep(left - _SPIN_S)
        else:
            while time.perf_counter() < deadline:
                pass
            return


def _jitter(value_ms: int, ratio: float) -> int:
    """按比例抖动（**对数正态**，右偏，均值保持）。

    为什么不是 uniform：对称抖动的"长按"和"短按"概率相等，而真人按键是
    "松得快"更常见（右偏）；而且 uniform 的支撑集固定，样本一多方差收敛。
    σ 用 ratio 本身，均值用 exp(μ+σ²/2) 反推 μ，保证不会整体变长。
    """
    if ratio <= 0:
        return value_ms
    v = max(1.0, float(value_ms))
    sigma = max(0.01, min(1.0, float(ratio)))
    mu = math.log(v) - 0.5 * sigma * sigma
    return max(10, int(round(math.exp(random.gauss(mu, sigma)))))


def _rand_in(pair: Sequence[int] | None, default: tuple[int, int]) -> int:
    """从 [lo, hi] 区间取随机整数（区间缺失/非法时回落到 default）。"""
    lo, hi = default
    if pair:
        vals = list(pair)[:2]
        if len(vals) == 2:
            lo, hi = vals
    lo, hi = int(min(lo, hi)), int(max(lo, hi))
    return random.randint(max(0, lo), max(0, hi))


class BaseExecutor(abc.ABC):
    """执行器基类：限速 + 急停 + 统计 + 人类化鼠标轨迹。"""

    #: 后端是否支持绝对坐标鼠标移动（机械舵机按压后端为 False，走位需配 keys 模式）
    supports_mouse_move: bool = True

    def __init__(self, cfg: ExecutorConfig) -> None:
        self.cfg = cfg
        self._limiter = ActionRateLimiter(
            cfg.min_action_interval_ms, cfg.max_actions_per_sec,
            getattr(cfg, "rate_floor_jitter", 0.0))
        self._rate_sig = (cfg.min_action_interval_ms, cfg.max_actions_per_sec,
                          getattr(cfg, "rate_floor_jitter", 0.0))
        self._panic = threading.Event()
        self.sent = 0
        self.dropped = 0
        self.journal: list[dict] = []  # 最近动作日志（仪表盘/测试断言用）
        self._lock = threading.Lock()
        # 光标位置：作为下一次轨迹的起点（None=未知，首帧退化为单点）
        self.cursor: tuple[int, int] | None = None
        self.last_path: list[PathPoint] = []
        self.path_points_total = 0
        # 本次点击手势参数（execute 里定，后端只管执行）
        self.click_hover_ms = 0
        self.click_press_ms = 0
        #: 动作闸门（观察/暂停模式）。返回 False 时动作一律不下发，但照常计数。
        #: None = 总是放行（没装助手的时候就是这个状态）。
        self.action_gate: Callable[[], bool] | None = None
        #: 被闸门拦下的动作数。页面上要能看出"它想动但没动"，否则观察模式
        #: 看起来和"什么都不做"一模一样。
        self.suppressed = 0
        #: 连按/连点（技能狂按、双击、三连点）。输出层手势，见 motion.Mash。
        #: None = 关闭（老配置文件没有 mash 段时也走这条，行为与旧版一致）。
        self.mash: Mash | None = Mash(getattr(cfg, "mash", None), random.Random())
        #: 连按白名单之外还要排掉的键（拾取键/逃生键由决策层注入）。
        #: ⚠️ 拾取键必须在这里 —— 它连按等于绕开 A 框确认闸门。
        self.burst_forbid: set[str] = set()
        #: 本次连按实际按下几下（=1 表示没连按），复盘/审计用。
        self.last_burst_count = 1
        self.burst_taps = 0
        self.burst_aborted = 0
        #: 连按暂停计数器（可嵌套）。配方/UI 自动化期间置位。
        self._burst_pause = 0

    #: 后端名（子类覆盖）。放基类是为了探活的默认失败信息能说清是哪个后端。
    backend_name: str = "base"

    #: 这个后端会不会真的动手（dryrun = 不会）。自检要靠它判断"这台机器上
    #: 到底有没有人在按键盘"，否则联调通过 = 真机上没验证，是最常见的翻车方式。
    actuates: bool = True

    #: 这个后端吃不吃得下"连按"。舵机（pca9685）单次行程 100ms+，
    #: 60ms 的连按间隔它根本回不了位 —— 结果是按住不放 + 机械磨损，
    #: 比"没有连按"更假也更伤设备。所以由后端自己声明，不靠配置兜。
    supports_burst: bool = True

    def probe(self) -> dict:
        """探活：**这台机器上动作能不能真的发出去**。

        返回 `{ok, backend, actuates, detail, hint}`。`ok=False` 的意思是
        "现在上真机，动作会石沉大海"，`hint` 给出具体该查什么 —— 比把
        `SerialException` 原样抛给人有用得多。

        默认实现刻意返回失败而不是 True：新加的后端如果忘了实现探活，
        自检应该拦下来（或至少说"这个后端没有探活"），不能默默放行。
        """
        return {"ok": False, "backend": self.backend_name, "actuates": self.actuates,
                "detail": f"{self.backend_name} 还没实现探活",
                "hint": "这个后端没有探活接口，无法确认动作能不能发出去"}

    @property
    def panicked(self) -> bool:
        return self._panic.is_set()

    def panic(self) -> None:
        """急停：立刻置位并输出 stop_all。"""
        self._panic.set()
        try:
            self._emit(Command(op="stop_all"), bypass=True)
        except Exception:  # noqa: BLE001
            log.exception("急停输出失败")
        log.warning("执行器已急停")

    def reset(self) -> None:
        self._panic.clear()
        self._limiter.reset()
        log.info("执行器复位")

    def execute(self, cmd: Command) -> bool:
        if self._panic.is_set() and cmd.op != "stop_all":
            self.dropped += 1
            return False
        if cmd.op != "stop_all" and self.action_gate is not None \
                and not self.action_gate():
            # 观察/暂停模式：想动，但不动。丢掉的计数要留着 —— 复盘时要能说
            # "我当时想做 A 但被暂停了"，而不是看起来整段什么都没发生。
            self.suppressed += 1
            with self._lock:
                self.journal.append(
                    {"ts": time.time(), "op": cmd.op, "key": cmd.key,
                     "x": cmd.x, "y": cmd.y, "button": cmd.button,
                     "hold_ms": cmd.hold_ms, "path": 0, "press_ms": 0,
                     "suppressed": True})
                if len(self.journal) > 500:
                    del self.journal[:-300]
            return False
        if cmd.op != "stop_all":
            # 限速参数每帧跟随 config（配置页热更新后立即生效，不用重启）
            rate = (self.cfg.min_action_interval_ms, self.cfg.max_actions_per_sec,
                    getattr(self.cfg, "rate_floor_jitter", 0.0))
            if rate != self._rate_sig:
                self._rate_sig = rate
                self._limiter.sync(*rate)
            # 最小间隔：由限速器自己等到可发送（不依赖 sleep 精度，见 acquire 注释）
            if not self._limiter.acquire(0.25):
                self.dropped += 1
                return False
        # 鼠标类命令先规划人类化轨迹（WindMouse），后端按点输出
        self.last_path = self._plan_mouse_path(cmd)
        # 点击手势参数（悬停/按压）在这里定，后端只管执行 —— 日志才能如实记录
        if cmd.op == "mouse_click":
            self.click_hover_ms = _rand_in(self.cfg.mouse_path.hover_ms, (30, 110))
            self.click_press_ms = _rand_in(self.cfg.mouse_path.press_ms, (55, 135))
        else:
            self.click_hover_ms = 0
            self.click_press_ms = 0
        # 连按计划必须在发出**第一下之前**定下来：第一下的按压时长要跟着缩短
        # （连打时手指不在键上停留），事后再补就不算"这一次手势"了。
        plan = self._plan_mash(cmd)
        burst = plan.count if plan is not None else 1
        self.last_burst_count = burst
        if plan is not None and cmd.op == "key_tap":
            cmd = cmd.model_copy(update={"hold_ms": plan.first_hold_ms})
        try:
            self._emit(cmd)
            if plan is not None:
                self._run_burst(cmd, plan)
        except Exception:  # noqa: BLE001
            log.exception("执行命令失败 op=%s", cmd.op)
            return False
        self.sent += 1
        if self.last_path:
            self.cursor = (self.last_path[-1].x, self.last_path[-1].y)
        with self._lock:
            self.journal.append(
                {"ts": time.time(), "op": cmd.op, "key": cmd.key, "x": cmd.x, "y": cmd.y,
                 "button": cmd.button, "hold_ms": cmd.hold_ms, "path": len(self.last_path),
                 "press_ms": self.click_press_ms, "burst": burst}
            )
            if len(self.journal) > 500:
                del self.journal[:-300]
        return True

    # ------------- 连按 / 连点 -------------

    @contextlib.contextmanager
    def burst_paused(self):
        """临时关闭连按 —— 配方（交易行/处理装备/换技能）执行期间用。

        配方是按步骤精确执行的：多按两下就点错槽位、买卖错物品。
        "像人"在这类 UI 自动化里是负价值 —— 真人手点配方也会点得很小心。
        """
        self._burst_pause += 1
        try:
            yield
        finally:
            self._burst_pause = max(0, self._burst_pause - 1)

    def set_burst_forbid(self, keys: Sequence[str]) -> None:
        """注入永不连按的键（拾取键、逃生键）。决策层构造时调用。

        为什么不能只靠配置里的 `keys` 白名单：白名单是可编辑的，而拾取键
        连按会直接绕开 A 框确认（按 3 次 = 三次盲按的机会）。这条是铁律，
        必须由运行时注入，写死在白名单里早晚被一次"临时加个键"改掉。
        """
        self.burst_forbid = {(k or "").lower() for k in keys if k}
        if self.mash is not None:
            self.mash.forbid |= self.burst_forbid

    def _plan_mash(self, cmd: Command) -> MashPlan | None:
        """这条命令要不要连按。只对 key_tap / mouse_click 有意义。

        key_down/key_up 是长按（走位），把它们连按会变成"抽搐"而不是"连打"。
        """
        if not self.supports_burst or self._burst_pause:
            return None
        if self.mash is None or self.cfg.mash is None:
            return None
        if not getattr(self.cfg.mash, "enabled", True):
            return None
        if cmd.op not in ("key_tap", "mouse_click"):
            return None
        if (cmd.key or "").lower() in self.burst_forbid:
            return None
        return self.mash.plan(cmd.op, cmd.key)

    def _run_burst(self, cmd: Command, plan: MashPlan) -> None:
        """补发第 2..n 下（第一下已经在 execute 里发出）。

        时间轴用**绝对 deadline**（和 `_walk_path` 同理）：`_emit` 自身有耗时
        （串口写一次 1-3ms），相对 sleep 会把它累加进去，整簇越按越慢 ——
        这正好和"越按越快"的人类特征反号，等于换个方向暴露。

        每一下都查三道中断：急停、动作闸门（切观察/暂停）、以及后端异常。
        连按是**同一个逻辑动作**的重复，像轨迹点一样绕过限速器；速率上限由
        `Mash` 自己的时长预算（budget_ratio）兜，不靠 ActionRateLimiter。
        """
        now = time.perf_counter()
        for tap in plan.taps:
            now += tap.wait_s
            _sleep_until(now)
            if self._panic.is_set() or (
                    self.action_gate is not None and not self.action_gate()):
                # 闸门/急停优先于"把这一簇按完"：宁可少按两下，不能多按一下
                self.burst_aborted += 1
                return
            sub = cmd.model_copy(update={"hold_ms": tap.hold_ms})
            saved_path = self.last_path
            saved_hover, saved_press = self.click_hover_ms, self.click_press_ms
            if cmd.op == "mouse_click":
                # 连点不重新走轨迹：手已经在目标上。落点给 1-3px 高斯抖动 ——
                # 真人的连点不会像素级落在同一点，显示器/手都在动。
                self.last_path = []
                jx = int(round(random.gauss(0.0, 1.2)))
                jy = int(round(random.gauss(0.0, 1.2)))
                sub = sub.model_copy(update={"x": (cmd.x or 0) + jx, "y": (cmd.y or 0) + jy})
                self.click_hover_ms = 0
                # 连点时按压更短（手指在连击，不会停顿）
                self.click_press_ms = max(
                    22, int(self.click_press_ms * random.uniform(0.55, 0.85)))
            try:
                self._emit(sub)
            finally:
                self.last_path = saved_path
                self.click_hover_ms, self.click_press_ms = saved_hover, saved_press
            self.burst_taps += 1
            now = max(now, time.perf_counter())

        # 收尾停顿（看一眼技能出来没有）。分段睡，急停能立刻打断。
        end = now + max(0.0, plan.tail_s)
        while True:
            left = end - time.perf_counter()
            if left <= 0 or self._panic.is_set():
                break
            time.sleep(min(0.05, left))

    # ------------- 人类化鼠标轨迹 -------------

    def _plan_mouse_path(self, cmd: Command) -> list[PathPoint]:
        """为 mouse_move / mouse_click 规划轨迹；其他命令返回空列表。"""
        if cmd.op not in ("mouse_move", "mouse_click"):
            return []
        if cmd.x is None or cmd.y is None:
            return []
        tx, ty = int(cmd.x), int(cmd.y)
        start = self.cursor if self.cursor is not None else (tx, ty)
        path = build_mouse_path(start, (tx, ty), self.cfg.mouse_path)
        self.path_points_total += len(path)
        return path

    def _walk_path(self, path: list[PathPoint], move) -> None:
        """按轨迹逐点输出（move(x, y) 由后端实现），点间按 dt 等待。

        时间轴用**绝对 deadline 累加**而不是每点 sleep(dt)：`move` 本身要
        花时间（HID 写一次 1-3ms），相对 sleep 会把这部分误差累积进去，
        整段越走越慢；累加 deadline 则自动吸收。

        注意：轨迹点在 _emit 内部输出，绕过 ActionRateLimiter ——
        它们属于**同一条**逻辑命令，不应被限速器拆散或丢弃。
        """
        now = time.perf_counter()
        for p in path:
            if p.dt > 0:
                now += p.dt
                _sleep_until(now)
            move(p.x, p.y)
            # 万一 move 比 dt 还慢（后端卡顿），别把后续点堆成"一次性喷出"
            now = max(now, time.perf_counter())

    @abc.abstractmethod
    def _emit(self, cmd: Command, bypass: bool = False) -> None:
        """后端实际输出（绕过统计的急停路径由 panic 调用）。"""

    def close(self) -> None:
        pass




# ---------------- HID 报文构建（纯函数，字节级可测） ----------------

# /dev/hidg1 绝对鼠标报文：6 字节 [btn, x_lo, x_hi, y_lo, y_hi, wheel]
# 描述符见 scripts/setup_rpi5_hidg.sh（8按钮 + 16bit X/Y(1..32767) + 滚轮）
def build_abs_mouse_report(x: int, y: int, screen_w: int, screen_h: int,
                           button: int = 0x00, wheel: int = 0x00) -> bytes:
    ax = max(0, min(32767, int(x * 32767 / max(1, screen_w - 1))))
    ay = max(0, min(32767, int(y * 32767 / max(1, screen_h - 1))))
    return bytes([button & 0xFF, ax & 0xFF, ax >> 8, ay & 0xFF, ay >> 8, wheel & 0xFF])


# /dev/hidg0 标准键盘报文：8 字节 [mod, reserved, k1..k6]
def build_keyboard_report(key_code: int = 0x00, modifier: int = 0x00) -> bytes:
    report = bytearray(8)
    report[0] = modifier & 0xFF
    if key_code:
        report[2] = key_code & 0xFF
    return bytes(report)
