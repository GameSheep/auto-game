"""放行度 —— 系统里所有"不许动 / 慢点动 / 先问一下"的开关，一张表说清。

为什么要单独有这个模块：
  限制不是不存在，而是**散落在四十多个配置字段里**。用户在真机上感觉"跑得太慢、
  半天才捡一个"，但打开 config.yaml 只能看到一堆 20.0 / 0.5 / 8.0，不知道
  哪个才是瓶颈。于是只能凭手感一个个改，改完不知道有没有用，也不敢改回去。

  这里做三件事：
    1. **登记**：每条限制一行 —— 它防的是什么、当前值多少、放宽会付出什么代价。
    2. **三档**：strict / normal / unleashed 一键切，只动"节奏类"字段。
    3. **对账**：拿运行统计回答"这条限制到今天为止拦了你多少次"。
       没有这一步，"限制太多"永远是个感觉，说不清是哪一条。

一条都不能碰的东西（`SAFETY`）单独列出来，且**不在任何档位的 patch 里**：
  拾取三道闸、场景门 fail_closed、坐标归一化、拾取键不入方向键。
  这些不是"保守"，是正确性。放宽它们不会变快，只会变错。

为什么档位只改配置、不改代码：
  改代码要走整包换代（可回滚），而"今天想跑快点"是个日常操作。
  把可调的都做成配置，剩下的才值得动代码 —— 这条线松了，就没法回滚了。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import AppConfig

#: 档位，从紧到松。顺序有意义（`current_tier` 按这个顺序找最近的一档）。
TIERS: tuple[str, ...] = ("strict", "normal", "unleashed")

TIER_TITLES: dict[str, dict[str, str]] = {
    "strict": {
        "title": "保守（默认）",
        "summary": "按出厂值跑。每一步都留足余量，最适合第一次上真机、画面还没调好的阶段。",
        "cost": "慢。决策 1Hz、轨迹 0.45s、累了就休息 20-90s。",
    },
    "normal": {
        "title": "平衡",
        "summary": "节奏翻倍，拟人化保留但减半。稳定跑通、画面标定过之后用这一档。",
        "cost": "吞吐约 ×1.8；轨迹更短，落点抖动略小。",
    },
    "unleashed": {
        "title": "搬砖（最松）",
        "summary": "把节奏类的限制全部推到安全下限：决策 4Hz、轨迹 50ms、几乎不休息、"
                   "走位超时更长、模型问得更勤。",
        "cost": "吞吐约 ×3-4；拟人度下降，长期统计特征更明显；CPU 占用上升。",
    },
}

#: 限制类别 —— 页面按这个分组，用户才知道自己在调哪一层。
KINDS: dict[str, str] = {
    "pace": "节奏（每秒能做多少事）",
    "chain": "动作链（一个动作要等多久）",
    "persona": "拟人化（为了看起来像人而故意变慢）",
    "autonomy": "自救（卡住后多久才动手）",
    "model": "模型预算（多久才允许问一次）",
}


@dataclass(frozen=True)
class Limit:
    """一条可调限制。

    `values` 三档的值必须能在 pydantic 约束内（不然 apply_patch 会整批拒绝，
    而用户只看到一句"配置校验失败"，不知道是表格里哪一行写错了）。
    """

    id: str                      # 点分配置路径
    label: str
    kind: str
    why: str                     # 这条限制在防什么（不写清就没法判断该不该松）
    values: tuple[Any, Any, Any]  # strict / normal / unleashed

    def value_for(self, tier: str) -> Any:
        if tier not in TIERS:
            raise KeyError(f"未知档位: {tier}")
        return self.values[TIERS.index(tier)]


#: 可调限制。**顺序 = 页面显示顺序 = 从"最影响吞吐"到"最不影响"**。
LIMITS: tuple[Limit, ...] = (
    # ---------------- 节奏：直接决定每秒能做多少动作 ----------------
    Limit(
        "capture.fps", "采集帧率（=决策频率）", "pace",
        "代理每秒抓几帧，服务端**每来一帧决策一次**。这是真正的决策频率上限；"
        "改完会立刻推给已连接的代理。注意：一帧检测约 30~130ms（场景门占大头），"
        "fps 高过 1000/检测耗时 时，多出来的帧会被当过期帧丢掉 —— 那不是更快，只是更热。",
        (2.0, 5.0, 10.0),
    ),
    Limit(
        "decision.tick_hz", "走位节拍基准", "pace",
        "两次走位/探索/导航动作之间的基准间隔（1/tick_hz 秒）。它管的是"
        "「多久才允许再点一次地面」，不是决策频率 —— 决策是帧驱动的。",
        (1.0, 2.0, 4.0),
    ),
    Limit(
        "executor.min_action_interval_ms", "动作最小间隔", "pace",
        "两条命令之间强制等多久。防连点被游戏吞掉，也防串口/HID 缓冲溢出。",
        (25, 12, 5),
    ),
    Limit(
        "executor.max_actions_per_sec", "每秒动作上限", "pace",
        "限速器的硬上限，超过直接丢弃（丢的命令不进 journal，只在 dropped 里计数）。",
        (20, 35, 60),
    ),
    Limit(
        "decision.action_desync_ms", "动作相位打散", "pace",
        "点击/按键前随机等 0~N 毫秒，把动作时刻从采集帧边界上挪开。"
        "**三档都保持 400**：不加这条，capture.fps=2 时间隔只可能是 500ms 和 "
        "1000ms 两个值（锁相 R=1.0，实测），这是「操作间隔高度标准化」的根，"
        "而且加多少随机间隔都治不了 —— 量化发生在最后一公里。"
        "它不改变动作语义，只改相位；代价是每次点击最多多占用决策线程这么久。",
        (400, 400, 400),
    ),
    Limit(
        "executor.mash.budget_ratio", "连按预算", "pace",
        "窗口内「连按/连点」允许占总时长的比例上限。它约束的是**拟人化本身**："
        "没有这条，连按会慢慢吃掉一半时间，人就变成「一直狂按同一个键」——"
        "那比不连按更假，而且刷图吞吐也没了。",
        (0.22, 0.30, 0.38),
    ),
    Limit(
        "executor.rate_floor_jitter", "最小间隔抖动", "pace",
        "限速器的最小间隔**本身**每次重抽。下界固定时所有动作都贴着同一个值，"
        "间隔直方图有一堵陡边 —— 这是可测的量化特征。只影响下限，不动速率上限。",
        (0.15, 0.15, 0.18),
    ),
    Limit(
        "decision.rechallenge_cooldown_s", "结算冷却", "pace",
        "点完『再次挑战』后强制等多久才允许下一个动作，防结算动画没出就乱点。",
        (3.0, 2.0, 1.2),
    ),

    # ---------------- 动作链：一次拾取要串多久 ----------------
    Limit(
        "decision.tap_recheck_ms", "拾取复核冷却", "chain",
        "按完 A 等多久才允许再按。低于游戏拾取动画时间会按空（白按一下，还占动作配额）。",
        (500, 350, 220),
    ),
    Limit(
        "decision.aframe_retry_frames", "A框按住重走", "chain",
        "A框一直在但按不下去，多少帧后重新走位。太大会原地死等一个够不到的目标。",
        (8, 6, 4),
    ),
    Limit(
        "decision.approach_timeout_s", "走位超时", "chain",
        "一个目标最多走多久就放弃拉黑。放宽 = 愿意走更远（漏检少了，但可能白走）。",
        (8.0, 12.0, 16.0),
    ),
    Limit(
        "decision.abandon_blacklist_s", "放弃后拉黑时长", "chain",
        "放弃的目标/区域多久内不再考虑。太长会把「其实还能捡」的整片区域屏蔽掉。",
        (30.0, 12.0, 6.0),
    ),
    Limit(
        "decision.walk.budget_s", "单次走位预算", "chain",
        "点一下地面最多等人走多久。比真实走位慢就会反复重发点击（浪费动作）。",
        (3.0, 4.0, 5.0),
    ),
    Limit(
        "decision.walk.min_progress_px", "走位有效判定", "chain",
        "这一轮至少靠近多少像素才算「走动了」。太严会把正常走位判成没走，然后重复点。",
        (12.0, 9.0, 6.0),
    ),
    Limit(
        "decision.walk.settle_grace_s", "停下后等 A 框", "chain",
        "人物停下后再给游戏多久弹 A 框提示。",
        (0.5, 0.35, 0.25),
    ),

    # ---------------- 拟人化：为了"像人"主动变慢的部分 ----------------
    Limit(
        "decision.humanize.fatigue.per_action", "疲劳累积速度", "persona",
        "每次动作累积多少疲劳。0.003 ≈ 280 次动作就累到必须休息。",
        (0.003, 0.0015, 0.0006),
    ),
    Limit(
        "decision.humanize.fatigue.break_threshold", "疲劳停手阈值", "persona",
        "疲劳到多少就完全停手休息。这是「跑一晚上，实际只干了半晚」的元凶。",
        (0.85, 0.93, 0.98),
    ),
    Limit(
        "decision.humanize.fatigue.break_seconds", "单次休息时长", "persona",
        "每次累到停手要休息多久（区间）。",
        ([20.0, 90.0], [10.0, 30.0], [4.0, 10.0]),
    ),
    Limit(
        "decision.humanize.reaction_ms", "看到 A 框的反应", "persona",
        "检测到 A 框到真正按下拾取键之间的延时。1Hz 下这个延时会被放大成一整帧。",
        ([180, 620], [120, 400], [70, 220]),
    ),
    Limit(
        "decision.humanize.drift.mean_interval_s", "走神频率", "persona",
        "平均多久发一次纯噪音鼠标移动（不点击不按键）。只影响统计特征，不影响产出。",
        (45.0, 120.0, 300.0),
    ),
    Limit(
        "decision.humanize.mood.skip_prob", "懒得捡概率", "persona",
        "单个候选被无视的基础概率。0.02 看着小，捡几百次就是白丢几件。",
        (0.02, 0.005, 0.0),
    ),
    Limit(
        "executor.mouse_path.max_duration_s", "单次轨迹最长时长", "persona",
        "一次鼠标移动最多画多久。轨迹本身是**阻塞**的，0.45s 意味着每次点击都占半秒。",
        (0.45, 0.12, 0.06),
    ),
    Limit(
        "executor.mouse_path.max_points", "轨迹点数上限", "persona",
        "轨迹点越多越像人，但每个点都是一次串口/HID 报文。",
        (40, 16, 8),
    ),
    Limit(
        "executor.mouse_path.speed_px_s", "名义移动速度", "persona",
        "轨迹规划用的速度。越大轨迹越短，落到目标越快。",
        (900.0, 2000.0, 4000.0),
    ),
    Limit(
        "executor.mouse_path.hover_ms", "到位后悬停", "persona",
        "光标到位后停多久才按下。真人为「看清再点」，代价是每次点击多几十毫秒。",
        ([30, 110], [15, 45], [0, 15]),
    ),
    Limit(
        "executor.mouse_path.press_ms", "按键保持时长", "persona",
        "左键按下到抬起。游戏只认「按下」这个事件，保持多久基本不影响结果。",
        ([55, 135], [40, 90], [30, 60]),
    ),

    # ---------------- 自救：卡住之后多久才动手 ----------------
    Limit(
        "decision.autonomy.frozen_window_s", "画面静止判卡住", "autonomy",
        "画面完全不动多久算撞墙/失焦。太长 = 卡在墙角干等。",
        (12.0, 8.0, 6.0),
    ),
    Limit(
        "decision.autonomy.idle_window_s", "无目标判空闲", "autonomy",
        "多久没有任何可交互目标就开始自己找。",
        (20.0, 12.0, 8.0),
    ),
    Limit(
        "decision.autonomy.stall_window_s", "无进展判停滞", "autonomy",
        "多久没有任何有效进展就进恢复阶梯（这一条关系到「跑一夜原地打转」）。",
        (90.0, 60.0, 45.0),
    ),
    Limit(
        "decision.autonomy.gate_window_s", "不在战斗界面多久按 ESC", "autonomy",
        "弹窗/菜单挡屏多久后按 ESC 自救。必须小于 watchdog.no_game_stop_s（否则还没自救就被停）。",
        (25.0, 15.0, 10.0),
    ),
    Limit(
        "decision.autonomy.level_interval_s", "恢复升级间隔", "autonomy",
        "恢复手段多久升级一档（换方向 → 闪避 → ESC → 重开）。",
        (8.0, 5.0, 3.0),
    ),
    Limit(
        "decision.autonomy.cycle_backoff_s", "恢复轮次退避", "autonomy",
        "每轮恢复失败后额外等多久，防「卡住 → 疯狂乱按」。",
        (15.0, 6.0, 2.0),
    ),

    # ---------------- 模型预算：多久才允许问一次 ----------------
    Limit(
        "decision.arbiter.cooldown_s", "裁决最小间隔", "model",
        "两次向三级裁决求助之间至少隔多久。太长 = 规则解不了的局面里干等。",
        (20.0, 8.0, 3.0),
    ),
    Limit(
        "decision.arbiter.idle_trigger_s", "空闲多久求助", "model",
        "无事可做多少秒后问模型。",
        (8.0, 5.0, 3.0),
    ),
    Limit(
        "decision.arbiter.unmatched_trigger_s", "未标定多久求助", "model",
        "画面一直认不出持续多少秒后求助。",
        (15.0, 8.0, 5.0),
    ),
    Limit(
        "decision.arbiter.plan_ttl_s", "计划有效期", "model",
        "模型给的计划多久后作废。太短 = 还没执行完就丢；太长 = 局面变了还在走老路。",
        (10.0, 20.0, 30.0),
    ),
    Limit(
        "decision.vlm.max_calls_per_min", "视觉模型每分钟上限", "model",
        "截图问多模态模型的预算。",
        (6, 12, 24),
    ),
    Limit(
        "decision.vlm.min_interval_s", "视觉模型最小间隔", "model",
        "两次截图问答之间至少隔多久。",
        (8.0, 4.0, 2.0),
    ),
    Limit(
        "decision.supervisor.max_calls_per_min", "监督层每分钟上限", "model",
        "监督层对账的预算。",
        (3, 8, 16),
    ),
    Limit(
        "decision.supervisor.min_interval_s", "监督层最小间隔", "model",
        "两次监督判断之间至少隔多久。",
        (20.0, 8.0, 4.0),
    ),
    Limit(
        "decision.llm.cooldown_s", "文本顾问最小间隔", "model",
        "两次咨询文本模型之间至少隔多久。",
        (60.0, 20.0, 8.0),
    ),
)


@dataclass(frozen=True)
class Safety:
    """安全铁律：**任何档位都不动**。

    列出来不是为了让人调，是为了让"为什么这条不能松"有地方写。
    页面照原样展示，只读。
    """

    label: str
    what: str
    where: str


SAFETY: tuple[Safety, ...] = (
    Safety(
        "拾取三道闸",
        "场景门命中 + A框在目标附近 + 目标仍然存在，三条同时为真才允许按拾取键。"
        "少一条都会变成「盲按」，而盲按是唯一会直接导致封号的违规动作。",
        "decision/fsm.py:_try_pick",
    ),
    Safety(
        "A框连续确认帧数",
        "aframe_confirm_frames ≥ 1，且距离不能变大。帧是过去的信息，按空比慢一点严重得多。",
        "decision.aframe_confirm_frames",
    ),
    Safety(
        "场景门 fail_closed",
        "一个锚点都没命中时拦下，而不是当战斗处理。未标定的画面绝不动手。",
        "vision.scene_gate.fail_closed",
    ),
    Safety(
        "拾取键不得当方向键",
        "任何移动/恢复/探索指令里出现拾取键，一律拦截并计入 blind_tap_violations（必须恒为 0）。",
        "decision/fsm.py:_move_key",
    ),
    Safety(
        "鼠标坐标归一化",
        "像素 → 0..32767 只在一个出口换算，分母 screen_w-1。两处各算一次就会整体偏位。",
        "executor/__init__.py:_to_abs",
    ),
    Safety(
        "帧尺寸 = 屏幕尺寸",
        "抓到的帧宽高必须等于 executor.screen_w/h，否则点击全部按比例偏移（日志还是全绿）。",
        "preflight.py:check_frame_size",
    ),
    Safety(
        "模式闸门",
        "观察/暂停模式下动作一律不下发（只计数）。页面按钮、语音、模型建议都必须过它。",
        "assistant/modes.py:install_gate",
    ),
)


def _nest(path: str, value: Any) -> dict[str, Any]:
    """点分路径 → 嵌套 dict（只造结构，不校验）。"""
    out: dict[str, Any] = {}
    cur = out
    keys = path.split(".")
    for k in keys[:-1]:
        cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value
    return out


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def tier_patch(name: str) -> dict[str, Any]:
    """档位 → 嵌套 patch（可直接喂给 `settings.apply_patch`）。"""
    if name not in TIERS:
        raise KeyError(f"未知档位: {name}（可选 {', '.join(TIERS)}）")
    patch: dict[str, Any] = {}
    for lim in LIMITS:
        patch = _merge(patch, _nest(lim.id, lim.value_for(name)))
    return patch


def _get_path(cfg: AppConfig, path: str) -> Any:
    obj: Any = cfg
    for p in path.split("."):
        obj = getattr(obj, p)
    return obj


def current_tier(cfg: AppConfig) -> str:
    """当前配置最接近哪一档。

    判据是"三档距离之和最小"，不是"完全相等" —— 用户手改过两三个滑块之后，
    不该显示成"未知"。距离为 0 时就是那一档（顺序上 strict 优先）。
    """
    best, best_score = "strict", None
    for tier in TIERS:
        score = 0.0
        for lim in LIMITS:
            want = lim.value_for(tier)
            got = _get_path(cfg, lim.id)
            score += _distance(got, want)
        if best_score is None or score < best_score:
            best, best_score = tier, score
    return best


def _distance(got: Any, want: Any) -> float:
    """归一化距离：数值按相对差，列表按逐项相对差，其余按不等。"""
    if isinstance(want, (int, float)) and not isinstance(want, bool):
        denom = max(abs(float(want)), 1e-6)
        return abs(float(got) - float(want)) / denom
    if isinstance(want, list) and isinstance(got, list) and len(want) == len(got):
        return sum(_distance(g, w) for g, w in zip(got, want)) / max(1, len(want))
    return 0.0 if got == want else 1.0


def table(cfg: AppConfig) -> list[dict[str, Any]]:
    """当前值 + 三档值 + 是否已经是最松。页面直接渲染这一份。"""
    rows: list[dict[str, Any]] = []
    for lim in LIMITS:
        now = _get_path(cfg, lim.id)
        rows.append({
            "id": lim.id,
            "label": lim.label,
            "kind": lim.kind,
            "kind_title": KINDS.get(lim.kind, lim.kind),
            "why": lim.why,
            "now": now,
            "strict": lim.values[0],
            "normal": lim.values[1],
            "unleashed": lim.values[2],
            "relaxed": now == lim.values[2],
            "tightened": now == lim.values[0] or _distance(now, lim.values[0]) < 1e-9,
        })
    return rows


# ---------------- 对账：这条限制到底拦了你多少次 ----------------

#: 每个计数器的解释（数字本身没有意义，配一句话才看得懂）
_COUNTERS: tuple[tuple[str, str, str], ...] = (
    ("suppressed", "被模式闸门拦下", "观察/暂停模式下「想做但没做」的动作数。非 0 说明当时不在协作模式。"),
    ("dropped", "被限速丢弃", "超出 max_actions_per_sec 或急停后丢弃的命令。非 0 就是这条限速在卡你。"),
    ("blind_tap_violations", "铁律拦截", "尝试把拾取键当移动键的次数。必须恒为 0 —— 非 0 是 bug，不是限制。"),
    ("supervisor_blocks", "监督层拦下", "shadow 档下应恒为 0；enforce 下这才是它真拦的数量。"),
    ("targets_ignored", "懒得捡（主动放弃）", "被 mood.skip_prob 随机无视的候选数。这是纯粹白丢的产出。"),
    ("human_breaks", "疲劳停手", "累到阈值后停手的次数。每次都要休息 break_seconds。"),
    ("drifts", "走神", "非目标导向的噪音操作次数。只影响时间，不影响产出。"),
    ("walk_no_progress", "走位无效", "点了地面但距离没缩短的轮数。多 = 走位预算或判定阈值不合适。"),
    ("abandons", "放弃目标", "走位超时/按不下去/pin 不住的次数。多 = 目标太难够到。"),
    ("recoveries", "进入恢复", "判定卡住、开始自救的次数。多 = 环境在变（刷怪/移动的怪）。"),
    ("stale_frames", "旧帧不做决定", "帧年龄超过 decision.max_frame_age_ms 被跳过。"
                                  "非 0 = 机器/链路跟不上采集，画面是几秒前的世界。"),
    ("duplicate_frames", "同一帧重复决策", "同一帧被喂了多次。正常应为 0 —— 非 0 说明上游在重复投递。"),
)


def ledger(stats: dict[str, Any] | None = None,
           executor: Any | None = None) -> dict[str, Any]:
    """把"被拦了多少"翻译成人话。返回 {rows, summary, hint}。

    数据源刻意只用**已经在跑的那些计数器**，不新开统计口径 ——
    新口径意味着页面上的数字和日志里的数字对不上，然后又多一个需要解释的东西。
    """
    st = dict(stats or {})
    if executor is not None:
        st.setdefault("suppressed", getattr(executor, "suppressed", 0))
        st.setdefault("dropped", getattr(executor, "dropped", 0))

    rows = []
    for key, label, why in _COUNTERS:
        if key not in st:
            continue
        rows.append({"key": key, "label": label, "why": why, "count": int(st.get(key) or 0)})

    sent = getattr(executor, "sent", None) if executor is not None else st.get("sent")
    lost = sum(r["count"] for r in rows
               if r["key"] in ("suppressed", "dropped", "targets_ignored", "walk_no_progress"))
    total = (int(sent) if sent is not None else 0) + lost
    wasted = (lost / total) if total else 0.0

    if total and wasted >= 0.25:
        hint = (f"有 {wasted:.0%} 的动作配额没变成产出（被拦/丢弃/白丢）。"
                "这不是「跑得慢」，是「跑了没用」，先看是哪一条。")
    elif lost:
        hint = "有少量动作被拦，属正常摩擦。"
    else:
        hint = "没有动作被拦 —— 现在限制你的只有节奏本身（tick_hz / 轨迹时长）。"

    return {
        "rows": rows,
        "sent": int(sent) if sent is not None else None,
        "lost": lost,
        "wasted_ratio": round(wasted, 4),
        "summary": hint,
    }


def modes(cfg: AppConfig) -> list[dict[str, Any]]:
    """四道"它敢不敢自己决定"的开关。

    这些**不进三档 patch**，因为它们改的不是快慢，是"谁来做决定"：
      - arbiter / supervisor 的 enforce 需要先跑影子期看它想拦什么（见 config.yaml 注释）；
      - humanize 关掉会明显变快，但行为退化成确定性脚本 —— 这是策略选择，不是性能调优。
    所以这里只如实展示当前状态，让用户知道"它现在为什么不自己动"，而不是替他按下去。
    """
    d = cfg.decision
    return [
        {"key": "arbiter", "label": "三级裁决", "value": str(d.arbiter.mode),
         "note": "shadow = 只记账、不执行它的选择；要让模型真改动作才切 enforce"},
        {"key": "supervisor", "label": "监督层", "value": str(d.supervisor.mode),
         "note": "shadow = 只观察、不拦；enforce 才有踩刹车的权力"},
        {"key": "humanize", "label": "拟人化", "value": "on" if d.humanize.enabled else "off",
         "note": "关掉会更快，但行为完全确定、可复现（也更好认）"},
        {"key": "autonomy", "label": "自主性", "value": "on" if d.autonomy.enabled else "off",
         "note": "关掉之后卡住不会自救，只会停在那里"},
    ]


def report(cfg: AppConfig, stats: dict[str, Any] | None = None,
           executor: Any | None = None) -> dict[str, Any]:
    """一次拿全：当前档位、所有限制、安全铁律、自治开关、对账。"""
    tier = current_tier(cfg)
    return {
        "tier": tier,
        "tier_title": TIER_TITLES[tier]["title"],
        "tiers": [{"id": t, **TIER_TITLES[t]} for t in TIERS],
        "kinds": KINDS,
        "limits": table(cfg),
        "modes": modes(cfg),
        "safety": [{"label": s.label, "what": s.what, "where": s.where} for s in SAFETY],
        "ledger": ledger(stats, executor),
    }
