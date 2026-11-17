"""配置模块 —— YAML 主配置 + 环境变量覆盖。

安全默认值：
  - 服务端必须配置 token（未配置时自动生成并持久化到 .token 文件）。
  - 仪表盘默认只读；写操作（start/stop/panic）需要 token。
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


# 拟人化默认开关：单元测试/排障要可复现，可用 AUTOGAME_HUMANIZE=0 全局关掉。
# 部署时保持默认开启；显式写 config.yaml 的 humanize.enabled 优先级更高。
_HUMANIZE_ON = os.environ.get("AUTOGAME_HUMANIZE", "1").strip().lower() not in (
    "0", "false", "no", "off")


class CaptureConfig(BaseModel):
    """采集代理（游戏 PC 端）配置。"""

    fps: float = Field(2.0, ge=0.5, le=30, description="采集帧率，默认 2fps（决策 1Hz 已足够）")
    jpeg_quality: int = Field(80, ge=30, le=95)
    region: Optional[list[int]] = Field(
        None, description="采集区域 [left, top, width, height]，None=全屏"
    )
    backend: Literal["auto", "dxcam", "mss"] = "auto"
    monitor: int = Field(1, ge=1, description="显示器编号（dxcam 用）")
    reconnect_s: float = Field(2.0, ge=0.5)
    # 异动触发（对话定稿："如果发现异动，压缩成 JPEG 传输"）：
    # 画面近重复时跳帧不上传，每 motion_force_s 至少传一帧保活
    motion_trigger: bool = Field(False, description="启用画面异动检测（静止场景自动降频）")
    motion_hash_thresh: int = Field(4, ge=1, le=32, description="感知哈希汉明距离≤该值视为无异动")
    motion_force_s: float = Field(2.0, ge=0.5, description="无异动时强制上传间隔（秒）")


class VisionConfig(BaseModel):
    """视觉检测配置（服务端）。"""

    # 角色固定在屏幕中心（ARPG 通用设定，来自对话）
    character_at_center: bool = True

    # 光柱颜色 HSV 阈值（OpenCV H: 0-179, S/V: 0-255）
    hsv: dict[str, list[list[int]]] = Field(
        default_factory=lambda: {
            # 金色光柱（高品质掉落）
            "gold": [[18, 120, 180], [35, 255, 255]],
            # 粉色/紫色光柱（稀有掉落）
            "pink": [[140, 90, 160], [172, 255, 255]],
        }
    )
    pillar_min_area: int = Field(40, ge=5, description="光柱最小轮廓面积(px)")
    pillar_min_width: int = Field(2, ge=1, description="光柱最小宽度px（真实 TLI 光柱 2-6px 细条）")
    pillar_max_width: int = Field(70, ge=10, description="光柱最大宽度px")
    pillar_min_aspect: float = Field(
        2.5, ge=0.5,
        description="高/宽最小比例。光柱是竖直细条：暗黑4/火炬之光实测 3~8；"
                    "低于 2.5 的是团光/地图节点/技能光斑，不是柱（真实录像验证："
                    "火炬之光『异界』关卡选择页的星图节点高宽比 1.4~2.0，曾全被当光柱）",
    )
    pillar_max_aspect: float = Field(15.0, description="高/宽最大比例")

    # A框（近距交互提示）检测
    aframe: "AFrameConfig" = Field(default_factory=lambda: AFrameConfig())

    max_detect_width: int = Field(1280, ge=320, description="检测前等比缩放的最大宽度（0=不缩放）")

    # OCR 引擎：null/None=禁用（纯颜色模式）；tesseract/paddle=启用白名单识别
    # （PaddleOCR 中文效果更好：pip install paddleocr paddlepaddle）
    ocr: Literal["null", "tesseract", "paddle"] | None = None
    whitelist: list[str] = Field(
        default_factory=list, description="OCR 白名单物品名；空=所有光柱都视为目标"
    )
    whitelist_fuzzy: int = Field(78, ge=50, le=100, description="OCR 模糊匹配阈值(0-100)")
    # 稀有度白名单（**不依赖 OCR**）：按光柱颜色簇过滤，如 ["gold"] 只捡金色。
    # 对话定稿的 C3 铁律是"只捡白名单装备"；不装 OCR 时用颜色近似稀有度，
    # 让铁律在默认部署下也能真正生效（否则所有光柱都算目标）。
    whitelist_colors: list[str] = Field(
        default_factory=list,
        description="只捡这些颜色的光柱（对应 vision.hsv 的键，如 gold/pink）；空=全部",
    )

    rechallenge: "RechallengeConfig" = Field(default_factory=lambda: RechallengeConfig())

    # 右侧 UI 列表掉落方案（对话定稿的高级方案：部分掉落显示为屏幕一侧的
    # 物品列表而非地面光柱 —— 识别列表条目并点击条目让角色走过去拾取）
    item_list: "ItemListConfig" = Field(default_factory=lambda: ItemListConfig())

    # 地面掉落标签检测（TLI 类游戏：掉落物=黄色标签条，点击标签下方走过去拾取）
    ground_labels: "GroundLabelConfig" = Field(default_factory=lambda: GroundLabelConfig())

    minimap: "MinimapConfig" = Field(default_factory=lambda: MinimapConfig())

    # HUD 固定区域排除（小地图/任务/聊天/技能栏/ Buff 条——掉落物不可能出现在这里）
    # 每项 [x, y, w, h]，原分辨率坐标；默认按《火炬之光：无限》1280x720 布局
    hud_exclusions: list[list[int]] = Field(
        default_factory=lambda: [
            [0, 55, 245, 430],     # 小地图 + 任务追踪
            [0, 540, 265, 180],    # 世界聊天
            [295, 585, 710, 135],  # 技能栏（含拾取键提示）
            [1035, 0, 245, 70],    # Buff/功能栏
            [0, 0, 165, 55],       # FPS/网络
            [380, 0, 520, 70],     # 顶部 Boss 血条/目标条（批量验证发现的误报源）
        ]
    )

    # 场景门：技能栏模板匹配（战斗 HUD 存在才允许出命令，防 UI 界面误操作）
    scene_gate: "SceneGateConfig" = Field(default_factory=lambda: SceneGateConfig())

    # 光柱白芯校验：掉落光柱有白热芯（近白像素），怪物特效是纯饱和色
    require_white_core: bool = Field(True, description="光柱必须含白热芯（抗 Boss/技能特效误报）")
    white_core_min_ratio: float = Field(0.04, ge=0.0, le=0.5)
    white_core_s_max: int = Field(70, description="白芯饱和度上限（光柱白热芯 S=59-70；Boss触手 S≥144 被拒）")
    white_core_v_min: int = Field(225, description="白芯亮度下限")

    # 光柱时域确认：跨帧持续出现的候选才算真光柱（抗单帧特效闪烁）
    beam_persistence_frames: int = Field(2, ge=1, le=5, description="连续 N 帧可见才确认")
    beam_track_tol_px: int = Field(260, ge=40, le=800, description="跨帧跟踪位置容差（2Hz×420px/s 走位）")
    # 稳定光柱：连续 ≥N 帧可见标记 stable（掉落物存续分钟级，技能特效秒级）
    # 0=不标记；FSM 优先以 stable 光柱为目标，无 stable 时回退全部候选
    beam_stable_frames: int = Field(3, ge=0, le=20)

    # 加载/过场画面判定：暗像素占比超过该值视为加载/黑屏过场（FSM 保持不动）
    loading_dark_ratio: float = Field(0.85, ge=0.5, le=1.0, description="画面暗像素占比阈值")
    loading_dark_v: int = Field(15, ge=1, le=80, description="暗像素亮度上限（加载画面接近纯黑）")


class SceneGateConfig(BaseModel):
    """战斗 HUD 门：反例优先 → 正向锚点 → 兜底拦下。

    判定顺序（顺序即优先级，negative_templates 命中直接判非战斗）：
      1. 反例模板（整屏 UI：关卡选择/背包/结算/装备面板）任一命中 → in_game=False
      2. 技能栏模板（skillbar*.png）任一命中 → in_game=True
      3. 红色生命球结构锚点 → in_game=True
      4. 一个都没命中 → fail_closed=True 时判 False（未标定的游戏必须拦住）

    threshold 0.6 对应半分辨率匹配（实测：战斗帧 ≥0.668，UI 帧 ≤0.26，间隔充足）。
    """

    enabled: bool = True
    template: str = Field("skillbar.png", description="技能栏模板（templates/ 目录）")
    threshold: float = Field(0.6, ge=0.4, le=0.99)
    roi: Optional[list[int]] = Field(
        None, description="匹配区域 [x,y,w,h]（原分辨率）；None=底部 35% 全宽"
    )
    # 反例（负样本）：整屏 UI 的静态裁块。空 → 自动加载 templates/neg_*.png
    negative_templates: list[str] = Field(
        default_factory=list,
        description="反例模板路径（templates/ 目录名或绝对路径）；命中即判非战斗",
    )
    negative_threshold: float = Field(0.80, ge=0.4, le=0.99)
    fail_closed: bool = Field(
        True, description="无任何正向锚点命中时是否拦下（未标定游戏必须拦，防盲操作）"
    )


class ItemListConfig(BaseModel):
    """右侧 UI 掉落列表配置。

    条目判定：ROI 内横向亮色/带框条行（行宽 > ROI 40%、行高 12-40px），
    每行中心即点击坐标（点击列表条目角色会走过去拾取）。
    OCR 白名单启用时，仅命中白名单的条目返回。
    """

    enabled: bool = False
    # 列表 ROI [x, y, w, h]（原分辨率坐标）；None=默认屏幕右侧 1/4 竖条
    roi: Optional[list[int]] = None
    row_min_height: int = Field(10, ge=4)
    row_max_height: int = Field(44, ge=6)
    row_min_width_ratio: float = Field(0.4, ge=0.1, le=1.0, description="行宽占 ROI 宽度最小比例")
    max_rows: int = Field(6, ge=1, le=12)
    # 条目亮色 HSV（物品名/框的高亮色，色相无关按亮+饱和判定）
    bright_s_min: int = Field(110)
    bright_v_min: int = Field(170)


class UiTaskConfig(BaseModel):
    """UI 场景任务配置（交易行/装备/天赋/技能 —— 配方引擎驱动）。"""

    grid_roi: list[int] = Field([150, 80, 900, 340], description="槽位网格 ROI")
    cell_min: int = Field(24)
    cell_max: int = Field(140)
    row_roi: list[int] = Field([150, 380, 900, 180], description="文本/价格行 ROI")
    step_timeout_ms: int = Field(6000, ge=1000, description="单步超时")
    templates_dir: str | None = Field(None, description="UI 模板目录（缺省用 templates/）")


class GroundLabelConfig(BaseModel):
    """地面掉落标签检测（黄色横条=掉落物，战斗区域内、排除 HUD 后）。"""

    enabled: bool = True
    hue_lo: int = Field(12, description="标签黄色 hue 下限")
    hue_hi: int = Field(45, description="标签黄色 hue 上限")
    s_min: int = Field(90)
    v_min: int = Field(150)
    row_min_h: int = Field(10, ge=4)
    row_max_h: int = Field(30, ge=6)
    row_min_w: int = Field(50, ge=10)
    max_labels: int = Field(8, ge=1, le=16)
    click_below_px: int = Field(24, ge=0, description="点击点=标签中心下方偏移（地面）")


class MinimapConfig(BaseModel):
    """小地图目标点检测（对话定稿：根据左上角地形判断往哪里走）。

    小地图 ROI 内检测目标标记（高亮色点），角色假设在小地图中心（ARPG 通用），
    输出标记相对中心的方向 → 决策器换算为方向键走位。
    """

    enabled: bool = False
    # 小地图 ROI [x, y, w, h]（原分辨率坐标）；None=默认左上角 320x320
    roi: Optional[list[int]] = None
    # 目标标记 HSV（例如亮青色任务点）
    marker_hsv_lo: list[int] = Field([80, 150, 200])
    marker_hsv_hi: list[int] = Field([100, 255, 255])
    marker_min_area: int = Field(9, ge=1)
    marker_max_area: int = Field(900, ge=2)
    arrived_ratio: float = Field(
        0.12, ge=0.01, le=0.9, description="标记距小地图中心 < 该比例(×半径) 视为已到达"
    )


class RechallengeConfig(BaseModel):
    """『再次挑战』按钮检测配置（对话定稿：是否通过再次挑战 → 刷图循环）。

    优先模板匹配；未提供模板时用 颜色+横条形 状启发式（按钮高亮色宽横条）。
    """

    enabled: bool = True
    template: Optional[str] = Field(None, description="按钮模板图路径；None=启发式")
    template_threshold: float = Field(0.75, ge=0.5, le=0.99)
    template_scales: list[float] = Field([0.9, 1.0, 1.1])
    # 按钮 ROI（原分辨率坐标 [x, y, w, h]）；None=屏幕中部居中区域（避开底部技能栏）
    roi: Optional[list[int]] = None
    # 启发式：按钮高亮色（黄橙色系）HSV
    color_lo: list[int] = Field([15, 140, 200])
    color_hi: list[int] = Field([35, 255, 255])
    min_area: int = Field(2000, ge=200)
    max_area: int = Field(60000, ge=500)
    min_aspect: float = Field(2.5, description="宽/高最小比例（横条按钮）")
    max_aspect: float = Field(12.0)


class AFrameConfig(BaseModel):
    """A框（按A拾取提示）检测配置。

    两种模式（可同时启用，命中其一即可）：
      template: 模板匹配（需要提供模板图 templates/aframe.png）
      color:    颜色+形状启发式（框线高亮、内部镂空 —— 区别于实心光柱）
    """

    template: Optional[str] = Field(None, description="模板图路径；None=禁用模板匹配")
    template_threshold: float = Field(0.78, ge=0.5, le=0.99)
    template_scales: list[float] = Field(
        [0.9, 1.1],
        description="多尺度模板匹配（游戏 UI 缩放后仍可命中）；"
                    "实测 0.85/1.0/1.2 三个尺度已覆盖，过多尺度只增加耗时",
    )
    # 色相无关模式（推荐）：A框按"亮+饱和+镂空+方形"判定，颜色随稀有度变化也能检出
    hue_agnostic: bool = Field(True)
    bright_s_min: int = Field(100, description="色相无关模式下 S 下限")
    bright_v_min: int = Field(200, description="色相无关模式下 V 下限")
    color_lo: list[int] = Field([15, 100, 200], description="A框高亮色 HSV 下限（hue_agnostic=False 时用）")
    color_hi: list[int] = Field([45, 255, 255], description="A框高亮色 HSV 上限（hue_agnostic=False 时用）")
    min_area: int = Field(60, ge=10)
    max_area: int = Field(6000, ge=100)
    min_aspect: float = Field(0.7, description="宽/高最小比例（方框）")
    max_aspect: float = Field(2.2, description="宽/高最大比例")
    min_size: int = Field(12, ge=4, description="框宽/高最小px")
    max_size: int = Field(90, ge=12, description="框宽/高最大px")
    hole_min_ratio: float = Field(
        0.10, ge=0.0, le=0.9,
        description="镂空率下限：A框是空心轮廓(镂空率~0.5+)，实心光柱~0 → 排除误报",
    )
    scale: float = Field(1.0, ge=0.3, le=2.0, description="检测缩放（画面缩放后）")

    # ---- 结构校验（真实帧精度关键）----
    # A框的真实形态：**暗底小方框 + 内部一个亮字形（A）**。
    # 伤害数字/技能图标/血条高亮虽然是"亮+镂空"，但内部是亮色填充，必须被拒。
    # 实测：100 张真实帧上，未加校验时 11 个候选全是伤害数字/图标误报；
    # 加校验后真实 A框保留、误报全部拒绝。
    verify_dark_core: bool = Field(True, description="校验框内必须是暗底（拒绝实心亮色块）")
    dark_core_v_max: int = Field(110, ge=20, le=200, description="框内平均亮度上限（暗底判据）")
    glyph_min_ratio: float = Field(
        0.03, ge=0.0, le=0.5, description="框内亮像素占比下限（字形存在判据）"
    )
    glyph_max_ratio: float = Field(
        0.60, ge=0.1, le=1.0, description="框内亮像素占比上限（过亮=实心块/数字，非空心提示框）"
    )
    glyph_white_only: bool = Field(
        False,
        description="字形必须是**白色**（真实游戏 A 提示是白字暗底；实测可再排除红色 Buff 图标）。"
                    "模拟器用彩色字形，故默认关；现场部署建议开",
    )
    glyph_s_max: int = Field(70, ge=10, le=180, description="字形白色的饱和度上限（S<=该值视为白）")


class VLMConfig(BaseModel):
    """视觉模型（多模态）顾问配置 —— 卡壳时"看一眼画面"。

    与文本 LLM 的分工：
      - LLM（llm_advisor）：看**摘要**，判断"该不该换策略"，成本低、频率可高；
      - VLM（本类）：看**截图**，判断"我现在在哪个界面 / 目标在哪"，成本高、必须省着用。

    安全与预算（搬砖机器人会跑几小时，模型调用不能失控）：
      - 动作白名单 + 坐标边界钳制，模型不能让我们按拾取键或点屏幕外；
      - 每分钟调用上限 + 最小间隔 + 连续失败熔断（默认 3 次失败停 2 分钟）；
      - 后台线程执行，绝不阻塞决策循环；超时/失败静默降级为启发式恢复。
    """

    enabled: bool = False
    provider: Literal["ollama", "openai"] = Field(
        "ollama", description="ollama=本机/内网多模态；openai=任意 OpenAI 兼容 /v1/chat/completions"
    )
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5vl:7b"
    api_key: Optional[str] = Field(None, description="openai 兼容服务的 key（留空则读环境变量）")
    api_key_env: str = "AUTOGAME_VLM_KEY"
    timeout_s: float = Field(20.0, ge=2.0, le=120.0)
    max_calls_per_min: int = Field(6, ge=1, le=60, description="每分钟最多调用次数（预算）")
    min_interval_s: float = Field(8.0, ge=0.5, description="两次调用最小间隔")
    jpeg_quality: int = Field(70, ge=30, le=95, description="上传前重新编码质量（省带宽/显存）")
    max_width: int = Field(960, ge=320, le=1920, description="上传前等比缩小到该宽度")
    min_confidence: float = Field(0.45, ge=0.0, le=1.0, description="低于该置信度的建议只记录不执行")
    failure_threshold: int = Field(3, ge=1, description="连续失败 N 次触发熔断")
    cooldown_after_failure_s: float = Field(120.0, ge=5.0, description="熔断后冷却时长")
    allowed_actions: list[str] = Field(
        ["walk_to", "key", "esc", "wait", "explore", "reset_stage"],
        description="允许 VLM 触发的动作（其余仅记录）",
    )
    # 允许 VLM 建议的按键（白名单，绝不含拾取键 —— 拾取必须由 A框检测裁决）
    allowed_keys: list[str] = Field(["w", "a", "s", "d", "space", "esc", "enter", "tab"])


class AutonomyConfig(BaseModel):
    """自主性配置 —— 进度看门狗 + 分级恢复阶梯 + 空闲策略。

    设计目标（用户要求："作为搬砖机器人，要有更高智能，绝对不能卡壳"）：
      1. 不卡死：任何状态都有上限时长，超时即升级恢复动作；
      2. 不空转：没有目标时自己去找（探索/导航/清怪），而不是站着等；
      3. 不瞎按：恢复动作全部在安全白名单内，**永远不按拾取键**（铁律不变）；
      4. 不断线：恢复失败就换招，循环重试并降速告警，绝不"停在那里"。
    """

    enabled: bool = True

    # ---- 看门狗窗口（秒，均基于帧时间戳）----
    motion_delta_thresh: float = Field(2.0, ge=0.1, description="帧变化率(%)低于该值视为画面没动")
    frozen_window_s: float = Field(12.0, ge=1.0, description="画面完全不动多久算卡住")
    idle_window_s: float = Field(20.0, ge=1.0, description="没有任何可交互目标多久算空闲")
    stall_window_s: float = Field(90.0, ge=3.0, description="无任何有效进展多久算停滞")
    gate_window_s: float = Field(25.0, ge=1.0, description="长时间不在战斗界面(弹窗/菜单)多久后按 ESC")

    # ---- 恢复阶梯 ----
    level_interval_s: float = Field(8.0, ge=0.2, description="恢复等级升级间隔")
    step_interval_s: float = Field(1.2, ge=0.05, description="同一步骤内两次动作最小间隔")
    max_cycles: int = Field(0, ge=0, description="完整恢复循环上限；0=无限重试（绝不放弃）")
    cycle_backoff_s: float = Field(15.0, ge=0.0, description="每轮循环后的额外等待（降速避免乱按）")
    pause_after_cycles: int = Field(0, ge=0, description="连续 N 轮仍失败则暂停并告警；0=从不暂停")
    dodge_key: str = Field("space", description="脱困/位移键（空格=闪避类）；空=不用")
    dodge_taps: int = Field(2, ge=1, le=5)
    escape_key: str = Field("esc", description="关闭弹窗/菜单键")

    # ---- 空闲自主策略（提高自主性，不再干等）----
    auto_explore: bool = Field(True, description="空闲无目标时自动探索（临时启用 explore）")
    auto_attack: bool = Field(True, description="空闲且检测到怪物时自动清怪")
    explore_step_ms: int = Field(600, ge=150, le=1500, description="自主探索单步时长")
    explore_dirs: list[str] = Field(["w", "d", "s", "a"], description="自主探索方向序列")
    # 探路恢复：点击角色四周这些方向（像素偏移，屏幕坐标系）
    probe_offsets: list[list[int]] = Field(
        [[0, -140], [140, 0], [0, 140], [-140, 0]],
        description="脱困探路点击偏移（相对角色），逐个尝试找出可行方向",
    )

    # ---- 程序自身看门狗（帧流断了也不能傻等）----
    frame_stall_s: float = Field(10.0, ge=0.5, description="运行中超过该时长无新帧 → 告警")


class MousePathConfig(BaseModel):
    """鼠标轨迹配置（WindMouse：重力吸附 + 风力扰动 + 端点减速）。

    为什么不用两点一线：直线匀速移动是自动化最容易被统计识别的特征之一。
    人类鼠标轨迹有弧度、有加减速、落点差 1-3px、偶尔冲过头再拉回。

    归属执行器（输出层）：这一层只决定"手怎么动"，不决定"往哪动"。
    """

    enabled: bool = Field(True, description="关闭后退化为单点移动（与旧版行为一致）")
    gravity: float = Field(9.0, ge=1.0, le=30.0, description="向目标的重力吸附强度")
    wind: float = Field(3.0, ge=0.0, le=20.0, description="风力扰动幅度（越大轨迹越飘）")
    max_step_px: float = Field(10.0, ge=2.0, le=60.0, description="单步最大位移（越小轨迹越细）")
    damping: float = Field(0.86, ge=0.3, le=1.0, description="速度阻尼，<1 防抖振")
    speed_px_s: float = Field(900.0, ge=100.0, le=5000.0, description="名义移动速度")
    min_duration_s: float = Field(0.06, ge=0.0, le=0.5, description="整段轨迹最短时长")
    max_duration_s: float = Field(0.45, ge=0.05, le=2.0, description="整段轨迹最长时长（防阻塞）")
    max_points: int = Field(40, ge=2, le=200, description="轨迹点上限（USB HID/串口带宽预算）")
    endpoint_jitter_px: int = Field(2, ge=0, le=12, description="落点偏差像素（真人不点数学中心）")
    overshoot_prob: float = Field(0.12, ge=0.0, le=0.5, description="冲过头再拉回的概率")
    overshoot_max_px: float = Field(10.0, ge=0.0, le=40.0, description="冲过头的最大像素")
    micro_pause_prob: float = Field(0.25, ge=0.0, le=1.0, description="中途微小停顿概率")
    micro_pause_ms: list[int] = Field([20, 70], description="停顿区间 [lo, hi] 毫秒")
    max_micro_pauses: int = Field(2, ge=0, le=5, description="单次轨迹最多停顿次数")
    # ---- 点击手势（一次点击不是一个"事件"，是 悬停→按下→保持→抬起 四个阶段）----
    hover_ms: list[int] = Field([30, 110], description="光标到位后悬停多久再按下（先看清再点）")
    press_ms: list[int] = Field([55, 135], description="左键按下保持时长（真人不会 45ms 就松手）")


class WalkConfig(BaseModel):
    """走位等待 —— 把"点地面 → 人物走过去 → A框出现 → 按拾取键"拆成多步。

    为什么必须拆：点地面只是**发起**移动，人物实际走过去是游戏侧的耗时步骤
    （1-3s，且 A框要到一定距离内才出现）。点完立刻按固定节拍重复点，有两个问题：
      1. 机械特征明显：每秒一次、同一落点，统计上一眼假；
      2. 语义错误：真人点一次就盯着人走，到了看到提示才按键，不会每秒重发同一指令。

    这里用"画面是否在动 + 距离是否真的缩短"两个判据，把走位拆成
    发起 → 等待 → （到位 | 走完预算 | 卡住）→ 决策 的显式链条。
    """

    enabled: bool = Field(True, description="关闭后退回『每 tick 重复点』的旧行为")
    budget_s: float = Field(3.0, ge=0.5, le=15.0, description="单次点击后最多等多久（超时=没走到）")
    motion_delta_thresh: float = Field(2.5, ge=0.1, description="帧变化率(%)高于此值视为人物还在走")
    settle_frames: int = Field(2, ge=1, le=6, description="连续 N 帧不动视为人物停下")
    settle_grace_s: float = Field(0.5, ge=0.0, le=3.0, description="停下后再给 A框多久出现（提示有延迟）")
    min_progress_px: float = Field(12.0, ge=0.0, description="这一轮至少靠近这么多像素才算有效走位")
    key_extra_s: float = Field(0.25, ge=0.0, le=2.0, description="keys 模式：按键时长之外的额外等待")


class FatigueConfig(BaseModel):
    """疲劳度配置 —— 让延时**均值**和**方差**都随时间漂移。

    固定区间随机的方差有上界，样本一多就收敛；真人不会。
    这里用"慢变量疲劳 + 快变量唤醒度"两层状态来破这个盒子。
    """

    enabled: bool = True
    per_action: float = Field(0.003, ge=0.0, le=0.05, description="每次动作累积的疲劳（0.003≈280次到满）")
    rest_decay_per_s: float = Field(0.02, ge=0.0, le=0.5, description="空闲时每秒衰减的疲劳")
    idle_after_s: float = Field(8.0, ge=1.0, description="连续多久没动作算'歇着'，开始衰减")
    break_threshold: float = Field(0.85, ge=0.1, le=1.0, description="疲劳到该值就停手休息")
    break_seconds: list[float] = Field([20.0, 90.0], description="单次休息时长区间 [lo, hi] 秒")
    break_recovery: float = Field(0.55, ge=0.0, le=1.0, description="一次休息恢复的疲劳比例")
    latency_boost: float = Field(0.35, ge=0.0, le=3.0, description="满疲劳时延时放大比例")
    jitter_boost: float = Field(0.8, ge=0.0, le=3.0, description="满疲劳时抖动比例放大倍数")
    error_boost: float = Field(0.04, ge=0.0, le=0.3, description="满疲劳时额外失误概率")
    arousal_sigma: float = Field(0.05, ge=0.0, le=0.5, description="瞬时唤醒度随机游走强度")
    arousal_decay: float = Field(0.9, ge=0.5, le=0.999, description="唤醒度向 0 回归的速度（每秒）")


class DriftConfig(BaseModel):
    """走神配置 —— 非目标导向的"噪音"操作。

    真人的行为噪音不会被任务逻辑自然生成，恰恰是最难伪造的部分。
    安全默认：只发 mouse_move（不点击、不按键）。
    """

    enabled: bool = True
    mean_interval_s: float = Field(45.0, ge=5.0, description="平均多久走神一次（指数分布，非固定周期）")
    cooldown_s: float = Field(10.0, ge=1.0, description="两次走神最小间隔")
    startup_grace_s: float = Field(60.0, ge=0.0, description="启动后多久才开始可能走神（真人不会一上线就发呆）")
    moves: list[int] = Field([1, 2], description="每次走神发几次鼠标移动 [min, max]")
    margin_px: int = Field(40, ge=0, description="漂移点距画面边缘的内边距")
    noise_keys: list[str] = Field([], description="走神时可选按的键（默认空：只动鼠标不按键）")
    noise_key_hold_ms: int = Field(120, ge=20, le=800)
    hesitate_prob: float = Field(0.35, ge=0.0, le=1.0, description="进关卡/点按钮前'愣一下'的概率")
    hesitate_s: list[float] = Field([0.3, 1.6], description="犹豫时长区间 [lo, hi] 秒")


class SocialConfig(BaseModel):
    """社交噪音配置（**默认关闭**，请先读完风险再决定）。

    ⚠️ 为什么默认关闭：
      1. 打字前必须先聚焦聊天框；焦点没对时，每个字符都会变成游戏热键。
         台词里只要有一个拾取键/移动键字符，就等于"盲按拾取键"，直接违反铁律。
         （代码里有硬闸门：含禁用字符的台词会被拒绝发送，不"尽力发送"。）
      2. 污染的是账号社交记录，出问题比少捡两件装备严重得多。
      3. 收益有限：一句无意义的话不改变"零社交网络"的结构性特征。

    仍要开的话：台词只用 ASCII（无中文输入法），且只含数字和标点更安全。
    """

    enabled: bool = False
    phrases: list[str] = Field([], description="候选台词（ASCII，不含移动/拾取键字符）")
    chat_open_key: str = Field("enter", description="打开聊天输入框的键")
    chat_send_key: str = Field("enter", description="发送消息的键")
    min_interval_s: float = Field(600.0, ge=30.0, description="两条消息最小间隔（默认 10 分钟）")
    forbidden_chars: str = Field("wasd", description="禁用字符（拾取键会自动加入）")
    char_hold_ms: int = Field(60, ge=20, le=300, description="单个字符按压时长")
    max_phrase_len: int = Field(20, ge=1, le=80)


class MoodConfig(BaseModel):
    """情绪/偏好模型 —— 让"选哪个目标"也有不确定性。

    为什么这是最硬的破绽：原来的 `_pick_target` 是 `stables[0]` —— **永远选第一个**
    稳定光柱。这个确定性比任何时序抖动都致命：真人不会每次都用同一条规则挑目标。

    三个慢变量（0..1，随机游走 + 向基线回归 + 疲劳耦合）：
      patience 耐心 → 愿意走多远（累的时候只捡近的）
      greed    贪心 → 多看重稀有度（精神好时愿意绕远捡贵的）
      novelty  新鲜感 → 多讨厌重复（刚捡过同色/同位置会腻）

    选择用 softmax 而不是 argmax：偶尔挑个"不是最优"的目标，本身就是人味。
    """

    enabled: bool = True
    patience: float = Field(0.6, ge=0.0, le=1.0, description="耐心基线：高=愿意走远")
    greed: float = Field(0.5, ge=0.0, le=1.0, description="贪心基线：高=偏好高价值")
    novelty: float = Field(0.5, ge=0.0, le=1.0, description="新鲜感基线：高=讨厌重复")
    drift_sigma: float = Field(0.04, ge=0.0, le=0.3, description="情绪随机游走强度")
    reversion: float = Field(0.15, ge=0.0, le=1.0, description="情绪向基线回归的速度")
    fatigue_patience_drop: float = Field(0.5, ge=0.0, le=1.0, description="满疲劳时耐心下降比例")
    fatigue_greed_drop: float = Field(0.3, ge=0.0, le=1.0, description="满疲劳时贪心下降比例")
    temperature: float = Field(0.35, gt=0.0, le=3.0, description="softmax 温度：越大越随机")
    distance_scale_px: float = Field(500.0, ge=50.0, description="距离衰减尺度（像素）")
    color_value: dict[str, float] = Field(
        {"gold": 1.0, "pink": 0.7}, description="各稀有度颜色主观价值（0..1）"
    )
    skip_prob: float = Field(0.02, ge=0.0, le=0.5, description="单个候选被『懒得捡』无视的基础概率")
    skip_ignore_s: list[float] = Field([8.0, 30.0], description="无视某目标后多久重新考虑 [lo, hi] 秒")
    recent_memory: int = Field(8, ge=1, le=30, description="新鲜感记忆的最近拾取条数")
    recent_halflife_s: float = Field(120.0, ge=5.0, description="新鲜感衰减半衰期（秒）")


class HumanizeConfig(BaseModel):
    """拟人化行为总配置（决策层）。

    定位：**降低短期统计判定风险**的手段，不是安全方案。
    每加一层都在增加复杂度与调试成本，而封号风险不会归零。
    """

    enabled: bool = _HUMANIZE_ON
    reaction_ms: list[int] = Field([180, 620], description="看到A框→按下拾取键的反应时间基准区间")
    reaction_sigma: float = Field(0.35, ge=0.01, le=1.5, description="反应时间对数正态 σ（越大长尾越明显）")
    fatigue: "FatigueConfig" = Field(default_factory=lambda: FatigueConfig())
    drift: "DriftConfig" = Field(default_factory=lambda: DriftConfig())
    social: "SocialConfig" = Field(default_factory=lambda: SocialConfig())
    mood: "MoodConfig" = Field(default_factory=lambda: MoodConfig())


class DecisionConfig(BaseModel):
    """决策状态机配置。"""

    tick_hz: float = Field(1.0, ge=0.5, le=10, description="决策频率（对话定稿：1秒1次）")

    pickup_key: str = "a"
    # 拾取键复核冷却：按一次 A 后等游戏处理拾取动画再观察（防止连续按 A 捡垃圾）
    tap_recheck_ms: int = Field(500, ge=100, le=2000)

    # 走位方式：mouse_click=点地面寻路（推荐）; keys=方向键走位（纯键盘兜底）
    approach_mode: Literal["mouse_click", "keys"] = "mouse_click"

    # 点地面时，点击点 = 光柱底部中心 + 向下偏移（光柱悬空，点地面人物才会走过去）
    ground_click_offset_px: int = 28
    # 视为"足够近"的中心距离（超过则必须先走位；A框出现才是最终裁决）
    far_distance_px: int = 420
    # A框必须出现在当前目标附近才有效（对话定稿：A框出现在就近物品名字上）
    aframe_near_target_px: int = 200
    # 按拾取键前 A框至少连续出现几帧。人物还在走（画面在动）时，这一帧的 A框
    # 可能是几百毫秒前的旧信息 —— 边走边按最容易"按空"（模拟器里就是 blind_a_press）。
    aframe_confirm_frames: int = Field(2, ge=1, le=5, description="A框连续确认帧数（走位中防按空）")
    # A框长期可见却按不下去（光柱丢失检测/够不到/属于旁边物品）→ 重新走位，别死等
    aframe_retry_frames: int = Field(8, ge=2, le=60, description="A框按不下去多少帧后重新走位")

    approach_timeout_s: float = Field(8.0, ge=2.0, description="单目标走位超时→放弃+拉黑")
    stuck_delta_thresh: float = Field(1.5, ge=0.1, description="帧变化像素占比%%低于该值视为撞墙(0-100)")
    stuck_frames: int = Field(3, ge=2)
    abandon_blacklist_s: float = Field(30.0, description="放弃目标后该区域拉黑时长")

    # 键盘走位参数
    key_step_ms: int = Field(450, ge=100, le=1500, description="单次方向键按压时长")
    diagonal_keys: bool = True

    # 轻量拟人随机化（工程可靠性：避免完全等间隔的机械输入）
    jitter_ratio: float = Field(0.15, ge=0.0, le=0.5, description="时序抖动比例")

    # 主动通关行为（目标=通关一直刷关卡）
    attack_monsters: bool = Field(False, description="发现怪物 → 点击怪物（默认关：怪物检测需按实际画面校准后开启）")
    explore_enabled: bool = Field(False, description="无目标时按方向序列探索（默认关，配置开启）")
    explore_step_ms: int = Field(500, ge=100, le=1500, description="探索单步按压时长")
    explore_dirs: list[str] = Field(["w", "d", "s", "a"], description="探索方向序列")

    # 『再次挑战』刷图循环
    rechallenge_enabled: bool = Field(True, description="检测到再次挑战按钮时自动点击")
    rechallenge_cooldown_s: float = Field(3.0, ge=1.0, description="点按钮后的结算冷却")

    # 小地图导航（空闲时朝小地图目标点走）
    nav_enabled: bool = Field(False, description="空闲(SCANNING无目标)时按小地图标记走位")
    nav_step_ms: int = Field(400, ge=100, le=1500, description="导航单次方向键按压时长")

    # Ollama 决策顾问（可选）：连续放弃 N 次等异常局面时咨询局域网 LLM
    llm: "LLMConfig" = Field(default_factory=lambda: LLMConfig())

    # 自主性：进度看门狗 + 分级恢复阶梯（默认开启 —— 搬砖机器人不能卡壳）
    autonomy: "AutonomyConfig" = Field(default_factory=lambda: AutonomyConfig())
    # 视觉模型顾问（可选）：卡壳时截图问多模态模型"我现在在哪、该往哪走"
    vlm: "VLMConfig" = Field(default_factory=lambda: VLMConfig())
    # 拟人化行为（疲劳/反应时间/走神/社交噪音）—— 降短期统计风险，非安全方案
    humanize: "HumanizeConfig" = Field(default_factory=lambda: HumanizeConfig())
    # 走位等待：把"点地面→人物走过去→A框出现→按拾取键"拆成显式多步
    walk: "WalkConfig" = Field(default_factory=lambda: WalkConfig())


class LLMConfig(BaseModel):
    """Ollama 决策顾问配置（对话定稿：pc 上跑 ollama 决策模型）。

    定位：顾问而非决策者 —— 输出建议记录到事件流/仪表盘，只允许执行白名单安全动作。
    默认关闭；内网 Ollama 无需出网。
    """

    enabled: bool = False
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5:7b"
    timeout_s: float = Field(8.0, ge=1.0, le=30)
    trigger_abandons: int = Field(3, ge=1, description="连续放弃多少次后咨询")
    cooldown_s: float = Field(60.0, ge=0.5, description="两次咨询最小间隔")
    # 允许 LLM 直接触发的动作白名单（其余仅记录建议）
    # explore=换区域找目标 / reset_stage=重开一局 / wait=观察 / panic=交人工
    allowed_actions: list[str] = Field(
        ["wait", "continue", "explore", "reset_stage"])


class ExecutorConfig(BaseModel):
    """执行器配置。"""

    backend: Literal["dryrun", "rp2040", "rpi5_hidg", "pca9685"] = "dryrun"
    # rp2040
    serial_port: str = "COM5"
    serial_baud: int = 115200
    # rpi5_hidg
    hidg_keyboard: str = "/dev/hidg0"
    hidg_mouse: str = "/dev/hidg1"
    screen_w: int = 1920
    screen_h: int = 1080
    # pca9685（机械舵机按压，对话定稿的机械方案）
    pca9685_bus: int = Field(1, description="I2C 总线号")
    pca9685_addr: int = Field(0x40, description="PCA9685 I2C 地址")
    pca9685_mock: bool = Field(False, description="mock 总线（测试）")
    servo_map: dict[str, dict[str, int]] = Field(
        default_factory=lambda: {
            "a": {"ch": 0, "down": 110, "up": 60},
            "w": {"ch": 1, "down": 110, "up": 60},
            "s": {"ch": 2, "down": 110, "up": 60},
            "d": {"ch": 3, "down": 110, "up": 60},
            "mouse_left": {"ch": 8, "down": 120, "up": 70},
            "mouse_right": {"ch": 9, "down": 120, "up": 70},
        },
        description="键/鼠标按钮 → 舵机通道与按压角度",
    )

    # 安全限速
    min_action_interval_ms: int = Field(25, ge=0, description="两次输出动作最小间隔")
    max_actions_per_sec: int = Field(20, ge=1, description="动作速率上限")

    # 鼠标轨迹人类化（WindMouse）—— 输出层：只决定"手怎么动"
    mouse_path: "MousePathConfig" = Field(default_factory=lambda: MousePathConfig())


class WatchdogConfig(BaseModel):
    """视觉看门狗 —— 看不到游戏画面就自动停止。

    两类"看不到"要分开处理，否则会互相打架：
      - 完全没有新帧：相机被关/代理掉线/游戏退出 → 等 no_frame_stop_s 后停；
      - 有帧但判定不在游戏里：菜单/弹窗/相机被挪开。autonomy 会先按 ESC 自救
        （gate_window_s 默认 25s），所以这里要**等更久**才停，否则会把自救打断。
    """

    enabled: bool = True
    no_frame_stop_s: float = Field(
        30.0, ge=5.0, description="完全没有新帧多久 → 自动停止（相机被关/代理掉线）")
    no_game_stop_s: float = Field(
        45.0, ge=10.0, description="有帧但持续判定『不在游戏里』多久 → 自动停止")
    auto_stop: bool = Field(True, description="True=自动停止；False=只告警不动手")
    resume_after_s: float = Field(
        0.0, ge=0.0, description="画面恢复后多久自动重新开始；0=不自动恢复（人工确认更安全）")


class NotifyRule(BaseModel):
    """一条告警规则：某个指标跨过阈值就提醒一次。"""

    name: str = "未命名规则"
    enabled: bool = True
    metric: Literal[
        "rechallenges", "picks", "runtime_s", "abandons", "recoveries",
        "manual_needed", "no_pick_with_targets_s", "blind_taps", "dropped_commands",
    ] = "rechallenges"
    threshold: float = 0.0
    message: str = ""
    repeat: bool = Field(False, description="跨过阈值后是否每次都再提醒（默认只提醒一次）")


class NotifyConfig(BaseModel):
    """告警通知：满足条件时推 webhook / 邮件。

    为什么默认关闭：告警渠道涉及外部网络与凭据，属于部署决定，不该由框架替用户打开。
    密码只从环境变量读（`smtp_password_env`），不写进配置文件、不进运行时覆盖文件。
    """

    enabled: bool = False
    webhook_url: str = ""
    webhook_format: Literal["generic", "wecom", "dingtalk", "slack", "bark"] = "generic"
    # Bark（iOS 推送）：bark_key 填 App 里那串 key，bark_server 一般不用改
    bark_key: str = ""
    bark_server: str = "https://api.day.app"
    smtp_host: str = ""
    smtp_port: int = 465
    smtp_ssl: bool = True
    smtp_user: str = ""
    smtp_password_env: str = "AUTOGAME_SMTP_PASS"
    smtp_to: list[str] = Field(default_factory=list)
    min_interval_s: float = Field(600.0, ge=10.0, description="同一规则最短提醒间隔")
    rules: list[NotifyRule] = Field(
        default_factory=lambda: [
            NotifyRule(name="刷图 100 次", metric="rechallenges", threshold=100,
                       message="刷满 100 次，可以看一眼收益了"),
            NotifyRule(name="连续运行 2 小时", metric="runtime_s", threshold=7200,
                       message="已连续运行 2 小时，建议检查画面与背包"),
            NotifyRule(name="疑似包满", metric="no_pick_with_targets_s", threshold=180,
                       message="有目标但 3 分钟没捡到东西，可能背包满了"),
            NotifyRule(name="需要人工判断", metric="manual_needed", threshold=1,
                       message="出现模型/规则解决不掉的情况，需要人工确认"),
            NotifyRule(name="盲按违规", metric="blind_taps", threshold=1,
                       message="出现盲按拾取键（铁律违规），请立即检查"),
        ]
    )


class ServerConfig(BaseModel):
    """服务端（局域网小主机）配置。"""

    host: str = "0.0.0.0"
    port: int = 8765
    # 本地程序默认**不需要 token**（局域网自用，多一道门槛只是给自己添堵）。
    # 要暴露到不可信网络时再打开：require_token=true（token 见 AUTOGAME_TOKEN / .token）
    require_token: bool = False
    # token：优先环境变量 AUTOGAME_TOKEN；其次 config；最后自动生成持久化 .token
    token: Optional[str] = None
    # 仪表盘 MJPEG 参数
    dash_jpeg_quality: int = 70
    dash_fps: float = Field(4.0, ge=1.0, le=15)
    # 事件持久化（JSONL 审计日志，None=关闭）
    event_log: Optional[str] = "logs/events.jsonl"
    event_log_max_mb: float = Field(5.0, ge=0.1, description="单文件上限，超过滚动为 .1")
    # 历史保留
    history_size: int = Field(300, ge=10)


class AppConfig(BaseModel):
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    sim: "SimConfig" = Field(default_factory=lambda: SimConfig())
    ui_task: "UiTaskConfig" = Field(default_factory=lambda: UiTaskConfig())
    watchdog: "WatchdogConfig" = Field(default_factory=lambda: WatchdogConfig())
    notify: "NotifyConfig" = Field(default_factory=lambda: NotifyConfig())


class SimConfig(BaseModel):
    """合成场景模拟器配置（测试/联调用）。"""

    width: int = 1280
    height: int = 720
    walk_speed_px_s: float = 260.0
    pickup_radius_px: float = 130.0
    pillar_spawn_interval_s: float = 2.2
    picks_until_clear: int = Field(4, ge=0, description="拾满 N 个后地图结算出『再次挑战』；0=关闭刷图循环")


# 解决前向引用
UiTaskConfig.model_rebuild()
GroundLabelConfig.model_rebuild()
RechallengeConfig.model_rebuild()
MinimapConfig.model_rebuild()
ItemListConfig.model_rebuild()
SceneGateConfig.model_rebuild()
AFrameConfig.model_rebuild()
LLMConfig.model_rebuild()
VLMConfig.model_rebuild()
AutonomyConfig.model_rebuild()
HumanizeConfig.model_rebuild()
MoodConfig.model_rebuild()
MousePathConfig.model_rebuild()
WalkConfig.model_rebuild()
DecisionConfig.model_rebuild()
ExecutorConfig.model_rebuild()
VisionConfig.model_rebuild()
WatchdogConfig.model_rebuild()
NotifyConfig.model_rebuild()
AppConfig.model_rebuild()


def _deep_merge(base: dict, extra: dict) -> dict:
    """递归合并：extra 覆盖 base（dict 逐层合并，其余整体替换）。"""
    out = dict(base or {})
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def runtime_overlay_path(path: str | os.PathLike) -> Path:
    """运行时覆盖文件路径：`config.yaml` → `config.runtime.yaml`。

    为什么不直接改写 config.yaml：那个文件是手写的、带注释、按节组织。
    程序回写会把注释和排版全部碾平，用户下次打开就认不出来了。
    所以页面上的修改单独存一个覆盖文件，加载时叠加在 config.yaml 之上。
    """
    p = Path(path)
    return p.with_name(f"{p.stem}.runtime{p.suffix}")


def load_config(path: str | os.PathLike | None = None) -> AppConfig:
    """加载配置：YAML 文件（可选）+ 运行时覆盖 + 环境变量覆盖。"""
    data: dict = {}
    if path:
        p = Path(path)
        if p.exists():
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            overlay = runtime_overlay_path(p)
            if overlay.exists():
                try:
                    ov = yaml.safe_load(overlay.read_text(encoding="utf-8")) or {}
                    data = _deep_merge(data, ov)
                except Exception:  # noqa: BLE001
                    import logging

                    logging.getLogger(__name__).warning(
                        "运行时覆盖文件解析失败（已忽略）：%s", overlay, exc_info=True)
    cfg = AppConfig.model_validate(data)

    # token 解析优先级：环境变量 > 配置文件 > 自动生成持久化
    env_token = os.environ.get("AUTOGAME_TOKEN")
    if env_token:
        cfg.server.token = env_token
    elif cfg.server.token:
        pass
    else:
        token_file = _project_root() / ".token"
        if token_file.exists():
            cfg.server.token = token_file.read_text(encoding="utf-8").strip()
        else:
            cfg.server.token = secrets.token_urlsafe(32)
            token_file.write_text(cfg.server.token, encoding="utf-8")
            try:
                if os.name == "nt":
                    os.system(f'attrib +h "{token_file}"')
                else:
                    os.chmod(token_file, 0o600)
            except OSError:
                pass
    return cfg
