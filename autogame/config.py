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
    pillar_min_aspect: float = Field(1.1, ge=0.5, description="高/宽最小比例（细高条状）")
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
    """战斗 HUD 门：技能栏模板命中 → in_game=True。

    threshold 0.6 对应半分辨率匹配（实测：战斗帧 ≥0.668，UI 帧 ≤0.26，间隔充足）。
    """

    enabled: bool = True
    template: str = Field("skillbar.png", description="技能栏模板（templates/ 目录）")
    threshold: float = Field(0.6, ge=0.4, le=0.99)
    roi: Optional[list[int]] = Field(
        None, description="匹配区域 [x,y,w,h]（原分辨率）；None=底部 35% 全宽"
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
    allowed_actions: list[str] = Field(["wait", "continue"])


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


class ServerConfig(BaseModel):
    """服务端（局域网小主机）配置。"""

    host: str = "0.0.0.0"
    port: int = 8765
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
VisionConfig.model_rebuild()
AppConfig.model_rebuild()


def load_config(path: str | os.PathLike | None = None) -> AppConfig:
    """加载配置：YAML 文件（可选）+ 环境变量覆盖。"""
    data: dict = {}
    if path:
        p = Path(path)
        if p.exists():
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
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
