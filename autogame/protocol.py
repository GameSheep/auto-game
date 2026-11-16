"""通信协议 —— 代理/服务端/仪表盘之间的消息格式。

设计原则：
  - 帧（图像）走二进制 WebSocket 消息：固定小头部 + JPEG，避免 base64 开销。
  - 控制消息走 JSON 文本：可读、易调试、schema 用 pydantic 校验。

二进制帧头（小端）：
  byte 0    : msg_type（0x01=JPEG 帧）
  uint32    : 时间戳 ms（代理端采集时刻）
  uint16    : 宽
  uint16    : 高
  后续字节  : JPEG 数据
"""

from __future__ import annotations

import json
import struct
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

MSG_JPEG: int = 0x01
_HEADER = struct.Struct("<BIHH")  # type, ts_ms, w, h


def pack_jpeg_frame(jpeg: bytes, width: int, height: int, ts_ms: int) -> bytes:
    return _HEADER.pack(MSG_JPEG, ts_ms & 0xFFFFFFFF, width, height) + jpeg


def unpack_message(raw: bytes | str) -> tuple[str, dict[str, Any]]:
    """解析一条 WS 消息，返回 (kind, payload)。

    kind: "jpeg" | "json"
    """
    if isinstance(raw, str):
        obj = json.loads(raw)
        if not isinstance(obj, dict) or "type" not in obj:
            raise ValueError("bad json message")
        return "json", obj
    if len(raw) <= _HEADER.size:
        raise ValueError("binary message too short")
    msg_type, ts_ms, w, h = _HEADER.unpack(raw[: _HEADER.size])
    if msg_type != MSG_JPEG:
        raise ValueError(f"unknown binary msg_type={msg_type}")
    return "jpeg", {
        "ts_ms": ts_ms,
        "w": w,
        "h": h,
        "jpeg": raw[_HEADER.size :],
    }


# ---------------- JSON 控制消息（代理 ↔ 服务端） ----------------

class HelloMsg(BaseModel):
    type: Literal["hello"] = "hello"
    agent: str = "unknown"
    version: str = ""
    screen: Optional[list[int]] = None  # [w, h]


class PingMsg(BaseModel):
    type: Literal["ping"] = "ping"
    ts_ms: int = 0


class PongMsg(BaseModel):
    type: Literal["pong"] = "pong"
    ts_ms: int = 0


class AckMsg(BaseModel):
    type: Literal["ack"] = "ack"
    ok: bool = True
    error: str = ""


class SetCaptureMsg(BaseModel):
    """服务端 → 代理：调整采集参数。"""

    type: Literal["set_capture"] = "set_capture"
    fps: Optional[float] = None
    jpeg_quality: Optional[int] = None
    region: Optional[list[int]] = None


# ---------------- 服务端内部数据结构 ----------------

class Pillar(BaseModel):
    """检测到的光柱。"""

    x: int
    y: int            # 底部中心（地面点击点参考）
    cx: int
    cy: int           # 外接框中心
    w: int
    h: int
    color: str        # gold / pink
    area: int
    distance: float   # 距角色中心的像素距离
    whitelisted: bool = True  # 无 OCR 白名单时恒为 True
    label: str = ""
    seen_frames: int = 0  # 连续可见帧数（时域跟踪）
    stable: bool = False  # 达到 beam_stable_frames → 稳定光柱（掉落物特征，特效短命）""


class AFrame(BaseModel):
    """检测到的 A 框（拾取交互提示）。"""

    x: int
    y: int
    w: int
    h: int
    score: float


class ButtonHit(BaseModel):
    """检测到的 UI 按钮（如『再次挑战』）。"""

    x: int
    y: int          # 中心点（点击坐标）
    w: int
    h: int
    score: float


class ListEntry(BaseModel):
    """右侧 UI 掉落列表的条目（点击条目=走过去拾取）。"""

    x: int
    y: int          # 条目中心（点击坐标）
    w: int
    h: int


class Monster(BaseModel):
    """战斗区域内识别到的怪物（点击=移动+攻击）。"""

    x: int
    y: int
    w: int
    h: int


class MinimapInfo(BaseModel):
    """小地图目标标记信息（导航输入）。"""

    center: tuple[int, int]       # 小地图中心（屏幕坐标）
    radius: int                   # 小地图半径（px）
    marker: Optional[tuple[int, int]] = None  # 目标标记点（屏幕坐标）
    angle_deg: Optional[float] = None         # 标记相对中心的方向（0=右, 90=下）
    distance_ratio: Optional[float] = None    # 距中心距离 / 半径
    arrived: bool = False


class SceneState(BaseModel):
    """一帧画面解析后的完整场景状态（决策输入）。"""

    ts_ms: int
    w: int
    h: int
    character: tuple[int, int]
    pillars: list[Pillar] = Field(default_factory=list)
    aframe: Optional[AFrame] = None
    rechallenge: Optional[ButtonHit] = None
    minimap: Optional[MinimapInfo] = None
    item_list: list[ListEntry] = Field(default_factory=list)
    ground_labels: list[ListEntry] = Field(default_factory=list)  # 地面掉落标签（黄条=掉落物）
    monsters: list[Monster] = Field(default_factory=list)  # 战斗区域怪物
    in_game: bool = True  # 场景门：战斗 HUD 存在才为 True（False 时 FSM 禁止输出命令）
    loading: bool = False  # 加载/黑屏过场（FSM 保持不动）
    frame_delta: float = 0.0  # 与上一帧的变化像素占比（防撞墙用）
    ocr_used: bool = False


# ---------------- 决策 → 执行器命令 ----------------

class Command(BaseModel):
    """单条执行器命令。"""

    op: Literal["key_tap", "key_down", "key_up", "mouse_move", "mouse_click", "stop_all"]
    key: Optional[str] = None
    hold_ms: int = 60
    button: Literal["left", "right"] = "left"
    x: Optional[int] = None
    y: Optional[int] = None


Command.model_rebuild()


def dumps(obj: BaseModel | dict) -> str:
    if isinstance(obj, BaseModel):
        return obj.model_dump_json(exclude_none=True)
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def format_command(cmd: Command) -> str:
    """命令 → 人类可读文本（文本输出模式）。"""
    if cmd.op == "key_tap":
        return f"KEY_TAP {cmd.key} hold={cmd.hold_ms}ms"
    if cmd.op == "key_down":
        return f"KEY_DOWN {cmd.key}"
    if cmd.op == "key_up":
        return f"KEY_UP {cmd.key}"
    if cmd.op == "mouse_move":
        return f"MOUSE_MOVE ({cmd.x},{cmd.y})"
    if cmd.op == "mouse_click":
        pos = f" ({cmd.x},{cmd.y})" if cmd.x is not None and cmd.y is not None else ""
        return f"MOUSE_CLICK {cmd.button}{pos}"
    if cmd.op == "stop_all":
        return "STOP_ALL"
    return f"{cmd.op}"
