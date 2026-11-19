"""autogame —— 视觉驱动的游戏自动化框架（局域网架构）。

架构（对应 DeepSeek 对话定稿）：
  游戏 PC (采集代理, 只看不动手)
      │  WebSocket (JPEG 帧, token 鉴权, 局域网)
      ▼
  局域网小主机 (本框架服务端: OpenCV 视觉 + FSM 决策 + 监控仪表盘)
      │  命令 (串口 / HID gadget / 干跑)
      ▼
  执行器 (RP2040 组合HID 键盘+绝对鼠标 / RPi5 USB gadget / 干跑日志)

铁律（来自对话结论）：
  1. 绝不盲按拾取键 —— 只在检测到目标物品的 "A框" 交互提示时才按。
  2. 走位超时/撞墙自动放弃并拉黑目标，绝不原地卡死。
  3. 一键急停 (panic)，执行器立刻停止一切输出。
"""

from .version import ENGINE_VERSION as __version__  # noqa: E402  (单一版本来源)

__all__ = ["__version__"]
