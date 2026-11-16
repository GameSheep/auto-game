# -*- mode: python ; coding: utf-8 -*-
"""autogame 采集代理打包配置（对话定稿的产品化需求）。

产品形态：轻量 Windows 采集小软件 —— 只负责"看"（屏幕采集+上传），
不含任何按键/鼠标模拟库；用户另配 RP2040/RPi5 硬件执行器。

构建：  scripts/build_agent_exe.bat
产物：  dist/autogame-agent.exe
运行：  autogame-agent.exe --server http://192.168.1.20:8765 --token xxx
"""

a = Analysis(
    ["../run_agent.py"],
    pathex=[".."],
    binaries=[],
    datas=[],
    hiddenimports=["mss", "yaml"],
    hookspath=[],
    # 安全要求：排除按键模拟相关库（对话定稿 —— 采集程序绝不模拟输入）；
    # 服务端/视觉模块（fastapi/opencv 检测等）只跑在局域网小主机，不进代理包
    excludes=[
        "pynput", "pyautogui", "keyboard", "mouseinfo", "mouse",
        "PyMsgBox", "pygetwindow", "pyrect", "pyscreeze",
        "fastapi", "uvicorn", "starlette", "pydantic_settings",
        "pytest", "PyInstaller",
        "tkinter",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="autogame-agent",
    debug=False,
    strip=False,
    upx=False,
    console=True,           # 保留控制台便于查看采集日志
    disable_windowed_traceback=False,
)
