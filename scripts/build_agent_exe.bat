@echo off
REM 构建 autogame 采集代理 EXE（对话定稿的产品化交付形态）
REM 产物: dist\autogame-agent.exe —— 只采集画面上传，不含任何按键模拟
REM 用户侧部署: 双击运行 autogame-agent.exe --server http://<小主机IP>:8765 --token <token>

cd /d "%~dp0.."
.venv\Scripts\python -m PyInstaller packaging\agent.spec --noconfirm --distpath dist --workpath build
if errorlevel 1 (
    echo 构建失败
    exit /b 1
)
echo 构建完成: dist\autogame-agent.exe
echo 验证: dist\autogame-agent.exe --help
