#!/usr/bin/env bash
# install_j4125.sh —— 局域网小主机（Debian/Ubuntu, J4125/RPi5 通用）一键部署
# 功能：系统依赖 + Python venv + 开机自启（systemd）
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE_NAME=autogame
RUN_USER="${SUDO_USER:-$USER}"

echo "[1/4] 系统依赖"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip tesseract-ocr tesseract-ocr-chi-sim

echo "[2/4] Python 环境（$APP_DIR/.venv）"
cd "$APP_DIR"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt -q
.venv/bin/pip install pytesseract rapidfuzz -q || echo "(可选 OCR 依赖安装失败，服务仍可运行)"

echo "[3/4] 生成配置与 token"
if [ ! -f "$APP_DIR/.token" ]; then
    .venv/bin/python -c "from autogame.config import load_config; load_config('config.yaml')"
    echo "token -> $APP_DIR/.token"
fi

echo "[4/4] systemd 服务"
sudo tee "/etc/systemd/system/${SERVICE_NAME}.service" >/dev/null <<EOF
[Unit]
Description=autogame vision/decision server
After=network-online.target

[Service]
User=${RUN_USER}
WorkingDirectory=${APP_DIR}
ExecStart=${APP_DIR}/.venv/bin/python run_server.py --config ${APP_DIR}/config.yaml
Restart=always
RestartSec=3
# 安全加固：只读文件系统 + 限制权限（.token 与历史日志需要写，故仅保护 /usr /boot /etc）
ProtectSystem=full
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now "${SERVICE_NAME}"

echo "完成。状态: systemctl status ${SERVICE_NAME}"
echo "token: $(cat "$APP_DIR/.token")"
