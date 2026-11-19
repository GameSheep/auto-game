#!/usr/bin/env bash
# 首次安装（可升级布局）—— 只在**新机器**上跑一次，之后升级只用一条命令：
#
#     sudo /opt/autogame/venv/bin/python /opt/autogame/releases/$(cat /opt/autogame/current)/tools/upgrade.py \
#          install autogame-0.2.1.tar.gz
#
# 或者用没有版本号的固定入口（推荐）：
#
#     sudo autogame-install autogame-0.2.1.tar.gz
#
# 用法：sudo deploy/install.sh dist/autogame-0.2.0.tar.gz
set -euo pipefail

PKG=${1:-}
if [ -z "$PKG" ] || [ ! -f "$PKG" ]; then
    echo "用法: sudo deploy/install.sh <升级包.tar.gz>" >&2
    exit 2
fi
PKG=$(cd "$(dirname "$PKG")" && pwd)/$(basename "$PKG")

APP=/opt/autogame
DATA=/var/lib/autogame
USER_NAME=autogame
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "[1/6] 系统依赖"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip tar tesseract-ocr tesseract-ocr-chi-sim \
    || echo "(OCR 依赖装不上不影响主流程，只影响读字)"

echo "[2/6] 用户与目录"
id "$USER_NAME" >/dev/null 2>&1 || sudo useradd --system --home "$APP" --shell /usr/sbin/nologin "$USER_NAME"
sudo mkdir -p "$APP/releases" "$APP/bin" "$APP/venv"
sudo mkdir -p "$DATA/profiles" "$DATA/logs" "$DATA/snapshots" "$DATA/packages"
sudo chown -R "$USER_NAME:$USER_NAME" "$DATA"
# releases / current 归 root，服务用户只读 —— "引擎不能改"的权限基础
sudo chown -R root:root "$APP/releases" "$APP/bin"

echo "[3/6] 引擎环境（venv 跨版本共用，升级不用重装依赖）"
if [ ! -x "$APP/venv/bin/pip" ]; then
    sudo python3 -m venv "$APP/venv"
fi
sudo tar -xzf "$PKG" -C "$TMP"          # 只为拿 requirements 和 tools/upgrade.py
sudo "$APP/venv/bin/pip" install -q --upgrade pip
sudo "$APP/venv/bin/pip" install -q -r "$TMP/requirements.txt"

echo "[4/6] 数据根初始化（已有数据一律不覆盖）"
[ -f "$DATA/config.yaml" ] || sudo cp "$TMP/config.yaml" "$DATA/config.yaml"
for p in _template generic_1280x720; do
    [ -d "$DATA/profiles/$p" ] || sudo cp -r "$TMP/profiles/$p" "$DATA/profiles/$p"
done
sudo chown -R "$USER_NAME:$USER_NAME" "$DATA"

echo "[5/6] 装 release（校验 → 自检 → 切指针）"
sudo cp "$PKG" "$DATA/packages/" 2>/dev/null || true
sudo env AUTOGAME_HOME="$APP" AUTOGAME_DATA="$DATA" \
    "$APP/venv/bin/python" "$TMP/tools/upgrade.py" install "$PKG" || {
        echo "安装失败：上面写了哪一步没过。current 指针没动，现场是干净的。" >&2
        exit 1
    }
# 装的时候是 root，快照/.schema 会带 root 属主；交回服务用户，免得以后写不进数据根
sudo chown -R "$USER_NAME:$USER_NAME" "$DATA"

echo "[6/6] 服务"
sudo cp "$TMP/deploy/autogame" "$APP/bin/autogame"
sudo chmod 755 "$APP/bin/autogame"
sudo cp "$TMP/deploy/autogame.service" /etc/systemd/system/autogame.service
# 固定入口：以后升级不用记 release 目录
sudo tee /usr/local/bin/autogame-install >/dev/null <<'EOF'
#!/bin/sh
# 升级到新包：sudo autogame-install autogame-x.y.z.tar.gz
set -eu
APP=/opt/autogame
VER=$(cat "$APP/current")
env AUTOGAME_HOME="$APP" AUTOGAME_DATA=/var/lib/autogame \
    "$APP/venv/bin/python" "$APP/releases/$VER/tools/upgrade.py" install "$@"
chown -R autogame:autogame /var/lib/autogame 2>/dev/null || true
echo "重启服务后生效：systemctl restart autogame"
EOF
sudo chmod 755 /usr/local/bin/autogame-install

sudo systemctl daemon-reload
sudo systemctl enable --now autogame
sleep 1

VER=$(cat "$APP/current")
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo
echo "完成。引擎 $VER 跑在 $IP"
echo "  配置页   http://${IP:-<设备IP>}:8765/settings"
echo "  token    sudo cat $DATA/.token"
echo "  升级     sudo autogame-install autogame-x.y.z.tar.gz"
echo "  回退     sudo env AUTOGAME_HOME=$APP AUTOGAME_DATA=$DATA $APP/venv/bin/python $APP/releases/$VER/tools/upgrade.py rollback"
echo "  日志     journalctl -u autogame -f"
