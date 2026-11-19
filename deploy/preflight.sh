#!/usr/bin/env bash
# 设备体检 —— 只读，不装任何东西、不改任何文件。
#
#     bash preflight.sh
#
# 目的是在动手装之前分清三种情况：
#   1. 这台机器根本装不了（架构/系统/Python 版本不对）—— 现在就知道，别装一半
#   2. 能装但要先补东西（没 venv/没 tesseract/磁盘不够/串口权限）
#   3. 已经装过了（/opt/autogame 存在）—— 那不是安装，是升级，路子完全不同
#
# 为什么单列这个脚本：install.sh 一旦跑到一半失败，现场是"半个系统"，
# 排查成本远高于先花 5 秒体检。

set -uo pipefail
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

RC=0
say_block() { printf '\n\033[1m%s\033[0m\n' "$*"; }

say_block "1. 机器"
ARCH=$(uname -m)
case "$ARCH" in
    aarch64|armv8l|arm64) ok "架构 $ARCH（aarch64，pip 有预编译轮子）" ;;
    armv7l) warn "架构 $ARCH（32 位 ARM）—— numpy/opencv 轮子基本没有，要现编，很可能编不动" ;;
    *) warn "架构 $ARCH —— 不是 ARM，能跑但不是目标机器" ;;
esac
if [ -r /etc/os-release ]; then
    . /etc/os-release
    ok "系统 ${PRETTY_NAME:-$ID $VERSION_ID}"
    case "${ID:-}" in
        debian)
            major=${VERSION_ID%%.*}
            [ "${major:-0}" -ge 12 ] 2>/dev/null && ok "Debian $VERSION_ID（python3 ≥ 3.11）" \
                || bad "Debian $VERSION_ID 自带 python3 是 3.9，装不了（要求 ≥3.10）：升到 12 或装 backports"
            ;;
        ubuntu)
            major=${VERSION_ID%%.*}
            [ "${major:-0}" -ge 22 ] 2>/dev/null && ok "Ubuntu $VERSION_ID" \
                || warn "Ubuntu $VERSION_ID 的 python3 偏旧，确认 ≥3.10"
            ;;
        *) warn "系统 ${ID:-未知}：install.sh 用的是 apt，非 Debian 系要自己换包管理器" ;;
    esac
else
    bad "没有 /etc/os-release，系统太老或不是 Linux"
    RC=1
fi
MEM_MB=$(awk '/MemTotal/{printf "%d", $2/1024}' /proc/meminfo 2>/dev/null || echo 0)
[ "${MEM_MB:-0}" -ge 900 ] && ok "内存 ${MEM_MB}MB" || warn "内存 ${MEM_MB}MB —— 偏紧，pip 装 opencv 时别开别的"

say_block "2. Python"
PY=""
for c in python3.12 python3.11 python3.10 python3; do
    have "$c" && { PY=$c; break; }
done
if [ -z "$PY" ]; then
    bad "没有 python3"
    RC=1
else
    PYV=$("$PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)
    PYOK=$("$PY" - <<'EOF' 2>/dev/null
import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)
EOF
    ) && ok "$PY 是 $PYV" || { bad "$PY 是 $PYV，要求 ≥3.10"; RC=1; }
    "$PY" -c 'import venv' 2>/dev/null && ok "venv 模块在" \
        || warn "缺 venv（apt install python3-venv）"
    "$PY" -c 'import ensurepip' 2>/dev/null || warn "缺 ensurepip（apt install python3-venv python3-pip）"
fi

say_block "3. 系统工具"
for t in tar systemctl curl; do
    have "$t" && ok "$t" || { bad "缺 $t"; RC=1; }
done
have tesseract && ok "tesseract $(tesseract --version 2>&1 | head -1 | awk '{print $2}')（读字用，可选）" \
    || warn "没有 tesseract（只影响 OCR，不影响主流程：apt install tesseract-ocr tesseract-ocr-chi-sim）"
have ss && { ss -ltn 2>/dev/null | grep -q ':8765 ' && bad "8765 已被占用（服务起不来）" || ok "8765 空闲"; } \
    || warn "没有 ss，端口占用自己确认一下"

say_block "4. 磁盘"
for d in /opt /var/lib; do
    avail=$(df -Pk "$d" 2>/dev/null | awk 'NR==2{print int($4/1024)}')
    if [ -z "$avail" ]; then warn "$d 不存在或不可查"; continue; fi
    # venv 全套依赖约 400MB（opencv 自己就 130MB），再加两个 release 各约 20MB
    [ "$avail" -ge 800 ] && ok "$d 剩 ${avail}MB" \
        || { bad "$d 只剩 ${avail}MB，装不下 venv（需要 ≥800MB）"; RC=1; }
done

say_block "5. 现场是否已装过"
if [ -d /opt/autogame ]; then
    warn "/opt/autogame 已存在 —— 这是**升级**不是首装，别跑 install.sh"
    [ -f /opt/autogame/current ] && ok "current = $(cat /opt/autogame/current 2>/dev/null)"
    if have systemctl; then
        st=$(systemctl is-active autogame 2>/dev/null || true)
        [ "$st" = "active" ] && ok "服务在跑" || warn "服务状态：${st:-未知}"
    fi
else
    ok "/opt/autogame 干净（首装）"
fi
[ -d /var/lib/autogame/profiles ] && warn "/var/lib/autogame/profiles 已有数据 —— 首装时**不会**被覆盖" || ok "/var/lib/autogame 干净"

say_block "6. 执行器（决定 service 里要开哪些设备）"
found_exec=0
for dev in /dev/ttyACM0 /dev/ttyUSB0 /dev/hidg0; do
    [ -e "$dev" ] || continue
    found_exec=1
    owner=$(stat -c '%U:%G' "$dev" 2>/dev/null)
    perm=$(stat -c '%A' "$dev" 2>/dev/null)
    case "$dev" in
        /dev/hidg*) ok "$dev $perm $owner —— USB gadget 键鼠（config 里用 rpi5_hidg）" ;;
        *)          ok "$dev $perm $owner —— 串口执行器（config 里用 rp2040，用户要在 dialout 组）" ;;
    esac
done
[ "$found_exec" = 0 ] && warn "没看到 ttyACM0/ttyUSB0/hidg0：执行器没插，或这台机器只做视觉决策（dryrun）"

say_block "7. 网络"
if have curl; then
    # 只查能不能解析并握手，不下载
    if curl -sS -m 6 -o /dev/null -w '%{http_code}' https://pypi.org/simple/ 2>/dev/null | grep -qE '^[23]'; then
        ok "能到 pypi（pip 装依赖没问题）"
    else
        warn "到不了 pypi：要么没网，要么要走代理/镜像（pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple）"
    fi
fi

say_block "结论"
if [ "$RC" -eq 0 ]; then
    echo "  能装。下一步：把 dist/autogame-x.y.z.tar.gz 传上来，"
    echo "  tar -xzf 包 deploy/install.sh && sudo bash deploy/install.sh 包"
else
    echo "  上面有 ✗：先解决再装。装到一半失败的现场比现在难收拾。"
fi
exit "$RC"
