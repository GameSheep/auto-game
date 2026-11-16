#!/usr/bin/env bash
# setup_rpi5_hidg.sh —— 树莓派5 USB gadget 模式配置
# 把 RPi5 的 USB-C 口配置成 组合HID 设备（键盘 hidg0 + 绝对鼠标 hidg1），
# 用一根 USB-C 数据线直连游戏 PC，即可毫秒级模拟键鼠。
#
# 前置：
#   1) /boot/firmware/config.txt  dtoverlay=dwc2
#   2) /boot/firmware/cmdline.txt 首行添加 modules-load=dwc2,libcomposite
#   3) 重启后以 root 运行本脚本: sudo bash setup_rpi5_hidg.sh
#
# 用另一台机器的 autogame RPi5HidgExecutor 写 /dev/hidg0(键盘) /dev/hidg1(鼠标)。

set -euo pipefail

G=/sys/kernel/config/usb_gadget/autogame

# 标准键盘报告描述符（无 Report ID）
KB_DESC=$(printf '\x05\x01\x09\x06\xa1\x01\x05\x07\x19\xe0\x29\xe7\x15\x00\x25\x01\x75\x01\x95\x08\x81\x02\x95\x01\x75\x08\x81\x01\x95\x06\x75\x08\x15\x00\x25\x65\x05\x07\x19\x00\x29\x65\x81\x00\xc0')
# 绝对鼠标报告描述符：8按钮 + 16bit X/Y(1..32767) + 滚轮（无 Report ID，报文6字节）
MS_DESC=$(printf '\x05\x01\x09\x02\xa1\x01\x05\x09\x19\x01\x29\x08\x15\x00\x25\x01\x95\x08\x75\x01\x81\x02\x05\x01\x09\x30\x09\x31\x16\x01\x00\x26\xff\x7f\x36\x01\x00\x46\xff\x7f\x75\x10\x95\x02\x81\x02\x09\x38\x15\x81\x25\x7f\x75\x08\x95\x01\x81\x06\xc0')

modprobe libcomposite

if [ -d "$G" ]; then
    echo "[*] gadget 已存在，先卸载"
    echo "" > "$G/UDC" 2>/dev/null || true
fi

mkdir -p "$G"
cd "$G"

# 设备描述符（合法 VID/PID 组合，标准键盘鼠标类）
echo 0x1d6b > idVendor   # Linux Foundation
echo 0x0104 > idProduct  # Multifunction Composite Gadget
echo 0x0100 > bcdUSB
echo 0x0100 > bcdDevice
echo 0x01   > bDeviceClass

mkdir -p strings/0x409
echo "autogame"        > strings/0x409/manufacturer
echo "autogame hid"    > strings/0x409/product
echo "000000000001"    > strings/0x409/serialnumber

# 功能1：键盘
mkdir -p functions/hidg0
echo 1 > functions/hidg0/protocol   # report protocol
echo 1 > functions/hidg0/subclass   # boot subclass
echo 8  > functions/hidg0/report_length
printf '%b' "$KB_DESC" > functions/hidg0/report_desc

# 功能2：绝对鼠标
mkdir -p functions/hidg1
echo 1 > functions/hidg1/protocol
echo 1 > functions/hidg1/subclass
echo 6  > functions/hidg1/report_length
printf '%b' "$MS_DESC" > functions/hidg1/report_desc

# 配置
mkdir -p configs/c.1
echo 120 > configs/c.1/MaxPower
mkdir -p configs/c.1/strings/0x409
echo "HID combo" > configs/c.1/strings/0x409/configuration

ln -sf functions/hidg0 configs/c.1/
ln -sf functions/hidg1 configs/c.1/

# 绑定 UDC
UDC=$(ls /sys/class/udc | head -n1)
if [ -z "$UDC" ]; then
    echo "[!] 未找到 UDC，检查 dtoverlay=dwc2 是否生效" >&2
    exit 1
fi
echo "$UDC" > UDC

echo "[✓] gadget 已启用: /dev/hidg0=键盘 /dev/hidg1=绝对鼠标 (UDC=$UDC)"
echo "    测试: printf '\\x00\\x00\\x04\\x00\\x00\\x00\\x00\\x00' > /dev/hidg0  (按a)"
