# RP2040 组合 HID 执行器固件

服务端通过 USB 串口发 JSON 行命令，固件转成 **键盘 + 16bit 绝对坐标鼠标** 的 HID 输出。
游戏 PC 上不需要装任何驱动（识别为标准键盘/鼠标）。

## 协议

主机 → 固件（每行一条 JSON，`\n` 结尾）：

| 命令 | 说明 |
|---|---|
| `{"op":"key_tap","key":"a","hold_ms":60}` | 按下并释放一个键 |
| `{"op":"key_down","key":"w"}` / `{"op":"key_up","key":"w"}` | 按住 / 松开（走位用） |
| `{"op":"mouse_move","x":640,"y":400}` | 绝对坐标移动（0..屏幕宽-1） |
| `{"op":"mouse_click","button":"left","x":640,"y":800}` | 移动 + 点击 |
| `{"op":"stop_all"}` | 松开所有键与鼠标按钮（急停） |

固件 → 主机：`{"ok":true}` 或 `{"ok":false,"error":"..."}`。

**协议一致性由 `tests/test_firmware_protocol.py` 静态校验**（Report ID、报文长度、
坐标范围 32767、op 名称、键码表），改协议时该测试会立刻报错。

## 编译

### PlatformIO（推荐）

```bash
cd firmware/rp2040
pio run                 # 编译
pio run -t upload       # 烧录（板子处于 BOOTSEL 或已跑过 Arduino 固件时可直接上传）
```

`platformio.ini` 已固定板型与 TinyUSB 栈，无需手工选。

### Arduino IDE

1. 板管理器装 **Raspberry Pi Pico/RP2040**（Earle Philhower core）
2. 库管理器装 **Adafruit TinyUSB Library**
3. Tools → USB Stack 选 **Adafruit TinyUSB**
4. 打开 `hid_bridge/hid_bridge.ino` → 上传

## 烧录后自检（不发任何游戏按键）

```bash
# 在服务端机器上
python tools/preflight.py            # 会向串口发一条 stop_all 探活，期望 {"ok":true}
```

或手工确认设备枚举：

```bash
# Windows
Get-PnpDevice -Class HIDClass | Where-Object {$_.FriendlyName -match "Keyboard|Mouse"}
# Linux
ls /dev/ttyACM*        # 串口
dmesg | tail -20       # 应看到 HID 设备注册
```

## 接线与注意事项

- 板子用 **USB 数据线**（不是充电线）接游戏 PC；同一根线既做串口又做 HID。
- 服务端机器接板子的串口（`config.yaml: executor.serial_port`，Windows 形如 `COM5`）。
  > 注意：一块板子同时接两台机器需要 USB 双设备方案；常见做法是板子接**游戏 PC**（出 HID），
  > 串口也走这条 USB（服务端通过 USB/IP 或网络串口访问），或改用 RPi5 HID gadget 方案。
- 首次烧录后建议先跑 `executor.backend: dryrun` 的完整闭环，确认视觉/决策无误再接真机。
- 绝对鼠标坐标系是**整块虚拟桌面**；多显示器时请把游戏放在主显示器或调整
  `capture.region` / 坐标映射。
