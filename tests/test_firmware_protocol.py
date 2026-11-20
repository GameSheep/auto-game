"""固件 ↔ 主机协议一致性测试（静态解析 + 字节级断言）。

为什么需要：`firmware/rp2040/hid_bridge.ino` 是 C++ 固件，不在 pytest 覆盖范围内。
历史上它只是"写完没编译过"的状态 —— 一旦主机侧改了 Report ID、报文长度或 op 名称，
固件不会报错，只会**静默不响应**，现场表现是"按键没反应"，极难排查。

本测试把固件源码当协议契约来读：
  1. 描述符里的 Report ID（键盘=1 / 鼠标=2）必须与主机报文一致
  2. 报告数组长度必须等于主机报文长度 + 1（Report ID 字节）
  3. 鼠标逻辑范围 32767 必须与主机坐标映射一致
  4. 主机发出的每个 op 名称，固件必须都能识别
  5. ASCII→HID 键码表必须覆盖主机 FSM 会按的键
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from autogame.executor.base import build_abs_mouse_report, build_keyboard_report

ROOT = Path(__file__).resolve().parent.parent
INO = ROOT / "firmware" / "rp2040" / "hid_bridge" / "hid_bridge.ino"

# 主机侧会发出的全部 op（protocol.Command.op 的实际取值）
HOST_OPS = ["key_tap", "key_down", "key_up", "mouse_move", "mouse_click", "stop_all"]


@pytest.fixture(scope="module")
def ino() -> str:
    assert INO.exists(), f"固件缺失: {INO}"
    return INO.read_text(encoding="utf-8", errors="ignore")


def test_report_ids_match(ino: str):
    """描述符 Report ID 必须是键盘=1、鼠标=2（与主机报文首字节一致）。"""
    ids = re.findall(r"0x85,\s*0x([0-9A-Fa-f]{2})", ino)
    assert ids[:2] == ["01", "02"], f"Report ID 顺序异常: {ids[:2]}"
    kb = build_keyboard_report(0x04)
    ms = build_abs_mouse_report(10, 10, 100, 100)
    # 主机报文不含 Report ID（hidg 路径由描述符区分；RP2040 固件自己补 ID 字节）
    assert len(kb) == 8 and len(ms) == 6


def test_report_buffer_sizes_match_host(ino: str):
    """固件缓冲区长度 = 主机报文长度 + 1（Report ID）。"""
    m_kb = re.search(r"kb_report\[(\d+)\]", ino)
    m_ms = re.search(r"ms_report\[(\d+)\]", ino)
    assert m_kb and m_ms, "找不到报告缓冲区定义"
    assert int(m_kb.group(1)) == len(build_keyboard_report()) + 1, "键盘报告长度不一致"
    assert int(m_ms.group(1)) == len(build_abs_mouse_report(0, 0, 100, 100)) + 1, \
        "鼠标报告长度不一致"


def test_mouse_logical_range_matches_host(ino: str):
    """鼠标逻辑上限 32767 必须与主机坐标映射一致（否则点击会偏）。"""
    assert "0x26, 0xFF, 0x7F" in ino, "描述符 Logical Max 应为 32767 (0xFF7F)"
    # 主机映射：x * 32767 / (w-1)
    r = build_abs_mouse_report(99, 49, 100, 50)
    ax = r[1] | (r[2] << 8)
    ay = r[3] | (r[4] << 8)
    assert ax == 32767, f"右下角应映射到 32767，实际 {ax}"
    assert ay == 32767, f"底部应映射到 32767，实际 {ay}"


def test_all_host_ops_handled_by_firmware(ino: str):
    """主机发出的每个 op，固件都必须有对应分支（否则静默丢弃）。"""
    missing = [op for op in HOST_OPS if f'"{op}"' not in ino]
    assert not missing, f"固件未处理这些 op: {missing}"


def test_pickup_key_present_in_keymap(ino: str):
    """拾取键 a / 走位键 wasd 必须在固件键码表里。"""
    for key in ["a", "w", "s", "d"]:
        assert f'{{"{key}",0x' in ino.replace(" ", ""), f"键码表缺 {key}"


def test_firmware_declares_tinyusb_stack(ino: str):
    """固件必须声明 TinyUSB（否则 USB HID 描述符不生效）。"""
    assert "Adafruit_TinyUSB.h" in ino
    assert "Adafruit_USBD_HID" in ino


# ---------------- 安全：卡住的键必须自己松开 ----------------

def test_hold_watchdog_exists_and_polls_in_loop(ino: str):
    """`key_down` 之后主机崩了，键会一直按着 —— 只有固件自己能救。

    主机侧的 stop_all 只能救"主机还活着"的场合；进程被 kill、USB 松一下，
    那条 stop_all 永远发不出来。所以固件必须有一个纯本地的超时释放。
    """
    assert "HOLD_WATCHDOG_MS" in ino, "固件缺少按键看门狗"
    assert "static void watchdog_tick" in ino
    # 必须在主循环里被调用，否则等于没写
    loop = ino[ino.index("void loop()"):]
    assert "watchdog_tick()" in loop


def test_hold_flag_set_on_key_down_and_cleared_by_release(ino: str):
    """看门狗只该在"有键按着"时动手；抬起/停全部要清掉这个标记。"""
    body = ino[ino.index('op == "key_down"'):]
    body = body[:body.index("} else if")] if "} else if" in body else body
    assert "hold_dirty = true" in body

    rel = ino[ino.index("static void release_all"):]
    rel = rel[:rel.index("static void watchdog_tick")]
    assert "hold_dirty = false" in rel, "release_all 必须清掉按下标记"


def test_watchdog_reports_itself(ino: str):
    """它自己松的手要主动报一句，否则现场只会看到"莫名少走一步"。"""
    assert "watchdog_release" in ino


def test_every_command_refreshes_last_rx(ino: str):
    """任何一条命令都算"主机还活着"，否则长时间只发 mouse_move 会被误判超时。"""
    body = ino[ino.index("static bool handle_line"):]
    assert "last_rx_ms = millis()" in body[:400]
