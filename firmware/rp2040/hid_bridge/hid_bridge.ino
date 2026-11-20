// hid_bridge.ino —— RP2040 组合 HID 执行器固件
// 平台：Raspberry Pi Pico / RP2040（Arduino-Pico core + Adafruit TinyUSB）
//
// 功能：USB 串口收 JSON 行命令 → 键盘 + 绝对坐标鼠标输出
// 协议（与服务端 autogame/protocol.py Command 对齐）：
//   {"op":"key_tap","key":"a","hold_ms":60}
//   {"op":"key_down","key":"w"} / {"op":"key_up","key":"w"}
//   {"op":"mouse_move","x":640,"y":400}
//   {"op":"mouse_click","button":"left","x":640,"y":800}
//   {"op":"stop_all"}
//   {"op":"ping"}  → {"ok":true,"fw":"1.1.0","up_ms":...,"watchdog_ms":5000}
// 每条命令回 {"ok":true} 或 {"ok":false,"error":"..."}
// 命令解析顺序即优先级：ping 放在最后但**独立**，不受其它分支影响。
//
// Arduino IDE 设置：
//   Board: Raspberry Pi Pico (Arduino Mbed OS RP2040 Boards 不支持 TinyUSB 时，
//          推荐 Earle Philhower core) ; USB Stack: "Adafruit TinyUSB"
//   Library: Adafruit TinyUSB Library
// PlatformIO 用户见同目录 platformio.ini。

#include <Arduino.h>
#include <Adafruit_TinyUSB.h>

// ---------------- HID 报告描述符（单接口双 Report ID） ----------------
// Report ID 1: 标准 6KRO 键盘
// Report ID 2: 绝对坐标鼠标（16bit X/Y, 0..32767 → 全屏映射）+ 滚轮
static const uint8_t desc_hid_report[] = {
  // 键盘
  0x05, 0x01,       // Usage Page (Generic Desktop)
  0x09, 0x06,       // Usage (Keyboard)
  0xA1, 0x01,       // Collection (Application)
  0x85, 0x01,       //   Report ID (1)
  0x05, 0x07,       //   Usage Page (Key Codes)
  0x19, 0xE0, 0x29, 0xE7,  //   Usage Min 0xE0 Max 0xE7 (修饰键)
  0x15, 0x00, 0x25, 0x01,
  0x75, 0x01, 0x95, 0x08,
  0x81, 0x02,       //   Input (Data, Var, Abs) 修饰键位图
  0x95, 0x01, 0x75, 0x08,
  0x81, 0x01,       //   Input (Const) 保留字节
  0x95, 0x06, 0x75, 0x08,
  0x15, 0x00, 0x25, 0x65,
  0x05, 0x07, 0x19, 0x00, 0x29, 0x65,
  0x81, 0x00,       //   Input (Data, Array) 6 键位
  0xC0,             // End Collection

  // 绝对鼠标
  0x05, 0x01,       // Usage Page (Generic Desktop)
  0x09, 0x02,       // Usage (Mouse)
  0xA1, 0x01,       // Collection (Application)
  0x85, 0x02,       //   Report ID (2)
  0x09, 0x01,       //   Usage (Pointer)
  0xA1, 0x00,       //   Collection (Physical)
  0x05, 0x09,       //     Usage Page (Buttons)
  0x19, 0x01, 0x29, 0x03,
  0x15, 0x00, 0x25, 0x01,
  0x95, 0x03, 0x75, 0x01,
  0x81, 0x02,       //     Input (Data, Var, Abs) 3 按钮
  0x95, 0x05, 0x75, 0x01,
  0x81, 0x03,       //     Input (Const) 填充
  0x05, 0x01,
  0x09, 0x30, 0x09, 0x31,   //     Usage X / Y
  0x16, 0x01, 0x00,         //     Logical Min 1
  0x26, 0xFF, 0x7F,         //     Logical Max 32767
  0x36, 0x01, 0x00,         //     Physical Min 1
  0x46, 0xFF, 0x7F,         //     Physical Max 32767
  0x75, 0x10, 0x95, 0x02,
  0x81, 0x02,       //     Input (Data, Var, Abs) 16bit X/Y
  0x09, 0x38,       //     Usage Wheel
  0x15, 0x81, 0x25, 0x7F,
  0x75, 0x08, 0x95, 0x01,
  0x81, 0x06,       //     Input (Data, Var, Rel)
  0xC0, 0xC0        // End Collections
};

// 键盘报告: [id=1][mod][reserved][k1..k6]
static uint8_t kb_report[9] = {1, 0, 0, 0, 0, 0, 0, 0, 0};
// 鼠标报告: [id=2][btn][x_lo][x_hi][y_lo][y_hi][wheel]
static uint8_t ms_report[7] = {2, 0, 0, 0, 0, 0, 0};

Adafruit_USBD_HID usb_hid;

// ---------------- ASCII → HID 键码 ----------------
struct KeyMap { const char* name; uint8_t code; };
static const KeyMap KEYMAP[] = {
  {"a",0x04},{"b",0x05},{"c",0x06},{"d",0x07},{"e",0x08},{"f",0x09},{"g",0x0A},
  {"h",0x0B},{"i",0x0C},{"j",0x0D},{"k",0x0E},{"l",0x0F},{"m",0x10},{"n",0x11},
  {"o",0x12},{"p",0x13},{"q",0x14},{"r",0x15},{"s",0x16},{"t",0x17},{"u",0x18},
  {"v",0x19},{"w",0x1A},{"x",0x1B},{"y",0x1C},{"z",0x1D},
  {"1",0x1E},{"2",0x1F},{"3",0x20},{"4",0x21},{"5",0x22},
  {"6",0x23},{"7",0x24},{"8",0x25},{"9",0x26},{"0",0x27},
  {"enter",0x28},{"esc",0x29},{"backspace",0x2A},{"tab",0x2B},{"space",0x2C},
  {"capslock",0x39},{"f1",0x3A},{"f2",0x3B},{"f3",0x3C},{"f4",0x3D},
  {"f5",0x3E},{"f6",0x3F},{"f7",0x40},{"f8",0x41},{"f9",0x42},
  {"f10",0x43},{"f11",0x44},{"f12",0x45},
  {"up",0x52},{"down",0x51},{"left",0x50},{"right",0x4F},
};
// 修饰键 bit（HID 修饰键位图顺序）
static const struct { const char* n; uint8_t b; } MODS[] = {
  {"ctrl",0x01},{"shift",0x02},{"alt",0x04},{"win",0x08},
};

static uint8_t lookup_key(const String& k) {
  String lk = k; lk.toLowerCase(); lk.trim();
  for (auto& m : KEYMAP) if (lk == m.name) return m.code;
  return 0;
}

// ---------------- JSON 行解析（轻量手写，避免引库） ----------------
// 提取 "key":"value" 或 "key":number
static String json_get(const String& line, const char* key) {
  String pat = String("\"") + key + "\":";
  int i = line.indexOf(pat);
  if (i < 0) return "";
  i += pat.length();
  while (i < (int)line.length() && line[i] == ' ') i++;
  if (line[i] == '"') {
    int j = line.indexOf('"', i + 1);
    return (j < 0) ? "" : line.substring(i + 1, j);
  }
  int j = i;
  while (j < (int)line.length() && (isDigit(line[j]) || line[j]=='-' || line[j]=='+' || line[j]=='.')) j++;
  return line.substring(i, j);
}

static bool modifier_of(const String& k, uint8_t& bit) {
  String lk = k; lk.toLowerCase();
  for (auto& m : MODS) if (lk == m.n) { bit = m.b; return true; }
  return false;
}

// ---------------- 命令执行 ----------------
static void send_json(Stream& out, bool ok, const char* err = "") {
  out.print(ok ? "{\"ok\":true}" : String("{\"ok\":false,\"error\":\"") + err + "\"}");
  out.print("\n");
}

// ---------------- 通信看门狗 ----------------
// 为什么这一条必须在**固件**里，而不是主机里：
//   主机侧的 stop_all 只能救"主机还活着"的场合。而 key_down 之后主机崩了、
//   Python 进程被 kill、USB 线松了一下 —— 这时候键盘上那个键是**真的按着**的，
//   游戏里的人物会一直往前走，撞墙也在走，直到有人去拔 USB。
//   能救这种情况的只有对端：超时自己松手。
//
// 阈值取 5 秒的依据：本项目走位全是 key_tap + hold_ms（按压在固件里 delay 后抬起），
// 没有"长按走路"的路径，所以正常运行时按下状态存在的时间是几百毫秒。5 秒足够宽松，
// 只有真的断了才会触发。将来若加了长按走位，这个值要跟着调大。
static const uint32_t HOLD_WATCHDOG_MS = 5000;
static uint32_t last_rx_ms = 0;
static bool hold_dirty = false;      // 有键/鼠标处于按下状态且没被抬起

static void release_all() {
  kb_report[1] = 0;
  for (int i = 3; i < 9; i++) kb_report[i] = 0;
  usb_hid.sendReport(1, kb_report, sizeof(kb_report));
  ms_report[1] = 0;
  usb_hid.sendReport(2, ms_report, sizeof(ms_report));
  hold_dirty = false;
}

// ---------------- 探活 ----------------
// 主机的自检靠这一条区分三种完全不同的故障：
//   回 {"ok":true,...}                   → 对面就是这套固件，版本够新
//   回 {"ok":false,"error":"unknown op"} → 串口对面确实是这套固件，但太旧（要重刷）
//   什么都不回 / 回乱码                   → 对面是别的 CDC 设备，或者固件卡死了
// 没有这一条的话，串口号配错（打开了另一个 USB 串口设备）时的表现是
// "命令全部石沉大海，主机以为一切正常" —— 上真机最难查的一类问题。
//
// 带上 fw / up_ms：同一台机器刷了两份固件时，能看出跑的是哪一份。
static const char* FW_VERSION = "1.1.0";
static void send_ping() {
  Serial.print("{\"ok\":true,\"fw\":\"");
  Serial.print(FW_VERSION);
  Serial.print("\",\"up_ms\":");
  Serial.print((unsigned long)millis());
  Serial.print(",\"watchdog_ms\":");
  Serial.print((unsigned long)HOLD_WATCHDOG_MS);
  Serial.print("}\n");
}

static void watchdog_tick() {
  if (!hold_dirty || last_rx_ms == 0) return;
  if (millis() - last_rx_ms < HOLD_WATCHDOG_MS) return;
  release_all();
  // 主动报一句：主机重连后能看见"上一轮是我自己松的手"，而不是以为一切正常
  send_json(Serial, false, "watchdog_release");
}

static bool handle_line(const String& line) {
  last_rx_ms = millis();          // 看门狗的"还活着"信号：任何一条命令都算
  String op = json_get(line, "op");
  if (op.length() == 0) return false;
  String key = json_get(line, "key");
  long hold_ms = json_get(line, "hold_ms").toInt();
  long x = json_get(line, "x").toInt();
  long y = json_get(line, "y").toInt();
  String button = json_get(line, "button"); if (button.length() == 0) button = "left";

  if (op == "key_tap") {
    uint8_t code = lookup_key(key);
    if (!code) { send_json(Serial, false, "unknown key"); return true; }
    uint8_t mod = 0;
    modifier_of(key, mod);  // 支持组合键名（shift/ctrl…），一般不用
    kb_report[1] = mod; kb_report[3] = code;
    usb_hid.sendReport(1, kb_report, sizeof(kb_report));
    delay(constrain(hold_ms, 10, 2000));
    kb_report[1] = 0; kb_report[3] = 0;
    usb_hid.sendReport(1, kb_report, sizeof(kb_report));
    send_json(Serial, true);
  } else if (op == "key_down") {
    uint8_t code = lookup_key(key);
    if (!code) { send_json(Serial, false, "unknown key"); return true; }
    kb_report[3] = code;
    usb_hid.sendReport(1, kb_report, sizeof(kb_report));
    hold_dirty = true;            // 从这一刻起，主机不来消息就得我自己松手
    send_json(Serial, true);
  } else if (op == "key_up") {
    for (int i = 3; i < 9; i++) kb_report[i] = 0;
    usb_hid.sendReport(1, kb_report, sizeof(kb_report));
    send_json(Serial, true);
  } else if (op == "mouse_move") {
    uint32_t ax = constrain(x, 0, 32767);
    uint32_t ay = constrain(y, 0, 32767);
    ms_report[2] = ax & 0xFF; ms_report[3] = ax >> 8;
    ms_report[4] = ay & 0xFF; ms_report[5] = ay >> 8;
    usb_hid.sendReport(2, ms_report, sizeof(ms_report));
    send_json(Serial, true);
  } else if (op == "mouse_click") {
    if (x >= 0 && y >= 0) {
      uint32_t ax = constrain(x, 0, 32767);
      uint32_t ay = constrain(y, 0, 32767);
      ms_report[2] = ax & 0xFF; ms_report[3] = ax >> 8;
      ms_report[4] = ay & 0xFF; ms_report[5] = ay >> 8;
      usb_hid.sendReport(2, ms_report, sizeof(ms_report));
    }
    ms_report[1] = (button == "right") ? 0x02 : 0x01;
    usb_hid.sendReport(2, ms_report, sizeof(ms_report));
    // 按压时长由主机下发（人类化：55-135ms 随机）；缺省回落 45ms 兼容旧客户端
    delay(constrain(hold_ms > 0 ? hold_ms : 45, 10, 500));
    ms_report[1] = 0;
    usb_hid.sendReport(2, ms_report, sizeof(ms_report));
    send_json(Serial, true);
  } else if (op == "stop_all") {
    release_all();
    send_json(Serial, true);
  } else if (op == "ping") {
    // 自检探活，见上面 send_ping() 的注释。**不碰键盘鼠标**：探活本身
    // 不能产生任何输入，否则"自检一下"就等于按了一下键。
    send_ping();
  } else {
    send_json(Serial, false, "unknown op");
  }
  return true;
}

// ---------------- setup / loop ----------------
static String rx_buf;

void setup() {
  Serial.begin(115200);
  // 等待 USB 枚举（限 3s，无主机也能跑）
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) {}
  usb_hid.setPollInterval(2);
  usb_hid.setReportDescriptor(desc_hid_report, sizeof(desc_hid_report));
  usb_hid.begin();
  release_all();
}

void loop() {
  watchdog_tick();                 // 先看门狗：卡住的键比晚 1ms 收命令更要紧
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      rx_buf.trim();
      if (rx_buf.length()) handle_line(rx_buf);
      rx_buf = "";
    } else if (c != '\r') {
      rx_buf += c;
      if (rx_buf.length() > 256) rx_buf = "";  // 防溢出
    }
  }
}
