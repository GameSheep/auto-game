# autogame —— 视觉驱动的局域网游戏自动化框架

根据 DeepSeek 对话定稿的架构实现：**游戏 PC 只看（WGC/DXGI 采集），局域网小主机想（视觉+决策+LLM顾问），硬件执行器动手（组合 HID）**。

```
┌───────────────────────────┐         ┌──────────────────────────────────────┐
│ 游戏 PC (8745HS, Windows) │         │ 局域网小主机 (J4125 / RPi5)           │
│                           │         │                                      │
│  run_agent.py             │         │  run_server.py                       │
│  ├─ dxcam / mss 采集      │  JPEG   │  ├─ OpenCV 视觉检测                  │
│  │  (只读画面, 零注入)    │ ──────► │  │   光柱(HSV) · A框(模板/镂空)      │
│  └─ WebSocket 客户端      │  token  │  │   再次挑战按钮 · 小地图标记        │
└───────────────────────────┘  鉴权   │  ├─ FSM 决策状态机                   │
                                      │  │   拾取闭环 + 刷图循环 + 导航      │
                                      │  ├─ Ollama 决策顾问(可选)            │
                                      │  ├─ 监控仪表盘 (MJPEG)               │
                                      │  └─ 事件审计日志 (JSONL)             │
                                      └───────────┬──────────────────────────┘
                                                  │ 命令 (串口/HID gadget)
                                      ┌───────────▼──────────────────────────┐
                                      │ 执行器                               │
                                      │  RP2040 固件 (键盘+绝对鼠标)        │
                                      │  RPi5 USB gadget (/dev/hidg*)       │
                                      │  dryrun (测试, 零输出)              │
                                      └──────────────────────────────────────┘
```

## 实测性能（16核 x86 开发机，J4125 量级更充裕）

| 测项 | 数值 |
|---|---|
| 检测器单帧 @720p / 1080p / 1440p / 2160p（含场景门+新检测项） | 12-16 ms（≥60 fps 等效） |
| JPEG 编码 @1440p（代理侧） | ~10 ms |
| JPEG 解码 @1440p（服务端侧） | ~12 ms |
| **端到端单帧预算（解码+检测）** | **37 ms → 1Hz 决策 ≈ 3.7% 单核**（含场景门/加载判定/列表检测） |
| 内存（100帧处理驻留/峰值） | 6.8 / 14.8 MB |
| 带宽 @1440p q80, 2fps | ~207 KB/s |

复现：`.venv/Scripts/python scripts/benchmark.py`（结果写 `benchmark_result.json`）。

## 核心铁律（代码层面强制，测试断言）

1. **绝不盲按拾取键**：只有检测到目标物品的 **A框**（近距交互提示）且 A框邻近当前目标时才按 A。
   - 镂空率过滤（空心框线 vs 实心光柱）、邻近目标约束、150ms 防闪烁确认。
   - A框闪现即消失且从未按过键 → 记为误检闪烁，**绝不算拾取成功**（`picks ≤ a_taps` 不变量）。
2. **绝不原地卡死**：走位超时（默认 8s）放弃并拉黑区域 30s；帧差过低判定撞墙 →
   侧向脱困一次 → 仍卡则放弃。
3. **一键急停**：`POST /api/control/panic` 或仪表盘 ⛔ → 执行器立即停止一切输出。
4. **执行器限速**：最小动作间隔（自适应等待不丢命令）+ 每秒上限，所有后端统一生效。

## 完整刷图主循环（对话定稿的产品闭环）

```
拾取光柱物品 → 拾满/通关 → 结算画面检出『再次挑战』按钮(连续200ms防抖)
  → 点击按钮 → 下一张图 → 继续拾取 ……
  ↳ 无目标时小地图导航：目标标记方向 → 8方向键走位 → 到达目标点
  ↳ 连续放弃 ≥N 次 → 咨询 Ollama 顾问（建议记录仪表盘，白名单动作才执行）
```

端到端验证：`tests/test_e2e.py::test_farming_loop_full_stack` 走真实服务端栈完成
拾满→结算→点按钮→下一图→继续拾取，全程零盲按。

## 真实截图验证（网上找的实机素材，非自产画面）

`assets/samples/` 收录 8 张真实《火炬之光：无限》实机截图（来源：B站实机视频抽帧、17173 评测文章），
覆盖 Boss战+完整HUD / 粉色Boss触手(类光柱) / 技能特效满屏 / 暗黑刷图 / 装备面板 / 背包界面。

```bash
# 真实图 → 自适应标定 → 检测 → FSM 决策 → 文本输出命令
.venv/Scripts/python tools/scenario_runner.py --dir assets/samples
```

实测结论（tests/test_real_scenarios.py，25 个断言）：
- 场景门（技能栏多模板锚点随仓库分发）：战斗 HUD 放行，装备面板/背包/对话框拦截 → UI 界面零命令
- HUD 排除区：小地图/任务/聊天/技能栏区域强制排除（技能栏上的拾取键提示不会变成假 A框）
- 白芯校验：掉落光柱有白热芯；Boss 触手/技能光束是纯饱和色 → 全部拒绝（实测 Boss 触手 4 帧零误报、零按 A）
- 时域确认：光柱需连续 2 帧出现（单帧特效闪烁不触发）
- 拾取复核冷却：按 A 后 500ms 内禁止再按（真实素材回放中发现的连按捡垃圾时序 bug）
- 自适应标定 auto_hsv_ranges：从任意实机帧自动推断高亮色簇（无需手工标定即可工作）
- 暗图真实疑似光柱 → 仅输出 `MOUSE_CLICK left (x,y)` 走位命令，无 A框 绝不按 A
- 视频回放支持 `--csv` 导出逐 tick 判定时间线；100 图批量自检同步导出 `assets/batch_list.csv`

**100 张真实帧批量自检**（`tools/batch_validate.py`，10 条自动判定规则 R1-R10）：
PASS 97 / WARN 3 / FAIL 0，全程零盲按、UI 帧零命令、确定性验证通过、平均 100ms/图。
过程中借真实数据修复：A框色相无关检测（粉/紫稀有度框漏检 bug）、Boss 血条区误检、场景门半分辨率性能。

## 快速开始（本机模拟闭环，无需游戏）

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Windows
# Linux: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 终端1: 服务端 + 内置模拟器（完整刷图循环自动运行）
.venv/Scripts/python run_server.py --sim

# 浏览器 http://127.0.0.1:8765/ ，token 见 .token 文件
# 点 ▶ 启动 → 看拾取/刷图/导航统计与事件流
```

运行测试（143 个用例，含端到端闭环/真实截图回归/UI 场景任务/审查页/硬件协议/fuzz 加固/性能守卫）：

```bash
.venv/Scripts/python -m pytest tests/ -q
```

## 真实部署

### 1. 游戏 PC 采集代理（Windows）

```bash
.venv/Scripts/pip install dxcam          # 推荐：DXGI GPU 直出
.venv/Scripts/python run_agent.py --server http://192.168.1.20:8765 \
    --token <服务端token> --fps 2
# 开机自启（可选）:
# schtasks /create /tn autogame-agent /tr "C:\Code\auto-game\.venv\Scripts\python run_agent.py ..." /sc onlogon
```

安全边界：代理只读屏幕画面（与 OBS/Discord 同一系统 API），不注入进程、不读内存、
**不模拟任何键鼠输入**。

### 2. 局域网小主机服务端（J4125 / RPi5, Debian）

```bash
bash scripts/install_j4125.sh     # 系统依赖 + venv + systemd 自启 + tesseract OCR
systemctl status autogame
cat /opt/auto-game/.token
```

### 3. 视觉标定（三步，全部有工具）

```bash
# ① 游戏里出现目标光柱 → 抓图
.venv/Scripts/python tools/grab_frame.py --out shot.png

# ② 对光柱矩形采样 → 输出 HSV 范围（粘进 config.yaml）
.venv/Scripts/python tools/calibrate_hsv.py --image shot.png --rect 600,300,80,160 --name gold

# ③ 裁 A框 小图存 templates/aframe.png → 启用多尺度模板匹配（精度最高）
#    同理裁『再次挑战』按钮存 templates/rechallenge.png
```

### 3.5 采集代理打包成 EXE（产品化交付）

```bash
scriptsuild_agent_exe.bat        # 或: .venv/Scripts/python -m PyInstaller packaging/agent.spec
distutogame-agent.exe --server http://<小主机IP>:8765 --token <token>
```
单文件、零安装、已排除 pynput/pyautogui/keyboard 等按键模拟库（安全验证在 spec excludes）。

### 4. 执行器（四选一）

| 后端 | 配置 | 说明 |
|---|---|---|
| `dryrun` | 默认 | 只记录日志，零真实输出。测试/联调必用 |
| `rp2040` | `serial_port`, `serial_baud` | RP2040 刷入 `firmware/rp2040/`（Arduino-Pico + TinyUSB），USB 串口 JSON 行协议，组合 HID（键盘 + 16bit 绝对鼠标） |
| `rpi5_hidg` | `hidg_keyboard`, `hidg_mouse` | RPi5 USB-C 数据线直连 PC，`sudo bash scripts/setup_rpi5_hidg.sh` |
| `pca9685` | `servo_map`, `pca9685_bus` | PCA9685+舵机机械按压（对话最初的机械方案；仅按键无鼠标移动，配 approach_mode=keys） |

报文字节序列/JSON 帧有字节级测试覆盖（`tests/test_executor_hw.py`）。

## 关键配置（config.yaml）

- `vision.hsv`：光柱颜色（标定工具产出）；`vision.rechallenge`：再次挑战按钮（ROI/模板/颜色）
- `vision.minimap.enabled: true` + `marker_hsv_*`：小地图导航；`decision.nav_enabled: true`
- `decision.approach_mode`: `mouse_click`（点地面寻路，推荐）/ `keys`（纯键盘兜底）
- `decision.llm`：Ollama 顾问（默认关闭；`enabled: true, base_url, model, trigger_abandons: 3`）
- `capture.motion_trigger: true`：异动触发上传（对话定稿"发现异动才传图"，静止场景自动降频省带宽）
- `vision.item_list.enabled: true`：右侧 UI 掉落列表方案（点击条目走过去拾取）
- `executor.backend: pca9685`：PCA9685+舵机机械按压方案（对话最初的机械控制路线）
- `vision.beam_stable_frames: 3`：光柱稳定分级（连续 3 帧标记 stable，FSM 优先以稳定掉落光柱为目标，压低技能特效误走位）
- `server.event_log`：JSONL 审计日志（默认 `logs/events.jsonl`，5MB 滚动）
- OCR 白名单（可选）：`vision.ocr: tesseract` + `vision.whitelist: ["物品名"]`

## 仪表盘

`http://<小主机IP>:8765/` —— 实时检测标注流（光柱/A框/按钮/导航标记）、FSM 状态、
拾取/刷图/导航统计、事件流、▶⏸⛔♻ 控制。事件同时落盘 `logs/events.jsonl` 供事后审计。

## UI 场景自动化（交易行 / 处理装备 / 点天赋 / 换技能）

通用 UI 元素检测器 + 配方(Recipe)引擎（`vision/ui_panels.py` + `decision/recipe.py`）：

- **网格槽位检测**：交易物品格 / 装备格 / 天赋节点 / 技能槽通用（等尺寸矩形聚类 + 行优先排序）
- **高亮态识别**：选中/可买/可学/悬停槽位（亮度相对偏差，配色无关）
- **文本行检测**：价格行 / 物品名行（横向亮色行）
- **按钮模板匹配**：购买/装备/确认/排序按钮（全帧半分辨率搜索）
- **配方引擎**：步骤 = 期望视觉验证 + 动作 + 超时重试；激活后接管决策循环，完成/失败自动恢复 FSM
- 预置配方：`trade_buy`（交易行购买）/ `equip_best`（装备最优）/ `talent_alloc`（天赋连点）/ `skill_swap`（学技能换技能）

```bash
# 启动 UI 任务（先在 templates/ui/ 放面板锚点与按钮模板，从自己截图裁剪）
curl -X POST http://<小主机>:8765/api/task -H "X-Token: xxx" -d '{"recipe": "trade_buy"}'
curl http://<小主机>:8765/api/task/status -H "X-Token: xxx"   # 进度/失败原因
```

## 图片审查页（图上步骤下）

浏览所有已判定图片：**上面图片（带判定标注）+ 下面操作步骤（键盘/鼠标命令逐条列出）**。

```bash
.venv/Scripts/python run_server.py            # 启动服务端
# 浏览器打开 http://<IP>:8765/review （token 同仪表盘）
#   数据源：批量自检帧(100) / 视频回放帧（video_replay.py --frames-dir 导出）
#   过滤：全部 / 仅看有命令 / 仅看异常
```

- 批量自检帧：`tools/batch_validate.py` 生成报告后即可浏览
- 视频回放帧：`tools/video_replay.py 视频.mp4 --frames-dir assets/frames_replay/<名称>` 导出逐决策标注帧

## 对话需求追溯

DeepSeek 对话逐条提取的 **60 项需求**全部实现（58 直接 + 2 架构演进等价），逐条实现证据见
[docs/需求清单.md 需求追溯
需求清单.md](docs/需求清单.md)。补齐项：PaddleOCR 插件、POST /api/locate 位置查询、
右侧 UI 列表掉落方案、PCA9685 舵机机械执行器、PyInstaller 采集代理打包、加载/过场画面保持。

## 项目结构

```
autogame/
├── config.py            # pydantic 配置模型（YAML + 环境变量）
├── protocol.py          # WS 协议（二进制 JPEG 帧 + JSON 控制）
├── security.py          # token 常数时间比较 + 执行器限速
├── vision/
│   ├── detector.py      # 光柱 + A框(多尺度模板/镂空) + 再次挑战按钮 + 小地图 + 帧差
│   ├── ocr.py           # 可插拔 OCR（null / tesseract + 白名单模糊匹配）
│   └── pipeline.py      # 检测 → 白名单过滤 → SceneState
├── decision/
│   ├── fsm.py           # FSM：拾取闭环 + 刷图循环 + 导航（帧时间驱动，确定性可测）
│   └── llm_advisor.py   # Ollama 决策顾问（可选，白名单动作）
├── executor/            # dryrun / rp2040 串口 / rpi5_hidg（报文纯函数字节级可测）
├── server/              # FastAPI (WS 上行 + REST 控制 + MJPEG 仪表盘 + JSONL 审计)
├── agent/               # dxcam/mss 采集 + WS 客户端
└── sim/scene.py         # 合成 ARPG 世界（拾取/刷图/小地图全闭环联调）
tools/                   # grab_frame 截图 / calibrate_hsv 标定
firmware/rp2040/         # RP2040 组合 HID 固件 (.ino + platformio.ini)
scripts/                 # J4125 部署 / RPi5 gadget / benchmark
web/dashboard.html       # 监控仪表盘
tests/                   # 143 用例
assets/samples/          # 真实游戏截图素材（B站/17173 实机帧）
```

## 安全设计

- 所有 WS/REST 入口强制 token（常数时间比较）；无 token 连接 1008/401 拒绝（测试覆盖）。
- 帧只走局域网，无云依赖（Ollama 顾问也仅访问内网）；`.token` 隐藏属性/600 权限。
- systemd 单元 `ProtectSystem=full`、`NoNewPrivileges`。
- 执行器四层保险：token 控制 → FSM 铁律（不盲按/不卡死/picks≤taps）→ 限速 + panic → LLM 建议仅白名单动作可执行。
- 事件 JSONL 审计日志：每次按键/点击/决策都可事后追溯。

## 风险提示

游戏自动化可能违反目标游戏的服务条款，导致账号受限。请在单机/测试环境或自有服务器上验证，
自行承担使用风险。本框架不包含任何内存读写、进程注入或反检测对抗模块。
