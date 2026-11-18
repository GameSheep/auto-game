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

> 数字由 `scripts/benchmark.py` 自动生成，且**用 `config.yaml` 的部署配置测**
> （含 A框模板匹配 / 场景门 / 加载判定），不是"关掉模板的默认值"。

| 测项 | 数值 |
|---|---|
| 检测器单帧 @720p / 1080p / 1440p / 2160p | 32.6 / 41.6 / 41.8 / 43.3 ms |
| JPEG 编码 @1440p（代理侧） | ~10 ms |
| JPEG 解码 @1440p（服务端侧） | ~12 ms |
| **端到端单帧预算（解码+检测）** | **53 ms → 1Hz 决策 ≈ 5.3% 单核** |
| 内存（100帧处理驻留/峰值） | 6.7 / 16.6 MB |
| 带宽 @1440p q80, 2fps | ~234 KB/s |
| 真实屏幕实测（2560x1440, mss） | 检测中位 **37 ms** / p95 42 ms（`tools/e2e_real_screen.py`） |

**为什么比早期版本慢**：新增的 A框模板匹配是真实成本（早期只有形状启发式，
而它在真实帧上 11/11 全是误报）。模板匹配已做粗到精两级搜索：17×17 模板直接
全彩匹配要 75ms/尺度，改成灰度 1/2 分辨率粗搜（7ms/尺度）+ 全分辨率局部精修，
精度不变、耗时降一个数量级。1Hz 决策下 5.3% 单核，J4125 完全够用。

复现：`.venv/Scripts/python scripts/benchmark.py`（结果写 `benchmark_result.json`）。

## 核心铁律（代码层面强制，测试断言）

1. **绝不盲按拾取键**：只有检测到目标物品的 **A框**（近距交互提示）且 A框邻近当前目标时才按 A。
   - **A框结构校验**：真实 A 提示是"暗底 + 白色字形"。只按形状（亮+镂空）判定时，
     真实帧上 11 个候选 **全部是伤害数字/技能图标/Boss 特效**误报；加入
     "框内暗底 + 白色字形占比" 双判据后，100 张真实帧只剩 1 个命中，且经人工核验是真 A 提示。
   - 镂空率过滤、邻近目标约束、150ms 防闪烁确认。
   - A框闪现即消失且从未按过键 → 记为误检闪烁，**绝不算拾取成功**（`picks ≤ a_taps` 不变量）。
2. **只捡指定目标**：`vision.whitelist_colors: ["gold"]` 按稀有度过滤（不需要 OCR），
   `vision.whitelist: ["物品名"]` + `vision.ocr` 按名称过滤。两者都空时启动会**明确告警**
   —— 因为此时所有颜色的光柱都会被当成目标（C3 铁律实际上是关着的）。
3. **绝不原地卡死**：走位超时（默认 8s）放弃并拉黑区域 30s；帧差过低判定撞墙 →
   侧向脱困一次 → 仍卡则放弃。**长时间无进展则进自主恢复（见下节），绝不"停在那里"**。
4. **一键急停**：`POST /api/control/panic` 或仪表盘 ⛔ → 执行器立即停止一切输出。
5. **执行器限速**：最小动作间隔（自适应等待不丢命令）+ 每秒上限，所有后端统一生效。
6. **恢复动作同样受铁律约束**：自主恢复/模型建议**永远不能按拾取键**（
   `RecoveryLadder` 二次过滤 + `_emit_move_keys` 闸门），拾取只能由 A框检测裁决。

## 自主性：卡壳自愈 + 模型顾问（默认开启）

搬砖机器人真正的难点是"连续跑 8 小时不卡住"，而不是"能捡一次"。系统有三层防线：

**① 进度看门狗（规则，零依赖，永远可用）**

| 判定 | 含义 | 触发动作 |
|---|---|---|
| `frozen` | 画面完全不动（撞墙/失焦/游戏假死） | 进恢复阶梯 |
| `idle` | 画面在动但没有任何可交互目标（房间走空） | 自己去找（探索/导航/清怪） |
| `stalled` | 有目标但长时间零进展（走不到/来回绕） | 进恢复阶梯 |
| `gated` | 长时间不在战斗界面（弹窗/菜单/结算） | 按 ESC 自救（最多 6 次） |
| `loading` | 加载/黑屏过场 | 保持不动（正常等待） |

**② 分级恢复阶梯（由轻到重，永不放弃）**

```
重定位(清拉黑/丢目标) → 四向探路点击 → ESC 关弹窗 → 脱困键(空格闪避)
  → 重新聚焦(点角色) → 八向脱困 → 点『再次挑战』重开 → 视觉模型看画面
  → 文本模型换策略 → 新一轮（每轮降速，最多等 60s，不停止尝试）
```

每级默认 8s 升级；动作有最小间隔，避免"卡住→疯狂乱按"这种更糟的失败模式。
`max_cycles: 0` 表示无限重试；`pause_after_cycles: N` 是可选安全阀（默认不暂停）。

**③ 模型顾问（可选，看懂规则覆盖不到的局面）**

- **VLM（视觉，多模态）**：卡壳时把截图发给 Ollama / OpenAI 兼容服务，问"我在哪个界面、该往哪走"，
  返回结构化动作（`walk_to / key / esc / explore / reset_stage`）。
  安全边界：动作白名单 + 坐标钳制到画面内 + 置信度门槛 + 每分钟调用上限 + 连续失败熔断，
  **后台线程调用，绝不阻塞决策循环**。
- **LLM（文本，策略）**：看场景摘要判断"该不该换策略"，可下发的动作只有
  `wait / continue / explore / reset_stage / panic`（不能给坐标、不能按键）。

```yaml
decision:
  autonomy:
    enabled: true            # 看门狗 + 恢复阶梯
    frozen_window_s: 12      # 画面不动多久算卡住
    idle_window_s: 20        # 没有目标多久算空闲
    stall_window_s: 90       # 零进展多久算停滞
    auto_explore: true       # 空闲自动探索（点地面，不用键盘，避开拾取键冲突）
    auto_attack: true        # 空闲自动清怪
    max_cycles: 0            # 0=无限重试
  vlm:
    enabled: true
    provider: ollama         # ollama / openai（任意兼容 /v1 服务）
    model: qwen2.5vl:7b      # ollama pull qwen2.5vl:7b
    max_calls_per_min: 6     # 预算：一分钟最多问 6 次
  llm:
    enabled: true
    model: qwen2.5:7b
```

排障接口：`GET /api/autonomy` 直接看"卡在哪一级、模型说了什么"；仪表盘右侧"自主恢复"面板
显示判定/等级/循环/模型建议；`tools/preflight.py` 会检查模型是否已拉取。


## 拟人化行为：手抖 / 疲劳 / 走神（默认开启）

**先把话说清楚**：这一层只提高"纯统计模型判定"的门槛，改变不了"程序驱动"的本质。
无论加多少层随机，行为分布的形状都是我们自己设计的，不是由一个会累、会烦、会分心的人
自然长出来的。行为数据积累到足够量级并触发**人工复核**时，"摄像头看屏 → 程序判断 →
机械臂执行 → 无人值守"这套结构在人工眼里就是自动化系统。**它是降低短期统计风险的手段，
不是安全方案，封号风险不会归零。** 每加一层都在增加复杂度与调试成本 —— 收益递减，风险不归零。

代码位置：`autogame/humanize.py`（纯逻辑，无视觉/无业务依赖），分四层：

| 层 | 位置 | 做什么 | 为什么这么做 |
|---|---|---|---|
| **轨迹** | `executor/mouse_path` | WindMouse：目标当重力源吸附 + 风力扰动，两端慢中段快，落点差 1-3px，偶尔冲过头拉回，中途微小停顿 | 两点一线的匀速移动是最容易被统计识别的特征 |
| **疲劳** | `decision/humanize.fatigue` | 慢变量（连续操作累积、休息衰减）+ 快变量（唤醒度随机游走），同时漂移**均值**和**方差** | 固定区间随机的方差有上界，样本一多就收敛到那个盒子；真人不会 |
| **走神** | `decision/humanize.drift` | 泊松过程触发的非目标导向操作（默认只移动鼠标，不点击不按键） | 这类"噪音"不会被任务逻辑自然生成，恰恰是真人行为里最常见的部分 |
| **社交** | `decision/humanize.social` | 偶尔在公共频道发一句废话（**默认关闭**） | 打字前要先聚焦聊天框，焦点不对时每个字符都变成游戏热键 —— 台词含拾取键就等于盲按 |
| **偏好** | `decision/humanize.mood` | 耐心/贪心/新鲜感三个慢变量漂移 + softmax 选择（不是 argmax）+ 偶尔"懒得捡" | 原来 `_pick_target` 是 `stables[0]`：**永远选第一个**，这是确定性规则，比任何时序抖动都致命 |

关键设计取舍：

- **反应时间用对数正态，不用均匀分布**：人的反应时间右偏（偶尔"手慢"到区间外），
  均匀分布两头截断，样本一多统计上就不像人。
- **疲劳到阈值就真的停手**：休息窗口内**不输出任何命令**（`human_break` 事件可见），
  休息后重新计时看门狗，避免"刚睁眼就被停滞判据抓回去"。
- **走神有启动宽限期**：真人不会一上线就发呆，默认 60s 后才可能走神。
- **社交有硬闸门**：台词含移动/拾取键字符、或含执行器键盘映射不支持的字符 → **拒绝发送**，
  绝不"尽力发送"。这是唯一一处宁可少做也不做错的地方。
- **拟人化关闭时完全退化为旧行为**：`AUTOGAME_HUMANIZE=0` 或 `humanize.enabled: false`
  可全局关掉（单元测试套件默认关，保证断言确定性；拟人化本身由 `tests/test_humanize.py`
  显式打开并逐条断言）。

```yaml
executor:
  mouse_path:                # 手怎么动（输出层）
    enabled: true
    gravity: 9.0             # 重力吸附强度
    wind: 3.0                # 风力扰动（越大轨迹越飘）
    max_points: 40           # 轨迹点上限（USB HID/串口带宽预算）
    endpoint_jitter_px: 2    # 落点偏差
    overshoot_prob: 0.12     # 冲过头再拉回的概率

decision:
  humanize:                  # 人累不累（决策层）
    enabled: true
    reaction_ms: [180, 620]  # 看到A框 → 按下拾取键
    fatigue:
      per_action: 0.003      # 每次动作累积（≈280 次到满）
      break_threshold: 0.85  # 到阈值就停手
      break_seconds: [20, 90]
    drift:
      mean_interval_s: 45    # 平均多久走神一次（指数分布）
      noise_keys: []         # 走神时按的键（默认空：只动鼠标）
    social:
      enabled: false         # 默认关；开之前请读完上面的风险
      phrases: []            # 只用 ASCII，且不含 wasd/拾取键字符
    mood:                    # 选哪个目标（最硬的破绽在这里）
      patience: 0.6          # 耐心：累的时候自动下降 → 只捡近的
      greed: 0.5             # 贪心：高 → 愿意绕远捡贵的
      novelty: 0.5           # 新鲜感：讨厌连续同色/同位置
      temperature: 0.35      # softmax 温度：越大越随性
      skip_prob: 0.02        # 偶尔"懒得捡"，直接无视某个掉落
```

**偏好层解决的是另一个问题**：前面几层都在改"怎么动"，但**"选哪个目标"原本是确定性规则**
（`(stables or candidates)[0]`，永远第一个）。样本一多，这比任何抖动都更容易被识别。
现在用效用函数（距离/稀有度/新鲜感）+ softmax 加权随机 + 情绪漂移，
并且保留"偶尔挑次优""偶尔懒得捡"这两种真人特征（统计项 `off_pick` / `targets_ignored` 可观测）。

观测：`GET /api/autonomy` 的 `humanize` 字段（疲劳度/唤醒度/延时倍数/休息次数/走神次数），
事件流里能看到 `human_break` / `human_resume` / `drift` / `social_msg`。


## 动作链：一次交互不是一条命令

早期实现里 `mouse_click` 是一条**原子命令**：移动 → 点击，结束。然后 FSM 按固定 1 秒节拍
继续点同一落点。这在真实游戏里是错的 —— 点地面只是**发起**移动，后面还有一串游戏侧步骤：

```
① 光标移到目标（人类化轨迹：弧度/加减速/落点偏差）
② 悬停 30-110ms（先看清再点，不是到位就扣扳机）
③ 按下 → 保持 55-135ms → 抬起（真人不会 45ms 就松手）
④ 人物在游戏里自己走过去（1-3s，游戏侧耗时，程序只能等）
⑤ 走到一定距离内 → A框出现（游戏提示本身也有延迟）
⑥ 反应 180-620ms（对数正态）→ 按拾取键
⑦ A框消失 = 拾取成功（否则判误检/放弃）
```

**②③ 在执行器**（`executor.mouse_path.hover_ms / press_ms`，RP2040 固件已支持主机下发
`hold_ms`，**旧固件需要重新烧录**才能吃到可变按压时长，否则回落 45ms）。

**④⑤ 在状态机**（`decision.walk`）—— 这段最容易做错，所以拆成显式等待：

| 观察到的现象 | 判定 | 动作 |
|---|---|---|
| 画面在动（帧差 ≥ `motion_delta_thresh`） | 人还在走 | 什么都不做，**不重复点击** |
| 画面停下连续 `settle_frames` 帧 | 人停了 | 再给 A框 `settle_grace_s` 宽限（提示有延迟） |
| 宽限内 A框出现 | 到位了 | 结束等待 → 进入反应时间 → 按键 |
| 停下但 `target.distance` 缩短 ≥ `min_progress_px` | 走了一段没到 | 立刻再点一次继续接近（**不算撞墙**） |
| 停下且距离没变 | 撞墙/寻路失败 | 进撞墙判定（侧向脱困 → 放弃拉黑） |
| 等满 `budget_s` 还没到 | 没走到 | 重新发起走位 |

关键点：**用"距离是否真的缩短"而不是只看帧差判断有没有进展**。帧差既可能是人物在走，
也可能是背景特效；只看帧差会把"原地被特效糊住的卡死"误判成"正在移动"。

`budget_s` 与 `approach_timeout_s` 联动（每轮预算不超过剩余超时的 1/2），
避免"预算 3s + 超时 8s"变成只能试 2 次的尴尬局面。

### 按拾取键前的三道闸（铁律：绝不盲按）

帧是"过去"的信息（采集 → 编码 → 传输 → 检测，通常几百毫秒）。按 A 的那一刻，
人物可能早已离开拾取半径 —— 模拟器会记为 `blind_a_press`（铁律违规）。实测踩过三个坑：

| 坑 | 现象 | 修法 |
|---|---|---|
| 单帧误检/旧帧 | A框只出现一帧就按 → 按空 | `aframe_confirm_frames`：连续确认 N 帧才按 |
| 人物正在远离（路过/走过头） | 距离在变大还按 → 必然按空 | 用**当前帧**的 `target.distance` 与首帧比较，在远离就不按 |
| A框在但按不下去（光柱丢失检测/够不到） | 一直等 → 等到 A框消失 → 判定失败 | `aframe_retry_frames`：连续 N 帧按不下去就**再走一步**，别死等 |

第三坑的根因值得一提：光柱检测偶尔丢失，`_reacquire_target` 返回 `None`，
距离就只能用上一帧的旧值 —— 于是"在远离"判据永远成立，既不敢按也不肯重走，
形成死锁。修法是把**当前帧的对象**替换进 `self._target`，并给"按不下去"加超时重走。

```yaml
decision:
  walk:
    enabled: true
    budget_s: 3.0              # 单次点击后最多等多久
    motion_delta_thresh: 2.5   # 帧变化高于此值 = 还在走
    settle_frames: 2           # 连续 N 帧不动 = 停下
    settle_grace_s: 0.5        # 停下后给 A框的宽限
    min_progress_px: 12.0      # 至少靠近这么多像素才算有效走位
```

观测：事件流 `walk_end`（`reason` = `aframe` / `settled` / `budget` / `timeout`，
带 `waited` 与 `gained`），统计项 `walk_starts / walk_ends / walk_progress /
walk_no_progress / walk_settles`。探索（无目标时点地面）走同一条链，不再每秒重发指令。


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
- 场景门：**反例模板优先 → 技能栏模板 → 都没命中就拦下**（未标定就不动手）。
  战斗 HUD 放行；关卡选择/角色面板由反例模板拦；未标定的 HUD 一律拦，不再拿形状猜
- HUD 排除区：小地图/任务/聊天/技能栏区域强制排除（技能栏上的拾取键提示不会变成假 A框）
- 白芯校验：掉落光柱有白热芯；Boss 触手/技能光束是纯饱和色 → 全部拒绝（实测 Boss 触手 4 帧零误报、零按 A）
- 时域确认：光柱需连续 2 帧出现（单帧特效闪烁不触发）
- 拾取复核冷却：按 A 后 500ms 内禁止再按（真实素材回放中发现的连按捡垃圾时序 bug）
- 自适应标定 auto_hsv_ranges：从任意实机帧自动推断高亮色簇（无需手工标定即可工作）
- 暗图真实疑似光柱 → 仅输出 `MOUSE_CLICK left (x,y)` 走位命令，无 A框 绝不按 A
- 视频回放支持 `--csv` 导出逐 tick 判定时间线；100 图批量自检同步导出 `assets/batch_list.csv`

**100 张真实帧批量自检**（`tools/batch_validate.py`，10 条自动判定规则 R1-R10，
用 `config.yaml` 的部署配置跑）：
**PASS 100 / WARN 0 / FAIL 0**，零盲按、确定性验证通过、平均 120ms/图。
过程中借真实数据修复：A框色相无关检测（粉/紫稀有度框漏检）、Boss 血条区误检、
场景门半分辨率性能、**A框在真实帧上 11/11 全误报**（伤害数字/技能图标被当成拾取提示）。

## 真实屏幕端到端验证（真机链路，非模拟画面）

模拟器和单元测试都在"合成画面"上跑，真实屏幕引入的变量（多显示器、DPI 缩放、
窗口失焦、真实 UI 密度）完全没覆盖。这个工具把生产代码路径搬进真机链路：

```bash
.venv/Scripts/python tools/e2e_real_screen.py --seconds 12
```

```
服务端已启动 http://127.0.0.1:5523（dryrun 执行器，零真实输入）
采集后端: MsseCapture
真实帧: 2560x1440  抓取 55.1ms
检测耗时: 中位 37.0ms  p95 42.1ms
真实屏幕链路: 26 帧 / 10s  (有效 2.6 fps)
场景门: in_game=False → 命令 0 条 (非游戏画面应为 0)
铁律: blind_tap_violations=0
结论: PASS（真实屏幕链路可用）
```

证据落盘 `assets/real_screen_e2e.json`。链路是：真实屏幕 → JPEG → WebSocket → 服务端
→ 检测 → FSM 决策 → dryrun 执行器。**已验证到"真实采集+真实检测+真实决策"，
未验证的是"真实游戏画面 + 真实硬件执行器"**（见 `docs/完成度审查.md` 的剩余缺口）。

## 真实录像语料验证（别人的刷图录像 → 逐帧完整指令）

模拟器和静态截图都缺一样东西：**时序**。真实刷图是连续操作，卡壳、走位、
拾取窗口期都只在连续帧里才出现。所以用公开的刷图录像当"离线现场"：

```bash
# 1) 抽帧 + 跑检测/决策（用 config.yaml 的部署配置，DryRun 执行器零输出）
.venv/Scripts/python tools/video_batch.py 录像.mp4 --hz 2 --max-min 5
#    → assets/frames_replay/<录像名>/{frames/*.jpg, manifest.json, summary.json}

# 2) 汇总成离线审查页（双击即开，不需要起服务/填 token）
.venv/Scripts/python tools/review_offline.py assets/frames_replay
#    → assets/frames_replay/report.html
```

每张截图旁边就是**该帧的完整指令**（op/参数/协议文本）+ 意图 + 检出明细，
叠加层用 DOM 按 manifest 坐标画（不烧字，坐标可对着原始帧核对），
顶部可按"有指令/无指令/违规/有光柱/有A框/门禁拦截"筛选。

本轮两段真实录像（暗黑4 梦魇地下城 300s / 火炬之光手游 162s，2Hz → **906 帧**）的结论：

| 项 | 结果 |
|---|---|
| 场景门 | 13 帧被**反例模板**拦下且零命令：暗黑4 角色面板 7 帧、火炬之光「异界」星图 6 帧 |
| 铁律（绝不盲按拾取键） | **0 违规**（素材里没有 A 框 → 全程 0 次按 A） |
| 命令 | **288 条**（走位点击为主），坐标全部在画面内，单帧 ≤ 6 条 |
| 光柱检测 | 高宽比下限 1.1 → 2.5 后：暗黑4 **0/580** 帧误报，火炬 233/326 → **140/326**（剩下的多是技能光效，见下） |
| 怪物检测 | 仍偏高：暗黑4 341/580、火炬 254/326（会动的技能特效也算"活物"） |

> 上表是 `--gate-unmatched pass`（离线分析模式）跑出来的：这两段录像的 HUD 都不在
> 仓库现有模板覆盖范围内，生产默认会 fail-closed 拦下（见下面"场景门"一节）。
> 审查页页头有黄条标注这个模式，别把这里的指令当成线上行为。

借这两段录像修掉的真 bug（都有回归测试）：
1. `_detect_monsters` 文档说"排除 HUD 排除区"，代码**根本没排除** → 手游技能轮盘
   被当成怪物点掉。现在先抹 HUD 再检测，并与掉落标签/光柱互斥。
2. 怪物只有形状判据 → 静态图标也算怪。加时域确认（要动/要变尺寸才算活物）。
3. 恢复阶梯走到 `escape` 级时**在战斗界面按 ESC**（暗黑4 实战帧上连按 6 次：
   等于自己打开暂停菜单）。现在只在门禁判定"已不在战斗界面"时才按。
4. 模拟器 `_advance` 在 dt≤0 时直接 return → 同一 tick 内二次推进被吞，
   "点了再次挑战却没刷怪"（e2e 刷图循环用例偶发失败）。
5. `video_replay.py` 用默认 `VisionConfig` 而不是 `config.yaml` → 验证的不是线上参数。

**必须说清的局限**：光柱误报的根因不是形状判据失效，而是**标定前提不成立** ——
`auto_hsv_ranges` 把"画面里最亮最饱和的颜色"当光柱色，换一个游戏（技能是粉色光束）
它就必然把技能当掉落。真正的解法是在**你自己的游戏画面上**用配置页的相机标定
（`/api/calibrate`）标一次 HSV 与 A 框模板；这两段录像验证的是**决策层**
（门禁/铁律/命令生成/恢复策略），不是感知阈值。

## 场景门：只信证据，不信「大概像」

门禁是这套系统的安全底座：它说「现在不是战斗界面」，FSM 就一条命令都不发。判定顺序：

```
1. 反例模板命中（templates/neg_*.png，≥0.80） → 非战斗（确定的菜单/面板，任何时候都拦）
2. 技能栏模板命中（templates/skillbar*.png，≥0.60） → 战斗
3. 都没命中 → fail_closed=true 时拦下，原因记为 unmatched（= 你的游戏还没标定）
```

**为什么砍掉两条兜底判据**：`q_letter.png`（键位字母）实测是一块技能图标的角落，
对 8 张真实截图里 7 张给 0.77~0.99；红血球色斑在星图/装备面板上到处命中。
这类「给不出证据却敢放行」的判据，正是「关卡选择页显示接近掉落光柱」那次事故的根因。

标定一个游戏只要两步（各 1 分钟，工具会当场给分数）：

```bash
# 正向：从战斗截图裁一块「只有战斗才有」的 HUD 部件（血球／技能栏外框／摇杆环）
.venv/Scripts/python tools/make_template.py --src 战斗.png --rect 262,450,344,528 \
    --out templates/skillbar_mygame.png --check "assets/ref/*.jpg"
# 反例：从菜单截图裁一块固定的 UI（标题栏／标签条／按钮行）
.venv/Scripts/python tools/make_template.py --src 菜单.png --rect 598,50,960,96 \
    --out templates/neg_mygame_panel.png --against "assets/ref/*.jpg"
```

正向模板要在**所有**战斗图上 ≥0.6；反例要在菜单上 ≥0.8、在战斗图上明显更低。
做不到就换个区域——**不要靠调阈值把它调过去**（这次事故就是这么来的）。

## 参考图台账：把「人给的答案」变成回归用例

`assets/ref/labels.json` 是人工标定过的正确答案（截图 + 场景 + 期望检出），
`tools/ref_check.py` 逐字段比对，`tests/test_ref_labels.py` 把 `strict: true` 变成门槛。

```bash
.venv/Scripts/python tools/ref_check.py -v
# [OK  ] tli_nether_map_01.jpg   异界 · 关卡选择（星图）
#        门禁=negative:neg_tli_nether_map.png (模板 0.00 / 反例 1.00)  光柱=0 ... 指令=0
# [GAP ] tli_combat_01.jpg       战斗 · 异界深处  ✗ in_game: 标答=True 实测=False
```

`GAP` 不是通过，是「还没标定」——补齐方式写在条目备注里。术语、判据、
「像光柱但不是」的清单、截图规范见 [docs/游戏术语与参考图.md](docs/游戏术语与参考图.md)，
换机器接手见 [docs/交接-2026-11-17.md](docs/交接-2026-11-17.md)。

## 换游戏：profiles —— 引擎不动，只换标定

`autogame/` 里的检测、决策、走位、执行**不区分游戏**。区分游戏的只有
`profiles/<id>/` 里的三样东西：战斗 HUD 长什么样、哪些界面绝不能动手、目标物在屏幕上是什么样。

```
profiles/
  README.md            # 三步操作手册（放什么 → 我算什么 → 怎么验）
  _template/           # 新游戏脚手架，cp -r 改个名就能用
  generic_1280x720/    # 仓库默认（复用根 templates/ 与 assets/ref/labels.json）
  tli_phone960/        # 火炬之光手机端 960x432，带完整标定范例
    profile.yaml       # 元信息 + 对该游戏的 config 覆盖
    templates/         # gate_pos_*.png 正向锚点 / gate_neg_*.png 反例 / _index.json
    labels.json        # 本游戏的人工标答台账
    inbox/  refs/      # 待标定截图 / 已标定截图
```

```bash
# 探针：逐张出分数 + 列出所有「像光柱」的色块（带坐标和高宽比）+ 待办清单
python tools/profile_init.py --profile tli_phone960

# 按台账核对（strict 不符即 exit 1，CI 也在跑）
python tools/ref_check.py --profile tli_phone960

# 生产
python run_server.py --config config.yaml --profile tli_phone960
```

**分工是硬性的**：界面的类别（这是战斗/这是星图/这是背包）由人给，工具不猜；
"拿哪块像素当证据"由我出候选，但必须过分数裁决——正向模板要在**所有**战斗帧
≥0.6、所有菜单帧明显更低，反例反过来。过不了就换一块，不调阈值硬凑。

真踩过的三个坑（都写进了测试）：

| 坑 | 现象 | 现在怎么防 |
|---|---|---|
| 默认 ROI 是"底部 35% 全宽" | 手机端技能在右下角，不传 `--roi` 分数从 0.99 掉到 0.3，像"模板失效" | `profile.yaml` 显式给 `scene_gate.roi` |
| 把 `template_base_width` 改成录制分辨率 | 模板是 1280 归一化存储的，改成 960 后反例从 1.00 掉到 0.41，菜单拦不住 | `tests/test_profile.py` 断言它恒为 1280 |
| 先做正例再做反例 | 门禁反例优先，正例先上等于给菜单发通行证 | `profile_init` 的待办清单把反例排第一 |

## 快速开始（本机模拟闭环，无需游戏）

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Windows
# Linux: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 0) 部署前自检（不发任何游戏按键）：依赖/采集/模板/执行器/服务端逐项检查
.venv/Scripts/python tools/preflight.py

# 终端1: 服务端 + 内置模拟器（完整刷图循环自动运行）
.venv/Scripts/python run_server.py --sim

# 浏览器 http://127.0.0.1:8765/ ，token 见 .token 文件
# 点 ▶ 启动 → 看拾取/刷图/导航统计与事件流
# 右侧「自主恢复」面板：卡壳判定 / 恢复等级 / 循环轮数 / 模型建议

# 1) 卡壳自愈演示（人为制造卡死 → 看它自己脱困 → 恢复拾取）
.venv/Scripts/python tools/stuck_demo.py
```

运行测试（**366 个用例**，含端到端闭环/真实帧回归/A框精度回归/自主恢复与卡壳自愈/
模型顾问预算熔断/监督层权力边界/固件协议一致性/UI 场景任务/审查页/fuzz 加固）：

```bash
.venv/Scripts/python -m pytest tests/ -q                  # 本地：跑测试 + 出覆盖率
.venv/Scripts/python -m pytest tests/ -q --cov-fail-under=78  # 同 CI：低于门槛失败
```

CI：`.github/workflows/ci.yml`（py3.11/3.13 跑测试 + preflight），依赖锁 `requirements.lock`。


## LLM 监督层：让模型盯结果，不盯画面

`autogame/decision/supervisor.py`。快通道（OpenCV + 状态机）每帧都跑，但它只能回答
"有没有光柱 / A框在不在"，回答不了**"我刚做的这一串动作到底有没有让局面变好"**。
监督层补的就是这一句。

### 它为什么不能当驾驶员

把 1Hz 的控制权直接交给大模型是错的，三个原因：

| 问题 | 现实 |
|---|---|
| 延迟 | 本地 7B 一次 1~3s，云端 2~15s。每帧都问，1Hz 会变成 0.3Hz |
| 成本 | 一小时 3600 帧 |
| **方差** | 视觉阈值确定、可回归；LLM 每帧给的答案会飘。拿飘的东西逐帧控制，比开环更差 |

所以它是**断路器 + 审计员**：只踩刹车，不打方向。

### 权力表（这是安全边界，不是建议）

| 结论 | 能做什么 | 触发门槛 |
|---|---|---|
| `allow` | 什么都不做（绝大多数） | — |
| `throttle` | 走位/点击间隔 ×2 | 置信度 ≥0.60 |
| `hold` | 本帧不发命令，**最长 6s 自动放行** | 置信度 ≥0.60 |
| `takeover` | 只许 `esc` / `wait` / `reset_stage` 三选一 | 置信度 ≥0.80 + evidence 必须复述输入里的数字 |
| `panic` | 停车交人工 | ≥0.80 + 证据 + **连续 2 次一致** |

`stop_all` 永远放行（想停必须停得下来）。接管动作白名单外的、带坐标的一律降级为 `hold`。

### "结果是否正向"怎么算：预测-实测对账

不问模型"你觉得对吗"——那是让它编。改成让状态机在关键动作后登记一条预期，到点用实测值结算：

| 动作 | 登记的预期 | 实测 | 判定 |
|---|---|---|---|
| 点地面走位 | 距离应缩短 | 距离 237→241 | **miss** |
| 按拾取键 | A框应消失 | A框仍在 | **miss** |
| 按 ESC 自救 | 应回到战斗界面 | 仍不在战斗 | **miss** |

喂给模型的是一串这样的对账结果 + 健康指标（拾取增量、空转时长、意图抖动次数），
不是截图。**文本就够，也更快更便宜。**

### 三档模式，默认 `shadow`

```yaml
decision:
  supervisor:
    mode: shadow     # off / shadow（只观察）/ enforce（真生效）
```

`shadow` 下照常判断、照常记账、照常出结论，但**一条命令都不拦**。
跑几小时看 `logs/supervisor.jsonl`（每条裁决都落盘：输入指标、模型结论、证据），
确认它想拦的每条都站得住，再切 `enforce`。

**没有影子期的监督层不该上线**——它会成为新的故障源。

### 什么时候才会去问模型

不是每帧都问。本地体检每 5s 一次（零成本），只有命中才发起调用：

- 预测连续落空 ≥3 次（走位没走近 / 按了没捡到）
- 在动但 60s 没捡到东西（最危险的假动作：命令在发、界面没变）
- 未识别画面持续 45s
- 每分钟意图切换 ≥6 次（决策在抖）

预算：每分钟 ≤3 次、最小间隔 20s、连续失败 3 次熔断 5 分钟。失败静默降级为纯规则。

### 不用装模型也能验证

```bash
py -3.12 tools/supervisor_demo.py            # 跑一遍"假走位"场景，看它怎么发现并降级
py -3.12 tools/supervisor_demo.py --enforce  # 切到真的拦，对比命令数
```

### 四个层别搞混

| 层 | 看什么 | 什么时候问 | 权力 | 成本 |
|---|---|---|---|---|
| `llm_advisor` | 摘要 | 连续放弃 ≥3 次 | 只能建议 | 低 |
| `vlm` | 截图 | 卡壳恢复阶梯轮到它 | 只能建议 | 高 |
| `supervisor` | **预测-实测对账** | 体检发现异常 | 只能刹车（hold/panic） | 中 |
| `arbiter` | **局面 + 动作清单** | 规则答不上来时 | 能发动作（但从清单里选） | 低~中 |

前两个是顾问（说归说，不执行），后两个有实权，所以都有 `shadow` 档和影子日志。
`supervisor` 管"我做的事有没有用"，`arbiter` 管"现在该做什么"——它们是正交的：
一个在动作之后对账，一个在动作之前选择。


## 三级裁决：认得就自己做，拿不准才问模型

用户的原话是这三句，代码就是照这个写的：

> 先分析页面出现的东西，如果有我之前截图和文字介绍的，那么就可以直接处理；
> 稍微有点不明白，这才需要 laya 去判断，问 laya 的时候也是提前准备好系统已经支持的行为，
> 如果行为不满足选择 others，这时候就要请求大模型了；
> 问的时候人物也不是站着一动不动的等着，该干啥干啥，别被怪打死了。
> 然后这次行为要作为经验存储出来，和我最开始整理的游戏内容放一起。

### 三级分别是什么

| 级 | 触发 | 输出 | 模型调用 |
|---|---|---|---|
| **L0 经验直出** | 同局面指纹 + `confirmed` 经验 | 直接执行 | **0** |
| **L1 laya** | 规则答不上来（闲置/未标定/连续落空） | 从动作清单**单选**一个 | 1 次小模型 |
| **L2 大模型** | L1 返回 `others` / 选了清单外的 / 置信度不够 | 最多 5 步计划 | 1 次大模型 |

L1 的输入里有**系统已支持的行为清单**（`autogame/decision/actions.py`），
清单里每一条都对应 FSM 一个真实方法。所以"模型选了一个程序做不到的动作"这种事
在结构上就不会发生；即使发生了，`action_table()` 找不到实现会直接拒执行并留痕。

### 三条硬约束（写在代码里，不是文档里）

1. **不许指路**。参数里出现 `x` / `y` / `px` / `pos` 一律拒绝。
   去哪由像素证据决定，模型只决定"做什么"。
2. **不执行残缺计划**。L2 的计划里哪怕只有一步做不到，整条丢弃。
   前一步动了、后一步没动，局面会比现在更糟。
3. **经验先验证再生效**。见下。

### 经验为什么不能记了就用

```
candidate  --对账有效×2-->  confirmed  --再失败×2-->  rejected
candidate  --对账无效×2-->  rejected（保留，作为"别这么做"的反例）
```

只有 `confirmed` 能在 L0 直出。没有这条，模型第一次给出的"看起来很有道理"的做法，
第二天就会变成每帧执行的"真理"，而且你查不出来是谁决定的。

对账用的是实测数字，不是模型自评：`pickup` → 拾取数涨了没；`attack_nearest` → 怪少了没；
`esc` → 回到战斗界面没。判不出来（比如 `explore` 走了多远）就记 `unknown`，不硬判——
宁可不升级，也不要让噪声把一条错经验推成 confirmed。

### 等下的时候人在干什么

`arbiter.tick()` **永远不阻塞**。请求在后台线程，主循环该跑什么跑什么。
如果 FSM 是真的闲着（`idle_trigger_s` 内没发过命令）而且有怪，会插一个保命动作
（`keepalive_after_s` 之后，`keepalive_interval_s` 限频）——"先砍一刀再等模型"。

### 人能写进去的部分

`profiles/<id>/knowledge/game.md`。它会**原样**进入模型上下文，并且明确告诉模型
"与你的经验冲突时以此为准"。这是你唯一能直接给模型下达指令的入口。

写完的 `game.md` 加上 `experiences.jsonl` 就构成这个游戏的"依据库"，
两者在同一个目录 —— 你要"和最开始整理的游戏内容放一起"，就是字面意思上的放一起。

### 不用装模型也能验证

```bash
py -3.12 tools/arbiter_demo.py               # 三个场景全跑
py -3.12 tools/arbiter_demo.py --scenario c  # 只看"经验从无到有"那一段
```

场景 C 会演完整条链：第一次不认识 → laya 说 others → 大模型给计划 → 执行 → 对账有效
→ 第二次遇到同局面时 **0 次模型调用**，直接执行。

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

# ③ A框模板：从实机截图自动/手动裁出（仓库已带一份，换版本要重裁）
.venv/Scripts/python tools/auto_template.py --image shot.png --kind aframe \
    --rect 845,447,20,20        # 不给 --rect 会尝试自动定位
```

> `templates/aframe.png` 是**标定资产，必须入库**（历史踩坑：模板没入库 →
> 现场只剩形状启发式 → 真实帧 11/11 误报）。同理 `skillbar*.png` 场景门锚点。

### 3.5 采集代理打包成 EXE（产品化交付）

```bash
scripts\build_agent_exe.bat        # 或: .venv/Scripts/python -m PyInstaller packaging/agent.spec
dist\autogame-agent.exe --server http://<小主机IP>:8765 --token <token>
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
- **稀有度白名单（不需要 OCR）**：`vision.whitelist_colors: ["gold"]` —— 只捡指定颜色；
  空=全部颜色都捡（启动会告警提示"白名单未配置"）
- `vision.aframe.template`：A框模板（已随仓库分发）+ `verify_dark_core` / `glyph_white_only`
  结构校验（拒绝伤害数字与彩色图标误报）

## 仪表盘

`http://<小主机IP>:8765/` —— 实时检测标注流（光柱/A框/按钮/导航标记）、FSM 状态、
拾取/刷图/导航统计、事件流、▶⏸⛔♻ 控制。事件同时落盘 `logs/events.jsonl` 供事后审计。

## 控制台：配置 / 标定 / 试跑 / 告警

`http://<小主机IP>:8765/settings`（仪表盘右上角「⚙️ 设置」）。**本地默认不需要 token**
——局域网自用还要粘一串密钥纯属给自己添堵；要暴露到不可信网络再开 `server.require_token: true`。

三个页签：

### ① 控制台（运行 + 画面 + 标定 + 试跑）

左边实时画面 + ▶⏸⛔♻ 控制，中间运行状态，右边标定评分。**画面和标定放一起**，
因为调相机的时候眼睛必须盯着画面、同时看分数变没变。

标定给的是**可量化的"差多少"**，不是"检不到"：

| 指标 | 低了说明什么 | 页面直接告诉你 |
|---|---|---|
| 亮度 | <55 HSV 阈值开始漏检；>190 高光糊成白 | 调显示器亮度/曝光 |
| 清晰度（Laplacian） | <60 对焦/距离不对 | **先解决对焦**，再谈标定 |
| 场景门最高分 | 低于阈值=没对准游戏画面 | 挪相机或重做技能栏模板 |
| A框模板最高分 | 低于 `template_threshold` | 分辨率变了，重裁模板 |
| 检出光柱 | 游戏里有却 0 个 | 点「🎨 采样 HSV」 |

**试跑**（这是"我到底配对了没"最直接的答案）：

- **🧪 试跑一步**：用一份影子状态机在当前帧上跑一拍，输出「接下来几步会做什么」，
  每步都带 **RP2040 JSON 行** 和 **原始 HID 报文（十六进制）**。只算不按，绝不影响线上状态机。
- **⚡ 真发这些命令**：把上面那几条真的发给执行器（RP2040 串口 / hidg）。
- **手动发一条 HID 指令**：选 op（move/click/key_tap/key_down/key_up/stop_all）+ 参数，
  点发送，立刻看到 JSON 与 HID 字节。手动按拾取键不会被拦（你可能就是在测这个键通不通），
  但会写一条 `manual_pickup_test` 事件留痕。

### ② 拟人度：三档一键切换 + 细调

| 预设 | 内容 | 代价 |
|---|---|---|
| **全都不要（稳定快速）** | 随机全关：轨迹直连、无抖动、无疲劳、无走神、无偏好随机 | 统计特征非常"机器" |
| **放飞自我（绝对像人）** | 反应 220-900ms、走位预算 4s、风力 4.5、落点偏差 3px、懒得捡 5%、会累（30-150s 休息）、会走神、进关卡前会犹豫 | 吞吐降 20-40%，难复现 |
| **自定义** | 12 个滑块，保存即生效 | 组合没人替你验证过 |

预设只改拟人度字段，不碰视觉阈值/执行器后端 —— 否则点一下"放飞自我"把 HSV 改掉，
直接一条都捡不到，属于最糟的体验。

### ③ 告警：Bark 优先

**Bark（iOS）是默认推荐**：手机装 App → 复制 key → 填进页面 → 点「发测试告警」立刻收到。
不用注册服务、不用配 SMTP、不用建机器人。

其它通道：通用 Webhook（JSON）/ 企业微信机器人 / 钉钉 / Slack / SMTP 邮件（密码读环境变量）。

规则可勾选，指标里有几个是专门为"程序判断不了的事"设计的：

| 指标 | 用途 |
|---|---|
| `rechallenges` / `runtime_s` | 刷够 N 次 / 跑够 N 小时提醒 |
| `no_pick_with_targets_s` | **有目标却一直捡不到** → 疑似背包满 |
| `manual_needed` | 规则/模型解决不掉、已自动停止 → 需要你看一眼 |
| `blind_taps` | 盲按拾取键（铁律违规），一旦非 0 立刻告警 |
| `dropped_commands` | 执行器丢命令 |

### 配置怎么落地（为什么不是直接改 config.yaml）

页面上的修改**原地生效**（不替换配置对象 —— 否则 FSM 还在读旧对象，页面显示"已生效"而程序纹丝不动），
写入 `config.runtime.yaml` 覆盖文件，加载时叠在 `config.yaml` 之上：

```
config.yaml              # 手写、带注释 —— 程序永不回写，注释不会被碾平
config.runtime.yaml      # 只存页面改过的项，可读、可删（已 gitignore）
```

校验走 pydantic：任何一项非法则**整批不生效**，错误带字段路径；需要重启的字段会明确列出。

### 视觉看门狗：看不到游戏就停

- **完全没新帧**（相机被关/代理掉线）→ `no_frame_stop_s`（默认 30s）自动停。没决策依据，多等一秒都是白跑。
- **有帧但不在游戏里**（菜单/相机被挪开）→ `no_game_stop_s`（默认 45s）。必须比 autonomy 的
  `gate_window_s`（25s）长，否则会在 ESC 自救进行到一半时把程序掐掉。

`auto_stop: false` 时只告警不动手。停完状态切 PAUSED、写事件 `vision_lost`，页面顶部红块说明原因。

## UI 场景自动化（交易行 / 处理装备 / 点天赋 / 换技能）

通用 UI 元素检测器 + 配方(Recipe)引擎（`vision/ui_panels.py` + `decision/recipe.py`）：

- **网格槽位检测**：交易物品格 / 装备格 / 天赋节点 / 技能槽通用（等尺寸矩形聚类 + 行优先排序）
- **高亮态识别**：选中/可买/可学/悬停槽位（亮度相对偏差，配色无关）
- **文本行检测**：价格行 / 物品名行（横向亮色行）
- **按钮模板匹配**：购买/装备/确认/排序按钮（全帧半分辨率搜索）
- **配方引擎**：步骤 = 期望视觉验证 + 动作 + 超时重试；激活后接管决策循环，完成/失败自动恢复 FSM
- 预置配方：`trade_buy`（交易行购买）/ `equip_best`（装备最优）/ `talent_alloc`（天赋连点）/ `skill_swap`（学技能换技能）

```bash
# ① 看配方需要哪些模板
.venv/Scripts/python tools/crop_template.py --list all

# ② 从自己的截图裁剪（坐标=截图像素），带 4 倍预览图人工确认
.venv/Scripts/python tools/crop_template.py --image shot.png --rect 950,110,120,40 \
    --out templates/ui/sort.png --preview assets/tpl_preview.png

# ③ 启动任务（缺模板会 409 拒绝启动，不会静默空转）
curl -X POST http://<小主机>:8765/api/task -H "X-Token: xxx" -d '{"recipe": "trade_buy"}'
curl http://<小主机>:8765/api/task/status -H "X-Token: xxx"   # 进度/失败原因
```

> **缺模板 = 409 快速失败**：历史上 `templates/ui/` 不存在时，`trade_buy` 5 步全部
> `skip_panel_check`/`skip_button`，接口返回 `ok:true`、任务"跑完"却一条命令没发。
> 现在启动前预检，缺哪个文件、怎么裁都写在响应里。

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
[docs/需求清单.md](docs/需求清单.md)。补齐项：PaddleOCR 插件、POST /api/locate 位置查询、
右侧 UI 列表掉落方案、PCA9685 舵机机械执行器、PyInstaller 采集代理打包、加载/过场画面保持。

需求之外的验证驱动迭代（H1-H22，含本轮修复）同样逐条记录在该文件末尾。

**完成度审查与剩余缺口**见 [docs/完成度审查.md](docs/完成度审查.md)：
已补标定资产、A框精度、稀有度白名单、UI 预检、硬件自检、真实屏幕链路、git/CI/覆盖率门槛；
**仍未验证的只有"真实游戏画面 + 真实硬件执行器"这一段**（需要实机与硬件）。

## 项目结构

```
autogame/
├── config.py            # pydantic 配置模型（YAML + 环境变量）
├── protocol.py          # WS 协议（二进制 JPEG 帧 + JSON 控制）
├── security.py          # token 常数时间比较 + 执行器限速
├── humanize.py          # 拟人化行为层：WindMouse轨迹 / 疲劳 / 走神 / 社交 / 偏好（纯逻辑可测）
├── presets.py           # 拟人度预设（稳定快速 / 绝对像人 / 自定义）
├── vision/
│   ├── detector.py      # 光柱 + A框(模板粗到精 + 暗底白字校验) + 再挑战 + 小地图 + 帧差
│   ├── ocr.py           # 可插拔 OCR（null / tesseract / paddle + 白名单模糊匹配）
│   ├── pipeline.py      # 检测 → 稀有度白名单 → OCR 白名单 → SceneState
│   ├── ui_panels.py     # UI 网格/高亮/文本行/按钮检测
│   └── scene_classifier.py  # 关卡内/菜单分类
├── decision/
│   ├── fsm.py           # FSM：拾取闭环 + 刷图循环 + 导航 + 自主恢复（帧时间驱动）
│   ├── recovery.py      # 进度看门狗 + 分级恢复阶梯（永不放弃，绝不按拾取键）
│   ├── vlm.py           # 视觉模型顾问（多模态，白名单/钳制/预算/熔断）
│   ├── recipe.py        # UI 配方引擎 + 启动前模板预检
│   └── llm_advisor.py   # 文本策略顾问（Ollama，白名单动作）
├── executor/            # dryrun / rp2040 / rpi5_hidg / pca9685（报文纯函数字节级可测）
├── server/              # FastAPI (WS 上行 + REST 控制 + MJPEG 仪表盘 + JSONL 审计)
│   ├── settings.py      # 配置热更新（原地生效 + 覆盖文件持久化）
│   ├── calib.py         # 相机标定反馈（亮度/清晰度/场景门/A框分数 + 建议）
│   └── notify.py        # 告警规则引擎（webhook / SMTP，去重限流）
├── agent/               # dxcam/mss 采集 + WS 客户端
└── sim/scene.py         # 合成 ARPG 世界（拾取/刷图/小地图全闭环联调）
tools/                   # preflight 自检 / stuck_demo 卡壳自愈演示 / e2e_real_screen
                         # / auto_template / crop_template / grab_frame / calibrate_hsv
                         # / batch_validate / video_replay / soak
firmware/rp2040/         # RP2040 组合 HID 固件 (.ino + platformio.ini + README)
scripts/                 # J4125 部署 / RPi5 gadget / benchmark / 打包 EXE
web/                     # 监控仪表盘 + 图片审查页
tests/                   # 286 用例（fixtures 内含 9 张真实帧，12MB）
templates/               # 标定资产：aframe.png / skillbar*.png / ui/（必须入库）
assets/samples/          # 真实游戏截图素材（B站/17173 实机帧）
.github/workflows/ci.yml # CI：pytest（覆盖率门槛 78%）+ preflight
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
