# profiles —— 一个游戏一个目录，引擎代码不动

这个目录就是"换游戏"的全部工作量。`autogame/` 里的检测、决策、走位、执行不区分游戏；
区分游戏的只有这里的三样东西：**画面长什么样、哪些界面绝不能动手、目标物在屏幕上是什么样子**。

## 你要做的三步

### 1. 丢截图

```
profiles/<游戏id>/inbox/
    battle_01.png       # 正常战斗，HUD 完整
    battle_02.png       # 战斗，掉了一个品质好的东西（有光柱）
    map_select_01.png   # 关卡选择/星图/世界地图
    backpack_01.png     # 背包/角色面板/商店/结算
```

同一个游戏用同一个目录，**每种界面至少 2 张**（不同地图/不同光效，防止我按一张图过拟合）。

### 2. 一句话告诉我"这张该判成什么"

在 `deliveries.md` 里写，或者直接跟我说。格式就一句：

```
battle_01.png     战斗，没有掉落物 → 应该能打怪，但没有"接近光柱"这类指令
battle_02.png     战斗，画面右侧偏下有个金色细光柱 → 应该朝它走
map_select_01.png 异界星图，7 个发光地图节点 → 绝对不能动，那不是光柱
backpack_01.png   背包界面 → 绝对不能点
```

**没有这句话，截图只是截图。** 有这句话，它才变成回归用例——以后我改任何阈值，
`tools/ref_check.py` 都会拿它验一遍，不对就报错。这是我保证"不瞎猜"的唯一手段。

### 3. 我跑标定

```bash
# 扫 inbox，逐张出探针分数 + 生成待填台账
py -3.12 tools/profile_init.py --profile tli_phone960

# 按台账核对（strict 不符就 exit 1）
py -3.12 tools/ref_check.py --profile tli_phone960

# 用这个 profile 跑生产
py -3.12 run_server.py --config config.yaml --profile tli_phone960
```

## 自动 / 必须人工，边界在哪

我**不能**凭一张图"看懂"这是什么，因为历史已经证明我会看错（把技能图标角落当成 "Q"，
把星图节点当成光柱，把角色面板下的血球当成在战斗）。所以流程是这样分工的：

| 环节 | 谁做 | 说明 |
|---|---|---|
| 这张图属于哪类界面 | **你** | 一句话，见上面第 2 步 |
| 战斗 HUD 拿哪一块当锚点 | 我做候选，**分数裁决** | 圈出候选 → 对全部真实帧算分数 → 战斗帧 ≥0.6、菜单帧 ≤0.4 才算通过 |
| 光柱颜色/形状阈值 | 我做候选，**分数裁决** | 同上，误报一张就作废 |
| HUD 排除区（哪些矩形不可能是掉落物） | 我做候选，**你眼校** | 页面上的"相机标定"能看框 |
| 角色是否在屏幕中心 | **你** | 一眼的事，我猜错整个走位方向就反了 |
| 交互键位（确认/ESC/拾取） | **你** | 键位没法从画面推 |
| 落地成 profile + 回归用例 | 我 | 你不碰代码 |

一句话总结：**"这是什么"你定，"拿什么像素证据判定"我算，"算得对不对"用你给的答案验。**
中间任何一步没有分数支撑，我不会写进 profile——宁可报"这块标不了"，也不塞一个看起来
合理的阈值进去。

## 目录约定

```
profiles/<id>/
    profile.yaml      # 元信息 + 对这个游戏的 config 覆盖（与 config.yaml 同构）
    templates/        # gate_pos_*.png 正向锚点 / gate_neg_*.png 反例 / _index.json
    labels.json       # 人工标答台账（ref_check 逐字段核对）
    inbox/            # 你丢进来的新截图
    refs/             # 标定完成的截图存档（inbox 里清空后挪这儿）
    deliveries.md     # 你写的"这张该判成什么"
```

`templates/_index.json` 记录每张模板"是按多宽的帧裁的"。默认路径
（`tools/make_template.py`）会把裁块统一缩放存成 1280 宽，这时不用管它；
只有你手工裁了张原生尺寸的图（比如直接从 960 录屏截一块没缩放），才需要在这里声明
`base_width`，否则会被按 1280 反推缩放，分数掉到噪声水平。

顺带澄清一个容易混的点：**模板跨分辨率大致能用，跨游戏基本不能**。把技能栏模板放在
同一游戏的另一分辨率上，分数是 0.9→0.7 这种衰减；放到另一个游戏上是 0.25——那不是
缩放错了，是两块 HUD 本来就长得不一样。所以每个游戏都要有自己的 `gate_pos_*.png`，
没有捷径。

## 已有 profile

| id | 说明 | 状态 |
|---|---|---|
| `generic_1280x720` | 仓库自带默认（复用根 `templates/` 与 `assets/ref/labels.json`） | 已标定 |
| `tli_phone960` | 火炬之光：无限 手机端录屏 960x432 | **partial**：正例（右下技能轮盘 0.99/0.88）+ 星图反例（1.00）已就位，缺背包/角色面板/结算反例 |

新建一个：`cp -r profiles/_template profiles/<新id>`，改 `profile.yaml` 里的 id 与
`reference_resolution`，然后把截图丢进 `inbox/`。

## 一个真实标定的完整过程（tli_phone960）

留在这里当范例，照着做就行：

```bash
# 1) 探针：看每张图的分数、列出所有"像光柱"的色块（含坐标和高宽比）
py -3.12 tools/profile_init.py --profile tli_phone960

# 2) 先做反例（门禁反例优先，顺序反了会让菜单直接放行）
py -3.12 tools/make_template.py --src profiles/tli_phone960/refs/tli_nether_map_01.jpg \
    --rect 8,2,205,55 --out profiles/tli_phone960/templates/gate_neg_nether_map.png \
    --check "profiles/tli_phone960/refs/tli_nether*.jpg"

# 3) 再做正例。手机端技能在右下角，必须带 --roi，否则默认的"底部 35%"会裁掉一半
py -3.12 tools/make_template.py --src profiles/tli_phone960/refs/tli_combat_01.jpg \
    --rect 640,255,900,390 --roi 600,235,960,432 \
    --out profiles/tli_phone960/templates/gate_pos_skillwheel.png \
    --check "profiles/tli_phone960/refs/tli_combat_0*.jpg" \
    --against "profiles/tli_phone960/refs/tli_nether*.jpg"

# 4) 回归
py -3.12 tools/ref_check.py --profile tli_phone960
```

第 3 步实测：自匹配 0.993、另一张战斗帧 0.883、星图 0.281。三条都过才写进 templates/。

## 常见坑

- **正例没轮上 ROI 就下结论"模板不管用"**：默认 ROI 是底部 35% 全宽，那是 PC ARPG
  技能栏横在底部的假设。手机端技能在右下角，不传 `--roi` 分数会从 0.99 掉到 0.3。
- **把 `template_base_width` 改成录制分辨率**：模板是 1280 归一化存储的，改成 960 会
  让反例从 1.00 掉到 0.41，表现是"菜单拦不住"，但看起来像"模板失效"。
- **先做正例再做反例**：门禁反例优先，正例先上就等于给菜单发了通行证。
- **裁到动态元素**：技能图标会随冷却变灰、光环会闪。要裁位置固定且外观稳定的
  部件（边框、外圈、标题栏），分数不稳就是裁错了。
