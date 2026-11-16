# templates/ui —— UI 场景自动化模板

配方引擎（交易行/装备/天赋/技能）靠**模板匹配**定位面板与按钮，模板必须来自
**你自己的分辨率 / 画质 / 语言**的游戏截图。放错或缺失时服务端会直接拒绝启动任务
（HTTP 409 + 缺失清单），不会静默空转。

## 每个配方需要哪些模板

| 配方 | 模板文件 | 说明 |
|---|---|---|
| `trade_buy` | `panel_trade.png` | 交易行面板锚点（标题栏或面板边框的稳定区域） |
| | `sort.png` | 排序按钮 |
| | `buy.png` | 购买按钮 |
| | `confirm.png` | 确认按钮 |
| `equip_best` | `panel_equip.png` / `equip.png` / `confirm.png` | 装备面板 + 装备 + 确认 |
| `talent_alloc` | `panel_talent.png` / `confirm.png` | 天赋面板 + 确认 |
| `skill_swap` | `panel_skill.png` / `learn.png` / `confirm.png` | 技能面板 + 学习 + 确认 |

随时用命令查看最新清单：

```bash
python tools/crop_template.py --list all
```

## 裁剪方法

```bash
# 1) 游戏里打开对应面板 → 截图（Win+Shift+S 或 tools/grab_frame.py）
python tools/grab_frame.py --out shot.png

# 2) 按截图像素坐标裁（左上角 x,y + 宽高），带 4 倍预览图人工确认
python tools/crop_template.py --image shot.png --rect 950,110,120,40 \
    --out templates/ui/sort.png --preview assets/tpl_preview.png
```

## 注意事项

- **同一分辨率**：模板与运行时画面必须同分辨率；换分辨率/画质/语言后要重裁。
- **带内容的按钮**：模板不要是纯色块（TM_CCOEFF_NORMED 对零方差区域退化），
  必须包含文字或边框。
- **面板锚点**：选面板中**不随内容变化**的区域（标题栏、边框），否则内容一变就匹配失败。
- **不要裁到会动的元素**（滚动条、倒计时、闪烁高亮）。
