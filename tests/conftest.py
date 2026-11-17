"""测试全局隔离 —— 让单元测试保持**确定性**。

拟人化层（疲劳 / 反应时间 / 走神 / 进关卡前犹豫）是刻意引入随机时序的：
"看到 A框后 180-620ms 才按下""偶尔跳过一拍去发呆"这类行为，
会让"第 2 帧必须点击"这类断言随机失败 —— 那不是 bug，是测试假设变了。

处理方式（与 autonomy 的隔离思路一致，见 tests/test_fsm.py::_mk_engine）：
  1. 整个测试套件默认关闭拟人化（AUTOGAME_HUMANIZE=0）；
  2. 拟人化本身由 tests/test_humanize.py 显式打开并逐条断言。
"""

import os

# 必须在 autogame.config 被导入之前设置（config.py 在导入时读取该变量）
os.environ.setdefault("AUTOGAME_HUMANIZE", "0")
