"""版本三元组 —— 引擎会换，数据不能跟着坏，所以分开记数。

三个号回答三个不同的问题，任何一个对不上就拒绝启动/拒绝升级：

    ENGINE_VERSION   引擎（代码）版本。升级就是把它从 0.2.0 换成 0.2.1。
    DATA_SCHEMA      数据布局版本。profiles/logs 的目录结构与字段约定。
                     引擎只**向后读**：能读 `MIN_DATA_SCHEMA..DATA_SCHEMA` 之间的数据。
    RECIPE_SCHEMA    配方版本（evolve/rules.yaml 的 DSL 字段集）。
                     新引擎必须能跑旧配方，否则升级会让大家手里那份学出来的规则失效。

## 为什么数据要有版本

"引擎不能改但会不停升级"意味着升级是常态。常态操作必须便宜且可逆，
所以升级前要能一眼判断："这个新引擎认不认我现在的数据？"
不认就别装，而不是装完发现手册读不进来。

向后兼容的口径：**小版本升级只允许加字段，不改语义**。加字段时把 DATA_SCHEMA
加一，并在 upgrade.py 的 MIGRATIONS 里补一条单向迁移；删字段/改语义属于大版本，
必须人工确认（升级工具会拦下来）。
"""

from __future__ import annotations

ENGINE_VERSION = "0.2.0"

#: 本引擎写出的数据布局版本
DATA_SCHEMA = 2

#: 本引擎还能读的最旧数据（低于它的数据必须先由支持它的引擎升上来，不能跳）
MIN_DATA_SCHEMA = 1

#: 配方 DSL 版本（rules.yaml / templates 索引）
RECIPE_SCHEMA = 1

#: 数据目录里记录当前布局版本的文件名
SCHEMA_FILE = ".schema"


def info() -> dict:
    return {
        "engine_version": ENGINE_VERSION,
        "data_schema": DATA_SCHEMA,
        "min_data_schema": MIN_DATA_SCHEMA,
        "recipe_schema": RECIPE_SCHEMA,
    }
