"""两个根：引擎根（只读，整包换）与数据根（用户的图、手册、日志，升级不碰）。

## 为什么必须分开

"引擎绝对不能改，但可以升级" —— 升级是**换整个目录**，不是就地改文件。
如果 profiles/ 和 logs/ 长在引擎目录里，换目录就等于把用户的截图、手册、
进化代数一起换掉。所以数据必须住在另一个根上，升级只动代码那一半。

## 设备上的布局

    /opt/autogame/releases/0.2.0/   引擎（解包出来的只读副本）
    /opt/autogame/current           一行文本："0.2.0"
    /opt/autogame/bin/autogame      唯一入口，读指针决定跑哪个 release
    /var/lib/autogame/              数据根
        profiles/                   图、手册、模板、进化代数
        logs/                       supervisor/arbiter/事件台账
        config.yaml                 这份机器自己的配置
        snapshots/                  升级前的元数据快照
        .schema                     数据布局版本（整数，见 autogame/version.py）

systemd 里 `ReadOnlyPaths=/opt/autogame/releases`，服务进程写不进去 ——
"不能改"是内核保证的，不是靠自觉。

## 开发机上

不设 `AUTOGAME_DATA` 时数据根 = 引擎根，行为与以前完全一致（仓库根就是数据根）。
这样 `pytest`、`tools/*.py` 全部不用改。
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_DATA = "AUTOGAME_DATA"
_ENV_HOME = "AUTOGAME_HOME"


def code_root() -> Path:
    """引擎代码所在目录。设备上是 `/opt/autogame/releases/<ver>`。"""
    return Path(__file__).resolve().parent.parent


def data_root() -> Path:
    """可变数据所在目录。设备上是 `/var/lib/autogame`，开发机上等于 code_root。"""
    raw = os.environ.get(_ENV_DATA, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return code_root()


def install_root() -> Path:
    """整个安装目录（含 releases/ 与 current）。

    设备上由 `AUTOGAME_HOME` 显式给出；否则从代码位置反推：
    代码根是 `.../releases/<ver>` 时取上上级，其他情况就当代码根即安装根。
    """
    raw = os.environ.get(_ENV_HOME, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    root = code_root()
    if root.parent.name == "releases":
        return root.parent.parent
    return root


def data(*parts: str | os.PathLike) -> Path:
    return data_root().joinpath(*parts)


def code(*parts: str | os.PathLike) -> Path:
    return code_root().joinpath(*parts)


def resolve(raw: str | os.PathLike, *, base: Path | None = None) -> Path:
    """用户写的相对路径按数据根解析（绝对路径原样返回）。

    为什么基准不是 cwd：设备上 systemd 的 cwd 是 `/`，
    相对路径会解析到根目录，表现为"配置里的路径全都不存在"。
    """
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    return (base or data_root()) / p


def shared() -> bool:
    """两个根是不是同一个目录（开发机 / 扁平安装）。"""
    return data_root() == code_root()


def describe() -> dict:
    return {
        "code": str(code_root()),
        "data": str(data_root()),
        "install": str(install_root()),
        "shared": shared(),
    }
