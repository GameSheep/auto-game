"""落盘这件小事：要么旧文件，要么新文件，不能有半份。

## 为什么单独一个模块

在这套东西里，"状态写一半"的代价和别处不一样：

- 设备是 RK3566 + eMMC，断电是**常态**（不是意外）。用户拔电源、板子欠压复位、
  服务被 kill -9，都发生在写入途中。
- 有**多个进程**会写同一批文件：服务进程（页面点「现在学」、后台 tick）、
  `tools/` 下的命令行、升级脚本。两个"读-改-写"交错，后写的会把先写的吃掉。

原来的写法是 `p.write_text(...)` 或 `tmp.write_text(...); tmp.replace(p)`。
第二种看着像原子写，其实只做了一半：

    rename 只保证"别人看到的名字"是原子的，不保证"内容已经落到存储上"。
    ext4 延迟分配 + eMMC 自带缓存，断电后完全可能得到一个长度 0 的文件 ——
    而它名字是对的、时间戳是新的，读起来像"状态被清空了"，不像"坏了"。

所以这里写 + `fsync` + rename + **目录 fsync**，一个都不能少。目录 fsync 是为了
让"改名"这个动作本身落盘，否则断电后可能回到改名之前（指针指回上一代/空）。

## 三个原语

- `atomic_write_json`：临时文件 → fsync → os.replace → 目录 fsync
- `load_json_tolerant`：读坏了就挪到 `<名>.corrupt-<时间戳>` 再返回默认值，
  而不是静默当新文件 —— "学出来的状态被悄悄清空且没有证据"是最难查的一类 bug
- `file_lock`：`<路径>.lock` 旁路文件上的操作系统锁，**跨进程**有效。
  Windows 用 `msvcrt.locking`，其他用 `fcntl.flock`；外加一层进程内 RLock，
  因为 Windows 的字节锁在同一进程的不同句柄之间不互斥
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

log = logging.getLogger(__name__)

__all__ = [
    "atomic_write_bytes",
    "atomic_write_text",
    "atomic_write_json",
    "backup_corrupt",
    "load_json_tolerant",
    "file_lock",
    "locked_update_json",
]


# --------------------------------------------------------------------------- 原子写


def _fsync_dir(path: Path) -> None:
    """让"改名"本身落盘。Windows 上打不开目录，直接跳过 —— 那里 rename 是元数据操作。"""
    if os.name == "nt":  # pragma: no cover - 平台分支
        return
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path | str, data: bytes, *, fsync: bool = True) -> None:
    """写 *data* 到 *path*，要么全成，要么原文件一个字节都不动。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                    dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            if fsync:
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass      # 有些文件系统不给 fsync；rename 至少还能保证可见性
        for attempt in range(6):
            try:
                os.replace(tmp, path)
                break
            except OSError as exc:
                # Windows 上杀毒/索引器会短暂占住句柄，导致替换失败（错误码 5/32/33）。
                # 只重试"已经写好内容"的改名，绝不重试写和改 —— 那会真的写坏。
                if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 5:
                    raise
                time.sleep(0.01 * (2 ** attempt))
        if fsync:
            _fsync_dir(path.parent)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def atomic_write_text(path: Path | str, text: str, *, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: Path | str, obj: Any, *, indent: int | None = None,
                      ensure_ascii: bool = False, sort_keys: bool = False) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent,
                                       sort_keys=sort_keys))


# --------------------------------------------------------------------------- 坏了怎么办


def backup_corrupt(path: Path | str) -> Path | None:
    """把读不出来的文件挪到一边，返回挪到哪了。挪不动就返回 None（不抛）。"""
    path = Path(path)
    if not path.exists():
        return None
    stamp = int(time.time())
    target = path.with_name(f"{path.name}.corrupt-{stamp}")
    n = 1
    while target.exists():
        target = path.with_name(f"{path.name}.corrupt-{stamp}-{n}")
        n += 1
    try:
        os.replace(path, target)
    except OSError:
        return None
    # 数据被移到一边是"发生过事故"，必须留痕迹，不能只在代码里悄悄返回默认值
    log.warning("文件坏了，已备份: %s → %s", path.name, target.name)
    return target


def load_json_tolerant(path: Path | str, default: Any = None) -> Any:
    """读 JSON；读不出来就备份原件并返回 *default*。

    "备份"这个动作是关键：不备份的容错等于把数据静默丢掉 —— 用户只看到
    "学习进度归零了"，而没有任何东西能查。
    """
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - 半份/乱码/权限，一律当坏
        backup_corrupt(path)
        return default


# --------------------------------------------------------------------------- 锁

try:                                    # Windows
    import msvcrt

    def _lock_fd(fd: int) -> bool:
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _unlock_fd(fd: int) -> None:
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass

except ImportError:                     # POSIX
    import fcntl

    def _lock_fd(fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _unlock_fd(fd: int) -> None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass


_PROC_LOCKS: dict[str, threading.RLock] = {}
_PROC_GUARD = threading.Lock()


def _proc_lock(path: Path) -> threading.RLock:
    key = os.path.normcase(str(path))
    with _PROC_GUARD:
        lock = _PROC_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PROC_LOCKS[key] = lock
        return lock


@contextmanager
def file_lock(path: Path | str, timeout: float = 10.0, poll: float = 0.05) -> Iterator[None]:
    """*path* 的独占锁，线程之间和**进程之间**都互斥。

    两层缺一不可：进程内 RLock 管同进程的多线程（Windows 的字节锁在同一个进程的
    两个句柄之间不冲突，光有它管不住线程），`<path>.lock` 上的系统锁管别的进程。

    注意：**不要嵌套**同一个 path。进程内 RLock 是可重入的，会放行，但系统锁是
    按"打开的文件描述"算的，第二次开同一个文件会自己等自己，直到超时。
    需要"读-改-写"一整段，用 `locked_update_json`，别在锁里再调会加锁的函数。
    """
    lock_path = Path(f"{path}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    plock = _proc_lock(lock_path)
    if not plock.acquire(timeout=max(0.0, timeout)):
        raise TimeoutError(f"获取文件锁超时（{timeout}s）: {lock_path}")

    fd = -1
    acquired = False
    try:
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o666)
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            if _lock_fd(fd):
                acquired = True
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"锁被别的进程占着（{timeout}s）: {lock_path}")
            time.sleep(poll)
        yield
    finally:
        if acquired:
            _unlock_fd(fd)
        if fd >= 0:
            os.close(fd)
        plock.release()


def locked_update_json(path: Path | str, mutator: Callable[[Any], Any], *,
                       default: Any = None, timeout: float = 10.0,
                       indent: int | None = 2, sort_keys: bool = False) -> Any:
    """读 → mutator → 原子写，整段在锁里。两个进程同时改一个状态文件不会互相吃掉。"""
    path = Path(path)
    with file_lock(path, timeout=timeout):
        current = load_json_tolerant(path, default)
        updated = mutator(current)
        atomic_write_json(path, updated, indent=indent, sort_keys=sort_keys)
        return updated
