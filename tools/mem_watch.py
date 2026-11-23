"""跑一条命令并采样它的内存/耗时，用来看"到底是涨内存还是涨时间"。

用法：
  py -3.12 tools/mem_watch.py -- py -3.12 tools/video_replay.py assets/replay/src/x.mp4
"""
from __future__ import annotations

import subprocess
import sys
import time

import psutil


def main() -> None:
    if "--" not in sys.argv:
        raise SystemExit("用法: python tools/mem_watch.py -- <命令...>")
    cmd = sys.argv[sys.argv.index("--") + 1:]
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd)
    proc = psutil.Process(p.pid)
    peak = 0.0
    samples = 0
    while p.poll() is None:
        try:
            # 必须连子进程一起算：`py -3.12 xxx.py` 会再 fork 一个 python.exe，
            # 只量启动器会得到"峰值 13MB"这种显然不对的数。
            rss = sum(c.memory_info().rss for c in proc.children(recursive=True)
                      if c.is_running()) / 1024 / 1024
            rss += proc.memory_info().rss / 1024 / 1024
        except psutil.NoSuchProcess:
            break
        peak = max(peak, rss)
        samples += 1
        if samples % 10 == 1:
            print(f"  [{time.perf_counter() - t0:6.1f}s] RSS {rss:7.0f}MB 峰值 {peak:7.0f}MB",
                  flush=True)
        time.sleep(0.5)
    print(f"退出码 {p.returncode}  总耗时 {time.perf_counter() - t0:.1f}s  "
          f"峰值 RSS {peak:.0f}MB")


if __name__ == "__main__":
    main()
