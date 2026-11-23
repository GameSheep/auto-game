"""内存探针：按真实链路喂帧，看 RSS 是否随帧数线性增长。

用法：
  py -3.12 tools/mem_probe.py --frames 300 --w 1920 --h 1080
  py -3.12 tools/mem_probe.py --frames 300 --path full     # 检测+决策+hub 记账
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
import tracemalloc
from pathlib import Path

import cv2
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autogame.config import AppConfig                     # noqa: E402
from autogame.decision.fsm import DecisionEngine          # noqa: E402
from autogame.executor import DryRunExecutor              # noqa: E402
from autogame.executor.base import ExecutorConfig         # noqa: E402
from autogame.vision.detector import Detector, ROOT_TEMPLATES  # noqa: E402


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 / 1024


def make_frame(w: int, h: int, i: int) -> np.ndarray:
    """合成一帧：深底色 + 一根光柱 + 一点噪声，保证检测路径真的走完整。"""
    f = np.full((h, w, 3), (30, 26, 24), np.uint8)
    x = int(w * 0.3 + (i * 7) % (w // 2))
    cv2.rectangle(f, (x - 8, int(h * 0.55)), (x + 8, int(h * 0.75)), (60, 200, 240), -1)
    cv2.rectangle(f, (x - 3, int(h * 0.56)), (x + 3, int(h * 0.74)), (255, 255, 255), -1)
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--path", choices=["detect", "full"], default="full")
    ap.add_argument("--hz", type=float, default=2.0)
    args = ap.parse_args()

    cfg = AppConfig()
    det = Detector(cfg.vision, ROOT_TEMPLATES if ROOT_TEMPLATES.exists() else None)
    if args.path == "full":
        ex = DryRunExecutor(ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000))
        eng = DecisionEngine(cfg.decision, ex)
        eng.start()

    gc.collect()
    base = rss_mb()
    tracemalloc.start()
    print(f"起点 RSS {base:.0f}MB  帧 {args.w}x{args.h}  共 {args.frames} 帧  路径={args.path}")
    sample_every = max(1, args.frames // 10)
    t0 = time.perf_counter()
    for i in range(args.frames):
        frame = make_frame(args.w, args.h, i)
        ts = int(i / args.hz * 1000)
        scene = det.process(frame, ts)
        if args.path == "full":
            eng.on_scene(scene)
        del frame
        if (i + 1) % sample_every == 0:
            cur = rss_mb()
            gc.collect()
            print(f"  {i + 1:>5} 帧  RSS {cur:7.0f}MB  (+{cur - base:6.0f})  "
                  f"{time.perf_counter() - t0:6.1f}s")
    gc.collect()
    end = rss_mb()
    cur_alloc, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    dt = time.perf_counter() - t0
    print(f"结束 RSS {end:.0f}MB  净增 {end - base:+.0f}MB  "
          f"tracemalloc 当前 {cur_alloc / 1e6:.0f}MB 峰值 {peak / 1e6:.0f}MB")
    print(f"均耗时 {dt / args.frames * 1000:.1f}ms/帧")


if __name__ == "__main__":
    main()
