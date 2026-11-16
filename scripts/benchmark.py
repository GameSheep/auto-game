"""性能基准 —— 用实测数据回答"压力大不大"（对话原话：opencv 压力大不大）。

测项：
  1. 检测器单帧耗时 @ 720p / 1080p / 1440p / 2160p（含降采样路径）
  2. JPEG 编码耗时（代理侧）/ 解码耗时（服务端侧）
  3. 端到端单帧预算（解码+检测+决策占位）
  4. 内存占用（进程 RSS 增量近似）

用法：
  .venv/Scripts/python scripts/benchmark.py            # 全部测项
  .venv/Scripts/python scripts/benchmark.py --runs 100 # 提高采样次数
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autogame.config import VisionConfig  # noqa: E402
from autogame.sim.scene import SimWorld  # noqa: E402
from autogame.vision.detector import Detector  # noqa: E402


def bench(fn, runs: int, warmup: int = 5) -> dict:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "runs": runs,
        "mean_ms": round(statistics.mean(times), 3),
        "p95_ms": round(sorted(times)[int(len(times) * 0.95) - 1], 3),
        "min_ms": round(min(times), 3),
        "max_ms": round(max(times), 3),
        "fps_eq": round(1000 / statistics.mean(times), 1),
    }


def make_frame(w: int, h: int) -> np.ndarray:
    """生成带光柱/A框的基准帧（以 720p sim 世界为基底，按需放大）。"""
    world = SimWorld(width=1280, height=720, seed=1234)
    world.pillars.clear()
    from autogame.sim.scene import SimPillar

    world.pillars.append(SimPillar(160, 40, "gold"))
    world.pillars.append(SimPillar(-90, -60, "pink"))
    frame = world.render()
    if (w, h) != (1280, 720):
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()
    runs = args.runs

    # 用**部署配置**（含 A框模板/场景门）测，否则数字比现场好看：
    # 模板匹配是新增的真实成本，不能只测"没有模板的默认值"。
    try:
        from autogame.config import load_config
        from pathlib import Path

        cfg = load_config(str(Path(__file__).resolve().parent.parent / "config.yaml")).vision
        print(" 配置: config.yaml（含 A框模板/场景门）")
    except Exception as e:  # noqa: BLE001
        print(f" 配置: 默认值（config.yaml 加载失败: {e}）")
        cfg = VisionConfig()
    results: dict[str, dict] = {}

    print("=" * 70)
    print(" autogame 性能基准  (CPU:", os.cpu_count(), "cores )")
    print("=" * 70)

    # 1) 检测器单帧耗时（各分辨率）
    for (w, h) in [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)]:
        frame = make_frame(w, h)
        det = Detector(cfg)
        r = bench(lambda: det.process(frame), runs)
        results[f"detector_{w}x{h}"] = r
        print(f"检测器 @{w}x{h:<9} 平均 {r['mean_ms']:>7.2f} ms  p95 {r['p95_ms']:>7.2f} ms"
              f"  ≈{r['fps_eq']} fps")

    # 2) JPEG 编码（代理侧，2560x1440 实战常见）
    frame = make_frame(2560, 1440)
    for q in (60, 80, 90):
        r = bench(lambda q=q: cv2.imencode(".jpg", frame,
                                           [int(cv2.IMWRITE_JPEG_QUALITY), q]), runs)
        results[f"jpeg_encode_q{q}_1440p"] = r
        print(f"JPEG编码 q{q} @1440p     平均 {r['mean_ms']:>7.2f} ms"
              f"  ≈{r['fps_eq']} fps")

    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    jpeg = buf.tobytes()
    print(f"JPEG 体积 @1440p q80: {len(jpeg) / 1024:.0f} KB/帧"
          f" → 2fps 带宽 ≈ {len(jpeg) * 2 / 1024:.0f} KB/s")

    # 3) JPEG 解码（服务端侧）
    r = bench(lambda: cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR), runs)
    results["jpeg_decode_1440p"] = r
    print(f"JPEG解码 @1440p          平均 {r['mean_ms']:>7.2f} ms  ≈{r['fps_eq']} fps")

    # 4) 端到端单帧预算（服务端：解码+检测，决策 <1ms）
    det = Detector(cfg)

    def e2e():
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        det.process(img)

    r = bench(e2e, runs)
    results["e2e_decode_detect_1440p"] = r
    print(f"端到端(解码+检测) @1440p 平均 {r['mean_ms']:>7.2f} ms"
          f"  → 1Hz 决策下 CPU 占用 ≈ {r['mean_ms'] / 1000 * 100:.1f}% 单核")

    # 5) 内存增量（处理 100 帧后的近似驻留）
    import tracemalloc

    tracemalloc.start()
    frame = make_frame(1920, 1080)
    det = Detector(cfg)
    for _ in range(100):
        det.process(frame)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results["memory_100frames_1080p"] = {
        "current_mb": round(current / 1024 / 1024, 2),
        "peak_mb": round(peak / 1024 / 1024, 2),
    }
    print(f"内存(100帧@1080p)        驻留 {current / 1024 / 1024:.1f} MB"
          f"  峰值 {peak / 1024 / 1024:.1f} MB")

    out = ROOT / "benchmark_result.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main()
