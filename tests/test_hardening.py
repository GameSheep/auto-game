"""加固测试 —— 随机帧 fuzz（不崩溃）+ 性能回归守卫。"""

import time

import cv2
import numpy as np
import pytest

from autogame.config import VisionConfig
from autogame.vision.detector import Detector, ROOT_TEMPLATES


def _random_frame(rng, mode):
    """构造各类极端随机帧。"""
    h, w = 720, 1280
    if mode == "noise":
        return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    if mode == "flat":
        v = int(rng.integers(0, 256))
        return np.full((h, w, 3), v, dtype=np.uint8)
    if mode == "stripes":
        f = np.zeros((h, w, 3), dtype=np.uint8)
        for x in range(0, w, int(rng.integers(4, 60))):
            c = rng.integers(0, 256, 3)
            f[:, x:x + int(rng.integers(1, 10))] = c
        return f
    if mode == "dark":
        return rng.integers(0, 30, (h, w, 3), dtype=np.uint8)
    if mode == "bright":
        return rng.integers(220, 256, (h, w, 3), dtype=np.uint8)
    # blobs: 随机色块
    f = np.full((h, w, 3), (28, 24, 22), dtype=np.uint8)
    for _ in range(rng.integers(3, 20)):
        x, y = rng.integers(0, w - 60), rng.integers(0, h - 60)
        bw, bh = int(rng.integers(5, 60)), int(rng.integers(5, 200))
        c = rng.integers(0, 256, 3)
        f[y:y + bh, x:x + bw] = c
    return f


MODES = ["noise", "flat", "stripes", "dark", "bright", "blobs"]


@pytest.mark.parametrize("mode", MODES)
def test_fuzz_random_frames_never_crash(mode):
    """随机极端帧 fuzz：检测器必须稳定返回合法 SceneState。"""
    rng = np.random.default_rng(abs(hash(mode)) % (2**32))
    det = Detector(VisionConfig(), ROOT_TEMPLATES if ROOT_TEMPLATES.exists() else None)
    for i in range(6):
        frame = _random_frame(rng, mode)
        st = det.process(frame, 1000 + i * 1000)
        # 输出结构合法性
        assert 0 <= st.character[0] <= st.w
        assert 0 <= st.character[1] <= st.h
        for p in st.pillars:
            assert 0 <= p.cx <= st.w and 0 <= p.cy <= st.h
            assert p.w > 0 and p.h > 0
        if st.aframe:
            assert 0 <= st.aframe.x <= st.w
        assert isinstance(st.frame_delta, float)
        assert st.in_game in (True, False)


def test_fuzz_two_frame_consistency():
    """连续两帧 fuzz：跟踪状态不崩溃、无异常抛出。"""
    rng = np.random.default_rng(123)
    det = Detector(VisionConfig(), ROOT_TEMPLATES if ROOT_TEMPLATES.exists() else None)
    for i in range(20):
        f1 = _random_frame(rng, MODES[i % len(MODES)])
        f2 = _random_frame(rng, MODES[(i + 1) % len(MODES)])
        det.process(f1, 1000 + i * 200)
        det.process(f2, 1100 + i * 200)


def _bench(det, frame, n: int = 8, warm: int = 1) -> tuple[float, float]:
    """返回 (最快帧耗时, 中位耗时) 毫秒。

    为什么用最快帧而不是平均：这些断言在开发机上跑，CI 并行/热降频会让个别帧
    被抢占几十毫秒，均值和中位都跟着飘（实测同一提交在不同负载下 62~103ms）。
    最快的帧最接近"纯计算耗时"，噪声最小；真出现 1.5 倍级退化时它也会上升。
    """
    ts = 1000
    for _ in range(warm):
        det.process(frame, ts)
        ts += 500
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        det.process(frame, ts)
        times.append((time.perf_counter() - t0) * 1000)
        ts += 500
    times.sort()
    return times[0], times[len(times) // 2]


def test_perf_guard_detector_720p():
    """性能守卫：720p 单帧检测最快帧 ≤ 90ms（基准线 ~40ms，J4125 预算 1Hz=1000ms）。

    阈值给到 90 而不是 60：开发机在跑全量测试时会热降频，实测干净提交也能到 62ms；
    60 的断言等于每跑几次就红一次，久而久之就没人看了——那比没有守卫更糟。
    """
    frame = np.full((720, 1280, 3), (28, 24, 22), dtype=np.uint8)
    frame[400:510, 800:814] = (36, 190, 255)
    frame[406:506, 805:809] = (255, 255, 255)
    det = Detector(VisionConfig(), ROOT_TEMPLATES if ROOT_TEMPLATES.exists() else None)
    best, med = _bench(det, frame)
    assert best <= 90, (f"检测器 720p 最快帧 {best:.1f}ms 超出 90ms 守卫预算（性能退化）；"
                        f"中位 {med:.1f}ms")


def test_perf_guard_scene_gate_halfres():
    """性能守卫：真实 720p 帧的最快帧 ≤ 90ms（含场景门匹配 + 光柱/怪物检测）。"""
    frame = cv2.imread("assets/samples/real_hud_boss.png") if (ROOT_TEMPLATES.parent / "assets" / "samples").exists() else None
    if frame is None:
        pytest.skip("素材缺失")
    cfg = VisionConfig()
    det = Detector(cfg, ROOT_TEMPLATES if ROOT_TEMPLATES.exists() else None)
    best, med = _bench(det, frame, n=6)
    assert best <= 90, (f"真实 1280x720 帧最快 {best:.1f}ms 超出 90ms 守卫预算；"
                        f"中位 {med:.1f}ms")
