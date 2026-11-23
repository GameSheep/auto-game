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


# 标定工作量：固定尺寸的模板匹配。用它把"机器快慢"和"代码快慢"分开 ——
# 同一次运行里，机器降频会同时拖慢标定和检测；只有代码退化才会"标定没变、检测变慢"。
_CALIB_IMG = np.random.RandomState(20261123).randint(0, 255, (360, 640), dtype=np.uint8)
_CALIB_TPL = _CALIB_IMG[100:164, 200:264].copy()
#: 标定在"没降频的机器"上的典型最快耗时（本机实测 3.1~4.2ms）。
_CALIB_NOMINAL_MS = 4.0
#: 检测预算（干净机器上的基准：整帧 ~50~70ms；J4125 按 3 倍估仍留 1Hz 余量）
_BUDGET_MS = 90.0


def _calib_ms(n: int = 6) -> float:
    """固定计算量的最快耗时（毫秒）。"""
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        cv2.matchTemplate(_CALIB_IMG, _CALIB_TPL, cv2.TM_CCOEFF_NORMED)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[0]


def _budget_for_machine() -> tuple[float, float, float]:
    """按当前机器工况放大预算 → (预算ms, 标定ms, 缩放倍数)。

    为什么必须缩放：本机 CPU 满载几分钟后会从 2600MHz 掉到 1496MHz（实测），
    同一份代码的整帧检测从 52ms 变成 160ms —— **2~3 倍**。全量测试跑五六分钟，
    所以这两条守卫以前在整套里**必然红**，单独跑却绿。一条"总是红"的守卫等于没有，
    而且会让人学会忽略它（本项目已经吃过一次"静默失效"的亏）。

    缩放只跟随**标定**，不跟随检测本身：代码真退化时标定不变、检测变慢，
    照样会红。想让它变绿只能改回去，不能靠"机器慢"蒙过去。
    """
    calib = _calib_ms()
    scale = max(1.0, calib / _CALIB_NOMINAL_MS)
    return _BUDGET_MS * scale, calib, scale


def test_perf_guard_detector_720p():
    """性能守卫：720p 单帧检测最快帧 ≤ 90ms（按降频程度放大，见 `_budget_for_machine`）。

    阈值给到 90 而不是 60：干净机器实测整帧 50~70ms，90 留了余量；
    60 的断言等于每跑几次就红一次，久而久之就没人看了——那比没有守卫更糟。
    """
    frame = np.full((720, 1280, 3), (28, 24, 22), dtype=np.uint8)
    frame[400:510, 800:814] = (36, 190, 255)
    frame[406:506, 805:809] = (255, 255, 255)
    det = Detector(VisionConfig(), ROOT_TEMPLATES if ROOT_TEMPLATES.exists() else None)
    best, med = _bench(det, frame)
    budget, calib, scale = _budget_for_machine()
    assert best <= budget, (
        f"检测器 720p 最快帧 {best:.1f}ms 超出预算 {budget:.0f}ms"
        f"（预算 90ms × 降频倍数 {scale:.2f}，标定 {calib:.1f}ms≤标称 "
        f"{_CALIB_NOMINAL_MS}ms 时倍数为 1）；中位 {med:.1f}ms")


def test_perf_guard_scene_gate_halfres():
    """性能守卫：真实 720p 帧的最快帧 ≤ 90ms（含场景门匹配 + 光柱/怪物检测）。"""
    frame = cv2.imread("assets/samples/real_hud_boss.png") if (ROOT_TEMPLATES.parent / "assets" / "samples").exists() else None
    if frame is None:
        pytest.skip("素材缺失")
    cfg = VisionConfig()
    det = Detector(cfg, ROOT_TEMPLATES if ROOT_TEMPLATES.exists() else None)
    best, med = _bench(det, frame, n=6)
    budget, calib, scale = _budget_for_machine()
    assert best <= budget, (f"真实 1280x720 帧最快 {best:.1f}ms 超出预算 {budget:.0f}ms"
                            f"（降频倍数 {scale:.2f}，标定 {calib:.1f}ms）；中位 {med:.1f}ms")


def test_perf_budget_scales_with_machine_state_only(monkeypatch):
    """预算只跟随**标定**，不跟随检测 —— 这是"不靠机器慢蒙过去"的保证。

    本机降频实测：2600→1496MHz 时整帧 52→160ms（2~3 倍），全量测试里这两条
    守卫必然红。改成按标定缩放后，降频时预算同步放大（校验仍活着但不再误报）；
    代码真退化时标定不变 → 预算不变 → 照样红。
    """
    import sys

    mod = sys.modules[__name__]
    monkeypatch.setattr(mod, "_calib_ms", lambda *a, **k: 4.0)
    budget, calib, scale = mod._budget_for_machine()
    assert (budget, scale) == (90.0, 1.0), "标称工况下预算就是 90ms，不许偷偷放宽"

    monkeypatch.setattr(mod, "_calib_ms", lambda *a, **k: 12.0)
    budget, calib, scale = mod._budget_for_machine()
    assert abs(scale - 3.0) < 1e-9, f"标定慢 3 倍 → 倍数应 3，实际 {scale}"
    assert abs(budget - 270.0) < 1e-6, f"预算应同步放大到 270ms，实际 {budget}"

    # 比标称更快（新机器）不能把预算压到 90 以下
    monkeypatch.setattr(mod, "_calib_ms", lambda *a, **k: 1.0)
    budget, _c, scale = mod._budget_for_machine()
    assert budget == 90.0 and scale == 1.0, "更快也不能把预算压下去（否则新机器上乱红）"
