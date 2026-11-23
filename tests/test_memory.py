"""内存红线：这些东西**不许**随运行时长增长。

以前"跑一夜会不会涨爆"只能跑一夜才知道；4GB 的 RK3566 上炸的是整台机器。
这里把三处最容易被写坏的地方钉成用例：

  1. FSM 事件日志（每帧都可能追加，必须自己截断）
  2. 服务端帧历史（deque 的 maxlen 是唯一的闸门，改了配置就得仍然成立）
  3. 检测器的 Python 堆（每帧若干临时数组，只要有人把某一帧存进 self 就露馅）

第 3 条只看 Python 堆：OpenCV 的原生分配 tracemalloc 看不见，所以它抓的是
"有人把 frame 存到实例上"这类真正会导致线性增长的错误。
"""

from __future__ import annotations

import gc
import tracemalloc

import cv2
import numpy as np

from autogame.config import AppConfig, DecisionConfig
from autogame.decision.fsm import DecisionEngine
from autogame.executor import DryRunExecutor
from autogame.executor.base import ExecutorConfig
from autogame.server.hub import Hub
from autogame.vision.detector import Detector
from autogame.vision.pipeline import VisionPipeline


def _engine() -> DecisionEngine:
    dcfg = DecisionConfig()
    dcfg.autonomy.enabled = False
    return DecisionEngine(dcfg, DryRunExecutor(
        ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000)))


def _frame(w: int = 640, h: int = 360, i: int = 0) -> np.ndarray:
    f = np.full((h, w, 3), (30, 26, 24), np.uint8)
    x = 40 + (i * 5) % (w - 80)
    cv2.rectangle(f, (x - 4, 120), (x + 4, 220), (60, 200, 240), -1)
    cv2.rectangle(f, (x - 2, 126), (x + 2, 214), (255, 255, 255), -1)
    return f


def test_fsm_event_log_is_bounded():
    """FSM 每帧都可能发事件：一夜下来必须有上限，否则就是线性泄漏。"""
    eng = _engine()
    eng.start()
    for i in range(2000):
        eng._emit_event("unit_test", i=i)
    assert len(eng.events) <= 400, f"事件日志涨到 {len(eng.events)} 条，没有截断"


def test_hub_frame_history_is_bounded():
    """帧历史是给仪表盘画曲线的，超过 history_size 必须丢最旧的。"""
    cfg = AppConfig()
    cfg.server.event_log = ""
    hub = Hub(cfg, DryRunExecutor(cfg.executor), VisionPipeline(cfg.vision))
    ok, buf = cv2.imencode(".jpg", _frame())
    assert ok
    jpeg = buf.tobytes()
    for i in range(cfg.server.history_size + 150):
        hub.on_frame(jpeg, 1000 + i * 100)
    assert len(hub._history) <= cfg.server.history_size, \
        f"帧历史 {len(hub._history)} 超过上限 {cfg.server.history_size}"


def test_detector_python_heap_is_flat():
    """检测器每帧都造临时数组，但**不许**把帧本身留在实例上。

    只看稳定期（前 60 帧算热身：模板缩放/首帧标定会一次性占内存），
    稳定期再跑 200 帧，Python 堆增长要接近 0。抓到过的最典型错误是
    `self._last_frame = frame` —— 一帧 1080p 就是 6MB，一小时就是几 GB。
    """
    det = Detector(AppConfig().vision)
    for i in range(60):
        det.process(_frame(i=i), 1000 + i * 100)
    gc.collect()
    tracemalloc.start()
    base, _ = tracemalloc.get_traced_memory()
    for i in range(200):
        det.process(_frame(i=i), 7000 + i * 100)
    gc.collect()
    now, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    grown_mb = (now - base) / 1e6
    assert grown_mb < 3.0, (
        f"检测器稳定期堆增长 {grown_mb:.1f}MB（峰值 {peak / 1e6:.0f}MB）—— "
        f"有一帧被留在实例上了？"
    )
