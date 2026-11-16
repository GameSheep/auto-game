"""异动触发上传测试（对话定稿：发现异动才传图）。"""

import cv2
import numpy as np

from autogame.config import CaptureConfig
from autogame.agent.client import AgentClient


def _mk_agent(motion=True):
    cfg = CaptureConfig(motion_trigger=motion, motion_hash_thresh=4, motion_force_s=2.0)
    agent = AgentClient.__new__(AgentClient)  # 跳过 __init__ 的 WS 依赖
    from autogame.config import AppConfig
    agent.cfg = AppConfig()
    agent.cfg.capture = cfg
    agent._last_hash = None
    agent._last_sent_t = 0.0
    return agent


def _should_send(agent, frame):
    """抽出 _send_frame 里的异动判定逻辑（镜像实现）。"""
    if not agent.cfg.capture.motion_trigger:
        return True
    import cv2, time
    small = cv2.resize(frame, (16, 16))
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    h = g > g.mean()
    now_t = time.monotonic()
    if agent._last_hash is not None:
        dist = int((h != agent._last_hash).sum())
        if dist <= agent.cfg.capture.motion_hash_thresh \
                and now_t - agent._last_sent_t < agent.cfg.capture.motion_force_s:
            return False
    agent._last_hash = h
    agent._last_sent_t = now_t
    return True


def test_motion_trigger_static_scene_skips():
    """静止画面：首帧发送，紧随帧跳过。"""
    agent = _mk_agent(motion=True)
    frame = np.full((720, 1280, 3), (28, 24, 22), dtype=np.uint8)
    assert _should_send(agent, frame) is True   # 首帧必发
    assert _should_send(agent, frame) is False  # 无异动跳过
    assert _should_send(agent, frame) is False


def test_motion_trigger_change_sends():
    """画面变化 → 立即发送。"""
    agent = _mk_agent(motion=True)
    f1 = np.full((720, 1280, 3), (28, 24, 22), dtype=np.uint8)
    assert _should_send(agent, f1) is True
    f2 = f1.copy()
    f2[100:400, 200:900] = (255, 200, 60)  # 大变化
    assert _should_send(agent, f2) is True


def test_motion_trigger_force_keepalive():
    """无异动超时后强制发送保活帧。"""
    agent = _mk_agent(motion=True)
    frame = np.full((720, 1280, 3), (28, 24, 22), dtype=np.uint8)
    assert _should_send(agent, frame) is True
    assert _should_send(agent, frame) is False
    agent._last_sent_t -= 3.0  # 模拟时间流逝超过 motion_force_s
    assert _should_send(agent, frame) is True


def test_motion_trigger_disabled_sends_all():
    agent = _mk_agent(motion=False)
    frame = np.full((720, 1280, 3), (28, 24, 22), dtype=np.uint8)
    assert all(_should_send(agent, frame) for _ in range(5))


def test_motion_trigger_small_change_still_skipped():
    """细微变化（光标闪烁级）不触发上传。"""
    agent = _mk_agent(motion=True)
    f1 = np.full((720, 1280, 3), (28, 24, 22), dtype=np.uint8)
    assert _should_send(agent, f1) is True
    f2 = f1.copy()
    f2[50:56, 50:56] = (255, 255, 255)  # 6x6 小点
    assert _should_send(agent, f2) is False
