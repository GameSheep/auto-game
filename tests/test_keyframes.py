"""关键帧 —— "刚才那一帧"必须有东西可指。

对话里那条要求是「最近 60 秒事件记录（识别结果、判断依据、实际动作、关键帧）」。
文字记录解决"发生了什么"，关键帧解决"当时画面长什么样"：光柱在屏幕哪一侧、
界面是不是结算页、是不是有弹窗挡住 —— 这些用文字描述永远会漏。

三条约束，用例就是照这三条钉的：

1. **不能每帧都存**。1Hz 全存一天能把 eMMC 写穿，所以只存"画面变了"和"出事了"。
2. **存了要能找到**。文件名带毫秒时间戳，按时间取最近的一张，取不到退回当前帧。
3. **写盘失败不能拖死决策链**。盘满、没权限的时候截图失败是小事，停下来才是大事。
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from autogame.assistant.context import RecentLog
from autogame.config import AppConfig
from autogame.executor import DryRunExecutor
from autogame.server.hub import Hub
from autogame.vision.pipeline import VisionPipeline


def _jpeg(color: int = 128) -> bytes:
    img = np.full((60, 80, 3), color, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _hub(tmp_path, *, recent: bool = True) -> Hub:
    cfg = AppConfig()
    cfg.server.event_log = str(tmp_path / "logs" / "events.jsonl")
    cfg.executor.min_action_interval_ms = 0
    hub = Hub(cfg, DryRunExecutor(cfg.executor), VisionPipeline(cfg.vision))
    if recent:
        hub.recent = RecentLog()
    return hub


def _scene(pillars: int = 0, gate: str = ""):
    return SimpleNamespace(in_game=True, pillars=[object()] * pillars, aframe=None,
                           gate_reason=gate, monsters=[], w=960, h=540)


def test_scene_change_saves_a_keyframe_and_records_it(tmp_path):
    hub = _hub(tmp_path)
    hub._latest_jpeg = _jpeg()

    hub._note_scene(_scene(pillars=0))
    assert hub.recent.frame_ref() is not None, "画面变了却没留图"
    path = hub.recent.frame_ref()["path"]
    assert path.endswith(".jpg")
    assert (tmp_path / "logs" / "frames" / path.split("frames")[-1].lstrip("\\/")).exists()


def test_same_scene_does_not_write_again(tmp_path):
    hub = _hub(tmp_path)
    hub._latest_jpeg = _jpeg()
    hub._note_scene(_scene(pillars=1))
    first = hub.recent.frame_ref()["path"]
    hub._last_keyframe_ts = 0            # 排除最小间隔的影响，只看"有没有变化"
    hub._note_scene(_scene(pillars=1))
    assert hub.recent.frame_ref()["path"] == first


def test_minimum_gap_prevents_writing_on_every_flicker(tmp_path):
    """光柱数来回跳的时候不能每帧写一张 —— 那等于每帧都存。"""
    hub = _hub(tmp_path)
    hub._latest_jpeg = _jpeg()
    hub._keyframe_min_gap_s = 60.0
    hub._note_scene(_scene(pillars=1))
    assert hub._last_keyframe_ts > 0
    before = len(list(hub._frame_dir.glob("f-*.jpg")))
    hub._note_scene(_scene(pillars=2))          # 变化了，但太快
    assert len(list(hub._frame_dir.glob("f-*.jpg"))) == before
    hub.save_keyframe("forced", force=True)      # 事故类可以无视间隔
    assert len(list(hub._frame_dir.glob("f-*.jpg"))) == before + 1


def test_notable_event_leaves_a_frame_even_without_scene_change(tmp_path):
    """急停/卡死这类事故，画面可能一帧都没变，但那一帧正是要看的。"""
    hub = _hub(tmp_path)
    hub._latest_jpeg = _jpeg()
    hub._keyframe_min_gap_s = 600.0
    hub._on_fsm_event({"event": "panic", "state": "PANIC", "reason": "测试"})
    assert hub.keyframe() is not None
    assert "panic" in hub.keyframe().name


def test_ordinary_events_do_not_write_frames(tmp_path):
    hub = _hub(tmp_path)
    hub._latest_jpeg = _jpeg()
    hub._on_fsm_event({"event": "explore_move", "state": "EXPLORING"})
    assert hub.keyframe() is None, "走路也截图的话，关键帧就不关键了"


def test_old_frames_are_pruned(tmp_path):
    hub = _hub(tmp_path)
    hub._latest_jpeg = _jpeg()
    hub._keyframe_limit = 3
    for _ in range(6):
        hub.save_keyframe("x", force=True)
        time.sleep(0.002)
    assert len(list(hub._frame_dir.glob("f-*.jpg"))) == 3


def test_keyframe_picks_the_nearest_to_a_timestamp(tmp_path):
    hub = _hub(tmp_path)
    hub._latest_jpeg = _jpeg()
    a = hub.save_keyframe("a", force=True)
    time.sleep(0.01)
    b = hub.save_keyframe("b", force=True)
    assert hub.keyframe(a["ts"]) == hub._frame_dir / f"f-{int(a['ts'] * 1000)}-a.jpg" or \
        hub.keyframe(a["ts"]).name.endswith("-a.jpg")
    assert hub.keyframe(b["ts"]).name.endswith("-b.jpg")
    assert hub.keyframe().name.endswith("-b.jpg"), "不传就是最近一张"


def test_frame_write_failure_does_not_raise(tmp_path, monkeypatch):
    """盘满了也不能让决策链挂掉 —— 截图是附加品，不是运行前提。"""
    hub = _hub(tmp_path)
    hub._latest_jpeg = _jpeg()

    def boom(self, data):
        raise OSError("No space left on device")

    monkeypatch.setattr("pathlib.Path.write_bytes", boom)
    assert hub.save_keyframe("x", force=True) is None
    hub._note_scene(_scene(pillars=3))     # 不能抛


def test_no_frame_yet_is_not_an_error(tmp_path):
    hub = _hub(tmp_path)
    assert hub.save_keyframe("x", force=True) is None
    assert hub.keyframe() is None


def test_http_frame_endpoint_returns_jpeg(tmp_path):
    from fastapi.testclient import TestClient

    from autogame.server.app import create_app

    cfg = AppConfig()
    cfg.server.event_log = str(tmp_path / "logs" / "events.jsonl")
    cfg.server.token = "t"
    cfg.executor.min_action_interval_ms = 0
    hub = Hub(cfg, DryRunExecutor(cfg.executor), VisionPipeline(cfg.vision))
    hub.recent = RecentLog()            # 关键帧只为"给助手看"而存在
    hub._latest_jpeg = _jpeg()
    hub.save_keyframe("scene", force=True)
    app = create_app(cfg, hub, "<h1>x</h1>")
    c = TestClient(app)
    r = c.get("/api/assistant/frame", headers={"X-Token": "t"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/jpeg")
    assert r.content[:2] == b"\xff\xd8"


def test_http_frame_endpoint_falls_back_to_live_frame(tmp_path):
    from fastapi.testclient import TestClient

    from autogame.server.app import create_app

    cfg = AppConfig()
    cfg.server.event_log = str(tmp_path / "logs" / "events.jsonl")
    cfg.server.token = "t"
    cfg.executor.min_action_interval_ms = 0
    hub = Hub(cfg, DryRunExecutor(cfg.executor), VisionPipeline(cfg.vision))
    hub._latest_jpeg = _jpeg()
    app = create_app(cfg, hub, "<h1>x</h1>")
    r = TestClient(app).get("/api/assistant/frame", headers={"X-Token": "t"})
    assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"


def test_no_frame_returns_404(tmp_path):
    from fastapi.testclient import TestClient

    from autogame.server.app import create_app

    cfg = AppConfig()
    cfg.server.event_log = str(tmp_path / "logs" / "events.jsonl")
    cfg.server.token = "t"
    cfg.executor.min_action_interval_ms = 0
    hub = Hub(cfg, DryRunExecutor(cfg.executor), VisionPipeline(cfg.vision))
    app = create_app(cfg, hub, "<h1>x</h1>")
    r = TestClient(app).get("/api/assistant/frame", headers={"X-Token": "t"})
    assert r.status_code == 404


def test_fsm_events_reach_the_recent_log_even_without_event_file(tmp_path):
    """不开审计文件时，急停/恢复也必须出现在"最近一分钟"里。

    之前 FSM 的事件回调只在 `event_log` 非空时才挂上 —— 于是关掉审计日志的
    机器上，助手看到的现场里**没有急停、没有卡死、没有进恢复**，
    最该被解释的几件事恰好全丢了。
    """
    hub = _hub(tmp_path)
    hub._event_log_path = None          # 真的关掉审计文件这条路
    hub.recent.clear()
    hub._on_fsm_event({"event": "stuck_alarm", "state": "RECOVERING", "reason": "打转"})
    assert any(it["kind"] == "stuck_alarm" and "打转" in str(it["data"])
               for it in hub.recent.items())


@pytest.mark.parametrize("tag,expect", [("scene", "scene"), ("", "frame"),
                                        ("a b/c", "abc")])
def test_tags_are_filesystem_safe(tmp_path, tag, expect):
    hub = _hub(tmp_path)
    hub._latest_jpeg = _jpeg()
    hub.save_keyframe(tag, force=True)
    assert hub.keyframe().name.split("-", 2)[-1].startswith(expect)


def test_no_assistant_means_no_keyframes(tmp_path):
    """关键帧唯一的用途是"你刚才说的是哪一帧"。

    没装助手就没人会来取，存下来只是白写盘 —— 以前跑测试会在仓库 logs/ 下
    堆出几十张没人看的 jpg，就是缺了这一条。
    """
    hub = _hub(tmp_path, recent=False)
    hub._latest_jpeg = _jpeg()
    assert hub.save_keyframe("x", force=True) is None
    assert not list(hub._frame_dir.glob("f-*.jpg"))


def test_no_assistant_means_no_keyframes(tmp_path):
    """关键帧唯一的用途是"你刚才说的是哪一帧"。

    没装助手就没人会来取，存下来只是白写盘 —— 以前跑测试会在仓库 logs/ 下
    堆出几十张没人看的 jpg，就是缺了这一条。
    """
    hub = _hub(tmp_path, recent=False)
    hub._latest_jpeg = _jpeg()
    assert hub.save_keyframe("x", force=True) is None
    assert not list(hub._frame_dir.glob("f-*.jpg"))
