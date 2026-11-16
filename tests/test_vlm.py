"""视觉模型顾问（VLM）测试 —— 桩服务 + 白名单/预算/熔断，全部离线。"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

from autogame.config import VLMConfig
from autogame.decision.vlm import VisionAdvisor


class _VLMStub(BaseHTTPRequestHandler):
    """最小 Ollama /api/chat 多模态桩服务。"""

    reply = json.dumps({
        "action": "walk_to", "target": [900, 420],
        "scene": "battle", "confidence": 0.8, "reason": "右侧有光柱",
    })
    seen_images = 0

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        msg = body["messages"][0]
        assert body["model"] == "test-vlm"
        assert body["stream"] is False
        assert isinstance(msg.get("images"), list) and msg["images"], "必须带图片"
        type(self).seen_images += 1
        content = json.dumps({"message": {"content": self.reply}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, *args):  # 静默
        pass


def _start_stub() -> tuple[HTTPServer, int]:
    srv = HTTPServer(("127.0.0.1", 0), _VLMStub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _jpeg(w: int = 640, h: int = 360) -> bytes:
    img = np.zeros((h, w, 3), np.uint8)
    cv2.rectangle(img, (w // 3, h // 3), (w // 2, h // 2), (0, 200, 255), -1)
    ok, enc = cv2.imencode(".jpg", img)
    assert ok
    return enc.tobytes()


def _wait(cond, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


def _advisor(**over) -> VisionAdvisor:
    cfg = VLMConfig(enabled=True, model="test-vlm", min_interval_s=0.5)
    for k, v in over.items():
        setattr(cfg, k, v)
    adv = VisionAdvisor(cfg, frame_source=lambda: _jpeg())
    return adv


# ---------------- 建议清洗（安全边界） ----------------

def test_sanitize_rejects_unknown_action():
    adv = _advisor()
    out = adv.sanitize({"action": "press_pickup", "confidence": 0.9})
    assert out["valid"] is False and "白名单" in out["why"]


def test_sanitize_rejects_low_confidence():
    adv = _advisor(min_confidence=0.6)
    out = adv.sanitize({"action": "esc", "confidence": 0.3})
    assert out["valid"] is False and "置信度" in out["why"]


def test_sanitize_clamps_target_into_screen():
    """模型幻觉坐标（负数/超大）必须钳到画面内，不能点到屏幕外。"""
    adv = _advisor()
    out = adv.sanitize({"action": "walk_to", "target": [-999, 99999], "confidence": 0.9},
                       {"screen_w": 1280, "screen_h": 720})
    assert out["valid"] is True
    assert out["target"] == [0, 719]


def test_sanitize_rejects_non_whitelisted_key():
    adv = _advisor()
    out = adv.sanitize({"action": "key", "key": "f12", "confidence": 0.9})
    assert out["valid"] is False and "按键" in out["why"]


def test_sanitize_accepts_safe_key():
    adv = _advisor()
    out = adv.sanitize({"action": "key", "key": "space", "confidence": 0.9})
    assert out["valid"] is True and out["key"] == "space"


# ---------------- 图像预处理 ----------------

def test_encode_downscales_to_max_width():
    adv = _advisor(max_width=320)
    b64 = adv._encode(_jpeg(1920, 1080))
    import base64

    raw = base64.b64decode(b64)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == 320
    assert img.shape[0] == 180  # 等比


# ---------------- 端到端（桩服务） ----------------

def test_request_returns_advice_from_stub():
    srv, port = _start_stub()
    try:
        adv = _advisor(base_url=f"http://127.0.0.1:{port}")
        assert adv.request({"state": "RECOVERING"}, now=1000.0) is True
        assert _wait(lambda: adv.take_advice() is not None)
        advice = adv.last_advice
        assert advice["action"] == "walk_to" and advice["valid"] is True
        assert advice["target"] == [900, 420]
        assert adv.stats["ok"] == 1
    finally:
        srv.shutdown()


def test_budget_min_interval_blocks_second_call():
    srv, port = _start_stub()
    try:
        adv = _advisor(base_url=f"http://127.0.0.1:{port}", min_interval_s=30.0)
        assert adv.request({}, now=1000.0) is True
        assert _wait(lambda: not adv._busy)
        assert adv.request({}, now=1001.0) is False, "最小间隔内必须拒绝"
    finally:
        srv.shutdown()


def test_budget_per_minute_blocks():
    srv, port = _start_stub()
    try:
        adv = _advisor(base_url=f"http://127.0.0.1:{port}", max_calls_per_min=1,
                       min_interval_s=0.5)
        assert adv.request({}, now=1000.0) is True
        assert _wait(lambda: not adv._busy)
        assert adv.request({}, now=1001.0) is False, "每分钟预算用尽必须拒绝"
    finally:
        srv.shutdown()


def test_circuit_breaker_after_failures():
    adv = _advisor(failure_threshold=1, cooldown_after_failure_s=60.0)

    def _boom(*a, **k):
        raise RuntimeError("model down")

    adv._post = _boom  # type: ignore[method-assign]
    assert adv.request({}, now=1000.0) is True
    assert _wait(lambda: not adv._busy)
    assert adv.stats["fail"] == 1
    assert adv.available(time.monotonic()) is False, "熔断期内不得再调用"
    assert adv.snapshot()["tripped"] is True


def test_disabled_never_calls():
    adv = _advisor(enabled=False)
    assert adv.request({}) is False
    assert adv.stats["calls"] == 0
