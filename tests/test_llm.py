"""Ollama 决策顾问测试 —— mock HTTP 服务验证（不需要真实 Ollama）。"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from autogame.config import DecisionConfig, ExecutorConfig, LLMConfig
from autogame.decision.fsm import DecisionEngine
from autogame.decision.llm_advisor import LLMAdvisor
from autogame.executor import DryRunExecutor


class _OllamaStub(BaseHTTPRequestHandler):
    """最小 Ollama /api/chat 桩服务。"""

    response_content = '{"action": "wait", "reason": "连续放弃，建议等待"}'

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        assert body["model"] == "test-model"
        assert body["stream"] is False
        content = json.dumps({
            "message": {"content": _OllamaStub.response_content}
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, *args):  # 静默
        pass


def _start_stub():
    server = HTTPServer(("127.0.0.1", 0), _OllamaStub)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, port


def _mk_engine(llm_cfg: LLMConfig):
    dcfg = DecisionConfig(tick_hz=10.0, llm=llm_cfg)
    ecfg = ExecutorConfig(min_action_interval_ms=0, max_actions_per_sec=1000)
    ex = DryRunExecutor(ecfg)
    return DecisionEngine(dcfg, ex), ex


def _scene(ts=1000):
    from autogame.protocol import Pillar, SceneState

    return SceneState(
        ts_ms=ts, w=1280, h=720, character=(640, 360),
        pillars=[Pillar(x=900, y=455, cx=900, cy=400, w=14, h=110, color="gold",
                        area=1500, distance=300, whitelisted=True)],
        frame_delta=50.0,
    )


def test_parse_advice_variants():
    advisor = LLMAdvisor(LLMConfig(enabled=True))
    assert advisor._parse_advice('{"action":"wait","reason":"r"}')["action"] == "wait"
    assert advisor._parse_advice('好的\n```json\n{"action":"continue","reason":"ok"}\n```')[
        "action"] == "continue"
    import pytest

    with pytest.raises(ValueError):
        advisor._parse_advice("没有 JSON 的回复")
    with pytest.raises(ValueError):
        advisor._parse_advice('{"foo": 1}')


def test_should_consult_gates():
    advisor = LLMAdvisor(LLMConfig(enabled=False, trigger_abandons=2))
    advisor.note_abandon()
    advisor.note_abandon()
    assert not advisor.should_consult(), "未启用不得咨询"
    advisor2 = LLMAdvisor(LLMConfig(enabled=True, trigger_abandons=3, cooldown_s=60))
    advisor2.note_abandon()
    advisor2.note_abandon()
    assert not advisor2.should_consult(), "未达触发次数不得咨询"
    advisor2.note_abandon()
    assert advisor2.should_consult()
    advisor2.note_success()
    assert not advisor2.should_consult(), "拾取成功后计数归零"


def test_fsm_triggers_llm_after_consecutive_abandons():
    server, port = _start_stub()
    try:
        llm = LLMConfig(enabled=True, base_url=f"http://127.0.0.1:{port}",
                        model="test-model", trigger_abandons=2, cooldown_s=0.5)
        eng, ex = _mk_engine(llm)
        eng.start()
        # 连续超时放弃 2 次（每次超时需要不同时间戳）
        eng.on_scene(_scene(1000))
        eng.on_scene(_scene(11000))   # 超时 → 放弃1
        eng.on_scene(_scene(41000))   # 冷却结束，但位置被拉黑 → 换新目标（同位置不选）
        eng.pillars_blacklist_clear = None if False else None
        # 手动清拉黑再触发第二次放弃
        eng._blacklist.clear()
        eng.on_scene(_scene(45000))
        eng.on_scene(_scene(75000))   # 超时 → 放弃2 → 触发咨询
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if any(e["event"] == "llm_advice" for e in eng.events):
                break
            time.sleep(0.05)
        adv = [e for e in eng.events if e["event"] == "llm_advice"]
        assert adv, "连续放弃必须触发 LLM 咨询"
        assert adv[0]["advice"]["action"] == "wait"
        assert adv[0]["allowed"] is True
        assert eng.advisor.queries == 1
    finally:
        server.shutdown()


def test_fsm_wait_advice_extends_cooldown():
    server, port = _start_stub()
    try:
        llm = LLMConfig(enabled=True, base_url=f"http://127.0.0.1:{port}",
                        model="test-model", trigger_abandons=1, cooldown_s=0.5)
        eng, ex = _mk_engine(llm)
        eng.start()
        eng.on_scene(_scene(1000))
        eng.on_scene(_scene(11000))   # 一次放弃即触发
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if eng.advisor.last_advice:
                break
            time.sleep(0.05)
        time.sleep(0.3)  # 等回调写入冷却
        # wait 建议 → 冷却至少延长到放弃时刻 + 3s
        assert eng._cooldown_until >= 11.0 + 2.5, "wait 建议必须延长冷却"
    finally:
        server.shutdown()


def test_llm_disabled_zero_queries():
    llm = LLMConfig(enabled=False)
    eng, ex = _mk_engine(llm)
    eng.start()
    eng.on_scene(_scene(1000))
    eng.on_scene(_scene(11000))
    eng.on_scene(_scene(41000))
    eng._blacklist.clear()
    eng.on_scene(_scene(45000))
    eng.on_scene(_scene(75000))
    assert eng.advisor.queries == 0, "关闭时绝不发请求"
