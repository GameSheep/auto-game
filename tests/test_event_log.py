"""事件持久化测试 —— JSONL 审计日志 + 轮转。"""

import json
from pathlib import Path

from autogame.config import AppConfig, ExecutorConfig
from autogame.executor import DryRunExecutor
from autogame.server.hub import Hub
from autogame.vision.pipeline import VisionPipeline


def _mk_hub(tmp_path: Path, max_mb: float = 5.0) -> Hub:
    cfg = AppConfig()
    cfg.server.event_log = str(tmp_path / "logs" / "events.jsonl")
    cfg.server.event_log_max_mb = max_mb
    return Hub(cfg, DryRunExecutor(ExecutorConfig()), VisionPipeline(cfg.vision))


def test_event_log_written(tmp_path):
    hub = _mk_hub(tmp_path)
    hub.fsm.start()
    log_file = Path(hub.cfg.server.event_log)
    assert log_file.exists()
    lines = [json.loads(l) for l in log_file.read_text(encoding="utf-8").splitlines()]
    assert any(e["event"] == "started" for e in lines)


def test_event_log_rotates(tmp_path):
    hub = _mk_hub(tmp_path, max_mb=0.1)  # 100KB 触发轮转
    hub.fsm.start()
    for i in range(3000):
        hub.fsm._emit_event("noise", i=i)
    log_file = Path(hub.cfg.server.event_log)
    rotated = log_file.with_suffix(log_file.suffix + ".1")
    assert rotated.exists(), "超限后必须滚动出 .1 文件"
    assert log_file.exists()
    for f in (log_file, rotated):
        for line in f.read_text(encoding="utf-8").splitlines():
            json.loads(line)


def test_event_log_survives_fsm_growing_list(tmp_path):
    """FSM 事件列表裁剪（>400 清理）后回调直写保证零丢失。"""
    hub = _mk_hub(tmp_path)
    hub.fsm.start()
    for i in range(500):
        hub.fsm._emit_event("noise", i=i)
    log_file = Path(hub.cfg.server.event_log)
    events = [json.loads(l) for l in log_file.read_text(encoding="utf-8").splitlines()]
    noises = [e for e in events if e["event"] == "noise"]
    assert len(noises) == 500, f"回调直写不得丢事件，实际 {len(noises)}"


def test_event_log_disabled(tmp_path):
    cfg = AppConfig()
    cfg.server.event_log = None
    hub = Hub(cfg, DryRunExecutor(ExecutorConfig()), VisionPipeline(cfg.vision))
    hub.fsm.start()
    assert not (tmp_path / "logs").exists()
