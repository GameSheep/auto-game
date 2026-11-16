"""补充测试 —— OCR 白名单模糊匹配 / 配置回环 / 协议边界。"""

import pytest
from pydantic import ValidationError

from autogame.config import AppConfig, load_config
from autogame.protocol import SceneState, unpack_message
from autogame.vision.ocr import NullOCR, fuzzy_contains


# ---------------- 白名单模糊匹配 ----------------

def test_fuzzy_exact_substring():
    assert fuzzy_contains("拾取 破城斧", ["破城斧"], 78) == "破城斧"


def test_fuzzy_partial_with_rapidfuzz_or_fallback():
    hit = fuzzy_contains("破 城 斧（传奇）", ["破城斧"], 78)
    assert hit == "破城斧"


def test_fuzzy_no_match():
    assert fuzzy_contains(" rusty sword ", ["破城斧"], 78) is None
    assert fuzzy_contains("", ["破城斧"], 78) is None


def test_fuzzy_multiple_candidates_first_match():
    hit = fuzzy_contains("卓越水晶剑", ["水晶剑", "破城斧"], 78)
    assert hit == "水晶剑"


def test_null_ocr_no_text():
    assert NullOCR().read_text(None, 0, 0, 10, 10) == ""


# ---------------- 配置模型 ----------------

def test_config_defaults_valid():
    cfg = AppConfig()
    assert cfg.capture.fps >= 0.5
    assert cfg.server.token is None  # load_config 阶段才生成
    assert cfg.executor.backend == "dryrun"
    assert cfg.decision.tick_hz == 1.0


def test_config_rejects_bad_values():
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"capture": {"fps": 100}})  # >30 上限
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"executor": {"backend": "notexist"}})
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"decision": {"approach_timeout_s": 0.5}})  # <2 下限


def test_config_yaml_roundtrip(tmp_path):
    import yaml

    cfg = AppConfig()
    data = yaml.safe_load(cfg.model_dump_json())
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    cfg2 = load_config(p)
    assert cfg2.capture.fps == cfg.capture.fps
    assert cfg2.vision.hsv == cfg.vision.hsv
    assert cfg2.decision.approach_mode == cfg.decision.approach_mode


def test_load_config_env_token(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOGAME_TOKEN", "env-token-xyz")
    cfg = load_config(tmp_path / "missing.yaml")
    assert cfg.server.token == "env-token-xyz"


# ---------------- 协议边界 ----------------

def test_oversized_scene_roundtrip():
    import json as _json

    scene = SceneState(
        ts_ms=1, w=3840, h=2160, character=(1920, 1080),
        pillars=[],
    )
    msg = {"type": "scene", **scene.model_dump()}
    kind, payload = unpack_message(_json.dumps(msg))
    assert kind == "json"
    assert payload["w"] == 3840


def test_binary_frame_max_size_boundary():
    from autogame.protocol import pack_jpeg_frame

    big = b"\xff" * (2**20)
    raw = pack_jpeg_frame(big, 3840, 2160, 4294967295)  # ts 上限
    kind, payload = unpack_message(raw)
    assert len(payload["jpeg"]) == 2**20
    assert payload["ts_ms"] == 4294967295
