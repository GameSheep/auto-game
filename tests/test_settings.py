"""设置页后端测试 —— 配置热更新 / 预设 / 标定反馈 / 视觉看门狗 / 告警。

这里验证的是"页面点了之后程序真的变了"，而不是"接口返回 200"。
历史坑：替换 cfg 对象导致 FSM 还在读旧配置 —— 页面显示已生效，程序纹丝不动。
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from autogame.config import AppConfig, load_config, runtime_overlay_path
from autogame.executor import DryRunExecutor
from autogame.executor import ExecutorConfig
from autogame.presets import PRESETS, apply_preset, preset_patch
from autogame.protocol import Pillar, SceneState
from autogame.server.app import create_app
from autogame.server.hub import Hub
from autogame.server.notify import Notifier
from autogame.server.settings import ConfigError, apply_patch, public_config
from autogame.vision.pipeline import VisionPipeline

TOKEN = "test-token"


def _cfg(**over) -> AppConfig:
    cfg = AppConfig()
    cfg.server.token = TOKEN
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _hub(cfg: AppConfig) -> Hub:
    # 用 cfg.executor 本体（而不是另建一个）：执行器持有的是这个对象的引用，
    # 页面改配置要能作用到它身上 —— 这也是"原地改"测试的前提。
    cfg.executor.min_action_interval_ms = 0
    cfg.executor.max_actions_per_sec = 1000
    ex = DryRunExecutor(cfg.executor)
    return Hub(cfg, ex, VisionPipeline(cfg.vision))


def _client(cfg: AppConfig, hub: Hub, config_path: str = "") -> TestClient:
    app = create_app(cfg, hub, "<h1>dash</h1>", config_path=config_path)
    return TestClient(app)


def _h(token: str = TOKEN) -> dict:
    return {"X-Token": token}


# ═══════════════════════ 配置热更新 ═══════════════════════


def test_patch_is_applied_in_place_not_replaced():
    """必须原地改：FSM/执行器持有的是子配置对象引用，换对象等于没生效。"""
    cfg = _cfg()
    hub = _hub(cfg)
    decision_obj = hub.fsm.cfg
    executor_obj = hub.executor.cfg
    assert decision_obj is cfg.decision and executor_obj is cfg.executor

    # 测试环境默认关掉拟人化（conftest），这里打开以产生一次真实变更
    res = apply_patch(cfg, {"decision": {"humanize": {"enabled": True}}})
    assert "decision.humanize.enabled" in res["changed"]
    # 引用没变，值变了 → FSM 下一帧就能读到新值
    assert hub.fsm.cfg is decision_obj
    assert hub.fsm.cfg.humanize.enabled is True
    assert hub.fsm.human.cfg.enabled is True, "Humanizer 持有同一个对象，必须同步"
    assert hub.executor.cfg is executor_obj


def test_patch_validates_and_rejects_whole_batch():
    """校验失败整批不生效（半套配置比不配置更危险）。"""
    cfg = _cfg()
    before = cfg.decision.humanize.mood.temperature
    with pytest.raises(ConfigError):
        apply_patch(cfg, {"decision": {"humanize": {"mood": {"temperature": -5}}}})
    assert cfg.decision.humanize.mood.temperature == before


def test_patch_reports_restart_required_fields():
    cfg = _cfg()
    res = apply_patch(cfg, {"executor": {"backend": "rp2040"}})
    assert "executor.backend" in res["restart_required"], "执行器后端需要重启"


def test_public_config_hides_token():
    cfg = _cfg()
    d = public_config(cfg)
    assert d["server"]["token"] in ("***", "")


def test_api_patch_persists_overlay_and_survives_reload(tmp_path):
    """页面改的值要落到覆盖文件，并且下次 load_config 能读回来。"""
    base = tmp_path / "config.yaml"
    base.write_text("decision:\n  tick_hz: 1.0\n", encoding="utf-8")
    cfg = load_config(base)
    cfg.server.token = TOKEN
    hub = _hub(cfg)
    c = _client(cfg, hub, config_path=str(base))

    r = c.post("/api/config", headers=_h(),
               json={"patch": {"decision": {"humanize": {"mood": {"temperature": 0.9}}}}})
    assert r.status_code == 200, r.text
    assert "decision.humanize.mood.temperature" in r.json()["changed"]

    overlay = runtime_overlay_path(base)
    assert overlay.exists(), "必须写运行时覆盖文件"
    assert "0.9" in overlay.read_text(encoding="utf-8")

    # 重新加载：覆盖文件叠在 config.yaml 之上
    again = load_config(base)
    assert again.decision.humanize.mood.temperature == 0.9


def test_api_patch_rejects_bad_value_with_field_path():
    cfg = _cfg()
    hub = _hub(cfg)
    c = _client(cfg, hub)
    r = c.post("/api/config", headers=_h(),
               json={"patch": {"decision": {"humanize": {"mood": {"temperature": 99}}}}})
    assert r.status_code == 400
    assert "temperature" in r.text, "错误信息要能指到具体字段"


def test_local_mode_needs_no_token():
    """本地自用默认**不需要 token**：粘 token 只是给自己添堵。"""
    cfg = _cfg()
    assert cfg.server.require_token is False
    hub = _hub(cfg)
    c = _client(cfg, hub)
    assert c.get("/api/config").status_code == 200
    assert c.get("/api/calibrate").status_code == 200
    assert c.post("/api/test/plan").status_code == 200


def test_token_still_enforced_when_enabled():
    """要暴露到不可信网络时打开 require_token，行为与之前一致。"""
    cfg = _cfg()
    cfg.server.require_token = True
    hub = _hub(cfg)
    c = _client(cfg, hub)
    assert c.get("/api/config").status_code == 401
    assert c.get("/api/calibrate").status_code == 401
    assert c.post("/api/alerts/test").status_code == 401
    assert c.get("/api/config", headers=_h()).status_code == 200


# ═══════════════════════ 预设 ═══════════════════════


def test_presets_cover_three_modes():
    assert set(PRESETS) == {"stable", "human", "custom"}
    assert preset_patch("custom") == {}


def test_stable_preset_turns_humanize_off():
    cfg = _cfg()
    hub = _hub(cfg)
    apply_patch(cfg, preset_patch("stable"))
    assert cfg.decision.humanize.enabled is False
    assert cfg.decision.walk.enabled is False
    assert cfg.executor.mouse_path.enabled is False
    assert hub.fsm.human.cfg.enabled is False, "FSM 里的 Humanizer 必须同步关掉"
    assert hub.fsm.human.mood_active is False


def test_human_preset_makes_it_slower_and_more_random():
    cfg = _cfg()
    apply_patch(cfg, preset_patch("human"))
    h = cfg.decision.humanize
    assert h.enabled is True
    assert h.reaction_ms[1] >= 600, "反应时间上限要明显拉长"
    assert h.drift.hesitate_prob > 0.3
    assert h.mood.skip_prob >= 0.03
    assert cfg.executor.mouse_path.wind >= 4.0
    assert cfg.decision.walk.budget_s >= 3.5


def test_preset_api_applies_and_reports_changed():
    cfg = _cfg()
    hub = _hub(cfg)
    c = _client(cfg, hub)
    r = c.post("/api/preset/stable", headers=_h())
    assert r.status_code == 200
    assert len(r.json()["changed"]) > 5
    assert hub.fsm.human.cfg.enabled is False
    r2 = c.post("/api/preset/nope", headers=_h())
    assert r2.status_code == 404


# ═══════════════════════ 相机标定反馈 ═══════════════════════


def _fake_frame(w=1280, h=720) -> np.ndarray:
    """造一张"像游戏画面"的帧：暗背景 + 一根亮光柱。"""
    frame = np.full((h, w, 3), (28, 24, 22), dtype=np.uint8)
    frame[300:520, 640:656] = (36, 190, 255)
    frame[306:514, 645:651] = (255, 255, 255)
    return frame


def test_calibrate_without_frame_explains_why():
    cfg = _cfg()
    hub = _hub(cfg)
    c = _client(cfg, hub)
    r = c.get("/api/calibrate", headers=_h())
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert "尚无画面" in body["reason"]


def test_calibrate_returns_scores_and_advice():
    cfg = _cfg()
    hub = _hub(cfg)
    import cv2

    ok, enc = cv2.imencode(".jpg", _fake_frame())
    assert ok
    hub.on_frame(enc.tobytes(), 1000)
    c = _client(cfg, hub)
    body = c.get("/api/calibrate", headers=_h()).json()
    assert body["ok"] is True
    assert 0 <= body["scores"]["overall"] <= 100
    assert body["metrics"]["frame_w"] == 1280
    assert body["advice"], "必须给出可执行的建议"
    assert body["verdict"]


def test_calibrate_advice_points_at_blur():
    """糊掉的画面必须明确提示"先解决对焦"，而不是让用户去瞎调 HSV。"""
    cfg = _cfg()
    hub = _hub(cfg)
    import cv2

    blurry = cv2.GaussianBlur(_fake_frame(), (41, 41), 0)
    ok, enc = cv2.imencode(".jpg", blurry)
    hub.on_frame(enc.tobytes(), 1000)
    c = _client(cfg, hub)
    body = c.get("/api/calibrate", headers=_h()).json()
    assert any("模糊" in a for a in body["advice"]), body["advice"]


def test_sample_hsv_keeps_color_names():
    """HSV 的键就是颜色名，whitelist_colors 靠它工作 —— 采样不能瞎起名。"""
    cfg = _cfg()
    cfg.vision.whitelist_colors = ["gold"]
    hub = _hub(cfg)
    import cv2

    ok, enc = cv2.imencode(".jpg", _fake_frame())
    hub.on_frame(enc.tobytes(), 1000)
    c = _client(cfg, hub)
    body = c.post("/api/calibrate/hsv", headers=_h()).json()
    if body.get("ok"):
        assert "gold" in body["hsv"], f"颜色名被改掉了: {list(body['hsv'])}"
    else:
        pytest.skip(f"该测试帧未采到足够色簇: {body.get('reason')}")


# ═══════════════════════ 视觉看门狗 ═══════════════════════


def _scene(in_game: bool = True) -> SceneState:
    return SceneState(ts_ms=1000, w=1280, h=720, character=(640, 360),
                      pillars=[], in_game=in_game, frame_delta=50.0)


def test_watchdog_stops_when_no_frames():
    """完全没有新帧（相机被关/代理掉线）→ 自动停止。"""
    cfg = _cfg()
    cfg.watchdog.no_frame_stop_s = 1.0
    hub = _hub(cfg)
    hub.control("start")
    hub._latest_ts = time.time() - 10.0     # 10 秒没新帧
    hub.check_vision_lost()
    assert hub.fsm.state.value == "PAUSED"
    assert hub._auto_stopped is True
    assert any(e["event"] == "vision_lost" for e in hub.fsm.events)


def test_watchdog_waits_longer_for_menu_than_for_no_frames():
    """有帧但不在游戏里 → 必须等更久（autonomy 的 ESC 自救要时间）。"""
    cfg = _cfg()
    cfg.watchdog.no_game_stop_s = 5.0
    hub = _hub(cfg)
    hub.control("start")
    hub._latest_ts = time.time()                 # 帧是新鲜的
    hub._latest_scene = _scene(in_game=False)
    hub.check_vision_lost()
    assert hub.fsm.state.value != "PAUSED", "刚开始看不到游戏不能立刻停"
    # 假装已经持续 6 秒
    hub._vision_lost_since = time.time() - 6.0
    hub.check_vision_lost()
    assert hub.fsm.state.value == "PAUSED"


def test_watchdog_resets_when_picture_comes_back():
    cfg = _cfg()
    cfg.watchdog.no_game_stop_s = 5.0
    hub = _hub(cfg)
    hub.control("start")
    hub._latest_ts = time.time()
    hub._latest_scene = _scene(in_game=False)
    hub.check_vision_lost()
    assert hub._vision_lost_since > 0
    hub._latest_scene = _scene(in_game=True)
    hub.check_vision_lost()
    assert hub._vision_lost_since == 0.0
    assert hub.fsm.state.value != "PAUSED"


def test_watchdog_alert_only_mode_does_not_stop():
    cfg = _cfg()
    cfg.watchdog.no_frame_stop_s = 1.0
    cfg.watchdog.auto_stop = False
    hub = _hub(cfg)
    hub.control("start")
    hub._latest_ts = time.time() - 10.0
    hub.check_vision_lost()
    assert hub.fsm.state.value != "PAUSED", "auto_stop=False 时只告警不动手"


# ═══════════════════════ 告警 ═══════════════════════


def test_notify_rules_fire_once_and_respect_interval():
    from autogame.config import NotifyConfig, NotifyRule

    cfg = NotifyConfig(enabled=True, min_interval_s=60.0,
                       rules=[NotifyRule(name="刷图", metric="rechallenges", threshold=10)])
    n = Notifier(cfg)
    assert n.check({"rechallenges": 5}, now=1000.0) == []
    fired = n.check({"rechallenges": 12}, now=1001.0)
    assert len(fired) == 1 and fired[0]["rule"] == "刷图"
    assert n.check({"rechallenges": 13}, now=1002.0) == [], "一次性规则不得重复提醒"
    assert len(n.history) == 1


def test_notify_disabled_does_nothing():
    from autogame.config import NotifyConfig

    n = Notifier(NotifyConfig(enabled=False))
    assert n.check({"rechallenges": 999}, now=1.0) == []
    assert n.history == []


def test_metrics_include_bag_full_proxy():
    """有目标却一直捡不到 → 疑似包满（游戏玩法内程序判断不了的东西）。"""
    cfg = _cfg()
    hub = _hub(cfg)
    hub.control("start")
    hub._latest_scene = SceneState(
        ts_ms=1000, w=1280, h=720, character=(640, 360),
        pillars=[Pillar(x=700, y=400, cx=700, cy=350, w=14, h=100,
                        color="gold", area=900, distance=90.0)],
        frame_delta=50.0,
    )
    hub.metrics()                       # 第一次：建立计时
    hub._no_pick_since = time.time() - 200.0
    m = hub.metrics()
    assert m["no_pick_with_targets_s"] >= 150


def test_alert_test_endpoint_records_history():
    cfg = _cfg()
    hub = _hub(cfg)
    c = _client(cfg, hub)
    r = c.post("/api/alerts/test", headers=_h())
    assert r.status_code == 200
    assert r.json()["notify"]["history"], "测试告警要能在页面看到"


def test_settings_page_served():
    cfg = _cfg()
    hub = _hub(cfg)
    c = _client(cfg, hub)
    r = c.get("/settings")
    assert r.status_code == 200
    # 三页结构：参数（三档预设 + 三个 0-10 旋钮）/ 摄像头 / 告警
    for s in ("识别灵敏度", "执行速度", "安全兜底"):
        assert s in r.text, f"参数页缺少旋钮 {s}"
    for s in ("放飞自我", "稳定运行", "极致安全"):
        assert s in r.text, f"缺少预设档 {s}"
    assert "触发规则" in r.text, "告警页缺少规则表"
    assert "id=\"page-camera\"" in r.text


# ═══════════════════════ 试跑 / 直发 HID ═══════════════════════


def _scene_with_pillar() -> SceneState:
    return SceneState(
        ts_ms=1000, w=1280, h=720, character=(640, 360),
        pillars=[Pillar(x=900, y=455, cx=900, cy=400, w=14, h=110,
                        color="gold", area=1500, distance=320.0, whitelisted=True)],
        frame_delta=50.0,
    )


def test_plan_without_frame_explains():
    cfg = _cfg()
    hub = _hub(cfg)
    c = _client(cfg, hub)
    d = c.post("/api/test/plan").json()
    assert d["ok"] is False and "尚无画面" in d["reason"]


def test_plan_returns_commands_with_hid_preview():
    """试跑要能看到「接下来几步」以及每步对应的 RP2040 JSON / HID 报文。"""
    cfg = _cfg()
    hub = _hub(cfg)
    hub._latest_scene = _scene_with_pillar()
    c = _client(cfg, hub)
    d = c.post("/api/test/plan").json()
    assert d["ok"] is True
    assert d["scene"]["pillars"] == 1
    assert d["commands"], "有光柱时必须给出走位命令"
    first = d["commands"][0]
    assert first["op"] in ("mouse_click", "key_tap")
    assert "hid_hex" in first and "serial_json" in first
    if first["op"] == "mouse_click":
        assert len(first["hid_hex"].split()) == 6, "绝对鼠标报文是 6 字节"
        assert '"op":"mouse_click"' in first["serial_json"].replace(" ", "")


def test_plan_does_not_touch_real_executor():
    """试跑只算不按：真执行器不能收到任何命令。"""
    cfg = _cfg()
    hub = _hub(cfg)
    hub._latest_scene = _scene_with_pillar()
    c = _client(cfg, hub)
    before = hub.executor.sent
    c.post("/api/test/plan")
    assert hub.executor.sent == before, "试跑绝不能真的发命令"


def test_send_executes_on_real_executor():
    cfg = _cfg()
    hub = _hub(cfg)
    c = _client(cfg, hub)
    before = hub.executor.sent
    d = c.post("/api/test/send", json={"commands": [
        {"op": "mouse_click", "x": 700, "y": 500, "button": "left"},
        {"op": "key_tap", "key": "space", "hold_ms": 60},
    ]}).json()
    assert d["ok"] is True and d["sent"] == 2
    assert hub.executor.sent == before + 2, "真发必须打到执行器"
    assert all("hid_hex" in r for r in d["results"])


def test_send_marks_blind_pickup_but_does_not_block():
    """手动测拾取键是合理的（就是在验证这个键通不通），但要留痕。"""
    cfg = _cfg()
    hub = _hub(cfg)
    c = _client(cfg, hub)
    d = c.post("/api/test/send", json={"commands": [
        {"op": "key_tap", "key": "a", "hold_ms": 60}]}).json()
    r = d["results"][0]
    assert r["ok"] is True
    assert r["blind_risk"] is True
    assert any(e["event"] == "manual_pickup_test" for e in hub.fsm.events)


def test_send_rejects_empty_and_bad_body():
    cfg = _cfg()
    hub = _hub(cfg)
    c = _client(cfg, hub)
    assert c.post("/api/test/send", json={"commands": []}).json()["ok"] is False
    assert c.post("/api/test/send", json={}).status_code == 400


# ═══════════════════════ Bark 推送 ═══════════════════════


def test_bark_uses_url_encoded_path(monkeypatch):
    """Bark 是 GET /{key}/{title}/{body}，中文标题必须 URL 编码，否则 400。"""
    from autogame.config import NotifyConfig
    from autogame.server import notify as notify_mod

    cfg = NotifyConfig(enabled=True, webhook_format="bark",
                       bark_key="abc123", bark_server="https://api.day.app")
    n = Notifier(cfg)
    captured = {}

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, *a): return b'{"code":200}'

    def fake_urlopen(req, timeout=0):  # noqa: ARG001
        captured["url"] = req.full_url
        return _Resp()

    monkeypatch.setattr(notify_mod.urllib.request, "urlopen", fake_urlopen)
    n._send_bark("刷图完成", "刷了 100 次")
    url = captured["url"]
    assert url.startswith("https://api.day.app/abc123/")
    assert "%E5%88%B7%E5%9B%BE" in url, f"中文未编码: {url}"
    assert n._sent_ok == 1 and n._sent_fail == 0


def test_bark_snapshot_reports_channel():
    from autogame.config import NotifyConfig

    n = Notifier(NotifyConfig(enabled=True, webhook_format="bark", bark_key="k"))
    s = n.snapshot()
    assert s["bark"] is True and s["format"] == "bark"


def test_notify_config_accepts_bark_through_api():
    cfg = _cfg()
    hub = _hub(cfg)
    c = _client(cfg, hub)
    r = c.post("/api/config", json={"patch": {"notify": {
        "enabled": True, "webhook_format": "bark", "bark_key": "mykey"}}})
    assert r.status_code == 200, r.text
    assert cfg.notify.webhook_format == "bark"
    assert cfg.notify.bark_key == "mykey"
