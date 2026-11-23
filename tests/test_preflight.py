"""上真机前自检 —— 把"静默失效"变成"上机前就能看见"。

这一组用例守的不是"函数返回什么"，而是**真机上最容易翻车的那几种情况，
自检会不会拦下来**：

    串口号配错，对面是别的设备        → 必须 fail，且说清"该查什么"
    旧固件不认识 ping                 → 必须 fail，且说"重刷"
    帧 1280×720，screen_w 还是 1920   → 必须 fail，并算出"点中心会落到哪"
    采集区域不从 (0,0) 开始           → 必须 fail（下发坐标时没加偏移）
    一帧都没有 / 帧太旧               → 必须 fail
    后端是 dryrun                     → 不能报"可以上真机"
    服务在跑                          → 不能去串口插 ping（会抢走命令回显）

全部用假串口 / 假 hub，不需要硬件。
"""

from __future__ import annotations

import json
import sys
import types

from autogame import preflight
from autogame.config import AppConfig
from autogame.executor import DryRunExecutor


# ---------------------------------------------------------------- 夹具

class _FakeSer:
    def __init__(self, reply: str = ""):
        self.reply = reply
        self.timeout = 0.5
        self.in_waiting = 0
        self.lines: list[str] = []

    def write(self, b) -> None:
        self.lines.append(b.decode() if isinstance(b, bytes) else str(b))

    def flush(self) -> None:
        return None

    def readline(self) -> bytes:
        return (self.reply + "\n").encode() if self.reply else b""

    def read(self, n: int) -> bytes:
        return b""

    def close(self) -> None:
        return None


def _fake_serial(monkeypatch, reply: str) -> _FakeSer:
    ser = _FakeSer(reply)
    mod = types.ModuleType("serial")
    mod.Serial = lambda *a, **k: ser  # noqa: ARG005
    monkeypatch.setitem(sys.modules, "serial", mod)
    return ser


def _rp2040(cfg: AppConfig):
    from autogame.executor import RP2040Executor

    cfg.executor.backend = "rp2040"
    cfg.executor.serial_port = "FAKE"
    return RP2040Executor(cfg.executor)


class _Hub:
    """只提供 preflight 用到的那几个面。"""

    def __init__(self, executor, *, w: int = 1280, h: int = 720,
                 character=(640, 360), in_game: bool = True, gap: float = 0.4,
                 seq: int = 12, paused: bool = False):
        from autogame.assistant.modes import Mode, ModeManager

        self.executor = executor
        self.modes = ModeManager()
        if paused:
            self.modes.set(Mode.PAUSED, reason="测试")
        self._w, self._h = w, h
        self._ch = list(character) if character else None
        self._in_game = in_game
        self._gap = gap
        self._seq = seq

    def snapshot(self) -> dict:
        return {
            "frame_seq": self._seq, "frame_gap_s": self._gap,
            "scene": {"w": self._w, "h": self._h, "character": self._ch,
                      "in_game": self._in_game,
                      "gate_reason": "skillbar" if self._in_game else "unmatched"},
        }


def _cfg(**kw) -> AppConfig:
    cfg = AppConfig()
    cfg.executor.min_action_interval_ms = 0
    for k, v in kw.items():
        obj, _, attr = k.rpartition("__")
        target = cfg
        for part in (obj.split("__") if obj else []):
            target = getattr(target, part)
        setattr(target, attr, v)
    return cfg


def _by(res: dict, name: str) -> dict:
    for c in res["checks"]:
        if c["name"] == name:
            return c
    raise AssertionError(f"没有这一项：{name}（有 {[c['name'] for c in res['checks']]}）")


# ---------------------------------------------------------------- 执行器探活

def test_dryrun_says_it_will_not_actually_press_anything(monkeypatch):
    """dryrun 探活"通过"是可以的，但必须自报**不会动手** ——
    否则"联调跑通"会被当成"真机验证过"。"""
    cfg = _cfg()
    ex = DryRunExecutor(cfg.executor)
    got = ex.probe()
    assert got["ok"] is True and got["actuates"] is False
    assert "不会真按键盘" in got["detail"]
    assert "rp2040" in got["hint"]


def test_dryrun_run_is_not_reported_as_ready(monkeypatch):
    cfg = _cfg()
    cfg.executor.screen_w, cfg.executor.screen_h = 1280, 720
    hub = _Hub(DryRunExecutor(cfg.executor), w=1280, h=720)
    res = preflight.run(cfg=cfg, hub=hub)
    assert res["verdict"] == "dry", res["summary"]
    assert res["actuates"] is False
    assert "不会真的动手" in res["summary"]


def test_rp2040_probe_sends_ping_and_reads_the_version(monkeypatch):
    ser = _fake_serial(monkeypatch, '{"ok":true,"fw":"1.1.0","up_ms":9000,'
                                    '"watchdog_ms":5000}')
    cfg = _cfg()
    ex = _rp2040(cfg)
    got = ex.probe()
    assert got["ok"] is True and got["actuates"] is True
    assert got["fw"] == "1.1.0" and got["watchdog_ms"] == 5000
    sent = [json.loads(x) for x in ser.lines]
    assert sent == [{"op": "ping"}], "探活只能发 ping，不能碰键盘鼠标"
    assert "已运行 9 秒" in got["detail"]


def test_old_firmware_is_told_to_reflash(monkeypatch):
    """旧固件对 ping 回 unknown op —— 这跟"对面不是这块板子"是两种毛病，
    修法完全不同（刷固件 vs 查线），不能合成一个"不通"。"""
    _fake_serial(monkeypatch, '{"ok":false,"error":"unknown op"}')
    cfg = _cfg()
    got = _rp2040(cfg).probe()
    assert got["ok"] is False
    assert "太旧" in got["detail"] and "重刷" in got["hint"]


def test_silent_port_is_not_the_same_as_an_old_firmware(monkeypatch):
    _fake_serial(monkeypatch, "")
    cfg = _cfg()
    got = _rp2040(cfg).probe()
    assert got["ok"] is False
    assert "没回话" in got["detail"]
    assert "串口号" in got["hint"]


def test_wrong_device_shows_up_as_not_json(monkeypatch):
    _fake_serial(monkeypatch, "hello from some other gadget")
    cfg = _cfg()
    got = _rp2040(cfg).probe()
    assert got["ok"] is False and "不是 JSON" in got["detail"]


def test_probe_reports_a_broken_port_with_an_actionable_hint(monkeypatch):
    """句柄还在、但写不出去（线松了 / 驱动卡了）—— 这是真机上"它突然不动了"
    最常见的原因，探活要报 fail 并给出能照着做的提示，而不是抛异常。"""
    ser = _fake_serial(monkeypatch, '{"ok":true,"fw":"1.1.0"}')

    def _boom(_b):
        raise OSError("device disconnected")

    ser.write = _boom  # type: ignore[method-assign]
    cfg = _cfg()
    got = _rp2040(cfg).probe()
    assert got["ok"] is False
    assert "打不通" in got["detail"] and "device disconnected" in got["detail"]
    assert got["hint"], "光说坏了没用，要告诉人查什么"


def test_pca9685_mock_admits_it_does_not_move_servos(monkeypatch):
    from autogame.executor.pca9685 import Pca9685Executor

    cfg = _cfg()
    cfg.executor.backend = "pca9685"
    cfg.executor.pca9685_mock = True
    got = Pca9685Executor(cfg.executor).probe()
    assert got["ok"] is True and got["actuates"] is False
    assert "mock" in got["detail"]


def test_hidg_probe_reports_a_missing_node(monkeypatch):
    from autogame.executor import RPi5HidgExecutor

    cfg = _cfg()
    cfg.executor.backend = "rpi5_hidg"
    cfg.executor.hidg_keyboard = "/nonexistent/hidg0"
    cfg.executor.hidg_mouse = "/nonexistent/hidg1"
    ex = RPi5HidgExecutor.__new__(RPi5HidgExecutor)   # 不走 open，只测 probe
    ex.cfg = cfg.executor
    got = ex.probe()
    assert got["ok"] is False and "hidg 节点不在" in got["detail"]
    assert "setup_rpi5_hidg" in got["hint"]


# ---------------------------------------------------------------- 几何：点击全偏

def test_frame_size_mismatch_is_blocked_with_the_miss_coordinates(monkeypatch):
    """真机上"点了没反应"最常见的原因就是这一条。报错必须给出落点，
    否则人只会看到"点击好像没生效"，根本想不到是分辨率。"""
    _fake_serial(monkeypatch, '{"ok":true,"fw":"1.1.0","up_ms":1000}')
    cfg = _cfg()
    cfg.executor.screen_w, cfg.executor.screen_h = 1920, 1080
    hub = _Hub(_rp2040(cfg), w=1280, h=720)
    res = preflight.run(cfg=cfg, hub=hub)
    assert res["verdict"] == "blocked"
    geo = _by(res, "画面尺寸")
    assert geo["level"] == "fail"
    assert "1280×720" in geo["detail"] and "1920×1080" in geo["detail"]
    assert "(960,540)" in geo["detail"], geo["detail"]
    assert "screen_w" in geo["hint"]


def test_matching_sizes_pass(monkeypatch):
    _fake_serial(monkeypatch, '{"ok":true,"fw":"1.1.0","up_ms":1000}')
    cfg = _cfg()
    cfg.executor.screen_w, cfg.executor.screen_h = 1280, 720
    hub = _Hub(_rp2040(cfg), w=1280, h=720)
    res = preflight.run(cfg=cfg, hub=hub)
    assert _by(res, "画面尺寸")["level"] == "ok"
    assert res["verdict"] == "ok", res["blockers"]


def test_capture_region_with_an_offset_is_blocked(monkeypatch):
    """采集区域不从 (0,0) 开始时，下发坐标没有加偏移 —— 所有点击都会偏。"""
    _fake_serial(monkeypatch, '{"ok":true,"fw":"1.1.0","up_ms":1000}')
    cfg = _cfg()
    cfg.executor.screen_w, cfg.executor.screen_h = 1280, 720
    cfg.capture.region = [100, 50, 1280, 720]
    hub = _Hub(_rp2040(cfg), w=1280, h=720)
    res = preflight.run(cfg=cfg, hub=hub)
    geo = _by(res, "画面尺寸")
    assert geo["level"] == "fail" and "(100,50)" in geo["detail"]
    assert res["verdict"] == "blocked"


def test_capture_region_size_mismatch_is_blocked(monkeypatch):
    _fake_serial(monkeypatch, '{"ok":true,"fw":"1.1.0","up_ms":1000}')
    cfg = _cfg()
    cfg.executor.screen_w, cfg.executor.screen_h = 1920, 1080
    cfg.capture.region = [0, 0, 1280, 720]
    hub = _Hub(_rp2040(cfg), w=1280, h=720)
    res = preflight.run(cfg=cfg, hub=hub)
    assert _by(res, "画面尺寸")["level"] == "fail"
    assert res["verdict"] == "blocked"


# ---------------------------------------------------------------- 帧

def test_no_frames_at_all_is_blocked(monkeypatch):
    cfg = _cfg()
    hub = _Hub(DryRunExecutor(cfg.executor), seq=0, gap=None)
    res = preflight.run(cfg=cfg, hub=hub)
    f = _by(res, "采集在流")
    assert f["level"] == "fail" and "一帧都没收到过" in f["detail"]
    assert "agent" in f["hint"]
    assert res["verdict"] == "blocked"


def test_stale_frames_are_blocked(monkeypatch):
    cfg = _cfg()
    hub = _Hub(DryRunExecutor(cfg.executor), gap=9.5)
    res = preflight.run(cfg=cfg, hub=hub)
    f = _by(res, "采集在流")
    assert f["level"] == "fail" and "9.5 秒前" in f["detail"]


def test_character_outside_the_frame_is_blocked(monkeypatch):
    cfg = _cfg()
    hub = _Hub(DryRunExecutor(cfg.executor), w=1280, h=720, character=(1400, 360))
    res = preflight.run(cfg=cfg, hub=hub)
    assert _by(res, "角色位置")["level"] == "fail"
    assert res["verdict"] == "blocked"


def test_missing_character_is_only_a_warning(monkeypatch):
    """这一帧没定位到角色可能只是动画/遮挡，不该拦住上机 —— 但它会
    让门禁一直不过，得说出来。"""
    cfg = _cfg()
    hub = _Hub(DryRunExecutor(cfg.executor), character=None)
    res = preflight.run(cfg=cfg, hub=hub)
    assert _by(res, "角色位置")["level"] == "warn"
    assert "角色位置" in res["warns"]


def test_not_in_game_is_only_a_warning(monkeypatch):
    cfg = _cfg()
    hub = _Hub(DryRunExecutor(cfg.executor), in_game=False)
    res = preflight.run(cfg=cfg, hub=hub)
    assert _by(res, "当前画面")["level"] == "warn"


def test_no_hub_skips_everything_that_needs_frames():
    """服务还没起来时也要能跑 —— 只查配置侧，不要因为拿不到帧就崩。"""
    cfg = _cfg()
    res = preflight.run(cfg=cfg, hub=None, executor=None)
    for name in ("采集在流", "画面尺寸", "角色位置", "当前画面"):
        assert _by(res, name)["level"] == "skip", name
    assert res["verdict"] in ("dry", "blocked")


# ---------------------------------------------------------------- 模式与闸门

def test_paused_mode_is_blocked(monkeypatch):
    _fake_serial(monkeypatch, '{"ok":true,"fw":"1.1.0","up_ms":1000}')
    cfg = _cfg()
    cfg.executor.screen_w, cfg.executor.screen_h = 1280, 720
    hub = _Hub(_rp2040(cfg), paused=True)
    res = preflight.run(cfg=cfg, hub=hub)
    m = _by(res, "运行模式")
    assert m["level"] == "fail" and "不允许动作" in m["detail"]
    assert res["verdict"] == "blocked"


def test_action_gate_closed_is_also_caught(monkeypatch):
    """只有模式管、执行器闸门关着时，自检也必须说出来 ——
    两者不一致本身就是"页面显示在跑、动作全被拦"的那类假运行。"""
    cfg = _cfg()
    ex = DryRunExecutor(cfg.executor)
    ex.action_gate = lambda: False
    res = preflight.run(cfg=cfg, executor=ex, hub=None)
    g = _by(res, "动作闸门")
    assert g["level"] == "fail" and "一步都不会走" in g["detail"]


def test_running_service_does_not_touch_the_serial_port(monkeypatch):
    """服务在跑的时候去串口插一句 ping，会把正在下发命令的回显抢走。
    这时候探活必须跳过，而不是"顺便查一下"。"""
    ser = _fake_serial(monkeypatch, '{"ok":true,"fw":"1.1.0"}')
    cfg = _cfg()
    cfg.executor.screen_w, cfg.executor.screen_h = 1280, 720
    ex = _rp2040(cfg)
    ser.lines.clear()
    res = preflight.run(cfg=cfg, hub=_Hub(ex), allow_io=False)
    assert _by(res, "执行器探活")["level"] == "skip"
    assert ser.lines == [], "运行中不许往串口写任何东西"
    # 跳过的结论不能是"会动手"，也不能说成"确定不会动"
    assert res["actuates"] is False
    assert res["verdict"] == "dry"
    assert "没验证硬件" in res["summary"]


def test_stop_all_is_used_as_the_release_probe(monkeypatch):
    """松手检查用 stop_all：它只松不按，是最安全的探针。"""
    _fake_serial(monkeypatch, '{"ok":true,"fw":"1.1.0"}')
    cfg = _cfg()
    ex = _rp2040(cfg)
    ex._ser.lines.clear()
    res = preflight.run(cfg=cfg, executor=ex, hub=None)
    assert _by(res, "能松手")["level"] == "ok"
    ops = [json.loads(x).get("op") for x in ex._ser.lines if x.strip().startswith("{")]
    assert "stop_all" in ops
    assert "key_tap" not in ops and "mouse_click" not in ops, "自检不能按键"


# ---------------------------------------------------------------- 汇总与文案

def test_summary_says_something_actionable(monkeypatch):
    _fake_serial(monkeypatch, '{"ok":true,"fw":"1.1.0","up_ms":1000}')
    cfg = _cfg()
    cfg.executor.screen_w, cfg.executor.screen_h = 1920, 1080
    res = preflight.run(cfg=cfg, hub=_Hub(_rp2040(cfg), w=1280, h=720))
    assert "先别上真机" in res["summary"]
    assert res["blockers"] == ["画面尺寸"]


def test_text_output_lists_hints_for_failures(monkeypatch):
    _fake_serial(monkeypatch, "")
    cfg = _cfg()
    res = preflight.run(cfg=cfg, executor=_rp2040(cfg), hub=None)
    text = preflight.as_text(res)
    assert "[失败] 执行器探活" in text
    assert "→" in text, "失败项要带做法，不然人只知道坏了不知道怎么修"
    assert "结论：" in text


def test_http_endpoint_returns_the_same_verdict(tmp_path):
    from fastapi.testclient import TestClient

    from autogame.server.app import create_app
    from autogame.server.hub import Hub
    from autogame.vision.pipeline import VisionPipeline

    cfg = AppConfig()
    cfg.server.token = "t"
    cfg.server.event_log = ""
    cfg.executor.min_action_interval_ms = 0
    hub = Hub(cfg, DryRunExecutor(cfg.executor), VisionPipeline(cfg.vision))
    app = create_app(cfg, hub, "<h1>x</h1>")
    r = TestClient(app)
    d = r.get("/api/preflight", headers={"X-Token": "t"}).json()
    assert "verdict" in d and "checks" in d and "text" in d
    assert d["verdict"] == "blocked"      # 没帧
    assert any(c["name"] == "采集在流" for c in d["checks"])


def test_http_endpoint_requires_the_token(tmp_path):
    from fastapi.testclient import TestClient

    from autogame.server.app import create_app
    from autogame.server.hub import Hub
    from autogame.vision.pipeline import VisionPipeline

    cfg = AppConfig()
    cfg.server.token = "t"
    cfg.server.require_token = True    # 本地默认免鉴权，这条测的是打开之后的行为
    cfg.server.event_log = ""
    hub = Hub(cfg, DryRunExecutor(cfg.executor), VisionPipeline(cfg.vision))
    app = create_app(cfg, hub, "<h1>x</h1>")
    r = TestClient(app)
    assert r.get("/api/preflight").status_code == 401


def test_capture_fps_lower_than_tick_hz_is_a_warning():
    """决策比采集快 = 同一个画面被决策多次。不拦，但要指出来。"""
    cfg = _cfg()
    cfg.capture.fps = 0.5
    cfg.decision.tick_hz = 2.0
    res = preflight.run(cfg=cfg, hub=None, executor=None)
    c = _by(res, "采集配置")
    assert c["level"] == "warn" and "同一个画面" in c["detail"]


# ---------------------------------------------------------------- 依赖 / 构造失败

def test_missing_pyserial_is_a_blocker_not_a_silent_skip(monkeypatch):
    """缺 pyserial 的原文是 `No module named 'serial'`。

    这句在真机上分不清"没装"还是"装错解释器"，而且以前会退化成
    "没有运行中的执行器，跳过" —— 一条 fail 变成一句无害的 skip。
    """
    cfg = _cfg()
    cfg.executor.backend = "rp2040"
    monkeypatch.setattr(preflight, "_can_import",
                        lambda name: name not in ("serial",))
    res = preflight.run(cfg=cfg, hub=None, executor=None, allow_io=False)
    c = _by(res, "后端依赖")
    assert c["level"] == "fail"
    assert "pyserial" in c["detail"]
    assert "pip install pyserial" in c["hint"]
    assert res["verdict"] == "blocked"


def test_deps_check_also_runs_offline(monkeypatch):
    """离线（不碰硬件）时最该先查的就是依赖 —— 那时还没有执行器实例。"""
    cfg = _cfg()
    cfg.executor.backend = "rp2040"
    monkeypatch.setattr(preflight, "_can_import", lambda name: True)
    res = preflight.run(cfg=cfg, hub=None, executor=None, allow_io=False)
    assert _by(res, "后端依赖")["level"] == "ok"


def test_no_capture_backend_at_all_is_a_warning(monkeypatch):
    """dxcam/mss 都没有 = 一帧也采不到。真机上是瞎子，但不该冒充"配置错"。"""
    cfg = _cfg()
    cfg.capture.backend = "auto"
    monkeypatch.setattr(preflight, "_can_import",
                        lambda name: name not in ("dxcam", "mss"))
    res = preflight.run(cfg=cfg, hub=None, executor=None, allow_io=False)
    c = _by(res, "采集依赖")
    assert c["level"] == "warn" and "瞎子" in c["detail"]


def test_build_failure_is_reported_instead_of_looking_like_not_started():
    """build_executor 抛异常时，以前只说"没有运行中的执行器"——
    用户会以为"服务还没起"，实际是后端起不来。现在必须是 fail。"""
    cfg = _cfg()
    res = preflight.run(cfg=cfg, hub=None, executor=None, allow_io=False,
                        build_error="RuntimeError: 打不开串口 'COM5'：PermissionError")
    c = _by(res, "执行器构造")
    assert c["level"] == "fail" and "COM5" in c["detail"]
    assert res["verdict"] == "blocked"


def test_rp2040_without_pyserial_says_what_to_install():
    """构造时的报错要带修法，不能只有一句 ModuleNotFoundError。"""
    import pytest

    from autogame.executor import RP2040Executor

    cfg = _cfg()
    cfg.executor.backend = "rp2040"
    cfg.executor.serial_port = "COM_TEST_MISSING"
    try:
        import serial  # noqa: F401
        pytest.skip("这台机器装了 pyserial，跳过缺包路径")
    except ImportError:
        pass
    with pytest.raises(RuntimeError) as ei:
        RP2040Executor(cfg.executor)
    assert "pip install pyserial" in str(ei.value)
