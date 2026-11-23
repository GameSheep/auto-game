"""端到端闭环测试 —— 真实服务端 + 真实 WebSocket + 合成世界。

链路：SimWorld渲染帧 → AgentClient(WS) → 服务端视觉检测 → FSM决策
      → DryRunExecutor → 桥接线程 → SimWorld输入 → 拾取判定

验证目标（对话定稿验收）：
  1. 光柱 → 走位 → A框出现 → 按A → 拾取成功 的完整闭环可运行。
  2. 全程无盲按（blind_a_press == 0，模拟器违规计数）。
  3. token 鉴权生效（无 token 连接被拒绝）。
"""

import asyncio
import logging
import socket
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import pytest
import uvicorn

from autogame.config import AppConfig
from autogame.agent.client import AgentClient
from autogame.executor import DryRunExecutor
from autogame.server.app import create_app
from autogame.server.hub import Hub
from autogame.sim.scene import SimCapture, SimWorld
from autogame.vision.pipeline import VisionPipeline

log = logging.getLogger(__name__)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class ServerFixture:
    """独立线程运行真实 uvicorn 服务端。

    用完必须 stop：每个实例都是一个真在跑的 uvicorn + 决策循环 + 看门狗。
    以前这里没有 stop，7 个 e2e 各留一个服务端在后台跑到进程结束 ——
    后面的用例被前面的服务端抢 CPU，于是"偶发"超时、性能守卫莫名其红。
    """

    _live: list["ServerFixture"] = []

    def __init__(self) -> None:
        self.port = _free_port()
        self.cfg = AppConfig()
        self.cfg.server.token = "e2e-test-token"
        # 本地默认免鉴权；这些 e2e 专门验证"开鉴权时"的行为，所以显式打开
        self.cfg.server.require_token = True
        self.cfg.server.port = self.port
        # 2fps = 生产采集率（config.yaml 默认）。以前这里是 6fps"加速闭环"，
        # 但一轮检测要 ~130ms（场景门 19 张模板占 65%），6fps 只有 166ms 预算，
        # 一次尖峰就积压 —— 队列里全是 1~3 秒前的帧，按键必然按空。
        # 测试要跑的是**能实时跟上的工况**，不是刻意造一个追不上的队列。
        self.cfg.capture.fps = 2.0
        self.cfg.vision.max_detect_width = 1280
        self.cfg.decision.tick_hz = 2.0
        self.cfg.decision.approach_timeout_s = 8.0
        self.executor = DryRunExecutor(self.cfg.executor)
        self.hub = Hub(self.cfg, self.executor, VisionPipeline(self.cfg.vision))
        self.app = create_app(self.cfg, self.hub, "<html>test</html>")
        self.server = uvicorn.Server(
            uvicorn.Config(self.app, host="127.0.0.1", port=self.port,
                           log_level="error", access_log=False, ws="websockets")
        )
        self._thread = threading.Thread(target=self.server.run, daemon=True)
        ServerFixture._live.append(self)

    def stop(self) -> None:
        """关掉服务端并等线程退出（不关就是留一个后台进程在抢 CPU）。"""
        try:
            self.server.should_exit = True
        except Exception:  # noqa: BLE001
            pass
        self._thread.join(timeout=5)
        if self in ServerFixture._live:
            ServerFixture._live.remove(self)

    def start(self) -> None:
        self._thread.start()
        for _ in range(100):
            if self.server.started:
                return
            time.sleep(0.05)
        raise RuntimeError("服务端启动超时")

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@pytest.fixture(autouse=True)
def _stop_servers_after_test():
    """每个用例结束都关掉它开的服务端（见 ServerFixture 的说明）。"""
    yield
    for srv in list(ServerFixture._live):
        srv.stop()


def _run_bridge(executor, world, stop_event: threading.Event) -> None:
    idx = 0
    while not stop_event.is_set():
        journal = executor.journal
        while idx < len(journal):
            e = journal[idx]
            idx += 1
            world.on_command(e["op"], key=e.get("key"), button=e.get("button"),
                             x=e.get("x"), y=e.get("y"), hold_ms=e.get("hold_ms") or 0)
        stop_event.wait(0.02)


def test_full_closed_loop_picks_pillars():
    srv = ServerFixture()
    srv.start()

    world = SimWorld(seed=2026)
    agent_cfg = srv.cfg.model_copy(deep=True)
    client = AgentClient(agent_cfg, srv.url, agent_name="e2e-sim", capture=SimCapture(world))

    stop_event = threading.Event()
    bridge = threading.Thread(target=_run_bridge, args=(srv.executor, world, stop_event), daemon=True)
    bridge.start()
    agent_thread_holder = {}

    async def run_agent():
        await asyncio.wait_for(client.run(), timeout=40)

    async def main():
        # 启动 FSM（通过 REST，验证 token 控制）
        import httpx

        async with httpx.AsyncClient(timeout=5) as hc:
            r = await hc.post(f"{srv.url}/api/control/start",
                              headers={"X-Token": "e2e-test-token"})
            assert r.status_code == 200, "带 token 控制应成功"
            # 无 token 必须被拒
            r = await hc.post(f"{srv.url}/api/control/start")
            assert r.status_code == 401, "无 token 控制必须 401"

        agent_task = asyncio.create_task(run_agent())
        agent_thread_holder["task"] = agent_task
        # 等待拾取成功
        deadline = time.monotonic() + 35
        while time.monotonic() < deadline:
            if world.picked >= 2:
                break
            await asyncio.sleep(0.5)
        client.stop()
        try:
            await asyncio.wait_for(agent_task, timeout=5)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        # 急停复位
        async with httpx.AsyncClient(timeout=5) as hc:
            await hc.post(f"{srv.url}/api/control/stop",
                          headers={"X-Token": "e2e-test-token"})

    asyncio.run(main())
    stop_event.set()
    time.sleep(0.1)

    # ---- 验收断言 ----
    assert world.picked >= 2, f"闭环必须完成拾取，实际 {world.picked}，FSM事件={srv.hub.fsm.events[-10:]}"
    assert world.blind_a_press == 0, "绝不允许盲按 A（模拟器违规计数必须为0）"
    assert srv.hub.fsm.stats["blind_tap_violations"] == 0
    assert srv.hub.fsm.stats["a_taps"] > 0
    picks = srv.hub.fsm.stats["picks"]
    assert picks >= 1, "FSM 必须确认至少一次拾取"
    assert picks <= srv.hub.fsm.stats["a_taps"], "拾取数不得超过按A数（铁律）"
    assert picks <= world.picked, "FSM 拾取数不得超过模拟器实际移除数"
    snap = srv.hub.snapshot()
    assert snap["executor"]["sent"] > 0


def test_unauthorized_ws_rejected():
    srv = ServerFixture()
    srv.start()

    import websockets

    async def try_connect():
        try:
            async with websockets.connect(
                f"ws://127.0.0.1:{srv.port}/ws/agent", max_size=2**20
            ) as ws:
                # 无 token → 服务端应关闭连接
                try:
                    await asyncio.wait_for(ws.recv(), timeout=3)
                    return "received"
                except websockets.ConnectionClosed:
                    return "closed"
                except asyncio.TimeoutError:
                    return "timeout"
        except Exception as e:  # noqa: BLE001
            return f"error:{type(e).__name__}"

    result = asyncio.run(try_connect())
    assert result in ("closed", "timeout") or result.startswith("error:"), \
        f"无 token 连接必须被拒绝，实际: {result}"


def test_farming_loop_full_stack():
    """完整刷图闭环（走真实服务端）：拾满 → 结算 → FSM点『再次挑战』→ 下一图 → 继续拾取。

    验证对话定稿的产品主循环在真实 WS/视觉/决策栈上运转。
    """
    srv = ServerFixture()
    srv.start()
    # 与生产一致：2fps 采集 + 2Hz 决策（见 ServerFixture 里的说明）
    srv.cfg.capture.fps = 2.0
    srv.cfg.decision.tick_hz = 2.0

    world = SimWorld(seed=33, picks_until_clear=3, walk_speed_px_s=420.0)
    agent_cfg = srv.cfg.model_copy(deep=True)
    client = AgentClient(agent_cfg, srv.url, agent_name="farm-sim", capture=SimCapture(world))

    stop_event = threading.Event()
    bridge = threading.Thread(target=_run_bridge, args=(srv.executor, world, stop_event), daemon=True)
    bridge.start()

    async def main():
        import httpx

        async with httpx.AsyncClient(timeout=5) as hc:
            r = await hc.post(f"{srv.url}/api/control/start",
                              headers={"X-Token": "e2e-test-token"})
            assert r.status_code == 200
        agent_task = asyncio.create_task(asyncio.wait_for(client.run(), timeout=130))
        # 等待至少完成一次完整刷图循环
        deadline = time.monotonic() + 110
        while time.monotonic() < deadline:
            if world.map_clears >= 1 and world.picked >= 4:
                break
            await asyncio.sleep(0.5)
        client.stop()
        try:
            await asyncio.wait_for(agent_task, timeout=5)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

    asyncio.run(main())
    stop_event.set()
    time.sleep(0.1)

    assert world.map_clears >= 1, (
        f"必须完成至少一次完整刷图循环，实际 map_clears={world.map_clears} "
        f"picked={world.picked} events={srv.hub.fsm.events[-8:]}"
    )
    assert world.picked >= 4, "下一张图必须继续拾取"
    assert srv.hub.fsm.stats["rechallenges"] >= 1, "FSM 必须点击过『再次挑战』"
    assert world.blind_a_press == 0, "全程零盲按"
    assert srv.hub.fsm.stats["blind_tap_violations"] == 0


def test_locate_api_real_http():
    """对话定稿的 POST /api/locate —— 真实 HTTP 调用验证响应结构。

    链路：SimCapture（含固定位置金柱）→ AgentClient → hub 最新帧缓存
         → POST /api/locate {"target":"gold"} → found/center/ground/click_point。
    同时验证：未鉴权 401、无此颜色 found=False + available 列表。
    """
    srv = ServerFixture()
    srv.start()
    srv.cfg.capture.fps = 4.0

    # 固定金柱：世界 (100,50) → 屏幕地面 ≈ (740,410)，禁自动刷怪保证确定性
    world = SimWorld(seed=7, pillar_spawn_interval_s=999)
    from autogame.sim.scene import SimPillar

    world.pillars = [SimPillar(100, 50, "gold")]
    agent_cfg = srv.cfg.model_copy(deep=True)
    client = AgentClient(agent_cfg, srv.url, agent_name="locate-test",
                         capture=SimCapture(world))

    async def main():
        import httpx

        async with httpx.AsyncClient(timeout=10) as hc:
            # 1) 未鉴权必须 401
            r = await hc.post(f"{srv.url}/api/locate", json={"target": "gold"})
            assert r.status_code == 401, f"无 token 必须 401，实际 {r.status_code}"

            # 2) 启动采集代理，等 hub 缓存到帧
            task = asyncio.create_task(asyncio.wait_for(client.run(), timeout=30))
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                snap = srv.hub.snapshot()
                if snap["frame_seq"] >= 3:
                    break
                await asyncio.sleep(0.2)
            client.stop()
            try:
                await asyncio.wait_for(task, timeout=8)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

            # 3) 正向查询 gold → found=True + 完整响应结构
            r = await hc.post(f"{srv.url}/api/locate",
                              json={"target": "gold"},
                              headers={"X-Token": "e2e-test-token"})
            assert r.status_code == 200
            body = r.json()
            assert body["found"] is True, f"金柱在画面内必须 found：{body}"
            assert body["target"] == "gold"
            assert set(body.keys()) == {
                "found", "target", "center", "ground", "bbox",
                "area", "distance", "click_point",
            }
            cx, cy = body["center"]
            gx, gy = body["ground"]
            assert abs(cx - 740) <= 25, f"center.x 应≈740（世界100+中心640），实际 {cx}"
            assert abs(gy - 410) <= 25, f"ground.y 应≈410（世界50+中心360），实际 {gy}"
            # click_point = ground + ground_click_offset(28)
            assert body["click_point"] == [gx, gy + 28]
            # bbox 结构 [x, y, w, h]
            bx, by, bw, bh = body["bbox"]
            assert bw > 0 and bh > 0
            assert abs((bx + bw / 2) - cx) <= 2
            assert body["area"] > 0
            assert body["distance"] >= 0

            # 4) 查询不存在的颜色 → found=False + available 含 gold
            r = await hc.post(f"{srv.url}/api/locate",
                              json={"target": "pink"},
                              headers={"X-Token": "e2e-test-token"})
            body2 = r.json()
            assert body2["found"] is False
            assert "gold" in body2.get("available", [])

    asyncio.run(main())


def test_agent_reconnect_recovery():
    """可靠性：代理断线重连后，服务端恢复收帧、旧连接被踢、静止场景首帧必发。"""
    srv = ServerFixture()
    srv.start()
    world = SimWorld(seed=5, pillar_spawn_interval_s=999)
    from autogame.sim.scene import SimPillar

    world.pillars = [SimPillar(100, 50, "gold")]
    agent_cfg = srv.cfg.model_copy(deep=True)

    async def run_session(client, until_seq, timeout=20):
        task = asyncio.create_task(asyncio.wait_for(client.run(), timeout=timeout))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if srv.hub.snapshot()["frame_seq"] >= until_seq:
                break
            await asyncio.sleep(0.2)
        client.stop()
        try:
            await asyncio.wait_for(task, timeout=5)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        return srv.hub.snapshot()["frame_seq"]

    async def main():
        client1 = AgentClient(agent_cfg, srv.url, agent_name="rc-1",
                              capture=SimCapture(world))
        seq1 = await run_session(client1, 3)
        assert seq1 >= 3, f"会话1必须收到帧，实际 {seq1}"

        # 会话2：新连接（服务端踢旧连接逻辑覆盖）
        client2 = AgentClient(agent_cfg, srv.url, agent_name="rc-2",
                              capture=SimCapture(world))
        seq2 = await run_session(client2, seq1 + 3)
        assert seq2 >= seq1 + 3, f"重连后必须恢复收帧 {seq1} → {seq2}"
        # 门禁状态恢复（红血球/模板锚点在新帧上正常工作）
        snap = srv.hub.snapshot()
        assert snap["state"] in ("PAUSED", "SCANNING")

    asyncio.run(main())


def test_ui_task_trade_buy_full_stack(tmp_path):
    """UI 任务全栈：POST /api/task 启动交易行配方 → hub 决策循环执行配方
    → 按钮模板匹配/行点击/槽位点击全链路 → 完成后自动恢复 FSM。"""
    srv = ServerFixture()
    srv.start()
    srv.cfg.capture.fps = 4.0

    # 生成交易面板帧源 + 按钮模板
    import sys as _sys

    _sys.path.insert(0, str(ROOT / "tests"))
    from test_ui_tasks import _crop_templates, _mk_trade_panel

    panel = _mk_trade_panel()
    _crop_templates(panel, tmp_path)
    srv.cfg.ui_task.templates_dir = str(tmp_path)

    class TradeCapture:
        name = "trade"

        def grab(self):
            return panel.copy()

        def close(self):
            pass

    agent_cfg = srv.cfg.model_copy(deep=True)
    client = AgentClient(agent_cfg, srv.url, agent_name="ui-task",
                         capture=TradeCapture())

    async def main():
        import httpx

        # 1) 未鉴权必须 400/401
        async with httpx.AsyncClient(timeout=10) as hc0:
            r = await hc0.post(f"{srv.url}/api/task", json={"recipe": "trade_buy"})
            assert r.status_code == 401, "无 token 必须 401"

        # 2) 启动采集代理（帧源 = 交易面板）
        task = asyncio.create_task(asyncio.wait_for(client.run(), timeout=40))

        async with httpx.AsyncClient(timeout=10) as hc:
            # 等帧就绪
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if srv.hub.snapshot()["frame_seq"] >= 3:
                    break
                await asyncio.sleep(0.2)
            assert srv.hub.snapshot()["frame_seq"] >= 3, "代理必须已上传帧"

            # 3) 启动任务
            r = await hc.post(f"{srv.url}/api/task",
                              json={"recipe": "trade_buy"},
                              headers={"X-Token": "e2e-test-token"})
            assert r.status_code == 200, f"任务启动失败: {r.text}"
            assert r.json()["steps"] == 5

            # 4) 等任务完成（配方引擎在 hub 决策循环里推进）
            deadline = time.monotonic() + 25
            body = {}
            last_status = None
            while time.monotonic() < deadline:
                st = await hc.get(f"{srv.url}/api/task/status",
                                  headers={"X-Token": "e2e-test-token"})
                body = st.json()
                if body["status"] is not None:
                    last_status = body["status"]
                if not body["active"]:
                    break
                await asyncio.sleep(0.1)
            assert body["active"] is False, f"任务应自动结束: {body}"
            # 权威完成记录：hub.ui_task_ends（快照）+ JSONL 审计双写
            # （完成即清除 ui_task，轮询可能停在完成前，以完成记录为准）
            snap = srv.hub.snapshot()
            end_events = list(snap.get("ui_task_ends", []))
            assert end_events, "完成事件必须写入快照与审计日志"
            end = end_events[-1]
            assert end["failed"] == "", f"配方不应失败: {end}"
            assert end["step"] == end["total_steps"] == 5, f"5 步应全部完成: {end}"
            # 过程证据：最后一次轮询至少推进到第 4 步
            assert last_status is not None and last_status["step"] >= 4, \
                f"配方必须推进到尾部: {last_status}"
            # 5) 停止接口
            r = await hc.post(f"{srv.url}/api/task/stop",
                              headers={"X-Token": "e2e-test-token"})
            assert r.json()["ok"] is True
        client.stop()
        try:
            await asyncio.wait_for(task, timeout=5)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

    asyncio.run(main())


def test_ui_task_without_templates_fails_fast(tmp_path):
    """缺 UI 模板时必须**拒绝启动**，而不是静默 skip 全部步骤假装成功。

    历史 bug：templates/ui/ 不存在时 trade_buy 5 步全部 skip_panel_check/skip_button，
    接口返回 ok=true、任务"跑完"却一条命令都没发 —— 最难排查的假成功。
    """
    srv = ServerFixture()
    srv.start()
    srv.cfg.ui_task.templates_dir = str(tmp_path)  # 空目录

    async def main():
        import httpx

        async with httpx.AsyncClient(timeout=10) as hc:
            r = await hc.post(f"{srv.url}/api/task", json={"recipe": "trade_buy"},
                              headers={"X-Token": "e2e-test-token"})
            assert r.status_code == 409, f"缺模板必须 409，实际 {r.status_code}: {r.text}"
            detail = r.json()["detail"]
            need = {"panel_trade.png", "sort.png", "buy.png", "confirm.png"}
            assert need <= set(detail["missing"]), f"缺失清单不完整: {detail['missing']}"
            assert "crop_template" in detail["how_to"], "必须给出裁剪指引"
            assert srv.hub.ui_task is None, "预检失败不得挂上任务"
            # 状态接口也必须显示无活动任务
            st = await hc.get(f"{srv.url}/api/task/status",
                              headers={"X-Token": "e2e-test-token"})
            assert st.json()["active"] is False

    asyncio.run(main())
