"""FastAPI 服务端 —— 代理 WS 上行 + 仪表盘 + REST 控制。

安全：
  - 所有连接必须携带 token（WS: query ?token= / REST: X-Token 头）。
  - 常数时间比较；无 token 的请求一律 401/1008 关闭。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from pathlib import Path as _Path
from typing import Any

import cv2
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.responses import Response, StreamingResponse

from ..config import AppConfig, VisionConfig
from ..protocol import AckMsg, HelloMsg, PongMsg, SetCaptureMsg, dumps, unpack_message
from ..security import check_token
from ..vision.detector import Detector, ROOT_TEMPLATES

# 审查页素材根目录（assets/）
_ASSETS = _Path(__file__).resolve().parent.parent.parent / "assets"
from ..vision.ui_panels import GridConfig
from pathlib import Path
from .hub import Hub

log = logging.getLogger(__name__)

# 全局 Hub（单进程单游戏PC；多代理可在部署层多实例）
_HUB: Hub | None = None
_CFG: AppConfig | None = None


async def _push_capture(hub: Any, cfg: AppConfig) -> bool:
    """把当前采集参数推给**正在连接的那个代理**。

    为什么必须推：代理只在 hello 时拉一次参数（见 agent_ws 里的 SetCaptureMsg）。
    页面改完 fps 如果不推，服务端配置写着 10fps、代理还按 2fps 采图 ——
    决策看着"变快了"，其实一直在反复用旧帧。这种假生效比不生效更糟：
    用户会得出"调了没用"的结论，然后把本来有效的开关关掉。

    没有代理连接时返回 False（这是正常状态，不是错误）。
    """
    ws = getattr(hub, "agent_ws", None)
    if ws is None:
        return False
    try:
        await ws.send_text(dumps(SetCaptureMsg(
            fps=cfg.capture.fps, jpeg_quality=cfg.capture.jpeg_quality)))
        return True
    except Exception:  # noqa: BLE001
        log.warning("推送采集参数失败（代理可能已断开）")
        return False


def create_app(cfg: AppConfig, hub: Hub, dashboard_html: str,
               config_path: str = "", profile: Any = None) -> FastAPI:
    global _HUB, _CFG
    _HUB = hub
    _CFG = cfg
    token = cfg.server.token or ""

    # 对话助手：装配时就把动作闸门装到执行器上 —— 这样"先停下"从服务起来
    # 的第一毫秒就有效，而不是等第一次请求进来。
    from ..assistant import build as _build_assistant
    if profile is not None:
        # 体检发现退化时要知道该回滚谁 —— 没有 profile 就没有可回滚的配方，
        # 那时候监控只报不修（报总比假装正常好）。
        hub.profile = profile
    asst = _build_assistant(cfg, profile, hub, config_path=config_path)
    # 换代（promote / rollback）时让运行中的进程跟上：重载规则 + 重置体检基线。
    # 不注册的话，"学完一代"在运行上等于没发生，直到下次重启。
    from ..evolve import set_generation_hook

    set_generation_hook(hub.on_generation_change)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()
        hub.bind_loop(loop)
        decide_task = asyncio.create_task(hub.decision_loop(), name="decision-loop")
        watch_task = asyncio.create_task(hub.watchdog_loop(), name="frame-watchdog")
        tasks = [decide_task, watch_task]
        if profile is not None:
            tasks.append(asyncio.create_task(_learner_loop(), name="learner-loop"))
        log.info("决策循环已启动（含帧流看门狗）")
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t

    async def _learner_loop() -> None:
        """持续学习：材料变了就自己学，不用人记得点按钮。

        为什么在服务里跑而不是 cron：它要知道"现在是不是在打怪"。换代会让正在
        执行的动作失去依据，所以只在这个进程空闲时才动手 —— 这个状态只有这里知道。
        """
        from .. import learner as lr

        interval = max(5.0, float(getattr(cfg.learner, "interval_s", 15.0)))
        log.info("持续学习已启动（每 %.0fs 看一次材料）", interval)
        while True:
            try:
                busy_now = hub.fsm.state.value != "PAUSED"
                use_model = bool(getattr(cfg.learner, "use_model", True))
                res = await asyncio.to_thread(
                    lr.tick, profile, cfg, use_model=use_model, busy=busy_now)
                if res.get("action") == "learn":
                    log.info("自动学习：%s", res.get("message") or res.get("code"))
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("自动学习这一轮出错（下一轮继续）")
            await asyncio.sleep(interval)

    app = FastAPI(title="autogame", docs_url=None, redoc_url=None, lifespan=lifespan)

    def _require_token_header(request: Request) -> None:
        """本地默认**免鉴权**：局域网自用还要粘 token 只是给自己添堵。

        需要暴露到不可信网络时把 server.require_token 打开，行为与之前一致
        （常数时间比较、无 token 401/1008）。
        """
        if not cfg.server.require_token:
            return
        provided = request.headers.get("x-token") or request.query_params.get("token")
        if not check_token(provided, token):
            raise HTTPException(status_code=401, detail="unauthorized")

    # ---------------- 代理上行通道 ----------------

    @app.websocket("/ws/agent")
    async def agent_ws(ws: WebSocket) -> None:
        if cfg.server.require_token:
            provided = ws.query_params.get("token")
            if not check_token(provided, token):
                await ws.close(code=1008, reason="unauthorized")
                return
        await ws.accept()
        # 踢除旧代理连接（重连风暴防护：同一时间只允许一路帧源）
        old = getattr(hub, "agent_ws", None)
        if old is not None and old is not ws:
            try:
                await old.close(code=1012, reason="replaced by new agent")
                log.info("旧代理连接已踢除")
            except Exception:  # noqa: BLE001
                pass
        hub.agent_ws = ws
        hello_seen = False
        agent_name = "unknown"
        log.info("代理已连接")
        try:
            while True:
                raw = await ws.receive()
                if raw.get("type") == "websocket.disconnect":
                    break
                msg = raw.get("bytes") or raw.get("text")
                if msg is None:
                    continue
                kind, payload = unpack_message(msg)
                if kind == "json":
                    mtype = payload.get("type")
                    if mtype == "hello":
                        h = HelloMsg.model_validate(payload)
                        hello_seen = True
                        agent_name = h.agent
                        # 下发采集参数（服务端为准）
                        await ws.send_text(
                            dumps(
                                SetCaptureMsg(
                                    fps=cfg.capture.fps,
                                    jpeg_quality=cfg.capture.jpeg_quality,
                                )
                            )
                        )
                        await ws.send_text(dumps(AckMsg(ok=True)))
                    elif mtype == "ping":
                        await ws.send_text(dumps(PongMsg(ts_ms=payload.get("ts_ms", 0))))
                    else:
                        await ws.send_text(dumps(AckMsg(ok=False, error="unknown type")))
                elif kind == "jpeg":
                    if not hello_seen:
                        await ws.send_text(dumps(AckMsg(ok=False, error="hello first")))
                        continue
                    try:
                        summary = hub.on_frame(payload["jpeg"], payload["ts_ms"])
                    except ValueError as e:
                        await ws.send_text(dumps(AckMsg(ok=False, error=str(e))))
                        continue
                    # 每帧回执（小 JSON，代理可据此自适应）
                    await ws.send_text(dumps({"type": "frame_ack", **summary}))
        except WebSocketDisconnect:
            pass
        finally:
            if hub.agent_ws is ws:
                hub.agent_ws = None
            log.info("代理断开: %s", agent_name)

    # ---------------- 仪表盘 ----------------

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> str:
        return dashboard_html

    @app.get("/api/state")
    async def state(request: Request) -> JSONResponse:
        # 只读状态也要求 token，避免局域网内信息泄露
        _require_token_header(request)
        return JSONResponse(hub.snapshot())

    @app.get("/api/autonomy")
    async def autonomy(request: Request) -> JSONResponse:
        """自主恢复 / 模型调用状态（排障用：卡在哪一级、模型说了什么）。"""
        _require_token_header(request)
        snap = hub.snapshot()
        return JSONResponse({
            "state": snap["state"],
            "autonomy": snap["autonomy"],
            "frame_gap_s": snap["frame_gap_s"],
            "frame_stalled": snap["frame_stalled"],
        })

    @app.get("/api/stream.mjpg")
    async def stream_mjpg(request: Request):
        _require_token_header(request)
        interval = 1.0 / max(1.0, cfg.server.dash_fps)
        boundary = "autogameframe"

        async def gen():
            last = 0.0
            while True:
                now = time.monotonic()
                if now - last < interval:
                    await asyncio.sleep(max(0.0, interval - (now - last)))
                last = time.monotonic()
                jpg = hub.annotated_jpeg()
                if jpg is None:
                    await asyncio.sleep(0.2)
                    continue
                yield (
                    b"--" + boundary.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
                )

        return StreamingResponse(
            gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}"
        )

    # ---------------- 控制接口（写操作，必须 token） ----------------

    @app.post("/api/control/{action}")
    async def control(action: str, request: Request) -> JSONResponse:
        _require_token_header(request)
        try:
            result = hub.control(action)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return JSONResponse(result)

    @app.post("/api/capture")
    async def set_capture(request: Request) -> JSONResponse:
        """调整代理采集参数。**会立刻推给已连接的代理**，不用等它重连。"""
        _require_token_header(request)
        body = await request.json()
        m = SetCaptureMsg.model_validate({**body, "type": "set_capture"})
        if m.fps is not None:
            cfg.capture.fps = m.fps
        if m.jpeg_quality is not None:
            cfg.capture.jpeg_quality = m.jpeg_quality
        pushed = await _push_capture(hub, cfg)
        return JSONResponse({"ok": True, "pushed": pushed,
                             "note": "" if pushed else "当前没有代理连接，参数会在它连上时下发"})

    @app.post("/api/locate")
    async def locate(request: Request) -> JSONResponse:
        """对话定稿的位置查询接口：
        POST {"target": "red"|"gold"|"pink"|...} → {"found":true, "center":[x,y], ...}
        在最新一帧上运行检测，返回目标颜色光柱的地面坐标（无目标 found=false）。
        Detector 实例跨调用复用（时域确认依赖连续帧）。
        """
        _require_token_header(request)
        body = await request.json()
        target = str(body.get("target", "")).strip().lower()
        hub_ = _HUB
        with hub_._lock:  # noqa: SLF001 同包访问
            jpeg = hub_._latest_jpeg
        if jpeg is None:
            return JSONResponse({"found": False, "reason": "尚无画面"})
        import numpy as np

        buf = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return JSONResponse({"found": False, "reason": "帧解码失败"})
        # 复用 Hub 管线的检测器实例：它已持续处理历史帧，
        # 时域确认（beam_persistence）状态与在线决策完全一致
        st = hub_.pipeline.detector.process(frame)
        want_all = target in ("", "any", "all")
        hits = [p for p in st.pillars if want_all or p.color == target]
        if not hits:
            return JSONResponse({"found": False, "target": target,
                                 "available": sorted({p.color for p in st.pillars})})
        p = hits[0]  # 距中心最近
        return JSONResponse({
            "found": True,
            "target": p.color,
            "center": [p.cx, p.cy],
            "ground": [p.x, p.y],
            "bbox": [p.cx - p.w // 2, p.cy - p.h // 2, p.w, p.h],
            "area": p.area,
            "distance": round(p.distance, 1),
            "click_point": [p.x, p.y + _CFG.decision.ground_click_offset_px],
        })

    # ---------------- 设置页：配置 / 预设 / 标定 / 告警 ----------------

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page() -> str:
        return (_ASSETS.parent / "web" / "settings.html").read_text(encoding="utf-8")

    @app.get("/api/config")
    async def get_config(request: Request) -> JSONResponse:
        from .settings import LIVE_FIELDS, agent_report, input_report, public_config
        from ..presets import DESCRIPTIONS

        _require_token_header(request)
        return JSONResponse({
            "config": public_config(cfg),
            "input_report": input_report(cfg),
            # 游戏者：目标/技能这类"知识"的状态。改知识文件需要重启（不像键位能就地派生），
            # 所以这里只报状态，不进 LIVE_FIELDS。
            "agent_report": agent_report(cfg, getattr(profile, "root", None)),
            "presets": DESCRIPTIONS,
            "live_fields": list(LIVE_FIELDS),
            "config_path": config_path,
        })

    @app.post("/api/config")
    async def patch_config(request: Request) -> JSONResponse:
        """局部更新配置：校验 → 原地生效 → 写入覆盖文件。

        校验失败整批不生效（半套配置比不配置更危险），错误字段回给页面。
        """
        from .settings import ConfigError, apply_patch, save_overlay

        _require_token_header(request)
        body = await request.json()
        patch = body.get("patch") if isinstance(body, dict) else None
        if not isinstance(patch, dict):
            raise HTTPException(status_code=400, detail="body 需要 {\"patch\": {...}}")
        try:
            res = apply_patch(cfg, patch)
        except ConfigError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        saved = ""
        if config_path:
            try:
                saved = save_overlay(config_path, patch)
            except OSError as e:
                log.warning("配置已生效但写盘失败: %s", e)
        if res["changed"]:
            log.info("配置已更新 %d 项: %s", len(res["changed"]),
                     ", ".join(res["changed"][:8]))
        return JSONResponse({"ok": True, "saved": saved, **res})

    # ---------------- 二次进化 ----------------

    @app.get("/api/evolve")
    async def evolve_state(request: Request) -> JSONResponse:
        """当前代数、材料体检、它认为缺什么、以及"持续学习"现在在干什么。"""
        _require_token_header(request)
        if profile is None:
            return JSONResponse({"ok": False, "error": "没有绑定 profile（单游戏模式）"})
        from .. import evolve as ev
        from .. import learner as lr

        audit = await asyncio.to_thread(ev.audit_materials, profile)
        busy_now = hub.fsm.state.value != "PAUSED"
        learn = await asyncio.to_thread(lr.snapshot, profile, cfg, busy=busy_now)
        wanted_path = ev.evolve_root(profile) / "wanted.json"
        wanted = {}
        if wanted_path.exists():
            try:
                wanted = json.loads(wanted_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                wanted = {}

        def _row(g) -> dict:
            """一代 + 它带来的变化。页面要能回答"这次多认了什么"。"""
            d = g.to_dict()
            dp = Path(g.root) / "delta.json"
            if dp.exists():
                try:
                    raw = json.loads(dp.read_text(encoding="utf-8"))
                    kn = raw.get("knowledge") or {}
                    d["delta"] = {
                        "ma": len(raw.get("added") or []),
                        "mc": len(raw.get("changed") or []),
                        "mr": len(raw.get("removed") or []),
                        "ka": int(kn.get("n_added") or 0),
                        "kr": int(kn.get("n_removed") or 0),
                        "blind": bool(raw.get("blind")),
                        "vs": raw.get("vs") or "",
                    }
                except Exception:  # noqa: BLE001
                    pass
            return d

        return JSONResponse({
            "ok": True,
            "profile": profile.id,
            "active": ev.active_id(profile),
            "generations": [_row(g) for g in ev.list_generations(profile)],
            "audit": audit.to_dict(),
            "audit_summary": audit.summary(),
            "learner": learn,
            "wanted": wanted.get("asks") or [],
            "history": ev.history(profile, limit=8),
        })

    @app.post("/api/evolve/run")
    async def evolve_run(request: Request) -> JSONResponse:
        """手动催一次（"现在学"）。跑在后台线程：回归要动检测器和 FSM，几十秒。

        不走自动那套防抖/间隔 —— 人点了就是现在要，哪怕机器人正在跑。
        """
        _require_token_header(request)
        if profile is None:
            return JSONResponse({"ok": False, "error": "没有绑定 profile"})
        from .. import learner as lr

        body = {}
        with contextlib.suppress(Exception):
            body = await request.json() or {}
        use_model = bool(body.get("use_model", True))
        res = await asyncio.to_thread(lr.learn, profile, cfg,
                                     source="manual", use_model=use_model)
        return JSONResponse({"ok": bool(res.get("ok")), **res})

    @app.post("/api/evolve/auto")
    async def evolve_auto(request: Request) -> JSONResponse:
        """开关持续学习。存在 learner.json（数据根），重启和升级都不丢。"""
        _require_token_header(request)
        if profile is None:
            return JSONResponse({"ok": False, "error": "没有绑定 profile"})
        from .. import learner as lr

        body = {}
        with contextlib.suppress(Exception):
            body = await request.json() or {}
        st = await asyncio.to_thread(lr.set_enabled, profile, cfg,
                                     bool(body.get("enabled", True)))
        return JSONResponse({"ok": True, "enabled": bool(st.enabled)})

    @app.post("/api/evolve/note")
    async def evolve_note(request: Request) -> JSONResponse:
        """补一条经验：问两个问题，写进手册认得的格式，剩下的交给持续学习。

        不做自由文本 —— 解析器不认的自由文本，写进去也学不到，比不给入口更糟。
        """
        _require_token_header(request)
        if profile is None:
            return JSONResponse({"ok": False, "error": "没有绑定 profile"})
        from ..teach import append_note

        body = {}
        with contextlib.suppress(Exception):
            body = await request.json() or {}
        try:
            path, block = await asyncio.to_thread(
                append_note, profile, str(body.get("when") or ""),
                str(body.get("todo") or ""))
        except ValueError as e:
            return JSONResponse({"ok": False, "error": str(e)})
        except Exception as e:  # noqa: BLE001
            log.exception("写补充经验失败")
            return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"})
        return JSONResponse({"ok": True, "file": str(path.name), "added": block})

    @app.post("/api/evolve/rollback")
    async def evolve_rollback(request: Request) -> JSONResponse:
        """退回上一代。事故现场的第一动作，所以它必须比"重新进化"更快。"""
        _require_token_header(request)
        if profile is None:
            return JSONResponse({"ok": False, "error": "没有绑定 profile"})
        from .. import evolve as ev

        ok, msg = await asyncio.to_thread(ev.rollback, profile)
        return JSONResponse({"ok": ok, "message": msg, "active": ev.active_id(profile)})

    # ---------------- 指令组：喊一句就跑整套 ----------------

    def _group_tpl_dir():
        """组用哪套模板。有激活代就用代的模板，否则用 profile 的。"""
        if profile is None:
            return None
        from .. import evolve as ev

        gen = ev.active_generation(profile)
        if gen is not None:
            return gen.resolved_templates(profile)
        return Path(profile.templates_dir)

    @app.get("/api/groups")
    async def groups_state(request: Request) -> JSONResponse:
        """列出能喊的组、每个缺什么，以及世界层的规模。

        `ready=False` 的组也列出来 —— 它缺的是模板，不是不存在。藏起来会让人
        以为自己没教会，其实是没截图。
        """
        _require_token_header(request)
        if profile is None:
            return JSONResponse({"ok": False, "error": "没有绑定 profile"})
        from .. import groups as gp

        def _read():
            world = gp.load_world(profile)
            tpl = _group_tpl_dir()
            rows, conflicts, orphans = [], [], []
            for g in gp.load_groups(profile):
                gaps = g.gaps(world, tpl)
                rows.append({**g.to_dict(), "gaps": gaps, "ready": not gaps,
                             "say_line": g.say_line()})
            root = gp.groups_root(profile)
            for name, key in (("_conflicts.json", "conflicts"),
                              ("_orphans.json", "orphans")):
                p = root / name
                if p.exists():
                    try:
                        obj = json.loads(p.read_text(encoding="utf-8"))
                    except Exception:  # noqa: BLE001
                        obj = {}
                    if key == "conflicts":
                        conflicts = obj.get("conflicts") or []
                    else:
                        orphans = obj.get("orphans") or []
            return {
                "ok": True,
                "world": {"scenes": world.known_scenes(),
                          "objects": sorted(world.objects),
                          "keys": world.keys,
                          "terms": len(world.terms),
                          "taboos": len(world.taboos)},
                "groups": rows, "conflicts": conflicts, "orphans": orphans,
                "active_task": (_HUB.ui_task.status() if _HUB and _HUB.ui_task else None),
            }

        return JSONResponse(await asyncio.to_thread(_read))

    @app.post("/api/groups/sync")
    async def groups_sync(request: Request) -> JSONResponse:
        """手册改了立刻同步，不等出代。组是数据，不吃回归。"""
        _require_token_header(request)
        if profile is None:
            return JSONResponse({"ok": False, "error": "没有绑定 profile"})
        from .. import groups as gp

        try:
            info = await asyncio.to_thread(gp.sync_from_lessons, profile)
        except Exception as e:  # noqa: BLE001
            log.exception("同步指令组失败")
            return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"})
        return JSONResponse({"ok": True, **info})

    @app.post("/api/groups/route")
    async def groups_route(request: Request) -> JSONResponse:
        """只试匹配，不执行。页面上的"喊一句"用它 —— 想先看它听懂了没有再跑。"""
        _require_token_header(request)
        if profile is None:
            return JSONResponse({"ok": False, "error": "没有绑定 profile"})
        from .. import groups as gp

        body = {}
        with contextlib.suppress(Exception):
            body = await request.json() or {}
        text = str(body.get("text") or "").strip()

        def _do():
            world = gp.load_world(profile)
            r = gp.route(text, gp.load_groups(profile), world)
            out = r.to_dict()
            # ok 表示"认出这一组了吗"，不是"接口通不通"。覆盖它会害页面把
            # "没听懂"显示成"还跑不了" —— 两种状态对用户是完全不同的两件事。
            out["routed"] = r.ok
            # 缺口一起带回去：只"听"的时候最该看到的就是"认出来了，但缺什么"，
            # 不然用户点了「开始」才知道跑不了。
            gaps = r.group.gaps(world, _group_tpl_dir()) if r.group else []
            out["gaps"] = gaps
            out["ready"] = bool(r.group) and not gaps
            return out

        return JSONResponse(await asyncio.to_thread(_do))

    @app.post("/api/command")
    async def run_command(request: Request) -> JSONResponse:
        """用户的命令进来：认出是哪一组就交给组执行器，认不出就把候选摆出来。

        认不出时**不执行任何动作**。猜错的代价是几十秒到几分钟的错误操作，
        而"没听懂"只丢几秒钟。这一条比召回率重要。
        """
        _require_token_header(request)
        if profile is None:
            return JSONResponse({"ok": False, "error": "没有绑定 profile"})
        from .. import groups as gp

        body = {}
        with contextlib.suppress(Exception):
            body = await request.json() or {}
        text = str(body.get("text") or "").strip()
        if not text:
            return JSONResponse({"ok": False, "error": "命令是空的"})

        world = gp.load_world(profile)
        all_groups = gp.load_groups(profile)
        r = gp.route(text, all_groups, world)
        out = r.to_dict()
        if not r.ok:
            out.update({"ok": False, "routed": False,
                        "error": r.why or "没认出是哪一组"})
            return JSONResponse(out)
        tpl_dir = _group_tpl_dir()
        gaps = r.group.gaps(world, tpl_dir)
        if gaps and not bool(body.get("force")):
            # 缺模板还硬跑 = 每一步都点不出去，然后在超时里说"失败"，比不做更糟
            out.update({"ok": False, "routed": True, "ready": False, "gaps": gaps,
                        "error": "这一组还缺东西，跑不了：" + "；".join(gaps[:3])})
            return JSONResponse(out)

        fsm = _HUB.fsm if _HUB else None
        # 喊一句"刷k7"就是让它开始干活，不该再让人去别的页面点一次「开始」。
        # 但急停（PANIC）不自动解 —— 那是事故现场，必须人看过再解。
        if fsm is not None:
            from ..decision.fsm import State

            if fsm.state == State.PANIC:
                out.update({"ok": False, "routed": True,
                            "error": "现在是急停状态，先解除急停再喊"})
                return JSONResponse(out)
            if fsm.state == State.PAUSED:
                fsm.start()
                log.info("指令组启动前自动恢复运行（原状态 PAUSED）")

        runner = gp.GroupRunner(r.group, world, tpl_dir,
                                executor=(_HUB.executor if _HUB else None),
                                fsm=fsm)
        if _HUB is not None:
            _HUB.ui_task = runner
        log.info("指令组启动: %s（%d 步）← %r", r.group.name, len(r.group.steps), text)
        out.update({"ok": True, "routed": True, "ready": True,
                    "steps": len(r.group.steps),
                    "message": f"开始跑「{r.group.name}」，共 {len(r.group.steps)} 步"})
        return JSONResponse(out)

    # ---------------- 对话助手 ----------------
    # 四个入口对应四种东西：说话、切模式、看现场、改规则。
    # 注意 `/api/command`（喊组名）和这里不是一回事：那个是发起任务，
    # 这里是"和人交流并把结论变成可回滚的改动"。

    @app.get("/api/assistant")
    async def assistant_state(request: Request) -> JSONResponse:
        _require_token_header(request)
        from ..assistant import voice

        snap = asst.modes.snapshot()
        return JSONResponse({
            "ok": True,
            "mode": snap,
            "dialog": asst.dialog.snapshot(limit=12),
            "recent": asst.recent.snapshot(seconds=60, limit=20),
            "patches": (asst.patches.history(limit=8) if asst.patches else []),
            "voice": voice.capabilities(cfg),
            "suppressed": int(getattr(getattr(hub, "executor", None),
                                      "suppressed", 0) or 0),
            "model_ready": asst._model_ready(),
            "health": hub.health.status(),
            "experience": (hub.experience.summary()
                           if getattr(hub, "experience", None) else None),
        })

    @app.get("/api/health")
    async def health_state(request: Request) -> JSONResponse:
        """上线后体检：现在跟基线比是快了还是慢了、卡没卡、有没有自己回滚过。"""
        _require_token_header(request)
        return JSONResponse({"ok": True, **hub.health.status(),
                             "incidents": list(hub._health_incidents)})

    # ---------------- 上真机前自检 ----------------
    # doctor 管"能不能起来"，这里管"起来之后会不会真的动手、动得对不对"。
    # 两者在页面上放在一起看，但结论分开说 —— 混成一个"通过/不通过"会让人
    # 以为 dryrun 跑通就等于真机验证过了。

    @app.get("/api/preflight")
    async def preflight(request: Request) -> JSONResponse:
        _require_token_header(request)
        from .. import preflight as pf

        # 在跑的时候不碰串口：探活插一句 ping 会抢走正在下发命令的回显，
        # 既可能把好固件误判成故障，也会打乱那条命令的 ack。
        running = hub.fsm.state.value != "PAUSED"
        res = await asyncio.to_thread(
            pf.run, hub.cfg, hub, hub.executor, "", allow_io=not running)
        res["text"] = pf.as_text(res)
        return JSONResponse(res)

    # ---------------- 经验闭环（候选 → 晋升 / 驳回） ----------------
    # 体检发现问题只是"候选"，不直接改手册：同一个问题要复现够了才算规律。
    # 晋升走的是补丁那条路（预览/校验/留原件/可回滚），没有特权通道。

    @app.get("/api/experience")
    async def experience_list(request: Request) -> JSONResponse:
        _require_token_header(request)
        if hub.experience is None:
            return JSONResponse({"ok": False, "error": "没有绑定 profile",
                                 "items": [], "summary": {}})
        return JSONResponse({"ok": True,
                             "items": hub.experience.as_public(limit=30),
                             "rules": hub.experience.rules_learned(),
                             "summary": hub.experience.summary()})

    @app.post("/api/experience/promote")
    async def experience_promote(request: Request) -> JSONResponse:
        _require_token_header(request)
        from ..experience import ExperienceError

        body = {}
        with contextlib.suppress(Exception):
            body = await request.json() or {}
        if hub.experience is None:
            return JSONResponse({"ok": False, "error": "没有绑定 profile"})
        try:
            res = await asyncio.to_thread(hub.experience.promote,
                                          str(body.get("id") or ""), asst.patches)
        except ExperienceError as e:
            return JSONResponse({"ok": False, "error": str(e)})
        # 规则/配置刚写进文件，正在跑的那套还是旧的 —— 不推一遍就等于没学。
        live_msgs: list[str] = []
        if res.get("rule_patch"):
            live_msgs.append(str(await asyncio.to_thread(asst.make_live, "rule", {})))
        if res.get("config_patch"):
            # 配置补丁要带 payload 才对：apply_patch 是"把这份值套进内存"，
            # 不是"重读文件"。传空字典等于什么都没做，却会回"当场生效"。
            cfg_payload = (res["candidate"] or {}).get("config") or {}
            live_msgs.append(str(await asyncio.to_thread(
                asst.make_live, "config", cfg_payload)))
        live = "；".join(x for x in live_msgs if x)

        msg = "已经写进手册，编号 " + res["patch"]["id"]
        if res.get("rule_patch"):
            msg += f"；同时加了一条规则（{res['rule_patch']['id']}）"
        elif res.get("rule_error"):
            msg += "；" + str(res["rule_error"])
        if res.get("config_patch"):
            msg += f"；另外改了配置（{res['config_patch']['id']}）"
        elif res.get("config_error"):
            msg += "；" + str(res["config_error"])
        if live:
            msg += "。" + live
        return JSONResponse({"ok": True, **res, "live": live, "message": msg})

    @app.post("/api/experience/reject")
    async def experience_reject(request: Request) -> JSONResponse:
        _require_token_header(request)
        from ..experience import ExperienceError

        body = {}
        with contextlib.suppress(Exception):
            body = await request.json() or {}
        if hub.experience is None:
            return JSONResponse({"ok": False, "error": "没有绑定 profile"})
        try:
            row = await asyncio.to_thread(hub.experience.reject,
                                          str(body.get("id") or ""),
                                          str(body.get("why") or ""))
        except ExperienceError as e:
            return JSONResponse({"ok": False, "error": str(e)})
        return JSONResponse({"ok": True, "item": row, "message": "已驳回，不再自动提"})

    @app.post("/api/assistant/interpret", include_in_schema=False)
    @app.post("/api/assistant/ask")
    async def assistant_ask(request: Request) -> JSONResponse:
        """说一句话。本地命令会立刻生效，其余交给模型（模型不可用时离线回答）。

        `/api/assistant/interpret` 是别名：对话里给的接口清单用的是这个名字。
        """
        _require_token_header(request)
        body = {}
        with contextlib.suppress(Exception):
            body = await request.json() or {}
        text = str(body.get("text") or "").strip()
        if not text:
            return JSONResponse({"ok": False, "error": "说点什么"})
        use_model = bool(body.get("use_model", True))
        res = await asyncio.to_thread(asst.handle, text, source="text",
                                      use_model=use_model)
        return JSONResponse(res)

    @app.post("/api/assistant/mode")
    async def assistant_mode(request: Request) -> JSONResponse:
        """页面上四选一。和语音说"先停下"走的是同一个 set_mode。"""
        _require_token_header(request)
        body = {}
        with contextlib.suppress(Exception):
            body = await request.json() or {}
        name = str(body.get("mode") or "").strip()
        try:
            st = await asyncio.to_thread(
                asst.set_mode, name, seconds=body.get("seconds"),
                reason=str(body.get("reason") or "页面切换"))
        except ValueError:
            return JSONResponse({"ok": False, "error": f"没有这个模式: {name}"})
        return JSONResponse({"ok": True, "mode": st})

    @app.post("/api/voice/transcribe", include_in_schema=False)
    @app.post("/api/assistant/voice")
    async def assistant_voice(request: Request) -> JSONResponse:
        """上行一段录音 → 文字 → 走同一套对话。

        `/api/voice/transcribe` 是同一条路的别名：对话里给的接口清单用的是
        这个名字，页面上用的是 `/api/assistant/voice`。两个都留着，
        免得接第三方客户端时为了一个路径名改代码。

        服务端没装 sherpa 时返回 `unsupported=True`，页面据此改用浏览器识别。
        这里**不**尝试"降级识别"：把 webm 硬喂给识别器只会得到一段看着像话的错字。
        """
        _require_token_header(request)
        from ..assistant import voice

        audio = await request.body()
        if not audio:
            return JSONResponse({"ok": False, "error": "没收到音频"})
        try:
            text = await asyncio.to_thread(voice.transcribe, cfg, audio)
        except voice.VoiceUnavailable as e:
            return JSONResponse({"ok": False, "unsupported": True,
                                 "error": str(e), "hint": e.hint})
        if not text:
            return JSONResponse({"ok": False, "error": "没听清，再说一遍？"})
        res = await asyncio.to_thread(asst.handle, text, source="voice")
        res["heard"] = text
        return JSONResponse(res)

    @app.post("/api/assistant/tts")
    async def assistant_tts(request: Request):  # noqa: ANN201 - 返回 wav 或 JSON
        """文字 → wav。页面在浏览器没有 SpeechSynthesis 时才用这条。"""
        _require_token_header(request)
        from ..assistant import voice

        body = {}
        with contextlib.suppress(Exception):
            body = await request.json() or {}
        try:
            wav = await asyncio.to_thread(voice.synthesize, cfg,
                                          str(body.get("text") or ""))
        except voice.VoiceUnavailable as e:
            return JSONResponse({"ok": False, "unsupported": True,
                                 "error": str(e), "hint": e.hint})
        return StreamingResponse(iter([wav]), media_type="audio/wav")

    @app.get("/api/assistant/context")
    async def assistant_context(request: Request) -> JSONResponse:
        """最近 N 秒的现场（页面画时间线用，也是"刚才"的原样）。"""
        _require_token_header(request)
        seconds = float(request.query_params.get("seconds") or 60)
        return JSONResponse({"ok": True, **asst.recent.snapshot(seconds=seconds, limit=80)})

    @app.get("/api/assistant/frame")
    async def assistant_frame(request: Request):
        """"刚才那一帧"到底是哪一帧 —— 按时间戳取最近的一张关键帧。

        存的是**画面变化点**，不是每一帧：1Hz 全存一天下来能把 eMMC 写穿。
        取不到就退回当前这一帧（还活着的话），再没有就是 404。
        """
        _require_token_header(request)
        raw = request.query_params.get("ts") or ""
        ts: float | None = None
        with contextlib.suppress(ValueError):
            ts = float(raw) if raw else None
        path = await asyncio.to_thread(hub.keyframe, ts)
        if path is not None:
            return FileResponse(str(path), media_type="image/jpeg")
        jpeg = hub._latest_jpeg_bytes()
        if jpeg:
            return Response(content=jpeg, media_type="image/jpeg")
        return JSONResponse({"ok": False, "error": "还没有画面"}, status_code=404)

    # ---------------- 改规则（预览 / 应用 / 回滚） ----------------

    @app.post("/api/scripts/patch/preview", include_in_schema=False)
    @app.post("/api/assistant/patch/preview")
    async def patch_preview(request: Request) -> JSONResponse:
        _require_token_header(request)
        from ..assistant.patch import PatchError

        body = {}
        with contextlib.suppress(Exception):
            body = await request.json() or {}
        if asst.patches is None:
            return JSONResponse({"ok": False, "error": "没有绑定 profile"})
        try:
            pv = await asyncio.to_thread(
                asst.patches.preview, str(body.get("kind") or ""),
                body.get("payload") or {}, note=str(body.get("note") or ""),
                text=str(body.get("text") or ""))
        except PatchError as e:
            return JSONResponse({"ok": False, "error": str(e)})
        return JSONResponse({"ok": pv["ok"], **pv})

    @app.post("/api/scripts/patch/apply", include_in_schema=False)
    @app.post("/api/assistant/patch/apply")
    async def patch_apply(request: Request) -> JSONResponse:
        _require_token_header(request)
        from ..assistant.patch import PatchError

        body = {}
        with contextlib.suppress(Exception):
            body = await request.json() or {}
        if asst.patches is None:
            return JSONResponse({"ok": False, "error": "没有绑定 profile"})
        try:
            rec = await asyncio.to_thread(
                asst.patches.apply, str(body.get("kind") or ""),
                body.get("payload") or {}, note=str(body.get("note") or ""),
                text=str(body.get("text") or ""), stamp=str(body.get("stamp") or ""))
        except PatchError as e:
            return JSONResponse({"ok": False, "error": str(e)})
        live = await asyncio.to_thread(asst.make_live, rec["kind"],
                                       body.get("payload") or {})
        return JSONResponse({"ok": True, "patch": rec, "live": live})

    @app.post("/api/scripts/rollback", include_in_schema=False)
    @app.post("/api/scripts/patch/rollback", include_in_schema=False)
    @app.post("/api/assistant/patch/rollback")
    async def patch_rollback(request: Request) -> JSONResponse:
        _require_token_header(request)
        from ..assistant.patch import PatchError

        body = {}
        with contextlib.suppress(Exception):
            body = await request.json() or {}
        if asst.patches is None:
            return JSONResponse({"ok": False, "error": "没有绑定 profile"})
        try:
            rec = await asyncio.to_thread(asst.patches.rollback, str(body.get("id") or ""))
        except PatchError as e:
            return JSONResponse({"ok": False, "error": str(e)})
        # 回滚写回了文件，也要让正在跑的东西跟上（规则能重载；配置要重启）
        live = ""
        if rec.get("kind") == "rule":
            live = await asyncio.to_thread(asst.make_live, "rule", {})
        elif rec.get("kind") == "config":
            live = "配置文件已退回；当前进程里还是改过的那套，重启服务后彻底恢复。"
        return JSONResponse({"ok": True, "patch": rec, "live": live})

    @app.get("/api/assistant/patch/{pid}")
    async def patch_detail(pid: str, request: Request) -> JSONResponse:
        """看某一次改动到底改了什么（页面点开那一条时用）。

        账本里每条都存了 diff 和校验结果 —— 只给人看，不给它自己回读，
        免得"上次这样改过就照着来"变成一条隐形的规则。
        """
        _require_token_header(request)
        if asst.patches is None:
            return JSONResponse({"ok": False, "error": "没有绑定 profile"})
        rec = await asyncio.to_thread(asst.patches.get, pid)
        if rec is None:
            return JSONResponse({"ok": False, "error": f"没有这个补丁: {pid}"})
        return JSONResponse({"ok": True, "patch": rec})

    @app.get("/api/assistant/history")
    async def assistant_history(request: Request) -> JSONResponse:
        _require_token_header(request)
        limit = int(request.query_params.get("limit") or 30)
        return JSONResponse({"ok": True,
                             "turns": asst.dialog.snapshot(limit=limit)["turns"],
                             "patches": asst.patches.history(limit) if asst.patches else []})

    @app.delete("/api/groups/{gid}")
    async def groups_forget(gid: str, request: Request) -> JSONResponse:
        """忘掉一组。只由人触发 —— 学习过程永远不删组。"""
        _require_token_header(request)
        if profile is None:
            return JSONResponse({"ok": False, "error": "没有绑定 profile"})
        from .. import groups as gp

        ok = await asyncio.to_thread(gp.forget_group, profile, gid)
        return JSONResponse({"ok": ok, "message": "已删掉" if ok else "没有这一组"})

    # ---------------- 引擎版本 ----------------
    # 和"进化"是两件事：进化换配方（设备自己长），升级换引擎（只有人能推）。
    # 页面只做"看"和"退"：装新包要 sudo + 重启服务，从网页点一下只会让人
    # 以为已经换了。所以安装留在 shell，页面负责把话说清楚。

    @app.get("/api/upgrade")
    async def upgrade_state(request: Request) -> JSONResponse:
        _require_token_header(request)
        from .. import upgrade as upg
        from .. import version

        st = await asyncio.to_thread(upg.status, deep=True)
        st["running"] = version.ENGINE_VERSION
        # 指针和正在跑的代码不是同一个 → 说明切过了但还没重启
        st["pending"] = bool(st.get("engine_version")) and st["engine_version"] != st["running"]
        st["can_rollback"] = bool(st.get("previous")) and st["previous"] != st.get("engine_version")
        return JSONResponse(st)

    @app.post("/api/upgrade/rollback")
    async def upgrade_rollback(request: Request) -> JSONResponse:
        """退回上一版引擎。返回 restart_required=True 时，页面会提示"重启后生效"。"""
        _require_token_header(request)
        from .. import upgrade as upg

        res = await asyncio.to_thread(upg.rollback)
        return JSONResponse({"ok": bool(res.get("ok")), **res})

    @app.post("/api/preset/{name}")
    async def apply_named_preset(name: str, request: Request) -> JSONResponse:
        """一键切换拟人度预设（stable / human / custom）。"""
        from .settings import ConfigError, apply_patch, save_overlay
        from ..presets import preset_patch

        _require_token_header(request)
        try:
            patch = preset_patch(name)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        try:
            res = apply_patch(cfg, patch) if patch else {"changed": [],
                                                        "restart_required": []}
        except ConfigError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        saved = ""
        if patch and config_path:
            try:
                saved = save_overlay(config_path, patch)
            except OSError as e:
                log.warning("预设已生效但写盘失败: %s", e)
        log.info("已应用预设 %s（%d 项生效）", name, len(res["changed"]))
        return JSONResponse({"ok": True, "preset": name, "saved": saved, **res})

    # ---------------- 放行度：所有"慢一点/不许动"的开关 ----------------
    # 三档只动节奏类字段（tick_hz、轨迹时长、休息阈值…），安全铁律不在任何
    # 档位的 patch 里 —— 见 limits.SAFETY，那些放宽不会变快，只会变错。

    @app.get("/api/limits")
    async def limits_state(request: Request) -> JSONResponse:
        """当前档位 + 全部限制的当前值/三档值 + 安全铁律 + 被拦计数。"""
        _require_token_header(request)
        from .. import limits as lm

        return JSONResponse(lm.report(hub.cfg, hub.fsm.stats, hub.executor))

    @app.post("/api/limits/{tier}")
    async def limits_apply(tier: str, request: Request) -> JSONResponse:
        """切档。和改配置走同一条路：校验 → 原地生效 → 写覆盖文件。"""
        from .settings import ConfigError, apply_patch, save_overlay
        from .. import limits as lm

        _require_token_header(request)
        try:
            patch = lm.tier_patch(tier)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        try:
            res = apply_patch(hub.cfg, patch)
        except ConfigError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        saved = ""
        if config_path:
            try:
                saved = save_overlay(config_path, patch)
            except OSError as e:
                log.warning("放行度已生效但写盘失败: %s", e)
        # 采集帧率是"决策频率"的真正上限：配置改了必须推到代理，否则就是假生效
        pushed = await _push_capture(hub, hub.cfg)
        log.info("放行度切到 %s（%d 项生效，采集参数已推送=%s）",
                 tier, len(res["changed"]), pushed)
        body = lm.report(hub.cfg, hub.fsm.stats, hub.executor)
        return JSONResponse({"ok": True, "saved": saved, "pushed": pushed,
                             **res, "state": body})

    @app.delete("/api/config/overlay")
    async def drop_overlay(request: Request) -> JSONResponse:
        """删除运行时覆盖文件（回滚到 config.yaml 的原值，需重启生效）。"""
        from .settings import clear_overlay

        _require_token_header(request)
        if not config_path:
            raise HTTPException(status_code=400, detail="未配置 config_path")
        removed = clear_overlay(config_path)
        return JSONResponse({"ok": True, "removed": removed,
                             "note": "已删除覆盖文件，重启后回到 config.yaml 的原始值"})

    @app.get("/api/calibrate")
    async def calibrate(request: Request) -> JSONResponse:
        """相机标定反馈：亮度/清晰度/场景门/A框模板分数 + 文字建议。"""
        _require_token_header(request)
        return JSONResponse(hub.calibration_report())

    @app.post("/api/calibrate/hsv")
    async def calibrate_hsv(request: Request) -> JSONResponse:
        """从当前画面自动采样 HSV 范围（立即生效，并写入覆盖文件）。"""
        from .settings import save_overlay

        _require_token_header(request)
        res = hub.sample_hsv()
        if res.get("ok") and config_path:
            try:
                save_overlay(config_path, {"vision": {"hsv": res["hsv"]}})
            except OSError as e:
                log.warning("HSV 已生效但写盘失败: %s", e)
        return JSONResponse(res)

    @app.post("/api/test/plan")
    async def test_plan(request: Request) -> JSONResponse:
        """试跑：在当前画面上算"接下来几步会做什么"（不产生真实输入）。"""
        _require_token_header(request)
        return JSONResponse(hub.test_plan())

    @app.post("/api/test/send")
    async def test_send(request: Request) -> JSONResponse:
        """把命令真发给执行器（含 RP2040 串口 / hidg），返回 HID 报文。"""
        _require_token_header(request)
        body = await request.json()
        cmds = body.get("commands") if isinstance(body, dict) else None
        if not isinstance(cmds, list):
            raise HTTPException(status_code=400, detail="body 需要 {\"commands\": [...]}")
        return JSONResponse(hub.test_send(cmds))

    @app.get("/api/alerts")
    async def alerts(request: Request) -> JSONResponse:
        _require_token_header(request)
        return JSONResponse(hub.notifier.snapshot())

    @app.post("/api/alerts/test")
    async def alert_test(request: Request) -> JSONResponse:
        """发一条测试告警，验证 webhook / 邮件是否通。"""
        _require_token_header(request)
        hub.notifier.alert("测试告警", "这是一条来自 autogame 设置页的测试消息。",
                           level="info")
        return JSONResponse({"ok": True, "notify": hub.notifier.snapshot()})

    # ---------------- 图片审查页（图上步骤下） ----------------

    @app.get("/review", response_class=HTMLResponse)
    async def review_page() -> str:
        return (_ASSETS.parent / "web" / "review.html").read_text(encoding="utf-8")

    @app.get("/api/review/sources")
    async def review_sources(request: Request) -> JSONResponse:
        _require_token_header(request)
        batch_n = 0
        if (_ASSETS / "batch_report.json").exists():
            d = json.loads((_ASSETS / "batch_report.json").read_text(encoding="utf-8"))
            batch_n = len(d.get("reports", []))
        replays = sorted(p.name for p in (_ASSETS / "frames_replay").glob("*")
                         if (p / "manifest.json").exists())             if (_ASSETS / "frames_replay").exists() else []
        return JSONResponse({"batch": batch_n, "replays": replays})

    def _review_items(src: str, name: str | None) -> list[dict]:
        items: list[dict] = []
        if src == "batch":
            fp = _ASSETS / "batch_report.json"
            if not fp.exists():
                return items
            d = json.loads(fp.read_text(encoding="utf-8"))
            for i, r in enumerate(d.get("reports", []), 1):
                cmds = [c for tick in r["commands"] for c in tick]
                items.append({
                    "image": f"/review/img?src=batch&name={r['image']}",
                    "name": r["image"],
                    "size": r["size"],
                    "gate": "战斗" if r["in_game"][-1] else "UI拦截",
                    "verdict": r["verdict"],
                    "pillars": r["pillars"][-1] if r["pillars"] else [],
                    "aframe": r["aframe"],
                    "steps": [{"tick": f"tick{ti+1}",
                               "intent": (r.get("intents", [None] * 10)[ti]
                                          if ti < len(r.get("intents", [])) else None),
                               "cmds": tick}
                              for ti, tick in enumerate(r["commands"]) if tick],
                    "intents": r.get("intents", []),
                    "no_action": len(cmds) == 0,
                })
        elif src == "replay" and name:
            safe = _Path(name).name
            fp = _ASSETS / "frames_replay" / safe / "manifest.json"
            if not fp.exists():
                return items
            d = json.loads(fp.read_text(encoding="utf-8"))
            for it in d.get("items", []):
                # 命令兼容两种写法：旧回放写字符串，tools/video_batch.py 写完整对象
                raw = it.get("commands") or []
                texts = [c.get("text", "") if isinstance(c, dict) else str(c) for c in raw]
                items.append({
                    "image": f"/review/img?src=replay&name={safe}/{it['image']}",
                    "name": f"{safe}/{it['image']}",
                    "gate": it.get("gate", ""),
                    "pillars": it.get("pillars", []),
                    "aframe": it.get("aframe"),
                    "intent": it.get("intent", ""),
                    "size": it.get("frame_size"),
                    "verdict": "FAIL" if it.get("violations") else None,
                    "steps": [{"tick": it["t"], "intent": it.get("intent", ""),
                               "cmds": texts}] if texts else [],
                    "no_action": not texts,
                })
        return items

    @app.get("/api/review/manifest")
    async def review_manifest(request: Request, src: str = "batch",
                              name: str | None = None) -> JSONResponse:
        _require_token_header(request)
        items = _review_items(src, name)
        n_cmds = sum(1 for it in items for s in it["steps"] for _c in s["cmds"])
        n_a = sum(1 for it in items for s in it["steps"]
                  for c in s["cmds"] if c.startswith("KEY_TAP a"))
        return JSONResponse({"src": src, "total": len(items),
                             "with_commands": sum(1 for it in items if it["steps"]),
                             "commands": n_cmds, "a_press": n_a, "items": items})

    @app.get("/review/img")
    async def review_img(request: Request, src: str = "batch", name: str = ""):
        _require_token_header(request)
        clean = name.replace("\\", "/")
        if src == "replay":
            # name 形如 "<视频目录>/<帧文件>.png"，逐段取 basename 防目录穿越
            parts = [Path(p).name for p in clean.split("/") if p]
            if len(parts) != 2:
                raise HTTPException(status_code=400, detail="bad name")
            fp = _ASSETS / "frames_replay" / parts[0] / parts[1]
        else:
            fp = _ASSETS / "batch" / Path(clean).name
        if not fp.exists():
            raise HTTPException(status_code=404, detail="图片不存在")
        from fastapi.responses import FileResponse

        media = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                 ".png": "image/png"}.get(fp.suffix.lower(), "application/octet-stream")
        return FileResponse(fp, media_type=media)

    @app.post("/api/task")
    async def start_task(request: Request) -> JSONResponse:
        """启动 UI 场景任务（交易行/装备/天赋/技能配方）。

        body: {"recipe": "trade_buy|equip_best|talent_alloc|skill_swap"}
        激活后配方引擎接管决策循环（FSM 暂停），完成/失败自动恢复。
        """
        _require_token_header(request)
        body = await request.json()
        recipe_name = str(body.get("recipe", "")).strip()
        from ..decision.recipe import PRESET_RECIPES, RecipeRunner, required_templates
        steps = PRESET_RECIPES.get(recipe_name)
        if steps is None:
            raise HTTPException(status_code=400, detail=f"未知配方: {recipe_name}")
        ut = cfg.ui_task
        gcfg = GridConfig(roi=ut.grid_roi, cell_min=ut.cell_min, cell_max=ut.cell_max)
        tdir = Path(ut.templates_dir) if ut.templates_dir else (
            ROOT_TEMPLATES / "ui" if (ROOT_TEMPLATES / "ui").exists() else None)
        runner = RecipeRunner(recipe_name, steps, gcfg, tdir,
                              step_timeout_ms=ut.step_timeout_ms,
                              executor=_HUB.executor if _HUB else None,
                              rows_roi=ut.row_roi)
        # 预检：缺模板就**不启动**。历史行为是静默 skip 每一步 → 任务"成功启动"
        # 却一条命令都不发（实测 trade_buy 5 步全 skip），属于最难排查的假成功。
        missing = runner.missing_templates()
        if missing:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "缺少 UI 模板，任务未启动（避免空转假成功）",
                    "recipe": recipe_name,
                    "missing": missing,
                    "templates_dir": str(tdir) if tdir else str(ROOT_TEMPLATES / "ui"),
                    "how_to": "从自己的截图裁剪：python tools/crop_template.py "
                              "--image shot.png --rect x,y,w,h --out templates/ui/<文件名>",
                },
            )
        _HUB.ui_task = runner
        log.info("UI 任务启动: %s", recipe_name)
        return JSONResponse({"ok": True, "task": recipe_name,
                             "steps": len(steps),
                             "templates": [str(runner.template_path(n))
                                           for n in required_templates(steps)]})

    @app.post("/api/task/stop")
    async def stop_task(request: Request) -> JSONResponse:
        """停手。页面上这个按钮的含义是"立刻别动了"，所以任务和引擎一起停。

        只清任务不停引擎的后果：点了停，人走开了，它还在自由刷 ——
        按钮叫「停」，那就必须是真的停。
        """
        _require_token_header(request)
        hub_ = _HUB
        t = hub_.ui_task if hub_ else None
        if hub_ is not None:
            hub_.ui_task = None
            if t is not None and getattr(t, "wants_scene", False) and hub_.fsm is not None:
                hub_.fsm.stop()
        return JSONResponse({"ok": True, "stopped": t.name if t else None,
                             "engine": "paused" if (t is not None and
                                                    getattr(t, "wants_scene", False)) else ""})

    @app.get("/api/task/status")
    async def task_status(request: Request) -> JSONResponse:
        _require_token_header(request)
        hub_ = _HUB
        t = hub_.ui_task if hub_ else None
        return JSONResponse({"active": t is not None,
                             "status": t.status() if t else None})

    @app.exception_handler(Exception)
    async def on_error(request: Request, exc: Exception) -> JSONResponse:
        log.exception("请求异常 %s %s", request.method, request.url.path)
        return JSONResponse({"error": str(exc)}, status_code=500)

    return app


def get_hub() -> Hub:
    if _HUB is None:
        raise RuntimeError("Hub 未初始化")
    return _HUB
