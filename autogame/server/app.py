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

import cv2
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.responses import StreamingResponse

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


def create_app(cfg: AppConfig, hub: Hub, dashboard_html: str,
               config_path: str = "") -> FastAPI:
    global _HUB, _CFG
    _HUB = hub
    _CFG = cfg
    token = cfg.server.token or ""

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()
        hub.bind_loop(loop)
        decide_task = asyncio.create_task(hub.decision_loop(), name="decision-loop")
        watch_task = asyncio.create_task(hub.watchdog_loop(), name="frame-watchdog")
        log.info("决策循环已启动（含帧流看门狗）")
        try:
            yield
        finally:
            for t in (decide_task, watch_task):
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t

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
        """调整代理采集参数（代理 hello 时会拉取最新值）。"""
        _require_token_header(request)
        body = await request.json()
        m = SetCaptureMsg.model_validate({**body, "type": "set_capture"})
        if m.fps is not None:
            cfg.capture.fps = m.fps
        if m.jpeg_quality is not None:
            cfg.capture.jpeg_quality = m.jpeg_quality
        return JSONResponse({"ok": True})

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
        from .settings import LIVE_FIELDS, public_config
        from ..presets import DESCRIPTIONS

        _require_token_header(request)
        return JSONResponse({
            "config": public_config(cfg),
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
        _require_token_header(request)
        hub_ = _HUB
        t = hub_.ui_task if hub_ else None
        hub_.ui_task = None
        return JSONResponse({"ok": True, "stopped": t.name if t else None})

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
