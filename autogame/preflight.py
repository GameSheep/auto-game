"""上真机前自检 —— 别等插上硬件才发现"它根本没动"。

## 它和 doctor.py 的分工

`doctor.py` 回答"这台机器能不能**起来**"：数据根可写、配置能解析、模板能解码、
磁盘够。它的时机是升级前，拦的是"起不来"。

本模块回答另一个问题："它会不会真的动手，而且**动得对不对**"。时机是上机前。

两者不能合成一个，因为失败形态完全不同：起不来有异常栈，看得见；而真机上真正
难查的是**静默失效** ——

    串口号配对了一半，打开的是另一个 CDC 设备   → 命令石沉大海，主机当成功
    帧是 1280×720，screen_w 还是默认 1920×1080  → 点击整体偏到别处
    采集代理没跑                                → 一帧都没有，循环空转
    后端还是 dryrun                             → 一切正常，只是没人按键
    模式停在"暂停"                              → 页面显示运行中，动作全被拦

这几种在日志里都干干净净。所以自检不能"等错误发生"，只能主动去问。

## 结论为什么是三档

    blocked  有硬伤，上机就是白跑
    dry      链路通，但**不会真的动手**（dryrun / mock 总线）
    ok       会动手，且该对的都对上了

`dry` 是刻意保留的：先用 dryrun 在真机环境跑通链路是完全合理的用法，
把它算成失败会逼人说谎。但它绝不能显示成 `ok` —— 那正是"以为在跑"的来源。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

Check = dict[str, Any]

#: 多久没有新帧就算"采集断了"。采集默认 2fps，3 秒 = 漏了 6 帧。
FRAME_GAP_FAIL_S = 3.0

#: 模式/门禁这些"现在不动"的状态，只算提示不算拦。
_NOWARN_LEVELS = ("ok", "warn", "fail", "skip")


def _c(name: str, level: str, detail: str, hint: str = "") -> Check:
    assert level in _NOWARN_LEVELS, level
    return {"name": name, "level": level, "detail": detail, "hint": hint}


# ---------------------------------------------------------------- 执行器

def _check_executor(cfg, executor, *, allow_io: bool, timeout_s: float) -> tuple[Check, dict]:
    """问一次后端：动作能不能真的发出去。返回 (检查项, 探活原文)。"""
    backend = str(getattr(getattr(cfg, "executor", None), "backend", "dryrun"))
    if executor is None:
        return _c("执行器探活", "skip",
                  "没有运行中的执行器（服务没起，或还没建）",
                  "先启动服务再自检，或用 --offline 只看配置"), {}

    if not allow_io:
        # 正在跑的时候**不能**去串口里插一句 ping：回显会被正在下发的命令抢走，
        # 既可能把正常固件误判成故障，也会打乱那条命令的 ack。
        return _c("执行器探活", "skip",
                  "服务正在跑，跳过串口探活（避免抢走命令回显）",
                  "先点「暂停」再自检，或在启动前用命令行 --offline 检查"), {}

    try:
        got = executor.probe(timeout_s=timeout_s) if _accepts_timeout(executor) \
            else executor.probe()
    except Exception as e:  # noqa: BLE001
        return _c("执行器探活", "fail", f"{backend} 探活抛异常：{type(e).__name__}: {e}",
                  "这通常是后端实现或驱动的问题，看服务日志"), {}

    if got.get("ok"):
        lvl = "ok"
    else:
        lvl = "fail"
    return _c("执行器探活", lvl, str(got.get("detail") or ""), str(got.get("hint") or "")), got


def _accepts_timeout(obj) -> bool:
    try:
        import inspect

        return "timeout_s" in inspect.signature(obj.probe).parameters
    except (TypeError, ValueError):
        return False


def _check_release(executor, *, allow_io: bool) -> Check:
    """能不能松手 —— 这条比"能不能按键"更要紧。

    按下之后松不掉是全项目最坏的一种故障（人物一直往前走，撞墙也在走）。
    检查方式就是发一次 stop_all：它本身**只松不按**，是最安全的探针。
    """
    if executor is None or not allow_io:
        return _c("能松手", "skip", "没有可用的执行器实例，跳过")
    from .protocol import Command

    before = int(getattr(executor, "dropped", 0) or 0)
    try:
        executor.execute(Command(op="stop_all"))
    except Exception as e:  # noqa: BLE001
        return _c("能松手", "fail", f"stop_all 发不出去：{type(e).__name__}: {e}",
                  "按下键之后松不掉是最坏的情况，先修这个再上机")
    if getattr(executor, "panicked", False):
        return _c("能松手", "warn",
                  "执行器处于急停状态：除了 stop_all，其它动作都会被丢掉",
                  "确认现场安全后点「复位」")
    if int(getattr(executor, "dropped", 0) or 0) > before:
        return _c("能松手", "fail", "stop_all 被丢弃了", "急停/闸门把松手也拦了，这是 bug")
    return _c("能松手", "ok", "stop_all 下发成功")


def _check_gate(executor) -> Check:
    """动作闸门开着没有。关着的话"能动手"是假的。"""
    gate = getattr(executor, "action_gate", None) if executor is not None else None
    if executor is None:
        return _c("动作闸门", "skip", "没有执行器实例")
    if gate is None:
        return _c("动作闸门", "ok", "没有装助手闸门，动作总是放行")
    try:
        allowed = bool(gate())
    except Exception as e:  # noqa: BLE001
        return _c("动作闸门", "fail", f"闸门求值抛异常：{type(e).__name__}: {e}", "")
    if allowed:
        return _c("动作闸门", "ok", "当前模式允许动作")
    return _c("动作闸门", "fail",
              "当前模式不允许动作（暂停/观察）—— 现在上机它一步都不会走",
              "把模式切回「协作」再开始")


# ---------------------------------------------------------------- 帧与几何

def _check_frames(hub) -> Check:
    if hub is None:
        return _c("采集在流", "skip", "服务没在跑，看不到帧", "启动服务后再自检")
    snap = hub.snapshot()
    gap = snap.get("frame_gap_s")
    if not snap.get("frame_seq"):
        return _c("采集在流", "fail", "一帧都没收到过",
                  "采集代理（autogame.agent）没在跑？它在游戏 PC 上，要单独启动")
    if gap is None:
        return _c("采集在流", "fail", "没有最近一帧的时间戳", "重启采集代理试试")
    if gap > FRAME_GAP_FAIL_S:
        return _c("采集在流", "fail",
                  f"最近一帧是 {gap:.1f} 秒前（超过 {FRAME_GAP_FAIL_S:.0f} 秒没新帧）",
                  "采集代理挂了 / 网络断了 / 游戏窗口最小化了")
    return _c("采集在流", "ok", f"{gap:.1f} 秒前刚收到一帧（共 {snap['frame_seq']} 帧）")


def _check_geometry(cfg, hub) -> Check:
    """帧尺寸和 screen_w/h 对不对得上 —— 真机上"点击全偏"的唯一来源。

    这条必须拦死，不能只提示：坐标是**帧坐标**，而 HID 绝对坐标是按
    `executor.screen_w/h` 归一化的。两者不一致时，点画面中心会落到屏幕别处，
    而所有环节都"成功"，没有一条日志能告诉你点歪了。
    """
    if hub is None:
        return _c("画面尺寸", "skip", "服务没在跑，拿不到帧尺寸")
    snap = hub.snapshot()
    scene = snap.get("scene") or {}
    fw, fh = int(scene.get("w") or 0), int(scene.get("h") or 0)
    if not fw or not fh:
        return _c("画面尺寸", "skip", "还没有有效的帧，尺寸未知")
    ecfg = getattr(cfg, "executor", None)
    sw = int(getattr(ecfg, "screen_w", 0) or 0)
    sh = int(getattr(ecfg, "screen_h", 0) or 0)

    region = list(getattr(getattr(cfg, "capture", None), "region", None) or [])
    if region and len(region) == 4:
        left, top, rw, rh = (int(x) for x in region)
        if left or top:
            return _c("画面尺寸", "fail",
                      f"采集区域从 ({left},{top}) 开始，但下发坐标时没加这个偏移 —— "
                      f"所有点击都会偏 ({left},{top}) 个像素",
                      "把 capture.region 改成 [0, 0, 宽, 高]，或去掉 region 用全屏")
        if (rw, rh) != (sw, sh):
            return _c("画面尺寸", "fail",
                      f"采集区域是 {rw}×{rh}，而 executor 按 {sw}×{sh} 归一化",
                      f"把 executor.screen_w/h 改成 {rw}/{rh}")

    if (fw, fh) != (sw, sh):
        # 算给人看：点帧中心会落到屏幕哪里
        def _map(v, f, s):
            return int(v * s / max(1, f))

        tx = _map(fw // 2, fw, sw)
        ty = _map(fh // 2, fh, sh)
        return _c("画面尺寸", "fail",
                  f"帧是 {fw}×{fh}，executor 按 {sw}×{sh} 归一化 —— "
                  f"点画面中心 ({fw // 2},{fh // 2}) 会落到屏幕 ({tx},{ty})",
                  f"把 executor.screen_w/screen_h 改成 {fw}/{fh}（这是游戏窗口的实际分辨率）")
    return _c("画面尺寸", "ok", f"{fw}×{fh}，与 executor 归一化用的尺寸一致")


def _check_character(cfg, hub) -> Check:
    if hub is None:
        return _c("角色位置", "skip", "服务没在跑")
    scene = (hub.snapshot().get("scene") or {})
    w, h = int(scene.get("w") or 0), int(scene.get("h") or 0)
    ch = scene.get("character")
    if not w or not h:
        return _c("角色位置", "skip", "还没有有效的帧")
    if not ch or len(ch) < 2:
        return _c("角色位置", "warn", "这一帧没定位到角色",
                  "角色被 UI 挡住 / 血条颜色变了 / 模板不匹配；门禁可能一直不过")
    x, y = int(ch[0]), int(ch[1])
    if not (0 <= x < w and 0 <= y < h):
        return _c("角色位置", "fail",
                  f"角色坐标 ({x},{y}) 落在 {w}×{h} 画面外",
                  "标定的参考分辨率与实际画面不一致，重新标定")
    if x < 8 or y < 8 or x > w - 8 or y > h - 8:
        return _c("角色位置", "warn", f"角色贴在画面边缘 ({x},{y})",
                  "贴边时导航容易误判方向，确认这是真的在角落")
    return _c("角色位置", "ok", f"({x},{y})，在 {w}×{h} 画面内")


def _check_gate_scene(hub) -> Check:
    """现在画面上是不是游戏内 —— 不是的话跑起来也不会动，但这不是故障。"""
    if hub is None:
        return _c("当前画面", "skip", "服务没在跑")
    scene = (hub.snapshot().get("scene") or {})
    if not scene.get("w"):
        return _c("当前画面", "skip", "还没有有效的帧")
    if scene.get("in_game"):
        return _c("当前画面", "ok", "识别为游戏内可操作画面")
    why = str(scene.get("gate_reason") or "")
    return _c("当前画面", "warn",
              f"现在不在游戏内（门禁结论：{why or '未知'}）—— 跑起来也不会动作",
              "把游戏切到主界面再自检，否则这条只是说明现状")


# ---------------------------------------------------------------- 配置侧

def _check_mode(hub) -> Check:
    if hub is None or getattr(hub, "modes", None) is None:
        return _c("运行模式", "skip", "没有模式管理器")
    snap = hub.modes.snapshot()
    if snap.get("allows_action"):
        return _c("运行模式", "ok", f"{snap.get('label') or snap.get('mode')}")
    return _c("运行模式", "fail",
              f"模式是「{snap.get('label') or snap.get('mode')}」，不允许动作",
              "切回「协作」模式")


def _check_arbiter(cfg) -> Check:
    """arbiter 在 enforce 但模型端点配错 —— 上机后每帧都超时，看着像卡死。"""
    arb = getattr(getattr(cfg, "decision", None), "arbiter", None)
    if arb is None:
        return _c("裁决层", "skip", "没有 arbiter 配置")
    if getattr(arb, "mode", "off") != "enforce":
        return _c("裁决层", "ok", f"mode={getattr(arb, 'mode', '?')}（不会自己抢动作）")
    planner = getattr(arb, "planner", None)
    base = str(getattr(planner, "base_url", "") or "")
    if not base:
        return _c("裁决层", "warn", "arbiter 是 enforce，但 planner 没配 base_url",
                  "每次兜底都会失败，白白拖慢每一帧")
    if "127.0.0.1" in base or "localhost" in base:
        return _c("裁决层", "warn", f"arbiter 是 enforce，planner 指向本机（{base}）",
                  "确认本机真的跑着模型服务；没跑的话每帧都会超时")
    return _c("裁决层", "ok", f"mode=enforce，planner={base}")


def _check_capture_cfg(cfg) -> Check:
    cap = getattr(cfg, "capture", None)
    if cap is None:
        return _c("采集配置", "skip", "没有采集配置")
    fps = float(getattr(cap, "fps", 0) or 0)
    hz = float(getattr(getattr(cfg, "decision", None), "tick_hz", 0) or 0)
    if fps <= 0:
        return _c("采集配置", "fail", "capture.fps 是 0，采不到帧", "设成 2 或更高")
    if hz > fps:
        return _c("采集配置", "warn",
                  f"决策 {hz:g}Hz 高于采集 {fps:g}fps —— 同一个画面会被决策多次",
                  f"把 decision.tick_hz 降到 {fps:g} 以下，或提高 capture.fps")
    return _c("采集配置", "ok", f"采集 {fps:g}fps / 决策 {hz:g}Hz")


# ---------------------------------------------------------------- 汇总

def run(cfg=None, hub=None, executor=None, profile: str = "", *,
        allow_io: bool = True, timeout_s: float = 1.5) -> dict:
    """跑全套上机自检。

    `cfg` 不传时按运行时同样的方式加载（带 profile）。`hub` 不传就只能查配置侧，
    帧/模式相关的项会报 skip —— 这对应"服务还没启动，先看看配的对不对"。
    """
    from . import doctor

    pid = profile
    if cfg is None:
        pid, _note = doctor._resolve_profile_id(profile)
        cfg_check, cfg = doctor._check_config(pid)
    else:
        cfg_check = _c("配置能解析", "ok", "使用传入的运行中配置")

    if executor is None and hub is not None:
        executor = getattr(hub, "executor", None)

    checks: list[Check] = [cfg_check]

    cli_ok = False
    if cfg is not None:
        ex_check, probe = _check_executor(cfg, executor, allow_io=allow_io,
                                          timeout_s=timeout_s)
        checks.append(ex_check)
        cli_ok = bool(probe.get("actuates")) and bool(probe.get("ok"))
        checks.append(_check_release(executor, allow_io=allow_io))
        checks.append(_check_gate(executor))
        checks.append(_check_frames(hub))
        checks.append(_check_geometry(cfg, hub))
        checks.append(_check_character(cfg, hub))
        checks.append(_check_gate_scene(hub))
        checks.append(_check_mode(hub))
        checks.append(_check_arbiter(cfg))
        checks.append(_check_capture_cfg(cfg))
        # 内容侧：模板/标答/配方/磁盘 —— 和 doctor 同一套判断，不重复实现
        _p, tpl, lab, warns = doctor._check_profile(cfg, pid)
        checks.extend([tpl, lab])
        checks.append(doctor._check_recipes(cfg))
        checks.extend(_c("数据提示", "warn", w) for w in warns)
        checks.append(doctor._check_disk())

    fails = [c for c in checks if c["level"] == "fail"]
    warns = [c for c in checks if c["level"] == "warn"]

    # actuates：这台机器到底会不会真的按键。dryrun/mock 总线都会说 False。
    actuates = bool(executor is not None and getattr(executor, "actuates", True))
    probed = next((c for c in checks if c["name"] == "执行器探活"), None)
    if probed and probed["level"] == "skip":
        # 没探活成功就不能宣称"会动手"，但也不能因此判 blocked
        actuates = False

    if fails:
        verdict = "blocked"
        summary = f"先别上真机：{len(fails)} 项硬伤（{fails[0]['name']} 等）"
    elif not actuates:
        verdict = "dry"
        skipped = bool(probed and probed["level"] == "skip")
        if skipped:
            # "没验证"和"确定不会动"是两件事，混成一句话会让人以为链路已经通了
            summary = ("没验证硬件：执行器探活被跳过了，现在没法断言它会动手 —— "
                       "停掉服务、不做 --offline，再自检一次")
        else:
            summary = ("链路是通的，但**不会真的动手**（dryrun / mock 总线）"
                       "—— 想真跑要换成真后端")
    else:
        verdict = "ok"
        summary = "可以上真机" + (f"，另有 {len(warns)} 项提示" if warns else "")
    _ = cli_ok  # 保留：将来要做"多次采样取多数"时用

    return {
        "verdict": verdict,
        "ok": verdict == "ok",
        "actuates": actuates,
        "summary": summary,
        "checks": checks,
        "blockers": [c["name"] for c in fails],
        "warns": [c["name"] for c in warns],
        "at": round(time.time(), 1),
    }


def as_text(res: dict) -> str:
    icon = {"ok": "OK  ", "warn": "注意", "fail": "失败", "skip": "跳过"}
    lines = [f"[{icon.get(c['level'], c['level'])}] {c['name']}  {c['detail']}"
             for c in res["checks"]]
    lines.append("")
    for c in res["checks"]:
        if c.get("hint") and c["level"] in ("fail", "warn"):
            lines.append(f"       → {c['hint']}")
    lines.append("")
    lines.append("结论：" + {"ok": "可以上真机", "dry": "能跑但不会动手",
                             "blocked": "不能上机"}[res["verdict"]]
                 + " —— " + res["summary"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="autogame 上真机前自检")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--profile", default="", help="要检查哪个 profile")
    ap.add_argument("--offline", action="store_true",
                    help="不去碰硬件（只查配置和文件，不打开串口）")
    ap.add_argument("--timeout", type=float, default=1.5, help="探活超时（秒）")
    args = ap.parse_args(argv)

    if args.offline:
        res = run(profile=args.profile, allow_io=False)
    else:
        # 命令行自检：临时建一个执行器去问，问完关掉（不占用服务运行的串口）
        from .config import load_config
        from .executor import build_executor
        from . import doctor

        pid, _ = doctor._resolve_profile_id(args.profile)
        path = doctor._config_path()
        cfg = load_config(str(path) if path else None, profile=pid or None)
        ex = None
        try:
            ex = build_executor(cfg.executor)
        except Exception as e:  # noqa: BLE001
            ex = None
            cfg_note = e  # noqa: F841
        try:
            res = run(cfg=cfg, executor=ex, profile=pid, timeout_s=args.timeout)
        finally:
            if ex is not None:
                try:
                    ex.close()
                except Exception:  # noqa: BLE001
                    pass

    if args.json:
        print(json.dumps(res, ensure_ascii=False))
    else:
        print(as_text(res))
    return 0 if res["verdict"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
