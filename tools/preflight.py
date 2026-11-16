"""部署前自检（preflight）—— 一命令确认"这台机器能不能跑起来"。

解决的真实问题：硬件/依赖/标定资产任何一项缺失，系统都不会崩，只会**静默降级**
（dxcam 缺失降级 mss、模板缺失降级启发式、串口缺失启动即失败），
现场排查成本极高。本工具把每一项显式检查并给出修复命令。

安全性：**不会按下任何游戏按键**。RP2040 后端只发一条 `stop_all`（确保执行器空闲），
dryrun/hidg 只做设备存在性与可写性检查。

用法：
  python tools/preflight.py                      # 本机全项检查（按 config.yaml）
  python tools/preflight.py --server http://192.168.1.20:8765 --token xxx
  python tools/preflight.py --json               # 机器可读输出（CI/部署脚本用）
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS: list[dict] = []


def check(name: str, ok: bool, detail: str = "", fix: str = "",
          warn_only: bool = False) -> bool:
    level = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    RESULTS.append({"item": name, "level": level, "detail": detail, "fix": fix})
    mark = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}[level]
    print(f" [{mark}] {name}" + (f" — {detail}" if detail else ""))
    if not ok and fix:
        print(f"       → {fix}")
    return ok


def check_config() -> dict | None:
    from autogame.config import load_config

    try:
        cfg = load_config(str(ROOT / "config.yaml"))
    except Exception as e:  # noqa: BLE001
        check("配置加载 config.yaml", False, str(e), "检查 YAML 语法与字段名")
        return None
    check("配置加载 config.yaml", True, f"server.port={cfg.server.port}")
    return cfg


def check_deps() -> None:
    for mod, why, fix in [
        ("cv2", "视觉检测", "pip install opencv-python"),
        ("numpy", "数值计算", "pip install numpy"),
        ("mss", "跨平台采集兜底", "pip install mss"),
        ("fastapi", "服务端", "pip install fastapi uvicorn"),
        ("yaml", "配置解析", "pip install PyYAML"),
    ]:
        try:
            __import__(mod)
            check(f"依赖 {mod}", True, why)
        except Exception:  # noqa: BLE001
            check(f"依赖 {mod}", False, f"缺失（{why}）", fix)

    # 可选依赖：缺失只降级，不算失败
    for mod, why, fix in [
        ("dxcam", "Windows 高性能采集（DXGI）", "pip install dxcam"),
        ("serial", "RP2040 串口执行器", "pip install pyserial"),
        ("smbus2", "PCA9685 机械执行器", "pip install smbus2"),
        ("pytesseract", "OCR 白名单", "pip install pytesseract + apt install tesseract-ocr"),
    ]:
        try:
            __import__(mod)
            check(f"可选依赖 {mod}", True, why)
        except Exception:  # noqa: BLE001
            check(f"可选依赖 {mod}", False, f"未安装（{why} 不可用）", fix, warn_only=True)


def check_capture(cfg) -> None:  # noqa: ANN001
    """试抓一帧（真实调用采集后端，不做任何输入）。"""
    try:
        from autogame.agent.capture import build_capture

        cap = build_capture(cfg.capture.backend, cfg.capture.region)
        frame = cap.grab()
        cap.close()
        if frame is None:
            check("采集后端试抓帧", False, "grab() 返回 None",
                  "确认屏幕未锁定 / 显卡驱动正常", warn_only=True)
            return
        h, w = frame.shape[:2]
        check("采集后端试抓帧", True, f"{type(cap).__name__} → {w}x{h}")
    except Exception as e:  # noqa: BLE001
        check("采集后端试抓帧", False, str(e)[:120], "检查 capture.backend 配置", warn_only=True)


def check_vision_assets(cfg) -> None:  # noqa: ANN001
    tpl_dir = ROOT / "templates"
    af = cfg.vision.aframe.template
    ok = bool(af) and (ROOT / af).exists()
    check("A框模板（templates/aframe.png）", ok,
          af or "未配置",
          "python tools/auto_template.py --image <截图> --kind aframe --rect x,y,w,h")

    skillbars = sorted(tpl_dir.glob("skillbar*.png"))
    check("场景门模板（templates/skillbar*.png）", bool(skillbars),
          f"{len(skillbars)} 张",
          "从战斗画面裁剪技能栏存为 templates/skillbar.png")

    if cfg.vision.whitelist or cfg.vision.whitelist_colors:
        check("白名单已配置", True,
              f"名称={cfg.vision.whitelist or '空'} 颜色={cfg.vision.whitelist_colors or '空'}")
    else:
        check("白名单已配置", False, "未配置 → 所有颜色光柱都会被捡（C3 铁律未生效）",
              'config.yaml 里设 vision.whitelist_colors: ["gold"]', warn_only=True)

    if cfg.vision.ocr and cfg.vision.ocr != "null":
        try:
            from autogame.vision.ocr import build_ocr

            build_ocr(cfg.vision)
            check(f"OCR 引擎 {cfg.vision.ocr}", True, "可加载")
        except Exception as e:  # noqa: BLE001
            check(f"OCR 引擎 {cfg.vision.ocr}", False, str(e)[:120],
                  "安装对应依赖或改回 ocr: null")


def check_executor(cfg) -> None:  # noqa: ANN001
    be = cfg.executor.backend
    if be == "dryrun":
        check("执行器后端 dryrun", True, "零真实输出（联调/测试用）")
        return

    if be == "rp2040":
        port = cfg.executor.serial_port
        if not port:
            check("执行器 rp2040", False, "未配置 serial_port", "config.yaml: executor.serial_port: COM5")
            return
        if platform.system() == "Windows" and not port.startswith("COM"):
            check("执行器 rp2040", False, f"Windows 端口应形如 COM5，当前 {port}")
            return
        try:
            import serial  # noqa: F401
        except Exception:  # noqa: BLE001
            check("执行器 rp2040", False, "pyserial 未安装", "pip install pyserial")
            return
        try:
            import serial

            with serial.Serial(port, cfg.executor.serial_baud, timeout=2) as ser:
                ser.write(b'{"op":"stop_all"}\n')  # 唯一安全的探活命令
                resp = ser.readline().decode("utf-8", "ignore").strip()
            ok = '"ok":true' in resp.replace(" ", "")
            check("执行器 rp2040 探活", ok, resp or "无响应",
                  "确认固件已烧录、串口未被其他程序占用")
        except Exception as e:  # noqa: BLE001
            check("执行器 rp2040 探活", False, str(e)[:120],
                  "确认设备已插入、端口号正确、驱动正常")
        return

    if be == "rpi5_hidg":
        for dev in [cfg.executor.hidg_keyboard, cfg.executor.hidg_mouse]:
            p = Path(dev)
            exists = p.exists()
            writable = exists and os.access(p, os.W_OK)
            check(f"HID gadget {dev}", exists and writable,
                  "存在且可写" if writable else ("存在但不可写" if exists else "不存在"),
                  "sudo bash scripts/setup_rpi5_hidg.sh（需 root 权限）")
        return

    if be == "pca9685":
        try:
            from smbus2 import SMBus
        except Exception:  # noqa: BLE001
            check("执行器 pca9685", False, "smbus2 未安装", "pip install smbus2")
            return
        try:
            bus = SMBus(cfg.executor.pca9685_bus)
            mode1 = bus.read_byte_data(0x40, 0x00)
            bus.close()
            check("执行器 pca9685 探活", True, f"PCA9685 在线 MODE1=0x{mode1:02X}")
        except Exception as e:  # noqa: BLE001
            check("执行器 pca9685 探活", False, str(e)[:120],
                  "确认 I2C 已启用（raspi-config）、接线 SDA/SCL、地址 0x40")
        return

    check("执行器后端", False, f"未知后端 {be}", "可选: dryrun/rp2040/rpi5_hidg/pca9685")


def check_models(cfg) -> None:  # noqa: ANN001
    """模型顾问自检（可选能力，缺失只 WARN 不 FAIL）。

    搬砖机器人"不卡壳"靠两层：规则恢复（总是可用）+ 模型顾问（看得懂画面）。
    模型没配好不会让系统崩，只会退化成纯启发式 —— 所以这里只提示，不阻断部署。
    """
    try:
        import httpx
    except Exception:  # noqa: BLE001
        check("模型顾问依赖", False, "httpx 未安装", "pip install httpx", warn_only=True)
        return

    vlm = cfg.decision.vlm
    if not vlm.enabled:
        check("视觉模型顾问(VLM)", True, "未启用（卡壳时用启发式恢复）", warn_only=True)
    else:
        base = vlm.base_url.rstrip("/")
        try:
            if vlm.provider == "ollama":
                r = httpx.get(f"{base}/api/tags", timeout=5)
                r.raise_for_status()
                names = [m.get("name", "") for m in r.json().get("models", [])]
                hit = any(n.split(":")[0] == vlm.model.split(":")[0] for n in names)
                check("视觉模型顾问(VLM)", hit,
                      f"{vlm.provider}/{vlm.model} 已就绪" if hit
                      else f"模型未拉取，现有: {', '.join(names[:5]) or '无'}",
                      f"ollama pull {vlm.model}", warn_only=True)
            else:
                r = httpx.get(f"{base}/v1/models", timeout=5,
                              headers={"Authorization": f"Bearer {vlm.api_key or ''}"})
                check("视觉模型顾问(VLM)", r.status_code < 500,
                      f"{vlm.provider} HTTP {r.status_code}", warn_only=True)
        except Exception as e:  # noqa: BLE001
            check("视觉模型顾问(VLM)", False, f"不可达: {str(e)[:80]}",
                  f"确认 {base} 已启动（ollama serve）", warn_only=True)

    llm = cfg.decision.llm
    if not llm.enabled:
        check("文本策略顾问(LLM)", True, "未启用（不影响自主恢复）", warn_only=True)
    else:
        try:
            r = httpx.get(f"{llm.base_url.rstrip('/')}/api/tags", timeout=5)
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
            hit = any(n.split(":")[0] == llm.model.split(":")[0] for n in names)
            check("文本策略顾问(LLM)", hit,
                  f"ollama/{llm.model} 已就绪" if hit
                  else f"模型未拉取，现有: {', '.join(names[:5]) or '无'}",
                  f"ollama pull {llm.model}", warn_only=True)
        except Exception as e:  # noqa: BLE001
            check("文本策略顾问(LLM)", False, f"不可达: {str(e)[:80]}",
                  "确认 ollama 已启动", warn_only=True)

    auto = cfg.decision.autonomy
    check("自主恢复", auto.enabled,
          f"启用（冻结{auto.frozen_window_s:.0f}s / 空闲{auto.idle_window_s:.0f}s / "
          f"停滞{auto.stall_window_s:.0f}s，"
          f"{'无限重试' if auto.max_cycles == 0 else f'最多{auto.max_cycles}轮'}）",
          "config.yaml: decision.autonomy.enabled: true")


def check_server(url: str | None, token: str | None) -> None:
    if not url:
        return
    try:
        import httpx

        headers = {"X-Token": token} if token else {}
        r = httpx.get(f"{url.rstrip('/')}/api/state", headers=headers, timeout=5)
        if r.status_code == 401:
            check("服务端可达 + 鉴权", False, "401（token 错误或未提供）",
                  "使用 .token 文件里的 token：--token <token>")
        elif r.status_code == 200:
            st = r.json()
            check("服务端可达 + 鉴权", True,
                  f"state={st.get('state')} frame_seq={st.get('frame_seq')} "
                  f"executor={st.get('executor', {}).get('backend')}")
        else:
            check("服务端可达 + 鉴权", False, f"HTTP {r.status_code}")
    except Exception as e:  # noqa: BLE001
        check("服务端可达 + 鉴权", False, str(e)[:120], "确认服务端已启动、IP/端口正确")


def main() -> None:
    ap = argparse.ArgumentParser(description="部署前自检（不发任何游戏按键）")
    ap.add_argument("--server", default=None, help="服务端 URL（检查可达性与鉴权）")
    ap.add_argument("--token", default=None, help="服务端 token（默认读 .token）")
    ap.add_argument("--skip-capture", action="store_true", help="跳过屏幕试抓（无显示环境）")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    if not args.json:
        print("=" * 62)
        print(" autogame preflight — 部署前自检（不会按下任何游戏按键）")
        print("=" * 62)

    cfg = check_config()
    check_deps()
    if cfg is not None:
        if not args.skip_capture:
            check_capture(cfg)
        check_vision_assets(cfg)
        check_executor(cfg)
        check_models(cfg)
    token = args.token
    if token is None and (ROOT / ".token").exists():
        token = (ROOT / ".token").read_text(encoding="utf-8").strip()
    check_server(args.server, token)

    fails = [r for r in RESULTS if r["level"] == "FAIL"]
    warns = [r for r in RESULTS if r["level"] == "WARN"]
    if args.json:
        print(json.dumps({"results": RESULTS,
                          "fail": len(fails), "warn": len(warns)},
                         ensure_ascii=False, indent=1))
    else:
        print("-" * 62)
        print(f" 结论: PASS {len(RESULTS) - len(fails) - len(warns)} / "
              f"WARN {len(warns)} / FAIL {len(fails)}")
        if fails:
            print(" 必须修复的项：")
            for r in fails:
                print(f"   ✗ {r['item']}: {r['detail']}")
                if r["fix"]:
                    print(f"     → {r['fix']}")
        elif warns:
            print(" 可用，但有降级项（见上方 ! 标记）")
        else:
            print(" 全部通过，可以启动。")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
