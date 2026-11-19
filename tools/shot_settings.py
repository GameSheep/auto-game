"""给设置页拍一张图 —— 用来确认"我改的这块在真浏览器里长什么样"。

    python tools/shot_settings.py --url http://127.0.0.1:8791/settings#evolve \
        --token $(cat .token) --out /tmp/evolve.png

为什么不用 `chrome --headless --screenshot`：页面把 token 存在 localStorage 里，
命令行截图没法先写 localStorage，于是截出来的永远是一张"未连接服务"。
这里走 CDP：先导航、再 evaluate 写 token、再 reload、再截 —— 截的就是用户看到的。

依赖：本机有 Chrome/Edge + python 的 websocket-client（`pip install websocket-client`，
或者用 `websockets` 也行）。找不到浏览器时会打印清楚原因，不静默失败。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def find_browser() -> str:
    for c in _CANDIDATES:
        if Path(c).exists():
            return c
    for name in ("chrome", "chromium", "google-chrome", "msedge"):
        p = shutil.which(name)
        if p:
            return p
    raise SystemExit("找不到 Chrome/Edge。装一个，或用 --browser 指定路径。")


def _http_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read().decode())


class Cdp:
    """最小 CDP 客户端（只要 navigate / evaluate / screenshot）。"""

    def __init__(self, ws_url: str):
        try:
            import websocket  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise SystemExit("缺 websocket-client：pip install websocket-client") from e
        self.ws = websocket.create_connection(ws_url, timeout=30)
        self._id = 0

    def send(self, method: str, **params) -> dict:
        self._id += 1
        self.ws.send(json.dumps({"id": self._id, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self._id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result") or {}

    def close(self) -> None:
        try:
            self.ws.close()
        except Exception:  # noqa: BLE001
            pass


def shoot(url: str, out: Path, token: str = "", width: int = 1280,
          height: int = 1800, wait: float = 2.5, browser: str = "",
          js: str = "") -> Path:
    exe = browser or find_browser()
    profile = tempfile.mkdtemp(prefix="ag-shot-")
    port = 9333
    proc = subprocess.Popen(
        [exe, "--headless=new", "--disable-gpu", "--hide-scrollbars",
         f"--remote-debugging-port={port}", f"--user-data-dir={profile}",
         f"--window-size={width},{height}", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ws_url = ""
        for _ in range(60):
            try:
                pages = _http_json(f"http://127.0.0.1:{port}/json")
                page = next((p for p in pages if p.get("type") == "page"), None)
                if page:
                    ws_url = page["webSocketDebuggerUrl"]
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.2)
        if not ws_url:
            raise SystemExit("浏览器没起来（CDP 端口没响应）")

        c = Cdp(ws_url)
        c.send("Page.enable")
        c.send("Runtime.enable")
        c.send("Emulation.setDeviceMetricsOverride", width=width, height=height,
               deviceScaleFactor=1, mobile=False)
        c.send("Page.navigate", url=url)
        time.sleep(1.2)
        if token:
            setup_js = ("try{localStorage.setItem('ag_token','%s');"
                        "const t=document.getElementById('token');if(t)t.value='%s';"
                        "'ok'}catch(e){String(e)}" % (token, token))
            c.send("Runtime.evaluate", expression=setup_js, awaitPromise=False)
            c.send("Page.reload")
        time.sleep(wait)   # 等 fetch 回来渲染完
        if js:
            # 带上 js 才能截到"点了按钮之后"的样子。只看静态页会漏掉最该验的东西：
            # 回执、禁用态、错误提示。
            r = c.send("Runtime.evaluate", expression=js, awaitPromise=False,
                       returnByValue=True)
            val = (r.get("result") or {}).get("value")
            if r.get("exceptionDetails"):
                print("JS 报错:", json.dumps(r["exceptionDetails"], ensure_ascii=False)[:400])
            elif val is not None:
                print("JS:", val)
            time.sleep(1.5)
        res = c.send("Page.captureScreenshot", format="png", captureBeyondViewport=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(res["data"]))
        c.close()
        return out
    finally:
        proc.terminate()
        with __import__("contextlib").suppress(Exception):
            proc.wait(timeout=5)
        shutil.rmtree(profile, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="给设置页截图（带 token）")
    ap.add_argument("--url", default="http://127.0.0.1:8765/settings")
    ap.add_argument("--token", default=os.environ.get("AUTOGAME_TOKEN", ""))
    ap.add_argument("--out", default="shot.png")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=1800)
    ap.add_argument("--wait", type=float, default=2.5)
    ap.add_argument("--browser", default="")
    ap.add_argument("--js", default="", help="截图前先执行这段 JS（用于截交互态）")
    a = ap.parse_args(argv)
    p = shoot(a.url, Path(a.out), a.token, a.width, a.height, a.wait, a.browser, a.js)
    print(f"截图: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
