"""告警通知 —— 满足规则时推 webhook / 邮件。

为什么用 stdlib（urllib/smtplib）而不是加依赖：
通知是"锦上添花"的功能，不该为了它给部署环境多装一个包。

设计要点：
  - **发送不阻塞**：webhook/邮件都在后台线程里发，决策循环绝不等网络。
  - **去重限流**：同一规则在 `min_interval_s` 内只发一次，否则"包满"会每秒提醒。
  - **规则一次性**：跨过阈值只提醒一次（除非 `repeat: true`），避免刷屏。
  - **失败静默**：通知失败只记日志 + 页面可见，绝不因此影响主流程。
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import threading
import time
import urllib.error
import urllib.request
from email.mime.text import MIMEText
from email.header import Header
from typing import Any

from ..config import NotifyConfig

log = logging.getLogger(__name__)


class Notifier:
    """规则引擎 + 多通道发送。"""

    def __init__(self, cfg: NotifyConfig) -> None:
        self.cfg = cfg
        self.history: list[dict] = []      # 最近告警（页面展示）
        self._fired: dict[str, float] = {}  # rule_name → 上次触发时间
        self._sent_ok = 0
        self._sent_fail = 0
        self._lock = threading.Lock()

    # ---------------- 规则检查 ----------------

    def check(self, metrics: dict[str, float], now: float | None = None) -> list[dict]:
        """检查所有规则，返回本次触发的告警列表（同时异步发送）。"""
        if not self.cfg.enabled:
            return []
        now = time.time() if now is None else now
        fired: list[dict] = []
        for rule in self.cfg.rules:
            if not rule.enabled:
                continue
            value = float(metrics.get(rule.metric, 0.0) or 0.0)
            if value < rule.threshold:
                continue
            last = self._fired.get(rule.name, 0.0)
            if not rule.repeat and last > 0:
                continue                      # 一次性规则：只提醒一次
            if now - last < self.cfg.min_interval_s:
                continue                      # 限流
            self._fired[rule.name] = now
            item = {
                "ts": now,
                "rule": rule.name,
                "metric": rule.metric,
                "value": value,
                "threshold": rule.threshold,
                "text": rule.message or f"{rule.name}（{rule.metric}={value:g}）",
            }
            fired.append(item)
            self._record(item)
            self.send(f"[autogame] {rule.name}", item["text"])
        return fired

    def alert(self, title: str, text: str, level: str = "warn") -> None:
        """即时告警（不走规则，比如"自动停止""需要人工"）。"""
        item = {"ts": time.time(), "rule": title, "metric": "event",
                "value": 0, "threshold": 0, "text": text, "level": level}
        self._record(item)
        if self.cfg.enabled:
            self.send(f"[autogame] {title}", text)

    def _record(self, item: dict) -> None:
        with self._lock:
            self.history.append(item)
            if len(self.history) > 200:
                del self.history[:-120]
        log.warning("告警: %s — %s", item["rule"], item["text"])

    # ---------------- 发送 ----------------

    def send(self, title: str, text: str) -> None:
        """异步发送（Bark / webhook / 邮件，配了就都发）。"""
        if self.cfg.webhook_format == "bark" and self.cfg.bark_key:
            threading.Thread(target=self._send_bark, args=(title, text),
                             daemon=True).start()
        elif self.cfg.webhook_url:
            threading.Thread(target=self._send_webhook, args=(title, text),
                             daemon=True).start()
        if self.cfg.smtp_host and self.cfg.smtp_to:
            threading.Thread(target=self._send_mail, args=(title, text),
                             daemon=True).start()

    def _payload(self, title: str, text: str) -> bytes:
        fmt = self.cfg.webhook_format
        if fmt == "wecom":
            body: Any = {"msgtype": "text", "text": {"content": f"{title}\n{text}"}}
        elif fmt == "dingtalk":
            body = {"msgtype": "text", "text": {"content": f"{title}\n{text}"}}
        elif fmt == "slack":
            body = {"text": f"*{title}*\n{text}"}
        else:
            body = {"title": title, "text": text, "ts": time.time(), "source": "autogame"}
        return json.dumps(body, ensure_ascii=False).encode("utf-8")

    def _send_webhook(self, title: str, text: str) -> None:
        try:
            req = urllib.request.Request(
                self.cfg.webhook_url,
                data=self._payload(title, text),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=8) as resp:  # noqa: S310
                resp.read(2048)
            self._sent_ok += 1
        except (urllib.error.URLError, OSError, ValueError) as e:
            self._sent_fail += 1
            log.warning("webhook 告警发送失败: %s", e)

    def _send_bark(self, title: str, text: str) -> None:
        """Bark 推送：GET {server}/{key}/{title}/{body}（路径段必须 URL 编码）。

        Bark 是 iOS 上最省事的推送方式：装 App 拿一串 key 就能收，不用注册服务、
        不用配 SMTP。路径里的中文必须 quote，否则服务端会 400。
        """
        from urllib.parse import quote

        server = (self.cfg.bark_server or "https://api.day.app").rstrip("/")
        url = f"{server}/{quote(self.cfg.bark_key, safe='')}/{quote(title, safe='')}/{quote(text, safe='')}"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=8) as resp:  # noqa: S310
                resp.read(2048)
            self._sent_ok += 1
        except (urllib.error.URLError, OSError, ValueError) as e:
            self._sent_fail += 1
            log.warning("Bark 推送失败: %s", e)

    def _send_mail(self, title: str, text: str) -> None:
        pwd = os.environ.get(self.cfg.smtp_password_env, "")
        if not pwd:
            self._sent_fail += 1
            log.warning("SMTP 密码未设置（环境变量 %s），跳过邮件告警",
                        self.cfg.smtp_password_env)
            return
        msg = MIMEText(text, "plain", "utf-8")
        msg["Subject"] = Header(title, "utf-8")
        msg["From"] = self.cfg.smtp_user
        msg["To"] = ", ".join(self.cfg.smtp_to)
        try:
            if self.cfg.smtp_ssl:
                server = smtplib.SMTP_SSL(self.cfg.smtp_host, self.cfg.smtp_port, timeout=10)
            else:
                server = smtplib.SMTP(self.cfg.smtp_host, self.cfg.smtp_port, timeout=10)
                server.starttls()
            with server:
                if self.cfg.smtp_user:
                    server.login(self.cfg.smtp_user, pwd)
                server.sendmail(self.cfg.smtp_user, self.cfg.smtp_to, msg.as_string())
            self._sent_ok += 1
        except (smtplib.SMTPException, OSError) as e:
            self._sent_fail += 1
            log.warning("邮件告警发送失败: %s", e)

    # ---------------- 观测 ----------------

    def snapshot(self) -> dict:
        with self._lock:
            hist = list(self.history[-20:])
        return {
            "enabled": self.cfg.enabled,
            "webhook": bool(self.cfg.webhook_url),
            "bark": bool(self.cfg.bark_key),
            "format": self.cfg.webhook_format,
            "mail": bool(self.cfg.smtp_host and self.cfg.smtp_to),
            "sent_ok": self._sent_ok,
            "sent_fail": self._sent_fail,
            "rules": [{"name": r.name, "metric": r.metric, "threshold": r.threshold,
                       "enabled": r.enabled} for r in self.cfg.rules],
            "fired": dict(self._fired),
            "history": hist,
        }
