"""模型调用客户端 —— ollama 与 OpenAI 兼容接口的唯一出口。

抽出来的原因：supervisor（对账/刹车）和 arbiter（三级裁决）都要调模型，
如果各写一份 HTTP，迟早会出现"改了一边忘了另一边"（比如超时、温度、JSON 模式）。
这里只做一件事：把 (system, user) 变成一段文本。解析、校验、降级都在调用方。
"""

from __future__ import annotations

import json
import os
from typing import Any


def chat(cfg: Any, system: str, user: str, *, temperature: float = 0.0,
         json_mode: bool = True) -> str:
    """发一次对话请求，返回模型输出的文本。

    cfg 需要具备：provider / base_url / model / timeout_s，
    以及可选的 api_key / api_key_env。用 duck typing 是为了让不同的配置类
    （SupervisorConfig / ArbiterModelConfig）都能直接传进来。
    """
    import httpx

    provider = str(getattr(cfg, "provider", "ollama"))
    base = str(getattr(cfg, "base_url", "")).rstrip("/")
    model = str(getattr(cfg, "model", ""))
    timeout = float(getattr(cfg, "timeout_s", 12.0))
    if not base or not model:
        raise ValueError("模型配置不完整（base_url / model 为空）")

    if provider == "ollama":
        url = f"{base}/api/chat"
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"
        headers: dict[str, str] = {}
    else:
        url = f"{base}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        key = getattr(cfg, "api_key", None) or os.environ.get(
            str(getattr(cfg, "api_key_env", "")), "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}

    resp = httpx.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if provider == "ollama":
        return data["message"]["content"]
    return data["choices"][0]["message"]["content"]


def extract_json(text: str) -> dict[str, Any]:
    """从模型输出里抠出第一个 JSON 对象。

    不开 json_mode 的模型经常会写"好的，分析如下：{...} 希望有帮助"，
    所以不能直接 json.loads。但也**不能**宽松到"猜它想说什么"——
    找不到 JSON 就抛错，让调用方降级，而不是用正则拼一个猜测出来。
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("模型返回空")
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"模型返回里没有 JSON 对象: {text[:120]}")
    obj = json.loads(text[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("模型返回的 JSON 不是对象")
    return obj
