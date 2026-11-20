"""运行时配置 —— 校验 → 原地生效 → 持久化到覆盖文件。

三条设计约束（都是踩过坑才这么定的）：

1. **必须原地改，不能替换对象。**
   状态机持有的是 `cfg.decision` 这个对象引用，执行器持有 `cfg.executor`。
   如果把 `cfg.decision` 换成新对象，FSM 还在读旧的 —— 页面显示"已生效"，
   实际程序纹丝不动。这是最难发现的假生效。

2. **必须走 pydantic 校验。**
   直接 setattr 一个 `temperature: -5` 进去，报错会在几百帧之后以
   "除零/负指数"的形式出现，离现场十万八千里。

3. **只写覆盖文件，不改写 config.yaml。**
   那个文件是手写带注释的，程序回写会碾平注释和排版。

改完之后**哪些字段真的会影响运行**也标出来了（`LIVE_FIELDS`），
不生效的字段（比如采集后端、执行器后端）页面会明确提示"需重启"。
"""

from __future__ import annotations

import logging
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from ..atomic import atomic_write_text
from ..config import AppConfig, runtime_overlay_path

log = logging.getLogger(__name__)

#: 改了立刻生效的字段前缀（其余字段页面会提示"需重启"）。
LIVE_FIELDS: tuple[str, ...] = (
    "decision.",
    "executor.mouse_path.",
    "executor.min_action_interval_ms",
    "executor.max_actions_per_sec",
    "vision.whitelist",
    "vision.whitelist_colors",
    "vision.hsv",
    "vision.pillar_min_area",
    "vision.aframe.",
    "vision.scene_gate.threshold",
    "capture.fps",
    "capture.jpeg_quality",
    "watchdog.",
    "notify.",
    "server.dash_fps",
)

#: 永远不回显/不回写的敏感字段
SECRET_FIELDS: tuple[str, ...] = ("server.token",)


class ConfigError(ValueError):
    """配置校验失败（带字段级错误信息，页面直接展示）。"""


def _deep_merge(base: dict, extra: dict) -> dict:
    out = dict(base or {})
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _copy_into(dst: BaseModel, src: BaseModel, prefix: str = "") -> list[str]:
    """把 src 的值原地写回 dst，返回**实际变化**的点分路径列表。"""
    changed: list[str] = []
    for name in type(src).model_fields:
        a = getattr(dst, name)
        b = getattr(src, name)
        path = f"{prefix}{name}"
        if isinstance(a, BaseModel) and isinstance(b, BaseModel):
            changed.extend(_copy_into(a, b, f"{path}."))
        elif a != b:
            setattr(dst, name, b)
            changed.append(path)
    return changed


def apply_patch(cfg: AppConfig, patch: dict) -> dict:
    """校验并原地应用 patch。返回 {"changed": [...], "restart_required": [...]}。

    校验失败抛 ConfigError（带字段路径），绝不部分应用 —— 半套配置比不配置更危险。
    """
    if not isinstance(patch, dict) or not patch:
        raise ConfigError("patch 必须是非空对象")
    merged = _deep_merge(cfg.model_dump(), patch)
    try:
        new_cfg = AppConfig.model_validate(merged)
    except ValidationError as e:
        errs = [
            {"field": ".".join(str(x) for x in err["loc"]), "msg": err["msg"]}
            for err in e.errors()
        ]
        raise ConfigError(f"配置校验失败: {errs}") from e

    changed = _copy_into(cfg, new_cfg)
    restart = [p for p in changed
               if not any(p == f.rstrip(".") or p.startswith(f) for f in LIVE_FIELDS)]
    return {"changed": changed, "restart_required": restart}


def public_config(cfg: AppConfig) -> dict:
    """页面可读的当前配置（敏感字段置空）。"""
    d = cfg.model_dump()
    for path in SECRET_FIELDS:
        cur = d
        parts = path.split(".")
        for p in parts[:-1]:
            cur = cur.get(p, {})
        if parts[-1] in cur:
            cur[parts[-1]] = "***" if cur[parts[-1]] else ""
    return d


def save_overlay(config_path: str, patch: dict) -> str:
    """把 patch 累加进运行时覆盖文件，返回文件路径。

    只存"页面改过的项"，不存全量配置 —— 这样覆盖文件很短、可读、可手工删掉回滚。
    """
    path = runtime_overlay_path(config_path)
    existing: dict = {}
    if path.exists():
        try:
            existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            log.warning("覆盖文件损坏，将被重写: %s", path)
            existing = {}
    merged = _deep_merge(existing, patch)
    # 敏感字段绝不落盘
    for secret in SECRET_FIELDS:
        parts = secret.split(".")
        cur = merged
        for p in parts[:-1]:
            if not isinstance(cur.get(p), dict):
                cur = None
                break
            cur = cur[p]
        if isinstance(cur, dict):
            cur.pop(parts[-1], None)
    # 原子写：半份 YAML 会让服务下次起不来，而页面上没有任何地方能看出"配置文件坏了"
    atomic_write_text(
        path,
        "# 页面改过的配置（自动生成，可删除以回滚到 config.yaml）\n"
        + yaml.safe_dump(merged, allow_unicode=True, sort_keys=False),
    )
    return str(path)


def clear_overlay(config_path: str) -> bool:
    path = runtime_overlay_path(config_path)
    if path.exists():
        path.unlink()
        return True
    return False
