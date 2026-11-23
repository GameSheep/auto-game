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
from ..config import AppConfig, apply_input_scheme, runtime_overlay_path
from ..input import InputSchemeError

log = logging.getLogger(__name__)

#: 改了立刻生效的字段前缀（其余字段页面会提示"需重启"）。
LIVE_FIELDS: tuple[str, ...] = (
    "decision.",
    # 操作方案：改键位本身不生效，但 apply_patch 会就地重新派生到 decision.* /
    # executor.mash.keys，而 FSM 读的正是那些字段 → 实际是即时生效的。
    # 不列在这里的话，页面会提示"需重启"，而它其实已经生效了（比不生效更误导）。
    "input.",
    "executor.mouse_path.",
    "executor.min_action_interval_ms",
    "executor.max_actions_per_sec",
    "executor.rate_floor_jitter",
    "executor.mash.",
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


#: 有操作方案时，这些字段是**派生值**：改它们会被下次加载覆盖回方案里的值。
#: 典型症状是"页面上改了拾取键，跑起来了，重启后又变回去"——比报错难查得多，
#: 所以这里直接拒，并告诉人该改哪儿。
_DERIVED_BY_SCHEME: frozenset[str] = frozenset({
    "decision.pickup_key",
    "decision.interact_key",
    "decision.approach_mode",
    "decision.move_button",
    "executor.mash.keys",
})


def _flat_paths(d: dict, prefix: str = ""):
    for k, v in (d or {}).items():
        p = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            yield from _flat_paths(v, f"{p}.")
        else:
            yield p


def apply_patch(cfg: AppConfig, patch: dict) -> dict:
    """校验并原地应用 patch。返回 {"changed": [...], "restart_required": [...]}。

    校验失败抛 ConfigError（带字段路径），绝不部分应用 —— 半套配置比不配置更危险。
    """
    if not isinstance(patch, dict) or not patch:
        raise ConfigError("patch 必须是非空对象")
    if cfg.input is not None:
        locked = sorted(p for p in _flat_paths(patch) if p in _DERIVED_BY_SCHEME)
        if locked:
            raise ConfigError(
                "这些字段由操作方案决定，直接改会在下次加载时被覆盖回去："
                f"{', '.join(locked)}。请改 input.bindings / input.move / input.primary "
                "（对应页面「操作方案」，或 profiles/<id>/scheme.yaml）")
    merged = _deep_merge(cfg.model_dump(), patch)
    try:
        new_cfg = AppConfig.model_validate(merged)
    except ValidationError as e:
        errs = [
            {"field": ".".join(str(x) for x in err["loc"]), "msg": err["msg"]}
            for err in e.errors()
        ]
        raise ConfigError(f"配置校验失败: {errs}") from e

    # 改了键位就必须**立刻重新派生**到 decision.* / mash.keys，否则是假生效：
    # 页面显示"拾取键已改成 F"，FSM 还在按旧的 a —— 日志里一切正常，
    # 游戏里一直捡空。派生在临时对象上做，冲突就整笔拒绝，不做半套。
    if "input" in (patch or {}):
        try:
            apply_input_scheme(new_cfg)
        except InputSchemeError as e:
            raise ConfigError(str(e)) from e

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


def agent_report(cfg: AppConfig, profile_root: Any = None) -> dict | None:
    """游戏者知识的状态（页面"这游戏现在能玩到什么程度"）。

    返回 `None` = 这个游戏还没教（没有 goals/skills）—— 页面据此隐藏卡片，
    而不是显示一张空表让人以为"能玩但没内容"。
    """
    try:
        from ..agent import AgentKnowledge, resolve_knowledge_dir

        kn = AgentKnowledge.load(resolve_knowledge_dir(cfg.agent, profile_root))
    except Exception as e:  # noqa: BLE001
        return {"source": "", "error": f"{type(e).__name__}: {e}"}
    if not kn:
        return None
    scheme = cfg.input
    if scheme is None:
        return {"source": kn.source_dir, "error": "没有操作方案（scheme.yaml），"
                                                "动作不知道按哪个键"}
    rep = kn.report(scheme)
    rep["enabled"] = bool(getattr(cfg.agent, "enabled", False))
    return rep


def input_report(cfg: AppConfig) -> dict | None:
    """操作方案的可视化数据（页面"这个游戏能跑到什么程度"就靠它）。

    为什么不把 `input` 原样丢给页面：原始结构是"键盘怎么配的"，页面要回答的是
    "哪些做得了、哪些做不了、做不了会怎样"。后者的答案在 `capability_table()` 里
    （它包含引擎的支持度判断），不能靠前端拼。
    """
    scheme = getattr(cfg, "input", None)
    if scheme is None:
        return None
    return {
        "describe": scheme.describe(),
        "source": scheme.source,
        "move_kind": scheme.move.kind,
        "capabilities": scheme.capability_table(),
        "conflicts": [{"severity": c.severity, "message": c.message, "fix": c.fix}
                      for c in scheme.conflicts()],
    }


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
