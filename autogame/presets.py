"""拟人化预设 —— 一键切换"要多像人"。

三个预设对应三种真实取舍，别指望有"又快又绝对安全"的选项：

  stable  稳定快速：拟人化全关。轨迹直连、无抖动、无疲劳、无走神、无偏好随机。
          → 行为完全确定、可复现、最好调；代价是统计特征非常"机器"。
  human   绝对像人：全开，并且把停顿/错误率往人的方向推（更长反应时间、
          更长走位预算、更大的轨迹扰动、更高的"懒得捡"概率）。
          → 更像人；代价是吞吐下降、行为更难复现、排障更难。
  custom  自定义：不动任何参数，由调用方自己逐项调。

预设只改**与拟人度相关的字段**，不碰视觉阈值/执行器后端/服务端这些"部署参数" ——
否则用户点一下"绝对像人"把 HSV 阈值改掉，画面直接检不出东西，属于最糟的体验。
"""

from __future__ import annotations

from typing import Any

from .config import AppConfig

#: 每个预设 = 一组 (点分路径, 值)。用点分路径而不是嵌套 dict，
#: 是为了让"哪些字段被预设改了"在日志/页面上一目了然。
PRESETS: dict[str, dict[str, Any]] = {
    "stable": {
        "decision.humanize.enabled": False,
        "decision.humanize.fatigue.enabled": False,
        "decision.humanize.drift.enabled": False,
        "decision.humanize.mood.enabled": False,
        "decision.humanize.social.enabled": False,
        "decision.humanize.reaction_ms": [60, 90],
        "decision.walk.enabled": False,
        "executor.mouse_path.enabled": False,
        "executor.mouse_path.overshoot_prob": 0.0,
        "executor.mouse_path.micro_pause_prob": 0.0,
        "executor.mouse_path.endpoint_jitter_px": 0,
        "executor.mouse_path.hover_ms": [0, 0],
        "executor.mouse_path.press_ms": [45, 60],
        "decision.jitter_ratio": 0.0,
        "decision.action_desync_ms": 0,
        "executor.rate_floor_jitter": 0.0,
        "decision.humanize.rhythm.enabled": False,
        "executor.mouse_path.model": "windmouse",
        "executor.mash.enabled": False,
    },
    "human": {
        "decision.humanize.enabled": True,
        "decision.humanize.fatigue.enabled": True,
        "decision.humanize.fatigue.per_action": 0.004,
        "decision.humanize.fatigue.break_threshold": 0.8,
        "decision.humanize.fatigue.break_seconds": [30.0, 150.0],
        "decision.humanize.fatigue.latency_boost": 0.5,
        "decision.humanize.fatigue.jitter_boost": 1.0,
        "decision.humanize.drift.enabled": True,
        "decision.humanize.drift.mean_interval_s": 30.0,
        "decision.humanize.drift.startup_grace_s": 45.0,
        "decision.humanize.drift.hesitate_prob": 0.5,
        "decision.humanize.drift.hesitate_s": [0.4, 2.2],
        "decision.humanize.mood.enabled": True,
        "decision.humanize.mood.temperature": 0.5,
        "decision.humanize.mood.skip_prob": 0.05,
        "decision.humanize.reaction_ms": [220, 900],
        "decision.humanize.reaction_sigma": 0.45,
        "decision.walk.enabled": True,
        "decision.walk.budget_s": 4.0,
        "decision.walk.settle_grace_s": 0.8,
        "executor.mouse_path.enabled": True,
        "executor.mouse_path.model": "kinematic",
        "executor.mouse_path.bow": 0.11,
        "executor.mouse_path.wobble_px": 2.6,
        "executor.mouse_path.noise_px": 0.8,
        "executor.mouse_path.speed_noise": 0.16,
        "executor.mouse_path.max_points": 72,
        "executor.mouse_path.overshoot_prob": 0.2,
        "executor.mouse_path.micro_pause_prob": 0.35,
        "executor.mouse_path.endpoint_jitter_px": 3,
        "executor.mouse_path.hover_ms": [50, 180],
        "executor.mouse_path.press_ms": [70, 170],
        "decision.jitter_ratio": 0.22,
        "decision.action_desync_ms": 550,
        "executor.rate_floor_jitter": 0.2,
        "decision.humanize.rhythm.enabled": True,
        "decision.humanize.rhythm.cv": 0.3,
        "decision.humanize.rhythm.tail_prob": 0.03,
        "executor.mash.prob_key": 0.5,
        "executor.mash.prob_click": 0.22,
        "executor.mash.budget_ratio": 0.34,
    },
    "custom": {},
}

#: 预设的人类可读说明（页面直接渲染）
DESCRIPTIONS: dict[str, dict[str, str]] = {
    "stable": {
        "title": "全都不要（稳定快速）",
        "summary": "随机全部关掉：轨迹直连、无抖动、无疲劳、无走神、无偏好随机。行为完全确定、可复现。",
        "cost": "统计特征非常“机器”，长期数据里很好认。",
    },
    "human": {
        "title": "放飞自我（绝对像人）",
        "summary": "全开并加强：反应 220-900ms、走位预算 4s、风力 4.5、落点偏差 3px、"
                   "懒得捡 5%、会累（30-150s 休息）、会走神、进关卡前会犹豫。",
        "cost": "吞吐降 20-40%，行为难复现，排障更难。",
    },
    "custom": {
        "title": "自定义",
        "summary": "不动任何参数，由你逐项调。页面上的滑块就是当前生效值。",
        "cost": "参数组合没人替你验证过，出问题自己扛。",
    },
}


def _set_path(cfg: AppConfig, path: str, value: Any) -> None:
    """按点分路径写值（只写已存在的字段，避免拼错路径静默无效）。"""
    obj: Any = cfg
    parts = path.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    if not hasattr(obj, parts[-1]):
        raise AttributeError(f"预设路径不存在: {path}")
    setattr(obj, parts[-1], value)


def preset_patch(name: str) -> dict[str, Any]:
    """预设 → 嵌套 patch（供 /api/config 复用同一套校验/应用逻辑）。"""
    if name not in PRESETS:
        raise KeyError(f"未知预设: {name}")
    patch: dict[str, Any] = {}
    for path, value in PRESETS[name].items():
        cur = patch
        keys = path.split(".")
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        cur[keys[-1]] = value
    return patch


def apply_preset(cfg: AppConfig, name: str) -> list[str]:
    """把预设应用到**运行中的配置对象**（原地改，保持引用不变）。

    返回被改动的点分路径列表，便于页面/日志展示"这次到底动了什么"。
    """
    if name not in PRESETS:
        raise KeyError(f"未知预设: {name}")
    changed: list[str] = []
    for path, value in PRESETS[name].items():
        _set_path(cfg, path, value)
        changed.append(path)
    return changed
