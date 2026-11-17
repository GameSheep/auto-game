"""相机标定反馈 —— 告诉用户"你放得准不准、为什么检不出来"。

摄像头方向/距离是物理的，程序没法替你调。能做的是把**失败原因**量化出来：
"画面太暗""对焦糊了""A框模板分数只有 0.4（阈值 0.72）""没看到游戏 HUD"。
否则用户只知道"捡不到东西"，只能瞎试。

所有分数都是 0-100 的启发式指标，用于**相对比较**（调之前/调之后），
不是绝对精度 —— 页面上的文字建议比分数更重要。
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from ..config import VisionConfig
from ..protocol import SceneState

# 经验阈值（720p 游戏画面实测）
BRIGHT_DARK = 55.0       # 低于此值：HSV 阈值开始失效
BRIGHT_OVER = 190.0      # 高于此值：高光压成一片白
SHARP_BLUR = 60.0        # Laplacian 方差低于此值：明显糊
SHARP_GOOD = 150.0       # 达到此值算清晰


def _score_range(value: float, lo: float, hi: float) -> float:
    """value 落在 [lo, hi] 内满分，越界线性衰减。"""
    if lo <= value <= hi:
        return 100.0
    if value < lo:
        return max(0.0, value / max(1e-6, lo) * 100.0)
    return max(0.0, (255.0 - value) / max(1e-6, 255.0 - hi) * 100.0)


def _score_ge(value: float, good: float) -> float:
    return max(0.0, min(100.0, value / max(1e-6, good) * 100.0))


def analyze(
    frame: np.ndarray,
    scene: SceneState | None,
    diag: dict | None,
    cfg: VisionConfig,
    frame_age_s: float | None = None,
) -> dict:
    """一帧的标定诊断。返回 {scores, metrics, advice, ...}。"""
    diag = diag or {}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    aframe_best = float(diag.get("aframe_best", 0.0))
    aframe_thr = float(diag.get("aframe_threshold", cfg.aframe.template_threshold))
    gate_best = float(diag.get("gate_best", 0.0))
    gate_thr = float(diag.get("gate_threshold", cfg.scene_gate.threshold))

    in_game = bool(scene.in_game) if scene is not None else False
    loading = bool(scene.loading) if scene is not None else False
    pillars = list(scene.pillars) if scene is not None else []
    aframe = scene.aframe if scene is not None else None

    s_bright = _score_range(brightness, BRIGHT_DARK, BRIGHT_OVER)
    s_sharp = _score_ge(sharpness, SHARP_GOOD)
    s_gate = 100.0 if in_game else _score_ge(gate_best, max(1e-6, gate_thr))
    s_aframe = max(0.0, min(100.0, aframe_best / max(1e-6, aframe_thr) * 100.0))
    s_pillar = 100.0 if pillars else 0.0

    overall = (s_gate * 0.35 + s_aframe * 0.25 + s_sharp * 0.15
               + s_bright * 0.15 + s_pillar * 0.10)

    advice: list[str] = []
    if brightness < BRIGHT_DARK:
        advice.append(
            f"画面偏暗（亮度 {brightness:.0f}）：光柱的 HSV 阈值会开始漏检。"
            "调高显示器亮度/关闭夜间模式，或提高采集端曝光。")
    if brightness > BRIGHT_OVER:
        advice.append(
            f"画面过曝（亮度 {brightness:.0f}）：高光会糊成一片白，光柱和 A 框都会变形。"
            "降低显示器亮度或对比度。")
    if sharpness < SHARP_BLUR:
        advice.append(
            f"画面模糊（清晰度 {sharpness:.0f} < {SHARP_BLUR:.0f}）：对焦/距离不对，"
            "模板匹配分数会掉到阈值以下。先解决对焦，再谈标定。")
    if aframe_best < aframe_thr:
        advice.append(
            f"A框模板匹配最高只有 {aframe_best:.2f}（阈值 {aframe_thr:.2f}）："
            "可能是分辨率/窗口尺寸变了，需要从当前画面重新裁 A 框模板"
            "（tools/auto_template.py）。")
    if not in_game:
        reason = str(diag.get("gate_reason", "") or "")
        if reason.startswith("negative:"):
            advice.append(
                f"命中反例模板（{reason.split(':', 1)[1]}）：当前画面被我标成整屏 UI"
                "（关卡选择/背包/结算），这是预期拦截，不是故障。")
        elif reason == "unmatched":
            advice.append(
                f"服务端没有认出这块 HUD（场景门最高 {gate_best:.2f} / 阈值 {gate_thr:.2f}，"
                "反例最高 "
                f"{float(diag.get('gate_neg_best', 0.0)):.2f}）：这是**未标定**，"
                "不是'相机没对准'。用 tools/make_template.py 从当前画面裁一块技能栏，"
                "存成 templates/skillbar_<游戏名>.png 再试。")
        else:
            advice.append(
                f"未识别到游戏 HUD（场景门最高 {gate_best:.2f} / 阈值 {gate_thr:.2f}）："
                "相机可能没对准游戏画面，或者技能栏模板与实际画质不匹配。")
    if not pillars and in_game:
        advice.append(
            "画面里没检出光柱：如果游戏里明明有光柱，说明 HSV 范围需要重新标定"
            "（页面下方『从当前画面采样 HSV』）。")
    if loading:
        advice.append("当前是加载/黑屏过场，属正常等待，不是标定问题。")
    if not advice:
        advice.append("各项指标正常。建议在真实战斗画面下再看一眼，确认光柱与 A 框都能检出。")

    return {
        "ts": time.time(),
        "frame_age_s": None if frame_age_s is None else round(frame_age_s, 2),
        "metrics": {
            "brightness": round(brightness, 1),
            "contrast": round(contrast, 1),
            "sharpness": round(sharpness, 1),
            "aframe_best": round(aframe_best, 3),
            "aframe_threshold": round(aframe_thr, 3),
            "gate_best": round(gate_best, 3),
            "gate_threshold": round(gate_thr, 3),
            "gate_reason": str(diag.get("gate_reason", "") or ""),
            "gate_neg_best": round(float(diag.get("gate_neg_best", 0.0)), 3),
            "gate_neg_name": str(diag.get("gate_neg_name", "") or ""),
            "in_game": in_game,
            "loading": loading,
            "pillars": len(pillars),
            "pillar_colors": sorted({p.color for p in pillars}),
            "aframe_found": aframe is not None,
            "frame_w": int(frame.shape[1]),
            "frame_h": int(frame.shape[0]),
        },
        "scores": {
            "overall": round(overall, 0),
            "gate": round(s_gate, 0),
            "aframe": round(s_aframe, 0),
            "sharpness": round(s_sharp, 0),
            "brightness": round(s_bright, 0),
            "pillar": round(s_pillar, 0),
        },
        "advice": advice,
    }


def verdict_label(overall: float) -> str:
    """总分 → 一句话结论（页面顶部大字）。"""
    if overall >= 80:
        return "可以开跑"
    if overall >= 60:
        return "能用，但建议先修掉提示里的问题"
    if overall >= 35:
        return "勉强，误检/漏检概率高"
    return "先别跑 —— 现在大概率一条都捡不到"
