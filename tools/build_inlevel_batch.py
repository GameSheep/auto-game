"""从实机视频重建 100 张"关卡内"批量图集。

筛选链（对话定稿判据 + 用户修正）：
  1) 视频级预筛：背景运动中位数 ≥150（关卡内世界滚动；城镇打桩 ≈52-84 → 排除）
     —— 用户明确指出 Tartarus 系列是城镇打桩，不是关卡
  2) 1fps 采样 + 静止排除（连续帧哈希几乎不变 → 菜单/过场）
  3) 帧级四要素：小地图(≥3) + 怪物(≥1) + 伤害数字(≥2) + 红血球HUD(≥1500px)
  4) 场景门（技能栏/血球可见）
  5) 感知哈希去重 → 前 100 张写入 assets/batch/（img_*.png）
"""

import cv2
import glob
import numpy as np
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autogame.config import VisionConfig
from autogame.vision.detector import Detector, ROOT_TEMPLATES
from autogame.vision.scene_classifier import in_level_features, is_in_level


def ahash(img):
    g = cv2.cvtColor(cv2.resize(img, (16, 16)), cv2.COLOR_BGR2GRAY)
    return g > g.mean()


def hamming(a, b):
    return int((a != b).sum())


def bg_hash(frame):
    """暗部背景哈希：抹掉高亮 FX/文字，只留场景结构。"""
    small = cv2.resize(frame, (32, 18))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    g[hsv[:, :, 2] > 140] = 0
    return g > np.mean(g) if g.mean() > 0 else g > 0


def video_bg_motion(video: str) -> float:
    """视频背景运动中位数（1s 采样）。"""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(fps))
    hashes = []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            hashes.append(bg_hash(f))
        i += 1
    cap.release()
    if len(hashes) < 2:
        return 0.0
    return float(statistics.median(hamming(hashes[k], hashes[k + 1])
                                   for k in range(len(hashes) - 1)))


def main():
    cap_frames = 100
    motion_min = 180  # tartarus(153)=城镇打桩擦线，user_farm(221)/legend_c(195)/bb114(196) 保持
    out_dir = ROOT / "assets" / "batch"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("img_*.png"):
        f.unlink()

    videos = [v for v in sorted(glob.glob(str(ROOT / "assets" / "videos" / "*.mp4")))
              if "judged" not in Path(v).stem]
    print("== 视频级预筛（背景运动中位数 ≥%d = 关卡内）==" % motion_min)
    inlevel_videos = []
    for v in videos:
        m = video_bg_motion(v)
        verdict = "关卡内 ✓" if m >= motion_min else "城镇/站立 ✗"
        print(f"  {Path(v).stem:16s} 中位数={m:.0f} → {verdict}")
        if m >= motion_min:
            inlevel_videos.append(v)
    if not inlevel_videos:
        raise SystemExit("没有关卡内视频")

    gate_det = [None]

    def gate_check(frame):
        if gate_det[0] is None:
            gate_det[0] = Detector(VisionConfig(),
                                   ROOT_TEMPLATES if ROOT_TEMPLATES.exists() else None)
        return gate_det[0].process(frame).in_game

    caps = {v: cv2.VideoCapture(v) for v in inlevel_videos}
    kept_hashes: list = []
    prev_hash = {}
    stats = {"sampled": 0, "static_skip": 0, "not_inlevel": 0,
             "gate_block": 0, "dup_skip": 0, "kept": 0}
    kept = 0
    active = dict.fromkeys(inlevel_videos, True)
    idx = {v: 0 for v in inlevel_videos}
    fps = {v: (caps[v].get(cv2.CAP_PROP_FPS) or 30) for v in inlevel_videos}

    print("== 帧级筛选 ==")
    while kept < cap_frames and any(active.values()):
        for v in inlevel_videos:
            if not active[v] or kept >= cap_frames:
                continue
            step = max(1, int(fps[v]))  # 1fps
            target = idx[v] + step
            caps[v].set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = caps[v].read()
            idx[v] = target
            if not ok:
                active[v] = False
                continue
            stats["sampled"] += 1
            h = ahash(frame)
            ph = prev_hash.get(v)
            if ph is not None and hamming(h, ph) <= 2:
                stats["static_skip"] += 1
                continue
            prev_hash[v] = h
            feat = in_level_features(frame)
            ok_in = (feat["minimap_density"] >= 3.0 and feat["damage_numbers"] >= 2
                     and feat["monsters"] >= 1 and feat["hp_orb_area"] >= 1500)
            if not ok_in:
                stats["not_inlevel"] += 1
                continue
            if not gate_check(frame):
                stats["gate_block"] += 1
                continue
            if any(hamming(h, kh) <= 4 for kh in kept_hashes):
                stats["dup_skip"] += 1
                continue
            kept_hashes.append(h)
            out = out_dir / f"img_{kept:03d}_{Path(v).stem}_f{target}.png"
            cv2.imwrite(str(out), frame)
            kept_hashes.append(ahash(frame))
            kept += 1
            stats["kept"] += 1

    for c in caps.values():
        c.release()
    print("统计:", stats)
    print(f"图集: {kept} 张关卡内帧 → {out_dir}")


if __name__ == "__main__":
    main()
