"""真实录像审查链路测试 —— manifest → 离线审查页（每帧完整指令必须落在页面上）。

这套工具是"用真实录像验证程序"的交付物：manifest 里少了指令、审查页里漏了
命令串，验证就等于没做。所以这里用一份合成 manifest 把渲染链路钉死。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import review_offline  # noqa: E402


def _item(**over) -> dict:
    base = {
        "image": "t00000.jpg", "t": "00:00.00", "ms": 0,
        "frame_size": [1280, 720], "image_size": [960, 540],
        "gate": "战斗", "in_game": True, "loading": False, "frame_delta": 12.5,
        "state": "SCANNING", "intent": "向上方向点地面探索（寻找怪物/掉落）",
        "pillars": [], "aframe": None, "rechallenge": None, "monsters": [],
        "ground_labels": [], "item_list": [],
        "commands": [], "violations": [],
    }
    base.update(over)
    return base


def test_report_renders_full_command_per_frame(tmp_path):
    d = tmp_path / "demo"
    (d / "frames").mkdir(parents=True)
    cv2.imwrite(str(d / "frames" / "t00000.jpg"),
                np.zeros((54, 96, 3), dtype=np.uint8))
    cmd = {"op": "mouse_click", "key": None, "button": "left", "x": 640, "y": 68,
           "hold_ms": 60, "text": "MOUSE_CLICK left (640,68)"}
    items = [
        _item(commands=[cmd],
              pillars=[{"x": 700, "y": 400, "cx": 700, "cy": 330, "w": 14, "h": 110,
                        "color": "gold", "distance": 66.0, "stable": True, "seen": 3}]),
        _item(image="t00001.jpg", t="00:00.50", ms=500, intent=""),
    ]
    (d / "manifest.json").write_text(json.dumps({
        "summary": {"name": "demo", "video": "demo.mp4", "src": {"size": [1280, 720],
                                                                 "duration_s": 1.0},
                    "hz": 2.0, "ticks": 2, "gate_pass": 2, "loading_ticks": 0,
                    "ticks_with_commands": 1, "commands": 1, "per_op": {"mouse_click": 1},
                    "a_press": 0, "ticks_with_pillars": 1, "ticks_with_aframe": 0,
                    "ticks_with_monsters": 0, "violations": []},
        "items": items}, ensure_ascii=False), encoding="utf-8")

    out = review_offline.build(tmp_path)
    html = out.read_text(encoding="utf-8")
    assert "MOUSE_CLICK left (640,68)" in html, "每帧的完整指令必须出现在审查页上"
    assert 'x=640' in html and 'y=68' in html, "参数要单独列出（便于核对坐标）"
    assert "本帧无命令" in html, "无指令的帧也要说明为什么没动作"
    assert html.count('class="card"') == 2, "manifest 里几帧就渲染几张卡片"
    # 叠加层坐标按 frame_size 折算：光柱左边 = (700-14/2)/1280 = 54.14%
    assert "54.14%" in html and 'class="p st"' in html


def test_report_empty_root_is_loud(tmp_path):
    try:
        review_offline.build(tmp_path)
    except SystemExit as e:
        assert "manifest.json" in str(e)
    else:  # pragma: no cover
        raise AssertionError("没有 manifest 时必须直接报错，而不是生成一个空页面")
