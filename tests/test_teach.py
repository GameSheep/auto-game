"""教学编译器测试 —— 钉的是"人写的一句话怎么变成可执行/可核对的东西"。

重点不在"能解析"，而在四条边界：

1. 只写类型没写标题、只写标题没写类型，都得认得出来；真认不出要报警，不能静默丢
2. 核对通过才 strict；不通过照样入库，但 strict=false + 原因写在备注里
3. 已有条目绝不覆盖 —— 人的话比生成的更值钱
4. 已有受管块就替换，没有才追加 —— 重跑十次也只该有一个块
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from autogame.config import AppConfig
from autogame.profile import Profile
from autogame.teach import (
    KIND_KEY,
    KIND_OBJECT,
    KIND_SCENE,
    build_glossary,
    check_claims,
    compile_lessons,
    load_lessons,
    measure_region,
    parse_lessons,
    parse_rect,
    write_game_md,
    write_labels,
)

LESSON = """# 测试游戏

## 术语

| 我说的 | 标准说法 | 备注 |
|---|---|---|
| 搬砖 | 重复刷图 | 主要玩法 |
| 星图 | 关卡选择界面 | |

## 界面 · 关卡选择页

- 图: img/map.png
- 判定: 反例
- 一眼认出: 左上角标题块
- 框: 0,0,300,80

## 界面 · 战斗

- 图: img/combat.png
- 判定: 正例

## 物体 · 掉落光柱

- 图: img/pillar.png
- 框: 50,50,6,60
- 颜色: 金色

## 流程 · 一局

1. 点节点进图
2. 清怪
3. 点再次挑战

## 键位

- 拾取: A
- 返回: ESC
- 走位: 点地面

## 禁忌

- 未标定画面不要按 ESC
- 背包不要按确认键

## 图 A 某个没写类型的段

- 图: img/map.png
- 判定: 反例
"""


@pytest.fixture()
def prof(tmp_path: Path) -> Profile:
    p = Profile(id="test", root=tmp_path, name="测试", game="测试游戏")
    (tmp_path / "lessons" / "img").mkdir(parents=True)
    (tmp_path / "lessons" / "t.md").write_text(LESSON, encoding="utf-8")
    (tmp_path / "refs").mkdir()
    _make_images(tmp_path / "lessons" / "img")
    return p


def _gold_bar(w: int = 200, h: int = 200) -> np.ndarray:
    """画一束金色细高条：HSV 约 H25 S255 V255 → BGR (0,214,255)。"""
    img = np.full((h, w, 3), (30, 26, 24), dtype=np.uint8)
    img[50:110, 50:56] = (0, 214, 255)
    return img


def _make_images(d: Path) -> None:
    cv2.imwrite(str(d / "pillar.png"), _gold_bar())
    cv2.imwrite(str(d / "map.png"), np.full((200, 200, 3), 40, dtype=np.uint8))
    cv2.imwrite(str(d / "combat.png"), np.full((200, 200, 3), 70, dtype=np.uint8))


def _probe_stub(in_game_by_name: dict[str, bool]):
    """按图名返回门禁结果的假探针——不依赖模板文件。"""
    def _run(img: np.ndarray) -> dict:
        del img
        return {"in_game": False, "gate_reason": "unmatched", "gate_best": 0.0,
                "gate_neg": 0.0, "pillars": 0, "pillar_colors": [],
                "labels": 0, "monsters": 0}
    return _run


# ---------------- 解析 ----------------

def test_parse_sections_and_types(prof):
    ls = parse_lessons(prof.root / "lessons" / "t.md")
    kinds = [c.kind for c in ls.claims]
    assert KIND_SCENE in kinds and KIND_OBJECT in kinds and KIND_KEY in kinds
    assert ls.title == "测试游戏"
    # 术语走表格，不算 claim
    assert ("搬砖", "重复刷图", "主要玩法") in ls.terms


def test_scene_claim_fields(prof):
    ls = parse_lessons(prof.root / "lessons" / "t.md")
    scene = next(c for c in ls.claims if c.kind == KIND_SCENE)
    assert scene.title == "关卡选择页"
    assert scene.file == "img/map.png"
    assert scene.get("verdict") == "反例"
    assert scene.rect == [0, 0, 300, 80]


def test_rect_variants():
    assert parse_rect("0,0,300,80") == [0, 0, 300, 80]
    assert parse_rect("(10 20 30 40)") == [10, 20, 30, 40]
    assert parse_rect("[5,6,7,8]") == [5, 6, 7, 8]
    assert parse_rect("") is None
    assert parse_rect("1,2,3") is None


def test_heading_without_type_is_inferred(prof):
    """『## 图 A 某个没写类型的段』没有类型词，但字段是 图+判定 → 认成界面。"""
    ls = parse_lessons(prof.root / "lessons" / "t.md")
    tail = [c for c in ls.claims if "没写类型" in c.title]
    assert tail and tail[0].kind == KIND_SCENE


def test_unknown_section_warns_not_silent(tmp_path: Path):
    p = tmp_path / "x.md"
    p.write_text("# t\n\n## 随便写点什么\n\n一句话描述\n", encoding="utf-8")
    ls = parse_lessons(p)
    assert not ls.claims
    assert ls.warnings and "没看懂类型" in ls.warnings[0]


def test_key_section_keeps_raw_action_names(prof):
    """``- 拾取: A`` 的字段名就是动作名，不能被通用别名过滤掉。"""
    ls = parse_lessons(prof.root / "lessons" / "t.md")
    keys = next(c for c in ls.claims if c.kind == KIND_KEY)
    assert keys.fields.get("拾取") == "A"
    assert keys.fields.get("返回") == "ESC"


# ---------------- 核对 ----------------

def test_scene_check_pass_and_fail(prof):
    ls = load_lessons(prof.root / "lessons")
    cfg = AppConfig()
    # 全判"不在战斗"：反例 claim 一致，正例（战斗）不一致
    rows = check_claims(ls, prof, cfg, probe=_probe_stub({}))
    by = {r["title"]: r for r in rows}
    assert by["关卡选择页"]["status"] == "ok"
    assert by["战斗"]["status"] == "fail"
    assert "缺一块固定 HUD 锚点" in by["战斗"]["detail"]


def test_scene_check_fail_when_menu_passes_gate(prof):
    """反例被放行 = 最危险的一类：会在菜单上乱按键。"""
    ls = load_lessons(prof.root / "lessons")

    def _probe(img):
        return {"in_game": True, "gate_reason": "skillbar", "gate_best": 0.8,
                "gate_neg": 0.1, "pillars": 0, "pillar_colors": [],
                "labels": 0, "monsters": 0}

    rows = check_claims(ls, prof, AppConfig(), probe=_probe)
    by = {r["title"]: r for r in rows}
    assert by["关卡选择页"]["status"] == "fail"
    assert "gate_neg" in by["关卡选择页"]["detail"]


def test_object_rect_measures_threshold(prof):
    ls = load_lessons(prof.root / "lessons")
    rows = check_claims(ls, prof, AppConfig(), probe=_probe_stub({}))
    obj = next(r for r in rows if r["kind"] == KIND_OBJECT)
    assert obj["status"] == "ok", obj
    assert "高宽比=10.0" in obj["evidence"]
    assert "gold" in obj["evidence"]


def test_object_too_wide_is_flagged(prof, tmp_path: Path):
    """宽而短的光不是光柱（星图节点/Boss 光环就长这样），必须报出来。"""
    img = np.full((200, 200, 3), (30, 26, 24), dtype=np.uint8)
    img[50:70, 50:90] = (0, 214, 255)
    cv2.imwrite(str(tmp_path / "lessons" / "img" / "pillar.png"), img)
    # 框也要跟着改成宽条：高宽比由人给的框决定，不是由图上画了什么决定
    (tmp_path / "lessons" / "t.md").write_text(
        LESSON.replace("框: 50,50,6,60", "框: 50,50,40,20"), encoding="utf-8")
    ls = load_lessons(tmp_path / "lessons")
    rows = check_claims(ls, prof, AppConfig(), probe=_probe_stub({}))
    obj = next(r for r in rows if r["kind"] == KIND_OBJECT)
    assert obj["status"] == "fail"
    assert "高宽比" in obj["detail"]


def test_object_without_rect_is_gap(prof, tmp_path: Path):
    nr = tmp_path / "lessons" / "norect.md"
    nr.write_text("## 物体 · 光柱\n\n- 图: img/pillar.png\n- 颜色: 金色\n",
                  encoding="utf-8")
    ls = [parse_lessons(nr)]
    rows = check_claims(ls, prof, AppConfig(), probe=_probe_stub({}))
    obj = next(r for r in rows if r["kind"] == KIND_OBJECT)
    assert obj["status"] == "gap"
    assert "缺框" in obj["detail"]


def test_key_conflict_is_fail(prof, tmp_path: Path):
    k = tmp_path / "lessons" / "k.md"
    k.write_text("## 键位\n\n- 拾取: B\n- 返回: ESC\n- 走位: 点地面\n",
                 encoding="utf-8")
    # 只喂这一份：目录里还有 t.md（拾取=A），一起编译会互相覆盖同名条目
    ls = [parse_lessons(k)]
    rows = check_claims(ls, prof, AppConfig(), probe=_probe_stub({}))
    by = {r["title"]: r for r in rows}
    assert by["拾取"]["status"] == "fail"          # 配置默认 a
    assert by["返回"]["status"] == "ok"            # autonomy.escape_key = esc
    assert by["走位"]["status"] == "skip"          # 不是键，别报成缺口


def test_measure_region_clips_out_of_bounds():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    m = measure_region(img, [90, 90, 50, 50], AppConfig().vision)
    assert m["x"] + m["w"] <= 100 and m["y"] + m["h"] <= 100


# ---------------- 编译产物 ----------------

def test_labels_strict_only_when_check_passes(prof):
    ls = load_lessons(prof.root / "lessons")
    res = compile_lessons(ls, prof, AppConfig(), labels_data={},
                          probe=_probe_stub({}))
    by = {Path(it["file"]).name: it for it in res.labels_new}
    assert by["map.png"]["strict"] is True          # 反例，程序也判非战斗
    assert by["combat.png"]["strict"] is False      # 正例，程序没认出来
    assert "待标定" in by["combat.png"]["note"]
    assert by["map.png"]["expect"] == {"in_game": False}


def test_existing_label_not_overwritten(prof):
    ls = load_lessons(prof.root / "lessons")
    existing = {"items": [{"file": "lessons/img/map.png", "scene": "人手写的",
                           "strict": True, "expect": {"in_game": False},
                           "note": "人的话"}]}
    res = compile_lessons(ls, prof, AppConfig(), labels_data=existing,
                          probe=_probe_stub({}))
    assert "lessons/img/map.png" in res.labels_seen
    assert all(it["scene"] != "人手写的" for it in res.labels_new)
    assert not any(it["file"] == "lessons/img/map.png" for it in res.labels_new)


def test_write_labels_appends_without_touching_existing(prof, tmp_path: Path):
    lp = tmp_path / "labels.json"
    lp.write_text(json.dumps({"version": 1, "items": [
        {"file": "x.png", "scene": "旧", "strict": True, "expect": {},
         "note": "不能动"}]}, ensure_ascii=False), encoding="utf-8")
    ls = load_lessons(prof.root / "lessons")
    res = compile_lessons(ls, prof, AppConfig(), labels_data=json.loads(
        lp.read_text(encoding="utf-8")), probe=_probe_stub({}))
    n, _ = write_labels(res, lp)
    data = json.loads(lp.read_text(encoding="utf-8"))
    assert n == len(res.labels_new)
    assert data["items"][0]["note"] == "不能动"
    assert len(data["items"]) == 1 + n


def test_game_md_block_replaced_not_duplicated(prof):
    ls = load_lessons(prof.root / "lessons")
    res = compile_lessons(ls, prof, AppConfig(), labels_data={},
                          probe=_probe_stub({}))
    kdir = prof.root / "knowledge"
    write_game_md(res, kdir)
    first = (kdir / "game.md").read_text(encoding="utf-8")
    write_game_md(res, kdir)
    second = (kdir / "game.md").read_text(encoding="utf-8")
    assert second.count("<!-- teach:begin") == 1
    assert first.count("<!-- teach:begin") == 1


def test_game_md_keeps_human_text_outside_block(prof):
    kdir = prof.root / "knowledge"
    kdir.mkdir(exist_ok=True)
    (kdir / "game.md").write_text("# 我写的\n\n别动我\n", encoding="utf-8")
    ls = load_lessons(prof.root / "lessons")
    res = compile_lessons(ls, prof, AppConfig(), labels_data={},
                          probe=_probe_stub({}))
    write_game_md(res, kdir)
    text = (kdir / "game.md").read_text(encoding="utf-8")
    assert "别动我" in text
    assert "### 禁忌" in text


def test_glossary_from_terms(prof):
    ls = load_lessons(prof.root / "lessons")
    g = build_glossary(ls)
    assert g["搬砖"]["canonical"] == "重复刷图"
    assert g["星图"]["note"] == ""


def test_md_block_contains_taboo_and_flow(prof):
    ls = load_lessons(prof.root / "lessons")
    res = compile_lessons(ls, prof, AppConfig(), labels_data={},
                          probe=_probe_stub({}))
    assert "未标定画面不要按 ESC" in res.md_block
    assert "点节点进图" in res.md_block
    assert "拾取：A" in res.md_block


def test_html_comment_is_not_parsed_as_section(tmp_path: Path):
    """注释里常写着「## 界面」这种示例，不剔掉会解析出幽灵条目。"""
    p = tmp_path / "c.md"
    p.write_text(
        "# t\n\n<!--\n## 界面 · 假的\n- 判定: 反例\n-->\n\n## 禁忌\n- 别按 ESC\n",
        encoding="utf-8")
    ls = parse_lessons(p)
    assert [c.kind for c in ls.claims] == ["taboo"]
    assert "别按 ESC" in ls.claims[0].body[0]


def test_html_comment_is_not_parsed_as_section(tmp_path: Path):
    """注释里常写着「## 界面」这种示例，不剔掉会解析出幽灵条目。"""
    p = tmp_path / "c.md"
    p.write_text(
        "# t\n\n<!--\n## 界面 · 假的\n- 判定: 反例\n-->\n\n## 禁忌\n- 别按 ESC\n",
        encoding="utf-8")
    ls = parse_lessons(p)
    assert [c.kind for c in ls.claims] == ["taboo"]
    assert "别按 ESC" in ls.claims[0].body[0]


def test_real_profile_tutorial_matches_program():
    """真台账：`tli_phone960` 的教学手册与程序现状必须一致。

    这条是"教程即回归"的锚——谁改了阈值/模板让星图又被判成战斗，这里直接红，
    不用等到跑录像才发现。
    """
    from autogame.config import load_config
    from autogame.profile import load_profile

    root = Path(__file__).resolve().parent.parent
    prof = load_profile("tli_phone960")
    lessons = load_lessons(prof.root / "lessons")
    assert lessons, "教学手册不能是空的"
    cfg = load_config(root / "config.yaml", profile=prof)
    res = compile_lessons(lessons, prof, cfg)
    assert not res.failures, "\n".join(
        f"{c['title']}: {c['detail']}（{c['evidence']}）" for c in res.failures)
    assert res.glossary, "术语表不能是空的"


def test_real_profile_tutorial_matches_program():
    """真台账：`tli_phone960` 的教学手册与程序现状必须一致。

    这条是"教程即回归"的锚——谁改了阈值/模板让星图又被判成战斗，这里直接红，
    不用等到跑录像才发现。
    """
    from autogame.config import load_config
    from autogame.profile import load_profile

    root = Path(__file__).resolve().parent.parent
    prof = load_profile("tli_phone960")
    lessons = load_lessons(prof.root / "lessons")
    assert lessons, "教学手册不能是空的"
    cfg = load_config(root / "config.yaml", profile=prof)
    res = compile_lessons(lessons, prof, cfg)
    assert not res.failures, "\n".join(
        f"{c['title']}: {c['detail']}（{c['evidence']}）" for c in res.failures)
    assert res.glossary, "术语表不能是空的"
