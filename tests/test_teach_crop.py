"""文档 → 模板 的中间层测试。

钉的不是"算法能跑"，而是七条人味边界：

1. 方位词必须和它后面的名词绑在一起 —— "右下角技能轮盘 + 左下角摇杆"是两件事
2. 否定句不算数 —— "右下没有技能轮盘"不是让你去右下角找
3. 高宽比按各自的名词定 —— 同一句里"标题块"和"圆形节点"差一倍
4. 说"轮盘"就真的去找圆 —— 否则第一次在真实数据上选中了聊天文字
5. 框不能太大太小 —— 锚点是 HUD 的一个零件，不是小半屏，也不是一个按钮
6. 自动量出来的框要能对上人手标的框（这条最重要，是整个工具存在的理由）
7. 人写的框一律优先，回填只填空着的地方

第 6 条用真实截图跑，阈值故意放得宽（IoU>0.4）：要钉的是"自动量的框能当锚点
用"，不是"和人手的框一模一样"。差一点没关系——预览页就是给人改这一步用的。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from autogame import teach_crop as tc
from autogame.config import load_config
from autogame.profile import load_profile
from autogame.teach import Lesson, load_lessons, parse_lessons

PROFILE_ID = "tli_phone960"


# ---------------- 夹具 ----------------

@pytest.fixture(scope="module")
def prof():
    try:
        return load_profile(PROFILE_ID)
    except FileNotFoundError:
        pytest.skip(f"缺少 profile: {PROFILE_ID}")


@pytest.fixture(scope="module")
def cfg(prof):
    return load_config("config.yaml", profile=prof)


@pytest.fixture(scope="module")
def lessons(prof):
    return load_lessons(prof.root / "lessons")


@pytest.fixture(scope="module")
def pool(prof, lessons):
    return tc.load_pool(lessons, prof)


def _intent(prof, lessons, key: str) -> tc.Intent:
    for it in tc.plan_intents(prof, lessons):
        if key in it.title:
            return it
    pytest.skip(f"手册里没有包含『{key}』的界面段")


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - ix * iy
    return ix * iy / union if union > 0 else 0.0


# ---------------- 1. 方位绑名词 ----------------

def test_hints_bind_each_noun_to_its_direction():
    hints = tc.parse_hints("右下角技能轮盘（一圆圈图标）+ 左下角摇杆，左上小地图")
    assert [h.region_key for h in hints] == ["BR", "BL", "TL"]
    assert "轮盘" in hints[0].noun
    assert "摇杆" in hints[1].noun
    assert "小地图" in hints[2].noun


def test_negated_direction_is_not_a_target():
    """『右下没有技能轮盘』是描述"这里什么都没有"，不该去右下角找一个轮盘。"""
    hints = tc.parse_hints("左上角「‹ 异界」标题块，中间 7 个发光圆形节点，右下没有技能轮盘")
    assert [h.region_key for h in hints] == ["TL", "MID"]


def test_direction_without_noun_still_works():
    """人只写 `- 在: 右下` 也得能用——那是"往那边搜"，不是"找不到东西"。"""
    hints = tc.parse_hints("右下")
    assert len(hints) == 1
    assert hints[0].region_key == "BR"
    assert hints[0].noun == ""


def test_longest_direction_wins_once():
    """『右下角』命中之后不能再算一次『右下』，否则同一处会出现两个区域。"""
    hints = tc.parse_hints("右下角的血条")
    assert len(hints) == 1


# ---------------- 2. 形状与高宽比 ----------------

def test_aspects_follow_each_noun_not_the_whole_sentence():
    hints = tc.parse_hints("左上角「‹ 异界」标题块 + 中间 7 个发光圆形节点")
    bar, circle = hints[0], hints[1]
    # 标题块是扁的（h/w < 1），圆形节点接近方
    assert max(bar.aspects) <= 0.6
    assert min(circle.aspects) >= 0.7


def test_shape_word_decides_the_operator():
    assert tc.parse_hints("右下角技能轮盘")[0].kind == "circle"
    assert tc.parse_hints("左上角标题块")[0].kind == "bar"
    assert tc.parse_hints("左上小地图")[0].kind == ""      # 没形状词 → 走通用路径


def test_aspects_from_unknown_text_returns_defaults():
    assert tc.aspects_from("这块我也说不清") == tc._DEFAULT_ASPECTS


# ---------------- 3. 框的尺寸边界 ----------------

def test_proposed_rects_respect_size_caps(prof, lessons):
    """没有一个候选能大到"小半屏"，也没有一个能小到一个按钮。"""
    it = _intent(prof, lessons, "异界深处")
    img = cv2.imread(str(it.image))
    H, W = img.shape[:2]
    rects = tc.propose_rects(it, img)
    assert rects, "右下角应该能铺出候选"
    for r, _src, _w in rects:
        x, y, w, h = r
        assert w <= int(W * tc._MAX_W_FRAC) or w > W * 0.2, f"框太宽: {r}"
        assert x >= 0 and y >= 0 and x + w <= W and y + h <= H, f"框越界: {r}"


def test_size_fit_prefers_hud_sized_boxes():
    """人的锚点落在整帧 4%~10%；太小的按钮和太大的半屏都该被压下去。"""
    assert tc._size_fit(0.06) == 1.0
    assert tc._size_fit(0.085) == 1.0
    assert tc._size_fit(0.012) < 0.2      # 70x70 的技能按钮
    assert tc._size_fit(0.30) < 0.2       # 小半屏


# ---------------- 4. 圆检测 ----------------

def test_circle_group_finds_the_skill_wheel(prof):
    """真实战斗帧的右下角，4 个技能图标要能被聚成"一圈"。"""
    img = cv2.imread(str(prof.root / "refs" / "tli_combat_01.jpg"))
    assert img is not None
    H, W = img.shape[:2]
    sub = img[int(H * 0.54):, int(W * 0.54):]
    groups = tc._circle_groups(sub)
    assert groups, "右下角至少该找到一组圆"
    assert max(n for _box, n in groups) >= 3, (
        f"技能盘是 4 个图标围一圈，应聚成 3 个以上，实际 {[n for _, n in groups]}")

    box, _n = groups[0]
    gx, gy, gw, gh = box
    # 整圈应该在右下角、并且落在画面内
    assert gx + int(W * 0.54) > W * 0.5
    assert gw > 100 and gh > 100


def test_circle_group_box_stays_inside_region(prof):
    """外扩之后不能越界——越界的框会被尺寸规则整条丢掉，表现为"只剩一个单按钮"。"""
    img = cv2.imread(str(prof.root / "refs" / "tli_combat_01.jpg"))
    H, W = img.shape[:2]
    sub = img[int(H * 0.54):, int(W * 0.54):]
    for (x, y, w, h), _n in tc._circle_groups(sub):
        assert x >= 0 and y >= 0
        assert x + w <= sub.shape[1] and y + h <= sub.shape[0], \
            f"框超出搜索区域: {(x, y, w, h)} vs {sub.shape[1]}x{sub.shape[0]}"


# ---------------- 5. 自动量出的框能当锚点用 ----------------

def test_auto_rect_matches_human_rect_on_combat(prof, cfg, lessons, pool):
    """把手册里的框清掉，纯靠"右下角技能轮盘"这句话，也要框到技能盘上。

    这条是整个工具存在的理由。IoU 阈值 0.4 是有意放松的：自动框通常比人的框
    完整（人常常只框技能盘下半部分），重叠够了就能当锚点。
    """
    it = _intent(prof, lessons, "异界深处")
    it.given_rect = None
    res = tc.search(it, pool, cfg, tpl_dir=prof.templates_dir)
    assert res.best is not None, res.reason
    human = [640, 255, 260, 135]          # 手册里人标的框
    assert _iou(res.best.rect, human) > 0.4, (
        f"自动框 {res.best.rect} 与人手框 {human} 差太远")
    assert res.best.ev["ok"], "自动框必须能通过跨图验证"


def test_auto_rect_finds_whole_wheel_not_single_button(prof, cfg, lessons, pool):
    """找到的是"一圈图标"，不是一个按钮——单按钮会随冷却数字变化而失配。"""
    it = _intent(prof, lessons, "Boss")
    it.given_rect = None
    res = tc.search(it, pool, cfg, tpl_dir=prof.templates_dir)
    assert res.best is not None, res.reason
    w, h = res.best.rect[2], res.best.rect[3]
    area = w * h / (960 * 432)
    assert area > 0.05, f"选中的是一小块（{w}x{h}，占帧 {area:.1%}），不是整圈"


def test_auto_rect_passes_thresholds_on_menus(prof, cfg, lessons, pool):
    """反例模板的 hit/miss 是反的：该命中菜单、不该命中战斗。"""
    it = _intent(prof, lessons, "星图")
    it.given_rect = None
    res = tc.search(it, pool, cfg, tpl_dir=prof.templates_dir)
    assert res.best is not None, res.reason
    ev = res.best.ev
    assert ev["ok"], f"反例模板没通过：{ev}"
    assert ev["hit_min"] >= ev["hit_thr"], "在星图上必须命中"
    if ev["miss_max"] is not None:
        assert ev["miss_max"] < ev["miss_thr"], "在战斗图上不该命中"


def test_search_never_picks_chat_text(prof, cfg, lessons, pool):
    """第一次在真实数据上跑，第一名是左下角聊天记录（两张连续帧逐字相同）。

    挡住它的是"说轮盘就只在圆里找"，不是文字识别——所以这里钉的是"别再选到
    左下那一长条"。
    """
    it = _intent(prof, lessons, "异界深处")
    it.given_rect = None
    res = tc.search(it, pool, cfg, tpl_dir=prof.templates_dir)
    assert res.best is not None
    x, y, w, h = res.best.rect
    is_chat_bar = y > 350 and w > 250 and h < 70
    assert not is_chat_bar, f"又选中了聊天文字那一长条: {res.best.rect}"


# ---------------- 6. 验证集的语义 ----------------

def test_evaluate_swaps_hit_and_miss_for_negative_templates(prof, cfg, lessons, pool):
    """反例模板把"该命中"和"不该命中"对调，阈值也跟着换。"""
    it = _intent(prof, lessons, "星图")
    tpl = cv2.imread(str(prof.templates_dir / "gate_neg_nether_title.png"))
    if tpl is None:
        pytest.skip("缺少现成的反例模板")
    assert not it.hit_positive
    ev = tc.evaluate(tpl, it, pool, cfg)
    assert ev["hit_thr"] == cfg.vision.scene_gate.negative_threshold
    assert ev["miss_thr"] == cfg.vision.scene_gate.threshold
    assert ev["n_hit"] >= 1 and ev["n_miss"] >= 1


def test_evaluate_reports_sample_counts(prof, cfg, lessons, pool):
    it = _intent(prof, lessons, "异界深处")
    tpl = cv2.imread(str(prof.templates_dir / "gate_pos_skillwheel.png"))
    if tpl is None:
        pytest.skip("缺少现成的正向模板")
    ev = tc.evaluate(tpl, it, pool, cfg)
    assert ev["n_hit"] == 2 and ev["n_miss"] == 1


# ---------------- 7. ROI 反推 ----------------

def test_suggested_roi_covers_the_positive_templates(prof, cfg, lessons, pool):
    """ROI 由"锚点在哪"反推，不能沿用 PC 的"底部 35% 全宽"。"""
    results = []
    for key in ("异界深处",):
        it = _intent(prof, lessons, key)
        results.append(tc.search(it, pool, cfg, tpl_dir=prof.templates_dir))
    roi = tc.suggest_roi(results, prof)
    assert roi is not None
    x, y, w, h = roi
    best = results[0].best.rect
    assert x <= best[0] and y <= best[1], "ROI 必须包含锚点"
    assert x + w >= best[0] + best[2] and y + h >= best[1] + best[3]
    # 现有的手工 ROI 是 [600,235,360,197]，反推出来的应该和它大体重合
    assert _iou(roi, [600, 235, 360, 197]) > 0.4


# ---------------- 8. 命名与落盘 ----------------

def test_template_name_is_ascii_safe(prof, lessons):
    """cv2.imwrite 写不了中文路径，所以文件名必须 ASCII。"""
    it = tc.Intent(id="x", title="界面 · 掉落光柱（金色）", role=tc.ROLE_POS,
                   kind="scene", source="a.md", line=1)
    name = tc.template_name(it, 2)
    assert name.startswith("gate_pos_")
    assert name.isascii(), name
    assert name.endswith(".png")


def test_template_name_prefers_human_given_slug():
    it = tc.Intent(id="x", title="战斗", role=tc.ROLE_POS, kind="scene",
                   source="a.md", line=1, tpl_name="skillwheel")
    assert tc.template_name(it, 7) == "gate_pos_skillwheel.png"


def test_save_template_records_source_in_index(prof, lessons, tmp_path):
    it = _intent(prof, lessons, "异界深处")
    cand = tc.Cand(rect=[640, 255, 260, 135],
                   tpl=np.full((60, 120, 3), 128, np.uint8))
    fake = tc.ProfileLike = type("P", (), {})()          # 简易替身，避免动真目录
    fake.templates_dir = tmp_path
    fake.reference_resolution = (960, 432)
    out = tc.save_template(fake, it, cand, "gate_pos_x.png")
    assert out.exists()
    import json
    idx = json.loads((tmp_path / "_index.json").read_text(encoding="utf-8"))
    assert idx["gate_pos_x.png"]["rect"] == [640, 255, 260, 135]
    assert idx["gate_pos_x.png"]["source"]


# ---------------- 9. 回填只填空 ----------------

LESSON_WITH_HOLE = """# 测试

## 界面 · 战斗

- 图: refs/tli_combat_01.jpg
- 判定: 正例
- 一眼认出: 右下角技能轮盘
- 框: 待补充
"""

LESSON_WITH_HUMAN_RECT = """# 测试

## 界面 · 战斗

- 图: refs/tli_combat_01.jpg
- 判定: 正例
- 框: 1,2,3,4
"""


def _write_back_on(tmp_path, text: str, prof, rect=(640, 255, 260, 135)):
    p = tmp_path / "l.md"
    p.write_text(text, encoding="utf-8")
    lesson = parse_lessons(p)
    intent = tc.Intent(id="i", title=lesson.claims[0].title, role=tc.ROLE_POS,
                       kind="scene", source=p.name, line=lesson.claims[0].line)
    cand = tc.Cand(rect=list(rect), tpl=np.zeros((10, 10, 3), np.uint8))
    report = tc.CropReport(profile="t", results=[tc.CropResult(intent, best=cand)])
    tc.write_back(prof, [lesson], report)
    return p.read_text(encoding="utf-8")


def test_write_back_fills_placeholder(prof, tmp_path):
    out = _write_back_on(tmp_path, LESSON_WITH_HOLE, prof)
    assert "- 框: 640,255,260,135" in out


def test_write_back_never_touches_human_rect(prof, tmp_path):
    """人写的框比自动量的值钱。回填是填空，不是纠正。"""
    out = _write_back_on(tmp_path, LESSON_WITH_HUMAN_RECT, prof)
    assert "- 框: 1,2,3,4" in out
    assert "640,255" not in out


def test_write_back_inserts_line_when_missing(prof, tmp_path):
    text = LESSON_WITH_HOLE.replace("- 框: 待补充\n", "")
    out = _write_back_on(tmp_path, text, prof)
    assert "- 框: 640,255,260,135" in out


# ---------------- 10. --set 解析 ----------------

def test_parse_set_accepts_four_numbers():
    got = tc._parse_set(["星图=0,0,300,80", "战斗=640,255,260,135"])
    assert got["星图"] == [0, 0, 300, 80]
    assert got["战斗"] == [640, 255, 260, 135]


def test_parse_set_rejects_bad_input():
    with pytest.raises(SystemExit):
        tc._parse_set(["星图"])
    with pytest.raises(SystemExit):
        tc._parse_set(["星图=1,2"])


# ---------------- 11. 打分口径不能漂 ----------------

def test_gate_score_matches_production_detector(prof, cfg, tmp_path):
    """`gate_score.match_best` 必须和 Detector 内部算出的分数一模一样。

    这条是防漂移锚点。找框、裁模板、生产门禁三处都用同一个函数；如果哪天有人
    只改了其中一侧，工具说"0.58 不够"而生产 0.60 放行，人就会开始不信任何数字。
    """
    import shutil

    from autogame.vision.detector import Detector
    from autogame.vision.gate_score import match_best

    src_tpl = prof.templates_dir / "gate_pos_skillwheel.png"
    img_path = prof.root / "refs" / "tli_combat_01.jpg"
    if not src_tpl.exists() or not img_path.exists():
        pytest.skip("缺少现成模板或截图")

    one = tmp_path / "tpl"
    one.mkdir()
    shutil.copy(src_tpl, one / src_tpl.name)

    img = cv2.imread(str(img_path))
    det = Detector(cfg.vision, one)
    det.process(img, 1000)
    det.process(img, 2000)          # 两拍：门禁有缓存，第二拍才是稳态
    got = float(det.diag["gate_best"])
    assert det.diag["gate_reason"] in ("skillbar", "unmatched"), det.diag["gate_reason"]

    mine = match_best(img, cv2.imread(str(src_tpl)),
                      roi=list(cfg.vision.scene_gate.roi),
                      ref_res=list(cfg.vision.reference_resolution),
                      max_detect_width=int(cfg.vision.max_detect_width or 0))
    assert abs(mine - got) < 1e-3, f"工具算 {mine}，生产算 {got}"


def test_gate_score_matches_detector_on_negative(prof, cfg, tmp_path):
    import shutil

    from autogame.vision.detector import Detector
    from autogame.vision.gate_score import match_best

    src_tpl = prof.templates_dir / "gate_neg_nether_title.png"
    img_path = prof.root / "refs" / "tli_nether_map_01.jpg"
    if not src_tpl.exists() or not img_path.exists():
        pytest.skip("缺少现成模板或截图")

    one = tmp_path / "tpl"
    one.mkdir()
    shutil.copy(src_tpl, one / src_tpl.name)
    shutil.copy(prof.templates_dir / "gate_pos_skillwheel.png", one / "gate_pos_skillwheel.png")

    img = cv2.imread(str(img_path))
    det = Detector(cfg.vision, one)
    det.process(img, 1000)
    det.process(img, 2000)
    got = float(det.diag["gate_neg_best"])
    mine = match_best(img, cv2.imread(str(src_tpl)), whole_frame=True,
                      ref_res=list(cfg.vision.reference_resolution),
                      max_detect_width=int(cfg.vision.max_detect_width or 0))
    assert abs(mine - got) < 1e-3, f"工具算 {mine}，生产算 {got}"


# ---------------- 11. 打分口径不能漂 ----------------

def test_gate_score_matches_production_detector(prof, cfg, tmp_path):
    """`gate_score.match_best` 必须和 Detector 内部算出的分数一模一样。

    这条是防漂移锚点。找框、裁模板、生产门禁三处都用同一个函数；如果哪天有人
    只改了其中一侧，工具说"0.58 不够"而生产 0.60 放行，人就会开始不信任何数字。
    """
    import shutil

    from autogame.vision.detector import Detector
    from autogame.vision.gate_score import match_best

    src_tpl = prof.templates_dir / "gate_pos_skillwheel.png"
    img_path = prof.root / "refs" / "tli_combat_01.jpg"
    if not src_tpl.exists() or not img_path.exists():
        pytest.skip("缺少现成模板或截图")

    one = tmp_path / "tpl"
    one.mkdir()
    shutil.copy(src_tpl, one / src_tpl.name)

    img = cv2.imread(str(img_path))
    det = Detector(cfg.vision, one)
    det.process(img, 1000)
    det.process(img, 2000)          # 两拍：门禁有缓存，第二拍才是稳态
    got = float(det.diag["gate_best"])
    assert det.diag["gate_reason"] in ("skillbar", "unmatched"), det.diag["gate_reason"]

    mine = match_best(img, cv2.imread(str(src_tpl)),
                      roi=list(cfg.vision.scene_gate.roi),
                      ref_res=list(cfg.vision.reference_resolution),
                      max_detect_width=int(cfg.vision.max_detect_width or 0))
    assert abs(mine - got) < 1e-3, f"工具算 {mine}，生产算 {got}"


def test_gate_score_matches_detector_on_negative(prof, cfg, tmp_path):
    import shutil

    from autogame.vision.detector import Detector
    from autogame.vision.gate_score import match_best

    src_tpl = prof.templates_dir / "gate_neg_nether_title.png"
    img_path = prof.root / "refs" / "tli_nether_map_01.jpg"
    if not src_tpl.exists() or not img_path.exists():
        pytest.skip("缺少现成模板或截图")

    one = tmp_path / "tpl"
    one.mkdir()
    shutil.copy(src_tpl, one / src_tpl.name)
    shutil.copy(prof.templates_dir / "gate_pos_skillwheel.png", one / "gate_pos_skillwheel.png")

    img = cv2.imread(str(img_path))
    det = Detector(cfg.vision, one)
    det.process(img, 1000)
    det.process(img, 2000)
    got = float(det.diag["gate_neg_best"])
    mine = match_best(img, cv2.imread(str(src_tpl)), whole_frame=True,
                      ref_res=list(cfg.vision.reference_resolution),
                      max_detect_width=int(cfg.vision.max_detect_width or 0))
    assert abs(mine - got) < 1e-3, f"工具算 {mine}，生产算 {got}"
