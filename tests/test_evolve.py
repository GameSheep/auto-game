"""二次进化的边界测试。

这些用例钉的不是"功能能跑"，而是几条**不许松动的规则**：

    材料不够不许出代       —— 把不确定性藏进分数里，比明说"学不了"糟得多
    材料没变不许出新代     —— 否则每跑一次多一代，回滚列表很快变成垃圾场
    规则有一处错整份不用   —— 半份规则跑起来，行为无法从文档推断
    新代退化了不许上线     —— 这是"持续学习"和"越学越坏"的唯一分界
    学坏了必须一条命令能退 —— 设备上没有 git，退不回去等于变砖

临时 profile 全部造在 tmp_path 里，不碰真实的 tli_phone960。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from autogame import evolve
from autogame.config import load_config
from autogame.decision.knowledge import KnowledgeBase
from autogame.profile import load_profile

LESSON = """# 测试用手册

## 界面 · 战斗
- 图: refs/combat_a.jpg
- 判定: 正例
- 在: 右下角技能轮盘

## 界面 · 战斗（另一张）
- 图: refs/combat_b.jpg
- 判定: 正例
- 在: 右下角技能轮盘

## 界面 · 背包
- 图: refs/bag_a.jpg
- 判定: 反例

## 界面 · 结算
- 图: refs/result_a.jpg
- 判定: 反例

## 流程 · 没子弹了
- 一眼认出: 连续三次攻击都没打掉怪
- 该做什么: 先按 esc 退开，再等两秒
"""


def _frame(seed: int, h: int = 200, w: int = 360) -> np.ndarray:
    """造一张"长得不一样"的图。种子不同 → 相关度低，不会被当成重复样本。"""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    cv2.circle(img, (w // 2, h // 2), 30, (40, 200, 240), -1)
    return img


@pytest.fixture()
def tmp_profile(tmp_path):
    """造一个材料齐全的临时 profile。

    故意不复制真实截图：这些用例验的是"流转和闸门"，不是识别精度。
    用真图会让用例变慢，还会让"改了阈值就红"混淆"改了流程就红"。
    """
    root = tmp_path / "profiles" / "unit_test"
    (root / "lessons").mkdir(parents=True)
    (root / "refs").mkdir()
    (root / "templates").mkdir()

    (root / "profile.yaml").write_text(
        "id: unit_test\nname: 单元测试\nstatus: partial\n"
        "reference_resolution: [360, 200]\nconfig: {}\n", encoding="utf-8")
    (root / "lessons" / "t.md").write_text(LESSON, encoding="utf-8")

    for i, name in enumerate(("combat_a.jpg", "combat_b.jpg", "bag_a.jpg",
                             "result_a.jpg")):
        cv2.imwrite(str(root / "refs" / name), _frame(i))

    labels = {"version": 1, "items": [
        {"file": "refs/combat_a.jpg", "scene": "战斗", "strict": False,
         "expect": {"in_game": True}},
    ]}
    (root / "labels.json").write_text(json.dumps(labels), encoding="utf-8")

    p = load_profile(root)
    yield p


# ---------------- 1. 材料体检 ----------------

def test_audit_blocks_when_positives_missing(tmp_profile):
    """正例只有 1 张时必须判"不够"。

    一张图过阈值毫无信息量：模板就是从它身上裁的，自匹配必然接近 1.0。
    这不是"稍微少一点"，是"这条验证根本不成立"。
    """
    lesson = tmp_profile.root / "lessons" / "t.md"
    text = lesson.read_text(encoding="utf-8")
    # 删掉第二个正例
    text = text.replace("""## 界面 · 战斗（另一张）
- 图: refs/combat_b.jpg
- 判定: 正例
- 在: 右下角技能轮盘
""", "")
    lesson.write_text(text, encoding="utf-8")

    a = evolve.audit_materials(tmp_profile)
    pos = next(n for n in a.needs if "正例" in n.what)
    assert pos.have == 1 and not pos.ok and pos.blocking
    assert not a.ready


def test_audit_flags_images_not_referenced(tmp_profile):
    """放进 refs 但手册没引用 → 必须列出来。这是"图给了却没用"的头号原因。"""
    cv2.imwrite(str(tmp_profile.root / "refs" / "orphan.jpg"), _frame(9))
    a = evolve.audit_materials(tmp_profile)
    assert "orphan.jpg" in a.unused_images
    # 被引用的四张不能误报
    assert "combat_a.jpg" not in a.unused_images


def test_audit_detects_duplicate_samples(tmp_profile):
    """同一张图存两份 → 判冗余。加起来不算"两张不同帧"。"""
    shutil.copy(tmp_profile.root / "refs" / "combat_a.jpg",
                tmp_profile.root / "refs" / "combat_a_copy.jpg")
    a = evolve.audit_materials(tmp_profile)
    assert any("combat_a" in d for d in a.duplicates)


def test_digest_ignores_inbox(tmp_profile):
    """inbox 里的图不算材料 —— 否则"丢一张待标定的图"就触发一整代回归，白烧设备。"""
    (tmp_profile.root / "inbox").mkdir()
    before = evolve.materials_digest(tmp_profile)
    cv2.imwrite(str(tmp_profile.root / "inbox" / "new.jpg"), _frame(77))
    assert evolve.materials_digest(tmp_profile) == before


def test_digest_changes_when_lessons_change(tmp_profile):
    before = evolve.materials_digest(tmp_profile)
    lesson = tmp_profile.root / "lessons" / "t.md"
    lesson.write_text(lesson.read_text(encoding="utf-8") + "\n## 术语 · 光柱\n- 别名: 光条\n",
                      encoding="utf-8")
    assert evolve.materials_digest(tmp_profile) != before


# ---------------- 2. 出代 ----------------

def _cfg(prof):
    return load_config(None, profile=prof)


def test_build_refuses_when_materials_unchanged(tmp_profile):
    """材料指纹没变就不出新代。这是防"跑十次十代"的唯一一道闸。"""
    gen, note = evolve.build(tmp_profile, _cfg(tmp_profile), use_model=False)
    assert gen is not None, note
    again, why = evolve.build(tmp_profile, _cfg(tmp_profile), use_model=False)
    assert again is None
    assert "完全一样" in why


def test_build_writes_knowledge_and_rules_skeleton(tmp_profile):
    gen, note = evolve.build(tmp_profile, _cfg(tmp_profile), use_model=False)
    assert gen is not None, note
    assert gen.rules_path.exists()
    doc = yaml.safe_load(gen.rules_path.read_text(encoding="utf-8"))
    assert doc["rules"] == []
    assert (gen.root / "knowledge" / "game.md").exists()
    # 还没有上线：出代和上线是两步，中间必须夹着回归
    assert evolve.active_id(tmp_profile) is None


def test_build_blocked_when_audit_fails(tmp_profile, monkeypatch):
    """体检不过时 build 直接拒绝；--force 才继续（排障用）。"""
    monkeypatch.setattr(evolve, "audit_materials",
                        lambda p: evolve.Audit(needs=[evolve.Need(
                            what="正例", have=0, need=2, why="", fix="")]))
    gen, why = evolve.build(tmp_profile, _cfg(tmp_profile), use_model=False)
    assert gen is None and "材料不够" in why


# ---------------- 3. 规则校验 ----------------

def test_validate_rejects_unlisted_fact(tmp_profile):
    """引用不存在的事实 → 整份拒绝，并且说清可用的是哪些。"""
    doc = {"rules": [{"id": "r1", "when": {"fact": "hp_frac", "op": "<", "value": 0.3},
                      "then": {"action": "wait"}}]}
    rules, errors = evolve.validate_rules(doc)
    assert rules == [] and errors
    assert "hp_frac" in str(errors[0])
    assert "pillars" in str(errors[0])      # 报错要带上可用清单，否则人得去翻代码


def test_validate_rejects_unlisted_action():
    doc = {"rules": [{"id": "r1", "when": {"fact": "pillars", "op": ">", "value": 0},
                      "then": {"action": "dodge"}}]}
    rules, errors = evolve.validate_rules(doc)
    assert rules == []
    assert any("dodge" in str(e) for e in errors)


def test_validate_rejects_type_mismatch():
    """整数事实给字符串 → 拒绝。放过它会在求值时抛 TypeError 或永远不命中。"""
    doc = {"rules": [{"id": "r1", "when": {"fact": "pillars", "op": "==", "value": "many"},
                      "then": {"action": "wait"}}]}
    rules, errors = evolve.validate_rules(doc)
    assert rules == []
    assert any("整数" in str(e) for e in errors)


def test_validate_rejects_unknown_param():
    doc = {"rules": [{"id": "r1", "when": {"fact": "pillars", "op": ">", "value": 0},
                      "then": {"action": "pickup", "params": {"x": 100}}}]}
    rules, errors = evolve.validate_rules(doc)
    assert rules == []
    assert any("不认参数" in str(e) for e in errors)


def test_one_bad_rule_voids_whole_file():
    """有一条错就整份不用。

    "能用的先用上"会让实际行为无法从文档推断 —— 人看到自己写的第一条没生效，
    会以为是求值问题，查半天才发现是第三条把整份废了（反过来也一样糟）。
    """
    doc = {"rules": [
        {"id": "good", "when": {"fact": "pillars", "op": ">", "value": 0},
         "then": {"action": "pickup"}},
        {"id": "bad", "when": {"fact": "nope", "op": ">", "value": 0},
         "then": {"action": "wait"}},
    ]}
    rules, errors = evolve.validate_rules(doc)
    assert rules == [] and errors


# ---------------- 4. 规则求值 ----------------

def test_rule_order_is_priority():
    """顺序即优先级：第一条命中就返回，不看完整个列表。"""
    doc = {"rules": [
        {"id": "a", "when": {"fact": "pillars", "op": ">", "value": 0},
         "then": {"action": "wait"}},
        {"id": "b", "when": {"fact": "pillars", "op": ">", "value": 0},
         "then": {"action": "pickup"}},
    ]}
    rules, errors = evolve.validate_rules(doc)
    assert not errors
    hit = evolve.evaluate(rules, {"pillars": 3})
    assert hit["rule_id"] == "a" and hit["action"] == "wait"


def test_missing_fact_does_not_fire():
    """字段缺失 → 不命中（不是当 False 命中）。

    当 False 的后果：写 `pillars >= 1 → pickup` 的人会看到"这张图上没捡"，
    以为是规则写错；真正的原因是 probe 里压根没这个字段。
    """
    doc = {"rules": [{"id": "a", "when": {"fact": "pillars", "op": ">=", "value": 1},
                      "then": {"action": "pickup"}}]}
    rules, _ = evolve.validate_rules(doc)
    hit = evolve.evaluate(rules, {"in_game": True})
    assert not hit["hit"]
    assert hit["unknown"] == ["a"]


def test_all_and_any_combinations():
    doc = {"rules": [{"id": "a", "when": {"any": [
        {"fact": "gate", "op": "in", "value": ["unmatched", "no_template"]},
        {"all": [{"fact": "monsters", "op": ">=", "value": 3},
                 {"fact": "pillars", "op": "==", "value": 0}]},
    ]}, "then": {"action": "esc"}}]}
    rules, _ = evolve.validate_rules(doc)
    assert evolve.evaluate(rules, {"gate": "unmatched"})["hit"]
    assert evolve.evaluate(rules, {"gate": "skillbar", "monsters": 5, "pillars": 0})["hit"]
    assert not evolve.evaluate(rules, {"gate": "skillbar", "monsters": 1,
                                      "pillars": 0})["hit"]


def test_ruleset_loads_bad_file_as_empty_without_raising(tmp_path):
    """规则文件坏了 → 空集 + 报错日志，不能把决策链拖down。"""
    p = tmp_path / "rules.yaml"
    p.write_text("rules:\n  - id: x\n    when: {fact: nope, op: '>', value: 1}\n"
                 "    then: {action: wait}\n", encoding="utf-8")
    rs = evolve.RuleSet.load(p)
    assert not rs and rs.decide({"pillars": 1}) is None


def test_ruleset_missing_file_is_empty(tmp_path):
    assert not evolve.RuleSet.load(tmp_path / "nope.yaml")


# ---------------- 5. 回归闸门 ----------------

def _fake_score(passes: int, total: int = 3, fails=None):
    return {"strict_pass": passes, "strict_total": total,
            "ratio": passes / total if total else 0.0,
            "fail": fails or [], "gap": [], "items": []}


def test_first_generation_must_pass_all(tmp_profile, monkeypatch):
    """第一代不满分不定基线。否则问题会被固化成"基线"，以后再也看不出退化。"""
    gen, _ = evolve.build(tmp_profile, _cfg(tmp_profile), use_model=False)
    ok, why = evolve.promote(tmp_profile, gen, score=_fake_score(2, 3))
    assert not ok and "别急着定基线" in why
    assert evolve.active_id(tmp_profile) is None


def test_promote_first_generation_when_clean(tmp_profile):
    gen, _ = evolve.build(tmp_profile, _cfg(tmp_profile), use_model=False)
    ok, why = evolve.promote(tmp_profile, gen, score=_fake_score(3, 3))
    assert ok
    assert evolve.active_id(tmp_profile) == gen.id


def test_promote_rejects_regression(tmp_profile, monkeypatch):
    """新代通过数下降 → 拒绝上线。这是"持续学习"最常翻的车。"""
    g0, _ = evolve.build(tmp_profile, _cfg(tmp_profile), use_model=False)
    evolve.promote(tmp_profile, g0, score=_fake_score(3, 3))
    assert evolve.active_id(tmp_profile) == g0.id

    # 改手册 → 指纹变 → 能出第二代
    lesson = tmp_profile.root / "lessons" / "t.md"
    lesson.write_text(lesson.read_text(encoding="utf-8") + "\n## 禁忌 · 别乱点\n",
                      encoding="utf-8")
    g1, note = evolve.build(tmp_profile, _cfg(tmp_profile), use_model=False)
    assert g1 is not None, note

    ok, why = evolve.promote(tmp_profile, g1, score=_fake_score(2, 3))
    assert not ok and "跌" in why
    # 关键：拒绝之后 active 还是旧的那代
    assert evolve.active_id(tmp_profile) == g0.id
    assert g1.status == "rejected"


def test_promote_rejects_new_failures_even_if_count_same(tmp_profile):
    """通过数持平但挂了另一条 → 也是退化。只看总数会被"此消彼长"骗过去：
    学会认新图、同时把老图搞错，总数不变，但真实能力是"换了个错法"。
    """
    g0, _ = evolve.build(tmp_profile, _cfg(tmp_profile), use_model=False)
    evolve.promote(tmp_profile, g0, score=_fake_score(3, 3))

    lesson = tmp_profile.root / "lessons" / "t.md"
    lesson.write_text(lesson.read_text(encoding="utf-8") + "\n## 禁忌 · 别乱点\n",
                      encoding="utf-8")
    g1, _ = evolve.build(tmp_profile, _cfg(tmp_profile), use_model=False)
    # 总数 4、通过 3：通过数没变（3 == 3），但新增了一条挂掉的
    ok, why = evolve.promote(tmp_profile, g1, score=_fake_score(
        3, 4, fails=[{"file": "b.jpg"}]))
    assert not ok and "新增失败" in why


def test_promote_refuses_without_labels(tmp_profile):
    """没有 strict 标答 → 不上线。不能把"没有验证"当成"全过"。"""
    (tmp_profile.root / "labels.json").write_text(
        json.dumps({"items": []}), encoding="utf-8")
    gen, _ = evolve.build(tmp_profile, _cfg(tmp_profile), use_model=False)
    ok, why = evolve.promote(tmp_profile, gen, score=_fake_score(0, 0))
    assert not ok and "无法验证" in why


# ---------------- 6. 回滚 ----------------

def test_rollback_returns_to_previous(tmp_profile):
    """设备上没有 git，退不回去等于变砖 —— 所以这条必须一条命令可用。"""
    g0, _ = evolve.build(tmp_profile, _cfg(tmp_profile), use_model=False)
    evolve.promote(tmp_profile, g0, score=_fake_score(3, 3))

    lesson = tmp_profile.root / "lessons" / "t.md"
    lesson.write_text(lesson.read_text(encoding="utf-8") + "\n## 禁忌 · 别乱点\n",
                      encoding="utf-8")
    g1, _ = evolve.build(tmp_profile, _cfg(tmp_profile), use_model=False)
    evolve.promote(tmp_profile, g1, score=_fake_score(3, 3))
    assert evolve.active_id(tmp_profile) == g1.id

    ok, msg = evolve.rollback(tmp_profile)
    assert ok and evolve.active_id(tmp_profile) == g0.id
    assert "g0" not in msg          # 别把内部变量名漏到给人看的消息里


def test_rollback_on_baseline_says_so(tmp_profile):
    g0, _ = evolve.build(tmp_profile, _cfg(tmp_profile), use_model=False)
    evolve.promote(tmp_profile, g0, score=_fake_score(3, 3))
    ok, msg = evolve.rollback(tmp_profile)
    assert not ok and "没有上一代" in msg


def test_rollback_without_active(tmp_profile):
    ok, msg = evolve.rollback(tmp_profile)
    assert not ok and "没有 active" in msg


def test_promote_retires_previous(tmp_profile):
    """上线新代时旧代标 retired —— 否则 history 里全是 active，看不出谁在跑。"""
    g0, _ = evolve.build(tmp_profile, _cfg(tmp_profile), use_model=False)
    evolve.promote(tmp_profile, g0, score=_fake_score(3, 3))
    lesson = tmp_profile.root / "lessons" / "t.md"
    lesson.write_text(lesson.read_text(encoding="utf-8") + "\n## 禁忌 · 别乱点\n",
                      encoding="utf-8")
    g1, _ = evolve.build(tmp_profile, _cfg(tmp_profile), use_model=False)
    evolve.promote(tmp_profile, g1, score=_fake_score(3, 3))

    gens = {g.id: g for g in evolve.list_generations(tmp_profile)}
    assert gens[g0.id].status == "retired"
    assert gens[g1.id].status == "active"


def test_history_records_both_outcomes(tmp_profile):
    g0, _ = evolve.build(tmp_profile, _cfg(tmp_profile), use_model=False)
    evolve.promote(tmp_profile, g0, score=_fake_score(3, 3))
    lesson = tmp_profile.root / "lessons" / "t.md"
    lesson.write_text(lesson.read_text(encoding="utf-8") + "\n## 禁忌 · 别乱点\n",
                      encoding="utf-8")
    g1, _ = evolve.build(tmp_profile, _cfg(tmp_profile), use_model=False)
    evolve.promote(tmp_profile, g1, score=_fake_score(1, 3))
    actions = [r["action"] for r in evolve.history(tmp_profile)]
    assert actions == ["promote", "reject"]


# ---------------- 7. 装到设备后的接线 ----------------

def test_arbiter_uses_rules_when_fsm_idle(tmp_profile):
    """FSM 停手够久 → 规则接上，走 l05_rule 这一级。"""
    from autogame.decision.actions import ActionRegistry
    from autogame.decision.arbiter import Arbiter, Stage
    from autogame.config import ArbiterConfig

    rules_file = tmp_profile.root / "rules.yaml"
    rules_file.write_text("""rules:
  - id: r_pick
    when: {fact: pillars, op: ">=", value: 1}
    then: {action: pickup}
""", encoding="utf-8")

    cfg = ArbiterConfig(mode="enforce", rules_path=str(rules_file))
    arb = Arbiter(cfg, ActionRegistry(), KnowledgeBase(tmp_profile.root / "knowledge"))
    call = arb.tick({"gate_reason": "skillbar", "in_game": True, "pillars": 2,
                     "monsters": 1, "aframe": False, "ground_labels": 0,
                     "intent": "explore", "miss_streak": 0}, now=10.0, idle_s=2.0)
    assert call is not None and call.stage == Stage.RULE.value
    assert call.action_id == "pickup"
    assert arb.stats["rule_hits"] == 1


def test_arbiter_does_not_let_rules_take_over_running_fsm(tmp_profile):
    """FSM 刚干完活（idle_s 很小）→ 规则不许插手。

    这是"补位不夺权"的边界。规则写错时，如果它能长期抢过 FSM，整个角色行为
    会跑偏，而且很难从日志看出来是谁干的。
    """
    from autogame.decision.actions import ActionRegistry
    from autogame.decision.arbiter import Arbiter
    from autogame.config import ArbiterConfig

    rules_file = tmp_profile.root / "rules.yaml"
    rules_file.write_text("""rules:
  - id: r_pick
    when: {fact: pillars, op: ">=", value: 1}
    then: {action: pickup}
""", encoding="utf-8")

    cfg = ArbiterConfig(mode="enforce", rules_path=str(rules_file))
    arb = Arbiter(cfg, ActionRegistry(), KnowledgeBase(tmp_profile.root / "knowledge"))
    probe = {"gate_reason": "skillbar", "in_game": True, "pillars": 2,
             "monsters": 1, "aframe": False, "ground_labels": 0,
             "intent": "explore", "miss_streak": 0}
    call = arb.tick(probe, now=10.0, idle_s=0.2)
    assert call is None
    assert arb.stats["rule_hits"] == 0


def test_profile_injects_active_rules_path(tmp_profile):
    """换 profile 时规则路径必须跟着换。

    手写死路径会指向上一款游戏的规则 —— 那比没有规则更糟：它会在错误的画面上
    做出"有依据"的动作。
    """
    g0, _ = evolve.build(tmp_profile, _cfg(tmp_profile), use_model=False)
    evolve.promote(tmp_profile, g0, score=_fake_score(3, 3))
    cfg = load_config(None, profile=tmp_profile)
    assert cfg.decision.arbiter.rules_path.endswith(
        f"{g0.id}/rules.yaml".replace("/", str(Path("/"))))


# ---------------- 8. 自我纠正 ----------------

def test_collect_wanted_turns_logs_into_asks(tmp_profile, monkeypatch, tmp_path):
    """日志里的 hold/panic/L2 → 具体"请你补哪种材料"。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "supervisor.jsonl").write_text(
        json.dumps({"verdict": "panic", "reason": "画面完全陌生", "scene": "?"}) + "\n"
        + json.dumps({"verdict": "hold", "reason": "拿不准该不该点"}) + "\n",
        encoding="utf-8")
    doc = evolve.collect_wanted(tmp_profile, logs_dir=tmp_path / "logs")
    assert doc["signals"]["supervisor:panic"] == 1
    assert any("急停" in a for a in doc["asks"])
    assert any("反例图" in a for a in doc["asks"])
    assert (tmp_profile.root / "evolve" / "wanted.json").exists()


def test_collect_wanted_is_quiet_without_logs(tmp_profile, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    doc = evolve.collect_wanted(tmp_profile, logs_dir=tmp_path / "logs")
    assert doc["signals"] == {}
    assert not any("急停" in a for a in doc["asks"])


# ---------------- 9. 模型在环（规则草稿） ----------------

def test_model_draft_writes_valid_rules(tmp_profile, monkeypatch):
    """模型给的合法规则要落进 rules.yaml，并能在运行时求值。

    这是"生成针对这个游戏的代码"的正路：模型只产出**数据**（条件 → 动作），
    由引擎解释执行。它写不出任意代码，所以写坏了最多是"这条规则不生效"，
    而不是"设备变砖"。
    """
    from autogame.decision import llm_client

    payload = json.dumps({"rules": [
        {"id": "r_pick", "when": {"fact": "pillars", "op": ">=", "value": 1},
         "then": {"action": "pickup"}, "why": "有光柱就捡"},
    ]})
    monkeypatch.setattr(llm_client, "chat", lambda *a, **k: payload)

    gen, note = evolve.build(tmp_profile, _cfg(tmp_profile), use_model=True)
    assert gen is not None, note
    assert gen.rules_source == "model"
    doc = yaml.safe_load(gen.rules_path.read_text(encoding="utf-8"))
    assert doc["rules"][0]["id"] == "r_pick"

    # 落盘之后能被运行时加载并命中
    rs = evolve.RuleSet.load(gen.rules_path)
    hit = rs.decide({"pillars": 2, "in_game": True})
    assert hit and hit["action"] == "pickup"


def test_model_draft_rejects_bad_rules_and_stays_empty(tmp_profile, monkeypatch):
    """模型写了不存在的事实/动作 → 整份被拒，但**不阻断出代**。

    知识（knowledge/）那一半照样生效，所以这一代仍然有价值，只是没有硬规则。
    如果这里抛错，模型一抽风整个进化流程就断了 —— 那是把模型当成了必要条件。
    """
    from autogame.decision import llm_client

    payload = json.dumps({"rules": [
        {"id": "r_bad", "when": {"fact": "hp_frac", "op": "<", "value": 0.3},
         "then": {"action": "dodge"}},
    ]})
    monkeypatch.setattr(llm_client, "chat", lambda *a, **k: payload)

    gen, note = evolve.build(tmp_profile, _cfg(tmp_profile), use_model=True)
    assert gen is not None, note
    assert gen.rules_source == "none"
    doc = yaml.safe_load(gen.rules_path.read_text(encoding="utf-8"))
    assert doc["rules"] == []
    # 错误要落盘，人才能看到"模型写了什么被拒了"
    assert (gen.root / "rule_errors.txt").exists()


# ---------------- 9. 模型在环（规则草稿） ----------------

def test_model_draft_writes_valid_rules(tmp_profile, monkeypatch):
    """模型给的合法规则要落进 rules.yaml，并能在运行时求值。

    这是"生成针对这个游戏的代码"的正路：模型只产出**数据**（条件 → 动作），
    由引擎解释执行。它写不出任意代码，所以写坏了最多是"这条规则不生效"，
    而不是"设备变砖"。
    """
    from autogame.decision import llm_client

    payload = json.dumps({"rules": [
        {"id": "r_pick", "when": {"fact": "pillars", "op": ">=", "value": 1},
         "then": {"action": "pickup"}, "why": "有光柱就捡"},
    ]})
    monkeypatch.setattr(llm_client, "chat", lambda *a, **k: payload)

    gen, note = evolve.build(tmp_profile, _cfg(tmp_profile), use_model=True)
    assert gen is not None, note
    assert gen.rules_source == "model"
    doc = yaml.safe_load(gen.rules_path.read_text(encoding="utf-8"))
    assert doc["rules"][0]["id"] == "r_pick"

    # 落盘之后能被运行时加载并命中
    rs = evolve.RuleSet.load(gen.rules_path)
    hit = rs.decide({"pillars": 2, "in_game": True})
    assert hit and hit["action"] == "pickup"


def test_model_draft_rejects_bad_rules_and_stays_empty(tmp_profile, monkeypatch):
    """模型写了不存在的事实/动作 → 整份被拒，但**不阻断出代**。

    知识（knowledge/）那一半照样生效，所以这一代仍然有价值，只是没有硬规则。
    如果这里抛错，模型一抽风整个进化流程就断了 —— 那是把模型当成了必要条件。
    """
    from autogame.decision import llm_client

    payload = json.dumps({"rules": [
        {"id": "r_bad", "when": {"fact": "hp_frac", "op": "<", "value": 0.3},
         "then": {"action": "dodge"}},
    ]})
    monkeypatch.setattr(llm_client, "chat", lambda *a, **k: payload)

    gen, note = evolve.build(tmp_profile, _cfg(tmp_profile), use_model=True)
    assert gen is not None, note
    assert gen.rules_source == "none"
    doc = yaml.safe_load(gen.rules_path.read_text(encoding="utf-8"))
    assert doc["rules"] == []
    # 错误要落盘，人才能看到"模型写了什么被拒了"
    assert (gen.root / "rule_errors.txt").exists()
