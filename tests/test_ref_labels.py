"""参考图台账回归 —— 人工标答必须全对。

两本台账，都进 CI：
  1. `assets/ref/labels.json` —— 仓库级（沿用老结构，PC 1280 样本）
  2. `profiles/<id>/labels.json` —— 每个游戏自己的标答

这本台账是"用真实画面验证程序"的唯一可信来源：截图是人给的，答案是人写的。
strict 条目对不上就说明感知/决策退化了，直接红。

加样本的方式（不需要改这个文件）：
  1. 截图丢进 assets/ref/ 或 profiles/<id>/refs/
  2. 在该台账里加一条，写清楚场景和 expect（能确定的字段才写，其余 null）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import ref_check  # noqa: E402

LABELS = ROOT / "assets" / "ref" / "labels.json"
PROFILE_LEDGERS = sorted(
    p for p in (ROOT / "profiles").glob("*/labels.json")
    if not p.parent.name.startswith("_")
)


def test_ref_ledger_exists_and_has_samples():
    data = json.loads(LABELS.read_text(encoding="utf-8"))
    items = data.get("items", [])
    assert items, "台账不能是空的"
    strict = [it for it in items if it.get("strict")]
    assert strict, "至少要有 1 条 strict 条目，否则这套东西不会拦住任何回归"
    for it in items:
        assert (ROOT / it["file"]).exists(), f"台账里的图不存在: {it['file']}"


def test_strict_items_match(capsys):
    fails = ref_check.run(LABELS)
    out = capsys.readouterr().out
    assert fails == 0, f"{fails} 条 strict 标答不符：\n{out}"
    assert "GAP" in out or "已知缺口 0 条" in out


def test_profile_ledgers_have_samples():
    assert PROFILE_LEDGERS, "至少要有一个 profile 有自己的台账"
    for lp in PROFILE_LEDGERS:
        data = json.loads(lp.read_text(encoding="utf-8"))
        for it in data.get("items", []):
            # 台账里的路径相对 profile 目录解析
            assert (lp.parent / it["file"]).exists(), \
                f"{lp.parent.name}: 图不存在 {it['file']}"


@pytest.mark.parametrize("ledger", PROFILE_LEDGERS, ids=lambda p: p.parent.name)
def test_profile_strict_items_match(ledger, capsys):
    fails = ref_check.run(ledger, profile_id=ledger.parent.name)
    out = capsys.readouterr().out
    assert fails == 0, f"{ledger.parent.name} 有 {fails} 条 strict 标答不符：\n{out}"

