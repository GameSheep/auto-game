"""参考图台账（assets/ref/labels.json）回归 —— 人工标答必须全对。

这本台账是"用真实画面验证程序"的唯一可信来源：截图是人给的，答案是人写的。
strict 条目对不上就说明感知/决策退化了，直接红。

加样本的方式（不需要改这个文件）：
  1. 截图丢进 assets/ref/
  2. 在 labels.json 里加一条，写清楚场景和 expect（能确定的字段才写，其余 null）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import ref_check  # noqa: E402

LABELS = ROOT / "assets" / "ref" / "labels.json"


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
