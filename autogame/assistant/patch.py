"""改规则，不改代码 —— 自然语言落到"数据"，而且能退回去。

## 为什么不让模型直接改 Python

用户的原话是"说一句话，AI 就可靠地修改内部逻辑并生效"。这里最危险的不是
模型改错一行，而是**改对了却没验证、改坏了却回不去**。所以三件事：

1. 只动**数据**（配置、手册、规则），不动 `.py`。数据错了看得出来是哪一条，
   代码错了要读 diff 猜。这也正是 `evolve` 已经在做的事：配方是数据。
2. 每次修改都有**预览 → 校验 → 应用 → 可回滚**四步。校验不过就不写；
   写了就有原件，一条命令能退回去。
3. 预览和应用之间**核对同一份底稿**。模型给的方案你看了 3 分钟才点确认，
   这 3 分钟里材料可能被另一边改了 —— 那就要重看，而不是照着旧底稿硬写。

## 三种补丁

| 种类 | 改哪儿 | 校验 |
|---|---|---|
| `config` | `config.runtime.yaml`（页面同一条路） | pydantic 全量校验 |
| `lesson` | `lessons/*.md` 追加一条「流程」 | 重新解析，条目数不能变少 |
| `rule` | 当前这一代的 `rules.yaml` | `evolve.validate_rules`，原有规则不能少 |

`rule` 特意只改**当前代**：改了它就是改正在用的配方，所以它带回归风险，
应用时允许 `learner`/`evolve` 之后再叠一代把它验一遍。
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import yaml

from .. import evolve as ev
from ..atomic import atomic_write_json, atomic_write_text, file_lock

log = logging.getLogger(__name__)

KIND_CONFIG = "config"
KIND_LESSON = "lesson"
KIND_RULE = "rule"

KINDS = (KIND_CONFIG, KIND_LESSON, KIND_RULE)

#: 预览里最多显示多少行 diff（页面是给人看的，不是给 diff 工具看的）
MAX_DIFF_LINES = 200


class PatchError(ValueError):
    """补丁不可用（校验没过 / 底稿变了 / 找不到）。"""


def _sha(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:12]


def _diff(old: str, new: str, name: str) -> str:
    rows = list(difflib.unified_diff(
        (old or "").splitlines(), (new or "").splitlines(),
        fromfile=f"{name}（现在）", tofile=f"{name}（改后）", lineterm="", n=3))
    if len(rows) > MAX_DIFF_LINES:
        rows = rows[:MAX_DIFF_LINES] + [f"... 还有 {len(rows) - MAX_DIFF_LINES} 行"]
    return "\n".join(rows)


class PatchStore:
    """一个 profile 的补丁账本。"""

    def __init__(self, profile: Any, cfg: Any = None, config_path: str = "") -> None:
        self.profile = profile
        self.cfg = cfg
        self.config_path = config_path
        # 没有 profile 也要能改**全局配置**（页面上的设置就是这么改的）。
        # 账本跟着被改的文件走，不落在临时目录 —— 备份和正本必须同生共死。
        self.root = self._ledger_root()
        self.bak_dir = self.root / "bak"

    def _ledger_root(self) -> Path:
        if self.profile is not None:
            return Path(self.profile.root) / "patches"
        if self.config_path:
            return Path(self.config_path).resolve().parent / "patches"
        return Path("logs") / "patches"

    # ---------------- 路径与底稿 ----------------

    def target_of(self, kind: str) -> Path:
        if kind == KIND_CONFIG:
            from ..config import runtime_overlay_path
            return Path(runtime_overlay_path(self.config_path))
        if kind == KIND_LESSON:
            if self.profile is None:
                raise PatchError("没有绑定 profile，改不了手册。")
            d = Path(self.profile.root) / "lessons"
            files = sorted(p for p in d.glob("*.md")
                           if p.name.lower() not in ("readme.md", "index.md"))
            return files[0] if files else (d / "补充.md")
        if kind == KIND_RULE:
            if self.profile is None:
                raise PatchError("没有绑定 profile，改不了规则。")
            gen = ev.active_generation(self.profile)
            if gen is None:
                raise PatchError("还没有任何一代配方，规则没地方放。先学一遍再来改规则")
            return gen.rules_path
        raise PatchError(f"不认识的补丁种类: {kind}")

    def _read(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""

    # ---------------- 预览（不写任何东西） ----------------

    def preview(self, kind: str, payload: dict, *, note: str = "",
                text: str = "") -> dict:
        """算出"改完会变成什么样"，并跑校验。**不落盘、不改文件。**"""
        if kind not in KINDS:
            raise PatchError(f"不认识的补丁种类: {kind}")
        if not isinstance(payload, dict) or not payload:
            raise PatchError("补丁内容不能为空")

        target = self.target_of(kind)
        old = self._read(target)
        new, checks = self._build(kind, old, payload)

        ok = all(c["ok"] for c in checks)
        return {
            "ok": ok,
            "kind": kind,
            "target": str(target),
            "note": note,
            "text": text,
            "stamp": _sha(old),
            "diff": _diff(old, new, target.name),
            "checks": checks,
            "payload": payload,
            "file_exists": target.exists(),
        }

    # ---------------- 应用 ----------------

    def apply(self, kind: str, payload: dict, *, note: str = "", text: str = "",
              stamp: str = "", actor: str = "user") -> dict:
        """校验后写盘，并把原件存起来。`stamp` 对不上就拒绝 —— 底稿已经变了。"""
        pv = self.preview(kind, payload, note=note, text=text)
        if not pv["ok"]:
            bad = [c["name"] for c in pv["checks"] if not c["ok"]]
            raise PatchError("校验没过：" + "；".join(bad))
        if stamp and stamp != pv["stamp"]:
            raise PatchError("你要应用的这一版和预览时看到的已经不是同一份文件了"
                             "（这期间它被改过）。请重新看一眼再确认。")

        target = Path(pv["target"])
        old = self._read(target)
        new, _ = self._build(kind, old, payload)

        pid = f"p-{time.strftime('%m%d%H%M%S')}-{uuid.uuid4().hex[:4]}"
        self.bak_dir.mkdir(parents=True, exist_ok=True)
        bak = self.bak_dir / f"{pid}.txt"

        with file_lock(target):
            # 锁里再核一次底稿：预览到落盘之间有别的进程写过就放弃
            if _sha(self._read(target)) != pv["stamp"]:
                raise PatchError("文件在锁里被人改过了，这次不写。请重新预览。")
            if old:
                atomic_write_text(bak, old)
            atomic_write_text(target, new)

        rec = {"id": pid, "kind": kind, "target": str(target), "note": note or text,
               "payload": payload, "text": text, "actor": actor,
               "created": time.time(), "status": "applied",
               "backup": str(bak) if old else "",
               "diff": pv["diff"], "checks": pv["checks"],
               "before_sha": _sha(old), "after_sha": _sha(new)}
        atomic_write_json(self.root / f"{pid}.json", rec, indent=2)
        self._append_index(rec)
        log.info("补丁已应用 %s %s（%s）", pid, kind, note or text[:30])
        return rec

    # ---------------- 回滚 ----------------

    def rollback(self, pid: str) -> dict:
        p = self.root / f"{pid}.json"
        if not p.exists():
            raise PatchError(f"没有这个补丁: {pid}")
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("status") == "rolled_back":
            raise PatchError("这个补丁已经回滚过了")

        target = Path(rec["target"])
        bak = rec.get("backup") or ""
        with file_lock(target):
            cur = self._read(target)
            if rec.get("after_sha") and _sha(cur) != rec["after_sha"]:
                raise PatchError("这个文件在补丁之后又被改过，回滚会覆盖那些改动。"
                                 "先看清楚再决定。")
            atomic_write_text(target, self._read(Path(bak)) if bak else "")

        rec["status"] = "rolled_back"
        rec["rolled_back_at"] = time.time()
        atomic_write_json(p, rec, indent=2)
        self._append_index(rec)
        log.info("补丁已回滚 %s", pid)
        return rec

    # ---------------- 账本 ----------------

    def get(self, pid: str) -> dict | None:
        """按编号取一条补丁的完整记录（页面点开看 diff 用）。"""
        p = self.root / f"{pid}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def history(self, limit: int = 30) -> list[dict]:
        p = self.root / "history.jsonl"
        if not p.exists():
            return []
        rows = []
        for line in p.read_text(encoding="utf-8").splitlines()[-limit:]:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def _append_index(self, rec: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({k: rec.get(k) for k in
                                ("id", "kind", "note", "created", "status",
                                 "target")}, ensure_ascii=False) + "\n")

    # ---------------- 各类型的"改法 + 校验" ----------------

    def _build(self, kind: str, old: str, payload: dict) -> tuple[str, list[dict]]:
        if kind == KIND_CONFIG:
            return self._build_config(old, payload)
        if kind == KIND_LESSON:
            return self._build_lesson(old, payload)
        return self._build_rule(old, payload)

    # --- config ---

    def _build_config(self, old: str, payload: dict) -> tuple[str, list[dict]]:
        from ..server.settings import ConfigError, _deep_merge, public_config

        header = "# 页面改过的配置（自动生成，可删除以回滚到 config.yaml）\n"
        try:
            existing = yaml.safe_load(old) or {} if old.strip() else {}
        except Exception:  # noqa: BLE001
            existing = {}
        merged = _deep_merge(existing, payload)
        text = header + yaml.safe_dump(merged, allow_unicode=True, sort_keys=False)

        checks = []
        try:
            cfg = self.cfg
            if cfg is not None:
                base = public_config(cfg)
                from ..config import AppConfig
                AppConfig.model_validate(_deep_merge(base, merged))
                checks.append({"name": "配置能解析", "ok": True, "detail": "字段和取值都合法"})
            else:
                checks.append({"name": "配置能解析", "ok": True,
                               "detail": "没有传当前配置，跳过全量校验"})
        except ConfigError as e:  # pragma: no cover - 兼容旧签名
            checks.append({"name": "配置能解析", "ok": False, "detail": str(e)})
        except Exception as e:  # noqa: BLE001
            checks.append({"name": "配置能解析", "ok": False,
                           "detail": f"{type(e).__name__}: {e}"})
        return text, checks

    # --- lesson ---

    def _build_lesson(self, old: str, payload: dict) -> tuple[str, list[dict]]:
        from ..teach import append_note  # noqa: F401  （说明入口是同一条）

        when = str(payload.get("when") or "").strip()
        todo = str(payload.get("do") or payload.get("todo") or "").strip()
        checks = []
        if len(when) < 2 or len(todo) < 2:
            checks.append({"name": "两个问题都答了", "ok": False,
                           "detail": "「什么时候用得上」和「该怎么做」都要写"})
            return old, checks
        checks.append({"name": "两个问题都答了", "ok": True, "detail": "条件 + 动作"})

        stamp = time.strftime("%Y-%m-%d %H:%M")
        block = f"\n## 流程 · 补充 {stamp}\n- 一眼认出: {when}\n- 该做什么: {todo}\n"
        new = old.rstrip() + "\n" + block if old.strip() else "# 玩家补充\n" + block

        # 校验：重新解析一遍，条目数不能变少（写坏了立刻能看出来）。
        # 解析器吃的是文件，所以先落到临时文件 —— 预览阶段绝不碰真手册。
        try:
            from ..teach import parse_lessons

            def _claims(text: str) -> int:
                if not text.strip():
                    return 0
                fd, name = tempfile.mkstemp(suffix=".md")
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        f.write(text)
                    return len(parse_lessons(Path(name)).claims)
                finally:
                    Path(name).unlink(missing_ok=True)

            before, after = _claims(old), _claims(new)
            checks.append({"name": "手册还能解析", "ok": after >= before,
                           "detail": f"条目 {before} → {after}"})
        except Exception as e:  # noqa: BLE001
            checks.append({"name": "手册还能解析", "ok": False,
                           "detail": f"{type(e).__name__}: {e}"})
        return new, checks

    # --- rule ---

    def _build_rule(self, old: str, payload: dict) -> tuple[str, list[dict]]:
        rule = payload.get("rule") if isinstance(payload.get("rule"), dict) else payload
        try:
            doc = yaml.safe_load(old) or {} if old.strip() else {}
        except Exception as e:  # noqa: BLE001
            return old, [{"name": "现有规则能读", "ok": False,
                          "detail": f"{type(e).__name__}: {e}"}]
        if not isinstance(doc, dict):
            doc = {"version": 1, "howto": [], "rules": []}
        rules = doc.setdefault("rules", [])
        before_ids = {str(r.get("id") or "") for r in rules if isinstance(r, dict)}

        rid = str(rule.get("id") or "").strip()
        checks = [{"name": "规则有 id", "ok": bool(rid), "detail": rid or "缺 id"}]
        if rid and rid in before_ids:
            # 必须写回 doc：只重绑局部变量的话，doc["rules"] 还是原来那个列表，
            # 于是"覆盖"变成静默不生效 —— 校验还说"原有规则都在"，补丁账本也
            # 记成已应用，文件一个字没变。这类"报成功、没改动"最难查。
            doc["rules"] = [rule if str(r.get("id") or "") == rid else r for r in rules]
            checks.append({"name": "是覆盖已有规则", "ok": True, "detail": rid})
        else:
            rules.append(rule)
            checks.append({"name": "是新增规则", "ok": True, "detail": rid or "（无 id）"})

        text = yaml.safe_dump(doc, allow_unicode=True, sort_keys=False)
        try:
            ok_rules, errors = ev.validate_rules(yaml.safe_load(text))
            checks.append({"name": "规则能过校验", "ok": not errors,
                           "detail": "；".join(str(e) for e in errors[:3]) or
                                     f"{len(ok_rules)} 条"})
            after_ids = {str(r.get("id") or "") for r in ok_rules}
            lost = sorted(before_ids - after_ids)
            checks.append({"name": "原有规则没丢", "ok": not lost,
                           "detail": ("少了 " + "、".join(lost)) if lost
                                     else f"{len(before_ids)} 条都在"})
        except Exception as e:  # noqa: BLE001
            checks.append({"name": "规则能过校验", "ok": False,
                           "detail": f"{type(e).__name__}: {e}"})
        return text, checks


def summarize(pv: dict) -> str:
    """把预览结果说成人话（语音/页面上直接用）。"""
    if not pv.get("ok"):
        bad = [c for c in pv.get("checks", []) if not c["ok"]]
        return "这版还不行：" + "；".join(f"{c['name']}（{c['detail']}）" for c in bad)
    adds = sum(1 for ln in (pv.get("diff") or "").splitlines()
               if ln.startswith("+") and not ln.startswith("+++"))
    dels = sum(1 for ln in (pv.get("diff") or "").splitlines()
               if ln.startswith("-") and not ln.startswith("---"))
    return f"改完了：新增 {adds} 行、去掉 {dels} 行。要现在应用吗？"
