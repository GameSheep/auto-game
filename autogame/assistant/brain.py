"""对话主流程：一句话进来，先本地、再模型、最后离线。

## 顺序就是设计

    本地命令（0ms，无模型）  →  模型的提案（秒级）  →  离线兜底（不需要网络）

三道**都失败**时它会说"我现在连不上模型"，而不是沉默或者假装答应了。
最后一段尤其重要：设备装在电视盒子旁边、网线拔了的时候，"谁在跑、跑到哪、
上一分钟干了什么"这些事实仍然答得出来，因为它们在本地。

## 一条硬规矩：模型只提案，不执行

模型返回的 `action` 只是**建议**。所有动作都要过一遍这里的白名单再执行：

| 模型能要的 | 谁来落实 |
|---|---|
| `pause` / `resume` / `observe` / `auto` | 模式管理器（和语音走同一个开关） |
| `patch`（改配置/手册/规则） | 先算 diff → 问你 → 你说"确认"才写 |
| `undo` | 补丁账本 |
| `command`（跑某个指令组） | `groups.route`，认不出就不跑 |
| `none` | 只是说话 |

模型要什么就执行什么，等于把"能不能停"交给了网络。所以白名单之外的一律
当作 `none`，并把"你想让我做什么"反问回去。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from . import localintent as li
from .context import RecentLog
from .dialog import Dialog
from .modes import Mode, ModeManager
from .patch import KIND_CONFIG, KIND_LESSON, KIND_RULE, PatchError, PatchStore, summarize

log = logging.getLogger(__name__)

#: 模型可以要的动作。**白名单之外的一律忽略**。
ACTIONS = ("none", "pause", "resume", "observe", "auto", "status",
           "patch", "undo", "command")

SYSTEM_PROMPT = """你是这台游戏机器人（RK3566 + OpenCV + RP2040）的操作助手。
你能看到"最近一分钟的现场记录"，能用中文回答，能提出改进方案。

## 你的位置
- 快决策（走路、打怪、捡东西）由确定性规则和视觉负责，**你不参与每一步**。
- 你负责：解释刚才发生了什么、听懂人的纠正、提出改进方案。
- 你**不能**直接动手。要停/继续请用 action；要改配置或规则请用 action=patch，
  系统会先给人看差异，人确认了才生效。

## 必须遵守
1. 只用下面的事实回答。记录里没有的事就说"记录里没有"，不要推测成事实。
2. 解释要指向具体时间点和具体证据（"t-12.4s 那一帧识别到…"）。
3. 拿不准或需要人补充时，用 question + options 提问，不要自己替他决定。
4. 改规则的方案要给具体数值和条件，不要写"优化一下"。

## 输出（严格 JSON，不要额外文字）
{
  "reply": "给人看的话（可以是几行）",
  "speak": "要念出来的一句（20 字以内，口语）",
  "question": "需要人回答时的问题，没有就 null",
  "options": ["选项一", "选项二"],
  "action": "none|pause|resume|observe|auto|status|patch|undo|command",
  "action_arg": {},
  "patch": {"kind": "config|lesson|rule", "payload": {...}} 或 null
}

patch 的 payload 形状：
- kind=config:  {"vision": {"whitelist_colors": ["gold"]}, ...}  只写要改的字段
- kind=lesson:  {"when": "什么情况下用得上", "do": "该怎么做"}
- kind=rule:    {"rule": {"id": "r_xxx", "when": {...}, "do": {...}}}
"""


def _json_from(text: str) -> dict:
    """模型输出 → dict。抠不出来就抛，让调用方走离线路径。"""
    from ..decision.llm_client import extract_json
    return extract_json(text)


class Assistant:
    """一个会话的完整门面。页面和语音都只跟它打交道。"""

    def __init__(self, cfg: Any, profile: Any = None, hub: Any = None,
                 *, modes: ModeManager | None = None,
                 recent: RecentLog | None = None,
                 config_path: str = "") -> None:
        self.cfg = cfg
        self.profile = profile
        self.hub = hub
        self.modes = modes or ModeManager()
        # 模式一变，副作用必须跟着走（停引擎/松按键/重新点亮），而且**到点自动
        # 恢复也算一次变化** —— 否则"暂停十分钟"到点了，画面上还是停着。
        self.modes.on_change = self._on_mode_change
        self.recent = recent or RecentLog(
            window_s=float(getattr(getattr(cfg, "assistant", None), "window_s", 60.0)))
        self.dialog = Dialog()
        # 没有 profile 也要能改**全局配置**：PatchStore 自己会处理账本落在哪。
        # 之前这里写成"没 profile 就没有 patches"，于是没绑定游戏的机器上
        # 连"把阈值调一下"都做不了，而这条恰恰是最不需要 profile 的。
        self.patches = PatchStore(profile, cfg, config_path=config_path)
        self._last_patch_id = ""
        self._acl = getattr(cfg, "assistant", None)

    # ================= 对外入口 =================

    def set_mode(self, name: str, *, seconds: float | None = None,
                 reason: str = "页面切换") -> dict:
        """切模式，并把该做的副作用一起做掉。

        **副作用才是重点**：切到不能动的模式时，必须同时停任务、停 FSM、
        松开按键；切回能动的模式时，必须把 FSM 从 PAUSED 点亮。只改一个枚举值
        等于什么都没做 —— 用户看到的仍然是"按了暂停它还在走"。
        """
        m = Mode(str(name))
        st = (self.modes.pause(seconds, reason=reason) if m is Mode.PAUSED
              else self.modes.set(m, reason=reason))
        self.recent.add("assistant", f"模式切到{m.label}", mode=m.value, reason=reason)
        return st

    def handle(self, text: str, *, source: str = "text", use_model: bool = True
               ) -> dict:
        text = str(text or "").strip()
        if not text:
            return self._out("", "你还没说话呢。", source="local")

        self.dialog.say("user", text, source=source)
        self.recent.add("assistant", f"用户说：{text}", source=source)

        # 1) 在等回答吗？（"确认""取消""第二个"）
        #    过期也算"在等过"：直接回"没有在等确认的事"会让人以为刚才那句
        #    确认白说了，而屏幕上明明还有一个问题框。要说清"过期了，重新问"。
        ans = self.dialog.resolve(text)
        if ans["matched"] or ans.get("kind") == "expired":
            return self._on_answer(text, ans, source=source)

        # 2) 本地命令（不等模型）
        it = li.parse(text)
        if it.is_command:
            return self._on_intent(text, it, source=source)

        # 3) 问模型；不可用则离线回答
        if use_model and self._model_ready():
            try:
                return self._ask_model(text, source=source)
            except Exception as e:  # noqa: BLE001
                log.warning("问模型失败，改走离线回答: %s", e)
                return self._offline(text, source=source, why=str(e))
        return self._offline(text, source=source)

    # ================= 回答上一句提问 =================

    def _on_answer(self, text: str, ans: dict, *, source: str) -> dict:
        p = ans.get("pending") or {}
        if ans["kind"] == "expired":
            return self._out(text, "刚才那个问题（" + str(p.get("question", "")) +
                            "）已经过去太久了，我不拿旧问题当准。你再说一遍要做什么，"
                            "我重新问。", source="local")

        if ans["kind"] == "cancel":
            self.dialog.clear_pending()
            return self._out(text, "好，这次不动。", source="local")

        if ans["kind"] == "bad_choice":
            opts = p.get("options") or []
            return self._out(text, f"没有第 {int(ans['option']) + 1} 个，现在有 "
                            f"{len(opts)} 个选项。", source="local")

        if ans["kind"] == "choose":
            opts = p.get("options") or []
            idx = int(ans["option"])
            payload = (p.get("payload") or {})
            choice_payload = ((payload.get("choices") or [{}])[idx]
                              if payload.get("choices") else payload)
            self.dialog.clear_pending()
            return self._do_choice(text, opts[idx] if idx < len(opts) else "",
                                   choice_payload, source=source)

        if ans["kind"] == "confirm":
            payload = p.get("payload") or {}
            self.dialog.clear_pending()
            if payload.get("patch"):
                return self._apply_pending_patch(text, payload, source=source)
            if payload.get("action"):
                res = self._apply_action(payload["action"],
                                         payload.get("action_arg") or {}, source=source)
                res["reply"] = ("好，已" + res.get("reply", "")) if res.get("ok") else res.get("reply", "")
                return res
            return self._out(text, "好。", source="local")

        # kind == "text"：开放问题的回答，继续交给模型理解
        if ans["kind"] == "text":
            self.dialog.clear_pending()
            if self._model_ready():
                try:
                    return self._ask_model(text, source=source, extra_hint=
                                           f"（用户在回答你的问题：{p.get('question')}）")
                except Exception:  # noqa: BLE001
                    pass
            return self._offline(text, source=source)
        return self._out(text, "嗯。", source="local")

    def _do_choice(self, text: str, option: str, payload: dict, *, source: str) -> dict:
        if payload.get("patch"):
            return self._apply_pending_patch(text, payload, source=source)
        if payload.get("action"):
            return self._apply_action(payload["action"], payload.get("action_arg") or {},
                                      source=source)
        return self._out(text, f"好，按「{option}」来。", source="local")

    # ================= 本地命令 =================

    def _on_intent(self, text: str, it: li.Intent, *, source: str) -> dict:
        k = it.kind
        if k == li.STOP:
            st = self.set_mode("paused", reason="语音/页面：停下")
            self.recent.add("assistant", "本地命令：停止", intent=k)
            return self._out(text, "已经停下，按键都松开了。", source="local",
                             intent=it, action={"kind": "stop"}, mode=st)

        if k == li.PAUSE:
            secs = it.arg
            st = self.set_mode("paused", seconds=secs, reason="语音/页面：暂停")
            msg = (f"暂停 {int(secs // 60)} 分钟，到点自己接着跑。"
                   if secs and secs >= 60 else
                   (f"暂停 {int(secs)} 秒。" if secs else "已经暂停。"))
            return self._out(text, msg, source="local", intent=it,
                             action={"kind": "pause", "seconds": secs}, mode=st)

        if k == li.RESUME:
            st = self.modes.resume(reason="语音/页面：继续")
            self._engine_start()
            return self._out(text, f"好，继续（{st['label']}）。", source="local",
                             intent=it, action={"kind": "resume"}, mode=st)

        if k == li.OBSERVE:
            st = self.set_mode("observe", reason="语音/页面：只观察")
            return self._out(text, "切到观察模式：还会看画面、会做判断，但一个动作都不发。",
                             source="local", intent=it,
                             action={"kind": "observe"}, mode=st)

        if k == li.AUTO:
            st = self.set_mode("auto", reason="语音/页面：自动")
            return self._out(text, "切到自动模式，按脚本自己跑。", source="local",
                             intent=it, action={"kind": "auto"}, mode=st)

        if k == li.UNDO:
            return self._undo(text, source=source, intent=it)

        if k == li.STATUS:
            return self._out(text, self._status_line(), source="local", intent=it,
                             action={"kind": "status"})

        if k in (li.CONFIRM, li.CANCEL):
            return self._out(text, "现在没有在等确认的事。", source="local", intent=it)

        # CHOOSE 但没有提问上下文
        if k == li.CHOOSE:
            return self._out(text, "现在没有在选项里挑的事。", source="local", intent=it)

        return self._offline(text, source=source)

    # ================= 模型 =================

    def _model_ready(self) -> bool:
        a = self._acl
        if a is None or not getattr(a, "enabled", True):
            return False
        return bool(getattr(a, "base_url", "")) and bool(getattr(a, "model", ""))

    def _ask_model(self, text: str, *, source: str, extra_hint: str = "") -> dict:
        from ..decision.llm_client import chat

        user = "\n".join([
            "## 最近一分钟的现场",
            self.recent.brief(
                seconds=float(getattr(self._acl, "window_s", 60.0)),
                limit=int(getattr(self._acl, "max_events", 40))),
            "",
            "## 当前状态",
            json.dumps(self._state_facts(), ensure_ascii=False, indent=1),
            "",
            "## 对话上文",
            self.dialog.history_text(limit=8),
            "",
            "## 用户这句话",
            extra_hint + text,
        ])
        raw = chat(self._acl, SYSTEM_PROMPT, user, temperature=0.2, json_mode=True)
        obj = _json_from(raw)
        return self._on_model_obj(text, obj, source=source)

    def _on_model_obj(self, text: str, obj: dict, *, source: str) -> dict:
        reply = str(obj.get("reply") or "").strip() or "（模型没说话）"
        speak = str(obj.get("speak") or "").strip()[:60] or reply[:40]
        action = str(obj.get("action") or "none").strip()
        arg = obj.get("action_arg") if isinstance(obj.get("action_arg"), dict) else {}
        question = str(obj.get("question") or "").strip()
        options = [str(o) for o in (obj.get("options") or [])][:6]
        patch = obj.get("patch") if isinstance(obj.get("patch"), dict) else None

        if action not in ACTIONS:
            log.warning("模型要了一个不在白名单里的动作：%r（忽略）", action)
            action, arg = "none", {}

        # 模型说要改东西 → 只算预览，不写
        if patch or action == "patch":
            return self._preview_patch(text, reply, speak, patch or arg, source=source)

        self.recent.add("assistant", f"模型回答：{reply[:60]}", source=source)

        out = self._out(text, reply, source="model")
        out["speak"] = speak

        if question:
            p = self.dialog.ask(question, options=options, kind="question",
                                payload={"action": action if action != "none" else "",
                                         "action_arg": arg})
            out["question"] = question
            out["pending"] = p.to_dict()
            out["speak"] = speak if not question else (speak or question)
            return out

        if action != "none":
            res = self._apply_action(action, arg, source=source)
            out.update({k: v for k, v in res.items() if k not in ("reply", "speak")})
            out["reply"] = reply + ("\n\n" + res["reply"] if res.get("reply") else "")
        return out

    # ================= 改规则 =================

    def _preview_patch(self, text: str, reply: str, speak: str, patch: dict,
                       *, source: str) -> dict:
        if self.patches is None or not getattr(self._acl, "allow_patch", True):
            return self._out(text, "我这边没有打开改规则的权限，你可以在设置页手动改。",
                             source="model")
        kind = str(patch.get("kind") or "").strip()
        payload = patch.get("payload") if isinstance(patch.get("payload"), dict) else {}
        if kind not in (KIND_CONFIG, KIND_LESSON, KIND_RULE):
            return self._out(text, "这个改动我没法落成一条配置或规则，需要你手动处理。",
                             source="model")
        try:
            pv = self.patches.preview(kind, payload, note=reply[:120], text=text)
        except PatchError as e:
            return self._out(text, f"改不了：{e}", source="model")

        says = summarize(pv)
        q = f"{says}（{pv['target']}）"
        p = self.dialog.ask(q, options=["确认", "取消"], kind="confirm",
                            payload={"patch": {"kind": kind, "payload": payload,
                                               "note": reply[:120], "text": text,
                                               "stamp": pv["stamp"]},
                                     "preview": {"diff": pv["diff"],
                                                 "target": pv["target"]}})
        out = self._out(text, reply or says, source="model")
        out.update({"speak": speak or says, "question": q,
                    "pending": p.to_dict(), "patch": pv})
        return out

    def _apply_pending_patch(self, text: str, payload: dict, *, source: str) -> dict:
        info = payload.get("patch") or {}
        if self.patches is None:
            return self._out(text, "没有可改的对象（没绑定 profile）。", source="local")
        try:
            rec = self.patches.apply(info.get("kind", ""), info.get("payload") or {},
                                     note=info.get("note", ""), text=info.get("text", ""),
                                     stamp=info.get("stamp", ""))
        except PatchError as e:
            return self._out(text, f"没写成：{e}", source="local")
        self._last_patch_id = rec["id"]

        # 写完文件不等于生效：规则是数据，进程不会自己知道它变了；
        # 配置在内存里还有一份。这两件事必须在同一句话里做完，
        # 否则用户听到的"改好了"只是磁盘上的文件改了。
        live = self.make_live(rec["kind"], info.get("payload") or {})
        msg = (f"改好了，编号 {rec['id']}，旧版本已经存起来。"
               "说「撤销」可以退回去。")
        if live:
            msg += " " + live
        self.recent.add("assistant", f"应用补丁 {rec['id']}（{rec['kind']}）")
        out = self._out(text, msg, source="local")
        out["applied"] = {"id": rec["id"], "kind": rec["kind"], "target": rec["target"],
                          "live": live}
        return out

    def make_live(self, kind: str, payload: dict) -> str:
        """把刚写下的改动推给正在跑的那套东西。返回一句人话（可为空）。

        - `rule`：`fsm.arbiter` 里的规则是启动时加载的，必须重载；
        - `config`：`cfg` 是内存对象，页面改配置走 `apply_patch`，这里必须一样；
          有些字段（端口、串口、后端）只能重启，如实说，不假装已生效。
        """
        if kind == KIND_RULE:
            hub = self.hub
            if hub is None or not hasattr(hub, "_reload_rules"):
                return "（没接引擎，重启后生效）"
            try:
                hub._reload_rules()
            except Exception as e:  # noqa: BLE001
                log.exception("重载规则失败")
                return f"（规则已写盘，但重载失败：{e}；重启后生效）"
            return "当前这一代规则已经当场重载。"

        if kind == KIND_CONFIG:
            from ..server.settings import ConfigError, apply_patch

            try:
                res = apply_patch(self.cfg, payload)
            except ConfigError as e:
                return f"（配置文件已写，但这套值不能在运行时套用：{e}）"
            except Exception as e:  # noqa: BLE001
                log.exception("热应用配置失败")
                return f"（配置已写盘，热应用失败：{e}；重启后生效）"
            restart = res.get("restart_required") or []
            if restart:
                return ("当场生效了 " + str(len(res.get("changed") or [])) +
                        " 项；" + "、".join(restart[:4]) + " 要重启服务才生效。")
            return "当场生效。"
        return ""

    def _undo(self, text: str, *, source: str, intent: Any = None) -> dict:
        if self.patches is None:
            return self._out(text, "没有可撤销的改动。", source="local", intent=intent)
        pid = self._last_patch_id
        if not pid:
            hist = [r for r in self.patches.history(limit=20)
                    if r.get("status") == "applied"]
            pid = hist[-1]["id"] if hist else ""
        if not pid:
            return self._out(text, "还没改过东西，没有可撤销的。", source="local")
        try:
            rec = self.patches.rollback(pid)
        except PatchError as e:
            return self._out(text, f"没退回去：{e}", source="local")
        self._last_patch_id = ""
        # 回滚同样有个"文件回去了、内存没回去"的问题。规则能当场重载；
        # 配置只写回了文件 —— 内存里那套要等重启，不能含糊过去。
        note = ""
        if rec.get("kind") == KIND_RULE:
            hub = self.hub
            if hub is not None and hasattr(hub, "_reload_rules"):
                try:
                    hub._reload_rules()
                    note = "规则已当场重载。"
                except Exception:  # noqa: BLE001
                    log.exception("回滚后重载规则失败")
                    note = "规则文件已退回，重载失败，重启后生效。"
        elif rec.get("kind") == KIND_CONFIG:
            note = "配置文件已退回，当前进程里还是改过的那套，重启服务后彻底恢复。"
        return self._out(text, f"已经退回到改之前（{pid}）。" + note, source="local",
                         intent=intent, action={"kind": "undo", "id": pid})

    # ================= 执行动作（白名单） =================

    def _apply_action(self, action: str, arg: dict, *, source: str) -> dict:
        action = str(action or "none")
        if action == "pause":
            st = self.set_mode("paused", seconds=arg.get("seconds"), reason="模型建议")
            return self._out("", f"已暂停（{st['label']}）。", source="local",
                             mode=st, action={"kind": "pause"})
        if action == "resume":
            st = self.modes.resume(reason="模型建议")
            self._engine_start()
            return self._out("", f"已恢复（{st['label']}）。", source="local",
                             mode=st, action={"kind": "resume"})
        if action == "observe":
            st = self.set_mode("observe", reason="模型建议")
            return self._out("", "已切到观察模式。", source="local", mode=st,
                             action={"kind": "observe"})
        if action == "auto":
            st = self.set_mode("auto", reason="模型建议")
            return self._out("", "已切到自动模式。", source="local", mode=st,
                             action={"kind": "auto"})
        if action == "status":
            return self._out("", self._status_line(), source="local",
                             action={"kind": "status"})
        if action == "undo":
            return self._undo("", source=source)
        if action == "command":
            return self._run_group(str(arg.get("text") or ""), source=source)
        return self._out("", "", source="local", action={"kind": action})

    def _run_group(self, text: str, *, source: str) -> dict:
        """喊一句就跑某个指令组。认不出就不跑 —— 和 `/api/command` 同一条规矩。"""
        if not text:
            return self._out("", "要跑哪个？说个组名。", source="local")
        if self.profile is None:
            return self._out("", "没有绑定 profile。", source="local")
        from .. import groups as gp
        world = gp.load_world(self.profile)
        r = gp.route(text, gp.load_groups(self.profile), world)
        if not r.ok:
            return self._out("", f"没认出「{text}」是哪一组。", source="local")
        if self.hub is not None:
            runner = gp.GroupRunner(r.group, world, self._tpl_dir(),
                                    executor=self.hub.executor, fsm=self.hub.fsm)
            self.hub.ui_task = runner
        return self._out("", f"开始跑「{r.group.name}」。", source="local",
                         action={"kind": "command", "group": r.group.name})

    # ================= 小工具 =================

    def _tpl_dir(self):
        from pathlib import Path
        if self.profile is not None and getattr(self.profile, "templates_dir", None):
            return Path(self.profile.templates_dir)
        return Path("templates")

    def _on_mode_change(self, old: Mode, new: Mode) -> None:
        """模式切换的副作用：该停的停、该点亮的点亮。

        观察和暂停的差别在这里：观察要**继续决策**（好让"当时想做什么"留得下来），
        只把引擎停掉的话，日志里就只剩"它什么都没想"。所以只有暂停才停 FSM，
        观察只松手、清任务。
        """
        if new.allows_action:
            self._engine_start()
            return
        self._release(stop_engine=new is Mode.PAUSED)

    def _hard_stop(self) -> None:
        """真的停：清任务 + 停 FSM + 保证没有按键还按着。

        这三件事必须一起做。只改模式是"以后不发动作"，而**已经按下去的键**
        不会自己弹起来 —— 用户看到的仍然是"它还在动"。
        """
        self._release(stop_engine=True)

    def _release(self, *, stop_engine: bool) -> None:
        hub = self.hub
        if hub is None:
            return
        try:
            hub.ui_task = None
            fsm = getattr(hub, "fsm", None)
            if stop_engine and fsm is not None:
                fsm.stop()
        except Exception:  # noqa: BLE001
            log.exception("停引擎失败")
        try:
            from ..protocol import Command
            ex = getattr(hub, "executor", None)
            if ex is not None:
                # stop_all 是唯一**绕过动作闸门**的命令（基类里专门放行）：
                # 闸门关的是"新的动作"，而这一条是"把已经按下的松开"。
                ex.execute(Command(op="stop_all"))
        except Exception:  # noqa: BLE001
            log.exception("释放按键失败")

    def _engine_start(self) -> None:
        hub = self.hub
        if hub is None:
            return
        fsm = getattr(hub, "fsm", None)
        try:
            from ..decision.fsm import State
            if fsm is not None and fsm.state == State.PAUSED:
                fsm.start()
        except Exception:  # noqa: BLE001
            log.debug("启动引擎失败", exc_info=True)

    def _state_facts(self) -> dict:
        facts: dict[str, Any] = {"模式": self.modes.snapshot()["label"],
                                 "模式原因": self.modes.snapshot()["reason"]}
        hub = self.hub
        if hub is not None:
            fsm = getattr(hub, "fsm", None)
            if fsm is not None:
                facts["引擎状态"] = getattr(fsm.state, "value", str(fsm.state))
                facts["引擎统计"] = {k: v for k, v in (fsm.stats or {}).items()
                                     if k in ("picks", "a_taps", "recoveries",
                                              "llm_calls", "decisions")}
                task = getattr(hub, "ui_task", None)
                if task is not None:
                    facts["正在跑的指令组"] = getattr(task, "name", "")
            h = getattr(hub, "health", None)
            if h is not None:
                hs = h.status()
                facts["体检"] = {
                    "结论": "正常" if hs["ok"] else f"连续 {hs['bad_streak']} 个窗口不对",
                    "发现": [f["label"] for f in hs["findings"]],
                    "基线就绪": hs["baseline_ready"],
                    "最近回滚": hs["rollback_reason"] or None,
                }
            exp = getattr(hub, "experience", None)
            if exp is not None:
                try:
                    es = exp.summary()
                    if es["pending"]:
                        facts["待定经验"] = {
                            "候选": es["pending"], "已够格": len(es["ready"]),
                            "说明": "够格的可以在对话页「攒下来的经验」里记进手册",
                        }
                    learned = exp.rules_learned()
                    if learned:
                        # 只给 id 和它治什么：模型要说"我现在会自己做什么"，
                        # 不能瞎编一条它没学会的规则。
                        facts["学会的规则"] = [
                            {"id": r["rule"].get("id"), "动作":
                             (r["rule"].get("then") or {}).get("action"),
                             "为什么": str(r["rule"].get("why") or "")[:60]}
                            for r in learned][:6]
                except Exception:  # noqa: BLE001
                    log.debug("读经验候选失败", exc_info=True)
        if self.profile is not None:
            try:
                from .. import evolve as ev
                g = ev.active_generation(self.profile)
                facts["配方"] = {"代": g.id if g else "无",
                                 "分数": (g.score if g else {})}
            except Exception:  # noqa: BLE001
                pass
        return facts

    def _status_line(self) -> str:
        s = self.modes.snapshot()
        parts = [f"现在是{s['label']}（{s['reason']}，{s['age_s']:.0f} 秒前切的）。"]
        hub = self.hub
        if hub is not None:
            fsm = getattr(hub, "fsm", None)
            if fsm is not None:
                parts.append(f"引擎在 {getattr(fsm.state, 'value', '?')}。")
                st = fsm.stats or {}
                parts.append(f"累计捡了 {st.get('picks', 0)} 个、按了 {st.get('a_taps', 0)} 次 A。")
            task = getattr(hub, "ui_task", None)
            if task is not None:
                parts.append(f"正在跑「{getattr(task, 'name', '')}」。")
            h = getattr(hub, "health", None)
            if h is not None:
                hs = h.status()
                if hs["findings"]:
                    parts.append("体检：" + "；".join(
                        f["detail"] for f in hs["findings"]) + "。")
                if hs["rollback_reason"]:
                    parts.append(f"最近因为「{hs['rollback_reason']}」自动退回过一版配方。")
            exp = getattr(hub, "experience", None)
            if exp is not None:
                try:
                    es = exp.summary()
                    if es["pending"]:
                        parts.append(f"攒了 {es['pending']} 条候选经验，"
                                     f"其中 {len(es['ready'])} 条够格写进手册了。")
                    learned = exp.rules_learned()
                    if learned:
                        parts.append(f"已经学会 {len(learned)} 条规则，"
                                     "下一次同样的局面它会自己换招。")
                except Exception:  # noqa: BLE001
                    log.debug("读经验候选失败", exc_info=True)
        n = len(self.recent.items(30))
        parts.append(f"最近 30 秒有 {n} 条记录。")
        return "".join(parts)

    def _out(self, text: str, reply: str, *, source: str, intent: Any = None,
             action: dict | None = None, mode: dict | None = None) -> dict:
        t = time.time()
        if reply:
            self.dialog.say("assistant", reply, source=source)
        out = {"ok": True, "ts": t, "text": text, "reply": reply, "speak": reply,
               "source": source, "mode": mode or self.modes.snapshot(),
               "intent": intent.to_dict() if intent is not None else None,
               "action": action, "pending": self.dialog.snapshot()["pending"],
               "patch": None}
        return out

    # ================= 离线兜底 =================

    def _offline(self, text: str, *, source: str, why: str = "") -> dict:
        """没有模型时的回答：只说本地事实，不猜。

        这是刻意的：设备断网、模型没配的时候，最坏的体验不是"它不会聊"，
        而是"它编得跟真的一样"。所以这里只复述记录，并把"怎么改规则"指清楚。
        """
        rows = self.recent.items(seconds=30.0)
        if rows:
            head = "最近 30 秒我这边记到 " + f"{len(rows)} 条：" + "；".join(
                f"{r['text']}" for r in rows[-3:])
        else:
            head = "最近 30 秒我什么都没记到 —— 可能还没开始跑，或者帧流断了。"
        tail = ("（现在连不上模型，" + (why[:60] if why else "没配置") +
                "。要改规则可以直接说「遇到 X 就 Y」，那个不需要模型。）")
        reply = head + " " + self._status_line() + " " + tail
        return self._out(text, reply, source="offline")
