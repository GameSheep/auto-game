"""操作方案学习向导 —— 把"这个游戏怎么操作"问清楚，写进 profile。

    # 看现在是什么（不写文件）
    py -3.12 tools/learn_input.py --profile tli_phone960 --print

    # 从预设起步，直接落盘
    py -3.12 tools/learn_input.py --profile tli_pc --preset tli_pc

    # 一项一项问（回车保留当前值，输入 - 清空，输入 ? 看说明）
    py -3.12 tools/learn_input.py --profile tli_phone960 --preset tli_pc --ask

    # 边按边学：切到游戏里按一下真实按键，向导抓键名
    py -3.12 tools/learn_input.py --profile tli_phone960 --ask --capture

为什么是"问"而不是"看录像自动推断"：
    自动推断看到的是"某段时间里 A 被按了 12 次"，分不清那是捡东西还是
    往左走 —— 而这个区别正是铁律要守的东西（按 A 要么捡，要么走位）。
    猜错的代价不是"少捡一件"，是"一边走一边捡空气"，而且日志全绿。
    所以按键这种确定性知识，问一次比推断一百次便宜。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autogame.input import (  # noqa: E402
    CAPABILITY_CN,
    PRESETS,
    Binding,
    InputScheme,
    MoveScheme,
    SkillSlot,
    normalize_key,
    preset,
)

#: 提问顺序 = 在实际游戏里"学一遍"的顺序：先走路，再打，再捡。
ASK_ORDER = ("move", "primary", "pickup", "interact", "cancel", "map", "skill")

_HINT = {
    "move": "走路方式。click=点地面（火炬/暗黑类）；wasd=方向键（魔兽/FPS）；stick=虚拟摇杆（手游）",
    "primary": "最主要的输出键。火炬是鼠标右键（按住连发），魔兽常是 1",
    "pickup": "捡东西的键。没有就填 -（比如魔兽是自动拾取/右键尸体）",
    "interact": "开箱子/进房/点 NPC 的键。火炬是 D，魔兽是右键点目标",
    "cancel": "关面板/返回。基本都是 ESC",
    "map": "打开地图/关卡面板。没有就填 -",
    "skill": "辅助技能，逗号分隔（q,w,e,r）。顺序会影响以后的循环，按你实际用得多的排",
}


def _load_existing(profile_dir: Path) -> InputScheme | None:
    p = profile_dir / "scheme.yaml"
    if not p.exists():
        return None
    import yaml

    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return InputScheme.model_validate(raw)


def _save(profile_dir: Path, scheme: InputScheme) -> Path:
    import yaml

    profile_dir.mkdir(parents=True, exist_ok=True)
    out = profile_dir / "scheme.yaml"
    header = (
        "# 操作方案（这个游戏怎么被操作）\n"
        "# 由 tools/learn_input.py 生成；也可以手改。\n"
        "#\n"
        "# 这里只写「哪个键是干什么的」。什么时候允许按由决策层的三道闸决定，\n"
        "# 不在这份文件里，也不能靠改这里放宽（拾取永远要过场景门 + A框 + 目标存在）。\n"
        "#\n"
        "# 详细说明：docs/操作方案.md\n"
        "\n")
    body = yaml.safe_dump(scheme.to_yaml_dict(), allow_unicode=True, sort_keys=False)
    out.write_text(header + body, encoding="utf-8")
    return out


# ---------------------------------------------------------------- 交互


def _capture_key(timeout: float = 6.0) -> str | None:
    """抓一次真实输入（键盘或鼠标）。装不上 pynput 就返回 None（退回手输）。

    键盘和鼠标要**同时听**：这款游戏的主攻是鼠标右键，只监听键盘的话，
    用户对着屏幕按半天右键，向导一点反应都没有。
    """
    try:
        from pynput import keyboard, mouse  # type: ignore
    except Exception:  # noqa: BLE001
        return None

    import threading

    got: list[str] = []
    fired = threading.Event()

    def _take(k: str) -> bool:
        k = normalize_key(k)
        if not k:
            return False
        got.append(k)
        fired.set()
        return True

    def on_press(key):
        name = getattr(key, "name", None)
        return not _take(name or getattr(key, "char", "") or "")

    def on_click(_x, _y, button, pressed):
        if pressed:
            return not _take(f"mouse_{getattr(button, 'name', 'left')}")
        return True

    kl = keyboard.Listener(on_press=on_press)
    ml = mouse.Listener(on_click=on_click)
    kl.start()
    ml.start()
    try:
        fired.wait(timeout)
    finally:
        kl.stop()
        ml.stop()
    return got[0] if got else None


def ask_move(base: MoveScheme) -> MoveScheme:
    cur = base.kind
    while True:
        v = input(f"走路方式 [click/wasd/stick/none]（回车={cur}，?=说明）：").strip().lower()
        if not v:
            v = cur
        if v == "?":
            print("  " + _HINT["move"])
            continue
        if v in ("click", "ground_click"):
            btn = input(f"  用哪个鼠标键点地面 [left/right]（回车={base.button.split('_')[-1]}）：").strip()
            return MoveScheme(kind="ground_click",
                              button=f"mouse_{btn or base.button.split('_')[-1]}")
        if v == "wasd":
            cam = input("  视角由鼠标控制吗？[y/N]（回车=N，即固定俯视）：").strip().lower()
            # 四个方向逐个问，不按"以前进键为基准推菱形"—— 改键玩家不一定守规矩。
            keys = {
                "forward": normalize_key(input("  前进键（回车=w）：").strip() or "w"),
                "back": normalize_key(input("  后退键（回车=s）：").strip() or "s"),
                "left": normalize_key(input("  左移键（回车=a）：").strip() or "a"),
                "right": normalize_key(input("  右移键（回车=d）：").strip() or "d"),
            }
            return MoveScheme(kind="wasd", keys=keys,
                              camera="mouse_look" if cam == "y" else "fixed")
        if v in ("stick", "none"):
            return MoveScheme(kind=v)  # type: ignore[arg-type]
        print("  只认 click / wasd / stick / none")


def ask_binding(cap: str, cur: Binding | None, capture: bool) -> Binding | None:
    label = CAPABILITY_CN[cap]
    shown = cur.label if cur else "（未学）"
    tip = "（直接按一下游戏里的键，或输入键名；- 清空）" if capture else "（键名，- 清空）"
    raw = input(f"{label} [{shown}] {tip}：").strip()
    if capture and not raw:
        got = _capture_key()
        if got:
            raw = got
            print(f"  抓到：{got}")
        else:
            print("  没抓到（pynput 不可用或超时），请直接输入键名")
            raw = input(f"  {label} 键名：").strip()
    if raw == "-":
        return None
    if not raw:
        return cur
    return Binding(key=raw, action=(cur.action if cur else "tap"))


def ask_skills(cur: list[SkillSlot], capture: bool) -> list[SkillSlot]:
    shown = ",".join(s.key for s in cur) or "（无）"
    raw = input(f"辅助技能（逗号分隔，回车保留）[{shown}]：").strip()
    if not raw:
        return cur
    if raw == "-":
        return []
    out: list[SkillSlot] = []
    for i, k in enumerate([x.strip() for x in raw.replace("，", ",").split(",") if x.strip()], 1):
        out.append(SkillSlot(id=f"技能{i}", key=k, action="tap"))
    return out


def interactive(base: InputScheme, capture: bool = False) -> InputScheme:
    print(f"\n当前方案：{base.describe()}")
    print("回车 = 保留当前值，- = 清空该项\n")
    move = ask_move(base.move)
    primary = ask_binding("primary", base.primary, capture)
    binds = dict(base.bindings)
    for cap in ("pickup", "interact", "cancel", "map"):
        b = ask_binding(cap, binds.get(cap), capture)
        if b is None:
            binds.pop(cap, None)
        else:
            binds[cap] = b
    skills = ask_skills(base.skills, capture)
    name = input(f"方案名（回车={base.name or '未命名'}）：").strip() or base.name
    return InputScheme(name=name, game=base.game, source="learned",
                       move=move, primary=primary, skills=skills, bindings=binds)


# ---------------------------------------------------------------- 展示


def _w(s: str) -> int:
    """终端显示宽度：中日韩字符占两格，`str.ljust` 只按字符数补，会错位。"""
    return sum(2 if "\u4e00" <= ch <= "\u9fff" or ch in "（）" else 1 for ch in s)


def _pad(s: str, width: int) -> str:
    return s + " " * max(0, width - _w(s))


def show(scheme: InputScheme) -> None:
    print("\n" + scheme.describe())
    print("\n" + _pad("能力", 10) + _pad("状态", 8) + _pad("按键", 16) + "不能用会怎样")
    print("-" * 78)
    mark = {"ready": "OK", "unknown": "未学", "partial": "部分", "blocked": "冲突",
            "unsupported": "不支持"}
    for row in scheme.capability_table():
        print(_pad(row["name"], 10) + _pad(mark.get(row["state"], row["state"]), 8)
              + _pad(row["binding"] or "-", 16) + (row["consequence"] or ""))
    errs = scheme.errors()
    warns = [c for c in scheme.conflicts() if c.severity == "warn"]
    if warns:
        print("\n注意：")
        for c in warns:
            print(f"  - {c.message}")
    if errs:
        print("\n冲突（必须改，否则不加载）：")
        for c in errs:
            print(f"  - {c.message}\n    → {c.fix}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="操作方案学习向导")
    ap.add_argument("--profile", required=True, help="profile id（写进 profiles/<id>/scheme.yaml）")
    ap.add_argument("--preset", choices=sorted(PRESETS), help="从内置预设起步")
    ap.add_argument("--ask", action="store_true", help="逐项提问（否则只写预设/现有方案）")
    ap.add_argument("--capture", action="store_true", help="按真实按键抓键名（需要 pynput）")
    ap.add_argument("--print", dest="only_print", action="store_true", help="只展示，不写文件")
    args = ap.parse_args(argv)

    pdir = ROOT / "profiles" / args.profile
    existing = _load_existing(pdir) if pdir.exists() else None

    if args.preset:
        base = preset(args.preset)
    elif existing is not None:
        base = existing
    else:
        # 空白起步：走位未知、什么都没学。这样程序不会"猜一个默认值"去按。
        base = InputScheme(name=args.profile, source="learned",
                           move=MoveScheme(kind="none"))
        print(f"（没有 {pdir/'scheme.yaml'}，也没给 --preset：从空白开始，走位先设为 none）")

    scheme = interactive(base, capture=args.capture) if args.ask else base
    show(scheme)

    if scheme.errors():
        print("\n没有写文件：上面的冲突解决后再来一次。")
        return 1
    if args.only_print:
        return 0
    out = _save(pdir, scheme)
    print(f"\n已写入 {out}")
    if not scheme.learned("move"):
        print("提示：走位还没学，程序不会让角色移动（其余动作仍可用）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
