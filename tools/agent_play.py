"""陪练：在没有真游戏的时候，看"游戏者"会怎么玩、以及它凭什么这么玩。

两件事：

    --check   把知识编译一遍，列出每个目标现在**做得到/做不到、缺什么**
    默认      在一个模拟世界里真跑一段，看动作序列、效果账本、以及有没有盲按

为什么要有"模拟世界陪练"这一档：真机上一次试错要几分钟（启动游戏、走到位置、
死一次重开），而这里 10 秒就能看清"它是不是在乱按"。真正上机前，错的应该是
知识，不是脚本。

用法：
    py tools/agent_play.py --profile tli_phone960 --check
    py tools/agent_play.py --profile tli_phone960 --seconds 12
    py tools/agent_play.py --profile tli_phone960 --seconds 12 --events
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autogame.agent import AgentKnowledge, GamePlayer, resolve_knowledge_dir  # noqa: E402
from autogame.config import load_config  # noqa: E402
from autogame.profile import PROFILES_DIR, load_profile  # noqa: E402
from autogame.protocol import AFrame, Pillar, SceneState  # noqa: E402
from autogame.sim.scene import SimWorld  # noqa: E402

_ICON = {"ok": "OK  ", "warn": "注意", "blocked": "缺口", "error": "错误"}


def _load(profile: str, config_path: str):
    cfg = load_config(config_path, profile=profile or None)
    if cfg.input is None:
        print("这个配置没有 input 段（操作方案）—— 先写 scheme.yaml，"
              "否则动作不知道按哪个键。")
        print("  py tools/learn_input.py --profile %s --preset tli_pc --ask" % (profile or "<id>"))
        raise SystemExit(2)
    root = None
    if profile:
        try:
            root = load_profile(profile).root
        except Exception:  # noqa: BLE001
            root = None
    return cfg, AgentKnowledge.load(resolve_knowledge_dir(cfg.agent, root))


def check(cfg, kn: AgentKnowledge) -> int:
    print(f"知识来源：{kn.source_dir}")
    if not kn:
        print("还没有目标/技能 —— 模板见 profiles/_template/agent/")
        return 0
    print(kn.summary(cfg.input))
    print()
    rep = kn.report(cfg.input)
    print(f"{'目标':<20}{'状态':<8}说明")
    print("-" * 76)
    for g in rep["goals"]:
        state = "可用" if g["usable"] else "做不到"
        print(f"{g['id']:<20}{state:<8}{g['reason'] or '、'.join(g['skills'])}")
    print()
    if rep["cues"]:
        print("判据（画面里「该不该按」的证据）：")
        for c in rep["cues"]:
            bad = "  [写错了]" if c["problem"] else ""
            print(f"  {c['describe']}{bad}")
        print()
    if rep["issues"]:
        for i in rep["issues"]:
            print(f"[{_ICON.get(i['level'], i['level'])}] {i['where']}: {i['message']}")
            if i.get("fix"):
                print(f"        → {i['fix']}")
    else:
        print("知识检查：没有问题")
    return 1 if any(i["level"] == "error" for i in rep["issues"]) else 0


def _scene(world: SimWorld, now: float) -> SceneState:
    """把模拟世界翻译成决策输入（不渲染、不识别，只走逻辑）。"""
    ch = (world.w // 2, world.h // 2)
    pillars: list[Pillar] = []
    for p in world.pillars:
        sx, sy = p.screen_pos(world.char_wx, world.char_wy, world.w, world.h)
        pillars.append(Pillar(x=sx, y=sy, cx=sx, cy=sy - 40, w=14, h=110,
                              color=p.color, area=1, distance=0.0))
    afr = None
    near = world._near_pillars()  # noqa: SLF001  (陪练工具，直接用内部状态)
    if near:
        sx, sy = near[0].screen_pos(world.char_wx, world.char_wy, world.w, world.h)
        afr = AFrame(x=sx - 14, y=sy - 40, w=28, h=28, score=0.9)
    # 模拟世界自己报"画面有没有在动"：角色朝航点走 / 有键按着 → 45%，否则 0。
    # 真实链路里这个数由视觉层算（像素差占比），这里不能写死 —— 写死会让
    # 走位的效果永远是"没变化"，然后账本天天报假故障。
    moving = bool(getattr(world, "_keys_down", None)) or (
        world.waypoint is not None
        and ((world.waypoint[0] - world.char_wx) ** 2
             + (world.waypoint[1] - world.char_wy) ** 2) ** 0.5 > 8)
    return SceneState(ts_ms=int(now * 1000) & 0xFFFFFFFF, w=world.w, h=world.h,
                      character=ch, pillars=pillars, aframe=afr,
                      in_game=not world.map_cleared,
                      frame_delta=45.0 if moving else 0.0)


def play(cfg, kn: AgentKnowledge, seconds: float, show_events: bool) -> int:
    world = SimWorld(width=int(getattr(cfg.sim, "width", 1280)),
                     height=int(getattr(cfg.sim, "height", 720)),
                     seed=2026, pillar_spawn_interval_s=1.6)
    player = GamePlayer(
        knowledge=kn, scheme=cfg.input, reach_px=cfg.agent.reach_px,
        dead_after=cfg.agent.dead_after,
        max_frame_age_ms=cfg.agent.max_frame_age_ms,
        sink=lambda c: world.on_command(op=c.op, key=c.key, button=c.button,
                                        x=c.x, y=c.y, hold_ms=c.hold_ms) or True,
        cues=kn.cues or None,
    )
    wants_frame = bool(kn.cues)
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        now = time.monotonic()
        world.tick(now)
        # 有判据才渲染：判据是画面证据，不给画面就永远不成立（这是对的，
        # 但陪练就成了"这条知识没法验证"）。渲染只在有 cues 时才付这份成本。
        frame = world.render() if wants_frame else None
        for e in player.on_scene(_scene(world, now), now, frame=frame):
            if show_events:
                print("  ", e)
        time.sleep(0.1)

    s = player.stats
    print()
    print(f"跑了 {seconds:.0f}s：动作 {s['actions']} 次，"
          f"捡到 {world.picked} 个，盲按 {world.blind_a_press} 次")
    print(f"  被证据挡下 {s['blocked_evidence']} 次"
          f"（其中没 A 框 {s['pickup_refused_no_aframe']} 次）"
          f"，能力缺失 {s['blocked_capability']} 次")
    print(f"  技能成功 {s['skill_ok']} / 失败 {s['skill_failed']} / 做不到 {s['skill_blocked']}")
    print("效果账本：", json.dumps(player.ledger.to_dict(), ensure_ascii=False))
    if s.get("cue_frames"):
        print(f"  判据读了 {s['cue_frames']} 帧（无画面 {s.get('cue_no_frame', 0)} 帧，"
              f"读失败 {s.get('cue_errors', 0)} 次，成立过 {len(player.summary()['cue_on'])} 条）")
    hy = player.hypotheses()
    if hy:
        print("待验证假设：")
        for h in hy:
            print(f"  - {h['hypothesis']}")
            print(f"    → {h['hint']}")
    if world.blind_a_press:
        print("\n!! 出现盲按：证据闸没生效，这是硬伤，先别上真机")
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="游戏者陪练 / 知识检查")
    ap.add_argument("--profile", default="", help="用哪个 profile 的知识")
    ap.add_argument("--config", default="config.yaml", help="配置文件")
    ap.add_argument("--check", action="store_true", help="只检查知识，不跑")
    ap.add_argument("--seconds", type=float, default=12.0, help="跑多久（秒）")
    ap.add_argument("--events", action="store_true", help="把每一帧发生的事打出来")
    args = ap.parse_args(argv)

    profile = args.profile
    if not profile:
        from autogame.doctor import _resolve_profile_id
        profile, note = _resolve_profile_id("")
        if note:
            print(note)
    cfg, kn = _load(profile, args.config)
    if args.check:
        return check(cfg, kn)
    if not kn:
        print(f"知识目录是空的：{kn.source_dir}")
        print("先写 goals.yaml / skills.yaml（模板：profiles/_template/agent/）")
        return 2
    bad = check(cfg, kn)
    if bad:
        print("\n知识有错，先改完再跑（错误不会抛异常，只会让它一步都不动）")
        return bad
    return play(cfg, kn, args.seconds, args.events)


if __name__ == "__main__":
    raise SystemExit(main())
