"""审查页生成 —— 把 frames_replay/*/manifest.json 汇总成一份离线 HTML。

特点：
  * 单文件、离线可看（图片用相对路径，双击即开，不需要起服务/填 token）；
  * 每张截图旁边就是**该帧的完整指令**（op/参数/协议文本）+ 意图 + 检出明细；
  * 叠加层用 DOM 按 manifest 坐标画（截图上不烧字，坐标能对着原始帧核对）；
  * 顶部筛选：有指令 / 无指令 / 违规 / 有光柱 / 有A框 / 门禁拦截。

用法：
  python tools/review_offline.py assets/frames_replay
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CSS = """
:root{--bg:#0e1114;--panel:#161a1f;--line:#242a31;--fg:#e6e9ee;--dim:#8b939e;
      --accent:#3b82f6;--warn:#d97706;--bad:#dc2626;--ok:#16a34a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:14px/1.55 "Segoe UI","Microsoft YaHei",system-ui,sans-serif}
header{position:sticky;top:0;z-index:20;background:#12161a;border-bottom:1px solid var(--line);
       padding:10px 18px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
header h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.3px}
.stat{color:var(--dim);font-size:13px}
.mode{margin:8px 0 0;padding:7px 10px;font-size:12.5px;line-height:1.5;
      color:#fbbf24;background:#2a2113;border:1px solid #4a3a15;border-radius:5px}
.mode code{color:#fde68a}
.stat b{color:var(--fg);font-weight:600}
.stat b.bad{color:var(--bad)}
.sep{width:1px;height:18px;background:var(--line)}
.filters{display:flex;gap:6px;flex-wrap:wrap;margin-left:auto}
.filters button{font:inherit;font-size:12.5px;padding:4px 11px;border-radius:999px;cursor:pointer;
       background:#1b2027;color:var(--dim);border:1px solid var(--line)}
.filters button.on{background:var(--accent);border-color:var(--accent);color:#fff}
nav.jump{display:flex;gap:14px;padding:10px 18px;background:#12161a;border-bottom:1px solid var(--line);
     font-size:12.5px;color:var(--dim);flex-wrap:wrap;align-items:center}
nav.jump a{color:var(--accent);text-decoration:none}
nav.jump a:hover{text-decoration:underline}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--dim)}
.legend i{display:inline-block;width:10px;height:10px;border:1.5px solid;border-radius:2px;
     margin-right:5px;vertical-align:-1px}
.legend .lp i{border-color:#f59e0b} .legend .la i{border-color:#22c55e}
.legend .lm i{border-color:#c026d3;border-radius:50%} .legend .ll i{border-color:#38bdf8}
.legend .lc i{border:0;background:#fff;width:9px;height:9px;transform:rotate(45deg);border-radius:0}
main{max-width:1500px;margin:0 auto;padding:18px}
section.video{margin-bottom:34px}
section.video>h2{font-size:14px;margin:0 0 4px;font-weight:600}
section.video .sub{color:var(--dim);font-size:12.5px;margin-bottom:12px;
       display:flex;gap:12px;flex-wrap:wrap}
.card{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,1fr);gap:14px;
      background:var(--panel);border:1px solid var(--line);border-radius:10px;
      padding:12px;margin-bottom:12px}
@media(max-width:1080px){.card{grid-template-columns:1fr}}
.shot{position:relative;background:#000;border-radius:7px;overflow:hidden;align-self:start;
      cursor:zoom-in;line-height:0}
.shot img{width:100%;display:block}
.ov{position:absolute;inset:0;pointer-events:none}
.ov i{position:absolute;border:2px solid;border-radius:2px;
      box-shadow:0 0 0 1px rgba(0,0,0,.7), inset 0 0 0 1px rgba(0,0,0,.5)}
.ov .p{border-color:#f59e0b}
.ov .p.st{box-shadow:0 0 0 1px rgba(245,158,11,.35)}
.ov .a{border-color:#22c55e}
.ov .m{border-color:#c026d3;opacity:.75}
.ov .l{border-color:#38bdf8;opacity:.6;border-width:1px}
.ov .r{border-color:#eab308}
.ov .c{position:absolute;width:16px;height:16px;margin:-8px 0 0 -8px;border:0;
      border-radius:50%;box-shadow:0 0 0 1.5px #fff,0 0 0 3px rgba(0,0,0,.75)}
.ov .c:before,.ov .c:after{content:"";position:absolute;background:#fff;
      box-shadow:0 0 2px #000}
.ov .c:before{left:7px;top:-2px;width:1.5px;height:20px}
.ov .c:after{top:7px;left:-2px;height:1.5px;width:20px}
.side{min-width:0;display:flex;flex-direction:column;gap:8px}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.tag{font-size:11.5px;padding:1.5px 8px;border-radius:6px;border:1px solid var(--line);
     color:var(--dim);background:#12161a;white-space:nowrap}
.tag.t{color:var(--fg);font-variant-numeric:tabular-nums}
.tag.on{color:#bfdbfe;border-color:#1e3a8a;background:#111c30}
.tag.off{color:#9ca3af}
.tag.bad{color:#fecaca;border-color:#7f1d1d;background:#1e1010}
.tag.warn{color:#fde68a;border-color:#78350f;background:#1d1607}
.intent{color:#cbd5e1;font-size:13px}
.intent .k{color:var(--dim)}
.det{color:var(--dim);font-size:12.5px;font-variant-numeric:tabular-nums}
.det b{color:#cbd5e1;font-weight:600}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{color:var(--dim);font-weight:500;text-align:left;padding:3px 8px;border-bottom:1px solid var(--line);
   font-size:11.5px;text-transform:uppercase;letter-spacing:.4px}
td{padding:4px 8px;border-bottom:1px solid #1c2127;vertical-align:top}
tr:last-child td{border-bottom:0}
code{font:12.5px/1.4 "Cascadia Mono",Consolas,monospace;color:#fcd34d}
code.k{color:#93c5fd}
td.n{color:var(--dim);width:22px;font-variant-numeric:tabular-nums}
.none{color:var(--dim);font-size:12.5px;font-style:normal;padding:6px 0}
.none b{color:#9ca3af;font-weight:500}
.viol{color:#fca5a5;font-size:12px}
#box{position:fixed;inset:0;background:rgba(0,0,0,.94);display:none;z-index:50;
     padding:24px;overflow:auto}
#box.on{display:block}
#box img{max-width:100%;display:block;margin:0 auto;border-radius:6px}
#box .close{position:fixed;top:14px;right:20px;color:#fff;font-size:22px;cursor:pointer;
     background:none;border:0}
"""

JS = """
const on = new Set(['all']);
function apply(){
  const cards = document.querySelectorAll('.card');
  let n=0;
  cards.forEach(c=>{
    const flags=(c.dataset.flags||'').split(' ').filter(Boolean);
    const show = on.has('all') || flags.some(f=>on.has(f));
    c.style.display = show ? '' : 'none';
    if(show) n++;
  });
  document.getElementById('shown').textContent = n;
  document.querySelectorAll('.filters button').forEach(b=>
    b.classList.toggle('on', on.has(b.dataset.f)));
}
document.querySelectorAll('.filters button').forEach(b=>{
  b.addEventListener('click',()=>{
    const f=b.dataset.f;
    if(f==='all'){ on.clear(); on.add('all'); }
    else { on.delete('all'); on.has(f)?on.delete(f):on.add(f); if(!on.size) on.add('all'); }
    apply();
  });
});
document.querySelectorAll('.shot').forEach(s=>{
  s.addEventListener('click',()=>{
    const img=s.querySelector('img');
    const b=document.getElementById('box');
    b.querySelector('img').src=img.src;
    b.classList.add('on');
  });
});
document.getElementById('box').addEventListener('click',e=>{
  if(e.target.id!=='box' && e.target.className!=='close') return;
  document.getElementById('box').classList.remove('on');
});
document.addEventListener('keydown',e=>{ if(e.key==='Escape')
  document.getElementById('box').classList.remove('on'); });
"""

OP_CN = {
    "mouse_click": "点击",
    "mouse_move": "移动光标",
    "key_tap": "按键",
    "key_down": "按下",
    "key_up": "抬起",
    "stop_all": "急停",
}


def _pct(v: float, total: float) -> str:
    return f"{max(0.0, min(100.0, v / total * 100)):.2f}%"


def _overlay(it: dict) -> str:
    fw, fh = it["frame_size"]
    parts: list[str] = []
    for p in it["pillars"]:
        left = _pct(p["cx"] - p["w"] / 2, fw)
        top = _pct(p["cy"] - p["h"] / 2, fh)
        w = _pct(p["w"], fw)
        h = _pct(p["h"], fh)
        cls = "p st" if p.get("stable") else "p"
        parts.append(f'<i class="{cls}" style="left:{left};top:{top};width:{w};height:{h}"></i>')
    for m in it["monsters"]:
        parts.append(
            f'<i class="m" style="left:{_pct(m["x"] - m["w"] / 2, fw)};'
            f'top:{_pct(m["y"] - m["h"] / 2, fh)};width:{_pct(m["w"], fw)};'
            f'height:{_pct(m["h"], fh)};border-radius:50%"></i>')
    for e in it["ground_labels"]:
        parts.append(
            f'<i class="l" style="left:{_pct(e["x"] - e["w"] / 2, fw)};'
            f'top:{_pct(e["y"] - e["h"] / 2, fh)};width:{_pct(e["w"], fw)};'
            f'height:{_pct(e["h"], fh)}"></i>')
    for e in it["item_list"]:
        parts.append(
            f'<i class="l" style="left:{_pct(e["x"] - e["w"] / 2, fw)};'
            f'top:{_pct(e["y"] - e["h"] / 2, fh)};width:{_pct(e["w"], fw)};'
            f'height:{_pct(e["h"], fh)}"></i>')
    a = it.get("aframe")
    if a:
        parts.append(
            f'<i class="a" style="left:{_pct(a["x"] - a["w"] / 2, fw)};'
            f'top:{_pct(a["y"] - a["h"] / 2, fh)};width:{_pct(a["w"], fw)};'
            f'height:{_pct(a["h"], fh)}"></i>')
    r = it.get("rechallenge")
    if r:
        parts.append(f'<i class="r" style="left:{_pct(r["x"] - 60, fw)};'
                     f'top:{_pct(r["y"] - 18, fh)};width:{_pct(120, fw)};'
                     f'height:{_pct(36, fh)}"></i>')
    for c in it["commands"]:
        if c.get("x") is not None and c.get("y") is not None and c["op"] != "mouse_move":
            parts.append(f'<i class="c" style="left:{_pct(c["x"], fw)};'
                         f'top:{_pct(c["y"], fh)}"></i>')
    return "".join(parts)


def _gate_suffix(it: dict) -> str:
    """门禁拦截原因（反例命中 / 未标定 / 其他）—— 让人一眼知道是"该拦"还是"没认出来"。"""
    reason = str(it.get("gate_reason", "") or "")
    if reason.startswith("negative:"):
        return "·反例"
    if reason == "unmatched":
        return "·未标定"
    if reason in ("", "disabled", "no_template"):
        return ""
    return f"·{reason}"


def _det_line(it: dict) -> str:
    bits = []
    if it["pillars"]:
        cols = {}
        for p in it["pillars"]:
            cols[p["color"]] = cols.get(p["color"], 0) + 1
        desc = " ".join(f"{k}×{v}" for k, v in cols.items())
        bits.append(f"光柱 <b>{len(it['pillars'])}</b>（{desc}）")
    if it.get("aframe"):
        a = it["aframe"]
        bits.append(f"A框 <b>({a['x']},{a['y']})</b>")
    if it["monsters"]:
        bits.append(f"怪物 <b>{len(it['monsters'])}</b>")
    if it["ground_labels"]:
        bits.append(f"地面标签 <b>{len(it['ground_labels'])}</b>")
    if it["item_list"]:
        bits.append(f"掉落列表 <b>{len(it['item_list'])}</b>")
    if it.get("rechallenge"):
        bits.append("再次挑战 <b>有</b>")
    if it.get("loading"):
        bits.append("加载/黑场")
    bits.append(f"帧差 <b>{it.get('frame_delta', 0):.1f}%</b>")
    return " · ".join(bits)


def _cmd_table(it: dict) -> str:
    if not it["commands"]:
        why = it.get("intent") or "本帧没有可执行动作"
        return f'<div class="none"><b>本帧无命令</b> —— {why}</div>'
    rows = []
    for i, c in enumerate(it["commands"], 1):
        args = []
        if c.get("key"):
            args.append(f'key={c["key"]}')
        if c.get("x") is not None:
            args.append(f'x={c["x"]}')
        if c.get("y") is not None:
            args.append(f'y={c["y"]}')
        if c["op"] in ("mouse_click",):
            args.append(f'button={c.get("button")}')
        if c["op"] in ("key_tap", "key_down"):
            args.append(f'hold={c.get("hold_ms")}ms')
        rows.append(
            f'<tr><td class="n">{i}</td><td>{OP_CN.get(c["op"], c["op"])}</td>'
            f'<td><code class="k">{", ".join(args) or "—"}</code></td>'
            f'<td><code>{c["text"]}</code></td></tr>')
    return ('<table><tr><th>#</th><th>操作</th><th>参数</th><th>完整指令</th></tr>'
            + "".join(rows) + "</table>")


def build(out_root: Path) -> Path:
    vids = []
    for d in sorted(out_root.iterdir()):
        mp = d / "manifest.json"
        if d.is_dir() and mp.exists():
            vids.append((d, json.loads(mp.read_text(encoding="utf-8"))))
    if not vids:
        raise SystemExit(f"{out_root} 下没有 manifest.json，先跑 tools/video_batch.py")

    tot = {"ticks": 0, "cmds": 0, "with": 0, "a": 0, "viol": 0}
    analysis_mode = False  # 有录像被用 --gate-unmatched pass 跑（"没认出来"也当战斗）
    body: list[str] = []
    jump: list[str] = []
    for vi, (d, man) in enumerate(vids, 1):
        s = man["summary"]
        tot["ticks"] += s["ticks"]
        tot["cmds"] += s["commands"]
        tot["with"] += s["ticks_with_commands"]
        tot["a"] += s["a_press"]
        tot["viol"] += len(s["violations"])
        analysis_mode = analysis_mode or s.get("gate_unmatched") == "pass"
        src = s.get("src", {})
        # 门禁原因分布：反例命中 / 未标定 必须一眼分得清，否则"为什么一帧命令都没有"
        # 只能靠猜（真实事故：门禁常开时反而看不出问题）。
        reasons: dict[str, int] = {}
        for it in man["items"]:
            if it["in_game"]:
                continue
            r = str(it.get("gate_reason", "") or "")
            key = ("反例" if r.startswith("negative:")
                   else "未标定" if r == "unmatched" else (r or "其他"))
            reasons[key] = reasons.get(key, 0) + 1
        gate_desc = " ".join(f"{k} {v}" for k, v in sorted(reasons.items()))
        jump.append(f'<a href="#v{vi}">{s["name"]}</a>')
        body.append(
            f'<section class="video" id="v{vi}"><h2>{s["name"]}</h2>'
            f'<div class="sub"><span>{src.get("size", ["?", "?"])[0]}×'
            f'{src.get("size", ["?", "?"])[1]} · {src.get("duration_s", 0)}s · '
            f'{s["hz"]}Hz 回放</span>'
            f'<span>{s["ticks"]} 帧</span><span>命令 {s["commands"]} 条</span>'
            f'<span>有指令 {s["ticks_with_commands"]} 帧</span>'
            f'<span>按A {s["a_press"]}</span>'
            f'<span>光柱帧 {s["ticks_with_pillars"]}</span>'
            f'<span>A框帧 {s["ticks_with_aframe"]}</span>'
            f'<span>门禁拦截 {s["ticks"] - s["gate_pass"]}'
            f'{("（" + gate_desc + "）") if gate_desc else ""}</span>'
            f'<span style="color:{"#fca5a5" if s["violations"] else "var(--dim)"}">'
            f'违规 {len(s["violations"])}</span></div>')
        for it in man["items"]:
            flags = ["all"]
            flags.append("cmd" if it["commands"] else "nocmd")
            if it.get("violations"):
                flags.append("viol")
            if it["pillars"]:
                flags.append("pillar")
            if it.get("aframe"):
                flags.append("aframe")
            if not it["in_game"]:
                flags.append("gated")
            img = f'{d.name}/frames/{it["image"]}'
            tag_gate = ('<span class="tag on">战斗</span>' if it["in_game"]
                        else f'<span class="tag off">UI拦截{_gate_suffix(it)}</span>')
            vtag = ""
            if it.get("violations"):
                vtag = ('<span class="tag bad">违规 ' + "；".join(it["violations"]) + "</span>")
            intent = (it.get("intent") or "").strip()
            body.append(
                f'<article class="card" data-flags="{" ".join(flags)}">'
                f'<div class="shot"><img loading="lazy" src="{img}" alt="{it["t"]}">'
                f'<div class="ov">{_overlay(it)}</div></div>'
                f'<div class="side">'
                f'<div class="row"><span class="tag t">{it["t"]}</span>'
                f'<span class="tag">#{it["ms"] // 1000}s</span>{tag_gate}'
                f'<span class="tag">{it.get("state", "")}</span>{vtag}</div>'
                f'<div class="intent"><span class="k">意图</span> '
                f'{intent if intent else "（本帧未产生动作意图）"}</div>'
                f'<div class="det">{_det_line(it)}</div>'
                f'{_cmd_table(it)}</div></article>')
        body.append("</section>")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>真实录像逐帧指令审查</title><style>{CSS}</style></head><body>
<header>
  <h1>真实录像 · 逐帧指令审查</h1>
  {('<p class="mode">注意：本页用了 <code>--gate-unmatched pass</code>（离线分析模式）——'
    '门禁"没认出来"的帧被当成战斗来跑，好让你看清感知层想干什么。'
    '生产默认是拦下（未标定不动手），别把这里的指令当成线上行为。</p>')
   if analysis_mode else ''}
  <span class="stat">帧 <b>{tot["ticks"]}</b></span>
  <span class="stat">有指令 <b>{tot["with"]}</b></span>
  <span class="stat">命令 <b>{tot["cmds"]}</b></span>
  <span class="stat">按A <b>{tot["a"]}</b></span>
  <span class="stat">违规 <b class="bad">{tot["viol"]}</b></span>
  <span class="sep"></span>
  <span class="stat">显示 <b id="shown">{tot["ticks"]}</b> 帧</span>
  <div class="filters">
    <button data-f="all" class="on">全部</button>
    <button data-f="cmd">有指令</button>
    <button data-f="nocmd">无指令</button>
    <button data-f="viol">违规</button>
    <button data-f="pillar">有光柱</button>
    <button data-f="aframe">有A框</button>
    <button data-f="gated">门禁拦截</button>
  </div>
</header>
<nav class="jump">
  <span>跳到</span>{"".join(jump)}
  <span class="legend" style="margin-left:auto">
    <span class="lp"><i></i>光柱</span>
    <span class="la"><i></i>A框</span>
    <span class="lm"><i></i>怪物</span>
    <span class="ll"><i></i>掉落标签/列表</span>
    <span class="lc"><i></i>本帧点击点</span>
  </span>
</nav>
<main>{"".join(body)}</main>
<div id="box"><button class="close">×</button><img src="" alt=""></div>
<script>{JS}</script></body></html>"""
    out = out_root / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"审查页 → {out}  （{tot['ticks']} 帧 / {tot['cmds']} 条命令）")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="汇总 manifest → 离线审查页")
    ap.add_argument("out_root", nargs="?", default=str(ROOT / "assets" / "frames_replay"))
    args = ap.parse_args()
    build(Path(args.out_root))


if __name__ == "__main__":
    main()
