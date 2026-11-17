"""bilibili 关键词搜索 —— 找真实刷图录像（无需登录）。"""
from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"


def search(keyword: str, page: int = 1) -> list[dict]:
    q = urllib.parse.quote(keyword)
    url = (f"https://api.bilibili.com/x/web-interface/wbi/search/type?"
           f"search_type=video&keyword={q}&page={page}")
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Referer": "https://www.bilibili.com/"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read().decode("utf-8"))
    out = []
    for it in data.get("data", {}).get("result", []) or []:
        title = re.sub(r"<[^>]+>", "", it.get("title", ""))
        out.append({
            "bvid": it.get("bvid"),
            "title": title,
            "author": it.get("author"),
            "duration": it.get("duration"),      # "12:34"
            "play": it.get("play"),
            "pubdate": it.get("pubdate"),
            "desc": re.sub(r"<[^>]+>", "", it.get("description", ""))[:80],
        })
    return out


if __name__ == "__main__":
    kw = sys.argv[1] if len(sys.argv) > 1 else "暗黑4 刷图"
    for i, v in enumerate(search(kw), 1):
        print(f"{i:2d}. {v['bvid']}  {v['duration']:>7}  play={v['play']:>8}  {v['title'][:60]}")
