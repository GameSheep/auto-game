"""读图写图 —— 绕开 `cv2.imread` 在中文路径上的坑。

`cv2.imread` 在 Windows 上走的是窄字符 API，路径里有中文就返回 None（不报错，
只是"读不出来"）。这一路上恰好**必然**出现中文：物品名就是中文，字条文件名就是
物品名，底图路径大概率在中文目录下。所以统一走 `np.fromfile + imdecode`，
它只吃字节，与编码无关。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread_any(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    p = Path(path)
    try:
        buf = np.fromfile(str(p), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def imwrite_any(path: str | Path, img: np.ndarray) -> bool:
    p = Path(path)
    ok, buf = cv2.imencode(p.suffix or ".png", img)
    if not ok:
        return False
    try:
        buf.tofile(str(p))
    except OSError:
        return False
    return True
