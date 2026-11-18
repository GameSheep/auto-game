"""场景门匹配打分 —— 与 `Detector._check_scene_gate` 同口径的唯一实现。

为什么单独拆一个模块
--------------------
有三个地方都要回答"这块模板在这张图上多少分"：

    autogame/vision/detector.py    生产门禁（真正决定放不放行）
    tools/make_template.py         人工裁模板时的验证
    autogame/teach_crop.py         从文档自动找框时的候选评分

如果各自抄一份 matchTemplate，迟早会漂移：生产 0.60 放行，工具算出 0.58 说
"不合格"，人就开始不信任何一个数字了。所以缩放链路、ROI 换算、降采样倍数、
匹配算法**只写在这里**，另外三处都调它。tests/test_gate_score.py 用真实截图
跑一遍 Detector，逐帧比对 diag["gate_best"]，防止有人手改了某一侧。
"""

from __future__ import annotations

import cv2
import numpy as np

# 模板存储基准宽（与 tools/make_template.py 的 BASE_WIDTH 同一个值）。
# 单独写在这里是为了让本模块不依赖 tools/ 下的脚本。
DEFAULT_BASE_WIDTH = 1280


def _to_work(frame: np.ndarray, max_detect_width: int) -> tuple[np.ndarray, float]:
    """按 max_detect_width 等比缩放，返回 (work, scale)。与 Detector.process 一致。"""
    h0, w0 = frame.shape[:2]
    if max_detect_width and w0 > max_detect_width:
        scale = max_detect_width / float(w0)
        return cv2.resize(frame, (int(round(w0 * scale)), int(round(h0 * scale))),
                          interpolation=cv2.INTER_AREA), scale
    return frame, 1.0


def match_best(frame: np.ndarray, tpl: np.ndarray, *,
               whole_frame: bool = False,
               roi: list[int] | None = None,
               base_width: int = DEFAULT_BASE_WIDTH,
               ref_res: tuple[int, int] | list[int] | None = None,
               max_detect_width: int = 0) -> float:
    """返回模板在帧中的最高匹配分（TM_CCOEFF_NORMED），-1.0 表示算不了。

    三种模式，对应生产上的三条路径：

    whole_frame=True   整帧匹配 → 反例模板走这条（`_check_negative_ui`）
    roi=[x,y,w,h]      只在 ROI 内匹配 → 正向模板走这条，坐标按**基准分辨率**书写，
                       内部按 ref_res → 实际帧换算（`_check_scene_gate` + `_ref_scale`）
    两者都不给         底部 65% 以下全宽 → 历史默认，只有 PC ARPG 对得上

    返回值舍到 4 位小数：3 位会让 0.5999 和 0.6004 都印成 0.60，人看不出
    差在哪；4 位刚好够分辨而不至于让报表全是噪声位。
    """
    if frame is None or tpl is None:
        return -1.0
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if tpl.ndim == 2:
        tpl = cv2.cvtColor(tpl, cv2.COLOR_GRAY2BGR)

    h0, w0 = frame.shape[:2]
    work, scale = _to_work(frame, max_detect_width)
    wh, ww = work.shape[:2]

    if whole_frame:
        rx, ry, rw, rh = 0, 0, ww, wh
    elif roi:
        rw_ref, rh_ref = (list(ref_res or (w0, h0)) + [w0, h0])[:2]
        fx, fy = w0 / float(rw_ref or w0), h0 / float(rh_ref or h0)
        rx = int(roi[0] * fx * scale)
        ry = int(roi[1] * fy * scale)
        rw = int(roi[2] * fx * scale)
        rh = int(roi[3] * fy * scale)
    else:
        # 与 Detector 的历史默认逐字一致（含它用 wh 算 rh 的写法）
        ry, rh = int(wh * 0.65), wh - int(wh * 0.65)
        rx, rw = 0, ww

    rx, ry = max(0, rx), max(0, ry)
    rw = min(rw, ww - rx)
    rh = min(rh, wh - ry)
    if rw <= 10 or rh <= 10:
        return -1.0

    roi_img = cv2.resize(work[ry:ry + rh, rx:rx + rw], None, fx=0.5, fy=0.5,
                         interpolation=cv2.INTER_AREA)
    if min(roi_img.shape[:2]) < 12:
        return -1.0

    # 模板按裁制基准宽缩放：实际帧宽 / 模板基准宽，再乘 work 缩放与降采样
    norm = (w0 / float(base_width or DEFAULT_BASE_WIDTH)) * scale * 0.5
    th, tw = tpl.shape[:2]
    tw_s = max(2, int(round(tw * norm)))
    th_s = max(2, int(round(th * norm)))
    if th_s >= roi_img.shape[0] or tw_s >= roi_img.shape[1]:
        return -1.0

    t_s = cv2.resize(tpl, (tw_s, th_s), interpolation=cv2.INTER_AREA)
    res = cv2.matchTemplate(cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY),
                            cv2.cvtColor(t_s, cv2.COLOR_BGR2GRAY),
                            cv2.TM_CCOEFF_NORMED)
    return round(float(cv2.minMaxLoc(res)[1]), 4)


def match_peaks(frame: np.ndarray, tpl: np.ndarray, *,
                base_width: int = DEFAULT_BASE_WIDTH,
                max_detect_width: int = 0,
                suppress: float = 0.6) -> tuple[float, float]:
    """整帧匹配，返回 (最高分, 第二峰分)。

    第二峰是"这块独不独特"的度量，也是人裁锚点时的第一条直觉：

        星图中间那 7 个发光节点很显眼，但每张星图上都有好几个长得一样的
        → 第二峰接近最高分 → 拿它当锚点，随便一张星图都会命中，等于没标。
        左上角「‹ 异界」标题块每张图只有一处
        → 第二峰远低于最高分 → 这块能当锚点。

    抑制半径按模板尺寸的 `suppress` 倍取：同一块 UI 的相邻位置分数天然接近，
    不抑制掉就会把"同一个东西"误当成"第二个相似的东西"。
    """
    if frame is None or tpl is None:
        return -1.0, -1.0
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if tpl.ndim == 2:
        tpl = cv2.cvtColor(tpl, cv2.COLOR_GRAY2BGR)

    h0, w0 = frame.shape[:2]
    work, scale = _to_work(frame, max_detect_width)
    if min(work.shape[:2]) < 12:
        return -1.0, -1.0
    full = cv2.resize(work, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    if min(full.shape[:2]) < 12:
        return -1.0, -1.0

    norm = (w0 / float(base_width or DEFAULT_BASE_WIDTH)) * scale * 0.5
    th, tw = tpl.shape[:2]
    tw_s = max(2, int(round(tw * norm)))
    th_s = max(2, int(round(th * norm)))
    if th_s >= full.shape[0] or tw_s >= full.shape[1]:
        return -1.0, -1.0

    t_s = cv2.resize(tpl, (tw_s, th_s), interpolation=cv2.INTER_AREA)
    res = cv2.matchTemplate(cv2.cvtColor(full, cv2.COLOR_BGR2GRAY),
                            cv2.cvtColor(t_s, cv2.COLOR_BGR2GRAY),
                            cv2.TM_CCOEFF_NORMED)
    _, best, _, best_loc = cv2.minMaxLoc(res)
    # 把最高峰附近抹掉再取第二峰
    m = res.copy()
    bx, by = best_loc
    pad_x = max(1, int(tw_s * suppress))
    pad_y = max(1, int(th_s * suppress))
    m[max(0, by - pad_y):by + pad_y + 1, max(0, bx - pad_x):bx + pad_x + 1] = -1.0
    second = float(m.max()) if m.size else -1.0
    return round(float(best), 4), round(second, 4)


def normalize_crop(frame: np.ndarray, rect: list[int],
                   base_width: int = DEFAULT_BASE_WIDTH) -> np.ndarray:
    """裁出 rect（x,y,w,h，左上原点）并缩放到统一基准宽。

    所有模板都按同一基准宽存储，运行时按"实际帧宽 / 模板基准宽"缩放，所以换
    分辨率不用重裁。这和 tools/make_template.py 的 crop_normalized 是同一件事，
    那份脚本传的是 x0,y0,x1,y1，这里统一收 x,y,w,h（与手册的 `- 框:` 同口径）。
    """
    if frame is None or rect is None:
        return np.zeros((0, 0, 3), np.uint8)
    h_img, w_img = frame.shape[:2]
    x, y, w, h = [int(v) for v in rect]
    x0 = max(0, min(x, w_img - 1))
    y0 = max(0, min(y, h_img - 1))
    x1 = max(x0 + 1, min(x + w, w_img))
    y1 = max(y0 + 1, min(y + h, h_img))
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((0, 0, 3), np.uint8)
    factor = (base_width or DEFAULT_BASE_WIDTH) / float(w_img)
    return cv2.resize(crop, (max(6, int(round(crop.shape[1] * factor))),
                             max(6, int(round(crop.shape[0] * factor)))),
                      interpolation=cv2.INTER_AREA)


def usable(tpl: np.ndarray, min_std: float = 8.0) -> bool:
    """模板能否用于 NCC：要有真实结构，且不是一条线。

    这是 detector._tpl_usable 的同义函数。低方差模板上 TM_CCOEFF_NORMED 会
    退化成"哪里都像"——历史教训 q_letter.png（图标的一个角落）对全部画面都给
    0.88，把门禁变成常开。自动找框时更要拦：算法很乐意框一块纯色天空交差。
    """
    if tpl is None or min(tpl.shape[:2]) < 6:
        return False
    gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY) if tpl.ndim == 3 else tpl
    return float(gray.std()) >= min_std
