"""可插拔 OCR —— 白名单物品名识别。

默认 NullOCR（纯颜色模式即可工作，见对话定稿：不需要识别到文字级别）。
可选引擎（对话明确提到两者）：
  - Tesseract：轻量（pip install pytesseract + 系统装 tesseract-ocr-chi-sim）
  - PaddleOCR：中文识别效果更好（pip install paddleocr paddlepaddle，较重）
安装后配置 vision.ocr: tesseract / paddle 即启用。
"""

from __future__ import annotations

import logging
import re

from ..config import VisionConfig

log = logging.getLogger(__name__)


class BaseOCR:
    """OCR 接口：输入 BGR 帧与目标区域，返回识别文本。"""

    name = "base"

    def read_text(self, frame_bgr, x: int, y: int, w: int, h: int) -> str:  # noqa: ANN001
        raise NotImplementedError


class NullOCR(BaseOCR):
    name = "null"

    def read_text(self, frame_bgr, x, y, w, h) -> str:  # noqa: ANN001, ARG002
        return ""


class TesseractOCR(BaseOCR):
    """Tesseract OCR（需要 pip install pytesseract + 系统安装 tesseract-ocr）。"""

    name = "tesseract"

    def __init__(self, lang: str = "chi_sim+eng") -> None:
        import pytesseract  # 延迟导入，缺失时给出明确报错

        self._t = pytesseract
        self._lang = lang
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "tesseract 未安装或不在 PATH：Linux 执行 `sudo apt install "
                "tesseract-ocr tesseract-ocr-chi-sim`"
            ) from e

    def read_text(self, frame_bgr, x, y, w, h) -> str:  # noqa: ANN001
        h_img, w_img = frame_bgr.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w_img, x + w), min(h_img, y + h)
        if x1 <= x0 or y1 <= y0:
            return ""
        roi = frame_bgr[y0:y1, x0:x1]
        # 放大2倍 + 灰度 + Otsu 二值化，显著提升小字体识别率
        roi = cv2_resize2x(roi)
        gray = cvt_gray(roi)
        _, bw = cvt_threshold(gray)
        try:
            text = self._t.image_to_string(bw, lang=self._lang, config="--psm 7")
        except Exception:  # noqa: BLE001
            return ""
        return text.strip()


class PaddleOCR(BaseOCR):
    """PaddleOCR（对话定稿：中文效果优于 Tesseract；较重，按需安装）。

    pip install paddlepaddle paddleocr
    模型首次运行自动下载到 ~/.paddleocr。
    """

    name = "paddle"

    def __init__(self, lang: str = "ch") -> None:
        from paddleocr import PaddleOCR as _PaddleOCR  # 延迟导入

        self._ocr = _PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
        log.info("PaddleOCR 已加载 (lang=%s)", lang)

    def read_text(self, frame_bgr, x, y, w, h) -> str:  # noqa: ANN001
        h_img, w_img = frame_bgr.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w_img, x + w), min(h_img, y + h)
        if x1 <= x0 or y1 <= y0:
            return ""
        roi = frame_bgr[y0:y1, x0:x1]
        try:
            result = self._ocr.ocr(roi, cls=True)
        except Exception:  # noqa: BLE001
            return ""
        # paddleocr 新旧版本返回结构不同，统一提取文本
        texts: list[str] = []
        for line_group in result or []:
            for line in line_group or []:
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    item = line[1]
                    if isinstance(item, (list, tuple)) and item:
                        texts.append(str(item[0]))
                    else:
                        texts.append(str(item))
        return " ".join(texts).strip()


# OpenCV 延迟引用（避免模块顶层依赖传染）
def cv2_resize2x(img):  # noqa: ANN001, ANN201
    import cv2

    h, w = img.shape[:2]
    return cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)


def cvt_gray(img):  # noqa: ANN001, ANN201
    import cv2

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def cvt_threshold(gray):  # noqa: ANN001, ANN201
    import cv2

    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


def fuzzy_contains(text: str, whitelist: list[str], threshold: int) -> str | None:
    """白名单模糊匹配：子串命中或 rapidfuzz 相似度达标。返回命中的白名单项。"""
    if not text:
        return None
    text_norm = re.sub(r"\s+", "", text)
    try:
        from rapidfuzz import fuzz  # 可选依赖

        for item in whitelist:
            if item in text_norm:
                return item
            if fuzz.partial_ratio(item, text_norm) >= threshold:
                return item
        return None
    except ImportError:
        for item in whitelist:
            if item in text_norm:
                return item
        # 无 rapidfuzz 时退化为简单字符重合率
        common = set(item) & set(text_norm)
        if common and len(common) / max(1, len(set(item))) * 100 >= threshold:
            return item
        return None


def build_ocr(cfg: VisionConfig) -> BaseOCR:
    if cfg.ocr == "tesseract":
        try:
            ocr = TesseractOCR()
            log.info("OCR 引擎: tesseract")
            return ocr
        except RuntimeError as e:
            log.warning("tesseract 不可用(%s)，降级为 NullOCR", e)
    elif cfg.ocr == "paddle":
        try:
            ocr = PaddleOCR()
            log.info("OCR 引擎: paddleocr")
            return ocr
        except Exception as e:  # noqa: BLE001
            log.warning("paddleocr 不可用(%s)，降级为 NullOCR（pip install paddleocr paddlepaddle）", e)
    return NullOCR()
