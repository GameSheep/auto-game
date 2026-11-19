"""游戏适配包（profile）：一个游戏/一种画面规格一个目录，引擎代码不为游戏改动。

设计口径（与用户对齐过）：
    引擎只认"像素证据 + 分数阈值"，不认"游戏名"。所以每个游戏要补的不是代码，
    而是一份可复现的标定：哪个位置是战斗 HUD、哪些画面绝不能动手、目标物什么颜色
    什么形状、角色在屏幕哪里。这些全部落在 profiles/<id>/ 里。

目录约定：
    profiles/<id>/
        profile.yaml       # 元信息 + 对该游戏的 config 覆盖（与 config.yaml 同构）
        templates/         # 门禁模板 gate_pos_*.png / gate_neg_*.png（+ _index.json）
        labels.json        # 人工标答台账（tools/ref_check.py 逐字段核对）
        refs/              # 用户给的原始截图（只读，别改）
        inbox/             # 用户丢进来的新截图，等标定（tools/profile_init.py 扫它）

加载顺序（后者覆盖前者）：
    config.py 默认值 → config.yaml → profiles/<id>/profile.yaml 的 config 段
    → config.runtime.yaml（页面上的修改）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import paths

log = logging.getLogger(__name__)

ROOT = paths.code_root()
#: profile 是**数据**（用户的图、手册、模板），住在数据根；引擎升级不该动它。
PROFILES_DIR = paths.data("profiles")

# 基准分辨率：光柱/HUD 排除区等像素坐标都按它书写，运行时按实际帧等比缩放。
DEFAULT_REFERENCE_RESOLUTION = (1280, 720)


@dataclass
class Profile:
    """一个游戏适配包。"""

    id: str
    root: Path
    name: str = ""
    game: str = ""
    status: str = "uncalibrated"       # calibrated / partial / uncalibrated
    reference_resolution: tuple[int, int] = DEFAULT_REFERENCE_RESOLUTION
    config: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    # 可选的路径覆盖（相对 profile 目录解析）。用于"复用仓库现有 templates/ 与台账"。
    _templates_dir: str = ""
    _labels: str = ""

    # ---------- 约定路径 ----------
    @property
    def templates_dir(self) -> Path:
        if self._templates_dir:
            p = (self.root / self._templates_dir).resolve()
            if p.exists():
                return p
        return self.root / "templates"

    @property
    def labels_path(self) -> Path:
        if self._labels:
            p = (self.root / self._labels).resolve()
            if p.exists():
                return p
        return self.root / "labels.json"

    @property
    def refs_dir(self) -> Path:
        return self.root / "refs"

    @property
    def inbox_dir(self) -> Path:
        return self.root / "inbox"

    @property
    def is_calibrated(self) -> bool:
        return self.status == "calibrated"

    def resolve_rect(self, rect: list[int], frame_w: int, frame_h: int) -> tuple[float, float]:
        """把基准分辨率下的矩形换算到实际帧的缩放系数。

        用户标定时按截图原尺寸写像素（肉眼能画框、能核对），这里负责兼容别的分辨率。
        """
        rw, rh = self.reference_resolution
        return (frame_w / float(rw), frame_h / float(rh))

    def summary(self) -> str:
        pos = sorted(self.templates_dir.glob("gate_pos_*.png")) if self.templates_dir.exists() else []
        neg = sorted(self.templates_dir.glob("gate_neg_*.png")) if self.templates_dir.exists() else []
        n_labels = 0
        if self.labels_path.exists():
            try:
                data = yaml.safe_load(self.labels_path.read_text(encoding="utf-8")) or {}
                n_labels = len(data.get("items") or [])
            except Exception:  # noqa: BLE001
                pass
        rw, rh = self.reference_resolution
        return (f"{self.id}  [{self.status}]  {rw}x{rh}  "
                f"正例{len(pos)} 反例{len(neg)} 标答{n_labels}")


def profiles_root() -> Path:
    return PROFILES_DIR


def list_profiles() -> list[str]:
    """列出全部可用 profile id（按名字排序，跳过 _ 开头的脚手架模板）。"""
    if not PROFILES_DIR.exists():
        return []
    return sorted(p.name for p in PROFILES_DIR.iterdir()
                  if p.is_dir() and not p.name.startswith("_")
                  and (p / "profile.yaml").exists())


def active_profile_id() -> str:
    """这台机器当前玩哪个游戏 —— 数据根里的 `profile` 指针文件。

    为什么是指针文件而不是命令行参数：设备上服务由 systemd 拉起，
    改 unit 只为了换个游戏名，等于每次都要 sudo 编辑系统文件。
    一行文本改起来跟改配置一样便宜，而且写临时文件再 rename 就不会读到半截。
    """
    p = paths.data("profile")
    try:
        return p.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def load_profile(id_or_path: str | Path) -> Profile:
    """按 id 或路径加载 profile。找不到直接抛错——静默降级会让门禁悄悄失效。"""
    p = Path(id_or_path)
    root = p if p.is_dir() else (PROFILES_DIR / str(id_or_path))
    if not root.is_dir():
        avail = ", ".join(list_profiles()) or "（无）"
        raise FileNotFoundError(f"profile 不存在: {id_or_path}；可用: {avail}")
    yml = root / "profile.yaml"
    if not yml.exists():
        raise FileNotFoundError(f"profile 缺少 profile.yaml: {yml}")

    data = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
    rr = data.get("reference_resolution") or DEFAULT_REFERENCE_RESOLUTION
    try:
        ref_res = (int(rr[0]), int(rr[1]))
    except (TypeError, IndexError, ValueError):
        log.warning("profile %s 的 reference_resolution 非法（%r），回退默认", root.name, rr)
        ref_res = DEFAULT_REFERENCE_RESOLUTION

    return Profile(
        id=str(data.get("id") or root.name),
        root=root,
        name=str(data.get("name") or ""),
        game=str(data.get("game") or ""),
        status=str(data.get("status") or "uncalibrated"),
        reference_resolution=ref_res,
        config=dict(data.get("config") or {}),
        notes=list(data.get("notes") or []),
        _templates_dir=str(data.get("templates_dir") or ""),
        _labels=str(data.get("labels") or ""),
    )


def detect_profile_from_config(cfg_path: str | Path | None) -> Profile | None:
    """`--config profiles/x/profile.yaml` 这种写法也能识别出 profile。"""
    if not cfg_path:
        return None
    p = Path(cfg_path).resolve()
    if p.name == "profile.yaml" and p.parent.parent.name == "profiles":
        try:
            return load_profile(p.parent)
        except Exception:  # noqa: BLE001
            log.warning("从配置路径推断 profile 失败: %s", cfg_path, exc_info=True)
    return None
