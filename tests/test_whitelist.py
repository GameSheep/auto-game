"""稀有度白名单（C3 铁律的"无 OCR"实现）回归测试。

对话定稿的核心铁律是"只捡白名单装备，绝不瞎按捡垃圾"。
原实现只支持 OCR 名称白名单，而 OCR 是可选依赖 —— 默认部署（ocr=null、whitelist 空）
时这条铁律实际是**关着的**：所有颜色的光柱都会被当成目标。

本文件锁死新增的 `vision.whitelist_colors`：按光柱颜色簇过滤，不需要任何 OCR 依赖。
"""

from __future__ import annotations

from autogame.config import VisionConfig
from autogame.sim.scene import SimPillar, SimWorld
from autogame.vision.pipeline import VisionPipeline


def _world_with_two_colors() -> SimWorld:
    world = SimWorld(seed=11)
    world.pillars.clear()
    # 一金一粉，横向分开（纵向相邻会被光柱轮廓合并成一个）
    world.pillars.append(SimPillar(60, 20, "gold"))
    world.pillars.append(SimPillar(-40, 30, "pink"))
    return world


def _process_twice(pipe: VisionPipeline, world: SimWorld):  # noqa: ANN001
    """光柱需要跨帧确认（beam_persistence_frames=2）。"""
    pipe.process(world.render(), 1000)
    return pipe.process(world.render(), 1300)


def test_no_whitelist_keeps_all_colors():
    """白名单为空 → 所有颜色都是候选（保持原行为）。"""
    cfg = VisionConfig()
    state = _process_twice(VisionPipeline(cfg), _world_with_two_colors())
    colors = {p.color for p in state.pillars}
    assert colors == {"gold", "pink"}, f"未配置白名单时不应过滤，实际 {colors}"


def test_color_whitelist_keeps_only_gold():
    """whitelist_colors=[gold] → 粉色光柱必须被过滤掉，金色标记为白名单命中。"""
    cfg = VisionConfig(whitelist_colors=["gold"])
    state = _process_twice(VisionPipeline(cfg), _world_with_two_colors())
    assert state.pillars, "金色光柱被误过滤"
    assert all(p.color == "gold" for p in state.pillars), \
        f"仍混入非白名单颜色: {[p.color for p in state.pillars]}"
    assert all(p.whitelisted for p in state.pillars), "白名单命中应标记 whitelisted"
    assert all(p.label == "gold" for p in state.pillars), "白名单命中应回填 label"


def test_color_whitelist_can_be_multiple():
    """多色白名单：金+粉都保留。"""
    cfg = VisionConfig(whitelist_colors=["gold", "pink"])
    state = _process_twice(VisionPipeline(cfg), _world_with_two_colors())
    assert {p.color for p in state.pillars} == {"gold", "pink"}


def test_color_whitelist_unknown_color_drops_all():
    """填了不存在的颜色 → 全部过滤（宁可不捡，也不瞎捡）。"""
    cfg = VisionConfig(whitelist_colors=["rainbow"])
    state = _process_twice(VisionPipeline(cfg), _world_with_two_colors())
    assert state.pillars == [], "未知颜色白名单应过滤掉全部光柱"
