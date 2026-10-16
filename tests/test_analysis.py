"""颜色分析工具的单元测试。

这些测试只用手工构造的图，不依赖 ComfyUI 或真实照片，因此可离线运行。
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from chibitoy import analysis


def _solid_with_bands(hair_rgb, clothing_rgb, size=(100, 200)) -> Image.Image:
    """构造"窄头 + 宽身"的合成图：头用发色、身用服装色。

    **为什么不是简单的上下两块**：源照片的服装带现在按**解剖位置**取（见
    ``analysis.source_clothing_band``），需要"头部比肩窄"的形状才能定位头部下沿。
    均匀矩形没有肩部跳变，会退化成默认比例，四个取带就可能落到错误的色块上。
    """
    width, height = size
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    pixels = np.array(img)
    # 头占 40%：这是 Q 版公仔的比例，能让**四类取带**各自落在正确的色块内——
    # 发带 (0.02,0.30) 在头内、生成服装带 (0.38,0.68) 在身内、
    # 源解剖服装带（头下沿+0.02 起）也在身内。头设太小会让发带越界吃到身体。
    head_bottom = int(height * 0.40)
    head_w = max(2, width // 3)
    head_x0 = (width - head_w) // 2
    pixels[:head_bottom, head_x0:head_x0 + head_w, :3] = hair_rgb
    pixels[:head_bottom, head_x0:head_x0 + head_w, 3] = 255
    pixels[head_bottom:, :, :3] = clothing_rgb
    pixels[head_bottom:, :, 3] = 255
    return Image.fromarray(pixels, "RGBA")


def test_alpha_mask_uses_alpha_channel():
    img = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    img.putpixel((5, 5), (255, 0, 0, 255))
    mask = analysis.alpha_mask(img)
    assert mask.sum() == 1
    assert mask[5, 5]


def test_estimate_foreground_finds_subject_on_solid_background():
    img = Image.new("RGB", (40, 40), (0, 0, 255))          # 蓝底
    for x in range(10, 30):
        for y in range(10, 30):
            img.putpixel((x, y), (250, 220, 200))            # 肤色方块
    mask = analysis.estimate_foreground(img)
    assert mask[20, 20]
    assert not mask[2, 2]
    assert 0.2 < mask.mean() < 0.3


def test_band_mask_slices_by_relative_height():
    mask = np.ones((100, 10), dtype=bool)
    top = analysis.band_mask(mask, (0.0, 0.25))
    assert top[:25].all() and not top[25:].any()
    bottom = analysis.band_mask(mask, (0.8, 1.0))
    assert bottom[80:].all() and not bottom[:80].any()


def test_band_mask_respects_foreground_bounding_box():
    """分带应相对前景包围盒，而不是整张画布。"""
    mask = np.zeros((200, 10), dtype=bool)
    mask[50:150] = True            # 前景只占中间 100 行
    band = analysis.band_mask(mask, (0.0, 0.5))
    assert band[50:100].all()
    assert not band[:50].any() and not band[100:].any()


def test_dominant_colors_recovers_known_color():
    img = Image.new("RGB", (60, 60), (10, 20, 30))
    mask = np.ones((60, 60), dtype=bool)
    colors = analysis.dominant_colors(img, mask, top_n=1)
    assert len(colors) == 1
    rgb, share = colors[0]
    assert rgb == pytest.approx((10, 20, 30), abs=2)
    assert share == pytest.approx(1.0)


def test_dominant_colors_ranks_by_coverage():
    img = Image.new("RGB", (100, 100), (255, 0, 0))
    for x in range(10):
        for y in range(100):
            img.putpixel((x, y), (0, 255, 0))       # 10% 绿色
    colors = analysis.dominant_colors(img, np.ones((100, 100), dtype=bool), top_n=2)
    assert colors[0][0] == pytest.approx((255, 0, 0), abs=2)
    assert colors[0][1] > colors[1][1]
    assert colors[1][0] == pytest.approx((0, 255, 0), abs=2)


def test_dominant_colors_on_empty_mask_returns_empty():
    img = Image.new("RGB", (10, 10), (0, 0, 0))
    assert analysis.dominant_colors(img, np.zeros((10, 10), dtype=bool)) == []


def test_color_similarity_bounds():
    assert analysis.color_similarity((10, 20, 30), (10, 20, 30)) == pytest.approx(1.0)
    assert analysis.color_similarity((0, 0, 0), (255, 255, 255)) == pytest.approx(0.0)
    mid = analysis.color_similarity((0, 0, 0), (128, 0, 0))
    assert 0.0 < mid < 1.0


def test_analyze_pair_reports_similarity_for_matching_images():
    """源图与生成图配色一致时，相似度应接近 1。"""
    source = _solid_with_bands((20, 20, 20), (240, 240, 240))
    generated = _solid_with_bands((20, 20, 20), (240, 240, 240), size=(200, 400))
    result = analysis.analyze_pair(source, generated)
    assert result["hair_similarity"] == pytest.approx(1.0, abs=0.05)
    assert result["clothing_similarity"] == pytest.approx(1.0, abs=0.05)


def test_analyze_pair_detects_mismatched_clothing():
    """服装颜色明显不同时，相似度必须显著下降——这是本工具存在的意义。"""
    source = _solid_with_bands((20, 20, 20), (20, 20, 20))
    generated = _solid_with_bands((20, 20, 20), (250, 250, 250), size=(200, 400))
    result = analysis.analyze_pair(source, generated)
    assert result["clothing_similarity"] < 0.5
    assert result["hair_similarity"] == pytest.approx(1.0, abs=0.05)


# --- 解剖位置服装带（新增） --------------------------------------------------

def _head_then_body(head_h=140, head_w=100, body_w=700, body_h=700):
    """造一个"窄头 + 宽身"的前景掩码，模拟全身照。"""
    mask = np.zeros((1000, 1000), dtype=bool)
    mask[80:80 + head_h, 450:450 + head_w] = True          # 头
    mask[260:260 + body_h, 150:150 + body_w] = True        # 身
    return mask


def test_estimate_head_fraction_finds_the_shoulder_jump():
    mask = _head_then_body()
    fraction = analysis.estimate_head_fraction(mask)
    # 头部下沿在 y=260，主体从 y=80 起、到 y=960，故约 180/880 = 0.20
    assert 0.12 < fraction < 0.30, f"头部占比估计 {fraction} 不合理"


def test_source_clothing_band_targets_upper_body_not_shoes():
    """**核心断言**：源照片的服装带必须落在**上身**，不能落在腿/鞋上。

    固定带 (0.72, 1.00) 对头肩照是对的，但对全身照落在**腿和鞋**上 ——
    于是"服装主色相似度"变成"源图的鞋 vs 生成图的外套"。
    实测同一张图：比鞋得 0.257，比上衣得 0.529。
    """
    mask = _head_then_body()
    band = analysis.source_clothing_band(mask)
    assert band[0] < 0.5, f"服装带起点 {band[0]} 太低，会落到腿/鞋上"
    assert band[1] <= 1.0 and band[0] < band[1]


def test_source_clothing_band_is_below_the_head():
    mask = _head_then_body()
    head = analysis.estimate_head_fraction(mask)
    band = analysis.source_clothing_band(mask)
    assert band[0] >= head, "服装带不应与头部重叠"


def test_estimate_head_fraction_handles_degenerate_masks():
    assert analysis.estimate_head_fraction(np.zeros((10, 10), dtype=bool)) == 0.28
    assert analysis.estimate_head_fraction(np.ones((4, 4), dtype=bool)) == 0.28


def test_analyze_pair_uses_anatomical_source_band():
    """analyze_pair 必须使用解剖带，而不是固定的 SOURCE_CLOTHING_BAND。"""
    source = _solid_with_bands((20, 20, 20), (240, 240, 240))
    generated = _solid_with_bands((20, 20, 20), (240, 240, 240), size=(200, 400))
    pair = analysis.analyze_pair(source, generated)
    band = pair["source"].clothing_band
    assert band != analysis.SOURCE_CLOTHING_BAND or band[0] < 0.5, \
        "源照片服装带仍在使用固定的底部带"
