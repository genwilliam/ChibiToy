from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

RGB = tuple[int, int, int]

# 分带默认值 (相对前景包围盒高度的百分比)
# 源照片通常是"头肩构图"：头发在上部，服装压在最底部
SOURCE_HAIR_BAND = (0.02, 0.30)
SOURCE_CLOTHING_BAND = (0.72, 1.00)
# 生成的 Q 版公仔是"全身构图"：头部约占上 1/3，服装在中段
GENERATED_HAIR_BAND = (0.02, 0.30)
GENERATED_CLOTHING_BAND = (0.38, 0.68)


@dataclass
class ColorReport:
    # 图的颜色分析结果
    path: str
    foreground_ratio: float # 前景占整图比例
    hair_band: tuple[float, float]
    clothing_band: tuple[float, float]
    hair_colors: list[tuple[RGB, float]] = field(default_factory=list)
    clothing_colors: list[tuple[RGB, float]] = field(default_factory=list)
    hair_dominant: RGB | None = None
    clothing_dominant: RGB | None = None


def estimate_foreground(image: Image.Image, tolerance: int = 45) -> np.ndarray:
    # 用边框主色即背景估计前景掩码（源照片没有 alpha 时使用）
    arr = np.asarray(image.convert("RGB")).astype(np.int16)
    h, w, _ = arr.shape
    border = np.concatenate([
        arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :],
    ])
    background = np.median(border, axis=0)
    distance = np.abs(arr - background).sum(axis=2)
    return distance > tolerance * 3


def alpha_mask(image: Image.Image) -> np.ndarray:
    # 取 RGBA 图的 alpha 通道作为前景掩码（生成图使用）
    if image.mode in ("RGBA", "LA"):
        alpha = np.asarray(image.convert("RGBA").getchannel("A"))
        return alpha > 128
    return np.ones((image.height, image.width), dtype=bool)


def dominant_colors(image: Image.Image, mask: np.ndarray, top_n: int = 3) -> list[tuple[RGB, float]]:
    # 返回掩码区域内出现最多的前 ``top_n`` 种颜色及其占比
    # 每通道量化到 4bit（16 级），以抵抗生成图像的细微噪声
    arr = np.asarray(image.convert("RGB"))
    pixels = arr[mask]
    if pixels.size == 0:
        return []
    quantised = (pixels // 16).astype(np.int32)
    keys = quantised[:, 0] * 256 + quantised[:, 1] * 16 + quantised[:, 2]
    values, counts = np.unique(keys, return_counts=True)
    order = np.argsort(-counts)[:top_n]
    total = counts.sum()
    results: list[tuple[RGB, float]] = []
    for index in order:
        key = int(values[index])
        # 用该簇内像素的真实均值
        mean = pixels[keys == key].mean(axis=0)
        results.append((tuple(int(v) for v in mean), float(counts[index] / total)))
    return results


def estimate_head_fraction(mask: np.ndarray) -> float:
    # 从前景掩码估计头部占主体高度的比例（0~1）
    if mask.ndim != 2 or not mask.any():
        return 0.28
    ys, _ = np.nonzero(mask)
    top, bottom = int(ys.min()), int(ys.max()) + 1
    subject_h = bottom - top
    if subject_h < 8:
        return 0.28
    widths = mask[top:bottom].sum(axis=1)
    head_ref = widths[: max(1, subject_h // 10)]
    head_ref = head_ref[head_ref > 0]
    if not head_ref.size:
        return 0.28
    threshold = float(np.median(head_ref)) * 1.8
    for i in range(max(1, int(subject_h * 0.05)), subject_h):
        if widths[i] > threshold:
            return i / subject_h
    return 0.28


def source_hair_band(mask: np.ndarray, head_fraction: float | None = None) -> tuple[float, float]:
    # 给出源照片的发色取带
    head = estimate_head_fraction(mask) if head_fraction is None else head_fraction
    top = 0.02
    bottom = min(0.30, max(0.10, head * 0.65))
    return (top, round(bottom, 3))


def source_clothing_band(mask: np.ndarray, thickness: float = 0.25) -> tuple[float, float]:
    # 给出源照片的上身衣物取带
    head = estimate_head_fraction(mask)
    top = min(0.95, head + 0.02)
    return (round(top, 3), round(min(1.0, top + thickness), 3))


def band_mask(mask: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    # 取前景包围盒内垂直方向 ``[y0, y1)`` 比例区间的掩码部分
    rows = np.where(mask.any(axis=1))[0]
    if rows.size == 0:
        return np.zeros_like(mask)
    top, bottom = rows[0], rows[-1]
    height = bottom - top + 1
    y0 = top + int(height * band[0])
    y1 = top + int(height * band[1])
    out = np.zeros_like(mask)
    out[max(0, y0):max(0, y1)] = mask[max(0, y0):max(0, y1)]
    return out


def analyze(image: Image.Image, path: str, hair_band, clothing_band) -> ColorReport:
    # 对单张图做分带主色分析
    mask = alpha_mask(image) if image.mode in ("RGBA", "LA") else estimate_foreground(image)
    report = ColorReport(
        path=path,
        foreground_ratio=float(mask.mean()),
        hair_band=hair_band,
        clothing_band=clothing_band,
    )
    report.hair_colors = dominant_colors(image, band_mask(mask, hair_band))
    report.clothing_colors = dominant_colors(image, band_mask(mask, clothing_band))
    report.hair_dominant = report.hair_colors[0][0] if report.hair_colors else None
    report.clothing_dominant = report.clothing_colors[0][0] if report.clothing_colors else None
    return report


def color_similarity(a: RGB, b: RGB) -> float:
    # 归一化 RGB 相似度（1.0 = 完全相同）。用于主色还原度的客观近似
    distance = float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))
    return max(0.0, 1.0 - distance / (255.0 * np.sqrt(3)))


def analyze_pair(
    source: Image.Image,
    generated: Image.Image,
    source_path: str = "source",
    generated_path: str = "generated",
) -> dict:
    # 同时分析源照片与生成图，给出发色与服装主色的相似度
    src_mask = (alpha_mask(source) if source.mode in ("RGBA", "LA")
                else estimate_foreground(source))
    src_band = source_clothing_band(src_mask)
    src_hair_band = source_hair_band(src_mask)
    src = analyze(source, source_path, src_hair_band, src_band)
    gen = analyze(generated, generated_path, GENERATED_HAIR_BAND, GENERATED_CLOTHING_BAND)
    result: dict = {"source": src, "generated": gen}
    if src.hair_dominant and gen.hair_dominant:
        result["hair_similarity"] = color_similarity(src.hair_dominant, gen.hair_dominant)
    if src.clothing_dominant and gen.clothing_dominant:
        result["clothing_similarity"] = color_similarity(
            src.clothing_dominant, gen.clothing_dominant
        )
    return result
