#!/usr/bin/env python
"""把历轮产物渲染成带标注的「基准样图」对照样张

每格显示：缩略图、轮次标签、该轮配置、结论，以及客观配色指标

用法::

    .venv/bin/python scripts/make_benchmark_sheet.py \\
        --manifest docs/benchmark_manifest.json \\
        --out docs/benchmark_sheet.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PIL import Image, ImageDraw, ImageFont

from chibitoy import analysis

# macOS 自带的中文字体；缺失时回退到 PIL 默认字体
CJK_FONTS = [
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]

CELL = 380
PAD = 16
COLS = 3
BG = (250, 250, 252)
TEXT = (28, 28, 34)
MUTED = (110, 110, 120)
OK = (22, 130, 70)
BAD = (190, 45, 45)


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in CJK_FONTS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _fit(image: Image.Image, size: int) -> Image.Image:
    """等比缩放并居中贴到白色方块上。"""
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    copy = image.convert("RGBA")
    copy.thumbnail((size - 8, size - 8), Image.LANCZOS)
    flat = Image.new("RGBA", copy.size, (255, 255, 255, 255))
    flat.alpha_composite(copy)
    canvas.paste(flat.convert("RGB"), ((size - copy.width) // 2, (size - copy.height) // 2))
    return canvas


def _metrics(entry_path: Path, source_path: Path) -> tuple[float | None, float | None]:
    """现场计算发色 / 服装主色相似度（失败时返回 None，不阻塞样张生成）。"""
    try:
        with Image.open(source_path) as s, Image.open(entry_path) as g:
            result = analysis.analyze_pair(s, g, str(source_path), str(entry_path))
        return result.get("hair_similarity"), result.get("clothing_similarity")
    except Exception:
        return None, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="docs/benchmark_manifest.json")
    parser.add_argument("--out", default="docs/benchmark_sheet.png")
    parser.add_argument("--cell", type=int, default=CELL)
    parser.add_argument("--cols", type=int, default=COLS)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    source = Path(manifest["source"])
    entries = [e for e in manifest["entries"] if Path(e["file"]).is_file()]
    missing = [e["file"] for e in manifest["entries"] if not Path(e["file"]).is_file()]
    for name in missing:
        print(f"[跳过] 文件不存在: {name}", file=sys.stderr)
    if not entries:
        print("[错误] 没有任何可渲染的条目", file=sys.stderr)
        return 2

    cell = args.cell
    text_h = 132
    rows = (len(entries) + args.cols - 1) // args.cols
    width = args.cols * cell + (args.cols + 1) * PAD
    height = rows * (cell + text_h) + (rows + 1) * PAD + 136
    sheet = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(sheet)

    title_font, label_font = _font(30), _font(20)
    small_font = _font(17)

    draw.text((PAD, 24), "ChibiToy 基准样图（历轮对照）", font=title_font, fill=TEXT)
    draw.text((PAD, 62), f"源照片: {manifest['source']}    配色指标由 scripts/color_check.py 同源算法现场计算",
              font=small_font, fill=MUTED)
    draw.text((PAD, 84), "⚠️ 配色指标只反映颜色，会被退化图像欺骗（见「候选 LoRA 失败」格：画面已崩坏却得 0.995），必须与结构类一票否决项配合判读",
              font=small_font, fill=BAD)
    draw.text((PAD, 104), "身份维度另见 scripts/identity_stability.py（同人跨种子极差 0.068，稳定）—— 它是通用视觉相似度，不是人脸识别",
              font=small_font, fill=MUTED)

    for index, entry in enumerate(entries):
        row, col = divmod(index, args.cols)
        x = PAD + col * (cell + PAD)
        y = 136 + PAD + row * (cell + text_h + PAD)
        path = Path(entry["file"])

        with Image.open(path) as img:
            sheet.paste(_fit(img, cell), (x, y))
        draw.rectangle([x - 1, y - 1, x + cell, y + cell], outline=(225, 225, 232))

        hair, clothing = _metrics(path, source)
        ty = y + cell + 6
        draw.text((x, ty), entry["label"], font=label_font, fill=TEXT)
        draw.text((x, ty + 26), entry["config"], font=small_font, fill=MUTED)
        draw.text((x, ty + 48), entry.get("verdict", ""), font=small_font, fill=MUTED)
        line = (f"发色 {hair:.3f}" if hair is not None else "发色 n/a")
        draw.text((x, ty + 74), line, font=small_font,
                  fill=OK if (hair or 0) >= 0.9 else BAD)
        line2 = (f"服装主色 {clothing:.3f}" if clothing is not None else "服装主色 n/a")
        draw.text((x + 130, ty + 74), line2, font=small_font,
                  fill=OK if (clothing or 0) >= 0.9 else BAD)
        draw.text((x, ty + 98), "（阈值：发色 0.90 / 服装主色 0.90）", font=small_font, fill=MUTED)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"[已生成] {out}  {sheet.width}x{sheet.height}  共 {len(entries)} 格")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
