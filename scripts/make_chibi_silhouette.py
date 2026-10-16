#!/usr/bin/env python
"""生成一张"Q 版公仔纯轮廓"参照图，供 ControlNet 锁定头身比使用

用法::

    .venv/bin/python scripts/make_chibi_silhouette.py --out tmp/chibi_silhouette.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

HEAD_RADIUS = 190
TOTAL_HEIGHT = 900
CANVAS = 1024


def build_silhouette(size: int = CANVAS) -> Image.Image:
    """画一个"大头小身、正面站立"的公仔剪影（纯黑，白底）。"""
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)
    cx = size // 2
    top = (size - TOTAL_HEIGHT) // 2
    head_r = HEAD_RADIUS
    head_cy = top + head_r
    body_top = head_cy + head_r - 30          # 与头部略微重叠，避免断颈
    body_bottom = top + TOTAL_HEIGHT - 150    # 留给腿脚
    body_w = 260

    # 头
    draw.ellipse([cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r], fill=0)
    # 躯干（圆角矩形）
    draw.rounded_rectangle(
        [cx - body_w // 2, body_top, cx + body_w // 2, body_bottom],
        radius=70, fill=0,
    )
    # 双腿
    leg_w, leg_h = 78, 150
    for sign in (-1, 1):
        lx = cx + sign * (leg_w // 2 + 8)
        draw.rounded_rectangle(
            [lx - leg_w // 2, body_bottom - 10, lx + leg_w // 2, body_bottom + leg_h],
            radius=leg_w // 2, fill=0,
        )
    # 脚
    foot_w, foot_h = 96, 46
    for sign in (-1, 1):
        fx = cx + sign * (leg_w // 2 + 8)
        fy = body_bottom + leg_h
        draw.rounded_rectangle(
            [fx - foot_w // 2, fy - 10, fx + foot_w // 2, fy - 10 + foot_h],
            radius=foot_h // 2, fill=0,
        )

    return img.convert("RGB")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="tmp/chibi_silhouette.png")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    build_silhouette().save(out)
    print(f"[已生成] {out}  ({', '.join(str(v) for v in build_silhouette().size)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
