#!/usr/bin/env python
"""客观测量发色 / 服装主色还原度

用法:

    .venv/bin/python scripts/color_check.py \\
        --source inputs/img/demo.jpg \\
        --generated results/cn_sil_00001_.png

输出源照片与生成图各自的发区/服装区主色，以及归一化相似度
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PIL import Image

from chibitoy import analysis


def _matte_source(path: str, base_url: str) -> "Image.Image":
    """用 ComfyUI 的 BiRefNet 对源照片做真实前景分割，返回带 alpha 的图。

    源照片前景原先靠"边框主色即背景"估计，背景一杂乱就会失准
    （实测 `inputs/头像.jpg` 估计出 92.4% 前景）。此处复用 ComfyUI 已在用的
    BiRefNet，不新增任何模型权重。
    """
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "src"))
    from chibitoy.client import ComfyUIClient

    client = ComfyUIClient(base_url)
    if not client.is_available():
        raise RuntimeError(f"ComfyUI 不可用: {base_url}(--matte-source 需要它)")
    uploaded = client.upload_image(_Path(path))
    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": uploaded}},
        "2": {"class_type": "LoadBackgroundRemovalModel",
              "inputs": {"bg_removal_name": "birefnet.safetensors"}},
        "3": {"class_type": "RemoveBackground",
              "inputs": {"bg_removal_model": ["2", 0], "image": ["1", 0]}},
        "4": {"class_type": "InvertMask", "inputs": {"mask": ["3", 0]}},
        "5": {"class_type": "JoinImageWithAlpha",
              "inputs": {"image": ["1", 0], "alpha": ["4", 0]}},
        "6": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "colorcheck_src", "images": ["5", 0]}},
    }
    import tempfile

    result = client.generate(workflow, output_dir=_Path(tempfile.mkdtemp(prefix="colorcheck_")))
    if not result.saved_paths:
        raise RuntimeError("源照片抠图未产出结果")
    return Image.open(result.saved_paths[0]).convert("RGBA")


def _fmt(colors) -> str:
    return "  ".join(f"RGB{c} {p * 100:.0f}%" for c, p in colors[:3]) or "（空）"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="源真人照片")
    parser.add_argument("--matte-source", action="store_true",
                        help="先用 ComfyUI 的 BiRefNet 对源照片做**真实前景分割**再分析。"
                             "源照片背景不纯净时（如白墙/杂物）必须开启，否则前景估计会失准")
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188",
                        help="ComfyUI 地址（配合 --matte-source 使用）")
    parser.add_argument("--generated", required=True, help="生成的公仔图(带 alpha 的 PNG)")
    parser.add_argument("--hair-threshold", type=float, default=0.90,
                        help="发色相似度合格线(需求单为 100%%，此处按颜色近似放宽，需人工复核)")
    parser.add_argument("--clothing-threshold", type=float, default=0.90,
                        help="服装主色相似度合格线(需求单 ≥90%%)")
    args = parser.parse_args()

    for path in (args.source, args.generated):
        if not Path(path).is_file():
            print(f"[错误] 文件不存在: {path}", file=sys.stderr)
            return 2

    if args.matte_source:
        try:
            src_img = _matte_source(args.source, args.comfy_url)
            print(f"[抠图] 源照片已用 BiRefNet 真实分割: {args.source}")
        except Exception as exc:
            print(f"[错误] 源照片抠图失败: {exc}", file=sys.stderr)
            return 2
    else:
        with Image.open(args.source) as _f:
            src_img = _f.convert("RGB")
        print("[提示] 未开启 --matte-source,源照片将用边框主色估计前景"
              "背景杂乱时建议开启")

    with Image.open(args.generated) as gen:
        result = analysis.analyze_pair(src_img, gen, args.source, args.generated)

    src, gen = result["source"], result["generated"]
    print("=== 源照片 ===")
    print(f"  前景占比 {src.foreground_ratio * 100:.1f}%  发区 {src.hair_band}  服装区 {src.clothing_band}")
    print(f"  发区主色   : {_fmt(src.hair_colors)}")
    print(f"  服装区主色 : {_fmt(src.clothing_colors)}")
    print("=== 生成图 ===")
    print(f"  前景占比 {gen.foreground_ratio * 100:.1f}%  发区 {gen.hair_band}  服装区 {gen.clothing_band}")
    print(f"  发区主色   : {_fmt(gen.hair_colors)}")
    print(f"  服装区主色 : {_fmt(gen.clothing_colors)}")

    print("=== 还原度（归一化 RGB 相似度，1.0 为完全一致）===")
    exit_code = 0
    hair = result.get("hair_similarity")
    clothing = result.get("clothing_similarity")
    if hair is None:
        print("  发色: 无法计算")
    else:
        ok = hair >= args.hair_threshold
        print(f"  发色     {hair:.3f}  {'达标' if ok else '不达标'}(阈值 {args.hair_threshold})")
        exit_code |= 0 if ok else 1
    if clothing is None:
        print("  服装主色: 无法计算")
    else:
        ok = clothing >= args.clothing_threshold
        print(f"  服装主色 {clothing:.3f}  {'达标' if ok else '不达标'}(阈值 {args.clothing_threshold})")
        exit_code |= 0 if ok else 1
    print()
    print("注意: 本工具是「位置分带 + 颜色聚类」的近似度量，不是语义分割")
    print("      发型/款式等形状信息仍需人工核对。")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
