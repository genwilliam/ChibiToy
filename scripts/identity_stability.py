#!/usr/bin/env python
"""测量身份特征在不同种子间是否稳定

注意：本脚本用的是通用视觉编码器（CLIP-ViT-H），度量的是整体视觉相似度，不是人脸识别

用法::

    .venv/bin/python scripts/identity_stability.py \\
        --group "photo1=results/stab20_s1_00001_.png,results/stab20_s42_00001_.png,..." \\
        --group "photo2=results/r27_photo2_00001_.png,results/r28_seed7_00001_.png"

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chibitoy import identity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", action="append", default=[],
                        help="格式 '名称=路径1,路径2,...'，可重复")
    parser.add_argument("--clip-vision", default=identity.DEFAULT_CLIP_VISION)
    args = parser.parse_args()

    if len(args.group) < 1:
        print("[错误] 至少需要一个 --group", file=sys.stderr)
        return 2

    groups: dict[str, list[Path]] = {}
    for spec in args.group:
        if "=" not in spec:
            print(f"[错误] --group 需为 '名称=路径,...' 格式: {spec}", file=sys.stderr)
            return 2
        name, _, raw = spec.partition("=")
        paths = [Path(p) for p in raw.split(",") if p]
        missing = [p for p in paths if not p.is_file()]
        if missing:
            print(f"[错误] 以下文件不存在: {missing}", file=sys.stderr)
            return 2
        groups[name] = paths

    print(f"[身份稳定性] 加载 CLIP 视觉编码器: {args.clip_vision}")
    result = identity.stability(groups, args.clip_vision)

    print(f"  分组: {result['vectors']}")
    within, cross = result["within"], result["cross"]
    print(f"  组内（同一主体，跨种子）: 均值 {within['mean']:.4f}  "
          f"范围 {within['min']:.4f}~{within['max']:.4f}  极差 {within['range']:.4f}  n={within['count']}")
    if cross["count"]:
        print(f"  跨组（不同主体）        : 均值 {cross['mean']:.4f}  "
              f"范围 {cross['min']:.4f}~{cross['max']:.4f}  n={cross['count']}")
        separated = within["mean"] > cross["mean"] + 0.02
        print(f"  -> 组内显著高于跨组: {'✅ 指标对身份有区分力' if separated else '❌ 无区分力'}")
    else:
        print(f"  跨组: 只给了一个分组，无法比较（仅报组内稳定性）")
    print()
    print("注意: 这是**通用视觉相似度**，不是人脸识别；不得用于「面部特征 ≥80%」验收项。")
    print("      且必须**同领域比较**（产物 ↔ 产物）；混入真人照片会被领域差异主导。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
