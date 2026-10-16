#!/usr/bin/env python
"""校验样本 manifest：字段数、文件存在性、SHA256 真实性、图像规格

用法::

    .venv/bin/python scripts/verify_manifest.py results/exp/stage_4_11/sample_manifest.csv

退出码：0 = 全部一致；1 = 存在不一致。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 允许的 PNG 规格
EXPECTED_SIZE = "1088x1088"
EXPECTED_FORMAT = "PNG-RGBA"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(csv_path: Path) -> int:
    from PIL import Image

    if not csv_path.is_file():
        print(f"[错误] manifest 不存在: {csv_path}")
        return 1

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        print(f"[错误] manifest 为空: {csv_path}")
        return 1

    problems: list[str] = []
    checked = 0

    for row in rows:
        sample = row.get("sample_id", "?")
        if None in row:
            problems.append(f"{sample}: 字段数多于表头（多出 {row[None]!r}）")
            continue
        if any(v is None for v in row.values()):
            problems.append(f"{sample}: 字段数少于表头（存在空缺列）")
            continue

        image_path = ROOT / row["image_path"]
        if not image_path.is_file():
            problems.append(f"{sample}: 输出文件不存在 {row['image_path']}")
            continue
        checked += 1

        claimed = (row.get("output_sha256") or "").strip()
        actual = sha256_of(image_path)
        if claimed and claimed != actual:
            problems.append(
                f"{sample}: output_sha256 不一致\n"
                f"    声明 {claimed}\n    实际 {actual}")
        elif not claimed:
            problems.append(f"{sample}: output_sha256 为空（应填实际哈希）")

        input_path = ROOT / row["input_path"]
        if not input_path.is_file():
            problems.append(f"{sample}: 输入文件不存在 {row['input_path']}")
        else:
            claimed_in = (row.get("input_sha256") or "").strip()
            actual_in = sha256_of(input_path)
            if claimed_in and claimed_in != actual_in:
                problems.append(
                    f"{sample}: input_sha256 不一致\n"
                    f"    声明 {claimed_in}\n    实际 {actual_in}")

        with Image.open(image_path) as img:
            size = f"{img.width}x{img.height}"
            mode = img.mode
        actual_format = f"PNG-{mode}"
        if row.get("size") and row["size"] != size:
            problems.append(
                f"{sample}: size 不一致 声明 {row['size']} 实际 {size}")
        if row.get("format") and row["format"] != actual_format:
            problems.append(
                f"{sample}: format 不一致 声明 {row['format']} 实际 {actual_format}")
        if size != EXPECTED_SIZE:
            problems.append(
                f"{sample}: 尺寸 {size} 未达验收目标 {EXPECTED_SIZE}")
        if actual_format != EXPECTED_FORMAT:
            problems.append(
                f"{sample}: 格式 {actual_format} 未达验收目标 {EXPECTED_FORMAT}")

    print(f"清单: {csv_path.relative_to(ROOT)}")
    print(f"样本: {len(rows)} 行，其中 {checked} 行通过文件存在性检查")
    if problems:
        print(f"\n发现 {len(problems)} 项问题：")
        for item in problems:
            print(f"  ❌ {item}")
        print("\n判定：❌ 存在不一致")
        return 1

    print("\n判定：✅ 全部一致（哈希/路径/尺寸/格式均已复核）")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="校验样本 manifest")
    parser.add_argument("manifest", type=Path, help="manifest CSV 路径")
    args = parser.parse_args()
    path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    return verify(path)


if __name__ == "__main__":
    sys.exit(main())
