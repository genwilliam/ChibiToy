#!/usr/bin/env python
"""照片样本登记与独立性检查


用法::

    # 1) 查看当前已登记样本
    .venv/bin/python scripts/sample_registry.py --list

    # 2) 登记一张新照片（自动分配 sample_id、算 SHA256、写 metadata.json）
    .venv/bin/python scripts/sample_registry.py --add ~/Desktop/new_photo.jpg

    # 3) 登记前只检查、不落盘
    .venv/bin/python scripts/sample_registry.py --check ~/Desktop/new_photo.jpg

退出码：0 = 成功/通过；1 = 校验失败或有重复；2 = 用法错误。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chibitoy import preprocess

# 每张照片一个子目录
SAMPLES_DIR = ROOT / "inputs" / "photos"
METADATA_NAME = "metadata.json"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_registry() -> list[dict]:
    """读取所有已登记样本的 metadata（跳过损坏的，并报告）"""
    records: list[dict] = []
    if not SAMPLES_DIR.is_dir():
        return records
    for meta_path in sorted(SAMPLES_DIR.glob(f"*/{METADATA_NAME}")):
        try:
            records.append(json.loads(meta_path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            print(f"[警告] 无法读取 {rel_to_root(meta_path)}: {exc}", file=sys.stderr)
    return records


def inspect_photo(path: Path) -> tuple[dict, list[str], list[str]]:
    """按项目既有规格检查一张照片

    Returns:
        (info, errors, warnings) errors 非空表示不可用。
    """
    errors: list[str] = []
    warnings: list[str] = []
    info: dict = {"source_path": str(path), "source_name": path.name}

    if not path.is_file():
        return info, [f"文件不存在: {path}"], warnings

    try:
        info["format"], info["original_size"] = (
            lambda d: (d["format"], d["size"]))(
            preprocess.validate_image(path))
    except preprocess.PreprocessError as exc:
        return info, [str(exc)], warnings

    info["file_bytes"] = path.stat().st_size
    info["sha256"] = sha256_of(path)

    width, height = info["original_size"]
    info["min_side"] = min(width, height)

    if width < preprocess.DEFAULT_MIN_SIDE or height < preprocess.DEFAULT_MIN_SIDE:
        warnings.append(
            f"低分辨率素材：{width}x{height} 的最短边 {min(width, height)}px "
            f"低于推荐边长 {preprocess.DEFAULT_MIN_SIDE}px。"
            f"项目现行策略**接受**该素材，但它**不满足**正式需求单的输入规格，"
            f"不得与合规素材混在同一份泛化结论里。"
        )
    if min(width, height) < preprocess.HARD_MIN_SIDE:
        errors.append(
            f"尺寸过小：{width}x{height} 低于可处理下限 "
            f"{preprocess.HARD_MIN_SIDE}px"
        )
    return info, errors, warnings


def check_duplicate(info: dict, registry: list[dict]) -> dict | None:
    """按内容 SHA256 查重；返回已存在的记录（无则 None）。"""
    digest = info.get("sha256")
    if not digest:
        return None
    for record in registry:
        if record.get("sha256") == digest:
            return record
    return None


def next_sample_id(registry: list[dict]) -> str:
    """分配下一个样本编号，如 S01、S02（按已登记最大编号递增）。"""
    highest = 0
    for record in registry:
        sid = str(record.get("sample_id", ""))
        if sid.startswith("S") and sid[1:].isdigit():
            highest = max(highest, int(sid[1:]))
    return f"S{highest + 1:02d}"


def cmd_list(registry: list[dict]) -> int:
    if not registry:
        print(f"（尚无已登记样本；目录: {rel_to_root(SAMPLES_DIR)}）")
        print("登记方式: .venv/bin/python scripts/sample_registry.py --add <照片路径>")
        return 0
    print(f"已登记样本（{len(registry)} 个独立来源）:\n")
    header = f"{'ID':<5} {'尺寸':<12} {'SHA256':<12} {'最低边':<7} 文件名"
    print(header)
    print("-" * len(header))
    for record in registry:
        size = record.get("original_size") or [0, 0]
        print(f"{record.get('sample_id', '?'):<5} "
              f"{f'{size[0]}x{size[1]}':<12} "
              f"{str(record.get('sha256', ''))[:10]:<12} "
              f"{record.get('min_side', '?'):<7} "
              f"{record.get('source_name', '?')}")
    low = [r for r in registry
           if (r.get("min_side") or 0) < preprocess.DEFAULT_MIN_SIDE]
    if low:
        print(f"\n⚠️ 其中 {len(low)} 张为低分辨率素材（低于推荐边长 "
              f"{preprocess.DEFAULT_MIN_SIDE}px）: "
              f"{', '.join(r.get('sample_id', '?') for r in low)}")
        print("   这些样本**不满足**正式输入规格；泛化结论必须分成两组陈述。")
    return 0


def cmd_check(path: Path, registry: list[dict]) -> int:
    """只检查、不落盘。"""
    info, errors, warnings = inspect_photo(path)
    print(f"检查: {path}")
    if "original_size" in info:
        width, height = info["original_size"]
        print(f"  格式      : {info['format']}")
        print(f"  尺寸      : {width}x{height}（最短边 {min(width, height)}px）")
        print(f"  文件大小  : {info['file_bytes']} 字节 "
              f"（上限 {preprocess.DEFAULT_MAX_FILE_SIZE} 字节）")
        print(f"  SHA256    : {info['sha256']}")
    for note in warnings:
        print(f"  [警告] {note}")
    for note in errors:
        print(f"  [错误] {note}")

    duplicate = check_duplicate(info, registry)
    if duplicate:
        print(f"  [重复] 该内容已登记为 {duplicate.get('sample_id')} "
              f"（{duplicate.get('source_name')}）")
        print("         **同一张照片不得计为两个独立样本。**")
        errors.append(f"内容与已登记样本 {duplicate.get('sample_id')} 完全相同")

    if errors:
        print("\n判定：❌ 不可登记")
        return 1
    print(f"\n判定：✅ 可登记（将分配 {next_sample_id(registry)}）")
    print("  注意：本检查为**逐字节**查重。同一张照片若被重编码、缩放或裁切，"
          "哈希会不同而**无法被自动识别**——此类情况必须人工确认来源同一性，"
          "并在 metadata 的 derived_from 字段中如实标注。")
    return 0


def cmd_add(path: Path, registry: list[dict], allow_duplicate: bool) -> int:
    info, errors, warnings = inspect_photo(path)
    duplicate = check_duplicate(info, registry)
    if duplicate and not allow_duplicate:
        print(f"[错误] 该内容已登记为 {duplicate.get('sample_id')} "
              f"（{duplicate.get('source_name')}）。", file=sys.stderr)
        print("       同一张照片不得计为两个独立样本。"
              "确需另存请用 --allow-duplicate 并说明理由。", file=sys.stderr)
        return 1
    if errors:
        for note in errors:
            print(f"[错误] {note}", file=sys.stderr)
        return 1

    sample_id = next_sample_id(registry)
    target_dir = SAMPLES_DIR / sample_id
    if target_dir.exists():
        print(f"[错误] 目标目录已存在，拒绝覆盖: {target_dir}", file=sys.stderr)
        return 1

    target_dir.mkdir(parents=True)
    target = target_dir / f"original{path.suffix.lower()}"
    shutil.copyfile(path, target)

    record = {
        "sample_id": sample_id,
        "source_name": path.name,
        "original_size": info.get("original_size"),
        "min_side": info.get("min_side"),
        "file_bytes": info.get("file_bytes"),
        "format": info.get("format"),
        "sha256": info.get("sha256"),
        "stored_path": rel_to_root(target),
        "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "meets_formal_input_spec": (info.get("min_side") or 0)
        >= preprocess.DEFAULT_MIN_SIDE,
        "warnings": warnings,
        "derived_from": None,
        "notes": "",
    }
    (target_dir / METADATA_NAME).write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[已登记] {sample_id} -> {rel_to_root(target_dir)}")
    print(f"         尺寸 {info['original_size'][0]}x{info['original_size'][1]}，"
          f"SHA256 {info['sha256']}")
    print(f"         满足正式输入规格: "
          f"{'是' if record['meets_formal_input_spec'] else '否（低分辨率素材）'}")
    for note in warnings:
        print(f"         [警告] {note}")
    print(f"\n下一步：把该样本加入基准测试")
    print(f"  .venv/bin/python scripts/batch_generate.py --inputs "
          f"{rel_to_root(target_dir)} --pattern 'original*' -- <生成参数>")
    print(f"注意：一个样本的多个 seed/变体**不是**多个独立样本。")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="照片样本登记与独立性检查",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="列出已登记样本")
    group.add_argument("--check", metavar="照片", help="只检查某张照片，不落盘")
    group.add_argument("--add", metavar="照片", help="登记一张新照片")
    parser.add_argument("--allow-duplicate", action="store_true",
                        help="允许登记内容重复的照片（需自行说明理由）")
    args = parser.parse_args(argv)

    registry = load_registry()
    if args.list:
        return cmd_list(registry)
    if args.check:
        return cmd_check(Path(args.check).expanduser(), registry)
    return cmd_add(Path(args.add).expanduser(), registry, args.allow_duplicate)


if __name__ == "__main__":
    raise SystemExit(main())
