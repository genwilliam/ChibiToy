#!/usr/bin/env python
"""人工盲评流程：把样本随机编号、打乱顺序，生成评分表

用法::

    # 1. 生成盲评材料（图片复制到 blind/ 并改名，附评分表）
    .venv/bin/python scripts/blind_eval.py --make \
        --samples results/exp/ablation_v2/A0_base_s42_00001_.png \
                  results/exp/ablation_v2/A1_accessory_s42_00001_.png \
        --out results/blind_eval --seed 7

    # 2. 评审填完盲评表后，揭盲并与客观指标合并
    .venv/bin/python scripts/blind_eval.py --reveal \
        --key results/blind_eval/key.json \
        --scores results/blind_eval/scores.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 评审逐项填写的字段
SCORE_FIELDS = {
    "翅膀袖伪影": ["无", "轻微", "明显"],
    "耳机状伪影": ["无", "轻微", "明显"],
    "头后圆环": ["无", "轻微", "明显"],
    "发型与发色保留": ["通过", "部分通过", "失败"],
    "服装款式与主色保留": ["通过", "部分通过", "失败"],
    "全身构图": ["通过", "失败"],
    "整体质量": ["1", "2", "3", "4", "5"],
}


def make(samples: list[Path], out_dir: Path, seed: int) -> dict:
    """生成盲评材料：随机编号 + 打乱顺序 + 写 key.json 与 scores.csv。"""
    missing = [str(p) for p in samples if not p.is_file()]
    if missing:
        raise SystemExit(f"[错误] 样本不存在: {missing}")
    out_dir.mkdir(parents=True, exist_ok=True)
    blind_dir = out_dir / "blind"
    blind_dir.mkdir(exist_ok=True)

    rng = random.Random(seed)
    order = list(range(len(samples)))
    rng.shuffle(order) # 打乱展示顺序
    labels = [f"P{i + 1:02d}" for i in range(len(samples))]
    rng.shuffle(labels) # 编号与顺序独立随机

    key = {"seed": seed, "items": []}
    for position, index in enumerate(order):
        label = labels[position]
        source = samples[index]
        target = blind_dir / f"{label}{source.suffix}"
        shutil.copyfile(source, target)
        # 组别只写进 key，不写进图片名或评分表
        key["items"].append({"label": label, "show_order": position + 1,
                             "group": source.stem, "source": str(source),
                             "blind_file": str(target.relative_to(out_dir))})
    (out_dir / "key.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    header = ["评审编号", "图片编号"] + list(SCORE_FIELDS)
    with (out_dir / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for label in labels:
            writer.writerow(["", label] + [""] * len(SCORE_FIELDS))
    print(f"[已生成] 盲评目录 {blind_dir}({len(samples)} 张，编号 {labels})")
    print(f"[已生成] {out_dir/'scores.csv'}（评审填写）与 {out_dir/'key.json'}(评审不得查看)")
    return key


def reveal(key_path: Path, scores_path: Path) -> int:
    # 把盲评结果与真实组别合并；同一组多份评分时报告一致率
    key = json.loads(key_path.read_text(encoding="utf-8"))
    label_to_group = {item["label"]: item["group"] for item in key["items"]}
    with scores_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print("[错误] 评分表为空", file=sys.stderr)
        return 2

    by_group: dict[str, list[dict]] = {}
    for row in rows:
        label = (row.get("图片编号") or "").strip()
        group = label_to_group.get(label)
        if group is None:
            print(f"[提示] 跳过未知编号 {label!r}")
            continue
        by_group.setdefault(group, []).append(row)

    print("=== 揭盲结果 ===")
    for group, items in sorted(by_group.items()):
        print(f"  {group}:")
        for field, allowed in SCORE_FIELDS.items():
            values = [(item.get(field) or "").strip() for item in items]
            values = [v for v in values if v]
            if not values:
                continue
            counts = {v: values.count(v) for v in dict.fromkeys(values)}
            agree = "（一致）" if len(counts) == 1 else f"(分歧 {counts})"
            print(f"    {field}: {counts} {agree}")
    reviewers = {(row.get("评审编号") or "").strip() for row in rows} - {""}
    if len(reviewers) < 2:
        print(f"\n  ⚠️ 仅 {len(reviewers) or 1} 名评审 —— 不得声称已达成多人一致性")
    else:
        print(f"\n  评审人数 {len(reviewers)}；同一图的多份评分已在上方按分歧标注")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--make", action="store_true", help="生成盲评材料")
    parser.add_argument("--reveal", action="store_true", help="揭盲并汇总")
    parser.add_argument("--samples", nargs="*", default=[])
    parser.add_argument("--out", default="results/blind_eval")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--key", default=None)
    parser.add_argument("--scores", default=None)
    args = parser.parse_args()

    if args.make:
        if not args.samples:
            print("[错误] --make 需要 --samples 至少一个样本", file=sys.stderr)
            return 2
        make([Path(p) for p in args.samples], Path(args.out), args.seed)
        return 0
    if args.reveal:
        key = Path(args.key or Path(args.out) / "key.json")
        scores = Path(args.scores or Path(args.out) / "scores.csv")
        return reveal(key, scores)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
