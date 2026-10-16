#!/usr/bin/env python
"""受控对照实验运行器

用途
    服装配色问题的排查需要每次只改一个变量，并把每次运行的完整参数与指标落盘

规格文件(JSON)::

    {
      "name": "crop",
      "common": ["--mode","txt2img","--width","1088","--height","1088",
                 "--input-image","inputs/img/test.jpg", ...],
      "variants": {
        "top-square":  ["--ipadapter-crop","top-square"],
        "head-square": ["--ipadapter-crop","head-square"]
      },
      "seeds": [42]
    }

用法::

    .venv/bin/python scripts/exp_run.py --spec experiments/crop.json
    .venv/bin/python scripts/exp_run.py --spec experiments/crop.json --steps 8 --dry-run

产出(写到 ``--out``，默认 ``results/exp/<name>``):
    ``<variant>_s<seed>_<n>.png`` 产物
    ``report.json`` / ``report.md``  每次运行的参数与指标

指标口径(与 ``scripts/color_check.py`` 一致):沿用在 ``chibitoy.analysis`` 中的
源/生成波段定义。必须注意：这是按位置先验取的波段，不是服装语义分割;
在没有语义分割的情况下，它是"服装区域颜色"的近似，不能称为服装颜色准确率
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))      # 以便 import scripts.color_check 复用同一抠图口径

from chibitoy import analysis
from chibitoy import generate as gen
from PIL import Image


def measure(source: Path, generated: Path, matted_source=None, base_url: str | None = None) -> dict:
    """算一对（源照片, 生成图）的客观指标

    ``matted_source`` 可由调用方传入以避免对源照片重复跑 BiRefNet
    """
    if matted_source is None:
        from scripts.color_check import _matte_source  # 复用同一条抠图口径

        matted_source = _matte_source(str(source), base_url or "http://127.0.0.1:8188")
    im = matted_source
    if isinstance(im, (str, Path)):
        with Image.open(im) as handle:
            im = handle.copy()
    with Image.open(generated) as handle:
        gen_image = handle.convert("RGBA")
    # 走 analyze_pair，保证源照片的服装带按解剖位置取（与 color_check 同一口径）
    pair = analysis.analyze_pair(im, gen_image, str(source), str(generated))
    src_report, gen_report = pair["source"], pair["generated"]
    return {
        "hair_similarity": round(pair["hair_similarity"], 4),
        "clothing_similarity": round(pair["clothing_similarity"], 4),
        "src_hair": src_report.hair_dominant,
        "src_clothing": src_report.clothing_dominant,
        "src_clothing_band": list(src_report.clothing_band),
        "gen_hair": gen_report.hair_dominant,
        "gen_clothing": gen_report.clothing_dominant,
        "gen_coverage": round(gen_report.foreground_ratio, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, help="实验规格 JSON")
    parser.add_argument("--out", default=None, help="输出目录（默认 results/exp/<name>）")
    parser.add_argument("--steps", type=int, default=None, help="覆盖步数（筛选阶段可调小）")
    parser.add_argument("--seeds", default=None, help="逗号分隔，覆盖规格里的 seeds")
    parser.add_argument("--variants", default=None, help="逗号分隔，只跑指定变体")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    name = spec.get("name", "exp")
    out_dir = Path(args.out or f"results/exp/{name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in (args.seeds.split(",") if args.seeds else spec.get("seeds", [42]))]
    variants = spec["variants"]
    if args.variants:
        keep = set(args.variants.split(","))
        variants = {k: v for k, v in variants.items() if k in keep}
    source = Path(spec.get("source", "inputs/img/test.jpg"))

    common = list(spec["common"])
    if args.steps is not None:
        if "--steps" in common:
            common[common.index("--steps") + 1] = str(args.steps)
        else:
            common += ["--steps", str(args.steps)]

    rows: list[dict] = []
    for variant, extra in variants.items():
        for seed in seeds:
            prefix = f"{variant}_s{seed}"
            argv = common + extra + ["--seed", str(seed), "--prefix", prefix,
                                     "--out", str(out_dir)]
            print(f"\n===== {name} / {variant} / seed={seed} =====")
            if args.dry_run:
                continue
            started = time.monotonic()
            code = gen.main(argv)
            elapsed = round(time.monotonic() - started, 2)
            produced = sorted(out_dir.glob(f"{prefix}_*.png"))
            row = {
                "variant": variant, "seed": seed, "extra_args": extra,
                "elapsed_seconds": elapsed, "exit_code": code,
                "output": str(produced[-1]) if produced else None,
            }
            if code == 0 and produced:
                try:
                    row.update(measure(source, produced[-1]))
                except Exception as exc:
                    row["measure_error"] = f"{type(exc).__name__}: {exc}"
            else:
                row["failure"] = "生成失败或未产出"
            rows.append(row)
            print(f"  -> {json.dumps({k: v for k, v in row.items() if k != 'extra_args'}, ensure_ascii=False)}")

    if args.dry_run:
        print("\n[dry-run] 未执行")
        return 0

    (out_dir / "report.json").write_text(
        json.dumps({"spec": spec, "steps": args.steps, "rows": rows},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    ok = [r for r in rows if "clothing_similarity" in r]
    lines = [f"# 实验对照：{name}", "",
             f"- 源照片：`{source}`", f"- 步数：{args.steps or '规格默认'}",
             f"- 样本数：{len(rows)}（有效 {len(ok)}）", "",
             "| 变体 | 种子 | 服装主色相似度 | 发色相似度 | 前景覆盖 | 耗时(s) | 输出 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['seed']} | "
            f"{r.get('clothing_similarity', '—')} | {r.get('hair_similarity', '—')} | "
            f"{r.get('gen_coverage', '—')} | {r['elapsed_seconds']} | "
            f"{'`' + r['output'] + '`' if r.get('output') else '失败'} |")
    if ok:
        clothes = [r["clothing_similarity"] for r in ok]
        lines += ["", f"- 服装主色：均值 **{sum(clothes)/len(clothes):.3f}**，"
                      f"范围 {min(clothes):.3f}~{max(clothes):.3f}"]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[已生成] {out_dir/'report.json'} 与 {out_dir/'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
