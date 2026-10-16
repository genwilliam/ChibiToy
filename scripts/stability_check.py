#!/usr/bin/env python
"""三维度重复稳定性判定

用法::

    .venv/bin/python scripts/stability_check.py \\
        --source inputs/photo.jpg \\
        --outputs results/s1.png,results/s42.png,results/s123.png,results/s777.png,results/s2024.png \\
        --timings 416,441,447,433,450 \\
        --out docs/stability_report.md --json docs/stability_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image

from chibitoy import analysis

THRESHOLD_PERCENT = 10.0


def _is_direct_child(candidate: Path, parent: Path) -> bool:
    return candidate.parent == parent


def relative_deviation(values: list[float]) -> float | None:
    """(max-min)/mean ×100。全等或空集返回 0 / None。"""
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return None
    mean = float(np.mean(clean))
    if mean == 0:
        return 0.0
    return float((max(clean) - min(clean)) / mean * 100.0)


def measure(path: Path, source: Path, matte_source, base_url: str) -> dict:
    with Image.open(path) as image:
        width, height = image.size
        coverage = (float(analysis.alpha_mask(image).mean())
                    if image.mode in ("RGBA", "LA")
                    else float(analysis.estimate_foreground(image).mean()))
    src_img = matte_source(source, base_url) if matte_source else Image.open(source).convert("RGB")
    with Image.open(path) as generated:
        result = analysis.analyze_pair(src_img, generated, str(source), str(path))
    return {
        "path": str(path),
        "width": width,
        "height": height,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "coverage": coverage,
        "hair": result.get("hair_similarity"),
        "clothing": result.get("clothing_similarity"),
    }


def _identity_similarities(paths: list[Path], clip_vision: str | None) -> list[float]:
    import itertools

    from chibitoy import identity as identity_module

    ckpt = clip_vision or identity_module.DEFAULT_CLIP_VISION
    model, torch_module = identity_module.load_encoder(ckpt)
    vectors = [identity_module.embed(model, torch_module, path) for path in paths]
    return [identity_module.cosine(a, b) for a, b in itertools.combinations(vectors, 2)]


def judge(rows: list[dict], timings: list[float] | None,
          identity_similarities: list[float] | None = None) -> dict:
    times = [float(t) for t in timings] if timings else []
    dims: list[dict] = []

    if len(times) == len(rows) and times:
        dims.append(_dim("工程", "耗时", times, "s"))
    dims.append(_dim("工程", "前景覆盖率", [r["coverage"] for r in rows], ""))
    dims.append(_dim("工程", "输出边长", [float(min(r["width"], r["height"])) for r in rows], "px"))
    dims.append(_dim("配色", "发色相似度", [r["hair"] for r in rows], ""))
    dims.append(_dim("配色", "服装主色相似度", [r["clothing"] for r in rows], ""))
    if identity_similarities:
        dims.append(_dim("身份", "同人跨种子相似度", identity_similarities, ""))
    return {
        "dimensions": dims,
        "threshold_percent": THRESHOLD_PERCENT,
        "passed": all(d["passed"] for d in dims if d["deviation"] is not None),
    }


def _generate_samples(args, passthrough: list[str]) -> list[Path] | None:
    from chibitoy import generate as gen

    results_dir = Path("results/stability_gen")
    results_dir.mkdir(parents=True, exist_ok=True)

    cleaned: list[str] = []
    skip_next = False
    for token in passthrough:
        if skip_next:
            skip_next = False
            continue
        name = token.split("=", 1)[0]
        if name == "--out":
            skip_next = "=" not in token
            continue
        cleaned.append(token)

    if args.warmup:
        seed = args.base_seed - 1
        print(f"[稳定性] 预热运行（seed={seed}，结果丢弃）…")
        code = gen.main([*cleaned, "--seed", str(seed), "--prefix", f"{args.prefix}_warmup",
                         "--out", str(results_dir)])
        if code != 0:
            print("[错误] 预热运行失败", file=sys.stderr)
            return None

        removed = 0
        for stale in results_dir.glob(f"{args.prefix}_warmup_*.png"):
            if not _is_direct_child(stale.resolve(), results_dir.resolve()):
                print(f"[警告] 跳过目标目录之外的文件（不删除）: {stale}",
                      file=sys.stderr)
                continue
            stale.unlink()
            removed += 1
        print(f"[稳定性] 预热完成，其结果已丢弃（删除 {removed} 个预热产物）")

    import time

    produced: list[Path] = []
    timings: list[float] = []
    for index in range(args.generate):
        seed = args.base_seed + index
        prefix = f"{args.prefix}_s{seed}"
        print(f"\n===== 样本 {index + 1}/{args.generate}（seed={seed}）=====")
        started = time.monotonic()
        code = gen.main([*cleaned, "--seed", str(seed), "--prefix", prefix,
                         "--out", str(results_dir)])
        elapsed = time.monotonic() - started
        found = sorted(results_dir.glob(f"{prefix}_*.png"))
        if code != 0 or not found:
            print(f"[错误] seed={seed} 未产出", file=sys.stderr)
            return None
        print(f"[稳定性] 本次耗时 {elapsed:.2f}s")
        produced.append(found[-1])
        timings.append(round(elapsed, 2))
    return produced, timings


def _dim(dimension: str, name: str, values: list, unit: str) -> dict:
    clean = [float(v) for v in values if v is not None]
    deviation = relative_deviation(clean)
    return {
        "dimension": dimension,
        "name": name,
        "unit": unit,
        "values": [round(v, 4) for v in clean],
        "deviation": None if deviation is None else round(deviation, 3),
        "passed": (deviation is None) or (deviation <= THRESHOLD_PERCENT),
    }


def render(rows: list[dict], verdict: dict, source: str, note: str) -> str:
    lines: list[str] = []
    add = lines.append
    add("# 三维度重复稳定性判定")
    add("")
    add(f"- 源照片：`{source}`")
    add(f"- 产物件数：{len(rows)}")
    add(f"- 阈值：任一维度偏差 >{THRESHOLD_PERCENT:.0f}% 即不达标（`docs/acceptance_criteria.md` §5）")
    if note:
        add(f"- 备注：{note}")
    add("")
    add("## 逐维度结果")
    add("")
    add("| 维度 | 指标 | 实测值 | 偏差 | 判定 |")
    add("| --- | --- | --- | --- | --- |")
    for d in verdict["dimensions"]:
        values = " / ".join(
            f"{v:g}" for v in d["values"][:8]) + (" …" if len(d["values"]) > 8 else "")
        dev = "n/a" if d["deviation"] is None else f"**{d['deviation']:.2f}%**"
        add(f"| {d['dimension']} | {d['name']} | {values}{d['unit']} | {dev} | "
            f"{'✅' if d['passed'] else '❌ 不达标'} |")
    add("")
    add(f"**总判定：{'✅ 通过' if verdict['passed'] else '❌ 不通过'}**")
    add("")
    add("## 产物明细")
    add("")
    add("| # | 文件 | 尺寸 | 覆盖率 | 发色 | 服装主色 |")
    add("| --- | --- | --- | --- | --- | --- |")
    for index, row in enumerate(rows, 1):
        add(f"| {index} | `{row['path']}` | {row['width']}×{row['height']} | "
            f"{row['coverage'] * 100:.1f}% | {_fmt(row['hair'])} | {_fmt(row['clothing'])} |")
    add("")
    has_identity = any(d["dimension"] == "身份" for d in verdict["dimensions"])
    if has_identity:
        add("> 身份维度已计入：同人跨种子的**两两视觉相似度**极差。")
    else:
        add("> **身份维度未判定** —— 加 `--identity` 可并入"
            "（需加载 2.5 GB CLIP 视觉权重，约 10~25 s）。")
    add("> 注意：该身份指标是**通用视觉相似度、不是人脸识别**，不得用于「面部特征 ≥80%」判定。")
    add("")
    return "\n".join(lines) + "\n"


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--outputs", default=None,
                        help="逗号分隔的产物路径（同一配置、不同种子）。与 --generate 二选一")
    parser.add_argument("--timings", default=None, help="逗号分隔的耗时（秒），与 --outputs 一一对应")
    parser.add_argument("--out", default=None, help="Markdown 输出路径")
    parser.add_argument("--json", default=None, help="JSON 输出路径")
    parser.add_argument("--matte-source", action="store_true", help="源照片先用 BiRefNet 真实分割")
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("--note", default="")
    parser.add_argument("--generate", type=int, default=0, metavar="N",
                        help="自己生成 N 个样本（配合 -- 传入生成参数）。"
                             "生成的样本会进入判定；`--warmup` 那次不计入")
    parser.add_argument("--warmup", action="store_true",
                        help="正式样本前先跑一次并**丢弃**。§5 实测：冷启动比热态慢约 15%%，"
                             "预热可把偏差由 14.83%% 降到约 1%%（无离群时）")
    parser.add_argument("--prefix", default="stabgen", help="生成样本的输出前缀")
    parser.add_argument("--base-seed", type=int, default=1, help="样本种子起始值")
    parser.add_argument("--identity", action="store_true",
                        help="额外判定**身份维度**（同人跨种子的视觉相似度极差）。"
                             "需加载 2.5 GB CLIP 视觉权重，约 10~25 s，故默认关闭以保持秒级。"
                             "注意：该指标是通用视觉相似度、**不是人脸识别**")
    parser.add_argument("--clip-vision", default=None, help="CLIP 视觉权重路径（配合 --identity）")
    args, passthrough = parser.parse_known_args(argv)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    if args.generate and not passthrough:
        print("[错误] --generate 需要配合 -- 传入生成参数", file=sys.stderr)
        return 2
    if any(tok.split("=", 1)[0] == "--seed" for tok in passthrough):
        print("[错误] 请勿在 -- 之后传 --seed，种子由本脚本逐个指定", file=sys.stderr)
        return 2

    source = Path(args.source)
    if not source.is_file():
        print(f"[错误] 源照片不存在: {source}", file=sys.stderr)
        return 2
    if not args.outputs and not args.generate:
        print("[错误] 必须提供 --outputs（已有产物）或 --generate N（现场生成），二者二选一",
              file=sys.stderr)
        return 2
    if args.generate and args.outputs:
        print("[错误] --generate 与 --outputs 不能同时使用", file=sys.stderr)
        return 2
    timings = None
    outputs = [Path(p) for p in (args.outputs or "").split(",") if p]
    missing = [p for p in outputs if not p.is_file()]
    if missing:
        print(f"[错误] 以下产物不存在: {missing}", file=sys.stderr)
        return 2
    if args.generate:
        generated = _generate_samples(args, passthrough)
        if generated is None:
            return 1
        outputs, generated_timings = generated
        if not args.timings:
            timings = generated_timings

    if len(outputs) < 2:
        print("[错误] 至少需要 2 件产物才能计算偏差", file=sys.stderr)
        return 2

    if args.timings:
        try:
            timings = [float(t) for t in args.timings.split(",") if t]
        except ValueError:
            print("[错误] --timings 需为逗号分隔的数字", file=sys.stderr)
            return 2
        if len(timings) != len(outputs):
            print(f"[错误] --timings 个数 {len(timings)} 与产物个数 {len(outputs)} 不一致",
                  file=sys.stderr)
            return 2

    matte_source = None
    if args.matte_source:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "color_check_helper", Path(__file__).with_name("color_check.py"))
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        matte_source = module._matte_source

    print(f"[稳定性] 判定 {len(outputs)} 件产物，阈值 {THRESHOLD_PERCENT:.0f}%")
    rows = [measure(p, source, matte_source, args.comfy_url) for p in outputs]

    identity_similarities = None
    if args.identity:
        print("[稳定性] 计算身份维度（加载 CLIP 视觉权重）…")
        try:
            identity_similarities = _identity_similarities(outputs, args.clip_vision)
        except Exception as exc:
            print(f"[错误] 身份维度计算失败: {exc}", file=sys.stderr)
            return 2

    verdict = judge(rows, timings, identity_similarities)

    for d in verdict["dimensions"]:
        dev = "n/a" if d["deviation"] is None else f"{d['deviation']:.2f}%"
        print(f"  [{d['dimension']}] {d['name']}: 偏差 {dev} "
              f"{'✅' if d['passed'] else '❌ 不达标'}")

    report = render(rows, verdict, str(source), args.note)
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(report, encoding="utf-8")
        print(f"[已生成] {destination}")
    if args.json:
        sidecar = Path(args.json)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps({"source": str(source), "rows": rows, "verdict": verdict},
                                      ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[已生成] {sidecar}")
    if not args.out and not args.json:
        print(report)

    print(f"[总判定] {'✅ 通过' if verdict['passed'] else '❌ 不通过'}")
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
