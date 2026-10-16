#!/usr/bin/env python
"""混合流水线稳定性评测：本地构图 -> 云端 API 重绘 -> 本地抠图，重复 N 次

关于随机种子:
    `qwen-image-2.0` 不支持指定随机种子（能力已在 apibackend.ProviderCapabilities 声明）
    因此无法"固定种子重复"，只能发独立请求

费用保护:
    - `--confirm-cost` 必须显式给出，否则只打印将要发起的请求数并退出
    - 单次请求前检查剩余次数，超过 `--max-calls` 立即停止；失败不自动重试

用法::

    .venv/bin/python scripts/hybrid_stability.py --runs 5 --confirm-cost \
        --base results/exp/levers/cloth_acc_s42_00001_.png --out results/stability_hybrid
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from PIL import Image

from chibitoy import apibackend, apiconfig
from chibitoy import config as project_config
from chibitoy.client import ComfyUIClient
from chibitoy import generate as gen

PROMPT = ("restyle this chibi figurine into a matte vinyl / soft clay collectible toy photo. "
          "Keep exactly the same pose, same full body including both legs and both feet, "
          "same head-to-body proportion, same clothing color, pure white background, "
          "soft studio lighting, no text, no watermark, no logo.")


def _stats(values: list[float]) -> dict:
    if not values:
        return {}
    mean = statistics.fmean(values)
    return {"n": len(values), "mean": round(mean, 4), "min": round(min(values), 4),
            "max": round(max(values), 4), "range": round(max(values) - min(values), 4),
            "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            # 相对偏差口径与项目一致：(max-min)/mean
            "relative_range_percent": round((max(values) - min(values)) / mean * 100, 2)
            if mean else None}


def run_one(index: int, *, base: Path, backend, out_dir: Path,
            prompt: str | None = None) -> dict:
    """一次混合生成；返回结构化记录。"""
    record: dict = {"run": index, "input_base": str(base),
                    "api_model": backend.provider.spec.model}
    started = time.monotonic()

    # 本地构图准备
    prepared = out_dir / f"in_{index}.png"
    with Image.open(base) as im:
        im.convert("RGB").resize((backend.cfg.generation.width,
                                  backend.cfg.generation.height), Image.LANCZOS).save(prepared)
    record["compose_seconds"] = round(time.monotonic() - started, 2)

    # 云端 API 重绘
    request = backend.build_request(prepared)
    request.prompt = prompt or PROMPT
    api_start = time.monotonic()
    try:
        result = backend.generate(request, output_dir=out_dir)
    except apibackend.ApiError as exc:
        record["api_ok"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"      # 异常信息不含密钥
        record["api_seconds"] = round(time.monotonic() - api_start, 2)
        record["elapsed_seconds"] = round(time.monotonic() - started, 2)
        return record
    record["api_ok"] = True
    record["api_seconds"] = round(time.monotonic() - api_start, 2)
    record["api_attempts"] = result.raw_history.get("attempts")
    api_out = result.saved_paths[0]
    record["api_output"] = str(api_out)

    # 本地抠图加 alpha
    alpha_start = time.monotonic()
    try:
        matted = gen._matte_reference_rgba(api_out, ComfyUIClient(), "birefnet.safetensors")
        final = out_dir / f"hybrid_run{index}_1088.png"
        matted.save(final)
        record["alpha_seconds"] = round(time.monotonic() - alpha_start, 2)
        record["final"] = str(final)
        with Image.open(final) as handle:
            record["size"] = list(handle.size)
            record["mode"] = handle.mode
        record["min_side_ok"] = min(record["size"]) >= 1080
    except Exception as exc:
        record["alpha_seconds"] = round(time.monotonic() - alpha_start, 2)
        record["alpha_error"] = f"{type(exc).__name__}: {exc}"
        record["final"] = str(api_out)
    record["elapsed_seconds"] = round(time.monotonic() - started, 2)
    return record


def measure(record: dict, source: Path, matted_source) -> None:
    """把客观指标补进记录（复用现有 analysis 口径，不另起一套）。"""
    final = record.get("final")
    if not final or not Path(final).is_file():
        return
    try:
        from scripts.exp_run import measure as exp_measure
        metrics = exp_measure(source, Path(final), matted_source=matted_source)
        record.update({"clothing_similarity": metrics["clothing_similarity"],
                       "hair_similarity": metrics["hair_similarity"],
                       "gen_coverage": metrics["gen_coverage"],
                       "gen_clothing": metrics["gen_clothing"]})
    except Exception as exc:
        record["measure_error"] = f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-calls", type=int, default=5,
                        help="硬上限：真实 API 调用次数（防止费用失控）")
    parser.add_argument("--confirm-cost", action="store_true",
                        help="**必须显式给出**才真正发起付费请求")
    parser.add_argument("--base", default="results/exp/levers/cloth_acc_s42_00001_.png",
                        help="本地构图的基准图（混合流水线的第一阶段产物）")
    parser.add_argument("--source", default="inputs/img/test.jpg", help="对比用的源照片")
    parser.add_argument("--out", default="results/stability_hybrid")
    parser.add_argument("--json", default=None)
    parser.add_argument("--prompt", default=None,
                        help="覆盖提示词（默认用本文件里的标准提示词）。"
                             "用于消融实验，例如去掉配饰：见 docs/api_backend.md §H.1")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(args.base)
    source = Path(args.source)
    if not base.is_file():
        print(f"[错误] 基准构图不存在: {base}", file=sys.stderr)
        return 2

    if args.runs > args.max_calls:
        print(f"[错误] --runs {args.runs} 超过 --max-calls {args.max_calls}，拒绝执行", file=sys.stderr)
        return 2

    if not args.confirm_cost:
        try:
            _model = apiconfig.load().active_provider().model or "（未配置）"
        except Exception as exc:
            _model = f"（配置不可读: {type(exc).__name__}）"
        print(f"[费用保护] 未给出 --confirm-cost。")
        print(f"           本次将发起 **{args.runs} 次真实付费 API 请求**"
              f"（模型 {_model}）。")
        print(f"           费用上界估计：0.25 元/张 × {args.runs} = ≤{0.25 * args.runs:.2f} 元"
              f"（依据同 family qwen-image-3.0-pro 的公开单价；2.0 的确切单价未逐字核实）。")
        print(f"           确认后请追加 --confirm-cost 重新运行。")
        return 0

    cfg = apiconfig.load(require_key=True)
    # ApiBackend 默认会重试（max_retries=3）
    cfg.active_provider().max_retries = 1
    backend = apibackend.ApiBackend(cfg)
    print(f"[后端] {cfg.describe()}  max_retries=1（失败不重试，避免费用翻倍）")
    print(f"[限制] 云端不支持固定随机种子 -> 只能发独立请求；失败不自动重试。")
    print(f"[计划] {args.runs} 次混合生成，硬上限 {args.max_calls} 次调用\n")

    matted_source = None
    try:
        from scripts.color_check import _matte_source
        matted_source = _matte_source(str(source), project_config.COMFY_BASE_URL)
    except Exception as exc:
        print(f"[提示] 源照片抠图失败（{type(exc).__name__}），服装/发色指标将改用位置先验估计")

    records: list[dict] = []
    calls = 0
    for index in range(1, args.runs + 1):
        if calls >= args.max_calls:
            print(f"[停止] 已达调用上限 {args.max_calls}")
            break
        print(f"===== 第 {index}/{args.runs} 次 =====")
        record = run_one(index, base=base, backend=backend, out_dir=out_dir,
                         prompt=args.prompt)
        calls += 1
        if record.get("api_ok"):
            measure(record, source, matted_source)
        records.append(record)
        print(f"  api_ok={record.get('api_ok')} 端到端={record['elapsed_seconds']}s "
              f"服装={record.get('clothing_similarity', '—')} 发色={record.get('hair_similarity', '—')}")
        if record.get("error"):
            print(f"  错误: {record['error'][:160]}")

    ok = [r for r in records if r.get("api_ok")]
    report = {
        "config": {"provider": cfg.provider, "model": backend.provider.spec.model,
                   "size": [cfg.generation.width, cfg.generation.height],
                   "seed_controllable": False,
                   "base": str(base), "source": str(source),
                   "prompt": args.prompt or PROMPT,
                   "prompt_overridden": bool(args.prompt)},
        "calls_made": calls,
        "success_count": len(ok),
        "success_rate": round(len(ok) / len(records), 4) if records else 0.0,
        "cost_estimate": {"unit_cny_upper_bound": 0.25, "calls": calls,
                          "upper_bound_cny": round(0.25 * calls, 2),
                          "basis": "同 family qwen-image-3.0-pro 公开单价；2.0 未逐字核实",
                          "verified": False},
        "records": records,
    }
    for key, label in (("elapsed_seconds", "端到端耗时"), ("clothing_similarity", "服装主色"),
                       ("hair_similarity", "发色")):
        values = [r[key] for r in ok if key in r]
        report[f"stats_{key}"] = _stats(values)
        if values:
            print(f"  [统计] {label}: {report[f'stats_{key}']}")

    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
    print(f"\n[已生成] {out_dir/'report.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
