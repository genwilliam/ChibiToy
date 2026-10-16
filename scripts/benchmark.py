#!/usr/bin/env python
"""性能与稳定性测试脚本

测量内容:
  * 冷启动: 服务重启后第一次生成（包含模型加载时间）
  * 热运行: 后续连续生成
  * 每次生成的端到端耗时，以及 ComfyUI 自身报告的推理耗时
  * 运行前后的统一内存占用

测量方法说明:
  * 端到端耗时 = 客户端从 POST /prompt 到拿到 history 结果的墙钟时间。
  * 推理耗时   = 从 ComfyUI 日志中解析 "Prompt executed in X seconds"。
  * 内存占用   = 通过 /system_stats 读取 ram_free 前后差值。
  * 冷/热区分  = 冷启动需在运行本脚本前重启 ComfyUI。

用法::

    # 重启服务后测量冷启动 + 4 次热运行
    ./scripts/benchmark.py --runs 5 --steps 30

    # 只测热运行
    ./scripts/benchmark.py --runs 5 --skip-cold
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chibitoy import config, graph, prompts  # noqa: E402
from chibitoy.client import ComfyUIClient, ComfyUIError  # noqa: E402

LOG_PATH = config.LOGS_DIR / "comfyui.log"
EXEC_TIME_RE = re.compile(r"Prompt executed in ([\d.]+) seconds")


def parse_exec_times(log_path: Path) -> list[float]:
    """从 ComfyUI 日志中解析每次执行的耗时（秒）。"""
    if not log_path.is_file():
        return []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return [float(m) for m in EXEC_TIME_RE.findall(text)]


def system_memory_gb() -> dict:
    """读取当前统一内存状态（优先用 psutil，缺失时退回 vm_stat）。"""
    stats: dict = {}
    try:
        import psutil

        vm = psutil.virtual_memory()
        stats["total_gb"] = round(vm.total / 1024**3, 2)
        stats["available_gb"] = round(vm.available / 1024**3, 2)
        stats["used_percent"] = vm.percent
        return stats
    except ImportError:
        pass

    try:
        out = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=10, check=False
        ).stdout
        page_size = 16384
        free_pages = 0
        for line in out.splitlines():
            if "page size of" in line:
                page_size = int(re.search(r"page size of (\d+)", line).group(1))
            if line.startswith("Pages free"):
                free_pages = int(re.search(r"(\d+)", line.split(":")[1]).group(1))
        stats["free_gb"] = round(free_pages * page_size / 1024**3, 2)
    except Exception:
        pass
    return stats


def run_once(
    client: ComfyUIClient,
    *,
    ckpt: str,
    appearance: str,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    width: int,
    height: int,
    seed: int,
    output_dir: Path,
) -> dict:
    """执行一次生成，返回本次的测量数据。"""
    workflow = graph.build_sdxl_txt2img(
        ckpt_name=ckpt,
        positive_prompt=prompts.build_positive(appearance),
        negative_prompt=prompts.NEGATIVE_PROMPT,
        seed=seed,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler,
        scheduler=scheduler,
        width=width,
        height=height,
        filename_prefix="benchmark",
    )

    mem_before = system_memory_gb()
    started = time.monotonic()
    try:
        result = client.generate(workflow, output_dir=output_dir, timeout=3600)
    except (ComfyUIError, TimeoutError) as exc:
        return {"ok": False, "error": str(exc), "client_seconds": time.monotonic() - started}
    elapsed = time.monotonic() - started
    mem_after = system_memory_gb()

    return {
        "ok": True,
        "seed": seed,
        "prompt_id": result.prompt_id,
        "client_seconds": elapsed,
        "images": [str(p) for p in result.saved_paths],
        "mem_before": mem_before,
        "mem_after": mem_after,
    }


def summarize(label: str, samples: list[float]) -> dict:
    if not samples:
        return {"label": label, "count": 0}
    ordered = sorted(samples)
    # 样本较少时 p95 就取最大值，避免给出误导性的分位点
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {
        "label": label,
        "count": len(samples),
        "min": round(min(samples), 2),
        "max": round(max(samples), 2),
        "mean": round(statistics.mean(samples), 2),
        "median": round(statistics.median(samples), 2),
        "p95": round(p95, 2),
        "stdev": round(statistics.stdev(samples), 2) if len(samples) > 1 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ChibiToy 性能测试")
    parser.add_argument("--runs", type=int, default=5, help="总运行次数")
    parser.add_argument("--skip-cold", action="store_true",
                        help="跳过冷启动分析（不把第一次视为冷启动）")
    parser.add_argument("--ckpt", default=config.SDXL_BASE_CKPT)
    parser.add_argument("--appearance", default=prompts.DEFAULT_APPEARANCE)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--sampler", default="dpmpp_2m")
    parser.add_argument("--scheduler", default="karras")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--target-seconds", type=float, default=30.0,
                        help="单张耗时目标（清单 11.1）")
    parser.add_argument("--report", default=None, help="把结果写入 JSON 文件")
    args = parser.parse_args()

    client = ComfyUIClient()
    if not client.is_available():
        print(f"[错误] 无法连接 ComfyUI: {config.COMFY_BASE_URL}", file=sys.stderr)
        return 2

    log_before = len(parse_exec_times(LOG_PATH))
    stats = client.system_stats()
    system = stats.get("system", {})
    devices = stats.get("devices", [{}])
    device = devices[0] if devices else {}

    print("=" * 72)
    print("ChibiToy 性能测试")
    print("=" * 72)
    print(f"ComfyUI      : {system.get('comfyui_version')}")
    print(f"Python       : {(system.get('python_version') or '').split()[0]}")
    print(f"PyTorch      : {system.get('pytorch_version')}")
    print(f"设备         : {device.get('name')} ({device.get('type')})")
    print(f"统一内存     : {round((device.get('vram_total') or 0) / 1024**3, 1)} GB")
    print(f"模型         : {args.ckpt}")
    print(f"分辨率       : {args.width}x{args.height}")
    print(f"采样         : {args.sampler} / {args.scheduler} / {args.steps} 步 / CFG {args.cfg}")
    print(f"运行次数     : {args.runs}")
    print("-" * 72)

    all_client: list[float] = []
    cold_sample: float | None = None
    warm_samples: list[float] = []

    for index in range(args.runs):
        label = "冷启动" if (index == 0 and not args.skip_cold) else f"热运行#{index}"
        print(f"[{index + 1}/{args.runs}] {label} ... ", end="", flush=True)
        record = run_once(
            client,
            ckpt=args.ckpt,
            appearance=args.appearance,
            steps=args.steps,
            cfg=args.cfg,
            sampler=args.sampler,
            scheduler=args.scheduler,
            width=args.width,
            height=args.height,
            seed=args.seed_base + index,
            output_dir=config.OUTPUTS_DIR / "benchmark",
        )
        if not record["ok"]:
            print(f"失败: {record['error']}")
            continue
        elapsed = record["client_seconds"]
        all_client.append(elapsed)
        if index == 0 and not args.skip_cold:
            cold_sample = elapsed
        else:
            warm_samples.append(elapsed)
        print(f"{elapsed:.2f}s -> {Path(record['images'][0]).name if record['images'] else '(无图)'}")

    # 等待日志落盘后再解析
    time.sleep(1.0)
    exec_times = parse_exec_times(LOG_PATH)[log_before:]

    print("-" * 72)
    if cold_sample is not None:
        print(f"冷启动（含模型加载）: {cold_sample:.2f}s")
    print(f"客户端端到端总耗时  : {summarize('端到端', all_client)}")
    if warm_samples:
        warm = summarize("热运行", warm_samples)
        print(f"热运行统计          : {warm}")
        model_load = (cold_sample - warm["median"]) if cold_sample else None
        if model_load is not None:
            print(f"推算模型加载时间    : {model_load:.2f}s"
                  f"  (冷启动 {cold_sample:.2f}s - 热运行中位数 {warm['median']:.2f}s)")
    if exec_times:
        print(f"ComfyUI 报告推理耗时: {summarize('推理', exec_times)}")

    print("-" * 72)
    if warm_samples:
        median = statistics.median(warm_samples)
        verdict = "达标" if median <= args.target_seconds else "未达标"
        print(f"目标: 热运行中位端到端耗时 <= {args.target_seconds:.0f}s -> {verdict}"
              f"（实测 {median:.2f}s）")
    mem = system_memory_gb()
    if mem:
        print(f"测试后内存状态      : {mem}")

    if args.report:
        report = {
            "environment": {
                "comfyui_version": system.get("comfyui_version"),
                "pytorch_version": system.get("pytorch_version"),
                "device": device.get("name"),
                "unified_memory_gb": round((device.get("vram_total") or 0) / 1024**3, 1),
            },
            "parameters": {
                "model": args.ckpt,
                "resolution": [args.width, args.height],
                "sampler": args.sampler,
                "scheduler": args.scheduler,
                "steps": args.steps,
                "cfg": args.cfg,
                "runs": args.runs,
            },
            "cold_seconds": cold_sample,
            "client_summary": summarize("端到端", all_client),
            "warm_summary": summarize("热运行", warm_samples),
            "comfy_exec_summary": summarize("推理", exec_times),
            "target_seconds": args.target_seconds,
            "memory_after": mem,
        }
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[已保存] {args.report}")

    return 0 if warm_samples else 1


if __name__ == "__main__":
    raise SystemExit(main())
