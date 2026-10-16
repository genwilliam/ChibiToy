#!/usr/bin/env python
"""环境自检脚本

一次性检查 macOS / Python / PyTorch / MPS / ComfyUI / 模型文件 / 磁盘空间

用法::

    .venv/bin/python scripts/check_env.py
    .venv/bin/python scripts/check_env.py --json   # 机器可读输出
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chibitoy import config

OK = "OK"
WARN = "警告"
FAIL = "失败"


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=20, check=False
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def check_system() -> dict:
    mem_bytes = int(_run(["sysctl", "-n", "hw.memsize"]) or 0)
    usage = shutil.disk_usage(PROJECT_ROOT)
    return {
        "macos_version": platform.mac_ver()[0],
        "chip": _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor(),
        "arch": platform.machine(),
        "unified_memory_gb": round(mem_bytes / 1024**3, 1),
        "disk_free_gb": round(usage.free / 1024**3, 1),
        "disk_total_gb": round(usage.total / 1024**3, 1),
    }


def check_python() -> dict:
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "in_venv": sys.prefix != sys.base_prefix,
        "venv_path": sys.prefix,
    }


def check_torch() -> dict:
    info: dict = {"installed": False}
    try:
        import torch
    except ImportError as exc:
        info["error"] = str(exc)
        return info

    info.update(
        installed=True,
        torch_version=torch.__version__,
        mps_available=bool(torch.backends.mps.is_available()),
        mps_built=bool(torch.backends.mps.is_built()),
        cuda_available=bool(torch.cuda.is_available()),
    )

    if info["mps_available"]:
        try:
            device = torch.device("mps")
            a = torch.randn(512, 512, device=device)
            b = torch.randn(512, 512, device=device)
            c = (a @ b).sum().item()
            torch.mps.synchronize()
            info["mps_compute_test"] = "通过" if c == c else "结果异常(NaN)"
            info["mps_fp16_test"] = "通过" if (a.half() @ b.half()) is not None else "失败"
        except Exception as exc:
            info["mps_compute_test"] = f"失败: {exc}"
    return info


def check_comfyui() -> dict:
    main_py = config.COMFYUI_DIR / "main.py"
    info: dict = {
        "installed": main_py.is_file(),
        "path": str(config.COMFYUI_DIR),
    }
    if not main_py.is_file():
        return info

    version_file = config.COMFYUI_DIR / "comfyui_version.py"
    if version_file.is_file():
        namespace: dict = {}
        exec(version_file.read_text(encoding="utf-8"), namespace)
        info["version"] = namespace.get("__version__")

    git_head = _run(["git", "-C", str(config.COMFYUI_DIR), "log", "-1", "--format=%H %ad", "--date=iso"])
    info["git_head"] = git_head
    return info


def check_service() -> dict:
    try:
        from chibitoy.client import ComfyUIClient
    except ImportError as exc:
        return {"reachable": False, "error": str(exc)}

    client = ComfyUIClient()
    if not client.is_available():
        return {"reachable": False, "url": config.COMFY_BASE_URL}

    stats = client.system_stats()
    devices = [
        {"name": d.get("name"), "type": d.get("type"),
         "vram_total_gb": round((d.get("vram_total") or 0) / 1024**3, 1),
         "vram_free_gb": round((d.get("vram_free") or 0) / 1024**3, 1)}
        for d in stats.get("devices", [])
    ]
    system = stats.get("system", {})
    return {
        "reachable": True,
        "url": config.COMFY_BASE_URL,
        "comfyui_version": system.get("comfyui_version"),
        "python_version": system.get("python_version"),
        "pytorch_version": system.get("pytorch_version"),
        "devices": devices,
    }


def check_models() -> dict:
    ckpt = config.CHECKPOINTS_DIR / config.SDXL_BASE_CKPT
    info: dict = {"checkpoints_dir": str(config.CHECKPOINTS_DIR), "files": []}

    if not config.CHECKPOINTS_DIR.is_dir():
        info["error"] = "checkpoints 目录不存在"
        return info

    for path in sorted(config.CHECKPOINTS_DIR.glob("*")):
        if path.is_file() and path.suffix in {".safetensors", ".ckpt", ".pt", ".sft"}:
            info["files"].append(
                {"name": path.name, "size_gb": round(path.stat().st_size / 1024**3, 2)}
            )

    if ckpt.is_file():
        info["sdxl_base_present"] = True
        info["sdxl_base_size_gb"] = round(ckpt.stat().st_size / 1024**3, 2)
        info["sdxl_base_size_expected_gb"] = 6.46
        info["sdxl_base_complete"] = ckpt.stat().st_size == 6938078334
    else:
        info["sdxl_base_present"] = False
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description="ChibiToy 环境自检")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()

    report = {
        "system": check_system(),
        "python": check_python(),
        "torch": check_torch(),
        "comfyui": check_comfyui(),
        "service": check_service(),
        "models": check_models(),
    }

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print_report(report)

    return _exit_code(report)


def _print_report(report: dict) -> None:
    sysinfo = report["system"]
    print("=" * 66)
    print("ChibiToy 环境自检")
    print("=" * 66)
    print(f"macOS        : {sysinfo['macos_version']} ({sysinfo['arch']})")
    print(f"芯片         : {sysinfo['chip']}")
    print(f"统一内存     : {sysinfo['unified_memory_gb']} GB")
    print(f"磁盘可用     : {sysinfo['disk_free_gb']} GB / {sysinfo['disk_total_gb']} GB")

    py = report["python"]
    print(f"Python       : {py['python_version']}  (venv: {py['in_venv']})")
    print(f"解释器       : {py['python_executable']}")

    torch_info = report["torch"]
    if not torch_info.get("installed"):
        print(f"PyTorch      : [{FAIL}] 未安装 ({torch_info.get('error')})")
    else:
        print(f"PyTorch      : {torch_info['torch_version']}")
        print(f"MPS 可用     : {torch_info['mps_available']}  (已构建: {torch_info['mps_built']})")
        print(f"MPS 实算     : {torch_info.get('mps_compute_test', '未测试')}")
        print(f"CUDA 可用    : {torch_info['cuda_available']}  (Apple Silicon 应为 False)")

    comfy = report["comfyui"]
    if comfy.get("installed"):
        print(f"ComfyUI      : v{comfy.get('version')} @ {comfy['path']}")
        print(f"Git HEAD     : {comfy.get('git_head')}")
    else:
        print(f"ComfyUI      : [{FAIL}] 未安装 @ {comfy.get('path')}")

    service = report["service"]
    if service.get("reachable"):
        print(f"服务状态     : [OK] {service['url']}")
        print(f"服务版本     : ComfyUI {service.get('comfyui_version')} / "
              f"PyTorch {service.get('pytorch_version')}")
        for device in service.get("devices", []):
            print(f"计算设备     : {device['name']} ({device['type']}), "
                  f"可用显存 {device['vram_free_gb']} GB / {device['vram_total_gb']} GB")
    else:
        print(f"服务状态     : [{WARN}] 无法连接 {service.get('url', config.COMFY_BASE_URL)}")
        print("               启动命令: ./scripts/start_comfyui.sh --background")

    models = report["models"]
    print("-" * 66)
    if models.get("sdxl_base_present"):
        state = OK if models.get("sdxl_base_complete") else "不完整(下载中?)"
        print(f"SDXL Base    : [{state}] {models['sdxl_base_size_gb']} GB "
              f"(期望 {models['sdxl_base_size_expected_gb']} GB)")
    else:
        print(f"SDXL Base    : [{FAIL}] 缺失，运行 ./scripts/download_models.sh")
    if models.get("files"):
        print(f"检出模型     : {[f['name'] for f in models['files']]}")
    print("=" * 66)


def _exit_code(report: dict) -> int:
    torch_info = report["torch"]
    if not torch_info.get("installed") or not torch_info.get("mps_available"):
        return 1
    if not report["comfyui"].get("installed"):
        return 1
    if not report["models"].get("sdxl_base_complete"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
