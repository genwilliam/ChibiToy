#!/usr/bin/env python
"""把「依赖清单 / 安装说明」从文字声明变成可执行的核对

只需运行本脚本即可确认：
  1. 环境是否符合文档声明（Python 版本、包数量、ComfyUI 提交号、自定义节点）；
  2. 模型权重是否齐全，SHA256 是否与文档一致（
  3. ComfyUI 是否在运行，且工作流所需的节点与模型都已被识别。

用法::

    .venv/bin/python scripts/verify_delivery.py            # 全量（含 15 GB 权重哈希，约 1~2 分钟）
    .venv/bin/python scripts/verify_delivery.py --skip-hash  # 快速（只查存在性）
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("CHIBITOY_ROOT") or Path(__file__).resolve().parent.parent)

MODELS = [
    ("models/checkpoints/sd_xl_base_1.0.safetensors", 6_938_078_334,
     "31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b", True),
    ("models/loras/Designer_BlindBox-000015.safetensors", 681_239_124,
     "218c7602c3c876e901f5e117d79411731254dd48b5d543382f7c49d9538a7d0e", True),
    ("models/ipadapter/ip-adapter-plus-face_sdxl_vit-h.safetensors", 847_517_512,
     "677ad8860204f7d0bfba12d29e6c31ded9beefdf3e4bbd102518357d31a292c1", True),
    ("models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors", 2_528_373_448,
     "6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030", True),
    ("models/background_removal/birefnet.safetensors", 444_473_596,
     "9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154", True),
    ("models/controlnet/canny_sdxl.safetensors", 2_502_139_104,
     "bf47cd757ceaf2572c53321329ef819ea38c09a6e3783588387913cd94dff47c", True),
    ("models/loras/figurine_slider.safetensors", 8_789_076,
     "5484762feed22fecdc1fd5007aa2ae7fea34891704ec3c1ce71545ecb740ff35", False),
    ("models/loras/toy_v1.safetensors", 93_065_304,
     "80d80f4998c34c9ac4ef55d9af28e6e25a867788c1c6158c5fc8f3683061fa2c", False),
]

DELIVERABLES = [
    "README.md",
    "config/extra_model_paths.yaml",
    "workflows/chibitoy_final.api.json",
    "workflows/chibitoy_final.ui.json",
    "src/chibitoy/config.py",
    "src/chibitoy/prompts.py",
    "src/chibitoy/graph.py",
    "src/chibitoy/client.py",
    "src/chibitoy/generate.py",
    "src/chibitoy/preprocess.py",
    "src/chibitoy/analysis.py",
    "src/chibitoy/identity.py",
    "scripts/batch_generate.py",
    "scripts/best_of_n.py",
    "scripts/make_chibi_silhouette.py",
    "scripts/color_check.py",
    "scripts/acceptance_record.py",
    "scripts/stability_check.py",
    "scripts/audit_docs.py",
    "scripts/verify_all.py",
    "scripts/identity_stability.py",
    "scripts/make_benchmark_sheet.py",
    "scripts/verify_delivery.py",
    "scripts/verify_manifest.py",
    "scripts/sample_registry.py",
    "scripts/blind_eval.py",
    "scripts/hybrid_stability.py",
    "docs/PROJECT_EXECUTION_STATUS.md",
    "docs/photo_intake_workflow.md",
    "tests/test_output_verification.py",
    "tests/test_sample_registry.py",
]

EXPECTED_PYTHON = "3.11"
EXPECTED_PACKAGES = 88
EXPECTED_COMFYUI_COMMIT = "e0a32bc"
REQUIRED_CUSTOM_NODES = ["ComfyUI_IPAdapter_plus"]

REQUIRED_NODES = [
    "CheckpointLoaderSimple", "LoraLoader", "CLIPTextEncode", "EmptyLatentImage",
    "KSampler", "VAEDecode", "SaveImage", "LoadImage", "IPAdapterModelLoader",
    "CLIPVisionLoader", "IPAdapterAdvanced", "LoadBackgroundRemovalModel",
    "RemoveBackground", "InvertMask", "JoinImageWithAlpha",
    "Canny", "ControlNetLoader", "ControlNetApplyAdvanced",
]
NODE_MODEL_EXPECTATIONS = [
    ("ControlNetLoader", "control_net_name", "canny_sdxl.safetensors"),
    ("LoadBackgroundRemovalModel", "bg_removal_name", "birefnet.safetensors"),
    ("LoraLoader", "lora_name", "Designer_BlindBox-000015.safetensors"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_models(skip_hash: bool) -> list[str]:
    problems: list[str] = []
    print("=== 模型权重 ===")
    for rel, size, expected_hash, required in MODELS:
        path = ROOT / rel
        tag = "必需" if required else "候选(不采用)"
        if not path.is_file():
            message = f"{rel}（{tag}）不存在"
            problems.append(message) if required else None
            print(f"  {'❌' if required else '⚠️ '} {rel} 缺失 [{tag}]")
            continue
        actual_size = path.stat().st_size
        size_ok = actual_size == size
        hash_note = ""
        if not skip_hash:
            actual_hash = sha256(path)
            hash_ok = actual_hash == expected_hash
            hash_note = "SHA256 ✅" if hash_ok else f"SHA256 ❌ 实际 {actual_hash[:16]}…"
            if not hash_ok and required:
                problems.append(f"{rel} SHA256 不匹配")
        if not size_ok and required:
            problems.append(f"{rel} 体积不符：{actual_size} != {size}")
        print(f"  {'✅' if size_ok else '❌'} {rel}  {actual_size:,} B  {hash_note}")
    return problems


def check_deliverables() -> list[str]:
    problems: list[str] = []
    print("=== 交付文件 ===")
    missing = [rel for rel in DELIVERABLES if not (ROOT / rel).is_file()]
    for rel in DELIVERABLES:
        if rel in missing:
            print(f"  ❌ {rel}")
        else:
            print(f"  ✅ {rel}")
    problems.extend(f"缺少交付文件 {rel}" for rel in missing)
    return problems


def check_environment() -> list[str]:
    problems: list[str] = []
    print("=== 环境 ===")
    version = sys.version.split()[0]
    ok = version.startswith(EXPECTED_PYTHON)
    print(f"  {'✅' if ok else '❌'} Python {version}（期望 {EXPECTED_PYTHON}.x）")
    if not ok:
        problems.append(f"Python 版本不符：{version}")

    count = len(list(metadata.distributions()))
    ok = count == EXPECTED_PACKAGES
    print(f"  {'✅' if ok else '⚠️ '} 已安装包 {count} 个（文档声明 {EXPECTED_PACKAGES}）")
    if not ok:
        problems.append(f"包数量与文档不符：{count} != {EXPECTED_PACKAGES}")

    head = ROOT / "ComfyUI" / ".git" / "HEAD"
    if head.is_file():
        try:
            import subprocess

            commit = subprocess.run(
                ["git", "-C", str(ROOT / "ComfyUI"), "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=10, check=False).stdout.strip()
            ok = commit.startswith(EXPECTED_COMFYUI_COMMIT)
            print(f"  {'✅' if ok else '⚠️ '} ComfyUI {commit}（文档声明 {EXPECTED_COMFYUI_COMMIT}）")
            if not ok:
                problems.append(f"ComfyUI 提交号与文档不符：{commit}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠️  无法读取 ComfyUI 提交号: {exc}")
    else:
        print("  ⚠️  ComfyUI/.git 不存在，跳过提交号核对")

    for name in REQUIRED_CUSTOM_NODES:
        exists = (ROOT / "ComfyUI" / "custom_nodes" / name).is_dir()
        print(f"  {'✅' if exists else '❌'} 自定义节点 {name}")
        if not exists:
            problems.append(f"缺少自定义节点 {name}")
    return problems


def check_record(path: Path) -> list[str]:
    problems: list[str] = []
    print(f"=== 验收记录核验（{path}）===")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"  ❌ 无法读取: {exc}")
        return [f"验收记录不可读: {exc}"]

    for role in ("input", "output"):
        info = data.get(role) or {}
        recorded = Path(info.get("path", ""))
        if not recorded.is_file():
            print(f"  ❌ {role} 文件不存在: {recorded}")
            problems.append(f"记录中的 {role} 文件不存在: {recorded}")
            continue
        actual_bytes = recorded.stat().st_size
        actual_hash = sha256(recorded)
        size_ok = actual_bytes == info.get("bytes")
        hash_ok = actual_hash == info.get("sha256")
        print(f"  {'✅' if size_ok else '❌'} {role} 字节数 {actual_bytes:,}"
              f"（记录 {info.get('bytes'):,}）")
        print(f"  {'✅' if hash_ok else '❌'} {role} SHA256 {actual_hash[:16]}…")
        if not size_ok:
            problems.append(f"{role} 字节数与记录不符")
        if not hash_ok:
            problems.append(f"{role} SHA256 与记录不符")

    out_info = data.get("output") or {}
    width, height = out_info.get("width"), out_info.get("height")
    if isinstance(width, int) and isinstance(height, int):
        ok = min(width, height) >= 1080
        print(f"  {'✅' if ok else '❌'} 输出尺寸 {width}×{height} ≥1080")
        if not ok:
            problems.append(f"输出尺寸 {width}×{height} 低于 1080")

    color = data.get("color") or {}
    hair, clothing = color.get("hair"), color.get("clothing")
    for name, value in (("发色", hair), ("服装主色", clothing)):
        if value is None:
            print(f"  ⚠️  {name} 未记录")
            continue
        print(f"  ℹ️  {name} 相似度 {value:.3f} "
              f"（{'达标' if value >= 0.9 else '不达标'}，这是结果不是记录错误）")

    elapsed = data.get("elapsed_seconds")
    if isinstance(elapsed, (int, float)):
        print(f"  ℹ️  端到端耗时 {elapsed:.2f} s"
              f"（{'达标' if elapsed <= 30 else '超 30 s 目标'}，同上）")

    memory = data.get("memory") or {}
    if memory.get("available"):
        print(f"  ✅ 峰值内存已记录: {memory.get('peak')}")
    else:
        print(f"  ⚠️  内存未采集: {memory.get('reason')}")
    print("  （核验的是**记录是否属实**，不是结果是否达标）")
    return problems


def check_comfyui(base_url: str) -> list[str]:
    problems: list[str] = []
    print(f"=== ComfyUI 服务（{base_url}）===")
    try:
        with urllib.request.urlopen(f"{base_url}/object_info", timeout=30) as response:
            info = json.load(response)
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠️  服务不可达，跳过节点核对: {exc}")
        return problems

    missing = [node for node in REQUIRED_NODES if node not in info]
    print(f"  {'✅' if not missing else '❌'} 工作流所需节点 {len(REQUIRED_NODES) - len(missing)}/{len(REQUIRED_NODES)}")
    for node in missing:
        print(f"      ❌ 缺少节点 {node}")
    problems.extend(f"缺少节点 {node}" for node in missing)

    for node, port, expected in NODE_MODEL_EXPECTATIONS:
        spec = info.get(node, {}).get("input", {}).get("required", {}).get(port)
        if spec is None:
            continue
        options = spec[0] if isinstance(spec[0], list) else (spec[1] or {}).get("options")
        if options is None:
            continue
        ok = expected in options
        print(f"  {'✅' if ok else '❌'} {node}.{port} 含 {expected}")
        if not ok:
            problems.append(f"{node}.{port} 未识别 {expected}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-hash", action="store_true", help="跳过 SHA256（快速检查）")
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("--check-record", default=None,
                        help="回读验收记录的 JSON sidecar，核对其数字是否与磁盘现实一致")
    args = parser.parse_args()

    problems: list[str] = []
    problems += check_environment()
    print()
    problems += check_deliverables()
    print()
    problems += check_models(args.skip_hash)
    print()
    problems += check_comfyui(args.comfy_url)

    if args.check_record:
        print()
        record_path = Path(args.check_record)
        if not record_path.is_file() and not record_path.is_absolute():
            record_path = ROOT / record_path
        problems += check_record(record_path)

    print()
    if problems:
        print(f"=== 核验未通过：{len(problems)} 项问题 ===")
        for item in problems:
            print(f"  - {item}")
        return 1
    print("=== 核验通过：环境、交付文件、模型权重、工作流节点全部与文档一致 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
