from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMFYUI_DIR = PROJECT_ROOT / "ComfyUI"
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_DIR = PROJECT_ROOT / "models"
WORKFLOWS_DIR = PROJECT_ROOT / "workflows"
INPUTS_DIR = PROJECT_ROOT / "inputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"
TMP_DIR = PROJECT_ROOT / "tmp"
PREPROCESSED_DIR = TMP_DIR / "preprocessed"

CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
LORAS_DIR = MODELS_DIR / "loras"
CONTROLNET_DIR = MODELS_DIR / "controlnet"
CLIP_VISION_DIR = MODELS_DIR / "clip_vision"

EXTRA_MODEL_PATHS_CONFIG = CONFIG_DIR / "extra_model_paths.yaml"

COMFY_HOST = os.environ.get("CHIBITOY_HOST", "127.0.0.1")
COMFY_PORT = int(os.environ.get("CHIBITOY_PORT", "8188"))
COMFY_BASE_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"

SDXL_BASE_CKPT = "sd_xl_base_1.0.safetensors"

SDXL_RESOLUTIONS = {
    "square_1024": (1024, 1024),
    "portrait_896x1152": (896, 1152),
    "portrait_832x1216": (832, 1216),
}

FORMAL_OUTPUT_MIN_SIZE = (1080, 1080)

RECOMMENDED_OUTPUT_SIZE = (1088, 1088)


def ensure_runtime_dirs() -> None:
    """确保运行时目录存在"""
    for directory in (WORKFLOWS_DIR, INPUTS_DIR, OUTPUTS_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def unique_path(directory: Path, filename: str) -> Path:
    name = Path(filename).name
    candidate = directory / name
    if not candidate.exists():
        return candidate
    stem, suffix = Path(name).stem, Path(name).suffix
    counter = 1
    while True:
        candidate = directory / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1
