from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_CLIP_VISION = "models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"

_MEAN = (0.4815, 0.4578, 0.4082)
_STD = (0.2686, 0.2613, 0.2758)
_IMAGE_SIZE = 224


def strip_vision_prefix(state_dict: dict) -> dict:
    return {
        (key[len("vision_model."):] if key.startswith("vision_model.") else key): value
        for key, value in state_dict.items()
        if not key.endswith("position_ids")
    }


def load_encoder(ckpt: str | Path = DEFAULT_CLIP_VISION):
    """加载 CLIP 视觉编码器。返回 ``(model, torch_module)``。"""
    import torch
    from safetensors.torch import load_file
    from transformers import CLIPVisionConfig, CLIPVisionModel

    state = strip_vision_prefix(load_file(str(ckpt)))
    config = CLIPVisionConfig(
        hidden_size=1280, intermediate_size=5120, num_hidden_layers=32,
        num_attention_heads=16, image_size=_IMAGE_SIZE, patch_size=14,
        projection_dim=1024, num_channels=3,
    )
    model = CLIPVisionModel(config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"CLIP 视觉权重未完全加载，缺失 {len(missing)} 项（前 3: {missing[:3]}）")
    del unexpected
    model.eval()
    return model, torch


def embed(model, torch_module, path: str | Path) -> "np.ndarray":
    """把一张图编码成单位向量。"""
    with Image.open(path) as image:
        resized = image.convert("RGB").resize((_IMAGE_SIZE, _IMAGE_SIZE), Image.LANCZOS)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch_module.from_numpy(array).permute(2, 0, 1)
    tensor = (tensor - torch_module.tensor(_MEAN).view(3, 1, 1)) / torch_module.tensor(_STD).view(3, 1, 1)
    with torch_module.no_grad():
        output = model(pixel_values=tensor.unsqueeze(0))
    vector = output.pooler_output[0].cpu().numpy()
    return vector / np.linalg.norm(vector)


def cosine(a: "np.ndarray", b: "np.ndarray") -> float:
    return float(np.dot(a, b))


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None, "range": None}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
    }


def stability(groups: dict[str, list[str | Path]], ckpt: str | Path = DEFAULT_CLIP_VISION) -> dict:
    model, torch_module = load_encoder(ckpt)
    vectors = {
        name: [embed(model, torch_module, path) for path in paths]
        for name, paths in groups.items()
    }
    within = [
        cosine(a, b)
        for group in vectors.values()
        for a, b in itertools.combinations(group, 2)
    ]
    cross = [
        cosine(a, b)
        for (_, left), (_, right) in itertools.combinations(vectors.items(), 2)
        for a in left
        for b in right
    ]
    return {
        "within": _stats(within),
        "cross": _stats(cross),
        "vectors": {name: len(group) for name, group in vectors.items()},
    }
