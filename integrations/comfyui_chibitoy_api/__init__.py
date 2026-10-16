"""ChibiToy 云端 API 节点（ComfyUI 自定义节点）

作用：让 ComfyUI 工作流**直接调用云端生图 API**，不加载任何本地扩散模型。
也就是说，这条工作流只需要 ComfyUI 本体 + 本节点，**不需要 SDXL / LoRA / ControlNet**。

安装（一次性）：把本目录复制或软链到 ComfyUI 的 custom_nodes 下，然后重启 ComfyUI。

    ln -s "$(pwd)/integrations/comfyui_chibitoy_api" ComfyUI/custom_nodes/chibitoy_api

配置：读取项目根的 config/chibitoy.yaml（密钥只从该文件读，不打印）。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


class ChibiToyApiGenerate:
    """把输入图像交给云端 API 重绘为 Q 版潮玩手办，并把结果作为 IMAGE 返回。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "appearance": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "size": ("INT", {"default": 1088, "min": 512, "max": 2048, "step": 8}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "ChibiToy"

    def generate(self, image, appearance, prompt, size=1088):
        import numpy as np
        import torch
        from PIL import Image

        from chibitoy import apibackend, apiconfig

        # ComfyUI 的 IMAGE 是 [B,H,W,C] 的 0..1 张量，取第一张转成 PNG 临时文件
        array = (image[0].cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
        temp_dir = PROJECT_ROOT / "tmp" / "comfy_api"
        temp_dir.mkdir(parents=True, exist_ok=True)
        source = temp_dir / "input.png"
        Image.fromarray(array).save(source)

        config = apiconfig.load(require_key=True)      # 缺密钥会给出填写指引并报错
        backend = apibackend.ApiBackend(config)
        request = backend.build_request(source, appearance=appearance)
        if prompt.strip():
            request.prompt = prompt.strip()
        # 云端不支持指定随机种子，重复运行即为独立采样

        out_dir = PROJECT_ROOT / "outputs" / "comfy_api"
        result = backend.generate(request, output_dir=out_dir)
        with Image.open(result.saved_paths[0]) as handle:
            rendered = handle.convert("RGB")

        # 转回 IMAGE 张量
        data = np.asarray(rendered).astype("float32") / 255.0
        tensor = torch.from_numpy(data)[None,]
        print(f"[ChibiToy] 云端生成完成，用时 {result.elapsed_seconds:.1f}s -> {result.saved_paths[0]}")
        return (tensor,)


NODE_CLASS_MAPPINGS = {"ChibiToyApiGenerate": ChibiToyApiGenerate}
NODE_DISPLAY_NAME_MAPPINGS = {"ChibiToyApiGenerate": "ChibiToy 云端 API 生图"}
