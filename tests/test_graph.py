"""graph 模块的单元测试。

这些测试使用**手工固定的节点定义**，取自 ComfyUI 0.40.0 服务端真实的
/object_info 返回结果，因此不依赖服务是否运行，可离线复现。

如需用真实服务端定义回归测试，运行:
    .venv/bin/python -m pytest tests/ -m live
"""

from __future__ import annotations

import copy

import pytest

from chibitoy import graph, prompts

# --------------------------------------------------------------------------
# 节点定义 fixture —— 结构复制自真实 /object_info 响应（已精简部分候选值）
# --------------------------------------------------------------------------

SAMPLERS = ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_sde", "heun"]
SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform", "simple"]

NODE_DEFS = {
    "CheckpointLoaderSimple": {
        "input": {"required": {"ckpt_name": [["sd_xl_base_1.0.safetensors"]]}},
        "output": ["MODEL", "CLIP", "VAE"],
        "output_name": ["MODEL", "CLIP", "VAE"],
        "output_node": False,
    },
    "CLIPTextEncode": {
        "input": {
            "required": {
                "text": ["STRING", {"multiline": True, "dynamicPrompts": True}],
                "clip": ["CLIP"],
            }
        },
        "output": ["CONDITIONING"],
        "output_name": ["CONDITIONING"],
        "output_node": False,
    },
    "EmptyLatentImage": {
        "input": {
            "required": {
                "width": ["INT", {"default": 1024, "min": 16, "max": 16384}],
                "height": ["INT", {"default": 1024, "min": 16, "max": 16384}],
                "batch_size": ["INT", {"default": 1, "min": 1, "max": 4096}],
            }
        },
        "output": ["LATENT"],
        "output_name": ["LATENT"],
        "output_node": False,
    },
    "KSampler": {
        "input": {
            "required": {
                "model": ["MODEL"],
                "seed": ["INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}],
                "steps": ["INT", {"default": 20, "min": 1, "max": 10000}],
                "cfg": ["FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}],
                "sampler_name": [SAMPLERS],
                "scheduler": [SCHEDULERS],
                "positive": ["CONDITIONING"],
                "negative": ["CONDITIONING"],
                "latent_image": ["LATENT"],
                "denoise": ["FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}],
            }
        },
        "output": ["LATENT"],
        "output_name": ["LATENT"],
        "output_node": False,
    },
    "VAEDecode": {
        "input": {"required": {"samples": ["LATENT"], "vae": ["VAE"]}},
        "output": ["IMAGE"],
        "output_name": ["IMAGE"],
        "output_node": False,
    },
    "LoadImage": {
        "input": {"required": {"image": [["example.png", "reference.png", "base.png"]]}},
        "output": ["IMAGE", "MASK"],
        "output_name": ["IMAGE", "MASK"],
        "output_node": False,
    },
    "VAEEncode": {
        "input": {"required": {"pixels": ["IMAGE"], "vae": ["VAE"]}},
        "output": ["LATENT"],
        "output_name": ["LATENT"],
        "output_node": False,
    },
    "LoraLoader": {
        "input": {
            "required": {
                "model": ["MODEL"],
                "clip": ["CLIP"],
                "lora_name": [["example_lora.safetensors"]],
                "strength_model": ["FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0}],
                "strength_clip": ["FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0}],
            }
        },
        "output": ["MODEL", "CLIP"],
        "output_name": ["MODEL", "CLIP"],
        "output_node": False,
    },
    "SaveImage": {
        "input": {
            "required": {
                "images": ["IMAGE"],
                "filename_prefix": ["STRING", {"default": "ComfyUI"}],
            }
        },
        "output": [],
        "output_name": [],
        "output_node": True,
    },
    # IP-Adapter 三件套。注意 IPAdapterAdvanced 故意保留**可选输入**
    # image_negative / attn_mask（顺序在 clip_vision 之前）——这是
    # 端口序号错位缺陷的触发条件，见 test_ui_optional_input_slot_index。
    "IPAdapterModelLoader": {
        "input": {
            "required": {"ipadapter_file": [["ip-adapter-plus-face_sdxl_vit-h.safetensors"]]}
        },
        "output": ["IPADAPTER"],
        "output_name": ["IPADAPTER"],
        "output_node": False,
    },
    "CLIPVisionLoader": {
        "input": {
            "required": {"clip_name": [["CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"]]}
        },
        "output": ["CLIP_VISION"],
        "output_name": ["CLIP_VISION"],
        "output_node": False,
    },
    "IPAdapterAdvanced": {
        "input": {
            "required": {
                "model": ["MODEL"],
                "ipadapter": ["IPADAPTER"],
                "image": ["IMAGE"],
                "weight": ["FLOAT", {"default": 1.0, "min": -1, "max": 5, "step": 0.05}],
                "weight_type": [[
                    "linear", "ease in", "ease out", "ease in-out", "reverse in-out",
                    "weak input", "weak output", "weak middle", "strong middle",
                    "style transfer", "composition", "strong style transfer",
                ]],
                "combine_embeds": [["concat", "add", "subtract", "average", "norm average"]],
                "start_at": ["FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}],
                "end_at": ["FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}],
                "embeds_scaling": [["V only", "K+V", "K+V w/ C penalty", "K+mean(V) w/ C penalty"]],
            },
            "optional": {
                "image_negative": ["IMAGE"],
                "attn_mask": ["MASK"],
                "clip_vision": ["CLIP_VISION"],
            },
        },
        "output": ["MODEL"],
        "output_name": ["MODEL"],
        "output_node": False,
    },
    # 背景分割三件套。LoadBackgroundRemovalModel 使用 ComfyUI 0.40+ 的
    # **新式 COMBO** 形式 ["COMBO", {"options": [...]}]，见 test_combo_options_*。
    "LoadBackgroundRemovalModel": {
        "input": {
            "required": {
                "bg_removal_name": ["COMBO", {"options": ["birefnet.safetensors"]}]
            }
        },
        "output": ["BACKGROUND_REMOVAL"],
        "output_name": ["bg_model"],
        "output_node": False,
    },
    "RemoveBackground": {
        "input": {
            "required": {
                "bg_removal_model": ["BACKGROUND_REMOVAL"],
                "image": ["IMAGE"],
            }
        },
        "output": ["MASK"],
        "output_name": ["mask"],
        "output_node": False,
    },
    "JoinImageWithAlpha": {
        "input": {"required": {"image": ["IMAGE"], "alpha": ["MASK"]}},
        "output": ["IMAGE"],
        "output_name": ["IMAGE"],
        "output_node": False,
    },
    "ImageScale": {
        "input": {
            "required": {
                "image": ["IMAGE"],
                "upscale_method": [["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]],
                "width": ["INT", {"default": 512, "min": 0, "max": 16384}],
                "height": ["INT", {"default": 512, "min": 0, "max": 16384}],
                "crop": [["disabled", "center"]],
            }
        },
        "output": ["IMAGE"],
        "output_name": ["IMAGE"],
        "output_node": False,
    },
    "ImageScaleToMaxDimension": {
        "input": {
            "required": {
                "image": ["IMAGE"],
                "upscale_method": [["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]],
                "largest_size": ["INT", {"default": 512, "min": 0, "max": 16384}],
            }
        },
        "output": ["IMAGE"],
        "output_name": ["IMAGE"],
        "output_node": False,
    },
    "EmptyImage": {
        "input": {
            "required": {
                "width": ["INT", {"default": 512, "min": 1, "max": 16384}],
                "height": ["INT", {"default": 512, "min": 1, "max": 16384}],
                "batch_size": ["INT", {"default": 1, "min": 1, "max": 4096}],
                "color": ["INT", {"default": 0, "min": 0, "max": 16777215}],
            }
        },
        "output": ["IMAGE"],
        "output_name": ["IMAGE"],
        "output_node": False,
    },
    "ImageCompositeMasked": {
        "input": {
            "required": {
                "destination": ["IMAGE"],
                "source": ["IMAGE"],
                "x": ["INT", {"default": 0, "min": 0, "max": 16384}],
                "y": ["INT", {"default": 0, "min": 0, "max": 16384}],
                "resize_source": ["BOOLEAN", {"default": False}],
            },
            "optional": {"mask": ["MASK"]},
        },
        "output": ["IMAGE"],
        "output_name": ["IMAGE"],
        "output_node": False,
    },
    "InvertMask": {
        "input": {"required": {"mask": ["MASK"]}},
        "output": ["MASK"],
        "output_name": ["MASK"],
        "output_node": False,
    },
    "Canny": {
        "input": {
            "required": {
                "image": ["IMAGE"],
                "low_threshold": ["FLOAT", {"default": 0.4, "min": 0.01, "max": 0.99, "step": 0.01}],
                "high_threshold": ["FLOAT", {"default": 0.8, "min": 0.01, "max": 0.99, "step": 0.01}],
            }
        },
        "output": ["IMAGE"],
        "output_name": ["IMAGE"],
        "output_node": False,
    },
    "ControlNetLoader": {
        "input": {"required": {"control_net_name": [["canny_sdxl.safetensors"]]}},
        "output": ["CONTROL_NET"],
        "output_name": ["CONTROL_NET"],
        "output_node": False,
    },
    "ControlNetApplyAdvanced": {
        "input": {
            "required": {
                "positive": ["CONDITIONING"],
                "negative": ["CONDITIONING"],
                "control_net": ["CONTROL_NET"],
                "image": ["IMAGE"],
                "strength": ["FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}],
                "start_percent": ["FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}],
                "end_percent": ["FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}],
            }
        },
        "output": ["CONDITIONING", "CONDITIONING"],
        "output_name": ["positive", "negative"],
        "output_node": False,
    },
}


@pytest.fixture()
def workflow() -> dict:
    return graph.build_sdxl_txt2img(
        ckpt_name="sd_xl_base_1.0.safetensors",
        positive_prompt="chibi toy figure",
        negative_prompt="photorealistic",
    )


# --------------------------------------------------------------------------
# 构建
# --------------------------------------------------------------------------

def test_build_produces_expected_nodes(workflow):
    classes = {node["class_type"] for node in workflow.values()}
    assert classes == {
        "CheckpointLoaderSimple",
        "CLIPTextEncode",
        "EmptyLatentImage",
        "KSampler",
        "VAEDecode",
        "SaveImage",
    }


def test_build_respects_parameters(workflow):
    # 覆盖默认参数后应当体现在 KSampler 上
    wf = graph.build_sdxl_txt2img(
        ckpt_name="sd_xl_base_1.0.safetensors",
        positive_prompt="p",
        negative_prompt="n",
        seed=123,
        steps=7,
        cfg=3.5,
        width=832,
        height=1216,
    )
    sampler = next(n for n in wf.values() if n["class_type"] == "KSampler")
    assert sampler["inputs"]["seed"] == 123
    assert sampler["inputs"]["steps"] == 7
    assert sampler["inputs"]["cfg"] == 3.5

    latent = next(n for n in wf.values() if n["class_type"] == "EmptyLatentImage")
    assert (latent["inputs"]["width"], latent["inputs"]["height"]) == (832, 1216)


def test_build_does_not_share_mutable_state(workflow):
    first = graph.build_sdxl_txt2img("a", "p", "n")
    second = graph.build_sdxl_txt2img("b", "p", "n")
    first["4"]["inputs"]["ckpt_name"] = "mutated"
    assert second["4"]["inputs"]["ckpt_name"] == "b"


# --------------------------------------------------------------------------
# API 格式校验
# --------------------------------------------------------------------------

def test_valid_workflow_passes(workflow):
    assert graph.validate_api_workflow(workflow, NODE_DEFS) == []


def test_detects_unknown_class_type(workflow):
    workflow["4"]["class_type"] = "NotARealNode"
    problems = graph.validate_api_workflow(workflow, NODE_DEFS)
    assert any("未知的 class_type" in p for p in problems)


def test_detects_missing_required_input(workflow):
    del workflow["3"]["inputs"]["steps"]
    problems = graph.validate_api_workflow(workflow, NODE_DEFS)
    assert any("缺少必填输入 'steps'" in p for p in problems)


def test_detects_unknown_input_name(workflow):
    workflow["3"]["inputs"]["not_a_real_input"] = 1
    problems = graph.validate_api_workflow(workflow, NODE_DEFS)
    assert any("存在未知输入 'not_a_real_input'" in p for p in problems)


def test_detects_invalid_combo_value(workflow):
    workflow["3"]["inputs"]["sampler_name"] = "definitely_not_a_sampler"
    problems = graph.validate_api_workflow(workflow, NODE_DEFS)
    assert any("不在候选列表中" in p for p in problems)


def test_detects_type_mismatch_on_link(workflow):
    # 把 KSampler 的 latent_image 指向 CLIPTextEncode 的 CONDITIONING 输出
    workflow["3"]["inputs"]["latent_image"] = ["6", 0]
    problems = graph.validate_api_workflow(workflow, NODE_DEFS)
    assert any("需要 LATENT" in p and "CONDITIONING" in p for p in problems)


def test_detects_dangling_link(workflow):
    workflow["3"]["inputs"]["model"] = ["999", 0]
    problems = graph.validate_api_workflow(workflow, NODE_DEFS)
    assert any("不存在的节点" in p for p in problems)


def test_detects_bad_slot_index(workflow):
    workflow["3"]["inputs"]["model"] = ["4", 99]
    problems = graph.validate_api_workflow(workflow, NODE_DEFS)
    assert any("不存在的输出序号" in p for p in problems)


def test_detects_wrong_literal_type(workflow):
    workflow["3"]["inputs"]["steps"] = "twenty"
    problems = graph.validate_api_workflow(workflow, NODE_DEFS)
    assert any("应为 INT" in p for p in problems)


def test_detects_missing_output_node(workflow):
    del workflow["9"]
    problems = graph.validate_api_workflow(workflow, NODE_DEFS)
    assert any("没有输出节点" in p for p in problems)


# --------------------------------------------------------------------------
# UI 格式转换与校验
# --------------------------------------------------------------------------

def test_ui_conversion_validates(workflow):
    ui = graph.api_to_ui_workflow(workflow, NODE_DEFS)
    assert graph.validate_ui_workflow(ui, NODE_DEFS) == []


def test_ui_top_level_schema(workflow):
    ui = graph.api_to_ui_workflow(workflow, NODE_DEFS)
    for key in ("last_node_id", "last_link_id", "nodes", "links", "groups", "config", "extra", "version"):
        assert key in ui, f"缺少顶层字段 {key}"
    assert ui["version"] == 0.4
    assert len(ui["links"]) == 9  # SDXL 文生图共 9 条连线
    assert all(len(link) == 6 for link in ui["links"])


def test_ui_ksampler_widget_order_matches_comfyui_convention(workflow):
    """KSampler 的控件顺序必须是
    [seed, control_after_generate, steps, cfg, sampler_name, scheduler, denoise]。

    依据 ComfyUI 自带 default_workflow.json 中同样的 7 元素顺序。
    """
    wf = graph.build_sdxl_txt2img(
        "sd_xl_base_1.0.safetensors", "p", "n",
        seed=42, steps=30, cfg=7.5, sampler_name="dpmpp_2m", scheduler="karras",
    )
    ui = graph.api_to_ui_workflow(wf, NODE_DEFS)
    node = next(n for n in ui["nodes"] if n["type"] == "KSampler")
    assert node["widgets_values"] == [42, "fixed", 30, 7.5, "dpmpp_2m", "karras", 1.0]


def test_ui_widget_values_for_other_nodes(workflow):
    ui = graph.api_to_ui_workflow(workflow, NODE_DEFS)
    by_type = {n["type"]: n for n in ui["nodes"]}
    assert by_type["CheckpointLoaderSimple"]["widgets_values"] == ["sd_xl_base_1.0.safetensors"]
    assert by_type["EmptyLatentImage"]["widgets_values"] == [1024, 1024, 1]
    assert by_type["SaveImage"]["widgets_values"] == ["chibitoy"]
    assert by_type["VAEDecode"]["widgets_values"] == []


def test_ui_links_reference_real_slots(workflow):
    ui = graph.api_to_ui_workflow(workflow, NODE_DEFS)
    nodes = {n["id"]: n for n in ui["nodes"]}
    for link_id, src, src_slot, dst, dst_slot, link_type in ui["links"]:
        assert src_slot < len(nodes[src]["outputs"])
        assert dst_slot < len(nodes[dst]["inputs"])
        assert nodes[src]["outputs"][src_slot]["type"] == link_type
        assert nodes[dst]["inputs"][dst_slot]["type"] == link_type


def test_ui_validation_catches_broken_link(workflow):
    ui = graph.api_to_ui_workflow(workflow, NODE_DEFS)
    ui["links"][0][3] = 99999  # 指向不存在的目标节点
    problems = graph.validate_ui_workflow(ui, NODE_DEFS)
    assert any("不存在" in p for p in problems)


def test_ui_validation_catches_type_mismatch(workflow):
    ui = graph.api_to_ui_workflow(workflow, NODE_DEFS)
    copy_ui = copy.deepcopy(ui)
    copy_ui["links"][0][5] = "WRONG_TYPE"
    problems = graph.validate_ui_workflow(copy_ui, NODE_DEFS)
    assert any("不一致" in p for p in problems)


def test_ui_validation_catches_undeclared_link_reference(workflow):
    ui = graph.api_to_ui_workflow(workflow, NODE_DEFS)
    for node in ui["nodes"]:
        if node["inputs"]:
            node["inputs"][0]["link"] = 424242
            break
    problems = graph.validate_ui_workflow(ui, NODE_DEFS)
    assert any("未声明的连线" in p for p in problems)


def test_ui_node_ids_unique(workflow):
    ui = graph.api_to_ui_workflow(workflow, NODE_DEFS)
    ids = [n["id"] for n in ui["nodes"]]
    assert len(ids) == len(set(ids))


def test_topological_order_places_loader_first(workflow):
    ui = graph.api_to_ui_workflow(workflow, NODE_DEFS)
    by_id = {n["id"]: n for n in ui["nodes"]}
    loader = next(n for n in ui["nodes"] if n["type"] == "CheckpointLoaderSimple")
    save = next(n for n in ui["nodes"] if n["type"] == "SaveImage")
    sampler = next(n for n in ui["nodes"] if n["type"] == "KSampler")
    assert loader["order"] < sampler["order"] < save["order"]
    assert len(by_id) == 7


# --------------------------------------------------------------------------
# 图生图（img2img）
# --------------------------------------------------------------------------

@pytest.fixture()
def img2img_workflow() -> dict:
    return graph.build_sdxl_img2img(
        ckpt_name="sd_xl_base_1.0.safetensors",
        input_image="example.png",
        positive_prompt="chibi toy figure",
        negative_prompt="photorealistic",
        denoise=0.55,
    )


def test_img2img_uses_loadimage_and_vaeencode(img2img_workflow):
    classes = {node["class_type"] for node in img2img_workflow.values()}
    assert "LoadImage" in classes
    assert "VAEEncode" in classes
    assert "EmptyLatentImage" not in classes  # img2img 不应生成空 latent


def test_img2img_validates_api_and_ui(img2img_workflow):
    assert graph.validate_api_workflow(img2img_workflow, NODE_DEFS) == []
    ui = graph.api_to_ui_workflow(img2img_workflow, NODE_DEFS)
    assert graph.validate_ui_workflow(ui, NODE_DEFS) == []


def test_img2img_links_loadimage_to_vaeencode(img2img_workflow):
    vae_encode = next(n for n in img2img_workflow.values() if n["class_type"] == "VAEEncode")
    # 原图必须**先经 ImageScale 缩放**再进 VAE：手机原图直接进 VAE 会爆显存
    assert vae_encode["inputs"]["pixels"] == ["10", 0]
    scale = img2img_workflow["10"]
    assert scale["class_type"] == "ImageScale"
    assert scale["inputs"]["image"] == ["1", 0]
    assert scale["inputs"]["width"] == scale["inputs"]["height"] == 1088


def test_img2img_uses_checkpoint_vae(img2img_workflow):
    vae_encode = next(n for n in img2img_workflow.values() if n["class_type"] == "VAEEncode")
    # CheckpointLoaderSimple 输出槽 2 为 VAE
    assert vae_encode["inputs"]["vae"] == ["2", 2]


def test_img2img_respects_denoise():
    wf = graph.build_sdxl_img2img("c", "example.png", "p", "n", denoise=0.75)
    sampler = next(n for n in wf.values() if n["class_type"] == "KSampler")
    assert sampler["inputs"]["denoise"] == 0.75


def test_img2img_does_not_share_state_with_txt2img():
    a = graph.build_sdxl_txt2img("ckpt_a", "p", "n")
    b = graph.build_sdxl_img2img("ckpt_b", "example.png", "p", "n")
    a["4"]["inputs"]["ckpt_name"] = "mutated"
    assert b["2"]["inputs"]["ckpt_name"] == "ckpt_b"


def test_img2img_ui_node_count(img2img_workflow):
    ui = graph.api_to_ui_workflow(img2img_workflow, NODE_DEFS)
    assert len(ui["nodes"]) == 9  # +LoadImage +VAEEncode +ImageScale，-EmptyLatentImage
    # 11 条连线: LoadImage->VAEEncode, ckpt->VAEEncode, ckpt->CLIP×2, ckpt->KSampler,
    # pos/neg->KSampler, VAEEncode->KSampler, KSampler->VAEDecode, ckpt->VAEDecode,
    # VAEDecode->SaveImage
    assert len(ui["links"]) == 12   # 新增 LoadImage->ImageScale


# --------------------------------------------------------------------------
# img2img + LoRA
# --------------------------------------------------------------------------

def test_img2img_with_lora_inserts_loader():
    wf = graph.build_sdxl_img2img(
        "sd_xl_base_1.0.safetensors", "example.png", "p", "n",
        lora_name="example_lora.safetensors",
        lora_strength_model=0.8, lora_strength_clip=0.7,
    )
    classes = {n["class_type"] for n in wf.values()}
    assert "LoraLoader" in classes

    loader = next(n for n in wf.values() if n["class_type"] == "LoraLoader")
    assert loader["inputs"]["lora_name"] == "example_lora.safetensors"
    assert loader["inputs"]["strength_model"] == 0.8
    assert loader["inputs"]["strength_clip"] == 0.7

    # KSampler 的 model 与 CLIPTextEncode 的 clip 都应来自 LoraLoader
    sampler = next(n for n in wf.values() if n["class_type"] == "KSampler")
    assert sampler["inputs"]["model"] == ["9", 0]
    for encode in (n for n in wf.values() if n["class_type"] == "CLIPTextEncode"):
        assert encode["inputs"]["clip"] == ["9", 1]


def test_img2img_without_lora_has_no_loader(img2img_workflow):
    classes = {n["class_type"] for n in img2img_workflow.values()}
    assert "LoraLoader" not in classes
    sampler = next(n for n in img2img_workflow.values() if n["class_type"] == "KSampler")
    assert sampler["inputs"]["model"] == ["2", 0]


def test_img2img_lora_validates():
    wf = graph.build_sdxl_img2img(
        "sd_xl_base_1.0.safetensors", "example.png", "p", "n",
        lora_name="example_lora.safetensors",
    )
    assert graph.validate_api_workflow(wf, NODE_DEFS) == []
    ui = graph.api_to_ui_workflow(wf, NODE_DEFS)
    assert graph.validate_ui_workflow(ui, NODE_DEFS) == []
    assert len(ui["nodes"]) == 10  # 多一个 LoraLoader（基础 9 + LoRA）
    assert len(ui["links"]) == 14  # 基础 12（含 LoadImage->ImageScale）+ LoRA 带来的净增 2


# --------------------------------------------------------------------------
# 提示词
# --------------------------------------------------------------------------

def test_positive_prompt_embeds_appearance():
    text = prompts.build_positive("short black hair, red hoodie")
    assert "short black hair, red hoodie" in text
    assert "1.75 head-to-body proportion" in text
    assert "matte PVC" in text


def test_positive_prompt_avoids_white_bleed():
    """回归：正向提示词里不得出现 "white"。

    实测（阶段 1B-3B-2 阶段A）：原本连续三次 "pure white background / isolated on
    pure white / seamless white studio backdrop"，结果**要求黑发黑衣却生成白发白衣**
    —— 白色 token 渗漏到了主体。背景已改由 BiRefNet 真实分割负责，
    提示词不再描述白色背景。
    """
    assert prompts.POSITIVE_PROMPT.lower().count("white") == 0
    assert "studio backdrop" in prompts.POSITIVE_PROMPT


# --------------------------------------------------------------------------
# 可选: 用真实运行中的 ComfyUI 定义回归
# --------------------------------------------------------------------------

@pytest.mark.live
def test_matches_live_server_definitions(workflow):
    """用真实服务端的 /object_info 校验工作流。需要 ComfyUI 正在运行。"""
    from chibitoy.client import ComfyUIClient

    client = ComfyUIClient()
    if not client.is_available():
        pytest.skip("ComfyUI 服务未启动")

    live_defs = {}
    for class_type in {n["class_type"] for n in workflow.values()}:
        info = client.object_info(class_type)
        live_defs.update(info)

    assert graph.validate_api_workflow(workflow, live_defs) == []
    ui = graph.api_to_ui_workflow(workflow, live_defs)
    assert graph.validate_ui_workflow(ui, live_defs) == []


@pytest.mark.live
def test_img2img_matches_live_server(tmp_path):
    """上传一张小图后，用真实服务端定义校验 img2img 工作流。"""
    from PIL import Image

    from chibitoy.client import ComfyUIClient

    client = ComfyUIClient()
    if not client.is_available():
        pytest.skip("ComfyUI 服务未启动")

    image_path = tmp_path / "live_img2img_test.png"
    Image.new("RGB", (64, 64), (255, 0, 0)).save(image_path)
    uploaded = client.upload_image(image_path)

    workflow = graph.build_sdxl_img2img(
        ckpt_name="sd_xl_base_1.0.safetensors",
        input_image=uploaded,
        positive_prompt="chibi toy figure",
        negative_prompt="photorealistic",
        denoise=0.55,
    )
    live_defs = {}
    for class_type in {n["class_type"] for n in workflow.values()}:
        live_defs.update(client.object_info(class_type))

    assert graph.validate_api_workflow(workflow, live_defs) == []
    ui = graph.api_to_ui_workflow(workflow, live_defs)
    assert graph.validate_ui_workflow(ui, live_defs) == []


# ---------------------------------------------------------------------------
# IP-Adapter 接入 txt2img（阶段 A/B：完整公仔 + 身份特征注入）
# ---------------------------------------------------------------------------

def test_txt2img_defaults_unchanged_without_optional_components():
    """不传 LoRA / IP-Adapter 时，输出必须与旧版逐类一致（6 类节点）。"""
    wf = graph.build_sdxl_txt2img("sd_xl_base_1.0.safetensors", "p", "n")
    classes = {n["class_type"] for n in wf.values()}
    assert classes == {
        "CheckpointLoaderSimple", "CLIPTextEncode", "EmptyLatentImage",
        "KSampler", "VAEDecode", "SaveImage",
    }
    assert wf["3"]["inputs"]["model"] == ["4", 0]


def test_txt2img_with_lora_rewires_clip_and_model():
    wf = graph.build_sdxl_txt2img("sd_xl_base_1.0.safetensors", "p", "n", lora_name="example_lora.safetensors")
    assert "LoraLoader" in {n["class_type"] for n in wf.values()}
    assert wf["3"]["inputs"]["model"] == ["10", 0]
    assert wf["6"]["inputs"]["clip"] == ["10", 1]
    assert wf["7"]["inputs"]["clip"] == ["10", 1]
    assert graph.validate_api_workflow(wf, NODE_DEFS) == []


def test_txt2img_with_ipadapter_inserts_chain_before_sampler():
    wf = graph.build_sdxl_txt2img(
        "sd_xl_base_1.0.safetensors", "p", "n",
        ipadapter_name="ip-adapter-plus-face_sdxl_vit-h.safetensors", ipadapter_image="example.png",
        clip_vision_name="CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors")
    classes = {n["class_type"] for n in wf.values()}
    assert {"LoadImage", "CLIPVisionLoader", "IPAdapterModelLoader",
            "IPAdapterAdvanced"} <= classes
    # 采样器的模型来自 IP-Adapter 输出，而不是直接来自 checkpoint
    assert wf["3"]["inputs"]["model"] == ["14", 0]
    assert wf["14"]["inputs"]["model"] == ["4", 0]
    assert wf["14"]["inputs"]["clip_vision"] == ["12", 0]
    assert wf["14"]["inputs"]["image"] == ["11", 0]
    assert graph.validate_api_workflow(wf, NODE_DEFS) == []


def test_txt2img_with_lora_and_ipadapter_chains_correctly():
    """链路顺序必须是 checkpoint -> LoRA -> IP-Adapter -> KSampler。"""
    wf = graph.build_sdxl_txt2img(
        "sd_xl_base_1.0.safetensors", "p", "n", lora_name="example_lora.safetensors",
        ipadapter_name="ip-adapter-plus-face_sdxl_vit-h.safetensors", ipadapter_image="example.png",
        clip_vision_name="CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors")
    assert wf["10"]["inputs"]["model"] == ["4", 0]
    assert wf["14"]["inputs"]["model"] == ["10", 0]
    assert wf["3"]["inputs"]["model"] == ["14", 0]
    assert wf["6"]["inputs"]["clip"] == ["10", 1]
    assert graph.validate_api_workflow(wf, NODE_DEFS) == []
    ui = graph.api_to_ui_workflow(wf, NODE_DEFS)
    assert graph.validate_ui_workflow(ui, NODE_DEFS) == []


def test_ui_optional_input_slot_index():
    """回归：省略可选连线输入时，UI 端口序号不得错位。

    ``IPAdapterAdvanced`` 定义了可选输入 ``image_negative`` / ``attn_mask``，
    位置在 ``clip_vision`` 之前。若 ``_slot_index`` 按节点定义**全量**计数，
    ``clip_vision`` 会被算成端口 5，而实际端口数组只有 4 个（0..3），
    于是 UI 工作流校验报「目标节点没有输入端口 5」。
    """
    wf = graph.build_sdxl_txt2img(
        "sd_xl_base_1.0.safetensors", "p", "n",
        ipadapter_name="ip-adapter-plus-face_sdxl_vit-h.safetensors", ipadapter_image="example.png",
        clip_vision_name="CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors")
    ui = graph.api_to_ui_workflow(wf, NODE_DEFS)
    ipa_node = next(n for n in ui["nodes"] if n["type"] == "IPAdapterAdvanced")
    assert [p["name"] for p in ipa_node["inputs"]] == [
        "model", "ipadapter", "image", "clip_vision"
    ]
    assert graph.validate_ui_workflow(ui, NODE_DEFS) == []


# ---------------------------------------------------------------------------
# 提示词模板类型（回归：阶段 1B-3B-2 曾因多余逗号把负向提示词写成 tuple）
# ---------------------------------------------------------------------------

def test_prompt_templates_are_nonempty_strings():
    """正向/负向提示词必须是**字符串**且非空。

    曾出现的真实错误：在括号内做隐式字符串拼接时多写一个逗号，
    整个表达式就变成 tuple，ComfyUI 校验会报「输入 'text' 应为 STRING，实际为 tuple」。
    """
    for name, value in (("POSITIVE_PROMPT", prompts.POSITIVE_PROMPT),
                        ("NEGATIVE_PROMPT", prompts.NEGATIVE_PROMPT)):
        assert isinstance(value, str), f"{name} 必须是 str，实际为 {type(value).__name__}"
        assert value.strip(), f"{name} 不得为空"


def test_build_positive_inserts_appearance_as_string():
    text = prompts.build_positive("short black hair, black t-shirt")
    assert isinstance(text, str)
    assert "short black hair, black t-shirt" in text
    assert "{appearance}" not in text      # 占位符必须已被替换


def test_negative_prompt_covers_text_and_logo_suppression():
    """需求禁止文字/Logo；负向必须覆盖衣物上的印刷文字（第 2 轮实测出现过乱码文字）。"""
    assert "logo" in prompts.NEGATIVE_PROMPT
    assert "text on shirt" in prompts.NEGATIVE_PROMPT


# ---------------------------------------------------------------------------
# object_info 的两种 COMBO 形式（回归：ComfyUI 0.40+ 新式 ["COMBO", {...}]）
# ---------------------------------------------------------------------------

def test_combo_options_supports_both_object_info_forms():
    """旧式与新式 COMBO 都必须能取出候选值。

    只认旧式会把新式候选误读成字符串 "COMBO"，于是把所有合法取值判为非法
    （实测报错：输入 'bg_removal_name' 的值 'birefnet.safetensors' 不在候选列表中
    （可选值: COMBO））。
    """
    assert graph.combo_options([["a", "b"]]) == ["a", "b"]
    assert graph.combo_options(["COMBO", {"options": ["x", "y"]}]) == ["x", "y"]
    assert graph.combo_options(["COMBO", {}]) == []
    assert graph.combo_options([]) == []


def test_new_style_combo_is_widget_not_port():
    """新式 COMBO 是控件，不得被当成必填连线端口。"""
    assert graph.is_widget_input(["COMBO", {"options": ["birefnet.safetensors"]}]) is True
    assert graph.is_widget_input([["birefnet.safetensors"]]) is True
    # forceInput 的 COMBO 仍然是连线端口
    assert graph.is_widget_input(["COMBO", {"options": ["a"], "forceInput": True}]) is False


def test_txt2img_with_bg_removal_inserts_mask_chain():
    """背景分割链路：解码图 -> RemoveBackground -> MASK -> JoinImageWithAlpha。"""
    wf = graph.build_sdxl_txt2img(
        "sd_xl_base_1.0.safetensors", "p", "n", bg_removal_model="birefnet.safetensors")
    classes = {n["class_type"] for n in wf.values()}
    assert {"LoadBackgroundRemovalModel", "RemoveBackground", "JoinImageWithAlpha"} <= classes
    assert wf["16"]["inputs"]["bg_removal_model"] == ["15", 0]
    assert wf["16"]["inputs"]["image"] == ["8", 0]
    assert wf["17"]["inputs"]["image"] == ["8", 0]
    assert wf["18"]["inputs"]["mask"] == ["16", 0]      # 前景掩码先取反
    assert wf["17"]["inputs"]["alpha"] == ["18", 0]     # 再交给 JoinImageWithAlpha
    # SaveImage 保存的是带 alpha 的合成结果，而不是原始解码图
    assert wf["9"]["inputs"]["images"] == ["17", 0]
    assert graph.validate_api_workflow(wf, NODE_DEFS) == []
    ui = graph.api_to_ui_workflow(wf, NODE_DEFS)
    assert graph.validate_ui_workflow(ui, NODE_DEFS) == []


# ---------------------------------------------------------------------------
# img2img 的 IP-Adapter（阶段 A/B：结构 / 身份分离）
# ---------------------------------------------------------------------------

def test_img2img_with_ipadapter_and_bg_removal():
    wf = graph.build_sdxl_img2img(
        "sd_xl_base_1.0.safetensors", "base.png", "p", "n",
        ipadapter_name="ip-adapter-plus-face_sdxl_vit-h.safetensors",
        ipadapter_image="reference.png",
        clip_vision_name="CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
        bg_removal_model="birefnet.safetensors")
    assert wf["6"]["inputs"]["model"] == ["14", 0]      # 采样器接 IP-Adapter 输出
    assert wf["14"]["inputs"]["model"] == ["2", 0]      # 无 LoRA 时接 checkpoint
    assert wf["8"]["inputs"]["images"] == ["17", 0]     # 保存的是带 alpha 的合成图
    assert graph.validate_api_workflow(wf, NODE_DEFS) == []
    ui = graph.api_to_ui_workflow(wf, NODE_DEFS)
    assert graph.validate_ui_workflow(ui, NODE_DEFS) == []


def test_img2img_ipadapter_reference_is_independent_of_base_image():
    """两阶段方案的关键：底图与身份参考图是**两张不同的图**。

    阶段 1 用 txt2img 产出 Q 版结构图作为底图；阶段 2 仍用**原始真人照片**
    作为身份参考。若二者被强行绑定成同一张，方案就不成立。
    """
    wf = graph.build_sdxl_img2img(
        "sd_xl_base_1.0.safetensors", "base.png", "p", "n",
        ipadapter_name="ip-adapter-plus-face_sdxl_vit-h.safetensors",
        ipadapter_image="reference.png",
        clip_vision_name="CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors")
    assert wf["1"]["inputs"]["image"] == "base.png"        # VAEEncode 的底图
    assert wf["11"]["inputs"]["image"] == "reference.png"  # IP-Adapter 的参考图
    assert graph.validate_api_workflow(wf, NODE_DEFS) == []


def test_img2img_ipadapter_with_lora_chains_after_lora():
    """同时用 LoRA 时，IP-Adapter 必须接在 LoRA 之后。"""
    wf = graph.build_sdxl_img2img(
        "sd_xl_base_1.0.safetensors", "base.png", "p", "n",
        lora_name="example_lora.safetensors",
        ipadapter_name="ip-adapter-plus-face_sdxl_vit-h.safetensors",
        ipadapter_image="reference.png",
        clip_vision_name="CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors")
    assert wf["9"]["inputs"]["model"] == ["2", 0]
    assert wf["14"]["inputs"]["model"] == ["9", 0]
    assert wf["6"]["inputs"]["model"] == ["14", 0]
    assert graph.validate_api_workflow(wf, NODE_DEFS) == []


# ---------------------------------------------------------------------------
# ControlNet 结构控制（阶段 A：把构图/头身比锁到参照图）
# ---------------------------------------------------------------------------

def test_txt2img_without_controlnet_keeps_direct_conditioning():
    """不启用 ControlNet 时，采样器必须直接吃文本编码结果（保持既有行为）。"""
    wf = graph.build_sdxl_txt2img("sd_xl_base_1.0.safetensors", "p", "n")
    assert wf["3"]["inputs"]["positive"] == ["6", 0]
    assert wf["3"]["inputs"]["negative"] == ["7", 0]
    assert "ControlNetApplyAdvanced" not in {n["class_type"] for n in wf.values()}


def test_txt2img_with_controlnet_rewires_conditioning():
    """启用 ControlNet 后，采样器的 conditioning 必须来自 ControlNetApplyAdvanced。"""
    wf = graph.build_sdxl_txt2img(
        "sd_xl_base_1.0.safetensors", "p", "n",
        controlnet_name="canny_sdxl.safetensors", controlnet_image="reference.png")
    assert wf["3"]["inputs"]["positive"] == ["22", 0]
    assert wf["3"]["inputs"]["negative"] == ["22", 1]
    assert wf["22"]["inputs"]["positive"] == ["6", 0]      # 正向文本编码
    assert wf["22"]["inputs"]["negative"] == ["7", 0]      # 负向文本编码
    assert wf["22"]["inputs"]["control_net"] == ["21", 0]  # ControlNetLoader
    assert wf["22"]["inputs"]["image"] == ["20", 0]        # Canny 边缘图
    assert wf["20"]["inputs"]["image"] == ["19", 0]        # 参照图
    assert graph.validate_api_workflow(wf, NODE_DEFS) == []
    ui = graph.api_to_ui_workflow(wf, NODE_DEFS)
    assert graph.validate_ui_workflow(ui, NODE_DEFS) == []


def test_txt2img_controlnet_coexists_with_lora_and_ipadapter():
    """LoRA / IP-Adapter / ControlNet 三者可同时启用且互不干扰。"""
    wf = graph.build_sdxl_txt2img(
        "sd_xl_base_1.0.safetensors", "p", "n",
        lora_name="example_lora.safetensors",
        ipadapter_name="ip-adapter-plus-face_sdxl_vit-h.safetensors",
        ipadapter_image="reference.png",
        clip_vision_name="CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
        controlnet_name="canny_sdxl.safetensors", controlnet_image="example.png")
    assert wf["3"]["inputs"]["model"] == ["14", 0]         # 模型链走 IP-Adapter
    assert wf["3"]["inputs"]["positive"] == ["22", 0]      # 条件链走 ControlNet
    assert wf["6"]["inputs"]["clip"] == ["10", 1]          # 文本编码走 LoRA 之后
    assert graph.validate_api_workflow(wf, NODE_DEFS) == []
    ui = graph.api_to_ui_workflow(wf, NODE_DEFS)
    assert graph.validate_ui_workflow(ui, NODE_DEFS) == []


def test_validate_api_workflow_rejects_out_of_range_number():
    """回归：数值越界必须被本地校验拦住。

    实测教训：ComfyUI 0.40 的 Canny 阈值是 0.01~0.99 的 FLOAT。曾按旧习惯传 100/200，
    服务端报 "Value 100.0 bigger than max of 0.99" 并**丢弃整个输出**，
    而本地校验当时只查类型、不查范围，直接放行了。
    """
    wf = graph.build_sdxl_txt2img(
        "sd_xl_base_1.0.safetensors", "p", "n",
        controlnet_name="canny_sdxl.safetensors", controlnet_image="reference.png",
        canny_low=100.0, canny_high=200.0)
    problems = graph.validate_api_workflow(wf, NODE_DEFS)
    assert any("超过上限" in p for p in problems), problems


def test_canny_defaults_within_server_range():
    """默认阈值必须落在服务端允许的 0.01~0.99 内。"""
    wf = graph.build_sdxl_txt2img(
        "sd_xl_base_1.0.safetensors", "p", "n",
        controlnet_name="canny_sdxl.safetensors", controlnet_image="reference.png")
    canny = wf["20"]["inputs"]
    assert 0.01 <= canny["low_threshold"] <= 0.99
    assert 0.01 <= canny["high_threshold"] <= 0.99
    assert graph.validate_api_workflow(wf, NODE_DEFS) == []


# --- CLI 的 --backend 开关 ---------------------------------------------------

def test_backend_flag_exists_and_defaults_to_comfyui():
    """默认必须是 comfyui：既有 CLI 行为与 338 项测试都建立在本地路径上。

    文档里的 `backend.mode` 由 `apibackend.build_backend()` 读取；
    CLI 直接调用时保持本地为默认，避免破坏既有用法。
    """
    from chibitoy import generate

    parser = generate.build_parser()
    ns = parser.parse_args([])
    assert ns.backend == "comfyui"
    assert parser.parse_args(["--backend", "api"]).backend == "api"


def test_api_backend_path_requires_input_image(capsys):
    """`--backend api` 缺 --input-image 时应报错退出，而不是走进本地流程。"""
    from chibitoy import generate

    assert generate.main(["--backend", "api"]) == 2
    assert "--input-image" in capsys.readouterr().err


def test_api_backend_reports_missing_key_without_calling(capsys, monkeypatch, tmp_path):
    """缺密钥时**不得伪造调用**：应报错并提示如何填写。

    注意：`--confirm-cost` 是到达该分支的前提 —— 费用门在读取密钥之前，
    这正是阶段 4.15 加强的费用保护。
    """
    from chibitoy import apiconfig, generate

    monkeypatch.setattr(apiconfig, "load",
                        lambda *a, **k: (_ for _ in ()).throw(
                            apiconfig.ConfigError("服务商 'aliyun' 尚未配置 API Key。")))
    code = generate.main(["--backend", "api", "--confirm-cost",
                          "--input-image", "inputs/img/test.jpg"])
    assert code == 2
    assert "API Key" in capsys.readouterr().err


def test_api_backend_cost_gate_precedes_key_check(capsys, monkeypatch):
    """**回归**：未确认费用时，即使密钥缺失也只提示费用，不读密钥、不报密钥错。"""
    from chibitoy import apiconfig, generate

    loaded = {"n": 0}

    def _counting_load(*a, **k):
        loaded["n"] += 1
        raise apiconfig.ConfigError("不该走到这里：费用门必须先拦下")

    monkeypatch.setattr(apiconfig, "load", _counting_load)
    code = generate.main(["--backend", "api", "--input-image", "inputs/img/test.jpg"])

    assert code == 3, "未确认费用应返回 3（未执行）"
    out = capsys.readouterr()
    assert "费用保护" in out.out
    assert "API Key" not in out.err


def test_cli_tolerates_empty_argv_token():
    """**回归**：`--opt val \\ `（反斜杠后多一个空格）会多传一个空参数。

    shell 转义的是那个空格而非换行，argparse 于是报
    `unrecognized arguments: `（冒号后为空）—— 这个报错极难自查，故直接忽略空参数。
    """
    import sys as _sys

    from chibitoy import generate

    original = _sys.argv
    try:
        _sys.argv = ["generate.py", "--mode", "txt2img", "--dry-run", ""]
        assert generate.main() in (0, 1, 2)          # 不得因空参数抛 SystemExit
    except SystemExit as exc:                        # argparse 的退出码 2 = 参数错误
        raise AssertionError(f"空参数仍被当作无法识别的参数: {exc}") from None
    finally:
        _sys.argv = original
