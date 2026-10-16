from __future__ import annotations

from typing import Any

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN"}

CONTROL_AFTER_WIDGETS = {"seed", "noise_seed"}
CONTROL_AFTER_VALUE = "fixed"

def build_sdxl_txt2img(
    ckpt_name: str,
    positive_prompt: str,
    negative_prompt: str,
    seed: int = 0,
    steps: int = 20,
    cfg: float = 7.5,
    sampler_name: str = "euler",
    scheduler: str = "normal",
    width: int = 1024,
    height: int = 1024,
    batch_size: int = 1,
    denoise: float = 1.0,
    filename_prefix: str = "chibitoy",
    lora_name: str | None = None,
    lora_strength_model: float = 1.0,
    lora_strength_clip: float = 1.0,
    ipadapter_name: str | None = None,
    ipadapter_image: str | None = None,
    clip_vision_name: str | None = None,
    ipadapter_weight: float = 1.0,
    ipadapter_weight_type: str = "linear",
    ipadapter_start_at: float = 0.0,
    ipadapter_end_at: float = 1.0,
    bg_removal_model: str | None = None,
    controlnet_name: str | None = None,
    controlnet_image: str | None = None,
    controlnet_strength: float = 0.8,
    controlnet_start: float = 0.0,
    controlnet_end: float = 1.0,
    canny_low: float = 0.4,
    canny_high: float = 0.8,
    preprocess_photo: bool = False,
    photo_side: int = 1024,
) -> dict[str, Any]:
    wf: dict[str, Any] = {
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": ckpt_name},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": batch_size},
        },
    }

    model_src: list[Any] = ["4", 0]
    clip_src: list[Any] = ["4", 1]

    if lora_name:
        wf["10"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": model_src,
                "clip": clip_src,
                "lora_name": lora_name,
                "strength_model": lora_strength_model,
                "strength_clip": lora_strength_clip,
            },
        }
        model_src, clip_src = ["10", 0], ["10", 1]

    wf["6"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": positive_prompt, "clip": clip_src},
    }
    wf["7"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": negative_prompt, "clip": clip_src},
    }

    positive_ref: list[Any] = ["6", 0]
    negative_ref: list[Any] = ["7", 0]
    if controlnet_name and controlnet_image:
        wf["19"] = {"class_type": "LoadImage", "inputs": {"image": controlnet_image}}
        wf["20"] = {
            "class_type": "Canny",
            "inputs": {"image": ["19", 0], "low_threshold": canny_low, "high_threshold": canny_high},
        }
        wf["21"] = {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": controlnet_name},
        }
        wf["22"] = {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": ["6", 0],
                "negative": ["7", 0],
                "control_net": ["21", 0],
                "image": ["20", 0],
                "strength": controlnet_strength,
                "start_percent": controlnet_start,
                "end_percent": controlnet_end,
            },
        }
        positive_ref, negative_ref = ["22", 0], ["22", 1]

    photo_ref: list[Any] | None = None
    if ipadapter_name and ipadapter_image and clip_vision_name:
        wf["11"] = {"class_type": "LoadImage", "inputs": {"image": ipadapter_image}}
        photo_ref = ["11", 0]
        if preprocess_photo:
            wf["30"] = {
                "class_type": "ImageScale",
                "inputs": {"image": ["11", 0], "upscale_method": "lanczos",
                           "width": photo_side, "height": photo_side, "crop": "center"},
            }
            photo_ref = ["30", 0]
            if bg_removal_model:
                wf["32"] = {"class_type": "RemoveBackground",
                            "inputs": {"bg_removal_model": ["15", 0], "image": photo_ref}}
                wf["33"] = {"class_type": "InvertMask", "inputs": {"mask": ["32", 0]}}
                wf["34"] = {"class_type": "EmptyImage",
                            "inputs": {"width": photo_side, "height": photo_side,
                                       "batch_size": 1, "color": 0xFFFFFF}}
                wf["35"] = {
                    "class_type": "ImageCompositeMasked",
                    "inputs": {"destination": ["34", 0], "source": photo_ref,
                               "x": 0, "y": 0, "resize_source": False, "mask": ["33", 0]},
                }
                photo_ref = ["35", 0]
        wf["12"] = {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": clip_vision_name}}
        wf["13"] = {
            "class_type": "IPAdapterModelLoader",
            "inputs": {"ipadapter_file": ipadapter_name},
        }
        wf["14"] = {
            "class_type": "IPAdapterAdvanced",
            "inputs": {
                "model": model_src,
                "ipadapter": ["13", 0],
                "image": photo_ref,
                "clip_vision": ["12", 0],
                "weight": ipadapter_weight,
                "weight_type": ipadapter_weight_type,
                "combine_embeds": "concat",
                "start_at": ipadapter_start_at,
                "end_at": ipadapter_end_at,
                "embeds_scaling": "V only",
            },
        }
        model_src = ["14", 0]

    wf["3"] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": denoise,
            "model": model_src,
            "positive": positive_ref,
            "negative": negative_ref,
            "latent_image": ["5", 0],
        },
    }
    wf["8"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
    }

    images_src: list[Any] = ["8", 0]
    if bg_removal_model:
        wf["15"] = {
            "class_type": "LoadBackgroundRemovalModel",
            "inputs": {"bg_removal_name": bg_removal_model},
        }
        wf["16"] = {
            "class_type": "RemoveBackground",
            "inputs": {"bg_removal_model": ["15", 0], "image": ["8", 0]},
        }
        wf["18"] = {"class_type": "InvertMask", "inputs": {"mask": ["16", 0]}}
        wf["17"] = {
            "class_type": "JoinImageWithAlpha",
            "inputs": {"image": ["8", 0], "alpha": ["18", 0]},
        }
        images_src = ["17", 0]

    wf["9"] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": filename_prefix, "images": images_src},
    }
    return wf


def build_sdxl_img2img(
    ckpt_name: str,
    input_image: str,
    positive_prompt: str,
    negative_prompt: str,
    seed: int = 0,
    steps: int = 20,
    cfg: float = 7.5,
    sampler_name: str = "euler",
    scheduler: str = "normal",
    denoise: float = 0.55,
    filename_prefix: str = "chibitoy_img2img",
    lora_name: str | None = None,
    lora_strength_model: float = 1.0,
    lora_strength_clip: float = 1.0,
    ipadapter_name: str | None = None,
    ipadapter_image: str | None = None,
    clip_vision_name: str | None = None,
    ipadapter_weight: float = 1.0,
    ipadapter_weight_type: str = "linear",
    ipadapter_start_at: float = 0.0,
    ipadapter_end_at: float = 1.0,
    bg_removal_model: str | None = None,
    input_side: int = 1088,
) -> dict[str, Any]:
    """构建 SDXL 图生图（img2img）的 API 格式工作流。

    节点构成: 加载输入照片 -> 基础模型 -> (可选 LoRA) -> VAE 编码图像为 latent ->
             正/负向文本编码 -> 采样(denoise<1) -> VAE 解码 -> 保存 PNG。

    Args:
        input_image: 已上传到 ComfyUI 输入目录的图片文件名（LoadImage 的 image 取值）。
        denoise: 去噪强度，1.0 等价于文生图，0.0 等价于原图复现；越低越保留输入结构。
        lora_name: 可选，加载到 model 与 CLIP 的风格 LoRA 文件名；None 表示不加载。
        lora_strength_model / lora_strength_clip: LoRA 对模型 / 文本编码器的权重。
    """
    model_ref: list = ["2", 0]
    clip_ref: list = ["2", 1]

    workflow: dict[str, Any] = {
        "1": {"class_type": "LoadImage", "inputs": {"image": input_image}},
        "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt_name}},
        "10": {
            "class_type": "ImageScale",
            "inputs": {"image": ["1", 0], "upscale_method": "lanczos",
                       "width": input_side, "height": input_side, "crop": "center"},
        },
        "3": {"class_type": "VAEEncode", "inputs": {"pixels": ["10", 0], "vae": ["2", 2]}},
    }

    if lora_name:
        workflow["9"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["2", 0],
                "clip": ["2", 1],
                "lora_name": lora_name,
                "strength_model": lora_strength_model,
                "strength_clip": lora_strength_clip,
            },
        }
        model_ref = ["9", 0]
        clip_ref = ["9", 1]

    if ipadapter_name and ipadapter_image and clip_vision_name:
        workflow["11"] = {"class_type": "LoadImage", "inputs": {"image": ipadapter_image}}
        workflow["12"] = {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": clip_vision_name}}
        workflow["13"] = {
            "class_type": "IPAdapterModelLoader",
            "inputs": {"ipadapter_file": ipadapter_name},
        }
        workflow["14"] = {
            "class_type": "IPAdapterAdvanced",
            "inputs": {
                "model": model_ref,
                "ipadapter": ["13", 0],
                "image": ["11", 0],
                "clip_vision": ["12", 0],
                "weight": ipadapter_weight,
                "weight_type": ipadapter_weight_type,
                "combine_embeds": "concat",
                "start_at": ipadapter_start_at,
                "end_at": ipadapter_end_at,
                "embeds_scaling": "V only",
            },
        }
        model_ref = ["14", 0]

    workflow.update(
        {
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": positive_prompt, "clip": clip_ref},
            },
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": negative_prompt, "clip": clip_ref},
            },
            "6": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": sampler_name,
                    "scheduler": scheduler,
                    "denoise": denoise,
                    "model": model_ref,
                    "positive": ["4", 0],
                    "negative": ["5", 0],
                    "latent_image": ["3", 0],
                },
            },
            "7": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["6", 0], "vae": ["2", 2]},
            },
        }
    )

    images_src: list[Any] = ["7", 0]
    if bg_removal_model:
        workflow["15"] = {
            "class_type": "LoadBackgroundRemovalModel",
            "inputs": {"bg_removal_name": bg_removal_model},
        }
        workflow["16"] = {
            "class_type": "RemoveBackground",
            "inputs": {"bg_removal_model": ["15", 0], "image": ["7", 0]},
        }

        workflow["18"] = {"class_type": "InvertMask", "inputs": {"mask": ["16", 0]}}
        workflow["17"] = {
            "class_type": "JoinImageWithAlpha",
            "inputs": {"image": ["7", 0], "alpha": ["18", 0]},
        }
        images_src = ["17", 0]

    workflow["8"] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": filename_prefix, "images": images_src},
    }
    return workflow

def _input_type(spec: Any) -> str:
    """从 object_info 的输入定义中取出类型字符串。

    COMBO 类型在 object_info 中表现为一个候选值列表，统一记作 "COMBO"。
    """
    if isinstance(spec, list) and spec:
        first = spec[0]
        return "COMBO" if isinstance(first, list) else str(first)
    return str(spec)


def _input_options(spec: Any) -> dict[str, Any]:
    """取出输入定义中的选项字典（default、min、max、forceInput 等）。"""
    if isinstance(spec, list) and len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1]
    return {}


def combo_options(spec: Any) -> list[Any]:
    """取出 COMBO 输入的候选值列表，兼容 object_info 的两种形式。

    - 旧式: ``[["a", "b"]]`` —— 候选值本身就是第一个元素；
    - 新式: ``["COMBO", {"options": [...]}]`` —— ComfyUI 0.40+ 的部分节点
      （如 ``LoadBackgroundRemovalModel``）使用该形式。

    只认旧式会让候选值被误读成字符串 ``"COMBO"``，从而把所有合法取值都判为非法。
    """
    if not isinstance(spec, list) or not spec:
        return []
    if isinstance(spec[0], list):
        return spec[0]
    options = _input_options(spec).get("options")
    return options if isinstance(options, list) else []


def iter_inputs(node_def: dict[str, Any]):
    """按 required -> optional 的顺序遍历节点输入，产出 (名称, 定义, 是否必填)。"""
    for section, required in (("required", True), ("optional", False)):
        for name, spec in (node_def.get("input", {}).get(section) or {}).items():
            yield name, spec, required


def is_widget_input(spec: Any) -> bool:
    """判断某个输入是否渲染为控件（而非连线端口）。"""
    if _input_options(spec).get("forceInput"):
        return False
    return _input_type(spec) in WIDGET_TYPES or _input_type(spec) == "COMBO"

def validate_api_workflow(
    workflow: dict[str, Any],
    node_defs: dict[str, Any],
) -> list[str]:
    """用服务端节点定义校验 API 工作流，返回问题列表（空列表表示通过）。"""
    problems: list[str] = []

    for node_id, node in workflow.items():
        class_type = node.get("class_type")
        if class_type not in node_defs:
            problems.append(f"节点 {node_id}: 未知的 class_type '{class_type}'")
            continue
        node_def = node_defs[class_type]
        inputs = node.get("inputs", {})
        known = {name: spec for name, spec, _ in iter_inputs(node_def)}

        for name, spec, required in iter_inputs(node_def):
            if required and name not in inputs:
                problems.append(
                    f"节点 {node_id} ({class_type}): 缺少必填输入 '{name}'"
                )

        # 未知输入名
        for name in inputs:
            if name not in known:
                problems.append(
                    f"节点 {node_id} ({class_type}): 存在未知输入 '{name}'"
                )

        # 输入取值校验
        for name, value in inputs.items():
            if name not in known:
                continue
            spec = known[name]
            expected = _input_type(spec)

            if isinstance(value, list):
                if len(value) != 2:
                    problems.append(
                        f"节点 {node_id} ({class_type}): 输入 '{name}' 的连线格式非法 {value}"
                    )
                    continue
                src_id, src_slot = value
                if src_id not in workflow:
                    problems.append(
                        f"节点 {node_id} ({class_type}): 输入 '{name}' 指向不存在的节点 '{src_id}'"
                    )
                    continue
                src_def = node_defs.get(workflow[src_id].get("class_type"))
                if src_def is None:
                    continue
                src_outputs = src_def.get("output", [])
                if not isinstance(src_slot, int) or not (0 <= src_slot < len(src_outputs)):
                    problems.append(
                        f"节点 {node_id} ({class_type}): 输入 '{name}' 引用了 "
                        f"节点 {src_id} 上不存在的输出序号 {src_slot}"
                    )
                    continue
                actual = src_outputs[src_slot]
                if actual != expected:
                    problems.append(
                        f"节点 {node_id} ({class_type}): 输入 '{name}' 需要 {expected}，"
                        f"但节点 {src_id} 的输出 {src_slot} 是 {actual}"
                    )
                continue

            # 字面量取值
            if expected == "COMBO":
                options = combo_options(spec)
                if options and value not in options:
                    hint = f"（可选值示例: {options[:5]}）" if len(options) > 5 else f"（可选值: {options}）"
                    problems.append(
                        f"节点 {node_id} ({class_type}): 输入 '{name}' 的值 '{value}' 不在候选列表中{hint}"
                    )
            elif expected == "INT" and not isinstance(value, int):
                problems.append(
                    f"节点 {node_id} ({class_type}): 输入 '{name}' 应为 INT，实际为 {type(value).__name__}"
                )
            elif expected == "FLOAT" and not isinstance(value, (int, float)):
                problems.append(
                    f"节点 {node_id} ({class_type}): 输入 '{name}' 应为 FLOAT，实际为 {type(value).__name__}"
                )
            elif expected == "STRING" and not isinstance(value, str):
                problems.append(
                    f"节点 {node_id} ({class_type}): 输入 '{name}' 应为 STRING，实际为 {type(value).__name__}"
                )
            elif expected == "BOOLEAN" and not isinstance(value, bool):
                problems.append(
                    f"节点 {node_id} ({class_type}): 输入 '{name}' 应为 BOOLEAN，实际为 {type(value).__name__}"
                )

            if expected in ("INT", "FLOAT") and isinstance(value, (int, float)) and not isinstance(value, bool):
                options = _input_options(spec)
                low, high = options.get("min"), options.get("max")
                if isinstance(low, (int, float)) and value < low:
                    problems.append(
                        f"节点 {node_id} ({class_type}): 输入 '{name}' 的值 {value} 小于下限 {low}"
                    )
                if isinstance(high, (int, float)) and value > high:
                    problems.append(
                        f"节点 {node_id} ({class_type}): 输入 '{name}' 的值 {value} 超过上限 {high}"
                    )

    # 至少需要一个输出节点
    has_output = any(
        node_defs.get(n.get("class_type"), {}).get("output_node")
        for n in workflow.values()
    )
    if not has_output:
        problems.append("工作流中没有输出节点（如 SaveImage），执行后不会产生任何结果")

    return problems

def _topological_order(workflow: dict[str, Any]) -> list[str]:
    """按依赖关系排序，得到执行顺序（Kahn 算法）。"""
    deps: dict[str, set[str]] = {nid: set() for nid in workflow}
    for nid, node in workflow.items():
        for value in node.get("inputs", {}).values():
            if isinstance(value, list) and len(value) == 2 and value[0] in workflow:
                deps[nid].add(value[0])

    order: list[str] = []
    remaining = dict(deps)
    while remaining:
        ready = sorted(n for n, d in remaining.items() if not (d - set(order)))
        if not ready:
            order.extend(sorted(remaining))
            break
        order.append(ready[0])
        remaining.pop(ready[0])
    return order


def api_to_ui_workflow(
    workflow: dict[str, Any],
    node_defs: dict[str, Any],
    origin: tuple[float, float] = (60.0, 80.0),
) -> dict[str, Any]:

    ui_id: dict[str, int] = {}
    for nid in workflow:
        ui_id[nid] = int(nid) if str(nid).isdigit() else len(ui_id) + 1000
    # 处理数字 id 冲突
    used: set[int] = set()
    for nid in workflow:
        candidate = ui_id[nid]
        while candidate in used:
            candidate += 1000
        ui_id[nid] = candidate
        used.add(candidate)

    order = _topological_order(workflow)
    order_index = {nid: i for i, nid in enumerate(order)}

    level: dict[str, int] = {}
    for nid in order:
        parents = [
            v[0]
            for v in workflow[nid].get("inputs", {}).values()
            if isinstance(v, list) and len(v) == 2 and v[0] in workflow
        ]
        level[nid] = 0 if not parents else max(level.get(p, 0) for p in parents) + 1

    column_row: dict[int, int] = {}
    link_counter = 0
    links: list[list[Any]] = []
    link_of_input: dict[tuple[str, str], int] = {}
    output_links: dict[tuple[str, int], list[int]] = {}

    # 分配连线 id
    for nid in order:
        node = workflow[nid]
        present_inputs = set(node.get("inputs", {}))
        for name, value in node.get("inputs", {}).items():
            if isinstance(value, list) and len(value) == 2:
                link_counter += 1
                src_id, src_slot = value
                links.append(
                    [
                        link_counter,
                        ui_id[src_id],
                        src_slot,
                        ui_id[nid],
                        _slot_index(node_defs, node["class_type"], name, present_inputs),
                        _link_type(node_defs, node["class_type"], name),
                    ]
                )
                link_of_input[(nid, name)] = link_counter
                output_links.setdefault((src_id, src_slot), []).append(link_counter)

    nodes: list[dict[str, Any]] = []
    for nid in order:
        node = workflow[nid]
        class_type = node["class_type"]
        node_def = node_defs[class_type]
        inputs = node.get("inputs", {})

        slot_inputs: list[dict[str, Any]] = []
        widgets_values: list[Any] = []
        for name, spec, _required in iter_inputs(node_def):
            if name not in inputs:
                continue
            value = inputs[name]
            if isinstance(value, list):
                slot_inputs.append(
                    {
                        "name": name,
                        "type": _input_type(spec),
                        "link": link_of_input[(nid, name)],
                    }
                )
            elif is_widget_input(spec):
                widgets_values.append(value)
                if name in CONTROL_AFTER_WIDGETS:
                    widgets_values.append(CONTROL_AFTER_VALUE)

        for name, value in inputs.items():
            if name in {s["name"] for s in slot_inputs}:
                continue
            if not isinstance(value, list) and not any(
                n == name for n, _s, _r in iter_inputs(node_def)
            ):
                widgets_values.append(value)

        output_names = node_def.get("output_name") or node_def.get("output", [])
        outputs = [
            {
                "name": output_names[i] if i < len(output_names) else f"out{i}",
                "type": output_type,
                "slot_index": i,
                "links": output_links.get((nid, i), []),
            }
            for i, output_type in enumerate(node_def.get("output", []))
        ]

        row = column_row.get(level[nid], 0)
        column_row[level[nid]] = row + 1
        widget_count = len(widgets_values)
        nodes.append(
            {
                "id": ui_id[nid],
                "type": class_type,
                "pos": [
                    origin[0] + level[nid] * 380.0,
                    origin[1] + row * 260.0,
                ],
                "size": [330.0, max(120.0, 60.0 + widget_count * 30.0)],
                "flags": {},
                "order": order_index[nid],
                "mode": 0,
                "inputs": slot_inputs,
                "outputs": outputs,
                "properties": {"Node name for S&R": class_type},
                "widgets_values": widgets_values,
            }
        )

    return {
        "last_node_id": max(ui_id.values()) if ui_id else 0,
        "last_link_id": link_counter,
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {"ds": {"scale": 0.9, "offset": [100, 60]}},
        "version": 0.4,
    }


def _link_type(node_defs: dict[str, Any], class_type: str, input_name: str) -> str:
    """取某个输入的类型字符串（用于 links 数组）。"""
    for name, spec, _r in iter_inputs(node_defs[class_type]):
        if name == input_name:
            return _input_type(spec)
    return "*"


def _slot_index(
    node_defs: dict[str, Any],
    class_type: str,
    input_name: str,
    present: set[str] | None = None,
) -> int:
    index = 0
    for name, spec, _r in iter_inputs(node_defs[class_type]):
        if is_widget_input(spec):
            continue
        if present is not None and name not in present:
            continue
        if name == input_name:
            return index
        index += 1
    return index

def validate_ui_workflow(
    ui: dict[str, Any],
    node_defs: dict[str, Any],
) -> list[str]:
    problems: list[str] = []

    for key in ("last_node_id", "last_link_id", "nodes", "links", "version"):
        if key not in ui:
            problems.append(f"缺少顶层字段 '{key}'")

    nodes = {n["id"]: n for n in ui.get("nodes", [])}
    if not nodes:
        problems.append("nodes 为空")
        return problems

    # 连线表本身的自洽性
    declared_links = {link[0] for link in ui.get("links", [])}
    for link in ui.get("links", []):
        if len(link) != 6:
            problems.append(f"连线 {link} 字段数不为 6")
            continue
        link_id, src, src_slot, dst, dst_slot, link_type = link
        if src not in nodes:
            problems.append(f"连线 {link_id}: 来源节点 {src} 不存在")
            continue
        if dst not in nodes:
            problems.append(f"连线 {link_id}: 目标节点 {dst} 不存在")
            continue
        src_node, dst_node = nodes[src], nodes[dst]
        if src_slot >= len(src_node.get("outputs", [])):
            problems.append(f"连线 {link_id}: 来源节点 {src} 没有输出端口 {src_slot}")
        if dst_slot >= len(dst_node.get("inputs", [])):
            problems.append(f"连线 {link_id}: 目标节点 {dst} 没有输入端口 {dst_slot}")
            continue
        # 端口类型一致性
        if src_slot < len(src_node.get("outputs", [])):
            src_type = src_node["outputs"][src_slot].get("type")
            dst_type = dst_node["inputs"][dst_slot].get("type")
            if src_type != dst_type:
                problems.append(
                    f"连线 {link_id}: 类型不匹配 {src_type} -> {dst_type}"
                )
        if link_type != dst_node["inputs"][dst_slot].get("type"):
            problems.append(
                f"连线 {link_id}: 连线记录的类型 {link_type} 与目标端口类型 "
                f"{dst_node['inputs'][dst_slot].get('type')} 不一致"
            )

    # 逐节点检查
    for node in ui.get("nodes", []):
        class_type = node.get("type")
        if class_type not in node_defs:
            problems.append(f"节点 {node.get('id')}: 未知类型 '{class_type}'")
            continue
        node_def = node_defs[class_type]
        connected = {i["name"] for i in node.get("inputs", [])}

        for name, spec, required in iter_inputs(node_def):
            if not required:
                continue
            if is_widget_input(spec):
                # 控件型输入: 需要出现在 inputs(被转成端口) 或由 widgets_values 提供
                if name in connected:
                    continue
                if not node.get("widgets_values"):
                    problems.append(
                        f"节点 {node.get('id')} ({class_type}): 控件输入 '{name}' 没有取值"
                    )
            elif name not in connected:
                problems.append(
                    f"节点 {node.get('id')} ({class_type}): 必填端口 '{name}' 未连线"
                )

        # 端口引用反查连线表
        for port in node.get("inputs", []):
            if port.get("link") not in declared_links:
                problems.append(
                    f"节点 {node.get('id')} ({class_type}): 输入 '{port.get('name')}' "
                    f"引用了未声明的连线 {port.get('link')}"
                )
        for port in node.get("outputs", []):
            for link_id in port.get("links", []) or []:
                if link_id not in declared_links:
                    problems.append(
                        f"节点 {node.get('id')} ({class_type}): 输出 '{port.get('name')}' "
                        f"引用了未声明的连线 {link_id}"
                    )

    # 未使用的连线
    referenced = {
        port["link"]
        for node in ui.get("nodes", [])
        for port in node.get("inputs", [])
        if port.get("link") is not None
    }
    for link_id in declared_links - referenced:
        problems.append(f"连线 {link_id} 已声明但没有任何输入端口引用")

    return problems
