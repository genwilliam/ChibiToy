from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import config, graph, preprocess, prompts
from .client import ComfyUIClient, ComfyUIError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ChibiToy: SDXL 文生图 / 图生图（真人照片转 Q 版潮玩）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", default=config.SDXL_BASE_CKPT, help="基础模型文件名")
    parser.add_argument("--appearance", default=prompts.DEFAULT_APPEARANCE,
                        help="需要保留的发型/发饰/服装描述")
    parser.add_argument("--positive", default=None, help="直接指定正向提示词（覆盖 --appearance）")
    parser.add_argument("--negative", default=prompts.NEGATIVE_PROMPT, help="负向提示词")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--steps", type=int, default=30, help="采样步数")
    parser.add_argument("--cfg", type=float, default=7.5, help="CFG 强度")
    parser.add_argument("--sampler", default="dpmpp_2m", help="采样器名称")
    parser.add_argument("--scheduler", default="karras", help="调度器名称")
    parser.add_argument("--mode", choices=["txt2img", "img2img"], default="txt2img",
                        help="生成模式：txt2img 文生图 / img2img 图生图")
    parser.add_argument("--input-image", default=None,
                        help="img2img 模式的输入照片路径（JPG/PNG/WEBP）")
    parser.add_argument("--denoise", type=float, default=0.55,
                        help="img2img 去噪强度：1.0≈文生图，0.0≈原图复现；越低越保留输入结构")
    parser.add_argument("--lora", default=None,
                        help="img2img 模式加载的风格 LoRA 文件名（models/loras/ 下）")
    parser.add_argument("--lora-strength-model", type=float, default=1.0,
                        help="LoRA 对模型的权重")
    parser.add_argument("--lora-strength-clip", type=float, default=1.0,
                        help="LoRA 对文本编码器的权重")
    parser.add_argument("--ipadapter", default=None,
                        help="IP-Adapter 模型文件名（models/ipadapter/ 下）。"
                             "启用后需用 --input-image 提供参考照片；"
                             "仅注入身份特征，不改变构图")
    parser.add_argument("--clip-vision", default="CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
                        help="CLIP Vision 编码器文件名（models/clip_vision/ 下）")
    parser.add_argument("--ipadapter-weight", type=float, default=1.0,
                        help="IP-Adapter 权重（越大越贴近参考照片，过高会压过造型提示词）")
    parser.add_argument("--ipadapter-weight-type", default="linear",
                        help="IP-Adapter 权重类型（linear/ease in/style transfer 等）")
    parser.add_argument("--ipadapter-start-at", type=float, default=0.0,
                        help="IP-Adapter 生效起始步（0~1）。调大 = 让提示词先决定构图，"
                             "IP-Adapter 只在后段补身份细节，可削弱写实比例拉力")
    parser.add_argument("--ipadapter-end-at", type=float, default=1.0,
                        help="IP-Adapter 生效结束步（0~1）")
    parser.add_argument("--ipadapter-image", default=None,
                        help="IP-Adapter 身份参考照片路径。**可与 --input-image 不同**——"
                             "两阶段方案（先定 Q 版结构、再注入身份）必须分别指定；"
                             "未提供时回退到 --input-image")
    parser.add_argument("--ipadapter-prep-bg", action="store_true",
                        help="先把身份参考图做**真实前景分割并合成到白底**再喂给 IP-Adapter。"
                             "实测：蓝底证件照直接当参考会把蓝色注入产物（生成蓝衣 + 蓝色伪影），"
                             "因为 IP-Adapter 读的是整张参考图")
    parser.add_argument("--ipadapter-crop",
                        choices=["none", "top-square", "head-square"], default="top-square",
                        help="IP-Adapter 参考图裁剪：top-square=只取顶部正方形（**默认**，"
                             "实测在本次受控对照中服装主色最好）；head-square=按 BiRefNet 的 alpha "
                             "自适应裁到头部（对头部特写更合理，但实测在本张全身照上指标更低，"
                             "见 results/exp/crop/report.md）；none=整图"
                             "（证件照的头肩区域）。IP-Adapter 会把参考图的背景与身形比例一并"
                             "注入，导致 Q 版比例被拉向写实；裁剪到头部可显著缓解")
    parser.add_argument("--controlnet", default=None,
                        help="ControlNet 模型文件名（models/controlnet/ 下，如 canny_sdxl.safetensors）")
    parser.add_argument("--controlnet-image", default=None,
                        help="ControlNet 结构参照图路径（内部先跑 Canny 取边缘）。"
                             "用一张 Q 版手办图作参照，可把构图与头身比锁到目标区间")
    parser.add_argument("--preprocess-in-graph", action="store_true",
                        help="把照片预处理（缩放到固定边长 + 真实抠图合成白底）**放进工作流图里**。"
                             "导出的工作流若要**在网页端**直接用（用户把手机原图拖进 LoadImage），"
                             "必须开启——否则 4000x3000 的原图会让 VAE/后续节点爆显存")
    parser.add_argument("--head-side-scale", type=float, default=2.0,
                        help="自适应头部裁切的边长 = 头部区域宽度 × 该系数（默认 2.0，"
                             "越大留的肩部越多、越小越贴近脸）")
    parser.add_argument("--head-probe-side", type=int, default=1280,
                        help="自适应裁切前把参考图缩到该长边，避免对 12MP 原图跑分割")
    parser.add_argument("--photo-side", type=int, default=1024,
                        help="图内预处理时照片缩放到的边长（配合 --preprocess-in-graph）")
    parser.add_argument("--controlnet-strength", type=float, default=0.8,
                        help="ControlNet 强度（越高越严格遵循参照图结构）")
    parser.add_argument("--bg-removal", default=None,
                        help="背景移除模型文件名（models/background_removal/ 下，如 "
                             "birefnet.safetensors）。启用后输出为带 alpha 的透明背景 PNG，"
                             "走真实前景分割而非补边/纯色覆盖")
    parser.add_argument("--target-size", default="1024x1024",
                        help="img2img 预处理目标尺寸：1024（正方形）或 宽x高，如 896x1152")
    parser.add_argument("--pad-color", default="#ffffff",
                        help="img2img 预处理补边颜色：#RRGGBB 或 R,G,B；默认白色中性背景")
    parser.add_argument("--max-pixels", type=int, default=preprocess.DEFAULT_MAX_PIXELS,
                        help="输入图片像素总量上限（超过则报错，避免异常大图）")
    parser.add_argument("--min-side", type=int, default=preprocess.DEFAULT_MIN_SIDE,
                        help="输入图片**推荐**最低边长（px）：低于此值只给出低分辨率警告，"
                             "不再拒绝（阶段 1B-3A-1 起）")
    parser.add_argument("--max-file-size", type=int,
                        default=preprocess.DEFAULT_MAX_FILE_SIZE,
                        help="输入文件大小上限（字节），默认 10000000（img2img 模式）")
    parser.add_argument("--no-preprocess", action="store_true",
                        help="跳过预处理，直接上传原图（img2img 模式）")
    parser.add_argument("--test-mode", action="store_true",
                        help="[兼容保留] 低分辨率素材现已默认被接受（仅警告），本开关不再必要；"
                             "文件大小/格式/解码安全/像素总量限制始终生效")
    parser.add_argument("--strict-min-side", action="store_true",
                        help="严格模式：恢复旧行为，低于 --min-side 直接拒绝（默认关闭）")
    parser.add_argument("--width", type=int, default=1024, help="生成宽度")
    parser.add_argument("--height", type=int, default=1024, help="生成高度")
    parser.add_argument("--batch-size", type=int, default=1, help="批量张数")
    parser.add_argument("--prefix", default="chibitoy", help="输出文件名前缀")
    parser.add_argument("--out", default=str(config.OUTPUTS_DIR), help="输出目录")
    parser.add_argument("--save-workflow", default=None,
                        help="把工作流保存到该前缀（同时生成 .api.json 与 .ui.json）")
    parser.add_argument("--workflow", default=None,
                        help="直接执行一个已导出的 API 格式工作流文件（忽略其他生成参数）")
    parser.add_argument("--timeout", type=float, default=1800.0, help="执行超时（秒）")
    parser.add_argument("--dry-run", action="store_true", help="只校验工作流，不执行")
    parser.add_argument("--backend", choices=["comfyui", "api"], default="comfyui",
                        help="生图后端：comfyui=本地工作流（默认，保持既有行为）；"
                             "api=云端 API（配置见 config/chibitoy.yaml，密钥只从该文件读取）")
    parser.add_argument("--confirm-cost", action="store_true",
                        help="**仅 --backend api 需要**：确认知晓本次将发起一次真实付费调用。"
                             "未给出时只打印费用提示并退出，**不创建客户端、不发起请求**。")
    parser.add_argument("--max-retries", type=int, default=None, metavar="N",
                        help="**仅 --backend api 需要**：失败后最多尝试 N 次（含首次）。"
                             "默认沿用配置值，而配置默认已是 1（不自动重试）。"
                             "调大即意味着一次失败可能变成 N 次付费调用。")
    return parser


def _upload_image(path: Path, client) -> str:
    """上传一张图片到 ComfyUI 输入目录，返回其文件名。"""
    uploaded = client.upload_image(path)
    print(f"[上传] {path} -> ComfyUI 输入目录 ({uploaded})")
    return uploaded


def _matte_reference_rgba(path: Path, client, bg_model: str):
    """用 BiRefNet 去背景，返回**保留 alpha** 的 PIL 图像（RGBA）。

    与 :func:`_matte_reference_to_white` 的区别：这里**不丢 alpha** ——
    自适应裁切需要 alpha 来求主体包围盒。
    """
    uploaded = client.upload_image(path)
    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": uploaded}},
        "2": {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": bg_model}},
        "3": {"class_type": "RemoveBackground",
              "inputs": {"bg_removal_model": ["2", 0], "image": ["1", 0]}},
        "4": {"class_type": "InvertMask", "inputs": {"mask": ["3", 0]}},
        "5": {"class_type": "JoinImageWithAlpha",
              "inputs": {"image": ["1", 0], "alpha": ["4", 0]}},
        "6": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "ipa_ref_alpha", "images": ["5", 0]}},
    }
    result = client.generate(workflow, output_dir=config.PREPROCESSED_DIR)
    if not result.saved_paths:
        raise ValueError("参考图抠图未产出结果")
    from PIL import Image as _Img

    with _Img.open(result.saved_paths[0]) as _m:
        return _m.convert("RGBA").copy()


def _matte_reference_to_white(path: Path, client, bg_model: str) -> Path:
    """用 ComfyUI 的 BiRefNet 去背景并合成到白底，返回新文件路径。

    目的：IP-Adapter 读取的是**整张参考图**。蓝底证件照会把蓝色一并注入，
    实测表现为"生成蓝衣 + 头部蓝色伪影"。先做真实前景分割再喂，可消除该污染。
    """
    uploaded = client.upload_image(path)
    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": uploaded}},
        "2": {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": bg_model}},
        "3": {"class_type": "RemoveBackground",
              "inputs": {"bg_removal_model": ["2", 0], "image": ["1", 0]}},
        "4": {"class_type": "InvertMask", "inputs": {"mask": ["3", 0]}},
        "5": {"class_type": "JoinImageWithAlpha",
              "inputs": {"image": ["1", 0], "alpha": ["4", 0]}},
        "6": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "ipa_ref_matted", "images": ["5", 0]}},
    }
    result = client.generate(workflow, output_dir=config.PREPROCESSED_DIR)
    if not result.saved_paths:
        raise ValueError("参考图抠图未产出结果")
    from PIL import Image as _Img

    with _Img.open(result.saved_paths[0]) as _m:
        rgba = _m.convert("RGBA")
        flat = _Img.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.getchannel("A"))
    out = config.PREPROCESSED_DIR / f"{path.stem}_matted_white.png"
    flat.save(out)
    print(f"[抠图] 身份参考图去背景并合成白底 -> {out} ({flat.size[0]}x{flat.size[1]})")
    return out


def _upload_ipadapter_reference(args, client) -> str:
    """上传 IP-Adapter 参考图（必要时先裁到头部），返回 ComfyUI 侧文件名。

    ``--ipadapter-image`` 优先，未提供时回退到 ``--input-image``。
    txt2img 与 img2img 共用，避免两处逻辑不一致。
    """
    raw = args.ipadapter_image or args.input_image
    if not raw:
        raise ValueError("使用 --ipadapter 时必须用 --ipadapter-image 或 --input-image 提供参考照片")
    ref = Path(raw)
    if not ref.is_file():
        raise ValueError(f"IP-Adapter 参考照片不存在: {ref}")
    if args.ipadapter_crop == "head-square":
        from PIL import Image as _Img

        from chibitoy import preprocess as _pp

        probe_side = getattr(args, "head_probe_side", 1280)
        config.PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        with _Img.open(ref) as _im:
            _rgb = _im.convert("RGB")
            _w, _h = _rgb.size
            _scale = min(1.0, probe_side / max(_w, _h))
            if _scale < 1.0:
                _rgb = _rgb.resize((max(1, round(_w * _scale)), max(1, round(_h * _scale))),
                                   _Img.LANCZOS)
                print(f"[裁剪] 参考图先缩放到 {_rgb.size[0]}x{_rgb.size[1]} 再做分割")
            _probe = config.PREPROCESSED_DIR / f"{ref.stem}_ipa_probe.png"
            _rgb.save(_probe)
        _rgba = _matte_reference_rgba(_probe, client, args.bg_removal or "birefnet.safetensors")
        _cropped, _note = _pp.head_square_crop_from_alpha(
            _rgba, side_scale=getattr(args, "head_side_scale", 2.0))
        _flat = _Img.new("RGB", _cropped.size, (255, 255, 255))
        _flat.paste(_cropped, mask=_cropped.getchannel("A"))
        out = config.PREPROCESSED_DIR / f"{ref.stem}_ipa_head.png"
        _flat.save(out)
        print(f"[裁剪] IP-Adapter 参考图 自适应头部裁切 -> {out} "
              f"({_flat.size[0]}x{_flat.size[1]})")
        print(f"[裁剪] {_note}")
        ref = out
    if args.ipadapter_crop == "top-square":
        from PIL import Image as _Img

        with _Img.open(ref) as _im:
            _w, _h = _im.size
            _side = min(_w, _h)
            _left = (_w - _side) // 2
            _cropped = _im.convert("RGB").crop((_left, 0, _left + _side, _side))
        config.PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out = config.PREPROCESSED_DIR / f"{ref.stem}_ipa_crop.png"
        _cropped.save(out)
        print(f"[裁剪] IP-Adapter 参考图 -> {out} ({_cropped.size[0]}x{_cropped.size[1]})")
        ref = out
    if getattr(args, "ipadapter_prep_bg", False):
        ref = _matte_reference_to_white(
            ref, client, args.bg_removal or "birefnet.safetensors")
    uploaded = client.upload_image(ref)
    print(f"[上传] IP-Adapter 参考照片 {ref} -> ComfyUI 输入目录 ({uploaded})")
    return uploaded


def load_node_defs(client: ComfyUIClient, workflow: dict) -> dict:
    """按工作流实际用到的节点类型，逐个拉取节点定义。"""
    classes = sorted({node["class_type"] for node in workflow.values()})
    defs: dict = {}
    for class_type in classes:
        info = client.object_info(class_type)
        if class_type not in info:
            raise ComfyUIError(f"服务端不存在节点类型: {class_type}")
        defs[class_type] = info[class_type]
    return defs


def _print_preprocess(result: preprocess.PreprocessResult) -> None:
    """打印预处理摘要（原始/处理后尺寸、补边参数与警告）。"""
    ow, oh = result.original_size
    rw, rh = result.resized_size
    tw, th = result.target_size
    ori_note = f"EXIF方向={result.exif_orientation} 已纠正" if result.orientation_applied else "无需纠正"
    print(f"[预处理] {result.original_format} {ow}x{oh} -> 等比缩放 {rw}x{rh} -> 补边到 {tw}x{th}")
    print(f"         补边 L/T/R/B = {result.pad_left}/{result.pad_top}/"
          f"{result.pad_right}/{result.pad_bottom}，颜色={result.pad_color}（{ori_note}）")
    print(f"         输出: {result.dest}")
    for note in result.warnings:
        print(f"[警告] {note}")


def _run_api(args) -> int:
    if not args.input_image:
        print("[错误] --backend api 需要 --input-image 指定真人照片", file=sys.stderr)
        return 2

    from PIL import Image as _Img

    from chibitoy import apibackend, apiconfig

    if not args.confirm_cost:
        print("[费用保护] 未给出 --confirm-cost，未发起任何请求。")
        print("           本次将发起 **1 次真实付费 API 请求**（--backend api 单次生成）。")
        try:
            _cfg = apiconfig.load()
            _provider = _cfg.active_provider()
            _model = _provider.model or "（未配置）"
            _retries = (args.max_retries if args.max_retries is not None
                        else _provider.max_retries)
        except Exception as exc:
            _model = f"（配置不可读: {type(exc).__name__}）"
            _retries = 1
        print(f"           模型 {_model}，最多尝试 {_retries} 次（含首次）。")
        print("           费用上界估计：0.25 元/张 × 1 = ≤0.25 元"
              "（依据同 family qwen-image-3.0-pro 的公开单价；"
              "本模型的官方单价**未逐字核实**，此数不能当作实际账单）。")
        print("           确认后请追加 --confirm-cost 重新运行。")
        return 3

    if args.dry_run:
        print("[dry-run] --backend api：只校验参数与输入，未发起任何请求。")
        if not args.input_image:
            print("[错误] --backend api 需要 --input-image 指定真人照片", file=sys.stderr)
            return 2
        try:
            _info = preprocess.validate_image(
                args.input_image, max_pixels=args.max_pixels,
                max_file_size=args.max_file_size)
        except preprocess.PreprocessError as exc:
            print(f"[预处理失败] {exc}", file=sys.stderr)
            return 2
        print(f"[输入] {_info['format']} "
              f"{_info['size'][0]}x{_info['size'][1]} 校验通过")
        print("[dry-run] 未调用云端 API，未产生费用。")
        return 0

    try:
        cfg = apiconfig.load(require_key=True)
    except apiconfig.ConfigError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    if args.max_retries is not None:
        if args.max_retries < 1:
            print(f"[错误] --max-retries 必须 ≥1（收到 {args.max_retries}）", file=sys.stderr)
            return 2
        cfg.active_provider().max_retries = args.max_retries
    try:
        backend = apibackend.ApiBackend(cfg)
    except apiconfig.ConfigError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    attempts = backend.provider.spec.max_retries
    print(f"[后端] api  {cfg.describe()}  最多尝试 {attempts} 次（含首次）")
    if attempts > 1:
        print(f"[费用提示] --max-retries={attempts} 意味着一次失败可能变成 "
              f"{attempts} 次付费调用。")
    if not backend.capabilities.controlnet:
        print("[能力] 云端不具备 ControlNet / LoRA —— 这是与本地后端的已知差异，"
              "完整全身与头身比不由云端保证")

    prepared = config.PREPROCESSED_DIR / f"{Path(args.input_image).stem}_api_input.png"
    try:
        info = preprocess.preprocess_image(
            args.input_image, prepared, target_size=(args.width, args.height),
            min_side=64 if args.test_mode else preprocess.DEFAULT_MIN_SIDE,
            max_file_size=args.max_file_size, test_mode=args.test_mode)
    except Exception as exc:
        print(f"[错误] 输入照片不合格: {exc}", file=sys.stderr)
        return 2
    print(f"[输入] {info.original_format} {info.original_size[0]}x{info.original_size[1]} "
          f"-> {info.final_size[0]}x{info.final_size[1]}")
    for warning in info.warnings:
        print(f"[提示] {warning}")

    request = backend.build_request(prepared, appearance=args.appearance)
    out_dir = Path(args.out) if args.out else config.OUTPUTS_DIR
    try:
        result = backend.generate(request, output_dir=out_dir)
    except apibackend.ApiError as exc:
        print(f"[错误] API 调用失败: {exc}", file=sys.stderr)
        return 1

    for path in result.saved_paths:
        with _Img.open(path) as handle:
            width, height = handle.size
            mode = handle.mode
        print(f"[输出] {path} [{width}x{height} {mode}]  API 耗时 {result.elapsed_seconds:.1f}s")
        problems = preprocess.validate_output_size(width, height,
                                                   min_size=config.FORMAL_OUTPUT_MIN_SIZE)
        if problems:
            print("[规格] 输出尺寸不达标：" + "；".join(problems), file=sys.stderr)
            print("       注意：**不得**用放大来伪装达标；请改用更大 size 的模型。", file=sys.stderr)
            return 1
        print("[规格] 尺寸达标，且为 API 原生输出（未经放大）")
    return 0


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args([token for token in raw if token != ""])

    if args.backend == "api":
        return _run_api(args)

    client = ComfyUIClient()
    if not client.is_available():
        print(f"[错误] 无法连接 ComfyUI: {config.COMFY_BASE_URL}", file=sys.stderr)
        print("       请先启动: ./scripts/start_comfyui.sh --background", file=sys.stderr)
        return 2
    expected_size: tuple[int, int] | None = None
    size_source = ""
    if args.workflow:
        workflow_path = Path(args.workflow)
        if not workflow_path.is_file():
            print(f"[错误] 工作流文件不存在: {workflow_path}", file=sys.stderr)
            return 2
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        print(f"[工作流] 从文件载入: {workflow_path}")
        for node in workflow.values():
            if isinstance(node, dict) and node.get("class_type") == "EmptyLatentImage":
                inputs = node.get("inputs", {})
                width_in, height_in = inputs.get("width"), inputs.get("height")
                if isinstance(width_in, int) and isinstance(height_in, int):
                    expected_size = (width_in, height_in)
                    size_source = "工作流文件 EmptyLatentImage"
                break
    else:
        positive = args.positive or prompts.build_positive(args.appearance)
        if args.mode == "img2img":
            if not args.input_image:
                print("[错误] img2img 模式需要 --input-image 指定输入照片", file=sys.stderr)
                return 2
            input_path = Path(args.input_image)
            if not input_path.is_file():
                print(f"[错误] 输入照片不存在: {input_path}", file=sys.stderr)
                return 2

            ipa_image: str | None = None
            if args.ipadapter:
                try:
                    ipa_image = _upload_ipadapter_reference(args, client)
                except ValueError as exc:
                    print(f"[错误] {exc}", file=sys.stderr)
                    return 2

            upload_path = input_path
            if args.no_preprocess:
                print("[预处理] 已跳过（--no-preprocess），直接使用原图")
                try:
                    from PIL import Image as _Image
                    with _Image.open(input_path) as _im:
                        expected_size = _im.size
                    size_source = "img2img 原图（--no-preprocess，尺寸取自原图）"
                except Exception:
                    expected_size = None
            else:
                strict_note = "（严格模式：低于推荐值将直接拒绝）" if args.strict_min_side else "（低于推荐值仅警告，不拒绝）"
                print(f"[模式] 推荐最低边长 {args.min_side}px {strict_note}")
                try:
                    target = preprocess.parse_target_size(args.target_size)
                    pad_color = preprocess.parse_pad_color(args.pad_color)
                    config.PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
                    dest = config.PREPROCESSED_DIR / f"{input_path.stem}_{target[0]}x{target[1]}.png"
                    result = preprocess.preprocess_image(
                        input_path, dest, target_size=target,
                        pad_color=pad_color, max_pixels=args.max_pixels,
                        min_side=args.min_side,
                        max_file_size=args.max_file_size,
                        test_mode=args.test_mode,
                        strict_min_side=args.strict_min_side,
                    )
                except preprocess.PreprocessError as exc:
                    print(f"[预处理失败] {exc}", file=sys.stderr)
                    return 2
                _print_preprocess(result)
                expected_size = result.final_size
                size_source = f"img2img 预处理输出（--target-size {args.target_size}）"
                sidecar = dest.with_suffix(".preprocess.json")
                sidecar.write_text(
                    json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
                )
                print(f"[元数据] {sidecar}")
                upload_path = dest

            uploaded_name = client.upload_image(upload_path)
            print(f"[上传] {upload_path} -> ComfyUI 输入目录 ({uploaded_name})")
            workflow = graph.build_sdxl_img2img(
                ckpt_name=args.ckpt,
                input_image=uploaded_name,
                positive_prompt=positive,
                negative_prompt=args.negative,
                seed=args.seed,
                steps=args.steps,
                cfg=args.cfg,
                sampler_name=args.sampler,
                scheduler=args.scheduler,
                denoise=args.denoise,
                filename_prefix=args.prefix,
                lora_name=args.lora,
                lora_strength_model=args.lora_strength_model,
                lora_strength_clip=args.lora_strength_clip,
                ipadapter_name=args.ipadapter,
                ipadapter_image=ipa_image,
                clip_vision_name=args.clip_vision,
                ipadapter_weight=args.ipadapter_weight,
                ipadapter_weight_type=args.ipadapter_weight_type,
                ipadapter_start_at=args.ipadapter_start_at,
                ipadapter_end_at=args.ipadapter_end_at,
                bg_removal_model=args.bg_removal,
            )
            print(f"[工作流] 现场构建 SDXL 图生图 (denoise={args.denoise})"
                  f"{' + LoRA' if args.lora else ''}"
                  f"{' + IP-Adapter' if args.ipadapter else ''}"
                  f"{' + 背景分割' if args.bg_removal else ''}")
        else:
            ipa_image: str | None = None
            if args.ipadapter:
                try:
                    ipa_image = _upload_ipadapter_reference(args, client)
                except ValueError as exc:
                    print(f"[错误] {exc}", file=sys.stderr)
                    return 2

            cn_image: str | None = None
            if args.controlnet:
                if not args.controlnet_image:
                    print("[错误] 使用 --controlnet 时必须用 --controlnet-image 提供参照图",
                          file=sys.stderr)
                    return 2
                cn_path = Path(args.controlnet_image)
                if not cn_path.is_file():
                    print(f"[错误] ControlNet 参照图不存在: {cn_path}", file=sys.stderr)
                    return 2
                cn_image = _upload_image(cn_path, client)

            workflow = graph.build_sdxl_txt2img(
                ckpt_name=args.ckpt,
                positive_prompt=positive,
                negative_prompt=args.negative,
                seed=args.seed,
                steps=args.steps,
                cfg=args.cfg,
                sampler_name=args.sampler,
                scheduler=args.scheduler,
                width=args.width,
                height=args.height,
                batch_size=args.batch_size,
                filename_prefix=args.prefix,
                lora_name=args.lora,
                lora_strength_model=args.lora_strength_model,
                lora_strength_clip=args.lora_strength_clip,
                ipadapter_name=args.ipadapter,
                ipadapter_image=ipa_image,
                clip_vision_name=args.clip_vision,
                ipadapter_weight=args.ipadapter_weight,
                ipadapter_weight_type=args.ipadapter_weight_type,
                ipadapter_start_at=args.ipadapter_start_at,
                ipadapter_end_at=args.ipadapter_end_at,
                bg_removal_model=args.bg_removal,
                controlnet_name=args.controlnet,
                controlnet_image=cn_image,
                controlnet_strength=args.controlnet_strength,
                preprocess_photo=args.preprocess_in_graph,
                photo_side=args.photo_side,
            )
            print(f"[工作流] 现场构建 SDXL 文生图"
                  f"{' + LoRA' if args.lora else ''}"
                  f"{' + IP-Adapter' if args.ipadapter else ''}"
                  f"{' + 背景分割' if args.bg_removal else ''}"
                  f"{' + ControlNet' if args.controlnet else ''}"
                  f"{' + 图内预处理' if args.preprocess_in_graph else ''}")
            expected_size = (args.width, args.height)
            size_source = "txt2img --width/--height"

    node_defs = load_node_defs(client, workflow)
    problems = graph.validate_api_workflow(workflow, node_defs)
    if problems:
        print("[校验失败] API 工作流存在以下问题:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"[校验通过] API 工作流 ({len(workflow)} 个节点)")

    ui_workflow = graph.api_to_ui_workflow(workflow, node_defs)
    ui_problems = graph.validate_ui_workflow(ui_workflow, node_defs)
    if ui_problems:
        print("[校验失败] UI 工作流存在以下问题:", file=sys.stderr)
        for problem in ui_problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"[校验通过] UI 工作流 ({len(ui_workflow['nodes'])} 个节点, "
          f"{len(ui_workflow['links'])} 条连线)")

    if args.save_workflow:
        prefix = Path(args.save_workflow)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        api_path = prefix.with_suffix(".api.json")
        ui_path = prefix.with_suffix(".ui.json")
        api_path.write_text(json.dumps(workflow, indent=2, ensure_ascii=False), encoding="utf-8")
        ui_path.write_text(json.dumps(ui_workflow, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[已保存] {api_path}")
        print(f"[已保存] {ui_path}")

    min_w, min_h = config.FORMAL_OUTPUT_MIN_SIZE
    rec_w, rec_h = config.RECOMMENDED_OUTPUT_SIZE
    if expected_size is None:
        print("[尺寸] 预期输出尺寸无法可靠推断，跳过生成前预警（以生成后实测为准）")
    elif expected_size[0] < min_w or expected_size[1] < min_h:
        print(
            f"[预警] 预期输出 {expected_size[0]}x{expected_size[1]}（来源：{size_source}）"
            f"低于最低输出要求 {min_w}x{min_h}，本次产物将**不达标**。"
            f"如需达标，请改用原生生成尺寸 {rec_w}x{rec_h}"
            f"（本工具不会自动更改默认尺寸）",
            file=sys.stderr,
        )
    else:
        print(f"[尺寸] 预期输出 {expected_size[0]}x{expected_size[1]}（来源：{size_source}）"
              f"≥ {min_w}x{min_h}")

    if args.dry_run:
        print("[dry-run] 未执行生成。")
        return 0

    out_dir = Path(args.out)
    print(f"[执行] 提交到 {config.COMFY_BASE_URL} ...")
    started = time.monotonic()
    last_note = [0.0]

    def on_progress(elapsed: float) -> None:
        if elapsed - last_note[0] >= 15.0:
            last_note[0] = elapsed
            print(f"  ... 已等待 {elapsed:.0f}s")

    try:
        result = client.generate(
            workflow, output_dir=out_dir, timeout=args.timeout, on_progress=on_progress
        )
    except ComfyUIError as exc:
        print(f"[执行失败] {exc}", file=sys.stderr)
        return 1
    except TimeoutError as exc:
        print(f"[执行超时] {exc}", file=sys.stderr)
        client.interrupt()
        return 1

    total = time.monotonic() - started
    print(f"[完成] prompt_id = {result.prompt_id}")
    print(f"[耗时] 端到端 {total:.2f}s（客户端计时）/ {result.elapsed_seconds:.2f}s（含等待）")
    for path in result.saved_paths:
        verdict = preprocess.verify_output(path)
        if verdict["readable"]:
            width, height = verdict["size"]
            print(f"[输出] {path} [{width}x{height} {verdict['mode']}]")
        else:
            print(f"[输出] {path} [无法读取: {verdict['reasons'][0]}]")

        if verdict["readable"] and not verdict["ok"]:
            print(f"[规格] 产物校验：**不达标**（最低要求 {min_w}x{min_h}，"
                  f"且正式规格要求透明背景 PNG）", file=sys.stderr)
            for reason in verdict["reasons"]:
                print(f"        - {reason}", file=sys.stderr)
            print("        注意：本次生成执行成功，但**不等于**项目验收通过；"
                  "尺寸/alpha 只是其中一部分指标。", file=sys.stderr)
        elif verdict["readable"]:
            print(f"[规格] 产物校验：尺寸与 alpha 达标（{width}x{height} ≥ "
                  f"{min_w}x{min_h}，含 alpha）")
            print("       注意：尺寸与 alpha 达标**不代表**人脸还原、风格、耗时等其余指标达标。")
        else:
            print("[规格] 产物**无法读取**，跳过尺寸/alpha 校验；"
                  "本次执行不能视为产出合格结果。", file=sys.stderr)

    if not result.saved_paths:
        print("[警告] 执行成功但没有产出图片", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
