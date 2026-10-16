from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from . import config

# 支持的扩展名与 Pillow 实际格式
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}
# MPO 是手机相机常见的"多图 JPEG"容器（主图就是普通 JPEG）
# inputs/img/test.jpg 被 Pillow 报为 MPO，若直接拒绝，普通手机照片就全部不可用
JPEG_FAMILY = {"JPEG", "MPO"}

# EXIF Orientation 标签 ID
_ORIENTATION_TAG = 0x0112

DEFAULT_TARGET_SIZE = (1024, 1024)
# 白色中性背景：贴合项目"纯白背景/白底抠图"目标，也避免深色补边干扰 img2img。
DEFAULT_PAD_COLOR = (255, 255, 255)
# 40 MP：覆盖常规相机/手机照片（约 12~48 MP），拦截异常大图，避免解压时内存暴涨。
DEFAULT_MAX_PIXELS = 40_000_000
# 推荐最低边长（px）：低于此值**不再拒绝输入**，改为给出「低分辨率」警告。
# 依据正式需求单「输入分辨率 ≥500×500」；阶段 1B-3A-1 起该值降级为**推荐值**，
# 使 1 寸证件照（如 194x259、295x413）等低分辨率真实照片能够进入生成流程。
DEFAULT_MIN_SIDE = 500
# 绝对下限（px）：低于此值视为「尺寸过小 / 无效」，无法有效处理，直接拒绝并给出**具体原因**。
# 与上面的推荐值职责分离：推荐值只影响是否告警，本下限才决定是否接受。
HARD_MIN_SIDE = 64
# 最大文件大小（字节）：10,000,000 字节（十进制），在解码前用 stat() 检查，
# 避免异常大文件占满内存。
# 依据正式需求单「单张 ≤10MB」。注意：需求写的是 10,000,000 字节，
# 而非 10 MiB（10,485,760 字节），此处按需求原值取用。
DEFAULT_MAX_FILE_SIZE = 10_000_000


class PreprocessError(ValueError):
    """输入图片校验或预处理失败。"""


@dataclass
class PreprocessResult:
    """预处理结果元数据，用于记录与排查。"""

    source: str
    dest: str
    original_format: str
    original_size: tuple[int, int]        # 原始 (宽, 高)
    exif_orientation: int | None          # 原始 EXIF 方向标签（无则 None）
    orientation_applied: bool             # 是否实际执行了方向纠正
    oriented_size: tuple[int, int]        # 方向纠正后的 (宽, 高)
    resized_size: tuple[int, int]         # 等比缩放后、补边前的 (宽, 高)
    target_size: tuple[int, int]
    pad_color: tuple[int, int, int]
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    final_size: tuple[int, int]
    # 非致命提示（如测试模式下低于正式规格）；正式模式下为空列表
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_pad_color(value: str) -> tuple[int, int, int]:
    """解析补边色，支持 ``#RRGGBB`` 或 ``R,G,B``。"""
    value = (value or "").strip().lower()
    if value.startswith("#"):
        hexv = value.lstrip("#")
        if len(hexv) != 6:
            raise PreprocessError(f"补边色十六进制格式错误: {value!r}，应为 #RRGGBB")
        try:
            return tuple(int(hexv[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
        except ValueError as exc:
            raise PreprocessError(f"补边色十六进制格式错误: {value!r}") from exc

    parts = value.split(",")
    if len(parts) == 3:
        try:
            rgb = tuple(int(p.strip()) for p in parts)  # type: ignore[assignment]
        except ValueError as exc:
            raise PreprocessError(f"补边色格式错误: {value!r}") from exc
        if all(0 <= c <= 255 for c in rgb):
            return rgb  # type: ignore[return-value]
    raise PreprocessError(f"补边色格式错误: {value!r}，应为 #RRGGBB 或 R,G,B")


def parse_target_size(value: str) -> tuple[int, int]:
    """解析目标尺寸，支持 ``1024``（正方形）或 ``1024x1024`` / ``896x1152``。"""
    value = (value or "").strip().lower()
    if "x" in value:
        w_s, h_s = value.split("x", 1)
    else:
        w_s = h_s = value
    try:
        w, h = int(w_s.strip()), int(h_s.strip())
    except ValueError as exc:
        raise PreprocessError(f"目标尺寸格式错误: {value!r}，应为 1024 或 1024x1024") from exc
    if w <= 0 or h <= 0:
        raise PreprocessError(f"目标尺寸必须为正: {w}x{h}")
    return (w, h)


def is_supported_format(fmt: str | None) -> bool:
    """Pillow 报出的格式是否为受支持的输入格式。

    单独抽出来是为了能直接测这条判定 —— 真实的 MPO 文件难以合成
    （Pillow 只对真正含多图结构的文件报 MPO）。
    """
    return fmt in SUPPORTED_FORMATS or fmt in JPEG_FAMILY


def validate_image(
    path: Path | str,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> dict[str, Any]:
    """校验输入图片，返回 ``{"format": ..., "size": (w, h)}``；不合法则抛 ``PreprocessError``。

    校验顺序: 存在性 -> 扩展名 -> 文件字节数 -> 解码/格式 -> 像素总量。
    文件字节数在 ``Image.open``（解码）之前用 ``stat()`` 检查，不读取文件内容。
    """
    path = Path(path)
    try:
        if path.is_dir():
            raise PreprocessError(f"输入是目录而不是图片文件: {path}")
    except PermissionError:
        pass
    try:
        if not path.is_file():
            raise PreprocessError(f"输入图片不存在: {path}")
    except PermissionError as exc:
        raise PreprocessError(
            f"无权访问输入图片（权限不足）: {path} ({exc})") from exc

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise PreprocessError(f"不支持的扩展名 {suffix or '(无)'}；支持 {sorted(SUPPORTED_SUFFIXES)}")

    try:
        size_bytes = path.stat().st_size
    except PermissionError as exc:
        raise PreprocessError(
            f"无权读取输入图片（权限不足）: {path} ({exc})") from exc
    except OSError as exc:
        raise PreprocessError(f"无法读取输入图片信息: {path} ({exc})") from exc
    if size_bytes > max_file_size:
        raise PreprocessError(
            f"文件大小 {size_bytes} 字节（{size_bytes / (1024 * 1024):.2f} MiB）"
            f"超过上限 {max_file_size} 字节（{max_file_size / (1024 * 1024):.2f} MiB）"
        )

    try:
        with Image.open(path) as im:
            fmt = im.format
            width, height = im.size
            if width * height > max_pixels:
                raise PreprocessError(
                    f"图片像素总量 {width * height}（{width}x{height}）超过上限 {max_pixels}"
                )
            im.load()
    except PreprocessError:
        raise
    except Image.DecompressionBombError as exc:
        raise PreprocessError(f"图片像素总量超过 Pillow 安全上限（疑似解压炸弹）: {exc}") from exc
    except Exception as exc:
        raise PreprocessError(f"无法解码图片（可能已损坏）: {path} ({exc})") from exc

    if not is_supported_format(fmt):
        raise PreprocessError(
            f"不支持的图片格式 {fmt or '(未知)'}；支持 {sorted(SUPPORTED_FORMATS | JPEG_FAMILY)}")
    if width <= 0 or height <= 0:
        raise PreprocessError(f"图片尺寸非法: {width}x{height}")

    return {"format": fmt, "size": (width, height)}


def validate_min_dimensions(width: int, height: int, min_side: int = DEFAULT_MIN_SIDE) -> None:
    if width < min_side or height < min_side:
        raise PreprocessError(
            f"图像尺寸 {width}x{height} 小于最小要求 {min_side}x{min_side}"
            f"（当前为 strict_min_side=True 的严格模式；"
            f"如需接受低分辨率素材，请去掉该选项，届时只会给出警告）。"
        )


def validate_output_size(
    width: int,
    height: int,
    min_size: tuple[int, int] = config.FORMAL_OUTPUT_MIN_SIZE,
) -> list[str]:
    min_w, min_h = min_size
    reasons: list[str] = []
    if width < min_w:
        reasons.append(f"宽度 {width}px 小于最低要求 {min_w}px（差 {min_w - width}px）")
    if height < min_h:
        reasons.append(f"高度 {height}px 小于最低要求 {min_h}px（差 {min_h - height}px）")
    if reasons:
        rec_w, rec_h = config.RECOMMENDED_OUTPUT_SIZE
        reasons.append(
            f"产物 {width}x{height} 未达正式输出规格 {min_w}x{min_h}；"
            f"建议改用原生生成尺寸 {rec_w}x{rec_h}（SDXL 64 网格）；"
            f"靠放大低分辨率图片凑尺寸不算达标"
        )
    return reasons


def verify_output(
    path: Path | str,
    min_size: tuple[int, int] = config.FORMAL_OUTPUT_MIN_SIZE,
    require_alpha: bool = True,
) -> dict[str, Any]:

    path = Path(path)
    result: dict[str, Any] = {
        "path": str(path),
        "readable": False,
        "size": None,
        "mode": None,
        "has_alpha": False,
        "size_reasons": [],
        "reasons": [],
        "ok": False,
    }

    try:
        with Image.open(path) as image:      # 只读打开，绝不修改产物
            width, height, mode = image.width, image.height, image.mode
            has_alpha = "A" in mode or (mode == "P" and "transparency" in image.info)
    except Exception as exc:                  # 损坏 / 缺失 / 权限 —— 一律如实报告
        result["reasons"] = [f"无法读取产物: {type(exc).__name__}: {exc}"]
        return result

    result.update(readable=True, size=(width, height), mode=mode, has_alpha=has_alpha)

    size_reasons = validate_output_size(width, height, min_size=min_size)
    result["size_reasons"] = size_reasons
    reasons = list(size_reasons)
    if require_alpha and not has_alpha:
        reasons.append(
            f"产物缺少 alpha 通道（模式 {mode}）—— 项目正式规格要求**透明背景 PNG**；"
            f"当前产物只能算**不合规**，但**可读**"
            f"（未启用 --bg-removal 时 ComfyUI 的 SaveImage 本就输出 RGB，这是已知情况）"
        )
    result["reasons"] = reasons
    result["ok"] = not reasons
    return result


def head_square_crop_from_alpha(
    image: Image.Image,
    *,
    head_band: float = 0.35,
    side_scale: float = 2.0,
    alpha_threshold: int = 128,
) -> tuple[Image.Image, str]:
    if image.mode != "RGBA":
        return image, "输入不是 RGBA，回退为整图"
    alpha = np.asarray(image.getchannel("A"))
    mask = alpha >= alpha_threshold
    coverage = float(mask.mean())
    if coverage < 0.01 or coverage > 0.95:
        return image, f"alpha 前景占比 {coverage * 100:.1f}% 不可信，回退为整图"

    ys, xs = np.nonzero(mask)
    top, bottom = int(ys.min()), int(ys.max()) + 1
    subject_h = bottom - top

    widths = mask[top:bottom].sum(axis=1)
    head_ref = widths[:max(1, subject_h // 10)]
    head_ref = head_ref[head_ref > 0]
    jump = None
    if head_ref.size:
        threshold = float(np.median(head_ref)) * 1.8
        for i in range(max(1, int(subject_h * 0.05)), subject_h):
            if widths[i] > threshold:
                jump = i
                break
    band_bottom = min(bottom, top + (jump if jump is not None
                                     else max(1, int(subject_h * head_band))))

    band = mask[top:band_bottom]
    band_cols = np.nonzero(band.any(axis=0))[0]
    if band_cols.size == 0:
        return image, "头部区域为空，回退为整图"
    head_left, head_right = int(band_cols.min()), int(band_cols.max()) + 1
    head_w = head_right - head_left
    head_h = band_bottom - top
    center_x = (head_left + head_right) // 2
    center_y = (top + band_bottom) // 2

    width, height = image.size
    side = int(max(head_w * side_scale, head_h * 1.25))
    side = max(1, min(side, width, height))
    cx0 = min(max(0, center_x - side // 2), width - side)
    cy0 = min(max(0, center_y - side // 2), height - side)
    cropped = image.crop((cx0, cy0, cx0 + side, cy0 + side))
    note = (f"alpha 主体 y[{top},{bottom}] 占 {coverage * 100:.1f}%；"
            f"头部区域宽 {head_w}px 高 {head_h}px；裁切边长 {side}px 起点 ({cx0},{cy0})")
    return cropped, note


def apply_exif_orientation(image: Image.Image) -> tuple[Image.Image, int | None, bool]:
    """按 EXIF Orientation 纠正方向，返回 (纠正后的图, 方向标签, 是否真的纠正过)。"""
    orientation: int | None = None
    try:
        orientation = image.getexif().get(_ORIENTATION_TAG)
    except Exception:
        orientation = None
    transposed = ImageOps.exif_transpose(image)
    applied = orientation not in (None, 1)
    return transposed, orientation, applied


def _flatten_to_rgb(image: Image.Image, background: tuple[int, int, int]) -> Image.Image:
    """把任意模式归一为 RGB；带透明通道的先叠加到背景色上。"""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, background + (255,))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return image.convert("RGB")


def resize_and_pad(
    image: Image.Image,
    target_size: tuple[int, int],
    pad_color: tuple[int, int, int],
) -> tuple[Image.Image, dict[str, Any]]:
    tw, th = target_size
    w, h = image.size

    scale = min(tw / w, th / h)
    nw = max(1, round(w * scale))
    nh = max(1, round(h * scale))
    resized = image.resize((nw, nh), Image.Resampling.LANCZOS)

    pad_left = (tw - nw) // 2
    pad_top = (th - nh) // 2
    pad_right = tw - nw - pad_left
    pad_bottom = th - nh - pad_top

    canvas = Image.new("RGB", (tw, th), pad_color)
    canvas.paste(resized, (pad_left, pad_top))
    return canvas, {
        "resized_size": (nw, nh),
        "pad_left": pad_left,
        "pad_top": pad_top,
        "pad_right": pad_right,
        "pad_bottom": pad_bottom,
    }


def preprocess_image(
    source: Path | str,
    dest: Path | str,
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    pad_color: tuple[int, int, int] = DEFAULT_PAD_COLOR,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    min_side: int = DEFAULT_MIN_SIDE,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    test_mode: bool = False,
    hard_min_side: int = HARD_MIN_SIDE,
    strict_min_side: bool = False,
) -> PreprocessResult:
    """校验并预处理一张输入照片，写入 ``dest``（PNG），返回元数据。
    校验顺序:
        1. 文件存在性 / 扩展名 / 文件字节数 / 解码 / 像素总量（validate_image）
        2. EXIF 方向纠正
        3. 绝对下限（hard_min_side）：低于则拒绝，并给出具体原因
        4. 推荐边长（min_side）：低于则仅警告（阶段 1B-3A-1 起不再拒绝）
        5. 等比例缩放 + 补边（不拉伸变形）

    Args:
        min_side: 推荐最低边长，仅用于触发低分辨率警告。
        hard_min_side: 绝对可处理下限，低于此值拒绝。
        strict_min_side: 置 ``True`` 恢复旧行为（低于 ``min_side`` 直接拒绝），用于兼容与显式严格场景。
        test_mode: **兼容保留**。新策略下低分辨率默认即「接受 + 警告」，本参数不再改变尺寸行为；
            文件大小、格式、解码安全、像素总量限制始终生效。
            注意：任何情况下都**不会**通过放大来绕过校验，也**不会**因放大而声称细节变好。
    """
    source = Path(source)
    info = validate_image(source, max_pixels=max_pixels, max_file_size=max_file_size)

    notes: list[str] = []
    with Image.open(source) as im:
        original_size = im.size
        oriented, orientation, applied = apply_exif_orientation(im)
        oriented_size = oriented.size
        width, height = oriented_size

        if width < hard_min_side or height < hard_min_side:
            raise PreprocessError(
                f"图片尺寸过小，无法有效处理：{width}x{height}，最短边 {min(width, height)}px "
                f"低于可处理下限 {hard_min_side}px。"
                f"（该下限用于拦截尺寸无效或过小的图片，与推荐分辨率 {min_side}x{min_side} 无关——"
                f"后者只触发警告、不影响是否接受）"
            )

        if width < min_side or height < min_side:
            if strict_min_side:
                validate_min_dimensions(width, height, min_side)
            scale = min(target_size[0] / width, target_size[1] / height)
            new_w = max(1, round(width * scale))
            new_h = max(1, round(height * scale))
            notes.append(
                f"低分辨率输入：{width}x{height} 低于推荐边长 {min_side}px，"
                f"将按 {scale:.2f}x 等比放大到 {new_w}x{new_h} 后再补边至 "
                f"{target_size[0]}x{target_size[1]}。"
                f"注意：放大与补边**不会**提升原始细节，面部/发饰/服装细节可能不足；"
                f"如条件允许，建议改用最短边不小于 {min_side}px 的照片"
            )
        rgb = _flatten_to_rgb(oriented, pad_color)

    canvas, geom = resize_and_pad(rgb, target_size, pad_color)

    dest = Path(dest)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(dest, format="PNG")
    except OSError as exc:
        raise PreprocessError(
            f"无法写入预处理结果 {dest}: {type(exc).__name__}: {exc}"
            f"（常见原因：目标目录不可写、磁盘空间不足、文件名过长）"
        ) from exc

    return PreprocessResult(
        source=str(source),
        dest=str(dest),
        original_format=info["format"],
        original_size=original_size,
        exif_orientation=orientation,
        orientation_applied=applied,
        oriented_size=oriented_size,
        resized_size=geom["resized_size"],
        target_size=tuple(target_size),
        pad_color=tuple(pad_color),
        pad_left=geom["pad_left"],
        pad_top=geom["pad_top"],
        pad_right=geom["pad_right"],
        pad_bottom=geom["pad_bottom"],
        final_size=canvas.size,
        warnings=notes,
    )
