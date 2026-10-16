"""preprocess 模块的单元测试。

只依赖 Pillow 与临时目录，不依赖 ComfyUI 服务，可离线复现。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from chibitoy import preprocess
from chibitoy.preprocess import PreprocessError, preprocess_image


def _solid(w: int, h: int, color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (w, h), color)


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def test_parse_pad_color_hex():
    assert preprocess.parse_pad_color("#FFFFFF") == (255, 255, 255)
    assert preprocess.parse_pad_color("#0a1b2c") == (10, 27, 44)


def test_parse_pad_color_rgb():
    assert preprocess.parse_pad_color("10,20,30") == (10, 20, 30)


@pytest.mark.parametrize("bad", ["#12", "#GGGGGG", "1,2", "300,0,0", "abc", "1,2,3,4"])
def test_parse_pad_color_invalid(bad):
    with pytest.raises(PreprocessError):
        preprocess.parse_pad_color(bad)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1024", (1024, 1024)), ("1024x1024", (1024, 1024)), ("896x1152", (896, 1152))],
)
def test_parse_target_size(value, expected):
    assert preprocess.parse_target_size(value) == expected


@pytest.mark.parametrize("bad", ["0", "-1x10", "1024x", "x1024", "abc", "10x0"])
def test_parse_target_size_invalid(bad):
    with pytest.raises(PreprocessError):
        preprocess.parse_target_size(bad)


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------

def test_validate_missing_file(tmp_path):
    with pytest.raises(PreprocessError, match="不存在"):
        preprocess.validate_image(tmp_path / "nope.png")


def test_validate_unsupported_extension(tmp_path):
    p = tmp_path / "img.txt"
    p.write_bytes(b"hello")
    with pytest.raises(PreprocessError, match="扩展名"):
        preprocess.validate_image(p)


def test_validate_corrupted_file(tmp_path):
    p = tmp_path / "broken.png"
    p.write_bytes(b"\x89PNG not really a png garbage")
    with pytest.raises(PreprocessError, match="无法解码|损坏"):
        preprocess.validate_image(p)


def test_validate_oversized(tmp_path):
    p = tmp_path / "big.png"
    _solid(200, 200, (255, 0, 0)).save(p)
    with pytest.raises(PreprocessError, match="超过上限"):
        preprocess.validate_image(p, max_pixels=100)


def test_validate_supported_formats(tmp_path):
    for suffix, fmt in [(".png", "PNG"), (".jpg", "JPEG"), (".webp", "WEBP")]:
        p = tmp_path / f"img{suffix}"
        _solid(64, 64, (1, 2, 3)).save(p, format=fmt)
        info = preprocess.validate_image(p)
        assert info["format"] == fmt
        assert info["size"] == (64, 64)


# ---------------------------------------------------------------------------
# EXIF 方向
# ---------------------------------------------------------------------------

def _save_with_orientation(path, orientation, size=(100, 200)):
    im = Image.new("RGB", size)
    im.paste((255, 0, 0), (0, 0, size[0] // 2, size[1]))       # 左半红
    im.paste((0, 0, 255), (size[0] // 2, 0, size[0], size[1]))  # 右半蓝
    exif = Image.Exif()
    exif[0x0112] = orientation
    im.save(path, format="JPEG", exif=exif)
    return im


def test_exif_orientation_6_is_applied(tmp_path):
    src = tmp_path / "rotated.jpg"
    _save_with_orientation(src, 6, size=(100, 200))
    dst = tmp_path / "out.png"
    result = preprocess_image(src, dst, target_size=(200, 100), min_side=1)
    assert result.exif_orientation == 6
    assert result.orientation_applied is True
    assert result.oriented_size == (200, 100)  # 90° 旋转导致宽高互换


def test_exif_orientation_1_no_change(tmp_path):
    src = tmp_path / "normal.jpg"
    _save_with_orientation(src, 1, size=(100, 200))
    dst = tmp_path / "out.png"
    result = preprocess_image(src, dst, target_size=(100, 200), min_side=1)
    assert result.exif_orientation == 1
    assert result.orientation_applied is False
    assert result.oriented_size == (100, 200)


# ---------------------------------------------------------------------------
# 等比例缩放与补边
# ---------------------------------------------------------------------------

def test_landscape_fits_with_vertical_pads(tmp_path):
    src = tmp_path / "landscape.png"
    _solid(800, 400, (0, 255, 0)).save(src)  # 2:1 横向
    dst = tmp_path / "out.png"
    result = preprocess_image(src, dst, target_size=(1024, 1024), min_side=1)

    assert result.resized_size == (1024, 512)  # 等比缩放，不拉伸
    assert result.pad_left == 0 and result.pad_right == 0
    assert result.pad_top == 256 and result.pad_bottom == 256

    out = Image.open(dst).convert("RGB")
    assert out.size == (1024, 1024)
    assert out.getpixel((0, 0)) == (255, 255, 255)       # 顶部补边为白
    assert out.getpixel((512, 512)) == (0, 255, 0)       # 中部为内容绿


def test_portrait_fits_with_horizontal_pads(tmp_path):
    src = tmp_path / "portrait.png"
    _solid(400, 800, (0, 0, 255)).save(src)
    dst = tmp_path / "out.png"
    result = preprocess_image(src, dst, target_size=(1024, 1024), min_side=1)

    assert result.resized_size == (512, 1024)
    assert result.pad_left == 256 and result.pad_right == 256
    assert result.pad_top == 0 and result.pad_bottom == 0

    out = Image.open(dst).convert("RGB")
    assert out.getpixel((0, 512)) == (255, 255, 255)     # 左补边为白
    assert out.getpixel((512, 512)) == (0, 0, 255)       # 中部为内容蓝


def test_pad_color_is_configurable(tmp_path):
    src = tmp_path / "small.png"
    _solid(100, 100, (255, 0, 0)).save(src)
    dst = tmp_path / "out.png"
    # 目标 2:1，与 1:1 源图不同宽高比 -> 必然产生左右补边
    result = preprocess_image(src, dst, target_size=(200, 100), pad_color=(10, 20, 30), min_side=1)

    assert result.pad_color == (10, 20, 30)
    assert result.pad_left == 50 and result.pad_right == 50
    out = Image.open(dst).convert("RGB")
    assert out.getpixel((0, 50)) == (10, 20, 30)       # 左侧补边为指定色
    assert out.getpixel((100, 50)) == (255, 0, 0)      # 中部为内容红


def test_square_input_no_padding(tmp_path):
    src = tmp_path / "square.png"
    _solid(300, 300, (1, 2, 3)).save(src)
    dst = tmp_path / "out.png"
    result = preprocess_image(src, dst, target_size=(300, 300), min_side=1)
    assert result.resized_size == (300, 300)
    assert (result.pad_left, result.pad_top, result.pad_right, result.pad_bottom) == (0, 0, 0, 0)


# ---------------------------------------------------------------------------
# 其他行为
# ---------------------------------------------------------------------------

def test_does_not_overwrite_original(tmp_path):
    src = tmp_path / "orig.png"
    _solid(100, 100, (9, 8, 7)).save(src)   # 尺寸与断言无关，取 ≥HARD_MIN_SIDE 即可
    before = _sha(src)
    preprocess_image(src, tmp_path / "out.png", target_size=(100, 100), min_side=1)
    assert _sha(src) == before


def test_alpha_png_flattened_to_pad_color(tmp_path):
    src = tmp_path / "alpha.png"
    rgba = Image.new("RGBA", (100, 100), (255, 0, 0, 0))  # 全透明
    rgba.save(src)
    dst = tmp_path / "out.png"
    preprocess_image(src, dst, target_size=(100, 100), pad_color=(255, 255, 255), min_side=1)
    out = Image.open(dst).convert("RGB")
    assert out.getpixel((50, 50)) == (255, 255, 255)  # 透明区被填成白，而非黑


def test_result_metadata_fields(tmp_path):
    src = tmp_path / "m.png"
    _solid(800, 400, (5, 6, 7)).save(src)
    dst = tmp_path / "out.png"
    result = preprocess_image(src, dst, target_size=(1024, 1024), min_side=1)
    d = result.to_dict()
    for key in ("original_size", "resized_size", "target_size", "pad_color",
                "pad_left", "pad_top", "pad_right", "pad_bottom", "final_size"):
        assert key in d
    assert result.final_size == (1024, 1024)
    assert result.original_size == (800, 400)


def test_jpeg_preprocess_roundtrip(tmp_path):
    src = tmp_path / "photo.jpg"
    _solid(1200, 900, (100, 150, 200)).save(src, format="JPEG", quality=90)
    dst = tmp_path / "out.png"
    result = preprocess_image(src, dst, target_size=(1024, 1024))
    assert result.original_format == "JPEG"
    assert dst.suffix == ".png"
    assert Image.open(dst).size == (1024, 1024)


# ---------------------------------------------------------------------------
# 文件大小上限
# ---------------------------------------------------------------------------

def test_file_size_at_limit_accepted(tmp_path):
    src = tmp_path / "exact.png"
    _solid(64, 64, (1, 2, 3)).save(src)
    size = src.stat().st_size
    # 恰好等于上限 -> 应通过（校验用严格大于）
    assert preprocess.validate_image(src, max_file_size=size)["size"] == (64, 64)


def test_file_size_over_limit_rejected(tmp_path):
    src = tmp_path / "over.png"
    _solid(64, 64, (1, 2, 3)).save(src)
    size = src.stat().st_size
    with pytest.raises(PreprocessError, match="文件大小.*超过上限"):
        preprocess.validate_image(src, max_file_size=size - 1)


def test_file_size_checked_before_decode(tmp_path):
    """文件大小超限时，应在解码前被拒绝（即使内容不是合法图片）。"""
    src = tmp_path / "huge_garbage.png"
    src.write_bytes(b"not an image at all" * 1000)  # 约 20KB，内容非法
    with pytest.raises(PreprocessError, match="文件大小.*超过上限"):
        preprocess.validate_image(src, max_file_size=100)  # 100 字节上限
    # 不应走到"无法解码"那条错误


# ---------------------------------------------------------------------------
# 最小尺寸
# ---------------------------------------------------------------------------

def test_min_side_exact_boundary_accepted(tmp_path):
    src = tmp_path / "exact.png"
    _solid(256, 256, (1, 2, 3)).save(src)
    preprocess_image(src, tmp_path / "out.png", target_size=(256, 256), min_side=256)


def test_min_side_width_insufficient_strict_mode(tmp_path):
    """strict_min_side=True 时保留旧行为：低于 min_side 直接拒绝。"""
    src = tmp_path / "narrow.png"
    _solid(255, 400, (1, 2, 3)).save(src)
    with pytest.raises(PreprocessError, match="255x400 小于最小要求 256x256"):
        preprocess_image(src, tmp_path / "out.png", target_size=(256, 256),
                         min_side=256, strict_min_side=True)


def test_min_side_width_insufficient_warns_by_default(tmp_path):
    """阶段 1B-3A-1 起：默认不再拒绝低于 min_side 的图片，只给低分辨率警告。"""
    src = tmp_path / "narrow.png"
    _solid(255, 400, (1, 2, 3)).save(src)
    result = preprocess_image(src, tmp_path / "out.png", target_size=(256, 256), min_side=256)
    assert len(result.warnings) == 1
    assert "低分辨率输入" in result.warnings[0]


def test_min_side_height_insufficient_strict_mode(tmp_path):
    src = tmp_path / "short.png"
    _solid(400, 255, (1, 2, 3)).save(src)
    with pytest.raises(PreprocessError, match="400x255 小于最小要求 256x256"):
        preprocess_image(src, tmp_path / "out.png", target_size=(256, 256),
                         min_side=256, strict_min_side=True)


def test_min_side_reports_oriented_dimensions(tmp_path):
    """严格模式下，EXIF 旋转后的尺寸错误信息应反映纠正后的实际宽高，而非原始宽高。"""
    src = tmp_path / "rotated_small.jpg"
    _save_with_orientation(src, 6, size=(300, 200))  # 旋转 90° 后为 200x300
    with pytest.raises(PreprocessError, match="200x300 小于最小要求 256x256"):
        preprocess_image(src, tmp_path / "out.png", target_size=(256, 256),
                         min_side=256, strict_min_side=True)


def test_min_side_warning_reports_oriented_dimensions(tmp_path):
    """默认模式下，低分辨率警告同样应反映 EXIF 纠正后的尺寸。"""
    src = tmp_path / "rotated_small.jpg"
    _save_with_orientation(src, 6, size=(300, 200))  # 旋转 90° 后为 200x300
    result = preprocess_image(src, tmp_path / "out.png", target_size=(256, 256), min_side=256)
    assert "200x300" in result.warnings[0]


def test_min_side_not_masked_by_other_limits(tmp_path):
    """严格模式下，小尺寸图即使像素总量、文件大小都合规，仍被最小尺寸校验拦截。"""
    src = tmp_path / "small.png"
    _solid(200, 200, (1, 2, 3)).save(src)  # 200x200: 像素 4 万、文件很小
    with pytest.raises(PreprocessError, match="200x200 小于最小要求 256x256"):
        preprocess_image(src, tmp_path / "out.png", target_size=(256, 256),
                         min_side=256, strict_min_side=True)


def test_limits_are_independent(tmp_path):
    """三项上限互不替代：文件大小、像素总量、最小尺寸各自独立生效。"""
    # 像素总量超限，但文件大小与最小尺寸都合规
    src = tmp_path / "many_pixels.png"
    _solid(4000, 4000, (1, 2, 3)).save(src)  # 1600 万像素
    with pytest.raises(PreprocessError, match="像素总量.*超过上限"):
        preprocess.validate_image(src, max_pixels=1_000_000)


# ---------------------------------------------------------------------------
# 正式需求默认值（阶段 1A 变更：最小 500px、文件 ≤10,000,000 字节）
# ---------------------------------------------------------------------------

def test_default_limits_match_formal_spec():
    """默认限制必须与正式需求单一致。"""
    assert preprocess.DEFAULT_MIN_SIDE == 500
    assert preprocess.DEFAULT_MAX_FILE_SIZE == 10_000_000  # 十进制 10MB，非 10 MiB


def test_min_side_500_exact_accepted(tmp_path):
    """恰好 500×500 应通过（使用默认 min_side）。"""
    src = tmp_path / "exact500.png"
    _solid(500, 500, (1, 2, 3)).save(src)
    result = preprocess_image(src, tmp_path / "out.png", target_size=(500, 500))
    assert result.oriented_size == (500, 500)


def test_min_side_499_width_strict_mode_rejected(tmp_path):
    """strict_min_side=True 时，宽 499 仍被拒绝，错误信息给出实际尺寸与限制。"""
    src = tmp_path / "w499.png"
    _solid(499, 600, (1, 2, 3)).save(src)
    with pytest.raises(PreprocessError, match="499x600 小于最小要求 500x500"):
        preprocess_image(src, tmp_path / "out.png", target_size=(500, 500),
                         strict_min_side=True)


def test_min_side_499_width_accepted_with_warning_by_default(tmp_path):
    """阶段 1B-3A-1 起：宽 499 默认**通过**，只给低分辨率警告。"""
    src = tmp_path / "w499.png"
    _solid(499, 600, (1, 2, 3)).save(src)
    result = preprocess_image(src, tmp_path / "out.png", target_size=(500, 500))
    assert result.oriented_size == (499, 600)     # 原图未被改动
    assert len(result.warnings) == 1
    assert "低分辨率输入" in result.warnings[0]


def test_min_side_499_height_strict_mode_rejected(tmp_path):
    """strict_min_side=True 时，高 499 仍被拒绝。"""
    src = tmp_path / "h499.png"
    _solid(600, 499, (1, 2, 3)).save(src)
    with pytest.raises(PreprocessError, match="600x499 小于最小要求 500x500"):
        preprocess_image(src, tmp_path / "out.png", target_size=(500, 500),
                         strict_min_side=True)


def test_file_size_default_boundary(tmp_path):
    """默认上限 10,000,000 字节：恰好等于不触发大小错误；超过 1 字节即触发。

    内容用无意义字节，因此 '恰好等于' 的情形会在随后的解码阶段失败——
    这正好证明文件大小检查发生在解码之前。
    """
    at_limit = tmp_path / "at_limit.png"
    at_limit.write_bytes(b"x" * preprocess.DEFAULT_MAX_FILE_SIZE)
    with pytest.raises(PreprocessError, match="无法解码|损坏"):
        preprocess.validate_image(at_limit)

    over_limit = tmp_path / "over_limit.png"
    over_limit.write_bytes(b"x" * (preprocess.DEFAULT_MAX_FILE_SIZE + 1))
    with pytest.raises(PreprocessError, match="文件大小.*超过上限"):
        preprocess.validate_image(over_limit)


# ---------------------------------------------------------------------------
# 低分辨率输入策略（阶段 1B-3A-1：移除「宽高必须均 ≥500px」硬门槛）
# ---------------------------------------------------------------------------

def test_low_res_499x500_accepted_with_warning(tmp_path):
    """宽 499 不再被拒绝，改为低分辨率警告。"""
    src = tmp_path / "w499.png"
    _solid(499, 500, (1, 2, 3)).save(src)
    result = preprocess_image(src, tmp_path / "o.png", target_size=(500, 500))
    assert result.oriented_size == (499, 500)
    assert len(result.warnings) == 1
    assert "低分辨率输入" in result.warnings[0]


def test_low_res_500x499_accepted_with_warning(tmp_path):
    """高 499 不再被拒绝，改为低分辨率警告。"""
    src = tmp_path / "h499.png"
    _solid(500, 499, (1, 2, 3)).save(src)
    result = preprocess_image(src, tmp_path / "o.png", target_size=(500, 500))
    assert result.oriented_size == (500, 499)
    assert len(result.warnings) == 1


def test_strict_mode_error_explains_how_to_relax(tmp_path):
    """严格模式报错必须说明如何接受低分辨率素材（即不要用 strict_min_side）。"""
    src = tmp_path / "small.png"
    _solid(300, 300, (1, 2, 3)).save(src)
    with pytest.raises(PreprocessError, match="strict_min_side"):
        preprocess_image(src, tmp_path / "o.png", target_size=(500, 500),
                         strict_min_side=True)


@pytest.mark.parametrize(("w", "h"), [(295, 413), (194, 259)])
def test_real_id_photo_sizes_pass_size_gate(tmp_path, w, h):
    """1 寸证件照尺寸（295x413、194x259）必须通过尺寸门槛 —— 核心验收项。"""
    src = tmp_path / f"idphoto_{w}x{h}.png"
    _solid(w, h, (10, 20, 30)).save(src)
    dst = tmp_path / "o.png"
    result = preprocess_image(src, dst, target_size=(1024, 1024))

    assert result.oriented_size == (w, h)          # 原尺寸被如实记录，未被修改
    assert result.final_size == (1024, 1024)
    assert len(result.warnings) == 1               # 必须出现低分辨率警告
    assert "低分辨率输入" in result.warnings[0]
    assert f"{w}x{h}" in result.warnings[0]
    assert "不会" in result.warnings[0]            # 必须声明"放大不提升细节"
    assert Image.open(dst).size == (1024, 1024)


def test_test_mode_flag_still_accepted(tmp_path):
    """--test-mode 为兼容保留：新策略下与默认行为一致（接受 + 低分辨率警告）。"""
    src = tmp_path / "small.png"
    _solid(295, 413, (1, 2, 3)).save(src)
    result = preprocess_image(src, tmp_path / "o.png", target_size=(500, 500), test_mode=True)
    assert len(result.warnings) == 1
    assert "低分辨率输入" in result.warnings[0]


def test_test_mode_does_not_warn_for_large_image(tmp_path):
    """测试模式下，尺寸达标时不应产生警告。"""
    src = tmp_path / "big.png"
    _solid(600, 600, (1, 2, 3)).save(src)
    result = preprocess_image(src, tmp_path / "o.png", target_size=(600, 600), test_mode=True)
    assert result.warnings == []


def test_formal_mode_no_warning_for_large_image(tmp_path):
    src = tmp_path / "big.png"
    _solid(600, 600, (1, 2, 3)).save(src)
    result = preprocess_image(src, tmp_path / "o.png", target_size=(600, 600))
    assert result.warnings == []


# ---------------------------------------------------------------------------
# 双模式：测试模式只豁免最小边长，其他限制一律不豁免
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("test_mode", [False, True])
def test_both_modes_reject_oversize_file(tmp_path, test_mode):
    """两种模式都必须拒绝超过 10,000,000 字节的文件。"""
    src = tmp_path / "over.png"
    src.write_bytes(b"x" * (preprocess.DEFAULT_MAX_FILE_SIZE + 1))
    with pytest.raises(PreprocessError, match="文件大小.*超过上限"):
        preprocess_image(src, tmp_path / "o.png", test_mode=test_mode)


@pytest.mark.parametrize("test_mode", [False, True])
def test_both_modes_reject_unsupported_format(tmp_path, test_mode):
    src = tmp_path / "a.txt"
    src.write_bytes(b"hello")
    with pytest.raises(PreprocessError, match="扩展名"):
        preprocess_image(src, tmp_path / "o.png", test_mode=test_mode)


@pytest.mark.parametrize("test_mode", [False, True])
def test_both_modes_reject_corrupted_file(tmp_path, test_mode):
    src = tmp_path / "broken.png"
    src.write_bytes(b"\x89PNG garbage not a real png")
    with pytest.raises(PreprocessError, match="无法解码|损坏"):
        preprocess_image(src, tmp_path / "o.png", test_mode=test_mode)


@pytest.mark.parametrize("test_mode", [False, True])
def test_both_modes_reject_over_max_pixels(tmp_path, test_mode):
    src = tmp_path / "many.png"
    _solid(1000, 1000, (1, 2, 3)).save(src)
    with pytest.raises(PreprocessError, match="像素总量.*超过上限"):
        preprocess_image(src, tmp_path / "o.png", max_pixels=10_000, test_mode=test_mode)


def test_test_mode_still_checks_file_size_before_decode(tmp_path):
    """测试模式下，文件大小仍必须在完整解码之前检查。"""
    src = tmp_path / "over_garbage.png"
    src.write_bytes(b"not an image" * 2000)  # 内容非法但体积超限
    with pytest.raises(PreprocessError, match="文件大小.*超过上限"):
        preprocess_image(src, tmp_path / "o.png", max_file_size=100, test_mode=True)


# ---------------------------------------------------------------------------
# CLI 层：模式选择、退出码与警告行为
# ---------------------------------------------------------------------------

def test_cli_parser_default_is_formal_mode():
    from chibitoy import generate
    args = generate.build_parser().parse_args([])
    assert args.test_mode is False                      # 默认正式模式


def test_cli_parser_test_mode_flag():
    from chibitoy import generate
    args = generate.build_parser().parse_args(["--test-mode"])
    assert args.test_mode is True


class _StubClient:
    """替代 ComfyUIClient，使 CLI 测试可离线运行且无上传副作用。"""

    def __init__(self, *a, **k):
        pass

    def is_available(self):
        return True

    def upload_image(self, path, subfolder="", overwrite=True):
        return "example.png"          # 必须命中测试夹具的 LoadImage 候选项

    def object_info(self, node_class=None):
        from test_graph import NODE_DEFS
        if node_class is None:
            return dict(NODE_DEFS)
        return {node_class: NODE_DEFS[node_class]}


def _run_cli(monkeypatch, tmp_path, extra):
    from chibitoy import generate
    monkeypatch.setattr(generate, "ComfyUIClient", _StubClient)
    monkeypatch.setattr(generate.config, "PREPROCESSED_DIR", tmp_path / "pre")
    return generate.main(extra)


def test_cli_accepts_small_image_with_warning(tmp_path, monkeypatch, capsys):
    """阶段 1B-3A-1：默认不再因低分辨率拒绝，改为警告（退出码 0）。"""
    src = tmp_path / "small.png"
    _solid(295, 413, (1, 2, 3)).save(src)
    rc = _run_cli(monkeypatch, tmp_path,
                  ["--mode", "img2img", "--input-image", str(src), "--dry-run"])
    out = capsys.readouterr()
    assert rc == 0
    assert "推荐最低边长" in out.out
    assert "仅警告" in out.out
    assert "[警告]" in out.out
    assert "低分辨率输入" in out.out
    assert "295x413" in out.out


def test_cli_strict_min_side_rejects_small_image(tmp_path, monkeypatch, capsys):
    """--strict-min-side 恢复旧行为：低于推荐值直接拒绝（退出码 2）。"""
    src = tmp_path / "small.png"
    _solid(295, 413, (1, 2, 3)).save(src)
    rc = _run_cli(monkeypatch, tmp_path,
                  ["--mode", "img2img", "--input-image", str(src),
                   "--strict-min-side", "--dry-run"])
    out = capsys.readouterr()
    assert rc == 2
    assert "strict_min_side" in out.err
    assert "严格模式" in out.out


def test_cli_test_mode_flag_still_accepted(tmp_path, monkeypatch, capsys):
    """--test-mode 兼容保留：与默认行为一致（接受 + 警告，退出码 0）。"""
    src = tmp_path / "small.png"
    _solid(295, 413, (1, 2, 3)).save(src)
    rc = _run_cli(monkeypatch, tmp_path,
                  ["--mode", "img2img", "--input-image", str(src),
                   "--test-mode", "--dry-run"])
    out = capsys.readouterr()
    assert rc == 0
    assert "[警告]" in out.out
    assert "低分辨率输入" in out.out
    assert "295x413" in out.out


# ---------------------------------------------------------------------------
# 绝对下限 HARD_MIN_SIDE：尺寸过小/无效 → 具体原因（阶段 1B-3A-1）
# ---------------------------------------------------------------------------

def test_hard_min_side_constant_exposed():
    """绝对下限必须存在，且严格小于推荐边长（两者职责分离）。"""
    assert preprocess.HARD_MIN_SIDE == 64
    assert preprocess.HARD_MIN_SIDE < preprocess.DEFAULT_MIN_SIDE


@pytest.mark.parametrize(("w", "h"), [(32, 32), (63, 300), (300, 63)])
def test_below_hard_floor_rejected_with_specific_reason(tmp_path, w, h):
    """低于绝对下限必须拒绝，理由是「尺寸过小/无法有效处理」，而非笼统的「图片尺寸不足」。"""
    src = tmp_path / f"tiny_{w}x{h}.png"
    _solid(w, h, (1, 2, 3)).save(src)
    with pytest.raises(PreprocessError) as ei:
        preprocess_image(src, tmp_path / "o.png", target_size=(1024, 1024))
    msg = str(ei.value)
    assert "尺寸过小" in msg
    assert "无法有效处理" in msg
    assert f"{w}x{h}" in msg           # 报出实际尺寸
    assert "64" in msg                 # 报出具体下限
    assert "推荐分辨率" in msg          # 明确与推荐值区分
    assert "图片尺寸不足" not in msg     # 不得使用笼统措辞


def test_hard_floor_is_overridable(tmp_path):
    """hard_min_side 可显式下调，用于特殊小图场景。"""
    src = tmp_path / "tiny.png"
    _solid(32, 32, (1, 2, 3)).save(src)
    result = preprocess_image(src, tmp_path / "o.png", target_size=(64, 64), hard_min_side=16)
    assert result.final_size == (64, 64)
    assert len(result.warnings) == 1


# ---------------------------------------------------------------------------
# 低分辨率：不得非等比例拉伸
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("w", "h"), [(295, 413), (194, 259)])
def test_low_res_resize_preserves_aspect_ratio(tmp_path, w, h):
    """低分辨率图必须等比缩放：缩放后宽高比与原图一致（容差 1%），且完整落入目标框。"""
    src = tmp_path / f"id_{w}x{h}.png"
    _solid(w, h, (1, 2, 3)).save(src)
    result = preprocess_image(src, tmp_path / "o.png", target_size=(1024, 1024))
    rw, rh = result.resized_size
    assert abs(rw / rh - w / h) < 0.01        # 比例保持 → 未拉伸变形
    assert rw <= 1024 and rh <= 1024          # 完全放进目标框
    assert max(rw, rh) == 1024                # 长边贴满，缩放确实生效
    assert result.final_size == (1024, 1024)


@pytest.mark.parametrize(("w", "h"), [(295, 413), (194, 259)])
def test_low_res_warning_states_scale_and_no_detail_gain(tmp_path, w, h):
    """低分辨率警告必须写明放大倍数，并明确「放大不会提升原始细节」。"""
    src = tmp_path / f"id_{w}x{h}.png"
    _solid(w, h, (1, 2, 3)).save(src)
    result = preprocess_image(src, tmp_path / "o.png", target_size=(1024, 1024))
    msg = result.warnings[0]
    assert f"{w}x{h}" in msg                  # 原始尺寸
    assert "x 等比放大" in msg or "等比放大" in msg
    assert "不会" in msg and "细节" in msg      # 明确不提升细节
    assert "建议" in msg                       # 给出改进建议
    assert "500" in msg                        # 提到推荐边长


def test_low_res_png_jpeg_webp_all_accepted(tmp_path):
    """低分辨率下三种受支持格式都应被接受（仅警告）。"""
    import io
    from PIL import Image as _I
    for ext, fmt in [(".png", "PNG"), (".jpg", "JPEG"), (".webp", "WEBP")]:
        src = tmp_path / f"small{ext}"
        _I.new("RGB", (194, 259), (10, 20, 30)).save(src, format=fmt)
        result = preprocess_image(src, tmp_path / f"out{ext}.png", target_size=(512, 512))
        assert result.final_size == (512, 512), ext
        assert len(result.warnings) == 1, ext


# ---------------------------------------------------------------------------
# 输出尺寸校验（阶段 1B-3A-2 —— 与输入侧校验职责分离）
# ---------------------------------------------------------------------------

def test_output_size_constants_are_formal_not_lowered():
    """阈值必须是正式要求 1080，且**不得**被下调到当前默认的 1024。"""
    from chibitoy import config
    assert config.FORMAL_OUTPUT_MIN_SIZE == (1080, 1080)
    assert config.RECOMMENDED_OUTPUT_SIZE == (1088, 1088)
    assert config.FORMAL_OUTPUT_MIN_SIZE != (1024, 1024)


@pytest.mark.parametrize(("w", "h", "ok"), [
    (1024, 1024, False),   # 当前默认输出：不达标
    (1080, 1080, True),    # 恰好达标
    (1088, 1088, True),    # 推荐原生尺寸
    (1080, 1024, False),   # 仅高度不足
    (1024, 1080, False),   # 仅宽度不足
    (2048, 2048, True),
])
def test_validate_output_size_matrix(w, h, ok):
    reasons = preprocess.validate_output_size(w, h)
    assert (reasons == []) is ok
    if not ok:
        assert all(isinstance(r, str) and r.strip() for r in reasons)


def test_validate_output_size_reports_only_offending_side():
    """宽度达标、高度不足时，只应报高度。"""
    joined = " ".join(preprocess.validate_output_size(1080, 1024))
    assert "高度" in joined
    assert "宽度" not in joined
    # 反向：高度达标、宽度不足
    joined2 = " ".join(preprocess.validate_output_size(1024, 1080))
    assert "宽度" in joined2
    assert "高度" not in joined2


def test_validate_output_size_custom_threshold():
    """阈值可覆盖（用于不同验收档位）。"""
    assert preprocess.validate_output_size(600, 600, min_size=(512, 512)) == []
    assert preprocess.validate_output_size(600, 600, min_size=(2048, 2048)) != []


def test_validate_output_size_boundary_1079():
    assert preprocess.validate_output_size(1079, 1080) != []
    assert preprocess.validate_output_size(1080, 1079) != []


def test_validate_output_size_is_pure_and_never_resizes(tmp_path):
    """校验器不得做 I/O、不得修改任何图片文件。"""
    p = tmp_path / "img.png"
    _solid(64, 64, (1, 2, 3)).save(p)
    before = _sha(p)
    preprocess.validate_output_size(1024, 1024)
    preprocess.validate_output_size(4096, 4096)
    assert _sha(p) == before


def test_validate_output_size_does_not_raise_on_absurd_values():
    """只判断、不抛异常：极端值也必须返回原因列表而非异常。"""
    assert preprocess.validate_output_size(1, 1) != []
    assert preprocess.validate_output_size(0, 0) != []
    assert preprocess.validate_output_size(99999, 99999) == []


# ---------------------------------------------------------------------------
# 生成前预警（CLI 层）—— 只预警，不改变生成默认值
# ---------------------------------------------------------------------------

def test_cli_preflight_warns_for_txt2img_default(tmp_path, monkeypatch, capsys):
    """txt2img 默认 1024：应预警，且尺寸来源必须标为 txt2img。"""
    rc = _run_cli(monkeypatch, tmp_path, ["--mode", "txt2img", "--dry-run"])
    out = capsys.readouterr()
    assert rc == 0
    assert "[预警]" in out.err
    assert "txt2img --width/--height" in out.err    # 尺寸来源正确
    assert "预期输出 1024x1024" in out.err
    assert "1080x1080" in out.err


def test_cli_preflight_silent_when_meets_spec(tmp_path, monkeypatch, capsys):
    """txt2img 1088：不得出现预警。"""
    rc = _run_cli(monkeypatch, tmp_path,
                  ["--mode", "txt2img", "--width", "1088", "--height", "1088", "--dry-run"])
    out = capsys.readouterr()
    assert rc == 0
    assert "[预警]" not in out.err
    assert "预期输出 1088x1088" in out.out


def test_cli_preflight_img2img_uses_preprocessed_target(tmp_path, monkeypatch, capsys):
    """img2img 的预期尺寸来源是"预处理输出"，(须) 随 --target-size 变化。"""
    src = tmp_path / "p.png"
    _solid(600, 600, (1, 2, 3)).save(src)

    # 默认 1024 → 预警
    rc = _run_cli(monkeypatch, tmp_path,
                  ["--mode", "img2img", "--input-image", str(src), "--dry-run"])
    out = capsys.readouterr()
    assert rc == 0
    assert "[预警]" in out.err
    assert "img2img 预处理输出" in out.err
    assert "预期输出 1024x1024" in out.err

    # 1088 → 无预警
    rc2 = _run_cli(monkeypatch, tmp_path,
                   ["--mode", "img2img", "--input-image", str(src),
                    "--target-size", "1088x1088", "--dry-run"])
    out2 = capsys.readouterr()
    assert rc2 == 0
    assert "[预警]" not in out2.err
    assert "预期输出 1088x1088" in out2.out


def test_cli_preflight_img2img_no_preprocess_uses_original_size(tmp_path, monkeypatch, capsys):
    """--no-preprocess 时预期尺寸必须取自**原图**，不得误报为 --target-size。"""
    src = tmp_path / "small.png"
    _solid(194, 259, (1, 2, 3)).save(src)
    rc = _run_cli(monkeypatch, tmp_path,
                  ["--mode", "img2img", "--input-image", str(src),
                   "--no-preprocess", "--dry-run"])
    out = capsys.readouterr()
    assert rc == 0
    assert "预期输出 194x259" in out.err          # 原图尺寸
    assert "预期输出 1024x1024" not in out.err     # 不得误报为预处理目标尺寸
    assert "原图" in out.err


def test_cli_preflight_does_not_change_defaults(tmp_path, monkeypatch, capsys):
    """预警不得擅自修改任何默认尺寸：工作流里的 EmptyLatentImage 仍应为 1024。"""
    from chibitoy import generate
    captured = {}

    class _CapStub(_StubClient):
        def queue_prompt(self, workflow):
            captured["wf"] = workflow
            raise RuntimeError("stop-before-execute")   # 只看工作流，不真跑

        def generate(self, workflow, output_dir=None, timeout=0.0, on_progress=None):
            self.queue_prompt(workflow)                 # 必定抛错，阻止真实提交

    monkeypatch.setattr(generate, "ComfyUIClient", _CapStub)
    monkeypatch.setattr(generate.config, "PREPROCESSED_DIR", tmp_path / "pre")
    try:
        generate.main(["--mode", "txt2img"])
    except RuntimeError:
        pass
    latent = [n for n in captured["wf"].values()
              if n.get("class_type") == "EmptyLatentImage"][0]
    assert latent["inputs"]["width"] == 1024      # 默认未被预警逻辑改动
    assert latent["inputs"]["height"] == 1024


# ---------------------------------------------------------------------------
# 生成后真实产物校验（CLI 层）—— 仅报告，不改退出码
# ---------------------------------------------------------------------------

def _run_cli_with_output(monkeypatch, tmp_path, extra, out_size, mode="RGBA"):
    """用桩客户端返回一张指定尺寸/模式的"产物"，避免真实生图。

    ``mode`` 默认 ``RGBA``（正式规格要求的透明背景 PNG）；要模拟
    "未启用 --bg-removal 时的 RGB 产物"就显式传 ``mode="RGB"``。
    """
    from chibitoy import generate
    from chibitoy.client import GenerationResult

    product = tmp_path / f"product_{out_size[0]}x{out_size[1]}_{mode}.png"
    if mode == "RGBA":
        Image.new("RGBA", out_size, (5, 6, 7, 255)).save(product)
    else:
        Image.new("RGB", out_size, (5, 6, 7)).save(product)

    class _GenStub(_StubClient):
        def generate(self, workflow, output_dir=None, timeout=0.0, on_progress=None):
            return GenerationResult(prompt_id="stub-prompt", saved_paths=[product],
                                    elapsed_seconds=0.0)

    monkeypatch.setattr(generate, "ComfyUIClient", _GenStub)
    monkeypatch.setattr(generate.config, "PREPROCESSED_DIR", tmp_path / "pre")
    rc = generate.main(extra)
    return rc, product


def test_cli_output_below_spec_reports_but_exit_code_unchanged(tmp_path, monkeypatch, capsys):
    """产物 1024x1024：必须报告不达标，但退出码保持 0（本阶段不改退出码）。"""
    rc, _ = _run_cli_with_output(monkeypatch, tmp_path, ["--mode", "txt2img"],
                                 out_size=(1024, 1024))
    out = capsys.readouterr()
    assert rc == 0
    assert "[规格]" in out.err
    assert "不达标" in out.err
    assert "1080x1080" in out.err
    assert "不等于" in out.err          # 不得把生成成功说成验收成功


def test_cli_output_rgb_without_alpha_is_reported_as_noncompliant(tmp_path, monkeypatch, capsys):
    """未启用 --bg-removal 时产物是 RGB：必须指出缺少 alpha，且仍不改退出码。"""
    rc, _ = _run_cli_with_output(monkeypatch, tmp_path,
                                 ["--mode", "txt2img", "--width", "1088", "--height", "1088"],
                                 out_size=(1088, 1088), mode="RGB")
    out = capsys.readouterr()
    assert rc == 0, "规格问题只报告，不改退出码（与既有约定一致）"
    assert "不达标" in out.err
    assert "alpha" in out.err.lower() or "透明" in out.err


def test_cli_output_meets_spec_reports_pass(tmp_path, monkeypatch, capsys):
    """预期尺寸与实际产物都为 1088：生成前无预警，生成后报达标。"""
    rc, _ = _run_cli_with_output(monkeypatch, tmp_path,
                                 ["--mode", "txt2img", "--width", "1088", "--height", "1088"],
                                 out_size=(1088, 1088))
    out = capsys.readouterr()
    assert rc == 0
    assert "达标" in out.out
    assert "[预警]" not in out.err
    assert "不达标" not in out.err
    assert "不代表" in out.out          # 尺寸达标 ≠ 其他指标达标


def test_cli_output_check_does_not_modify_product(tmp_path, monkeypatch, capsys):
    """校验必须只读：产物图片逐字节不变，尺寸也不被"修正"。"""
    rc, product = _run_cli_with_output(monkeypatch, tmp_path, ["--mode", "txt2img"],
                                       out_size=(1024, 1024))
    capsys.readouterr()
    assert rc == 0
    with Image.open(product) as im:
        assert im.size == (1024, 1024)     # 未被放大到 1080
    # 再次运行，确认文件哈希稳定（无写入）
    h1 = _sha(product)
    rc2, _ = _run_cli_with_output(monkeypatch, tmp_path, ["--mode", "txt2img"],
                                  out_size=(1024, 1024))
    capsys.readouterr()
    assert rc2 == 0
    assert _sha(product) == h1


# --- 基于 alpha 的自适应头部裁切（新增） ------------------------------------

def _alpha_image(size, box, color=(120, 90, 70)):
    """造一张 RGBA：``box`` 内不透明，其余全透明。"""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    solid = Image.new("RGBA", (box[2] - box[0], box[3] - box[1]), (*color, 255))
    img.paste(solid, (box[0], box[1]))
    return img


def test_head_crop_uses_head_size_not_body_size():
    """**核心断言**：边长必须由**头部**尺寸决定，不能用整个身体包围盒。

    全身照的身体包围盒又宽又高（实测 781x674）；若用它的尺寸定边长，
    正方形会撑满画面，退化成近似 top-square —— 那正是第一版实现没修好的原因。
    """
    # 窄头(x 430~530) + 宽身(x 150~850)：必须按"头"定边长，不能按"身"
    img = Image.new("RGBA", (1000, 1000), (0, 0, 0, 0))
    img.paste(Image.new("RGBA", (100, 140), (120, 90, 70, 255)), (430, 80))    # 头
    img.paste(Image.new("RGBA", (700, 700), (120, 90, 70, 255)), (150, 260))   # 身
    cropped, note = preprocess.head_square_crop_from_alpha(img, side_scale=2.0)
    assert cropped.size[0] == cropped.size[1], "裁切必须是正方形"
    assert cropped.size[0] < 600, \
        f"边长 {cropped.size[0]} 过大 —— 说明用了身体尺寸而不是头部尺寸"
    assert "头部区域宽" in note


def test_head_crop_keeps_the_head_inside():
    """裁切必须把头部完整包含进去 —— 不能为了去掉颜色污染而把脸裁掉。"""
    img = _alpha_image((1000, 1000), (400, 60, 560, 220))     # 头在 y60~220
    cropped, _ = preprocess.head_square_crop_from_alpha(img, side_scale=2.0)
    assert cropped.getchannel("A").getbbox() is not None, "裁切后头部丢失"


def test_head_crop_falls_back_when_alpha_is_unusable():
    """掩码全透或全实心时不可信，必须回退整图而不是乱裁。"""
    empty = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    cropped, note = preprocess.head_square_crop_from_alpha(empty)
    assert cropped.size == (100, 100) and "回退" in note

    solid = Image.new("RGBA", (100, 100), (10, 20, 30, 255))
    cropped, note = preprocess.head_square_crop_from_alpha(solid)
    assert cropped.size == (100, 100) and "回退" in note


def test_head_crop_rejects_non_rgba():
    cropped, note = preprocess.head_square_crop_from_alpha(Image.new("RGB", (50, 50)))
    assert cropped.size == (50, 50) and "回退" in note


# --- MPO（手机多图 JPEG）必须按 JPEG 接受 -----------------------------------

def test_mpo_is_accepted_as_jpeg(tmp_path):
    """**回归**：手机相机的 MPO 容器必须可用。

    实测用户的合规样张 `inputs/img/test.jpg` 被 Pillow 报为 `MPO`；
    若直接拒绝，普通手机照片就全部不可用 —— 与需求"支持 JPG"矛盾。
    """
    # 先测判定本身（真实 MPO 文件难以合成：Pillow 只对含多图结构的文件报 MPO）
    assert preprocess.is_supported_format("MPO") is True
    assert preprocess.is_supported_format("JPEG") is True
    assert preprocess.is_supported_format("PNG") is True
    assert preprocess.is_supported_format("WEBP") is True
    assert preprocess.is_supported_format("TIFF") is False
    assert preprocess.is_supported_format(None) is False

    # 再测真实的手机 MPO 样张（存在才测，缺失则跳过，避免依赖用户文件）
    real = Path(__file__).resolve().parent.parent / "inputs" / "img" / "test.jpg"
    if not real.is_file():
        pytest.skip("缺少真实 MPO 样张 inputs/img/test.jpg")
    with Image.open(real) as probe:
        detected = probe.format
    assert preprocess.is_supported_format(detected) is True, f"真实样张格式 {detected} 未被接受"
    dest = tmp_path / "out.png"
    info = preprocess.preprocess_image(real, dest, target_size=(512, 512))
    assert dest.exists() and info.final_size == (512, 512)


def test_unsupported_format_still_rejected(tmp_path):
    src = tmp_path / "x.tiff"
    Image.new("RGB", (100, 100)).save(src, format="TIFF")
    # 扩展名检查会先触发，故两种提示都接受
    with pytest.raises(preprocess.PreprocessError, match="不支持"):
        preprocess.preprocess_image(src, tmp_path / "o.png", target_size=(64, 64))


# --- 阶段 4.15：输入诊断的准确性 ---------------------------------------------

def test_directory_as_input_reports_directory_not_missing(tmp_path):
    """目录曾被报成"输入图片不存在"，误导排错；必须明确说是目录。"""
    from chibitoy.preprocess import PreprocessError, validate_image

    with pytest.raises(PreprocessError, match="目录"):
        validate_image(tmp_path)


def test_permission_error_is_reported_readably_not_as_traceback(tmp_path):
    """无权访问曾直接抛 PermissionError 栈；必须转成可读的 PreprocessError。"""
    import os
    import stat as stat_mod

    from chibitoy.preprocess import PreprocessError, validate_image

    if os.geteuid() == 0:
        pytest.skip("以 root 运行时权限位不生效")

    blocked = tmp_path / "blocked"
    blocked.mkdir()
    image = blocked / "a.jpg"
    image.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    blocked.chmod(0)
    try:
        with pytest.raises(PreprocessError):
            validate_image(image)
    finally:
        blocked.chmod(0o755)          # 必须恢复，否则 tmp_path 清理会失败


def test_write_failure_raises_readable_preprocess_error(tmp_path):
    """写盘失败（此处用"目标父路径是文件"制造 OSError）必须是可读的 PreprocessError。

    **回归**：长文件名会触发 ENAMETOOLONG，而调用方只捕获 PreprocessError，
    原先会直接冒泡成裸栈。这里不发散到具体的 errno，只验证类型与提示。
    """
    from chibitoy.preprocess import PreprocessError, preprocess_image

    src = tmp_path / "src.png"
    Image.new("RGB", (100, 100), (1, 2, 3)).save(src)

    blocker = tmp_path / "blocker"
    blocker.write_text("I am a file, not a directory", encoding="utf-8")
    dest = blocker / "cannot" / "write.png"      # 父路径是文件 -> mkdir 失败

    with pytest.raises(PreprocessError, match="无法写入预处理结果"):
        preprocess_image(src, dest, target_size=(64, 64))
