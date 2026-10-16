"""身份参考图处理链的单元测试（`--ipadapter-crop` 与 `--ipadapter-prep-bg`）。

这两步修过两个真实缺陷，必须有回归保护：

1. **第 4 轮掩码反转**：`RemoveBackground` 返回的是"1=主体"的前景掩码，
   而 `JoinImageWithAlpha` 内部会取反。若中间不插入 `InvertMask`，主体会变透明、背景被保留。
2. **第 20 轮蓝底注入**：IP-Adapter 读的是整张参考图，蓝底证件照会把蓝色注入产物
   （生成蓝衣 + 头部蓝色伪影）。故需先真实抠图并合成白底。

测试用假客户端，不触发任何真实生图。
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from chibitoy import config, generate


class FakeClient:
    """只记录调用、返回预置抠图结果的最小客户端。"""

    def __init__(self, matte: Path | None = None):
        self.matte = matte
        self.uploaded: list[Path] = []
        self.workflows: list[dict] = []

    def is_available(self) -> bool:
        return True

    def upload_image(self, path: Path, **kwargs) -> str:
        self.uploaded.append(Path(path))
        return Path(path).name

    def generate(self, workflow, output_dir=None, **kwargs):
        self.workflows.append(workflow)
        return SimpleNamespace(saved_paths=[self.matte] if self.matte else [], images=[])


def _write_photo(path: Path, size=(200, 300), rgb=(120, 140, 200)) -> Path:
    Image.new("RGB", size, rgb).save(path)
    return path


def _args(**overrides) -> Namespace:
    base = dict(
        ipadapter="ip-adapter-plus-face_sdxl_vit-h.safetensors",
        ipadapter_image=None,
        input_image=None,
        ipadapter_crop="none",
        ipadapter_prep_bg=False,
        bg_removal="birefnet.safetensors",
    )
    base.update(overrides)
    return Namespace(**base)


@pytest.fixture(autouse=True)
def _isolate_preprocessed(tmp_path, monkeypatch):
    """把中间产物目录指到临时目录，避免污染仓库的 tmp/preprocessed。"""
    monkeypatch.setattr(config, "PREPROCESSED_DIR", tmp_path)
    return tmp_path


def test_falls_back_to_input_image(tmp_path):
    photo = _write_photo(tmp_path / "p.jpg")
    client = FakeClient()
    name = generate._upload_ipadapter_reference(_args(input_image=str(photo)), client)
    assert name == "p.jpg"
    assert client.workflows == []


def test_requires_some_reference():
    with pytest.raises(ValueError, match="参考照片"):
        generate._upload_ipadapter_reference(_args(), FakeClient())


def test_missing_reference_file_raises(tmp_path):
    with pytest.raises(ValueError, match="不存在"):
        generate._upload_ipadapter_reference(
            _args(ipadapter_image=str(tmp_path / "nope.jpg")), FakeClient())


def test_top_square_crop_takes_head_region(tmp_path):
    """top-square 裁的是顶部正方形（头肩区域），不是中心裁剪。"""
    photo = Image.new("RGB", (200, 400), (0, 255, 0))
    photo.paste(Image.new("RGB", (200, 200), (255, 0, 0)), (0, 0))   # 顶部红色
    path = tmp_path / "p.jpg"
    photo.save(path)

    client = FakeClient()
    generate._upload_ipadapter_reference(
        _args(ipadapter_image=str(path), ipadapter_crop="top-square"), client)
    uploaded = client.uploaded[0]
    with Image.open(uploaded) as img:
        assert img.size == (200, 200)
    arr = np.array(Image.open(uploaded).convert("RGB"))
    # JPEG 往返会让 255 变成 254，故用通道主导性判断而非精确相等
    r, g, b = arr[0, 0]
    assert r > 200 and g < 50 and b < 50, f"裁出来的应当是顶部（红）而不是中间，实际 {arr[0, 0].tolist()}"


def test_prep_bg_workflow_has_invert_mask_before_join(tmp_path):
    """**回归第 4 轮缺陷**：必须在 RemoveBackground 与 JoinImageWithAlpha 之间插入 InvertMask。

    否则 alpha 语义反了 —— 主体变透明、背景被保留。
    """
    photo = _write_photo(tmp_path / "p.jpg")
    matte = tmp_path / "matte.png"
    Image.new("RGBA", (32, 32), (255, 255, 255, 255)).save(matte)

    client = FakeClient(matte=matte)
    generate._upload_ipadapter_reference(
        _args(ipadapter_image=str(photo), ipadapter_prep_bg=True), client)

    assert client.workflows, "开启了 prep-bg 却没有触发抠图工作流"
    workflow = client.workflows[0]
    chain = {node["class_type"]: node for node in workflow.values()}
    assert "RemoveBackground" in chain
    assert "InvertMask" in chain, "缺少 InvertMask —— 第 4 轮的掩码反转缺陷会复发"
    assert "JoinImageWithAlpha" in chain

    invert_src = chain["InvertMask"]["inputs"]["mask"][0]
    join_alpha_src = chain["JoinImageWithAlpha"]["inputs"]["alpha"][0]
    assert chain["RemoveBackground"] is workflow[invert_src], "InvertMask 必须接在 RemoveBackground 之后"
    assert chain["InvertMask"] is workflow[join_alpha_src], "JoinImageWithAlpha 的 alpha 必须来自 InvertMask"


def test_prep_bg_composites_onto_white(tmp_path):
    """抠图结果必须合成到**白底**：透明角落要变白，否则仍会把暗色带进参考图。"""
    photo = _write_photo(tmp_path / "p.jpg")
    matte = tmp_path / "matte.png"
    rgba = Image.new("RGBA", (32, 32), (0, 0, 0, 0))          # 全透明
    rgba.paste(Image.new("RGBA", (16, 16), (10, 200, 30, 255)), (0, 0))
    rgba.save(matte)

    client = FakeClient(matte=matte)
    name = generate._upload_ipadapter_reference(
        _args(ipadapter_image=str(photo), ipadapter_prep_bg=True), client)

    assert client.uploaded[-1].name == name
    with Image.open(client.uploaded[-1]) as out:
        assert out.mode == "RGB"
        arr = np.array(out)
    assert arr[31, 31].tolist() == [255, 255, 255], "透明区域应合成为白底"
    assert arr[0, 0].tolist() == [10, 200, 30], "主体像素应被保留"


def test_prep_bg_uses_configured_background_model(tmp_path):
    photo = _write_photo(tmp_path / "p.jpg")
    matte = tmp_path / "matte.png"
    Image.new("RGBA", (8, 8), (255, 255, 255, 255)).save(matte)

    client = FakeClient(matte=matte)
    generate._upload_ipadapter_reference(
        _args(ipadapter_image=str(photo), ipadapter_prep_bg=True,
              bg_removal="custom_bg.safetensors"), client)
    loader = next(n for n in client.workflows[0].values()
                  if n["class_type"] == "LoadBackgroundRemovalModel")
    assert loader["inputs"]["bg_removal_name"] == "custom_bg.safetensors"


def test_prep_bg_raises_when_no_result(tmp_path):
    photo = _write_photo(tmp_path / "p.jpg")
    with pytest.raises(ValueError, match="抠图"):
        generate._upload_ipadapter_reference(
            _args(ipadapter_image=str(photo), ipadapter_prep_bg=True), FakeClient(matte=None))
