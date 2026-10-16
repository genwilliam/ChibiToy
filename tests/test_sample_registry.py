"""样本登记与独立性检查的回归测试（阶段 4.17）。

守护的承诺:
    1. 同一张照片的**内容**不得被登记为两个独立样本（否则会伪造泛化证据）。
    2. 低分辨率素材可登记但必须**标注不满足正式输入规格**。
    3. 低于硬下限的图片必须**拒绝**。
    4. 登记时**绝不覆盖**已有样本目录，也**不动**源文件。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sample_registry as registry  # noqa: E402


@pytest.fixture
def samples_dir(tmp_path, monkeypatch):
    """把登记目录重定向到 tmp，避免污染真实 inputs/photos。"""
    target = tmp_path / "photos"
    monkeypatch.setattr(registry, "SAMPLES_DIR", target)
    return target


def _make_photo(path: Path, size: tuple[int, int], color=(120, 80, 60)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


# --- 规格检查 ---------------------------------------------------------------

def test_formal_spec_photo_has_no_warning(tmp_path):
    photo = _make_photo(tmp_path / "ok.jpg", (1200, 1600))
    info, errors, warnings = registry.inspect_photo(photo)
    assert errors == []
    assert warnings == []
    assert info["min_side"] == 1200
    assert info["sha256"]


def test_low_resolution_is_accepted_but_flagged(tmp_path):
    """295x413 是文档里 demo.jpg 的真实尺寸：接受，但必须标注不满足正式规格。"""
    photo = _make_photo(tmp_path / "demo.jpg", (295, 413))
    info, errors, warnings = registry.inspect_photo(photo)
    assert errors == []                       # **不拒绝**
    assert len(warnings) == 1
    assert "低分辨率" in warnings[0]
    assert "不满足" in warnings[0]


def test_below_hard_min_side_is_rejected(tmp_path):
    photo = _make_photo(tmp_path / "tiny.jpg", (32, 32))
    _info, errors, _warnings = registry.inspect_photo(photo)
    assert errors, "低于硬下限必须拒绝"
    assert "尺寸过小" in errors[0]


def test_missing_file_is_reported_not_raised(tmp_path):
    _info, errors, _warnings = registry.inspect_photo(tmp_path / "nope.jpg")
    assert errors and "不存在" in errors[0]


def test_oversized_file_is_rejected(tmp_path):
    """超过 10MB 上限必须在解码前被拒绝（与 preprocess 同一套常量）。"""
    photo = tmp_path / "big.jpg"
    photo.write_bytes(b"\xff\xd8\xff" + b"\x00" * 10_000_100)
    _info, errors, _warnings = registry.inspect_photo(photo)
    assert errors and "超过上限" in errors[0]


# --- 查重（核心承诺） -------------------------------------------------------

def test_duplicate_content_is_detected_by_hash(tmp_path):
    first = _make_photo(tmp_path / "a.jpg", (800, 800))
    second = tmp_path / "b.jpg"
    second.write_bytes(first.read_bytes())          # 逐字节相同，不同文件名
    info, errors, _warnings = registry.inspect_photo(second)
    assert errors == []
    existing = [{"sample_id": "S01", "source_name": "a.jpg", "sha256": info["sha256"]}]
    found = registry.check_duplicate(info, existing)
    assert found is not None and found["sample_id"] == "S01"


def test_add_rejects_duplicate_without_allow_flag(tmp_path, samples_dir, capsys):
    first = _make_photo(tmp_path / "a.jpg", (800, 800))
    duplicate = tmp_path / "b.jpg"
    duplicate.write_bytes(first.read_bytes())

    assert registry.cmd_add(first, registry.load_registry(), False) == 0
    before = sorted(p.name for p in samples_dir.iterdir())

    assert registry.cmd_add(duplicate, registry.load_registry(), False) == 1
    after = sorted(p.name for p in samples_dir.iterdir())
    assert before == after, "重复登记不得新建目录"
    assert "已登记为 S01" in capsys.readouterr().err


def test_add_registers_with_metadata(tmp_path, samples_dir):
    photo = _make_photo(tmp_path / "new.jpg", (1000, 1000))
    assert registry.cmd_add(photo, registry.load_registry(), False) == 0

    meta = json.loads((samples_dir / "S01" / "metadata.json").read_text(encoding="utf-8"))
    assert meta["sample_id"] == "S01"
    assert meta["meets_formal_input_spec"] is True
    assert meta["derived_from"] is None
    assert meta["sha256"] == registry.sha256_of(photo)
    # 存储副本内容必须与源一致
    assert (samples_dir / "S01" / "original.jpg").read_bytes() == photo.read_bytes()
    # **源文件不得被移动或删除**
    assert photo.is_file()


def test_add_marks_low_resolution_sample_as_not_meeting_spec(tmp_path, samples_dir):
    photo = _make_photo(tmp_path / "low.jpg", (295, 413))
    assert registry.cmd_add(photo, registry.load_registry(), False) == 0
    meta = json.loads((samples_dir / "S01" / "metadata.json").read_text(encoding="utf-8"))
    assert meta["meets_formal_input_spec"] is False
    assert meta["warnings"]


def test_sample_ids_increment(tmp_path, samples_dir):
    for index, color in enumerate([(10, 10, 10), (200, 20, 20)], 1):
        _make_photo(tmp_path / f"p{index}.jpg", (700, 700), color)
        assert registry.cmd_add(tmp_path / f"p{index}.jpg",
                                registry.load_registry(), False) == 0
    ids = sorted(p.name for p in samples_dir.iterdir() if p.is_dir())
    assert ids == ["S01", "S02"]


def test_add_never_overwrites_existing_dir(tmp_path, samples_dir, monkeypatch):
    """即使编号被算错，也绝不能覆盖已有样本目录。"""
    photo = _make_photo(tmp_path / "x.jpg", (800, 800))
    assert registry.cmd_add(photo, registry.load_registry(), False) == 0
    original_meta = (samples_dir / "S01" / "metadata.json").read_text(encoding="utf-8")

    monkeypatch.setattr(registry, "next_sample_id", lambda _r: "S01")
    assert registry.cmd_add(photo, [], True) == 1          # allow_duplicate + 已存在目录
    assert (samples_dir / "S01" / "metadata.json").read_text(encoding="utf-8") == original_meta


def test_list_reports_low_resolution_group(tmp_path, samples_dir, capsys):
    _make_photo(tmp_path / "ok.jpg", (900, 900))
    _make_photo(tmp_path / "low.jpg", (300, 300), (5, 5, 5))
    registry.cmd_add(tmp_path / "ok.jpg", registry.load_registry(), False)
    registry.cmd_add(tmp_path / "low.jpg", registry.load_registry(), False)

    registry.cmd_list(registry.load_registry())
    out = capsys.readouterr().out
    assert "S01" in out and "S02" in out
    assert "低分辨率素材" in out
    assert "泛化结论必须分成两组陈述" in out


def test_load_registry_skips_corrupt_metadata(tmp_path, samples_dir, capsys):
    (samples_dir / "S01").mkdir(parents=True)
    (samples_dir / "S01" / "metadata.json").write_text("{ not json", encoding="utf-8")
    assert registry.load_registry() == []
    assert "无法读取" in capsys.readouterr().err
