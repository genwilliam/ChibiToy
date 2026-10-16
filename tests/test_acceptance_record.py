"""验收记录自动采集脚本的单元测试。

该脚本的作用是让"记录里每个数字都有出处"由工具保证。因此测试重点是：
- 采集项确实落在输出里（不能漏项）；
- 主观项**必须留白**，不得被脚本代填；
- 输入不合规（如 <500px）时必须**明确标注**，不得默认通过。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "acceptance_record", ROOT / "scripts" / "acceptance_record.py")
assert SPEC and SPEC.loader
record = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(record)


@pytest.fixture()
def sample(tmp_path):
    src = tmp_path / "src.jpg"
    out = tmp_path / "out.png"
    Image.new("RGB", (300, 400), (200, 180, 170)).save(src)
    Image.new("RGBA", (1088, 1088), (255, 255, 255, 255)).save(out)
    return src, out


def _render(sample, elapsed=12.5, width=1088, extra=""):
    src, out = sample
    src_info = record.describe_image(src)
    out_info = record.describe_image(out)
    out_info["width"] = out_info["height"] = width
    return record.render(
        {"time": "T", "cpu": "CPU", "model": "M", "memory_gb": 32.0,
         "os": "macOS", "python": "3.11", "torch": "2.x", "mps": True,
         "packages": 88, "comfyui_commit": "abc"},
        src_info, out_info,
        {"transparent": 0.7, "semi": 0.01, "opaque": 0.29, "subject": 0.3},
        {"available": True, "current": "1.1G", "peak": "24.2G", "pid": "1"},
        {"hair": 0.99, "clothing": 0.15, "source_hair": (2, 3, 4),
         "source_clothing": (1, 1, 5), "generated_hair": (2, 3, 3),
         "generated_clothing": (212, 219, 230), "generated_coverage": 0.258,
         "note": "n"},
        "cmd --x", elapsed, extra,
    )


def test_sha256_matches_hashlib(tmp_path):
    import hashlib
    target = tmp_path / "f.bin"
    target.write_bytes(b"chibitoy")
    assert record.sha256_of(target) == hashlib.sha256(b"chibitoy").hexdigest()


def test_describe_image_reports_size_and_bytes(sample):
    src, _ = sample
    info = record.describe_image(src)
    assert (info["width"], info["height"]) == (300, 400)
    assert info["bytes"] > 0 and len(info["sha256"]) == 64


def test_alpha_distribution_on_opaque_image(sample):
    _, out = sample
    dist = record.alpha_distribution(out)
    assert dist["opaque"] == pytest.approx(1.0)
    assert dist["transparent"] == pytest.approx(0.0)


def test_alpha_distribution_returns_none_without_alpha(tmp_path):
    flat = tmp_path / "flat.jpg"
    Image.new("RGB", (10, 10)).save(flat)
    assert record.alpha_distribution(flat) is None


def test_record_contains_all_objective_sections(sample):
    text = _render(sample)
    # 章节号在加入稳定性一节后整体后移（7=稳定性(可选) / 8=逐项判定 / 9=未测试项）
    for marker in ("## 1. 测试环境", "## 2. 输入照片", "## 4. 输出照片",
                   "## 5. 耗时与内存", "## 6. 客观配色指标", "## 8. 逐项判定"):
        assert marker in text, marker
    assert "deead876" not in text          # 不得凭空出现别的产物的哈希
    assert "24.2G" in text                 # 峰值内存必须落盘


def test_subjective_items_are_left_blank(sample):
    """主观项必须留白——脚本绝不代填主观结论。"""
    text = _render(sample)
    for item in ("| 1 | 完整公仔", "| 2 | 无畸形", "| 6 | 头身比",
                 "| 8 | 材质", "| 12 | 表情", "| 13 | 无文字"):
        row = next(line for line in text.splitlines() if line.startswith(item))
        assert "待人工填写" in row, row


def test_face_metric_is_marked_untested_not_guessed(sample):
    """面部特征一项必须标「未测试」，不得用颜色/视觉相似度冒充。"""
    text = _render(sample)
    row = next(line for line in text.splitlines() if line.startswith("| 9 |"))
    assert "未测试" in row
    assert "不可替代" in row


def test_undersized_input_is_flagged(sample):
    """源照片 <500px 时必须明确标注，不得默认合规。"""
    text = _render(sample)
    assert "低于需求单要求的 ≥500×500" in text


def test_undersized_output_fails_the_size_item(sample):
    text = _render(sample, width=512)
    row = next(line for line in text.splitlines() if line.startswith("| 4 | 输出 PNG"))
    assert "不通过" in row


def test_over_30s_fails_the_speed_item(sample):
    row = next(line for line in _render(sample, elapsed=393.65).splitlines()
               if line.startswith("| 14 |"))
    assert "不通过" in row


def test_color_items_use_objective_scores(sample):
    text = _render(sample)
    hair = next(line for line in text.splitlines() if line.startswith("| 10 |"))
    clothing = next(line for line in text.splitlines() if line.startswith("| 11 |"))
    assert "0.990" in hair and "通过" in hair
    assert "0.150" in clothing and "不通过" in clothing


def test_missing_files_return_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["acceptance_record.py",
                                     "--input", str(tmp_path / "nope.jpg"),
                                     "--output", str(tmp_path / "nope.png"),
                                     "--out", str(tmp_path / "r.md")])
    assert record.main() == 2
    assert "不存在" in capsys.readouterr().err


def test_memory_unavailable_is_reported_not_faked(sample, monkeypatch):
    """内存采集失败时必须写明原因，绝不能编一个数字。"""
    monkeypatch.setattr(record, "memory_footprint",
                        lambda: {"available": False, "reason": "测试用：无进程"})
    src, out = sample
    text = record.render(
        {"time": "T", "cpu": "C", "model": "M", "memory_gb": 32.0, "os": "O",
         "python": "P", "torch": "T", "mps": True, "packages": 1, "comfyui_commit": "x"},
        record.describe_image(src), record.describe_image(out), None,
        record.memory_footprint(),
        {"hair": 1.0, "clothing": 1.0, "source_hair": (0, 0, 0),
         "source_clothing": (0, 0, 0), "generated_hair": (0, 0, 0),
         "generated_clothing": (0, 0, 0), "generated_coverage": 0.2, "note": "n"},
        "", None, "")
    assert "采集失败" in text and "测试用：无进程" in text


def test_matting_helper_keeps_alpha():
    """**回归**：必须复用保留 alpha 的抠图。

    本轮踩过的坑：误用 `generate._matte_reference_to_white`（它把结果合成到白底存为 RGB），
    alpha 掩码丢失后 `analysis` 退化为边框估计，源照片发色被识别成 `(254,254,254)` 白色、
    发色相似度变成 0.014（正确值约 0.998）。
    """
    source = (ROOT / "scripts" / "acceptance_record.py").read_text(encoding="utf-8")
    assert "_matte_source_keeping_alpha" in source
    assert "alpha 掩码丢失" in source, "必须在代码里写明为何不能用另一个抠图函数"


def test_stability_section_is_embedded_when_provided(sample):
    """提供 --stability-json 时，记录必须包含三维度稳定性一节（同源）。"""
    payload = {
        "source": "inputs/img/demo.jpg",
        "verdict": {
            "threshold_percent": 10.0, "passed": False,
            "dimensions": [
                {"dimension": "工程", "name": "耗时", "deviation": 10.918, "passed": False},
                {"dimension": "配色", "name": "发色相似度", "deviation": 0.624, "passed": True},
            ],
        },
    }
    text = _render(sample)
    assert "三维度重复稳定性" not in text          # 不传时不应出现

    src, out = sample
    with_stability = record.render(
        {"time": "T", "cpu": "C", "model": "M", "memory_gb": 32.0, "os": "O",
         "python": "P", "torch": "T", "mps": True, "packages": 1, "comfyui_commit": "x"},
        record.describe_image(src), record.describe_image(out), None,
        {"available": False, "reason": "测试"},
        {"hair": 1.0, "clothing": 1.0, "source_hair": (0, 0, 0), "source_clothing": (0, 0, 0),
         "generated_hair": (0, 0, 0), "generated_clothing": (0, 0, 0),
         "generated_coverage": 0.2, "note": "n"},
        "", None, "", payload)
    assert "三维度重复稳定性" in with_stability
    assert "10.92%" in with_stability and "0.62%" in with_stability
    assert "不是人脸识别" in with_stability
    assert "❌ 不通过" in with_stability
