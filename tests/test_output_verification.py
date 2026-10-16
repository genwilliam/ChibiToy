"""产物校验的回归测试：绝不把损坏或未生成的产物报成"达标"。

守护的承诺（每一条都来自审查中发现并已复现的真实缺陷）:

    1. 一个**无法读取**的产物，绝不能打印 `[规格] …达标`，也不能抛出
       `UnboundLocalError`。历史缺陷：`width/height` 只在 `try` 里绑定，
       却在校验失败分支之外被打印 —— 单个损坏产物直接崩，多个产物时
       **复用上一张的尺寸**并报"达标"，即**假通过**。
    2. 无 alpha 通道的 RGB 产物必须被指出（项目要求透明背景 PNG）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chibitoy import config, generate, client, preprocess  # noqa: E402


# --- 产物校验函数（纯函数，直接测） -----------------------------------------

def test_verify_output_reads_valid_rgba_product(tmp_path):
    good = tmp_path / "ok.png"
    Image.new("RGBA", (1088, 1088), (1, 2, 3, 255)).save(good)

    verdict = preprocess.verify_output(good)

    assert verdict["readable"] is True
    assert verdict["size"] == (1088, 1088)
    assert verdict["mode"] == "RGBA"
    assert verdict["has_alpha"] is True
    assert verdict["size_reasons"] == []
    assert verdict["ok"] is True


def test_verify_output_flags_rgb_product_without_alpha(tmp_path):
    """项目要求透明背景 PNG；RGB 产物必须被标记，不能被当成合规。"""
    flat = tmp_path / "flat.png"
    Image.new("RGB", (1088, 1088), (255, 255, 255)).save(flat)

    verdict = preprocess.verify_output(flat)

    assert verdict["readable"] is True
    assert verdict["has_alpha"] is False
    assert verdict["ok"] is False
    assert any("alpha" in reason.lower() or "透明" in reason
               for reason in verdict["reasons"])


def test_verify_output_handles_corrupt_product(tmp_path):
    """损坏产物：必须如实报 readable=False，且**绝不**抛异常。"""
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not a real png at all")

    verdict = preprocess.verify_output(corrupt)

    assert verdict["readable"] is False
    assert verdict["size"] is None
    assert verdict["ok"] is False
    assert verdict["reasons"]


def test_verify_output_handles_missing_product(tmp_path):
    verdict = preprocess.verify_output(tmp_path / "nope.png")
    assert verdict["readable"] is False
    assert verdict["ok"] is False


def test_verify_output_flags_undersized_product(tmp_path):
    small = tmp_path / "small.png"
    Image.new("RGBA", (512, 512), (1, 2, 3, 255)).save(small)
    verdict = preprocess.verify_output(small)
    assert verdict["readable"] is True
    assert verdict["size_reasons"]
    assert verdict["ok"] is False


# --- 端到端：损坏产物不得导致崩溃或假通过 ------------------------------------

class _StubResult:
    def __init__(self, saved_paths):
        self.prompt_id = "p-1"
        self.saved_paths = saved_paths
        self.elapsed_seconds = 0.5
        self.images = []


def _run_local_with_products(tmp_path, monkeypatch, products):
    """用桩替换 ComfyUIClient，跑完整本地路径，返回 (退出码, stdout, stderr)。"""
    class StubClient:
        def __init__(self, *a, **k):
            pass

        def is_available(self):
            return True

        def upload_image(self, path):
            return Path(path).name

        def object_info(self, class_type):
            return {class_type: {"input": {"required": {}}}}

        def generate(self, workflow, output_dir=None, timeout=None, on_progress=None):
            return _StubResult(list(products))

        def interrupt(self):
            pass

    monkeypatch.setattr(generate, "ComfyUIClient", StubClient)
    # 让节点校验一路通过，聚焦产物校验逻辑
    monkeypatch.setattr(generate.graph, "validate_api_workflow", lambda *a, **k: [])
    monkeypatch.setattr(generate.graph, "api_to_ui_workflow",
                        lambda *a, **k: {"nodes": [], "links": []})
    monkeypatch.setattr(generate.graph, "validate_ui_workflow", lambda *a, **k: [])

    return generate.main(["--mode", "txt2img", "--width", "1088", "--height", "1088",
                          "--out", str(tmp_path / "out")])


def test_corrupt_only_product_does_not_crash_and_is_not_reported_ok(
        tmp_path, monkeypatch, capsys):
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not a real png")

    code = _run_local_with_products(tmp_path, monkeypatch, [corrupt])
    captured = capsys.readouterr()

    assert code == 0                    # 按项目约定：本块只报告、不改退出码
    assert "UnboundLocalError" not in captured.err, "损坏产物不得导致崩溃"
    assert "达标" not in captured.out, "损坏产物绝不能被报成达标"
    assert "无法读取" in captured.out or "无法校验" in captured.out
    assert "不能视为产出合格结果" in captured.err


def test_corrupt_second_product_does_not_reuse_first_products_size(
        tmp_path, monkeypatch, capsys):
    """**核心回归**：多个产物时，损坏的那个曾静默复用上一张的尺寸并报达标。"""
    good = tmp_path / "good.png"
    Image.new("RGBA", (1088, 1088), (1, 2, 3, 255)).save(good)
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not a real png")

    code = _run_local_with_products(tmp_path, monkeypatch, [good, corrupt])
    captured = capsys.readouterr()

    assert "无法读取" in captured.out or "无法校验" in captured.out, \
        "损坏产物必须被单独报出"
    assert code == 0

    # 损坏产物那一行之后不得出现"达标"
    lines = captured.out.splitlines()
    bad_index = next(i for i, line in enumerate(lines)
                     if "无法读取" in line or "无法校验" in line)
    assert not any("达标" in line for line in lines[bad_index:]), \
        "损坏产物之后不得再报达标"


def test_valid_rgba_product_still_reports_ok(tmp_path, monkeypatch, capsys):
    """正常路径不能被修坏：合规产物仍应报达标并返回 0。"""
    good = tmp_path / "good.png"
    Image.new("RGBA", (1088, 1088), (1, 2, 3, 255)).save(good)

    code = _run_local_with_products(tmp_path, monkeypatch, [good])
    captured = capsys.readouterr()

    assert code == 0
    assert "达标" in captured.out
    assert "1088x1088" in captured.out


def test_rgb_product_without_alpha_is_reported_not_ok(tmp_path, monkeypatch, capsys):
    """无 alpha 的 RGB 产物不满足透明背景要求，必须被指出。"""
    flat = tmp_path / "flat.png"
    Image.new("RGB", (1088, 1088), (255, 255, 255)).save(flat)

    code = _run_local_with_products(tmp_path, monkeypatch, [flat])
    captured = capsys.readouterr()

    assert code == 0                    # 规格问题只报告，不改退出码
    assert "不达标" in captured.err
    assert "alpha" in captured.err.lower() or "透明" in captured.err


# --- 绝不覆盖已有产物（数据丢失回归） ----------------------------------------

def test_unique_path_returns_original_when_free(tmp_path):
    target = config.unique_path(tmp_path, "a.png")
    assert target == tmp_path / "a.png"


def test_unique_path_avoids_collision_instead_of_overwriting(tmp_path):
    existing = tmp_path / "a.png"
    existing.write_bytes(b"USER DATA - MUST SURVIVE")

    target = config.unique_path(tmp_path, "a.png")

    assert target != existing
    assert target == tmp_path / "a_1.png"
    assert existing.read_bytes() == b"USER DATA - MUST SURVIVE"


def test_unique_path_increments_until_free(tmp_path):
    (tmp_path / "a.png").write_bytes(b"1")
    (tmp_path / "a_1.png").write_bytes(b"2")
    (tmp_path / "a_2.png").write_bytes(b"3")
    assert config.unique_path(tmp_path, "a.png") == tmp_path / "a_3.png"


def test_unique_path_does_not_create_files(tmp_path):
    config.unique_path(tmp_path, "a.png")
    assert list(tmp_path.iterdir()) == [], "unique_path 不得创建文件"


def test_unique_path_strips_directory_traversal(tmp_path):
    """只取 basename：服务端返回的路径不得穿越到目标目录之外。"""
    target = config.unique_path(tmp_path, "../../etc/passwd")
    assert target.parent == tmp_path
    assert target.name == "passwd"


def test_download_image_never_overwrites_existing_file(tmp_path, monkeypatch):
    """**核心回归**：已存在的同名产物曾被下一次下载静默覆盖（实测数据丢失）。"""
    import chibitoy.client as client_mod

    sentinel = tmp_path / "chibitoy_00001_.png"
    sentinel.write_bytes(b"USER PRECIOUS RESULT")

    class FakeResponse:
        status_code = 200
        content = b"\x89PNG\r\n\x1a\n" + b"new" * 8

        def raise_for_status(self):
            pass

    client = client_mod.ComfyUIClient.__new__(client_mod.ComfyUIClient)
    monkeypatch.setattr(client, "_request",
                        lambda *a, **k: FakeResponse(), raising=False)

    written = client.download_image(
        {"filename": "chibitoy_00001_.png", "subfolder": "", "type": "output"}, tmp_path)

    assert sentinel.read_bytes() == b"USER PRECIOUS RESULT", "已有产物被覆盖了"
    assert written != sentinel
    assert written.exists()


def test_download_write_failure_raises_readable_error(tmp_path, monkeypatch):
    """输出目录不可写（此处用"目录路径其实是个文件"制造 OSError）必须给出可读错误。

    **回归**：原先 PermissionError / ENOSPC / IsADirectoryError 会直接冒泡成裸栈，
    而调用方只捕获 ComfyUIError。
    """
    import chibitoy.client as client_mod

    blocker = tmp_path / "blocker"
    blocker.write_text("file, not a directory", encoding="utf-8")

    class FakeResponse:
        status_code = 200
        content = b"\x89PNG\r\n\x1a\n" + b"x" * 16

        def raise_for_status(self):
            pass

    client = client_mod.ComfyUIClient.__new__(client_mod.ComfyUIClient)
    monkeypatch.setattr(client, "_request", lambda *a, **k: FakeResponse(), raising=False)

    with pytest.raises(client_mod.ComfyUIError, match="无法写入产物"):
        client.download_image(
            {"filename": "out.png", "subfolder": "", "type": "output"}, blocker)
