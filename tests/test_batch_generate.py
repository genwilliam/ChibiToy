"""批量生成脚本的单元测试。

`scripts/batch_generate.py` 是生产链路上跑批的入口，此前从未有过自动化测试。
本文件用 monkeypatch 替换真正的 `chibitoy.generate.main`，
只验证**参数改写逻辑**（不触发任何真实生图）。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "batch_generate", ROOT / "scripts" / "batch_generate.py")
assert SPEC and SPEC.loader
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


@pytest.fixture()
def photos(tmp_path) -> Path:
    for name in ("b.jpg", "a.jpg", "c.png"):
        (tmp_path / name).write_bytes(b"\xff\xd8\xff")   # 内容无关紧要，不会被真正读取
    return tmp_path


@pytest.fixture()
def captured(monkeypatch) -> list[list[str]]:
    """替换真正的生成入口，记录每次调用收到的 argv。"""
    calls: list[list[str]] = []

    def fake_main(argv):
        calls.append(list(argv))
        return 0

    monkeypatch.setattr(batch.gen, "main", fake_main)
    return calls


def _run(argv):
    return batch.main(argv)


def test_rewrites_input_image_per_photo(photos, captured):
    """每张照片都必须被改成自己的 --input-image，否则 IP-Adapter 会用错参考图。"""
    code = _run(["--inputs", str(photos), "--pattern", "*.jpg",
                 "--", "--mode", "txt2img", "--input-image", "old.jpg", "--steps", "8"])
    assert code == 0
    assert len(captured) == 2
    for call, name in zip(captured, ("a.jpg", "b.jpg")):
        assert call[call.index("--input-image") + 1] == str(photos / name)
        assert "old.jpg" not in call


def test_prefix_is_indexed_and_does_not_collide(photos, captured):
    _run(["--inputs", str(photos), "--pattern", "*.jpg", "--", "--steps", "8"])
    prefixes = [call[call.index("--prefix") + 1] for call in captured]
    assert prefixes == ["batch01_a", "batch02_b"]
    assert len(set(prefixes)) == len(prefixes)


def test_drops_dry_run(photos, captured):
    """批量必须真跑，透传里的 --dry-run 要被丢掉。"""
    _run(["--inputs", str(photos), "--pattern", "*.jpg", "--", "--dry-run", "--steps", "8"])
    assert all("--dry-run" not in call for call in captured)


def test_drops_save_workflow_and_its_value(photos, captured):
    """--save-workflow 带一个取值，必须**连值一起**丢掉。

    否则该值会作为游离参数留在后面，被 generate 的 argparse 判为无法识别。
    """
    _run(["--inputs", str(photos), "--pattern", "*.jpg",
          "--", "--save-workflow", "workflows/foo", "--steps", "8"])
    for call in captured:
        assert "--save-workflow" not in call
        assert "workflows/foo" not in call


def test_keeps_unrelated_passthrough(photos, captured):
    _run(["--inputs", str(photos), "--pattern", "*.jpg",
          "--", "--mode", "txt2img", "--width", "1088", "--seed", "42", "--steps", "8"])
    call = captured[0]
    assert call[:2] == ["--mode", "txt2img"]
    assert call[call.index("--width") + 1] == "1088"
    assert call[call.index("--seed") + 1] == "42"


def test_limit_and_only_filters(photos, captured):
    _run(["--inputs", str(photos), "--pattern", "*.jpg", "--limit", "1", "--", "--steps", "8"])
    assert len(captured) == 1
    captured.clear()
    _run(["--inputs", str(photos), "--pattern", "*.jpg", "--only", "b", "--", "--steps", "8"])
    assert len(captured) == 1
    assert "b.jpg" in captured[0][captured[0].index("--input-image") + 1]


def test_missing_directory_returns_2(tmp_path, captured):
    assert _run(["--inputs", str(tmp_path / "nope"), "--"]) == 2
    assert captured == []


def test_no_matching_photos_returns_2(photos, captured):
    assert _run(["--inputs", str(photos), "--pattern", "*.webp", "--"]) == 2
    assert captured == []


def test_png_pattern_is_selectable(photos, captured):
    _run(["--inputs", str(photos), "--pattern", "*.png", "--", "--steps", "8"])
    assert len(captured) == 1
    assert captured[0][captured[0].index("--input-image") + 1].endswith("c.png")


def test_reports_failure_and_returns_1(photos, monkeypatch):
    """任一张失败时，必须返回 1 并在 stderr 里点名是哪张。"""
    def fake_main(argv):
        return 1 if "a.jpg" in " ".join(argv) else 0

    monkeypatch.setattr(batch.gen, "main", fake_main)
    code = _run(["--inputs", str(photos), "--pattern", "*.jpg", "--", "--steps", "8"])
    assert code == 1


def test_drops_save_workflow_in_equals_form(photos, captured):
    """`--save-workflow=path` 这种写法也必须被完整丢弃。"""
    _run(["--inputs", str(photos), "--pattern", "*.jpg",
          "--", "--save-workflow=workflows/foo", "--steps", "8"])
    for call in captured:
        assert not any("save-workflow" in tok for tok in call)
        assert "workflows/foo" not in call


def test_drops_input_image_in_equals_form(photos, captured):
    """`--input-image=old.jpg` 也必须被丢掉，且不能残留取值。"""
    _run(["--inputs", str(photos), "--pattern", "*.jpg",
          "--", "--input-image=old.jpg", "--steps", "8"])
    for call in captured:
        assert not any("old.jpg" in tok for tok in call)
        assert call[call.index("--input-image") + 1].endswith((".jpg", ".png"))


# --- 阶段 4.15 修复的回归 ----------------------------------------------------

def test_batch_reports_each_image_and_writes_machine_readable_report(
        tmp_path, monkeypatch, capsys):
    """**回归**：单张异常不得终止整批；且必须写出逐张的机器可读报告。"""
    photos = tmp_path / "photos"
    photos.mkdir()
    for name in ("a.jpg", "b.jpg"):
        (photos / name).write_bytes(b"x")

    calls = []

    def fake_main(argv):
        calls.append(argv)
        if len(calls) == 1:
            raise RuntimeError("模拟单张意外异常")
        return 0

    monkeypatch.setattr(batch.gen, "main", fake_main)
    report = tmp_path / "batch_report.json"
    code = batch.main(["--inputs", str(photos), "--pattern", "*.jpg",
                    "--report", str(report)])

    assert len(calls) == 2, "第一张异常后，第二张仍必须被处理"
    assert code == 1, "有失败项时整体必须非零退出"

    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["total"] == 2
    assert data["succeeded"] == 1
    assert data["failed"] == 1
    assert len(data["records"]) == 2
    assert data["records"][0]["error"].startswith("RuntimeError")
    assert data["records"][1]["status"] == "ok"


def test_batch_counts_cost_gate_exit_code_as_failure(tmp_path, monkeypatch, capsys):
    """`--backend api` 未确认费用时返回 3（未执行）—— 批量不得记成成功。"""
    photos = tmp_path / "photos"
    photos.mkdir()
    (photos / "a.jpg").write_bytes(b"x")

    monkeypatch.setattr(batch.gen, "main", lambda argv: 3)
    report = tmp_path / "r.json"
    code = batch.main(["--inputs", str(photos), "--pattern", "*.jpg",
                    "--report", str(report)])

    assert code == 1
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["succeeded"] == 0
    assert data["failed"] == 1
