"""轮廓图生成器的单元测试。

背景：第 18–26 轮的对照实验证明，轮廓图里**任何合成的肢体或服饰细节**都会被模型
当作"设计"渲染，从而产生泡泡袖、方块肩、白色缺口等异常。

这些结论此前只写在文档与代码注释里 —— 若有人改动轮廓图，不会有任何测试报警。
本文件把它们变成**可执行的断言**，让"泡泡袖不会回来"由测试而非记忆保证。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "make_chibi_silhouette", ROOT / "scripts" / "make_chibi_silhouette.py")
assert SPEC and SPEC.loader
silhouette = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(silhouette)


@pytest.fixture(scope="module")
def mask() -> np.ndarray:
    """轮廓图的前景掩码（True = 剪影内部）。"""
    return np.array(silhouette.build_silhouette().convert("L")) < 128


def _row_spans(mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    rows = np.where(mask.any(axis=1))[0]
    return np.array([mask[r].sum() for r in range(rows[0], rows[-1] + 1)]), rows[0], rows[-1]


def test_output_is_rgb_square():
    image = silhouette.build_silhouette()
    assert image.size == (silhouette.CANVAS, silhouette.CANVAS)
    assert image.mode == "RGB"


def test_build_is_deterministic():
    """同样的参数必须产出完全相同的图——否则无法复现基准样图。"""
    first = np.array(silhouette.build_silhouette())
    second = np.array(silhouette.build_silhouette())
    assert np.array_equal(first, second)


def test_foreground_is_connected_and_centred(mask):
    """主体应当是居中的单一连通块（允许两腿间有间隙）。"""
    cols = np.where(mask.any(axis=0))[0]
    centre = (cols[0] + cols[-1]) / 2
    assert abs(centre - silhouette.CANVAS / 2) < 6, "剪影未居中"


def test_no_arm_protrusions(mask):
    """**回归：轮廓图不得自带手臂形状。**

    第 26 轮实测：画了手臂时，躯干区最宽处会与头部同宽，模型随即把它渲染成
    "泡泡袖/方块肩"。**阈值 0.85 有实测标定**（躯干最宽/头部最宽）：

    | 轮廓图版本 | 比值 | 本测试判定 |
    | --- | --- | --- |
    | v2（手臂从肩线外凸） | 0.98 | ❌ 失败（正是要抓的回归） |
    | v3（手臂下移内收） | 0.87 | ❌ 失败 |
    | v4（不画手臂，当前） | 0.69 | ✅ 通过 |

    **重要限定（第 27 轮的反证）**：本断言只是**必要条件，不是充分条件**。
    第 27 轮用另一张照片（`inputs/头像.jpg`）跑同一套 v4 轮廓图时，
    产物**仍然出现了泡泡袖** —— 说明泡泡袖的成因不止轮廓图，风格 LoRA 与提示词同样会引入。
    本断言保证的是"轮廓图不主动制造手臂造型"，**不能**保证"成品不会出现泡泡袖"。
    """
    widths, top, bottom = _row_spans(mask)
    height = bottom - top + 1
    head_end = int(height * 0.45)          # 头部占上 45%
    head_peak = widths[:head_end].max()
    torso_peak = widths[head_end:int(height * 0.75)].max()
    assert torso_peak < head_peak * 0.85, (
        f"躯干区最宽 {torso_peak}px 接近头部最宽 {head_peak}px —— 疑似又画了手臂"
    )


def test_no_internal_holes_above_legs(mask):
    """**回归：上半身不得有内部空洞。**

    第 24 轮实测：画的内部辅助线会在生成结果里显影成黑线；v3 中更直接表现为白色缺口。
    标定：v2 上半身空洞 36 行、v3 有 5 行、v4 为 0 行。
    """
    rows = np.where(mask.any(axis=1))[0]
    leg_start = rows[0] + int((rows[-1] - rows[0]) * 0.72)
    for r in range(rows[0], leg_start):
        span = np.where(mask[r])[0]
        if span.size:
            assert mask[r, span[0]:span[-1] + 1].all(), f"第 {r} 行上半身存在内部空洞"


def test_leg_gap_exists(mask):
    """两腿之间应当有间隙——这是正常分腿，不是内部空洞。"""
    rows = np.where(mask.any(axis=1))[0]
    leg_zone = mask[rows[0] + int((rows[-1] - rows[0]) * 0.88):rows[-1]]
    holed = [
        r for r in leg_zone
        if (lambda s: s.size and not r[s[0]:s[-1] + 1].all())(np.where(r)[0])
    ]
    assert holed, "腿部分应存在两腿间隙"


def test_head_to_body_ratio_is_chibi(mask):
    """头身比必须落在 Q 版区间。

    轮廓图刻意做得比最终目标略高一点（模型会把头略微收小），
    但必须明显小于写实的 7~8 头身。
    """
    rows = np.where(mask.any(axis=1))[0]
    total = rows[-1] - rows[0] + 1
    head_diameter = silhouette.HEAD_RADIUS * 2
    ratio = total / head_diameter
    assert 2.0 <= ratio <= 3.0, f"头身比 {ratio:.2f} 不在 Q 版区间"


def test_head_is_wider_than_torso(mask):
    """大头小身是 Q 版的核心特征，必须成立。"""
    widths, top, bottom = _row_spans(mask)
    height = bottom - top + 1
    head_peak = widths[:int(height * 0.45)].max()
    torso_peak = widths[int(height * 0.45):int(height * 0.75)].max()
    assert head_peak > torso_peak, "头部应当宽于躯干"


def test_cli_writes_file(tmp_path, monkeypatch, capsys):
    out = tmp_path / "sil.png"
    monkeypatch.setattr("sys.argv", ["make_chibi_silhouette.py", "--out", str(out)])
    assert silhouette.main() == 0
    assert out.is_file()
    with Image.open(out) as img:
        assert img.size == (silhouette.CANVAS, silhouette.CANVAS)
    assert "已生成" in capsys.readouterr().out
