"""多候选择优脚本的单元测试。

该脚本存在的理由有实测依据：第 20 轮证明同一配置仅换种子，服装主色就在
0.108~0.993 之间跳变（达标率 1/5）。这些测试保证择优逻辑本身可靠。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "best_of_n", ROOT / "scripts" / "best_of_n.py")
assert SPEC and SPEC.loader
best_of_n = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(best_of_n)


def _figure(path: Path, clothing_rgb, coverage: float, size=(200, 200)) -> Path:
    """造一张"占画布一定比例"的 RGBA 假产物：**窄头(肤色) + 宽身(给定服装色)**。

    为什么不是简单的上下两块：源照片的服装带现在按**解剖位置**取
    （``analysis.source_clothing_band``），需要"头部比肩窄"的形状才能定位头部下沿。
    均匀方形没有肩部跳变，取带会落到肤色与服装的交界上，令评分失去意义。
    """
    side = int(size[0] * coverage ** 0.5)
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    body = Image.new("RGBA", (side, side), (240, 220, 210, 255))
    head_bottom = int(side * 0.40)
    head_w = max(2, side // 2)
    head_x0 = (side - head_w) // 2
    for x in range(side):
        for y in range(head_bottom, side):
            body.putpixel((x, y), (*clothing_rgb, 255))
    for x in range(head_x0, head_x0 + head_w):
        for y in range(head_bottom):
            body.putpixel((x, y), (240, 220, 210, 255))     # 头部窄 -> 有肩部跳变
    img.paste(body, ((size[0] - side) // 2, (size[1] - side) // 2))
    img.save(path)
    return path


def test_score_rewards_matching_clothing(tmp_path):
    source = _figure(tmp_path / "src.png", (20, 20, 20), 0.25)
    good = _figure(tmp_path / "good.png", (20, 20, 20), 0.25)
    bad = _figure(tmp_path / "bad.png", (250, 250, 250), 0.25)
    assert best_of_n.score(good, source)[0] > best_of_n.score(bad, source)[0]


def test_score_penalises_degenerate_coverage(tmp_path):
    """**退化守卫**：第 10 轮的教训是纯颜色指标会被崩坏画面骗到满分。

    覆盖率明显不合理的产物（本例只有 1%）必须直接判 0 分。
    """
    source = _figure(tmp_path / "src.png", (20, 20, 20), 0.25)
    degenerate = _figure(tmp_path / "deg.png", (20, 20, 20), 0.01)
    similarity, coverage = best_of_n.score(degenerate, source)
    assert coverage < best_of_n.MIN_COVERAGE
    assert similarity == 0.0


def test_score_handles_image_without_alpha(tmp_path):
    """没有 alpha 通道的候选也要能评分（退化为边框估计前景）。"""
    source = _figure(tmp_path / "src.png", (20, 20, 20), 0.25)
    flat = tmp_path / "flat.png"
    Image.new("RGB", (200, 200), (255, 255, 255)).save(flat)
    similarity, coverage = best_of_n.score(flat, source)
    assert 0.0 <= similarity <= 1.0
    assert 0.0 <= coverage <= 1.0


def test_rejects_seed_in_passthrough(tmp_path, capsys):
    """种子必须由脚本掌控，否则候选之间不再独立。"""
    source = _figure(tmp_path / "src.png", (20, 20, 20), 0.25)
    code = best_of_n.main(["--source", str(source), "--", "--seed", "42"])
    assert code == 2
    assert "--seed" in capsys.readouterr().err


def test_rejects_non_positive_candidates(tmp_path, capsys):
    source = _figure(tmp_path / "src.png", (20, 20, 20), 0.25)
    assert best_of_n.main(["--source", str(source), "--candidates", "0"]) == 2
    assert "至少为 1" in capsys.readouterr().err


def test_rejects_missing_source(tmp_path, capsys):
    assert best_of_n.main(["--source", str(tmp_path / "nope.jpg")]) == 2
    assert "不存在" in capsys.readouterr().err


def test_strips_user_out_so_candidates_land_in_known_dir(tmp_path, monkeypatch):
    """用户传的 --out（生成器的输出目录）必须被剔除。

    否则候选文件会写到脚本不知道的位置，`results_dir.glob` 找不到任何产出。
    """
    source = _figure(tmp_path / "src.png", (20, 20, 20), 0.25)
    seen: list[list[str]] = []
    monkeypatch.setattr(best_of_n.gen, "main", lambda argv: (seen.append(list(argv)), 1)[1])
    best_of_n.main(["--source", str(source), "--candidates", "1",
                    "--", "--out", "/tmp/elsewhere", "--steps", "8"])
    assert seen, "生成器未被调用"
    assert "/tmp/elsewhere" not in seen[0]
    assert seen[0][seen[0].index("--out") + 1] == "results/best_of_n"


def _real_candidates() -> list[tuple[int, Path]]:
    found = [(seed, ROOT / "results" / f"stab20_s{seed}_00001_.png")
             for seed in (1, 42, 123, 777, 2024)]
    return [(seed, path) for seed, path in found if path.is_file()]


def test_ranking_matches_visual_verification():
    """用第 20 轮的**真实候选**验证择优结果。

    第 21 轮做过目视对照：seed 777（客观 0.993）确实是黑色服装、与源图相符；
    seed 1（客观 0.108）确实是白色连帽衫。择优必须选出 seed 777。
    """
    candidates = _real_candidates()
    if len(candidates) < 5:
        pytest.skip("第 20 轮的 5 张稳定性候选不齐全，跳过真实数据校验")
    source = ROOT / "inputs" / "img" / "demo.jpg"
    if not source.is_file():
        pytest.skip("源照片缺失")
    ranked = sorted(((best_of_n.score(path, source)[0], seed)
                     for seed, path in candidates), reverse=True)
    assert ranked[0][1] == 777, f"择优结果与目视对照不符: {ranked}"
    assert ranked[0][0] >= 0.9, "胜出候选应当达到达标线"
