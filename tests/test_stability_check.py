"""三维度稳定性判定器的单元测试。

该脚本把 §5 的判定规则（任一维度偏差 >10% 即不达标）变成可执行代码，
此前这件事只靠手工串命令做过一次。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "stability_check", ROOT / "scripts" / "stability_check.py")
assert SPEC and SPEC.loader
stability = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stability)

STAB20 = [ROOT / "results" / f"stab20_s{s}_00001_.png" for s in (1, 42, 123, 777, 2024)]
TIMINGS = [416, 441, 447, 433, 450]
SOURCE = ROOT / "inputs" / "img" / "demo.jpg"


def _row(path="p.png", width=1088, coverage=0.26, hair=0.99, clothing=0.99):
    return {"path": path, "width": width, "height": width, "sha256": "x" * 64,
            "coverage": coverage, "hair": hair, "clothing": clothing}


# --- 偏差计算 ---------------------------------------------------------------

def test_relative_deviation_uses_full_precision():
    """**回归**：偏差必须用未舍入的原始值计算。

    第 29 轮发现：此前手工文档里的 3.9% 是用四舍五入后的 26.2/25.2 算的，
    用精确值应为 4.12%。用舍入值会让"精度"悄悄丢失。
    """
    # 真实覆盖率（第 20 轮 5 件产物的 alpha>128 占比，单位 %）
    values = [26.232614, 25.796963, 25.519876, 25.173517, 25.873669]
    assert stability.relative_deviation(values) == pytest.approx(4.1179, abs=1e-3)
    # 若改用四舍五入到 1 位小数的值算，会得到 3.89% —— 正是当时文档里的错误数字
    rounded = [26.2, 25.8, 25.5, 25.2, 25.9]
    assert stability.relative_deviation(rounded) == pytest.approx(3.89, abs=0.02)


def test_relative_deviation_is_zero_for_identical_values():
    assert stability.relative_deviation([5.0, 5.0, 5.0]) == pytest.approx(0.0)


def test_relative_deviation_needs_two_points():
    assert stability.relative_deviation([1.0]) is None
    assert stability.relative_deviation([]) is None


def test_relative_deviation_ignores_none():
    assert stability.relative_deviation([1.0, None, 1.1]) == pytest.approx(
        (1.1 - 1.0) / 1.05 * 100, abs=0.01)


# --- 判定 -------------------------------------------------------------------

def test_dim_passes_within_threshold():
    dim = stability._dim("工程", "耗时", [100, 105, 108], "s")
    assert dim["deviation"] <= stability.THRESHOLD_PERCENT
    assert dim["passed"] is True


def test_dim_fails_beyond_threshold():
    dim = stability._dim("配色", "服装主色", [0.1, 0.99], "")
    assert dim["deviation"] > stability.THRESHOLD_PERCENT
    assert dim["passed"] is False


def test_judge_passes_when_all_dimensions_are_stable():
    rows = [_row(coverage=0.26, hair=0.99, clothing=0.98),
            _row(coverage=0.261, hair=0.991, clothing=0.981)]
    verdict = stability.judge(rows, [100.0, 101.0])
    assert verdict["passed"] is True


def test_judge_fails_when_clothing_swings():
    """配色大幅波动时，即使工程维度完好，总判定也必须不通过。"""
    rows = [_row(clothing=0.108), _row(clothing=0.993)]
    verdict = stability.judge(rows, [100.0, 100.5])
    assert verdict["passed"] is False
    failing = [d for d in verdict["dimensions"] if not d["passed"]]
    assert [d["name"] for d in failing] == ["服装主色相似度"]


def test_judge_covers_all_three_dimensions():
    verdict = stability.judge([_row(), _row()], [100.0, 101.0])
    dims = {d["dimension"] for d in verdict["dimensions"]}
    assert dims == {"工程", "配色"}
    names = {d["name"] for d in verdict["dimensions"]}
    assert {"耗时", "前景覆盖率", "输出边长", "发色相似度", "服装主色相似度"} <= names


# --- 真实数据（第 20 轮 5 件产物） -------------------------------------------

@pytest.mark.skipif(not all(p.is_file() for p in STAB20) or not SOURCE.is_file(),
                    reason="第 20 轮稳定性产物不齐全")
def test_real_data_judgement_matches_documented_numbers():
    """**用真实数据校验判定器**，并锁定文档里记录的偏差数值。

    文档中的 7.77% / 4.12% / 0.62% / 293.45% 必须与判定器的输出一致 ——
    这正是第 29 轮修正掉的手工精度误差（原先记为 3.9% / 0.60%）。
    """
    rows = [stability.measure(p, SOURCE, None, "http://127.0.0.1:8188") for p in STAB20]
    verdict = stability.judge(rows, TIMINGS)
    got = {d["name"]: d["deviation"] for d in verdict["dimensions"]}
    assert got["耗时"] == pytest.approx(7.773, abs=0.01)
    assert got["前景覆盖率"] == pytest.approx(4.118, abs=0.01)
    assert got["输出边长"] == pytest.approx(0.0, abs=0.001)
    assert got["发色相似度"] == pytest.approx(0.623, abs=0.01)
    assert got["服装主色相似度"] == pytest.approx(293.451, abs=0.01)
    assert verdict["passed"] is False


def test_report_renders_threshold_and_verdict():
    rows = [_row(clothing=0.1), _row(clothing=0.99)]
    verdict = stability.judge(rows, [440.0, 441.0])
    text = stability.render(rows, verdict, "src.jpg", "备注")
    assert "10" in text and "总判定" in text and "不通过" in text
    assert "不是人脸识别" in text, "必须写明身份指标的边界"


# --- 参数校验 ---------------------------------------------------------------

def test_rejects_fewer_than_two_outputs(tmp_path, capsys):
    src = tmp_path / "s.jpg"; Image.new("RGB", (10, 10)).save(src)
    out = tmp_path / "o.png"; Image.new("RGBA", (10, 10)).save(out)
    import sys
    argv = sys.argv
    sys.argv = ["stability_check.py", "--source", str(src), "--outputs", str(out)]
    try:
        assert stability.main() == 2
    finally:
        sys.argv = argv
    assert "至少需要 2 件" in capsys.readouterr().err


def test_rejects_timings_length_mismatch(tmp_path, capsys):
    src = tmp_path / "s.jpg"; Image.new("RGB", (10, 10)).save(src)
    outs = []
    for i in (1, 2):
        p = tmp_path / f"o{i}.png"; Image.new("RGBA", (10, 10)).save(p); outs.append(str(p))
    import sys
    argv = sys.argv
    sys.argv = ["stability_check.py", "--source", str(src),
                "--outputs", ",".join(outs), "--timings", "1,2,3"]
    try:
        assert stability.main() == 2
    finally:
        sys.argv = argv
    assert "不一致" in capsys.readouterr().err


# --- 身份维度（--identity） --------------------------------------------------

def test_identity_dimension_is_opt_in():
    """默认不加载 CLIP 权重（保持秒级）；只有 --identity 才计入身份维度。"""
    rows = [_row(), _row()]
    without = stability.judge(rows, [100.0, 101.0])
    assert not any(d["dimension"] == "身份" for d in without["dimensions"])
    with_id = stability.judge(rows, [100.0, 101.0], [0.93, 0.94, 0.95])
    assert any(d["dimension"] == "身份" for d in with_id["dimensions"])


def test_identity_dimension_fails_when_similarities_swing():
    rows = [_row(), _row()]
    verdict = stability.judge(rows, [100.0, 101.0], [0.5, 0.95])
    identity = next(d for d in verdict["dimensions"] if d["dimension"] == "身份")
    assert identity["passed"] is False


def test_identity_note_differs_when_included():
    """报告必须如实说明身份维度是否已计入。"""
    rows = [_row(), _row()]
    without = stability.render(rows, stability.judge(rows, [1.0, 1.0]), "s", "")
    assert "身份维度未判定" in without
    with_id = stability.render(rows, stability.judge(rows, [1.0, 1.0], [0.9, 0.91]), "s", "")
    assert "身份维度已计入" in with_id
    assert "不是人脸识别" in with_id


def test_relative_deviation_is_scale_sensitive_near_zero():
    """**方法学提醒**：相似度接近 0 时，相对偏差会被放大。

    实测：服装主色 0.1085 / 0.1448 / 0.1111（三个都"完全不达标"）的相对偏差
    仍高达 29.9%。这不代表工具出错，而是"相对偏差"在近零区间的固有性质 ——
    因此该数值应与**是否达到 0.90 阈值**一起看，不能只看偏差百分比。
    """
    deviation = stability.relative_deviation([0.1085, 0.1448, 0.1111])
    assert deviation is not None and deviation > 25
    # 但三者都远低于 0.90 阈值 → 结论一致：完全不达标
    assert all(v < 0.90 for v in (0.1085, 0.1448, 0.1111))


@pytest.mark.slow
@pytest.mark.skipif(not all(p.is_file() for p in STAB20) or not SOURCE.is_file(),
                    reason="第 20 轮稳定性产物不齐全")
def test_identity_dimension_on_real_data_is_stable():
    """真实数据：身份维度应当通过（与第 23 轮 0.068 极差的结论一致）。"""
    from chibitoy import identity as identity_module
    if not Path(identity_module.DEFAULT_CLIP_VISION).is_file():
        pytest.skip("本地 CLIP 视觉权重不存在")
    sims = stability._identity_similarities(STAB20, None)
    rows = [stability.measure(p, SOURCE, None, "http://127.0.0.1:8188") for p in STAB20]
    verdict = stability.judge(rows, TIMINGS, sims)
    identity = next(d for d in verdict["dimensions"] if d["dimension"] == "身份")
    assert identity["passed"] is True
    assert identity["deviation"] == pytest.approx(7.27, abs=0.2)


# --- 生成与预热（--generate / --warmup） -------------------------------------

GEN_ARGS = ["--mode", "txt2img", "--width", "1088", "--steps", "2"]


def _make_source(tmp_path) -> str:
    """脚本会先校验源文件存在，故测试必须先造一个真实文件。"""
    src = tmp_path / "s.jpg"
    Image.new("RGB", (32, 32), (200, 180, 170)).save(src)
    return str(src)


def _fake_generator(monkeypatch, created: list):
    """假的生成入口：按 --prefix 造出文件，记录 argv。"""
    from chibitoy import generate as gen_module

    def fake_main(argv):
        created.append(list(argv))
        prefix = argv[argv.index("--prefix") + 1]
        out_dir = Path(argv[argv.index("--out") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        # 必须写**真实可解码**的 PNG —— 判定阶段会打开它并读 alpha
        Image.new("RGBA", (64, 64), (255, 255, 255, 255)).save(out_dir / f"{prefix}_00001_.png")
        return 0

    monkeypatch.setattr(gen_module, "main", fake_main)


def test_warmup_sample_is_discarded_and_not_counted(tmp_path, monkeypatch):
    """**核心断言**：预热那次必须被丢弃，且不进入判定。

    若预热产物混入样本，判定结果就会被一次冷启动污染 —— 那正是 §5 要避免的。
    """
    monkeypatch.chdir(tmp_path)
    created: list = []
    _fake_generator(monkeypatch, created)
    code = stability.main(["--source", _make_source(tmp_path), "--generate", "2", "--warmup",
                           "--base-seed", "10", "--prefix", "t", "--"] + GEN_ARGS)
    try:
        prefixes = [argv[argv.index("--prefix") + 1] for argv in created]
        assert prefixes == ["t_warmup", "t_s10", "t_s11"], prefixes
        # 预热文件必须被删除，只剩 2 个计入判定的样本
        left = sorted(p.name for p in (tmp_path / "results" / "stability_gen").glob("*.png"))
        assert left == ["t_s10_00001_.png", "t_s11_00001_.png"], left
    finally:
        assert code in (0, 1)      # 判定结果本身不关心，行为正确即可


def test_without_warmup_no_extra_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    created: list = []
    _fake_generator(monkeypatch, created)
    stability.main(["--source", _make_source(tmp_path), "--generate", "2",
                    "--prefix", "t", "--"] + GEN_ARGS)
    prefixes = [argv[argv.index("--prefix") + 1] for argv in created]
    assert prefixes == ["t_s1", "t_s2"], prefixes


def test_generate_strips_user_out_dir(tmp_path, monkeypatch):
    """生成器的 --out 是输出目录；用户传入会被剔除，否则脚本找不到样本。"""
    monkeypatch.chdir(tmp_path)
    created: list = []
    _fake_generator(monkeypatch, created)
    stability.main(["--source", _make_source(tmp_path), "--generate", "1", "--prefix", "t",
                    "--"] + GEN_ARGS + ["--out", "/tmp/elsewhere"])
    assert "/tmp/elsewhere" not in created[0]
    assert created[0][created[0].index("--out") + 1] == "results/stability_gen"


def test_generate_requires_passthrough(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert stability.main(["--source", _make_source(tmp_path), "--generate", "2"]) == 2
    assert "生成参数" in capsys.readouterr().err


def test_generate_and_outputs_are_mutually_exclusive(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code = stability.main(["--source", _make_source(tmp_path), "--generate", "2",
                           "--outputs", "a.png", "--"] + GEN_ARGS)
    assert code == 2
    assert "不能同时使用" in capsys.readouterr().err


def test_requires_outputs_or_generate(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert stability.main(["--source", _make_source(tmp_path)]) == 2
    assert "二选一" in capsys.readouterr().err


def test_generate_rejects_seed_passthrough(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code = stability.main(["--source", _make_source(tmp_path), "--generate", "2",
                           "--"] + GEN_ARGS + ["--seed", "9"])
    assert code == 2
    assert "--seed" in capsys.readouterr().err


def test_generation_captures_timings_into_the_verdict(tmp_path, monkeypatch):
    """**回归**：用 --generate 时，脚本必须自己采集耗时并纳入判定。

    第 36 轮踩过的坑：`timings = None` 的初始化排在了生成块**之后**，
    把刚采集到的耗时覆盖成 None，导致判定里**整个耗时维度消失**而无人察觉。
    这条测试断言耗时维度确实出现且取自实测。
    """
    monkeypatch.chdir(tmp_path)
    created: list = []
    _fake_generator(monkeypatch, created)
    result = {}
    original_judge = stability.judge          # 必须先取出原函数，否则会自我递归

    def capturing_judge(rows, timings, identity=None):
        result["timings"] = timings
        return original_judge(rows, timings, identity)

    monkeypatch.setattr(stability, "judge", capturing_judge)
    stability.main(["--source", _make_source(tmp_path), "--generate", "2", "--prefix", "t",
                    "--"] + GEN_ARGS)
    assert result["timings"] is not None, "耗时未被采集 —— 又出现了覆盖为 None 的问题"
    assert len(result["timings"]) == 2
    assert all(t >= 0 for t in result["timings"])


def test_timings_from_generation_appear_as_a_dimension(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _fake_generator(monkeypatch, [])
    rows = [{"path": "a", "width": 1088, "height": 1088, "sha256": "x",
             "coverage": 0.26, "hair": 0.9, "clothing": 0.9},
            {"path": "b", "width": 1088, "height": 1088, "sha256": "y",
             "coverage": 0.261, "hair": 0.9, "clothing": 0.9}]
    verdict = stability.judge(rows, [100.0, 101.0])
    assert any(d["name"] == "耗时" for d in verdict["dimensions"])
