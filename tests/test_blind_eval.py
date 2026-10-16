"""人工盲评流程测试：**不需要任何模型或网络**。"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image


def _module():
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("blind_eval", root / "scripts" / "blind_eval.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def samples(tmp_path):
    paths = []
    for name, color in (("A0_base", (10, 10, 10)), ("A1_accessory", (200, 30, 30)),
                        ("A2_structure", (30, 200, 30))):
        path = tmp_path / f"{name}_s42_00001_.png"
        Image.new("RGB", (32, 32), color).save(path)
        paths.append(path)
    return paths


def test_make_hides_group_and_shuffles(tmp_path, samples):
    """**核心断言**：评分表与盲评文件名里**不得出现组别**，否则盲评无效。"""
    module = _module()
    out = tmp_path / "blind_eval"
    key = module.make(samples, out, seed=7)

    scores_text = (out / "scores.csv").read_text(encoding="utf-8")
    for group in ("A0_base", "A1_accessory", "A2_structure"):
        assert group not in scores_text, f"评分表泄露了组别 {group}"
    for item in key["items"]:
        assert item["group"] not in Path(item["blind_file"]).name, "盲评文件名泄露了组别"

    labels = [item["label"] for item in key["items"]]
    assert len(set(labels)) == len(samples)          # 编号唯一
    assert sorted(item["show_order"] for item in key["items"]) == [1, 2, 3]

    rows = list(csv.DictReader((out / "scores.csv").open(encoding="utf-8")))
    assert len(rows) == len(samples)
    assert set(rows[0]) >= {"评审编号", "图片编号", "翅膀袖伪影", "整体质量"}


def test_make_same_seed_is_reproducible(tmp_path, samples):
    module = _module()
    first = module.make(samples, tmp_path / "a", seed=3)
    second = module.make(samples, tmp_path / "b", seed=3)
    assert [i["group"] for i in first["items"]] == [i["group"] for i in second["items"]]


def test_make_rejects_missing_sample(tmp_path):
    module = _module()
    with pytest.raises(SystemExit, match="样本不存在"):
        module.make([tmp_path / "nope.png"], tmp_path / "o", seed=1)


def test_reveal_reports_disagreement_and_warns_single_reviewer(tmp_path, samples, capsys):
    """只有一名评审时**必须**提示不得声称多人一致性；两名评审时须标出分歧。"""
    module = _module()
    out = tmp_path / "blind_eval"
    key = module.make(samples, out, seed=11)
    labels = [item["label"] for item in key["items"]]

    header = ["评审编号", "图片编号"] + list(module.SCORE_FIELDS)
    with (out / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index, label in enumerate(labels):
            row = ["R1", label] + ["无" if f != "整体质量" else "4" for f in module.SCORE_FIELDS]
            writer.writerow(row)
        # 第二名评审对同一张图给出不同判断 -> 应被标为分歧
        row = ["R2", labels[0]] + ["明显" if f == "翅膀袖伪影" else ("无" if f != "整体质量" else "2")
                                   for f in module.SCORE_FIELDS]
        writer.writerow(row)

    assert module.reveal(out / "key.json", out / "scores.csv") == 0
    printed = capsys.readouterr().out
    assert "评审人数 2" in printed or "2" in printed
    assert "分歧" in printed
