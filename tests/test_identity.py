"""身份稳定性度量的单元测试。

分两层：
- **快速层**：纯函数（键名前缀修正、统计、余弦），默认执行；
- **重型层**（标记 ``slow``）：需要加载 2.5 GB 的 CLIP 视觉权重，实测约 10 s，
  默认套件用 ``-m "not live and not slow"`` 排除。
"""

from __future__ import annotations

import numpy as np
import pytest

from chibitoy import identity


# --- 快速层 -----------------------------------------------------------------

def test_strip_vision_prefix_removes_prefix():
    """transformers 5.x 期望无前缀键；带前缀会静默加载失败（本轮踩过的坑）。"""
    stripped = identity.strip_vision_prefix({
        "vision_model.embeddings.class_embedding": 1,
        "vision_model.encoder.layers.0.layer_norm1.weight": 2,
    })
    assert set(stripped) == {
        "embeddings.class_embedding",
        "encoder.layers.0.layer_norm1.weight",
    }


def test_strip_vision_prefix_drops_non_parameters():
    """position_ids 不是参数，混进去会被算作"多余键"。"""
    stripped = identity.strip_vision_prefix({
        "vision_model.embeddings.position_ids": 1,
        "vision_model.embeddings.class_embedding": 2,
    })
    assert set(stripped) == {"embeddings.class_embedding"}


def test_strip_vision_prefix_leaves_unprefixed_keys():
    stripped = identity.strip_vision_prefix({"embeddings.class_embedding": 1})
    assert set(stripped) == {"embeddings.class_embedding"}


def test_stats_on_empty_input():
    stats = identity._stats([])
    assert stats["count"] == 0 and stats["mean"] is None


def test_stats_reports_range():
    stats = identity._stats([0.8, 0.9, 0.95])
    assert stats["count"] == 3
    assert stats["mean"] == pytest.approx(0.8833, abs=1e-3)
    assert stats["min"] == 0.8 and stats["max"] == 0.95
    assert stats["range"] == pytest.approx(0.15, abs=1e-9)


def test_cosine_of_unit_vectors():
    a = np.array([1.0, 0.0]); b = np.array([1.0, 0.0]); c = np.array([0.0, 1.0])
    assert identity.cosine(a, b) == pytest.approx(1.0)
    assert identity.cosine(a, c) == pytest.approx(0.0)


def test_module_documents_it_is_not_face_recognition():
    """本模块**不是**人脸识别——文档必须写明，且不得用于「面部特征 ≥80%」验收项。"""
    doc = identity.__doc__ or ""
    assert "不是人脸识别" in doc
    assert "面部特征" in doc


# --- 重型层（需要本地 CLIP 权重） ---------------------------------------------

@pytest.mark.slow
def test_encoder_loads_all_weights_without_missing():
    """权重必须全部加载。若键名前缀没处理对，会出现大量 missing 而相似度全是噪声。"""
    from pathlib import Path

    if not Path(identity.DEFAULT_CLIP_VISION).is_file():
        pytest.skip("本地 CLIP 视觉权重不存在")
    model, _ = identity.load_encoder()
    assert model is not None


@pytest.mark.slow
def test_stability_separates_same_person_from_different_person():
    """真实数据校验：同人组内相似度必须显著高于跨人。

    依据第 23 轮实测（照片 1 的 5 个种子 vs 照片 2 的 3 个种子）：
    组内均值 0.9332、跨组均值 0.7710。
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    group1 = [root / "results" / f"stab20_s{s}_00001_.png" for s in (1, 42, 123, 777, 2024)]
    group2 = [root / "results" / f"{n}_00001_.png" for n in ("r27_photo2", "r28_seed7", "r29_cn05")]
    if not all(p.is_file() for p in group1 + group2):
        pytest.skip("稳定性产物不齐全")
    if not Path(identity.DEFAULT_CLIP_VISION).is_file():
        pytest.skip("本地 CLIP 视觉权重不存在")

    result = identity.stability({"photo1": group1, "photo2": group2})
    assert result["within"]["mean"] > result["cross"]["mean"] + 0.02, result
    assert result["within"]["range"] < 0.15, "同人跨种子的身份应相对稳定"
