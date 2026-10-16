#!/usr/bin/env python
"""多候选择优：同一张照片生成 N 个种子，按客观指标自动挑出最好的一张

实测依据:
    第 20 轮实测：同一张照片、同一套生产配置，仅换种子，服装主色还原度在
    0.108 / 0.145 / 0.111 / **0.993** / 0.144 之间跳变 —— 达标率仅 1/5、偏差 295%
    配色由采样随机性主导，无法通过调参收敛
    在没有本项目专属 LoRA 的前提下，"多生成几张再择优"是用算力换可靠性的唯一可行折中

真实端到端验证:
    以 N=3、base-seed=1 真跑一次（未挑选已知好种子）：三个候选得分 0.141 / 0.140 / 0.108
    全部不达标；脚本正确排名、正确胜出 seed=2,并如实打印"没有任何候选达到 0.9"的警告
    最后把胜出者复制到指定路径。即"生成 → 评分 → 排名 → 复制"
    失败路径也表现正确

用法::

    .venv/bin/python scripts/best_of_n.py --candidates 5 \\
        --source inputs/img/demo.jpg --out results/best.png -- \\
        --mode txt2img --width 1088 --height 1088 \\
        --input-image inputs/img/demo.jpg \\
        --ipadapter ip-adapter-plus-face_sdxl_vit-h.safetensors --ipadapter-weight 0.6 \\
        --ipadapter-crop top-square --ipadapter-prep-bg \\
        --lora Designer_BlindBox-000015.safetensors \\
        --controlnet canny_sdxl.safetensors --controlnet-image tmp/chibi_silhouette.png \\
        --bg-removal birefnet.safetensors --steps 30 \\
        --appearance "short black hair, black t-shirt"

注意：不要把 ``--seed`` 写在 ``--`` 之后 —— 种子由本脚本逐个指定。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image  # noqa: E402

from chibitoy import analysis  # noqa: E402
from chibitoy import generate as gen  # noqa: E402

# 前景覆盖率合理性区间。实测正常产物为 25.2%~26.2%
# 留出余量，用于剔除退化产物
MIN_COVERAGE = 0.12
MAX_COVERAGE = 0.45


def score(candidate: Path, source: Path) -> tuple[float, float]:
    """返回 (服装主色相似度, 前景覆盖率)。覆盖率越界时相似度按 0 处理。"""
    with Image.open(source) as src, Image.open(candidate) as gen_img:
        coverage = float(analysis.alpha_mask(gen_img).mean()) if gen_img.mode in ("RGBA", "LA") \
            else float(analysis.estimate_foreground(gen_img).mean())
        if not MIN_COVERAGE <= coverage <= MAX_COVERAGE:
            return 0.0, coverage
        result = analysis.analyze_pair(src, gen_img, str(source), str(candidate))
    return float(result.get("clothing_similarity") or 0.0), coverage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--source", required=True, help="源真人照片（评分基准）")
    parser.add_argument("--base-seed", type=int, default=1)
    parser.add_argument("--winner-out", default=None,
                        help="把胜出的一张复制到该路径（注意：不要与传给生成器的 --out 混淆）")
    parser.add_argument("--threshold", type=float, default=0.90, help="达标线")
    args, passthrough = parser.parse_known_args(argv)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    if args.candidates < 1:
        print("[错误] --candidates 至少为 1", file=sys.stderr)
        return 2
    if not Path(args.source).is_file():
        print(f"[错误] 源照片不存在: {args.source}", file=sys.stderr)
        return 2
    if any(tok.split("=", 1)[0] == "--seed" for tok in passthrough):
        print("[错误] 请勿在 -- 之后传 --seed, 种子由本脚本逐个指定", file=sys.stderr)
        return 2

    # 生成器的 --out 是"输出目录"
    # 剔除用户传入的 --out 并改用下面这个专属目录
    results_dir = Path("results/best_of_n")
    cleaned: list[str] = []
    skip_next = False
    for token in passthrough:
        if skip_next:
            skip_next = False
            continue
        name = token.split("=", 1)[0]
        if name == "--out":
            skip_next = "=" not in token
            continue
        cleaned.append(token)

    source = Path(args.source)
    print(f"[择优] 候选 {args.candidates} 张，基准种子 {args.base_seed}，候选目录 {results_dir}")
    scored: list[tuple[float, float, Path, int]] = []
    for index in range(args.candidates):
        seed = args.base_seed + index
        prefix = f"bon_seed{seed}"
        print(f"\n===== 候选 {index + 1}/{args.candidates}(seed={seed})=====")
        code = gen.main([*cleaned, "--seed", str(seed), "--prefix", prefix,
                         "--out", str(results_dir)])
        produced = sorted(results_dir.glob(f"{prefix}_*.png"))
        if code != 0 or not produced:
            print(f"[择优] seed={seed} 未产出，跳过", file=sys.stderr)
            continue
        similarity, coverage = score(produced[-1], source)
        print(f"[择优] seed={seed} 服装主色={similarity:.3f} 覆盖率={coverage * 100:.1f}%")
        scored.append((similarity, coverage, produced[-1], seed))

    if not scored:
        print("[错误] 没有任何候选产出", file=sys.stderr)
        return 1

    scored.sort(key=lambda item: -item[0])
    print("\n=== 候选排名（按服装主色相似度）===")
    for rank, (similarity, coverage, path, seed) in enumerate(scored, 1):
        flag = "达标" if similarity >= args.threshold else "不达标"
        print(f"  {rank}. seed={seed:<6} {similarity:.3f} {flag}  覆盖率 {coverage * 100:.1f}%  {path}")

    best_similarity, _, best_path, best_seed = scored[0]
    print(f"\n[择优] 胜出: seed={best_seed}  得分 {best_similarity:.3f}  {best_path}")
    if best_similarity < args.threshold:
        print(f"[择优] 注意：没有任何候选达到 {args.threshold} —— "
              f"建议加大 --candidates")
    if args.winner_out:
        Path(args.winner_out).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(best_path, args.winner_out)
        print(f"[择优] 已复制到 {args.winner_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
