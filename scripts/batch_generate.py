#!/usr/bin/env python
"""批量生成：对目录中的多张真人照片依次运行同一套 ChibiToy 管线

用法:

    .venv/bin/python scripts/batch_generate.py --inputs inputs/photos \\
        --pattern "*.jpg" \\
        -- --mode txt2img --width 1088 --height 1088 \\
        --ipadapter ip-adapter-plus-face_sdxl_vit-h.safetensors --ipadapter-weight 0.6 \\
        --ipadapter-crop top-square --bg-removal birefnet.safetensors \\
        --steps 30 --seed 42 --appearance "short black hair, black t-shirt"

说明:

- ``--`` 之后的参数**原样透传**给 ``chibitoy.generate``，因此批量与单张使用完全相同的参数集，
  不会出现两套逻辑不一致的问题。
- ``--ipadapter`` 需要为每张照片单独指定参考图，本脚本会自动把 ``--input-image``
  改写为当前处理的那张照片。
- 每张照片的输出前缀自动加序号与文件名，互不覆盖。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chibitoy import generate as gen  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--inputs", required=True, help="输入照片所在的目录")
    parser.add_argument("--pattern", default="*.jpg", help="匹配模式（默认 *.jpg）")
    parser.add_argument("--limit", type=int, default=0, help="最多处理几张（0 表示不限制）")
    parser.add_argument("--only", default=None, help="只处理文件名包含该子串的照片")
    parser.add_argument("--report", default=None,
                        help="逐张结果的 JSON 报告路径"
                             "（默认 <项目根>/outputs/batches/batch_report.json，"
                             "不随当前工作目录变化）")
    args, passthrough = parser.parse_known_args(argv)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    src_dir = Path(args.inputs)
    if not src_dir.is_dir():
        print(f"[错误] 输入目录不存在: {src_dir}", file=sys.stderr)
        return 2

    photos = sorted(p for p in src_dir.glob(args.pattern) if p.is_file())
    if args.only:
        photos = [p for p in photos if args.only in p.name]
    if args.limit:
        photos = photos[: args.limit]
    if not photos:
        print(f"[错误] {src_dir} 下没有匹配 {args.pattern} 的照片", file=sys.stderr)
        return 2

    # 透传参数里去掉：
    #   --input-image   由本脚本逐张指定
    #   --save-workflow 批量导出工作流没有意义，且必须连它的取值一起丢掉
    #   --dry-run       批量要真跑
    DROP_WITH_VALUE = ("--input-image", "--save-workflow")
    DROP_FLAG = ("--dry-run",)
    cleaned: list[str] = []
    skip_next = False
    for token in passthrough:
        if skip_next:
            skip_next = False
            continue
        name = token.split("=", 1)[0]
        if name in DROP_WITH_VALUE:
            skip_next = "=" not in token # `--opt=v` 形式没有独立取值
            continue
        if name in DROP_FLAG:
            continue
        cleaned.append(token)

    print(f"[批量] 共 {len(photos)} 张，参数: {' '.join(cleaned)}")
    failures: list[str] = []
    records: list[dict] = []
    started = time.monotonic()
    for index, photo in enumerate(photos, 1):
        prefix = f"batch{index:02d}_{photo.stem}"
        print(f"\n===== [{index}/{len(photos)}] {photo.name} -> 前缀 {prefix} =====")
        t0 = time.monotonic()
        # 单张的意外异常不得终止整批：否则前面已完成的产物无法汇总，
        # 后面的照片也不会被处理（历史缺陷：无 try/except，一张异常整批崩）
        try:
            code = gen.main([
                *cleaned,
                "--input-image", str(photo),
                "--prefix", prefix,
            ])
            error = ""
        except Exception as exc:
            code = 1
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.monotonic() - t0
        status = "ok" if code == 0 else "failed"
        print(f"[批量] {photo.name} 退出码={code} 耗时={elapsed:.1f}s")
        if error:
            print(f"[批量] {photo.name} 异常: {error}", file=sys.stderr)
        records.append({
            "index": index,
            "input": str(photo),
            "prefix": prefix,
            "exit_code": code,
            "status": status,
            "elapsed_seconds": round(elapsed, 2),
            "error": error,
        })
        if code != 0:
            failures.append(photo.name)

    total = time.monotonic() - started
    succeeded = len(photos) - len(failures)
    print(f"\n[批量完成] 成功 {succeeded}/{len(photos)}，总耗时 {total:.1f}s")
    # 机器可读的逐张结果：便于后续统计与排错，不依赖解析 stdout
    report = {
        "total": len(photos),
        "succeeded": succeeded,
        "failed": len(failures),
        "failed_inputs": failures,
        "total_elapsed_seconds": round(total, 2),
        "params": cleaned,
        "records": records,
    }
    report_path = (Path(args.report) if args.report
                   else PROJECT_ROOT / "results" / "batch_report.json")
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        print(f"[批量报告] {report_path}")
    except OSError as exc:
        print(f"[警告] 批量报告写入失败: {exc}", file=sys.stderr)
    if failures:
        print(f"[批量失败] {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
