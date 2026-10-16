#!/usr/bin/env python
"""交付文档的结构体检
检查项:
    1. 代码围栏是否配平
    2. Markdown 表格每块的列数是否一致
    3. 章节编号是否连续
    4. Markdown 内部链接指向的本地文件是否存在
    5. 反引号里形如 `路径/文件名.ext` 的本地文件引用是否存在
       跳过 URL、通配符、明显是示例的路径

用法::
    .venv/bin/python scripts/audit_docs.py            # 体检全部交付文档
    .venv/bin/python scripts/audit_docs.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 体检范围：交付文档（不含历史阶段的调研笔记）
DOCS = [
    "README.md",
    "docs/delivery.md",
    "docs/acceptance_criteria.md",
    "docs/issues.md",
    "docs/HANDOVER.md",
    "docs/phase1_baseline.md",
    "docs/phase2_experiments.md",
    "docs/api_provider_comparison.md",
    "docs/api_backend.md",
    "docs/face_model_license_survey.md",
    "docs/artifact_routes.md",
    "docs/report_1b3b2.md",
    "docs/environment.md",
    "docs/phase3_api_first.md",
    "docs/env_snapshot_2026-10-16.txt",
    "docs/acceptance_record_2026-10-15.md",
    "docs/acceptance_record_auto.md",
    "docs/stability_report.md",
    "docs/benchmark_manifest.json",
    "docs/benchmark_sheet.png",
    "docs/PROJECT_EXECUTION_STATUS.md",
    "docs/photo_intake_workflow.md",
]

# 必须以字母或数字开头，否则会把 `workflows/x.api.json` 里的 `.api.json` 也当成一个路径
FILE_REF = re.compile(r"`([A-Za-z0-9][A-Za-z0-9_./\-]*\.(?:py|md|json|png|txt|yaml|sh|safetensors))`")
SKIP_REF_MARKERS = ("*", "<", ">", "...", "…", "$", "http")


def check_fences(text: str) -> list[str]:
    count = text.count("\n```") + (1 if text.startswith("```") else 0)
    return [] if count % 2 == 0 else [f"代码围栏数为奇数（{count}），存在未闭合的代码块"]


def check_tables(text: str) -> list[str]:
    """检查每个表格块内部的列数是否一致。"""
    problems: list[str] = []
    block: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("|") and line.rstrip().endswith("|"):
            block.append((lineno, line))
            continue
        problems += _check_block(block)
        block = []
    problems += _check_block(block)
    return problems


def _check_block(block: list[tuple[int, str]]) -> list[str]:
    if len(block) < 2:
        return []
    widths = {lineno: len(line.split("|")) - 2 for lineno, line in block}
    expected = widths[block[0][0]]
    return [
        f"第 {lineno} 行所属表格列数为 {width}，与表头的 {expected} 不一致"
        for lineno, width in widths.items() if width != expected
    ]


def check_sections(text: str) -> list[str]:
    """检查 `## N.` 形式的章节编号是否从 1 开始连续递增。"""
    numbers = [int(m.group(1)) for m in re.finditer(r"^## (\d+)\.", text, re.M)]
    if not numbers:
        return []
    # 允许一个 0 号「前言/关系说明」章节
    if numbers and numbers[0] == 0:
        numbers = numbers[1:]
    expected = list(range(1, len(numbers) + 1))
    if numbers != expected:
        return [f"章节编号不连续：实际 {numbers}，期望 {expected}"]
    return []


def check_links(path: Path, text: str) -> list[str]:
    problems: list[str] = []
    # 先剔除行内代码（反引号）内容：文档里描述链接语法时会出现 `](path)` 这样的片段
    stripped = re.sub(r"`[^`]*`", "", text)
    for target in re.findall(r"\]\(([^)#\s]+)\)", stripped):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        resolved = (path.parent / target).resolve()
        if not resolved.exists():
            problems.append(f"Markdown 链接指向不存在的文件: {target}")
    return problems


def _known_suffixes() -> set[str]:
    """仓库内所有文件的路径后缀集合，用于后缀匹配

    文档里常写裸文件名（如 `generate.py`），其真实位置是 `src/chibitoy/generate.py`
    也会写相对 `models/` 的路径（如 `loras/xxx.safetensors`）
    故只要仓库内存在以该字符串结尾的文件，就认为引用有效
    """
    suffixes: set[str] = set()
    for path in ROOT.rglob("*"):
        # 只排除虚拟环境（内容庞杂且不是交付物）；ComfyUI 内的文件（如 requirements.txt）
        if not path.is_file() or ".venv" in path.parts:
            continue
        rel = path.relative_to(ROOT).as_posix()
        parts = rel.split("/")
        for i in range(len(parts)):
            suffixes.add("/".join(parts[i:]))
    return suffixes

DELETED_MARKERS = ("已删除", "已清理", "已被清理", "不再存在")

def check_file_refs(text: str, known: set[str]) -> list[str]:
    problems: list[str] = []
    for line in text.splitlines():
        if any(marker in line for marker in DELETED_MARKERS):
            continue
        for name in FILE_REF.findall(line):
            if any(marker in name for marker in SKIP_REF_MARKERS):
                continue
            if name.startswith("/"):
                continue
            if name not in known:
                problems.append(f"引用了不存在的本地文件: {name}")
    return problems


def audit(path: Path, known: set[str] | None = None) -> list[str]:
    if path.suffix != ".md":
        return [] if path.exists() else [f"文件不存在: {path}"]
    if not path.is_file():
        return [f"文件不存在: {path}"]
    text = path.read_text(encoding="utf-8")
    return (check_fences(text) + check_tables(text) + check_sections(text)
            + check_links(path, text) + check_file_refs(text, known or set()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=None)
    parser.add_argument("--docs", nargs="*", default=DOCS)
    args = parser.parse_args()

    known = _known_suffixes()
    report: dict[str, list[str]] = {}
    for rel in args.docs:
        problems = audit(ROOT / rel, known)
        report[rel] = problems
        status = "✅" if not problems else f"❌ {len(problems)} 项"
        print(f"  {status:8s} {rel}")
        for item in problems:
            print(f"           - {item}")

    total = sum(len(v) for v in report.values())
    print()
    print(f"=== 体检完成：{len(args.docs)} 份文档，共 {total} 项问题 ===")
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
        print(f"  报告已写入 {args.json}")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
