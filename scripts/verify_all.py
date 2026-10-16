#!/usr/bin/env python
"""
把四条各自独立的校验链路串起来，并明确列出唯一未验证的环节

链路:
    1. `verify_delivery.py`  交付文件齐全 / 模型 SHA256 / 环境事实 / ComfyUI 节点
    2. `audit_docs.py`       交付文档结构（围栏、表格、章节、链接、文件引用）
    3. `verify_delivery.py --check-record`  验收记录与磁盘现实是否一致
    4. `pytest`              自动化测试套件
    5. 规范稳定性报告是否存在且自洽（`stability_report.json` ↔ `.md`）

用法::

    .venv/bin/python scripts/verify_all.py            # 全量（约 1~2 分钟）
    .venv/bin/python scripts/verify_all.py --skip-hash --skip-tests   # 快速
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def run(title: str, argv: list[str]) -> tuple[bool, str]:
    print(f"\n{'=' * 72}\n▶ {title}\n{'=' * 72}")
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    output = (result.stdout or "") + (result.stderr or "")
    print(output.rstrip())
    return result.returncode == 0, output


def check_stability_report() -> bool:
    json_path = ROOT / "docs" / "stability_report.json"
    md_path = ROOT / "docs" / "stability_report.md"
    print(f"\n{'=' * 72}\n▶ 规范稳定性报告自洽性\n{'=' * 72}")
    if not (json_path.is_file() and md_path.is_file()):
        print("  ❌ 缺少 docs/stability_report.{json,md}")
        return False
    data = json.loads(json_path.read_text(encoding="utf-8"))
    text = md_path.read_text(encoding="utf-8")
    ok = True
    for dim in data["verdict"]["dimensions"]:
        if dim["deviation"] is None:
            continue
        wanted = f"{dim['deviation']:.2f}%"
        hit = wanted in text
        ok &= hit
        print(f"  {'✅' if hit else '❌'} {dim['name']}: {wanted}")
    print(f"  → 规范报告自洽: {'✅' if ok else '❌'}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-hash", action="store_true", help="跳过模型 SHA256（更快）")
    parser.add_argument("--skip-tests", action="store_true", help="跳过测试套件")
    args = parser.parse_args()

    checks: list[tuple[str, bool]] = []

    delivery_argv = [PYTHON, "scripts/verify_delivery.py"]
    if args.skip_hash:
        delivery_argv.append("--skip-hash")
    ok, _ = run("① 交付核验（文件 / 模型 / 环境 / ComfyUI 节点）", delivery_argv)
    checks.append(("交付核验", ok))

    ok, _ = run("② 文档结构体检", [PYTHON, "scripts/audit_docs.py"])
    checks.append(("文档结构", ok))

    record = ROOT / "docs" / "acceptance_record_auto.json"
    if record.is_file():
        ok, _ = run("③ 验收记录与磁盘现实一致性",
                    [PYTHON, "scripts/verify_delivery.py", "--skip-hash",
                     "--check-record", str(record.relative_to(ROOT))])
        checks.append(("验收记录", ok))
    else:
        print("\n▶ ③ 验收记录：⚠️ 不存在 docs/acceptance_record_auto.json，跳过")
        checks.append(("验收记录", False))

    checks.append(("规范稳定性报告", check_stability_report()))

    if args.skip_tests:
        print("\n▶ ④ 测试套件：已按 --skip-tests 跳过")
    else:
        ok, output = run("④ 自动化测试套件",
                         [PYTHON, "-m", "pytest", "tests/", "-q",
                          "-p", "no:cacheprovider", "-m", "not live and not slow"])
        checks.append(("测试套件", ok))

    print(f"\n{'=' * 72}\n▶ 汇总\n{'=' * 72}")
    for name, passed in checks:
        print(f"  {'✅ 通过' if passed else '❌ 未通过'}  {name}")

    print("\n  ⚠️ **唯一未验证的环节：从未在干净环境里按 README 从零安装过。**")
    print("     本机环境是长期迭代出来的（88 个包、模型已就位），")
    print("     README §3 的安装步骤只做过**逐条命令验证**，未做过**整机重装演练**。")
    print("     第三方若在干净环境复现，请以 README §3 为准，并把结果反馈回来。")

    all_ok = all(passed for _, passed in checks)
    print(f"\n=== 总判定：{'✅ 全部通过' if all_ok else '❌ 存在未通过项'} ===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
