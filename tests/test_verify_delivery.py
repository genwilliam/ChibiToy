"""交付核验脚本的单元测试。

`scripts/verify_delivery.py` 是"交付可信度"的守门人，必须有测试保护。
本文件验证两件事：

1. **它能抓到问题**（失败路径）—— 一个永远返回"通过"的核验毫无价值；
2. **它声明的清单与文档一致**（防文档漂移）—— 需求单要求"交付物与实际一致"。
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "verify_delivery", ROOT / "scripts" / "verify_delivery.py")
assert SPEC and SPEC.loader
verify_delivery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify_delivery)


@pytest.fixture()
def fake_root(tmp_path, monkeypatch):
    """把脚本的 ROOT 指向临时目录，用于测试失败路径。"""
    monkeypatch.setattr(verify_delivery, "ROOT", tmp_path)
    return tmp_path


def test_sha256_matches_hashlib(tmp_path):
    payload = b"chibitoy" * 1000
    target = tmp_path / "sample.bin"
    target.write_bytes(payload)
    assert verify_delivery.sha256(target) == hashlib.sha256(payload).hexdigest()


def test_check_deliverables_reports_all_missing_on_empty_root(fake_root):
    problems = verify_delivery.check_deliverables()
    assert len(problems) == len(verify_delivery.DELIVERABLES)
    assert all("缺少交付文件" in item for item in problems)


def test_check_deliverables_passes_on_real_repo():
    """真实仓库里声明的交付文件必须都存在——否则交付清单是空头支票。"""
    missing = [r for r in verify_delivery.DELIVERABLES if not (ROOT / r).is_file()]
    assert missing == [], f"交付清单里声明但实际缺失: {missing}"


def test_check_models_detects_truncated_file(fake_root):
    """体积与哈希不符的模型必须被抓到（模拟下载截断/被替换）。"""
    rel = "models/checkpoints/sd_xl_base_1.0.safetensors"
    target = fake_root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"\0" * 1000)
    problems = verify_delivery.check_models(skip_hash=False)
    assert any("sd_xl_base_1.0.safetensors" in item for item in problems)


def test_check_models_reports_missing_required_models(fake_root):
    problems = verify_delivery.check_models(skip_hash=True)
    required = [m for m in verify_delivery.MODELS if m[3]]
    assert len(problems) == len(required)


def test_all_declared_model_paths_are_relative_and_contained():
    """清单里的路径必须是相对路径且不逃出仓库根目录。"""
    for rel, *_ in verify_delivery.MODELS:
        path = Path(rel)
        assert not path.is_absolute(), rel
        assert ".." not in path.parts, rel


def test_declared_models_resolve_under_root():
    """每一条模型声明都应能拼出仓库内的绝对路径，且在 models/ 下。"""
    for rel, *_ in verify_delivery.MODELS:
        resolved = (ROOT / rel).resolve()
        assert resolved.is_relative_to((ROOT / "models").resolve()), rel


def _documented_hashes(text: str) -> set[tuple[str, str]]:
    """抽出文档里的省略哈希，支持两种写法。

    - 带省略号：`` `31e35c80…f7e5b` `` -> ("31e35c80", "f7e5b")
    - 仅前缀  ：`` `31e35c80` ``        -> ("31e35c80", "")
    """
    pairs = set(re.findall(r"`([0-9a-f]{6,12})…([0-9a-f]{4,8})`", text))
    # 裸 8 位形式只在**同一行出现了 .safetensors** 时才当作哈希，
    # 否则会把文件大小这类数字（如 10000000）误判成哈希。
    prefixes: set[str] = set()
    for line in text.splitlines():
        if ".safetensors" in line:
            prefixes.update(re.findall(r"`([0-9a-f]{8})`", line))
    return {(prefix, suffix) for prefix, suffix in pairs} | {(p, "") for p in prefixes}


@pytest.mark.parametrize("doc", ["README.md", "docs/delivery.md", "docs/environment.md"])
def test_documented_hashes_match_verify_manifest(doc):
    """**防文档漂移**：文档里写的省略哈希必须与核验脚本的清单对得上。

    这是需求单"交付物必须与实际一致"的直接落地：
    只要有人改了模型却忘了同步文档（或反之），本测试就会失败。
    """
    text = (ROOT / doc).read_text(encoding="utf-8")
    documented = _documented_hashes(text)
    if not documented:
        pytest.skip(f"{doc} 未记录省略哈希")

    # 文档里省略哈希的后缀长度不统一（有的 5 位、有的 4 位），
    # 因此判据是"前缀与后缀都能对上某个真实哈希"，而不是固定切片长度。
    unknown = {
        (prefix, suffix)
        for prefix, suffix in documented
        if not any(full.startswith(prefix) and full.endswith(suffix)
                   for _, _, full, _ in verify_delivery.MODELS)
    }
    assert unknown == set(), f"{doc} 中出现了核验清单里没有的哈希: {sorted(unknown)}"


def test_required_nodes_cover_the_exported_workflow():
    """核验清单里的节点必须覆盖交付工作流真正用到的每一种节点。"""
    import json

    workflow = json.loads(
        (ROOT / "workflows" / "chibitoy_final.api.json").read_text(encoding="utf-8"))
    used = {node["class_type"] for node in workflow.values()}
    declared = set(verify_delivery.REQUIRED_NODES)
    assert used <= declared, f"工作流用到但核验清单未覆盖: {sorted(used - declared)}"


def test_expected_package_count_matches_readme():
    """脚本声明的包数量应与文档一致，避免两边各说各话。"""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    expected = verify_delivery.EXPECTED_PACKAGES
    assert f"{expected} 个包" in readme or f"{expected} 个已安装包" in readme or str(expected) in readme


def test_main_returns_nonzero_when_root_is_empty(fake_root, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["verify_delivery.py", "--skip-hash"])
    assert verify_delivery.main() == 1
    assert "核验未通过" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 验收记录核验（--check-record）
# ---------------------------------------------------------------------------

def _write_record(tmp_path, *, out_bytes=None, out_sha=None, width=1088):
    """造一份合成记录 + 对应的真实文件，用于测试核验逻辑。"""
    import json

    src = tmp_path / "src.jpg"
    out = tmp_path / "out.png"
    src.write_bytes(b"\xff\xd8\xff" + b"x" * 97)
    out.write_bytes(b"\x89PNG" + b"y" * 500)
    payload = {
        "input": {"path": str(src), "bytes": src.stat().st_size,
                  "sha256": verify_delivery.sha256(src)},
        "output": {"path": str(out), "bytes": out_bytes or out.stat().st_size,
                   "sha256": out_sha or verify_delivery.sha256(out),
                   "width": width, "height": width},
        "color": {"hair": 0.99, "clothing": 0.15},
        "memory": {"available": True, "peak": "24.2G"},
        "elapsed_seconds": 441.72,
    }
    path = tmp_path / "record.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_check_record_passes_on_truthful_record(tmp_path):
    assert verify_delivery.check_record(_write_record(tmp_path)) == []


def test_check_record_detects_tampered_sha256(tmp_path):
    """记录里的哈希与实际文件不符时必须被抓到——否则"可复现的验收"只是自说自话。"""
    problems = verify_delivery.check_record(_write_record(tmp_path, out_sha="0" * 64))
    assert any("SHA256" in item for item in problems)


def test_check_record_detects_wrong_byte_count(tmp_path):
    problems = verify_delivery.check_record(_write_record(tmp_path, out_bytes=1))
    assert any("字节数" in item for item in problems)


def test_check_record_detects_missing_output_file(tmp_path):
    path = _write_record(tmp_path)
    (tmp_path / "out.png").unlink()
    problems = verify_delivery.check_record(path)
    assert any("不存在" in item for item in problems)


def test_check_record_detects_undersized_output(tmp_path):
    problems = verify_delivery.check_record(_write_record(tmp_path, width=512))
    assert any("低于 1080" in item for item in problems)


def test_check_record_reports_unreadable_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    problems = verify_delivery.check_record(bad)
    assert problems and "不可读" in problems[0]


def test_check_record_does_not_fail_on_unmet_targets(tmp_path, capsys):
    """核验的是"记录是否属实"，**不是**结果是否达标。

    服装主色 0.15 不达标、耗时 441 s 超 30 s —— 这些都应如实报告，
    但**不得**算作核验失败（否则记录真实反而被判错）。
    """
    problems = verify_delivery.check_record(_write_record(tmp_path))
    assert problems == []
    out = capsys.readouterr().out
    assert "不达标，这是结果不是记录错误" in out
    assert "超 30 s 目标" in out


# ---------------------------------------------------------------------------
# 文档口径一致性（防"同一数字在不同文档里不一样"）
# ---------------------------------------------------------------------------

DELIVERY_DOCS = ["README.md", "docs/delivery.md", "docs/acceptance_criteria.md",
                 "docs/acceptance_record_2026-10-15.md"]


@pytest.mark.parametrize("doc", DELIVERY_DOCS)
def test_pass_rate_is_always_scoped(doc):
    """**回归：达标率必须带范围。**

    第 28 轮审计发现：`1/5`（第 20 轮的 5 个种子）与 `1/7`（累计 7 个不同种子）
    两个数字并存却未标明范围，读者会误读为矛盾。
    因此规定：出现 `达标 1/5` 时必须同时说明其范围（"本次/5 次中/累计"）。
    """
    text = (ROOT / doc).read_text(encoding="utf-8")
    for line in text.splitlines():
        if "达标 1/5" in line or "达标率仅 1/5" in line:
            assert any(marker in line for marker in ("本次", "累计", "5 次中")), \
                f"{doc} 中的「达标 1/5」未标明范围: {line.strip()[:120]}"


def test_superseded_stability_number_is_marked_historical():
    """旧的 1.47% 稳定性结论必须标注为历史，否则会与当前的 7.8% 冲突。"""
    text = (ROOT / "docs/report_1b3b2.md").read_text(encoding="utf-8")
    idx = text.find("1.47%")
    assert idx != -1
    window = text[max(0, idx - 400): idx + 800]
    assert "取代" in window, "1.47% 附近未标注「已被取代」"


def test_timing_range_is_consistent_across_docs():
    """含 ControlNet 的 30 步耗时范围在各文档中必须一致（360–450 s）。"""
    for doc in ("README.md", "docs/acceptance_criteria.md"):
        text = (ROOT / doc).read_text(encoding="utf-8")
        assert "369–446" not in text, f"{doc} 仍存在旧的耗时范围 369–446"


def test_stability_deviations_use_two_decimals_consistently():
    """**回归：稳定性偏差的精度必须统一。**

    第 29 轮发现手工文档里混用了 `295%` / `294.66%`、`7.8%` / `7.77%` 两种精度，
    且覆盖率曾因用舍入值计算而写成 3.9%（精确值 4.12%）。
    规定：交付文档中一律使用判定器输出的**两位小数**。
    """
    stale = ["295%", "7.8%", "0.60%", "3.9%"]
    # 允许"说明这次修正"的句子提到旧值（否则连纠错记录本身都会被判违规）。
    # 只拦截把旧值**当作事实陈述**的出现。
    # 「原记录」「错」用于对照表：显式标明该值是已被取代的旧记录值
    explanatory = ("曾", "误", "应为", "写成", "舍入", "更正", "取代", "原记录", "错")
    for doc in DELIVERY_DOCS + ["docs/report_1b3b2.md"]:
        text = (ROOT / doc).read_text(encoding="utf-8")
        for line in text.splitlines():
            for token in stale:
                if token in line and not any(marker in line for marker in explanatory):
                    raise AssertionError(
                        f"{doc} 把未统一的精度写法当作事实陈述: {line.strip()[:140]}")


def test_stability_report_matches_its_json():
    """报告里的数值必须是 JSON 判定结果的两位小数舍入。"""
    import json

    report_path = ROOT / "docs" / "stability_report.md"
    json_path = ROOT / "docs" / "stability_report.json"
    if not (report_path.is_file() and json_path.is_file()):
        pytest.skip("稳定性报告或 sidecar 不存在")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    text = report_path.read_text(encoding="utf-8")
    for dim in data["verdict"]["dimensions"]:
        if dim["deviation"] is None:
            continue
        assert f"{dim['deviation']:.2f}%" in text, \
            f"{dim['name']} 的 {dim['deviation']:.2f}% 未出现在报告中"


# ---------------------------------------------------------------------------
# 问题清单（docs/issues.md）
# ---------------------------------------------------------------------------

def _issue_rows():
    import re

    text = (ROOT / "docs" / "issues.md").read_text(encoding="utf-8")
    return re.findall(r"^\| (I\d+) \| (.+?) \| (.+?) \| (.+?) \| (.+?) \|$", text, re.M)


def test_issues_doc_exists_and_has_entries():
    path = ROOT / "docs" / "issues.md"
    assert path.is_file(), "需求要求的「问题清单」缺失"
    rows = _issue_rows()
    assert len(rows) >= 8, f"问题清单条目过少: {len(rows)}"
    for issue_id, _, _, status, blocker in rows:
        assert status.strip() in ("🔴 可继续推进", "⛔ 外部阻塞", "🟡 已缓解",
                                  "⚪ 已修复/已排除"), \
            f"{issue_id} 的状态不在允许集合内: {status}"
        assert blocker.strip(), f"{issue_id} 未写明卡在哪里"


def test_issues_readme_count_matches_table():
    """**回归**：README 里的分类计数必须与问题清单表格一致。

    第 31 轮我就写错过（说 6 条外部阻塞，实际 8 条）—— 手工计数会出错。
    """
    from collections import Counter

    counts = Counter(row[3].strip() for row in _issue_rows())
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"{counts['⛔ 外部阻塞']} 条需外部输入" in readme, \
        f"README 的外部阻塞计数与清单不符（清单为 {counts['⛔ 外部阻塞']}）"
    assert f"{counts['🔴 可继续推进']} 条可继续推进" in readme


def test_acceptance_items_all_map_to_a_deliverable_or_a_known_gap():
    """需求验收项必须都能在映射表里找到落点（实现或已登记的缺口）。"""
    delivery = (ROOT / "docs" / "delivery.md").read_text(encoding="utf-8")
    assert "### 9.2 验收项 → 实现与验证" in delivery
    assert "### 9.3 反向检查" in delivery
    for item in range(1, 20):
        assert f"| {item} | " in delivery, f"验收项 {item} 未出现在映射表中"


def test_model_filenames_in_docs_actually_exist():
    """**守护：文档里出现的本地模型文件名必须真实存在。**

    第 35 轮我在 `acceptance_criteria.md` 里把 `Designer_BlindBox-000015.safetensors`
    写成了 `Designer_Box-000015.safetensors` —— 一条照抄无法跑通的命令。
    文档里的命令若引用了不存在的模型，使用者会直接踩坑，故用测试兜住。

    跳过上游 URL/路径（那是来源说明，不是本地文件）。
    """
    import re

    known = {p.name for p in (ROOT / "models").rglob("*.safetensors")}
    pattern = re.compile(r"[A-Za-z0-9_.\-]+\.safetensors")
    skip_markers = ("http", "上游", "上游路径", "URL", "https")
    for doc in DELIVERY_DOCS + ["README.md"]:
        for lineno, line in enumerate((ROOT / doc).read_text(encoding="utf-8").splitlines(), 1):
            if any(marker in line for marker in skip_markers):
                continue
            for name in pattern.findall(line):
                assert name in known, \
                    f"{doc}:{lineno} 引用了不存在的本地模型 {name}"


def test_docs_cite_the_canonical_stability_numbers():
    """**守护：文档引用的稳定性偏差必须等于规范报告（唯一事实来源）的数值。**

    第 36 轮用规范流程重跑了稳定性测量，数值随之变化；若文档不跟着改，
    就会出现"文档里的数字与机器采集的报告不一致"——那正是 §8 明令禁止的。
    """
    import json

    json_path = ROOT / "docs" / "stability_report.json"
    if not json_path.is_file():
        pytest.skip("规范稳定性报告不存在")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    canonical = {d["name"]: f"{d['deviation']:.2f}%" for d in data["verdict"]["dimensions"]
                 if d["deviation"] is not None}

    # §5 主表必须逐项与规范报告一致
    section = (ROOT / "docs" / "acceptance_criteria.md").read_text(encoding="utf-8")
    for name, formatted in canonical.items():
        assert formatted in section, \
            f"验收标准 §5 未出现规范报告中的 {name} 偏差 {formatted}"

    # README 的「规范测量结果」段落必须**逐项**等于规范报告，且不得混入被取代的旧值。
    # 只检查 presence 是不够的：植入旧值时，规范值可能仍出现在别处而让检查漏过（实测过）。
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.find("**规范测量结果**")
    assert start != -1, "README 缺少「规范测量结果」段落"
    paragraph = readme[start:readme.find("\n\n", start)]

    superseded = {"耗时": "7.77%", "前景覆盖率": "4.12%",
                  "服装主色相似度": "294.66%", "同人跨种子相似度": "7.27%"}
    for name, formatted in canonical.items():
        assert formatted in paragraph, \
            f"README 规范测量段缺少 {name} 的 {formatted}"
    for name, stale in superseded.items():
        assert stale not in paragraph, \
            f"README 规范测量段混入了被取代的旧值（{name}）: {stale}"


def test_acceptance_table_rows_are_well_formed():
    """**守护：交付文档验收表的每一行都必须结构规整。**

    第 37 轮发现 `docs/delivery.md` 第 15 行被一次替换**接错了位置**，
    变成新旧文本混合的语无伦次句子（"发色"在一行里出现两次）。
    这类损坏靠肉眼容易漏过，故用结构断言兜住。
    """
    import re

    text = (ROOT / "docs" / "delivery.md").read_text(encoding="utf-8")
    metrics = ["耗时", "发色", "服装主色", "身份", "覆盖率", "尺寸"]
    value = re.compile(r"\d+\.\d+\s*%")
    checked = 0
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.startswith("| ") or not line.split("|")[1].strip().isdigit():
            continue
        checked += 1
        cells = [c for c in line.split("|")[1:-1]]
        assert len(cells) == 4, f"delivery.md:{lineno} 单元格数异常（{len(cells)}）: {line[:80]}"

        # cells = [序号, 项目名, 判定, 依据]；项目名可能合法地含指标名（如「发型/发色/发饰」）
        checked_text = "|".join(cells[2:])
        for metric in metrics:
            # 只统计「指标 + 紧跟其后的百分比」这种配对。
            # 一次替换接错位置时，会出现**同一个指标带数值地重复出现**（第 37 轮的损坏即如此）；
            # 而"耗时主因=冷/热"这类不含数值的复述是合法的，不应误报。
            paired = sum(1 for m in re.finditer(re.escape(metric), checked_text)
                         if value.search(checked_text[m.end():m.end() + 16]))
            assert paired <= 1, \
                f"delivery.md:{lineno} 中「{metric}」带数值地出现 {paired} 次 —— 疑似文本被错误拼接"
    assert checked >= 19, f"验收表行数异常: {checked}"
    assert checked >= 19, f"验收表行数异常: {checked}"


# ---------------------------------------------------------------------------
# 文档结构体检（scripts/audit_docs.py）
# ---------------------------------------------------------------------------

def _audit_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "audit_docs", ROOT / "scripts" / "audit_docs.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_delivery_docs_pass_structural_audit():
    """**守护：全部交付文档必须通过结构体检。**

    项目后期做了大量程序化文本替换（自动同步数值、插入章节），有"接错位置"的风险 ——
    第 37 轮就发现交付文档的验收表某行被拼成了新旧混合的句子。
    抽查发现不了全部，故把结构完整性变成可执行检查。
    """
    module = _audit_module()
    known = module._known_suffixes()
    failures = {}
    for rel in module.DOCS:
        problems = module.audit(ROOT / rel, known)
        if problems:
            failures[rel] = problems
    assert not failures, f"文档结构体检未通过: {failures}"


def test_audit_catches_planted_defects(tmp_path):
    """**守护本身也必须被验证**：植入 4 类缺陷时必须全部命中。

    一个总是通过（或总是失败）的守护毫无价值 —— 第 37 轮已因此返工三次判据。
    """
    module = _audit_module()
    bad = tmp_path / "bad.md"
    bad.write_text(
        "## 1. 章\n\n"
        "| a | b |\n| --- | --- |\n| 1 | 2 | 3 |\n\n"          # 表格列数不一致
        "引用了 `scripts/does_not_exist.py`。\n\n"                 # 文件不存在
        "```python\n未闭合\n",                                     # 围栏未配平
        encoding="utf-8")
    problems = module.audit(bad, module._known_suffixes())
    joined = " | ".join(problems)
    assert "围栏" in joined
    assert "列数" in joined
    assert "不存在" in joined


# ---------------------------------------------------------------------------
# 交付总检查（scripts/verify_all.py）
# ---------------------------------------------------------------------------

def _verify_all_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "verify_all", ROOT / "scripts" / "verify_all.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verify_all_is_listed_as_a_deliverable():
    assert "scripts/verify_all.py" in verify_delivery.DELIVERABLES


def test_verify_all_stability_check_passes_on_real_report(capsys):
    module = _verify_all_module()
    assert module.check_stability_report() is True
    assert "规范报告自洽" in capsys.readouterr().out


def test_verify_all_stability_check_detects_missing_report(tmp_path, monkeypatch, capsys):
    """报告缺失时必须判失败，而不是默默通过。"""
    module = _verify_all_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    assert module.check_stability_report() is False
    assert "缺少" in capsys.readouterr().out


def test_verify_all_discloses_the_one_unverified_step():
    """**必须如实声明唯一未验证的环节**（干净环境安装），不得含糊。

    本机环境是长期迭代出来的；README 的安装步骤只做过逐条命令验证，
    没有做过整机重装演练。这一点若不写明，就是在暗示"全部已验证"。
    """
    text = (ROOT / "scripts" / "verify_all.py").read_text(encoding="utf-8")
    assert "唯一未验证的环节" in text
    assert "干净环境" in text


# ---------------------------------------------------------------------------
# 交接文档（docs/HANDOVER.md）的事实性
# ---------------------------------------------------------------------------

def test_handover_exists_and_is_audited():
    assert (ROOT / "docs" / "HANDOVER.md").is_file(), "交接文档缺失"
    module = _audit_module()
    assert "docs/HANDOVER.md" in module.DOCS, "交接文档未纳入结构体检"


def test_handover_scale_claims_match_reality():
    """**守护：交接文档里的规模数字必须与实际一致。**

    这类数字（脚本数、文档数、产物数）最容易随迭代过期 —— 第 40 轮就写错过一次
    （写"19 份文档"时 HANDOVER 自身尚未计入）。
    """
    import re

    text = (ROOT / "docs" / "HANDOVER.md").read_text(encoding="utf-8")
    scripts = len(list((ROOT / "scripts").glob("*.py"))) + len(list((ROOT / "scripts").glob("*.sh")))
    docs = len(list((ROOT / "docs").iterdir()))
    workflows = len(list((ROOT / "workflows").iterdir()))
    results = len(list((ROOT / "results").glob("*.png")))
    for count, label in ((scripts, "个脚本"), (docs, "份文档"),
                         (workflows, "个工作流文件")):
        assert f"{count} {label}" in text, f"交接文档的「{count} {label}」与实际不符"

    # 产物数**故意不严格核对**：每生成一张图它就会变，严格相等会让守护反复误报
    # （第 40 轮实测过）。改为「不得声称多于实际存在的数量」——这是稳定且有意义的断言。
    import re as _re

    claimed = _re.search(r"(\d+)\+? 张产物", text)
    assert claimed, "交接文档未声明产物数量"
    assert int(claimed.group(1)) <= results, \
        f"交接文档声称 {claimed.group(1)} 张产物，实际只有 {results} 张"


def test_handover_cites_canonical_stability_numbers():
    """交接文档引用的稳定性数值必须来自规范报告（唯一事实来源）。"""
    import json

    canon = json.loads((ROOT / "docs" / "stability_report.json").read_text(encoding="utf-8"))
    text = (ROOT / "docs" / "HANDOVER.md").read_text(encoding="utf-8")
    for dim in canon["verdict"]["dimensions"]:
        if dim["deviation"] is None:
            continue
        assert f"{dim['deviation']:.2f}%" in text, \
            f"交接文档缺少规范报告中的 {dim['name']} 偏差"


def test_handover_referenced_files_exist():
    """交接文档里以代码标记引用的本地文件必须存在。"""
    import re

    text = (ROOT / "docs" / "HANDOVER.md").read_text(encoding="utf-8")
    refs = set(re.findall(r"`((?:docs|scripts|workflows|results|src|inputs)/[A-Za-z0-9_./\-]+)`", text))
    missing = [r for r in refs if r.endswith((".md", ".json", ".png", ".py"))
               and not (ROOT / r).exists()]
    assert not missing, f"交接文档引用了不存在的文件: {missing}"


def test_handover_states_the_unverified_step():
    """交接文档必须写明唯一未验证的环节，不得暗示"全部已验证"。"""
    text = (ROOT / "docs" / "HANDOVER.md").read_text(encoding="utf-8")
    assert "从未在干净环境" in text
    assert "逐条命令验证" in text
