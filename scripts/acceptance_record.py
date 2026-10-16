"""自动生成验收记录
它采集什么:
    环境（硬件/系统/Python/PyTorch/ComfyUI 提交/包数）
    输入照片：路径、尺寸、格式、字节数、SHA256
    输出照片：路径、尺寸、格式、字节数、SHA256、alpha 分布
    耗时（由调用方提供，来自生成时的客户端计时）
    内存：当前与峰值（`vmmap -summary`，因为 `ps` 在本环境被沙箱禁止）
    配色：发色 / 服装主色的客观相似度（`chibitoy.analysis`）

用法:

    .venv/bin/python scripts/acceptance_record.py \\
        --input inputs/img/demo.jpg --output results/ar30_00001.png \\
        --command 'python -m chibitoy.generate ...' --elapsed 393.65 \\
        --out docs/acceptance_record_auto.md
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from chibitoy import analysis, config  # noqa: E402


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_image(path: Path) -> dict:
    with Image.open(path) as image:
        return {
            "path": str(path),
            "width": image.width,
            "height": image.height,
            "format": image.format,
            "mode": image.mode,
            "bytes": path.stat().st_size,
            "sha256": sha256_of(path),
        }


def alpha_distribution(path: Path) -> dict | None:
    with Image.open(path) as image:
        if image.mode not in ("RGBA", "LA"):
            return None
        alpha = np.asarray(image.convert("RGBA").getchannel("A"))
    total = alpha.size
    return {
        "transparent": float((alpha < 10).sum() / total),
        "semi": float(((alpha >= 10) & (alpha <= 245)).sum() / total),
        "opaque": float((alpha > 245).sum() / total),
        "subject": float((alpha > 128).sum() / total),
    }


def environment() -> dict:
    import torch

    def sysctl(key: str) -> str:
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True,
                                  text=True, timeout=10, check=False).stdout.strip()
        except Exception:  # noqa: BLE001
            return "未知"

    commit = "未知"
    comfy = config.PROJECT_ROOT / "ComfyUI"
    if (comfy / ".git").is_dir():
        commit = subprocess.run(["git", "-C", str(comfy), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=15,
                                check=False).stdout.strip() or "未知"
    return {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cpu": sysctl("machdep.cpu.brand_string") or "未知",
        "model": sysctl("hw.model") or "未知",
        "memory_gb": round(int(sysctl("hw.memsize") or 0) / 1024 ** 3, 1) if sysctl("hw.memsize").isdigit() else "未知",
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "mps": bool(torch.backends.mps.is_available()),
        "packages": len(list(metadata.distributions())),
        "comfyui_commit": commit,
    }


def memory_footprint() -> dict:
    """用 `vmmap -summary` 取当前与峰值内存。

    `ps -o rss=` 在本环境被沙箱禁止（Operation not permitted），vmmap 可用且直接给峰值。
    """
    try:
        pid = subprocess.run(["pgrep", "-f", "python.*main.py"], capture_output=True,
                             text=True, timeout=10, check=False).stdout.split()
    except Exception:  # noqa: BLE001
        pid = []
    if not pid:
        return {"available": False, "reason": "未找到 ComfyUI 进程（pgrep 无结果）"}
    try:
        out = subprocess.run(["vmmap", "-summary", pid[0]], capture_output=True,
                             text=True, timeout=60, check=False).stdout
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"vmmap 执行失败: {exc}"}
    current = peak = None
    for line in out.splitlines():
        if "Physical footprint (peak)" in line:
            peak = line.split(":")[-1].strip()
        elif "Physical footprint:" in line and peak is None:
            current = line.split(":")[-1].strip()
    if peak is None and current is None:
        return {"available": False, "reason": "vmmap 输出中未找到 footprint 行"}
    return {"available": True, "current": current, "peak": peak, "pid": pid[0]}


def color_metrics(source: Path, generated: Path, matte: bool, base_url: str) -> dict:
    """配色相似度。matte=True 时先把源照片真实抠图。

    **必须复用 `color_check._matte_source`，不能用 `generate._matte_reference_to_white`**：
    后者会把抠图结果**合成到白底并存为 RGB**（那是 IP-Adapter 需要的不透明图），
    **alpha 掩码丢失**，导致 `analysis` 退化为"边框主色即背景"估计。
    实测该错误会让源照片发色被识别为 `(254,254,254)` 白色、发色相似度变成 0.014（正确值约 0.998）。
    """
    from chibitoy.client import ComfyUIClient

    src_img = None
    note = "源照片前景用「边框主色即背景」估计"
    if matte:
        client = ComfyUIClient(base_url)
        if client.is_available():
            src_img = _matte_source_keeping_alpha(source, base_url)
            note = "源照片已用 BiRefNet 真实分割并保留 alpha 掩码（--matte-source）"
        else:
            note = "ComfyUI 不可用，退化为边框主色估计"
    if src_img is None:
        src_img = Image.open(source).convert("RGB")
    with Image.open(generated) as gen_img:
        result = analysis.analyze_pair(src_img, gen_img, str(source), str(generated))
    return {
        "hair": result.get("hair_similarity"),
        "clothing": result.get("clothing_similarity"),
        "source_hair": result["source"].hair_dominant,
        "source_clothing": result["source"].clothing_dominant,
        "generated_hair": result["generated"].hair_dominant,
        "generated_clothing": result["generated"].clothing_dominant,
        "generated_coverage": result["generated"].foreground_ratio,
        "note": note,
    }


def _matte_source_keeping_alpha(source: Path, base_url: str):
    """复用 `scripts/color_check.py` 的抠图函数 —— 它保留 alpha，正是分析所需。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "color_check_helper", Path(__file__).with_name("color_check.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._matte_source(str(source), base_url)


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _stability_section(payload: dict | None) -> list[str]:
    """把 `stability_check.py` 的 JSON 结果渲染成记录里的一节
    这样记录里的稳定性数据与规范报告同源，不会出现"记录用旧样本、报告用新样本"的错配
    """
    if not payload:
        return []
    lines = ["## 7. 三维度重复稳定性（同源：`docs/stability_report.json`）", "",
             f"- 源照片：`{payload.get('source', 'n/a')}`",
             f"- 阈值：任一维度偏差 >{payload['verdict']['threshold_percent']:.0f}% 即不达标",
             f"- 总判定：**{'✅ 通过' if payload['verdict']['passed'] else '❌ 不通过'}**", "",
             "| 维度 | 指标 | 偏差 | 判定 |", "| --- | --- | --- | --- |"]
    for dim in payload["verdict"]["dimensions"]:
        dev = "n/a" if dim["deviation"] is None else f"**{dim['deviation']:.2f}%**"
        lines.append(f"| {dim['dimension']} | {dim['name']} | {dev} | "
                     f"{'✅' if dim['passed'] else '❌ 不达标'} |")
    lines += ["", "复现：`scripts/stability_check.py --generate 5 --warmup --identity …`；",
              "身份维度用的是**通用视觉相似度、不是人脸识别**，不得用于「面部特征 ≥80%」判定。", ""]
    return lines


def render(env, src, out, alpha, mem, color, command, elapsed, verdict_extra: str,
           stability: dict | None = None) -> str:
    lines: list[str] = []
    add = lines.append
    add("# 验收记录（自动采集）")
    add("")
    add("> 本文件的**客观部分由 `scripts/acceptance_record.py` 自动采集**，每个数字都有对应命令。")
    add("> **主观判定留白为 `待人工填写`** —— 脚本不代填主观结论。")
    add("")
    add("## 1. 测试环境")
    add("")
    add("| 项 | 值 |")
    add("| --- | --- |")
    add(f"| 采集时间 | {env['time']} |")
    add(f"| 硬件 | {env['cpu']}（{env['model']}），内存 {env['memory_gb']} GB |")
    add(f"| 系统 | {env['os']} |")
    add(f"| Python | {env['python']} |")
    add(f"| PyTorch | {env['torch']}，MPS 可用 = {env['mps']} |")
    add(f"| 已安装包 | {env['packages']} 个 |")
    add(f"| ComfyUI | `{env['comfyui_commit']}` |")
    add("")
    add("## 2. 输入照片")
    add("")
    add("| 项 | 值 |")
    add("| --- | --- |")
    add(f"| 路径 | `{src['path']}` |")
    add(f"| 尺寸 | {src['width']} × {src['height']}"
        f"{'  ⚠️ **低于需求单要求的 ≥500×500**' if min(src['width'], src['height']) < 500 else ''} |")
    add(f"| 格式 / 模式 | {src['format']} / {src['mode']} |")
    add(f"| 字节数 | {src['bytes']:,} |")
    add(f"| SHA256 | `{src['sha256']}` |")
    add("")
    add("## 3. 生成命令")
    add("")
    add("```bash")
    add(command or "（未提供，请手工补全）")
    add("```")
    add("")
    add("## 4. 输出照片")
    add("")
    add("| 项 | 值 |")
    add("| --- | --- |")
    add(f"| 路径 | `{out['path']}` |")
    add(f"| 尺寸 | {out['width']} × {out['height']}"
        f"{'  ✅ ≥1080' if min(out['width'], out['height']) >= 1080 else '  ❌ 低于 1080'} |")
    add(f"| 格式 / 模式 | {out['format']} / {out['mode']} |")
    add(f"| 字节数 | {out['bytes']:,} |")
    add(f"| SHA256 | `{out['sha256']}` |")
    if alpha:
        add(f"| alpha 分布 | 完全透明 {alpha['transparent'] * 100:.1f}% / 半透明 "
            f"{alpha['semi'] * 100:.1f}% / 完全实心 {alpha['opaque'] * 100:.1f}%"
            f"（alpha>128 占 {alpha['subject'] * 100:.1f}%） |")
    add("")
    add("## 5. 耗时与内存")
    add("")
    add("| 项 | 值 | 采集方式 |")
    add("| --- | --- | --- |")
    add(f"| 端到端耗时 | {_fmt(elapsed, 2)} s | 生成时的客户端计时 |")
    if mem.get("available"):
        add(f"| 服务进程内存（当前） | {mem['current']} | `vmmap -summary {mem['pid']}` |")
        add(f"| **服务进程内存（峰值）** | **{mem['peak']}** | 同上，`Physical footprint (peak)` |")
    else:
        add(f"| 内存 | 采集失败：{mem.get('reason')} | `vmmap -summary <pid>` |")
    add("")
    add("## 6. 客观配色指标")
    add("")
    add(f"采集方式：`chibitoy.analysis`（{color['note']}）")
    add("")
    add("| 指标 | 源主色 | 产物主色 | 相似度 | 判定 |")
    add("| --- | --- | --- | --- | --- |")
    hair_ok = (color["hair"] or 0) >= 0.90
    clo_ok = (color["clothing"] or 0) >= 0.90
    add(f"| 发色 | {color['source_hair']} | {color['generated_hair']} | "
        f"**{_fmt(color['hair'])}** | {'✅ 达标' if hair_ok else '❌ 不达标'} |")
    add(f"| 服装主色 | {color['source_clothing']} | {color['generated_clothing']} | "
        f"**{_fmt(color['clothing'])}** | {'✅ 达标' if clo_ok else '❌ 不达标'} |")
    add("")
    add(f"产物前景覆盖率：{color['generated_coverage'] * 100:.1f}%")
    add("")
    section = _stability_section(stability)
    if section:
        lines.extend(section)
    add("## 8. 逐项判定")
    add("")
    add("| # | 验收项 | 判定 | 依据 |")
    add("| --- | --- | --- | --- |")
    add(f"| 4 | 输出 PNG ≥1080×1080 | {'✅ 通过' if min(out['width'], out['height']) >= 1080 else '❌ 不通过'} "
        f"| 实测 {out['width']}×{out['height']} |")
    add(f"| 10 | 发色还原 | {'✅ 通过' if hair_ok else '❌ 不通过'} | 客观相似度 {_fmt(color['hair'])} |")
    add(f"| 11 | 服装主色 ≥90% | {'✅ 通过' if clo_ok else '❌ 不通过'} | 客观相似度 {_fmt(color['clothing'])} |")
    add(f"| 14 | 单张 ≤30 秒 | {'✅ 通过' if (elapsed or 1e9) <= 30 else '❌ 不通过'} | 实测 {_fmt(elapsed, 2)} s |")
    add("| 1 | 完整公仔（头/躯干/四肢/脚） | 待人工填写 | 目视 |")
    add("| 2 | 无畸形 / 无多余肢体 | 待人工填写 | 目视 |")
    add("| 3 | 纯白 / 透明背景、无蓝底残留 | 待人工填写 | 目视 + alpha 分布 |")
    add("| 5 | 原生分辨率（非放大） | 待人工填写 | 检查工作流无 upscale 节点 |")
    add("| 6 | 头身比 1.5–2 | 待人工填写 | 目视 |")
    add("| 7 | 正面 / 近正面、主体居中 | 待人工填写 | 目视 |")
    add("| 8 | 材质哑光 PVC / 软陶 / 搪胶 | 待人工填写 | 目视（主观项） |")
    add("| 9 | 面部特征 ≥80% | ⛔ **未测试** | 需人脸识别模型；本工具是颜色/视觉相似度，**不可替代** |")
    add("| 12 | 表情不得无故改变 | 待人工填写 | 目视 |")
    add("| 13 | 无文字 / 水印 / logo | 待人工填写 | 目视 |")
    add("| 15 | 5 次重复偏差 ≤10% | 待人工填写 | 需另行跑 §5 的三维度稳定性 |")
    add("| 16 | ControlNet 姿态 / 结构控制 | 待人工填写 | 检查工作流含 ControlNetApplyAdvanced |")
    add("| 17 | 头发 / 服饰分割 | ❌ 未实现 | 语义分割权重未安装 |")
    add("| 18 | 本项目专属 LoRA | ❌ 未训练 | 当前为第三方 LoRA |")
    add("| 19 | 批量处理 | 待人工填写 | 见 `scripts/batch_generate.py` |")
    add("")
    if verdict_extra:
        add(verdict_extra.rstrip())
        add("")
    add("## 9. 未测试项与边界（必读）")
    add("")
    add("- **面部特征 ≥80%**：本工具**不是人脸识别**，无法回答该问题；该项维持「未测试」。")
    add("- **配色指标**：位置分带 + 颜色聚类的近似度量，**不是语义分割**，"
        "且**会被退化图像欺骗**（第 10 轮那张完全崩坏的图曾得 0.995），必须与结构类一票否决项配合判读。")
    add("- 主观项（材质、头身比、表情、发型形状）必须人工填写，脚本不代填。")
    add("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--command", default="", help="本次使用的生成命令（原样记录）")
    parser.add_argument("--elapsed", type=float, default=None, help="端到端耗时（秒）")
    parser.add_argument("--out", required=True, help="Markdown 输出路径")
    parser.add_argument("--matte-source", action="store_true",
                        help="源照片先用 BiRefNet 真实分割（推荐）")
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("--extra", default="", help="追加到判定表后的一段自定文字")
    parser.add_argument("--stability-json", default=None,
                        help="`stability_check.py` 产出的 JSON；填入后记录会包含三维度稳定性一节，"
                             "使记录的稳定性数据与规范报告**同源**")
    parser.add_argument("--json", default=None,
                        help="同时把**原始采集数据**写成 JSON，供 verify_delivery.py 回读核验。"
                             "解析 Markdown 易碎，机器可读的 sidecar 才是可靠的事实来源")
    args = parser.parse_args()

    src_path, out_path = Path(args.input), Path(args.output)
    for path in (src_path, out_path):
        if not path.is_file():
            print(f"[错误] 文件不存在: {path}", file=sys.stderr)
            return 2

    src_info = describe_image(src_path)
    out_info = describe_image(out_path)
    alpha = alpha_distribution(out_path)
    memory = memory_footprint()
    color = color_metrics(src_path, out_path, args.matte_source, args.comfy_url)

    stability = None
    if args.stability_json:
        stability_path = Path(args.stability_json)
        if not stability_path.is_file():
            print(f"[错误] 稳定性 JSON 不存在: {stability_path}", file=sys.stderr)
            return 2
        stability = json.loads(stability_path.read_text(encoding="utf-8"))

    record = render(environment(), src_info, out_info, alpha, memory, color,
                    args.command, args.elapsed, args.extra, stability)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(record, encoding="utf-8")
    print(f"[已生成] {destination}  {len(record)} 字符")

    if args.json:
        payload = {
            "environment": environment(),
            "input": src_info,
            "output": out_info,
            "alpha": alpha,
            "memory": memory,
            "color": {k: (list(v) if isinstance(v, tuple) else v)
                      for k, v in color.items()},
            "elapsed_seconds": args.elapsed,
            "command": args.command,
            "stability_source": args.stability_json,
        }
        sidecar = Path(args.json)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
        print(f"[已生成] {sidecar}（机器可读 sidecar，供核验回读）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
