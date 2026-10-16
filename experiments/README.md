## 实验规格

| 文件                   | 内容                                                                   | 执行命令                                                     |
| ---------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------ |
| `crop.json`            | 对比三种取景方式：`top-square`、`head-square`、`none`                  | `scripts/exp_run.py --spec experiments/crop.json`            |
| `lora_strength.json`   | 对比 LoRA 强度：0.0、0.4、0.7、1.0                                     | `scripts/exp_run.py --spec experiments/lora_strength.json`   |
| `levers.json`          | 对比单变量调整，包括 IP-Adapter 权重和服装描述措辞                     | `scripts/exp_run.py --spec experiments/levers.json`          |
| `final_candidate.json` | 使用 5 个随机种子验证最终候选方案                                      | `scripts/exp_run.py --spec experiments/final_candidate.json` |
| `negative.json`        | 测试负面提示词对配饰的抑制效果；生成结果出现头后圆环，因此未采用该方案 | `scripts/exp_run.py --spec experiments/negative.json`        |

## 云端 API 消融实验

云端 API 实验使用 `scripts/hybrid_stability.py --prompt` 直接覆盖提示词，不需要单独定义实验规格。脚本提供费用确认机制（`--confirm-cost`），默认不自动重试，并记录生成指标。

### 配饰移除实验

**状态：已准备，尚未执行。**

实验目标：在重绘过程中移除头部及耳部的耳机状结构，同时保留发型、发色、面部特征、服装和全身构图。

预计调用次数为 1 次。费用暂估约 0.25 元，官方单价尚未核实，详见 `docs/api_backend.md` 第 E 节。

```bash
.venv/bin/python scripts/hybrid_stability.py --runs 1 --max-calls 1 --confirm-cost \
  --base results/exp/levers/cloth_acc_s42_00001_.png \
  --out results/exp/api_noaccessory \
  --prompt "Restyle this chibi figurine as a matte vinyl or soft clay collectible toy.
Keep the original pose, full-body composition, head-to-body proportion,
clothing style and color, hairstyle, and hair color unchanged.
Remove all headphones, earmuffs, headsets, and accessories attached to the head or ears.
Show plain hair only. Do not add any ring, halo, or hoop behind the head.
Use a pure white background and soft studio lighting.
No text, watermark, or logo."
```

### 验收标准

执行后逐项检查。任一指标不达标，即判定实验失败；不自动重试。

| 验收项     | 判定标准                                |
| ---------- | --------------------------------------- |
| 耳机状伪影 | 通过人工检查确认已消失                  |
| 头后圆环   | 通过人工检查确认未出现                  |
| 发型与发色 | 发色相似度 ≥ 0.90，发型通过人工检查     |
| 服装       | 服装主色相似度 ≥ 0.90                   |
| 全身构图   | 头部、躯干、双臂、双腿和双脚均完整可见  |
| 输出规格   | 1088×1088、RGBA 格式、四角 alpha 均为 0 |
