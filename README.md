# ChibiToy：真人照片转 Q 版潮玩手办

将真人照片转换为 Q 版潮玩手办风格图片（1.5–2 头身、哑光 PVC/软陶质感、纯白或透明背景、≥1080×1080 PNG）。

提供两条可切换的生图后端：

| 后端              | 说明                                                               | 单张耗时      |
| ----------------- | ------------------------------------------------------------------ | ------------- |
| `api`             | 调用阿里云百炼图像 API（`qwen-image-2.0`），无需在本机运行扩散模型 | **≈ 18–23 s** |
| `comfyui`（本地） | 通过本地 ComfyUI 工作流运行 ControlNet、LoRA 和 IP-Adapter         | 未测试        |

> 两种后端共用输入校验、预处理、背景处理和输出校验流程。

---

## 目录结构

```
ChibiToy/
├── README.md                  项目说明
├── requirements.txt           原生 Python 依赖（pip 用户）
├── pyproject.toml / uv.lock   uv 依赖与锁定（uv 用户）
├── config/
│   ├── chibitoy.example.yaml  API 后端配置模板
│   └── chibitoy.yaml          本地配置文件（包含 API Key，已加入 `.gitignore`）
├── src/chibitoy/              核心库
│   ├── generate.py            CLI 入口（--backend api|comfyui）
│   ├── graph.py               ComfyUI 工作流构建与校验
│   ├── client.py              ComfyUI HTTP 客户端
│   ├── apibackend.py          云端 API 后端与厂商适配
│   ├── apiconfig.py           配置加载与密钥卫生
│   ├── preprocess.py          输入校验 / EXIF / 缩放 / 输出校验
│   ├── analysis.py            颜色与解剖取带指标
│   ├── prompts.py             正/负向提示词
│   └── identity.py            CLIP 视觉相似度（**不是人脸识别**）
├── scripts/                   18 个可执行脚本（见下表）
├── workflows/                 8 份可直接导入 ComfyUI 的 .json
├── models/                    权重（.gitignore 排除）
├── inputs/img/                你的照片（.gitignore 排除）
├── results/                   生成的 Q 版图片（按日期命名）
├── outputs/                   脚本生成的原图和中间文件
└── docs/                      交付文档（索引见下）
```

---

## 快速开始

### 1. 安装（二选一）

```bash
# 方式 A：uv
uv sync
# 方式 B：Python
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. 配置

```bash
cp config/chibitoy.example.yaml config/chibitoy.yaml
# 编辑 config/chibitoy.yaml，填写 providers.aliyun.api_key
```

### 3. 运行

```bash
# 1. 使用云端 API
.venv/bin/python -m chibitoy.generate --backend api \\
    --input-image inputs/img/test.jpg \\
    --appearance "short black hair, dark brown jacket"
# 2. 使用本地 ComfyUI
.venv/bin/python -m chibitoy.generate --mode txt2img --width 1088 --height 1088 \\
    --input-image inputs/img/test.jpg \\
    --ipadapter ip-adapter-plus-face_sdxl_vit-h.safetensors --ipadapter-weight 0.6 \\
    --ipadapter-crop top-square --ipadapter-prep-bg \\
    --lora Designer_BlindBox-000015.safetensors \\
    --controlnet canny_sdxl.safetensors \\
    --controlnet-image inputs/chibi_silhouette.png --controlnet-strength 0.7 \\
    --bg-removal birefnet.safetensors --seed 42 --steps 30 \\
    --appearance "short black hair, dark brown jacket"
# 3. 批量生成
.venv/bin/python scripts/batch_generate.py --inputs inputs/img --pattern "*.jpg" \\
    --steps 30 --seed 42 --mode txt2img --width 1088 --height 1088 --lora Designer_BlindBox-000015.safetensors
```

生成结果默认写入 `results/`；如需分开，用 `--out outputs/run_$(date +%F)`.

### 4. 输出目录约定

| 目录                | 内容                                                                      |
| ------------------- | ------------------------------------------------------------------------- |
| `results/`          | **最终成品**，命名 `YYYY-MM-DD_q版_<来源>.png`                            |
| `outputs/`          | 脚本生成的原始图和中间文件，包括批量生成、分析、稳定性评测和 API 原始返回 |
| `tmp/preprocessed/` | 参考图裁切与抠图等临时文件                                                |

脚本使用各自的 `--prefix` 命名文件；若文件名冲突，`config.unique_path()` 会生成不重复的路径，不会覆盖已有文件。

---

## 在 ComfyUI 网页界面中选择工作流

`workflows/` 中有多份工作流文件。日常在网页界面中使用时，主要涉及以下几份：

| 工作流                          | 什么时候用                                 | 说明                                                                                                                                                                                                                                      |
| ------------------------------- | ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`chibitoy_api_only.ui.json`** | 用于云端 API                               | LoadImage → ChibiToy 云端 API 节点 → SaveImage，不加载本地扩散模型。首次使用前需安装该自定义节点（见本节末的软链命令）                                                                                                                    |
| **`chibitoy_web.ui.json`**      | 用于网页端运行                             | 工作流内包含缩放和抠图白底合成。使用时将真人照片载入对应的 LoadImage 节点。图里有两个 `LoadImage`：上面放**你的照片**，下面放 **ControlNet 轮廓图** `inputs/chibi_silhouette.png`（用 `scripts/make_chibi_silhouette.py` 生成，仓库已带） |
| `chibitoy_final.ui.json`        | 需要在界面里查看与生产配置对应的工作流时用 | 对应 CLI 的 `--backend comfyui` 配置。该工作流依赖 CLI 预处理输入照片；直接载入高分辨率原图可能导致显存不足                                                                                                                               |

```bash
# 网页端操作：ComfyUI → Workflow → Open → 选择 `chibitoy_web.ui.json` → 替换照片 → Run
# 轮廓图缺失时先重建：
.venv/bin/python scripts/make_chibi_silhouette.py --out inputs/chibi_silhouette.png
```

> 两者都可由 CLI 用 `--save-workflow <前缀>` 重新导出

---

## 命令行参数（共 52 个）

`--backend comfyui`与 `--backend api` 共用同一套参数；下表按用途分组，**由 CLI 解析器自动生成**，以 `--help` 输出为准。

```bash
.venv/bin/python -m chibitoy.generate --help      # 查看实时参数表
```

**常用参数**：

```bash
--backend api|comfyui   --input-image 照片   --appearance "发型, 服装颜色"
--width/--height 1088   --seed 42            --out outputs/run_$(date +%F)
```

<details>
<summary><b>展开全部 52 个参数</b></summary>

### 输入与预处理

| 参数                | 默认值      | 说明                                                                                                         |
| ------------------- | ----------- | ------------------------------------------------------------------------------------------------------------ |
| `--input-image`     | —           | img2img 模式的输入照片路径（JPG/PNG/WEBP）                                                                   |
| `--min-side`        | `500`       | 输入图片**推荐**最低边长（px）：低于此值只给出低分辨率警告，不再拒绝（阶段 1B-3A-1 起）                      |
| `--max-file-size`   | `10000000`  | 输入文件大小上限（字节），默认 10000000（img2img 模式）                                                      |
| `--max-pixels`      | `40000000`  | 输入图片像素总量上限（超过则报错，避免异常大图）                                                             |
| `--no-preprocess`   | `False`     | 跳过预处理，直接上传原图（img2img 模式）                                                                     |
| `--test-mode`       | `False`     | [兼容保留] 低分辨率素材现已默认被接受（仅警告），本开关不再必要；文件大小/格式/解码安全/像素总量限制始终生效 |
| `--strict-min-side` | `False`     | 严格模式：恢复旧行为，低于 --min-side 直接拒绝（默认关闭）                                                   |
| `--target-size`     | `1024x1024` | img2img 预处理目标尺寸：1024（正方形）或 宽x高，如 896x1152                                                  |
| `--pad-color`       | `#ffffff`   | img2img 预处理补边颜色：#RRGGBB 或 R,G,B；默认白色中性背景                                                   |

### 尺寸与采样

| 参数           | 默认值     | 说明                                                           |
| -------------- | ---------- | -------------------------------------------------------------- |
| `--width`      | `1024`     | 生成宽度                                                       |
| `--height`     | `1024`     | 生成高度                                                       |
| `--steps`      | `30`       | 采样步数                                                       |
| `--cfg`        | `7.5`      | CFG 强度                                                       |
| `--seed`       | `42`       | 随机种子                                                       |
| `--sampler`    | `dpmpp_2m` | 采样器名称                                                     |
| `--scheduler`  | `karras`   | 调度器名称                                                     |
| `--batch-size` | `1`        | 批量张数                                                       |
| `--mode`       | `txt2img`  | 生成模式：txt2img 文生图 / img2img 图生图                      |
| `--denoise`    | `0.55`     | img2img 去噪强度：1.0≈文生图，0.0≈原图复现；越低越保留输入结构 |

### 模型与 LoRA

| 参数                    | 默认值                       | 说明                                                   |
| ----------------------- | ---------------------------- | ------------------------------------------------------ |
| `--ckpt`                | `sd_xl_base_1.0.safetensors` | 基础模型文件名                                         |
| `--lora`                | —                            | img2img 模式加载的风格 LoRA 文件名（models/loras/ 下） |
| `--lora-strength-model` | `1.0`                        | LoRA 对模型的权重                                      |
| `--lora-strength-clip`  | `1.0`                        | LoRA 对文本编码器的权重                                |

### 身份参考（IP-Adapter）

| 参数                      | 默认值                                        | 说明                                                                                                                                                                                                                                                                                                                                                          |
| ------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--ipadapter`             | —                                             | IP-Adapter 模型文件名（models/ipadapter/ 下）。启用后需用 --input-image 提供参考照片；仅注入身份特征，不改变构图                                                                                                                                                                                                                                              |
| `--ipadapter-image`       | —                                             | IP-Adapter 身份参考照片路径。**可与 --input-image 不同**——两阶段方案（先定 Q 版结构、再注入身份）必须分别指定；未提供时回退到 --input-image                                                                                                                                                                                                                   |
| `--ipadapter-weight`      | `1.0`                                         | IP-Adapter 权重（越大越贴近参考照片，过高会压过造型提示词）                                                                                                                                                                                                                                                                                                   |
| `--ipadapter-weight-type` | `linear`                                      | IP-Adapter 权重类型（linear/ease in/style transfer 等）                                                                                                                                                                                                                                                                                                       |
| `--ipadapter-start-at`    | `0.0`                                         | IP-Adapter 生效起始步（0\~1）。调大 = 让提示词先决定构图，IP-Adapter 只在后段补身份细节，可削弱写实比例拉力                                                                                                                                                                                                                                                   |
| `--ipadapter-end-at`      | `1.0`                                         | IP-Adapter 生效结束步（0\~1）                                                                                                                                                                                                                                                                                                                                 |
| `--ipadapter-crop`        | `top-square`                                  | IP-Adapter 参考图裁剪：top-square=只取顶部正方形（**默认**，实测在本次受控对照中服装主色最好）；head-square=按 BiRefNet 的 alpha 自适应裁到头部（对头部特写更合理，但实测在本张全身照上指标更低，见 results/exp/crop/report.md）；none=整图（证件照的头肩区域）。IP-Adapter 会把参考图的背景与身形比例一并注入，导致 Q 版比例被拉向写实；裁剪到头部可显著缓解 |
| `--ipadapter-prep-bg`     | `False`                                       | 先把身份参考图做**真实前景分割并合成到白底**再喂给 IP-Adapter。实测：蓝底证件照直接当参考会把蓝色注入产物（生成蓝衣 + 蓝色伪影），因为 IP-Adapter 读的是整张参考图                                                                                                                                                                                            |
| `--clip-vision`           | `CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors` | CLIP Vision 编码器文件名（models/clip_vision/ 下）                                                                                                                                                                                                                                                                                                            |
| `--head-side-scale`       | `2.0`                                         | 自适应头部裁切的边长 = 头部区域宽度 × 该系数（默认 2.0，越大留的肩部越多、越小越贴近脸）                                                                                                                                                                                                                                                                      |
| `--head-probe-side`       | `1280`                                        | 自适应裁切前把参考图缩到该长边，避免对 12MP 原图跑分割                                                                                                                                                                                                                                                                                                        |
| `--photo-side`            | `1024`                                        | 图内预处理时照片缩放到的边长（配合 --preprocess-in-graph）                                                                                                                                                                                                                                                                                                    |
| `--preprocess-in-graph`   | `False`                                       | 把照片预处理（缩放到固定边长 + 真实抠图合成白底）**放进工作流图里**。导出的工作流若要**在网页端**直接用（用户把手机原图拖进 LoadImage），必须开启——否则 4000x3000 的原图会让 VAE/后续节点爆显存                                                                                                                                                               |

### 结构控制（ControlNet）

| 参数                    | 默认值 | 说明                                                                                                      |
| ----------------------- | ------ | --------------------------------------------------------------------------------------------------------- |
| `--controlnet`          | —      | ControlNet 模型文件名（models/controlnet/ 下，如 canny_sdxl.safetensors）                                 |
| `--controlnet-image`    | —      | ControlNet 结构参照图路径（内部先跑 Canny 取边缘）。用一张 Q 版手办图作参照，可把构图与头身比锁到目标区间 |
| `--controlnet-strength` | `0.8`  | ControlNet 强度（越高越严格遵循参照图结构）                                                               |

### 背景与后处理

| 参数           | 默认值 | 说明                                                                                                                                               |
| -------------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--bg-removal` | —      | 背景移除模型文件名（models/background_removal/ 下，如 birefnet.safetensors）。启用后输出为带 alpha 的透明背景 PNG，走真实前景分割而非补边/纯色覆盖 |

### 提示词

| 参数           | 默认值                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | 说明                                    |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------- |
| `--appearance` | `short black hair, casual grey t-shirt`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | 需要保留的发型/发饰/服装描述            |
| `--positive`   | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | 直接指定正向提示词（覆盖 --appearance） |
| `--negative`   | `photorealistic, realistic photo, real human, adult proportions, normal head-to-body ratio, tall slender body, long legs, long arms, elongated limbs, tiny head, deformed face, distorted face, asymmetric eyes, cross-eyed, extra eyes, mutated hands, extra fingers, missing fingers, fused fingers, extra limbs, missing limbs, extra arms, extra legs, bad anatomy, extra hands, multiple hands, three hands, duplicated hands, duplicated limbs, malformed hands, mutated hands, misplaced hands, glossy plastic, shiny surface, strong specular highlights, harsh reflections, complex background, cluttered background, busy background, gradient background, vignette, dark background, textured background, patterned background, backdrop props, studio props, hard shadows, harsh lighting, dramatic lighting, cropped, out of frame, cut off head, cut off feet, close-up, blurry, out of focus, noise, grainy, jpeg artifacts, low quality, lowres, watermark, text, signature, logo, brand logo, artist name, text on shirt, printed text on clothes, lettering, words, caption, label, gibberish text, garbled writing, badge, patch, emblem, sticker, tag, sew-on label, walking, running, jumping, raised leg, dynamic pose, tilted body, skin pores, skin blemish, realistic skin texture, photograph of a person` | 负向提示词                              |

### 后端与执行

| 参数              | 默认值                                                           | 说明                                                                                                                                                |
| ----------------- | ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--backend`       | `comfyui`                                                        | 生图后端：comfyui=本地工作流（默认，保持既有行为）；api=云端 API（配置见 config/chibitoy.yaml，密钥只从该文件读取）                                 |
| `--confirm-cost`  | `False`                                                          | **仅 --backend api 需要**：确认知晓本次将发起一次真实付费调用。未给出时只打印费用提示并退出，**不创建客户端、不发起请求**                           |
| `--max-retries`   | —                                                                | **仅 --backend api 需要**：失败后最多尝试 N 次（含首次）。默认沿用配置值，而配置默认已是 1（不自动重试）。调大即意味着一次失败可能变成 N 次付费调用 |
| `--prefix`        | `chibitoy`                                                       | 输出文件名前缀                                                                                                                                      |
| `--out`           | `/Users/genwilliam/Developer/github/genwilliam/ChibiToy/results` | 输出目录                                                                                                                                            |
| `--workflow`      | —                                                                | 直接执行一个已导出的 API 格式工作流文件（忽略其他生成参数）                                                                                         |
| `--save-workflow` | —                                                                | 把工作流保存到该前缀（同时生成 .api.json 与 .ui.json）                                                                                              |
| `--dry-run`       | `False`                                                          | 只校验工作流，不执行                                                                                                                                |
| `--timeout`       | `1800.0`                                                         | 执行超时（秒）                                                                                                                                      |

</details>

## 常用脚本

| 脚本                                                             | 用途                                                                      |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `scripts/batch_generate.py`                                      | 批量生成                                                                  |
| `scripts/hybrid_stability.py`                                    | N 次稳定性评测（**含费用保护** `--confirm-cost`、失败不重试、分阶段计时） |
| `scripts/color_check.py`                                         | 服装/发色客观指标                                                         |
| `scripts/make_chibi_silhouette.py`                               | 生成 ControlNet 纯轮廓参照图                                              |
| `scripts/exp_run.py`                                             | 规格驱动的对照实验（`experiments/*.json`）                                |
| `scripts/blind_eval.py`                                          | 人工盲评（匿名编号、组别隔离、揭盲汇总）                                  |
| `scripts/sample_registry.py`                                     | 样本登记与内容哈希查重                                                    |
| `scripts/verify_delivery.py` / `verify_all.py` / `audit_docs.py` | 交付核验 / 一键全链路核验 / 文档结构体检                                  |

---

## 测试与核验

```bash
.venv/bin/python -m pytest tests/ -q -m "not live and not slow"
.venv/bin/python scripts/verify_all.py
.venv/bin/python scripts/audit_docs.py
```

---

## 文档索引

| 文档                                | 内容                                                   |
| ----------------------------------- | ------------------------------------------------------ |
| `docs/acceptance_criteria.md`       | **《风格验证验收标准》**：校验维度、判定规则、合格指标 |
| `docs/delivery.md`                  | 交付清单与逐项验收状态                                 |
| `docs/environment.md`               | 环境、模型 SHA256、节点依赖与安装                      |
| `docs/issues.md`                    | 已知问题清单（含外部阻塞与缓解状态）                   |
| `docs/api_backend.md`               | API 后端用法、实测数据、费用与隐私说明                 |
| `docs/api_provider_comparison.md`   | 国内生图 API 调研对比                                  |
| `docs/photo_intake_workflow.md`     | 补充新照片后的验证流程                                 |
| `docs/face_model_license_survey.md` | 人脸嵌入模型许可调研                                   |
| `docs/HANDOVER.md`                  | 交接说明与下一步                                       |

---

## 模型清单与校验

模型清单和哈希由 `scripts/verify_delivery.py` 检查。文档记录与磁盘文件不一致时，检查会失败。

下表为省略形式的 SHA256（前缀），完整值以核验脚本为准：

| 模型                                                      | 用途                   | 许可                    | 字节数        | SHA256（省略） |
| --------------------------------------------------------- | ---------------------- | ----------------------- | ------------- | -------------- |
| `checkpoints/sd_xl_base_1.0.safetensors`                  | 基础模型               | CreativeML OpenRAIL++-M | 6,938,078,334 | `31e35c80`     |
| `loras/Designer_BlindBox-000015.safetensors`              | 风格 LoRA（在用）      | **不明**                | 681,239,124   | `218c7602`     |
| `ipadapter/ip-adapter-plus-face_sdxl_vit-h.safetensors`   | 身份保持               | apache-2.0              | 847,517,512   | `677ad886`     |
| `clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors` | 视觉编码器             | apache-2.0              | 2,528,373,448 | `6ca9667d`     |
| `background_removal/birefnet.safetensors`                 | 前景分割               | MIT                     | 444,473,596   | `9ab37426`     |
| `controlnet/canny_sdxl.safetensors`                       | 结构控制               | apache-2.0              | 2,502,139,104 | `bf47cd75`     |
| `loras/figurine_slider.safetensors`                       | 候选（**不采用**）     | MIT                     | 8,789,076     | `5484762f`     |
| `loras/toy_v1.safetensors`                                | 候选（**格式不兼容**） | openrail++              | 93,065,304    | `80d80f49`     |

> 此处仅显示 SHA256 前缀，完整值见 `docs/environment.md`；可使用 `scripts/verify_manifest.py` 进行校验。

> 代码许可不代表模型权重也允许相同用途。使用第三方权重前，应确认其许可条款，尤其是商业用途（见 `docs/face_model_license_survey.md`）。

---

## 隐私与许可

本项目不提供照片托管服务，也不主动收集用户照片或生成结果。使用本地后端时，图像在本机处理；使用 API 后端时，输入照片会发送给所配置的云端服务，具体数据处理方式以服务商政策为准。
