# 本分支改进说明

本仓库基于上游 [jingyaogong/minimind-v](https://github.com/jingyaogong/minimind-v)（`041d528`，2026-10-27）。
改动只涉及代码健壮性、训练效率和工程化，**模型结构、默认超参数和已发布权重的格式都没有变**，上游的 `*.pth` 权重可以直接使用。

## 一、已完成的改动

### 修复的问题

| 问题 | 上游表现 | 本分支 |
|---|---|---|
| 训练数据中有灰度 (L) / 调色板 (P) / CMYK 等非 RGB 图片 | `image2tensor` 只转换 RGBA/LA，其余模式进入 SigLIP 预处理后报 `ValueError: Unsupported number of image dimensions: 2`，DataLoader 中途退出 | 所有非 RGB 图片统一转为 RGB |
| 两个 `<image>` 紧挨着（如 `<image><image>请比较两张图`） | 128 个连续占位符被当成一张图，只填入 64 个特征、序列被缩短 64，前向报 `RuntimeError: The size of tensor a (136) must match the size of tensor b (200)` | 按顺序把第 k 个占位符对应到第 k // 64 张图，序列长度不变 |
| 一个 batch 里各样本图片数量不同 | `vlm_collate_fn` 用 `torch.stack` 拼图片，直接报错 | 用全 0 图补齐图片维度（补出的图没有对应占位符，不会被用到），多图/单图样本可以混合训练 |
| 没下载视觉编码器就开始训练/评估 | DataLoader 里报 `TypeError: 'NoneType' object is not callable`，看不出原因 | 启动时直接提示缺少 `siglip2-base-p32-256-ve` 及下载方式 |
| 权重文件和模型配置对不上（选错文件、dense/MoE 不一致、key 前缀不同） | `load_state_dict(strict=False)` 静默跳过，模型带着随机初始化的层继续训练或推理 | 打印缺失/多余的参数；从纯 LLM 权重开始训练时缺少 `vision_proj` 属于正常情况，单独提示 |
| 整个 batch 没有任何监督 token（例如回答被 `max_seq_len` 整段截掉，`batch_size` 越小越容易遇到） | loss 为 NaN（梯度其实是 0，不影响权重，但日志和训练曲线里会出现 NaN） | 返回 0；有监督 token 时与原来的 mean 归约完全等价 |
| `model.train()` 之后 | 冻结的视觉编码器也被切到 train 模式（SigLIP 默认没有 dropout，所以目前无实际影响） | 视觉编码器始终保持 eval 模式 |

### 训练效率

- **动态 padding**：Dataset 会把每条样本补齐到 `max_seq_len`（Pretrain 450 / SFT 768）。`vlm_collate_fn` 现在把 batch 裁到“最后一个有监督 token”所在位置，并向上取整到 64 的倍数。因果注意力下后面的 token 影响不到前面的位置，所以 **loss 完全不变**——用同一份迷你数据和随机种子与上游代码对比，dense 模型 Pretrain 的 loss 日志和最终权重逐位一致；SFT 的 loss 日志逐行一致，最终权重最大差 3e-5（序列长度不同带来的浮点误差）。能省多少计算取决于数据的长度分布（batch 内最长的样本越短，省得越多）。
  - MoE 的 aux_loss 不再把 padding token 计入路由统计，这是唯一的行为变化。
- **图像特征注入向量化**：原 `count_vision_proj` 对每个样本 `.tolist()` 再逐 token 做 Python 循环，每一步前向都会触发 GPU→CPU 同步，并且用 `@torch.compiler.disable` 跳过了编译。现在改为 gather + where，没有同步，可以被 `torch.compile` 完整编译（测试里用 `fullgraph=True` 验证）。常规输入下与原实现逐元素一致。
- RoPE buffer 的检查只在第一次前向时做一次（原来每一步都会触发一次同步，并让 `torch.compile` 在这里断图）。

### Checkpoint

- 续训文件 `checkpoints/*_resume.pth` 不再保存冻结的视觉编码器（约 95M 参数，fp32 下每次约 380MB），续训时视觉编码器会从磁盘重新加载。旧的续训文件仍然可以加载。
- `out/*.pth` 改为“写临时文件 + 原子替换”（README 写的是原子保存，但 `out/` 这份原来是直接 `torch.save`），保存中断不会留下损坏的权重。

### 脚本

- `eval_vlm.py` 新增 `--image`（测单张图）、`--prompt`（自定义问题）、`--vision_model`：
  ```bash
  python eval_vlm.py --weight sft_vlm --image ./dataset/eval_images/image-04-yellow-car.jpg --prompt "这辆车是什么颜色？"
  ```
- `scripts/convert_vlm.py` 改为命令行参数（原来写死 `use_moe=True` 和路径，转换 dense 模型要改代码）：
  ```bash
  cd scripts
  python convert_vlm.py --weight sft_vlm                 # dense → ../minimind-3v
  python convert_vlm.py --weight sft_vlm --use_moe 1     # MoE   → ../minimind-3v-moe
  ```
- 训练脚本中已废弃的 `torch.cuda.amp.autocast/GradScaler` 换成 `torch.amp.*`。
- `.gitignore` 增加 `checkpoints/`、`dataset/*.parquet`、`model/siglip2-base-p32-256-ve/`，避免在 fork 里误提交几百 MB 的数据和权重。

### 测试

新增 `tests/`，用随机初始化的迷你 SigLIP（同样输出 64 个视觉 token）和迷你 LLM，在 CPU 上几秒内覆盖：
图像特征注入（与原实现对比、相邻多图、截断、`torch.compile`）、前向/反向、loss、非 RGB 图片、数据集标签只覆盖 assistant 回复、动态 padding 前后 loss 一致、多图混合 batch、KV cache 与无 cache 的贪心生成一致、MoE、三种冻结策略、权重加载报告、checkpoint 保存与续训。

```bash
pip install pytest
pytest tests -q
```

## 二、还可以继续改进的方向

下面这些需要 GPU 和真实数据来验证效果，没有直接改进代码里：

1. **按长度分组采样**（类似 `group_by_length`）：配合动态 padding 能省下更多计算。Pretrain 默认 `batch_size=16`，batch 内最长的样本往往接近上限，单靠裁剪省得有限。
2. **用 `<|vision_start|>` / `<|vision_end|>` 包住图像 token**：tokenizer 里已经有这两个特殊 token 但没有用上，加上后模型能明确知道图像边界，多图时尤其有用（需要重新训练，和已发布权重不兼容）。
3. **更高的分辨率 + token 压缩**：目前 256×256 只有 64 个视觉 token，对文档、图表类问题（ALLaVA-VFLAN 里不少）细节不够。可以换 384 分辨率的编码器，再用 2×2 pixel-shuffle（InternVL）或 PatchMerger（Qwen2-VL）把 token 数压回去，或者参考 LLaVA-NeXT 的 AnyRes 做动态分辨率切块。
4. **LoRA 微调 LLM 中间层**：`freeze_llm=1` 只解冻首尾两层，LoRA 能在不明显损伤语言能力的前提下提供更多可训练容量。
5. **纯文本样本跳过视觉编码**：SFT 中约 23 万条纯文本对话用 8×8 黑图占位，视觉编码器仍然会把它们编码一遍，可以跳过。
6. **定量评测**：目前只靠 6 张图主观看效果，可以加一个固定的验证集算 val loss，以及 POPE（幻觉）、MMBench 等基准的评测脚本。
7. **Web Demo 多轮对话**：现在每轮对话相互独立，上传图片后追问时模型既看不到之前的图片也看不到之前的对话。
8. **权重衰减**：AdamW 默认的 0.01 目前也作用在 norm、bias 和 embedding 上，通常会把它们排除。
