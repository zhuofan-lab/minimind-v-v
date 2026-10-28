# 在 8 张 3090 上加大 MiniMind-V

现在 64M 的语言主干是 MiniMind-V 的短板：视觉特征对 LLM 来说像一门“外语”，能学到什么程度受限于 LLM 本身的语言能力。8 张 3090（每张 24GB）的算力足够把主干加大一个数量级。本分支支持两条路线，可以同时使用：

| 路线 | 做法 | 优点 | 代价 |
|---|---|---|---|
| **A. Qwen 主干** | `--llm_path` 指定 Qwen2.5 / Qwen3 等现成模型，视觉部分不变 | 效果提升最大；0.5B～4B 都能训 | 主干不再是从零训练；权重依赖基座模型目录 |
| **B. 更大的 MiniMind** | 先在 [minimind](https://github.com/jingyaogong/minimind) 仓库预训练更大的语言模型，再用 `--hidden_size/--num_hidden_layers` 训 VLM | 保持全流程从零训练 | 要先花十几个小时训 LLM；受限于 minimind 的数据量（约 50 亿 token），2～4 亿参数比较合适 |

> 下文的显存和耗时是估算值（显存按 CPU 上实测的每层激活大小推算，耗时按每卡约 25 TFLOPS 的有效算力估），误差可能有 ±50%。正式训练前建议先用 `scripts/bench_vlm.py` 在你的卡上测一下，见[第 1 节](#1-先测一下显存和速度)。

## 0. 新增的训练参数

两个训练脚本（`train_pretrain_vlm.py` / `train_sft_vlm.py`）新增了这些参数，默认都关闭，不传时行为和原来一致：

| 参数 | 作用 | 什么时候用 |
|---|---|---|
| `--group_by_length 1` | 按长度分组采样：长度接近的样本放进同一个 batch，配合动态 padding 减少无效计算；同一步各卡的 batch 长度也接近；最长的 batch 排在第一个，显存不够会立刻报错 | 多卡训练建议都开 |
| `--grad_checkpoint 1` | 梯度检查点：反向时重算每层前向，激活显存降到原来的十分之一左右（层数越多降得越多），计算量增加约 1/3 | 显存不够、想开大 batch 时 |
| `--llm_path` | 用 HF 格式的预训练模型做主干（路线 A） | |
| `--lora_rank` | 在主干的所有线性层上加 LoRA（路线 A），常配合 `--freeze_llm 2` | 1.5B 以上的主干 |
| `--vision_model` | 视觉编码器路径；视觉 token 数和特征维度自动从编码器读取 | 换编码器时 |
| `--from_weight` | 除了前缀名，也可以直接给 `.pth` 路径 | |

另外几项多卡/显存优化是自动生效的：
- 开梯度累积（`--accumulation_steps` > 1）时，中间步不再跨卡同步梯度（原来每次 backward 都 all-reduce 一遍）：3090 之间走 PCIe，累积 N 步可以省掉 N-1 次梯度通信；
- DDP 的梯度直接放在通信 bucket 里（`gradient_as_bucket_view`），省掉一份和可训练参数一样大的梯度副本（0.6B 全参时约 2.4GB）；
- Qwen 主干：冻结的参数用 bf16 存（显存减半），只在回答部分计算 logits 并分块算交叉熵——Qwen 的词表有 15 万，直接算完整 logits 在 batch=8、seq=768 时要十几 GB；
- 没有 GPU 时 `torchrun` 自动改用 gloo 后端，可以在 CPU 上调试多卡逻辑。

**全局 batch = 每卡 batch_size × 卡数 × accumulation_steps。** 默认学习率是按 MiniMind 单卡调的，换主干后需要按下面的推荐值显式传 `--learning_rate`。

## 1. 先测一下显存和速度

`scripts/bench_vlm.py` 用随机数据跑几步完整的训练（前向 + 反向 + 优化器），报告每步耗时、吞吐和显存峰值，并按数据集规模估算一个 epoch 的耗时。不需要下载数据集：

```bash
cd scripts
# 从 batch_size=1 开始翻倍，直到显存不够
python bench_vlm.py --llm_path ../model/Qwen3-0.6B --freeze_llm 0 --grad_checkpoint 1 --find_max_batch 1
python bench_vlm.py --llm_path ../model/Qwen3-1.7B --freeze_llm 2 --lora_rank 64 --batch_size 8
python bench_vlm.py --hidden_size 1024 --num_hidden_layers 24 --freeze_llm 0 --batch_size 8   # MiniMind 主干，随机初始化即可
```

`--seq_len` 相当于 batch 内最长样本的长度（默认 768，即 SFT 的上限）；`--avg_tokens` 是动态 padding 后每条样本的平均长度，只影响 epoch 耗时的估算。

## 2. 路线 A：Qwen 主干

### 选哪个模型

| 主干 | 参数量 | 推荐训练方式 | 参数+梯度+优化器 | 激活（每条样本，seq=768） | 每卡 batch（不开 / 开检查点） | SFT 1 epoch 粗估 |
|---|---|---|---|---|---|---|
| minimind-3（现在） | 64M | `--freeze_llm 1` | 约 0.8GB | 约 0.15GB | 32+ | 约 0.5 小时 |
| Qwen2.5-0.5B-Instruct | 0.49B | 全参 `--freeze_llm 0` | 约 8GB | 1.1GB / 0.1GB | 4～8 / 32 | 约 4～6 小时 |
| **Qwen3-0.6B** | 0.6B | 全参 `--freeze_llm 0` | 约 10GB | 1.6GB / 0.1GB | 4 / 16 | 约 5～7 小时 |
| Qwen3-1.7B | 1.7B | LoRA r=64 + `--freeze_llm 2` | 约 5GB | 1.7GB / 0.2GB | 4 / 32 | 约 10～16 小时 |
| Qwen2.5-3B-Instruct | 3.1B | LoRA r=64 + `--freeze_llm 2` | 约 8GB | 2.3GB / 0.3GB | 4 / 16 | 约 19～29 小时 |
| Qwen3-4B | 4.0B | LoRA r=64 + `--freeze_llm 2` | 约 10GB | 约 3GB / 0.35GB | 2 / 16 | 约 24～36 小时 |

- 1.5B 以上全参训练放不下：仅参数、梯度和 AdamW 状态就要约 16 字节/参数（1.7B 约 27GB），而 DDP 每张卡都存一份完整的。需要 FSDP/ZeRO 把这部分切到 8 张卡上，本分支还没有实现；LoRA 或 `--freeze_llm 1` 都能放下。
- LoRA 虽然只训练很少的参数，但每个线性层的输入都要留着给 LoRA 算梯度，激活显存接近全参训练；1.5B 以上建议直接开 `--grad_checkpoint 1`。
- 8 个进程会各自先把基座模型读进内存再搬到显卡上：4B（bf16）加载时内存峰值约 8×8GB=64GB，1.7B 约 27GB。
- 推荐从 **Qwen3-0.6B 全参** 开始：一天之内能训完，效果相比 64M 已经会有明显提升，也方便和 1.7B/4B 对比。
- 需要带对话模板的版本（Qwen3 本身就是；Qwen2.5 选 `*-Instruct`）。目前只支持 ChatML（`<|im_start|>`）格式的模型：Qwen2/2.5/3、SmolLM2、Yi 等；Qwen 的词表自带 `<|image_pad|>`，其他模型会自动新增这个 token。
- 耗时假设 SFT 数据 290 万条、动态 padding 后平均约 330 token（约 10 亿 token/epoch）；`--epochs` 默认是 2。实际耗时以训练日志里的 `epoch_time` 为准。

### 下载

```bash
# 视觉编码器（和原来一样）
modelscope download --model gongjy/siglip2-base-p32-256-ve --local_dir ./model/siglip2-base-p32-256-ve
# 语言主干，任选其一
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./model/Qwen3-0.6B
modelscope download --model Qwen/Qwen3-1.7B --local_dir ./model/Qwen3-1.7B
```

### 训练（8 卡）

```bash
cd trainer

# 第一阶段（可选，但建议）：只训练投影层，让视觉特征先对齐到 Qwen 的语义空间。全局 batch 16×8=128
# 3B 以上的主干在这一步也加 --grad_checkpoint 1
torchrun --nproc_per_node 8 train_pretrain_vlm.py --llm_path ../model/Qwen3-0.6B \
    --freeze_llm 2 --batch_size 16 --learning_rate 1e-3 --epochs 1 --group_by_length 1

# 第二阶段：SFT。Qwen3-0.6B 全参，全局 batch 16×8=128
torchrun --nproc_per_node 8 train_sft_vlm.py --llm_path ../model/Qwen3-0.6B --from_weight pretrain_vlm \
    --freeze_llm 0 --batch_size 16 --grad_checkpoint 1 --learning_rate 2e-5 --epochs 1 --group_by_length 1

# Qwen3-1.7B / Qwen3-4B：LoRA。4B 把 batch_size 改成 8、accumulation_steps 改成 2
torchrun --nproc_per_node 8 train_sft_vlm.py --llm_path ../model/Qwen3-1.7B --from_weight pretrain_vlm \
    --freeze_llm 2 --lora_rank 64 --batch_size 16 --grad_checkpoint 1 --learning_rate 2e-4 --epochs 1 --group_by_length 1
```

学习率参考了 LLaVA-1.5 的设置：投影层预训练 1e-3，全参 SFT 2e-5，LoRA 2e-4。第一阶段的权重保存为 `out/pretrain_vlm_qwen3-0.6b.pth`，第二阶段为 `out/sft_vlm_qwen3-0.6b.pth`（文件名里是主干目录名）。

### 测试和 WebUI

```bash
python eval_vlm.py --llm_path ./model/Qwen3-0.6B --weight sft_vlm
python eval_vlm.py --llm_path ./model/Qwen3-0.6B --weight sft_vlm --image 你的图片.jpg --prompt "图里有几个人？"
cd scripts && python web_demo_vlm.py --llm_path ../model/Qwen3-0.6B   # 默认加载 ../out/sft_vlm_qwen3-0.6b.pth
```

### 权重文件只保存“和基座不同的部分”

`out/*_qwen3-0.6b.pth` 里只有投影层、训练过的层和 LoRA，冻结且没改过的参数不重复保存（例如 1.7B + LoRA 的 SFT 权重只有约 150MB）。加载时会先从 `--llm_path` 读基座模型再叠加，所以**测试和部署时需要同一个基座目录**。多阶段训练时，前一阶段改过的参数会一直跟着保存下去，不会丢。

## 3. 路线 B：更大的 MiniMind（从零训练）

### 第一步：在 minimind 仓库训练更大的语言模型

按 minimind 仓库的 README 下载 `pretrain_t2t.jsonl`（10GB）和 `sft_t2t.jsonl`（14GB），然后：

```bash
git clone https://github.com/jingyaogong/minimind && cd minimind/trainer

# 预训练：1024×24（3.2 亿参数）。单卡默认是 batch 32 × 累积 8 = 256；8 卡时用 16 × 8 卡 × 2 = 256 保持一致
torchrun --nproc_per_node 8 train_pretrain.py --hidden_size 1024 --num_hidden_layers 24 \
    --data_path ../dataset/pretrain_t2t.jsonl --max_seq_len 380 --batch_size 16 --accumulation_steps 2 --epochs 1

# SFT：seq=768 时每卡 batch 8 左右（minimind 仓库没有梯度检查点）
torchrun --nproc_per_node 8 train_full_sft.py --hidden_size 1024 --num_hidden_layers 24 --from_weight pretrain \
    --data_path ../dataset/sft_t2t.jsonl --batch_size 8 --epochs 1

# 作为 VLM 的基座
cp ../out/full_sft_1024.pth /path/to/minimind-v-v/out/llm_1024.pth
```

- 规模：minimind 的数据大约 50 亿 token，1024×16（2.2 亿）到 1024×24（3.2 亿）比较合适，再大就数据不够了。minimind 的 README 引用了 MobileLLM 的结论：参数量相同时“深而窄”比“宽而浅”好，所以优先加层数。
- 默认 8 个注意力头、4 个 KV 头，hidden_size=1024 时 head_dim=128，3090 上的 Flash Attention 可以直接用。
- 耗时：预训练和 SFT 各 1 epoch，8×3090 上大约 15 小时；minimind 默认各 2 epoch，约翻倍。

### 第二步：在本仓库训练 VLM

```bash
cd trainer
torchrun --nproc_per_node 8 train_pretrain_vlm.py --hidden_size 1024 --num_hidden_layers 24 --from_weight llm \
    --batch_size 16 --group_by_length 1
torchrun --nproc_per_node 8 train_sft_vlm.py --hidden_size 1024 --num_hidden_layers 24 --from_weight pretrain_vlm \
    --freeze_llm 1 --batch_size 8 --group_by_length 1

cd .. && python eval_vlm.py --hidden_size 1024 --num_hidden_layers 24 --weight sft_vlm
cd scripts && python convert_vlm.py --hidden_size 1024 --num_hidden_layers 24   # 需要 transformers 格式时
```

- 主干变大后，`--freeze_llm 1`（只训首尾层）可训练的比例变小了。3.2 亿参数全参也放得下（参数+梯度+优化器约 5GB），可以试 `--freeze_llm 0` 配合较小的学习率（如 1e-5），对比看语言能力有没有被图文数据冲掉。
- 权重文件按 `hidden_size` 命名（`sft_vlm_1024.pth`），1024×16 和 1024×24 会互相覆盖；同时训多个深度时用 `--save_weight` 区分，例如 `--save_weight sft_vlm_L24`，测试时 `--weight sft_vlm_L24`。

## 4. 在 CPU 上验证过的内容

这些改动在没有 GPU 的环境里用迷你模型（随机初始化的迷你 SigLIP + 迷你 Qwen2/Qwen3）验证过，见 `tests/test_scaling.py` 和 `tests/test_scripts_e2e.py`：

- Qwen 主干：只在回答位置分块算出的 loss 和梯度与完整 logits 的交叉熵一致；KV cache 与无 cache 生成一致；Qwen2.5 风格 tokenizer（bos 为 None、基座 eos 为 `<|endoftext|>`、自动加 system prompt）下标签只覆盖回答；
- LoRA 初始为恒等变换、只训练 LoRA 和投影层；增量权重保存后从基座重新加载，输出一致；多阶段训练不丢前一阶段的改动；
- 冻结参数 bf16、可训练参数 fp32 的混合存储在 autocast 下前向/反向正常；
- 梯度检查点开关前后，MiniMind（dense / MoE）和 Qwen 主干的 loss 和梯度完全一致；
- 用 `torchrun` 起 2 个进程（gloo）跑真实训练脚本：DDP + 梯度累积 + 按长度分组 + 断点续训，以及 Qwen 主干 + LoRA + 梯度检查点；
- 默认参数（MiniMind 主干、不开新功能）下，和上游代码的 loss 日志逐行一致。

GPU 上的显存和速度这里没法实测，以 `bench_vlm.py` 在你的 3090 上的结果为准。

## 5. 还没做的

- **FSDP / ZeRO**：1.5B 以上全参训练需要把参数、梯度和优化器状态切到 8 张卡上；
- **Qwen 主干导出为独立的 transformers 格式**：目前测试和 WebUI 需要基座目录 + 增量权重；
- **更强的视觉编码器**：主干变大后，256×256、64 个视觉 token 会成为新的瓶颈。视觉 token 数已经改为从编码器自动读取，换成 SigLIP2 p16（256 token）或 so400m 后，可以再加 2×2 pixel-shuffle 把 token 数压回 64～144。
