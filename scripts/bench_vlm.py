"""
在自己的显卡上测某个配置的显存峰值和训练速度，用来挑 batch_size、冻结策略和是否开梯度检查点。

用随机数据跑几步完整的训练（前向 + 反向 + 优化器更新），不需要下载数据集；结果按数据集规模估算一个 epoch 的耗时。

    cd scripts
    # MiniMind 主干（随机初始化即可，不需要 LLM 权重）
    python bench_vlm.py --hidden_size 1024 --num_hidden_layers 24 --freeze_llm 0 --batch_size 8
    # Qwen 主干（需要先下载到 ../model/）
    python bench_vlm.py --llm_path ../model/Qwen3-0.6B --freeze_llm 0 --batch_size 8
    python bench_vlm.py --llm_path ../model/Qwen3-1.7B --freeze_llm 2 --lora_rank 64 --grad_checkpoint 1 --batch_size 8
    # 从 batch_size=1 开始翻倍，直到显存不够
    python bench_vlm.py --llm_path ../model/Qwen3-0.6B --freeze_llm 0 --find_max_batch 1
"""
import os
import sys

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import time
import warnings
from contextlib import nullcontext

import torch

from model.model_vlm import VLMConfig
from trainer.trainer_utils import init_vlm_model

warnings.filterwarnings('ignore')


def make_batch(model, tokenizer, batch_size, seq_len, supervised_ratio, device):
    """随机 token + 一张图（64 个 <|image_pad|>），后 supervised_ratio 部分算 loss，和真实 SFT 样本的结构一致"""
    cfg = model.config
    high = min(len(tokenizer), cfg.vocab_size) - 500  # 避开词表末尾的特殊 token
    input_ids = torch.randint(100, max(high, 200), (batch_size, seq_len))
    input_ids[:, 1:1 + cfg.image_token_len] = cfg.image_ids[0]
    labels = input_ids.clone()
    labels[:, :int(seq_len * (1 - supervised_ratio))] = -100
    size = model.processor.size
    height, width = (size['height'], size['width']) if 'height' in size else (size['shortest_edge'],) * 2
    pixel_values = {'pixel_values': torch.randn(batch_size, 1, 3, height, width)}
    return input_ids.to(device), labels.to(device), {k: v.to(device) for k, v in pixel_values.items()}


def run(model, optimizer, scaler, autocast_ctx, batch, steps, warmup, device_type):
    input_ids, labels, pixel_values = batch
    if device_type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(warmup + steps):
        if device_type == 'cuda': torch.cuda.synchronize()
        start = time.time()
        with autocast_ctx:
            res = model(input_ids, labels=labels, pixel_values=pixel_values)
            loss = res.loss + res.aux_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if device_type == 'cuda': torch.cuda.synchronize()
        if i >= warmup: times.append(time.time() - start)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device_type == 'cuda' else float('nan')
    return sum(times) / len(times), peak


def main():
    parser = argparse.ArgumentParser(description="MiniMind-V 显存/速度测试")
    parser.add_argument('--llm_path', default=None, type=str, help="HF 主干路径（如 ../model/Qwen3-0.6B）；不传则用 MiniMind 主干")
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument('--vision_model', default='../model/siglip2-base-p32-256-ve', type=str)
    parser.add_argument('--freeze_llm', default=1, type=int, choices=[0, 1, 2])
    parser.add_argument('--lora_rank', default=0, type=int)
    parser.add_argument('--grad_checkpoint', default=0, type=int, choices=[0, 1])
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--seq_len', default=768, type=int, help="每条样本的长度（相当于 batch 内最长样本的长度）")
    parser.add_argument('--supervised_ratio', default=0.5, type=float, help="算 loss 的 token 比例")
    parser.add_argument('--steps', default=10, type=int)
    parser.add_argument('--warmup', default=3, type=int)
    parser.add_argument('--dtype', default='bfloat16', type=str)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--find_max_batch', default=0, type=int, choices=[0, 1], help="从 batch_size=1 开始翻倍测试，直到显存不够")
    parser.add_argument('--num_gpus', default=8, type=int, help="估算 epoch 耗时用：卡数")
    parser.add_argument('--num_samples', default=2_900_000, type=int, help="估算 epoch 耗时用：样本数（sft_i2t 约 290 万）")
    parser.add_argument('--avg_tokens', default=330, type=int, help="估算 epoch 耗时用：平均每条样本的 token 数（动态 padding 后）")
    args = parser.parse_args()

    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    autocast_ctx = torch.amp.autocast(device_type, dtype=dtype) if device_type == 'cuda' else nullcontext()
    vlm_config = VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    model, tokenizer, _ = init_vlm_model(vlm_config, from_weight='llm' if args.llm_path else 'none', vision_model_path=args.vision_model,
                                         device=args.device, freeze_llm=args.freeze_llm, llm_path=args.llm_path, lora_rank=args.lora_rank,
                                         grad_checkpoint=bool(args.grad_checkpoint), frozen_dtype=dtype if (args.llm_path and device_type == 'cuda') else None)
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-6)
    scaler = torch.amp.GradScaler(device_type, enabled=(args.dtype == 'float16'))
    if device_type == 'cuda':
        print(f'模型和优化器加载后已占用显存: {torch.cuda.memory_allocated() / 2 ** 30:.2f} GB / {torch.cuda.get_device_properties(args.device).total_memory / 2 ** 30:.1f} GB')

    batch_sizes = [2 ** i for i in range(10)] if args.find_max_batch else [args.batch_size]
    print(f"\n{'batch':>6} {'seq':>5} {'step(s)':>8} {'tokens/s':>10} {'peak_mem(GB)':>13} {'est_epoch(h)':>13}")
    for bs in batch_sizes:
        try:
            batch = make_batch(model, tokenizer, bs, args.seq_len, args.supervised_ratio, args.device)
            step_time, peak = run(model, optimizer, scaler, autocast_ctx, batch, args.steps, args.warmup, device_type)
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            print(f'{bs:>6} {args.seq_len:>5}  显存不足（OOM）')
            break
        tokens_per_s = bs * args.seq_len / step_time
        epoch_hours = args.num_samples * args.avg_tokens / (tokens_per_s * args.num_gpus * 0.9) / 3600  # 多卡按 90% 线性加速估算
        print(f'{bs:>6} {args.seq_len:>5} {step_time:>8.3f} {tokens_per_s:>10.0f} {peak:>13.2f} {epoch_hours:>13.1f}')
    print(f'\n估算条件：{args.num_gpus} 张卡、{args.num_samples} 条样本 × 平均 {args.avg_tokens} token；实际耗时以训练日志里的 epoch_time 为准')


if __name__ == '__main__':
    main()
