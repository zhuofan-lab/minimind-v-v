import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import datasets
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from model.model_vlm import MiniMindVLM, VLMConfig
from model.model_vlm_hf import llm_tag
from dataset.lm_dataset import VLMDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, init_distributed_mode, setup_seed, init_vlm_model, vlm_checkpoint, SkipBatchSampler, LengthGroupedBatchSampler, vlm_collate_fn, save_weights, weight_tag

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    for step, (input_ids, labels, pixel_values) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        pixel_values = {k: v.to(args.device) for k, v in pixel_values.items()} if isinstance(pixel_values, dict) else pixel_values.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        is_update_step = step % args.accumulation_steps == 0 or step == iters
        # 梯度累积的中间步不做跨卡同步（DDP 默认每次 backward 都 all-reduce 一遍梯度），只在更新参数的那一步同步一次
        sync_ctx = model.no_sync() if isinstance(model, DistributedDataParallel) and not is_update_step else nullcontext()
        with sync_ctx:
            with autocast_ctx:
                res = model(input_ids, labels=labels, pixel_values=pixel_values)
                loss = res.loss + res.aux_loss
                loss = loss / args.accumulation_steps

            scaler.scale(loss).backward()

        if is_update_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            ckp = f'{args.save_dir}/{args.save_weight}_{weight_tag(vlm_config)}.pth'
            save_weights(model, ckp)  # 半精度、去掉冻结的vision_encoder、临时文件+原子替换
            vlm_checkpoint(vlm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            model.train()

        del input_ids, labels, pixel_values, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-V Pretrain")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain_vlm', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=2, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=450, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_i2t.parquet", help="训练数据路径")
    parser.add_argument('--from_weight', default='llm', type=str, help="基于哪个权重训练：前缀名（在../out下找 {前缀}_{hidden_size或主干名}.pth）或 .pth 路径；为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--freeze_llm', default=2, type=int, choices=[0, 1, 2], help="冻结策略（0=完全可训练，1=冻结+解冻首尾层，2=完全冻结仅训练proj）")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument('--llm_path', default=None, type=str, help="用HF格式的预训练LLM做主干（如 ../model/Qwen3-0.6B），此时忽略hidden_size/num_hidden_layers/use_moe")
    parser.add_argument('--vision_model', default='../model/siglip2-base-p32-256-ve', type=str, help="视觉编码器路径")
    parser.add_argument('--lora_rank', default=0, type=int, help="LoRA的秩（0=不用；仅用于--llm_path主干，常配合--freeze_llm 2）")
    parser.add_argument('--grad_checkpoint', default=0, type=int, choices=[0, 1], help="梯度检查点：激活显存大幅下降，计算量增加约1/3（0=否，1=是）")
    parser.add_argument('--group_by_length', default=0, type=int, choices=[0, 1], help="按长度分组采样，减少padding（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-V-Pretrain", help="wandb项目名")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized() and torch.cuda.is_available(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    vlm_config = VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, max_seq_len=args.max_seq_len, use_moe=bool(args.use_moe),
                           **({'llm_tag': llm_tag(args.llm_path)} if args.llm_path else {}))
    ckp_data = vlm_checkpoint(vlm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type, dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-V-Pretrain-{weight_tag(vlm_config)}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer, preprocess = init_vlm_model(vlm_config, from_weight=args.from_weight, vision_model_path=args.vision_model, device=args.device, freeze_llm=args.freeze_llm,
                                                  llm_path=args.llm_path, lora_rank=args.lora_rank, grad_checkpoint=bool(args.grad_checkpoint),
                                                  frozen_dtype=dtype if (args.llm_path and device_type == 'cuda') else None)
    vlm_config = model.config  # 视觉 token 数等由实际加载的视觉编码器决定；HF 主干的配置来自 llm_path
    train_ds = VLMDataset(args.data_path, tokenizer, preprocess=preprocess, image_special_token=vlm_config.image_special_token, image_token_len=vlm_config.image_token_len, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler(device_type, enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        # broadcast_buffers=False：模型里的 buffer（RoPE 频率、position_ids）都是确定的常量，不需要每步从 rank0 广播
        # gradient_as_bucket_view=True：梯度直接放在 DDP 的通信 bucket 里，省掉一份和可训练参数同样大小的梯度副本
        model = DistributedDataParallel(model, device_ids=[local_rank] if device_type == 'cuda' else None,
                                        broadcast_buffers=False, gradient_as_bucket_view=True)
    
    # ========== 8. 开始训练 ==========
    lengths = train_ds.estimate_lengths() if args.group_by_length else None
    for epoch in range(start_epoch, args.epochs):
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        if args.group_by_length:
            setup_seed(42 + epoch)
            batch_sampler = LengthGroupedBatchSampler(lengths, args.batch_size, dist.get_world_size() if dist.is_initialized() else 1,
                                                      dist.get_rank() if dist.is_initialized() else 0, seed=42, epoch=epoch, skip_batches=skip)
        else:
            train_sampler and train_sampler.set_epoch(epoch)
            setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
            batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=vlm_collate_fn)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()
