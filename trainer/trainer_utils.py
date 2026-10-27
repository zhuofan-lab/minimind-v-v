"""
训练工具函数集合
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from model.model_vlm import MiniMindVLM


def get_model_params(model, config, ignore_patterns=['vision_encoder']):
    def should_count(n): return not any(p in n for p in ignore_patterns)
    total = sum(p.numel() for n, p in model.named_parameters() if should_count(n)) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n and should_count(n)) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n and should_count(n)) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式
    
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def unwrap_model(model):
    """去掉 DDP / torch.compile 的包装，拿到原始模型。"""
    model = model.module if isinstance(model, DistributedDataParallel) else model
    return getattr(model, '_orig_mod', model)


def get_vlm_state_dict(model):
    """不含 vision_encoder 的 state_dict：视觉编码器全程冻结、可以从磁盘重新加载，没必要每次都存一遍（~95M 参数）。"""
    return {k: v for k, v in unwrap_model(model).state_dict().items() if not k.startswith('vision_encoder.')}


def save_weights(model, path):
    """半精度保存模型权重（不含 vision_encoder）；先写临时文件再原子替换，防止保存中断导致权重损坏。"""
    state_dict = {k: v.half().cpu() for k, v in get_vlm_state_dict(model).items()}
    tmp_path = path + '.tmp'
    torch.save(state_dict, tmp_path)
    os.replace(tmp_path, path)


def load_vlm_weights(model, weight_path, device='cpu'):
    """加载权重并报告缺失/多余的参数。

    strict=False 会静默跳过对不上的参数：权重文件选错、dense/MoE 配置不一致时，
    模型会带着随机初始化的层继续训练或推理而没有任何提示。这里把它们打印出来。
    """
    weights = torch.load(weight_path, map_location=device)
    weights = {k: v for k, v in weights.items() if 'mask' not in k}  # 兼容旧版权重里的 attention mask buffer
    missing, unexpected = model.load_state_dict(weights, strict=False)
    missing = [k for k in missing if not k.startswith('vision_encoder.')]  # 视觉编码器单独从磁盘加载
    def preview(keys): return f"{keys[:6]}{' ...' if len(keys) > 6 else ''}"
    if missing and all(k.startswith('vision_proj.') for k in missing):
        Logger(f'{weight_path} 中没有 vision_proj 参数，Projector 将随机初始化（从纯 LLM 权重开始训练时属正常）')
    elif missing:
        Logger(f'[警告] {weight_path} 缺少 {len(missing)} 个参数，它们将保持随机初始化: {preview(missing)}')
    if unexpected:
        Logger(f'[警告] {weight_path} 中有 {len(unexpected)} 个参数未被模型使用: {preview(unexpected)}')
    return model


def init_vlm_model(vlm_config, from_weight='pretrain_vlm', tokenizer_path='../model', vision_model_path='../model/siglip2-base-p32-256-ve', save_dir='../out', device='cuda', freeze_llm=0):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindVLM(vlm_config, vision_model_path=vision_model_path)
    if model.vision_encoder is None:
        raise FileNotFoundError(f'未找到视觉编码器: {os.path.abspath(vision_model_path)}，'
                                f'请先下载 siglip2-base-p32-256-ve（见 README「第0步」或 model/model_init.md）')

    if from_weight != 'none':
        moe_suffix = '_moe' if vlm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{vlm_config.hidden_size}{moe_suffix}.pth'
        load_vlm_weights(model, weight_path, device=device)

    # 1、全部冻结，只打开vision_proj梯度
    for name, param in model.named_parameters():
        if 'vision_proj' not in name:
            param.requires_grad = False

    # 2、判断策略
    if freeze_llm == 0:
        for name, param in model.named_parameters():
            if 'vision_encoder' not in name:
                param.requires_grad = True
    elif freeze_llm == 1:
        last_idx = vlm_config.num_hidden_layers - 1
        for name, param in model.model.named_parameters():
            if 'layers.0.' in name or f'layers.{last_idx}.' in name:
                param.requires_grad = True
    elif freeze_llm == 2:
        pass

    get_model_params(model, vlm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    preprocess = model.processor
    return model.to(device), tokenizer, preprocess


def vlm_checkpoint(vlm_config, weight='pretrain_vlm', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if vlm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{vlm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{vlm_config.hidden_size}{moe_path}_resume.pth'
    
    if model is not None:
        # 移除vision_encoder参数（不需要保存，因为是预训练且冻结的，续训时会从磁盘重新加载）
        state_dict = get_vlm_state_dict(model)
        save_weights(model, ckp_path)

        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)
        
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value
        
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def vlm_collate_fn(batch, pad_multiple=64):
    input_ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    # 动态 padding：Dataset 把每条样本都补齐到 max_seq_len，但 batch 内最后一个监督 token 之后的列对 loss 毫无贡献
    # （因果注意力下前面的位置看不到后面的 token），裁掉后 loss 完全不变，只是省掉了这部分无效计算。
    # 长度向上取整到 pad_multiple 的倍数，限制不同 shape 的数量，对 torch.compile 和 kernel 选择更友好。
    supervised = (labels != -100).any(dim=0).nonzero()
    if len(supervised) > 0:
        keep = min(labels.size(1), (int(supervised[-1]) + pad_multiple) // pad_multiple * pad_multiple)
        input_ids, labels = input_ids[:, :keep].contiguous(), labels[:, :keep].contiguous()
    pixel_data = [b[2] for b in batch]
    if hasattr(pixel_data[0], 'keys'):
        # 允许 batch 内每条样本的图片数不同（多图、单图样本混合）：不足的用全 0 图补齐。
        # 补出来的图在文本里没有对应的 <|image_pad|>，count_vision_proj 不会用到它们
        num_images = max(next(iter(d.values())).size(0) for d in pixel_data)
        pad = lambda x: x if x.size(0) == num_images else torch.cat([x, x.new_zeros((num_images - x.size(0), *x.shape[1:]))])
        pixel_values = {k: torch.stack([pad(d[k]) for d in pixel_data]) for k in pixel_data[0].keys()}
    else:
        pixel_values = torch.stack(pixel_data)
    return input_ids, labels, pixel_values


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches
    
    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch
    
    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)

