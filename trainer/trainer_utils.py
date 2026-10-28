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
from model.model_vlm_hf import HFVLM, lora_rank_in


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

    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    else:  # 没有 GPU 时用 gloo 后端，可以在 CPU 上调试多卡逻辑（torchrun --nproc_per_node 2 ... --device cpu）
        dist.init_process_group(backend="gloo")
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


def weight_tag(config):
    """权重文件名中的模型标记：MiniMind 主干为 {hidden_size}[_moe]（如 768、768_moe），HF 主干为模型目录名（如 qwen3-0.6b）"""
    return getattr(config, 'llm_tag', None) or f"{config.hidden_size}{'_moe' if config.use_moe else ''}"


def resolve_weight_path(save_dir, from_weight, config):
    """--from_weight 既可以是前缀名（pretrain_vlm → {save_dir}/pretrain_vlm_{tag}.pth），也可以直接给 .pth 路径"""
    if from_weight.endswith('.pth') or os.path.isfile(from_weight):
        return from_weight
    return f'{save_dir}/{from_weight}_{weight_tag(config)}.pth'


def get_vlm_state_dict(model):
    """需要保存的参数，不含冻结的 vision_encoder（可以从磁盘重新加载，没必要每次都存一遍，~95M 参数）。
    HF 主干（Qwen 等）只保存与基座模型不同的部分，见 HFVLM.vlm_state_dict。"""
    raw = unwrap_model(model)
    if hasattr(raw, 'vlm_state_dict'):
        return raw.vlm_state_dict()
    return {k: v for k, v in raw.state_dict().items() if not k.startswith('vision_encoder.')}


def save_weights(model, path):
    """半精度保存模型权重（不含 vision_encoder）；先写临时文件再原子替换，防止保存中断导致权重损坏。
    MiniMind 主干沿用 fp16；HF 主干本身是 bf16 训练的，用 bf16 保存避免很小的权重在 fp16 下下溢。"""
    dtype = torch.bfloat16 if isinstance(unwrap_model(model), HFVLM) else torch.float16
    state_dict = {k: v.to(dtype).cpu() for k, v in get_vlm_state_dict(model).items()}
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
    is_hf = isinstance(model, HFVLM)
    if is_hf and lora_rank_in(weights):  # 权重里有 LoRA：先挂上同样秩的 LoRA 再加载
        model.apply_lora(lora_rank_in(weights))
    missing, unexpected = model.load_state_dict(weights, strict=False)
    missing = [k for k in missing if not k.startswith('vision_encoder.')]  # 视觉编码器单独从磁盘加载
    if is_hf:  # HF 主干的权重只含与基座不同的部分，其余参数来自 llm_path，不算缺失
        model.loaded_keys |= set(weights) - set(unexpected)
        missing = [k for k in missing if k.startswith('vision_proj.')]
    def preview(keys): return f"{keys[:6]}{' ...' if len(keys) > 6 else ''}"
    if missing and all(k.startswith('vision_proj.') for k in missing):
        Logger(f'{weight_path} 中没有 vision_proj 参数，Projector 将随机初始化（从纯 LLM 权重开始训练时属正常）')
    elif missing:
        Logger(f'[警告] {weight_path} 缺少 {len(missing)} 个参数，它们将保持随机初始化: {preview(missing)}')
    if unexpected:
        Logger(f'[警告] {weight_path} 中有 {len(unexpected)} 个参数未被模型使用: {preview(unexpected)}')
    return model


def cast_frozen_params(model, dtype):
    """冻结的参数用 bf16/fp16 存（显存减半；autocast 下矩阵乘本来就以半精度计算），可训练参数保持 fp32（优化器更新更稳定）"""
    for param in model.parameters():
        param.data = param.data.to(torch.float32 if param.requires_grad else dtype)
    return model


def init_vlm_model(vlm_config, from_weight='pretrain_vlm', tokenizer_path='../model', vision_model_path='../model/siglip2-base-p32-256-ve', save_dir='../out', device='cuda', freeze_llm=0,
                   llm_path=None, lora_rank=0, grad_checkpoint=False, frozen_dtype=None):
    """llm_path 为空时用 MiniMind 主干（vlm_config 决定大小）；否则用 llm_path 下的 HF 模型（Qwen 等）做主干。
    frozen_dtype：冻结参数的存储精度（GPU 上训练 HF 主干时传 bf16），None 表示全部 fp32。"""
    if lora_rank and not llm_path:
        raise ValueError('LoRA 用于 --llm_path 指定的大主干；MiniMind 主干参数量小，请直接用 --freeze_llm 0/1')
    if llm_path:
        model = HFVLM(llm_path, vision_model_path=vision_model_path, image_special_token=vlm_config.image_special_token,
                      dtype=frozen_dtype or torch.float32)
        tokenizer = model.tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = MiniMindVLM(vlm_config, vision_model_path=vision_model_path)
    if model.vision_encoder is None:
        raise FileNotFoundError(f'未找到视觉编码器: {os.path.abspath(vision_model_path)}，'
                                f'请先下载 siglip2-base-p32-256-ve（见 README「第0步」或 model/model_init.md）')

    # HF 主干的 'llm' 就是 llm_path 里的基座模型本身，已经加载过了
    if from_weight != 'none' and not (llm_path and from_weight == 'llm'):
        load_vlm_weights(model, resolve_weight_path(save_dir, from_weight, model.config), device=device)

    # 1、全部冻结，只打开vision_proj梯度
    for name, param in model.named_parameters():
        if 'vision_proj' not in name:
            param.requires_grad = False

    # 2、判断策略
    llm = model.llm if llm_path else model.model
    if freeze_llm == 0:
        for name, param in model.named_parameters():
            if 'vision_encoder' not in name:
                param.requires_grad = True
    elif freeze_llm == 1:
        last_idx = model.config.num_hidden_layers - 1
        for name, param in llm.named_parameters():
            if 'layers.0.' in name or f'layers.{last_idx}.' in name:
                param.requires_grad = True
    elif freeze_llm == 2:
        pass

    # 3、LoRA：在语言模型的所有线性层上加低秩旁路，只训练旁路
    if lora_rank:
        Logger(f'LoRA rank={lora_rank}，作用于 {model.apply_lora(lora_rank)} 个线性层')
        for name, param in model.named_parameters():
            if '.lora.' in name:
                param.requires_grad = True

    if frozen_dtype is not None:
        cast_frozen_params(model, frozen_dtype)
    if grad_checkpoint:
        model.set_grad_checkpointing(True)

    get_model_params(model, model.config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    preprocess = model.processor
    # from_pretrained 加载的 HF 模型默认是 eval 模式，而 HF 的梯度检查点只在 train 模式下生效，这里统一切到 train 模式
    # （视觉编码器仍保持 eval，见 VisionTowerMixin.train）
    model.train()
    return model.to(device), tokenizer, preprocess


def vlm_checkpoint(vlm_config, weight='pretrain_vlm', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    ckp_path = f'{save_dir}/{weight}_{weight_tag(vlm_config)}.pth'
    resume_path = f'{save_dir}/{weight}_{weight_tag(vlm_config)}_resume.pth'
    
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


class LengthGroupedBatchSampler(Sampler):
    """按长度分组的 batch 采样器（配合 vlm_collate_fn 的动态 padding），支持 DDP 和断点续训。

    每个 epoch：随机打乱 → 每 mega batch（batch_size × 卡数 × mega_batch_mult 条）内按长度排序后切成全局 batch
    → 打乱全局 batch 的顺序 → 第 i 张卡取全局 batch 的第 i, i+卡数, ... 条。
    这样同一个 batch 内的样本长度接近（padding 少），同一步各张卡的 batch 长度也接近（不用互相等）。
    最长的那个 batch 固定放在第一个：显存不够会在开头就报错，而不是跑了几个小时才 OOM。
    """

    def __init__(self, lengths, batch_size, num_replicas=1, rank=0, seed=42, epoch=0, skip_batches=0, mega_batch_mult=50):
        self.lengths = lengths
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = epoch
        self.skip_batches = skip_batches
        self.mega_batch_mult = mega_batch_mult

    def _global_batches(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        global_bs = self.batch_size * self.num_replicas
        perm = torch.randperm(len(self.lengths), generator=g).tolist()
        while len(perm) % global_bs:  # 与 DistributedSampler 一样重复开头的样本补齐，保证每张卡的 batch 数相同
            perm += perm[:global_bs - len(perm) % global_bs]
        mega = global_bs * self.mega_batch_mult
        batches = []
        for i in range(0, len(perm), mega):
            chunk = sorted(perm[i:i + mega], key=lambda j: self.lengths[j], reverse=True)
            batches += [chunk[j:j + global_bs] for j in range(0, len(chunk), global_bs)]
        order = torch.randperm(len(batches), generator=g).tolist()
        longest = max(range(len(batches)), key=lambda b: self.lengths[batches[b][0]])
        order.remove(longest)
        return [batches[b] for b in [longest] + order]

    def __iter__(self):
        for i, global_batch in enumerate(self._global_batches()):
            if i >= self.skip_batches:
                yield global_batch[self.rank::self.num_replicas]

    def __len__(self):
        return max(0, math.ceil(len(self.lengths) / (self.batch_size * self.num_replicas)) - self.skip_batches)
