import os
import torch
import warnings
from .model_minimind import *
from typing import Optional, Tuple, List, Union
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import SiglipImageProcessor, SiglipVisionModel
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

warnings.filterwarnings('ignore')


class VLMConfig(MiniMindConfig):
    model_type = "minimind-v"

    def __init__(self, image_special_token='<|image_pad|>', image_ids=[12], **kwargs):
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        self.image_hidden_size = kwargs.get("image_hidden_size", 768)
        self.image_token_len = kwargs.get("image_token_len", 64)
        super().__init__(**kwargs)

class MMVisionProjector(nn.Module):
    def __init__(self, in_dim, out_dim, source_tokens=64, target_tokens=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, x):
        return self.mlp(x)


def inject_vision_features(input_ids, inputs_embeds, vision_tensors, image_token_id):
    """用视觉特征替换图像占位符(<|image_pad|>)位置的 embedding。

    数据中每个 <image> 都被展开成 image_token_len 个连续占位符，所以样本内第 k 个占位符对应
    第 k // image_token_len 张图的第 k % image_token_len 个视觉 token。
    全程向量化：没有逐 token 的 Python 循环和 GPU→CPU 同步，序列长度保持不变，
    相邻的多张图、被截断在序列末尾的图都能正确对齐，也可以直接被 torch.compile 编译。
    """
    vf = vision_tensors if vision_tensors.dim() == 4 else vision_tensors.unsqueeze(1)  # [B, N_img, L_img, H]
    num_images, img_len = vf.shape[1], vf.shape[2]
    is_image = input_ids == image_token_id                                             # [B, S]
    k = is_image.long().cumsum(dim=1) - 1                                               # 占位符在本样本内的序号
    img_idx, tok_idx = torch.div(k, img_len, rounding_mode='floor'), k % img_len
    valid = is_image & (img_idx < num_images)                                           # 多于图片数量的占位符保持原 embedding
    batch_idx = torch.arange(vf.size(0), device=vf.device).unsqueeze(1)
    feats = vf[batch_idx, img_idx.clamp(0, num_images - 1), tok_idx]                    # [B, S, H]
    return torch.where(valid.unsqueeze(-1), feats.to(inputs_embeds.dtype), inputs_embeds)


def sync_vision_config(config, vision_encoder):
    """视觉 token 数和特征维度以实际加载的视觉编码器为准。

    默认的 siglip2-base-p32-256 是 768 维、(256/32)^2=64 个 token，与 VLMConfig 的默认值一致；
    换成 p16 或 so400m 等编码器时不需要再手动改配置。
    """
    if vision_encoder is not None:
        vc = vision_encoder.config
        config.image_hidden_size = vc.hidden_size
        config.image_token_len = (vc.image_size // vc.patch_size) ** 2
    return config


class VisionTowerMixin:
    """视觉部分（冻结的视觉编码器 + 投影 + 替换占位符），MiniMindVLM 和 HFVLM（Qwen 等主干）共用。"""

    @staticmethod
    def get_vision_model(model_path: str):
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        if not os.path.exists(model_path):
            return None, None
        try:
            model = SiglipVisionModel.from_pretrained(model_path)
        except (RuntimeError, ValueError):
            return None, None
        processor = SiglipImageProcessor.from_pretrained(model_path)
        # 冻结 vision_encoder 的所有参数
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processor

    def train(self, mode: bool = True):
        super().train(mode)
        # 视觉编码器全程冻结，model.train() 时也保持 eval 模式（不受 dropout 等训练态行为影响）
        if getattr(self, 'vision_encoder', None) is not None:
            self.vision_encoder.eval()
        return self

    @staticmethod
    def image2tensor(image, processor):
        # 灰度(L)/调色板(P)/CMYK/RGBA 等非 RGB 图像会让 SigLIP 预处理直接报错，统一转为 RGB
        if image.mode != 'RGB': image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        return inputs

    @staticmethod
    def get_image_embeddings(image_inputs, vision_model):
        if hasattr(image_inputs, 'keys'):
            image_inputs = {k: v.squeeze(1) if v.ndim > 2 and v.shape[1] == 1 else v for k, v in image_inputs.items()}
        with torch.no_grad():
            outputs = vision_model(**image_inputs)
        return outputs.last_hidden_state

    def encode_images(self, pixel_values):
        """图像 → 投影到 LLM 隐藏维度的视觉 token：[B, N_img, L_img, H]（单图输入时为 [B, L_img, H]）"""
        if hasattr(pixel_values, 'keys'):
            sample_val = next(iter(pixel_values.values()))
            if sample_val.ndim == 5:
                bs, num = sample_val.shape[:2]
                feats = self.get_image_embeddings({k: v.flatten(0, 1) for k, v in pixel_values.items()}, self.vision_encoder)
                return self.vision_proj(feats).view(bs, num, feats.size(1), -1)
            return self.vision_proj(self.get_image_embeddings(pixel_values, self.vision_encoder))
        if pixel_values.dim() == 6:
            pixel_values = pixel_values.squeeze(2)
        return torch.stack([self.vision_proj(self.get_image_embeddings(pixel_values[:, i], self.vision_encoder))
                            for i in range(pixel_values.size(1))], dim=1)

    def merge_vision_features(self, input_ids, inputs_embeds, pixel_values):
        if pixel_values is None or not self.config.image_ids:
            return inputs_embeds
        return inject_vision_features(input_ids, inputs_embeds, self.encode_images(pixel_values), self.config.image_ids[0])


# 继承自语言模型
class MiniMindVLM(VisionTowerMixin, MiniMindForCausalLM):
    config_class = VLMConfig

    def __init__(self, config: VLMConfig = None, vision_model_path="./model/siglip2-base-p32-256-ve"):
        self.config = config or VLMConfig()
        super().__init__(self.config)
        self.vision_encoder, self.processor = self.__class__.get_vision_model(vision_model_path)
        sync_vision_config(self.config, self.vision_encoder)
        self.vision_proj = MMVisionProjector(self.config.image_hidden_size, self.config.hidden_size, target_tokens=self.config.image_token_len)
        self.grad_checkpointing = False

    def set_grad_checkpointing(self, enable: bool = True):
        """梯度检查点：反向时重算每层的前向，激活显存从 O(层数) 降到约一层，代价是多约 1/3 的计算量。"""
        self.grad_checkpointing = enable
        return self

    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=None):
        """兼容旧接口（seqlen 参数不再使用），实现见 inject_vision_features"""
        if vision_tensors is None or not self.config.image_ids:
            return h
        return inject_vision_features(tokens, h, vision_tensors, self.config.image_ids[0])

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                labels: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.FloatTensor] = None,
                **args):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        if pixel_values is not None and start_pos == 0:
            hidden_states = self.merge_vision_features(input_ids, hidden_states, pixel_values)

        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        # 只在首次前向检查一次：该判断会触发 GPU→CPU 同步，并让 torch.compile 在此处断图
        if not getattr(self, '_rope_checked', False):
            if self.model.freqs_cos[0, 0] == 0:
                freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
                self.model.freqs_cos, self.model.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
            self._rope_checked = True
        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length]
        )

        use_ckpt = self.grad_checkpointing and self.training and torch.is_grad_enabled()
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            if use_ckpt:
                hidden_states, present = checkpoint(layer, hidden_states, position_embeddings, past_key_value, use_cache, attention_mask, use_reentrant=False)
            else:
                hidden_states, present = layer(
                    hidden_states,
                    position_embeddings,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    attention_mask=attention_mask
                )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)

        aux_loss = sum([l.mlp.aux_loss for l in self.model.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        aux_loss = aux_loss + sum(p.sum() for p in self.vision_proj.parameters()) * 0  # dummy gradient for DDP
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 等价于 mean 归约；区别是整个 batch 都没有监督 token（例如回答被 max_seq_len 整段截掉）时，
            # mean 会得到 0/0=NaN（梯度本身是 0，不影响权重，但日志和训练曲线里会出现 NaN），这里返回 0
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100, reduction='sum')
            loss = loss / (shift_labels != -100).sum().clamp(min=1)

        output = MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=presents, hidden_states=hidden_states)
        return output

    def generate(self, *args, num_return_sequences=1, **kwargs):
        if num_return_sequences > 1 and 'pixel_values' in kwargs:
            pv = kwargs['pixel_values']
            if hasattr(pv, 'keys'):
                kwargs['pixel_values'] = {k: v.repeat(num_return_sequences, *([1] * (v.ndim - 1))) for k, v in pv.items()}
            else:
                kwargs['pixel_values'] = pv.repeat(num_return_sequences, *([1] * (pv.ndim - 1)))
        return super().generate(*args, num_return_sequences=num_return_sequences, **kwargs)
