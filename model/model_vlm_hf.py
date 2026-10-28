"""
用现成的预训练语言模型（Qwen2.5 / Qwen3 等 ChatML 对话格式的 HF 模型）作为 MiniMind-V 的语言主干。

结构与 MiniMindVLM 相同（LLaVA 式）：冻结的 SigLIP 视觉编码器 → MLP 投影 → 替换 <|image_pad|> 占位符的 embedding，
区别只是语言模型换成了 AutoModelForCausalLM 加载的预训练模型。

保存的权重只包含「和基座模型不同」的部分（投影层、训练过的层、LoRA），加载时先从 llm_path 读基座，再叠加这部分权重。
"""
import math
import os
import re

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from .model_vlm import VLMConfig, MMVisionProjector, VisionTowerMixin, sync_vision_config

CHATML_START, CHATML_END = '<|im_start|>', '<|im_end|>'


def llm_tag(llm_path):
    """权重文件名里区分主干用的标记，例如 ../model/Qwen3-0.6B → qwen3-0.6b（得到 sft_vlm_qwen3-0.6b.pth）"""
    name = os.path.basename(os.path.normpath(llm_path))
    return re.sub(r'[^0-9a-zA-Z._-]+', '-', name).lower()


# ---------------------------------------------------------------- LoRA ----

class LoRA(nn.Module):
    """低秩旁路：y = W x + B(A(x))。B 初始化为 0，所以刚加上时模型输出完全不变。"""

    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.B(self.A(x))


def _lora_hook(module, inputs, output):
    return output + module.lora(inputs[0])


def apply_lora(root: nn.Module, rank: int):
    """给 root 下所有 nn.Linear 挂上 LoRA。用 forward hook 实现，原权重的参数名不变，新增参数为 *.lora.A/B.weight。"""
    targets = [m for n, m in root.named_modules()
               if isinstance(m, nn.Linear) and 'lora' not in n.split('.') and not hasattr(m, 'lora')]
    for m in targets:
        m.lora = LoRA(m.in_features, m.out_features, rank).to(m.weight.device)
        m.register_forward_hook(_lora_hook)
    return len(targets)


def lora_rank_in(state_dict):
    """从权重里推断 LoRA 的秩（没有 LoRA 返回 0）"""
    for k, v in state_dict.items():
        if k.endswith('.lora.A.weight'):
            return v.shape[0]
    return 0


# ---------------------------------------------------------------- loss ----

def _chunk_ce(h, targets, lm_head):
    return F.cross_entropy(lm_head(h).float(), targets, reduction='sum')


def supervised_lm_loss(hidden_states, labels, lm_head, chunk_size=None):
    """只在有监督的位置算 logits 和交叉熵，并分块重计算：峰值显存只有一个块的 logits。

    Qwen 的词表约 15 万：batch=8、seq=768 时，直接算完整 logits 再求交叉熵，
    bf16 logits + fp32 副本 + log_softmax + 反向梯度合计约 15GB，3090 上直接 OOM。
    这里只算回答部分（SFT 里通常不到一半），每块约 6400 万个 logit（峰值约 1GB），反向时重算该块。
    结果与在完整 logits 上做 mean 归约的交叉熵相同。
    """
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    targets = shift_labels[mask]
    if targets.numel() == 0:  # 整个 batch 没有监督 token：返回和计算图相连的 0，保证 DDP 下每个参数都有梯度
        return hidden_states.sum() * 0.0 + lm_head.weight.sum() * 0.0
    chunk_size = chunk_size or max(64, (1 << 26) // lm_head.out_features)
    h = hidden_states[:, :-1][mask]
    total = 0.0
    for i in range(0, targets.numel(), chunk_size):
        total = total + checkpoint(_chunk_ce, h[i:i + chunk_size], targets[i:i + chunk_size], lm_head, use_reentrant=False)
    return total / targets.numel()


# ---------------------------------------------------------------- model ----

class HFVLM(VisionTowerMixin, nn.Module):
    """视觉编码器 + 投影 + 任意 ChatML 格式的 HF CausalLM（Qwen2/2.5/3、SmolLM2、Yi 等）。"""

    def __init__(self, llm_path, vision_model_path='./model/siglip2-base-p32-256-ve',
                 image_special_token='<|image_pad|>', dtype=torch.float32):
        super().__init__()
        self.llm_path = llm_path
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        if CHATML_START not in (self.tokenizer.chat_template or ''):
            raise ValueError(f'{llm_path} 的对话模板不是 ChatML（<|im_start|>...<|im_end|>）格式。'
                             f'目前支持 Qwen2/Qwen2.5/Qwen3、SmolLM2、Yi 等 ChatML 模型（需要带对话模板的版本，如 *-Instruct）')
        self.llm = AutoModelForCausalLM.from_pretrained(llm_path, dtype=dtype)
        image_id = self._ensure_token(image_special_token)
        self.im_end_id = self.tokenizer.convert_tokens_to_ids(CHATML_END)
        llm_cfg = self.llm.config
        self.config = VLMConfig(hidden_size=llm_cfg.hidden_size, num_hidden_layers=llm_cfg.num_hidden_layers,
                                vocab_size=len(self.tokenizer), image_special_token=image_special_token, image_ids=[image_id],
                                llm_tag=llm_tag(llm_path), llm_path=llm_path)
        self.vision_encoder, self.processor = self.get_vision_model(vision_model_path)
        sync_vision_config(self.config, self.vision_encoder)
        self.vision_proj = MMVisionProjector(self.config.image_hidden_size, llm_cfg.hidden_size)
        self.loaded_keys = set()  # 从 VLM 权重中加载过的参数名，保存时要一并保存（见 vlm_state_dict）

    def _ensure_token(self, token):
        """Qwen 系列词表自带 <|image_pad|>；其他模型没有时新增一个 token 并扩充 embedding。
        这个 token 的 embedding 本身不会被用到（会被视觉特征替换），所以新增的那一行不需要训练和保存。"""
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == self.tokenizer.unk_token_id:
            self.tokenizer.add_tokens([token], special_tokens=True)
            token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id >= self.llm.get_input_embeddings().num_embeddings:
            self.llm.resize_token_embeddings(len(self.tokenizer))
        return token_id

    @property
    def device(self):
        return self.vision_proj.mlp[1].weight.device

    def apply_lora(self, rank):
        """在语言模型的所有解码层的线性层上加 LoRA，返回加了多少层"""
        return apply_lora(self.llm.base_model, rank)

    def set_grad_checkpointing(self, enable: bool = True):
        if enable:
            self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        else:
            self.llm.gradient_checkpointing_disable()
        return self

    def vlm_state_dict(self):
        """需要保存的参数：投影层 + 当前可训练的参数（含 LoRA）+ 之前从 VLM 权重里加载过的参数。
        其余参数和基座模型完全一致，加载时从 llm_path 读取即可，不重复保存。"""
        keep = {n for n, p in self.named_parameters() if p.requires_grad} | self.loaded_keys
        return {k: v for k, v in self.state_dict().items() if k.startswith('vision_proj.') or k in keep}

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0,
                labels=None, pixel_values=None, **kwargs):
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        if pixel_values is not None and past_key_values is None:
            inputs_embeds = self.merge_vision_features(input_ids, inputs_embeds, pixel_values)
        aux_loss = inputs_embeds.new_zeros((), dtype=torch.float32)
        if labels is None:
            out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, past_key_values=past_key_values,
                           use_cache=use_cache, logits_to_keep=logits_to_keep)
            return MoeCausalLMOutputWithPast(logits=out.logits, past_key_values=out.past_key_values, aux_loss=aux_loss)
        hidden_states = self.llm.base_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False).last_hidden_state
        loss = supervised_lm_loss(hidden_states, labels, self.llm.get_output_embeddings())
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss)

    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, pixel_values=None, **kwargs):
        """参数与 MiniMindVLM.generate 一致（max_new_tokens / temperature / top_p / streamer 等），返回 prompt + 新生成的 token"""
        input_ids = kwargs.pop('input_ids', inputs)
        inputs_embeds = self.merge_vision_features(input_ids, self.llm.get_input_embeddings()(input_ids), pixel_values)
        eos = kwargs.pop('eos_token_id', None)
        eos = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
        if self.im_end_id not in eos:  # 基座模型的 eos 通常是 <|endoftext|>，对话结束符是 <|im_end|>
            eos.append(self.im_end_id)
        if kwargs.get('pad_token_id') is None:
            kwargs['pad_token_id'] = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos[0]
        return self.llm.generate(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                 eos_token_id=eos, **kwargs)
