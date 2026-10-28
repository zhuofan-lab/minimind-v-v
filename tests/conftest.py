"""
测试公共夹具：用一个随机初始化的迷你 SigLIP（64 个 patch token，与正式模型的 token 数一致）
和迷你 LLM 配置，在 CPU 上几秒内跑完整条 VLM 流程，不需要下载任何权重或数据集。
"""
import io
import json
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from PIL import Image
from transformers import AutoTokenizer, SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel

from model.model_vlm import MiniMindVLM, VLMConfig

IMAGE_SIZE, PATCH_SIZE, VISION_HIDDEN = 64, 8, 32  # (64/8)^2 = 64 个视觉 token


@pytest.fixture(scope='session')
def tiny_vision_path(tmp_path_factory):
    path = tmp_path_factory.mktemp('tiny_siglip')
    torch.manual_seed(0)
    cfg = SiglipVisionConfig(hidden_size=VISION_HIDDEN, intermediate_size=64, num_hidden_layers=1,
                             num_attention_heads=2, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE)
    SiglipVisionModel(cfg).save_pretrained(path)
    SiglipImageProcessor(size={'height': IMAGE_SIZE, 'width': IMAGE_SIZE}).save_pretrained(path)
    return str(path)


@pytest.fixture(scope='session')
def tokenizer():
    return AutoTokenizer.from_pretrained(os.path.join(REPO_ROOT, 'model'))


def make_config(**overrides):
    kwargs = dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                  image_hidden_size=VISION_HIDDEN, max_position_embeddings=1024)
    kwargs.update(overrides)
    return VLMConfig(**kwargs)


@pytest.fixture
def tiny_config():
    return make_config()


@pytest.fixture
def tiny_model(tiny_vision_path):
    torch.manual_seed(0)
    return MiniMindVLM(make_config(), vision_model_path=tiny_vision_path)


def image_bytes(color=(200, 30, 30), mode='RGB', size=(80, 60), fmt='JPEG'):
    buf = io.BytesIO()
    Image.new('RGB', size, color).convert(mode).save(buf, format=fmt)
    return buf.getvalue()


def write_parquet(path, samples, ensure_ascii=False):
    """samples: [(conversations, image_bytes), ...]，格式与官方 pretrain/sft parquet 一致。"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({
        'conversations': [json.dumps(conv, ensure_ascii=ensure_ascii) for conv, _ in samples],
        'image_bytes': [img for _, img in samples],
    })
    pq.write_table(table, path)
    return str(path)


QWEN25_LIKE_TEMPLATE = (
    "{%- if messages[0]['role'] == 'system' %}{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
    "{%- else %}{{- '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}{%- endif %}"
    "{%- for message in messages %}{%- if not (message.role == 'system' and loop.first) %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)


def save_tiny_llm(path, arch='qwen3', template='minimind'):
    """随机初始化的迷你 Qwen3/Qwen2 + ChatML tokenizer，模拟 ../model/Qwen3-0.6B 这样的 HF 模型目录。

    template='qwen25' 时模拟 Qwen2.5 tokenizer 的特点：bos_token 为 None、eos_token 为 <|endoftext|>（基座模型），
    并且没有 system 消息时模板会自动加默认的 system prompt。
    """
    from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM
    tok = AutoTokenizer.from_pretrained(os.path.join(REPO_ROOT, 'model'))
    if template == 'qwen25':
        tok.bos_token = None
        tok.eos_token = '<|endoftext|>'
        tok.chat_template = QWEN25_LIKE_TEMPLATE
    elif template == 'llama3':
        tok.chat_template = "{% for m in messages %}<|start_header_id|>{{ m.role }}<|end_header_id|>\n\n{{ m.content }}<|eot_id|>{% endfor %}"
    common = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                  vocab_size=len(tok), tie_word_embeddings=True, max_position_embeddings=2048)
    torch.manual_seed(1)
    model = Qwen3ForCausalLM(Qwen3Config(head_dim=16, **common)) if arch == 'qwen3' else Qwen2ForCausalLM(Qwen2Config(**common))
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


@pytest.fixture(scope='session')
def tiny_qwen3_path(tmp_path_factory):
    return save_tiny_llm(tmp_path_factory.mktemp('llm') / 'Qwen3-Tiny', arch='qwen3', template='minimind')


@pytest.fixture(scope='session')
def tiny_qwen25_path(tmp_path_factory):
    return save_tiny_llm(tmp_path_factory.mktemp('llm') / 'Qwen2.5-Tiny-Base', arch='qwen2', template='qwen25')


@pytest.fixture(autouse=True)
def _isolated_hf_datasets_cache(tmp_path, monkeypatch):
    # HFDataset.from_parquet 会把 parquet 转成 Arrow 缓存，测试时写到临时目录里
    import datasets
    monkeypatch.setattr(datasets.config, 'HF_DATASETS_CACHE', str(tmp_path / 'hf_cache'))
