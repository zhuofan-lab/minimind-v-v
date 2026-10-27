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


def write_parquet(path, samples):
    """samples: [(conversations, image_bytes), ...]，格式与官方 pretrain/sft parquet 一致。"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({
        'conversations': [json.dumps(conv, ensure_ascii=False) for conv, _ in samples],
        'image_bytes': [img for _, img in samples],
    })
    pq.write_table(table, path)
    return str(path)


@pytest.fixture(autouse=True)
def _isolated_hf_datasets_cache(tmp_path, monkeypatch):
    # HFDataset.from_parquet 会把 parquet 转成 Arrow 缓存，测试时写到临时目录里
    import datasets
    monkeypatch.setattr(datasets.config, 'HF_DATASETS_CACHE', str(tmp_path / 'hf_cache'))
