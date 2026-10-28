"""端到端：用 torchrun 在 CPU 上起 2 个进程（gloo 后端）跑真实的训练脚本，再用 eval / bench 脚本跑一遍。

覆盖 DDP + 梯度累积（no_sync）+ 按长度分组采样 + 断点续训，以及 Qwen 主干 + LoRA + 梯度检查点。
每个用例十几秒，主要开销是起进程和 import。
"""
import os
import re
import shutil
import subprocess
import sys

import pytest
import torch
from PIL import Image

from conftest import REPO_ROOT, image_bytes, write_parquet

ENV = {**os.environ, 'OMP_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'}


@pytest.fixture
def workdir(tmp_path):
    """训练脚本用相对路径（../model 读 tokenizer，../checkpoints 写续训文件），在临时目录里搭一个同样的结构"""
    (tmp_path / 'trainer').mkdir()
    (tmp_path / 'model').mkdir()
    for name in ['tokenizer.json', 'tokenizer_config.json']:
        shutil.copy(os.path.join(REPO_ROOT, 'model', name), tmp_path / 'model' / name)
    return tmp_path


@pytest.fixture
def train_parquet(tmp_path):
    samples = []
    for i in range(14):
        answer = '图里有一只小狗在草地上奔跑。' * (1 + i % 5)
        conv = [{'role': 'user', 'content': '<image>\n描述这张图片。'}, {'role': 'assistant', 'content': answer}]
        if i % 4 == 0:
            conv += [{'role': 'user', 'content': '还有呢？'}, {'role': 'assistant', 'content': '天空是蓝色的。'}]
        samples.append((conv, image_bytes((i * 17 % 255, 100, 200), mode='L' if i % 3 == 0 else 'RGB')))
    return write_parquet(tmp_path / 'train.parquet', samples)


def run(cmd, cwd):
    proc = subprocess.run(cmd, cwd=cwd, env=ENV, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f'exit {proc.returncode}\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}'
    return proc.stdout


def torchrun(script, args, cwd, nproc=2):
    return run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', str(nproc),
                os.path.join(REPO_ROOT, 'trainer', script), '--device', 'cpu', '--num_workers', '0', '--log_interval', '1', *args], cwd)


def losses(stdout):
    values = [float(v) for v in re.findall(r'\), loss: ([0-9.naif]+),', stdout)]
    assert values and all(v == v and v != float('inf') for v in values), stdout[-2000:]
    return values


def test_ddp_minimind_sft_with_accumulation_grouping_and_resume(workdir, train_parquet, tiny_vision_path):
    common = ['--hidden_size', '64', '--num_hidden_layers', '2', '--from_weight', 'none', '--vision_model', tiny_vision_path,
              '--data_path', train_parquet, '--save_dir', str(workdir / 'out'), '--batch_size', '2', '--accumulation_steps', '2',
              '--max_seq_len', '256', '--learning_rate', '1e-3', '--group_by_length', '1', '--save_interval', '1000']
    first = torchrun('train_sft_vlm.py', common + ['--epochs', '1'], workdir / 'trainer')
    assert len(losses(first)) == 4  # 14 条样本 / (2 卡 × batch 2) → 每张卡 4 个 batch
    assert (workdir / 'out' / 'sft_vlm_64.pth').exists()
    assert (workdir / 'checkpoints' / 'sft_vlm_64_resume.pth').exists()

    resumed = torchrun('train_sft_vlm.py', common + ['--epochs', '2', '--from_resume', '1'], workdir / 'trainer')
    assert '跳过前4个step' in resumed
    assert re.search(r'Epoch:\[2/2\]\(4/4\)', resumed) and not re.search(r'Epoch:\[1/2\]\(\d/4\)', resumed)


def test_ddp_qwen_backbone_lora_pretrain(workdir, train_parquet, tiny_vision_path, tiny_qwen3_path):
    out = torchrun('train_pretrain_vlm.py', ['--llm_path', tiny_qwen3_path, '--freeze_llm', '2', '--lora_rank', '4', '--grad_checkpoint', '1',
                                             '--vision_model', tiny_vision_path, '--data_path', train_parquet, '--save_dir', str(workdir / 'out'),
                                             '--batch_size', '2', '--epochs', '1', '--max_seq_len', '256', '--group_by_length', '1'], workdir / 'trainer')
    losses(out)
    assert 'LoRA rank=4' in out
    weights = torch.load(workdir / 'out' / 'pretrain_vlm_qwen3-tiny.pth')
    assert {k.split('.')[0] for k in weights} == {'vision_proj', 'llm'}
    assert all(k.startswith('vision_proj.') or '.lora.' in k for k in weights)  # 冻结的基座参数不重复保存


def test_eval_script_with_qwen_backbone(tmp_path, tiny_vision_path, tiny_qwen3_path):
    from model.model_vlm_hf import HFVLM
    from trainer.trainer_utils import save_weights
    torch.manual_seed(0)
    weight = str(tmp_path / 'sft_vlm_qwen3-tiny.pth')
    save_weights(HFVLM(tiny_qwen3_path, vision_model_path=tiny_vision_path), weight)
    image = str(tmp_path / 'cat.png')
    Image.new('RGB', (90, 70), (200, 120, 40)).save(image)
    out = run([sys.executable, 'eval_vlm.py', '--llm_path', tiny_qwen3_path, '--weight', weight, '--vision_model', tiny_vision_path,
               '--image', image, '--prompt', '图里是什么？', '--max_new_tokens', '6', '--device', 'cpu'], REPO_ROOT)
    assert '[图像]: cat.png' in out and '[Speed]' in out


@pytest.mark.parametrize('backbone', ['minimind', 'qwen'])
def test_bench_script_runs(backbone, tiny_vision_path, tiny_qwen3_path):
    args = ['--hidden_size', '64', '--num_hidden_layers', '2'] if backbone == 'minimind' else ['--llm_path', tiny_qwen3_path, '--freeze_llm', '2', '--lora_rank', '4']
    out = run([sys.executable, os.path.join(REPO_ROOT, 'scripts', 'bench_vlm.py'), '--device', 'cpu', '--vision_model', tiny_vision_path,
               '--batch_size', '2', '--seq_len', '128', '--steps', '1', '--warmup', '0', *args], os.path.join(REPO_ROOT, 'scripts'))
    assert re.search(r'^\s+2\s+128\s+[0-9.]+\s+\d+', out, re.M), out
