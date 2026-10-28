"""加大模型相关的测试：HF 主干（Qwen 等）、LoRA、增量权重、混合精度存储、梯度检查点、按长度分组采样。"""
import math
import random

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from conftest import image_bytes, make_config, save_tiny_llm, write_parquet
from dataset.lm_dataset import VLMDataset
from model.model_vlm import MiniMindVLM, VLMConfig
from model.model_vlm_hf import HFVLM, llm_tag, supervised_lm_loss
from trainer.trainer_utils import (LengthGroupedBatchSampler, init_vlm_model, load_vlm_weights, resolve_weight_path,
                                   save_weights, weight_tag)


def chat_inputs(model, tokenizer, question='描述这张图片', answer=None, color=(10, 200, 30)):
    cfg = model.config
    messages = [{'role': 'user', 'content': cfg.image_special_token * cfg.image_token_len + '\n' + question}]
    if answer is not None:
        messages.append({'role': 'assistant', 'content': answer})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=answer is None)
    enc = tokenizer(text, return_tensors='pt')
    pixel_values = dict(MiniMindVLM.image2tensor(Image.new('RGB', (80, 60), color), model.processor))
    return enc.input_ids, enc.attention_mask, pixel_values


@pytest.fixture
def hf_model(tiny_qwen3_path, tiny_vision_path):
    torch.manual_seed(0)
    return HFVLM(tiny_qwen3_path, vision_model_path=tiny_vision_path)


def init_hf(llm_path, vision_path, **kwargs):
    torch.manual_seed(0)
    return init_vlm_model(VLMConfig(), from_weight=kwargs.pop('from_weight', 'llm'), vision_model_path=vision_path,
                          device='cpu', llm_path=llm_path, **kwargs)


# ---------------------------------------------------------------- HF 主干 ----

def test_llm_tag_and_weight_naming(hf_model):
    assert llm_tag('../model/Qwen3-0.6B') == 'qwen3-0.6b'
    assert llm_tag('/data/Qwen2.5-1.5B-Instruct/') == 'qwen2.5-1.5b-instruct'
    assert weight_tag(hf_model.config) == 'qwen3-tiny'
    assert weight_tag(make_config()) == '64' and weight_tag(make_config(use_moe=True)) == '64_moe'
    assert resolve_weight_path('../out', 'sft_vlm', hf_model.config) == '../out/sft_vlm_qwen3-tiny.pth'
    assert resolve_weight_path('../out', '/abs/my_weights.pth', hf_model.config) == '/abs/my_weights.pth'


def test_hf_config_follows_llm_and_vision_encoder(hf_model):
    cfg = hf_model.config
    assert cfg.image_ids == [hf_model.tokenizer.convert_tokens_to_ids('<|image_pad|>')]
    assert (cfg.hidden_size, cfg.num_hidden_layers) == (64, 3)
    assert (cfg.image_token_len, cfg.image_hidden_size) == (64, 32)  # 来自迷你 SigLIP：(64/8)^2 个 token，32 维


def test_hf_loss_matches_full_logits_cross_entropy(hf_model):
    ids, _, pv = chat_inputs(hf_model, hf_model.tokenizer, answer='一片绿色的草地。')
    labels = ids.clone()
    labels[:, :70] = -100
    loss = hf_model(ids, labels=labels, pixel_values=pv).loss
    logits = hf_model(ids, pixel_values=pv).logits
    expected = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(), labels[:, 1:].reshape(-1), ignore_index=-100)
    torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-5)


def test_supervised_loss_chunking_is_exact():
    torch.manual_seed(0)
    lm_head = torch.nn.Linear(16, 50, bias=False)
    hidden = torch.randn(3, 40, 16, requires_grad=True)
    labels = torch.randint(0, 50, (3, 40))
    labels[:, :10] = -100
    labels[1, 25:] = -100
    reference = F.cross_entropy(lm_head(hidden[:, :-1]).reshape(-1, 50), labels[:, 1:].reshape(-1), ignore_index=-100)
    ref_grads = torch.autograd.grad(reference, [hidden, lm_head.weight])
    for chunk_size in [7, 10_000]:
        loss = supervised_lm_loss(hidden, labels, lm_head, chunk_size=chunk_size)
        torch.testing.assert_close(loss, reference)
        for got, want in zip(torch.autograd.grad(loss, [hidden, lm_head.weight]), ref_grads):
            torch.testing.assert_close(got, want)


def test_hf_loss_without_supervised_tokens_is_zero_and_connected(hf_model):
    ids, _, pv = chat_inputs(hf_model, hf_model.tokenizer)
    out = hf_model(ids, labels=torch.full_like(ids, -100), pixel_values=pv)
    assert out.loss.item() == 0.0
    out.loss.backward()  # 计算图仍然连通（DDP 要求每个可训练参数都有梯度）
    assert hf_model.llm.get_output_embeddings().weight.grad is not None


def test_hf_image_changes_the_prediction(hf_model):
    ids, _, pv_red = chat_inputs(hf_model, hf_model.tokenizer, color=(255, 0, 0))
    _, _, pv_blue = chat_inputs(hf_model, hf_model.tokenizer, color=(0, 0, 255))
    hf_model.eval()
    with torch.no_grad():
        assert not torch.allclose(hf_model(ids, pixel_values=pv_red).logits[:, -1], hf_model(ids, pixel_values=pv_blue).logits[:, -1])


def test_hf_generate_cache_consistency_and_eos(hf_model, monkeypatch):
    ids, mask, pv = chat_inputs(hf_model, hf_model.tokenizer)
    hf_model.eval()
    kwargs = dict(attention_mask=mask, pixel_values=pv, max_new_tokens=10, do_sample=False)
    with_cache, without_cache = hf_model.generate(ids, use_cache=True, **kwargs), hf_model.generate(ids, use_cache=False, **kwargs)
    assert torch.equal(with_cache, without_cache)
    assert torch.equal(with_cache[:, :ids.size(1)], ids) and with_cache.size(1) > ids.size(1)  # 和 MiniMindVLM 一样返回 prompt + 新 token

    captured = {}
    monkeypatch.setattr(hf_model.llm, 'generate', lambda **kw: captured.update(kw) or kw['input_ids'])
    endoftext = hf_model.tokenizer.convert_tokens_to_ids('<|endoftext|>')
    hf_model.generate(ids, attention_mask=mask, pixel_values=pv, eos_token_id=endoftext)
    assert captured['eos_token_id'] == [endoftext, hf_model.im_end_id]  # 基座模型的 eos 之外，自动加上对话结束符
    assert captured['inputs_embeds'].shape[:2] == ids.shape


def test_hf_dataset_labels_with_qwen25_style_tokenizer(tmp_path, tiny_qwen25_path, tiny_vision_path):
    # Qwen2.5 的 tokenizer：bos_token=None、基座版本 eos=<|endoftext|>、没有 system 时自动加默认 system prompt。
    # 原来用 f'{bos_token}assistant\n' 定位回答，会变成 'Noneassistant\n'，所有样本都没有监督 token
    model = HFVLM(tiny_qwen25_path, vision_model_path=tiny_vision_path)
    tok = model.tokenizer
    assert tok.bos_token is None and tok.eos_token == '<|endoftext|>'
    samples = [([{'role': 'user', 'content': '<image>\n图里是什么？'}, {'role': 'assistant', 'content': '一只红色的气球。'},
                 {'role': 'user', 'content': 'What else?'}, {'role': 'assistant', 'content': 'A blue sky.'}], image_bytes())]
    random.seed(0)
    ds = VLMDataset(write_parquet(tmp_path / 'd.parquet', samples), tok, preprocess=model.processor, max_length=256,
                    image_special_token=model.config.image_special_token, image_token_len=model.config.image_token_len)
    input_ids, labels, _ = ds[0]
    assert int((input_ids == model.config.image_ids[0]).sum()) == 64
    supervised = tok.decode(input_ids[labels != -100])
    assert '一只红色的气球。' in supervised and 'A blue sky.' in supervised
    assert supervised.count('<|im_end|>') == 2
    assert all(text not in supervised for text in ['图里是什么', 'What else', 'helpful assistant', 'system', 'user'])


def test_hf_adds_missing_image_token(tiny_qwen3_path, tiny_vision_path):
    model = HFVLM(tiny_qwen3_path, vision_model_path=tiny_vision_path, image_special_token='<|my_image|>')
    token_id = model.config.image_ids[0]
    assert token_id == model.tokenizer.convert_tokens_to_ids('<|my_image|>') == 6400
    assert model.llm.get_input_embeddings().num_embeddings > token_id
    ids, _, pv = chat_inputs(model, model.tokenizer)
    assert int((ids == token_id).sum()) == 64
    assert torch.isfinite(model(ids, labels=ids, pixel_values=pv).loss)


def test_hf_rejects_non_chatml_template(tmp_path, tiny_vision_path):
    path = save_tiny_llm(tmp_path / 'Llama-Tiny', arch='qwen2', template='llama3')
    with pytest.raises(ValueError, match='ChatML'):
        HFVLM(path, vision_model_path=tiny_vision_path)


# ---------------------------------------------------------------- 冻结策略 / LoRA / 权重 ----

@pytest.mark.parametrize('freeze_llm', [0, 1, 2])
def test_init_hf_backbone_freeze_strategies(tiny_qwen3_path, tiny_vision_path, freeze_llm):
    model, tokenizer, preprocess = init_hf(tiny_qwen3_path, tiny_vision_path, freeze_llm=freeze_llm)
    assert tokenizer is model.tokenizer and preprocess is model.processor
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    proj = {n for n, _ in model.named_parameters() if n.startswith('vision_proj.')}
    assert proj <= trainable and not any(n.startswith('vision_encoder.') for n in trainable)
    llm = trainable - proj
    if freeze_llm == 2:
        assert llm == set()
    elif freeze_llm == 1:
        assert llm and all(n.startswith(('llm.model.layers.0.', 'llm.model.layers.2.')) for n in llm)
    else:
        assert {'llm.model.embed_tokens.weight', 'llm.model.layers.1.mlp.up_proj.weight'} <= llm


def saved_activation_bytes(fn):
    seen, total = set(), [0]
    def pack(t):
        key = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
        if key not in seen:
            seen.add(key)
            total[0] += key[1]
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn()
    return total[0]


def test_init_returns_training_mode_so_checkpointing_is_active(tiny_qwen3_path, tiny_vision_path):
    # from_pretrained 得到的 HF 模型是 eval 模式，HF 的梯度检查点只在 train 模式下生效。
    # 训练脚本在第一次保存之前不会调用 model.train()，所以 init_vlm_model 必须返回 train 模式的模型
    plain, tok, _ = init_hf(tiny_qwen3_path, tiny_vision_path, freeze_llm=0)
    ckpt, _, _ = init_hf(tiny_qwen3_path, tiny_vision_path, freeze_llm=0, grad_checkpoint=True)
    assert ckpt.training and ckpt.llm.training and not ckpt.vision_encoder.training
    ids, _, pv = chat_inputs(ckpt, tok, answer='好的')
    without = saved_activation_bytes(lambda: plain(ids, labels=ids, pixel_values=pv))
    with_ckpt = saved_activation_bytes(lambda: ckpt(ids, labels=ids, pixel_values=pv))
    assert with_ckpt < without / 3


def test_lora_starts_as_identity_and_only_adapters_train(hf_model, tiny_qwen3_path, tiny_vision_path):
    ids, _, pv = chat_inputs(hf_model, hf_model.tokenizer)
    hf_model.eval()
    with torch.no_grad():
        before = hf_model(ids, pixel_values=pv).logits
        assert hf_model.apply_lora(4) == 3 * 7  # 3 层 × (q,k,v,o,gate,up,down)
        assert hf_model.apply_lora(4) == 0      # 重复调用不会叠加
        torch.testing.assert_close(hf_model(ids, pixel_values=pv).logits, before)  # B 初始化为 0

    model, _, _ = init_hf(tiny_qwen3_path, tiny_vision_path, freeze_llm=2, lora_rank=4)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert all(n.startswith('vision_proj.') or '.lora.' in n for n in trainable)
    assert sum(n.endswith('.lora.A.weight') for n in trainable) == 21


def test_lora_is_rejected_for_minimind_backbone(tiny_vision_path):
    with pytest.raises(ValueError, match='LoRA'):
        init_vlm_model(make_config(), from_weight='none', vision_model_path=tiny_vision_path, device='cpu', lora_rank=4)


def train_steps(model, tokenizer, steps=2):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    ids, _, pv = chat_inputs(model, tokenizer, answer='好的')
    for _ in range(steps):
        model(ids, labels=ids, pixel_values=pv).loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return ids, pv


def test_hf_delta_weights_roundtrip(tmp_path, tiny_qwen3_path, tiny_vision_path):
    model, tok, _ = init_hf(tiny_qwen3_path, tiny_vision_path, freeze_llm=1, lora_rank=4)
    ids, pv = train_steps(model, tok)
    path = str(tmp_path / 'sft_vlm_qwen3-tiny.pth')
    save_weights(model, path)
    saved = torch.load(path)
    assert all(v.dtype == torch.bfloat16 for v in saved.values())
    assert not any(k.startswith('vision_encoder.') for k in saved)
    assert any(k.startswith('llm.model.layers.0.') and '.lora.' not in k for k in saved)      # 训练过的首层
    assert not any(k.startswith('llm.model.layers.1.') and '.lora.' not in k for k in saved)  # 冻结且没改过的中间层不保存
    assert any(k.startswith('llm.model.layers.1.') and '.lora.' in k for k in saved)          # 但它的 LoRA 要保存

    reloaded = HFVLM(tiny_qwen3_path, vision_model_path=tiny_vision_path)
    load_vlm_weights(reloaded, path)  # 自动识别 LoRA 的秩并挂上
    for name, param in model.named_parameters():  # 训练后的参数按保存精度（bf16）取整后再比较
        if name in saved:
            param.data = param.data.to(torch.bfloat16).float()
    model.eval(), reloaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(reloaded(ids, pixel_values=pv).logits, model(ids, pixel_values=pv).logits, rtol=1e-4, atol=1e-4)


def test_hf_stage_chaining_keeps_earlier_changes(tmp_path, tiny_qwen3_path, tiny_vision_path):
    # Pretrain 训练了首尾层（freeze_llm=1），SFT 只训练投影层（freeze_llm=2）：
    # SFT 保存的权重里必须还带着 Pretrain 改过的层，否则只加载 SFT 权重时这些改动会丢失
    model, _, _ = init_hf(tiny_qwen3_path, tiny_vision_path, freeze_llm=1)
    with torch.no_grad():
        model.llm.model.layers[0].mlp.up_proj.weight.add_(1.0)
    pretrain_path = str(tmp_path / 'pretrain_vlm_qwen3-tiny.pth')
    save_weights(model, pretrain_path)

    sft_model, _, _ = init_hf(tiny_qwen3_path, tiny_vision_path, freeze_llm=2, from_weight=pretrain_path)
    sft_path = str(tmp_path / 'sft_vlm_qwen3-tiny.pth')
    save_weights(sft_model, sft_path)
    assert 'llm.model.layers.0.mlp.up_proj.weight' in torch.load(sft_path)

    final = HFVLM(tiny_qwen3_path, vision_model_path=tiny_vision_path)
    load_vlm_weights(final, sft_path)
    torch.testing.assert_close(final.llm.model.layers[0].mlp.up_proj.weight,
                               model.llm.model.layers[0].mlp.up_proj.weight.to(torch.bfloat16).float())


def test_frozen_params_bf16_trainable_fp32(tiny_qwen3_path, tiny_vision_path):
    model, tok, _ = init_hf(tiny_qwen3_path, tiny_vision_path, freeze_llm=1, lora_rank=4, frozen_dtype=torch.bfloat16)
    for name, param in model.named_parameters():
        assert param.dtype == (torch.float32 if param.requires_grad else torch.bfloat16), name
    ids, _, pv = chat_inputs(model, tok, answer='好的')
    with torch.autocast('cpu', dtype=torch.bfloat16):
        out = model(ids, labels=ids, pixel_values=pv)
    out.loss.backward()
    assert torch.isfinite(out.loss)
    assert all(p.grad is not None and p.grad.dtype == torch.float32 for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize('backbone', ['minimind', 'minimind_moe', 'hf'])
def test_grad_checkpointing_gives_identical_gradients(backbone, tiny_vision_path, tiny_qwen3_path, tokenizer):
    results = []
    for enable in [False, True]:
        torch.manual_seed(0)
        if backbone == 'hf':
            model = HFVLM(tiny_qwen3_path, vision_model_path=tiny_vision_path)
            tok = model.tokenizer
        else:
            model, tok = MiniMindVLM(make_config(use_moe=backbone == 'minimind_moe'), vision_model_path=tiny_vision_path), tokenizer
        model.train()
        model.set_grad_checkpointing(enable)
        ids, _, pv = chat_inputs(model, tok, answer='好的')
        out = model(ids, labels=ids, pixel_values=pv)
        (out.loss + out.aux_loss).backward()
        results.append((out.loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}))
    (loss_a, grads_a), (loss_b, grads_b) = results
    torch.testing.assert_close(loss_a, loss_b)
    assert grads_a.keys() == grads_b.keys() and grads_a
    for name in grads_a:
        torch.testing.assert_close(grads_a[name], grads_b[name], rtol=1e-5, atol=1e-6, msg=name)


# ---------------------------------------------------------------- 按长度分组采样 ----

def padded_tokens(batches, lengths):
    return sum(max(lengths[i] for i in batch) * len(batch) for batch in batches)


def test_length_grouped_sampler_covers_data_and_balances_ranks():
    random.seed(0)
    lengths = [random.randint(10, 700) for _ in range(1003)]
    batch_size, ranks = 4, 3
    per_rank = [list(LengthGroupedBatchSampler(lengths, batch_size, ranks, r, seed=1, epoch=0, mega_batch_mult=5)) for r in range(ranks)]
    expected_batches = math.ceil(1003 / (batch_size * ranks))
    assert [len(b) for b in per_rank] == [expected_batches] * ranks == [len(LengthGroupedBatchSampler(lengths, batch_size, ranks, 0))] * ranks
    assert all(len(batch) == batch_size for rank in per_rank for batch in rank)
    seen = [i for rank in per_rank for batch in rank for i in batch]
    assert set(seen) == set(range(1003)) and len(seen) - 1003 < batch_size * ranks  # 只有补齐用的少量重复
    assert max(lengths) in [lengths[i] for rank in per_rank for i in rank[0]]       # 最长的 batch 排在第一个
    for step in range(expected_batches):  # 同一步各卡的 batch 来自同一个全局 batch，长度上限接近
        maxima = [max(lengths[i] for i in per_rank[r][step]) for r in range(ranks)]
        assert max(maxima) - min(maxima) <= 120


def test_length_grouped_sampler_reduces_padding():
    random.seed(0)
    lengths = [random.randint(80, 768) for _ in range(4000)]
    grouped = list(LengthGroupedBatchSampler(lengths, 8, 1, 0, seed=0, epoch=0))
    perm = torch.randperm(len(lengths), generator=torch.Generator().manual_seed(0)).tolist()
    shuffled = [perm[i:i + 8] for i in range(0, len(perm), 8)]
    assert padded_tokens(grouped, lengths) < 0.7 * padded_tokens(shuffled, lengths)


def test_length_grouped_sampler_is_deterministic_and_resumable():
    lengths = list(range(1, 202))
    full = list(LengthGroupedBatchSampler(lengths, 4, 2, 1, seed=3, epoch=2))
    assert full == list(LengthGroupedBatchSampler(lengths, 4, 2, 1, seed=3, epoch=2))
    assert full != list(LengthGroupedBatchSampler(lengths, 4, 2, 1, seed=3, epoch=3))
    resumed = LengthGroupedBatchSampler(lengths, 4, 2, 1, seed=3, epoch=2, skip_batches=5)
    assert list(resumed) == full[5:] and len(resumed) == len(full) - 5
    tiny = [list(LengthGroupedBatchSampler([5, 1, 9, 3, 7], 4, 2, r)) for r in range(2)]  # 样本数少于一个全局 batch
    assert [len(b) for b in tiny] == [1, 1] and all(len(b[0]) == 4 for b in tiny)


@pytest.mark.parametrize('ensure_ascii', [False, True])
def test_estimate_lengths_tracks_real_token_counts(tmp_path, tokenizer, tiny_model, ensure_ascii):
    answers = ['短', '这是一段比较长的中文回答，描述了图片里的很多细节。' * 12, 'A short English answer.',
               'A much longer English answer with many details about the picture. ' * 12, '中英混合 mixed text 数字 123！' * 8]
    convs = [[{'role': 'user', 'content': '<image>\n请描述这张图片。'}, {'role': 'assistant', 'content': a}] for a in answers]
    convs.append([{'role': 'user', 'content': '<image>\n<image>\n比较两张图'}, {'role': 'assistant', 'content': '第一张更亮。'}])
    path = write_parquet(tmp_path / 'd.parquet', [(c, image_bytes()) for c in convs], ensure_ascii=ensure_ascii)
    ds = VLMDataset(path, tokenizer, preprocess=tiny_model.processor, max_length=4096)
    estimated = ds.estimate_lengths()
    real = [len(tokenizer(ds.create_chat_prompt(c)).input_ids) for c in convs]
    assert all(0.75 < e / r < 1.3 for e, r in zip(estimated, real)), list(zip(estimated, real))
