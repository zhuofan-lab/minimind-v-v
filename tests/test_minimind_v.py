import os
import random

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from conftest import REPO_ROOT, image_bytes, make_config, write_parquet
from dataset.lm_dataset import VLMDataset
from model.model_vlm import MiniMindVLM
from trainer.trainer_utils import (get_vlm_state_dict, init_vlm_model, load_vlm_weights, save_weights,
                                   vlm_checkpoint, vlm_collate_fn)

IMG_TOKEN = 12  # <|image_pad|>


# ---------------------------------------------------------------- 视觉特征注入 ----

def reference_count_vision_proj(tokens, h, vf, marker=IMG_TOKEN):
    """原实现（逐 token 的 Python 循环），用于对比重构前后在常规输入上的结果是否一致。"""
    if vf.dim() == 3:
        vf = vf.unsqueeze(1)
    out = []
    for b in range(h.size(0)):
        hb, seq, k, i = h[b], tokens[b].tolist(), 0, 0
        while i < len(seq):
            if seq[i] == marker:
                start = i
                while i < len(seq) and seq[i] == marker:
                    i += 1
                if k < vf.size(1):
                    hb = torch.cat((hb[:start], vf[b][k][:i - start], hb[i:]), dim=0)[:h.size(1)]
                    k += 1
            else:
                i += 1
        out.append(hb)
    return torch.stack(out)


def make_tokens(layout, length):
    """layout: [(起始位置, 占位符个数), ...]，其余位置填普通文本 token。"""
    tokens = torch.randint(100, 6400, (length,))
    for start, n in layout:
        tokens[start:start + n] = IMG_TOKEN
    return tokens


def test_injection_matches_original_implementation(tiny_model):
    torch.manual_seed(0)
    seq_len, hidden = 200, tiny_model.config.hidden_size
    layouts = [[(5, 64)], [(30, 64)], [(0, 64), (100, 64)], []]  # 单图、偏移、两张不相邻的图、纯文本
    tokens = torch.stack([make_tokens(layout, seq_len) for layout in layouts])
    h = torch.randn(len(layouts), seq_len, hidden)
    vf = torch.randn(len(layouts), 2, 64, hidden)
    new = tiny_model.count_vision_proj(tokens, h, vision_tensors=vf)
    torch.testing.assert_close(new, reference_count_vision_proj(tokens, h, vf), rtol=0, atol=0)


def test_injection_adjacent_images_keep_sequence_length(tiny_model):
    # 两个 <image> 紧挨着 → 128 个连续占位符。原实现把它当成一张图，只填 64 个特征并把序列缩短 64
    hidden = tiny_model.config.hidden_size
    tokens = make_tokens([(10, 128)], 200).unsqueeze(0)
    h, vf = torch.randn(1, 200, hidden), torch.randn(1, 2, 64, hidden)
    out = tiny_model.count_vision_proj(tokens, h, vision_tensors=vf)
    assert out.shape == h.shape
    torch.testing.assert_close(out[0, 10:74], vf[0, 0])
    torch.testing.assert_close(out[0, 74:138], vf[0, 1])
    torch.testing.assert_close(out[0, :10], h[0, :10])
    torch.testing.assert_close(out[0, 138:], h[0, 138:])


def test_injection_truncated_and_surplus_placeholders(tiny_model):
    hidden = tiny_model.config.hidden_size
    # 第 1 条：图像占位符被 max_seq_len 截断在末尾，只剩 20 个
    # 第 2 条：文本里出现了两个 <image>，但只提供了一张图 → 多出来的占位符保留原 embedding
    tokens = torch.stack([make_tokens([(80, 20)], 100), make_tokens([(0, 64), (70, 30)], 100)])
    h, vf = torch.randn(2, 100, hidden), torch.randn(2, 1, 64, hidden)
    out = tiny_model.count_vision_proj(tokens, h, vision_tensors=vf)
    torch.testing.assert_close(out[0, 80:], vf[0, 0, :20])
    torch.testing.assert_close(out[0, :80], h[0, :80])
    torch.testing.assert_close(out[1, :64], vf[1, 0])
    torch.testing.assert_close(out[1, 64:], h[1, 64:])


def test_injection_compiles_without_graph_breaks(tiny_model):
    hidden = tiny_model.config.hidden_size
    tokens = torch.stack([make_tokens([(3, 64)], 90), make_tokens([], 90)])
    h, vf = torch.randn(2, 90, hidden), torch.randn(2, 1, 64, hidden)
    torch._dynamo.reset()
    compiled = torch.compile(tiny_model.count_vision_proj, backend='eager', fullgraph=True)
    torch.testing.assert_close(compiled(tokens, h, vision_tensors=vf), tiny_model.count_vision_proj(tokens, h, vision_tensors=vf))


# ---------------------------------------------------------------- 模型前向 / loss ----

def build_inputs(model, tokenizer, question='描述这张图片', color=(10, 200, 30)):
    cfg = model.config
    text = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': cfg.image_special_token * cfg.image_token_len + '\n' + question}],
        tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, return_tensors='pt')
    pixel_values = dict(MiniMindVLM.image2tensor(Image.new('RGB', (80, 60), color), model.processor))
    return enc.input_ids, enc.attention_mask, pixel_values


def test_forward_backward_reaches_projector(tiny_model, tokenizer):
    ids, _, pv = build_inputs(tiny_model, tokenizer)
    assert int((ids == IMG_TOKEN).sum()) == 64
    out = tiny_model(ids, labels=ids, pixel_values=pv)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in tiny_model.vision_proj.parameters())
    assert all(p.grad is None for p in tiny_model.vision_encoder.parameters())


def test_image_changes_the_prediction(tiny_model, tokenizer):
    ids, _, pv_red = build_inputs(tiny_model, tokenizer, color=(255, 0, 0))
    _, _, pv_blue = build_inputs(tiny_model, tokenizer, color=(0, 0, 255))
    tiny_model.eval()
    with torch.no_grad():
        a = tiny_model(ids, pixel_values=pv_red).logits[:, -1]
        b = tiny_model(ids, pixel_values=pv_blue).logits[:, -1]
    assert not torch.allclose(a, b)


def test_loss_equals_mean_reduction(tiny_model, tokenizer):
    ids, _, pv = build_inputs(tiny_model, tokenizer)
    labels = ids.clone()
    labels[:, :70] = -100
    out = tiny_model(ids, labels=labels, pixel_values=pv)
    expected = F.cross_entropy(out.logits[:, :-1].reshape(-1, out.logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)
    torch.testing.assert_close(out.loss, expected)


def test_loss_without_supervised_tokens_is_zero_not_nan(tiny_model, tokenizer):
    ids, _, pv = build_inputs(tiny_model, tokenizer)
    out = tiny_model(ids, labels=torch.full_like(ids, -100), pixel_values=pv)
    assert out.loss.item() == 0.0
    out.loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in tiny_model.parameters() if p.grad is not None)


def test_vision_encoder_stays_in_eval_mode(tiny_model):
    tiny_model.train()
    assert tiny_model.training and tiny_model.model.training
    assert not tiny_model.vision_encoder.training
    tiny_model.eval()
    assert not tiny_model.vision_encoder.training


@pytest.mark.parametrize('mode', ['L', 'P', 'CMYK', 'RGBA', 'LA', 'I;16', '1'])
def test_image2tensor_accepts_non_rgb_images(tiny_model, mode):
    image = Image.new('RGB', (40, 30), (120, 30, 200)).convert(mode)
    pixel_values = MiniMindVLM.image2tensor(image, tiny_model.processor)['pixel_values']
    assert pixel_values.shape == (1, 3, 64, 64)


def test_greedy_generation_with_and_without_kv_cache_match(tiny_model, tokenizer):
    ids, mask, pv = build_inputs(tiny_model, tokenizer)
    tiny_model.eval()
    kwargs = dict(inputs=ids, attention_mask=mask, max_new_tokens=12, do_sample=False, pixel_values=pv, eos_token_id=None)
    with_cache = tiny_model.generate(use_cache=True, **kwargs)
    without_cache = tiny_model.generate(use_cache=False, **kwargs)
    assert with_cache.shape == (1, ids.size(1) + 12)
    assert torch.equal(with_cache, without_cache)


def test_moe_forward_has_aux_loss(tiny_vision_path, tokenizer):
    torch.manual_seed(0)
    model = MiniMindVLM(make_config(use_moe=True, num_experts=4, num_experts_per_tok=1), vision_model_path=tiny_vision_path)
    ids, _, pv = build_inputs(model, tokenizer)
    out = model(ids, labels=ids, pixel_values=pv)
    assert torch.isfinite(out.loss) and out.aux_loss.item() > 0
    (out.loss + out.aux_loss).backward()


# ---------------------------------------------------------------- 数据集 / collate ----

def make_dataset(tmp_path, tokenizer, processor, samples, max_length=256):
    path = write_parquet(tmp_path / 'data.parquet', samples)
    return VLMDataset(path, tokenizer, preprocess=processor, max_length=max_length)


SAMPLES = [
    ([{'role': 'user', 'content': '<image>\n描述这张图片。'}, {'role': 'assistant', 'content': '一只狗在草地上奔跑。'}],
     image_bytes((200, 30, 30))),
    ([{'role': 'user', 'content': '图里主要是什么颜色？<image>'}, {'role': 'assistant', 'content': '主要是蓝色。'},
      {'role': 'user', 'content': '还有呢？'}, {'role': 'assistant', 'content': '还有一些白色。'}],
     image_bytes((30, 30, 200), mode='L')),  # 灰度 JPEG：原实现在 SigLIP 预处理时直接报错
    ([{'role': 'user', 'content': '<image>\n这是一张调色板 PNG。'}, {'role': 'assistant', 'content': '好的。'}],
     image_bytes((30, 200, 30), mode='P', fmt='PNG')),
]


def test_dataset_builds_image_tokens_and_assistant_only_labels(tmp_path, tokenizer, tiny_model):
    random.seed(0)
    ds = make_dataset(tmp_path, tokenizer, tiny_model.processor, SAMPLES)
    for i in range(len(ds)):
        input_ids, labels, pixel = ds[i]
        assert input_ids.shape == labels.shape == (256,)
        assert int((input_ids == IMG_TOKEN).sum()) == 64
        assert pixel['pixel_values'].shape == (1, 3, 64, 64)
        supervised = tokenizer.decode(input_ids[labels != -100])
        user_turns = [t['content'].replace('<image>', '').strip() for t in SAMPLES[i][0] if t['role'] == 'user']
        for turn in SAMPLES[i][0]:
            if turn['role'] == 'assistant':
                assert turn['content'] in supervised
        assert all(u not in supervised for u in user_turns)
        assert supervised.count('<|im_end|>') == sum(t['role'] == 'assistant' for t in SAMPLES[i][0])
        assert torch.equal(labels[labels != -100], input_ids[labels != -100])


def test_collate_trims_padding_without_changing_loss(tmp_path, tokenizer, tiny_model):
    random.seed(0)
    ds = make_dataset(tmp_path, tokenizer, tiny_model.processor, SAMPLES, max_length=384)
    batch = [ds[i] for i in range(len(ds))]
    ids_full = torch.stack([b[0] for b in batch])
    labels_full = torch.stack([b[1] for b in batch])
    ids, labels, pixel_values = vlm_collate_fn(batch)

    assert pixel_values['pixel_values'].shape == (3, 1, 3, 64, 64)
    assert ids.size(1) % 64 == 0 and ids.size(1) < ids_full.size(1)
    assert ids.is_contiguous() and labels.is_contiguous()
    assert (labels_full[:, ids.size(1):] == -100).all()

    tiny_model.eval()
    with torch.no_grad():
        loss_full = tiny_model(ids_full, labels=labels_full, pixel_values=pixel_values).loss
        loss_trim = tiny_model(ids, labels=labels, pixel_values=pixel_values).loss
    torch.testing.assert_close(loss_trim, loss_full, rtol=1e-5, atol=1e-6)


def test_collate_mixes_samples_with_different_image_counts(tmp_path, tokenizer, tiny_model):
    random.seed(0)
    samples = [
        ([{'role': 'user', 'content': '<image>\n<image>\n这两张图有什么不同？'}, {'role': 'assistant', 'content': '第一张是红色，第二张是蓝色。'}],
         [image_bytes((220, 20, 20)), image_bytes((20, 20, 220))]),
        ([{'role': 'user', 'content': '<image>\n描述这张图片。'}, {'role': 'assistant', 'content': '一片绿色。'}],
         [image_bytes((20, 220, 20))]),
    ]
    ds = make_dataset(tmp_path, tokenizer, tiny_model.processor, samples)
    assert int((ds[0][0] == IMG_TOKEN).sum()) == 128 and int((ds[1][0] == IMG_TOKEN).sum()) == 64
    ids, labels, pixel_values = vlm_collate_fn([ds[0], ds[1]])
    assert pixel_values['pixel_values'].shape == (2, 2, 3, 64, 64)
    assert (pixel_values['pixel_values'][1, 1] == 0).all()  # 第二条样本只有一张图，第二张是补齐用的全 0 图

    tiny_model.eval()
    with torch.no_grad():
        batched = tiny_model(ids, pixel_values=pixel_values).logits
        for i in range(2):
            ids_i, _, pv_i = vlm_collate_fn([ds[i]])
            single = tiny_model(ids_i, pixel_values=pv_i).logits
            torch.testing.assert_close(batched[i, :ids_i.size(1)], single[0], rtol=1e-4, atol=1e-5)


def test_collate_keeps_full_length_when_nothing_is_supervised(tmp_path, tokenizer, tiny_model):
    long_question = '这是一段非常长的问题。' * 200  # 回答被 max_length 完全截掉
    samples = [([{'role': 'user', 'content': '<image>\n' + long_question}, {'role': 'assistant', 'content': '回答'}], image_bytes())]
    ds = make_dataset(tmp_path, tokenizer, tiny_model.processor, samples, max_length=128)
    ids, labels, pixel_values = vlm_collate_fn([ds[0]])
    assert ids.shape == (1, 128) and (labels == -100).all()
    loss = tiny_model(ids, labels=labels, pixel_values=pixel_values).loss
    assert loss.item() == 0.0  # 不再是 NaN


# ---------------------------------------------------------------- 训练初始化 / 权重 ----

def trainable_names(model):
    return {n for n, p in model.named_parameters() if p.requires_grad}


@pytest.mark.parametrize('freeze_llm', [0, 1, 2])
def test_init_vlm_model_freeze_strategies(tiny_vision_path, freeze_llm):
    cfg = make_config()
    model, tokenizer, preprocess = init_vlm_model(cfg, from_weight='none', tokenizer_path=os.path.join(REPO_ROOT, 'model'),
                                                  vision_model_path=tiny_vision_path, device='cpu', freeze_llm=freeze_llm)
    names = trainable_names(model)
    assert preprocess is not None and tokenizer is not None
    assert not any(n.startswith('vision_encoder.') for n in names)
    assert {n for n in names if n.startswith('vision_proj.')} == {n for n, _ in model.named_parameters() if n.startswith('vision_proj.')}
    llm = {n for n in names if not n.startswith('vision_proj.')}
    if freeze_llm == 2:
        assert llm == set()
    elif freeze_llm == 1:
        assert llm and all(n.startswith(('model.layers.0.', f'model.layers.{cfg.num_hidden_layers - 1}.')) for n in llm)
    else:
        assert 'model.embed_tokens.weight' in llm and 'model.layers.0.self_attn.q_proj.weight' in llm


def test_init_vlm_model_reports_missing_vision_encoder(tmp_path):
    with pytest.raises(FileNotFoundError, match='视觉编码器'):
        init_vlm_model(make_config(), from_weight='none', tokenizer_path=os.path.join(REPO_ROOT, 'model'),
                       vision_model_path=str(tmp_path / 'missing'), device='cpu')


def test_load_vlm_weights_reports_mismatches(tmp_path, tiny_vision_path, capsys):
    torch.manual_seed(0)
    src = MiniMindVLM(make_config(), vision_model_path=tiny_vision_path)
    llm_only = {k: v for k, v in get_vlm_state_dict(src).items() if not k.startswith('vision_proj.')}
    torch.save(llm_only, tmp_path / 'llm.pth')
    dst = MiniMindVLM(make_config(), vision_model_path=tiny_vision_path)
    load_vlm_weights(dst, str(tmp_path / 'llm.pth'))
    assert 'Projector 将随机初始化' in capsys.readouterr().out
    torch.testing.assert_close(dst.model.layers[0].self_attn.q_proj.weight, src.model.layers[0].self_attn.q_proj.weight)

    broken = {('backbone.' + k if 'layers.1.' in k else k): v for k, v in get_vlm_state_dict(src).items()}
    torch.save(broken, tmp_path / 'broken.pth')
    load_vlm_weights(dst, str(tmp_path / 'broken.pth'))
    out = capsys.readouterr().out
    assert '[警告]' in out and '缺少' in out and '未被模型使用' in out


def test_checkpoint_roundtrip_excludes_vision_encoder(tmp_path, tiny_vision_path, tokenizer):
    cfg = make_config()
    torch.manual_seed(0)
    model = MiniMindVLM(cfg, vision_model_path=tiny_vision_path)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    ids, _, pv = build_inputs(model, tokenizer)
    model(ids, labels=ids, pixel_values=pv).loss.backward()
    optimizer.step()
    scaler = torch.amp.GradScaler('cpu', enabled=False)

    save_dir = str(tmp_path / 'ckpt')
    vlm_checkpoint(cfg, weight='unit', model=model, optimizer=optimizer, epoch=1, step=7, save_dir=save_dir, scaler=scaler)
    assert sorted(os.listdir(save_dir)) == ['unit_64.pth', 'unit_64_resume.pth']  # 没有残留 .tmp

    weights = torch.load(os.path.join(save_dir, 'unit_64.pth'))
    assert weights and not any(k.startswith('vision_encoder.') for k in weights)
    assert all(v.dtype == torch.float16 for v in weights.values())

    resume = vlm_checkpoint(cfg, weight='unit', save_dir=save_dir)
    assert resume['epoch'] == 1 and resume['step'] == 7 and 'scaler' in resume
    assert not any(k.startswith('vision_encoder.') for k in resume['model'])

    torch.manual_seed(123)
    restored = MiniMindVLM(cfg, vision_model_path=tiny_vision_path)
    restored.load_state_dict(resume['model'], strict=False)
    for (name, a), (_, b) in zip(model.state_dict().items(), restored.state_dict().items()):
        torch.testing.assert_close(a, b, msg=name)
    torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=1e-3).load_state_dict(resume['optimizer'])


def test_save_weights_is_atomic_and_half_precision(tmp_path, tiny_model):
    path = str(tmp_path / 'w.pth')
    save_weights(tiny_model, path)
    assert os.listdir(tmp_path) == ['w.pth']
    state = torch.load(path)
    assert set(state) == set(get_vlm_state_dict(tiny_model))
    assert all(v.dtype == torch.float16 for v in state.values())
