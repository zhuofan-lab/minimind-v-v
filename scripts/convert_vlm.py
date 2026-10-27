import os
import sys
import json

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import torch
import transformers
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
from model.model_vlm import MiniMindVLM, VLMConfig
from trainer.trainer_utils import load_vlm_weights

warnings.filterwarnings('ignore', category=UserWarning)


def convert_torch2transformers_minimind(torch_path, transformers_path, dtype=torch.bfloat16, vision_model_path="../model/siglip2-base-p32-256-ve", tokenizer_path='../model/'):
    VLMConfig.register_for_auto_class()
    MiniMindVLM.register_for_auto_class("AutoModelForCausalLM")
    lm_model = MiniMindVLM(lm_config, vision_model_path=vision_model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    load_vlm_weights(lm_model, torch_path, device=device)
    lm_model = lm_model.to(dtype)  # 转换模型权重精度
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    del lm_model.vision_encoder
    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(transformers_path)
    # 显式写入 tie_word_embeddings（save_pretrained 默认不序列化与默认值相同的字段）
    config_path = os.path.join(transformers_path, "config.json")
    config = json.load(open(config_path, 'r', encoding='utf-8'))
    config['tie_word_embeddings'] = True
    # ======= transformers-5.0的兼容低版本写法 =======
    if int(transformers.__version__.split('.')[0]) >= 5:
        tokenizer_config_path = os.path.join(transformers_path, "tokenizer_config.json")
        json.dump({**json.load(open(tokenizer_config_path, 'r', encoding='utf-8')), "tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}}, open(tokenizer_config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        config['rope_theta'] = lm_config.rope_theta; config['rope_scaling'] = None; config.pop('rope_parameters', None)
    json.dump(config, open(config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"模型已保存为 Transformers-MiniMind-V 格式: {transformers_path}")


def convert_transformers2torch(transformers_path, torch_path):
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
    torch.save(model.state_dict(), torch_path)
    print(f"模型已保存为 PyTorch 格式: {torch_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="把训练得到的 *.pth 权重转换为 Transformers 格式")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=8192, type=int, help="写入config的最大序列长度")
    parser.add_argument('--weight', default='sft_vlm', type=str, help="权重名称前缀（pretrain_vlm, sft_vlm）")
    parser.add_argument('--torch_path', default=None, type=str, help="输入权重路径，默认 ../out/{weight}_{hidden_size}[_moe].pth")
    parser.add_argument('--transformers_path', default=None, type=str, help="输出目录，默认 ../minimind-3v[-moe]")
    parser.add_argument('--vision_model', default='../model/siglip2-base-p32-256-ve', type=str, help="视觉编码器路径")
    parser.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'], help="保存精度")
    args = parser.parse_args()

    lm_config = VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, max_seq_len=args.max_seq_len, use_moe=bool(args.use_moe))
    moe_suffix = '_moe' if lm_config.use_moe else ''
    torch_path = args.torch_path or f"../out/{args.weight}_{lm_config.hidden_size}{moe_suffix}.pth"
    transformers_path = args.transformers_path or f"../minimind-3v{'-moe' if lm_config.use_moe else ''}"
    convert_torch2transformers_minimind(torch_path, transformers_path, dtype=getattr(torch, args.dtype), vision_model_path=args.vision_model)
