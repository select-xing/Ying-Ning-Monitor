from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModel
import argparse
from PIL import Image
import torch.nn as nn
from pathlib import Path
from typing import Union
import torch
from peft import PeftModelForCausalLM
import os
from flask import Flask, request, jsonify

# from .wanda import *
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
app = Flask(__name__)


def load_model_and_tokenizer(
    model_dir: Union[str, Path], trust_remote_code: bool = True
):
    model_dir = Path(model_dir).expanduser().resolve()
    # 用于加载微调后的模型
    if (model_dir / "adapter_config.json").exists():
        import json

        with open(model_dir / "adapter_config.json", "r", encoding="utf-8") as file:
            config = json.load(file)
        model = AutoModel.from_pretrained(
            config.get("base_model_name_or_path"),
            trust_remote_code=trust_remote_code,
            device_map="balanced",
            # cache_dir="/public/lzy/GLM-4/finetune_demo/output/glm-4v-9b/",
            torch_dtype=torch.bfloat16,
        )
        model = PeftModelForCausalLM.from_pretrained(
            model=model,
            model_id=model_dir,
            trust_remote_code=trust_remote_code,
        )
        tokenizer_dir = model.peft_config["default"].base_model_name_or_path
    # 用于加载无微调的模型
    else:
        model = AutoModel.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            trust_remote_code=trust_remote_code,
            device_map="balanced",
            # quantization_config=quantization_config
            # cache_dir="/public/lzy/GLM-4/finetune_demo/output/glm-4v-9b/",
            # load_in_8bit=True
        )
        # model.load_state_dict(torch.load(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=trust_remote_code,
        encode_special_tokens=True,
        use_fast=False,
    )
    return model, tokenizer

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if "Linear" in module.__class__.__name__:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def prune(args, model):
    prune_n, prune_m = 0, 0
    lm_encoder_layers = model.transformer.encoder.layers
    for i in range(len(lm_encoder_layers)):
        layer = lm_encoder_layers[i]
        subset = find_layers(layer)
        for name in subset:
            W = subset[name].weight.data
            W_metric = (torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(W) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1))
            print(f"pruning lm layer {i} name {name}")
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)
            subset[name].weight.data[W_mask] = 0

    vm_transformer_layers = model.transformer.vision.transformer.layers
    for i in range(len(vm_transformer_layers)):
        layer = vm_transformer_layers[i]
        subset = find_layers(layer)
        for name in subset:
            W = subset[name].weight.data
            W_metric = (torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(W) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1))
            print(f"pruning lm layer {i} name {name}")
            if prune_n != 0:
                W_mask = (torch.zeros_like(W) == 1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel() * args.sparsity_ratio)].cpu()
                W_mask = (W_metric <= thresh)
            subset[name].weight.data[W_mask] = 0

    vm_linear_proj_layers = model.transformer.vision.linear_proj
    subset = find_layers(vm_linear_proj_layers)
    for name in subset:
        W = subset[name].weight.data
        W_metric = (torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(W) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1))
        print(f"pruning lm layer name {name}")
        if prune_n != 0:
            W_mask = (torch.zeros_like(W) == 1)
            for ii in range(W_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = W_metric[:, ii:(ii + prune_m)].float()
                    W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
        else:
            thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel() * args.sparsity_ratio)].cpu()
            W_mask = (W_metric <= thresh)
        subset[name].weight.data[W_mask] = 0

# 主函数，用于启动对话
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="THUDM/glm-4v-9b", help='LLaMA model')
    parser.add_argument('--sparsity_ratio', type=float, default=0.5, help='Sparsity level')
    parser.add_argument('--save_path', type=str, default="/public/lzy/llm_weights/glm-4v-9b-quant-prune-8bit-2-4/",
                        help='Path to save the pruned model.')
    args = parser.parse_args()

    # 初始化LLM模型
    print("=========== initialization llm =============")

    # 量化剪枝后的模型加载
    # llm_model, tokenizer = load_model_and_tokenizer("/public/lzy/llm_weights/glm-4v-9b-pruner-0.6")

    # 原始模型加载
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="balanced",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True, encode_special_tokens=True, use_fast=False)

    # 模型剪枝
    prune(args, llm_model)


    llm_model.eval()

    print("=========== model inference =============")

    query = "图中婴儿的睡姿是什么姿势，从三种仰卧、侧卧、俯卧姿势中选择一种回答"
    image_path = "/home/user0/img_1.jpg"
    image = Image.open(image_path).convert('RGB')

    inputs = tokenizer.apply_chat_template([{"role": "user", "image": image, "content": query}],
                                           add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                           return_dict=True)  # chat mode
    inputs = {k: v.to(llm_model.device) for k, v in inputs.items()}

    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = llm_model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        print(tokenizer.decode(outputs[0]))


def predict():
    llm_model.eval()
    print("=========== model inference =============")

    query = "图中婴儿的睡姿是什么姿势，从三种仰卧、侧卧、俯卧姿势中选择一种回答"
    # image_path = "/home/user0/img_1.jpg"
    # image = Image.open(image_path).convert('RGB')
    image = Image.open(request.files['image'])

    inputs = tokenizer.apply_chat_template([{"role": "user", "image": image, "content": query}],
                                           add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                           return_dict=True)  # chat mode
    inputs = {k: v.to(llm_model.device) for k, v in inputs.items()}

    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = llm_model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        print(tokenizer.decode(outputs[0]))
    return tokenizer.decode(outputs[0])


# 程序入口点
if __name__ == "__main__":
    main()



