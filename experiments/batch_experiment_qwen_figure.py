
# general imports
import os
import json
import torch
from tqdm import tqdm
import plotly.express as px
import pandas as pd

torch.set_grad_enabled(False);

# package import
from torch import Tensor
from transformer_lens import utils
from functools import partial
from jaxtyping import Int, Float
import matplotlib.pyplot as plt
import pickle

device = ['cuda:4', 'cuda:5']


from transformer_lens import HookedTransformer
from sae_lens import SAE
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes

# Choose a layer you want to focus on
# For this tutorial, we're going to use layer 2
layer = 13

from transformers import AutoModelForCausalLM


local_model_0 = AutoModelForCausalLM.from_pretrained("/aifs4su/yaodong/models/qwen/Qwen1.5-0.5B-Chat")

# get model
model_0 = HookedTransformer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", hf_model = local_model_0, device = device[0])

sae_0 = SAE.load_from_pretrained(
    path = "/aifs4su/yaodong/projects/hantao/personal/models/sae/sae_qwen1.5-0.5b-chat_blocks.13.hook_resid_pre_32768",
    device = device[0]
)

# get hook point
hook_point_0 = sae_0.cfg.hook_name
print(hook_point_0)

import os
import json
import torch
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer
from sae_lens import SAE
import argparse

tasks = {
    "safety": {
        "dataset": "/aifs4su/yaodong/projects/hantao/personal/data/safety_700.json",
        "general": "/aifs4su/yaodong/projects/hantao/personal/data/base_700.json",
        "model_path": "/aifs4su/yaodong/projects/hantao/personal/models/output_qwen_chat_sft0801",
    },
    "math": {
        "dataset": "/aifs4su/yaodong/projects/hantao/personal/data/MATH_700.json",
        "general": "/aifs4su/yaodong/projects/hantao/personal/data/base_700.json",
        "model_path": "/aifs4su/yaodong/projects/hantao/personal/models/0813/0814_output_Qwen1.5-0.5B-Chat_MATH_cor_sft",
    },
    "reasoning": {
        "dataset": "/aifs4su/yaodong/projects/hantao/personal/data/reasoning_700.json",
        "general": "/aifs4su/yaodong/projects/hantao/personal/data/base_700.json",
        "model_path": "/aifs4su/yaodong/projects/hantao/personal/models/0813/0814_output_Qwen1.5-0.5B-Chat_reasoning_cor_sft",
    },
    "empathy": {
        "dataset": "/aifs4su/yaodong/projects/hantao/personal/data/empathy_700.json",
        "general": "/aifs4su/yaodong/projects/hantao/personal/data/base_700.json",
        "model_path": "/aifs4su/yaodong/projects/hantao/personal/models/0813/0814_output_Qwen1.5-0.5B-Chat_empathy_cor_sft",
    },
    "chinese": {
        "dataset": "/aifs4su/yaodong/projects/hantao/personal/data/zh_700.json",
        "general": "/aifs4su/yaodong/projects/hantao/personal/data/base_700.json",
        "model_path": "/aifs4su/yaodong/projects/hantao/personal/models/0813/0814_output_Qwen1.5-0.5B-Chat_alpaca_zh_sft",
    }
}


def task(task_data, template, model_0, sae_0, hook_point_0, model_1, sae_1, hook_point_1, topk_list):
    max_topk = max(topk_list)  # Determine the maximum topk value needed
    
    # Initialize lists to store feature activations for each topk
    feature_acts_0_all = []
    feature_acts_1_all = []
    
    # Process each piece of data
    for piece in tqdm(task_data, "Processing data"):
        prompt = piece.get('instruction', '') + piece['input']
        answer = piece['output'][:100]
        full = template.format(prompt=prompt, answer=answer)
        
        # Process using model 0
        sv_logits_0, cache_0 = model_0.run_with_cache(full, prepend_bos=True)
        sv_feature_acts_0 = sae_0.encode(cache_0[hook_point_0])
        topk_0 = torch.topk(sv_feature_acts_0, max_topk, dim=1)
        
        # Process using model 1
        sv_logits_1, cache_1 = model_1.run_with_cache(full, prepend_bos=True)
        sv_feature_acts_1 = sae_1.encode(cache_1[hook_point_1])
        topk_1 = torch.topk(sv_feature_acts_1, max_topk, dim=1)
        
        # 在 topk 处理后创建新的变量来存储到 CPU 的数据
        topk_0_values_cpu = topk_0.values.to('cpu')
        topk_0_indices_cpu = topk_0.indices.to('cpu')
        topk_1_values_cpu = topk_1.values.to('cpu')
        topk_1_indices_cpu = topk_1.indices.to('cpu')

        # 将这些 CPU 数据存储到列表中
        feature_acts_0_all.append((topk_0_values_cpu, topk_0_indices_cpu))
        feature_acts_1_all.append((topk_1_values_cpu, topk_1_indices_cpu))
    
    print("Finished generation, start analyzing")
    # Now process the stored results for each topk value required
    results = []
    for topk in tqdm(topk_list, "Processing topk"):
        intersection_counts = []
        value_changes = []
        
        # Iterate through all stored topk results
        for topk_0, topk_1 in zip(feature_acts_0_all, feature_acts_1_all):
            topk_0_values, topk_0_indices = topk_0
            topk_1_values, topk_1_indices = topk_1
            
            indices_0, values_0 = topk_0_indices[:, :topk], topk_0_values[:, :topk]
            indices_1, values_1 = topk_1_indices[:, :topk], topk_1_values[:, :topk]
            
            # Compute intersections
            mask = (indices_0.unsqueeze(-1) == indices_1.unsqueeze(-2)).any(-1)
            matching_values_0 = values_0[mask]
            matching_values_1 = values_1[mask]
            
            # Calculate intersection counts and value changes
            intersection_count = mask.sum().item()
            if intersection_count > 0:
                value_change = torch.abs(matching_values_0 - matching_values_1).mean() / torch.abs(matching_values_0 + matching_values_1).mean()
            else:
                value_change = torch.tensor(0.0)
            
            intersection_counts.append(intersection_count)
            value_changes.append(value_change)
        
        # Calculate means and variances for each topk
        intersection_counts_tensor = torch.tensor(intersection_counts, dtype=torch.float)  # Ensure tensor is float
        value_changes_tensor = torch.tensor(value_changes, dtype=torch.float)  # Ensure tensor is float

        intersection_mean = intersection_counts_tensor.mean().item()
        intersection_variance = intersection_counts_tensor.var().item()
        value_changes_mean = value_changes_tensor.mean().item()
        value_changes_variance = value_changes_tensor.var().item()
                
        results.append({
            'topk': topk,
            'intersection_mean': intersection_mean,
            'intersection_variance': intersection_variance,
            'value_changes_mean': value_changes_mean,
            'value_changes_variance': value_changes_variance
        })
    
    return results


def process_task(task_name, config, topk_range):
    print(f"Processing {task_name}")

    # Load datasets
    with open(config["dataset"], 'r') as f:
        task_data = json.load(f)
    with open(config["general"], 'r') as f:
        general_data = json.load(f)

    # Load model
    # Load model
    local_model = AutoModelForCausalLM.from_pretrained(config["model_path"])
    # get model
    model_1 = HookedTransformer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", hf_model = local_model, device=device[1])

    sae_1 = SAE.load_from_pretrained(
        path = "/aifs4su/yaodong/projects/hantao/personal/models/sae/sae_qwen1.5-0.5b-chat_blocks.13.hook_resid_pre_32768",
        device = device[1]
    )

    # get hook point
    hook_point_1 = sae_1.cfg.hook_name
    print(hook_point_1)
    template = """<|im_start|>system
    You are a helpful assistant.<|im_end|><|im_start|>user
    {prompt}<|im_end|>
    <|im_start|>assistant
    {answer}"""
    
    result_task = task(
            task_data, template, model_0, sae_0, hook_point_0, model_1, sae_1, hook_point_1, topk_range
        )
    result_general = task(
            general_data, template, model_0, sae_0, hook_point_0, model_1, sae_1, hook_point_1, topk_range
        )
    
    result = {
        "specific": result_task,
        "general": result_general
    }

    return result

def plot_results(results, topk_range):
    plt.figure(figsize=(12, 7))  # 更大的图形尺寸以提高清晰度

    markers = ['o', '^', 's', 'x', '*', '+']  # 每个任务的不同标记
    colors = ['b', 'g', 'r', 'c', 'm', 'y']  # 每个任务的颜色
    linestyles = ['-', '--', '-.', ':']  # 线条样式

    for idx, (task_name, task_data) in enumerate(results.items()):
        marker = markers[idx % len(markers)]
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]

        specific_data = task_data["specific"]
        general_data = task_data["general"]

        y_values = []
        for topk, specific, general in zip(topk_range, specific_data, general_data):
            intersection_diff = (general['intersection_mean'] - specific['intersection_mean']) / topk
            y_values.append(intersection_diff)

        plt.plot(topk_range, y_values, label=f"{task_name}",
                 marker=marker, color=color, linestyle=linestyle, markersize=8)

    plt.title("不同任务中TopK对交集均值变化的影响")
    plt.xlabel('TopK')
    plt.ylabel('交集均值变化')
    plt.legend(title="任务", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()  # 调整布局以为图例腾出空间
    plt.savefig("outputs/figures/qwen_mean_change.pdf")

def main():
    # Dictionary to store results from each task
    results = {}

    topk_range = range(1, 20)
    for task_name, config in tasks.items():
        task_results = process_task(task_name, config, topk_range)
        results[task_name] = task_results


    # save raw results with pickle
    with open("outputs/raw/results_qwen.pkl", "wb") as f:
        pickle.dump(results, f)
    plot_results(results, topk_range)

if __name__ == "__main__":
    main()