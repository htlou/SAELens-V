
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

device = ['cuda:2', 'cuda:3']


from transformer_lens import HookedTransformer
from sae_lens import SAE
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes

# Choose a layer you want to focus on
# For this tutorial, we're going to use layer 2
layer = 16

from transformers import AutoModelForCausalLM


local_model_0 = AutoModelForCausalLM.from_pretrained("/aifs4su/yaodong/models/mistral/Mistral-7B-Instruct-v0.1")

# get model
model_0 = HookedTransformer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", hf_model = local_model_0, device = device[0])

sae_0, _, _ = SAE.from_pretrained(
    release = "mistral-7b-res-wg",
    sae_id = f"blocks.{layer}.hook_resid_pre",
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
        "model_path": "/aifs4su/yaodong/projects/hantao/personal/models/output_mistral_instruct_sft0801",
    },
    "math": {
        "dataset": "/aifs4su/yaodong/projects/hantao/personal/data/MATH_700.json",
        "general": "/aifs4su/yaodong/projects/hantao/personal/data/base_700.json",
        "model_path": "/aifs4su/yaodong/projects/hantao/personal/models/0813/0814_output_Mistral-7B-Instruct-v0.1_MATH_cor_sft",
    },
    "reasoning": {
        "dataset": "/aifs4su/yaodong/projects/hantao/personal/data/reasoning_700.json",
        "general": "/aifs4su/yaodong/projects/hantao/personal/data/base_700.json",
        "model_path": "/aifs4su/yaodong/projects/hantao/personal/models/0813/0814_output_Mistral-7B-Instruct-v0.1_reasoning_cor_sft",
    },
    "empathy": {
        "dataset": "/aifs4su/yaodong/projects/hantao/personal/data/empathy_700.json",
        "general": "/aifs4su/yaodong/projects/hantao/personal/data/base_700.json",
        "model_path": "/aifs4su/yaodong/projects/hantao/personal/models/0813/0814_output_Mistral-7B-Instruct-v0.1_empathy_cor_sft",
    },
    "chinese": {
        "dataset": "/aifs4su/yaodong/projects/hantao/personal/data/zh_700.json",
        "general": "/aifs4su/yaodong/projects/hantao/personal/data/base_700.json",
        "model_path": "/aifs4su/yaodong/projects/hantao/personal/models/0813/0814_output_Mistral-7B-Instruct-v0.1_alpaca_zh_sft",
    }
}

def task(data, template, model_0, sae_0, hook_point_0, model_1, sae_1, hook_point_1, topk):
    intersection_counts_full = torch.empty(0)
    value_changes_full = torch.empty(0)
    for piece in data:
        if 'instruction' in piece.keys():
            prompt = piece['instruction'] + piece['input']
        else:
            prompt = piece['input']
        answer = piece['output'][:100]
        full = template.format(prompt=prompt, answer=answer)
        input = template.format(prompt=prompt, answer="")
        input_len = model_0.to_tokens(input).shape[1]
        
        sv_logits_0, cache_0 = model_0.run_with_cache(full, prepend_bos=True)
        tokens_0 = model_0.to_tokens(full)
        full_len = model_0.to_tokens(full).shape[1]
        # print(tokens_0)

        # get the feature activations from our SAE
        sv_feature_acts_0 = sae_0.encode(cache_0[hook_point_0])

        # get sae_out
        sae_out_0 = sae_0.decode(sv_feature_acts_0)

        # print out the top activations, focus on the indices
        topk1 = torch.topk(sv_feature_acts_0, topk)
        # print(torch.topk(sv_feature_acts_0, 5))


        sv_logits_1, cache_1 = model_1.run_with_cache(full, prepend_bos=True)
        tokens_1 = model_1.to_tokens(full)
        # print(tokens_1)

        # get the feature activations from our SAE
        sv_feature_acts_1 = sae_1.encode(cache_1[hook_point_1])

        # get sae_out
        sae_out_1 = sae_1.decode(sv_feature_acts_1)

        # print out the top activations, focus on the indices
        topk2 = torch.topk(sv_feature_acts_1, topk)
        # print(torch.topk(sv_feature_acts_1, 5))

        topk1_values_cpu = topk1.values.to('cpu')
        topk1_indices_cpu = topk1.indices.to('cpu')
        topk2_values_cpu = topk2.values.to('cpu')
        topk2_indices_cpu = topk2.indices.to('cpu')

        intersection_counts = []
        value_changes = []

        # for col in range(topk1_indices_cpu.size(1)):  # Assuming same number of columns
        for col in range(0, full_len):
            indices1 = topk1_indices_cpu[:,col,:]
            indices2 = topk2_indices_cpu[:,col,:]
            values1 = topk1_values_cpu[:,col,:]
            values2 = topk2_values_cpu[:,col,:]

            # Compute intersection manually
            mask = (indices1.unsqueeze(-1) == indices2.unsqueeze(-2)).any(-1)
            common_indices1 = indices1[mask]
            common_indices2 = indices2[mask]

            intersection_count = common_indices1.size(0)
            intersection_counts.append(intersection_count)

            # Compute value changes for intersected indices
            if intersection_count > 0:
                # Matching indices in both tensors
                matching_values1 = values1[mask]
                matching_values2 = values2[mask]

                # Calculate mean of absolute differences
                value_change = torch.abs(matching_values1 - matching_values2).mean() /  torch.abs(matching_values1 + matching_values2).mean()
            else:
                value_change = torch.tensor(0.0)  # If no intersection, set change to 0
            value_changes.append(value_change)

        # Output results
        intersection_counts = torch.tensor(intersection_counts)
        value_changes = torch.tensor(value_changes)

        intersection_counts_full = torch.cat((intersection_counts_full, intersection_counts))
        value_changes_full = torch.cat((value_changes_full, value_changes))
        # print("Intersection Counts:", intersection_counts)
        # print("Average Value Changes:", value_changes)

    # After the loop
    # Calculate the mean and variance for intersection counts
    intersection_mean = intersection_counts_full.mean()
    intersection_variance = intersection_counts_full.var()

    # Calculate the mean and variance for value changes
    value_changes_mean = value_changes_full.mean()
    value_changes_variance = value_changes_full.var()
    
    return intersection_mean, intersection_variance, value_changes_mean, value_changes_variance



def process_task(task_name, config):
    print(f"Processing {task_name}")

    # Load datasets
    with open(config["dataset"], 'r') as f:
        task_data = json.load(f)
    with open(config["general"], 'r') as f:
        general_data = json.load(f)

    # Load model
    local_model = AutoModelForCausalLM.from_pretrained(config["model_path"])
    # get model
    model = HookedTransformer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", hf_model = local_model, device=device[1])

    sae_1, _, _ = SAE.from_pretrained(
        release = "mistral-7b-res-wg",
        sae_id = f"blocks.{layer}.hook_resid_pre",
        device = device[1]
    )

    # get hook point
    hook_point_1 = sae_1.cfg.hook_name
    print(hook_point_1)
    template = """<s>[INST] {prompt} [/INST]"
    {answer} """

    intersection_mean, intersection_variance, value_changes_mean, value_changes_variance = task(task_data, template, model_0, sae_0, hook_point_0, model, sae_1, hook_point_1, config["topk"])

    # general data 
    intersection_mean_general, intersection_variance_general, value_changes_mean_general, value_changes_variance_general = task(general_data, template, model_0, sae_0, hook_point_0, model, sae_1, hook_point_1, config["topk"])
    
    df_task = pd.DataFrame({
        "Task": [task_name, task_name],
        "Data Type": ["Specific", "General"],
        "Intersection Mean": [intersection_mean, intersection_mean_general],
        "Intersection Variance": [intersection_variance, intersection_variance_general],
        "Value Changes Mean": [value_changes_mean, value_changes_mean_general],
        "Value Changes Variance": [value_changes_variance, value_changes_variance_general]
    })

    return df_task

def main():
    parser = argparse.ArgumentParser(description='批量实验参数')
    parser.add_argument('--topk', type=int, default=5, help='topk')
    args = parser.parse_args()
    # Dictionary to store results from each task
    results = []

    for task_name, config in tasks.items():
        config["topk"] = args.topk
        results.append(process_task(task_name, config))

    # Concatenate all DataFrames and save to CSV
    final_df = pd.concat(results)
    file_name = f"outputs/mistral_output_top_{args.topk}.csv"
    final_df.to_csv(file_name, index=False)
    print(f"Saved all task results to {file_name}")

if __name__ == "__main__":
    main()

# Mean of Intersection Counts: tensor(3.3914)
# Variance of Intersection Counts: tensor(0.8000)
# Mean of Average Value Changes: tensor(0.0792)
# Variance of Average Value Changes: tensor(0.0021)

# Mean of Intersection Counts: tensor(3.8875)
# Variance of Intersection Counts: tensor(0.6103)
# Mean of Average Value Changes: tensor(0.0662)
# Variance of Average Value Changes: tensor(0.0010)

# Mean of Intersection Counts: tensor(3.7037)
# Variance of Intersection Counts: tensor(1.2513)
# Mean of Average Value Changes: tensor(0.1209)
# Variance of Average Value Changes: tensor(0.0028)

# Mean of Intersection Counts: tensor(3.7493)
# Variance of Intersection Counts: tensor(1.5863)
# Mean of Average Value Changes: tensor(0.0980)
# Variance of Average Value Changes: tensor(0.0022)