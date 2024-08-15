

# try:
#   # for google colab users
#     import google.colab # type: ignore
#     from google.colab import output
#     COLAB = True
#     get_ipython().run_line_magic('pip', 'install sae-lens transformer-lens')
# except:
#   # for local setup
#     COLAB = False
#     from IPython import get_ipython # type: ignore
#     ipython = get_ipython(); assert ipython is not None
#     ipython.run_line_magic("load_ext", "autoreload")
#     ipython.run_line_magic("autoreload", "2")

# # Imports for displaying vis in Colab / notebook
# import webbrowser
# import http.server
# import socketserver
# import threading
# PORT = 8000

# general imports
import os
import torch
from tqdm import tqdm
import plotly.express as px

torch.set_grad_enabled(False);


def display_vis_inline(filename: str, height: int = 850):
    '''
    Displays the HTML files in Colab. Uses global `PORT` variable defined in prev cell, so that each
    vis has a unique port without having to define a port within the function.
    '''
    if not(COLAB):
        webbrowser.open(filename);

    else:
        global PORT

        def serve(directory):
            os.chdir(directory)

            # Create a handler for serving files
            handler = http.server.SimpleHTTPRequestHandler

            # Create a socket server with the handler
            with socketserver.TCPServer(("", PORT), handler) as httpd:
                print(f"Serving files from {directory} on port {PORT}")
                httpd.serve_forever()

        thread = threading.Thread(target=serve, args=("/content",))
        thread.start()

        output.serve_kernel_port_as_iframe(PORT, path=f"/{filename}", height=height, cache_in_notebook=True)

        PORT += 1


# package import
from torch import Tensor
from transformer_lens import utils
from functools import partial
from jaxtyping import Int, Float

device = ['cuda:1', 'cuda:2']


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

# get the SAE for this layer
# sae_0, cfg_dict_0, _ = SAE.from_pretrained(
#     release = "gemma-2b-it-res-jb",
#     sae_id = f"blocks.{layer}.hook_resid_post",
#     device = device[0]
# )
sae_0, cfg_dict_0, _ = SAE.from_pretrained(
    release = "mistral-7b-res-wg",
    sae_id = f"blocks.{layer}.hook_resid_pre",
    device = device[0]
)


# get hook point
hook_point_0 = sae_0.cfg.hook_name
print(hook_point_0)

    
local_model_1 = AutoModelForCausalLM.from_pretrained("/aifs4su/yaodong/projects/hantao/personal/models/0808_output_Mistral-7B-Instruct-v0.1_alpaca_zh_sft")

# get model
model_1 = HookedTransformer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", hf_model = local_model_1, device = device[1])

# get the SAE for this layer
# sae_1, cfg_dict_1, _ = SAE.from_pretrained(
#     release = "gemma-2b-it-res-jb",
#     sae_id = f"blocks.{layer}.hook_resid_post",
#     device = device[1]
# )
sae_1, cfg_dict_1, _ = SAE.from_pretrained(
    release = "mistral-7b-res-wg",
    sae_id = f"blocks.{layer}.hook_resid_pre",
    device = device[1]
)

# get hook point
hook_point_1 = sae_1.cfg.hook_name
print(hook_point_1)

# dataset_path = "/aifs4su/yaodong/projects/hantao/personal/data/MATH_test_1-5_700.json"
dataset_path = "/aifs4su/yaodong/projects/hantao/personal/data/base_700.json"
import json

with open(dataset_path, 'r') as f:
    data = json.load(f)

# template = " <start_of_turn>user\n {prompt} \n<start_of_turn>model\n {answer}"
template = " <s>[INST] {prompt} [/INST] {answer}"

intersection_counts_full = torch.empty(0)
value_changes_full = torch.empty(0)

for piece in data:
    
    prompt = piece.get('problem', piece['input'])
    answer = piece.get('solution', piece['output'])[:100]
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
    topk1 = torch.topk(sv_feature_acts_0, 5)
    # print(torch.topk(sv_feature_acts_0, 5))


    sv_logits_1, cache_1 = model_1.run_with_cache(full, prepend_bos=True)
    tokens_1 = model_1.to_tokens(full)
    # print(tokens_1)

    # get the feature activations from our SAE
    sv_feature_acts_1 = sae_1.encode(cache_1[hook_point_1])

    # get sae_out
    sae_out_1 = sae_1.decode(sv_feature_acts_1)

    # print out the top activations, focus on the indices
    topk2 = torch.topk(sv_feature_acts_1, 5)
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
    
    devices_to_clear = [1, 2]

    # 遍历设备列表并清除各自的显存
    for device_id in devices_to_clear:
        torch.cuda.set_device(device_id)  # 设置当前设备
        torch.cuda.empty_cache()  # 清除当前设备的显存
    # print("Intersection Counts:", intersection_counts)
    # print("Average Value Changes:", value_changes)

# After the loop
# Calculate the mean and variance for intersection counts
intersection_mean = intersection_counts_full.mean()
intersection_variance = intersection_counts_full.var()

# Calculate the mean and variance for value changes
value_changes_mean = value_changes_full.mean()
value_changes_variance = value_changes_full.var()

print("Mean of Intersection Counts:", intersection_mean)
print("Variance of Intersection Counts:", intersection_variance)
print("Mean of Average Value Changes:", value_changes_mean)
print("Variance of Average Value Changes:", value_changes_variance)

# Mean of Intersection Counts: tensor(3.3914)
# Variance of Intersection Counts: tensor(0.8000)
# Mean of Average Value Changes: tensor(0.0792)
# Variance of Average Value Changes: tensor(0.0021)

# Mean of Intersection Counts: tensor(3.8875)
# Variance of Intersection Counts: tensor(0.6103)
# Mean of Average Value Changes: tensor(0.0662)
# Variance of Average Value Changes: tensor(0.0010)

# Mean of Intersection Counts: tensor(3.2180)
# Variance of Intersection Counts: tensor(1.1234)
# Mean of Average Value Changes: tensor(0.0857)
# Variance of Average Value Changes: tensor(0.0096)

# Mean of Intersection Counts: tensor(3.6158)
# Variance of Intersection Counts: tensor(0.9563)
# Mean of Average Value Changes: tensor(0.0663)
# Variance of Average Value Changes: tensor(0.0053)


# Mean of Intersection Counts: tensor(2.7753)
# Variance of Intersection Counts: tensor(0.9575)
# Mean of Average Value Changes: tensor(0.0988)
# Variance of Average Value Changes: tensor(0.0065)

# Mean of Intersection Counts: tensor(3.2677)
# Variance of Intersection Counts: tensor(1.0797)
# Mean of Average Value Changes: tensor(0.0903)
# Variance of Average Value Changes: tensor(0.0124)