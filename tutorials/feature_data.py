import os
import pdb
from typing import Any, cast
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from PIL import Image
from sae_lens import SAE
from torchvision.transforms.functional import to_pil_image
from transformer_lens.HookedLlava import HookedLlava
from transformer_lens import HookedChameleon
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from transformers import ChameleonForConditionalGeneration, AutoTokenizer, ChameleonProcessor
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookPoint
import seaborn as sns
# 请确保以下导入的函数已正确实现
from sae_lens.activation_visualization import (
    load_llava_model,
    load_chameleon_model,
    load_sae,
    # prepare_input,  # 我们将重新定义这个函数
    # generate_with_saev,
    # run_model,      # 我们将重新定义这个函数
)
import tqdm
import random
# 设置模型和设备
MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"
model_path = "/mnt/data/changye/model/llava"
device = "cuda:6"
sae_device = "cuda:6"  # 将所有模型和数据放在同一设备上
sae_path = "/mnt/data/changye/checkpoints/checkpoints-V/kxpk98cr/final_122880000"
dataset_path = "/home/saev/changye/data/colors"
columns_to_read = ["input_ids", "pixel_values", "attention_mask", "image_sizes"]
example_prompt = "The color in this image is"
image_path = "/home/saev/changye/data/colors/color_images"
save_path = "/home/saev/changye/SAELens-V/activation_visualization/color_experiment"

# 加载模型
(
    processor,
    vision_model,
    vision_tower,
    multi_modal_projector,
    hook_language_model,
) = load_llava_model(MODEL_NAME, model_path, device)

sae = load_sae(sae_path, sae_device)

# 定义批处理的输入准备函数
def prepare_batch_input(processor, device, file_paths, example_prompt):
    images = []
    prompts = []
    for file_path in file_paths:

        image = Image.open(file_path)
        image = image.resize((336, 336))
        # example_prompt = example_prompt + " <image>"
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": example_prompt+"<image>"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=False)
        prompts.append(prompt)
        images.append(image)
    inputs = processor(
        text=prompts,
        images=images,
        return_tensors='pt',
        padding=True,
    ).to(device)
    return inputs


# 修改后的 run_model 函数，支持批处理
def run_model(inputs, hook_language_model, sae, sae_device: str):
    with torch.no_grad():
        image_indices, cache = hook_language_model.run_with_cache(
            input=inputs,
            model_inputs=inputs,
            vision=True,
            prepend_bos=True,
            names_filter=lambda name: name == sae.cfg.hook_name,
            return_type="generate_with_saev",
        )

        tmp_cache = cache[sae.cfg.hook_name]
        tmp_cache = tmp_cache.to(sae_device)
        feature_acts = sae.encode(tmp_cache)
        sae_out = sae.decode(feature_acts)
        del cache
    return image_indices, feature_acts

# 获取图像文件列表
files = os.listdir(image_path)
png_files = [f for f in files if f.lower().endswith('.png')]
png_files.sort()
feature_acts_count_table = np.zeros(65536, dtype=int)

# **新增：随机采样一部分数据**
sample_size = 1000  # 设置要采样的图像数量，您可以根据需要调整
if sample_size > len(png_files):
    sample_size = len(png_files)
sampled_png_files = random.sample(png_files, sample_size)


# 设置批大小
batch_size = 32
num_files = len(sampled_png_files)
print("开始处理...")

with tqdm.tqdm(total=num_files) as pbar:
    for i in range(0, num_files, batch_size):
        # if i>64:
        #     break
        batch_files = sampled_png_files[i:i+batch_size]
        batch_file_paths = [os.path.join(image_path, f) for f in batch_files]
        # 准备批量输入数据
        inputs = prepare_batch_input(processor, device, batch_file_paths, example_prompt)
        # 运行模型
        _, feature_act = run_model(inputs, hook_language_model, sae, sae_device)
        # feature_act 的形状：[batch_size, sequence_length, feature_dim]
        feature_act = feature_act.cpu().detach().numpy()
        # 获取每个样本最后一个 token 的激活值
        last_token_feature_act = feature_act[:, -1, :]  # 形状：[batch_size, feature_dim]
        # 找出激活值大于 1 的索引
        indices = np.where(last_token_feature_act > 1)
        # indices 是一个包含 (batch_indices, feature_indices) 的元组
        # 对每个特征索引计数
        feature_indices = indices[1]
        unique_features, counts = np.unique(feature_indices, return_counts=True)
        feature_acts_count_table[unique_features] += counts
        pbar.update(len(batch_files))

# 保存结果到文本文件
output_path = "count_table.txt"
with open(output_path, "w") as f:
    for index, count in enumerate(feature_acts_count_table):
        f.write(f"{index}: {count}\n")

print(f"统计表已保存到 {output_path}")
