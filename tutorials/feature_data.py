import os
import random
import argparse
import numpy as np
import torch
from PIL import Image
from sae_lens import SAE
from transformer_lens.HookedLlava import HookedLlava
from transformers import LlavaForConditionalGeneration, LlavaProcessor
import tqdm

# 请确保以下导入的函数已正确实现
from sae_lens.activation_visualization import (
    load_llava_model,
    load_sae,
)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some inputs.')
    parser.add_argument('--output_path', type=str, default='count_table.txt', help='保存结果的输出文件路径')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--sae_path', type=str, required=True, help='SAE模型路径')
    parser.add_argument('--sample_size', type=int, default=1000, help='采样的图像数量')
    parser.add_argument('--device', type=str, default='cuda:0', help='使用的设备，例如cuda:0')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    return args

def prepare_batch_input(processor, device, file_paths, example_prompt):
    images = []
    prompts = []
    for file_path in file_paths:
        image = Image.open(file_path)
        image = image.resize((336, 336))
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": example_prompt + "<image>"},
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

def run_model(inputs, hook_language_model, sae, sae_device):
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

def save_results(output_path, feature_acts_count_table):
    with open(output_path, "w") as f:
        for index, count in enumerate(feature_acts_count_table):
            f.write(f"{index}: {count}\n")
    print(f"统计表已保存到 {output_path}")

def process_images(sampled_png_files, image_path, processor, device, hook_language_model, sae, sae_device, example_prompt, batch_size=32):
    num_files = len(sampled_png_files)
    feature_acts_count_table = np.zeros(65536, dtype=int)
    print("开始处理...")

    with tqdm.tqdm(total=num_files) as pbar:
        for i in range(0, num_files, batch_size):
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
    return feature_acts_count_table

def main():
    args = parse_arguments()

    # 设置随机种子
    random_seed = args.seed
    random.seed(random_seed)

    # 设置设备
    device = args.device
    sae_device = args.device  # 将所有模型和数据放在同一设备上

    # 设置其他参数
    MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"
    model_path = args.model_path
    sae_path = args.sae_path
    example_prompt = "The color in this image is"
    image_path = "/home/saev/changye/data/colors/color_images"
    output_path = args.output_path
    sample_size = args.sample_size

    # 加载模型
    processor, vision_model, vision_tower, multi_modal_projector, hook_language_model = load_llava_model(
        MODEL_NAME, model_path, device
    )
    sae = load_sae(sae_path, sae_device)

    # 获取图像文件列表
    files = os.listdir(image_path)
    png_files = [f for f in files if f.lower().endswith('.png')]
    png_files.sort()

    # 随机采样
    if sample_size > len(png_files):
        sample_size = len(png_files)
    sampled_png_files = random.sample(png_files, sample_size)

    # 处理图像并获取结果
    feature_acts_count_table = process_images(
        sampled_png_files, image_path, processor, device, hook_language_model, sae, sae_device, example_prompt
    )

    # 保存结果
    save_results(output_path, feature_acts_count_table)

if __name__ == "__main__":
    main()
