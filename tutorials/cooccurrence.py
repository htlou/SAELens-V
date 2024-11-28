import os
import random
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset
import tqdm
from sae_lens.activation_visualization import (
    load_llava_model,
    load_sae,
    separate_feature,
    run_model,
)
import concurrent.futures
from transformers import AutoTokenizer, LlavaNextForConditionalGeneration, LlavaNextProcessor, AutoModelForCausalLM

# # 配置代理
# os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

def load_and_process_model(model_name: str, model_path: str, device: str, sae_path: str, sae_device: str, n_devices: int = 7):
    """加载 LLaVA 模型和 SAE"""
    processor, hook_language_model = load_llava_model(model_name, model_path, device, n_devices)
    sae = load_sae(sae_path, sae_device)
    return processor, hook_language_model, sae

def load_and_sample_dataset(dataset_path: str, sample_size: int = 2):
    """加载并从数据集中抽取样本"""
    train_dataset = load_dataset(dataset_path, split="train", trust_remote_code=True)
    total_size = len(train_dataset)
    if sample_size >= total_size:
        print(f"sample size is larger than total size {total_size}, returning full train dataset")
        return train_dataset
    random_indices = random.sample(range(total_size), sample_size)
    sampled_dataset = train_dataset.select(random_indices)
    return sampled_dataset

def format_sample(raw_sample: dict):
    """格式化样本，只提取 question 和 image 字段，并生成所需的 prompt"""
    system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    user_prompt = 'USER: \n<image> {input}'
    assistant_prompt = '\nASSISTANT: {output}'

    prompt = raw_sample['question'].replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')
    image = raw_sample['image']
    image = image.resize((336, 336)).convert('RGBA')

    formatted_prompt = f'{system_prompt}{user_prompt.format(input=prompt)}{assistant_prompt.format(output="")}'
    return {'prompt': formatted_prompt, 'image': image}

def process_dataset(dataset, num_proc: int = 80):
    """使用 map 方法处理数据集"""
    return dataset.map(format_sample, num_proc=num_proc, remove_columns=['chosen', 'rejected', 'question'])

def prepare_inputs(processor, formatted_sample, device):
    """处理样本并准备输入"""
    return processor(
        text=formatted_sample['prompt'],
        images=formatted_sample['image'],
        return_tensors='pt',
        padding='max_length',
        max_length=256,
    ).to(device)

def save_cooccurrence_features(cooccurrence_feature_list,inputs_name_list, batch_idx,batch_size=999):
    """保存共现特征"""
    data_dict = {}
    for i in range(batch_idx,max(batch_idx+batch_size,len(cooccurrence_feature_list))):
        data_dict[inputs_name_list['image_name'][i]] = cooccurrence_feature_list[i]  # 使用批次索引作为文件名的一部分
    file_path = os.path.join(f"/data/changye/data/SPA_VL_cooccur/", f'cooccurrence_batch_{batch_idx}.pt')
    torch.save(data_dict, file_path)
    
    print(f"Batch {batch_idx} saved.")

def process_batch(i, formatted_dataset, processor, args):
    batch = {
        "image": formatted_dataset['image'][i:i + args.batch_size],
        "prompt": formatted_dataset['prompt'][i:i + args.batch_size]
    }
    inputs = prepare_inputs(processor, batch, args.device)
    return inputs


def main(args):
    import pdb;pdb.set_trace()
    # 加载模型和数据
    processor, hook_language_model, sae = load_and_process_model(
        args.model_name, args.model_path, args.device, args.sae_path, args.sae_device, n_devices=args.n_devices
    )
    sampled_dataset = load_and_sample_dataset(args.dataset_path, sample_size=args.sample_size)
    formatted_dataset = process_dataset(sampled_dataset, num_proc=80)

    # 处理输入
    all_inputs = []

    # 按批次准备输入
    with tqdm.tqdm(total=len(formatted_dataset), desc="Processing batches") as pbar:
        max_threads = 100  # 设置你希望的线程数，可以根据需要调整
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = []

            # 提交所有任务到线程池
            for i in range(0, len(formatted_dataset), args.batch_size):
                future = executor.submit(process_batch, 
                                        i, 
                                        formatted_dataset, 
                                        processor, 
                                        args)
                futures.append(future)

            # 按照完成顺序获取任务结果并更新进度条
            for future in concurrent.futures.as_completed(futures):
                inputs = future.result()  # 获取每个批次处理后的结果
                all_inputs.append(inputs)  # 将处理的结果添加到 all_inputs
                pbar.update(args.batch_size)  # 更新进度条
        
    processed_count = 0
    total_processed = 0

    cooccurrence_feature_list=[]
    inputs_name_list=[]

    with tqdm.tqdm(total=len(all_inputs), desc="Processing batches") as pbar:
        for batch_idx, inputs_batch in enumerate(all_inputs):
            # 运行模型并提取特征
            torch.cuda.empty_cache()
            image_indices, feature_act = run_model(inputs_batch, hook_language_model, sae, args.sae_device)
            cooccurrence_feature = separate_feature(image_indices, feature_act)
            for feature,batch in zip(cooccurrence_feature,inputs_batch):
                cooccurrence_feature_list.append(feature)
                inputs_name_list.append(batch["image_name"])
            # 更新已处理的数据数量
            processed_count += args.batch_size
            total_processed += args.batch_size
            
            pbar.update(1)

            # 保存当前批次的特征（每1000条数据保存一次）
            if processed_count >= 999:
                save_cooccurrence_features(cooccurrence_feature_list,inputs_name_list, batch_idx, batch_size=999)
                print(f"Processed {total_processed} data and saved features.")

                # 重置处理计数
                processed_count = 0

        # 最后一批数据可能没有满1000条，确保保存它们
        if processed_count > 0:
            save_cooccurrence_features(cooccurrence_feature, inputs_name_list,batch_idx,batch_size=999)
            print(f"Processed {total_processed} data and saved features.")

    print("Feature extraction and saving complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and run the model with LLaVA and SAE.")

    # Model and device configurations
    parser.add_argument('--model_name', type=str, default="llava-hf/llava-v1.6-mistral-7b-hf", help="Name of the model.")
    parser.add_argument('--model_path', type=str, default="/data/models/llava-v1.6-mistral-7b-hf", help="Path to the model directory.")
    parser.add_argument('--sae_path', type=str, default="/data/changye/model/llavasae_obliec100k_SAEV", help="Path to the SAE model.")
    parser.add_argument('--sae_device', type=str, default="cuda:7", help="Device for SAE model.")
    parser.add_argument('--device', type=str, default="cuda:0", help="Device for main model.")
    parser.add_argument('--n_devices', type=int, default=8, help="Number of devices for model parallelism.")

    # Dataset configurations
    parser.add_argument('--dataset_path', type=str, default="/data/changye/data/SPA-VL", help="Path to the dataset.")
    parser.add_argument('--sample_size', type=int, default=4000, help="Number of samples to randomly select.")
    parser.add_argument('--batch_size', type=int, default=3, help="Batch size for each processing step.")

    args = parser.parse_args()
    
    main(args)
