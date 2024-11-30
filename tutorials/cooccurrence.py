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
os.environ['TMPDIR'] = '/data/changye/tmp'
os.environ['HF_DATASETS_CACHE']='/data/changye/tmp'
# export HF_DATASETS_CACHE='/data/changye/tmp'
# # 配置代理
# os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

def load_and_process_model(model_name: str, model_path: str, device: str, sae_path: str, sae_device: str, n_devices: int = 7,stop_at_layer:int=None):
    """加载 LLaVA 模型和 SAE"""
    processor, hook_language_model = load_llava_model(model_name, model_path, device, n_devices,stop_at_layer=stop_at_layer)
    sae = load_sae(sae_path, sae_device)
    return processor, hook_language_model, sae

def load_and_sample_dataset(dataset_path: str, start_idx,end_idx):
    """加载并从数据集中抽取样本"""
    train_dataset = load_dataset(dataset_path, split="train", trust_remote_code=True)
    total_size = len(train_dataset)
    if start_idx >= total_size:
        print(f"start index {start_idx} is beyond the dataset size {total_size}, returning empty dataset")
        return train_dataset.select([])  # 返回空的数据集

    if end_idx > total_size:
        end_idx = total_size  # 限制结束索引不超出数据集范围

    # 选择指定范围的样本
    sampled_dataset = train_dataset.select(range(start_idx, end_idx))
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
        padding=True, 
        truncation=True,
        max_length=256,
    )

def save_cooccurrence_features(cooccurrence_feature_list,inputs_name_list,l0_act_list,total_processed,save_path,start_idx):
    """保存共现特征"""
    cooccurrence_dict = {}
    l0_dict={}
    for i in range(len(inputs_name_list)):
        cooccurrence_dict[inputs_name_list[i]] = cooccurrence_feature_list[i]  # 使用批次索引作为文件名的一部分
        l0_dict[inputs_name_list[i]]=l0_act_list[i]
    count=total_processed+start_idx
    cooccurrence_file_path = os.path.join(save_path, f'cooccurrence_batch_{count}.pt')
    l0_file_path=os.path.join(save_path, f'l0_batch_{count}.pt')
    torch.save(cooccurrence_dict, cooccurrence_file_path)
    torch.save(l0_dict, l0_file_path)
    
    print(f"Batch {count} saved.")

def process_batch(i, formatted_dataset, processor, args,step):
    batch = {
        "image": formatted_dataset['image'][i:i + step],
        "prompt": formatted_dataset['prompt'][i:i + step],
    }
    inputs = prepare_inputs(processor, batch, args.device)
    inputs["image_name"]=formatted_dataset["image_name"][i:i + step]
    return inputs


def main(args):
    # import pdb;pdb.set_trace()
    # 加载模型和数据
    processor, hook_language_model, sae = load_and_process_model(
        args.model_name, args.model_path, args.device, args.sae_path, args.sae_device, n_devices=args.n_devices,stop_at_layer=args.stop_at_layer
    )
    sampled_dataset = load_and_sample_dataset(args.dataset_path, start_idx=args.start_idx,end_idx=args.end_idx)
    formatted_dataset = process_dataset(sampled_dataset, num_proc=80)

    # 处理输入
    all_inputs = []

    # 按批次准备输入
    with tqdm.tqdm(total=len(formatted_dataset), desc="Processing batches") as pbar:
        max_threads = 80  # 设置你希望的线程数，可以根据需要调整
        # print("os cpu counts",os.cpu_count())
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = []
            thread_step=200
            # 提交所有任务到线程池
            for i in range(0, len(formatted_dataset), thread_step):
                future = executor.submit(process_batch, 
                                        i, 
                                        formatted_dataset, 
                                        processor, 
                                        args,thread_step)
                futures.append(future)

            # 使用 as_completed 确保任务完成后及时更新进度条
            for future in concurrent.futures.as_completed(futures):
                inputs = future.result()  # 获取每个批次处理后的结果
                all_inputs.append(inputs)  # 将处理的结果添加到 all_inputs
                pbar.update(thread_step)  # 更新进度条
                pbar.refresh()  # 强制刷新进度条，确保显示更新

    batch_list = []
    for d in all_inputs:
            for i in range(0, thread_step, args.batch_size):
                tensor={}
                for key in d:
                    tensor[key]=d[key][i:(i+args.batch_size)]
                batch_list.append(tensor)

    processed_count = 0
    total_processed = 0

    cooccurrence_feature_list=[]
    inputs_name_list=[]
    l0_act_list=[]

    with tqdm.tqdm(total=len(batch_list), desc="Processing cooccurrence") as pbar:
        for batch_idx, inputs_batch in enumerate(batch_list):
            torch.cuda.empty_cache()
            inputs_batch={
                'input_ids':inputs_batch['input_ids'].to(args.device),
                'attention_mask':inputs_batch['attention_mask'].to(args.device),
                'pixel_values':inputs_batch['pixel_values'].to(args.device),
                'image_sizes':inputs_batch['image_sizes'].to(args.device),
                'image_name':inputs_batch['image_name'],
                          }
            # 运行模型并提取特征

            image_indices, feature_act = run_model(inputs_batch, hook_language_model, sae, args.sae_device,stop_at_layer=args.stop_at_layer)
            if image_indices is None or feature_act is None:
                print("No image!")
                continue
            feature_act=feature_act
            l0_acts = ((feature_act[:, 1:] > 0).sum(dim=-1).float()).mean(dim=-1)   
            cooccurrence_feature = separate_feature(image_indices, feature_act)
            # cooccurrence_feature=[feature.to('cpu') for feature in cooccurrence_feature]
            # image_name=[batch.to('cpu') for batch in inputs_batch["image_name"] ]
            # l0_acts=[l0_act.to('cpu') for l0_act in l0_acts]
            for feature,batch,l0_act in zip(cooccurrence_feature,inputs_batch["image_name"],l0_acts):
                cooccurrence_feature_list.append(feature)
                inputs_name_list.append(batch)
                l0_act_list.append(l0_act)
            # 更新已处理的数据数量
            processed_count += args.batch_size
            total_processed += args.batch_size
            
            pbar.update(1)

            # 保存当前批次的特征（每1000条数据保存一次）
            if processed_count >= 1000:
                save_cooccurrence_features(cooccurrence_feature_list,inputs_name_list,l0_act_list,total_processed,args.save_path,args.start_idx)
                print(f"Processed {total_processed} data and saved features.")

                # 重置处理计数
                processed_count = 0
                cooccurrence_feature_list=[]
                inputs_name_list=[]
                l0_act_list=[]
        # 最后一批数据可能没有满1000条，确保保存它们
        if processed_count > 0:
            save_cooccurrence_features(cooccurrence_feature, inputs_name_list,l0_act_list,total_processed,args.save_path,args.start_idx)
            print(f"Processed {total_processed} data and saved features.")

    print("Feature extraction and saving complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and run the model with LLaVA and SAE.")

    # Model and device configurations
    parser.add_argument('--model_name', type=str, default="llava-hf/llava-v1.6-mistral-7b-hf", help="Name of the model.")
    parser.add_argument('--model_path', type=str, default="/data/models/llava-v1.6-mistral-7b-hf", help="Path to the model directory.")
    parser.add_argument('--sae_path', type=str, default="/data/changye/model/llavasae_obliec100k_SAEV", help="Path to the SAE model.")
    parser.add_argument('--sae_device', type=str, default="cuda:5", help="Device for SAE model.")
    parser.add_argument('--device', type=str, default="cuda:0", help="Device for main model.")
    parser.add_argument('--n_devices', type=int, default=8, help="Number of devices for model parallelism.")

    # Dataset configurations
    parser.add_argument('--dataset_path', type=str, default="/data/changye/data/SPA-VL", help="Path to the dataset.")
    parser.add_argument('--start_idx', type=int, default=13000)
    parser.add_argument('--end_idx', type=int, default=40000)
    parser.add_argument('--batch_size', type=int, default=10, help="Batch size for each processing step.")
    parser.add_argument('--save_path', type=str, default="/data/changye/data/SPA_VL_cooccur/")
    parser.add_argument('--stop_at_layer', type=int, default=17)
    args = parser.parse_args()
    
    main(args)
