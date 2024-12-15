from datasets import load_dataset,load_from_disk
from transformer_lens import HookedTransformer
from typing import Tuple
from sae_lens import SAE
from transformers import (
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    AutoModelForCausalLM,
)
import torch
from transformer_lens.HookedLlava import HookedLlava
from transformer_lens.utils import tokenize_and_concatenate
import plotly.express as px
import threading
import tqdm
from transformer_lens import utils
from functools import partial
# import pdb;pdb.set_trace()

def reconstr_hook(activation, hook, sae_out):
    return sae_out

def zero_abl_hook(activation, hook):
    return torch.zeros_like(activation)

def text_reconstruction_test(hook_language_model, sae, token_dataset,description,batch_size=8):
    total_orig_loss = 0.0
    total_reconstr_loss = 0.0
    total_zero_loss = 0.0
    num_batches = 0

    total_tokens =50
    with torch.no_grad():
        with tqdm.tqdm(total=total_tokens, desc="Reconstruction Test") as pbar:
            for start_idx in range(0, total_tokens, batch_size):
                # torch.cuda.empty_cache()
                end_idx = min(start_idx + batch_size, total_tokens)
                batch_data = token_dataset[start_idx:end_idx]

                # 获取批量的 tokens
                batch_tokens = batch_data["tokens"]

                # 确保 tokens 在模型的设备上
                device = hook_language_model.cfg.device
                batch_tokens = batch_tokens.to(device)

                # 获取模型的激活并缓存
                _, cache = hook_language_model.run_with_cache(
                    batch_tokens,
                    prepend_bos=True,
                    names_filter=lambda name: name == sae.cfg.hook_name
                )

                # 使用 SAE 进行编码和解码
                activation = cache[sae.cfg.hook_name]
                activation =activation.to(sae.device)
                feature_acts = sae.encode(activation)
                sae_out = sae.decode(feature_acts)
                sae_out = sae_out.to(device)
                # 释放缓存以节省内存
                del cache

                # 计算原始损失
                orig_loss = hook_language_model(
                    batch_tokens, return_type="loss"
                ).item()
                total_orig_loss += orig_loss

                # 计算重建后的损失
                reconstr_loss = hook_language_model.run_with_hooks(
                    batch_tokens,
                    fwd_hooks=[
                        (
                            sae.cfg.hook_name,
                            partial(reconstr_hook, sae_out=sae_out),
                        )
                    ],
                    return_type="loss",
                ).item()
                total_reconstr_loss += reconstr_loss

                # 计算零置换后的损失
                zero_loss = hook_language_model.run_with_hooks(
                    batch_tokens,
                    fwd_hooks=[(sae.cfg.hook_name, zero_abl_hook)],
                    return_type="loss",
                ).item()
                total_zero_loss += zero_loss

                num_batches += 1

                # 清理内存
                del activation, feature_acts, sae_out
                torch.cuda.empty_cache()

                # 可选：打印每个批次的损失
                # print(f"Batch {num_batches}: Orig Loss = {orig_loss}, Reconstr Loss = {reconstr_loss}, Zero Loss = {zero_loss}")
                pbar.update(batch_size)

    # 计算平均损失
    avg_orig_loss = total_orig_loss / num_batches
    avg_reconstr_loss = total_reconstr_loss / num_batches
    avg_zero_loss = total_zero_loss / num_batches

    print(f"{description}_平均原始损失:", avg_orig_loss)
    print(f"{description}_平均重建损失:", avg_reconstr_loss)
    print(f"{description}_平均零置换损失:", avg_zero_loss)

def image_reconstruction_test(hook_language_model, sae, token_dataset,description,batch_size=8):
    total_orig_loss = 0.0
    total_reconstr_loss = 0.0
    total_zero_loss = 0.0
    num_batches = 0
    toks=0
    total_tokens =50
    with torch.no_grad():
        with tqdm.tqdm(total=total_tokens, desc="Reconstruction Test") as pbar:
            for data in token_dataset:
                toks+=1
                if toks>total_tokens:
                    break
                # torch.cuda.empty_cache()
                device = hook_language_model.cfg.device
                tokens = {
                    "input_ids": torch.tensor(data["input_ids"]).to(device),
                    "pixel_values": torch.tensor(data["pixel_values"]).to(device),
                    "attention_mask": torch.tensor(data["attention_mask"]).to(device),
                    "image_sizes": torch.tensor(data["image_sizes"]).to(device) 
                }
                # 确保 tokens 在模型的设备上
                
                # batch_tokens = batch_tokens

                # 获取模型的激活并缓存
                _, cache = hook_language_model.run_with_cache(
                    tokens,
                    prepend_bos=True,
                    names_filter=lambda name: name == sae.cfg.hook_name
                )

                # 使用 SAE 进行编码和解码
                activation = cache[sae.cfg.hook_name]
                activation =activation.to(sae.device)
                feature_acts = sae.encode(activation)
                sae_out = sae.decode(feature_acts)
                sae_out = sae_out.to(device)
                # 释放缓存以节省内存
                del cache

                # 计算原始损失
                orig_loss = hook_language_model(
                    tokens, return_type="loss"
                ).item()
                total_orig_loss += orig_loss

                # 计算重建后的损失
                reconstr_loss = hook_language_model.run_with_hooks(
                    tokens,
                    fwd_hooks=[
                        (
                            sae.cfg.hook_name,
                            partial(reconstr_hook, sae_out=sae_out),
                        )
                    ],
                    return_type="loss",
                ).item()
                total_reconstr_loss += reconstr_loss

                # 计算零置换后的损失
                zero_loss = hook_language_model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(sae.cfg.hook_name, zero_abl_hook)],
                    return_type="loss",
                ).item()
                total_zero_loss += zero_loss

                num_batches += 1

                # 清理内存
                del activation, feature_acts, sae_out
                torch.cuda.empty_cache()

                # 可选：打印每个批次的损失
                # print(f"Batch {num_batches}: Orig Loss = {orig_loss}, Reconstr Loss = {reconstr_loss}, Zero Loss = {zero_loss}")
                pbar.update(1)

    # 计算平均损失
    avg_orig_loss = total_orig_loss / num_batches
    avg_reconstr_loss = total_reconstr_loss / num_batches
    avg_zero_loss = total_zero_loss / num_batches

    print(f"{description}_平均原始损失:", avg_orig_loss)
    print(f"{description}_平均重建损失:", avg_reconstr_loss)
    print(f"{description}_平均零置换损失:", avg_zero_loss)

def load_vision_model(model_path: str, device: str) -> Tuple[LlavaNextForConditionalGeneration, torch.nn.Module, torch.nn.Module]:
    """加载视觉模型和相关组件"""
    processor = LlavaNextProcessor.from_pretrained(model_path)
    vision_model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.float32, 
        low_cpu_mem_usage=True,
    )
    vision_tower = vision_model.vision_tower.to(device)
    multi_modal_projector = vision_model.multi_modal_projector.to(device)
    return vision_model, vision_tower, multi_modal_projector

def load_hooked_llava(model_name: str, hf_model, device: str, vision_tower, multi_modal_projector, n_devices: int) -> HookedLlava:
    """加载 HookedLlava 模型"""
    hook_language_model = HookedLlava.from_pretrained(
        model_name,
        hf_model=hf_model,
        device=device, 
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        tokenizer=None,
        dtype=torch.float32,
        vision_tower=vision_tower,
        multi_modal_projector=multi_modal_projector,
        n_devices=n_devices,
    )
    return hook_language_model

def load_lm_model(model_path: str) -> AutoModelForCausalLM:
    """加载因果语言模型"""
    ori_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    return ori_model

def load_sae_model(path: str, device: str) -> SAE:
    """加载 SAE 模型"""
    sae_model = SAE.load_from_pretrained(
        path=path,
        device=device
    )
    return sae_model

def l0_test(sae, hook_language_model, token_dataset,description, batch_size=8):
    sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads
    total_tokens =200
    tok=0
    l0_list = []
    with torch.no_grad():
        # activation store can give us tokens.
        with tqdm.tqdm(total=total_tokens, desc="L0 Test") as pbar:
            for data in token_dataset:
                tok+=1
                if tok>total_tokens:
                    break
                # torch.cuda.empty_cache()
                device = hook_language_model.cfg.device
                tokens = {
                    "input_ids": torch.tensor(data["input_ids"]).to(device),
                    "pixel_values": torch.tensor(data["pixel_values"]).to(device),
                    "attention_mask": torch.tensor(data["attention_mask"]).to(device),
                    "image_sizes": torch.tensor(data["image_sizes"]).to(device) 
                }
                # 运行模型并获取缓存
                _, cache = hook_language_model.run_with_cache(
                    tokens,
                    prepend_bos=True,
                    names_filter=lambda name: name == sae.cfg.hook_name
                )
                tmp_cache=cache[sae.cfg.hook_name]
                tmp_cache=tmp_cache.to(sae.device)
                # 使用 SAE 编码和解码
                feature_acts = sae.encode(tmp_cache)
                sae_out = sae.decode(feature_acts)

                # 释放缓存以节省内存
                del cache

                # 计算每个批次的 L0 范数
                
                l0 = (feature_acts[:, 1:] > 0).float().sum(-1).detach()
                result_list = l0.flatten().tolist()
                # print(l0)
                for d in result_list:
                    l0_list.append(d)

                # 可选：打印每个批次的平均 L0 值
                # print(f"批次 {start_idx // batch_size + 1}: 平均 L0 = {l0.mean().item()}")
                pbar.update(1)

        # 汇总所有批次的 L0 范数
        l0_average = sum(l0_list) / len(l0_list)
        # 输出所有批次的平均l0带上描述
        print(
            f"平均 L0 for {description}: {l0_average}")
        
        # px.histogram(l0.flatten().cpu().numpy()).show()

def load_and_tokenize_dataset(dataset_path: str, split: str, tokenizer, max_length: int, add_bos_token: bool):
    """加载并对数据集进行标记化处理"""
    dataset = load_dataset(
        path=dataset_path,
        split=split,
        streaming=False,
        cache_dir="/mnt/file2/changye/tmp",
    )
    tokenized_dataset = tokenize_and_concatenate(
        dataset=dataset,
        tokenizer=tokenizer,
        streaming=True,
        max_length=max_length,
        add_bos_token=add_bos_token,
    )
    return tokenized_dataset

def run_l0_test(sae, hook_model, token_dataset, description,batch_size=8):
    print(f"L0 test {description}")
    l0_test(sae, hook_model, token_dataset,description,batch_size)
    
def run_reconstruction_test(hook_language_model, sae, token_dataset,description,batch_size=8):
    print(f"Reconstruction test {description}")
    image_reconstruction_test(hook_language_model, sae, token_dataset,description,batch_size)

def main():
    MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"
    model_path = "/mnt/file2/changye/model/llava"
    vision_device = "cuda:0"
    tokenized_dataset_llava = load_from_disk(
        "/mnt/file2/changye/dataset/llavasae_obelics3k-tokenized-4096_4image/batch_1",
        # split="train",
        # cache_dir="/mnt/file2/changye/tmp"
    )
    # 加载视觉模型
    vision_model, vision_tower, multi_modal_projector = load_vision_model(
        model_path=model_path,
        device=vision_device
    )

    # 加载 HookedLlava 模型
    hook_llava_model = load_hooked_llava(
        model_name=MODEL_NAME,
        hf_model=vision_model.language_model,
        device=vision_device,
        vision_tower=vision_tower,
        multi_modal_projector=multi_modal_projector,
        n_devices=8,
    )

    # ori_model_path = "/mnt/data/changye/model/Mistral-7B-Instruct-v0.2"
    # ori_model = load_lm_model(ori_model_path)

    # hook_ori_model = HookedLlava.from_pretrained(
    #     "mistralai/Mistral-7B-Instruct-v0.2",
    #     hf_model=ori_model,
    #     device="cuda:4",
    #     fold_ln=False,
    #     center_writing_weights=False,
    #     center_unembed=False,
    #     tokenizer=None,
    #     dtype=torch.float32,
    #     n_devices=4,
    # )

    # # 加载 SAE 模型
    sae_ori = load_sae_model(
        path="/mnt/file2/changye/model/SAE-Mistral-7b-v0.2",
        device="cuda:7"
    )
    # sae_V_llava = load_sae_model(
    #     path="/mnt/file2/changye/model/llavasae_obliec100k_SAEV",
    #     device="cuda:6"
    # )

    # sae_llava = load_sae_model(
    #     path="/mnt/file2/changye/model/SAE-Llava-mistral-pile100k/xepk4xea/final_163840000",
    #     device="cuda:6"
    # )
    
    # 加载并标记化数据集（适用于 LLAVA）


    # # 加载并标记化数据集（适用于原始模型）
    # token_dataset_ori = load_and_tokenize_dataset(
    #     dataset_path="NeelNanda/pile-10k",
    #     split="train",
    #     tokenizer=hook_ori_model.tokenizer,
    #     max_length=sae_ori.cfg.context_size,
    #     add_bos_token=sae_ori.cfg.prepend_bos,
    #     cache_dir="/mnt/file2/changye/tmp"
    # )
    del vision_model, vision_tower, multi_modal_projector

    threads = []
    thread1=threading.Thread(target=run_reconstruction_test,args=(hook_llava_model,sae_ori,tokenized_dataset_llava ,"sae_ori",5))
    # thread2=threading.Thread(target=run_reconstruction_test,args=(hook_llava_model,sae_llava,token_dataset_llava,"sae_llava",4))
    # thread3=threading.Thread(target=run_reconstruction_test,args=(hook_ori_model,sae_ori,token_dataset_ori,"sae_ori_ori",5))
    # thread4=threading.Thread(target=run_reconstruction_test,args=(hook_ori_model,sae_llava,token_dataset_ori,"sae_llava_ori",4))
    # thread5=threading.Thread(target=run_reconstruction_test,args=(hook_ori_model,sae_V_llava,token_dataset_ori,"sae_V_llava_ori",5))
    # thread6=threading.Thread(target=run_reconstruction_test,args=(hook_llava_model,sae_ori,token_dataset_llava,"sae_ori_llava",5))
    # threads.extend([thread1,thread2,thread3,thread4,thread5,thread6])
    thread1.start()
    # thread3.start()
    thread1.join()
    # thread3.join()
    # thread2.start()
    # thread4.start()
    # thread2.join()
    # thread4.join()
    # thread5.start()
    # thread6.start()
    # thread5.join()
    # thread6.join()


if __name__ == "__main__":
    main()
