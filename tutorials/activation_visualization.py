import os
import torch
from tqdm import tqdm
import numpy as np
import plotly.express as px
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset, load_from_disk
from transformer_lens import HookedTransformer
from typing import Any, Generator, Iterator, Literal, cast
from sae_lens import SAE
from transformers import (
    AutoTokenizer,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    AutoModelForCausalLM,
)
import matplotlib.pyplot as plt
from functools import partial
from transformers.feature_extraction_utils import BatchFeature
from transformer_lens.utils import tokenize_and_concatenate
import transformer_lens.utils as utils
from transformer_lens.HookedLlava import HookedLlava
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import pdb
# pdb.set_trace()
def load_models(model_name: str, model_path: str, device: str):
    processor = LlavaNextProcessor.from_pretrained(model_path)
    vision_model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    vision_tower = vision_model.vision_tower.to(device)
    multi_modal_projector = vision_model.multi_modal_projector.to(device)
    hook_language_model = HookedLlava.from_pretrained(
        model_name,
        hf_model=vision_model.language_model,
        device=device,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        tokenizer=None,
        dtype=torch.float32,
        vision_tower=vision_tower,
        multi_modal_projector=multi_modal_projector,
        n_devices=4,
    )
    # hook_language_model=None
    return (
        processor,
        vision_model,
        vision_tower,
        multi_modal_projector,
        hook_language_model,
    )


def load_sae(sae_path: str, sae_device: str):
    sae = SAE.load_from_pretrained(
        path=sae_path,
        device=sae_device,
    )
    return sae


def load_dataset_func(dataset_path: str, columns_to_read: list):
    try:
        dataset = (
            load_dataset(
                dataset_path,
                split="train",
                streaming=False,
                trust_remote_code=False,  # type: ignore
            )
            if isinstance(dataset_path, str)
            else dataset_path
        )
    except Exception as e:
        dataset = (
            load_from_disk(
                dataset_path,
            )
            if isinstance(dataset_path, str)
            else dataset_path
        )
    if isinstance(dataset, (Dataset, DatasetDict)):
        dataset = cast(Dataset | DatasetDict, dataset)
    ds_context_size = len(dataset["input_ids"])
    if hasattr(dataset, "set_format"):
        dataset.set_format(type="torch", columns=columns_to_read)
        print("Dataset format set.")
    return dataset


def prepare_input(processor, image_path: str, example_prompt: str):
    image = Image.open(image_path)
    image = image.resize((336, 336))
    example_prompt = example_prompt + " <image>"
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": example_prompt},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=False)
    # print("Generated Prompt:\n", prompt)
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    return inputs,image


def image_recover(inputs, processor):
    img_std = torch.tensor(processor.image_processor.image_std).view(3, 1, 1)
    img_mean = torch.tensor(processor.image_processor.image_mean).view(3, 1, 1)
    img_recover = inputs.pixel_values[0].cpu() * img_std + img_mean
    img_recover = to_pil_image(img_recover)
    return img_recover


def run_model(inputs, hook_language_model, sae, sae_device: str):
    with torch.no_grad():
        image_indice, cache = hook_language_model.run_with_cache(
            input=inputs,
            model_inputs=inputs,
            vision=True,
            prepend_bos=True,
            names_filter=lambda name: name == sae.cfg.hook_name,
            return_type="image_indices",
        )

        tmp_cache = cache[sae.cfg.hook_name]
        tmp_cache = tmp_cache.to(sae_device)
        feature_acts = sae.encode(tmp_cache)
        sae_out = sae.decode(feature_acts)
        del cache
    return image_indice, feature_acts


def patch_mapping(image_indice, feature_acts):
    assert image_indice.shape[0] == 1176
    newline_indices = torch.arange(604, 1180, 25)
    valid_indices = torch.tensor(
        [i for i in image_indice if i not in newline_indices]
    )
    patch_indices = torch.stack(
        (valid_indices[:576], valid_indices[576:]), dim=1
    )
    patch_features = feature_acts[:, patch_indices]
    return patch_features


def main():
    MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"
    model_path = "/mnt/data/changye/model/llava"
    device = "cuda:4"
    sae_device = "cuda:7"
    sae_path = "/mnt/data/changye/checkpoints/xepk4xea/final_163840000"
    dataset_path = "/mnt/data/changye/data/obelics3k-tokenized-llava4096"
    columns_to_read = ["input_ids", "pixel_values", "attention_mask", "image_sizes"]
    example_prompt = "The color of car in the image is"
    # example_answer = "Apple"
    image_path = "/home/saev/changye/OIP (1).jpg"

    torch.cuda.empty_cache()

    processor, vision_model, vision_tower, multi_modal_projector, hook_language_model = load_models(
        MODEL_NAME, model_path, device
    )

    sae = load_sae(sae_path, sae_device)

    dataset = load_dataset_func(dataset_path, columns_to_read)

    inputs,image = prepare_input(processor, image_path, example_prompt)
    inputs = inputs.to(device)

    # img_recover = image_recover(inputs, processor)

    image_indice, feature_acts = run_model(inputs, hook_language_model, sae, sae_device)

    image_indice = image_indice.to("cpu")
    feature_acts = feature_acts.to("cpu")

    patch_features = patch_mapping(image_indice, feature_acts)

    # Further processing can be added here
    def map_patches_to_image(patch_features):
        """
        Maps activation data from patches to the corresponding positions in the image.
        
        Args:
            patch_features (torch.Tensor): Activation data of shape (576, 2, 65536).
            
        Returns:
            Image: A PIL Image representing the activation map.
        """
        patch_features=patch_features.squeeze(0)
        # Step 1: Count non-zero elements in the last dimension (65536) for each activation
        counts = (patch_features != 0).sum(dim=2)  # Shape: (576, 2)
        
        # Step 2: Sum counts over the two activations for each patch
        total_counts = counts.sum(dim=1)  # Shape: (576,)
        # total_counts=counts[:,0]
        # Step 3: Reshape total_counts into a 24x24 grid
        counts_2d = total_counts.view(24, 24)
        
        # Step 4: Upsample the 24x24 grid to a 336x336 image by repeating each element into a 14x14 block
        counts_large = counts_2d.repeat_interleave(14, dim=0).repeat_interleave(14, dim=1)
        
        # Step 5: Normalize counts to [0, 255] for image representation
        counts_large = counts_large.float()
        counts_normalized = counts_large / counts_large.max()
        colormap = plt.get_cmap('bwr')  # 蓝-白-红渐变
        counts_colored = colormap(counts_normalized.cpu().numpy())  
        # Step 6: Convert to NumPy array and ensure data type is uint8
        counts_colored_uint8 = (counts_colored[:, :, :3] * 255).astype(np.uint8)
        
        # Step 7: Create a grayscale image from the array
        activation_map = Image.fromarray(counts_colored_uint8)
        
        return activation_map

    activation_map = map_patches_to_image(patch_features)
    def overlay_activation_on_image(image, activation_map):
   
        original_image = image.convert('RGBA')
        activation_map = activation_map.resize((336, 336)).convert('RGBA')

        # 调整激活图的透明度
        alpha = 128  # 0.5透明度，取值范围0-255
        activation_map.putalpha(alpha)

        # 将激活图覆盖在原图上
        combined = Image.alpha_composite(original_image, activation_map)

        return combined
    final_image=overlay_activation_on_image(image, activation_map)
    final_image.show()
    final_image.save("car.png")

if __name__ == "__main__":
    main()
