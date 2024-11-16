from typing import Any, cast

import torch
from transformer_lens import HookedTransformer
import os

try:
    from transformer_lens import HookedChameleon
except Exception as e:
    HookedChameleon = None
from transformer_lens.hook_points import HookedRootModule

from transformers import AutoModelForSeq2SeqLM

def load_model(
    model_class_name: str,
    model_name: str,
    device: str | torch.device | None = None,
    model_from_pretrained_kwargs: dict[str, Any] | None = None,
    local_model_path: str | None = None,
) -> HookedRootModule:
    model_from_pretrained_kwargs = model_from_pretrained_kwargs or {}

    if "n_devices" in model_from_pretrained_kwargs:
        n_devices = model_from_pretrained_kwargs["n_devices"]
        if n_devices >= 1:
            # get available devices from CUDA_VISIBLE_DEVICES
            available_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            use_devices = available_devices[:n_devices]
            print(f"Available devices: {available_devices}")
            print("MODEL LOADING:")
            print("Setting model device to cuda for d_devices")
            print(f"Will use cuda {use_devices}")
            device = "cuda:0"
            print("-------------")

    if local_model_path is not None:
        if model_class_name == "HookedChameleon":
            from transformers import ChameleonForConditionalGeneration
            hf_model = ChameleonForConditionalGeneration.from_pretrained(local_model_path)
        else:
            hf_model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path)
    else:
        hf_model = None
    if model_class_name == "HookedTransformer":
        return HookedTransformer.from_pretrained_no_processing(
            model_name=model_name, device=device, **model_from_pretrained_kwargs
        )
    elif model_class_name == "HookedMamba":
        try:
            from mamba_lens import HookedMamba
        except ImportError:  # pragma: no cover
            raise ValueError(
                "mamba-lens must be installed to work with mamba models. This can be added with `pip install sae-lens[mamba]`"
            )
        # HookedMamba has incorrect typing information, so we need to cast the type here
        return cast(
            HookedRootModule,
            HookedMamba.from_pretrained(
                model_name, device=cast(Any, device), **model_from_pretrained_kwargs
            ),
        )
    elif model_class_name == "HookedChameleon":
        if HookedChameleon is None:
            raise ValueError("HookedChameleon is not installed")
        if hf_model is None:
            return HookedChameleon.from_pretrained(
                model_name=model_name, device=device, **model_from_pretrained_kwargs
            )
        else:
            return HookedChameleon.from_pretrained(
                model_name=model_name, hf_model=hf_model, 
                device=device, **model_from_pretrained_kwargs
            )
    else:  # pragma: no cover
        raise ValueError(f"Unknown model class: {model_class_name}")
