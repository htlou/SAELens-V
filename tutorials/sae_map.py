import os
import random
import argparse  # 新增导入
import pdb
from typing import Any, cast
import torch.nn.functional as F
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from PIL import Image
from sae_lens import SAE
from transformer_lens.HookedLlava import HookedLlava
from transformer_lens import HookedChameleon
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from transformers import ChameleonForConditionalGeneration, AutoTokenizer, ChameleonProcessor
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookPoint
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
