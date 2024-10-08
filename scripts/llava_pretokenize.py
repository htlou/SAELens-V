import sys
sys.path.append("/data/changye/SAELens-V")
from sae_lens import PretokenizeRunner, PretokenizeRunnerConfig

cfg = PretokenizeRunnerConfig(
    tokenizer_name="llava-hf/llava-v1.6-mistral-7b-hf",
    dataset_path="/data/changye/pile10k", # this is just a tiny test dataset
    # data_files={"train": "obelics_10k_washed.json"},
    shuffle=True,
    num_proc=4, # increase this number depending on how many CPUs you have

    # tweak these settings depending on the model
    context_size=2048,
    begin_batch_token="bos",
    begin_sequence_token=None,
    sequence_separator_token="eos",
    image_column_name=None,
    column_name="text",
    # uncomment to upload to huggingface
    # hf_repo_id="your-username/c4-10k-tokenized-gpt2"

    # uncomment to save the dataset locally
    save_path="./obelic10k-tokenized-text-llava"
)

dataset = PretokenizeRunner(cfg).run()