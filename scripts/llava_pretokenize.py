import sys
sys.path.append("/home/saev/changye/SAELens-V")
from sae_lens import PretokenizeRunner, PretokenizeRunnerConfig
# import pdb;pdb.set_trace()
cfg = PretokenizeRunnerConfig(
    tokenizer_name="llava-hf/llava-v1.6-mistral-7b-hf",
    dataset_path="/home/saev/changye/data/", # this is just a tiny test dataset
    data_files={"train": "/home/saev/changye/data/obelics_3k_washed.json"},
    shuffle=True,
    num_proc=4, # increase this number depending on how many CPUs you have

    # tweak these settings depending on the model
    context_size=4096,
    begin_batch_token="bos",
    begin_sequence_token=None,
    sequence_separator_token="eos",
    image_column_name="images",
    column_name="texts",
    # uncomment to upload to huggingface
    # hf_repo_id="your-username/c4-10k-tokenized-gpt2"

    # uncomment to save the dataset locally
    save_path="/home/saev/changye/data/obelics3k-tokenized-llava4096"
)

dataset = PretokenizeRunner(cfg).run()