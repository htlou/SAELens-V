export HUGGINGFACE_TOKEN='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'
# export HF_MIRROR='https://hf-mirror.com/'
huggingface-cli login --token $HUGGINGFACE_TOKEN

# huggingface-cli download htlou/obelics_obelics_100k_tokenized_2048 --local-dir ./obelics_obelics_100k_tokenized_2048 --repo-type dataset
# huggingface-cli download htlou/obelics_obelics_10k_tokenized_2048 --local-dir ./obelics_obelics_10k_tokenized_2048 --repo-type dataset
# huggingface-cli download htlou/obelics_obelics_100k --local-dir ./obelics_obelics_100k --repo-type dataset
# huggingface-cli download htlou/obelics_obelics_10k --local-dir ./obelics_obelics_10k --repo-type dataset
# huggingface-cli download htlou/1006_stream_aligner --local-dir ./1006_stream_aligner --repo-type dataset
# huggingface-cli download Qwen/Qwen-Audio --local-dir /home/saev/changye/model/Qwen/Qwen-Audio --repo-type model
# huggingface-cli download llava-hf/llava-v1.6-mistral-7b-hf --local-dir /home/saev/changye/model/llava --repo-type model
huggingface-cli download EleutherAI/pile --local-dir /mnt/data/changye/data/pile --repo-type dataset
# huggingface-cli download Antoinegg1/PM-14B-10k --local-dir /home/saev/changye/model/PM-14B-10k --repo-type model
# huggingface-cli download --repo-type model mistralai/Mistral-7B-Instruct-v0.2 --local-dir /home/saev/changye/model/