export HUGGINGFACE_TOKEN='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'
export HF_ENDPOINT='https://hf-mirror.com/'
huggingface-cli login --token $HUGGINGFACE_TOKEN
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""

# huggingface-cli download htlou/obelics_obelics_100k_tokenized_2048 --local-dir ./obelics_obelics_100k_tokenized_2048 --repo-type dataset
# huggingface-cli download htlou/obelics_obelics_10k_tokenized_2048 --local-dir ./obelics_obelics_10k_tokenized_2048 --repo-type dataset
# huggingface-cli download htlou/obelics_obelics_100k --local-dir ./obelics_obelics_100k --repo-type dataset
# huggingface-cli download htlou/obelics_obelics_10k --local-dir ./obelics_obelics_10k --repo-type dataset
# huggingface-cli download htlou/1006_stream_aligner --local-dir ./1006_stream_aligner --repo-type dataset
# huggingface-cli download Antoinegg1/SAE-Mistral-7b-v0.2 --local-dir /home/changye/model/SAE-Mistral-7b-v0.2 --repo-type dataset
# huggingface-cli download llava-hf/llava-v1.6-mistral-7b-hf --local-dir /mnt/file1/models/llava --repo-type model
# huggingface-cli download Antoinegg1/llavasae_obliec100k_SAEV --local-dir /mnt/file1/models/llavasae_obliec100k_SAEV --repo-type dataset
# huggingface-cli download Antoinegg1/SAE-Llava-mistral-pile100k --local-dir /mnt/file1/models/SAE-Llava-mistral-pile100k --repo-type dataset
# huggingface-cli download  --repo-type model mistralai/Mistral-7B-Instruct-v0.2 --local-dir /mnt/file1/models/Mistral-7B-Instruct-v0.2

huggingface-cli download  --repo-type dataset Antoinegg1/llavasae_obelics3k-tokenized-4096_4image --local-dir /mnt/file2/changye/dataset/llavasae_obelics3k-tokenized-4096_4image 

# huggingface-cli download  --repo-type dataset NeelNanda/pile-10k --local-dir /mnt/file1/models/dataset/pile-10k

# huggingface-cli download  --repo-type dataset clane9/imagenet-100 --local-dir /mnt/file2/changye/dataset/imagenet-100

