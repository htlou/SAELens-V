from huggingface_hub import snapshot_download
snapshot_download(repo_id="Antoinegg1/llava_sae_pile10k_cp", revision="main", local_files_only=False, allow_patterns="*", use_auth_token=False,local_dir="./sae",repo_type="dataset")
