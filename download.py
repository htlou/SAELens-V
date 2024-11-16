from huggingface_hub import snapshot_download
snapshot_download(repo_id="burkelibbey/colors", revision="main", local_files_only=False, allow_patterns="*", use_auth_token=False,local_dir="./burkelibbey/colors",repo_type="dataset")
