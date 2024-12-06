import os
from huggingface_hub import HfApi

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"

def upload_files_in_directory(directory_path, repo_name, username, token, ignore_dirs=None):
    if ignore_dirs is None:
        ignore_dirs = []
    
    api = HfApi()
    repo_id = f"{username}/{repo_name}"
    
    # 创建新的私有仓库（如果已存在则忽略）
    try:
        api.create_repo(repo_id=repo_id, token=token, exist_ok=True, private=False, repo_type="dataset")
        print(f"仓库 {repo_id} 已创建或已存在。")
    except Exception as e:
        print(f"创建仓库 {repo_id} 时出错：{e}")
        return
    
    # 将忽略目录转换为相对于 directory_path 的模式
    ignore_patterns = []
    for dir_path in ignore_dirs:
        relative_ignore_dir = os.path.relpath(dir_path, directory_path)
        ignore_patterns.append(f"{relative_ignore_dir}/**")
    
    # 使用 upload_folder 上传整个目录
    try:
        api.upload_folder(
            folder_path=directory_path,
            path_in_repo="",  # 将文件上传到仓库的根目录
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            ignore_patterns=ignore_patterns,
        )
        print(f"成功将目录 {directory_path} 上传到仓库 {repo_id}。")
    except Exception as e:
        print(f"上传目录 {directory_path} 时出错：{e}")
    
def main():

    # Hugging Face 用户名和 Token
    username = "Antoinegg1"         # 请替换为您的用户名
    token = "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"          # 请替换为您的 Hugging Face 访问令牌

    # 指定目标仓库名称和目录
    name_groups = [
        {
            "directory_path": "/data/changye/data/RLAIF-V_Cosi",
            "repo_name": "RLAIF-V_Cosi",
            "ignore_dirs": []
        }
    ]

    for name_group in name_groups:
        upload_files_in_directory(
            name_group["directory_path"],
            name_group["repo_name"],
            username,
            token,
            ignore_dirs=name_group["ignore_dirs"]
        )

if __name__ == "__main__":
    main()