from datasets import load_dataset
from img2dataset import download
import os
import pandas as pd

def download_images_for_hf_dataset(dataset, image_column="images", output_dir="downloaded_images"):
    """
    从 Hugging Face 数据集下载图像并过滤下载失败的样本。

    参数:
    dataset: Hugging Face 数据集对象
    image_column: 包含图像 URL 的列名
    output_dir: 下载的本地图像存放目录

    返回:
    过滤后仅保留下载成功图像的样本
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 构建包含所有图像 URL 的列表，同时记录 URL 和样本的对应关系
    all_urls = []
    case_index_map = []  # 记录每个 URL 所属的 case 和 index
    for case_idx, case in enumerate(dataset):
        for img_idx, url in enumerate(case[image_column]):
            if url is not None:
                all_urls.append(url)
                case_index_map.append((case_idx, img_idx))

    # 创建一个临时的 parquet 文件
    temp_parquet_file = "temp_urls.parquet"
    df = pd.DataFrame({
        "URL": all_urls,
        "TEXT": [None] * len(all_urls)
    })
    df.to_parquet(temp_parquet_file)

    # 使用 img2dataset 下载图像
    download(
        processes_count=16,
        thread_count=32,
        url_list=temp_parquet_file,
        image_size=256,
        output_folder=output_dir,
        output_format="files",
        input_format="parquet",
        url_col="URL",
        caption_col="TEXT",
        enable_wandb=False,
        number_sample_per_shard=1000,
        distributor="multiprocessing",
    )

    # 删除临时 parquet 文件
    os.remove(temp_parquet_file)

    # 检查下载的图像文件，并替换 URL 为本地路径
    valid_cases = []
    downloaded_images_dir = os.path.join(output_dir, "00000")  # img2dataset 默认的第一个 shard 目录
    failed_cases = set()  # 记录下载失败的 case index

    for i, (case_idx, img_idx) in enumerate(case_index_map):
        url = all_urls[i]
        local_image_path = os.path.join(downloaded_images_dir, os.path.basename(url))
        
        if os.path.exists(local_image_path):
            # 替换 URL 为本地路径
            dataset[case_idx][image_column][img_idx] = local_image_path
        else:
            # 如果有一个图像下载失败，记录 case index
            failed_cases.add(case_idx)

    # 过滤掉包含下载失败图像的样本
    valid_cases = [case for idx, case in enumerate(dataset) if idx not in failed_cases]
    
    return valid_cases

# 示例：从 Hugging Face 数据集加载 JSON 数据
dataset = load_dataset("json", data_files="/data/changye/dataset/obelic10k/obelics_10k.json")

# 执行下载并过滤
valid_cases = download_images_for_hf_dataset(dataset)

# 打印结果
print(f"成功处理的样本数量: {len(valid_cases)}")
