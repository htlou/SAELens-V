import os
import torch

def merge_pt_files(folder_path, output_file):
    """
    读取文件夹中所有 .pt 文件，并将它们合成一个 .pt 文件。
    
    Args:
        folder_path (str): 包含 .pt 文件的文件夹路径。
        output_file (str): 合成后的输出文件路径。
    """
    merged_data = []  # 用于存储合并的数据

    # 遍历文件夹中所有 .pt 文件
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith('.pt'):
            file_path = os.path.join(folder_path, file_name)
            data = torch.load(file_path)
            merged_data.append(data)

    # 将合并的数据保存到一个新的 .pt 文件中
    torch.save(merged_data, output_file)
    print(f"所有 .pt 文件已合并到 {output_file}")

# 示例用法
folder_path = "/data/changye/data/Align-Anything/Align-Anything_cooccur"  # 替换为你的文件夹路径
output_file = "/data/changye/data/Align-Anything/Align-Anything_cooccur.pt"  # 替换为你的输出文件路径
merge_pt_files(folder_path, output_file)
