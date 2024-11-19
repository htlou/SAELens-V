import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import umap
import matplotlib.pyplot as plt
# 请确保以下导入的函数已正确实现
from sae_lens.activation_visualization import (
    load_llava_model,
    load_sae,
)


# 加载 SAE 模型
sae_path = "/mnt/data/changye/checkpoints/checkpoints-Llavatext/xepk4xea/final_163840000"
sae_device = "cuda:0"
sae = load_sae(sae_path, sae_device)

# 获取特征向量矩阵
features = sae.W_dec.cpu().detach().numpy()  # [65536, 4096]

# 读取激活次数
act_data_path = "/home/saev/changye/SAELens-V/activation_visualization/color_experiment/llava_sae_color30k_results.txt"
activation_counts = np.zeros(features.shape[0], dtype=int)

with open(act_data_path, "r") as f:
    for line in f:
        index, count = line.strip().split(":")
        activation_counts[int(index)] = int(count)

# 区分激活点和未激活点
active_indices = np.where(activation_counts > 0)[0]
inactive_indices = np.where(activation_counts == 0)[0]

# 从未激活点中随机采样 10,000 个点
sample_size = 70000
if sample_size > len(inactive_indices):
    sample_size = len(inactive_indices)  # 如果未激活点少于 10,000，选择所有未激活点
sampled_inactive_indices = np.random.choice(inactive_indices, size=sample_size, replace=False)

# 合并采样的未激活点和激活点
selected_indices = np.concatenate([active_indices, sampled_inactive_indices])
selected_features = features[selected_indices]
selected_activation_counts = activation_counts[selected_indices]

# 归一化特征向量
from sklearn.preprocessing import normalize
normalized_features = normalize(selected_features)

# 计算特征之间的余弦相似度矩阵
cos_sim_matrix = cosine_similarity(normalized_features)

# 转换为距离矩阵
distance_matrix = 1 - cos_sim_matrix  # 确保距离非负
distance_matrix = np.clip(distance_matrix, 0, None)  # 剔除负值

# 使用 UMAP 进行降维（支持 GPU 加速）
reducer = umap.UMAP(
    n_components=2,  # 降维到 2D
    metric="precomputed",  # 使用预计算的距离矩阵
    n_neighbors=15,  # UMAP 参数
    min_dist=0.1,
    n_jobs=64,
    verbose=True,  # 打印进度信息
    low_memory=True,  # 优化内存使用
)
embedded_points = reducer.fit_transform(distance_matrix)
# 根据激活次数排序点
sorted_indices = np.argsort(selected_activation_counts)  # 从低到高排序
embedded_points = embedded_points[sorted_indices]
sorted_activation_counts = selected_activation_counts[sorted_indices]
# 保存 UMAP 结果到文件
csv_output_path = "/home/saev/changye/SAELens-V/activation_visualization/color_experiment/umap_sae_30kresults_sampled_inactive.csv"
np.savetxt(
    csv_output_path,
    np.column_stack((embedded_points, selected_activation_counts)),
    delimiter=",",
    header="Component 1,Component 2,Activation Counts",
    comments=""
)
print(f"UMAP 降维结果已保存到 {csv_output_path}")

# 可视化降维结果并保存为图像
plt.figure(figsize=(10, 8))

# 创建颜色和大小
colors = sorted_activation_counts
sizes = np.where(sorted_activation_counts > 0, 20, 5)  # 激活点更大，零激活点更小

scatter = plt.scatter(
    embedded_points[:, 0], 
    embedded_points[:, 1], 
    c=colors, 
    cmap="viridis", 
    s=sizes, 
    alpha=0.7
)
plt.colorbar(scatter, label="Activation Counts")
plt.title("UMAP Visualization with Sampled Inactive Points")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")

# 保存为图片文件
image_output_path = "/home/saev/changye/SAELens-V/activation_visualization/color_experiment/umap_sae_30K_visualization_sampled_inactive.png"
plt.savefig(image_output_path, format="png", dpi=300)
print(f"UMAP 可视化图已保存到 {image_output_path}")

# 显示图像
plt.show()
