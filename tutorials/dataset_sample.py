from datasets import load_dataset, DatasetDict

# 加载数据集
dataset = load_dataset("/data/changye/data/Align-Anything-TI2T-Instruction-100K")

# 随机采样
sample_size = 1000
sampled_dataset = dataset['train'].shuffle(seed=42).select(range(sample_size))

# 重新保存采样后的数据集
sampled_dataset.save_to_disk("/data/changye/data/Align-Anything-TI2T-Instruction-1K")

