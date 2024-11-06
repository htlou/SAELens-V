import argparse
import json
import random

def sample_from_json(json_path, sample_size,output_path):
    # 加载 JSON 文件
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    # 检查样本数量并抽样
    if sample_size > len(data):
        print(f"样本数量大于数据总量，将返回全部数据。")
        sample_size = len(data)
    
    sampled_data = random.sample(data, sample_size)
    
    # 输出采样结果
    print(f"从 JSON 文件中抽取了 {sample_size} 条样本数据。")
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(sampled_data, file, ensure_ascii=False, indent=4)
    return sampled_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample data from a JSON file.")
    parser.add_argument("--json_path", type=str, help="JSON 文件路径")
    parser.add_argument("--sample_size", type=int, help="抽样数量")
    parser.add_argument("--output_path", type=str, help="输出文件路径")
    
    args = parser.parse_args()
    
    # 调用抽样函数
    sampled_data = sample_from_json(args.json_path, args.sample_size,args.output_path)
    # print(sampled_data)
