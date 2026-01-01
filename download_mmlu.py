# benchmark/download.py

import os
# 设置镜像（必须在导入 datasets 前）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset

# 指定缓存目录
cache_dir = "dataset/mmlu"
dataset_name = 'cais/mmlu'

# 加载数据集
test_dataset = load_dataset(
    dataset_name,
    'all',
    split="test",
    cache_dir=cache_dir,
    download_mode="reuse_cache_if_exists"  # 避免重复下载
)

test_dataset.save_to_disk(cache_dir + "/test")

print("Download complete!")
print("Test size:", len(test_dataset))
