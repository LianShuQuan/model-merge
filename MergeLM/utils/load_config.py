import torch

if torch.cuda.is_available():
    cache_dir = "/mnt/82_store/huggingface_cache/"
else:
    cache_dir = "/Users/yule/.cache"
