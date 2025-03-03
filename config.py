# WatermarkTools/config.py

class Args:
    # 指定要使用的數據集名稱，這裡假設使用 "cifar10"
    dataset = 'cifar10'
    # 如果數據集路徑為 None，表示沒有指定路徑，自動下載到預設路徑
    dataset_path = None  
    # 水印演算法名稱（示意用）
    watermark = 'default_watermark'
    # 模型名稱（示意用）
    model = 'default_model'

args = Args()

"""
arg list:
args.dataset: the name of dataset
args.dataset_path: the path of args.dataset, it will be automatically download if None
args.watermark: the name of the watermark algorithm
args.model: the name of the watermark algorithm

<<<<<<< Updated upstream
"""
=======
import argparse

def get_config():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--watermark", type=str, default="", help="the name of the watermark algorithm")
    parser.add_argument("--dataset", type=str, default="", help="the name of training / testing dataset")
    parser.add_argument("--dataset_path", type=str, default=None, help="the path of args.dataset, it will be automatically download if emptys")
    parser.add_argument("--model_name_or_path", type=str, default="", help="the model name or its local path")
    
    args = parser.parse_args()
    return args
>>>>>>> Stashed changes
