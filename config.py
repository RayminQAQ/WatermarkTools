# WatermarkTools/config.py
import argparse

def get_config():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--watermark", type=str, default=None, help="the name of the watermark algorithm")
    parser.add_argument("--dataset", type=str, default=None, help="the name of training / testing dataset")
    parser.add_argument("--dataset_path", type=str, default=None, help="the path of args.dataset, it will be automatically download if emptys")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="the model name or its local path")
    
    args = parser.parse_args()
    return args
