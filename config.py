"""
arg list:
- arg.watermark: the name of the watermark algorithm
- args.dataset: the name of dataset
- args.dataset_path: the path of args.dataset, it will be automatically download if None
"""

import argparse

def get_config():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--watermark", type=str, default="", help="the name of the watermark algorithm")
    parser.add_argument("--dataset", type=str, default="", help="the name of training / testing dataset")
    parser.add_argument("--dataset_path", type=str, default="", help="the path of args.dataset, it will be automatically download if emptys")
    
    args = parser.parse_args()
    return args