"""
File: get_dataset.py
Usage: dataset = get_datasets(args)
Ref: Idea is provoke from https://github.com/zeroQiaoba/MERTools/blob/master/MERBench/toolkit/data/__init__.py
"""

from typing import Optional
from torchvision import transforms
from torch.utils.data import Dataset
from .mnist import mnist_interface
from .cifar10 import cifar10_interface

def get_datasets(args, transform: Optional[transforms.Compose] = None) -> tuple[Dataset, Dataset]:
    """
    param:
    - transform: apply watermark algorithm and reshape image
    """

    MODEL_DATASET_MAP = {
        "mnist": mnist_interface(dataset_path=args.dataset_path, transform=transform),
        "cifar10": cifar10_interface(dataset_path=args.dataset_path, transform=transform)
        # "cbsd68": CBSD68(),
        # "kodak24": Kodak24(),
    }
    
    if args.dataset not in MODEL_DATASET_MAP:
        raise ValueError(f"WARNING: Dataset name _{args.dataset}_ is not supported by function get_datasets()")
    
    dataset_instance = MODEL_DATASET_MAP[args.dataset]
    train_set = dataset_instance.get_train_dataset()
    eval_set = dataset_instance.get_eval_dataset()  # 確保你的實作有這個方法
    
    if train_set is None or eval_set is None:
        raise ValueError(f"WARNING: _{args.dataset}_ dataset is empty. Please check your {args.dataset}'s implementation")
    
    return train_set, eval_set
