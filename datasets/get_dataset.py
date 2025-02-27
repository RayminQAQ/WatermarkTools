"""
File: get_dataset.py
Usage: dataset = get_datasets(self.args)
Ref: Idea is provoke from https://github.com/zeroQiaoba/MERTools/blob/master/MERBench/toolkit/data/__init__.py
"""

from torch.utils.data import Dataset
from .mnist import mnist

def get_datasets(args, transfrom):
    """
    param:
    - transfrom: apply watermark algorithm and reshape image
    """
    # WATERMARK_MAP = {
    #    # "algo": ?
    #}
    
    #transfrom = WATERMARK_MAP[arg.watermark] if arg.watermark in WATERMARK_MAP else None
    
    MODEL_DATASET_MAP = {
        "mnist": mnist(args, root_dir=args.dataset_path, transform=transfrom),
        #"cbsd68": CBSD68(),
        #"kodak24": Kodak24(),
    }
    
    dataset = MODEL_DATASET_MAP[args.dataset] # Allow one dataset at a time 
    return dataset