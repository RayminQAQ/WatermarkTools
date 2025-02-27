"""
File: get_dataset.py
Usage: dataset = get_datasets(self.args)
Ref: Idea is provoke from https://github.com/zeroQiaoba/MERTools/blob/master/MERBench/toolkit/data/__init__.py
"""

from torch.utils.data import Dataset
from .mnist import mnist

def get_datasets(args):
    """
    param:
    - transfrom: apply watermark algorithm and reshape image
    """
    WATERMARK_MAP = {
        # "algo": ?
    }
    
    transfrom = WATERMARK_MAP[arg.watermark] if arg.watermark in WATERMARK_MAP else None
    
    MODEL_DATASET_MAP = {
        "mnist": mnist(root_dir=args.dataset_path, transform=transfrom),
        "cbsd68": CBSD68(),
        "kodak24": Kodak24(),
    }
    
    dataset = MODEL_DATASET_MAP[args.dataset] # Allow one dataset at a time 
    return dataset

"""
class get_datasets(Dataset):
    def __init__(self, args):
        MODEL_DATASET_MAP = {
            "mnist": mnist,
            
        }
        
        self.dataset = MODEL_DATASET_MAP[args.dataset] # Allow one dataset at a time 
        
    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def get_featdim(self):
        return self.dataset.get_featdim()
"""