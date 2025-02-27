"""
File: get_dataset.py
Usage: dataset = get_datasets(self.args)
Ref: Idea is provoke from https://github.com/zeroQiaoba/MERTools/blob/master/MERBench/toolkit/data/__init__.py
"""

from .mnist import mnist


def get_datasets(args, transfrom):
    """
    param:
    - transfrom: apply watermark algorithm and reshape image
    """

    MODEL_DATASET_MAP = {
        "mnist": mnist(root_dir=args.dataset_path, transform=transfrom),
        #"cbsd68": CBSD68(),
        #"kodak24": Kodak24(),
    }
    
    train_set, test_set, eval_set = None, None, None # Initialize
    train_set, test_set, eval_set = MODEL_DATASET_MAP[args.dataset] # Allow one dataset at a time 
    
    # Warning
    if eval_set is None:
        raise ValueError("WARNING: evaluation dataset is empty")
    
    return train_set, test_set, eval_set