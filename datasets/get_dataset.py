"""
File: get_dataset.py
Usage: dataset = get_datasets(self.args)
Ref: Idea is provoke from https://github.com/zeroQiaoba/MERTools/blob/master/MERBench/toolkit/data/__init__.py
"""

from .mnist import mnist_interface


def get_datasets(args, transfrom):
    """
    param:
    - transfrom: apply watermark algorithm and reshape image
    """

    MODEL_DATASET_MAP = {
        "mnist": mnist_interface(dataset_path=args.dataset_path, transform=transfrom),
        #"cbsd68": CBSD68(),
        #"kodak24": Kodak24(),
    }
    
    train_set, eval_set = None, None # Initialize
    train_set, eval_set = MODEL_DATASET_MAP[args.dataset].get_train_dataset, MODEL_DATASET_MAP[args.dataset].get_eval_dataset # Allow one dataset at a time 
    
    # Warning
    if train_set is None or eval_set is None:
        raise ValueError(f"WARNING: {args.dataset} dataset is empty")
    
    return train_set, eval_set