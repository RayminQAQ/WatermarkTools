from config import get_config
from datasets.get_dataset import get_datasets
from models.get_model import get_models

# Testing dependency
import torch

def train():
    pass

def run():
    config = get_config()
    train_set, eval_set = get_datasets(args=config, transfrom=None)

    # TBD: fix the bug of get_featdim()
    # sample = train_set.get_featdim() # BUG: AttributeError: 'function' object has no attribute 'get_featdim'
    sample = torch.zeros(128, 3, 64, 12) 
    model = get_models(args=config, sample=sample)
    
    # TBD: Train / Evaluation for models + TBD: design for adding watermark into image
    

if __name__ == "__main__":
    run()