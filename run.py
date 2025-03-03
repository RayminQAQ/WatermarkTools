import torchvision.transforms as transforms

from config import get_config
from datasets.get_dataset import get_datasets
from models.get_model import get_models

# Testing dependency
import torch

def train():
    pass

def run():
    config = get_config()
    
    # 
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_set, eval_set = get_datasets(args=config, transfrom=transform)

    # TBD: fix the bug of get_featdim()
    sample = train_set.get_feat()
    model = get_models(args=config, sample=sample)
    
    # TBD: Train / Evaluation for models + TBD: design for adding watermark into image
    

if __name__ == "__main__":
    run()