import torchvision.transforms as transforms

from config import get_config
from datasets.get_dataset import get_datasets
from models.get_model import get_models

# Testing dependency
import torch

def train_or_eval():
    pass

def run():
    config = get_config()
    
    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # TBD: more dataset collections
    train_set, eval_set = get_datasets(args=config, transfrom=transform)

    # TBD: more models collections & dynamic support for tensor shape
    sample = train_set.get_feat()
    model = get_models(args=config, sample=sample)
    
    # TBD: Train / Evaluation for models + TBD: design for adding watermark into image
    

if __name__ == "__main__":
    run()