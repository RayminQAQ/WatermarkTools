from config import get_config
from datasets.get_dataset import get_datasets

def run():
    config = get_config()
    train_set, eval_set = get_datasets(args=config, transfrom=None)

    # TBD: design for adding watermark into image
    
    # TBD: load pretrain models / checkpoint 
    
    # TBD: Train / Evaluation for models
    

if __name__ == "__main__":
    run()