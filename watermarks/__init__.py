from torch.utils.data import Dataset
from .mnist import mnist

class get_watermark_dataset(Dataset):
    def __init__(self, args, dataset: Dataset):
        WATERMARK_MAP = {
            "watermarkName": algo(dataset),
            
        }
        
        transform = "function"
        
        self.dataset = Dataset
        