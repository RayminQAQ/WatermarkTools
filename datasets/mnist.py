from torch.utils.data import Dataset
from .base import BaseDataset

class mnist(Dataset):
    def __init__(self, args, names, labels):
        self.dataset = "true nn.dataset class here"
        
        # Process your dataset here
        
        # Error handling
        if self.dataset is None:
            raise ValueError("A dataset instance must be provided to BaseDataset.")
        
    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    def get_featdim(self):
        pass
