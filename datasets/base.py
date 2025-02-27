"""
    Template of the dataset class
"""

from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, root_dir: None, transform: None):
        super().__init__()
        
        # Variables
        self.dataset = None
        self.root_dir = root_dir # default: None
        self.transform = transform # default: None
        
        # Process or Read your dataset here
        
        # Error handling
        if self.dataset is None:
            raise ValueError("A dataset instance must be provided to BaseDataset.")
        
        if self.transform is None:
            print("Warning: BaseDataset has not transform.")
        
    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    def get_featdim(self):
        pass