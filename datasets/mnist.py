from torch.utils.data import Dataset
from torchvision import datasets

class mnist(Dataset):
    def __init__(self, root_dir: None, transform: None, is_train: True):
        # Local variable
        self.dataset = -1
        self.root_dir = root_dir # default: None
        self.transform = transform # default: None
        
        # Load dataset (TBD)
        if self.root_dir is None:
            self.dataset = ""
        else:
            self.dataset = ""
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        if self.transform:
            image = self.transform(image)
        return image

    def get_featdim(self):
        sample_image, _ = self.dataset[0]
        if self.transform:
            sample_image = self.transform(sample_image)
        return sample_image.shape