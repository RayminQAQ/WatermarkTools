from torch.utils.data import Dataset
from torchvision import datasets

class mnist(Dataset):
    def __init__(self, root_dir: None, transform):
        # Local variable
        self.dataset = -1
        self.transform = transform
        
        # Load dataset
        if root_dir is None:
            self.dataset = datasets.MNIST(
                root="./data",
                train=train,
                download=True,
                transform=self.transform
            )
        else:
            self.dataset = datasets.MNIST(
                root=root_dir,
                train=train,
                download=False,
                transform=self.transform
            )
        
        self.targets = self.dataset.targets
        
        
    def __len__(self):
        pass

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        label = self.targets[index]
        return image, label

    def get_featdim(self):
        pass
