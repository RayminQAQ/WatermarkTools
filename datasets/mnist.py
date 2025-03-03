from torch.utils.data import Dataset
from torchvision import datasets

class mnist(Dataset):
    def __init__(self, dataset_path=None, transform=None, is_train=True):
        """
        Custom MNIST dataset wrapper.

        Args:
            dataset_path (str or None): Path to the dataset directory. If None, it downloads automatically.
            transform (callable or None): Transformations to apply to the images.
            is_train (bool): Whether to load the training set (True) or test set (False).
        """
        super(mnist, self).__init__()
        print("here")
        # Local variable for storing the dataset
        self.dataset = -1  
        self.dataset_path = dataset_path  # Default: None
        self.transform = transform  # Default: None
        
        # Load dataset, automatically saves the file into the default location if dataset_path is not provided.
        if self.dataset_path is None:
            # If no dataset path is specified, download and store MNIST in "./data"
            self.dataset = datasets.MNIST(root="./data", train=is_train, download=True)
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